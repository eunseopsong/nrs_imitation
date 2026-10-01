#include "Y2RobMotion/robot_motion.hpp"

// ===== [ADD] TCP/Base coordinate helper (minimal) =====
static inline void mat3_mul(const double R[3][3], const double v[3], double out[3])
{
    // out = R * v
    for(int i=0;i<3;i++){
        out[i] = R[i][0]*v[0] + R[i][1]*v[1] + R[i][2]*v[2];
    }
}

static inline void mat3_T_mul(const double R[3][3], const double v[3], double out[3])
{
    // out = R^T * v
    for(int i=0;i<3;i++){
        out[i] = R[0][i]*v[0] + R[1][i]*v[1] + R[2][i]*v[2];
    }
}

static inline double exp_smoothing_alpha(const double control_period, const double tau)
{
    if(tau <= 0.0) return 1.0;
    return 1.0 - std::exp(-control_period / tau);
}

static inline double compute_initial_damping_ratio(double mass0, double damper0, double stiffness0)
{
    const double mk0 = mass0 * stiffness0;
    if(mk0 <= 0.0) return 0.0;
    return damper0 / (2.0 * std::sqrt(mk0));
}

static inline double signed_force_hold(double desired_force, double hold_force)
{
    if(desired_force < 0.0) return -hold_force;
    return hold_force;
}

static inline double vec3_dot(const double a[3], const double b[3])
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static inline double vec3_norm(const double v[3])
{
    return std::sqrt(vec3_dot(v, v));
}

static inline void vec3_cross(const double a[3], const double b[3], double out[3])
{
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static inline void vec3_copy(const double src[3], double dst[3])
{
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
}

static inline void vec3_normalize_or_fallback(
    double v[3],
    const double fallback[3],
    const double epsilon)
{
    const double n = vec3_norm(v);
    if(n <= epsilon) {
        vec3_copy(fallback, v);
        return;
    }

    v[0] /= n;
    v[1] /= n;
    v[2] /= n;
}

static inline void project_onto_plane(const double v[3], const double normal[3], double out[3])
{
    const double proj = vec3_dot(v, normal);
    out[0] = v[0] - proj * normal[0];
    out[1] = v[1] - proj * normal[1];
    out[2] = v[2] - proj * normal[2];
}

static inline double clamp_abs(double value, double limit)
{
    if(limit <= 0.0) return 0.0;
    if(value > limit) return limit;
    if(value < -limit) return -limit;
    return value;
}

static inline void build_force_aligned_basis(
    const double q3_source_base[3],
    const bool contact_active,
    const std::array<double, 9>& prev_basis,
    const double epsilon,
    double Q[3][3])
{
    const double identity[3][3] = {{1,0,0},{0,1,0},{0,0,1}};

    if(!contact_active) {
        for(int r=0; r<3; ++r) {
            for(int c=0; c<3; ++c) {
                Q[r][c] = identity[r][c];
            }
        }
        return;
    }

    double q3[3] = { q3_source_base[0], q3_source_base[1], q3_source_base[2] };
    const double prev_q3[3] = { prev_basis[6], prev_basis[7], prev_basis[8] };
    vec3_normalize_or_fallback(q3, prev_q3, epsilon);
    double q1_prev[3] = { prev_basis[0], prev_basis[1], prev_basis[2] };
    double q2_prev[3] = { prev_basis[3], prev_basis[4], prev_basis[5] };
    double q1[3] = {0.0, 0.0, 0.0};

    project_onto_plane(q1_prev, q3, q1);
    if(vec3_norm(q1) <= epsilon) {
        project_onto_plane(q2_prev, q3, q1);
    }
    if(vec3_norm(q1) <= epsilon) {
        const double base_axes[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
        int best_axis = 0;
        double best_score = std::fabs(vec3_dot(base_axes[0], q3));
        for(int axis = 1; axis < 3; ++axis) {
            const double score = std::fabs(vec3_dot(base_axes[axis], q3));
            if(score < best_score) {
                best_score = score;
                best_axis = axis;
            }
        }
        project_onto_plane(base_axes[best_axis], q3, q1);
    }

    const double default_q1[3] = {1.0, 0.0, 0.0};
    vec3_normalize_or_fallback(q1, default_q1, epsilon);

    double q2[3];
    vec3_cross(q3, q1, q2);
    const double default_q2[3] = {0.0, 1.0, 0.0};
    vec3_normalize_or_fallback(q2, default_q2, epsilon);

    vec3_cross(q2, q3, q1);
    vec3_normalize_or_fallback(q1, default_q1, epsilon);

    for(int r = 0; r < 3; ++r) {
        Q[r][0] = q1[r];
        Q[r][1] = q2[r];
        Q[r][2] = q3[r];
    }
}

static inline void return_to_initial_mdk(
    Yadmittance_control& controller,
    const SetupParameters& config,
    const double control_period,
    const double mass0,
    const double damper0,
    const double stiffness0)
{
    const double mass_now = controller.adm_MDK_monitor(0);
    const double damper_now = controller.adm_MDK_monitor(1);
    const double stiffness_now = controller.adm_MDK_monitor(2);

    const double alpha_m =
        exp_smoothing_alpha(control_period, config.force_switch_return_tau_m);
    const double alpha_d =
        exp_smoothing_alpha(control_period, config.force_switch_return_tau_d);
    const double alpha_k =
        exp_smoothing_alpha(control_period, config.force_switch_return_tau_k);

    const double target_mass = mass_now + alpha_m * (mass0 - mass_now);
    const double target_stiffness = stiffness_now + alpha_k * (stiffness0 - stiffness_now);
    const double target_damper_raw = damper_now + alpha_d * (damper0 - damper_now);

    const double zeta0 = compute_initial_damping_ratio(mass0, damper0, stiffness0);
    const double damping_floor = 2.0 * zeta0 * std::sqrt(std::max(0.0, target_mass * target_stiffness));
    const double target_damper = std::max(target_damper_raw, damping_floor);

    controller.adm_1D_MDK(target_mass, target_damper, target_stiffness);
}

static inline void return_md_to_initial_with_zero_stiffness(
    Yadmittance_control& controller,
    const SetupParameters& config,
    const double control_period,
    const double mass0,
    const double damper0,
    const double virtual_stiffness0)
{
    const double mass_now = controller.adm_MDK_monitor(0);
    const double damper_now = controller.adm_MDK_monitor(1);

    const double alpha_m =
        exp_smoothing_alpha(control_period, config.force_switch_return_tau_m);
    const double alpha_d =
        exp_smoothing_alpha(control_period, config.force_switch_return_tau_d);

    const double target_mass = mass_now + alpha_m * (mass0 - mass_now);
    const double target_damper_raw = damper_now + alpha_d * (damper0 - damper_now);

    const double zeta0 = compute_initial_damping_ratio(mass0, damper0, virtual_stiffness0);
    const double damping_floor = 2.0 * zeta0 * std::sqrt(std::max(0.0, target_mass * virtual_stiffness0));
    const double target_damper = std::max(target_damper_raw, damping_floor);

    controller.adm_1D_MDK(target_mass, target_damper, 0.0);
}

void robot_motion::initialize_force_control_state()
{
    target_pose = current_pose;
    target_angles = current_angles;
    force_q_basis_ = {1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0};

    FC_AC_desX = std::vector<double>(9, 0.0);
    for(int i = 0; i < 6; ++i) {
        FC_AC_desX[i] = current_pose[i];
    }
    FC_AC_desX[0] = FC_AC_desX[0]/1000;
    FC_AC_desX[1] = FC_AC_desX[1]/1000;
    FC_AC_desX[2] = FC_AC_desX[2]/1000;

    FC_MASS = {2, 2, 2, 0.5, 0.5, 0.5};
    FC_DAMPER = {6000, 6000, 6000, 100, 100, 100};
    FC_STIFFNESS = {2000,2000,2000,100,100,100};

    for(int i = 0; i < 6; ++i){
        AControl[i].adm_1D_MDK(FC_MASS[i],FC_DAMPER[i],FC_STIFFNESS[i]);
        AControl[i].reset(FC_AC_desX[i]);
    }

    for(int i = 0; i < 3; ++i){FAAC3step[i]->FAAC_Init(FC_MASS[i],FC_DAMPER[i],FC_STIFFNESS[i]);}

    for(int i = 0; i < 3; ++i){
        if(policy_context_naf_mdgradi[i]) policy_context_naf_mdgradi[i]->reset_state();
        if(policy_naf_mdgradi[i]) policy_naf_mdgradi[i]->reset_state();
        if(policy_naf_massvari[i]) policy_naf_massvari[i]->reset_state();
    }

    RCLCPP_INFO(node_->get_logger(),"Force control was initialized");
}

void robot_motion::update_force_target_from_joystick()
{
    for(int i = 0; i < 6; ++i)
    {
        joy_target_pose[i] += joy_velocity_command_[i] * Control_period_;
    }

    FC_AC_desX[0] = joy_target_pose[0] / 1000.0;
    FC_AC_desX[1] = joy_target_pose[1] / 1000.0;
    FC_AC_desX[2] = joy_target_pose[2] / 1000.0;
    FC_AC_desX[3] = joy_target_pose[3];
    FC_AC_desX[4] = joy_target_pose[4];
    FC_AC_desX[5] = joy_target_pose[5];

    for(int i = 0; i < 3; ++i)
    {
        FC_AC_desX[i + 6] = joy_force_command_[i];
    }
}

void robot_motion::execute_force_control()
{
    if(config_.force_control_mode == 0) force_con_mode = "Classic";
    else if(config_.force_control_mode == 1) force_con_mode = "FAAC";
    else if(config_.force_control_mode == 2) force_con_mode = "Naf-massVari";
    else if(config_.force_control_mode == 3) force_con_mode = "Naf-mdGradi";
    else if(config_.force_control_mode == 4) force_con_mode = "Context-NAF-mdGradi";

    /**** START OF FORCE-CONTROL ****/

    // Coordinate transform basis: base <- local
    double R_bt[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    if(config_.force_control_coordinate == 1)
    {
        // current_angles 기준 TCP 회전 가져오기
        YMatrix curHTM = kinematics_->forwardKinematics(current_angles);
        YMatrix curR   = curHTM.extract(0,0,3,3);

        // YMatrix -> double[3][3]
        for(int r=0;r<3;r++){
            for(int c=0;c<3;c++){
                R_bt[r][c] = curR[r][c];
            }
        }
    }

    // base 벡터들 (m 단위)
    double x_base[3]  = { current_pose[0]/1000.0, current_pose[1]/1000.0, current_pose[2]/1000.0 };
    double xd_base[3] = { FC_AC_desX[0],          FC_AC_desX[1],          FC_AC_desX[2]          };
    double xc_base_prev[3] = { AC_pose[0], AC_pose[1], AC_pose[2] }; // 이전 step AC_pose (m)

    double fext_base[3] = { ft1data[0], ft1data[1], ft1data[2] };
    double xc_minus_x_base[3] = {
        xc_base_prev[0] - x_base[0],
        xc_base_prev[1] - x_base[1],
        xc_base_prev[2] - x_base[2]
    };

    // tcp로 투영된 스칼라들
    double x_tcp[3]   = {0,0,0};
    double xd_tcp[3]  = {0,0,0};
    double xc_tcp_prev[3] = {0,0,0};
    double fext_tcp[3] = {0,0,0};

    if(config_.force_control_coordinate == 1 ||
       config_.force_control_coordinate == 2)
    {
        if(config_.force_control_coordinate == 2) {
            const bool contact_active =
                (vec3_norm(fext_base) >= config_.force_q_contact_threshold);
            build_force_aligned_basis(
                xc_minus_x_base,
                contact_active,
                force_q_basis_,
                config_.force_q_basis_eps,
                R_bt);
            for(int r = 0; r < 3; ++r) {
                for(int c = 0; c < 3; ++c) {
                    force_q_basis_[3 * c + r] = R_bt[r][c];
                }
            }
        }

        // local = R^T * base
        mat3_T_mul(R_bt, x_base,       x_tcp);
        mat3_T_mul(R_bt, xd_base,      xd_tcp);
        mat3_T_mul(R_bt, xc_base_prev, xc_tcp_prev);
        mat3_T_mul(R_bt, fext_base,    fext_tcp);
    }

    for(int i=0;i<6;i++) // Xd, Fd, Fext
    {
        // ----------------------------
        // 1) i<3 (Position-force axes) : 좌표계(Base/TCP)에 따라 입력만 통일
        // ----------------------------
        if(i < 3)
        {
            // 좌표계별로 스칼라 축 입력 통일
            double Xd   = 0.0;
            double x    = 0.0;
            double xc_p = 0.0;
            double Fext = 0.0;

            if(config_.force_control_coordinate == 1 ||
               config_.force_control_coordinate == 2) {
                // Local frame 기준 (TCP or force-aligned Q)
                Xd   = xd_tcp[i];
                x    = x_tcp[i];
                xc_p = xc_tcp_prev[i];
                Fext = fext_tcp[i];
            } else {
                // Base frame 기준 (기존 코드와 동일 의미)
                Xd   = FC_AC_desX[i];            // (m)
                x    = current_pose[i] / 1000.0; // (m)
                xc_p = AC_pose[i];               // (m) 이전 step 출력
                Fext = ft1data[i];               // (N)
            }

            double Fd = FC_AC_desX[i+6];
            bool desired_force_active =
                (std::fabs(Fd) > config_.force_switch_desired_force_threshold);
            bool actual_force_active =
                (std::fabs(Fext) > config_.force_switch_actual_force_threshold);
            bool force_control_active = desired_force_active && actual_force_active;
            double commanded_Fd = (desired_force_active && !actual_force_active)
                ? signed_force_hold(Fd, config_.force_switch_precontact_force_hold)
                : Fd;

            if(config_.force_control_coordinate == 2) {
                const double desired_force_base[3] = {
                    FC_AC_desX[6], FC_AC_desX[7], FC_AC_desX[8]
                };
                double desired_force_q[3] = {0.0, 0.0, 0.0};
                mat3_T_mul(R_bt, desired_force_base, desired_force_q);

                if(i < 2) {
                    Fd = 0.0;
                    desired_force_active = false;
                    force_control_active = false;
                    commanded_Fd = 0.0;
                } else {
                    const bool external_fd_active =
                        (std::fabs(desired_force_q[2]) >
                         config_.force_q_external_fd_threshold);
                    const bool contact_active =
                        (vec3_norm(fext_base) >= config_.force_q_contact_threshold);
                    const double internal_fd = clamp_abs(
                        config_.force_q_internal_force_gain * (Xd - x),
                        config_.force_q_internal_force_saturation);

                    Fd = external_fd_active ? desired_force_q[2] : (contact_active ? internal_fd : 0.0);
                    desired_force_active =
                        std::fabs(Fd) > config_.force_switch_desired_force_threshold;
                    actual_force_active = contact_active;
                    force_control_active = desired_force_active && actual_force_active;
                    commanded_Fd = (desired_force_active && !actual_force_active)
                        ? signed_force_hold(
                            Fd, config_.force_switch_precontact_force_hold)
                        : Fd;
                }
            }

            // ----------------------------
            // 2) Force_Con_Mode별 로직 (중복 없이 딱 1번만)
            // ----------------------------
            if(config_.force_control_mode == 0) { // Classical
                if(config_.force_control_coordinate == 2 && i < 2) {
                    return_to_initial_mdk(
                        AControl[i], config_, Control_period_, FC_MASS[i], FC_DAMPER[i], FC_STIFFNESS[i]);
                    AC_pose[i] = AControl[i].adm_1D_control(Xd, 0.0, Fext);
                    continue;
                }
                if(force_control_active) {
                    AControl[i].adm_1D_MDK(FC_MASS[i], FC_DAMPER[i], 0.0);
                } else if(desired_force_active) {
                    return_md_to_initial_with_zero_stiffness(
                        AControl[i], config_, Control_period_, FC_MASS[i], FC_DAMPER[i], FC_STIFFNESS[i]);
                } else {
                    return_to_initial_mdk(AControl[i], config_, Control_period_, FC_MASS[i], FC_DAMPER[i], FC_STIFFNESS[i]);
                }
                AC_pose[i] = AControl[i].adm_1D_control(Xd, commanded_Fd, Fext);

            }
            else if(config_.force_control_mode == 1) { // FAAC
                if(config_.force_control_coordinate == 2 && i < 2) {
                    FAAC_flag[i] = false;
                    return_to_initial_mdk(
                        AControl[i], config_, Control_period_, FC_MASS[i], FC_DAMPER[i], FC_STIFFNESS[i]);
                    AC_pose[i] = AControl[i].adm_1D_control(Xd, 0.0, Fext);
                    continue;
                }
                FAAC_flag[i] = force_control_active;
                FAAC_flag[abs(i-1)] = (FAAC_flag[i]) ? false : FAAC_flag[abs(i-1)];
                FAAC_flag[abs(i-2)] = (FAAC_flag[i]) ? false : FAAC_flag[abs(i-2)];

                if(FAAC_flag[i]) {
                    double Tank_energy = 5;
                    // (중요) Act_Pose(x), AC_Pose(xc_p), Fext, Fd 모두 같은 좌표계 스칼라
                    auto TSFAAC_MDK = FAAC3step[i]->FAAC_MDKob_RUN(
                        Tank_energy,
                        Fext,
                        Fd,
                        /*AC_Pose=*/xc_p,
                        /*Act_Pose=*/x
                    );
                    AControl[i].adm_1D_MDK(TSFAAC_MDK.Mass, TSFAAC_MDK.Damping, 0.0);
                } else if(desired_force_active) {
                    return_md_to_initial_with_zero_stiffness(
                        AControl[i], config_, Control_period_, FC_MASS[i], FC_DAMPER[i], FC_STIFFNESS[i]);
                } else {
                    return_to_initial_mdk(AControl[i], config_, Control_period_, FC_MASS[i], FC_DAMPER[i], FC_STIFFNESS[i]);
                }
                AC_pose[i] = AControl[i].adm_1D_control(Xd, commanded_Fd, Fext);

            }
            else if(config_.force_control_mode == 2) { // Naf_massVari
                if(config_.force_control_coordinate == 2 && i < 2) {
                    return_to_initial_mdk(
                        AControl[i], config_, Control_period_, FC_MASS[i], FC_DAMPER[i], FC_STIFFNESS[i]);
                    AC_pose[i] = AControl[i].adm_1D_control(Xd, 0.0, Fext);
                    continue;
                }
                if(force_control_active) {
                    auto out = policy_naf_massvari[i]->run(
                        /*xc=*/xc_p, /*x=*/x, /*Fd=*/Fd, /*Fext=*/Fext
                    );
                    AControl[i].adm_1D_MDK(out[0], policy_naf_massvari[i]->get_applied_damping(), 0.0);
                } else if(desired_force_active) {
                    return_md_to_initial_with_zero_stiffness(
                        AControl[i], config_, Control_period_, FC_MASS[i], FC_DAMPER[i], FC_STIFFNESS[i]);
                } else {
                    return_to_initial_mdk(AControl[i], config_, Control_period_, FC_MASS[i], FC_DAMPER[i], FC_STIFFNESS[i]);
                }
                AC_pose[i] = AControl[i].adm_1D_control(Xd, commanded_Fd, Fext);

            }
            else if(config_.force_control_mode == 3) { // Naf_mdGradi
                if(config_.force_control_coordinate == 2 && i < 2) {
                    return_to_initial_mdk(
                        AControl[i], config_, Control_period_, FC_MASS[i], FC_DAMPER[i], FC_STIFFNESS[i]);
                    AC_pose[i] = AControl[i].adm_1D_control(Xd, 0.0, Fext);
                    continue;
                }
                if(force_control_active) {
                    auto out = policy_naf_mdgradi[i]->run(xc_p, x, Fd, Fext);
                    AControl[i].adm_1D_MDK(out[0], policy_naf_mdgradi[i]->get_applied_damping(), 0.0);
                } else if(desired_force_active) {
                    return_md_to_initial_with_zero_stiffness(
                        AControl[i], config_, Control_period_, FC_MASS[i], FC_DAMPER[i], FC_STIFFNESS[i]);
                } else {
                    return_to_initial_mdk(AControl[i], config_, Control_period_, FC_MASS[i], FC_DAMPER[i], FC_STIFFNESS[i]);
                }
                AC_pose[i] = AControl[i].adm_1D_control(Xd, commanded_Fd, Fext);

            }
            else if(config_.force_control_mode == 4) {
                if(config_.force_control_coordinate == 2 && i < 2) {
                    auto out = policy_context_naf_mdgradi[i]->run(xc_p, x, 0.0, Fext);
                    (void)out;
                    return_to_initial_mdk(
                        AControl[i], config_, Control_period_, FC_MASS[i], FC_DAMPER[i], FC_STIFFNESS[i]);
                    AC_pose[i] = AControl[i].adm_1D_control(Xd, 0.0, Fext);
                    continue;
                }
                // Keep the recurrent policy/context state warm even before contact.
                auto out = policy_context_naf_mdgradi[i]->run(xc_p, x, Fd, Fext);

                if(force_control_active) {
                    // out = [mass_act, alpha_act]
                    double MD_Ratio = 1000; // kept same convention as training/inference wrapper default
                    AControl[i].adm_1D_MDK(out[0], MD_Ratio * out[0] * out[1], 0.0);
                } else if(desired_force_active) {
                    return_md_to_initial_with_zero_stiffness(
                        AControl[i], config_, Control_period_, FC_MASS[i], FC_DAMPER[i], FC_STIFFNESS[i]);
                } else {
                    return_to_initial_mdk(AControl[i], config_, Control_period_, FC_MASS[i], FC_DAMPER[i], FC_STIFFNESS[i]);
                }
                AC_pose[i] = AControl[i].adm_1D_control(Xd, commanded_Fd, Fext);
            }

            continue; // i<3 처리 끝
        }

        // ----------------------------
        // 3) i>=3 (Orientation axes) : 기존 동작 유지 (좌표계 영향 없음)
        // ----------------------------
        AC_pose[i] = AControl[i].adm_1D_control(FC_AC_desX[i], 0.0, 0.0);
    }

    // ===== [ADD] TCP에서 계산한 AC_pose(0..2)를 base로 복원 =====
    if(config_.force_control_coordinate == 1 ||
       config_.force_control_coordinate == 2)
    {
        // 현재 AC_pose[0..2]는 local 스칼라로 저장되어 있음 -> 벡터로 만들어 base로 회전
        double xc_tcp_new[3]  = { AC_pose[0], AC_pose[1], AC_pose[2] };
        double xc_base_new[3] = {0,0,0};

        // base = R * local
        mat3_mul(R_bt, xc_tcp_new, xc_base_new);

        AC_pose[0] = xc_base_new[0];
        AC_pose[1] = xc_base_new[1];
        AC_pose[2] = xc_base_new[2];
    }

    target_pose = AC_pose;
    target_pose[0] = target_pose[0]*1000; // m -> mm
    target_pose[1] = target_pose[1]*1000; // m -> mm
    target_pose[2] = target_pose[2]*1000; // m -> mm

    /**** END OF FORCE-CONTROL ****/

    /* Generate target HTM */
    std::vector<double> target_ori = {target_pose[3], target_pose[4], target_pose[5]};
    auto target_rot = YMatrix::fromSpatialAngle(target_ori);
    target_HTM = YMatrix::identity(4);
    target_HTM.insert(0, 0, target_rot);
    target_HTM[0][3] = target_pose[0]; // mm unit
    target_HTM[1][3] = target_pose[1]; // mm unit
    target_HTM[2][3] = target_pose[2]; // mm unit

    /* Inverse kinematics using QP-solver + execution */
    target_angles = kinematics_->solve_IK(target_angles, target_HTM);

    /* Data upload to past */
    pre_control_mode = control_mode; // Store previous control mode for comparison
}

void robot_motion::control_force()
{
    control_mode = "Force";

    if(pre_control_mode != control_mode)
    {
        initialize_force_control_state();
    }

    execute_force_control();
}
