// vr_calibration.cpp  (Option B: Auto-capture + Update R_Adj, T_AD, T_BC, T_SA in one YAML write)
// v9: T_FIX uses the same solved T_BC tool-center chain as runtime

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <array>
#include <string>
#include <mutex>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <ctime>   // for timestamp
#include <limits>

// ================= Constants =================
static constexpr double kPi = 3.14159265358979323846;

static inline double deg2rad(double d){ return d * kPi / 180.0; }
static inline double rad2deg(double r){ return r * 180.0 / kPi; }

static inline double clampd(double x, double lo, double hi)
{
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

// local time string: "YYYY.MM.DD HH:MM"
static std::string nowLocalString()
{
  std::time_t t = std::time(nullptr);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  std::ostringstream oss;
  oss << std::setfill('0')
      << (tm.tm_year + 1900) << "."
      << std::setw(2) << (tm.tm_mon + 1) << "."
      << std::setw(2) << tm.tm_mday << " "
      << std::setw(2) << tm.tm_hour << ":"
      << std::setw(2) << tm.tm_min;
  return oss.str();
}

static void ensureParentDirectoryExists(const std::string& file_path)
{
  const auto parent = std::filesystem::path(file_path).parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
}

// ================= Rodrigues (rotvec -> R) =================
static void rotvecToRotMatRad(const std::array<double,3>& w,
                              std::array<double,9>& R)
{
  double th = std::sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
  R = {1,0,0, 0,1,0, 0,0,1};
  if (th < 1e-12) return;

  double ux = w[0]/th, uy = w[1]/th, uz = w[2]/th;
  double s = std::sin(th), c = std::cos(th), v = 1.0 - c;

  R[0] = ux*ux*v + c;
  R[1] = ux*uy*v - uz*s;
  R[2] = ux*uz*v + uy*s;

  R[3] = uy*ux*v + uz*s;
  R[4] = uy*uy*v + c;
  R[5] = uy*uz*v - ux*s;

  R[6] = uz*ux*v - uy*s;
  R[7] = uz*uy*v + ux*s;
  R[8] = uz*uz*v + c;
}

static std::array<double,9> matMul3(const std::array<double,9>& A,
                                    const std::array<double,9>& B)
{
  std::array<double,9> C{};
  for (int r=0;r<3;r++){
    for (int c=0;c<3;c++){
      double s=0;
      for (int k=0;k<3;k++) s += A[r*3+k]*B[k*3+c];
      C[r*3+c]=s;
    }
  }
  return C;
}

static std::array<double,9> matT3(const std::array<double,9>& A)
{
  return {A[0],A[3],A[6],
          A[1],A[4],A[7],
          A[2],A[5],A[8]};
}

static double trace3(const std::array<double,9>& A)
{
  return A[0] + A[4] + A[8];
}

static double rotAngleBetweenRad(const std::array<double,9>& R_target,
                                 const std::array<double,9>& R_current)
{
  auto RtT  = matT3(R_target);
  auto Rrel = matMul3(RtT, R_current);
  double cosang = (trace3(Rrel) - 1.0) * 0.5;
  cosang = clampd(cosang, -1.0, 1.0);
  return std::acos(cosang);
}

// ================= Waypoint =================
struct Waypoint
{
  std::array<double,6> pose; // x y z wx wy wz
  double lin_vel;            // cmd_6D: desired_lin_vel
  double ang_vel;            // cmd_6D: desired_ang_vel
  double holding_time_s;     // cmd_6D: holding_time
};

// ================= helper: Eigen rigid transform =================
static Eigen::Matrix4d makeT(const Eigen::Matrix3d& R, const Eigen::Vector3d& p)
{
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3,3>(0,0) = R;
  T.block<3,1>(0,3) = p;
  return T;
}

static Eigen::Matrix4d invT(const Eigen::Matrix4d& T)
{
  Eigen::Matrix4d Ti = Eigen::Matrix4d::Identity();
  const Eigen::Matrix3d R = T.block<3,3>(0,0);
  const Eigen::Vector3d p = T.block<3,1>(0,3);
  Ti.block<3,3>(0,0) = R.transpose();
  Ti.block<3,1>(0,3) = -R.transpose()*p;
  return Ti;
}

static Eigen::Matrix3d projectToSO3(const Eigen::Matrix3d& R_in)
{
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_in, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  const Eigen::Matrix3d V = svd.matrixV();
  Eigen::Matrix3d R = U * V.transpose();
  if (R.determinant() < 0.0) {
    U.col(2) *= -1.0;
    R = U * V.transpose();
  }
  return R;
}

static double rotAngleBetweenRad(const Eigen::Matrix3d& R1,
                                 const Eigen::Matrix3d& R2)
{
  const Eigen::Matrix3d R = R1.transpose() * R2;
  double cosang = (R.trace() - 1.0) * 0.5;
  cosang = clampd(cosang, -1.0, 1.0);
  return std::acos(cosang);
}

static Eigen::Matrix<double,9,9> kron3(const Eigen::Matrix3d& A, const Eigen::Matrix3d& B)
{
  Eigen::Matrix<double,9,9> K;
  for (int i=0;i<3;i++){
    for (int j=0;j<3;j++){
      K.block<3,3>(3*i,3*j) = A(i,j) * B;
    }
  }
  return K;
}

static double rotDiffAngleDeg(const Eigen::Matrix3d& R1, const Eigen::Matrix3d& R2)
{
  Eigen::Matrix3d Rrel = R1.transpose() * R2;
  double cosang = (Rrel.trace() - 1.0) * 0.5;
  cosang = clampd(cosang, -1.0, 1.0);
  return rad2deg(std::acos(cosang));
}

// ================= Class =================
class VrCalibration : public rclcpp::Node
{
public:
  VrCalibration()
  : Node("vr_calibration_target_based"),
    steady_clock_(RCL_STEADY_TIME)
  {
    // ----------------------------
    // Files
    // ----------------------------
    const std::string vr_calibration_share =
      ament_index_cpp::get_package_share_directory("vr_calibration");
    const std::string vive_tracker_share =
      ament_index_cpp::get_package_share_directory("vive_tracker_ros2");
    const std::string txt_dir = vr_calibration_share + "/txt";

    waypoint_file_ = txt_dir + "/for_vr_calibration_point_v6.txt";
    ee_path_ = txt_dir + "/ur10_ee.txt";
    vr_path_ = txt_dir + "/ur10_vr.txt";
    const char* home_env = std::getenv("HOME");
    const std::string vive_tracker_src_yaml =
      std::string(home_env ? home_env : "") +
      "/nrs_imitation/behavior_ws/src/vive_tracker_ros2/yaml/calibration_matrix.yaml";
    calib_yaml_path_ = std::filesystem::exists(vive_tracker_src_yaml)
      ? vive_tracker_src_yaml
      : vive_tracker_share + "/yaml/calibration_matrix.yaml";

    // ----------------------------
    // Tunables
    // ----------------------------
    pos_enter_mm_ = 20.0;
    pos_exit_mm_  = 60.0;

    ori_enter_deg_ = 25.0;
    ori_exit_deg_  = 60.0;

    vel_thresh_mms_      = 15.0;  // mm/s
    angvel_thresh_dps_   = 8.0;   // deg/s

    hold_time_s_ = 2.0;

    cp_fresh_s_       = 1.0;
    vr_capture_age_s_ = 30.0;

    target_timeout_s_ = 300.0;
    loop_hz_          = 200.0;

    cp_unit_probe_N_ = 30;

    // ----------------------------
    // T_SA params
    // ----------------------------
    this->declare_parameter<double>("t_sa_w_des_z", 1.5707963267948966); // rad (≈ pi/2)
    this->declare_parameter<double>("t_sa_wait_timeout_s", 15.0);
    this->declare_parameter<double>("t_sa_hold_s", 0.25);
    this->declare_parameter<double>("t_sa_fresh_s", 1.0);

    // T_SA update mode
    // - keep   : YAML의 기존 T_SA 유지
    // - update : pre-phase에서 /calibrated_pose 기반으로 T_SA 재계산
    this->declare_parameter<std::string>("t_sa_mode", "update"); // "keep" or "update"
    this->declare_parameter<double>("t_sa_max_delta_deg", 180.0); // update 시 old->new 변화량 제한
    this->declare_parameter<bool>("z_fix_enable", true);
    this->declare_parameter<double>("z_fix_max_tilt_deg", 5.0);
    this->declare_parameter<bool>("z_residual_enable", true);
    this->declare_parameter<double>("z_residual_max_correction_mm", 10.0);
    this->declare_parameter<std::string>("waypoint_file", waypoint_file_);
    this->declare_parameter<std::string>("ee_output_file", ee_path_);
    this->declare_parameter<std::string>("vr_output_file", vr_path_);
    this->declare_parameter<std::string>("calib_yaml_file", calib_yaml_path_);
    this->declare_parameter<bool>("radj_enable", false);
    this->declare_parameter<int>("radj_sample_count", 0); // <=0: use all captured samples
    this->declare_parameter<double>("capture_hold_time_s", hold_time_s_);
    this->declare_parameter<double>("capture_min_hold_time_s", min_hold_time_s_);
    this->declare_parameter<double>("capture_window_s", 0.5);
    this->declare_parameter<int>("capture_min_clean_samples", 20);
    this->declare_parameter<double>("vr_capture_age_s", 0.2);
    this->declare_parameter<double>("max_capture_sync_dt_s", 0.05);
    this->declare_parameter<double>("capture_max_vr_std_mm", 10.0);
    this->declare_parameter<double>("max_calib_position_rms_mm", 50.0);
    this->declare_parameter<bool>("handeye_auto_trim_low_rotation_prefix", true);
    this->declare_parameter<double>("handeye_prefix_rotation_span_deg", 5.0);
    this->declare_parameter<int>("handeye_min_trim_prefix_samples", 4);

    t_sa_w_des_z_        = this->get_parameter("t_sa_w_des_z").as_double();
    t_sa_wait_timeout_s_ = this->get_parameter("t_sa_wait_timeout_s").as_double();
    t_sa_hold_s_         = this->get_parameter("t_sa_hold_s").as_double();
    t_sa_fresh_s_        = this->get_parameter("t_sa_fresh_s").as_double();

    t_sa_mode_           = this->get_parameter("t_sa_mode").as_string();
    std::transform(t_sa_mode_.begin(), t_sa_mode_.end(), t_sa_mode_.begin(), ::tolower);
    t_sa_max_delta_deg_  = this->get_parameter("t_sa_max_delta_deg").as_double();
    z_fix_enable_        = this->get_parameter("z_fix_enable").as_bool();
    z_fix_max_tilt_deg_  = this->get_parameter("z_fix_max_tilt_deg").as_double();
    z_residual_enable_   = this->get_parameter("z_residual_enable").as_bool();
    z_residual_max_correction_m_ =
      std::max(0.0, this->get_parameter("z_residual_max_correction_mm").as_double()) * 1e-3;
    waypoint_file_       = this->get_parameter("waypoint_file").as_string();
    ee_path_             = this->get_parameter("ee_output_file").as_string();
    vr_path_             = this->get_parameter("vr_output_file").as_string();
    calib_yaml_path_     = this->get_parameter("calib_yaml_file").as_string();
    radj_enable_         = this->get_parameter("radj_enable").as_bool();
    const int64_t radj_sample_count_param = this->get_parameter("radj_sample_count").as_int();
    radj_use_all_samples_ = (radj_sample_count_param <= 0);
    radj_sample_count_ = radj_use_all_samples_
      ? 0
      : static_cast<size_t>(std::max<int64_t>(3, radj_sample_count_param));
    hold_time_s_         = std::max(0.1, this->get_parameter("capture_hold_time_s").as_double());
    min_hold_time_s_     = std::max(0.0, this->get_parameter("capture_min_hold_time_s").as_double());
    min_hold_time_s_     = std::min(min_hold_time_s_, hold_time_s_);
    capture_window_s_    = std::max(0.05, this->get_parameter("capture_window_s").as_double());
    capture_min_clean_samples_ =
      static_cast<size_t>(std::max<int64_t>(1, this->get_parameter("capture_min_clean_samples").as_int()));
    vr_capture_age_s_    = std::max(0.01, this->get_parameter("vr_capture_age_s").as_double());
    max_capture_sync_dt_s_ = std::max(0.0, this->get_parameter("max_capture_sync_dt_s").as_double());
    capture_max_vr_std_mm_ = std::max(0.0, this->get_parameter("capture_max_vr_std_mm").as_double());
    max_calib_position_rms_mm_ = std::max(1.0, this->get_parameter("max_calib_position_rms_mm").as_double());
    handeye_auto_trim_low_rotation_prefix_ =
      this->get_parameter("handeye_auto_trim_low_rotation_prefix").as_bool();
    handeye_prefix_rotation_span_deg_ =
      std::max(0.0, this->get_parameter("handeye_prefix_rotation_span_deg").as_double());
    handeye_min_trim_prefix_samples_ =
      static_cast<size_t>(std::max<int64_t>(1, this->get_parameter("handeye_min_trim_prefix_samples").as_int()));

    // ----------------------------
    // Waypoints
    // ----------------------------
    loadWaypointsAndDecideWpUnits();
    buildTargetIndices();

    // ---- output file refresh ----
    ensureParentDirectoryExists(ee_path_);
    ensureParentDirectoryExists(vr_path_);
    ee_ofs_.open(ee_path_, std::ios::out | std::ios::trunc);
    vr_ofs_.open(vr_path_, std::ios::out | std::ios::trunc);

    if (!ee_ofs_.is_open() || !vr_ofs_.is_open())
      throw std::runtime_error("Failed to open output files (truncate mode)");

    // ----------------------------
    // subscriptions
    // ----------------------------
    sub_currentP_ =
      create_subscription<std_msgs::msg::Float64MultiArray>(
        "/ur10skku/currentP", 10,
        std::bind(&VrCalibration::cbCurrentP, this, std::placeholders::_1));

    sub_vr_ =
      create_subscription<geometry_msgs::msg::PoseStamped>(
        "/raw_pose", 10,
        std::bind(&VrCalibration::cbVR, this, std::placeholders::_1));

    // for T_SA update
    sub_calibrated_pose_ =
      create_subscription<std_msgs::msg::Float64MultiArray>(
        "/calibrated_pose", 10,
        std::bind(&VrCalibration::cbCalibratedPose, this, std::placeholders::_1));

    // load existing constants (T_CE, T_SA_old)
    loadExistingYamlConstants();

    RCLCPP_INFO(get_logger(),
      "[INIT] wp=%zu hold=%zu auto=on",
      waypoints_.size(), target_indices_.size());
    RCLCPP_INFO(get_logger(),
      "[FILES] waypoint=%s",
      waypoint_file_.c_str());
    RCLCPP_INFO(get_logger(),
      "[FILES] ee_out=%s vr_out=%s yaml=%s",
      ee_path_.c_str(), vr_path_.c_str(), calib_yaml_path_.c_str());

    RCLCPP_INFO(get_logger(),
      "[T_SA] mode=%s max=%.1fdeg",
      t_sa_mode_.c_str(), t_sa_max_delta_deg_);
    RCLCPP_INFO(get_logger(),
      "[HOLD] min=%.2fs fb=%.2fs",
      min_hold_time_s_, hold_time_s_);
    RCLCPP_INFO(get_logger(),
      "[STOP] v<%.1fmm/s w<%.1fdeg/s",
      vel_thresh_mms_, angvel_thresh_dps_);
    RCLCPP_INFO(get_logger(),
      "[CLEAN] win=%.2fs n=%zu std<%.1fmm",
      capture_window_s_, capture_min_clean_samples_, capture_max_vr_std_mm_);
    RCLCPP_INFO(get_logger(),
      "[SYNC] vr<%.2fs dt<%.3fs",
      vr_capture_age_s_, max_capture_sync_dt_s_);
    const std::string radj_sample_count_log =
      radj_use_all_samples_ ? "all" : std::to_string(radj_sample_count_);
    RCLCPP_INFO(get_logger(), "[R_ADJ] en=%s n=%s",
      radj_enable_ ? "true" : "false", radj_sample_count_log.c_str());
    RCLCPP_INFO(get_logger(), "[HAND_EYE] trim_prefix=%s span>=%.1fdeg min_prefix=%zu",
      handeye_auto_trim_low_rotation_prefix_ ? "true" : "false",
      handeye_prefix_rotation_span_deg_,
      handeye_min_trim_prefix_samples_);
  }

  void run()
  {
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(shared_from_this());

    if (target_indices_.empty()) {
      RCLCPP_WARN(get_logger(), "[INIT] no hold targets");
      return;
    }

    // ==========================================================
    // (0) Pre-phase: compute T_SA once BEFORE capture starts
    // ==========================================================
    computeTSAOnceBeforeCapture(exec);

    // Now start capture loop
    enum class State { WAIT_ENTER, IN_REGION };
    State state = State::WAIT_ENTER;

    size_t target_k = 0;
    rclcpp::Time target_start_time = tnow();

    resetMotionDetector();
    bool hold_active = false;
    rclcpp::Time hold_start_time = tnow();

    rclcpp::Rate rate(loop_hz_);

    while (rclcpp::ok() && target_k < target_indices_.size()) {
      exec.spin_some();

      std::array<double,6> cp;
      std::array<double,7> vr;
      rclcpp::Time cp_t, vr_t;
      uint64_t cp_seq = 0;

      if (!getLatestData(cp, vr, cp_t, vr_t, cp_seq)) {
        rate.sleep();
        continue;
      }
      if (!isCpFresh(cp_t)) {
        rate.sleep();
        continue;
      }

      const size_t wp_idx = target_indices_[target_k];
      const auto& target_pose = waypoints_[wp_idx].pose;

      if ((tnow() - target_start_time).seconds() > target_timeout_s_) {
        RCLCPP_WARN(get_logger(),
          "[TIMEOUT] target %zu/%zu wp=%zu",
          target_k+1, target_indices_.size(), wp_idx+1);
        target_k++;
        state = State::WAIT_ENTER;
        target_start_time = tnow();
        resetMotionDetector();
        hold_active = false;
        rate.sleep();
        continue;
      }

      const double dist_mm = posDistMm(cp, target_pose);
      const double ang_deg = oriErrDeg(cp, target_pose);

      if (state == State::WAIT_ENTER) {
        if (dist_mm <= pos_enter_mm_ && ang_deg <= ori_enter_deg_) {
          state = State::IN_REGION;
          hold_active = false;
          resetMotionDetector();
          target_start_time = tnow();
          resetCleanCaptureBuffer();

          RCLCPP_INFO(get_logger(),
            "[IN] target %zu/%zu wp=%zu",
            target_k+1, target_indices_.size(), wp_idx+1);
          RCLCPP_INFO(get_logger(),
            "[IN] d=%.2fmm a=%.2fdeg",
            dist_mm, ang_deg);
        }
        rate.sleep();
        continue;
      }

      if (dist_mm >= pos_exit_mm_ || ang_deg >= ori_exit_deg_) {
        state = State::WAIT_ENTER;
        hold_active = false;
        resetMotionDetector();
        resetCleanCaptureBuffer();
        RCLCPP_WARN(get_logger(),
          "[OUT] d=%.2fmm a=%.2fdeg",
          dist_mm, ang_deg);
        rate.sleep();
        continue;
      }

      if (!(dist_mm <= pos_enter_mm_ && ang_deg <= ori_enter_deg_)) {
        hold_active = false;
        resetCleanCaptureBuffer();
      }

      updateMotionIfNew(cp, cp_t, cp_seq);

      const bool stopped_now = isStoppedNow();
      if (!(dist_mm <= pos_enter_mm_ && ang_deg <= ori_enter_deg_)) {
        hold_active = false;
        resetCleanCaptureBuffer();
      } else if (!stopped_now) {
        hold_active = false;
        resetCleanCaptureBuffer();
      } else {
        if (!hold_active) {
          hold_active = true;
          hold_start_time = tnow();
          resetCleanCaptureBuffer();
        }
      }

      if (hold_active) {
        addCleanCaptureSampleIfNew(cp, vr, cp_t, vr_t, cp_seq, dist_mm, ang_deg);

        const double held = (tnow() - hold_start_time).seconds();
        if (held >= min_hold_time_s_) {

          std::array<double,6> cp_avg;
          std::array<double,7> vr_avg;
          double dist_avg_mm = 0.0;
          double ang_avg_deg = 0.0;
          if (!makeCleanCaptureAverage(target_pose, cp_avg, vr_avg, dist_avg_mm, ang_avg_deg)) {
            if (held < hold_time_s_) {
              RCLCPP_WARN_THROTTLE(
                get_logger(), steady_clock_, 2000,
                "[WAIT] hold=%.1f/%.1fs n=%zu/%zu",
                held, hold_time_s_,
                clean_capture_samples_.size(), capture_min_clean_samples_);
              rate.sleep();
              continue;
            }

            if (!makeBestEffortCaptureAverage(target_pose, cp, vr, cp_t, vr_t, cp_seq,
                                              dist_mm, ang_deg,
                                              cp_avg, vr_avg, dist_avg_mm, ang_avg_deg)) {
              RCLCPP_WARN_THROTTLE(
                get_logger(), steady_clock_, 2000,
                "[WAIT] no sample n=%zu/%zu win=%.3fs",
                clean_capture_samples_.size(), capture_min_clean_samples_,
                cleanCaptureWindowS());
              rate.sleep();
              continue;
            }
          }

          captureOnce(target_k, wp_idx, cp_avg, vr_avg, dist_avg_mm, ang_avg_deg);

          target_k++;
          state = State::WAIT_ENTER;
          target_start_time = tnow();
          resetMotionDetector();
          resetCleanCaptureBuffer();
          hold_active = false;
        }
      }

      rate.sleep();
    }

    RCLCPP_INFO(get_logger(), "[DONE] all targets processed");

    // ==========================================================
    // (final) compute T_BC / T_AD_avg and write YAML ONCE with:
    //         R_Adj, T_AD, T_BC, T_SA, T_CE
    // ==========================================================
    try {
      finalizeCalibrationAndSaveYaml();
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "[ERROR] finalize failed");
      RCLCPP_ERROR(get_logger(), "[ERROR] %s", e.what());
    }
  }

private:
  // ---------- clocks ----------
  mutable rclcpp::Clock steady_clock_;
  rclcpp::Time tnow() const { return steady_clock_.now(); }

  // ---------- capture params ----------
  double pos_enter_mm_{20.0};
  double pos_exit_mm_{60.0};
  double ori_enter_deg_{25.0};
  double ori_exit_deg_{60.0};
  double vel_thresh_mms_{15.0};
  double angvel_thresh_dps_{8.0};
  double hold_time_s_{2.0};
  double min_hold_time_s_{1.5};
  double cp_fresh_s_{1.0};
  double vr_capture_age_s_{30.0};
  double max_capture_sync_dt_s_{0.05};
  double capture_window_s_{0.5};
  size_t capture_min_clean_samples_{20};
  double capture_max_vr_std_mm_{10.0};
  double target_timeout_s_{300.0};
  double loop_hz_{200.0};
  size_t cp_unit_probe_N_{30};

  // ---------- T_SA params ----------
  double t_sa_w_des_z_{1.5707963267948966};
  double t_sa_wait_timeout_s_{15.0};
  double t_sa_hold_s_{0.25};
  double t_sa_fresh_s_{1.0};
  bool radj_enable_{false};
  size_t radj_sample_count_{8};
  bool radj_use_all_samples_{false};
  double max_calib_position_rms_mm_{50.0};
  bool handeye_auto_trim_low_rotation_prefix_{true};
  double handeye_prefix_rotation_span_deg_{5.0};
  size_t handeye_min_trim_prefix_samples_{4};

  // ✅ NEW
  std::string t_sa_mode_{"update"};   // keep/update
  double t_sa_max_delta_deg_{180.0};  // update guard

  // z-plane correction: left-multiplied rigid fix after base calibration
  bool z_fix_enable_{true};
  double z_fix_max_tilt_deg_{5.0};
  bool z_residual_enable_{true};
  double z_residual_max_correction_m_{0.010};

  // ---------- units ----------
  bool wp_rotvec_in_degrees_{false};
  bool cp_pos_unit_decided_{false};
  bool cp_pos_in_meters_{false};
  bool cp_rotvec_unit_decided_{false};
  bool cp_rotvec_in_degrees_{false};
  size_t cp_probe_cnt_{0};
  double cp_probe_max_abs_{0.0};

  // VR position unit auto-detect (meters vs mm)
  bool vr_pos_unit_decided_{false};
  bool vr_pos_in_mm_{false};

  // ---------- waypoints ----------
  std::vector<Waypoint> waypoints_;
  std::vector<size_t> target_indices_;

  // ---------- latest data ----------
  std::mutex mtx_;
  std::array<double,6> last_cp_{};
  std::array<double,7> last_vr_{};
  bool have_cp_{false};
  bool have_vr_{false};
  rclcpp::Time last_cp_time_;
  rclcpp::Time last_vr_time_;
  uint64_t cp_seq_{0};

  // calibrated_pose (for T_SA)
  bool have_cal_pose_{false};
  std::array<double,6> last_cal_pose_{};
  rclcpp::Time last_cal_pose_time_;

  // ---------- motion detector ----------
  bool have_prev_motion_{false};
  std::array<double,6> prev_motion_cp_{};
  rclcpp::Time prev_motion_time_;
  uint64_t prev_motion_seq_{0};
  double last_vnorm_mms_{1e9};
  double last_omega_dps_{1e9};

  struct CaptureSample
  {
    std::array<double,6> cp{};
    std::array<double,7> vr{};
    rclcpp::Time cp_t;
    rclcpp::Time vr_t;
    uint64_t cp_seq{0};
    double dist_mm{0.0};
    double ang_deg{0.0};
    double vnorm_mms{0.0};
    double omega_dps{0.0};
  };

  struct CaptureWindowStats
  {
    double window_s{0.0};
    double vr_std_mm{0.0};
    double avg_v_mms{0.0};
    double avg_w_dps{0.0};
    double avg_dist_mm{0.0};
    double avg_ang_deg{0.0};
    double score{0.0};
  };

  std::vector<CaptureSample> clean_capture_samples_;
  uint64_t last_clean_capture_cp_seq_{0};

  // ---------- ROS ----------
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_currentP_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_vr_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_calibrated_pose_;

  // ---------- files ----------
  std::string waypoint_file_;
  std::string ee_path_, vr_path_;
  std::ofstream ee_ofs_, vr_ofs_;

  // ---------- YAML / calibration outputs ----------
  std::string calib_yaml_path_;

  // existing constants from yaml
  Eigen::Matrix4d T_CE_ = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_SA_old_ = Eigen::Matrix4d::Identity();

  // computed outputs
  bool have_radj_{false};
  Eigen::Matrix3d R_adj_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix4d T_FIX_ = Eigen::Matrix4d::Identity();

  struct ZResidualModel
  {
    bool valid{false};
    double center_x{0.0};
    double center_y{0.0};
    double scale_xy{1.0};
    double max_abs_correction_m{0.012};
    std::array<double,6> coeff{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    double rms_before_m{0.0};
    double rms_after_m{0.0};
  };
  ZResidualModel z_residual_model_;

  bool t_sa_computed_{false};
  Eigen::Matrix4d T_SA_new_ = Eigen::Matrix4d::Identity();

  // store all captured samples for T_BC / T_AD
  std::vector<Eigen::Matrix4d> T_AB_all_; // arm
  std::vector<Eigen::Matrix4d> T_DC_all_; // tracker

  // storage for finalize step
  std::vector<Eigen::Vector3d> O_B0B1_list_;
  std::vector<Eigen::Vector3d> O_C0C1_list_;

  // ---------- callbacks ----------
  void cbCurrentP(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    if (msg->data.size() < 6) return;
    const auto ts = tnow();

    std::lock_guard<std::mutex> lk(mtx_);
    for (int i=0;i<6;i++) last_cp_[i] = msg->data[i];

        if (!cp_pos_unit_decided_) {
      double mabs = 0.0;
      mabs = std::max(mabs, std::fabs(last_cp_[0]));
      mabs = std::max(mabs, std::fabs(last_cp_[1]));
      mabs = std::max(mabs, std::fabs(last_cp_[2]));
      cp_pos_in_meters_ = (mabs < 10.0);
      cp_pos_unit_decided_ = true;
      RCLCPP_INFO(get_logger(),
        "[UNIT] currentP pos=%s max=%.3f",
        cp_pos_in_meters_ ? "M" : "MM", mabs);
    }

    // Internally currentP positions are kept in mm because waypoints are mm.
    if (cp_pos_in_meters_) {
      for (int i=0;i<3;i++) last_cp_[i] *= 1000.0;
    }

    have_cp_ = true;
    last_cp_time_ = ts;
    cp_seq_++;

    // auto-detect unit of currentP rotvec
    if (!cp_rotvec_unit_decided_) {
      cp_probe_cnt_++;
      cp_probe_max_abs_ = std::max(cp_probe_max_abs_, std::fabs(last_cp_[3]));
      cp_probe_max_abs_ = std::max(cp_probe_max_abs_, std::fabs(last_cp_[4]));
      cp_probe_max_abs_ = std::max(cp_probe_max_abs_, std::fabs(last_cp_[5]));

      if (cp_probe_cnt_ >= cp_unit_probe_N_) {
        cp_rotvec_in_degrees_ = (cp_probe_max_abs_ > 6.0);
        cp_rotvec_unit_decided_ = true;
        RCLCPP_INFO(get_logger(),
          "[UNIT] currentP rot=%s max=%.3f n=%zu",
          cp_rotvec_in_degrees_ ? "DEG" : "RAD", cp_probe_max_abs_, cp_probe_cnt_);
      }
    }
  }

  void cbVR(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    const auto ts = tnow();
    std::lock_guard<std::mutex> lk(mtx_);
    last_vr_[0]=msg->pose.position.x;
    last_vr_[1]=msg->pose.position.y;
    last_vr_[2]=msg->pose.position.z;
    last_vr_[3]=msg->pose.orientation.x;
    last_vr_[4]=msg->pose.orientation.y;
    last_vr_[5]=msg->pose.orientation.z;
    last_vr_[6]=msg->pose.orientation.w;
    have_vr_ = true;
    last_vr_time_ = ts;

    if (!vr_pos_unit_decided_) {
      double mabs = 0.0;
      mabs = std::max(mabs, std::fabs(last_vr_[0]));
      mabs = std::max(mabs, std::fabs(last_vr_[1]));
      mabs = std::max(mabs, std::fabs(last_vr_[2]));
      vr_pos_in_mm_ = (mabs > 10.0);
      vr_pos_unit_decided_ = true;
      RCLCPP_INFO(get_logger(),
        "[UNIT] raw_pose pos=%s max=%.3f",
        vr_pos_in_mm_ ? "MM" : "M", mabs);
    }
  }

  void cbCalibratedPose(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    if (msg->data.size() < 6) return;
    const auto ts = tnow();
    std::lock_guard<std::mutex> lk(mtx_);
    for (int i=0;i<6;i++) last_cal_pose_[i] = msg->data[i];
    last_cal_pose_time_ = ts;
    have_cal_pose_ = true;
  }

  // ---------- waypoint parsing ----------
  void loadWaypointsAndDecideWpUnits()
  {
    std::ifstream ifs(waypoint_file_);
    if (!ifs.is_open())
      throw std::runtime_error("Failed to load waypoint file: " + waypoint_file_);

    double max_abs_w = 0.0;
    std::string line;

    while (std::getline(ifs, line)) {
      if (line.empty()) continue;

      std::stringstream ss(line);
      Waypoint wp;

      for (int i=0;i<6;i++) {
        if (!(ss >> wp.pose[i]))
          throw std::runtime_error("Waypoint parse error in line: " + line);
      }

      if (!(ss >> wp.lin_vel >> wp.ang_vel >> wp.holding_time_s))
        throw std::runtime_error("Waypoint tail parse error in line: " + line);

      max_abs_w = std::max(max_abs_w, std::fabs(wp.pose[3]));
      max_abs_w = std::max(max_abs_w, std::fabs(wp.pose[4]));
      max_abs_w = std::max(max_abs_w, std::fabs(wp.pose[5]));

      waypoints_.push_back(wp);
    }

    wp_rotvec_in_degrees_ = (max_abs_w > 6.0);
  }

  void buildTargetIndices()
  {
    target_indices_.clear();
    for (size_t i=0; i<waypoints_.size(); ++i) {
      if (waypoints_[i].holding_time_s > 1e-12)
        target_indices_.push_back(i);
    }
  }

  // ---------- latest data fetch ----------
  bool getLatestData(std::array<double,6>& cp,
                     std::array<double,7>& vr,
                     rclcpp::Time& cp_t,
                     rclcpp::Time& vr_t,
                     uint64_t& cp_seq_out)
  {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!have_cp_) return false;

    cp = last_cp_;
    cp_t = last_cp_time_;
    cp_seq_out = cp_seq_;

    if (have_vr_) {
      vr = last_vr_;
      vr_t = last_vr_time_;
    } else {
      vr_t = rclcpp::Time(0,0,RCL_STEADY_TIME);
    }
    return true;
  }

  bool isCpFresh(const rclcpp::Time& cp_t) const
  {
    return (tnow() - cp_t).seconds() <= cp_fresh_s_;
  }

  bool isVrFreshForCapture(const rclcpp::Time& vr_t) const
  {
    if (!have_vr_) return false;
    return (tnow() - vr_t).seconds() <= vr_capture_age_s_;
  }

  bool isCalPoseFresh() const
  {
    if (!have_cal_pose_) return false;
    return (tnow() - last_cal_pose_time_).seconds() <= t_sa_fresh_s_;
  }

  // ---------- reach metrics ----------
  static double posDistMm(const std::array<double,6>& cp,
                          const std::array<double,6>& target)
  {
    double e2 = 0.0;
    for (int i=0;i<3;i++){
      double d = cp[i] - target[i];
      e2 += d*d;
    }
    return std::sqrt(e2);
  }

  std::array<double,3> toRotvecRad_WP(const std::array<double,6>& pose) const
  {
    std::array<double,3> w = {pose[3], pose[4], pose[5]};
    if (wp_rotvec_in_degrees_) {
      w[0] = deg2rad(w[0]);
      w[1] = deg2rad(w[1]);
      w[2] = deg2rad(w[2]);
    }
    return w;
  }

  std::array<double,3> toRotvecRad_CP(const std::array<double,6>& pose) const
  {
    std::array<double,3> w = {pose[3], pose[4], pose[5]};
    bool cp_deg = cp_rotvec_unit_decided_ ? cp_rotvec_in_degrees_ : false;
    if (cp_deg) {
      w[0] = deg2rad(w[0]);
      w[1] = deg2rad(w[1]);
      w[2] = deg2rad(w[2]);
    }
    return w;
  }

  double oriErrDeg(const std::array<double,6>& cp,
                   const std::array<double,6>& target) const
  {
    const auto w_c = toRotvecRad_CP(cp);
    const auto w_t = toRotvecRad_WP(target);

    std::array<double,9> Rc, Rt;
    rotvecToRotMatRad(w_c, Rc);
    rotvecToRotMatRad(w_t, Rt);

    const double ang_rad = rotAngleBetweenRad(Rt, Rc);
    return rad2deg(ang_rad);
  }

  // ---------- motion detector ----------
  void resetMotionDetector()
  {
    have_prev_motion_ = false;
    prev_motion_seq_  = 0;
    last_vnorm_mms_   = 1e9;
    last_omega_dps_   = 1e9;
  }

  void updateMotionIfNew(const std::array<double,6>& cp,
                         const rclcpp::Time& cp_time,
                         uint64_t cp_seq_in)
  {
    if (cp_seq_in == prev_motion_seq_) return;

    if (!have_prev_motion_) {
      prev_motion_cp_   = cp;
      prev_motion_time_ = cp_time;
      have_prev_motion_ = true;
      prev_motion_seq_  = cp_seq_in;
      return;
    }

    const double dt = (cp_time - prev_motion_time_).seconds();
    if (dt <= 1e-4) {
      prev_motion_cp_   = cp;
      prev_motion_time_ = cp_time;
      prev_motion_seq_  = cp_seq_in;
      return;
    }

    double v2 = 0.0;
    for (int i=0;i<3;i++){
      double v = (cp[i] - prev_motion_cp_[i]) / dt;
      v2 += v*v;
    }
    last_vnorm_mms_ = std::sqrt(v2);

    const auto w_prev = toRotvecRad_CP(prev_motion_cp_);
    const auto w_cur  = toRotvecRad_CP(cp);

    std::array<double,9> Rprev, Rcur;
    rotvecToRotMatRad(w_prev, Rprev);
    rotvecToRotMatRad(w_cur,  Rcur);

    const double dang_rad = rotAngleBetweenRad(Rprev, Rcur);
    last_omega_dps_ = rad2deg(dang_rad) / dt;

    prev_motion_cp_   = cp;
    prev_motion_time_ = cp_time;
    prev_motion_seq_  = cp_seq_in;
  }

  bool isStoppedNow() const
  {
    return (last_vnorm_mms_ <= vel_thresh_mms_) && (last_omega_dps_ <= angvel_thresh_dps_);
  }

  // ---------- clean capture buffer ----------
  void resetCleanCaptureBuffer()
  {
    clean_capture_samples_.clear();
    last_clean_capture_cp_seq_ = 0;
  }

  double captureSyncDtS(const rclcpp::Time& cp_t, const rclcpp::Time& vr_t) const
  {
    return std::fabs((cp_t - vr_t).seconds());
  }

  double cleanCaptureWindowS() const
  {
    if (clean_capture_samples_.size() < 2) return 0.0;
    return (clean_capture_samples_.back().cp_t - clean_capture_samples_.front().cp_t).seconds();
  }

  void addCleanCaptureSampleIfNew(const std::array<double,6>& cp,
                                  const std::array<double,7>& vr,
                                  const rclcpp::Time& cp_t,
                                  const rclcpp::Time& vr_t,
                                  uint64_t cp_seq,
                                  double dist_mm,
                                  double ang_deg)
  {
    if (cp_seq == last_clean_capture_cp_seq_) return;
    last_clean_capture_cp_seq_ = cp_seq;

    if (!isCpFresh(cp_t) || !isVrFreshForCapture(vr_t)) return;
    if (max_capture_sync_dt_s_ > 0.0 && captureSyncDtS(cp_t, vr_t) > max_capture_sync_dt_s_) return;
    if (dist_mm > pos_enter_mm_ || ang_deg > ori_enter_deg_) return;
    if (!isStoppedNow()) return;

    CaptureSample s;
    s.cp = cp;
    s.vr = vr;
    s.cp_t = cp_t;
    s.vr_t = vr_t;
    s.cp_seq = cp_seq;
    s.dist_mm = dist_mm;
    s.ang_deg = ang_deg;
    s.vnorm_mms = last_vnorm_mms_;
    s.omega_dps = last_omega_dps_;
    clean_capture_samples_.push_back(s);
  }

  bool computeCaptureWindowStats(size_t first_idx,
                                 size_t last_idx,
                                 CaptureWindowStats& stats) const
  {
    const size_t N = clean_capture_samples_.size();
    if (N == 0 || first_idx >= N || last_idx >= N || first_idx > last_idx) return false;

    const size_t K = last_idx - first_idx + 1;
    const double invK = 1.0 / static_cast<double>(K);
    stats = CaptureWindowStats{};
    stats.window_s =
      (K >= 2) ? (clean_capture_samples_[last_idx].cp_t - clean_capture_samples_[first_idx].cp_t).seconds() : 0.0;

    Eigen::Vector3d vr_pos_sum_m = Eigen::Vector3d::Zero();
    for (size_t sample_idx = first_idx; sample_idx <= last_idx; ++sample_idx) {
      const auto& s = clean_capture_samples_[sample_idx];
      double vx = s.vr[0], vy = s.vr[1], vz = s.vr[2];
      if (vr_pos_unit_decided_ && vr_pos_in_mm_) {
        vx *= 1e-3; vy *= 1e-3; vz *= 1e-3;
      }
      vr_pos_sum_m += Eigen::Vector3d(vx, vy, vz);
      stats.avg_v_mms += s.vnorm_mms;
      stats.avg_w_dps += s.omega_dps;
      stats.avg_dist_mm += s.dist_mm;
      stats.avg_ang_deg += s.ang_deg;
    }

    const Eigen::Vector3d vr_pos_mean_m = vr_pos_sum_m * invK;
    double vr_var = 0.0;
    for (size_t sample_idx = first_idx; sample_idx <= last_idx; ++sample_idx) {
      const auto& s = clean_capture_samples_[sample_idx];
      double vx = s.vr[0], vy = s.vr[1], vz = s.vr[2];
      if (vr_pos_unit_decided_ && vr_pos_in_mm_) {
        vx *= 1e-3; vy *= 1e-3; vz *= 1e-3;
      }
      vr_var += (Eigen::Vector3d(vx, vy, vz) - vr_pos_mean_m).squaredNorm();
    }

    stats.vr_std_mm = std::sqrt(vr_var * invK) * 1000.0;
    stats.avg_v_mms *= invK;
    stats.avg_w_dps *= invK;
    stats.avg_dist_mm *= invK;
    stats.avg_ang_deg *= invK;

    const double vr_norm = (capture_max_vr_std_mm_ > 0.0) ? stats.vr_std_mm / capture_max_vr_std_mm_ : 0.0;
    const double v_norm = (vel_thresh_mms_ > 0.0) ? stats.avg_v_mms / vel_thresh_mms_ : 0.0;
    const double w_norm = (angvel_thresh_dps_ > 0.0) ? stats.avg_w_dps / angvel_thresh_dps_ : 0.0;
    const double dist_norm = (pos_enter_mm_ > 0.0) ? stats.avg_dist_mm / pos_enter_mm_ : 0.0;
    const double ang_norm = (ori_enter_deg_ > 0.0) ? stats.avg_ang_deg / ori_enter_deg_ : 0.0;

    stats.score =
      3.0 * vr_norm +
      2.0 * v_norm +
      1.5 * w_norm +
      1.0 * dist_norm +
      0.5 * ang_norm;
    return true;
  }

  bool averageCaptureSampleRange(const std::array<double,6>& target_pose,
                                 size_t first_idx,
                                 size_t last_idx,
                                 bool enforce_vr_std,
                                 const char* log_tag,
                                 std::array<double,6>& cp_avg,
                                 std::array<double,7>& vr_avg,
                                 double& dist_avg_mm,
                                 double& ang_avg_deg)
  {
    const size_t N = clean_capture_samples_.size();
    if (N == 0 || first_idx >= N || last_idx >= N || first_idx > last_idx) return false;

    const size_t K = last_idx - first_idx + 1;
    const double avg_window_s =
      (K >= 2) ? (clean_capture_samples_[last_idx].cp_t - clean_capture_samples_[first_idx].cp_t).seconds() : 0.0;

    cp_avg = {0,0,0,0,0,0};
    vr_avg = {0,0,0,0,0,0,1};

    Eigen::Vector3d vr_pos_sum_m = Eigen::Vector3d::Zero();
    Eigen::Vector3d vr_pos_sum_raw = Eigen::Vector3d::Zero();
    Eigen::Vector4d q_sum = Eigen::Vector4d::Zero(); // coeffs: x,y,z,w
    bool have_q_ref = false;
    Eigen::Quaterniond q_ref(1,0,0,0);

    for (size_t sample_idx = first_idx; sample_idx <= last_idx; ++sample_idx) {
      const auto& s = clean_capture_samples_[sample_idx];
      for (int i=0; i<6; ++i) cp_avg[i] += s.cp[i];

      double vx = s.vr[0], vy = s.vr[1], vz = s.vr[2];
      if (vr_pos_unit_decided_ && vr_pos_in_mm_) {
        vx *= 1e-3; vy *= 1e-3; vz *= 1e-3;
      }
      vr_pos_sum_m += Eigen::Vector3d(vx, vy, vz);
      vr_pos_sum_raw += Eigen::Vector3d(s.vr[0], s.vr[1], s.vr[2]);

      Eigen::Quaterniond q(s.vr[6], s.vr[3], s.vr[4], s.vr[5]);
      q.normalize();
      if (!have_q_ref) {
        q_ref = q;
        have_q_ref = true;
      }
      if (q_ref.coeffs().dot(q.coeffs()) < 0.0) {
        q.coeffs() *= -1.0;
      }
      q_sum += q.coeffs();
    }

    const double invN = 1.0 / static_cast<double>(K);
    for (int i=0; i<6; ++i) cp_avg[i] *= invN;
    dist_avg_mm = posDistMm(cp_avg, target_pose);
    ang_avg_deg = oriErrDeg(cp_avg, target_pose);

    const Eigen::Vector3d vr_pos_mean_m = vr_pos_sum_m * invN;
    const Eigen::Vector3d vr_pos_mean_raw = vr_pos_sum_raw * invN;
    double vr_var = 0.0;
    for (size_t sample_idx = first_idx; sample_idx <= last_idx; ++sample_idx) {
      const auto& s = clean_capture_samples_[sample_idx];
      double vx = s.vr[0], vy = s.vr[1], vz = s.vr[2];
      if (vr_pos_unit_decided_ && vr_pos_in_mm_) {
        vx *= 1e-3; vy *= 1e-3; vz *= 1e-3;
      }
      vr_var += (Eigen::Vector3d(vx, vy, vz) - vr_pos_mean_m).squaredNorm();
    }
    const double vr_std_mm = std::sqrt(vr_var * invN) * 1000.0;
    if (enforce_vr_std && capture_max_vr_std_mm_ > 0.0 && vr_std_mm > capture_max_vr_std_mm_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), steady_clock_, 2000,
        "[WAIT] vr_std %.2f > %.2fmm",
        vr_std_mm, capture_max_vr_std_mm_);
      return false;
    }

    Eigen::Quaterniond q_avg;
    q_avg.coeffs() = q_sum * invN;
    q_avg.normalize();

    vr_avg[0] = vr_pos_mean_raw.x();
    vr_avg[1] = vr_pos_mean_raw.y();
    vr_avg[2] = vr_pos_mean_raw.z();
    vr_avg[3] = q_avg.x();
    vr_avg[4] = q_avg.y();
    vr_avg[5] = q_avg.z();
    vr_avg[6] = q_avg.w();

    RCLCPP_INFO(get_logger(),
      "[%s] n=%zu win=%.3fs",
      log_tag, K, avg_window_s);
    RCLCPP_INFO(get_logger(),
      "[%s] d=%.2fmm a=%.2f std=%.2f",
      log_tag, dist_avg_mm, ang_avg_deg, vr_std_mm);
    return true;
  }

  bool makeCleanCaptureAverage(const std::array<double,6>& target_pose,
                               std::array<double,6>& cp_avg,
                               std::array<double,7>& vr_avg,
                               double& dist_avg_mm,
                               double& ang_avg_deg)
  {
    const size_t N = clean_capture_samples_.size();
    if (N < capture_min_clean_samples_) return false;
    if (cleanCaptureWindowS() < capture_window_s_) return false;

    bool have_best = false;
    size_t best_first_idx = 0;
    size_t best_last_idx = 0;
    CaptureWindowStats best_stats;

    for (size_t last_idx = capture_min_clean_samples_ - 1; last_idx < N; ++last_idx) {
      size_t first_idx = last_idx;
      while (first_idx > 0 &&
             (clean_capture_samples_[last_idx].cp_t - clean_capture_samples_[first_idx].cp_t).seconds() < capture_window_s_) {
        --first_idx;
      }

      const size_t K = last_idx - first_idx + 1;
      if (K < capture_min_clean_samples_) continue;
      const double window_s =
        (clean_capture_samples_[last_idx].cp_t - clean_capture_samples_[first_idx].cp_t).seconds();
      if (window_s < capture_window_s_) continue;

      CaptureWindowStats stats;
      if (!computeCaptureWindowStats(first_idx, last_idx, stats)) continue;
      if (capture_max_vr_std_mm_ > 0.0 && stats.vr_std_mm > capture_max_vr_std_mm_) continue;

      if (!have_best || stats.score < best_stats.score) {
        have_best = true;
        best_first_idx = first_idx;
        best_last_idx = last_idx;
        best_stats = stats;
      }
    }

    if (!have_best) return false;

    RCLCPP_INFO(get_logger(),
      "[BEST] n=%zu win=%.3fs score=%.3f",
      best_last_idx - best_first_idx + 1,
      best_stats.window_s,
      best_stats.score);
    RCLCPP_INFO(get_logger(),
      "[BEST] std=%.2f v=%.2f w=%.2f",
      best_stats.vr_std_mm,
      best_stats.avg_v_mms,
      best_stats.avg_w_dps);
    RCLCPP_INFO(get_logger(),
      "[BEST] d=%.2fmm a=%.2fdeg",
      best_stats.avg_dist_mm,
      best_stats.avg_ang_deg);

    return averageCaptureSampleRange(target_pose, best_first_idx, best_last_idx, true, "CLEAN_CAPTURE",
                                     cp_avg, vr_avg, dist_avg_mm, ang_avg_deg);
  }

  bool makeBestEffortCaptureAverage(const std::array<double,6>& target_pose,
                                    const std::array<double,6>& cp,
                                    const std::array<double,7>& vr,
                                    const rclcpp::Time& cp_t,
                                    const rclcpp::Time& vr_t,
                                    uint64_t cp_seq,
                                    double dist_mm,
                                    double ang_deg,
                                    std::array<double,6>& cp_avg,
                                    std::array<double,7>& vr_avg,
                                    double& dist_avg_mm,
                                    double& ang_avg_deg)
  {
    if (!clean_capture_samples_.empty()) {
      RCLCPP_WARN(get_logger(),
        "[FAST] fallback n=%zu win=%.3fs",
        clean_capture_samples_.size(), cleanCaptureWindowS());
      return averageCaptureSampleRange(target_pose, 0, clean_capture_samples_.size() - 1,
                                       false, "FAST_CAPTURE",
                                       cp_avg, vr_avg, dist_avg_mm, ang_avg_deg);
    }

    if (!isCpFresh(cp_t) || !isVrFreshForCapture(vr_t)) return false;
    if (max_capture_sync_dt_s_ > 0.0 && captureSyncDtS(cp_t, vr_t) > max_capture_sync_dt_s_) return false;
    if (dist_mm > pos_enter_mm_ || ang_deg > ori_enter_deg_) return false;
    if (!isStoppedNow()) return false;

    CaptureSample s;
    s.cp = cp;
    s.vr = vr;
    s.cp_t = cp_t;
    s.vr_t = vr_t;
    s.cp_seq = cp_seq;
    s.dist_mm = dist_mm;
    s.ang_deg = ang_deg;
    s.vnorm_mms = last_vnorm_mms_;
    s.omega_dps = last_omega_dps_;
    clean_capture_samples_.push_back(s);

    RCLCPP_WARN(get_logger(),
      "[FAST] fallback latest sample");
    return averageCaptureSampleRange(target_pose, 0, 0, false, "FAST_CAPTURE",
                                     cp_avg, vr_avg, dist_avg_mm, ang_avg_deg);
  }

  // ==========================================================
  // T_SA computation (right-multiply)
  // ==========================================================
  bool computeTSAFromLatestCalPose()
  {
    if (!have_cal_pose_ || !isCalPoseFresh()) return false;

    // w_meas (rad) from /calibrated_pose: [x y z wx wy wz]
    std::array<double,3> w_meas = { last_cal_pose_[3], last_cal_pose_[4], last_cal_pose_[5] };

    // R_total from w_meas
    std::array<double,9> Rtot_arr;
    rotvecToRotMatRad(w_meas, Rtot_arr);
    Eigen::Matrix3d R_total;
    R_total << Rtot_arr[0], Rtot_arr[1], Rtot_arr[2],
               Rtot_arr[3], Rtot_arr[4], Rtot_arr[5],
               Rtot_arr[6], Rtot_arr[7], Rtot_arr[8];

    // desired R_des from w_des=[0,0,t_sa_w_des_z_]
    std::array<double,3> w_des = {0.0, 0.0, t_sa_w_des_z_};
    std::array<double,9> Rdes_arr;
    rotvecToRotMatRad(w_des, Rdes_arr);
    Eigen::Matrix3d R_des;
    R_des << Rdes_arr[0], Rdes_arr[1], Rdes_arr[2],
             Rdes_arr[3], Rdes_arr[4], Rdes_arr[5],
             Rdes_arr[6], Rdes_arr[7], Rdes_arr[8];

    const Eigen::Matrix3d R_SA_old = T_SA_old_.block<3,3>(0,0);
    const Eigen::Matrix3d R_SA_new = R_SA_old * R_total.transpose() * R_des;

    // ✅ guard: old->new 변화가 너무 크면 reject
    const double delta_deg = rotDiffAngleDeg(R_SA_old, R_SA_new);
    if (delta_deg > t_sa_max_delta_deg_) {
      RCLCPP_WARN(get_logger(),
        "[T_SA_REJECT] d=%.2f limit=%.2f",
        delta_deg, t_sa_max_delta_deg_);
      return false;
    }

    T_SA_new_ = Eigen::Matrix4d::Identity();
    T_SA_new_.block<3,3>(0,0) = R_SA_new;

    t_sa_computed_ = true;

    RCLCPP_INFO(get_logger(),
      "[T_SA_DONE] delta=%.2fdeg",
      delta_deg);
    RCLCPP_INFO(get_logger(),
      "[T_SA_W] meas=[%.4f %.4f %.4f]",
      w_meas[0], w_meas[1], w_meas[2]);
    RCLCPP_INFO(get_logger(),
      "[T_SA_W] des_z=%.4f",
      t_sa_w_des_z_);

    return true;
  }

  void computeTSAOnceBeforeCapture(rclcpp::executors::SingleThreadedExecutor& exec)
  {
    // ✅ keep 모드면: 기존 YAML의 T_SA를 그대로 유지
    if (t_sa_mode_ == "keep") {
      t_sa_computed_ = false;          // "새로 계산"은 안 함
      T_SA_new_ = Eigen::Matrix4d::Identity();
      RCLCPP_INFO(get_logger(), "[T_SA] keep yaml value");
      return;
    }

    // update 모드
    RCLCPP_INFO(get_logger(),
      "[T_SA] wait cal_pose %.2fs",
      t_sa_fresh_s_);
    RCLCPP_INFO(get_logger(),
      "[T_SA] stop hold %.2fs timeout %.1fs",
      t_sa_hold_s_, t_sa_wait_timeout_s_);

    const rclcpp::Time t0 = tnow();
    bool hold_active = false;
    rclcpp::Time hold_start = tnow();

    resetMotionDetector();

    rclcpp::Rate rate(std::min(200.0, loop_hz_));

    while (rclcpp::ok() && (tnow() - t0).seconds() < t_sa_wait_timeout_s_) {
      exec.spin_some();

      std::array<double,6> cp;
      std::array<double,7> vr;
      rclcpp::Time cp_t, vr_t;
      uint64_t cp_seq = 0;

      if (!getLatestData(cp, vr, cp_t, vr_t, cp_seq)) {
        rate.sleep();
        continue;
      }
      if (!isCpFresh(cp_t)) {
        rate.sleep();
        continue;
      }

      updateMotionIfNew(cp, cp_t, cp_seq);

      const bool stopped_now = isStoppedNow();
      if (!stopped_now) {
        hold_active = false;
        rate.sleep();
        continue;
      }

      if (!isCalPoseFresh()) {
        hold_active = false;
        rate.sleep();
        continue;
      }

      if (!hold_active) {
        hold_active = true;
        hold_start = tnow();
      }

      const double held = (tnow() - hold_start).seconds();
      if (held >= t_sa_hold_s_) {
        if (computeTSAFromLatestCalPose()) {
          RCLCPP_INFO(get_logger(), "[T_SA] pre-capture done");
          return;
        } else {
          // reject 등으로 실패하면 계속 기다려서 다시 시도
        }
      }

      rate.sleep();
    }

    // timeout: update 실패 -> old 유지
    RCLCPP_WARN(get_logger(),
      "[T_SA_WARN] timeout; keep yaml");
    t_sa_computed_ = false;
  }

  // ---------- capture ----------
  void captureOnce(size_t target_k, size_t wp_idx,
                   const std::array<double,6>& cp,
                   const std::array<double,7>& vr,
                   double dist_mm, double ang_deg)
  {
    // --- EE rotation from rotvec ---
    const auto w_c = toRotvecRad_CP(cp);
    std::array<double,9> Rarr;
    rotvecToRotMatRad(w_c, Rarr);

    // --- write files (meters) ---
    const double cp_x_m = cp[0] * 1e-3;
    const double cp_y_m = cp[1] * 1e-3;
    const double cp_z_m = cp[2] * 1e-3;

    double vr_x = vr[0], vr_y = vr[1], vr_z = vr[2];
    if (vr_pos_unit_decided_ && vr_pos_in_mm_) {
      vr_x *= 1e-3; vr_y *= 1e-3; vr_z *= 1e-3;
    }

    ee_ofs_
      << Rarr[0]<<" "<<Rarr[1]<<" "<<Rarr[2]<<" "<<cp_x_m<<" "
      << Rarr[3]<<" "<<Rarr[4]<<" "<<Rarr[5]<<" "<<cp_y_m<<" "
      << Rarr[6]<<" "<<Rarr[7]<<" "<<Rarr[8]<<" "<<cp_z_m<<"\n";

    vr_ofs_
      << vr_x<<" "<<vr_y<<" "<<vr_z<<" "
      << vr[3]<<" "<<vr[4]<<" "<<vr[5]<<" "<<vr[6]<<"\n";

    ee_ofs_.flush();
    vr_ofs_.flush();

    // --- store samples for calibration (meters) ---
    Eigen::Matrix3d R_ab;
    R_ab << Rarr[0], Rarr[1], Rarr[2],
            Rarr[3], Rarr[4], Rarr[5],
            Rarr[6], Rarr[7], Rarr[8];
    Eigen::Vector3d p_ab(cp_x_m, cp_y_m, cp_z_m);
    T_AB_all_.push_back(makeT(R_ab, p_ab));

    // VR transform from quaternion (ROS order: x y z w)
    Eigen::Quaterniond q_vr(vr[6], vr[3], vr[4], vr[5]); // ctor: (w,x,y,z)
    q_vr.normalize();
    Eigen::Matrix3d R_dc = q_vr.toRotationMatrix();
    Eigen::Vector3d p_dc(vr_x, vr_y, vr_z);
    T_DC_all_.push_back(makeT(R_dc, p_dc));

    RCLCPP_INFO(get_logger(),
      "[CAPTURE] target %zu/%zu wp=%zu",
      target_k+1, target_indices_.size(), wp_idx+1);
    RCLCPP_INFO(get_logger(),
      "[CAPTURE] d=%.2fmm a=%.2fdeg",
      dist_mm, ang_deg);
    RCLCPP_INFO(get_logger(),
      "[CAPTURE] v=%.2fmm/s w=%.2fdeg/s",
      last_vnorm_mms_, last_omega_dps_);
  }

  // ---------- YAML read (constants) ----------
  void loadExistingYamlConstants()
  {
    // defaults
    T_CE_ = Eigen::Matrix4d::Identity();
    T_CE_(2,3) = 0.222;

    T_SA_old_ = Eigen::Matrix4d::Identity();

    try {
      YAML::Node existing = YAML::LoadFile(calib_yaml_path_);

      if (existing["T_CE"]) {
        Eigen::Matrix4d tmp = Eigen::Matrix4d::Identity();
        if (readMat4(existing["T_CE"], tmp)) {
          T_CE_ = tmp;
          RCLCPP_INFO(get_logger(), "[YAML] Loaded existing T_CE.");
        }
      }

      if (existing["T_SA"]) {
        Eigen::Matrix4d tmp = Eigen::Matrix4d::Identity();
        if (readMat4(existing["T_SA"], tmp)) {
          T_SA_old_ = tmp;
          RCLCPP_INFO(get_logger(), "[YAML] Loaded existing T_SA (old).");
        }
      } else {
        RCLCPP_WARN(get_logger(), "[YAML] T_SA missing; use I");
      }

    } catch (...) {
      RCLCPP_WARN(get_logger(), "[YAML] load failed; create new");
      RCLCPP_WARN(get_logger(), "[YAML] path=%s", calib_yaml_path_.c_str());
    }
  }

  bool readMat4(const YAML::Node& n, Eigen::Matrix4d& T)
  {
    if (!n || !n.IsSequence() || n.size() != 4) return false;
    for (int r=0;r<4;r++){
      if (!n[r].IsSequence() || n[r].size() != 4) return false;
      for (int c=0;c<4;c++){
        T(r,c) = n[r][c].as<double>();
      }
    }
    return true;
  }

  bool readMat3(const YAML::Node& n, Eigen::Matrix3d& R)
  {
    if (!n || !n.IsSequence() || n.size() != 3) return false;
    for (int r=0;r<3;r++){
      if (!n[r].IsSequence() || n[r].size() != 3) return false;
      for (int c=0;c<3;c++){
        R(r,c) = n[r][c].as<double>();
      }
    }
    return true;
  }

  bool computeRAdjFromSamples()
  {
    const size_t N_all = T_AB_all_.size();
    if (N_all < 3 || T_DC_all_.size() != N_all) {
      RCLCPP_WARN(get_logger(), "[R_ADJ] need >=3; use I");
      R_adj_ = Eigen::Matrix3d::Identity();
      have_radj_ = false;
      return false;
    }

    const size_t N = radj_use_all_samples_
      ? N_all
      : std::min(radj_sample_count_, N_all);
    Eigen::Vector3d p_arm_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d p_vr_mean = Eigen::Vector3d::Zero();

    for (size_t i=0; i<N; ++i) {
      p_arm_mean += T_AB_all_[i].block<3,1>(0,3);
      p_vr_mean += T_DC_all_[i].block<3,1>(0,3);
    }
    p_arm_mean /= static_cast<double>(N);
    p_vr_mean /= static_cast<double>(N);

    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (size_t i=0; i<N; ++i) {
      const Eigen::Vector3d a = T_AB_all_[i].block<3,1>(0,3) - p_arm_mean;
      const Eigen::Vector3d v = T_DC_all_[i].block<3,1>(0,3) - p_vr_mean;
      H += v * a.transpose();
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.info() != Eigen::Success) {
      RCLCPP_WARN(get_logger(), "[R_ADJ] SVD failed; use I");
      R_adj_ = Eigen::Matrix3d::Identity();
      have_radj_ = false;
      return false;
    }

    Eigen::Matrix3d R_vr_to_arm = svd.matrixV() * svd.matrixU().transpose();
    if (R_vr_to_arm.determinant() < 0.0) {
      Eigen::Matrix3d V = svd.matrixV();
      V.col(2) *= -1.0;
      R_vr_to_arm = V * svd.matrixU().transpose();
    }

    double rms = 0.0;
    for (size_t i=0; i<N; ++i) {
      const Eigen::Vector3d a = T_AB_all_[i].block<3,1>(0,3) - p_arm_mean;
      const Eigen::Vector3d v = T_DC_all_[i].block<3,1>(0,3) - p_vr_mean;
      const Eigen::Vector3d e = a - R_vr_to_arm * v;
      rms += e.squaredNorm();
    }
    rms = std::sqrt(rms / static_cast<double>(N));

    // Runtime applies T_Adj = R_Adj.transpose() before the base calibration.
    R_adj_ = R_vr_to_arm.transpose();
    have_radj_ = true;

    RCLCPP_INFO(get_logger(),
      "[R_ADJ_DONE] n=%zu/%zu rms=%.3fmm\n"
      "R_Adj=\n"
      "[% .6f % .6f % .6f]\n"
      "[% .6f % .6f % .6f]\n"
      "[% .6f % .6f % .6f]",
      N, N_all, rms * 1000.0,
      R_adj_(0,0), R_adj_(0,1), R_adj_(0,2),
      R_adj_(1,0), R_adj_(1,1), R_adj_(1,2),
      R_adj_(2,0), R_adj_(2,1), R_adj_(2,2));

    return true;
  }

  Eigen::Matrix4d computeZPlaneFix(const Eigen::Matrix4d& T_AD,
                                   const Eigen::Matrix3d& R_Adj,
                                   const Eigen::Matrix4d& T_BC)
  {
    Eigen::Matrix4d T_fix = Eigen::Matrix4d::Identity();
    const size_t N = T_AB_all_.size();
    if (!z_fix_enable_) {
      RCLCPP_INFO(get_logger(), "[T_FIX] disabled; save I");
      return T_fix;
    }
    if (N < 3 || T_DC_all_.size() != N) {
      RCLCPP_WARN(get_logger(), "[T_FIX] need >=3; save I");
      return T_fix;
    }

    Eigen::Matrix4d T_Adj = Eigen::Matrix4d::Identity();
    T_Adj.block<3,3>(0,0) = R_Adj.transpose();
    const Eigen::Matrix4d T_CB = invT(T_BC);

    Eigen::MatrixXd A(static_cast<Eigen::Index>(N), 3);
    Eigen::VectorXd b(static_cast<Eigen::Index>(N));
    std::vector<Eigen::Vector3d> p_cal_list;
    std::vector<double> z_ref_list;
    p_cal_list.reserve(N);
    z_ref_list.reserve(N);

    double rms_before = 0.0;
    for (size_t i=0; i<N; ++i) {
      const Eigen::Matrix4d M_cal = T_AD * T_Adj * T_DC_all_[i] * T_CB;
      const Eigen::Vector3d p_cal = M_cal.block<3,1>(0,3);
      const double z_ref = T_AB_all_[i](2,3);
      const double dz = z_ref - p_cal.z();

      A(static_cast<Eigen::Index>(i), 0) = p_cal.x();
      A(static_cast<Eigen::Index>(i), 1) = p_cal.y();
      A(static_cast<Eigen::Index>(i), 2) = 1.0;
      b(static_cast<Eigen::Index>(i)) = dz;

      p_cal_list.push_back(p_cal);
      z_ref_list.push_back(z_ref);
      rms_before += dz * dz;
    }
    rms_before = std::sqrt(rms_before / static_cast<double>(N));

    const Eigen::Vector3d coeff = A.colPivHouseholderQr().solve(b);
    double rx = coeff.y();       // dz ~= rx*y - ry*x + tz
    double ry = -coeff.x();

    const double tilt = std::sqrt(rx*rx + ry*ry);
    const double max_tilt = deg2rad(std::max(0.0, z_fix_max_tilt_deg_));
    if (tilt > max_tilt && tilt > 1e-12) {
      const double scale = max_tilt / tilt;
      RCLCPP_WARN(get_logger(),
        "[T_FIX] tilt %.3f > %.3fdeg",
        rad2deg(tilt), z_fix_max_tilt_deg_);
      rx *= scale;
      ry *= scale;
    }

    const Eigen::Matrix3d R_fix =
      (Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX()) *
       Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY())).toRotationMatrix();

    double tz = 0.0;
    for (size_t i=0; i<N; ++i) {
      tz += z_ref_list[i] - (R_fix * p_cal_list[i]).z();
    }
    tz /= static_cast<double>(N);

    double rms_after = 0.0;
    for (size_t i=0; i<N; ++i) {
      const double dz_after = z_ref_list[i] - ((R_fix * p_cal_list[i]).z() + tz);
      rms_after += dz_after * dz_after;
    }
    rms_after = std::sqrt(rms_after / static_cast<double>(N));

    T_fix.block<3,3>(0,0) = R_fix;
    T_fix(2,3) = tz;

    RCLCPP_INFO(get_logger(),
      "[T_FIX] rx=%.4f ry=%.4f tz=%.3fmm",
      rad2deg(rx), rad2deg(ry), tz * 1000.0);
    RCLCPP_INFO(get_logger(),
      "[T_FIX] z_rms %.3f -> %.3fmm",
      rms_before * 1000.0, rms_after * 1000.0);

    return T_fix;
  }

  double evalZResidualCorrectionM(const ZResidualModel& model, double x, double y) const
  {
    if (!model.valid) return 0.0;
    const double s = std::max(1e-6, model.scale_xy);
    const double xn = (x - model.center_x) / s;
    const double yn = (y - model.center_y) / s;
    const auto& c = model.coeff;
    const double dz =
      c[0] + c[1] * xn + c[2] * yn + c[3] * xn * xn + c[4] * xn * yn + c[5] * yn * yn;
    return clampd(dz, -model.max_abs_correction_m, model.max_abs_correction_m);
  }

  Eigen::Vector3d applyZResidualToPoint(const Eigen::Vector3d& p) const
  {
    Eigen::Vector3d out = p;
    out.z() += evalZResidualCorrectionM(z_residual_model_, p.x(), p.y());
    return out;
  }

  ZResidualModel computeZResidualModel(const Eigen::Matrix4d& T_AD,
                                       const Eigen::Matrix3d& R_Adj,
                                       const Eigen::Matrix4d& T_BC,
                                       const Eigen::Matrix4d& T_FIX)
  {
    ZResidualModel model;
    model.max_abs_correction_m = z_residual_max_correction_m_;
    const size_t N = T_AB_all_.size();
    if (!z_residual_enable_) {
      RCLCPP_INFO(get_logger(), "[Z_RES] disabled");
      return model;
    }
    if (N < 6 || T_DC_all_.size() != N) {
      RCLCPP_WARN(get_logger(), "[Z_RES] need >=6; disabled");
      return model;
    }
    if (model.max_abs_correction_m <= 0.0) {
      RCLCPP_WARN(get_logger(), "[Z_RES] max correction zero");
      return model;
    }

    Eigen::Matrix4d T_Adj = Eigen::Matrix4d::Identity();
    T_Adj.block<3,3>(0,0) = R_Adj.transpose();
    const Eigen::Matrix4d T_CB = invT(T_BC);

    std::vector<Eigen::Vector3d> p_list;
    std::vector<double> dz_list;
    p_list.reserve(N);
    dz_list.reserve(N);

    Eigen::Vector2d xy_mean = Eigen::Vector2d::Zero();
    for (size_t i=0; i<N; ++i) {
      const Eigen::Matrix4d M_cal = T_FIX * T_AD * T_Adj * T_DC_all_[i] * T_CB;
      const Eigen::Vector3d p_cal = M_cal.block<3,1>(0,3);
      const double z_ref = T_AB_all_[i](2,3);
      p_list.push_back(p_cal);
      dz_list.push_back(z_ref - p_cal.z());
      xy_mean += p_cal.head<2>();
    }
    xy_mean /= static_cast<double>(N);

    double max_radius = 0.0;
    for (const auto& p : p_list) {
      max_radius = std::max(max_radius, (p.head<2>() - xy_mean).norm());
    }
    model.center_x = xy_mean.x();
    model.center_y = xy_mean.y();
    model.scale_xy = std::max(0.05, max_radius);

    Eigen::MatrixXd A(static_cast<Eigen::Index>(N), 6);
    Eigen::VectorXd b(static_cast<Eigen::Index>(N));
    for (size_t i=0; i<N; ++i) {
      const double xn = (p_list[i].x() - model.center_x) / model.scale_xy;
      const double yn = (p_list[i].y() - model.center_y) / model.scale_xy;
      A(static_cast<Eigen::Index>(i), 0) = 1.0;
      A(static_cast<Eigen::Index>(i), 1) = xn;
      A(static_cast<Eigen::Index>(i), 2) = yn;
      A(static_cast<Eigen::Index>(i), 3) = xn * xn;
      A(static_cast<Eigen::Index>(i), 4) = xn * yn;
      A(static_cast<Eigen::Index>(i), 5) = yn * yn;
      b(static_cast<Eigen::Index>(i)) = dz_list[i];
    }

    const Eigen::VectorXd coeff = A.colPivHouseholderQr().solve(b);
    if (coeff.size() != 6 || !coeff.allFinite()) {
      RCLCPP_WARN(get_logger(), "[Z_RES] solve failed");
      return model;
    }
    for (int i=0; i<6; ++i) model.coeff[static_cast<size_t>(i)] = coeff(i);
    model.valid = true;

    double max_abs_fit = 0.0;
    double rms_before = 0.0;
    for (size_t i=0; i<N; ++i) {
      const double dz_fit = evalZResidualCorrectionM(model, p_list[i].x(), p_list[i].y());
      max_abs_fit = std::max(max_abs_fit, std::fabs(dz_fit));
      rms_before += dz_list[i] * dz_list[i];
    }
    if (max_abs_fit > model.max_abs_correction_m && max_abs_fit > 1e-12) {
      const double scale = model.max_abs_correction_m / max_abs_fit;
      for (double& c : model.coeff) c *= scale;
      RCLCPP_WARN(get_logger(),
        "[Z_RES] fit %.3f > clamp %.3fmm",
        max_abs_fit * 1000.0, model.max_abs_correction_m * 1000.0);
    }

    double rms_after = 0.0;
    double max_after = 0.0;
    for (size_t i=0; i<N; ++i) {
      const double dz_after =
        dz_list[i] - evalZResidualCorrectionM(model, p_list[i].x(), p_list[i].y());
      rms_after += dz_after * dz_after;
      max_after = std::max(max_after, std::fabs(dz_after));
    }
    model.rms_before_m = std::sqrt(rms_before / static_cast<double>(N));
    model.rms_after_m = std::sqrt(rms_after / static_cast<double>(N));

    if (model.rms_after_m >= model.rms_before_m) {
      RCLCPP_WARN(get_logger(),
        "[Z_RES] no gain %.3f -> %.3fmm",
        model.rms_before_m * 1000.0, model.rms_after_m * 1000.0);
      model.valid = false;
      return model;
    }

    RCLCPP_INFO(get_logger(),
      "[Z_RES] z_rms %.3f -> %.3fmm",
      model.rms_before_m * 1000.0,
      model.rms_after_m * 1000.0);
    RCLCPP_INFO(get_logger(),
      "[Z_RES] max=%.3fmm clamp=%.1fmm",
      max_after * 1000.0,
      model.max_abs_correction_m * 1000.0);
    return model;
  }

  double validateCalibrationFitMm(const Eigen::Matrix4d& T_AD,
                                  const Eigen::Matrix3d& R_Adj,
                                  const Eigen::Matrix4d& T_BC,
                                  const Eigen::Matrix4d& T_FIX)
  {
    const size_t N = T_AB_all_.size();
    if (N == 0 || T_DC_all_.size() != N) {
      throw std::runtime_error("No samples available for calibration validation.");
    }

    Eigen::Matrix4d T_Adj = Eigen::Matrix4d::Identity();
    T_Adj.block<3,3>(0,0) = R_Adj.transpose();
    const Eigen::Matrix4d T_CB = invT(T_BC);

    double sum2 = 0.0;
    double max_err = 0.0;
    size_t max_i = 0;
    Eigen::Vector3d max_p_ref = Eigen::Vector3d::Zero();
    Eigen::Vector3d max_p_cal = Eigen::Vector3d::Zero();

    for (size_t i=0; i<N; ++i) {
      const Eigen::Matrix4d M_cal = T_FIX * T_AD * T_Adj * T_DC_all_[i] * T_CB;
      const Eigen::Vector3d p_cal = applyZResidualToPoint(M_cal.block<3,1>(0,3));
      const Eigen::Vector3d p_ref = T_AB_all_[i].block<3,1>(0,3);
      const double err = (p_ref - p_cal).norm();
      sum2 += err * err;
      if (err > max_err) {
        max_err = err;
        max_i = i;
        max_p_ref = p_ref;
        max_p_cal = p_cal;
      }
    }

    const double rms_mm = std::sqrt(sum2 / static_cast<double>(N)) * 1000.0;
    RCLCPP_INFO(get_logger(),
      "[VALID] rms=%.3fmm max=%.3fmm",
      rms_mm, max_err * 1000.0);
    RCLCPP_INFO(get_logger(),
      "[VALID] max_sample=%zu/%zu",
      max_i + 1, N);
    RCLCPP_INFO(get_logger(),
      "[VALID] max_ref=[%.1f %.1f %.1f]mm cal=[%.1f %.1f %.1f]mm err=[%.1f %.1f %.1f]mm",
      max_p_ref.x() * 1000.0, max_p_ref.y() * 1000.0, max_p_ref.z() * 1000.0,
      max_p_cal.x() * 1000.0, max_p_cal.y() * 1000.0, max_p_cal.z() * 1000.0,
      (max_p_ref.x() - max_p_cal.x()) * 1000.0,
      (max_p_ref.y() - max_p_cal.y()) * 1000.0,
      (max_p_ref.z() - max_p_cal.z()) * 1000.0);

    if (rms_mm > max_calib_position_rms_mm_) {
      std::ostringstream oss;
      oss << "Calibration rejected: position RMS " << std::fixed << std::setprecision(3)
          << rms_mm << "mm exceeds limit " << max_calib_position_rms_mm_
          << "mm. YAML was not overwritten.";
      throw std::runtime_error(oss.str());
    }

    return rms_mm;
  }

  size_t chooseHandeyeSolveStartIndex() const
  {
    const size_t N = T_AB_all_.size();
    if (!handeye_auto_trim_low_rotation_prefix_ || N < 3) return 0;

    const double span_thresh_rad = deg2rad(handeye_prefix_rotation_span_deg_);
    if (span_thresh_rad <= 0.0) return 0;

    const Eigen::Matrix3d R0 = T_AB_all_.front().block<3,3>(0,0);
    for (size_t i = 1; i < N; ++i) {
      const Eigen::Matrix3d Ri = T_AB_all_[i].block<3,3>(0,0);
      const double span_rad = rotAngleBetweenRad(R0, Ri);
      if (span_rad >= span_thresh_rad) {
        const size_t prefix_count = i;
        const size_t suffix_count = N - i;
        if (prefix_count >= handeye_min_trim_prefix_samples_ && suffix_count >= 3) {
          RCLCPP_WARN(get_logger(),
            "[HAND_EYE] trim low-rotation prefix samples 1..%zu; solve uses %zu..%zu",
            prefix_count, i + 1, N);
          RCLCPP_WARN(get_logger(),
            "[HAND_EYE] prefix span reached %.2fdeg at sample %zu",
            rad2deg(span_rad), i + 1);
          return i;
        }
        return 0;
      }
    }

    return 0;
  }

  // ---------- compute T_BC / T_AD_avg and save yaml ----------
  void finalizeCalibrationAndSaveYaml()
  {
    const size_t N_all = T_AB_all_.size();
    if (N_all < 2 || T_DC_all_.size() != N_all) {
      throw std::runtime_error("Not enough samples to compute calibration (need >=2).");
    }

    if (radj_enable_) {
      computeRAdjFromSamples();
    } else {
      R_adj_ = Eigen::Matrix3d::Identity();
      have_radj_ = false;
      RCLCPP_INFO(get_logger(),
        "[R_ADJ] disabled; use I");
    }

    Eigen::Matrix4d T_Adj = Eigen::Matrix4d::Identity();
    T_Adj.block<3,3>(0,0) =
      (have_radj_ ? R_adj_ : Eigen::Matrix3d::Identity()).transpose();

    std::vector<Eigen::Matrix4d> T_DC_adj_all;
    T_DC_adj_all.reserve(N_all);
    for (const auto& T_DC : T_DC_all_) {
      T_DC_adj_all.push_back(T_Adj * T_DC);
    }

    const size_t solve_start_idx = chooseHandeyeSolveStartIndex();
    const size_t N = N_all - solve_start_idx;
    if (N < 2) {
      throw std::runtime_error("Not enough samples after hand-eye trimming (need >=2).");
    }
    if (solve_start_idx > 0) {
      RCLCPP_WARN(get_logger(),
        "[HAND_EYE] solving with %zu/%zu samples; validation still uses all samples",
        N, N_all);
    }

    // clear buffers
    O_B0B1_list_.clear();
    O_C0C1_list_.clear();

    const size_t K = (N * (N - 1)) / 2;
    RCLCPP_INFO(get_logger(),
      "[HAND_EYE] using all-pairs motions K=%zu from N=%zu samples",
      K, N);

    Eigen::MatrixXd M(9 * K, 9);
    Eigen::MatrixXd K1(3 * K, 3);

    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    size_t k = 0;
    for (size_t s0=solve_start_idx; s0<N_all; s0++) {
      for (size_t s1=s0+1; s1<N_all; s1++) {
        const Eigen::Matrix4d& T_AB0 = T_AB_all_[s0];
        const Eigen::Matrix4d& T_AB1 = T_AB_all_[s1];

        const Eigen::Matrix4d& T_DC0 = T_DC_adj_all[s0];
        const Eigen::Matrix4d& T_DC1 = T_DC_adj_all[s1];

        const Eigen::Matrix4d T_B0B1 = invT(T_AB0) * T_AB1;
        const Eigen::Matrix4d T_C0C1 = invT(T_DC0) * T_DC1;

        const Eigen::Matrix3d R_B0B1 = T_B0B1.block<3,3>(0,0);
        const Eigen::Vector3d O_B0B1 = T_B0B1.block<3,1>(0,3);

        const Eigen::Matrix3d R_C0C1 = T_C0C1.block<3,3>(0,0);
        const Eigen::Vector3d O_C0C1 = T_C0C1.block<3,1>(0,3);

        Eigen::Matrix<double,9,9> m = kron3(I, R_B0B1) - kron3(R_C0C1.transpose(), I);
        M.block(9*k, 0, 9, 9) = m;

        K1.block(3*k, 0, 3, 3) = (I - R_B0B1);

        O_B0B1_list_.push_back(O_B0B1);
        O_C0C1_list_.push_back(O_C0C1);
        k++;
      }
    }
    if (k != K) {
      throw std::runtime_error("Internal error while building all-pairs hand-eye motions.");
    }

    Eigen::MatrixXd X = M.transpose() * M;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(X);
    if (es.info() != Eigen::Success) throw std::runtime_error("EigenSolver failed on X=M^T*M");

    Eigen::VectorXd vectX = es.eigenvectors().col(0);
    Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::ColMajor>> R_BC_raw(vectX.data());

    struct HandEyeCandidate
    {
      int sign = 1;
      Eigen::Matrix4d T_BC = Eigen::Matrix4d::Identity();
      Eigen::Matrix4d T_AD_avg = Eigen::Matrix4d::Identity();
      double fit_rms_m = std::numeric_limits<double>::infinity();
    };

    auto evaluateCandidate = [&](const Eigen::Matrix3d& R_raw, int sign) {
      HandEyeCandidate cand;
      cand.sign = sign;

      const Eigen::Matrix3d R_BC = projectToSO3(R_raw);
      Eigen::VectorXd K2_candidate(3 * K);
      for (size_t i=0; i<K; i++) {
        const Eigen::Vector3d& O_B0B1 = O_B0B1_list_[i];
        const Eigen::Vector3d& O_C0C1 = O_C0C1_list_[i];
        K2_candidate.segment<3>(3*i) = O_B0B1 - R_BC * O_C0C1;
      }

      const Eigen::Vector3d O_BC = K1.colPivHouseholderQr().solve(K2_candidate);
      cand.T_BC = makeT(R_BC, O_BC);

      std::vector<Eigen::Quaterniond> quats;
      quats.reserve(N);
      Eigen::Vector3d t_sum = Eigen::Vector3d::Zero();

      for (size_t s=solve_start_idx; s<N_all; s++) {
        const Eigen::Matrix4d& T_AB = T_AB_all_[s];
        const Eigen::Matrix4d& T_DC = T_DC_adj_all[s];
        const Eigen::Matrix4d T_AD = T_AB * cand.T_BC * invT(T_DC);

        Eigen::Quaterniond q(T_AD.block<3,3>(0,0));
        q.normalize();
        quats.push_back(q);
        t_sum += T_AD.block<3,1>(0,3);
      }

      Eigen::Quaterniond q_ref = quats.front();
      Eigen::Vector4d q_sum = Eigen::Vector4d::Zero();
      for (auto& q : quats) {
        if (q_ref.coeffs().dot(q.coeffs()) < 0) {
          q.coeffs() *= -1.0;
        }
        q_sum += q.coeffs();
      }
      q_sum /= static_cast<double>(quats.size());
      Eigen::Quaterniond q_mean;
      q_mean.coeffs() = q_sum;
      q_mean.normalize();

      const Eigen::Vector3d t_mean = t_sum / static_cast<double>(N);
      cand.T_AD_avg = makeT(q_mean.toRotationMatrix(), t_mean);

      const Eigen::Matrix4d T_CB = invT(cand.T_BC);
      double sum2 = 0.0;
      for (size_t i=0; i<N_all; ++i) {
        const Eigen::Matrix4d M_cal = cand.T_AD_avg * T_DC_adj_all[i] * T_CB;
        const Eigen::Vector3d p_cal = M_cal.block<3,1>(0,3);
        const Eigen::Vector3d p_ref = T_AB_all_[i].block<3,1>(0,3);
        const double err = (p_ref - p_cal).norm();
        sum2 += err * err;
      }
      cand.fit_rms_m = std::sqrt(sum2 / static_cast<double>(N_all));
      return cand;
    };

    const HandEyeCandidate cand_pos = evaluateCandidate(R_BC_raw, 1);
    const HandEyeCandidate cand_neg = evaluateCandidate(-R_BC_raw, -1);
    const HandEyeCandidate& best =
      (cand_pos.fit_rms_m <= cand_neg.fit_rms_m) ? cand_pos : cand_neg;
    const HandEyeCandidate& alt = (&best == &cand_pos) ? cand_neg : cand_pos;

    RCLCPP_INFO(get_logger(),
      "[HAND_EYE] sign=%+d fit_rms=%.3fmm alt=%+d %.3fmm",
      best.sign,
      best.fit_rms_m * 1000.0,
      alt.sign,
      alt.fit_rms_m * 1000.0);

    const Eigen::Matrix4d T_BC = best.T_BC;
    const Eigen::Matrix4d T_AD_avg = best.T_AD_avg;
    T_FIX_ = computeZPlaneFix(
      T_AD_avg,
      have_radj_ ? R_adj_ : Eigen::Matrix3d::Identity(),
      T_BC);
    z_residual_model_ = computeZResidualModel(
      T_AD_avg,
      have_radj_ ? R_adj_ : Eigen::Matrix3d::Identity(),
      T_BC,
      T_FIX_);

    validateCalibrationFitMm(
      T_AD_avg,
      have_radj_ ? R_adj_ : Eigen::Matrix3d::Identity(),
      T_BC,
      T_FIX_);

    // Decide final T_SA to save:
    // - keep mode: always save old
    // - update mode: if computed => save new else save old
    Eigen::Matrix4d T_SA_to_save = T_SA_old_;
    if (t_sa_mode_ == "update") {
      if (t_sa_computed_) T_SA_to_save = T_SA_new_;
      else T_SA_to_save = T_SA_old_;
    }

    writeCalibrationYamlAll(
      T_AD_avg,
      T_BC,
      have_radj_ ? R_adj_ : Eigen::Matrix3d::Identity(),
      T_FIX_,
      T_CE_,
      T_SA_to_save
    );

    RCLCPP_INFO(get_logger(),
      "[YAML_SAVED] R_Adj=%s",
      have_radj_ ? "computed" : "IDENTITY");
    RCLCPP_INFO(get_logger(),
      "[YAML_SAVED] T_SA %s %s",
      t_sa_mode_.c_str(),
      (t_sa_mode_=="update" ? (t_sa_computed_ ? "new" : "old") : "old"));
    RCLCPP_INFO(get_logger(),
      "[YAML_SAVED] %s",
      calib_yaml_path_.c_str());
  }

  void writeCalibrationYamlAll(const Eigen::Matrix4d& T_AD,
                               const Eigen::Matrix4d& T_BC,
                               const Eigen::Matrix3d& R_Adj,
                               const Eigen::Matrix4d& T_FIX,
                               const Eigen::Matrix4d& T_CE,
                               const Eigen::Matrix4d& T_SA)
  {
    ensureParentDirectoryExists(calib_yaml_path_);
    std::ofstream ofs(calib_yaml_path_, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) throw std::runtime_error("Failed to open yaml: " + calib_yaml_path_);

    const int prec = 12;

    ofs << "# VR calibration matrix setting\n";
    ofs << "# Auto-capture + One-shot YAML update (R_Adj, T_AD, T_BC, T_SA)\n";
    ofs << "# saved_at: " << nowLocalString() << "\n\n";

    ofs << "meta:\n";
    ofs << "  t_sa_w_des_z: " << std::fixed << std::setprecision(prec) << t_sa_w_des_z_ << "\n";
    ofs << "  z_fix_enable: " << (z_fix_enable_ ? "true" : "false") << "\n";
    ofs << "  z_fix_max_tilt_deg: " << std::fixed << std::setprecision(prec) << z_fix_max_tilt_deg_ << "\n";
    ofs << "  z_residual_enable: " << (z_residual_enable_ ? "true" : "false") << "\n";
    ofs << "  z_residual_max_correction_mm: " << std::fixed << std::setprecision(prec)
        << z_residual_max_correction_m_ * 1000.0 << "\n";
    ofs << "  note: \"Runtime tool correction uses inv(T_BC); T_SA is right-multiplied.\"\n\n";

    auto writeMat4 = [&](const std::string& key, const Eigen::Matrix4d& T){
      ofs << key << ":\n";
      ofs << std::fixed << std::setprecision(prec);
      for (int r=0;r<4;r++){
        ofs << "  - [";
        for (int c=0;c<4;c++){
          ofs << T(r,c);
          if (c<3) ofs << ", ";
        }
        ofs << "]\n";
      }
      ofs << "\n";
    };

    auto writeMat3 = [&](const std::string& key, const Eigen::Matrix3d& R){
      ofs << key << ":\n";
      ofs << std::fixed << std::setprecision(prec);
      for (int r=0;r<3;r++){
        ofs << "  - [";
        for (int c=0;c<3;c++){
          ofs << R(r,c);
          if (c<2) ofs << ", ";
        }
        ofs << "]\n";
      }
      ofs << "\n";
    };

    writeMat4("T_AD", T_AD);
    writeMat4("T_BC", T_BC);
    writeMat3("R_Adj", R_Adj);

    ofs << "# left-multiplied rigid z-plane correction (M_cal = T_FIX @ M_cal)\n";
    writeMat4("T_FIX", T_FIX);

    ofs << "# optional z-only residual correction after T_FIX: z += f((x-center_x)/scale_xy, (y-center_y)/scale_xy)\n";
    ofs << "Z_RESIDUAL:\n";
    ofs << "  enabled: " << (z_residual_model_.valid ? "true" : "false") << "\n";
    ofs << std::fixed << std::setprecision(prec);
    ofs << "  model: quadratic_xy\n";
    ofs << "  center_x: " << z_residual_model_.center_x << "\n";
    ofs << "  center_y: " << z_residual_model_.center_y << "\n";
    ofs << "  scale_xy: " << z_residual_model_.scale_xy << "\n";
    ofs << "  max_abs_correction_m: " << z_residual_model_.max_abs_correction_m << "\n";
    ofs << "  rms_before_m: " << z_residual_model_.rms_before_m << "\n";
    ofs << "  rms_after_m: " << z_residual_model_.rms_after_m << "\n";
    ofs << "  coeff: [";
    for (size_t i=0; i<z_residual_model_.coeff.size(); ++i) {
      ofs << z_residual_model_.coeff[i];
      if (i + 1 < z_residual_model_.coeff.size()) ofs << ", ";
    }
    ofs << "]\n\n";

    ofs << "# final constant offset; runtime applies this last when apply_T_CE_extra=true\n";
    writeMat4("T_CE", T_CE);

    ofs << "# spatial-angle frame alignment (right-multiply)\n";
    writeMat4("T_SA", T_SA);

    ofs.flush();
  }
};

// ================= main =================
int main(int argc, char** argv)
{
  setenv("RCUTILS_CONSOLE_OUTPUT_FORMAT", "[{severity}] {message}", 0);

  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<VrCalibration>();
    node->run();
  } catch (const std::exception& e) {
    std::cerr << "vr_calibration exception: " << e.what() << std::endl;
  }
  rclcpp::shutdown();
  return 0;
}
