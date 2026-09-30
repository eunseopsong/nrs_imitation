#include "Y2RobMotion/robot_motion.hpp"
#include <sstream>
#include <iomanip>
#include <cmath>

namespace fs = std::filesystem;

// ★ 추가: 경로 존재 + TorchScript load 확인
void robot_motion::check_model_file_or_throw(const std::string& path, const std::string& tag)
{
    RCLCPP_INFO(node_->get_logger(), "[%s] model path: %s", tag.c_str(), path.c_str());

    if (!fs::exists(path)) {
        RCLCPP_ERROR(node_->get_logger(), "[%s] MODEL FILE NOT FOUND: %s", tag.c_str(), path.c_str());
        throw std::runtime_error("Model file not found: " + path);
    }

    auto sz = fs::file_size(path);
    RCLCPP_INFO(node_->get_logger(), "[%s] model file exists (size=%ld bytes)", tag.c_str(), (long)sz);

    // TorchScript 로딩이 되는지까지 확인
    try {
        auto m = torch::jit::load(path, torch::kCPU);
        m.eval();
        RCLCPP_INFO(node_->get_logger(), "[%s] TorchScript load OK", tag.c_str());
    } catch (const c10::Error& e) {
        RCLCPP_ERROR(node_->get_logger(), "[%s] TorchScript load FAILED: %s", tag.c_str(), e.what());
        throw;
    }
}

/* Constructor */
robot_motion::robot_motion(
    rclcpp::Node::SharedPtr node,
    const SetupParameters& config)
: current_angles(config.number_of_joints,0.0)
, pre_current_angles(config.number_of_joints,0.0)
, current_angvel(config.number_of_joints,0.0)
, target_angles(config.number_of_joints,0.0)
, pre_target_angles(config.number_of_joints,0.0)
, target_angvel(config.number_of_joints,0.0)
, current_pose(6,0.0)
, pre_current_pose(6,0.0)
, current_carvel(6,0.0)
, target_pose(6,0.0)
, pre_target_pose(6,0.0)
, target_carvel(6,0.0)
, joy_target_pose(6,0.0)
, AC_pose(6,0.0)
, target_HTM(4,4)
, ft1data(6,0.0)
, robot_name(config.robot_name)
, numOfJoints(config.number_of_joints)
, node_(node)
, config_(config)
, Control_period_(config.control_period)
, HG_AC_desX(6,0.0)
, FC_AC_desX(9,0.0)
, FC_MASS(6,0.0)
, FC_DAMPER(6,0.0)
, FC_STIFFNESS(6,0.0)
, joy_move_axes_(8,0.0)
, joy_velocity_command_(6,0.0)
, joy_force_command_(3,0.0)
{
    if (config_.robot_kinematics == "kuka_iiwa") {
        kinematics_ = std::make_unique<KinematicsKUKAiiwa>(
            config_.control_period,
            static_cast<std::size_t>(config_.number_of_joints),
            config_.ee_to_tcp);
    }
    else if (config_.robot_kinematics == "ur10") {
        kinematics_ = std::make_unique<KinematicsUR10>(
            config_.control_period,
            static_cast<std::size_t>(config_.number_of_joints),
            config_.ee_to_tcp);
    }
    else {
        kinematics_ = std::make_unique<KinematicsUR10e>(
            config_.control_period,
            static_cast<std::size_t>(config_.number_of_joints),
            config_.ee_to_tcp);
    }

    joint_names = config_.joint_names;
    joy_axis_mapping_ = config_.joystick_axis_mapping;
    joy_axis_scales_ = config_.joystick_axis_scales;
    joy_force_input_axis_ = config_.joystick_force_input_axis;
    joy_force_input_scale_ = config_.joystick_force_input_scale;
    joy_force_target_axis_ = config_.joystick_force_target_axis;
    joy_force_input_neutral_ = config_.joystick_force_input_neutral;

    /* Target HTM  init */
    target_HTM = YMatrix::identity(4);

    /* Control mode init */
    control_mode = "Idling";
    pre_control_mode = "none";

    /* Force control mode init */
    force_con_mode = "Defualt";

    /* ROS Publisher init */
    std::string currentJ_TP = robot_name + "/currentJ";
    std::string currentP_TP = robot_name + "/currentP";
    std::string currentF_TP = robot_name + "/currentF";
    std::string currentMDK_TP = robot_name + "/currentMDK";
    std::string targetJ_TP = robot_name + "/targetJ";
    std::string targetP_TP = robot_name + "/targetP";
    std::string targetF_TP = robot_name + "/targetF";
    std::string ctlMode_TP = robot_name + "/ctlMode";
    std::string remapedCmd_TP = config_.remap_command_topic;

    currentJ_pub = node_->create_publisher<std_msgs::msg::Float64MultiArray>(currentJ_TP,1);
    currentP_pub = node_->create_publisher<std_msgs::msg::Float64MultiArray>(currentP_TP,1);
    currentF_pub = node_->create_publisher<std_msgs::msg::Float64MultiArray>(currentF_TP,1);
    currentF_provenance_pub_ = node_->create_publisher<std_msgs::msg::String>(
        robot_name + "/currentF_provenance", rclcpp::QoS(1).reliable());
    currentMDK_pub = node_->create_publisher<std_msgs::msg::Float64MultiArray>(currentMDK_TP,1);
    targetJ_pub = node_->create_publisher<std_msgs::msg::Float64MultiArray>(targetJ_TP,1);
    targetP_pub = node_->create_publisher<std_msgs::msg::Float64MultiArray>(targetP_TP,1);
    targetF_pub = node_->create_publisher<std_msgs::msg::Float64MultiArray>(targetF_TP,1);
    remapedCmd_pub = node_->create_publisher<std_msgs::msg::Float64MultiArray>(remapedCmd_TP,1);
    ctlMode_pub = node_->create_publisher<std_msgs::msg::String>(ctlMode_TP,10);
    RCLCPP_INFO(node_->get_logger(),"Publisher was generated");

    /* ROS Subscriber init */
    std::string cmdMotion_TP = robot_name + "/cmdMotion";
    std::string cmdMode_TP = robot_name + "/cmdMode";
    std::string joyMove_TP = robot_name + config_.joy_move_topic_suffix;
    std::string jointState_TP = robot_name + "/joint_states";
    std::string ftdata_TP = robot_name + "/ftdata";
    std::string remapedState_TP = config_.remap_state_topic;

    cmdMotion_sub = node_->create_subscription<std_msgs::msg::Float64MultiArray>(cmdMotion_TP, 1,
        std::bind(&robot_motion::cmdMotionCB, this, std::placeholders::_1));
    joyMove_sub = node_->create_subscription<std_msgs::msg::Float64MultiArray>(joyMove_TP, 1,
        std::bind(&robot_motion::joyMoveCB, this, std::placeholders::_1));
    cmdMode_sub = node_->create_subscription<std_msgs::msg::String>(cmdMode_TP, 10,
        std::bind(&robot_motion::cmdModeCB, this, std::placeholders::_1));

    JointState_sub = node_->create_subscription<sensor_msgs::msg::JointState>(jointState_TP, 1,
        std::bind(&robot_motion::JointStateCB, this, std::placeholders::_1));
    RemapedState_sub = node_->create_subscription<sensor_msgs::msg::JointState>(remapedState_TP, 1,
        std::bind(&robot_motion::RemapedStateCB, this, std::placeholders::_1));
    ftsensor_sub = node_->create_subscription<geometry_msgs::msg::WrenchStamped>(ftdata_TP, 1,
        std::bind(&robot_motion::ftsensorCB, this, std::placeholders::_1));

    /* ROS timer callback */
    timer_ = node_->create_wall_timer(
        std::chrono::duration<double>(config_.control_period),
        std::bind(&robot_motion::main_control,this));

    minitoring_timer_ = node_->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(100)),
        std::bind(&robot_motion::state_monitoring,this));

    RCLCPP_INFO(node_->get_logger(),"Subscription was generated");

    /* Admittance controller initialization */
    for(int i =0;i<6;i++){ AControl[i] = Yadmittance_control(config_.control_period); }

    /* FAAC instance initialization */
    std::vector<double> process_noise = {0.1,0.1,0.1};
    std::vector<double> measurement_noise = {10,10,10};
    for(int i =0;i<3;i++){
        FAAC3step[i] = std::make_unique<Nrs3StepFAAC>(
            config_.control_period, process_noise, measurement_noise);
    }

    /* RL instance initialization */
    std::string Y2ForceCon_share_dir = ament_index_cpp::get_package_share_directory("Y2ForceCon");

    std::string context_naf_mdgradi_model_path = Y2ForceCon_share_dir + "/checkpoints/ContextNAF_MDGradi/contextNAF_mdGradi_policy_script.pt";
    std::string naf_massvari_model_path = Y2ForceCon_share_dir + "/checkpoints/NAF_MassVari/NAF_massVari_policy_script.pt";
    std::string naf_mdgradi_model_path = Y2ForceCon_share_dir + "/checkpoints/NAF_MDGradi/NAF_mdGradi_policy_script.pt";

    // ★ 여기서 경로 + 로드 검증
    check_model_file_or_throw(context_naf_mdgradi_model_path,   "CONTEXT-NAF-MDGRADI");
    check_model_file_or_throw(naf_massvari_model_path,   "NAF-MASSVARI");
    check_model_file_or_throw(naf_mdgradi_model_path,   "NAF-MDGRADI");

    for(int i =0;i<3;i++){
        policy_context_naf_mdgradi[i] = std::make_unique<RL_ContextNAF_mdGradi>(
            context_naf_mdgradi_model_path,
            /*threads=*/1,
            torch::kCPU,
            /*dt=*/config_.control_period);

        policy_naf_massvari[i] = std::make_unique<RL_Naf_massVari>(
            naf_massvari_model_path,
            /*threads=*/1,
            torch::kCPU,
            /*dt=*/config_.control_period);

        policy_naf_mdgradi[i] = std::make_unique<RL_Naf_mdGradi>(
            naf_mdgradi_model_path,
            /*threads=*/1,
            torch::kCPU,
            /*dt=*/config_.control_period);
    }
}

/* ROS CMD Motion Callback */
void robot_motion::cmdMotionCB(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if(!strcmp(control_mode.c_str(), "Position"))
    {
        for (size_t i = 0; i < msg->data.size() && i < 6; ++i) {
            target_pose[i] = msg->data[i];
        }
    }
    else if(!strcmp(control_mode.c_str(), "Force"))
    {
        for (size_t i = 0; i < msg->data.size() && i < 9; ++i) {
            /* mm->m, AC input: m, rad */
            if(i<3) {FC_AC_desX[i] = (msg->data[i])/(1000.0);}
            else{FC_AC_desX[i] = msg->data[i];}
        }
    }
    else {printf("\033[33mWrong data was received \033[0m\n");}
}

void robot_motion::joyMoveCB(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    std::fill(joy_move_axes_.begin(), joy_move_axes_.end(), 0.0);
    for (size_t i = 0; i < msg->data.size() && i < joy_move_axes_.size(); ++i) {
        joy_move_axes_[i] = msg->data[i];
    }

    for (size_t i = 0; i < joy_velocity_command_.size(); ++i) {
        const int axis_index = joy_axis_mapping_[i];
        if (axis_index < 0 || axis_index >= static_cast<int>(joy_move_axes_.size())) {
            joy_velocity_command_[i] = 0.0;
            continue;
        }

        joy_velocity_command_[i] = joy_move_axes_[axis_index] * joy_axis_scales_[i];
    }

    std::fill(joy_force_command_.begin(), joy_force_command_.end(), 0.0);
    if (joy_force_target_axis_ >= 0 &&
        joy_force_target_axis_ < static_cast<int>(joy_force_command_.size()) &&
        joy_force_input_axis_ >= 0 &&
        joy_force_input_axis_ < static_cast<int>(joy_move_axes_.size())) {
        const double axis_delta =
            joy_force_input_neutral_ - joy_move_axes_[joy_force_input_axis_];
        joy_force_command_[joy_force_target_axis_] =
            axis_delta * joy_force_input_scale_;
    }
}

/* ROS CMD Mode Callback */
void robot_motion::cmdModeCB(const std_msgs::msg::String::SharedPtr msg)
{
    printf("Control mode callback \n");
    control_mode = msg->data;
}

/* ROS Joint State Callback */
void robot_motion::JointStateCB(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    if(!current_angles_received) {
        current_angles_received = true;
    }

    if (msg->position.size() < numOfJoints || msg->name.size() != msg->position.size()) {
        RCLCPP_WARN(node_->get_logger(), "Invalid joint state message");
        return;
    }

    // 첫 번째 메시지에서 매핑 테이블 생성
    if (!mapping_initialized_) {
        initializeJointMapping(msg);
    }

    // 매핑 테이블을 사용해서 빠르게 복사
    for (size_t i = 0; i < numOfJoints; ++i) {
        if (joint_mapping_[i] >= 0 && joint_mapping_[i] < static_cast<int>(msg->position.size())) {
            current_angles[i] = msg->position[joint_mapping_[i]];
        } else {
            RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                                 "Invalid mapping for joint %zu", i);
        }
    }
}

void robot_motion::RemapedStateCB(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    if(!config_.remapping_enabled) {
        return;
    }

    if(!current_angles_received) {
        current_angles_received = true;
    }

    if (msg->position.size() < numOfJoints || msg->name.size() != msg->position.size()) {
        RCLCPP_WARN(node_->get_logger(), "Invalid joint state message");
        return;
    }

    // 첫 번째 메시지에서 매핑 테이블 생성
    if (!mapping_initialized_) {
        initializeJointMapping(msg);
    }

    // 매핑 테이블을 사용해서 빠르게 복사
    for (size_t i = 0; i < numOfJoints; ++i) {
        if (joint_mapping_[i] >= 0 && joint_mapping_[i] < static_cast<int>(msg->position.size())) {
            current_angles[i] = msg->position[joint_mapping_[i]];
        } else {
            RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                                 "Invalid mapping for joint %zu", i);
        }
    }
}

/* ROS FT Sensor Callback */
void robot_motion::ftsensorCB(const geometry_msgs::msg::WrenchStamped::SharedPtr msg)
{
    ft1_source_ros_ns_ = static_cast<std::int64_t>(msg->header.stamp.sec) * 1000000000LL
        + msg->header.stamp.nanosec;
    ft1_source_contract_ = msg->header.frame_id == "robot_base_acquired_v1";
    ft1data[0] = msg->wrench.force.x;
    ft1data[1] = msg->wrench.force.y;
    ft1data[2] = msg->wrench.force.z;

    ft1data[3] = msg->wrench.torque.x;
    ft1data[4] = msg->wrench.torque.y;
    ft1data[5] = msg->wrench.torque.z;
}

/* joint state mapping function */
void robot_motion::initializeJointMapping(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    joint_mapping_.resize(numOfJoints, -1);

    for (size_t i = 0; i < numOfJoints; ++i) {
        for (size_t j = 0; j < msg->name.size(); ++j) {
            if (joint_names[i] == msg->name[j]) {
                joint_mapping_[i] = j;
                RCLCPP_INFO(node_->get_logger(),
                            "Joint %s mapped: expected_index=%zu -> msg_index=%zu",
                            joint_names[i].c_str(), i, j);
                break;
            }
        }

        if (joint_mapping_[i] == -1) {
            RCLCPP_ERROR(node_->get_logger(),
                         "Joint %s not found in joint state message!",
                         joint_names[i].c_str());
        }
    }
    mapping_initialized_ = true;
}

/* State Update */
void robot_motion::state_update()
{
    for(unsigned int i=0;i<numOfJoints;i++)
    {
        current_angvel[i] = (current_angles[i]-pre_current_angles[i])/Control_period_;
        target_angvel[i]  = (target_angles[i]-pre_target_angles[i])/Control_period_;
    }

    YMatrix current_HTM = kinematics_->forwardKinematics(current_angles);
    YMatrix current_orientation = current_HTM.extract(0,0,3,3);
    SpatialAngle current_SPangle = current_orientation.toSpatialAngle();

    current_pose[0] = current_HTM[0][3];
    current_pose[1] = current_HTM[1][3];
    current_pose[2] = current_HTM[2][3];
    current_pose[3] = current_SPangle.x;
    current_pose[4] = current_SPangle.y;
    current_pose[5] = current_SPangle.z;

    for(int i=0;i<6;i++)
    {
        current_carvel[i] = (current_pose[i]-pre_current_pose[i])/Control_period_;
        target_carvel[i]  = (target_pose[i]-pre_target_pose[i])/Control_period_;
    }

    pre_current_angles = current_angles;
    pre_target_angles  = target_angles;
    pre_current_pose   = current_pose;
    pre_target_pose    = target_pose;
}

/* ROS State Publisher */
void robot_motion::state_publisher()
{
    ctlMode_msg.data.clear();
    currentJ_msg.data.clear();
    currentp_msg.data.clear();
    currentF_msg.data.clear();
    currentMDK_msg.data.clear();

    targetJ_msg.data.clear();
    targetP_msg.data.clear();
    targetF_msg.data.clear();
    remapedCmd_msg.data.clear();

    for (int i=0;i<3;i++)
    {
        double norminal_MDK = sqrt(
            pow(AControl[0].adm_MDK_monitor(i),2)
            + pow(AControl[1].adm_MDK_monitor(i),2)
            + pow(AControl[2].adm_MDK_monitor(i),2)
        );
        currentMDK_msg.data.push_back(norminal_MDK);
    }

    for(int i =0; i<6; i++)
    {
        currentp_msg.data.push_back(current_pose[i]);
        currentF_msg.data.push_back(ft1data[i]);

        targetP_msg.data.push_back(target_pose[i]);
        if(i<3){ targetF_msg.data.push_back(FC_AC_desX[i+6]); }
        else {  targetF_msg.data.push_back(0.0); }
    }

    for(unsigned int i =0; i<numOfJoints; i++)
    {
        currentJ_msg.data.push_back(current_angles[i]);
        targetJ_msg.data.push_back(target_angles[i]);
        remapedCmd_msg.data.push_back(target_angles[i]);
    }

    ctlMode_msg.data = control_mode;

    ctlMode_pub->publish(ctlMode_msg);
    currentJ_pub->publish(currentJ_msg);
    currentP_pub->publish(currentp_msg);
    currentF_pub->publish(currentF_msg);
    // Retain the source stamp while currentF is periodically republished.
    // The legacy 6-value topic and control computations remain unchanged.
    bool valid_wrench = true;
    for (double value : ft1data) valid_wrench = valid_wrench && std::isfinite(value);
    std::ostringstream provenance;
    provenance << std::setprecision(17)
        << "{\"schema\":\"y2_wrench_provenance_v1\",\"source_contract\":"
        << (ft1_source_contract_ ? "true" : "false")
        << ",\"source_ros_ns\":" << ft1_source_ros_ns_
        << ",\"valid\":" << (valid_wrench ? "true" : "false") << ",\"base_wrench\":";
    if (valid_wrench) {
        provenance << '[';
        for (std::size_t i = 0; i < ft1data.size(); ++i) {
            if (i) provenance << ',';
            provenance << ft1data[i];
        }
        provenance << ']';
    } else {
        provenance << "null";
    }
    provenance << '}';
    std_msgs::msg::String provenance_msg;
    provenance_msg.data = provenance.str();
    currentF_provenance_pub_->publish(provenance_msg);
    currentMDK_pub->publish(currentMDK_msg);

    if (!config_.test_mode) {
        targetJ_pub->publish(targetJ_msg);
        targetP_pub->publish(targetP_msg);
        targetF_pub->publish(targetF_msg);
        if(config_.remapping_enabled){ remapedCmd_pub->publish(remapedCmd_msg); }
    }
}

/* Control Mode : Idling */
void robot_motion::control_idling()
{
    control_mode = "Idling";

    if(pre_control_mode != control_mode)
    {
        target_pose = current_pose;
        target_angles = current_angles;
    }

    if(control_mode == "Idling"){
        std::vector<double> target_ori = {target_pose[3], target_pose[4], target_pose[5]};
        auto target_rot = YMatrix::fromSpatialAngle(target_ori);

        target_HTM = YMatrix::identity(4);
        target_HTM.insert(0, 0, target_rot);
        target_HTM[0][3] = target_pose[0];
        target_HTM[1][3] = target_pose[1];
        target_HTM[2][3] = target_pose[2];

        auto current_HTM = kinematics_->forwardKinematics(target_angles);

        target_angles = kinematics_->solve_IK(target_angles, target_HTM);
        
    }

    pre_control_mode = control_mode;
}

/* Control Mode : Position */
void robot_motion::control_position()
{
    control_mode = "Position";

    if(pre_control_mode != control_mode)
    {
        target_pose = current_pose;
        target_angles = current_angles;
    }

    if(control_mode == "Position"){
        std::vector<double> target_ori = {target_pose[3], target_pose[4], target_pose[5]};
        auto target_rot = YMatrix::fromSpatialAngle(target_ori);

        target_HTM = YMatrix::identity(4);
        target_HTM.insert(0, 0, target_rot);
        target_HTM[0][3] = target_pose[0];
        target_HTM[1][3] = target_pose[1];
        target_HTM[2][3] = target_pose[2];

        target_angles = kinematics_->solve_IK(target_angles, target_HTM);
    }

    pre_control_mode = control_mode;
}

void robot_motion::control_joystick()
{
    control_mode = "Joystick";

    if(pre_control_mode != control_mode)
    {
        joy_target_pose = current_pose;
        target_pose = current_pose;
        target_angles = current_angles;
    }

    for(int i = 0; i < 6; ++i)
    {
        joy_target_pose[i] += joy_velocity_command_[i] * Control_period_;
    }

    target_pose = joy_target_pose;

    std::vector<double> target_ori = {target_pose[3], target_pose[4], target_pose[5]};
    auto target_rot = YMatrix::fromSpatialAngle(target_ori);

    target_HTM = YMatrix::identity(4);
    target_HTM.insert(0, 0, target_rot);
    target_HTM[0][3] = target_pose[0];
    target_HTM[1][3] = target_pose[1];
    target_HTM[2][3] = target_pose[2];

    target_angles = kinematics_->solve_IK(target_angles, target_HTM);

    pre_control_mode = control_mode;
}

void robot_motion::control_joystick_force()
{
    control_mode = "Joystick_force";

    if(pre_control_mode != control_mode)
    {
        joy_target_pose = current_pose;
        initialize_force_control_state();
    }

    update_force_target_from_joystick();
    execute_force_control();
}

void robot_motion::main_control()
{
    if(start_flag){
        state_update();

        if(control_mode == "Idling") control_idling();
        else if (control_mode == "Position") control_position();
        else if (control_mode == "Joystick") control_joystick();
        else if (control_mode == "Joystick_force") control_joystick_force();
        else if (control_mode == "Guiding") control_guiding();
        else if (control_mode == "Force") control_force();
        else control_idling();

        state_publisher();
    }
}
