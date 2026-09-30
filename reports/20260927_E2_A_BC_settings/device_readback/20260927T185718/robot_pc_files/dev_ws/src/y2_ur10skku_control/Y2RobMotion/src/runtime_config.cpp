#include "Y2RobMotion/runtime_config.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>

namespace
{
template<typename T>
T valueOr(const YAML::Node& node, const char* key, const T& fallback)
{
    const YAML::Node value = node[key];
    return value ? value.as<T>() : fallback;
}

template<typename T>
T flatOrNested(
    const YAML::Node& root,
    const char* flat_key,
    const YAML::Node& nested,
    const char* nested_key,
    const T& fallback)
{
    const YAML::Node flat_value = root[flat_key];
    return flat_value ? flat_value.as<T>() : valueOr<T>(nested, nested_key, fallback);
}

template<typename T, std::size_t N>
std::array<T, N> fixedArray(
    const YAML::Node& node,
    const char* key,
    const std::array<T, N>& fallback)
{
    const YAML::Node value = node[key];
    if (!value) {
        return fallback;
    }
    if (!value.IsSequence() || value.size() != N) {
        throw std::runtime_error(
            std::string(key) + " must contain exactly " + std::to_string(N) + " values");
    }

    std::array<T, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = value[i].as<T>();
    }
    return result;
}

template<typename T, std::size_t N>
std::array<T, N> flatOrNestedFixedArray(
    const YAML::Node& root,
    const char* flat_key,
    const YAML::Node& nested,
    const char* nested_key,
    const std::array<T, N>& fallback)
{
    return root[flat_key]
        ? fixedArray<T, N>(root, flat_key, fallback)
        : fixedArray<T, N>(nested, nested_key, fallback);
}

YMatrix matrix4x4(const YAML::Node& node, const char* key, const YMatrix& fallback)
{
    const YAML::Node value = node[key];
    if (!value) {
        return fallback;
    }
    if (!value.IsSequence() || value.size() != 4) {
        throw std::runtime_error(std::string(key) + " must be a 4x4 matrix");
    }

    YMatrix result(4, 4);
    for (std::size_t row = 0; row < 4; ++row) {
        if (!value[row].IsSequence() || value[row].size() != 4) {
            throw std::runtime_error(std::string(key) + " must be a 4x4 matrix");
        }
        for (std::size_t col = 0; col < 4; ++col) {
            result[row][col] = value[row][col].as<double>();
        }
    }
    return result;
}

YMatrix flatOrNestedMatrix4x4(
    const YAML::Node& root,
    const char* flat_key,
    const YAML::Node& nested,
    const char* nested_key,
    const YMatrix& fallback)
{
    return root[flat_key]
        ? matrix4x4(root, flat_key, fallback)
        : matrix4x4(nested, nested_key, fallback);
}

std::string lowercase(std::string value)
{
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string expandHomeDirectory(const std::string& path)
{
    if (path != "~" && path.rfind("~/", 0) != 0) {
        return path;
    }

    const char* home = std::getenv("HOME");
    if (home == nullptr || *home == '\0') {
        throw std::runtime_error(
            "HOME environment variable is not set; cannot expand path: " + path);
    }

    return path == "~"
        ? std::string(home)
        : (std::filesystem::path(home) / path.substr(2)).string();
}
}  // namespace

SetupParameters loadSetupParameters(rclcpp::Node& node)
{
    SetupParameters config;
    const std::string default_file =
        ament_index_cpp::get_package_share_directory("Y2RobMotion") +
        "/config/setup_parameters.yaml";
    config.setup_file =
        node.declare_parameter<std::string>("setup_file", default_file);

    if (!std::filesystem::exists(config.setup_file)) {
        throw std::runtime_error("Setup YAML file not found: " + config.setup_file);
    }

    const YAML::Node root = YAML::LoadFile(config.setup_file);
    const YAML::Node robot = root["robot"];
    const YAML::Node trajectory = root["trajectory"];
    const YAML::Node topics = root["topics"];
    const YAML::Node joystick = root["joystick"];
    const YAML::Node force = root["force_control"];
    const YAML::Node measurement = root["measurement"];

    config.robot_kinematics = lowercase(flatOrNested<std::string>(
        root, "ROBOT_KINEMATICS", robot, "kinematics", config.robot_kinematics));
    config.number_of_joints = flatOrNested<int>(
        root, "NUMBER_OF_JOINTS", robot, "number_of_joints", config.number_of_joints);
    config.package_bundle_directory = expandHomeDirectory(
        valueOr<std::string>(
            root, "PACKAGE_BUNDLE_DIR", config.package_bundle_directory));
    config.trajectory_mode = flatOrNested<int>(
        root, "TRAJECTORY_MODE", trajectory, "mode", config.trajectory_mode);

    config.force_control_coordinate = flatOrNested<int>(
        root, "Force_Con_Coordinate", force, "coordinate", config.force_control_coordinate);
    config.force_q_contact_threshold = flatOrNested<double>(
        root,
        "FORCE_Q_CONTACT_THRESHOLD",
        force,
        "q_contact_threshold",
        config.force_q_contact_threshold);
    config.force_q_external_fd_threshold = flatOrNested<double>(
        root,
        "FORCE_Q_EXTERNAL_FD_THRESHOLD",
        force,
        "q_external_fd_threshold",
        config.force_q_external_fd_threshold);
    config.force_q_internal_force_gain = flatOrNested<double>(
        root,
        "FORCE_Q_INTERNAL_FORCE_GAIN",
        force,
        "q_internal_force_gain",
        config.force_q_internal_force_gain);
    config.force_q_internal_force_saturation = flatOrNested<double>(
        root,
        "FORCE_Q_INTERNAL_FORCE_SATURATION",
        force,
        "q_internal_force_saturation",
        config.force_q_internal_force_saturation);
    config.force_q_basis_eps = flatOrNested<double>(
        root, "FORCE_Q_BASIS_EPS", force, "q_basis_epsilon", config.force_q_basis_eps);

    config.control_period = flatOrNested<double>(
        root, "CONTROL_PERIOD", robot, "control_period", config.control_period);
    config.robot_name = flatOrNested<std::string>(
        root, "ROBOT_NAME", robot, "name", config.robot_name);

    config.test_mode = flatOrNested<bool>(
        root, "TEST_MODE", topics, "test_mode", config.test_mode);
    config.remapping_enabled = flatOrNested<bool>(
        root,
        "REMAPPING_ENABLED",
        topics,
        "remapping_enabled",
        config.remapping_enabled);
    config.remap_state_topic = flatOrNested<std::string>(
        root, "REMAP_STATE_TOPIC", topics, "remap_state", config.remap_state_topic);
    config.remap_command_topic = flatOrNested<std::string>(
        root, "REMAP_COMMAND_TOPIC", topics, "remap_command", config.remap_command_topic);
    config.joy_move_topic_suffix = flatOrNested<std::string>(
        root,
        "JOY_MOVE_TOPIC_SUFFIX",
        topics,
        "joy_move_suffix",
        config.joy_move_topic_suffix);

    config.joystick_axis_mapping = flatOrNestedFixedArray<int, 6>(
        root,
        "JOYSTICK_AXIS_MAPPING",
        joystick,
        "axis_mapping",
        config.joystick_axis_mapping);
    config.joystick_axis_scales = flatOrNestedFixedArray<double, 6>(
        root,
        "JOYSTICK_AXIS_SCALES",
        joystick,
        "axis_scales",
        config.joystick_axis_scales);
    config.joystick_force_input_axis = flatOrNested<int>(
        root,
        "JOYSTICK_FORCE_INPUT_AXIS",
        joystick,
        "force_input_axis",
        config.joystick_force_input_axis);
    config.joystick_force_input_scale = flatOrNested<double>(
        root,
        "JOYSTICK_FORCE_INPUT_SCALE",
        joystick,
        "force_input_scale",
        config.joystick_force_input_scale);
    config.joystick_force_target_axis = flatOrNested<int>(
        root,
        "JOYSTICK_FORCE_TARGET_AXIS",
        joystick,
        "force_target_axis",
        config.joystick_force_target_axis);
    config.joystick_force_input_neutral = flatOrNested<double>(
        root,
        "JOYSTICK_FORCE_INPUT_NEUTRAL",
        joystick,
        "force_input_neutral",
        config.joystick_force_input_neutral);

    config.ee_to_tcp = flatOrNestedMatrix4x4(
        root, "EE2TCP", robot, "ee_to_tcp", config.ee_to_tcp);
    config.joint_names = flatOrNested<std::vector<std::string>>(
        root, "JOINT_NAMES", robot, "joint_names", config.joint_names);

    config.default_travel_time = flatOrNested<double>(
        root,
        "DEFAULT_TRAVEL_TIME",
        trajectory,
        "default_travel_time",
        config.default_travel_time);
    config.initial_transfer_speed = flatOrNested<double>(
        root,
        "INITIAL_TRANSFER_SPEED",
        trajectory,
        "initial_transfer_speed",
        config.initial_transfer_speed);
    config.angular_velocity_limit = flatOrNested<double>(
        root,
        "ANGULAR_VELOCITY_LIMIT",
        trajectory,
        "angular_velocity_limit",
        config.angular_velocity_limit);
    config.acceleration_time = flatOrNested<double>(
        root, "ACCELERATION_TIME", trajectory, "acceleration_time", config.acceleration_time);
    config.starting_time = flatOrNested<double>(
        root, "STARTING_TIME", trajectory, "starting_time", config.starting_time);
    config.last_resting_time = flatOrNested<double>(
        root,
        "LAST_RESTING_TIME",
        trajectory,
        "last_resting_time",
        config.last_resting_time);
    config.ptp_target_velocity = flatOrNested<double>(
        root,
        "PTP_TARGET_VELOCITY",
        trajectory,
        "ptp_target_velocity",
        config.ptp_target_velocity);

    config.force_control_mode = flatOrNested<int>(
        root, "Force_Con_Mode", force, "mode", config.force_control_mode);
    config.force_switch_desired_force_threshold = flatOrNested<double>(
        root,
        "FORCE_SWITCH_DESIRED_FORCE_THRESHOLD",
        force,
        "switch_desired_force_threshold",
        config.force_switch_desired_force_threshold);
    config.force_switch_actual_force_threshold = flatOrNested<double>(
        root,
        "FORCE_SWITCH_ACTUAL_FORCE_THRESHOLD",
        force,
        "switch_actual_force_threshold",
        config.force_switch_actual_force_threshold);
    config.force_switch_precontact_force_hold = flatOrNested<double>(
        root,
        "FORCE_SWITCH_PRECONTACT_FORCE_HOLD",
        force,
        "switch_precontact_force_hold",
        config.force_switch_precontact_force_hold);
    config.force_switch_return_tau_m = flatOrNested<double>(
        root,
        "FORCE_SWITCH_RETURN_TAU_M",
        force,
        "switch_return_tau_m",
        config.force_switch_return_tau_m);
    config.force_switch_return_tau_d = flatOrNested<double>(
        root,
        "FORCE_SWITCH_RETURN_TAU_D",
        force,
        "switch_return_tau_d",
        config.force_switch_return_tau_d);
    config.force_switch_return_tau_k = flatOrNested<double>(
        root,
        "FORCE_SWITCH_RETURN_TAU_K",
        force,
        "switch_return_tau_k",
        config.force_switch_return_tau_k);

    const std::string default_measurement_directory =
        config.package_bundle_directory.empty()
            ? ament_index_cpp::get_package_share_directory("Y2RobMotion") + "/measured"
            : config.package_bundle_directory + "/Y2RobMotion/measured";
    config.measurement_output_directory = expandHomeDirectory(flatOrNested<std::string>(
        root,
        "MEASUREMENT_OUTPUT_DIRECTORY",
        measurement,
        "output_directory",
        default_measurement_directory));
    config.record_start_topic = flatOrNested<std::string>(
        root,
        "RECORD_START_TOPIC",
        measurement,
        "record_start_topic",
        config.record_start_topic);
    config.record_stop_topic = flatOrNested<std::string>(
        root,
        "RECORD_STOP_TOPIC",
        measurement,
        "record_stop_topic",
        config.record_stop_topic);
    config.measurement_mode = lowercase(flatOrNested<std::string>(
        root, "MEASUREMENT_MODE", measurement, "mode", config.measurement_mode));

    if (config.number_of_joints <= 0) {
        throw std::runtime_error("NUMBER_OF_JOINTS must be positive");
    }
    if (config.control_period <= 0.0) {
        throw std::runtime_error("CONTROL_PERIOD must be positive");
    }
    if (config.joint_names.size() != static_cast<std::size_t>(config.number_of_joints)) {
        throw std::runtime_error(
            "JOINT_NAMES count must match NUMBER_OF_JOINTS");
    }
    if (config.robot_kinematics != "kuka_iiwa" &&
        config.robot_kinematics != "ur10" &&
        config.robot_kinematics != "ur10e") {
        throw std::runtime_error(
            "ROBOT_KINEMATICS must be one of: kuka_iiwa, ur10, ur10e");
    }
    if (config.force_control_coordinate < 0 || config.force_control_coordinate > 2) {
        throw std::runtime_error("Force_Con_Coordinate must be 0, 1, or 2");
    }
    if (config.force_control_mode < 0 || config.force_control_mode > 4) {
        throw std::runtime_error("Force_Con_Mode must be in the range 0..4");
    }

    RCLCPP_INFO(
        node.get_logger(),
        "Loaded setup YAML: %s "
        "(robot=%s, kinematics=%s, period=%.6fs, PTP velocity=%.3f mm/s)",
        config.setup_file.c_str(),
        config.robot_name.c_str(),
        config.robot_kinematics.c_str(),
        config.control_period,
        config.ptp_target_velocity);
    return config;
}
