#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <thread>
#include <mutex>
#include <algorithm>
#include <functional>
#include <cctype>
#include <exception>
#include <cstdio>
#include <future>

#include "std_srvs/srv/trigger.hpp"

#include "Y2Matrix/YMatrix.hpp"
#include "Y2Trajectory/MotionBlender6D.hpp"
#include "Y2RobMotion/robot_command.hpp"
#include "Y2RobMotion/runtime_config.hpp"

#include "y2_rob_motion_interfaces/srv/single_arm_command.hpp"

class singleArm_cmd : public rclcpp::Node
{
public:
    using SingleArmCommandSrv = y2_rob_motion_interfaces::srv::SingleArmCommand;
    using TriggerSrv = std_srvs::srv::Trigger;

    singleArm_cmd(
        const std::string& node_name)
    : Node(node_name)
    {
        config_ = loadSetupParameters(*this);
        load_file_ =
            (config_.trajectory_mode == 0) ? "txtcmd/cmd_6D.txt" :
            ((config_.trajectory_mode == 1) ? "txtcmd/cmd_9D.txt" :
                                             "txtcmd/cmd_continue9D.txt");

        PathGenParam pg_param;
        pg_param.defualt_travelTime = config_.default_travel_time;
        pg_param.initialTransferSpeed = config_.initial_transfer_speed;
        pg_param.angularVelocityLimit = config_.angular_velocity_limit;
        pg_param.accelerationTime = config_.acceleration_time;
        pg_param.startingTime = config_.starting_time;
        pg_param.lastRestingTime = config_.last_resting_time;
        pg_param.ptp_target_velocity = config_.ptp_target_velocity;
        pg_param.loadFileType =
            config_.trajectory_mode == 0 ? "cmd_6D" :
            (config_.trajectory_mode == 1 ? "cmd_9D" : "cmd_continue9D");

        rb_cmd = std::make_unique<robot_command>(
            this,
            config_.robot_name,
            config_.control_period,
            pg_param
        );

        package_path_ = config_.package_bundle_directory.empty()
            ? ament_index_cpp::get_package_share_directory("Y2RobMotion")
            : config_.package_bundle_directory + "/Y2RobMotion";

        command_service_ = this->create_service<SingleArmCommandSrv>(
            "~/single_arm_command",
            std::bind(
                &singleArm_cmd::CommandServiceCallback,
                this,
                std::placeholders::_1,
                std::placeholders::_2
            )
        );

        polishing_start_client_ =
            this->create_client<TriggerSrv>("/polishing_removal_node/start");

        polishing_end_client_ =
            this->create_client<TriggerSrv>("/polishing_removal_node/end");

        wait_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1000),
            std::bind(&singleArm_cmd::PrintRobotInitState, this)
        );

        RCLCPP_INFO(
            this->get_logger(),
            "singleArm_cmd service server is ready: /singleArm_cmd/single_arm_command"
        );

        RCLCPP_INFO(
            this->get_logger(),
            "Polishing removal trigger clients are ready: "
            "/polishing_removal_node/start, /polishing_removal_node/end"
        );
    }

    bool is_roboInit()
    {
        return rb_cmd->robot_init;
    }

    void sendCommand(const std::string& command_mode, const YMatrix& loaded_motion)
    {
        rb_cmd->sendCommand(command_mode, loaded_motion);
    }

    void setLoadFileType(const std::string& load_file_type)
    {
        rb_cmd->setLoadFileType(load_file_type);
    }

    void setPTPTargetVelocity(double target_velocity)
    {
        rb_cmd->setPTPTargetVelocity(target_velocity);
    }

    bool startStream(double target_velocity, std::string& message)
    {
        return rb_cmd->startStream(target_velocity, message);
    }

    bool appendStream(const YMatrix& loaded_motion, std::string& message)
    {
        return rb_cmd->appendStreamWaypoints(loaded_motion, message);
    }

    void stopStream()
    {
        rb_cmd->stopStream();
    }

    bool setStreamForce(double fx, double fy, double fz, std::string& message)
    {
        return rb_cmd->setStreamForce(fx, fy, fz, message);
    }

private:
    std::unique_ptr<robot_command> rb_cmd;
    SetupParameters config_;
    std::string load_file_;

    rclcpp::Service<SingleArmCommandSrv>::SharedPtr command_service_;
    rclcpp::TimerBase::SharedPtr wait_timer_;

    rclcpp::Client<TriggerSrv>::SharedPtr polishing_start_client_;
    rclcpp::Client<TriggerSrv>::SharedPtr polishing_end_client_;

    std::mutex command_mutex_;
    bool robot_ready_printed_ = false;
    std::string package_path_;

private:
    void PrintRobotInitState()
    {
        if (is_roboInit()) {
            if (!robot_ready_printed_) {
                RCLCPP_INFO(
                    this->get_logger(),
                    "Robot current position received. Command service is now available."
                );
                robot_ready_printed_ = true;
            }
        }
        else {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                3000,
                "Waiting for current position..."
            );
        }
    }

    std::string NormalizeCommandMode(const std::string& input)
    {
        std::string mode = input;

        std::transform(
            mode.begin(),
            mode.end(),
            mode.begin(),
            [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            }
        );

        if (mode == "i" || mode == "idling") {
            return "Idling";
        }
        else if (mode == "p" || mode == "ptp") {
            return "PTP";
        }
        else if (mode == "p9" || mode == "ptp9d" || mode == "ptp_9d" || mode == "ptp_force") {
            return "PTP9D";
        }
        else if (mode == "ptp9d_stream_start" || mode == "ptp9dstreamstart") {
            return "PTP9D_STREAM_START";
        }
        else if (mode == "ptp9d_stream_append" || mode == "ptp9dstreamappend") {
            return "PTP9D_STREAM_APPEND";
        }
        else if (mode == "ptp9d_stream_stop" || mode == "ptp9dstreamstop") {
            return "PTP9D_STREAM_STOP";
        }
        else if (mode == "ptp9d_stream_set_force" || mode == "ptp9dstreamsetforce") {
            return "PTP9D_STREAM_SET_FORCE";
        }
        else if (mode == "t" || mode == "txtload") {
            return "TxtLoad";
        }
        else if (mode == "g" || mode == "guiding") {
            return "Guiding";
        }
        else if (mode == "j" || mode == "joystick") {
            return "Joystick";
        }
        else if (mode == "jf" ||
                 mode == "joystick_force" ||
                 mode == "joystickforce") {
            return "Joystick_force";
        }
        else if (mode == "position") {
            return "Position";
        }
        else if (mode == "force") {
            return "Force";
        }

        return "Unknown";
    }

    bool MakePTPMotion(
        const std::vector<double>& target_pose,
        YMatrix& loaded_motion,
        std::string& message)
    {
        if (target_pose.size() != 6) {
            message =
                "PTP command requires target_pose with exactly 6 values: "
                "[x, y, z, rx, ry, rz].";
            return false;
        }

        loaded_motion.resize(1, 6);

        for (size_t i = 0; i < 6; ++i) {
            loaded_motion[0][i] = target_pose[i];
        }

        message = "PTP motion generated from service request.";
        return true;
    }

    bool MakePTP9DMotion(
        const std::vector<double>& target_pose,
        YMatrix& loaded_motion,
        std::string& message)
    {
        // One or more waypoints, each [x, y, z, rx, ry, rz, fx, fy, fz]
        // concatenated back-to-back. A single 9-value request keeps the
        // old one-point-per-call behavior; a longer multiple-of-9 request
        // is streamed as one continuously blended multi-point PTP9D move
        // (see PTP9D_command_gen) instead of stopping fully at each point.
        if (target_pose.empty() || target_pose.size() % 9 != 0) {
            message =
                "PTP9D command requires target_pose as one or more waypoints "
                "of 9 values each: [x, y, z, rx, ry, rz, fx, fy, fz].";
            return false;
        }

        const size_t num_waypoints = target_pose.size() / 9;
        loaded_motion.resize(static_cast<int>(num_waypoints), 9);

        for (size_t r = 0; r < num_waypoints; ++r) {
            for (size_t i = 0; i < 9; ++i) {
                loaded_motion[static_cast<int>(r)][i] = target_pose[r * 9 + i];
            }
        }

        message = "PTP9D motion generated from service request (" +
            std::to_string(num_waypoints) +
            (num_waypoints == 1 ? " waypoint)." : " waypoints).");
        return true;
    }

    bool MakeTxtLoadMotion(
        const std::string& requested_load_file,
        YMatrix& loaded_motion,
        std::string& load_file_type,
        std::string& message)
    {
        std::string load_file_path;

        if (requested_load_file.empty()) {
            load_file_path = package_path_ + "/" + load_file_;
        }
        else {
            if (!requested_load_file.empty() && requested_load_file[0] == '/') {
                load_file_path = requested_load_file;
            }
            else {
                load_file_path = package_path_ + "/" + requested_load_file;
            }
        }

        try {
            auto loaded_data = YMatrix::loadFromFile(load_file_path);

            loaded_motion.resize(loaded_data.rows(), loaded_data.cols());
            loaded_motion = loaded_data;

            if (loaded_motion.rows() < 2) {
                message = "TxtLoad failed: not enough positions to blend motion.";
                return false;
            }

            load_file_type = ResolveLoadFileType(load_file_path, loaded_motion);
            if (load_file_type.empty()) {
                message =
                    "TxtLoad failed: cannot infer command type from file path or column count: " +
                    load_file_path;
                return false;
            }
        }
        catch (const std::exception& e) {
            message = std::string("TxtLoad failed: ") + e.what();
            return false;
        }

        message =
            "TxtLoad motion loaded from: " + load_file_path +
            " as " + load_file_type;
        return true;
    }

    std::string ResolveLoadFileType(
        const std::string& load_file_path,
        const YMatrix& loaded_motion)
    {
        std::string normalized_path = load_file_path;
        std::replace(normalized_path.begin(), normalized_path.end(), '\\', '/');

        const std::string lower_path = ToLower(normalized_path);

        if (lower_path.find("cmd_continue6d") != std::string::npos) {
            return "cmd_continue6D";
        }
        if (lower_path.find("cmd_continue9d") != std::string::npos) {
            return "cmd_continue9D";
        }
        if (lower_path.find("cmd_6d") != std::string::npos) {
            return "cmd_6D";
        }
        if (lower_path.find("cmd_9d") != std::string::npos) {
            return "cmd_9D";
        }

        if (loaded_motion.cols() == 12) {
            return "cmd_9D";
        }
        if (loaded_motion.cols() == 9) {
            return "cmd_6D";
        }

        return "";
    }

    std::string ToLower(const std::string& input)
    {
        std::string lower = input;

        std::transform(
            lower.begin(),
            lower.end(),
            lower.begin(),
            [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            }
        );

        return lower;
    }

    bool CallTriggerService(
        const rclcpp::Client<TriggerSrv>::SharedPtr& client,
        const std::string& service_name,
        std::string& result_message,
        const std::chrono::milliseconds service_wait_timeout = std::chrono::milliseconds(1000),
        const std::chrono::seconds response_timeout = std::chrono::seconds(3))
    {
        if (!client->wait_for_service(service_wait_timeout)) {
            result_message =
                "Trigger service is not available: " + service_name;

            RCLCPP_WARN(
                this->get_logger(),
                "%s",
                result_message.c_str()
            );

            return false;
        }

        auto request = std::make_shared<TriggerSrv::Request>();
        auto future = client->async_send_request(request);

        const auto status = future.wait_for(response_timeout);

        if (status != std::future_status::ready) {
            result_message =
                "Trigger service response timeout: " + service_name;

            RCLCPP_WARN(
                this->get_logger(),
                "%s",
                result_message.c_str()
            );

            return false;
        }

        auto response = future.get();

        result_message =
            service_name + " response: success=" +
            std::string(response->success ? "true" : "false") +
            ", message=" + response->message;

        if (response->success) {
            RCLCPP_INFO(
                this->get_logger(),
                "%s",
                result_message.c_str()
            );
        }
        else {
            RCLCPP_WARN(
                this->get_logger(),
                "%s",
                result_message.c_str()
            );
        }

        return response->success;
    }

    void CommandServiceCallback(
        const std::shared_ptr<SingleArmCommandSrv::Request> request,
        std::shared_ptr<SingleArmCommandSrv::Response> response)
    {
        std::lock_guard<std::mutex> lock(command_mutex_);

        if (!is_roboInit()) {
            response->success = false;
            response->message =
                "Robot is not initialized yet. Waiting for current position.";

            RCLCPP_WARN(
                this->get_logger(),
                "%s",
                response->message.c_str()
            );
            return;
        }

        const std::string command_mode =
            NormalizeCommandMode(request->command_mode);

        if (command_mode == "Unknown") {
            response->success = false;
            response->message =
                "Unknown command_mode. Use one of: Idling, PTP, PTP9D, "
                "PTP9D_STREAM_START, PTP9D_STREAM_APPEND, PTP9D_STREAM_STOP, "
                "PTP9D_STREAM_SET_FORCE, TxtLoad, Guiding, Joystick, "
                "Joystick_force, Position, Force.";

            RCLCPP_ERROR(
                this->get_logger(),
                "%s",
                response->message.c_str()
            );
            return;
        }

        // PTP9D streaming commands have a different call/response shape
        // than the rest (fast, non-blocking; success/message comes
        // straight back from robot_command rather than through the
        // generic loaded_motion+sendCommand path below), so handle them
        // here and return early.
        if (command_mode == "PTP9D_STREAM_START") {
            std::string stream_message;
            const bool ok = startStream(request->target_velocity, stream_message);
            response->success = ok;
            response->message = "Command executed: " + command_mode + ". " + stream_message;
            RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
            return;
        }
        if (command_mode == "PTP9D_STREAM_APPEND") {
            YMatrix stream_motion(1, 9);
            std::string stream_message;
            if (!MakePTP9DMotion(request->target_pose, stream_motion, stream_message)) {
                response->success = false;
                response->message = stream_message;
                RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
                return;
            }
            const bool ok = appendStream(stream_motion, stream_message);
            response->success = ok;
            response->message = "Command executed: " + command_mode + ". " + stream_message;
            if (!ok) {
                RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
            }
            return;
        }
        if (command_mode == "PTP9D_STREAM_STOP") {
            stopStream();
            response->success = true;
            response->message = "Command executed: " + command_mode + ". PTP9D stream stopped.";
            RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
            return;
        }
        if (command_mode == "PTP9D_STREAM_SET_FORCE") {
            if (request->target_pose.size() != 3) {
                response->success = false;
                response->message =
                    "PTP9D_STREAM_SET_FORCE requires target_pose with exactly "
                    "3 values: [fx, fy, fz].";
                RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
                return;
            }
            std::string stream_message;
            const bool ok = setStreamForce(
                request->target_pose[0], request->target_pose[1], request->target_pose[2],
                stream_message);
            response->success = ok;
            response->message = "Command executed: " + command_mode + ". " + stream_message;
            if (!ok) {
                RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
            }
            return;
        }

        YMatrix loaded_motion(1, 6);
        std::string load_file_type;
        std::string message;

        if (command_mode == "PTP") {
            if (!MakePTPMotion(request->target_pose, loaded_motion, message)) {
                response->success = false;
                response->message = message;

                RCLCPP_ERROR(
                    this->get_logger(),
                    "%s",
                    response->message.c_str()
                );
                return;
            }

            setPTPTargetVelocity(
                request->target_velocity > 0.0
                    ? request->target_velocity
                    : config_.ptp_target_velocity);
        }
        else if (command_mode == "PTP9D") {
            if (!MakePTP9DMotion(request->target_pose, loaded_motion, message)) {
                response->success = false;
                response->message = message;

                RCLCPP_ERROR(
                    this->get_logger(),
                    "%s",
                    response->message.c_str()
                );
                return;
            }

            setPTPTargetVelocity(
                request->target_velocity > 0.0
                    ? request->target_velocity
                    : config_.ptp_target_velocity);
        }
        else if (command_mode == "TxtLoad") {
            if (!MakeTxtLoadMotion(request->load_file, loaded_motion, load_file_type, message)) {
                response->success = false;
                response->message = message;

                RCLCPP_ERROR(
                    this->get_logger(),
                    "%s",
                    response->message.c_str()
                );
                return;
            }

            setLoadFileType(load_file_type);
        }
        else if (command_mode == "Idling") {
            message = "Idling command requested.";
        }
        else if (command_mode == "Guiding") {
            message = "Guiding command requested.";
        }
        else {
            message = command_mode + " command requested.";
        }

        RCLCPP_INFO(
            this->get_logger(),
            "Service command received: %s",
            command_mode.c_str()
        );

        std::string polishing_start_message;
        std::string polishing_end_message;

        if (command_mode == "TxtLoad") {
            RCLCPP_INFO(
                this->get_logger(),
                "TxtLoad started. Calling /polishing_removal_node/start ..."
            );

            CallTriggerService(
                polishing_start_client_,
                "/polishing_removal_node/start",
                polishing_start_message
            );
        }

        try {
            sendCommand(command_mode, loaded_motion);
        }
        catch (const std::exception& e) {
            response->success = false;
            response->message =
                "Command failed: " + command_mode + ". " + e.what();
            RCLCPP_ERROR(
                this->get_logger(),
                "%s",
                response->message.c_str()
            );
            return;
        }

        if (command_mode == "TxtLoad") {
            RCLCPP_INFO(
                this->get_logger(),
                "TxtLoad finished. Calling /polishing_removal_node/end ..."
            );

            CallTriggerService(
                polishing_end_client_,
                "/polishing_removal_node/end",
                polishing_end_message
            );
        }

        response->success = true;
        response->message =
            "Command executed: " + command_mode + ". " + message;

        if (command_mode == "TxtLoad") {
            response->message +=
                " | polishing_start: " + polishing_start_message +
                " | polishing_end: " + polishing_end_message;
        }

        RCLCPP_INFO(
            this->get_logger(),
            "%s",
            response->message.c_str()
        );
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto cmd_node = std::make_shared<singleArm_cmd>("singleArm_cmd");

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(cmd_node);
    executor.spin();

    rclcpp::shutdown();

    return 0;
}
