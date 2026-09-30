#include "Y2RobMotion/robot_command.hpp"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <thread>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>

// ------------------------------------------------------------
// robot_command::LoadTxtToYMatrix (private static)
// ------------------------------------------------------------
YMatrix robot_command::LoadTxtToYMatrix(const std::string& path, int expected_cols)
{
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    std::vector<std::vector<double>> rows;
    std::string line;

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        if (!line.empty() && line[0] == '#') continue;

        // Support comma-separated too
        for (auto& ch : line) {
            if (ch == ',') ch = ' ';
        }

        std::istringstream iss(line);
        std::vector<double> vals;
        double v;
        while (iss >> v) vals.push_back(v);

        if (vals.empty()) continue;

        if ((int)vals.size() != expected_cols) {
            std::ostringstream oss;
            oss << "Invalid column count in " << path
                << " (got " << vals.size()
                << ", expected " << expected_cols << ")";
            throw std::runtime_error(oss.str());
        }

        rows.push_back(std::move(vals));
    }

    if (rows.empty()) {
        throw std::runtime_error("No valid data rows in file: " + path);
    }

    YMatrix mat((int)rows.size(), expected_cols);
    for (int r = 0; r < (int)rows.size(); ++r) {
        for (int c = 0; c < expected_cols; ++c) {
            mat[r][c] = rows[r][c];
        }
    }
    return mat;
}

// ------------------------------------------------------------

robot_command::robot_command(rclcpp::Node* node,
                             const std::string& robot_name,
                             double control_period,
                             const PathGenParam& pg_param_)
: current_position(6, 0.0)
, pg_param(pg_param_)          // copy
, robot_name_(robot_name)
, node_(node)
, control_period_(control_period)
{
    // Topic names
    const std::string current_p_sub_topic  = robot_name_ + "/currentP";
    const std::string cmd_motion_pub_topic = robot_name_ + "/cmdMotion";
    const std::string cmd_pub_topic        = robot_name_ + "/cmdMode";

    const std::string state_pub_topic      = robot_name_ + "/cmdState";

    // Subscriber
    current_p_sub_ = node_->create_subscription<std_msgs::msg::Float64MultiArray>(
        current_p_sub_topic, 1,
        std::bind(&robot_command::currentPCallback, this, std::placeholders::_1));

    // Publishers
    cmd_motion_pub_ = node_->create_publisher<std_msgs::msg::Float64MultiArray>(cmd_motion_pub_topic, 1);
    cmd_pub_        = node_->create_publisher<std_msgs::msg::String>(cmd_pub_topic, 10);

    state_pub_      = node_->create_publisher<std_msgs::msg::String>(state_pub_topic, 10);

    RCLCPP_INFO(node_->get_logger(), "UrCmd node initialized for robot: %s", robot_name_.c_str());
    RCLCPP_INFO(node_->get_logger(), "State topic  : %s  (String)", state_pub_topic.c_str());
}

robot_command::~robot_command()
{
    stopStream();
}

void robot_command::publish_state(const std::string& s)
{
    if (!state_pub_) return;
    std_msgs::msg::String m;
    m.data = s;
    state_pub_->publish(m);
}

void robot_command::setLoadFileType(const std::string& load_file_type)
{
    pg_param.loadFileType = load_file_type;
}

void robot_command::setPTPTargetVelocity(double target_velocity)
{
    pg_param.ptp_target_velocity = target_velocity;
}

void robot_command::currentPCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if (!robot_init) {
        robot_init = true;
        RCLCPP_INFO(node_->get_logger(), "CurrentP received! Robot initialized.");
    }

    for (size_t i = 0; i < 6 && i < msg->data.size(); ++i) {
        current_position[i] = msg->data[i];
    }
}

void robot_command::sendCommand(const std::string& command, const YMatrix& loaded_motion)
{
    if (command == "PTP") {
        publish_state("ptp started");
        PTP_command_gen(loaded_motion);
        publish_state("trajectory transfer done");
    }
    else if (command == "PTP9D") {
        publish_state("ptp9d started");
        PTP9D_command_gen(loaded_motion);
        publish_state("trajectory transfer done");
    }
    else if (command == "TxtLoad") {
        publish_state("txtload started");
        TxtLoad_command_gen(loaded_motion);
        publish_state("trajectory transfer done");
    }
    else if (command == "Idling" ||
             command == "Guiding" ||
             command == "Joystick" ||
             command == "Joystick_force" ||
             command == "Position" ||
             command == "Force") {
        RCLCPP_INFO(node_->get_logger(), "%s mode activated", command.c_str());
        cmd_msg_.data = command;
        cmd_pub_->publish(cmd_msg_);
        publish_state("mode changed: " + command);
    }
    else {
        RCLCPP_ERROR(node_->get_logger(), "Unknown command: %s", command.c_str());
    }
}

void robot_command::PTP_command_gen(const YMatrix& loaded_motion)
{
    cmd_msg_.data = "Position";
    cmd_pub_->publish(cmd_msg_);

    YMatrix position = {
        {current_position[0], current_position[1], current_position[2],
         current_position[3], current_position[4], current_position[5]},
        {loaded_motion[0][0], loaded_motion[0][1], loaded_motion[0][2],
         DegreeToRadian(loaded_motion[0][3]),
         DegreeToRadian(loaded_motion[0][4]),
         DegreeToRadian(loaded_motion[0][5])}
    };

    std::vector<double> velocity      = {0.0, pg_param.ptp_target_velocity};
    std::vector<double> ang_velocity  = {0.0, 0.0};
    std::vector<double> holding_time  = {0.0, 0.0};

    MotionBlender6D blender(position, velocity, ang_velocity, holding_time,
                            DegreeToRadian(pg_param.angularVelocityLimit),
                            pg_param.startingTime, pg_param.lastRestingTime,
                            pg_param.accelerationTime, control_period_);

    YMatrix blended_motion = blender.blendMotion(pg_param.defualt_travelTime);

    printf("Path Transferring...\n");

    auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(control_period_)
    );
    auto next_time = std::chrono::steady_clock::now();

    for (const auto& pos : blended_motion) {
        if (!rclcpp::ok()) {
            RCLCPP_WARN(node_->get_logger(), "ROS is not ok, stopping motion transfer.");
            return;
        }

        cmd_motion_msg_.data.clear();
        for (const auto& v : pos) cmd_motion_msg_.data.push_back(v);
        cmd_motion_pub_->publish(cmd_motion_msg_);

        next_time += period_ns;
        std::this_thread::sleep_until(next_time);
    }

    printf("Path Transferred.\n");
}

void robot_command::PTP9D_command_gen(const YMatrix& loaded_motion)
{
    // One or more 9D (pose + force) waypoints, blended the same way as the
    // cmd_9D TxtLoad path (MotionBlender9D) but driven directly from a
    // service request instead of a file -- avoids needing a shared
    // filesystem between the inference PC and this control PC, and lets
    // the caller re-target every point without a round trip through disk.
    // Unlike PTP_command_gen, this switches to Force mode: the blended
    // fx/fy/fz ride along through the same acceleration/deceleration
    // profiling as position, so force ramps in rather than stepping.
    //
    // Multiple waypoints in one call are blended into a single continuous
    // motion (one accel-decel profile spanning all of them, exactly as
    // MotionBlender9D already does for TxtLoad_command_gen) instead of
    // decelerating to a full stop at each one -- this is what lets the
    // caller stream a short lookahead segment per service call for smooth
    // motion while still keeping each call bounded/verifiable.
    cmd_msg_.data = "Force";
    cmd_pub_->publish(cmd_msg_);

    const int num_waypoints = loaded_motion.rows();

    YMatrix position(num_waypoints + 1, 9);
    position[0][0] = current_position[0];
    position[0][1] = current_position[1];
    position[0][2] = current_position[2];
    position[0][3] = current_position[3];
    position[0][4] = current_position[4];
    position[0][5] = current_position[5];
    position[0][6] = 0.0;
    position[0][7] = 0.0;
    position[0][8] = 0.0;

    for (int i = 0; i < num_waypoints; ++i) {
        position[i + 1][0] = loaded_motion[i][0];
        position[i + 1][1] = loaded_motion[i][1];
        position[i + 1][2] = loaded_motion[i][2];
        position[i + 1][3] = DegreeToRadian(loaded_motion[i][3]);
        position[i + 1][4] = DegreeToRadian(loaded_motion[i][4]);
        position[i + 1][5] = DegreeToRadian(loaded_motion[i][5]);
        position[i + 1][6] = loaded_motion[i][6];
        position[i + 1][7] = loaded_motion[i][7];
        position[i + 1][8] = loaded_motion[i][8];
    }

    std::vector<double> velocity(num_waypoints + 1, pg_param.ptp_target_velocity);
    velocity[0] = 0.0;
    std::vector<double> ang_velocity(num_waypoints + 1, 0.0);
    std::vector<double> holding_time(num_waypoints + 1, 0.0);

    MotionBlender9D blender(position, velocity, ang_velocity, holding_time,
                            DegreeToRadian(pg_param.angularVelocityLimit),
                            pg_param.startingTime, pg_param.lastRestingTime,
                            pg_param.accelerationTime, control_period_);

    YMatrix blended_motion = blender.blendMotion(pg_param.defualt_travelTime);

    printf("Path Transferring...\n");

    auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(control_period_)
    );
    auto next_time = std::chrono::steady_clock::now();

    for (const auto& pos : blended_motion) {
        if (!rclcpp::ok()) {
            RCLCPP_WARN(node_->get_logger(), "ROS is not ok, stopping motion transfer.");
            return;
        }

        cmd_motion_msg_.data.clear();
        for (const auto& v : pos) cmd_motion_msg_.data.push_back(v);
        cmd_motion_pub_->publish(cmd_motion_msg_);

        next_time += period_ns;
        std::this_thread::sleep_until(next_time);
    }

    printf("Path Transferred.\n");
}

// ------------------------------------------------------------
// PTP9D streaming: queued, non-stop-start continuous motion.
// See robot_command.hpp for the design rationale.
// ------------------------------------------------------------

bool robot_command::startStream(double target_velocity, std::string& message)
{
    if (stream_running_.load()) {
        message = "PTP9D stream already running.";
        return true;
    }

    stream_target_velocity_ = target_velocity > 0.0
        ? target_velocity
        : pg_param.ptp_target_velocity;

    {
        std::lock_guard<std::mutex> lock(stream_mutex_);
        stream_queue_.clear();
        // Seed the running "current command" from the live measured pose
        // so the first published point is a no-op step, not a jump.
        stream_current_[0] = current_position[0];
        stream_current_[1] = current_position[1];
        stream_current_[2] = current_position[2];
        stream_current_[3] = current_position[3];
        stream_current_[4] = current_position[4];
        stream_current_[5] = current_position[5];
        stream_current_[6] = 0.0;
        stream_current_[7] = 0.0;
        stream_current_[8] = 0.0;
    }
    {
        std::lock_guard<std::mutex> flock(stream_force_mutex_);
        stream_force_target_ = {0.0, 0.0, 0.0};
    }

    cmd_msg_.data = "Force";
    cmd_pub_->publish(cmd_msg_);

    stream_running_.store(true);
    stream_thread_ = std::thread(&robot_command::streamLoop, this);

    message = "PTP9D stream started.";
    return true;
}

bool robot_command::appendStreamWaypoints(const YMatrix& loaded_motion, std::string& message)
{
    if (!stream_running_.load()) {
        message = "PTP9D stream is not running -- call PTP9D_STREAM_START first.";
        return false;
    }

    const int num_waypoints = loaded_motion.rows();

    std::lock_guard<std::mutex> lock(stream_mutex_);
    if (stream_queue_.size() + static_cast<size_t>(num_waypoints) > kStreamMaxQueue) {
        message =
            "PTP9D stream queue is full (caller is appending faster than the "
            "robot can consume) -- dropping this append.";
        return false;
    }

    for (int i = 0; i < num_waypoints; ++i) {
        std::array<double, 9> wp{};
        wp[0] = loaded_motion[i][0];
        wp[1] = loaded_motion[i][1];
        wp[2] = loaded_motion[i][2];
        wp[3] = DegreeToRadian(loaded_motion[i][3]);
        wp[4] = DegreeToRadian(loaded_motion[i][4]);
        wp[5] = DegreeToRadian(loaded_motion[i][5]);
        wp[6] = loaded_motion[i][6];
        wp[7] = loaded_motion[i][7];
        wp[8] = loaded_motion[i][8];
        stream_queue_.push_back(wp);
    }

    message = "Appended " + std::to_string(num_waypoints) + " waypoint(s) to PTP9D stream.";
    return true;
}

void robot_command::stopStream()
{
    stream_running_.store(false);
    if (stream_thread_.joinable()) {
        stream_thread_.join();
    }
    std::lock_guard<std::mutex> lock(stream_mutex_);
    stream_queue_.clear();
    std::lock_guard<std::mutex> flock(stream_force_mutex_);
    stream_force_target_ = {0.0, 0.0, 0.0};
}

bool robot_command::setStreamForce(double fx, double fy, double fz, std::string& message)
{
    if (!stream_running_.load()) {
        message = "PTP9D stream is not running -- call PTP9D_STREAM_START first.";
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(stream_force_mutex_);
        stream_force_target_ = {fx, fy, fz};
    }
    message = "Force target updated.";
    return true;
}

void robot_command::streamLoop()
{
    const double max_pos_step = stream_target_velocity_ * control_period_;               // mm/tick safety cap
    const double max_ang_step = DegreeToRadian(pg_param.angularVelocityLimit) * control_period_; // rad/tick safety cap
    const double max_force_step = kStreamForceRateNs * control_period_;                  // N/tick safety cap
    const double gain_dt = kStreamGainHz * control_period_;                              // low-pass tracking fraction/tick

    auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(control_period_)
    );
    auto next_time = std::chrono::steady_clock::now();

    // "Bracket" playback state: interpolate between seg_a (departure point,
    // starts at the live pose) and seg_b (the next queued point) as elapsed
    // time advances over seg_duration, instead of waiting for stream_current_
    // to geometrically reach a point before moving on.
    std::array<double, 9> seg_a = stream_current_;
    std::array<double, 9> seg_b{};
    bool have_seg_b = false;
    double seg_duration = 0.0;
    double seg_elapsed = 0.0;

    while (stream_running_.load()) {
        if (!have_seg_b) {
            std::lock_guard<std::mutex> lock(stream_mutex_);
            if (!stream_queue_.empty()) {
                seg_b = stream_queue_.front();
                stream_queue_.pop_front();
                have_seg_b = true;
                seg_elapsed = 0.0;

                double dx = seg_b[0] - seg_a[0];
                double dy = seg_b[1] - seg_a[1];
                double dz = seg_b[2] - seg_a[2];
                double pos_dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                double drx = seg_b[3] - seg_a[3];
                double dry = seg_b[4] - seg_a[4];
                double drz = seg_b[5] - seg_a[5];
                double ang_dist = std::sqrt(drx * drx + dry * dry + drz * drz);
                double lin_time = pos_dist / std::max(1e-6, stream_target_velocity_);
                double ang_time = ang_dist / std::max(1e-6, DegreeToRadian(pg_param.angularVelocityLimit));
                seg_duration = std::max({lin_time, ang_time, kStreamMinSegTime});
            }
        }

        std::array<double, 9> instant_target{};
        if (have_seg_b) {
            seg_elapsed += control_period_;
            double t = std::min(1.0, seg_elapsed / seg_duration);
            // Position/orientation only -- force (6..8) is deliberately not
            // part of this bracket, see setStreamForce/stream_force_target_
            // below: it tracks the latest force decision immediately
            // instead of waiting out however much position lookahead
            // happens to be queued.
            for (int i = 0; i < 6; ++i) {
                instant_target[i] = seg_a[i] + (seg_b[i] - seg_a[i]) * t;
            }
            if (t >= 1.0) {
                // Segment fully interpolated -- shift the bracket forward.
                // The *next* loop iteration pulls a new seg_b (if queued),
                // continuing from here with no velocity discontinuity since
                // instant_target already equals seg_b at t=1.
                seg_a = seg_b;
                have_seg_b = false;
            }
        } else {
            // Queue is empty -- hold at the last departure point (safe
            // idle, never extrapolates past the last known-good target).
            for (int i = 0; i < 6; ++i) instant_target[i] = seg_a[i];
        }
        {
            std::lock_guard<std::mutex> flock(stream_force_mutex_);
            instant_target[6] = stream_force_target_[0];
            instant_target[7] = stream_force_target_[1];
            instant_target[8] = stream_force_target_[2];
        }

        // Rate-capped low-pass tracking toward instant_target (UR servoj's
        // lookahead_time+gain shape): normally just an exponential approach,
        // smoothing over per-sample noise in the queued points, but still
        // hard-capped per axis so a single bad/outlier target can't produce
        // an oversized step.
        for (int i = 0; i < 3; ++i) {
            double delta = gain_dt * (instant_target[i] - stream_current_[i]);
            delta = std::clamp(delta, -max_pos_step, max_pos_step);
            stream_current_[i] += delta;
        }
        for (int i = 3; i < 6; ++i) {
            double delta = gain_dt * (instant_target[i] - stream_current_[i]);
            delta = std::clamp(delta, -max_ang_step, max_ang_step);
            stream_current_[i] += delta;
        }
        for (int i = 6; i < 9; ++i) {
            double delta = gain_dt * (instant_target[i] - stream_current_[i]);
            delta = std::clamp(delta, -max_force_step, max_force_step);
            stream_current_[i] += delta;
        }

        cmd_motion_msg_.data.clear();
        for (double v : stream_current_) cmd_motion_msg_.data.push_back(v);
        cmd_motion_pub_->publish(cmd_motion_msg_);

        next_time += period_ns;
        std::this_thread::sleep_until(next_time);
    }
}

void robot_command::TxtLoad_command_gen(const YMatrix& loaded_motion)
{
    // ============================================================
    // cmd_6D
    // ============================================================
    if (pg_param.loadFileType == "cmd_6D") {

        cmd_msg_.data = "Position";
        cmd_pub_->publish(cmd_msg_);

        printf("\033[32mLoading 6D motion profile from %s...\033[0m\n", pg_param.loadFileType.c_str());
        loaded_motion.print();

        if (loaded_motion.cols() != 9) {
            printf("\033[33mthere is not proper setup inside of cmd_6D !! \033[0m\n");
            return;
        }

        YMatrix loaded_position(loaded_motion.rows() + 2, 6);
        std::vector<double> loaded_velocity(loaded_motion.rows() + 2, 0.0);
        std::vector<double> loaded_ang_velocity(loaded_motion.rows() + 2, 0.0);
        std::vector<double> loaded_holding_time(loaded_motion.rows() + 2, 0.0);

        loaded_position.insert(1, 0, loaded_motion.extract(0, 0, loaded_motion.rows(), 6));
        auto loaded_velocity_matrix     = loaded_motion.extract(0, 6, loaded_motion.rows(), 1);
        auto loaded_ang_velocity_matrix = loaded_motion.extract(0, 7, loaded_motion.rows(), 1);
        auto loaded_holding_time_matrix = loaded_motion.extract(0, 8, loaded_motion.rows(), 1);

        for (int i = 0; i < (int)loaded_velocity_matrix.rows(); i++) {
            loaded_velocity[i + 1]      = loaded_velocity_matrix[i][0];
            loaded_ang_velocity[i + 1]  = loaded_ang_velocity_matrix[i][0];
            loaded_holding_time[i + 1]  = loaded_holding_time_matrix[i][0];
        }

        // Convert degrees to radians (wx,wy,wz)
        for (int i = 0; i < (int)loaded_position.rows(); i++) {
            for (int j = 0; j < (int)loaded_position.cols(); j++) {
                if (j >= 3) loaded_position[i][j] = DegreeToRadian(loaded_position[i][j]);
            }
        }

        // Insert start & end pose
        loaded_position.insert(0, 0, {{current_position[0], current_position[1], current_position[2],
                                       current_position[3], current_position[4], current_position[5]}});
        loaded_position.insert(loaded_position.rows() - 1, 0, {{current_position[0], current_position[1], current_position[2],
                                                                current_position[3], current_position[4], current_position[5]}});

        printf("Loaded Position:\n");
        loaded_position.print();

        // Insert start & end (velocity & angular_velocity & holding_time)
        loaded_velocity[1] = pg_param.initialTransferSpeed;
        loaded_velocity.back() = pg_param.initialTransferSpeed;

        loaded_ang_velocity[1] = 0.0;
        loaded_ang_velocity.back() = 0.0;

        loaded_holding_time[1] = 0.0;
        loaded_holding_time.back() = 0.0;

        MotionBlender6D blender(loaded_position, loaded_velocity, loaded_ang_velocity, loaded_holding_time,
                                DegreeToRadian(pg_param.angularVelocityLimit),
                                pg_param.startingTime, pg_param.lastRestingTime,
                                pg_param.accelerationTime, control_period_);

        YMatrix blended_motion = blender.blendMotion(pg_param.defualt_travelTime);
        printf("Blended motion generated\n");

        printf("Path Transferring...\n");

        auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(control_period_)
        );
        auto next_time = std::chrono::steady_clock::now();

        for (const auto& pos : blended_motion) {
            if (!rclcpp::ok()) {
                RCLCPP_WARN(node_->get_logger(), "ROS is not ok, stopping motion transfer.");
                return;
            }

            cmd_motion_msg_.data.clear();
            for (const auto& v : pos) cmd_motion_msg_.data.push_back(v);
            cmd_motion_pub_->publish(cmd_motion_msg_);

            next_time += period_ns;
            std::this_thread::sleep_until(next_time);
        }

        printf("Path Transferred.\n");
    }

    // ============================================================
    // cmd_9D  (generate + save cmd_continue9D)
    // ============================================================
    else if (pg_param.loadFileType == "cmd_9D") {

        cmd_msg_.data = "Force";
        cmd_pub_->publish(cmd_msg_);

        printf("\033[32mLoading 9D motion profile from %s...\033[0m\n", pg_param.loadFileType.c_str());
        loaded_motion.print();

        if (loaded_motion.cols() != 12) {
            printf("\033[33mthere is not proper setup inside of cmd_9D !! \033[0m\n");
            return;
        }

        YMatrix loaded_position(loaded_motion.rows() + 2, 9);
        std::vector<double> loaded_velocity(loaded_motion.rows() + 2, 0.0);
        std::vector<double> loaded_ang_velocity(loaded_motion.rows() + 2, 0.0);
        std::vector<double> loaded_holding_time(loaded_motion.rows() + 2, 0.0);

        loaded_position.insert(1, 0, loaded_motion.extract(0, 0, loaded_motion.rows(), 9));
        auto loaded_velocity_matrix     = loaded_motion.extract(0, 9,  loaded_motion.rows(), 1);
        auto loaded_ang_velocity_matrix = loaded_motion.extract(0, 10, loaded_motion.rows(), 1);
        auto loaded_holding_time_matrix = loaded_motion.extract(0, 11, loaded_motion.rows(), 1);

        for (int i = 0; i < (int)loaded_velocity_matrix.rows(); i++) {
            loaded_velocity[i + 1]      = loaded_velocity_matrix[i][0];
            loaded_ang_velocity[i + 1]  = loaded_ang_velocity_matrix[i][0];
            loaded_holding_time[i + 1]  = loaded_holding_time_matrix[i][0];
        }

        // Convert degrees to radians only for wx,wy,wz (3..5)
        for (int i = 0; i < (int)loaded_position.rows(); i++) {
            for (int j = 0; j < (int)loaded_position.cols(); j++) {
                if (j >= 3 && j < 6) loaded_position[i][j] = DegreeToRadian(loaded_position[i][j]);
            }
        }

        // Insert start & end pose (pose part; force part is already included in 9 cols)
        loaded_position.insert(0, 0, {{current_position[0], current_position[1], current_position[2],
                                       current_position[3], current_position[4], current_position[5]}});
        loaded_position.insert(loaded_position.rows() - 1, 0, {{current_position[0], current_position[1], current_position[2],
                                                                current_position[3], current_position[4], current_position[5]}});

        printf("Loaded Position:\n");
        loaded_position.print();

        loaded_velocity[1] = pg_param.initialTransferSpeed;
        loaded_velocity.back() = pg_param.initialTransferSpeed;

        loaded_ang_velocity[1] = 0.0;
        loaded_ang_velocity.back() = 0.0;

        loaded_holding_time[1] = 0.0;
        loaded_holding_time.back() = 0.0;

        MotionBlender9D blender(loaded_position, loaded_velocity, loaded_ang_velocity, loaded_holding_time,
                                DegreeToRadian(pg_param.angularVelocityLimit),
                                pg_param.startingTime, pg_param.lastRestingTime,
                                pg_param.accelerationTime, control_period_);

        YMatrix blended_motion = blender.blendMotion(pg_param.defualt_travelTime);
        printf("Blended motion generated\n");

        // Save cmd_continue9D.txt
        std::filesystem::path source_path(__FILE__);
        std::string save_path = source_path.parent_path().string() + "/../txtcmd/cmd_continue9D.txt";
        blended_motion.saveToFile(save_path);
        RCLCPP_INFO(node_->get_logger(), "9D data was saved to cmd_continue9D: %s", save_path.c_str());

        // Transfer to robot
        printf("Path Transferring...\n");

        auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(control_period_)
        );
        auto next_time = std::chrono::steady_clock::now();

        for (const auto& row : blended_motion) {
            if (!rclcpp::ok()) {
                RCLCPP_WARN(node_->get_logger(), "ROS is not ok, stopping motion transfer.");
                return;
            }

            cmd_motion_msg_.data.clear();
            for (const auto& v : row) cmd_motion_msg_.data.push_back(v);
            cmd_motion_pub_->publish(cmd_motion_msg_);

            next_time += period_ns;
            std::this_thread::sleep_until(next_time);
        }

        printf("Path Transferred.\n");
    }

    // ============================================================
    // cmd_continue6D (not implemented)
    // ============================================================
    else if (pg_param.loadFileType == "cmd_continue6D") {
        RCLCPP_WARN(node_->get_logger(), "cmd_continue6D motion profile not implemented yet");
    }

    // ============================================================
    // cmd_continue9D (LOAD + CONNECT + REPLAY)
    // ============================================================
    else if (pg_param.loadFileType == "cmd_continue9D") {

        // Loaded 9D: x y z wx wy wz fx fy fz
        if (loaded_motion.cols() != 9) {
            RCLCPP_ERROR(node_->get_logger(),
                         "there is not proper setup inside of cmd_continue9D !! cols=%zu",
                         loaded_motion.cols());
            return;
        }

        YMatrix cont_motion = loaded_motion;

        // first pose (rad)
        std::vector<double> first_pose(6, 0.0);
        for (int i = 0; i < 6; ++i) first_pose[i] = cont_motion[0][i];

        // Safety thresholds
        const double MAX_TRANS_MM = 200.0;
        const double MAX_ROT_DEG  = 30.0;

        const double dx = first_pose[0] - current_position[0];
        const double dy = first_pose[1] - current_position[1];
        const double dz = first_pose[2] - current_position[2];
        const double trans_norm = std::sqrt(dx*dx + dy*dy + dz*dz);

        const double dwx_deg = RadianToDegree(first_pose[3] - current_position[3]);
        const double dwy_deg = RadianToDegree(first_pose[4] - current_position[4]);
        const double dwz_deg = RadianToDegree(first_pose[5] - current_position[5]);
        const double rot_norm_deg = std::sqrt(dwx_deg*dwx_deg + dwy_deg*dwy_deg + dwz_deg*dwz_deg);

        if (trans_norm > MAX_TRANS_MM || rot_norm_deg > MAX_ROT_DEG) {
            RCLCPP_ERROR(node_->get_logger(),
                         "Safety abort: current->first pose gap too large. trans=%.2fmm, rot=%.2fdeg",
                         trans_norm, rot_norm_deg);
            RCLCPP_ERROR(node_->get_logger(), "Move robot closer to the start pose, then retry.");
            return;
        }

        // 1) Connect path (Position mode)
        cmd_msg_.data = "Position";
        cmd_pub_->publish(cmd_msg_);

        YMatrix connect_position = {
            {current_position[0], current_position[1], current_position[2],
             current_position[3], current_position[4], current_position[5]},
            {first_pose[0], first_pose[1], first_pose[2],
             first_pose[3], first_pose[4], first_pose[5]}
        };

        // Mirror your "Insert start & end pose" concept: 2-point blend
        std::vector<double> connect_velocity     = {0.0, pg_param.initialTransferSpeed};
        std::vector<double> connect_ang_velocity = {0.0, 0.0};
        std::vector<double> connect_holding_time = {0.0, 0.0};

        MotionBlender6D connect_blender(connect_position,
                                       connect_velocity,
                                       connect_ang_velocity,
                                       connect_holding_time,
                                       DegreeToRadian(pg_param.angularVelocityLimit),
                                       pg_param.startingTime,
                                       pg_param.lastRestingTime,
                                       pg_param.accelerationTime,
                                       control_period_);

        YMatrix connect_motion = connect_blender.blendMotion(pg_param.defualt_travelTime);

        auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(control_period_)
        );
        auto next_time = std::chrono::steady_clock::now();

        RCLCPP_INFO(node_->get_logger(), "Connecting current pose -> first point (Position mode)...");

        for (const auto& pos6 : connect_motion) {
            if (!rclcpp::ok()) {
                RCLCPP_WARN(node_->get_logger(), "ROS not ok, stop connecting path.");
                return;
            }

            cmd_motion_msg_.data.clear();
            for (const auto& v : pos6) cmd_motion_msg_.data.push_back(v);
            cmd_motion_pub_->publish(cmd_motion_msg_);

            next_time += period_ns;
            std::this_thread::sleep_until(next_time);
        }

        // 2) Replay 9D in Force mode
        cmd_msg_.data = "Force";
        cmd_pub_->publish(cmd_msg_);

        RCLCPP_INFO(node_->get_logger(), "Replaying cmd_continue9D (Force mode)...");

        next_time = std::chrono::steady_clock::now();
        for (const auto& row9 : cont_motion) {
            if (!rclcpp::ok()) {
                RCLCPP_WARN(node_->get_logger(), "ROS not ok, stop continue9D replay.");
                return;
            }

            cmd_motion_msg_.data.clear();
            for (const auto& v : row9) cmd_motion_msg_.data.push_back(v);
            cmd_motion_pub_->publish(cmd_motion_msg_);

            next_time += period_ns;
            std::this_thread::sleep_until(next_time);
        }

        RCLCPP_INFO(node_->get_logger(), "cmd_continue9D finished.");
    }

    else {
        RCLCPP_ERROR(node_->get_logger(), "Unknown file type: %s", pg_param.loadFileType.c_str());
    }
}
