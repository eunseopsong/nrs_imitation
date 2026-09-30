#include <rclcpp/rclcpp.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <sstream>
#include <algorithm>  // [ADD] std::max
#include <memory>
#include <array>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>

#include "Y2FT_AQ/FT_EtherGet.hpp"
#include "Y2FT_AQ/FT_eCANGet.hpp"
#include "Y2FT_AQ/FT_msgGet.hpp"
#include "Y2Filters/Gen_filter.hpp"
#include "geometry_msgs/msg/wrench_stamped.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include <yaml-cpp/yaml.h>

#include "Y2Matrix/YMatrix.hpp"

// =============================
// Sensor / Robot 설정
// =============================

/* Sensor Communication Type
 * 0: Ethernet
 * 1: eCAN
 * 2: ROS message
 */
#define FT_SOURCE_MODE 0

/* Gravity Compensation Mode */
#define GRAVITY_COMPENSATION_MODE 1 // 0: Disabled, 1: Enabled

/* Sensor IP/PORT */
static const char* FT1_IP   = "192.168.0.100";
static const int   FT1_PORT = 8890;  // 4001: eCAN, 8890: Ethernet

/* Sensor Message Input */
static const char* FT1_MSG_TOPIC = "/ur10skku/ft_sensor";
static const char* FT1_MSG_TYPE  = "geometry_msgs/msg/WrenchStamped";
static const FTMsgArrayFormat FT1_MSG_ARRAY_FORMAT = {0, 1, 2, 3, 4, 5};

/* Sampling frequency */
static const int   SAMPLING_FREQ_HZ = 2000;

/* [ADD] Monitoring publish rate */
static const int   MONITORING_FREQ_HZ = 100;

/* Sensor 초기화를 위한 샘플링 개수 */
static const unsigned int SENSOR_INIT_COUNT_NUM = 1000;

/* MOV 필터 크기 */
static const int MOV_SIZE = 10/5;

/* Robot Namespace */
static const char* ROBOT_NAME = "/ur10skku";

/* FT local frame -> TCP local frame (FT 기준 벡터를 TCP 기준으로) */
YMatrix ROT_TCP2FT =
{
    { 1,  0,  0},
    { 0, -1,  0},
    { 0,  0, -1}
};

namespace
{
std::string expandUserPath(const std::string& path)
{
    if (path.rfind("~/", 0) != 0) {
        return path;
    }

    const char* home = std::getenv("HOME");
    if (home == nullptr || std::string(home).empty()) {
        return path;
    }
    return std::string(home) + path.substr(1);
}

} // namespace


// =============================
// 유틸 함수
// =============================

// ============================================
// FTGetMain Node 정의
// ============================================
class FTGetMain : public rclcpp::Node
{
public:
    FTGetMain(const std::string& ft_ip,
              int ft_port,
              const std::string& robot_name,
              int mov_size = 1);
    ~FTGetMain();

    /// 타이머 콜백: 센서 읽고, 필터 + 중력 보상 + 좌표 변환 + publish
    void transferData();

private:
    // ----- Parameters -----
    std::vector<Y_MovFilter> movFilter;         // MOV filter for each axis
    std::vector<double>      current_TCP_pose;  // [x,y,z,wx,wy,wz]
    YMatrix                  ROT_Base2TCP;      // Base → TCP 회전 행렬
    std::atomic<bool>        current_pose_received_{false};

    // 마지막 G-comp 결과 저장용 (Base frame)
    YMatrix                  last_GcompForce;   // 3x1
    YMatrix                  last_GcompMoment;  // 3x1

    // Reusable matrices to avoid per-loop allocations
    YMatrix                  SframeForce_;
    YMatrix                  SframeMoment_;
    YMatrix                  SGravityForce_;
    YMatrix                  SGravityMoment_;
    YMatrix                  TCPRawForce_;
    YMatrix                  TCPRawMoment_;
    YMatrix                  TCPForce_;
    YMatrix                  TCPMoment_;
    YMatrix                  RframeForce_;
    YMatrix                  RframeMoment_;
    YMatrix                  GcompForce_;
    YMatrix                  GcompMoment_;

    // Gravity compensation temporaries
    YMatrix                  g_base_;        // base frame gravity
    YMatrix                  g_tcp_;         // tcp frame gravity
    YMatrix                  g_tcp_init_;    // gravity reference after sensor zero
    bool                     g_tcp_init_set_;

    // Runtime gravity compensation parameters
    bool                     gravity_compensation_enabled_;
    bool                     gravity_matrix_loaded_;
    YMatrix                  gravity_matrix_; // 6x3, TCP wrench = H * g_tcp
    std::string              gravity_calibration_file_;

    std::string              log_file_path_;
    std::ofstream            log_ofs_;
    std::mutex               log_ofs_mutex_;
    // Async logger
    std::deque<std::string>  log_queue_;
    std::mutex               log_queue_mutex_;
    std::condition_variable  log_queue_cv_;
    std::thread              log_thread_;
    std::atomic<bool>        log_thread_running_{false};
    size_t                   log_queue_max_ = 10000; // bounded queue
    // Diagnostics counters
    std::atomic<uint64_t>    transfer_dbg_count_{0};
    std::atomic<uint64_t>    transfer_dbg_accum_us_{0};
    std::atomic<uint64_t>    ftget_dbg_accum_us_{0};

    // ----- [ADD] Monitoring downsample (decimation) -----
    uint32_t                 monitor_decim_{1};   // publish once per N steps
    uint32_t                 monitor_tick_{0};    // step counter

    // ----- FT Raw getter -----
    #if FT_SOURCE_MODE == 1
    FT_eCANGet FT1SensorGet;
    #elif FT_SOURCE_MODE == 0
    FT_EtherGet FT1SensorGet;
    #elif FT_SOURCE_MODE == 2
    std::unique_ptr<FT_msgGet> FT1SensorGet;
    #else
    #error "FT_SOURCE_MODE must be 0 (Ethernet), 1 (eCAN), or 2 (ROS message)"
    #endif

    // ----- Timer -----
    rclcpp::TimerBase::SharedPtr timer_;

    // ----- Publisher / Subscriber -----
    geometry_msgs::msg::WrenchStamped ft1_msg;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr ft1_pub;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr acquisition_pub_;
    std::uint64_t acquisition_published_sequence_ = 0;
    std::int64_t acquisition_ros_ns_ = 0;
    const std::string acquisition_boot_id_ = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());

    // Monitoring publishers (TCP / Base)
    geometry_msgs::msg::WrenchStamped ft_tcp_msg_;
    geometry_msgs::msg::WrenchStamped ft_base_msg_;
    geometry_msgs::msg::WrenchStamped ft_tcp_raw_msg_;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr ft_tcp_pub_;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr ft_base_pub_;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr ft_tcp_raw_pub_;

    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr currentP_sub;

    // ----- Service ----
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr print_param_srv_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr save_log_srv_;

    // ----- 내부 함수 -----
    bool initFTSensor();
    FTData readFTSensor();
    FTData filtering(const FTData& ftdata);
    void configureGravityCompensation();
    void currentPCB(const std_msgs::msg::Float64MultiArray::ConstSharedPtr msg);
    void printParamsCB(const std_srvs::srv::SetBool::Request::SharedPtr req,
                       std_srvs::srv::SetBool::Response::SharedPtr res);
    void saveLogCB(const std_srvs::srv::SetBool::Request::SharedPtr req,
                   std_srvs::srv::SetBool::Response::SharedPtr res);
};


// ============================================
// 생성자
// ============================================
FTGetMain::FTGetMain(const std::string& ft_ip,
                     int ft_port,
                     const std::string& robot_name,
                     int mov_size)
    : Node("FTGetMain")
    , movFilter()
    , current_TCP_pose(6, 0.0)
    , ROT_Base2TCP(3, 3)
    , last_GcompForce(3, 1)
    , last_GcompMoment(3, 1)
    #if FT_SOURCE_MODE != 2
    , FT1SensorGet(ft_ip, ft_port)
    #endif
    , SframeForce_(3, 1)
    , SframeMoment_(3, 1)
    , SGravityForce_(3, 1)
    , SGravityMoment_(3, 1)
    , TCPRawForce_(3, 1)
    , TCPRawMoment_(3, 1)
    , TCPForce_(3, 1)
    , TCPMoment_(3, 1)
    , RframeForce_(3, 1)
    , RframeMoment_(3, 1)
    , GcompForce_(3, 1)
    , GcompMoment_(3, 1)
    , g_base_(3, 1)
    , g_tcp_(3, 1)
    , g_tcp_init_(3, 1)
    , g_tcp_init_set_(false)
    , gravity_compensation_enabled_(GRAVITY_COMPENSATION_MODE != 0)
    , gravity_matrix_loaded_(false)
    , gravity_matrix_(6, 3)
{
    #if FT_SOURCE_MODE == 2
    FT1SensorGet = std::make_unique<FT_msgGet>(
        *this,
        FT1_MSG_TOPIC,
        FT1_MSG_TYPE,
        FT1_MSG_ARRAY_FORMAT
    );

    RCLCPP_INFO(this->get_logger(),
                "FT source mode: ROS message (topic=%s, type=%s)",
                FT1_MSG_TOPIC,
                FT1_MSG_TYPE);
    #elif FT_SOURCE_MODE == 1
    RCLCPP_INFO(this->get_logger(),
                "FT source mode: eCAN (ip=%s, port=%d)",
                ft_ip.c_str(),
                ft_port);
    #else
    RCLCPP_INFO(this->get_logger(),
                "FT source mode: Ethernet (ip=%s, port=%d)",
                ft_ip.c_str(),
                ft_port);
    #endif

    // ----- [ADD] Monitoring downsample 설정 -----
    // 2000Hz -> 100Hz면 20 step마다 1번 publish
    monitor_decim_ = static_cast<uint32_t>(std::max(1, SAMPLING_FREQ_HZ / MONITORING_FREQ_HZ));
    monitor_tick_  = 0;
    RCLCPP_INFO(this->get_logger(), "Monitoring downsample: %d Hz (decim=%u @ %d Hz loop)",
                MONITORING_FREQ_HZ, monitor_decim_, SAMPLING_FREQ_HZ);

    // ----- Publisher (기존) -----
    std::string ftdata_topic = robot_name + std::string("/ftdata");
    ft1_pub = this->create_publisher<geometry_msgs::msg::WrenchStamped>(ftdata_topic, 10);
    #if FT_SOURCE_MODE == 0
    acquisition_pub_ = this->create_publisher<std_msgs::msg::String>(
        robot_name + "/ft_acquisition", rclcpp::QoS(1).reliable());
    #endif

    // ----- Monitoring publishers -----
    std::string ftdata_tcp_topic  = robot_name + std::string("/ftdata_tcp");   // TCP(local) 기준 (g-comp)
    std::string ftdata_base_topic = robot_name + std::string("/ftdata_base");  // Base 기준 (g-comp)
    std::string ftdata_tcp_raw_topic = robot_name + std::string("/ftdata_tcp_raw");
    ft_tcp_pub_  = this->create_publisher<geometry_msgs::msg::WrenchStamped>(ftdata_tcp_topic, 10);
    ft_base_pub_ = this->create_publisher<geometry_msgs::msg::WrenchStamped>(ftdata_base_topic, 10);
    ft_tcp_raw_pub_ =
        this->create_publisher<geometry_msgs::msg::WrenchStamped>(ftdata_tcp_raw_topic, 10);

    // ----- Subscriber (현재 TCP 포즈) -----
    std::string currentP_topic = robot_name + std::string("/currentP");
    printf("\033[33m[FTGetMain] robot_topic_name: %s \033[0m\n", currentP_topic.c_str());

    currentP_sub = this->create_subscription<std_msgs::msg::Float64MultiArray>(
        currentP_topic,
        10,
        std::bind(&FTGetMain::currentPCB, this, std::placeholders::_1)
    );

    // ----- Service: print parameters -----
    print_param_srv_ = this->create_service<std_srvs::srv::SetBool>(
        "print_ft_params",
        std::bind(&FTGetMain::printParamsCB, this,
                  std::placeholders::_1, std::placeholders::_2)
    );

    // ----- log 파일 경로 설정 -----
    try {
        auto share_dir = ament_index_cpp::get_package_share_directory("Y2FT_AQ");
        std::filesystem::path log_dir =
            std::filesystem::path(share_dir).parent_path() / "log";
        std::filesystem::create_directories(log_dir);
        log_file_path_ = (log_dir / "ft_gcomp_log.txt").string();
        RCLCPP_INFO(this->get_logger(), "Log file: %s", log_file_path_.c_str());

        // Open log file once
        try {
            log_ofs_.open(log_file_path_, std::ios::app);
            if (!log_ofs_) {
                RCLCPP_ERROR(this->get_logger(), "Failed to open log file for appending: %s", log_file_path_.c_str());
            } else {
                log_ofs_ << std::fixed << std::setprecision(6);
            }
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Exception opening log file: %s", e.what());
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to resolve log dir: %s", e.what());
    }

    // Start async logger thread if file was opened
    if (log_ofs_) {
        log_thread_running_.store(true);
        log_thread_ = std::thread([this]() {
            std::vector<std::string> batch;
            batch.reserve(256);
            while (log_thread_running_.load() || !log_queue_.empty()) {
                {
                    std::unique_lock<std::mutex> ql(log_queue_mutex_);
                    log_queue_cv_.wait_for(ql, std::chrono::milliseconds(200), [this]() {
                        return !log_queue_.empty() || !log_thread_running_.load();
                    });

                    while (!log_queue_.empty() && batch.size() < 512) {
                        batch.push_back(std::move(log_queue_.front()));
                        log_queue_.pop_front();
                    }
                }

                if (!batch.empty()) {
                    std::lock_guard<std::mutex> lf(log_ofs_mutex_);
                    for (auto &line : batch) {
                        log_ofs_ << line;
                    }
                    log_ofs_.flush();
                    batch.clear();
                }
            }
            std::lock_guard<std::mutex> lf(log_ofs_mutex_);
            if (log_ofs_.is_open()) log_ofs_.flush();
        });
    }

    // ----- Service: save log -----
    save_log_srv_ = this->create_service<std_srvs::srv::SetBool>(
        "save_ft_log",
        std::bind(&FTGetMain::saveLogCB, this,
                  std::placeholders::_1, std::placeholders::_2)
    );

    // frame id (원하는 프레임명으로 바꿔도 됨)
    ft1_msg.header.frame_id      = "base";
    ft_tcp_msg_.header.frame_id  = "tcp";
    ft_base_msg_.header.frame_id = "base";

    // MOV 필터 초기화 (Fx, Fy, Fz, Mx, My, Mz)
    movFilter.reserve(6);
    for (int i = 0; i < 6; ++i) {
        movFilter.emplace_back(mov_size);
    }

    // Base → TCP 회전행렬 초기값: Identity
    ROT_Base2TCP = YMatrix::identity(3);

    configureGravityCompensation();

    // 타이머 시작
    auto period = std::chrono::duration<double>(1.0 / static_cast<double>(SAMPLING_FREQ_HZ));
    timer_ = this->create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        std::bind(&FTGetMain::transferData, this)
    );
}


// ============================================
// Gravity compensation parameter loading
// ============================================
void FTGetMain::configureGravityCompensation()
{
    gravity_compensation_enabled_ = this->declare_parameter<bool>(
        "gravity_compensation_enabled",
        GRAVITY_COMPENSATION_MODE != 0
    );

    const char* home = std::getenv("HOME");
    const std::string default_file =
        std::string(home ? home : "") +
        "/dev_ws/src/y2_ur10skku_control/Y2FT_AQ/config/spindle_gravity.yaml";
    gravity_calibration_file_ = expandUserPath(
        this->declare_parameter<std::string>("gravity_calibration_file", default_file));
    gravity_matrix_loaded_ = false;
    try {
        const YAML::Node root = YAML::LoadFile(gravity_calibration_file_);
        if (!root["enabled"] || !root["enabled"].as<bool>()) {
            RCLCPP_WARN(this->get_logger(),
                        "Spindle gravity calibration disabled in %s",
                        gravity_calibration_file_.c_str());
        } else if (!root["matrix_6x3"] || root["matrix_6x3"].size() != 6) {
            throw std::runtime_error("matrix_6x3 must contain 6 rows");
        } else {
            for (size_t r = 0; r < 6; ++r) {
                const YAML::Node row = root["matrix_6x3"][r];
                if (!row.IsSequence() || row.size() != 3) {
                    throw std::runtime_error("each matrix_6x3 row must contain 3 values");
                }
                for (size_t c = 0; c < 3; ++c) {
                    const double value = row[c].as<double>();
                    if (!std::isfinite(value)) {
                        throw std::runtime_error("matrix_6x3 contains non-finite data");
                    }
                    gravity_matrix_[static_cast<int>(r)][static_cast<int>(c)] = value;
                }
            }
            gravity_matrix_loaded_ = true;
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load spindle gravity YAML '%s': %s",
                     gravity_calibration_file_.c_str(), e.what());
    }

    RCLCPP_INFO(this->get_logger(),
                "Gravity compensation: requested=%s, matrix_loaded=%s, yaml=%s",
                gravity_compensation_enabled_ ? "true" : "false",
                gravity_matrix_loaded_ ? "true" : "false",
                gravity_calibration_file_.c_str());
}


// ============================================
// currentP Subscriber 콜백
// ============================================
void FTGetMain::currentPCB(const std_msgs::msg::Float64MultiArray::ConstSharedPtr msg)
{
    // [x,y,z,wx,wy,wz]
    for (size_t i = 0; i < 6 && i < msg->data.size(); ++i) {
        current_TCP_pose[i] = msg->data[i];
    }

    // Spatial Angle → 회전행렬 (Base -> TCP)
    SpatialAngle spatial_angle(
        current_TCP_pose[3],
        current_TCP_pose[4],
        current_TCP_pose[5]
    );

    ROT_Base2TCP.insert(0, 0, YMatrix::fromSpatialAngle(spatial_angle));
    current_pose_received_.store(true);
}


// ============================================
// print_ft_params Service 콜백
// ============================================
void FTGetMain::printParamsCB(
    const std_srvs::srv::SetBool::Request::SharedPtr req,
    std_srvs::srv::SetBool::Response::SharedPtr res)
{
    (void)req;  // unused

    std::ostringstream oss;
    oss << "\n";
    oss << "===== FTGetMain Parameters =====\n";
    oss << "FT_SOURCE_MODE           : " << FT_SOURCE_MODE << " (0: Ethernet, 1: eCAN, 2: ROS message)\n";
    oss << "GRAVITY_COMPENSATION_MODE: " << GRAVITY_COMPENSATION_MODE << " (compile default, 0: Disabled, 1: Enabled)\n";
    oss << "gravity_compensation_enabled: " << (gravity_compensation_enabled_ ? "true" : "false") << "\n";
    oss << "FT1_IP                   : " << FT1_IP << "\n";
    oss << "FT1_PORT                 : " << FT1_PORT << "\n";
    oss << "FT1_MSG_TOPIC            : " << FT1_MSG_TOPIC << "\n";
    oss << "FT1_MSG_TYPE             : " << FT1_MSG_TYPE << "\n";
    oss << "FT1_MSG_ARRAY_FORMAT     : ["
        << FT1_MSG_ARRAY_FORMAT.fx << ", "
        << FT1_MSG_ARRAY_FORMAT.fy << ", "
        << FT1_MSG_ARRAY_FORMAT.fz << ", "
        << FT1_MSG_ARRAY_FORMAT.mx << ", "
        << FT1_MSG_ARRAY_FORMAT.my << ", "
        << FT1_MSG_ARRAY_FORMAT.mz << "]\n";
    oss << "SAMPLING_FREQ_HZ         : " << SAMPLING_FREQ_HZ << " [Hz]\n";
    oss << "MONITORING_FREQ_HZ       : " << MONITORING_FREQ_HZ << " [Hz] (decim=" << monitor_decim_ << ")\n";
    oss << "SENSOR_INIT_COUNT_NUM    : " << SENSOR_INIT_COUNT_NUM << "\n";
    oss << "MOV_SIZE                 : " << MOV_SIZE << "\n";
    oss << "ROBOT_NAME               : " << ROBOT_NAME << "\n";

    // ROT_TCP2FT 출력 (3x3)
    oss << "ROT_TCP2FT (3x3):\n";
    for (int i = 0; i < 3; ++i)
    {
        oss << "  [ ";
        for (int j = 0; j < 3; ++j)
        {
            oss << ROT_TCP2FT[i][j];
            if (j < 2) oss << ", ";
        }
        oss << " ]\n";
    }

    oss << "gravity_matrix_loaded: " << (gravity_matrix_loaded_ ? "true" : "false") << "\n";
    oss << "gravity_calibration_file: " << gravity_calibration_file_ << "\n";

    std::string msg = oss.str();
    RCLCPP_INFO_STREAM(this->get_logger(), msg);

    res->success = true;
    res->message = msg;
}


bool FTGetMain::initFTSensor()
{
    #if FT_SOURCE_MODE == 2
    return FT1SensorGet ? FT1SensorGet->FT_init(SENSOR_INIT_COUNT_NUM) : false;
    #else
    return FT1SensorGet.FT_init(SENSOR_INIT_COUNT_NUM);
    #endif
}


FTData FTGetMain::readFTSensor()
{
    #if FT_SOURCE_MODE == 2
    return FT1SensorGet ? FT1SensorGet->FTGet() : FTData();
    #else
    return FT1SensorGet.FTGet();
    #endif
}


// ============================================
// save_ft_log Service 콜백
// ============================================
void FTGetMain::saveLogCB(
    const std_srvs::srv::SetBool::Request::SharedPtr req,
    std_srvs::srv::SetBool::Response::SharedPtr res)
{
    if (!req->data) {
        res->success = false;
        res->message = "data=false: nothing written";
        return;
    }

    if (!log_ofs_) {
        res->success = false;
        res->message = "log file not open";
        RCLCPP_ERROR(this->get_logger(), "Log file is not open: %s", log_file_path_.c_str());
        return;
    }

    std::ostringstream oss;
    oss << this->now().seconds() << " ";
    oss << last_GcompForce[0][0]  << " " << last_GcompForce[1][0]  << " " << last_GcompForce[2][0]  << " ";
    oss << last_GcompMoment[0][0] << " " << last_GcompMoment[1][0] << " " << last_GcompMoment[2][0] << " ";
    oss << current_TCP_pose[3]    << " " << current_TCP_pose[4]    << " " << current_TCP_pose[5]    << "\n";
    std::string line = oss.str();

    {
        std::lock_guard<std::mutex> ql(log_queue_mutex_);
        if (log_queue_.size() >= log_queue_max_) {
            res->success = false;
            res->message = "log queue full: dropped";
            RCLCPP_WARN(this->get_logger(), "Log queue full, dropping log entry");
            return;
        }
        log_queue_.push_back(std::move(line));
    }
    log_queue_cv_.notify_one();

    res->success = true;
    res->message = "Queued log entry";
}


// ============================================
// FT 데이터 MOV 필터링
// ============================================
FTData FTGetMain::filtering(const FTData& ftdata)
{
    FTData filtered_data;

    filtered_data.Fx = movFilter[0].MovFilter(ftdata.Fx);
    filtered_data.Fy = movFilter[1].MovFilter(ftdata.Fy);
    filtered_data.Fz = movFilter[2].MovFilter(ftdata.Fz);

    filtered_data.Mx = movFilter[3].MovFilter(ftdata.Mx);
    filtered_data.My = movFilter[4].MovFilter(ftdata.My);
    filtered_data.Mz = movFilter[5].MovFilter(ftdata.Mz);

    return filtered_data;
}


// ============================================
// 타이머 콜백: 메인 처리 루프
// ============================================
void FTGetMain::transferData()
{
    FTData filtered_out;

    // Reset vectors
    for (int i = 0; i < 3; ++i) {
        SframeForce_[i][0]   = 0.0;
        SframeMoment_[i][0]  = 0.0;
        SGravityForce_[i][0] = 0.0;
        SGravityMoment_[i][0]= 0.0;
        TCPRawForce_[i][0]   = 0.0;
        TCPRawMoment_[i][0]  = 0.0;
        TCPForce_[i][0]      = 0.0;
        TCPMoment_[i][0]     = 0.0;
        RframeForce_[i][0]   = 0.0;
        RframeMoment_[i][0]  = 0.0;
        GcompForce_[i][0]    = 0.0;
        GcompMoment_[i][0]   = 0.0;
    }

    // 센서 초기 offset 계산 (내부에서 한 번만 동작한다고 가정)
    initFTSensor();

    // Raw 데이터 읽고 필터링
    filtered_out = filtering(readFTSensor());

    #if FT_SOURCE_MODE == 0
    // Separate provenance channel: cached polls NEVER refresh sequence/stamp.
    // Timestamp describes host UDP reception, not the sensor's internal ADC.
    const auto& sample = FT1SensorGet.acquisition();
    if (sample.sequence != acquisition_published_sequence_) {
        const auto source_age = sample.age(std::chrono::steady_clock::now());
        acquisition_ros_ns_ = this->now().nanoseconds() -
            static_cast<std::int64_t>(source_age * 1e9);
        std::ostringstream body;
        body << std::setprecision(17)
             << "{\"schema\":\"aft_ether_acquisition_v1\",\"boot_id\":\""
             << acquisition_boot_id_ << "\",\"sequence\":" << sample.sequence
             << ",\"source_ros_ns\":" << acquisition_ros_ns_
             << ",\"valid\":" << ((sample.valid && !sample.invalid_packet_seen) ? "true" : "false")
             << ",\"initialized\":" << (FT1SensorGet.initialized() ? "true" : "false")
             << ",\"sensor_wrench\":";
        if (sample.valid) {
            body << '[';
            for (std::size_t i = 0; i < sample.raw.size(); ++i) {
                if (i) body << ',';
                body << sample.raw[i];
            }
            body << ']';
        } else {
            body << "null";
        }
        body << '}';
        std_msgs::msg::String message;
        message.data = body.str();
        acquisition_pub_->publish(message);
        acquisition_published_sequence_ = sample.sequence;
    }
    #endif

    // 센서 frame 힘/모멘트
    SframeForce_[0][0]  = filtered_out.Fx;
    SframeForce_[1][0]  = filtered_out.Fy;
    SframeForce_[2][0]  = filtered_out.Fz;

    SframeMoment_[0][0] = filtered_out.Mx;
    SframeMoment_[1][0] = filtered_out.My;
    SframeMoment_[2][0] = filtered_out.Mz;

    // Sensor axes -> TCP axes, before gravity compensation.
    for (int i = 0; i < 3; ++i) {
        for (int k = 0; k < 3; ++k) {
            TCPRawForce_[i][0] += ROT_TCP2FT[i][k] * SframeForce_[k][0];
            TCPRawMoment_[i][0] += ROT_TCP2FT[i][k] * SframeMoment_[k][0];
        }
    }

    // Common spindle model: delta_wrench_tcp = H(6x3) * delta_g_tcp.
    const bool apply_gcomp = gravity_compensation_enabled_ && gravity_matrix_loaded_ &&
                             current_pose_received_.load();
    if (apply_gcomp) {
        for (int i = 0; i < 3; ++i) g_base_[i][0] = 0.0;
        g_base_[2][0] = -9.81;

        for (int i = 0; i < 3; ++i) {
            g_tcp_[i][0] = 0.0;
            for (int k = 0; k < 3; ++k) {
                g_tcp_[i][0] += ROT_Base2TCP[k][i] * g_base_[k][0];
            }
        }

        if (!g_tcp_init_set_) {
            for (int i = 0; i < 3; ++i) {
                g_tcp_init_[i][0] = g_tcp_[i][0];
            }
            g_tcp_init_set_ = true;
        }

        for (int row = 0; row < 6; ++row) {
            double value = 0.0;
            for (int col = 0; col < 3; ++col) {
                const double dg = g_tcp_[col][0] - g_tcp_init_[col][0];
                value += gravity_matrix_[row][col] * dg;
            }
            if (row < 3) SGravityForce_[row][0] = value;
            else SGravityMoment_[row - 3][0] = value;
        }
    }

    for (int i = 0; i < 3; ++i) {
        TCPForce_[i][0] = TCPRawForce_[i][0] - SGravityForce_[i][0];
        TCPMoment_[i][0] = TCPRawMoment_[i][0] - SGravityMoment_[i][0];
    }

    // TCP(local) → Base frame
    for (int i = 0; i < 3; ++i) {
        RframeForce_[i][0] = 0.0;
        for (int k = 0; k < 3; ++k) {
            RframeForce_[i][0] += ROT_Base2TCP[i][k] * TCPForce_[k][0];
        }
    }
    for (int i = 0; i < 3; ++i) {
        RframeMoment_[i][0] = 0.0;
        for (int k = 0; k < 3; ++k) {
            RframeMoment_[i][0] += ROT_Base2TCP[i][k] * TCPMoment_[k][0];
        }
    }

    // 최종 publish용 (Base 기준)
    for (int i = 0; i < 3; ++i) {
        GcompForce_[i][0]  = RframeForce_[i][0];
        GcompMoment_[i][0] = RframeMoment_[i][0];
        last_GcompForce[i][0]  = TCPForce_[i][0];
        last_GcompMoment[i][0] = TCPMoment_[i][0];
    }

    // ----------------------------
    // 기존 publish (/ftdata): Base 기준 (full-rate)
    // ----------------------------
    #if FT_SOURCE_MODE == 0
    ft1_msg.header.stamp = rclcpp::Time(acquisition_ros_ns_);
    ft1_msg.header.frame_id = "robot_base_acquired_v1";
    #else
    ft1_msg.header.stamp = this->now();
    #endif
    ft1_msg.wrench.force.x  = GcompForce_[0][0];
    ft1_msg.wrench.force.y  = GcompForce_[1][0];
    ft1_msg.wrench.force.z  = GcompForce_[2][0];
    ft1_msg.wrench.torque.x = GcompMoment_[0][0];
    ft1_msg.wrench.torque.y = GcompMoment_[1][0];
    ft1_msg.wrench.torque.z = GcompMoment_[2][0];
    ft1_pub->publish(ft1_msg);

    // ----------------------------
    // Monitoring publish (/ftdata_tcp, /ftdata_base): ~100Hz downsample
    // ----------------------------
    monitor_tick_++;
    if (monitor_tick_ >= monitor_decim_) {
        monitor_tick_ = 0;

        const auto stamp = ft1_msg.header.stamp;

        // TCP(local)
        ft_tcp_msg_.header.stamp = stamp;
        ft_tcp_msg_.wrench.force.x  = TCPForce_[0][0];
        ft_tcp_msg_.wrench.force.y  = TCPForce_[1][0];
        ft_tcp_msg_.wrench.force.z  = TCPForce_[2][0];
        ft_tcp_msg_.wrench.torque.x = TCPMoment_[0][0];
        ft_tcp_msg_.wrench.torque.y = TCPMoment_[1][0];
        ft_tcp_msg_.wrench.torque.z = TCPMoment_[2][0];

        // Raw TCP-frame wrench used only for calibration/diagnostics.
        ft_tcp_raw_msg_.header.stamp = stamp;
        ft_tcp_raw_msg_.header.frame_id = "tcp";
        ft_tcp_raw_msg_.wrench.force.x = TCPRawForce_[0][0];
        ft_tcp_raw_msg_.wrench.force.y = TCPRawForce_[1][0];
        ft_tcp_raw_msg_.wrench.force.z = TCPRawForce_[2][0];
        ft_tcp_raw_msg_.wrench.torque.x = TCPRawMoment_[0][0];
        ft_tcp_raw_msg_.wrench.torque.y = TCPRawMoment_[1][0];
        ft_tcp_raw_msg_.wrench.torque.z = TCPRawMoment_[2][0];

        // Base
        ft_base_msg_.header.stamp = stamp;
        ft_base_msg_.wrench.force.x  = GcompForce_[0][0];
        ft_base_msg_.wrench.force.y  = GcompForce_[1][0];
        ft_base_msg_.wrench.force.z  = GcompForce_[2][0];
        ft_base_msg_.wrench.torque.x = GcompMoment_[0][0];
        ft_base_msg_.wrench.torque.y = GcompMoment_[1][0];
        ft_base_msg_.wrench.torque.z = GcompMoment_[2][0];

        // 구독자가 있을 때만 publish해서 부하 감소 (원하면 if 제거 가능)
        if (ft_tcp_pub_ && ft_tcp_pub_->get_subscription_count() > 0) {
            ft_tcp_pub_->publish(ft_tcp_msg_);
        }
        if (ft_base_pub_ && ft_base_pub_->get_subscription_count() > 0) {
            ft_base_pub_->publish(ft_base_msg_);
        }
        if (ft_tcp_raw_pub_ && ft_tcp_raw_pub_->get_subscription_count() > 0) {
            ft_tcp_raw_pub_->publish(ft_tcp_raw_msg_);
        }
    }
}


// ============================================
// main
// ============================================
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FTGetMain>(
        std::string(FT1_IP),
        FT1_PORT,
        std::string(ROBOT_NAME),
        MOV_SIZE
    );

    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();
    rclcpp::shutdown();
    return 0;
}


FTGetMain::~FTGetMain()
{
    // Stop logger thread
    log_thread_running_.store(false);
    log_queue_cv_.notify_all();
    if (log_thread_.joinable()) {
        log_thread_.join();
    }

    // Close log file if open
    std::lock_guard<std::mutex> lk(log_ofs_mutex_);
    if (log_ofs_.is_open()) {
        log_ofs_.flush();
        log_ofs_.close();
    }
}
