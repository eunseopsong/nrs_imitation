#ifndef FT_PROCESSING_H
#define FT_PROCESSING_H

#include <stdio.h>
#include <memory>
#include <vector>
#include <string>
#include <array>
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/wrench.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_srvs/srv/empty.hpp>

#include "nrs_filter_core.hpp"
#include "nrs_filter_applied.hpp"

#include "CAN_reader.hpp"

class FT_processing : public NRS_FTSensor
{
public:
    FT_processing(std::shared_ptr<rclcpp::Node> node,
                  double Ts,
                  unsigned char& HandleID_,
                  unsigned char& ContactID_,
                  bool HaccSwitch_,
                  bool CaccSwitch_);
    ~FT_processing();

    void FT_init(int sen_init_num);
    void FT_filtering();

    void FT_publish();
    void FT_print();
    void FT_record();

    bool SRV5_Handle(const std::shared_ptr<std_srvs::srv::Empty::Request> req,
                     std::shared_ptr<std_srvs::srv::Empty::Response> res);

    void FT_run();

private:
    using Vec3 = std::array<double, 3>;
    using Mat3 = std::array<std::array<double, 3>, 3>;

    void configureGravityCompensation();
    void calibratedPoseCB(const std_msgs::msg::Float64MultiArray::ConstSharedPtr msg);
    void resetGravityReference();
    void applyGravityCompensation(double force[3], double moment[3]);

    std::shared_ptr<rclcpp::Node> node_;
    double Ts_;
    double time_counter = 0;
    FILE *Data1_txt = nullptr;
    bool runnning = true;

    std::string YamlString_IP, YamlData1_path;
    int YamlData1_switch = 0;
    int YamlPrint_switch = 0;
    bool Hmov_switch = false, HLPF_switch = false, HBSF_switch = false;
    bool Cmov_switch = false, CLPF_switch = false, CBSF_switch = false;
    bool HaccSwitch = false;
    bool CaccSwitch = false;

    bool gravity_compensation_enabled_ = true;
    bool gravity_apply_handle_ = false;
    bool gravity_apply_contact_ = true;
    bool gravity_reference_set_ = false;
    bool missing_pose_warned_ = false;
    double tool_mass_ = 0.0;
    Vec3 tool_cog_ = {0.0, 0.0, 0.0};
    Vec3 gravity_sensor_init_ = {0.0, 0.0, 0.0};
    Vec3 gravity_compensation_axis_sign_ = {-1.0, 1.0, 1.0};
    Mat3 latest_world_to_tracker_rot_ = {{{1.0, 0.0, 0.0},
                                          {0.0, 1.0, 0.0},
                                          {0.0, 0.0, 1.0}}};
    Mat3 tracker_to_sensor_rot_ = {{{1.0, 0.0, 0.0},
                                    {0.0, 1.0, 0.0},
                                    {0.0, 0.0, 1.0}}};
    bool has_calibrated_pose_ = false;
    std::mutex pose_mutex_;
    std::string calibrated_pose_topic_;
    std::string tool_param_source_;
    std::string tool_stl_path_;
    double tool_stl_volume_m3_ = 0.0;
    Vec3 tool_stl_centroid_m_ = {0.0, 0.0, 0.0};

    rclcpp::Publisher<geometry_msgs::msg::Wrench>::SharedPtr ftsensor_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Wrench>::SharedPtr Cftsensor_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr vive_force_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr vive_moment_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr vive_acc_pub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr calibrated_pose_sub_;

    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr Aidin_gui_srv5;

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr aidinGui_statePub;
    std_msgs::msg::String aidinGui_stateMsg;

    geometry_msgs::msg::Wrench pub_data, Cpub_data;

    int Mov_num = 30;
    std::vector<NRS_MovFilter> movF, movM, movCF, movCM;

    double LPF_cutF = 2;
    double CLPF_cutF = 10;
    std::vector<NRS_FreqFilter> LPF_F, LPF_M, LPF_CF, LPF_CM;

    double BSF_cutF = 15;
    double BSF_BW = 5;
    double CBSF_cutF = 15;
    double CBSF_BW = 5;
    std::vector<NRS_FreqFilter> BSF_F, BSF_M, BSF_CF, BSF_CM;
};

#endif
