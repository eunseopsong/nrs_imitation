#include "FT_Processing.hpp"
#include <cstdlib>

namespace {
std::string expand_repo_path(const std::string& path) {
  if (path.empty()) {
    return path;
  }
  const char* home = std::getenv("HOME");
  if (home == nullptr || std::string(home).empty()) {
    return path;
  }
  if (path.rfind("~/", 0) == 0) {
    return std::string(home) + path.substr(1);
  }
  if (path[0] != '/') {
    return std::string(home) + "/nrs_imitation/" + path;
  }
  return path;
}
}
#include <cmath>
#include <iostream>

// [ADDED]
#include <thread>
#include <mutex>
#include <atomic>

namespace
{
// FT 데이터 공유 보호용 (단일 노드 전제)
std::mutex g_ft_mtx;

// 안전 가드
inline double clamp_positive(double v, double fallback)
{
  return (std::isfinite(v) && v > 1e-9) ? v : fallback;
}
} // namespace

FT_processing::FT_processing(std::shared_ptr<rclcpp::Node> node,
                             double Ts,
                             unsigned char& HandleID_,
                             unsigned char& ContactID_,
                             bool HaccSwitch_,
                             bool CaccSwitch_)
: NRS_FTSensor(HandleID_, ContactID_, HaccSwitch_, CaccSwitch_),
  node_(node),
  Ts_(Ts),
  HaccSwitch(HaccSwitch_),
  CaccSwitch(CaccSwitch_),
  movF(3, NRS_MovFilter(Mov_num)),
  movM(3, NRS_MovFilter(Mov_num)),
  movCF(3, NRS_MovFilter(Mov_num)),
  movCM(3, NRS_MovFilter(Mov_num)),
  LPF_F(3, NRS_FreqFilter(Ts_)),
  LPF_M(3, NRS_FreqFilter(Ts_)),
  LPF_CF(3, NRS_FreqFilter(Ts_)),
  LPF_CM(3, NRS_FreqFilter(Ts_)),
  BSF_F(3, NRS_FreqFilter(Ts_)),
  BSF_M(3, NRS_FreqFilter(Ts_)),
  BSF_CF(3, NRS_FreqFilter(Ts_)),
  BSF_CM(3, NRS_FreqFilter(Ts_))
{
  // 파라미터 읽기
  if (!node_->get_parameter("AFT80IP", YamlString_IP)) {
    RCLCPP_ERROR(node_->get_logger(), "Can't find AFT80IP!");
    YamlString_IP = "192.168.0.42";
  }
  if (!node_->get_parameter("Data1_path", YamlData1_path)) {
    RCLCPP_ERROR(node_->get_logger(), "Can't find Data1_path!");
    YamlData1_path = "tmp/data1.txt";
  }
  YamlData1_path = expand_repo_path(YamlData1_path);
  if (!node_->get_parameter("Data1_switch", YamlData1_switch)) {
    RCLCPP_ERROR(node_->get_logger(), "Can't find Data1_switch!");
    YamlData1_switch = 0;
  }
  if (!node_->get_parameter("Print_switch", YamlPrint_switch)) {
    RCLCPP_ERROR(node_->get_logger(), "Can't find Print_switch!");
    YamlPrint_switch = 0;
  }

  {
    std::vector<int64_t> tmp;
    if (node_->get_parameter("Handle_Sensor_Order", tmp)) {
      H_sen_order.assign(tmp.begin(), tmp.end());
    } else {
      RCLCPP_ERROR(node_->get_logger(), "Can't find Handle_Sensor_Order!");
    }
  }
  {
    std::vector<int64_t> tmp;
    if (node_->get_parameter("Handle_Sensor_sign", tmp)) {
      H_sen_sign.assign(tmp.begin(), tmp.end());
    } else {
      RCLCPP_ERROR(node_->get_logger(), "Can't find Handle_Sensor_sign!");
    }
  }
  {
    std::vector<int64_t> tmp;
    if (node_->get_parameter("Contact_Sensor_Order", tmp)) {
      C_sen_order.assign(tmp.begin(), tmp.end());
    } else {
      RCLCPP_ERROR(node_->get_logger(), "Can't find Contact_Sensor_Order!");
    }
  }
  {
    std::vector<int64_t> tmp;
    if (node_->get_parameter("Contact_Sensor_sign", tmp)) {
      C_sen_sign.assign(tmp.begin(), tmp.end());
    } else {
      RCLCPP_ERROR(node_->get_logger(), "Can't find Contact_Sensor_sign!");
    }
  }

  if (!node_->get_parameter("Hmov_switch", Hmov_switch)) {
    RCLCPP_ERROR(node_->get_logger(), "Can't find Hmov_switch!");
    Hmov_switch = false;
  }
  if (!node_->get_parameter("HLPF_switch", HLPF_switch)) {
    RCLCPP_ERROR(node_->get_logger(), "Can't find HLPF_switch!");
    HLPF_switch = false;
  }
  if (!node_->get_parameter("HBSF_switch", HBSF_switch)) {
    RCLCPP_ERROR(node_->get_logger(), "Can't find HBSF_switch!");
    HBSF_switch = false;
  }

  if (!node_->get_parameter("Cmov_switch", Cmov_switch)) {
    RCLCPP_ERROR(node_->get_logger(), "Can't find Cmov_switch!");
    Cmov_switch = false;
  }
  if (!node_->get_parameter("CLPF_switch", CLPF_switch)) {
    RCLCPP_ERROR(node_->get_logger(), "Can't find CLPF_switch!");
    CLPF_switch = false;
  }
  if (!node_->get_parameter("CBSF_switch", CBSF_switch)) {
    RCLCPP_ERROR(node_->get_logger(), "Can't find CBSF_switch!");
    CBSF_switch = false;
  }

  /* ROS2 publisher init*/
  ftsensor_pub_ = node_->create_publisher<geometry_msgs::msg::Wrench>("/ftsensor/measured_Hvalue", 10);
  Cftsensor_pub_ = node_->create_publisher<geometry_msgs::msg::Wrench>("/ftsensor/measured_Cvalue", 10);

  vive_force_pub_  = node_->create_publisher<geometry_msgs::msg::Vector3>("vive_force", 10);
  vive_moment_pub_ = node_->create_publisher<geometry_msgs::msg::Vector3>("vive_moment", 10);
  vive_acc_pub_    = node_->create_publisher<std_msgs::msg::Float64MultiArray>("vive_acc", 10);

  aidinGui_statePub = node_->create_publisher<std_msgs::msg::String>("Aidin_State_Text", 20);

  Aidin_gui_srv5 = node_->create_service<std_srvs::srv::Empty>(
    "sensor_zeroset",
    std::bind(&FT_processing::SRV5_Handle, this, std::placeholders::_1, std::placeholders::_2));

  CAN_sampling = Ts_;
}

FT_processing::~FT_processing()
{
  RCLCPP_ERROR(node_->get_logger(), "FT_processing was terminated");
  if (Data1_txt) {
    fclose(Data1_txt);
    Data1_txt = nullptr;
  }
}

void FT_processing::FT_init(int sen_init_num)
{
  if (YamlData1_switch == 1)
  {
    Data1_txt = fopen(YamlData1_path.c_str(), "wt");
  }

  char *AFT80_IP = const_cast<char *>(YamlString_IP.c_str());
  TCP_init(AFT80_IP, 4001);

  for (int i = 0; i < 3; i++)
  {
    LPF_F[i].LPF_cutF  = LPF_cutF;
    LPF_M[i].LPF_cutF  = LPF_cutF;
    LPF_CF[i].LPF_cutF = CLPF_cutF;
    LPF_CM[i].LPF_cutF = CLPF_cutF;

    BSF_F[i].BSF_cutF  = BSF_cutF;
    BSF_M[i].BSF_cutF  = BSF_cutF;
    BSF_CF[i].BSF_cutF = CBSF_cutF;
    BSF_CM[i].BSF_cutF = CBSF_cutF;

    BSF_F[i].BSF_BW  = BSF_BW;
    BSF_M[i].BSF_BW  = BSF_BW;
    BSF_CF[i].BSF_BW = CBSF_BW;
    BSF_CM[i].BSF_BW = CBSF_BW;
  }
  init_average_num = sen_init_num;
}

void FT_processing::FT_filtering()
{
  for (int i = 0; i < 3; i++)
  {
    if (Hmov_switch)
    {
      Force_val[i]  = movF[i].MovFilter(Force_val[i]);
      Moment_val[i] = movM[i].MovFilter(Moment_val[i]);
    }
    if (Cmov_switch)
    {
      Contact_Force_val[i]  = movCF[i].MovFilter(Contact_Force_val[i]);
      Contact_Moment_val[i] = movCM[i].MovFilter(Contact_Moment_val[i]);
    }

    if (HLPF_switch)
    {
      Force_val[i]  = LPF_F[i].LPF(Force_val[i]);
      Moment_val[i] = LPF_M[i].LPF(Moment_val[i]);
    }
    if (CLPF_switch)
    {
      Contact_Force_val[i]  = LPF_CF[i].LPF(Contact_Force_val[i]);
      Contact_Moment_val[i] = LPF_CM[i].LPF(Contact_Moment_val[i]);
    }

    if (HBSF_switch)
    {
      Force_val[i]  = BSF_F[i].BSF(Force_val[i]);
      Moment_val[i] = BSF_M[i].BSF(Moment_val[i]);
    }
    if (CBSF_switch)
    {
      Contact_Force_val[i]  = BSF_CF[i].BSF(Contact_Force_val[i]);
      Contact_Moment_val[i] = BSF_CM[i].BSF(Contact_Moment_val[i]);
    }
  }
}

void FT_processing::FT_publish()
{
  pub_data.force.x  = Force_val[0];
  pub_data.force.y  = Force_val[1];
  pub_data.force.z  = Force_val[2];
  pub_data.torque.x = Moment_val[0];
  pub_data.torque.y = Moment_val[1];
  pub_data.torque.z = Moment_val[2];
  ftsensor_pub_->publish(pub_data);

  Cpub_data.force.x  = Contact_Force_val[0];
  Cpub_data.force.y  = Contact_Force_val[1];
  Cpub_data.force.z  = Contact_Force_val[2];
  Cpub_data.torque.x = Contact_Moment_val[0];
  Cpub_data.torque.y = Contact_Moment_val[1];
  Cpub_data.torque.z = Contact_Moment_val[2];
  Cftsensor_pub_->publish(Cpub_data);

  geometry_msgs::msg::Vector3 force_msg;
  force_msg.x = Contact_Force_val[0];
  force_msg.y = Contact_Force_val[1];
  force_msg.z = Contact_Force_val[2];
  vive_force_pub_->publish(force_msg);

  geometry_msgs::msg::Vector3 moment_msg;
  moment_msg.x = Contact_Moment_val[0];
  moment_msg.y = Contact_Moment_val[1];
  moment_msg.z = Contact_Moment_val[2];
  vive_moment_pub_->publish(moment_msg);

  std_msgs::msg::Float64MultiArray acc_msg;
  acc_msg.data.reserve(9);
  acc_msg.data.push_back(CPos_acc_val[0]);
  acc_msg.data.push_back(CPos_acc_val[1]);
  acc_msg.data.push_back(CPos_acc_val[2]);
  acc_msg.data.push_back(CAng_acc_val[0]);
  acc_msg.data.push_back(CAng_acc_val[1]);
  acc_msg.data.push_back(CAng_acc_val[2]);
  acc_msg.data.push_back(CAng_vel_val[0]);
  acc_msg.data.push_back(CAng_vel_val[1]);
  acc_msg.data.push_back(CAng_vel_val[2]);
  vive_acc_pub_->publish(acc_msg);
}

void FT_processing::FT_print()
{
  if (YamlPrint_switch == 1)
  {
    printf("Fx:%10f, Fy:%10f, Fz:%10f \n", Force_val[0], Force_val[1], Force_val[2]);
    printf("Mx:%10f, My:%10f, Mz:%10f \n", Moment_val[0], Moment_val[1], Moment_val[2]);
    printf("CFx:%10f, CFy:%10f, CFz:%10f \n", Contact_Force_val[0], Contact_Force_val[1], Contact_Force_val[2]);
    printf("CMx:%10f, CMy:%10f, CMz:%10f \n", Contact_Moment_val[0], Contact_Moment_val[1], Contact_Moment_val[2]);
    if (HaccSwitch)
    {
      printf("Hacc_x:%10f, Hacc_y:%10f, Hacc_z:%10f \n", Pos_acc_val[0], Pos_acc_val[1], Pos_acc_val[2]);
      printf("Hang_acc_x:%10f, Hang_acc_y:%10f, Hang_acc_z:%10f \n", Ang_acc_val[0], Ang_acc_val[1], Ang_acc_val[2]);
      printf("Hang_vel_x:%10f, Hang_vel_y:%10f, Hang_vel_z:%10f \n", Ang_vel_val[0], Ang_vel_val[1], Ang_vel_val[2]);
    }
    if (CaccSwitch)
    {
      printf("Cacc_x:%10f, Cacc_y:%10f, Cacc_z:%10f \n", CPos_acc_val[0], CPos_acc_val[1], CPos_acc_val[2]);
      printf("Cang_acc_x:%10f, Cang_acc_y:%10f, Cang_acc_z:%10f \n", CAng_acc_val[0], CAng_acc_val[1], CAng_acc_val[2]);
      printf("Cang_vel_x:%10f, Cang_vel_y:%10f, Cang_vel_z:%10f \n", CAng_vel_val[0], CAng_vel_val[1], CAng_vel_val[2]);
    }
    printf("--------------------------------------------------\n");
  }
}

void FT_processing::FT_record()
{
  if (YamlData1_switch == 1)
  {
    if (Data1_txt != NULL)
    {
      fprintf(Data1_txt, "%10f %10f %10f %10f %10f %10f\n",
              Force_val[0], Force_val[1], Force_val[2],
              Moment_val[0], Moment_val[1], Moment_val[2]);
    }
    else
    {
      RCLCPP_ERROR(node_->get_logger(), "Data1 does not open : warning !!");
    }
  }
}

bool FT_processing::SRV5_Handle(
  const std::shared_ptr<std_srvs::srv::Empty::Request> /*req*/,
  const std::shared_ptr<std_srvs::srv::Empty::Response> /*res*/)
{
  // [CHANGED] 수신 스레드와 경쟁 방지
  {
    std::lock_guard<std::mutex> lk(g_ft_mtx);
    sensor_init_counter = 0;
  }

  aidinGui_stateMsg.data = "Sensor was initialized";
  aidinGui_statePub->publish(aidinGui_stateMsg);
  return true;
}

void FT_processing::FT_run()
{
  // ------------------------------------------------------------
  // 목표:
  //  - 센서 수신/필터링: 1/Ts_  (Ts_=0.002면 500Hz)
  //  - publish 루프:     1/Publish_sampling (기본 0.002 -> 500Hz)
  //
  // 사용법(YAML):
  //  - Sensor_sampling: 0.002    # 센서/필터 dt (500Hz)
  //  - Publish_sampling: 0.002   # publish 주기 (500Hz)
  // ------------------------------------------------------------

  // 센서 수신 주기(필터 dt): Ts_
  const double Ts_sensor = clamp_positive(Ts_, 0.002);

  // publish 주기: 별도 파라미터로 분리
  double Ts_pub = 0.002; // default 500Hz
  (void)node_->get_parameter("Publish_sampling", Ts_pub);
  Ts_pub = clamp_positive(Ts_pub, 0.002);

  RCLCPP_INFO(node_->get_logger(),
              "[FT_run] sensor dt=%.6f (%.1f Hz), publish dt=%.6f (%.1f Hz)",
              Ts_sensor, 1.0 / Ts_sensor, Ts_pub, 1.0 / Ts_pub);

  double init_sec = 5.0;
  FT_init(static_cast<int>(init_sec / Ts_sensor));
  std::cout << "Sensor was initialized" << std::endl;

  std::atomic<bool> stop_rx(false);

  // -----------------------------
  // (1) RX thread @ sensor rate
  // -----------------------------
  std::thread rx_thread([&]() {
    rclcpp::WallRate rx_rate(1.0 / Ts_sensor);

    while (rclcpp::ok() && !stop_rx.load())
    {
      // TCP_start()가 블로킹이어도, publish 루프는 영향 안 받게 분리함.
      if (TCP_start() != 0)
      {
        std::lock_guard<std::mutex> lk(g_ft_mtx);

        // 새 프레임 들어왔을 때만 init/filter 진행
        if (Sensor_value_init())
        {
          FT_filtering();
        }
      }

      // 수신 시도 주기 유지 (논블로킹이면 500Hz로 polling)
      rx_rate.sleep();
    }
  });

  // -----------------------------
  // (2) Publish loop 500Hz
  // -----------------------------
  rclcpp::WallRate pub_rate(1.0 / Ts_pub);

  while (rclcpp::ok())
  {
    // 스냅샷 복사 (락을 짧게)
    bool init_done = false;

    double F[3], M[3], CF[3], CM[3];
    double CPosAcc[3], CAngAcc[3], CAngVel[3];

    {
      std::lock_guard<std::mutex> lk(g_ft_mtx);

      // 초기화 완료 판단: sensor_init_counter가 init_average_num 이상이면 완료로 간주
      init_done = (sensor_init_counter >= init_average_num);

      if (init_done)
      {
        for (int i = 0; i < 3; i++)
        {
          F[i]  = Force_val[i];
          M[i]  = Moment_val[i];
          CF[i] = Contact_Force_val[i];
          CM[i] = Contact_Moment_val[i];

          CPosAcc[i] = CPos_acc_val[i];
          CAngAcc[i] = CAng_acc_val[i];
          CAngVel[i] = CAng_vel_val[i];
        }
      }
    }

    // 초기화 완료 이후에만 publish/print/record
    if (init_done)
    {
      // --- publish (snapshot 사용) ---
      geometry_msgs::msg::Wrench hw, cw;
      hw.force.x  = F[0];  hw.force.y  = F[1];  hw.force.z  = F[2];
      hw.torque.x = M[0];  hw.torque.y = M[1];  hw.torque.z = M[2];
      ftsensor_pub_->publish(hw);

      cw.force.x  = CF[0]; cw.force.y  = CF[1]; cw.force.z  = CF[2];
      cw.torque.x = CM[0]; cw.torque.y = CM[1]; cw.torque.z = CM[2];
      Cftsensor_pub_->publish(cw);

      geometry_msgs::msg::Vector3 force_msg;
      force_msg.x = CF[0];
      force_msg.y = CF[1];
      force_msg.z = CF[2];
      vive_force_pub_->publish(force_msg);

      geometry_msgs::msg::Vector3 moment_msg;
      moment_msg.x = CM[0];
      moment_msg.y = CM[1];
      moment_msg.z = CM[2];
      vive_moment_pub_->publish(moment_msg);

      std_msgs::msg::Float64MultiArray acc_msg;
      acc_msg.data.reserve(9);
      acc_msg.data.push_back(CPosAcc[0]);
      acc_msg.data.push_back(CPosAcc[1]);
      acc_msg.data.push_back(CPosAcc[2]);
      acc_msg.data.push_back(CAngAcc[0]);
      acc_msg.data.push_back(CAngAcc[1]);
      acc_msg.data.push_back(CAngAcc[2]);
      acc_msg.data.push_back(CAngVel[0]);
      acc_msg.data.push_back(CAngVel[1]);
      acc_msg.data.push_back(CAngVel[2]);
      vive_acc_pub_->publish(acc_msg);

      // --- print (snapshot 사용) ---
      if (YamlPrint_switch == 1)
      {
        printf("Fx:%10f, Fy:%10f, Fz:%10f \n", F[0], F[1], F[2]);
        printf("Mx:%10f, My:%10f, Mz:%10f \n", M[0], M[1], M[2]);
        printf("CFx:%10f, CFy:%10f, CFz:%10f \n", CF[0], CF[1], CF[2]);
        printf("CMx:%10f, CMy:%10f, CMz:%10f \n", CM[0], CM[1], CM[2]);
        printf("--------------------------------------------------\n");
      }

      // --- record (snapshot 사용) ---
      if (YamlData1_switch == 1)
      {
        if (Data1_txt != NULL)
        {
          fprintf(Data1_txt, "%10f %10f %10f %10f %10f %10f\n",
                  F[0], F[1], F[2],
                  M[0], M[1], M[2]);
        }
        else
        {
          RCLCPP_ERROR(node_->get_logger(), "Data1 does not open : warning !!");
        }
      }
    }

    rclcpp::spin_some(node_);
    pub_rate.sleep();
  }

  // 종료 처리
  stop_rx.store(true);
  if (rx_thread.joinable())
  {
    rx_thread.join();
  }
}
