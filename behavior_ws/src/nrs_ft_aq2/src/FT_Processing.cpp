#include "FT_Processing.hpp"
#include <cstdlib>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "ament_index_cpp/get_package_share_directory.hpp"

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
using Vec3 = std::array<double, 3>;
using Mat3 = std::array<std::array<double, 3>, 3>;

struct StlMassProperties
{
  double volume_m3 = 0.0;
  Vec3 centroid_m = {0.0, 0.0, 0.0};
  std::uint32_t triangle_count = 0;
};

// FT 데이터 공유 보호용 (단일 노드 전제)
std::mutex g_ft_mtx;

// 안전 가드
inline double clamp_positive(double v, double fallback)
{
  return (std::isfinite(v) && v > 1e-9) ? v : fallback;
}

std::string expand_home_path(const std::string& path)
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

std::string resolve_package_relative_path(const std::string& raw_path)
{
  const std::string expanded_path = expand_home_path(raw_path);
  if (expanded_path.empty()) {
    return expanded_path;
  }

  const std::filesystem::path path(expanded_path);
  if (path.is_absolute()) {
    return expanded_path;
  }

  try {
    const std::filesystem::path share_dir =
      ament_index_cpp::get_package_share_directory("nrs_ft_aq2");
    const std::filesystem::path package_relative = share_dir / path;
    if (std::filesystem::exists(package_relative)) {
      return package_relative.string();
    }

    if (!path.has_parent_path()) {
      const std::filesystem::path mesh_relative = share_dir / "mesh" / path;
      if (std::filesystem::exists(mesh_relative)) {
        return mesh_relative.string();
      }
    }

    return package_relative.string();
  } catch (const std::exception&) {
    return expanded_path;
  }
}

Vec3 cross_vec(const Vec3& a, const Vec3& b)
{
  return {
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  };
}

double dot_vec(const Vec3& a, const Vec3& b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void accumulate_triangle(const Vec3& a,
                         const Vec3& b,
                         const Vec3& c,
                         double& signed_volume,
                         Vec3& centroid_moment)
{
  const double volume = dot_vec(a, cross_vec(b, c)) / 6.0;
  signed_volume += volume;

  for (int i = 0; i < 3; ++i) {
    centroid_moment[i] += volume * (a[i] + b[i] + c[i]) / 4.0;
  }
}

float read_float_le(const char* data)
{
  float value = 0.0f;
  std::memcpy(&value, data, sizeof(float));
  return value;
}

bool read_binary_stl_mass_properties(const std::string& path,
                                     double unit_scale,
                                     std::uint32_t triangle_count,
                                     StlMassProperties& props,
                                     std::string& error_msg)
{
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    error_msg = "failed to open binary STL";
    return false;
  }

  ifs.seekg(84, std::ios::beg);

  double signed_volume = 0.0;
  Vec3 centroid_moment = {0.0, 0.0, 0.0};

  for (std::uint32_t tri = 0; tri < triangle_count; ++tri) {
    char record[50];
    if (!ifs.read(record, sizeof(record))) {
      error_msg = "unexpected EOF while reading binary STL triangles";
      return false;
    }

    Vec3 vertices[3];
    for (int v = 0; v < 3; ++v) {
      const int offset = 12 + v * 12;
      vertices[v] = {
        static_cast<double>(read_float_le(record + offset + 0)) * unit_scale,
        static_cast<double>(read_float_le(record + offset + 4)) * unit_scale,
        static_cast<double>(read_float_le(record + offset + 8)) * unit_scale
      };
    }

    accumulate_triangle(vertices[0], vertices[1], vertices[2],
                        signed_volume, centroid_moment);
  }

  if (std::abs(signed_volume) < 1e-15) {
    error_msg = "STL signed volume is too small; mesh may be open or invalid";
    return false;
  }

  props.volume_m3 = std::abs(signed_volume);
  for (int i = 0; i < 3; ++i) {
    props.centroid_m[i] = centroid_moment[i] / signed_volume;
  }
  props.triangle_count = triangle_count;
  return true;
}

bool read_ascii_stl_mass_properties(const std::string& path,
                                    double unit_scale,
                                    StlMassProperties& props,
                                    std::string& error_msg)
{
  std::ifstream ifs(path);
  if (!ifs) {
    error_msg = "failed to open ASCII STL";
    return false;
  }

  double signed_volume = 0.0;
  Vec3 centroid_moment = {0.0, 0.0, 0.0};
  Vec3 tri_vertices[3];
  int vertex_idx = 0;
  std::uint32_t triangle_count = 0;

  std::string token;
  while (ifs >> token) {
    if (token != "vertex") {
      continue;
    }

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (!(ifs >> x >> y >> z)) {
      error_msg = "failed to parse ASCII STL vertex";
      return false;
    }

    tri_vertices[vertex_idx] = {
      x * unit_scale,
      y * unit_scale,
      z * unit_scale
    };
    ++vertex_idx;

    if (vertex_idx == 3) {
      accumulate_triangle(tri_vertices[0], tri_vertices[1], tri_vertices[2],
                          signed_volume, centroid_moment);
      vertex_idx = 0;
      ++triangle_count;
    }
  }

  if (triangle_count == 0) {
    error_msg = "no triangles found in ASCII STL";
    return false;
  }
  if (std::abs(signed_volume) < 1e-15) {
    error_msg = "STL signed volume is too small; mesh may be open or invalid";
    return false;
  }

  props.volume_m3 = std::abs(signed_volume);
  for (int i = 0; i < 3; ++i) {
    props.centroid_m[i] = centroid_moment[i] / signed_volume;
  }
  props.triangle_count = triangle_count;
  return true;
}

bool read_stl_mass_properties(const std::string& raw_path,
                              double unit_scale,
                              StlMassProperties& props,
                              std::string& error_msg)
{
  if (unit_scale <= 0.0 || !std::isfinite(unit_scale)) {
    error_msg = "stl_unit_scale must be positive";
    return false;
  }

  const std::string path = expand_home_path(raw_path);
  std::ifstream probe(path, std::ios::binary | std::ios::ate);
  if (!probe) {
    error_msg = "failed to open STL path: " + path;
    return false;
  }

  const auto file_size_pos = probe.tellg();
  if (file_size_pos == std::ifstream::pos_type(-1)) {
    error_msg = "failed to determine STL file size: " + path;
    return false;
  }

  const auto file_size = static_cast<std::uintmax_t>(
    static_cast<std::streamoff>(file_size_pos));
  if (file_size < 84) {
    probe.close();
    return read_ascii_stl_mass_properties(path, unit_scale, props, error_msg);
  }

  probe.seekg(80, std::ios::beg);
  std::uint32_t triangle_count = 0;
  if (!probe.read(reinterpret_cast<char*>(&triangle_count), sizeof(triangle_count))) {
    error_msg = "failed to read binary STL triangle count";
    return false;
  }
  probe.close();

  const std::uintmax_t expected_binary_size =
    84ull + static_cast<std::uintmax_t>(triangle_count) * 50ull;

  if (expected_binary_size == file_size && triangle_count > 0) {
    return read_binary_stl_mass_properties(path, unit_scale, triangle_count,
                                           props, error_msg);
  }

  return read_ascii_stl_mass_properties(path, unit_scale, props, error_msg);
}

Mat3 identity_matrix()
{
  return {{{1.0, 0.0, 0.0},
           {0.0, 1.0, 0.0},
           {0.0, 0.0, 1.0}}};
}

Mat3 multiply_matrix(const Mat3& a, const Mat3& b)
{
  Mat3 result = {{{0.0, 0.0, 0.0},
                  {0.0, 0.0, 0.0},
                  {0.0, 0.0, 0.0}}};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return result;
}

Vec3 multiply_matrix_vector(const Mat3& matrix, const Vec3& vector)
{
  Vec3 result = {0.0, 0.0, 0.0};
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) {
      result[i] += matrix[i][k] * vector[k];
    }
  }
  return result;
}

Vec3 multiply_matrix_transpose_vector(const Mat3& matrix, const Vec3& vector)
{
  Vec3 result = {0.0, 0.0, 0.0};
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) {
      result[i] += matrix[k][i] * vector[k];
    }
  }
  return result;
}

Mat3 rpy_to_matrix(const Vec3& rpy)
{
  const double cr = std::cos(rpy[0]);
  const double sr = std::sin(rpy[0]);
  const double cp = std::cos(rpy[1]);
  const double sp = std::sin(rpy[1]);
  const double cy = std::cos(rpy[2]);
  const double sy = std::sin(rpy[2]);

  return {{
    {{cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr}},
    {{sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr}},
    {{-sp,     cp * sr,                cp * cr}}
  }};
}

Mat3 axis_angle_to_matrix(const Vec3& w)
{
  const double theta = std::sqrt(dot_vec(w, w));
  if (theta < 1e-8) {
    return identity_matrix();
  }

  const Vec3 k = {w[0] / theta, w[1] / theta, w[2] / theta};
  const Mat3 K = {{{0.0,   -k[2],  k[1]},
                   {k[2],   0.0,  -k[0]},
                   {-k[1],  k[0],  0.0}}};
  const Mat3 K2 = multiply_matrix(K, K);
  Mat3 result = identity_matrix();
  const double s = std::sin(theta);
  const double c = std::cos(theta);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      result[i][j] += s * K[i][j] + (1.0 - c) * K2[i][j];
    }
  }
  return result;
}

Vec3 transform_point(const Vec3& point, const Vec3& translation, const Vec3& rpy)
{
  const Mat3 R = rpy_to_matrix(rpy);
  Vec3 transformed = translation;
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) {
      transformed[i] += R[i][k] * point[k];
    }
  }
  return transformed;
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

  configureGravityCompensation();

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

void FT_processing::configureGravityCompensation()
{
  (void)node_->get_parameter("gravity_compensation_enabled", gravity_compensation_enabled_);
  (void)node_->get_parameter("gravity_compensation_apply_handle", gravity_apply_handle_);
  (void)node_->get_parameter("gravity_compensation_apply_contact", gravity_apply_contact_);

  calibrated_pose_topic_ = "/calibrated_pose";
  (void)node_->get_parameter("calibrated_pose_topic", calibrated_pose_topic_);

  double manual_mass = 0.0;
  (void)node_->get_parameter("tool_mass", manual_mass);
  tool_mass_ = manual_mass;

  std::vector<double> manual_cog = {0.0, 0.0, 0.0};
  (void)node_->get_parameter("tool_cog", manual_cog);
  if (manual_cog.size() == 3) {
    tool_cog_ = {manual_cog[0], manual_cog[1], manual_cog[2]};
  } else {
    RCLCPP_WARN(node_->get_logger(),
                "tool_cog must have exactly 3 elements. Using [0, 0, 0].");
    tool_cog_ = {0.0, 0.0, 0.0};
  }
  tool_param_source_ = "manual parameters";

  std::string raw_stl_path = "mesh/tcp_under_polishing.stl";
  (void)node_->get_parameter("tool_stl_path", raw_stl_path);
  tool_stl_path_ = resolve_package_relative_path(raw_stl_path);

  double stl_unit_scale = 0.001;  // mm -> m default
  (void)node_->get_parameter("stl_unit_scale", stl_unit_scale);

  double tool_density = 1240.0;  // PLA [kg/m^3]
  (void)node_->get_parameter("tool_density", tool_density);

  double tool_mass_override = 0.0;
  (void)node_->get_parameter("tool_mass_override", tool_mass_override);

  std::vector<double> stl_to_sensor_xyz_param = {0.0, 0.0, 0.0};
  std::vector<double> stl_to_sensor_rpy_param = {0.0, 0.0, 0.0};
  std::vector<double> tracker_to_sensor_rpy_param = {0.0, 0.0, 0.0};
  (void)node_->get_parameter("stl_to_sensor_xyz", stl_to_sensor_xyz_param);
  (void)node_->get_parameter("stl_to_sensor_rpy", stl_to_sensor_rpy_param);
  (void)node_->get_parameter("tracker_to_sensor_rpy", tracker_to_sensor_rpy_param);

  const auto vector_to_vec3 = [this](const std::vector<double>& value,
                                     const Vec3& fallback,
                                     const char* name) -> Vec3 {
    if (value.size() != 3) {
      RCLCPP_WARN(node_->get_logger(),
                  "%s must have exactly 3 elements. Using fallback.", name);
      return fallback;
    }
    return {value[0], value[1], value[2]};
  };

  const Vec3 stl_to_sensor_xyz =
    vector_to_vec3(stl_to_sensor_xyz_param, {0.0, 0.0, 0.0}, "stl_to_sensor_xyz");
  const Vec3 stl_to_sensor_rpy =
    vector_to_vec3(stl_to_sensor_rpy_param, {0.0, 0.0, 0.0}, "stl_to_sensor_rpy");
  const Vec3 tracker_to_sensor_rpy =
    vector_to_vec3(tracker_to_sensor_rpy_param, {0.0, 0.0, 0.0}, "tracker_to_sensor_rpy");

  tracker_to_sensor_rot_ = rpy_to_matrix(tracker_to_sensor_rpy);

  tool_stl_volume_m3_ = 0.0;
  tool_stl_centroid_m_ = {0.0, 0.0, 0.0};
  if (!tool_stl_path_.empty()) {
    StlMassProperties props;
    std::string error_msg;
    if (read_stl_mass_properties(tool_stl_path_, stl_unit_scale, props, error_msg)) {
      tool_stl_volume_m3_ = props.volume_m3;
      tool_stl_centroid_m_ = props.centroid_m;
      tool_cog_ = transform_point(props.centroid_m, stl_to_sensor_xyz, stl_to_sensor_rpy);

      if (tool_mass_override > 0.0 && std::isfinite(tool_mass_override)) {
        tool_mass_ = tool_mass_override;
        tool_param_source_ = "STL CoM + tool_mass_override";
      } else if (tool_density > 0.0 && std::isfinite(tool_density)) {
        tool_mass_ = tool_density * props.volume_m3;
        tool_param_source_ = "STL CoM + density*volume";
      } else {
        tool_param_source_ = "STL CoM + manual tool_mass";
        RCLCPP_WARN(node_->get_logger(),
                    "tool_density is not positive and tool_mass_override is not set. "
                    "Using tool_mass for mass.");
      }

      RCLCPP_INFO(node_->get_logger(),
                  "Loaded STL mass properties: path=%s, triangles=%u, volume=%.9e [m^3], "
                  "centroid_stl=[%.6f, %.6f, %.6f] [m]",
                  tool_stl_path_.c_str(),
                  props.triangle_count,
                  props.volume_m3,
                  props.centroid_m[0],
                  props.centroid_m[1],
                  props.centroid_m[2]);
    } else {
      RCLCPP_WARN(node_->get_logger(),
                  "Failed to load tool_stl_path='%s': %s. Using manual tool_mass/tool_cog.",
                  tool_stl_path_.c_str(),
                  error_msg.c_str());
    }
  }

  calibrated_pose_sub_ =
    node_->create_subscription<std_msgs::msg::Float64MultiArray>(
      calibrated_pose_topic_,
      10,
      std::bind(&FT_processing::calibratedPoseCB, this, std::placeholders::_1));

  RCLCPP_INFO(node_->get_logger(),
              "Gravity compensation: enabled=%s, apply_handle=%s, apply_contact=%s, "
              "pose_topic=%s, source=%s, mass=%.6f [kg], cog_sensor=[%.6f, %.6f, %.6f] [m]",
              gravity_compensation_enabled_ ? "true" : "false",
              gravity_apply_handle_ ? "true" : "false",
              gravity_apply_contact_ ? "true" : "false",
              calibrated_pose_topic_.c_str(),
              tool_param_source_.c_str(),
              tool_mass_,
              tool_cog_[0],
              tool_cog_[1],
              tool_cog_[2]);
}

void FT_processing::calibratedPoseCB(const std_msgs::msg::Float64MultiArray::ConstSharedPtr msg)
{
  if (msg->data.size() < 6) {
    RCLCPP_WARN(node_->get_logger(),
                "calibrated_pose message must contain [x, y, z, wx, wy, wz].");
    return;
  }

  const Vec3 axis_angle = {msg->data[3], msg->data[4], msg->data[5]};
  const Mat3 world_to_tracker = axis_angle_to_matrix(axis_angle);

  std::lock_guard<std::mutex> lk(pose_mutex_);
  latest_world_to_tracker_rot_ = world_to_tracker;
  has_calibrated_pose_ = true;
}

void FT_processing::resetGravityReference()
{
  gravity_reference_set_ = false;
  missing_pose_warned_ = false;
}

void FT_processing::applyGravityCompensation(double force[3], double moment[3])
{
  if (!gravity_compensation_enabled_ || tool_mass_ <= 0.0) {
    return;
  }

  Mat3 world_to_tracker;
  {
    std::lock_guard<std::mutex> lk(pose_mutex_);
    if (!has_calibrated_pose_) {
      if (!missing_pose_warned_) {
        RCLCPP_WARN(node_->get_logger(),
                    "Gravity compensation waiting for calibrated pose topic: %s",
                    calibrated_pose_topic_.c_str());
        missing_pose_warned_ = true;
      }
      return;
    }
    world_to_tracker = latest_world_to_tracker_rot_;
  }

  const Vec3 gravity_world = {0.0, 0.0, -9.81};
  const Vec3 gravity_tracker =
    multiply_matrix_transpose_vector(world_to_tracker, gravity_world);
  const Vec3 gravity_sensor =
    multiply_matrix_vector(tracker_to_sensor_rot_, gravity_tracker);

  if (!gravity_reference_set_) {
    gravity_sensor_init_ = gravity_sensor;
    gravity_reference_set_ = true;
  }

  Vec3 gravity_force = {0.0, 0.0, 0.0};
  for (int i = 0; i < 3; ++i) {
    gravity_force[i] = tool_mass_ * (gravity_sensor[i] - gravity_sensor_init_[i]);
  }

  const Vec3 gravity_moment = cross_vec(tool_cog_, gravity_force);
  for (int i = 0; i < 3; ++i) {
    force[i] -= gravity_force[i];
    moment[i] -= gravity_moment[i];
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
  resetGravityReference();
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
  resetGravityReference();

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
      if (gravity_apply_handle_) {
        applyGravityCompensation(F, M);
      }
      if (gravity_apply_contact_) {
        applyGravityCompensation(CF, CM);
      }

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
