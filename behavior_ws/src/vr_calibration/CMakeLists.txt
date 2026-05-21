cmake_minimum_required(VERSION 3.8)
project(vr_calibration)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# =========================
# Dependencies
# =========================
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

# =========================
# Solver library (NO main inside)
# =========================
add_library(vr_calibration_solver
  src/vr_calibration_solver.cpp
)

target_compile_features(vr_calibration_solver PUBLIC cxx_std_17)

target_include_directories(vr_calibration_solver PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(vr_calibration_solver
  rclcpp
  std_msgs
  geometry_msgs
  std_srvs
)

target_link_libraries(vr_calibration_solver
  yaml-cpp
  Eigen3::Eigen
)

# =========================
# Executable (ONLY main here)
# =========================
add_executable(vr_calibration
  src/vr_calibration.cpp
)

target_compile_features(vr_calibration PUBLIC cxx_std_17)

target_include_directories(vr_calibration PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(vr_calibration
  ament_index_cpp
  rclcpp
  std_msgs
  geometry_msgs
  std_srvs
)

target_link_libraries(vr_calibration
  vr_calibration_solver
  yaml-cpp
  Eigen3::Eigen
)

# =========================
# Install
# =========================
install(TARGETS vr_calibration vr_calibration_solver
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY data/
  DESTINATION share/${PROJECT_NAME}/yaml
)

install(DIRECTORY txt/
  DESTINATION share/${PROJECT_NAME}/txt
)

ament_package()
