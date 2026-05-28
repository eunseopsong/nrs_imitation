cmake_minimum_required(VERSION 3.8)
project(nrs_ft_aq2)

# 컴파일 옵션
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ROS2 의존성
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(ament_index_cpp REQUIRED)

# 우리 패키지가 볼 include 경로
# - src 에 헤더들(FT_Processing.hpp, CAN_reader.hpp, nrs_filter_*.hpp)이 있으니까 src 추가
# - include/nrs_ft_aq2 도 열어두면 나중에 옮겨도 됨
include_directories(
  src
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/nrs_ft_aq2
)

# 1) 공통 라이브러리 (예전 ros1의 FT_lib 역할)
#    필터 cpp 둘 + CAN_reader + FT_Processing 을 전부 여기 묶는다
add_library(ft_lib
  src/CAN_reader.cpp
  src/FT_Processing.cpp
  src/nrs_filter_core.cpp
  src/nrs_filter_applied.cpp
)

ament_target_dependencies(ft_lib
  rclcpp
  std_msgs
  geometry_msgs
  std_srvs
  ament_index_cpp
)

# 2) can sender 노드
add_executable(can_sender_node
  src/CAN_sender.cpp
)
target_include_directories(can_sender_node PUBLIC src)
ament_target_dependencies(can_sender_node
  rclcpp
  std_msgs
)

# 3) FT 메인 노드
add_executable(ft_aq_main_node
  src/node_ft_aq_main.cpp
)
target_include_directories(ft_aq_main_node PUBLIC src)
target_link_libraries(ft_aq_main_node
  ft_lib
)
ament_target_dependencies(ft_aq_main_node
  rclcpp
  std_msgs
  geometry_msgs
  std_srvs
)

# 4) 설치
install(TARGETS
  ft_lib
  can_sender_node
  ft_aq_main_node
  DESTINATION lib/${PROJECT_NAME}
)

# 헤더류 설치 (src 안에 있는 hpp 도 같이 내보내자)
install(DIRECTORY src/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# launch, config 있으면 같이
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

install(DIRECTORY mesh/
  DESTINATION share/${PROJECT_NAME}/mesh
  OPTIONAL
)

ament_package()
