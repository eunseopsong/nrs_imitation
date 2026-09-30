#!/usr/bin/env bash
# Source this file before direct ros2 launch/service commands; starts no nodes.
source /home/eunseop/miniconda3/etc/profile.d/conda.sh
conda activate nrs_imitation
source /opt/ros/humble/setup.bash
source /home/eunseop/dev_ws/install/setup.bash
source /home/eunseop/nrs_imitation/behavior_ws/install/setup.bash
export ROS_DOMAIN_ID=30 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
export ROS_LOG_DIR=/home/eunseop/nrs_imitation/results/ros_logs
