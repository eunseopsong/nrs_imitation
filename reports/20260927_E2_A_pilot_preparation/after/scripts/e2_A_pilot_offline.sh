#!/usr/bin/env bash
# Only Python fixtures/CPU inference. No ROS node, launch run, publisher or service.
set -e
source /opt/ros/humble/setup.bash
source /home/eunseop/dev_ws/install/setup.bash
source /home/eunseop/nrs_imitation/behavior_ws/install/setup.bash
export PYTHONPATH=/home/eunseop/nrs_imitation/behavior_ws/src/nrs_imitation:/home/eunseop/nrs_imitation/source:$PYTHONPATH
export HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
exec /home/eunseop/miniconda3/envs/nrs_imitation/bin/python3 /home/eunseop/nrs_imitation/scripts/e2_A_pilot_offline.py "$@"
