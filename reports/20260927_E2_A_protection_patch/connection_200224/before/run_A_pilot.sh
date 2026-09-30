#!/usr/bin/env bash
# Actual A pilot entry. The existing launch preflight and runtime guards apply.
# This script does not build, deploy, start a driver, zero a sensor, or set F0.
set -e

if (( $# > 1 )); then
    printf 'Usage: %s [session_id]\n' "$0" >&2
    exit 64
fi
nrs_a_session="${1:-E2_A_PILOT_01}"
if [[ ! "$nrs_a_session" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]]; then
    printf 'Invalid session_id: use 1-80 letters, digits, underscores, dots or hyphens.\n' >&2
    exit 64
fi

source /home/eunseop/miniconda3/etc/profile.d/conda.sh
conda activate nrs_imitation
source /opt/ros/humble/setup.bash
source /home/eunseop/dev_ws/install/setup.bash
source /home/eunseop/nrs_imitation/behavior_ws/install/setup.bash
export HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2

cd /home/eunseop/nrs_imitation
exec ros2 launch nrs_imitation e2_abc.launch.py \
    mode:=run condition:=A "session:=$nrs_a_session" \
    config:=/home/eunseop/nrs_imitation/experiments/e2_A_pilot_20260927/config.json
