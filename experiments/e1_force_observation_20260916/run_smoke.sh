#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 scripts/flow/e1_force_observation_check.py \
  --dataset_dir datasets/polishing/single_cam/20260910_90deg_rel_single/imitation_form \
  --output_dir experiments/e1_force_observation_20260916/results/smoke --smoke
