#!/usr/bin/env bash
# FLOW training on the 2-dot-row removal dataset 20260901_1505_calibpatch
# (42 EP: bad episodes dropped, positions calib-corrected to z=173 plane,
#  non-contact force zeroed via z<185 gate -- see
#  scripts/data/calibpatch_20260901_1505*.py).
# Config mirrors the per-direction FLOW runs (run_phasematch90_vs_90deg_20260827.sh):
# DINOv3 vit_small frozen + tcp_roi attention pooling, phase_resample, 500 ep, seed 0.
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 scripts/flow/train_flow_single_cam.py \
  --dataset_dir datasets/polishing/single_cam/20260901_1505_calibpatch/imitation_form \
  --ckpt_dir checkpoints/flow/polishing/single_cam/20260901_1652_2dot_calibpatch \
  --image_backbone dinov3 \
  --dino_model_name vit_small_patch16_dinov3.lvd1689m \
  --dino_roi_pooling attention \
  --freeze_image_backbone \
  --use_tcp_roi \
  --tcp_roi_area_fraction 0.25 \
  --phase_resample_enable \
  --num_epochs 500 \
  --seed 0
