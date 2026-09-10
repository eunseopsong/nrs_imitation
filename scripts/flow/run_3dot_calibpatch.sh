#!/usr/bin/env bash
# FLOW training on the 3-dot-row removal dataset 20260901_2141_calibpatch
# (32 EP: dropout episodes removed, per-regime calib correction to z=173,
#  xy aligned to the 2-dot diagonal, ep0-7 xy re-aligned to the body,
#  non-contact force zeroed via z<185 gate).
# 3-DOT DATA ONLY -- not merged with the 2-dot set.
# Config identical to the 2-dot run (run_2dot_calibpatch_*.sh) for comparability.
set -euo pipefail
cd "$(dirname "$0")/../.."
TS="${1:-$(date +%Y%m%d_%H%M)}"
python3 scripts/flow/train_flow_single_cam.py \
  --dataset_dir datasets/polishing/single_cam/20260901_2141_calibpatch/imitation_form \
  --ckpt_dir checkpoints/flow/polishing/single_cam/${TS}_3dot_calibpatch \
  --image_backbone dinov3 \
  --dino_model_name vit_small_patch16_dinov3.lvd1689m \
  --dino_roi_pooling attention \
  --freeze_image_backbone \
  --use_tcp_roi \
  --tcp_roi_area_fraction 0.25 \
  --phase_resample_enable \
  --num_epochs 500 \
  --seed 0
