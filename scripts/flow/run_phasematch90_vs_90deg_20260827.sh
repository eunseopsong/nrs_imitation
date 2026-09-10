#!/usr/bin/env bash
# Sequential per-direction FLOW training (DINOv3 + tcp_roi tokens).
# 20260827: edited 0deg (phasematch90) vs untouched 90deg, 42 EP each,
# identical hyperparameters, NOT merged. Config mirrors the balanced42
# run 20260825_1208 (area_fraction 0.25, phase_resample), num_epochs 500.
set -euo pipefail
cd "$(dirname "$0")/../.."

TS=20260827_2128
COMMON_ARGS=(
  --image_backbone dinov3
  --dino_model_name vit_small_patch16_dinov3.lvd1689m
  --dino_roi_pooling attention
  --freeze_image_backbone
  --use_tcp_roi
  --tcp_roi_area_fraction 0.25
  --phase_resample_enable
  --num_epochs 500
  --seed 0
)

echo "=================  RUN 1/2: 0deg_phasematch90  ================="
python3 scripts/flow/train_flow_single_cam.py \
  --dataset_dir datasets/polishing/single_cam/20260821_0deg_phasematch90/imitation_form \
  --ckpt_dir checkpoints/flow/polishing/single_cam/${TS}_phasematch90_0deg \
  "${COMMON_ARGS[@]}" \
  > logs/flow_train_phasematch90_0deg_${TS}.log 2>&1

echo "=================  RUN 2/2: 90deg_only  ================="
python3 scripts/flow/train_flow_single_cam.py \
  --dataset_dir datasets/polishing/single_cam/20260821_90deg_only/imitation_form \
  --ckpt_dir checkpoints/flow/polishing/single_cam/${TS}_only90deg \
  "${COMMON_ARGS[@]}" \
  > logs/flow_train_only90deg_${TS}.log 2>&1

echo "=================  BOTH RUNS DONE  ================="
