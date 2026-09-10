#!/usr/bin/env bash
# FLOW training on the stain-relative + fz-only-observation dataset.
# 20260909: 20260909_90deg_only_rel_fzobs/imitation_form (42 EP, 90deg only).
# Hyperparameters identical to the reference run 20260827_2128_only90deg
# (DINOv3 ViT-S frozen + attention tcp_roi, phase_resample, 500 epochs).
#
# NO model changes: state_dim=9, force_dim=3, action_dim=9 unchanged. The
# dataset carries relative x,y and observations/force with fx,fy=0; the training
# recomputes dataset_stats.pkl so normalisation adapts (relative x,y centred
# near 0; fx,fy degenerate -> sanitised, no signal).
set -euo pipefail
cd "$(dirname "$0")/../.."

TS=20260909
python3 scripts/flow/train_flow_single_cam.py \
  --dataset_dir datasets/polishing/single_cam/20260909_90deg_only_rel_fzobs/imitation_form \
  --ckpt_dir    checkpoints/flow/polishing/single_cam/${TS}_90deg_rel_fzobs \
  --image_backbone dinov3 \
  --dino_model_name vit_small_patch16_dinov3.lvd1689m \
  --dino_roi_pooling attention \
  --freeze_image_backbone \
  --use_tcp_roi \
  --tcp_roi_area_fraction 0.25 \
  --phase_resample_enable \
  --num_epochs 500 \
  --seed 0 \
  > logs/flow_train_90deg_rel_fzobs_${TS}.log 2>&1

echo "done -> checkpoints/flow/polishing/single_cam/${TS}_90deg_rel_fzobs"
