#!/usr/bin/env bash
# FLOW training on the stain-relative dataset, take 2.
#
# 20260910: 20260910_90deg_rel_single/imitation_form (42 EP, 90deg only).
#   - x,y relativized by ONE clean origin (457.4, 375.0) for all 42 episodes.
#     The per-episode dark detection (stain_origin_90deg_only.json) scattered
#     42mm -- shown to be an artifact of non-repeatable episode start poses +
#     eye-in-hand extrinsic error (corr(origin_err, start_pose_err)=0.79), not
#     real stain movement (the stain is fixed to +/-3.5mm). One origin => all
#     42 episodes kept, zero label noise.
#   - observation force fx,fy are NOT zeroed this time (the fzobs run
#     20260909_1530 drifted once the stain faded; the abs policy keeps going
#     on pos+force, so the shear channel matters -- keep it).
#
# Hyperparameters identical to 20260827_2128_only90deg / 20260909_1530
# (DINOv3 ViT-S frozen + attention tcp_roi, phase_resample, 500 epochs).
set -euo pipefail
cd "$(dirname "$0")/../.."

TS=20260910
python3 scripts/flow/train_flow_single_cam.py \
  --dataset_dir datasets/polishing/single_cam/20260910_90deg_rel_single/imitation_form \
  --ckpt_dir    checkpoints/flow/polishing/single_cam/${TS}_90deg_rel_single \
  --image_backbone dinov3 \
  --dino_model_name vit_small_patch16_dinov3.lvd1689m \
  --dino_roi_pooling attention \
  --freeze_image_backbone \
  --use_tcp_roi \
  --tcp_roi_area_fraction 0.25 \
  --phase_resample_enable \
  --num_epochs 500 \
  --seed 0 \
  > logs/flow_train_90deg_rel_single_${TS}.log 2>&1

echo "done -> checkpoints/flow/polishing/single_cam/${TS}_90deg_rel_single"
