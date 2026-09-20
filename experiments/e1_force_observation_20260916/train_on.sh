#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -c 'import torch; assert torch.cuda.is_available(), "E1 full training requires CUDA; GPU is not accessible in this environment. Refusing CPU fallback."; print("E1 GPU:", torch.cuda.get_device_name(0))'
python3 scripts/flow/train_flow_single_cam.py \
  --dataset_dir datasets/polishing/single_cam/20260910_90deg_rel_single/imitation_form \
  --ckpt_dir checkpoints/flow/polishing/single_cam/e1_force_observation_20260916/on \
  --image_backbone dinov3 --dino_model_name vit_small_patch16_dinov3.lvd1689m \
  --dino_roi_pooling attention --freeze_image_backbone --use_tcp_roi \
  --tcp_roi_area_fraction 0.25 --phase_resample_enable --num_epochs 500 --seed 0 \
  --use_force_observation
