#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train ACT policy on the latest dual-camera imitation_form dataset."""

from __future__ import annotations

from pathlib import Path

from train_act import build_arg_parser, main as run_training


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "polishing" / "dual_cam"
CKPT_ROOT = PROJECT_ROOT / "checkpoints" / "act" / "polishing"


def main() -> None:
    parser = build_arg_parser()
    parser.description = "Train ACT policy with dual-camera imitation_form data."
    parser.set_defaults(
        obs_mode="dual_cam",
        camera_names=["cam0", "cam1"],
        dataset_root=str(DATASET_ROOT),
        ckpt_root=str(CKPT_ROOT),
        debug_batches=-1,
    )
    args = parser.parse_args()
    args.obs_mode = "dual_cam"
    args.camera_names = ["cam0", "cam1"]
    args.dataset_root = str(DATASET_ROOT) if not args.dataset_root else args.dataset_root
    args.ckpt_root = str(CKPT_ROOT) if not args.ckpt_root else args.ckpt_root
    run_training(args)


if __name__ == "__main__":
    main()
