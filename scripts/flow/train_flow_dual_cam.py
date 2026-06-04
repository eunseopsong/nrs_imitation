#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train Flow policy on the latest dual-camera imitation_form dataset."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from flow_train_core import build_arg_parser, run_one


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "multi_cam"
CKPT_ROOT = PROJECT_ROOT / "checkpoints" / "flow" / "polishing"


def _timestamp_like(name: str) -> bool:
    for fmt in ("%Y%m%d_%H%M", "%Y%m%d%H%M", "%m%d_%H%M"):
        try:
            datetime.strptime(name, fmt)
            return True
        except ValueError:
            pass
    return False


def _count_episodes(path: Path) -> int:
    files = sorted(path.glob("episode_*.hdf5"))
    if not files:
        files = sorted(path.glob("episode_*.h5"))
    return len(files)


def _find_latest_imitation_form(root: Path) -> str:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    candidates = []
    for ep_dir in root.rglob("imitation_form"):
        if not ep_dir.is_dir():
            continue
        n = _count_episodes(ep_dir)
        if n <= 0:
            continue
        run_name = ep_dir.parent.name
        candidates.append((1 if _timestamp_like(run_name) else 0, run_name, ep_dir.stat().st_mtime, ep_dir, n))

    if not candidates:
        raise FileNotFoundError(f"No usable imitation_form/episode_*.hdf5 found under {root}")

    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return str(candidates[0][3])


def main() -> None:
    parser = build_arg_parser()
    parser.description = "Train Flow policy with dual-camera imitation_form data."
    parser.set_defaults(
        obs_mode="dual_cam",
        camera_names=["cam0", "cam1"],
        train_all_obs_modes=False,
        ckpt_root=str(CKPT_ROOT),
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(DATASET_ROOT),
        help="Root used to auto-select the latest imitation_form when --dataset_dir is omitted.",
    )
    args = parser.parse_args()
    args.obs_mode = "dual_cam"
    args.camera_names = ["cam0", "cam1"]
    args.train_all_obs_modes = False

    if not args.dataset_dir:
        args.dataset_dir = _find_latest_imitation_form(Path(args.dataset_root))
        print(f"[AUTO] dataset_dir not provided -> using latest imitation_form: {args.dataset_dir}")

    run_one(args, obs_mode="dual_cam", timestamp=None)


if __name__ == "__main__":
    main()
