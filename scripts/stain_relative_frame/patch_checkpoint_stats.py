#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-time: backfill the stain-relative-frame markers into a checkpoint's
dataset_stats.pkl.

Checkpoints trained BEFORE flow_train_core.carry_forward_relative_frame_stats
existed do not carry use_relative_position / relative_transform_version /
observation_force_xy_zeroed in their own dataset_stats.pkl -- the training
recomputes stats from scratch. inference_core._load_relative_frame_flags can
fall back to the source dataset's stats, but backfilling the checkpoint makes
the checkpoint self-describing (and independent of the dataset still being on
disk at the recorded path).

    python3 scripts/stain_relative_frame/patch_checkpoint_stats.py \
        checkpoints/flow/polishing/single_cam/20260909_90deg_rel_fzobs/20260909_1530

The source dataset stats path is read from the checkpoint's own
`dataset_dir`, or pass --dataset_stats explicitly.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

KEYS = (
    "use_relative_position",
    "relative_transform_version",
    "rotation_aligned",
    "stain_origin_report",
    "source_dataset_dir",
    "observation_force_xy_zeroed",
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt_dir", type=str, help="dir containing policy_best.ckpt + dataset_stats.pkl")
    ap.add_argument("--dataset_stats", type=str, default=None,
                    help="source (converted-dataset) dataset_stats.pkl; "
                         "default: <ckpt dataset_dir>/dataset_stats.pkl")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)

    ckpt_stats_path = Path(args.ckpt_dir).expanduser() / "dataset_stats.pkl"
    if not ckpt_stats_path.is_file():
        print(f"[patch] not found: {ckpt_stats_path}", file=sys.stderr)
        return 1
    with open(ckpt_stats_path, "rb") as f:
        ckpt_stats = pickle.load(f)

    if args.dataset_stats:
        src_path = Path(args.dataset_stats).expanduser()
    else:
        ds_dir = str(ckpt_stats.get("dataset_dir", "") or "")
        if not ds_dir:
            print("[patch] checkpoint stats has no dataset_dir; pass --dataset_stats", file=sys.stderr)
            return 1
        src_path = Path(ds_dir).expanduser() / "dataset_stats.pkl"
        if not src_path.is_file():
            src_path = Path(__file__).resolve().parents[2] / ds_dir / "dataset_stats.pkl"
    if not src_path.is_file():
        print(f"[patch] source dataset stats not found: {src_path}", file=sys.stderr)
        return 1
    with open(src_path, "rb") as f:
        ds_stats = pickle.load(f)

    if not ds_stats.get("use_relative_position", False):
        print(f"[patch] {src_path} has use_relative_position=False -> nothing to backfill")
        return 0

    changes = {}
    for k in KEYS:
        if k in ds_stats and ckpt_stats.get(k) != ds_stats[k]:
            changes[k] = ds_stats[k]
    if not changes:
        print("[patch] checkpoint stats already up to date")
        return 0

    print(f"[patch] {ckpt_stats_path}")
    for k, v in changes.items():
        print(f"[patch]   {k} = {v!r}")
    if args.dry_run:
        print("[patch] --dry_run: not written")
        return 0

    ckpt_stats.update(changes)
    backup = ckpt_stats_path.with_suffix(".pkl.pre_relframe_backup")
    if not backup.exists():
        backup.write_bytes(ckpt_stats_path.read_bytes())
        print(f"[patch] backup -> {backup}")
    with open(ckpt_stats_path, "wb") as f:
        pickle.dump(ckpt_stats, f)
    print("[patch] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
