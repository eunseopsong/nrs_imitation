#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step [0b]: capture the clean-specimen reference at the home pose.

Averages `--frames` (default 40) camera frames of a specimen with NO stain,
shot from the home pose, and records the lighting conditions alongside them.
Step [2] subtracts this reference from the home-pose frame to find the stain,
so the reference and the episodes must share illumination -- the lighting note
is mandatory (pass --lighting) and is written into the npz.

  ros2 run stain_relative_frame clean_reference_capture -- \
      --lighting "ring light 60%, blinds closed, 2026-09-07 14:30"

Output: <artifact_dir>/clean_reference.npz
  reference   (H, W) float32  -- mean grayscale
  frames_rgb  (N, H, W, 3) uint8 (unless --no_store_frames)
  pose_mean / pose_std (6,), lighting, stamp, image_topic, pose_topic
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import load_config
from .ros_utils import collect, make_grabber, pose_stats
from .stain_detect import build_clean_reference


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="[0b] clean-specimen reference capture")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--lighting", type=str, default=None,
                    help="REQUIRED: lighting conditions, recorded with the reference")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--image_topic", type=str, default=None)
    ap.add_argument("--pose_topic", type=str, default=None)
    ap.add_argument("--timeout_sec", type=float, default=20.0)
    ap.add_argument("--no_store_frames", action="store_true",
                    help="store only the mean, not the raw frames")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    lighting = args.lighting if args.lighting is not None else cfg.lighting_note
    if not lighting:
        print("[clean-ref] refusing to save a reference with no lighting note.\n"
              "            Pass --lighting \"...\" -- the reference is only valid under "
              "the illumination it was shot in, and step [2] silently degrades when "
              "that changes.", file=sys.stderr)
        return 1

    n_frames = int(args.frames or cfg.clean_reference_frames)
    image_topic = args.image_topic or cfg.image_topic
    pose_topic = args.pose_topic or cfg.pose_topic
    out_path = Path(args.out) if args.out else cfg.path("clean_reference_file")

    import rclpy

    rclpy.init(args=None)
    try:
        node = make_grabber(image_topic=image_topic, pose_topic=pose_topic,
                            node_name="srf_clean_reference_capture")
        print(f"[clean-ref] grabbing {n_frames} frames from {image_topic} ...")
        frames, poses, ok = collect(node, n_frames=n_frames, n_poses=5,
                                    timeout_sec=args.timeout_sec)
    finally:
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        rclpy.shutdown()

    frames = frames[:n_frames]
    if len(frames) < max(5, n_frames // 2):
        print(f"[clean-ref] only {len(frames)}/{n_frames} frames from {image_topic}",
              file=sys.stderr)
        return 1

    arr = np.stack(frames).astype(np.uint8)
    reference = build_clean_reference(arr)
    per_frame_std = float(np.stack([g for g in arr.astype(np.float32).mean(axis=-1)]).std(axis=0).mean())

    payload = dict(
        reference=reference.astype(np.float32),
        n_frames=int(arr.shape[0]),
        per_frame_gray_std=per_frame_std,
        lighting=str(lighting),
        stamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        image_topic=str(image_topic),
        pose_topic=str(pose_topic),
        image_shape=np.asarray(arr.shape[1:], dtype=np.int64),
    )
    if poses.shape[0] >= 1:
        st = pose_stats(poses)
        payload["pose_mean"] = np.asarray(st["mean"], dtype=np.float64)
        payload["pose_std"] = np.asarray(st["std"], dtype=np.float64)
    if not args.no_store_frames:
        payload["frames_rgb"] = arr

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)

    print(f"[clean-ref] {arr.shape[0]} frames  shape={arr.shape[1:]}  "
          f"mean gray={reference.mean():.2f}  temporal std={per_frame_std:.3f}")
    print(f"[clean-ref] lighting: {lighting}")
    if poses.shape[0] >= 1:
        print(f"[clean-ref] pose mean = {np.round(payload['pose_mean'], 4).tolist()}")
    print(f"[clean-ref] saved -> {out_path}")
    print("[clean-ref] next: [1] homography_collect")
    return 0


def load_clean_reference(path) -> tuple:
    """Read the reference back. Returns (reference_gray (H,W), meta dict)."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(
            f"clean reference not found: {p}\nRun step [0b] "
            "(ros2 run stain_relative_frame clean_reference_capture) first."
        )
    with np.load(p, allow_pickle=False) as z:
        ref = np.asarray(z["reference"], dtype=np.float32)
        meta = {
            k: (z[k].item() if z[k].shape == () else z[k].tolist())
            for k in z.files if k not in ("reference", "frames_rgb")
        }
    return ref, meta


if __name__ == "__main__":
    sys.exit(main())
