#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step [1a]: collect the pixel<->robot correspondences.

Two things are recorded, in this order:

  1. ONE home-pose image, with the calibration points visible. Every pixel
     coordinate is later read off THIS image (step [1b]), so it is saved
     alongside the robot points and never re-grabbed.
  2. For each of the 8 points: the operator touches the TCP to the point and
     presses ENTER; the node averages the pose topic over `--samples` samples
     and stores (x, y) in mm.

  ros2 run stain_relative_frame homography_collect -- --num_points 8

Output: <artifact_dir>/homography_points.npz
  home_image  (H, W, 3) uint8
  robot_pts   (N, 2) float64  mm
  robot_pose6 (N, 6) float64  full pose per touch, for auditing
  pose_std    (N, 6) float64  spread during each touch

Then: homography_pick_pixels -> homography_fit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import load_config
from .ros_utils import collect, make_grabber


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="[1a] collect homography correspondences")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--num_points", type=int, default=None)
    ap.add_argument("--samples", type=int, default=30, help="pose samples averaged per touch")
    ap.add_argument("--image_frames", type=int, default=10, help="frames averaged for the home image")
    ap.add_argument("--image_topic", type=str, default=None)
    ap.add_argument("--pose_topic", type=str, default=None)
    ap.add_argument("--touch_std_warn_mm", type=float, default=0.5,
                    help="warn when the TCP is not still during a touch")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    n_points = int(args.num_points or cfg.homography_num_points)
    image_topic = args.image_topic or cfg.image_topic
    pose_topic = args.pose_topic or cfg.pose_topic
    out_path = Path(args.out) if args.out else cfg.artifacts / "homography_points.npz"

    if n_points < 5:
        print(f"[homog] need >=5 points to fit 4+ and hold out 1, got {n_points}",
              file=sys.stderr)
        return 1

    print(f"[homog] {n_points} points, {args.samples} pose samples per touch")
    print(f"[homog] image={image_topic}  pose={pose_topic}")
    print("[homog] Pick points SPREAD ACROSS the whole work area -- a tight cluster "
          "fits well and extrapolates badly, which is what the held-out gate catches.")

    import rclpy

    rclpy.init(args=None)
    try:
        node = make_grabber(image_topic=image_topic, pose_topic=pose_topic,
                            node_name="srf_homography_collect")

        input("\n[homog] Put the arm at the HOME pose with all calibration points "
              "visible and unobstructed, then press ENTER to grab the home image... ")
        node.clear()
        frames, _, _ = collect(node, n_frames=args.image_frames, timeout_sec=15.0)
        if len(frames) < 1:
            print(f"[homog] no frames from {image_topic}", file=sys.stderr)
            return 1
        stack = np.stack(frames[: args.image_frames]).astype(np.float32)
        home_image = np.clip(stack.mean(axis=0), 0, 255).astype(np.uint8)
        print(f"[homog] home image: {home_image.shape} from {stack.shape[0]} frames")

        robot_pts, poses6, stds = [], [], []
        for i in range(n_points):
            input(f"\n[homog] point {i}/{n_points - 1}: touch the TCP to the point, "
                  f"hold still, press ENTER... ")
            node.clear()
            _, p, _ = collect(node, n_poses=args.samples,
                              timeout_sec=max(10.0, args.samples * 0.2))
            if p.shape[0] < 3:
                print(f"[homog] only {p.shape[0]} pose samples -- is {pose_topic} publishing?",
                      file=sys.stderr)
                return 1
            m, s = p.mean(axis=0), p.std(axis=0)
            robot_pts.append(m[:2])
            poses6.append(m)
            stds.append(s)
            flag = ""
            if max(s[0], s[1]) > args.touch_std_warn_mm:
                flag = f"  <-- MOVING during touch (xy std {max(s[0], s[1]):.3f}mm), redo advised"
            print(f"[homog]   x={m[0]:8.3f}  y={m[1]:8.3f}  z={m[2]:8.3f} mm   "
                  f"(n={p.shape[0]}, xy std={max(s[0], s[1]):.3f}mm){flag}")
    finally:
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        rclpy.shutdown()

    R = np.stack(robot_pts)
    span = R.max(axis=0) - R.min(axis=0)
    print(f"\n[homog] robot-point span: x={span[0]:.1f}mm  y={span[1]:.1f}mm")
    if min(span) < 30.0:
        print("[homog] WARNING: the points span <30mm on an axis. Expect the "
              "held-out gate to fail; spread them further.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        home_image=home_image,
        robot_pts=R,
        robot_pose6=np.stack(poses6),
        pose_std=np.stack(stds),
        image_topic=image_topic,
        pose_topic=pose_topic,
        stamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    print(f"[homog] saved -> {out_path}")
    print("[homog] next: ros2 run stain_relative_frame homography_pick_pixels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
