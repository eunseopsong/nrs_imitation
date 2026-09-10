#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Snapshot the current robot pose (/ur10skku/currentP) and camera frames
(/realsense/vr/color/image_raw) -- no move, no policy. Used to record what
'home pose' actually is and what the wrist camera sees there, e.g. for
using home pose as the direction-classification pose.

  python3 scripts/direction_classifier/snapshot_pose_and_camera.py --label 90

Saves one episode_<N>.hdf5 under --out_dir:
  observations/images/cam0  (N, H, W, 3) uint8
  observations/position     (P, 6)  float64   -- raw /ur10skku/currentP samples
  attrs: stain_direction_deg, pose_mean (6,), pose_topic, image_topic, stamp
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch  # noqa: F401  (import before rclpy)
import h5py
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

from pick_start_and_launch import _img_msg_to_numpy

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _Snap(Node):
    def __init__(self, image_topic: str, pose_topic: str):
        super().__init__("snapshot_pose_and_camera")
        self.frames = []
        self.poses = []
        self.create_subscription(Image, image_topic, self._on_img, qos_profile_sensor_data)
        self.create_subscription(Float64MultiArray, pose_topic, self._on_pose, qos_profile_sensor_data)

    def _on_img(self, msg: Image):
        try:
            self.frames.append(_img_msg_to_numpy(msg))
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"decode failed: {e}")

    def _on_pose(self, msg: Float64MultiArray):
        arr = np.asarray(msg.data, dtype=np.float64)
        if arr.shape[0] >= 6:
            self.poses.append(arr[:6])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", type=int, default=None,
                    help="stain_direction_deg drawn right now (optional): 0 / 90 for "
                         "the polishing directions, -1 for a novel non-direction type")
    ap.add_argument("--image_topic", type=str, default="/realsense/vr/color/image_raw")
    ap.add_argument("--pose_topic", type=str, default="/ur10skku/currentP")
    ap.add_argument("--num_frames", type=int, default=40)
    ap.add_argument("--grab_seconds", type=float, default=5.0)
    ap.add_argument("--out_dir", type=str,
                    default=str(PROJECT_ROOT / "datasets" / "direction_classifier"
                               / f"home_{time.strftime('%Y%m%d')}"))
    args = ap.parse_args()

    rclpy.init(args=None)
    node = _Snap(args.image_topic, args.pose_topic)
    try:
        deadline = time.monotonic() + args.grab_seconds
        while time.monotonic() < deadline and len(node.frames) < args.num_frames:
            rclpy.spin_once(node, timeout_sec=0.1)
        frames = list(node.frames)[: args.num_frames]
        poses = np.asarray(node.poses, dtype=np.float64)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if len(frames) < 3:
        raise RuntimeError(f"only {len(frames)} camera frames from {args.image_topic}")
    if poses.shape[0] < 1:
        raise RuntimeError(f"no pose samples from {args.pose_topic}")

    pose_mean = poses.mean(axis=0)
    pose_std = poses.std(axis=0)
    print(f"[snapshot] {len(frames)} frames  |  {poses.shape[0]} pose samples")
    print(f"[snapshot] pose_topic  = {args.pose_topic}")
    print(f"[snapshot] image_topic = {args.image_topic}")
    print(f"[snapshot] pose mean [x y z wx wy wz] = {np.round(pose_mean, 4).tolist()}")
    print(f"[snapshot] pose std                   = {np.round(pose_std, 4).tolist()}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = glob.glob(str(out_dir / "episode_*.hdf5"))
    idx = 1 + max([int(Path(p).stem.split("_")[1]) for p in existing], default=-1)
    path = out_dir / f"episode_{idx}.hdf5"
    arr = np.stack([np.asarray(f) for f in frames]).astype(np.uint8)
    with h5py.File(path, "w") as f:
        f.create_group("observations/images").create_dataset(
            "cam0", data=arr, compression="gzip", compression_opts=4)
        f.create_dataset("observations/position", data=poses)
        if args.label is not None:
            f.attrs["stain_direction_deg"] = int(args.label)
        f.attrs["pose_mean"] = pose_mean
        f.attrs["pose_std"] = pose_std
        f.attrs["pose_topic"] = args.pose_topic
        f.attrs["image_topic"] = args.image_topic
        f.attrs["stamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        f.attrs["source"] = "snapshot_pose_and_camera.py"
    print(f"[snapshot] saved -> {path}"
          + (f"  (label={args.label}deg)" if args.label is not None else "  (no label)"))


if __name__ == "__main__":
    main()
