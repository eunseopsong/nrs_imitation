#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capture direction-classifier training frames FROM THE NEUTRAL VIEWING POSE.

The stock classifier (checkpoints/direction_classifier/classifier.pt) was
trained on frames sampled along each direction's demo episodes, i.e. each
class only ever seen from its own trajectory's viewpoints. At autodir
inference every classification frame instead comes from ONE fixed neutral
pose, so the classifier keys on scene/background (identical for both classes
there) and pins its answer -- observed: a 0deg stain kept routing to 90deg.

Fix: retrain on frames captured HERE -- same neutral pose, same background,
only the stain differs -- so the head has to read the stain itself.

Workflow (repeat a few takes per label, nudging/redrawing the stain between
takes so the val split is meaningful):

  # draw a 0deg stain, then:
  python3 scripts/direction_classifier/capture_neutral_frames.py --label 0
  # redraw / shift it, run again (a few times)
  ...
  # wipe, draw a 90deg stain:
  python3 scripts/direction_classifier/capture_neutral_frames.py --label 90
  ...

Each run PTPs to the neutral pose (unless --no-ptp) and appends ONE
episode_<N>.hdf5 in the exact layout train_direction_classifier.py reads
(observations/images/cam0 + attrs['stain_direction_deg']). Then:

  python3 scripts/direction_classifier/train_direction_classifier.py \
    --dataset_dir datasets/direction_classifier/neutral_<date> \
    --frames_per_episode 20
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch  # noqa: F401  (import before rclpy -- see pick_start_and_launch)
import h5py
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from pick_start_and_launch import _img_msg_to_numpy, load_demo_start_pose_mean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_CKPT = str(
    PROJECT_ROOT / "checkpoints" / "flow" / "polishing" / "single_cam" / "20260826_1242"
)


class _Grabber(Node):
    def __init__(self, image_topic: str):
        super().__init__("capture_neutral_frames_grabber")
        self.frames = []
        self.create_subscription(Image, image_topic, self._cb, qos_profile_sensor_data)

    def _cb(self, msg: Image):
        try:
            self.frames.append(_img_msg_to_numpy(msg))
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"decode failed: {e}")


def _ptp_to(pose6: np.ndarray, velocity_mm_s: float,
            service_name: str = "/singleArm_cmd/single_arm_command") -> None:
    from y2_rob_motion_interfaces.srv import SingleArmCommand

    node = rclpy.create_node("capture_neutral_frames_ptp")
    cli = node.create_client(SingleArmCommand, service_name)
    try:
        if not cli.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f"{service_name} unavailable")
        req = SingleArmCommand.Request()
        req.command_mode = "PTP"
        req.target_pose = [
            float(pose6[0]), float(pose6[1]), float(pose6[2]),
            float(np.degrees(pose6[3])), float(np.degrees(pose6[4])), float(np.degrees(pose6[5])),
        ]
        req.target_velocity = float(velocity_mm_s)
        print(f"[capture] PTP -> neutral {np.round(pose6, 3).tolist()} @ {velocity_mm_s:.0f}mm/s")
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=60.0)
        resp = fut.result()
        print(f"[capture] PTP done: {getattr(resp, 'message', resp)}")
    finally:
        node.destroy_node()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", type=int, required=True, choices=[0, 90],
                    help="stain_direction_deg for this capture")
    ap.add_argument("--out_dir", type=str,
                    default=str(PROJECT_ROOT / "datasets" / "direction_classifier"
                               / f"neutral_{time.strftime('%Y%m%d')}"))
    ap.add_argument("--num_frames", type=int, default=40)
    ap.add_argument("--grab_seconds", type=float, default=4.0)
    ap.add_argument("--image_topic", type=str, default="/realsense/vr/color/image_raw")
    ap.add_argument("--base_ckpt_dir", type=str, default=DEFAULT_BASE_CKPT)
    ap.add_argument("--ptp_velocity_mm_s", type=float, default=20.0)
    ap.add_argument("--no-ptp", dest="ptp", action="store_false",
                    help="skip the move (arm already at the neutral pose)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init(args=None)
    try:
        if args.ptp:
            pose6 = load_demo_start_pose_mean(args.base_ckpt_dir, prefer_arithmetic=True)
            _ptp_to(pose6, args.ptp_velocity_mm_s)
            time.sleep(1.5)

        node = _Grabber(args.image_topic)
        deadline = time.monotonic() + args.grab_seconds
        while len(node.frames) < args.num_frames and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        frames = list(node.frames)[: args.num_frames]
        node.destroy_node()
    finally:
        rclpy.shutdown()

    if len(frames) < 5:
        raise RuntimeError(f"only got {len(frames)} frames from {args.image_topic}")

    arr = np.stack(frames).astype(np.uint8)  # (N, H, W, 3)
    existing = glob.glob(str(out_dir / "episode_*.hdf5"))
    idx = 1 + max([int(Path(p).stem.split("_")[1]) for p in existing], default=-1)
    path = out_dir / f"episode_{idx}.hdf5"
    with h5py.File(path, "w") as f:
        g = f.create_group("observations/images")
        g.create_dataset("cam0", data=arr, compression="gzip", compression_opts=4)
        f.attrs["stain_direction_deg"] = int(args.label)
        f.attrs["capture_pose"] = "neutral_arithmetic_mean"
        f.attrs["source"] = "capture_neutral_frames.py"

    counts = {}
    for p in glob.glob(str(out_dir / "episode_*.hdf5")):
        with h5py.File(p, "r") as f:
            counts[int(f.attrs["stain_direction_deg"])] = counts.get(
                int(f.attrs["stain_direction_deg"]), 0) + 1
    print(f"[capture] wrote {path}  ({arr.shape[0]} frames, label={args.label}deg)")
    print(f"[capture] {out_dir} now has episodes per label: {counts}")


if __name__ == "__main__":
    main()
