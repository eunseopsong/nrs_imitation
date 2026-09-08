#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin ROS 2 helpers shared by the stain_relative_frame nodes.

Kept deliberately small: image decode, a pose/frame grabber, and a PTP call.
Mirrors the conventions already used in scripts/direction_classifier so the
two paths cannot drift (pose topic is Float64MultiArray [x,y,z,rx,ry,rz] with
xyz in mm and the rotation triple in RADIANS; the PTP service wants that
triple in DEGREES).
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np


def img_msg_to_numpy(msg) -> np.ndarray:
    """sensor_msgs/Image -> (H, W, 3) uint8 RGB."""
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        arr = arr.reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)
    if msg.encoding == "mono8":
        return np.ascontiguousarray(arr.reshape(msg.height, msg.width))
    raise RuntimeError(f"unsupported image encoding: {msg.encoding}")


def make_grabber(image_topic: str = "", pose_topic: str = "", node_name: str = "srf_grabber"):
    """Build a Node that accumulates camera frames and/or pose samples."""
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float64MultiArray

    class _Grabber(Node):
        def __init__(self):
            super().__init__(node_name)
            self.frames: List[np.ndarray] = []
            self.poses: List[np.ndarray] = []
            self.collect = True
            if image_topic:
                self.create_subscription(Image, image_topic, self._on_img, qos_profile_sensor_data)
            if pose_topic:
                self.create_subscription(
                    Float64MultiArray, pose_topic, self._on_pose, qos_profile_sensor_data
                )

        def _on_img(self, msg):
            if not self.collect:
                return
            try:
                self.frames.append(img_msg_to_numpy(msg))
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"image decode failed: {exc}")

        def _on_pose(self, msg):
            if not self.collect:
                return
            arr = np.asarray(msg.data, dtype=np.float64)
            if arr.shape[0] >= 6:
                self.poses.append(arr[:6])

        def clear(self):
            self.frames.clear()
            self.poses.clear()

    return _Grabber()


def spin_until(node, predicate, timeout_sec: float, period: float = 0.05) -> bool:
    """Spin `node` until `predicate()` or the timeout. True if satisfied."""
    import rclpy

    deadline = time.monotonic() + float(timeout_sec)
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=period)
        if predicate():
            return True
    return predicate()


def collect(node, n_frames: int = 0, n_poses: int = 0, timeout_sec: float = 10.0):
    """Gather at least n_frames images and n_poses pose samples."""
    def done():
        return (len(node.frames) >= n_frames) and (len(node.poses) >= n_poses)

    ok = spin_until(node, done, timeout_sec)
    return list(node.frames), np.asarray(node.poses, dtype=np.float64), ok


def ptp_to(pose6, velocity_mm_s: float = 20.0,
           service_name: str = "/singleArm_cmd/single_arm_command",
           timeout_sec: float = 60.0, logger=None) -> str:
    """PTP the arm to a 6-DoF pose through SingleArmCommand.

    `pose6` is [x, y, z, rx, ry, rz] with xyz in mm and the rotation triple in
    RADIANS -- this converts to degrees for the service, which is the one
    field in this stack that is not radians.
    """
    import rclpy
    from y2_rob_motion_interfaces.srv import SingleArmCommand

    pose6 = np.asarray(pose6, dtype=np.float64).reshape(-1)
    if pose6.shape[0] < 6:
        raise ValueError(f"pose6 must have 6 elements, got {pose6.shape}")

    node = rclpy.create_node("srf_ptp_client")
    try:
        cli = node.create_client(SingleArmCommand, service_name)
        if not cli.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f"{service_name} unavailable")
        req = SingleArmCommand.Request()
        req.command_mode = "PTP"
        req.target_pose = [
            float(pose6[0]), float(pose6[1]), float(pose6[2]),
            float(np.degrees(pose6[3])), float(np.degrees(pose6[4])), float(np.degrees(pose6[5])),
        ]
        req.target_velocity = float(velocity_mm_s)
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout_sec)
        resp = fut.result()
        if resp is None:
            raise RuntimeError("PTP call timed out (no service response)")
        msg = str(getattr(resp, "message", resp))
        if logger is not None:
            logger(f"PTP done: {msg}")
        return msg
    finally:
        node.destroy_node()


def pose_stats(poses: np.ndarray) -> dict:
    """Mean/std of a (N,6) pose sample block, rotations reported in degrees."""
    p = np.asarray(poses, dtype=np.float64).reshape(-1, 6)
    mean = p.mean(axis=0)
    std = p.std(axis=0)
    return {
        "n": int(p.shape[0]),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "mean_deg": np.concatenate([mean[:3], np.degrees(mean[3:])]).tolist(),
        "std_deg": np.concatenate([std[:3], np.degrees(std[3:])]).tolist(),
    }
