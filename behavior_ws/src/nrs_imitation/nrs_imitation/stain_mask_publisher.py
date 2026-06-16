#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publish a live stain mask from an RGB camera stream."""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from nrs_imitation.hdf5_recorder_base import image_to_rgb_numpy
from nrs_imitation.hdf5_recorder_single_cam_stain_mask import (
    generate_stain_mask_from_rgb,
    make_stain_mask_overlay,
)


def _reliability_from_str(value: str) -> ReliabilityPolicy:
    s = str(value or "best_effort").strip().lower()
    if s in ("reliable", "rel"):
        return ReliabilityPolicy.RELIABLE
    return ReliabilityPolicy.BEST_EFFORT


def _image_qos(depth: int, reliability: ReliabilityPolicy) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=int(depth),
        reliability=reliability,
    )


def _mono8_to_image_msg(mask: np.ndarray, stamp=None, frame_id: str = "") -> Image:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise RuntimeError(f"mask must be 2D, got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    msg = Image()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(arr.shape[0])
    msg.width = int(arr.shape[1])
    msg.encoding = "mono8"
    msg.is_bigendian = 0
    msg.step = int(arr.shape[1])
    msg.data = arr.tobytes()
    return msg


def _rgb_to_image_msg(rgb: np.ndarray, stamp=None, frame_id: str = "") -> Image:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise RuntimeError(f"RGB image must be (H,W,3), got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    msg = Image()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(arr.shape[0])
    msg.width = int(arr.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = int(arr.shape[1] * 3)
    msg.data = arr.tobytes()
    return msg


class StainMaskPublisher(Node):
    def __init__(self):
        super().__init__("stain_mask_publisher")

        self.declare_parameter("image_topic", "/realsense/vr/color/image_raw")
        self.declare_parameter("mask_topic", "/inference_single_cam/stain_mask")
        self.declare_parameter("overlay_topic", "/inference_single_cam/stain_mask_overlay")
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("image_qos", "best_effort")
        self.declare_parameter("stain_dark_thresh", 80)
        self.declare_parameter("reflection_v_thresh", 235)
        self.declare_parameter("reflection_s_thresh", 60)
        self.declare_parameter("stain_min_area", 20)
        self.declare_parameter("stain_morph_kernel", 3)
        self.declare_parameter("overlay_alpha", 0.45)
        self.declare_parameter("log_every_n", 60)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.mask_topic = str(self.get_parameter("mask_topic").value)
        self.overlay_topic = str(self.get_parameter("overlay_topic").value)
        self.publish_overlay = bool(self.get_parameter("publish_overlay").value)
        self.stain_dark_thresh = int(self.get_parameter("stain_dark_thresh").value)
        self.reflection_v_thresh = int(self.get_parameter("reflection_v_thresh").value)
        self.reflection_s_thresh = int(self.get_parameter("reflection_s_thresh").value)
        self.stain_min_area = int(self.get_parameter("stain_min_area").value)
        self.stain_morph_kernel = int(self.get_parameter("stain_morph_kernel").value)
        self.overlay_alpha = float(self.get_parameter("overlay_alpha").value)
        self.log_every_n = max(1, int(self.get_parameter("log_every_n").value))

        img_qos = _image_qos(
            depth=1,
            reliability=_reliability_from_str(str(self.get_parameter("image_qos").value)),
        )

        self.pub_mask = self.create_publisher(Image, self.mask_topic, 1)
        self.pub_overlay = None
        if self.publish_overlay:
            self.pub_overlay = self.create_publisher(Image, self.overlay_topic, 1)

        self.create_subscription(Image, self.image_topic, self._on_image, img_qos)
        self._count = 0

        self.get_logger().info(
            "[STAIN-MASK-PUB] "
            f"image_topic={self.image_topic}, mask_topic={self.mask_topic}, "
            f"overlay_topic={self.overlay_topic if self.publish_overlay else '(disabled)'}, "
            f"dark_thresh={self.stain_dark_thresh}, min_area={self.stain_min_area}, "
            f"morph_kernel={self.stain_morph_kernel}"
        )

    def _on_image(self, msg: Image):
        try:
            rgb = image_to_rgb_numpy(msg)
            if rgb is None:
                raise RuntimeError(f"unsupported image encoding={msg.encoding}")

            mask = generate_stain_mask_from_rgb(
                rgb,
                stain_dark_thresh=self.stain_dark_thresh,
                reflection_v_thresh=self.reflection_v_thresh,
                reflection_s_thresh=self.reflection_s_thresh,
                stain_min_area=self.stain_min_area,
                stain_morph_kernel=self.stain_morph_kernel,
            )

            frame_id = msg.header.frame_id or "stain_mask"
            self.pub_mask.publish(_mono8_to_image_msg(mask, stamp=msg.header.stamp, frame_id=frame_id))

            if self.pub_overlay is not None:
                overlay = make_stain_mask_overlay(rgb, mask, alpha=self.overlay_alpha)
                self.pub_overlay.publish(_rgb_to_image_msg(overlay, stamp=msg.header.stamp, frame_id=frame_id))

            self._count += 1
            if self._count <= 3 or (self._count % self.log_every_n == 0):
                coverage = 100.0 * float(np.count_nonzero(mask)) / max(1, int(mask.size))
                self.get_logger().info(
                    f"[STAIN-MASK-PUB] #{self._count} mask_shape={tuple(mask.shape)} "
                    f"coverage={coverage:.2f}%"
                )
        except Exception as e:
            self.get_logger().warn(f"[STAIN-MASK-PUB] failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = StainMaskPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
