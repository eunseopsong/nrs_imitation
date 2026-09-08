#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steps [2]+[5] online: resolve `stain_origin` ONCE at the home pose and latch it.

Lifecycle, deliberately rigid:

  arm at home pose
    -> node averages `--frames` camera frames
    -> diff-mask against the clean reference from [0b]
    -> centroid -> H -> stain_origin
    -> FROZEN, latched on /stain_relative_frame/stain_origin (transient local)
    -> the image subscription is DESTROYED

After that the node holds no camera subscription at all, so there is no code
path that could recompute the origin as the episode runs -- the per-step
recomputation this design forbids is structurally impossible, not merely
avoided by convention. Starting a new episode requires an explicit
`~/new_episode` service call, which is logged.

The inference node consumes the latched value through
`relative_frame.RelativeFrameAdapter` -- the same functions the dataset
converter uses (see audit_inference_path.py).
"""

from __future__ import annotations

import sys
import time
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64MultiArray
from std_srvs.srv import Trigger

from .clean_reference_capture import load_clean_reference
from .config import load_config
from .homography import load_homography
from .relative_frame import StainOrigin
from .ros_utils import img_msg_to_numpy
from .stain_detect import detect_stain_origin


def _latched_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class StainOriginNode(Node):
    def __init__(self):
        super().__init__("stain_origin_node")

        self.declare_parameter("config", "")
        self.declare_parameter("image_topic", "")
        self.declare_parameter("frames", 10)
        self.declare_parameter("timeout_sec", 20.0)
        self.declare_parameter("origin_topic", "/stain_relative_frame/stain_origin")
        self.declare_parameter("ready_topic", "/stain_relative_frame/origin_ready")
        self.declare_parameter("homography", "")
        self.declare_parameter("clean_reference", "")

        cfg_path = str(self.get_parameter("config").value) or None
        self.cfg = load_config(cfg_path)
        self.image_topic = str(self.get_parameter("image_topic").value) or self.cfg.image_topic
        self.n_frames = max(1, int(self.get_parameter("frames").value))
        self.timeout_sec = float(self.get_parameter("timeout_sec").value)

        h_path = str(self.get_parameter("homography").value) or self.cfg.path("homography_file")
        r_path = str(self.get_parameter("clean_reference").value) or self.cfg.path("clean_reference_file")
        self.H, h_meta = load_homography(h_path, require_pass=True)
        self.ref, ref_meta = load_clean_reference(r_path)

        self.pub_origin = self.create_publisher(
            Float64MultiArray, str(self.get_parameter("origin_topic").value), _latched_qos()
        )
        self.pub_ready = self.create_publisher(
            Bool, str(self.get_parameter("ready_topic").value), _latched_qos()
        )
        self.create_service(Trigger, "~/new_episode", self._on_new_episode)
        self.create_service(Trigger, "~/get_stain_origin", self._on_get)

        self._origin: Optional[StainOrigin] = None
        self._frames: List[np.ndarray] = []
        self._sub = None
        self._deadline = 0.0

        self.get_logger().info(
            f"[STAIN-ORIGIN] H held-out max={h_meta.get('max_heldout_error_mm')}mm, "
            f"clean reference {tuple(self.ref.shape)} lighting={ref_meta.get('lighting')!r}"
        )
        self._arm("startup")

    # ------------------------------------------------------------------
    def _arm(self, reason: str) -> None:
        if self._sub is not None:
            self.get_logger().warn("[STAIN-ORIGIN] already arming; ignoring")
            return
        self._origin = None
        self._frames = []
        self._deadline = time.monotonic() + self.timeout_sec
        self.pub_ready.publish(Bool(data=False))
        self._sub = self.create_subscription(
            Image, self.image_topic, self._on_image, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"[STAIN-ORIGIN] armed ({reason}): put the arm at the HOME pose. "
            f"Collecting {self.n_frames} frames from {self.image_topic} ..."
        )

    def _disarm(self) -> None:
        """Drop the camera subscription so no later frame can move the origin."""
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None

    def _on_image(self, msg: Image) -> None:
        if self._origin is not None:
            return  # frozen; cannot happen, the sub is gone by then
        try:
            self._frames.append(img_msg_to_numpy(msg))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"[STAIN-ORIGIN] decode failed: {exc}")
            return

        if len(self._frames) < self.n_frames:
            if time.monotonic() > self._deadline:
                self.get_logger().error(
                    f"[STAIN-ORIGIN] timed out with {len(self._frames)}/{self.n_frames} frames"
                )
                self._disarm()
            return
        self._resolve()

    def _resolve(self) -> None:
        stack = np.stack(self._frames[: self.n_frames]).astype(np.float32)
        home_frame = np.clip(stack.mean(axis=0), 0, 255).astype(np.uint8)
        det = detect_stain_origin(
            home_frame, self.ref, self.H,
            diff_thresh=self.cfg.stain_diff_thresh,
            blur_sigma=self.cfg.stain_blur_sigma,
            min_area=self.cfg.stain_min_area,
            morph_kernel=self.cfg.stain_morph_kernel,
            max_components=self.cfg.stain_max_components,
        )
        if not det.ok:
            self.get_logger().error(
                f"[STAIN-ORIGIN] detection FAILED: {det.status} "
                f"(components={det.num_components}). Not publishing an origin. "
                f"Call ~/new_episode to retry."
            )
            self._frames = []
            self._deadline = time.monotonic() + self.timeout_sec
            return

        self._origin = StainOrigin(
            det.origin_mm, source=f"home_pose_mean_of_{self.n_frames}_frames", frame_index=0
        ).freeze()
        self._disarm()  # ★ no camera subscription survives -- origin cannot drift

        msg = Float64MultiArray()
        msg.data = [float(det.origin_mm[0]), float(det.origin_mm[1])]
        self.pub_origin.publish(msg)
        self.pub_ready.publish(Bool(data=True))
        self.get_logger().info(
            f"[STAIN-ORIGIN] FROZEN stain_origin = "
            f"({det.origin_mm[0]:.3f}, {det.origin_mm[1]:.3f}) mm  "
            f"centroid_px=({det.centroid_px[0]:.1f}, {det.centroid_px[1]:.1f})  "
            f"components={det.num_components} area={det.area_px}px  "
            f"| latched, camera subscription released"
        )

    # ------------------------------------------------------------------
    def _on_new_episode(self, request, response):
        self.get_logger().warn(
            "[STAIN-ORIGIN] ~/new_episode: discarding the frozen origin and re-arming. "
            "This is only valid BETWEEN episodes -- calling it mid-episode changes the "
            "frame the policy was conditioned on."
        )
        self._origin = None
        self._disarm()
        self._arm("new_episode service")
        response.success = True
        response.message = "re-armed; return the arm to the home pose"
        return response

    def _on_get(self, request, response):
        if self._origin is None:
            response.success = False
            response.message = "stain_origin not resolved yet"
        else:
            response.success = True
            response.message = f"{self._origin.xy[0]:.6f} {self._origin.xy[1]:.6f}"
        return response


def main(args=None) -> int:
    rclpy.init(args=args)
    try:
        node = StainOriginNode()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[STAIN-ORIGIN] cannot start: {exc}", file=sys.stderr)
        rclpy.shutdown()
        return 1
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
