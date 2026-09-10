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
from pathlib import Path
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
from .homography import load_homography, per_frame_homography
from .relative_frame import StainOrigin
from .ros_utils import img_msg_to_numpy
from .stain_detect import dark_cloud_origin, detect_stain_origin


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
        # auto -> "dark" when homography.json is method=depth_extrinsic (the
        # reference-free per-frame path stain_origin_offline uses), else "diff".
        self.declare_parameter("method", "auto")   # auto | diff | dark
        self.declare_parameter("pose_topic", "/ur10skku/currentP")  # dark: per-frame homography
        # dark: step [2] origin report (stain_origin_*.json) whose detect_params
        # were used to build THIS checkpoint's training origins. Pass it so the
        # online detector uses the exact same plate_roi / tool_box / dark_thresh
        # -- otherwise the online origin can land a few mm off the trained frame.
        # The path is stored in the checkpoint's dataset_stats.pkl["stain_origin_report"].
        self.declare_parameter("detect_params", "")

        cfg_path = str(self.get_parameter("config").value) or None
        self.cfg = load_config(cfg_path)
        self.image_topic = str(self.get_parameter("image_topic").value) or self.cfg.image_topic
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.n_frames = max(1, int(self.get_parameter("frames").value))
        self.timeout_sec = float(self.get_parameter("timeout_sec").value)

        h_path = str(self.get_parameter("homography").value) or self.cfg.path("homography_file")
        r_path = str(self.get_parameter("clean_reference").value) or self.cfg.path("clean_reference_file")
        self.H, h_meta = load_homography(h_path, require_pass=True)
        self._h_meta_inner = h_meta.get("meta", {}) or {}

        self.method = str(self.get_parameter("method").value).strip().lower()
        if self.method == "auto":
            self.method = "dark" if self._h_meta_inner.get("method") == "depth_extrinsic" else "diff"
        if self.method not in ("diff", "dark"):
            raise RuntimeError(f"method must be auto|diff|dark, got {self.method!r}")

        self.ref = None
        ref_meta = {}
        if self.method == "diff":
            self.ref, ref_meta = load_clean_reference(r_path)
        else:
            if self._h_meta_inner.get("method") != "depth_extrinsic":
                raise RuntimeError(
                    "method=dark needs a homography.json written by "
                    "homography_depth_calibrate (method=depth_extrinsic)"
                )
            self._plate_roi = tuple(int(x) for x in self.cfg.stain_plate_roi)
            self._tool_box = tuple(int(x) for x in self.cfg.stain_tool_box)
            self._dark_thresh = int(self.cfg.stain_dark_thresh)
            self._dark_min_area = int(self.cfg.stain_dark_min_area)
            dp_path = str(self.get_parameter("detect_params").value).strip()
            if dp_path:
                import json as _json
                rep = _json.loads(Path(dp_path).expanduser().read_text())
                dp = rep.get("detect_params", rep)
                if dp.get("plate_roi"):
                    self._plate_roi = tuple(int(x) for x in dp["plate_roi"])
                if dp.get("tool_box"):
                    self._tool_box = tuple(int(x) for x in dp["tool_box"])
                if dp.get("dark_thresh") is not None:
                    self._dark_thresh = int(dp["dark_thresh"])
                self.get_logger().info(
                    f"[STAIN-ORIGIN] dark detect_params from {dp_path}: "
                    f"plate_roi={self._plate_roi} tool_box={self._tool_box} dark_thresh={self._dark_thresh}"
                )
        self._latest_pose6: Optional[np.ndarray] = None
        self._frame_poses: List[np.ndarray] = []
        self._pose_sub = None

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

        if self.method == "diff":
            self.get_logger().info(
                f"[STAIN-ORIGIN] method=diff | H held-out max={h_meta.get('max_heldout_error_mm')}mm, "
                f"clean reference {tuple(self.ref.shape)} lighting={ref_meta.get('lighting')!r}"
            )
        else:
            self.get_logger().info(
                f"[STAIN-ORIGIN] method=dark (reference-free, per-frame homography from "
                f"{self.pose_topic}) | dark_thresh={self._dark_thresh} "
                f"plate_roi={self._plate_roi} tool_box={self._tool_box} Z0={self._h_meta_inner.get('z0_mm')}mm"
            )
        self._arm("startup")

    # ------------------------------------------------------------------
    def _arm(self, reason: str) -> None:
        if self._sub is not None:
            self.get_logger().warn("[STAIN-ORIGIN] already arming; ignoring")
            return
        self._origin = None
        self._frames = []
        self._frame_poses = []
        self._deadline = time.monotonic() + self.timeout_sec
        self.pub_ready.publish(Bool(data=False))
        if self.method == "dark":
            self._pose_sub = self.create_subscription(
                Float64MultiArray, self.pose_topic, self._on_pose, 10
            )
        self._sub = self.create_subscription(
            Image, self.image_topic, self._on_image, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"[STAIN-ORIGIN] armed ({reason}): put the arm at the HOME pose. "
            f"Collecting {self.n_frames} frames from {self.image_topic}"
            + (f" (+ pose from {self.pose_topic})" if self.method == "dark" else "")
            + " ..."
        )

    def _disarm(self) -> None:
        """Drop the camera subscription so no later frame can move the origin."""
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        if self._pose_sub is not None:
            self.destroy_subscription(self._pose_sub)
            self._pose_sub = None

    def _on_pose(self, msg: Float64MultiArray) -> None:
        arr = np.asarray(msg.data, dtype=np.float64).reshape(-1)
        if arr.size >= 6:
            self._latest_pose6 = arr[:6].copy()

    def _on_image(self, msg: Image) -> None:
        if self._origin is not None:
            return  # frozen; cannot happen, the sub is gone by then
        if self.method == "dark" and self._latest_pose6 is None:
            if time.monotonic() > self._deadline:
                self.get_logger().error(
                    f"[STAIN-ORIGIN] timed out waiting for a pose on {self.pose_topic}"
                )
                self._disarm()
            return
        try:
            self._frames.append(img_msg_to_numpy(msg))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"[STAIN-ORIGIN] decode failed: {exc}")
            return
        if self.method == "dark":
            self._frame_poses.append(self._latest_pose6.copy())

        if len(self._frames) < self.n_frames:
            if time.monotonic() > self._deadline:
                self.get_logger().error(
                    f"[STAIN-ORIGIN] timed out with {len(self._frames)}/{self.n_frames} frames"
                )
                self._disarm()
            return
        self._resolve()

    def _resolve(self) -> None:
        if self.method == "dark":
            self._resolve_dark()
        else:
            self._resolve_diff()

    def _resolve_diff(self) -> None:
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
            self._frame_poses = []
            self._deadline = time.monotonic() + self.timeout_sec
            return
        self._publish_origin(
            det.origin_mm,
            f"home_pose_mean_of_{self.n_frames}_frames",
            f"centroid_px=({det.centroid_px[0]:.1f}, {det.centroid_px[1]:.1f}) "
            f"components={det.num_components} area={det.area_px}px",
        )

    def _resolve_dark(self) -> None:
        frames = np.stack(self._frames[: self.n_frames])
        poses = self._frame_poses[: self.n_frames]
        Hs = np.stack([
            per_frame_homography(self._h_meta_inner, poses[i]) for i in range(len(frames))
        ])
        rep = dark_cloud_origin(
            frames, Hs, self._plate_roi, self._tool_box,
            episode="online", std_tol_mm=6.0,
            dark_thresh=self._dark_thresh, min_area=self._dark_min_area,
        )
        if rep.origin_mm is None:
            self.get_logger().error(
                f"[STAIN-ORIGIN] dark detection FAILED (ok {rep.n_ok}/{rep.n_frames} frames, "
                f"failures={rep.failures[:3]}...). Not publishing. Call ~/new_episode to retry."
            )
            self._frames = []
            self._frame_poses = []
            self._deadline = time.monotonic() + self.timeout_sec
            return
        spread = "n/a" if rep.std_norm_mm != rep.std_norm_mm else f"{rep.std_norm_mm:.2f}mm"
        ang = rep.angle_rad
        note = f"per-frame spread={spread} ok={rep.n_ok}/{rep.n_frames}" + (
            f" axis={np.degrees(ang):.1f}deg" if ang is not None else "")
        if rep.unstable:
            self.get_logger().warn(f"[STAIN-ORIGIN] dark origin flagged UNSTABLE ({note})")
        self._publish_origin(
            np.asarray(rep.origin_mm, float),
            f"home_pose_dark_cloud_of_{rep.n_ok}_frames", note, angle_rad=ang,
        )

    def _publish_origin(self, origin_mm: np.ndarray, source: str, note: str,
                        angle_rad: Optional[float] = None) -> None:
        self._origin = StainOrigin(origin_mm, source=source, frame_index=0).freeze()
        self._disarm()  # ★ no subscription survives -- origin cannot drift

        msg = Float64MultiArray()
        # [x, y] always; a 3rd element carries the strip's principal-axis angle
        # (base-frame rad, [0,pi)) when the detector produced one. Consumers
        # that only read [:2] are unaffected.
        msg.data = [float(origin_mm[0]), float(origin_mm[1])]
        if angle_rad is not None:
            msg.data.append(float(angle_rad))
        self.pub_origin.publish(msg)
        self.pub_ready.publish(Bool(data=True))
        self.get_logger().info(
            f"[STAIN-ORIGIN] FROZEN stain_origin = ({origin_mm[0]:.3f}, {origin_mm[1]:.3f}) mm  "
            f"[{self.method}] {note} | latched, subscriptions released"
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
