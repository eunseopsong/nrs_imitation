#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vr_demo_hdf5_recorder.py

Joystick-controlled multimodal merged-HDF5 recorder for nrs_imitation.

Existing single-cam recorder behavior is preserved, and this version adds:
  - global camera RGB stream as images/cam1
  - ArUco marker pose streams for id0 and id1

Default topics:
  cam0 tracker mode : /realsense/vr/color/image_raw
  cam0 robot mode   : /realsense/robot/color/image_raw
  cam1 global       : /realsense/global/color/image_raw
  marker id0        : /aruco/id_0/pose
  marker id1        : /aruco/id_1/pose

Marker convention:
  id0 = robot EE or VR tracker top marker
  id1 = workpiece/surface marker

Saved merged HDF5 layout:
  <repo>/datasets/ACT/YYYYMMDD_HHMM/merged_hdf5/
    vr_demo_merged_YYYYMMDD_HHMM.hdf5

  episodes/
    ep_0000/
      position             (T, 6) float32  [x_mm y_mm z_mm wx wy wz]
      ft                   (T, 3) float32  [fx fy fz]
      images/
        cam0               (T, H, W, 3) uint8 RGB
        cam1               (T, H, W, 3) uint8 RGB   optional/global
      marker/
        id0                (T, 7) float32 [x y z rx ry rz valid]
        id1                (T, 7) float32 [x y z rx ry rz valid]
        combined           (T,14) float32 [id0(7), id1(7)]
        id0_quat           (T, 8) float32 [x y z qx qy qz qw valid]
        id1_quat           (T, 8) float32 [x y z qx qy qz qw valid]
      gripper/
        present_position   (T,) int32
        present_current_mA (T,) float32
This file intentionally does NOT use cv_bridge. sensor_msgs/Image is converted
manually to numpy RGB to avoid cv_bridge / NumPy ABI issues.
"""

from __future__ import annotations

import os
import time
import atexit
import threading
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Set

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
ACT_ROOT_DEFAULT = os.path.join(REPO_ROOT, "datasets", "ACT")

import numpy as np
import h5py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Float64MultiArray, String, Int32, Float32
from geometry_msgs.msg import Wrench, PoseStamped
from sensor_msgs.msg import Image


# ============================================================
# QoS
# ============================================================

def make_qos(depth: int = 10, best_effort: bool = False) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=int(depth),
        reliability=ReliabilityPolicy.BEST_EFFORT if best_effort else ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


# ============================================================
# Image utilities
# ============================================================

def image_to_rgb_numpy(msg: Image) -> Optional[np.ndarray]:
    enc = (msg.encoding or "").lower()
    h, w, step = int(msg.height), int(msg.width), int(msg.step)

    if h <= 0 or w <= 0 or step <= 0:
        return None

    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if buf.size < h * step:
        return None

    row = buf[: h * step].reshape(h, step)

    if enc in ("rgb8", "bgr8"):
        need = w * 3
        if step < need:
            return None
        img = row[:, :need].reshape(h, w, 3)
        if enc == "bgr8":
            img = img[:, :, ::-1]
        return img.copy()

    if enc in ("rgba8", "bgra8"):
        need = w * 4
        if step < need:
            return None
        img4 = row[:, :need].reshape(h, w, 4)
        img = img4[:, :, :3]
        if enc == "bgra8":
            img = img[:, :, ::-1]
        return img.copy()

    if enc == "mono8":
        need = w
        if step < need:
            return None
        gray = row[:, :need].reshape(h, w)
        return np.repeat(gray[:, :, None], 3, axis=2).copy()

    return None


def pick_image_shape(frames: List[Optional[np.ndarray]]) -> Tuple[int, int]:
    for im in frames:
        if im is not None and im.ndim == 3 and im.shape[0] > 1 and im.shape[1] > 1 and im.shape[2] == 3:
            return int(im.shape[0]), int(im.shape[1])
    return 0, 0


def stack_images_repeat_last(frames: List[Optional[np.ndarray]], logger=None, tag: str = "IMAGE") -> Optional[np.ndarray]:
    H, W = pick_image_shape(frames)
    if H <= 0 or W <= 0:
        return None

    T = len(frames)
    out = np.zeros((T, H, W, 3), dtype=np.uint8)
    last = np.zeros((H, W, 3), dtype=np.uint8)
    valid_count = 0
    repeated_count = 0

    for i, im in enumerate(frames):
        if im is not None and im.ndim == 3 and im.shape == (H, W, 3):
            out[i] = im
            last = im
            valid_count += 1
        else:
            out[i] = last
            repeated_count += 1

    if logger is not None:
        logger.info(
            f"[{tag}] stacked: T={T}, H={H}, W={W}, "
            f"valid={valid_count}, repeated_or_invalid={repeated_count}"
        )

    return out


# ============================================================
# Marker utilities
# ============================================================

def quat_xyzw_to_rotvec(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.zeros(3, dtype=np.float32)
    q /= n

    qx, qy, qz, qw = [float(v) for v in q]
    qw = max(-1.0, min(1.0, qw))
    angle = 2.0 * np.arccos(qw)
    s = np.sqrt(max(0.0, 1.0 - qw * qw))

    if s < 1e-8:
        axis = np.asarray([qx, qy, qz], dtype=np.float64)
        an = float(np.linalg.norm(axis))
        if an < 1e-12:
            return np.zeros(3, dtype=np.float32)
        axis /= an
    else:
        axis = np.asarray([qx / s, qy / s, qz / s], dtype=np.float64)

    if angle > np.pi:
        angle -= 2.0 * np.pi

    return (axis * angle).astype(np.float32)


def pose_stamped_to_marker_vec(msg: PoseStamped) -> Tuple[np.ndarray, np.ndarray]:
    p = msg.pose.position
    q = msg.pose.orientation
    rv = quat_xyzw_to_rotvec(q.x, q.y, q.z, q.w)

    marker7 = np.asarray([p.x, p.y, p.z, rv[0], rv[1], rv[2], 1.0], dtype=np.float32)
    marker8 = np.asarray([p.x, p.y, p.z, q.x, q.y, q.z, q.w, 1.0], dtype=np.float32)
    return marker7, marker8


def stack_markers_repeat_last(frames: List[Optional[np.ndarray]], dim: int, logger=None, tag: str = "MARKER") -> np.ndarray:
    T = len(frames)
    out = np.zeros((T, dim), dtype=np.float32)
    last = np.zeros((dim,), dtype=np.float32)
    valid_count = 0
    repeated_count = 0

    for i, v in enumerate(frames):
        if v is not None:
            a = np.asarray(v, dtype=np.float32).reshape(-1)
            if a.size == dim:
                out[i] = a
                last = a
                valid_count += 1
                continue
        out[i] = last
        repeated_count += 1

    if logger is not None:
        logger.info(
            f"[{tag}] stacked: T={T}, dim={dim}, "
            f"valid={valid_count}, repeated_or_missing={repeated_count}"
        )

    return out


def stack_scalar_repeat_last(values: List[Optional[float]], dtype, logger=None, tag: str = "SCALAR") -> np.ndarray:
    T = len(values)
    out = np.zeros((T,), dtype=dtype)
    last = dtype.type(0) if hasattr(dtype, "type") else dtype(0)
    valid_count = 0
    repeated_count = 0

    for i, v in enumerate(values):
        if v is not None:
            out[i] = v
            last = out[i]
            valid_count += 1
        else:
            out[i] = last
            repeated_count += 1

    if logger is not None:
        logger.info(
            f"[{tag}] stacked: T={T}, "
            f"valid={valid_count}, repeated_or_missing={repeated_count}"
        )

    return out


# ============================================================
# Signal processing
# ============================================================

def ema_1d(y: np.ndarray, alpha: float) -> np.ndarray:
    if y.size == 0:
        return y.astype(np.float64).copy()
    z = y.astype(np.float64).copy()
    if alpha <= 0.0 or alpha >= 1.0:
        return z
    for i in range(1, y.size):
        z[i] = alpha * y[i] + (1.0 - alpha) * z[i - 1]
    return z


def ema_nd(X: np.ndarray, alpha: float) -> np.ndarray:
    if X.size == 0:
        return X.astype(np.float64).copy()
    Z = X.astype(np.float64).copy()
    if alpha <= 0.0 or alpha >= 1.0:
        return Z
    for i in range(1, Z.shape[0]):
        Z[i] = alpha * X[i] + (1.0 - alpha) * Z[i - 1]
    return Z


def process_force_keep_fz_with_ema_and_edge_zero(
    Fraw: np.ndarray,
    fz_ema_alpha: float,
    edge_zero_sec: float,
    sample_hz: float,
    zero_xy: bool = True,
    logger=None,
) -> np.ndarray:
    if Fraw.size == 0:
        return Fraw.astype(np.float64).copy()

    Fp = Fraw.astype(np.float64).copy()
    N = int(Fp.shape[0])

    if zero_xy:
        Fp[:, 0] = 0.0
        Fp[:, 1] = 0.0

    Fp[:, 2] = ema_1d(Fp[:, 2], alpha=float(fz_ema_alpha))

    edge_n = int(round(float(edge_zero_sec) * float(sample_hz)))
    edge_n = max(0, min(edge_n, N))
    if edge_n > 0:
        Fp[:edge_n, :] = 0.0
        Fp[max(0, N - edge_n):, :] = 0.0

    if logger is not None:
        raw_fz_abs_max = float(np.max(np.abs(Fraw[:, 2]))) if N > 0 else 0.0
        proc_fz_abs_max = float(np.max(np.abs(Fp[:, 2]))) if N > 0 else 0.0
        logger.info(
            f"[FORCE] zero_xy={zero_xy}, fz_ema_alpha={fz_ema_alpha}, "
            f"edge_zero_sec={edge_zero_sec}, edge_zero_samples={edge_n}, N={N}"
        )
        logger.info(
            f"[FORCE] raw |fz|max={raw_fz_abs_max:.3f} N, "
            f"processed |fz|max={proc_fz_abs_max:.3f} N"
        )

    return Fp


# ============================================================
# Main node
# ============================================================

class VRDemoHDF5Recorder(Node):
    def __init__(self):
        super().__init__("vr_demo_hdf5_recorder")

        # Save parameters
        self.declare_parameter("act_root_dir", ACT_ROOT_DEFAULT)
        self.declare_parameter("merged_subdir", "merged_hdf5")
        self.declare_parameter("file_prefix", "vr_demo_merged")
        self.declare_parameter("overwrite_file", False)
        self.declare_parameter("allow_overwrite_episode", False)
        self.declare_parameter("flush_each_episode", True)
        self.declare_parameter("num_episodes", 50)
        self.declare_parameter("min_samples", 10)

        # Topic parameters
        self.declare_parameter("recording_mode", "tracker")  # tracker | robot
        self.declare_parameter("tracker_pose_topic", "/calibrated_pose")
        self.declare_parameter("tracker_force_topic", "/ftsensor/measured_Cvalue")
        self.declare_parameter("tracker_image_topic", "/realsense/vr/color/image_raw")
        self.declare_parameter("robot_pose_topic", "/ur10skku/currentP")
        self.declare_parameter("robot_force_topic", "/ur10skku/currentF")
        self.declare_parameter("robot_image_topic", "/realsense/robot/color/image_raw")
        self.declare_parameter("pose_topic", "")
        self.declare_parameter("force_topic", "")
        self.declare_parameter("force_msg_type", "auto")  # auto | wrench | array
        self.declare_parameter("image_topic", "")
        self.declare_parameter("command_topic", "/vr_demo_recorder/command")

        # New multimodal streams
        self.declare_parameter("enable_global_cam", True)
        self.declare_parameter("global_image_topic", "/realsense/global/color/image_raw")
        self.declare_parameter("global_image_dataset_name", "cam1")
        self.declare_parameter("enable_aruco_markers", True)
        self.declare_parameter("aruco_id0_pose_topic", "/aruco/id_0/pose")
        self.declare_parameter("aruco_id1_pose_topic", "/aruco/id_1/pose")
        self.declare_parameter("enable_gripper_state", True)
        self.declare_parameter("gripper_position_topic", "/gripper/present_position")
        self.declare_parameter("gripper_current_topic", "/gripper/present_current_mA")

        # Sampling / freshness
        self.declare_parameter("sample_hz", 20.0)
        self.declare_parameter("require_pose_fresh_sec", 0.20)
        self.declare_parameter("require_force_fresh_sec", 0.20)
        self.declare_parameter("require_image_fresh_sec", 0.50)
        self.declare_parameter("require_global_image_fresh_sec", 0.80)
        self.declare_parameter("require_marker_fresh_sec", 0.80)
        self.declare_parameter("require_global_image", False)
        self.declare_parameter("require_aruco_id0", False)
        self.declare_parameter("require_aruco_id1", False)
        self.declare_parameter("recording_status_period_sec", 1.0)
        self.declare_parameter("idle_status_period_sec", 0.0)
        self.declare_parameter("command_dedupe_sec", 0.30)

        # Unit convention
        self.declare_parameter("pose_xyz_scale", 1000.0)  # m -> mm

        # Force processing
        self.declare_parameter("zero_xy_forces", True)
        self.declare_parameter("fz_ema_alpha", 0.2)
        self.declare_parameter("force_edge_zero_sec", 3.0)

        # Optional pose smoothing
        self.declare_parameter("pose_ema_enable", False)
        self.declare_parameter("pose_ema_alpha", 0.10)

        # Image save
        self.declare_parameter("image_dataset_name", "cam0")
        self.declare_parameter("image_compression", "gzip")  # gzip, lzf, none
        self.declare_parameter("image_gzip_level", 4)

        # Load parameters
        self.act_root_dir = str(self.get_parameter("act_root_dir").value)
        self.merged_subdir = str(self.get_parameter("merged_subdir").value)
        self.file_prefix = str(self.get_parameter("file_prefix").value)
        self.overwrite_file = bool(self.get_parameter("overwrite_file").value)
        self.allow_overwrite_episode = bool(self.get_parameter("allow_overwrite_episode").value)
        self.flush_each_episode = bool(self.get_parameter("flush_each_episode").value)
        self.num_episodes = int(self.get_parameter("num_episodes").value)
        self.min_samples = int(self.get_parameter("min_samples").value)

        self.recording_mode = str(self.get_parameter("recording_mode").value).strip().lower()
        self.tracker_pose_topic = str(self.get_parameter("tracker_pose_topic").value)
        self.tracker_force_topic = str(self.get_parameter("tracker_force_topic").value)
        self.tracker_image_topic = str(self.get_parameter("tracker_image_topic").value)
        self.robot_pose_topic = str(self.get_parameter("robot_pose_topic").value)
        self.robot_force_topic = str(self.get_parameter("robot_force_topic").value)
        self.robot_image_topic = str(self.get_parameter("robot_image_topic").value)

        pose_topic_override = str(self.get_parameter("pose_topic").value).strip()
        force_topic_override = str(self.get_parameter("force_topic").value).strip()
        force_msg_type_param = str(self.get_parameter("force_msg_type").value).strip().lower()
        image_topic_override = str(self.get_parameter("image_topic").value).strip()
        self.command_topic = str(self.get_parameter("command_topic").value)

        self.enable_global_cam = bool(self.get_parameter("enable_global_cam").value)
        self.global_image_topic = str(self.get_parameter("global_image_topic").value)
        self.global_image_dataset_name = str(self.get_parameter("global_image_dataset_name").value)
        self.enable_aruco_markers = bool(self.get_parameter("enable_aruco_markers").value)
        self.aruco_id0_pose_topic = str(self.get_parameter("aruco_id0_pose_topic").value)
        self.aruco_id1_pose_topic = str(self.get_parameter("aruco_id1_pose_topic").value)
        self.enable_gripper_state = bool(self.get_parameter("enable_gripper_state").value)
        self.gripper_position_topic = str(self.get_parameter("gripper_position_topic").value)
        self.gripper_current_topic = str(self.get_parameter("gripper_current_topic").value)

        if self.recording_mode not in ("tracker", "robot"):
            raise RuntimeError(f"recording_mode must be tracker or robot, got: {self.recording_mode}")

        default_pose_topic = self.tracker_pose_topic if self.recording_mode == "tracker" else self.robot_pose_topic
        default_force_topic = self.tracker_force_topic if self.recording_mode == "tracker" else self.robot_force_topic
        default_image_topic = self.tracker_image_topic if self.recording_mode == "tracker" else self.robot_image_topic
        default_force_msg_type = "wrench" if self.recording_mode == "tracker" else "array"
        self.pose_topic = pose_topic_override if pose_topic_override else default_pose_topic
        self.force_topic = force_topic_override if force_topic_override else default_force_topic
        self.force_msg_type = default_force_msg_type if force_msg_type_param in ("", "auto") else force_msg_type_param
        self.image_topic = image_topic_override if image_topic_override else default_image_topic
        if self.force_msg_type not in ("wrench", "array"):
            raise RuntimeError(
                f"force_msg_type must be auto, wrench, or array, got: {force_msg_type_param}"
            )

        self.sample_hz = float(self.get_parameter("sample_hz").value)
        self.dt = 1.0 / max(1e-9, self.sample_hz)
        self.require_pose_fresh_sec = float(self.get_parameter("require_pose_fresh_sec").value)
        self.require_force_fresh_sec = float(self.get_parameter("require_force_fresh_sec").value)
        self.require_image_fresh_sec = float(self.get_parameter("require_image_fresh_sec").value)
        self.require_global_image_fresh_sec = float(self.get_parameter("require_global_image_fresh_sec").value)
        self.require_marker_fresh_sec = float(self.get_parameter("require_marker_fresh_sec").value)
        self.require_global_image = bool(self.get_parameter("require_global_image").value)
        self.require_aruco_id0 = bool(self.get_parameter("require_aruco_id0").value)
        self.require_aruco_id1 = bool(self.get_parameter("require_aruco_id1").value)
        self.recording_status_period_sec = float(self.get_parameter("recording_status_period_sec").value)
        self.idle_status_period_sec = float(self.get_parameter("idle_status_period_sec").value)
        self.command_dedupe_sec = float(self.get_parameter("command_dedupe_sec").value)
        self.pose_xyz_scale = float(self.get_parameter("pose_xyz_scale").value)
        self.zero_xy_forces = bool(self.get_parameter("zero_xy_forces").value)
        self.fz_ema_alpha = float(self.get_parameter("fz_ema_alpha").value)
        self.force_edge_zero_sec = float(self.get_parameter("force_edge_zero_sec").value)
        self.pose_ema_enable = bool(self.get_parameter("pose_ema_enable").value)
        self.pose_ema_alpha = float(self.get_parameter("pose_ema_alpha").value)
        self.image_dataset_name = str(self.get_parameter("image_dataset_name").value)
        self.image_compression = str(self.get_parameter("image_compression").value).lower()
        self.image_gzip_level = int(self.get_parameter("image_gzip_level").value)

        # HDF5 setup
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.save_root = os.path.join(self.act_root_dir, self.timestamp)
        self.merged_dir = os.path.join(self.save_root, self.merged_subdir)
        os.makedirs(self.merged_dir, exist_ok=True)
        self.h5_path = os.path.join(self.merged_dir, f"{self.file_prefix}_{self.timestamp}.hdf5")
        if os.path.exists(self.h5_path) and not self.overwrite_file:
            raise RuntimeError(f"HDF5 file already exists: {self.h5_path}. Set overwrite_file:=true to overwrite.")

        self.h5_lock = threading.Lock()
        self.h5 = h5py.File(self.h5_path, "w")
        self.h5.attrs["created_unix"] = float(time.time())
        self.h5.attrs["created_time"] = str(datetime.now().isoformat())
        self.h5.attrs["recorder"] = "vr_demo_hdf5_recorder_multimodal"
        self.h5.attrs["schema_version"] = "multimodal_v2"
        self.h5.attrs["recording_mode"] = str(self.recording_mode)
        self.h5.attrs["pose_topic"] = str(self.pose_topic)
        self.h5.attrs["force_topic"] = str(self.force_topic)
        self.h5.attrs["force_msg_type"] = str(self.force_msg_type)
        self.h5.attrs["image_topic"] = str(self.image_topic)
        self.h5.attrs["global_image_topic"] = str(self.global_image_topic)
        self.h5.attrs["aruco_id0_pose_topic"] = str(self.aruco_id0_pose_topic)
        self.h5.attrs["aruco_id1_pose_topic"] = str(self.aruco_id1_pose_topic)
        self.h5.attrs["gripper_position_topic"] = str(self.gripper_position_topic)
        self.h5.attrs["gripper_current_topic"] = str(self.gripper_current_topic)
        self.grp_eps = self.h5.create_group("episodes")

        # Runtime state
        self.state_lock = threading.Lock()
        self.latest_pose: Optional[np.ndarray] = None
        self.latest_pose_t: float = 0.0
        self.latest_force: Optional[np.ndarray] = None
        self.latest_force_t: float = 0.0
        self.latest_image: Optional[np.ndarray] = None
        self.latest_image_t: float = 0.0
        self.latest_global_image: Optional[np.ndarray] = None
        self.latest_global_image_t: float = 0.0
        self.latest_marker0: Optional[np.ndarray] = None
        self.latest_marker0_quat: Optional[np.ndarray] = None
        self.latest_marker0_t: float = 0.0
        self.latest_marker1: Optional[np.ndarray] = None
        self.latest_marker1_quat: Optional[np.ndarray] = None
        self.latest_marker1_t: float = 0.0
        self.latest_gripper_position: Optional[int] = None
        self.latest_gripper_position_t: float = 0.0
        self.latest_gripper_current_mA: Optional[float] = None
        self.latest_gripper_current_t: float = 0.0

        self.episode_active = False
        self.finishing = False
        self.stop_requested = False
        self.current_ep_idx = 0
        self.saved_indices: Set[int] = set()

        self.P_buf: List[np.ndarray] = []
        self.F_buf: List[np.ndarray] = []
        self.I0_buf: List[Optional[np.ndarray]] = []
        self.I1_buf: List[Optional[np.ndarray]] = []
        self.M0_buf: List[Optional[np.ndarray]] = []
        self.M1_buf: List[Optional[np.ndarray]] = []
        self.M0Q_buf: List[Optional[np.ndarray]] = []
        self.M1Q_buf: List[Optional[np.ndarray]] = []
        self.GP_buf: List[Optional[int]] = []
        self.GC_buf: List[Optional[float]] = []
        self.sample_time_buf: List[float] = []
        self.last_status_t = 0.0
        self.last_idle_status_t = 0.0
        self.last_command: str = ""
        self.last_command_t: float = 0.0

        # ROS I/O
        image_qos = make_qos(depth=1, best_effort=True)
        reliable_qos = make_qos(depth=10, best_effort=False)
        self.create_subscription(Float64MultiArray, self.pose_topic, self._on_pose, reliable_qos)
        if self.force_msg_type == "wrench":
            self.create_subscription(Wrench, self.force_topic, self._on_force_wrench, reliable_qos)
        else:
            self.create_subscription(Float64MultiArray, self.force_topic, self._on_force_array, reliable_qos)
        self.create_subscription(Image, self.image_topic, self._on_image, image_qos)
        if self.enable_global_cam:
            self.create_subscription(Image, self.global_image_topic, self._on_global_image, image_qos)
        if self.enable_aruco_markers:
            self.create_subscription(PoseStamped, self.aruco_id0_pose_topic, self._on_marker0, reliable_qos)
            self.create_subscription(PoseStamped, self.aruco_id1_pose_topic, self._on_marker1, reliable_qos)
        if self.enable_gripper_state:
            self.create_subscription(Int32, self.gripper_position_topic, self._on_gripper_position, reliable_qos)
            self.create_subscription(Float32, self.gripper_current_topic, self._on_gripper_current, reliable_qos)
        self.create_subscription(String, self.command_topic, self._on_command, reliable_qos)
        self.timer = self.create_timer(self.dt, self._on_sample_timer)

        atexit.register(self._atexit_close)

        self.get_logger().info(
            "[READY] vr_demo_hdf5_recorder multimodal\n"
            f"  h5_path={self.h5_path}\n"
            f"  recording_mode={self.recording_mode}\n"
            f"  pose_topic={self.pose_topic}\n"
            f"  force_topic={self.force_topic} ({self.force_msg_type})\n"
            f"  cam0_topic={self.image_topic} -> images/{self.image_dataset_name}\n"
            f"  enable_global_cam={int(self.enable_global_cam)} cam1_topic={self.global_image_topic} -> images/{self.global_image_dataset_name}\n"
            f"  enable_aruco_markers={int(self.enable_aruco_markers)} id0={self.aruco_id0_pose_topic} id1={self.aruco_id1_pose_topic}\n"
            f"  enable_gripper_state={int(self.enable_gripper_state)} position={self.gripper_position_topic} current={self.gripper_current_topic}\n"
            f"  sample_hz={self.sample_hz}\n"
            f"  command_dedupe_sec={self.command_dedupe_sec}\n"
            f"  command_topic={self.command_topic}"
        )
        self._print_status("READY")

    # callbacks
    def _on_pose(self, msg: Float64MultiArray):
        arr = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if arr.size < 6:
            return
        pose = arr[:6].astype(np.float32).copy()
        pose[:3] *= np.float32(self.pose_xyz_scale)
        with self.state_lock:
            self.latest_pose = pose
            self.latest_pose_t = time.time()

    def _on_force_wrench(self, msg: Wrench):
        f = np.asarray([msg.force.x, msg.force.y, msg.force.z], dtype=np.float32)
        with self.state_lock:
            self.latest_force = f
            self.latest_force_t = time.time()

    def _on_force_array(self, msg: Float64MultiArray):
        arr = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            return
        with self.state_lock:
            self.latest_force = arr[:3].astype(np.float32).copy()
            self.latest_force_t = time.time()

    def _on_image(self, msg: Image):
        im = image_to_rgb_numpy(msg)
        if im is None:
            return
        with self.state_lock:
            self.latest_image = im
            self.latest_image_t = time.time()

    def _on_global_image(self, msg: Image):
        im = image_to_rgb_numpy(msg)
        if im is None:
            return
        with self.state_lock:
            self.latest_global_image = im
            self.latest_global_image_t = time.time()

    def _on_marker0(self, msg: PoseStamped):
        m7, m8 = pose_stamped_to_marker_vec(msg)
        with self.state_lock:
            self.latest_marker0 = m7
            self.latest_marker0_quat = m8
            self.latest_marker0_t = time.time()

    def _on_marker1(self, msg: PoseStamped):
        m7, m8 = pose_stamped_to_marker_vec(msg)
        with self.state_lock:
            self.latest_marker1 = m7
            self.latest_marker1_quat = m8
            self.latest_marker1_t = time.time()

    def _on_gripper_position(self, msg: Int32):
        with self.state_lock:
            self.latest_gripper_position = int(msg.data)
            self.latest_gripper_position_t = time.time()

    def _on_gripper_current(self, msg: Float32):
        with self.state_lock:
            self.latest_gripper_current_mA = float(msg.data)
            self.latest_gripper_current_t = time.time()

    def _on_command(self, msg: String):
        cmd = str(msg.data).strip().lower()
        if not cmd:
            return
        now = time.time()
        if (
            cmd == self.last_command
            and (now - self.last_command_t) < max(0.0, self.command_dedupe_sec)
        ):
            self.get_logger().warn(
                f"[COMMAND] duplicate ignored: {cmd} "
                f"(dt={now - self.last_command_t:.3f}s)"
            )
            return
        self.last_command = cmd
        self.last_command_t = now
        self.get_logger().warn(f"[COMMAND] {cmd}")
        if cmd == "start_recording":
            self.start_episode(reason="joystick_start")
        elif cmd == "end_recording":
            self.end_episode(reason="joystick_end")
        elif cmd == "prev_episode":
            self.prev_episode()
        elif cmd == "next_episode":
            self.next_episode()
        else:
            self.get_logger().warn(f"[COMMAND] unknown command ignored: {cmd}")

    # episode control
    def start_episode(self, reason: str = "start"):
        if self.stop_requested:
            self.get_logger().warn("Cannot start episode: stop already requested.")
            return
        if self.finishing:
            self.get_logger().warn("Cannot start episode: previous episode is still being saved.")
            return
        if self.episode_active:
            self.get_logger().warn("Episode already active.")
            return
        if not self.allow_overwrite_episode:
            prev_idx = int(self.current_ep_idx)
            self.current_ep_idx = self._next_available_ep_idx(start_idx=self.current_ep_idx)
            if self.current_ep_idx != prev_idx:
                self.get_logger().warn(
                    f"Episode index {prev_idx} already exists. "
                    f"Auto-advanced to {self._ep_name(self.current_ep_idx)}."
                )

        for buf in [self.P_buf, self.F_buf, self.I0_buf, self.I1_buf, self.M0_buf, self.M1_buf, self.M0Q_buf, self.M1Q_buf, self.GP_buf, self.GC_buf, self.sample_time_buf]:
            buf.clear()
        self.episode_active = True
        self.get_logger().warn(f"[EP {self._ep_name(self.current_ep_idx)}] START reason={reason}")
        self._print_status("RECORDING")

    def end_episode(self, reason: str = "end"):
        if not self.episode_active:
            self.get_logger().warn("No active episode to end.")
            return
        if self.finishing:
            self.get_logger().warn("Episode already finishing.")
            return
        self.episode_active = False
        self.finishing = True
        ep_idx = int(self.current_ep_idx)
        args = (
            ep_idx,
            list(self.P_buf), list(self.F_buf), list(self.I0_buf), list(self.I1_buf),
            list(self.M0_buf), list(self.M1_buf), list(self.M0Q_buf), list(self.M1Q_buf),
            list(self.GP_buf), list(self.GC_buf),
            list(self.sample_time_buf), reason,
        )
        self.get_logger().warn(f"[EP {self._ep_name(ep_idx)}] END requested. raw_samples={len(self.P_buf)}, reason={reason}")
        threading.Thread(target=self._finish_episode_worker, args=args, daemon=True).start()

    def prev_episode(self):
        if self.episode_active or self.finishing:
            self.get_logger().warn("Cannot change episode index while recording/saving.")
            return
        self.current_ep_idx = max(0, self.current_ep_idx - 1)
        self._print_status("PREV")

    def next_episode(self):
        if self.episode_active or self.finishing:
            self.get_logger().warn("Cannot change episode index while recording/saving.")
            return
        self.current_ep_idx += 1
        self._print_status("NEXT")

    def request_stop(self, reason: str = "stop"):
        self.stop_requested = True
        self.get_logger().warn(f"[STOP] requested: {reason}")
        if self.episode_active:
            self.end_episode(reason=f"stop_requested:{reason}")
        elif not self.finishing:
            self.finalize_and_shutdown()

    # sampling
    def _on_sample_timer(self):
        if not self.episode_active:
            if self.idle_status_period_sec > 0.0:
                now = time.time()
                if now - self.last_idle_status_t >= self.idle_status_period_sec:
                    self.last_idle_status_t = now
                    self._print_status("IDLE")
            return

        now = time.time()
        with self.state_lock:
            pose = None if self.latest_pose is None else self.latest_pose.copy()
            force = None if self.latest_force is None else self.latest_force.copy()
            image = None if self.latest_image is None else self.latest_image.copy()
            global_image = None if self.latest_global_image is None else self.latest_global_image.copy()
            marker0 = None if self.latest_marker0 is None else self.latest_marker0.copy()
            marker1 = None if self.latest_marker1 is None else self.latest_marker1.copy()
            marker0q = None if self.latest_marker0_quat is None else self.latest_marker0_quat.copy()
            marker1q = None if self.latest_marker1_quat is None else self.latest_marker1_quat.copy()
            gripper_position = self.latest_gripper_position
            gripper_current_mA = self.latest_gripper_current_mA
            pose_age = now - self.latest_pose_t if self.latest_pose_t > 0 else 1e9
            force_age = now - self.latest_force_t if self.latest_force_t > 0 else 1e9
            image_age = now - self.latest_image_t if self.latest_image_t > 0 else 1e9
            global_age = now - self.latest_global_image_t if self.latest_global_image_t > 0 else 1e9
            m0_age = now - self.latest_marker0_t if self.latest_marker0_t > 0 else 1e9
            m1_age = now - self.latest_marker1_t if self.latest_marker1_t > 0 else 1e9

        missing = []
        if pose is None or pose_age > self.require_pose_fresh_sec:
            missing.append(f"pose(age={pose_age:.3f})")
        if force is None or force_age > self.require_force_fresh_sec:
            missing.append(f"force(age={force_age:.3f})")
        if image is None or image_age > self.require_image_fresh_sec:
            missing.append(f"cam0(age={image_age:.3f})")
        if self.enable_global_cam and self.require_global_image and (global_image is None or global_age > self.require_global_image_fresh_sec):
            missing.append(f"cam1/global(age={global_age:.3f})")
        if self.enable_aruco_markers and self.require_aruco_id0 and (marker0 is None or m0_age > self.require_marker_fresh_sec):
            missing.append(f"aruco_id0(age={m0_age:.3f})")
        if self.enable_aruco_markers and self.require_aruco_id1 and (marker1 is None or m1_age > self.require_marker_fresh_sec):
            missing.append(f"aruco_id1(age={m1_age:.3f})")

        if missing:
            if now - self.last_status_t >= max(0.5, self.recording_status_period_sec):
                self.last_status_t = now
                self.get_logger().warn("[WAIT] " + ", ".join(missing))
            return

        self.P_buf.append(pose.astype(np.float32))
        self.F_buf.append(force[:3].astype(np.float32))
        self.I0_buf.append(image)
        self.I1_buf.append(global_image if self.enable_global_cam else None)
        self.M0_buf.append(marker0 if self.enable_aruco_markers else None)
        self.M1_buf.append(marker1 if self.enable_aruco_markers else None)
        self.M0Q_buf.append(marker0q if self.enable_aruco_markers else None)
        self.M1Q_buf.append(marker1q if self.enable_aruco_markers else None)
        self.GP_buf.append(gripper_position if self.enable_gripper_state else None)
        self.GC_buf.append(gripper_current_mA if self.enable_gripper_state else None)
        self.sample_time_buf.append(float(now))

        if now - self.last_status_t >= self.recording_status_period_sec:
            self.last_status_t = now
            self._print_status("RECORDING")

    # save worker
    def _finish_episode_worker(self, ep_idx, P_list, F_list, I0_list, I1_list, M0_list, M1_list, M0Q_list, M1Q_list, GP_list, GC_list, sample_time_list, reason):
        try:
            N = len(P_list)
            if N < max(1, self.min_samples):
                self.get_logger().warn(f"Episode dropped: raw_len={N} < min_samples={self.min_samples}, reason={reason}")
                return

            P = np.asarray(P_list, dtype=np.float32).reshape(N, 6)
            Fraw = np.asarray(F_list, dtype=np.float32).reshape(N, 3)
            P_out = ema_nd(P.astype(np.float64), self.pose_ema_alpha).astype(np.float32) if self.pose_ema_enable else P.copy()
            F_out = process_force_keep_fz_with_ema_and_edge_zero(
                Fraw, self.fz_ema_alpha, self.force_edge_zero_sec, self.sample_hz,
                zero_xy=self.zero_xy_forces, logger=self.get_logger()
            ).astype(np.float32)

            images0 = stack_images_repeat_last(I0_list, logger=self.get_logger(), tag="IMAGE/cam0")
            if images0 is None:
                self.get_logger().warn(f"Episode dropped: no valid cam0 frames. N={N}, reason={reason}")
                return
            images1 = stack_images_repeat_last(I1_list, logger=self.get_logger(), tag="IMAGE/cam1_global") if self.enable_global_cam else None
            if self.enable_global_cam and images1 is None:
                self.get_logger().warn("[IMAGE/cam1_global] no valid global frames. Saving cam0 only for this episode.")

            marker0 = stack_markers_repeat_last(M0_list, 7, logger=self.get_logger(), tag="MARKER/id0")
            marker1 = stack_markers_repeat_last(M1_list, 7, logger=self.get_logger(), tag="MARKER/id1")
            marker0q = stack_markers_repeat_last(M0Q_list, 8, logger=self.get_logger(), tag="MARKER/id0_quat")
            marker1q = stack_markers_repeat_last(M1Q_list, 8, logger=self.get_logger(), tag="MARKER/id1_quat")
            marker_combined = np.concatenate([marker0, marker1], axis=1).astype(np.float32)
            gripper_position = stack_scalar_repeat_last(GP_list, np.int32, logger=self.get_logger(), tag="GRIPPER/present_position")
            gripper_current_mA = stack_scalar_repeat_last(GC_list, np.float32, logger=self.get_logger(), tag="GRIPPER/present_current_mA")

            self._save_episode_to_hdf5(
                ep_idx, P_out, F_out, images0, images1,
                marker0, marker1, marker0q, marker1q, marker_combined,
                gripper_position, gripper_current_mA,
                np.asarray(sample_time_list), reason,
            )
            self.saved_indices.add(ep_idx)
            if ep_idx == self.current_ep_idx:
                self.current_ep_idx = self._next_available_ep_idx(start_idx=self.current_ep_idx + 1)

            self.get_logger().info(
                f"=== EPISODE SAVED ({self._ep_name(ep_idx)}) N={N}, "
                f"position={P_out.shape}, ft={F_out.shape}, cam0={images0.shape}, "
                f"cam1={None if images1 is None else images1.shape}, "
                f"marker0_valid={int(np.sum(marker0[:, -1] > 0.5))}/{N}, "
                f"marker1_valid={int(np.sum(marker1[:, -1] > 0.5))}/{N}, "
                f"gripper_position={gripper_position.shape}, gripper_current_mA={gripper_current_mA.shape}, "
                f"reason={reason} ==="
            )
            self._print_status("SAVED")
        except Exception as e:
            self.get_logger().error(f"Episode processing failed: {repr(e)}")
        finally:
            self.finishing = False
            if self.stop_requested and not self.episode_active:
                self.finalize_and_shutdown()

    def _compression_kwargs(self) -> Dict[str, object]:
        mode = str(self.image_compression).lower()
        if mode == "gzip":
            return dict(compression="gzip", compression_opts=int(self.image_gzip_level), shuffle=True)
        if mode == "lzf":
            return dict(compression="lzf", shuffle=True)
        return {}

    def _save_episode_to_hdf5(self, ep_idx, position, ft, images0, images1, marker0, marker1, marker0q, marker1q, marker_combined, gripper_position, gripper_current_mA, sample_times, reason):
        ep_name = self._ep_name(ep_idx)
        tmp_name = self._tmp_ep_name(ep_idx)
        with self.h5_lock:
            if ep_name in self.grp_eps:
                if not self.allow_overwrite_episode:
                    raise RuntimeError(f"{ep_name} already exists and allow_overwrite_episode=False")
                del self.grp_eps[ep_name]

            if tmp_name in self.grp_eps:
                del self.grp_eps[tmp_name]

            try:
                g = self.grp_eps.create_group(tmp_name)
                g.attrs["saved_unix"] = float(time.time())
                g.attrs["reason"] = str(reason)
                g.attrs["raw_len"] = int(position.shape[0])
                g.attrs["out_len"] = int(position.shape[0])
                g.attrs["record_hz"] = float(self.sample_hz)
                g.attrs["dt"] = float(self.dt)
                g.attrs["recording_mode"] = str(self.recording_mode)
                g.attrs["pose_topic"] = str(self.pose_topic)
                g.attrs["force_topic"] = str(self.force_topic)
                g.attrs["force_msg_type"] = str(self.force_msg_type)
                g.attrs["cam0_topic"] = str(self.image_topic)
                g.attrs["cam1_topic"] = str(self.global_image_topic)
                g.attrs["aruco_id0_pose_topic"] = str(self.aruco_id0_pose_topic)
                g.attrs["aruco_id1_pose_topic"] = str(self.aruco_id1_pose_topic)
                g.attrs["gripper_position_topic"] = str(self.gripper_position_topic)
                g.attrs["gripper_current_topic"] = str(self.gripper_current_topic)
                g.attrs["schema_version"] = "multimodal_v2"
                g.attrs["marker_format"] = "[x,y,z,rx,ry,rz,valid]"
                g.attrs["marker_quat_format"] = "[x,y,z,qx,qy,qz,qw,valid]"

                g.create_dataset("position", data=position.astype(np.float32), compression="gzip", compression_opts=4, shuffle=True)
                g.create_dataset("ft", data=ft.astype(np.float32), compression="gzip", compression_opts=4, shuffle=True)
                g.create_dataset("sample_time_unix", data=sample_times.astype(np.float64), compression="gzip", compression_opts=4, shuffle=True)

                g_img = g.create_group("images")
                g_img.create_dataset(self.image_dataset_name, data=images0.astype(np.uint8), **self._compression_kwargs())
                if images1 is not None:
                    g_img.create_dataset(self.global_image_dataset_name, data=images1.astype(np.uint8), **self._compression_kwargs())

                g_marker = g.create_group("marker")
                g_marker.create_dataset("id0", data=marker0.astype(np.float32), compression="gzip", compression_opts=4, shuffle=True)
                g_marker.create_dataset("id1", data=marker1.astype(np.float32), compression="gzip", compression_opts=4, shuffle=True)
                g_marker.create_dataset("combined", data=marker_combined.astype(np.float32), compression="gzip", compression_opts=4, shuffle=True)
                g_marker.create_dataset("id0_quat", data=marker0q.astype(np.float32), compression="gzip", compression_opts=4, shuffle=True)
                g_marker.create_dataset("id1_quat", data=marker1q.astype(np.float32), compression="gzip", compression_opts=4, shuffle=True)

                g_gripper = g.create_group("gripper")
                g_gripper.create_dataset("present_position", data=gripper_position.astype(np.int32), compression="gzip", compression_opts=4, shuffle=True)
                g_gripper.create_dataset("present_current_mA", data=gripper_current_mA.astype(np.float32), compression="gzip", compression_opts=4, shuffle=True)

                g.attrs["save_complete"] = 1
                self.grp_eps.move(tmp_name, ep_name)

                if self.flush_each_episode:
                    self.h5.flush()
            except Exception:
                if tmp_name in self.grp_eps:
                    del self.grp_eps[tmp_name]
                raise

    # status / shutdown
    def _ep_name(self, idx: int) -> str:
        return f"ep_{int(idx):04d}"

    def _tmp_ep_name(self, idx: int) -> str:
        return f"{self._ep_name(idx)}__writing"

    def _episode_exists(self, idx: int) -> bool:
        ep_name = self._ep_name(idx)
        with self.h5_lock:
            return ep_name in self.grp_eps

    def _next_available_ep_idx(self, start_idx: int) -> int:
        idx = max(0, int(start_idx))
        while (idx in self.saved_indices) or self._episode_exists(idx):
            idx += 1
        return idx

    def _print_status(self, tag: str):
        with self.state_lock:
            pose_ok = self.latest_pose is not None
            force_ok = self.latest_force is not None
            img0_ok = self.latest_image is not None
            img1_ok = self.latest_global_image is not None
            m0_ok = self.latest_marker0 is not None
            m1_ok = self.latest_marker1 is not None
            grip_pos_ok = self.latest_gripper_position is not None
            grip_cur_ok = self.latest_gripper_current_mA is not None
        self.get_logger().info(
            f"[{tag}] ep={self._ep_name(self.current_ep_idx)} active={int(self.episode_active)} "
            f"finishing={int(self.finishing)} samples={len(self.P_buf)} saved={sorted(self.saved_indices)} | "
            f"pose={int(pose_ok)} force={int(force_ok)} cam0={int(img0_ok)} cam1={int(img1_ok)} "
            f"aruco0={int(m0_ok)} aruco1={int(m1_ok)} "
            f"grip_pos={int(grip_pos_ok)} grip_cur={int(grip_cur_ok)}"
        )

    def finalize_and_shutdown(self):
        self.get_logger().warn("[FINALIZE] closing HDF5 and shutting down recorder.")
        try:
            with self.h5_lock:
                if self.h5 is not None:
                    self.h5.flush()
                    self.h5.close()
                    self.h5 = None
        except Exception as e:
            self.get_logger().error(f"HDF5 close failed: {repr(e)}")

    def _atexit_close(self):
        try:
            if getattr(self, "h5", None) is not None:
                self.h5.flush()
                self.h5.close()
                self.h5 = None
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = VRDemoHDF5Recorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        try:
            node.request_stop(reason="KeyboardInterrupt")
        except Exception:
            pass
        time.sleep(0.1)
        try:
            if rclpy.ok():
                node.finalize_and_shutdown()
        except Exception:
            pass
    finally:
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
