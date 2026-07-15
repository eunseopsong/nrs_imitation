#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hdf5_recorder_base.py

Shared multimodal merged-HDF5 recorder implementation for nrs_imitation.

Existing single-cam recorder behavior is preserved, and this version adds:
  - global camera RGB stream as images/cam1

Default topics:
  cam0 tracker mode : /realsense/vr/color/image_raw
  cam0 robot mode   : /realsense/robot/color/image_raw
  cam1 global       : /realsense/global/color/image_raw

Saved merged HDF5 layout:
  ~/nrs_imitation/datasets/polishing/<obs_mode>/YYYYMMDD_HHMM/merged_hdf5/
    vr_demo_merged_YYYYMMDD_HHMM.hdf5

  episodes/
    ep_0000/
      position             (T, 6) float32  [x_mm y_mm z_mm wx wy wz]
      ft                   (T, 3) float32  [fx fy fz]
      images/
        cam0               (T, H, W, 3) uint8 RGB
        cam1               (T, H, W, 3) uint8 RGB   optional/global

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

import numpy as np
import h5py

try:
    import cv2
except Exception:
    cv2 = None

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import Wrench
from sensor_msgs.msg import Image

from nrs_imitation.pretty_print import block, status
from nrs_imitation.stage1_filtering import (
    apply_stage1_filter,
    stage1_config_from_recorder,
    take_nearest_by_source_index,
)
from nrs_imitation.vr_demo_txt_recorder import (
    save_plot_1_lin_kinematics,
    save_plot_2_rotvec_kinematics,
    save_plot_3_forces,
)


REPO_ROOT = os.path.expanduser("~/nrs_imitation")
DEFAULT_DATASET_ROOT_DIR = os.path.join(REPO_ROOT, "datasets", "polishing")
VALID_OBS_MODES = ("single_cam", "dual_cam")


def infer_obs_mode(enable_global_cam: bool) -> str:
    if enable_global_cam:
        return "dual_cam"
    return "single_cam"


def normalize_obs_mode(obs_mode: str, enable_global_cam: bool) -> str:
    mode = str(obs_mode).strip().lower()
    if mode in ("", "auto"):
        return infer_obs_mode(enable_global_cam)
    if mode not in VALID_OBS_MODES:
        raise RuntimeError(f"obs_mode must be one of {VALID_OBS_MODES} or auto, got: {obs_mode}")
    return mode


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


def preprocess_rgb_image(
    image: Optional[np.ndarray],
    mode: str,
    specular_mask_mode: str,
    specular_v_thresh: int,
    specular_s_thresh: int,
    specular_dilate_px: int,
    specular_inpaint_radius: float,
    specular_attenuate_gain: float,
) -> Optional[np.ndarray]:
    if image is None:
        return None

    mode = str(mode or "raw").strip().lower()
    if mode in ("", "raw", "none", "off"):
        return image

    if mode not in ("specular_inpaint", "deglare", "highlight_inpaint", "highlight_attenuate", "specular_attenuate"):
        return image

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return image
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    if cv2 is None:
        return rgb.copy()

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask_mode = str(specular_mask_mode or "white").strip().lower()
    bright = val >= int(specular_v_thresh)
    if mask_mode in ("bright", "value", "v"):
        mask_bool = bright
    elif mask_mode in ("mixed", "white_or_bright"):
        mask_bool = bright | ((val >= int(specular_v_thresh)) & (sat <= int(specular_s_thresh)))
    else:
        mask_bool = bright & (sat <= int(specular_s_thresh))
    mask = mask_bool.astype(np.uint8) * 255

    dilate_px = max(0, int(specular_dilate_px))
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    if int(np.count_nonzero(mask)) == 0:
        return rgb.copy()

    if mode in ("highlight_attenuate", "specular_attenuate"):
        soft = mask.astype(np.float32) / 255.0
        if dilate_px > 0:
            sigma = max(0.1, float(dilate_px))
            soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=sigma, sigmaY=sigma)
            soft = np.clip(soft, 0.0, 1.0)

        gain = float(np.clip(float(specular_attenuate_gain), 0.0, 1.0))
        v = val.astype(np.float32)
        target_v = np.minimum(v, float(specular_v_thresh) + np.maximum(v - float(specular_v_thresh), 0.0) * gain)
        hsv_out = hsv.copy()
        hsv_out[:, :, 2] = np.clip((1.0 - soft) * v + soft * target_v, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB)

    radius = max(0.1, float(specular_inpaint_radius))
    return cv2.inpaint(rgb, mask, radius, cv2.INPAINT_TELEA)


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

class HDF5Recorder(Node):
    def __init__(self, node_name: str, fixed_defaults: Optional[Dict[str, object]] = None):
        super().__init__(node_name)
        self.recorder_name = str(node_name)
        self.fixed_defaults = dict(fixed_defaults or {})

        def declare(name: str, default):
            locked = name in self.fixed_defaults
            value = self.fixed_defaults.get(name, default)
            self.declare_parameter(name, value, ignore_override=locked)

        # Save parameters
        declare("act_root_dir", DEFAULT_DATASET_ROOT_DIR)
        declare("merged_subdir", "merged_hdf5")
        declare("file_prefix", node_name)
        declare("obs_mode", "auto")
        declare("overwrite_file", False)
        declare("allow_overwrite_episode", False)
        declare("flush_each_episode", True)
        declare("num_episodes", 50)
        declare("min_samples", 10)

        # Topic parameters
        declare("recording_mode", "tracker")  # tracker | robot
        declare("tracker_pose_topic", "/calibrated_pose")
        declare("tracker_force_topic", "/ftsensor/measured_Cvalue")
        declare("tracker_image_topic", "/realsense/vr/color/image_raw")
        declare("robot_pose_topic", "/ur10skku/currentP")
        declare("robot_force_topic", "/ur10skku/currentF")
        declare("robot_image_topic", "/realsense/robot/color/image_raw")
        declare("pose_topic", "")
        declare("force_topic", "")
        declare("force_msg_type", "auto")  # auto | wrench | array
        declare("image_topic", "")
        declare("command_topic", "/vr_demo_recorder/command")

        # New multimodal streams
        declare("enable_global_cam", True)
        declare("global_image_topic", "/realsense/global/color/image_raw")
        declare("global_image_dataset_name", "cam1")

        # Sampling / freshness
        declare("sample_hz", 30.0)
        declare("require_pose_fresh_sec", 0.20)
        declare("require_force_fresh_sec", 0.20)
        declare("require_image_fresh_sec", 0.50)
        declare("require_global_image_fresh_sec", 0.80)
        declare("require_global_image", False)
        declare("recording_status_period_sec", 1.0)
        declare("idle_status_period_sec", 0.0)

        # Unit convention
        declare("pose_xyz_scale", 1000.0)  # m -> mm

        # Stage-1-compatible force / pose trajectory filtering
        declare("force_filter_mode", "ema")  # ema | contact_cleanup
        declare("zero_xy_forces", False)
        declare("force_clamp_abs", 200.0)
        declare("force_ema_alpha", 0.2)
        declare("contact_thr_N", 5.0)
        declare("consec_on", 10)
        declare("consec_off", 10)
        declare("fz_contact_smooth_enable", True)
        declare("fz_contact_lam_d2", 4000.0)

        # Legacy HDF5-only force parameters are kept for ROS argument compatibility.
        declare("fz_ema_alpha", 0.2)
        declare("force_edge_zero_sec", 3.0)

        declare("filter_reference_hz", 125.0)
        declare("scale_filter_params_with_hz", True)
        declare("hampel_enable", True)
        declare("hampel_win", 16)
        declare("hampel_sig", 2.0)
        declare("lam_pos_d2", 250000.0)
        declare("lam_ang_d2", 6000.0)
        declare("pose_ema_enable", False)
        declare("pose_ema_alpha", 0.10)
        declare("retime_k", 1)
        declare("approach_slowdown_enable", False)
        declare("approach_pre_sec", 5.0)
        declare("approach_post_sec", 0.3)
        declare("approach_scale_max", 30.0)
        declare("approach_use_fz_ramp", True)
        declare("approach_fz_full", 20.0)
        declare("post_enable", True)
        declare("lam_pos_d3", 2.0e7)
        declare("lam_ang_d3", 6.0e5)
        declare("qp_guard_enable", True)
        declare("qp_guard_safety", 0.75)
        declare("qp_guard_max_iter", 8)
        declare("qp_guard_growth", 2.2)
        declare("max_dev_pos_mm", 8.0)
        declare("max_dev_ang_rad", 0.06)
        declare("cg_iters", 400)
        declare("cg_tol", 1e-8)
        declare("pos_vmax", 30.0)
        declare("pos_amax", 120.0)
        declare("ang_vmax", 0.6)
        declare("ang_amax", 3.0)
        declare("pos_jmax", 5000.0)
        declare("ang_jmax", 80.0)

        # Image save
        declare("image_dataset_name", "cam0")
        declare("image_compression", "gzip")  # gzip, lzf, none
        declare("image_gzip_level", 4)
        declare("image_preprocess_mode", "raw")  # raw | highlight_attenuate | specular_inpaint
        declare("image_specular_mask_mode", "white")  # white | bright
        declare("image_specular_v_thresh", 230)
        declare("image_specular_s_thresh", 80)
        declare("image_specular_dilate_px", 2)
        declare("image_specular_inpaint_radius", 3.0)
        declare("image_specular_attenuate_gain", 0.35)

        # Load parameters
        self.act_root_dir = os.path.expanduser(str(self.get_parameter("act_root_dir").value))
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
        self.obs_mode = normalize_obs_mode(
            str(self.get_parameter("obs_mode").value),
            self.enable_global_cam,
        )

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
        self.require_global_image = bool(self.get_parameter("require_global_image").value)
        self.recording_status_period_sec = float(self.get_parameter("recording_status_period_sec").value)
        self.idle_status_period_sec = float(self.get_parameter("idle_status_period_sec").value)
        self.pose_xyz_scale = float(self.get_parameter("pose_xyz_scale").value)
        if self.recording_mode == "robot" and abs(self.pose_xyz_scale - 1000.0) < 1e-9:
            self.pose_xyz_scale = 1.0
            self.get_logger().warn(
                "[UNIT] robot recording uses /ur10skku/currentP in mm; "
                "overriding pose_xyz_scale 1000.0 -> 1.0 to avoid x1000 datasets."
            )
        self.force_filter_mode = str(self.get_parameter("force_filter_mode").value)
        self.zero_xy_forces = bool(self.get_parameter("zero_xy_forces").value)
        self.force_clamp_abs = float(self.get_parameter("force_clamp_abs").value)
        self.force_ema_alpha = float(self.get_parameter("force_ema_alpha").value)
        self.contact_thr_N = float(self.get_parameter("contact_thr_N").value)
        self.consec_on = int(self.get_parameter("consec_on").value)
        self.consec_off = int(self.get_parameter("consec_off").value)
        self.fz_contact_smooth_enable = bool(self.get_parameter("fz_contact_smooth_enable").value)
        self.fz_contact_lam_d2 = float(self.get_parameter("fz_contact_lam_d2").value)
        self.fz_ema_alpha = float(self.get_parameter("fz_ema_alpha").value)
        self.force_edge_zero_sec = float(self.get_parameter("force_edge_zero_sec").value)
        self.filter_reference_hz = float(self.get_parameter("filter_reference_hz").value)
        self.scale_filter_params_with_hz = bool(self.get_parameter("scale_filter_params_with_hz").value)
        self.hampel_enable = bool(self.get_parameter("hampel_enable").value)
        self.hampel_win = int(self.get_parameter("hampel_win").value)
        self.hampel_sig = float(self.get_parameter("hampel_sig").value)
        self.lam_pos_d2 = float(self.get_parameter("lam_pos_d2").value)
        self.lam_ang_d2 = float(self.get_parameter("lam_ang_d2").value)
        self.pose_ema_enable = bool(self.get_parameter("pose_ema_enable").value)
        self.pose_ema_alpha = float(self.get_parameter("pose_ema_alpha").value)
        self.retime_k = int(self.get_parameter("retime_k").value)
        self.approach_slowdown_enable = bool(self.get_parameter("approach_slowdown_enable").value)
        self.approach_pre_sec = float(self.get_parameter("approach_pre_sec").value)
        self.approach_post_sec = float(self.get_parameter("approach_post_sec").value)
        self.approach_scale_max = float(self.get_parameter("approach_scale_max").value)
        self.approach_use_fz_ramp = bool(self.get_parameter("approach_use_fz_ramp").value)
        self.approach_fz_full = float(self.get_parameter("approach_fz_full").value)
        self.post_enable = bool(self.get_parameter("post_enable").value)
        self.lam_pos_d3 = float(self.get_parameter("lam_pos_d3").value)
        self.lam_ang_d3 = float(self.get_parameter("lam_ang_d3").value)
        self.qp_guard_enable = bool(self.get_parameter("qp_guard_enable").value)
        self.qp_guard_safety = float(self.get_parameter("qp_guard_safety").value)
        self.qp_guard_max_iter = int(self.get_parameter("qp_guard_max_iter").value)
        self.qp_guard_growth = float(self.get_parameter("qp_guard_growth").value)
        self.max_dev_pos_mm = float(self.get_parameter("max_dev_pos_mm").value)
        self.max_dev_ang_rad = float(self.get_parameter("max_dev_ang_rad").value)
        self.cg_iters = int(self.get_parameter("cg_iters").value)
        self.cg_tol = float(self.get_parameter("cg_tol").value)
        self.pos_vmax = float(self.get_parameter("pos_vmax").value)
        self.pos_amax = float(self.get_parameter("pos_amax").value)
        self.ang_vmax = float(self.get_parameter("ang_vmax").value)
        self.ang_amax = float(self.get_parameter("ang_amax").value)
        self.pos_jmax = float(self.get_parameter("pos_jmax").value)
        self.ang_jmax = float(self.get_parameter("ang_jmax").value)
        self.image_dataset_name = str(self.get_parameter("image_dataset_name").value)
        self.image_compression = str(self.get_parameter("image_compression").value).lower()
        self.image_gzip_level = int(self.get_parameter("image_gzip_level").value)
        self.image_preprocess_mode = str(self.get_parameter("image_preprocess_mode").value).strip().lower()
        self.image_specular_mask_mode = str(self.get_parameter("image_specular_mask_mode").value).strip().lower()
        if self.image_specular_mask_mode not in ("white", "bright", "value", "v", "mixed", "white_or_bright"):
            self.get_logger().warn(f"[IMAGE] unknown image_specular_mask_mode={self.image_specular_mask_mode}; using white")
            self.image_specular_mask_mode = "white"
        self.image_specular_v_thresh = int(self.get_parameter("image_specular_v_thresh").value)
        self.image_specular_s_thresh = int(self.get_parameter("image_specular_s_thresh").value)
        self.image_specular_dilate_px = int(self.get_parameter("image_specular_dilate_px").value)
        self.image_specular_inpaint_radius = float(self.get_parameter("image_specular_inpaint_radius").value)
        self.image_specular_attenuate_gain = float(self.get_parameter("image_specular_attenuate_gain").value)
        if self.image_preprocess_mode not in ("", "raw", "none", "off", "specular_inpaint", "deglare", "highlight_inpaint", "highlight_attenuate", "specular_attenuate"):
            self.get_logger().warn(f"[IMAGE] unknown image_preprocess_mode={self.image_preprocess_mode}; using raw")
            self.image_preprocess_mode = "raw"
        if self.image_preprocess_mode not in ("", "raw", "none", "off") and cv2 is None:
            self.get_logger().warn("[IMAGE] cv2 is not available; image preprocessing will fall back to raw RGB")

        # HDF5 setup
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.save_root = os.path.join(self.act_root_dir, self.obs_mode, self.timestamp)
        self.merged_dir = os.path.join(self.save_root, self.merged_subdir)
        os.makedirs(self.merged_dir, exist_ok=True)
        self.h5_path = os.path.join(self.merged_dir, f"{self.file_prefix}_{self.timestamp}.hdf5")
        if os.path.exists(self.h5_path) and not self.overwrite_file:
            raise RuntimeError(f"HDF5 file already exists: {self.h5_path}. Set overwrite_file:=true to overwrite.")

        self.h5_lock = threading.Lock()
        self.h5 = h5py.File(self.h5_path, "w")
        self.h5.attrs["created_unix"] = float(time.time())
        self.h5.attrs["created_time"] = str(datetime.now().isoformat())
        self.h5.attrs["recorder"] = str(self.recorder_name)
        self.h5.attrs["schema_version"] = "multimodal_v1"
        self.h5.attrs["obs_mode"] = str(self.obs_mode)
        self.h5.attrs["recording_mode"] = str(self.recording_mode)
        self.h5.attrs["pose_topic"] = str(self.pose_topic)
        self.h5.attrs["force_topic"] = str(self.force_topic)
        self.h5.attrs["force_msg_type"] = str(self.force_msg_type)
        self.h5.attrs["image_topic"] = str(self.image_topic)
        self.h5.attrs["global_image_topic"] = str(self.global_image_topic)
        self.h5.attrs["image_preprocess_mode"] = str(self.image_preprocess_mode)
        self.h5.attrs["image_specular_mask_mode"] = str(self.image_specular_mask_mode)
        self.h5.attrs["image_specular_v_thresh"] = int(self.image_specular_v_thresh)
        self.h5.attrs["image_specular_s_thresh"] = int(self.image_specular_s_thresh)
        self.h5.attrs["image_specular_dilate_px"] = int(self.image_specular_dilate_px)
        self.h5.attrs["image_specular_inpaint_radius"] = float(self.image_specular_inpaint_radius)
        self.h5.attrs["image_specular_attenuate_gain"] = float(self.image_specular_attenuate_gain)
        self.h5.attrs["trajectory_filter_source"] = "stage1_vr_filtering_pipeline"
        self.h5.attrs["trajectory_filter_reference_hz"] = float(self.filter_reference_hz)
        self.h5.attrs["trajectory_scale_filter_params_with_hz"] = int(bool(self.scale_filter_params_with_hz))
        self.h5.attrs["trajectory_force_filter_mode"] = str(self.force_filter_mode)
        self.h5.attrs["trajectory_zero_xy_forces"] = int(bool(self.zero_xy_forces))
        self.h5.attrs["trajectory_force_ema_alpha"] = float(self.force_ema_alpha)
        self.h5.attrs["trajectory_retime_k"] = int(self.retime_k)
        self.h5.attrs["trajectory_approach_slowdown_enable"] = int(bool(self.approach_slowdown_enable))
        self.h5.attrs["trajectory_pose_ema_enable"] = int(bool(self.pose_ema_enable))
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

        self.episode_active = False
        self.finishing = False
        self.stop_requested = False
        self.current_ep_idx = 0
        self.saved_indices: Set[int] = set()

        self.P_buf: List[np.ndarray] = []
        self.F_buf: List[np.ndarray] = []
        self.I0_buf: List[Optional[np.ndarray]] = []
        self.I1_buf: List[Optional[np.ndarray]] = []
        self.sample_time_buf: List[float] = []
        self.last_status_t = 0.0
        self.last_idle_status_t = 0.0

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
        self.create_subscription(String, self.command_topic, self._on_command, reliable_qos)
        self.timer = self.create_timer(self.dt, self._on_sample_timer)

        atexit.register(self._atexit_close)

        self.get_logger().info(block(f"{self.recorder_name} READY", [
            ("h5_path", self.h5_path),
            ("obs_mode", self.obs_mode),
            ("mode", self.recording_mode),
            ("pose_topic", self.pose_topic),
            ("force_topic", f"{self.force_topic} ({self.force_msg_type})"),
            ("cam0", f"{self.image_topic} -> images/{self.image_dataset_name}"),
            ("cam1", f"{int(self.enable_global_cam)} {self.global_image_topic} -> images/{self.global_image_dataset_name}"),
            ("image_preprocess", f"{self.image_preprocess_mode}/{self.image_specular_mask_mode}"),
            ("sample_hz", self.sample_hz),
            ("command", self.command_topic),
        ]))
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

    def _on_command(self, msg: String):
        cmd = str(msg.data).strip().lower()
        if not cmd:
            return
        self.get_logger().warn(f"[COMMAND] {cmd}")
        if cmd == "start_recording":
            self.start_episode(reason="joystick_start")
        elif cmd == "end_recording":
            self.end_episode(reason="joystick_end")
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
        if (not self.allow_overwrite_episode) and (self.current_ep_idx in self.saved_indices):
            self.get_logger().warn(f"Episode index {self.current_ep_idx} already exists. Set allow_overwrite_episode:=true to overwrite.")
            return

        for buf in [self.P_buf, self.F_buf, self.I0_buf, self.I1_buf, self.sample_time_buf]:
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
            list(self.sample_time_buf), reason,
        )
        self.get_logger().warn(f"[EP {self._ep_name(ep_idx)}] END requested. raw_samples={len(self.P_buf)}, reason={reason}")
        threading.Thread(target=self._finish_episode_worker, args=args, daemon=True).start()

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
            pose_age = now - self.latest_pose_t if self.latest_pose_t > 0 else 1e9
            force_age = now - self.latest_force_t if self.latest_force_t > 0 else 1e9
            image_age = now - self.latest_image_t if self.latest_image_t > 0 else 1e9
            global_age = now - self.latest_global_image_t if self.latest_global_image_t > 0 else 1e9

        missing = []
        if pose is None or pose_age > self.require_pose_fresh_sec:
            missing.append(f"pose(age={pose_age:.3f})")
        if force is None or force_age > self.require_force_fresh_sec:
            missing.append(f"force(age={force_age:.3f})")
        if image is None or image_age > self.require_image_fresh_sec:
            missing.append(f"cam0(age={image_age:.3f})")
        if self.enable_global_cam and self.require_global_image and (global_image is None or global_age > self.require_global_image_fresh_sec):
            missing.append(f"cam1/global(age={global_age:.3f})")

        if missing:
            if now - self.last_status_t >= max(0.5, self.recording_status_period_sec):
                self.last_status_t = now
                self.get_logger().warn("[WAIT] " + ", ".join(missing))
            return

        image = preprocess_rgb_image(
            image,
            mode=self.image_preprocess_mode,
            specular_mask_mode=self.image_specular_mask_mode,
            specular_v_thresh=self.image_specular_v_thresh,
            specular_s_thresh=self.image_specular_s_thresh,
            specular_dilate_px=self.image_specular_dilate_px,
            specular_inpaint_radius=self.image_specular_inpaint_radius,
            specular_attenuate_gain=self.image_specular_attenuate_gain,
        )
        if self.enable_global_cam and global_image is not None:
            global_image = preprocess_rgb_image(
                global_image,
                mode=self.image_preprocess_mode,
                specular_mask_mode=self.image_specular_mask_mode,
                specular_v_thresh=self.image_specular_v_thresh,
                specular_s_thresh=self.image_specular_s_thresh,
                specular_dilate_px=self.image_specular_dilate_px,
                specular_inpaint_radius=self.image_specular_inpaint_radius,
                specular_attenuate_gain=self.image_specular_attenuate_gain,
            )

        self.P_buf.append(pose.astype(np.float32))
        self.F_buf.append(force[:3].astype(np.float32))
        self.I0_buf.append(image)
        self.I1_buf.append(global_image if self.enable_global_cam else None)
        self.sample_time_buf.append(float(now))

        if now - self.last_status_t >= self.recording_status_period_sec:
            self.last_status_t = now
            self._print_status("RECORDING")

    # save worker
    def _finish_episode_worker(self, ep_idx, P_list, F_list, I0_list, I1_list, sample_time_list, reason):
        try:
            N = len(P_list)
            if N < max(1, self.min_samples):
                self.get_logger().warn(f"Episode dropped: raw_len={N} < min_samples={self.min_samples}, reason={reason}")
                return

            P = np.asarray(P_list, dtype=np.float32).reshape(N, 6)
            Fraw = np.asarray(F_list, dtype=np.float32).reshape(N, 3)
            raw_sample_times = np.asarray(sample_time_list, dtype=np.float64)

            images0 = stack_images_repeat_last(I0_list, logger=self.get_logger(), tag="IMAGE/cam0")
            if images0 is None:
                self.get_logger().warn(f"Episode dropped: no valid cam0 frames. N={N}, reason={reason}")
                return
            images1 = stack_images_repeat_last(I1_list, logger=self.get_logger(), tag="IMAGE/cam1_global") if self.enable_global_cam else None
            if self.enable_global_cam and images1 is None:
                self.get_logger().warn("[IMAGE/cam1_global] no valid global frames. Saving cam0 only for this episode.")

            filter_result = apply_stage1_filter(
                P,
                Fraw,
                stage1_config_from_recorder(self, self.sample_hz),
                sample_times=raw_sample_times,
                logger=self.get_logger(),
            )
            P_out = filter_result.position
            F_out = filter_result.force
            sample_times_out = filter_result.sample_time
            if sample_times_out is None:
                sample_times_out = raw_sample_times
            images0_out = take_nearest_by_source_index(images0, filter_result.source_index)
            images1_out = (
                take_nearest_by_source_index(images1, filter_result.source_index)
                if images1 is not None else None
            )
            self._save_first_episode_filter_plots(ep_idx, P, Fraw, P_out, F_out)

            self._save_episode_to_hdf5(
                ep_idx,
                P_out,
                F_out,
                images0_out,
                images1_out,
                sample_times_out,
                reason,
                raw_position=P,
                raw_ft=Fraw,
                raw_sample_times=raw_sample_times,
                source_index=filter_result.source_index,
                filter_meta=filter_result.meta,
            )
            self.saved_indices.add(ep_idx)
            if ep_idx == self.current_ep_idx:
                self.current_ep_idx += 1

            self.get_logger().info(block("EPISODE SAVED", [
                ("episode", self._ep_name(ep_idx)),
                ("samples", f"{N} -> {P_out.shape[0]}"),
                ("position", P_out.shape),
                ("ft", F_out.shape),
                ("cam0", images0_out.shape),
                ("cam1", None if images1_out is None else images1_out.shape),
                ("reason", reason),
            ]))
            self._print_status("SAVED")
        except Exception as e:
            self.get_logger().error(f"Episode processing failed: {repr(e)}")
        finally:
            self.finishing = False
            if self.stop_requested and not self.episode_active:
                self.finalize_and_shutdown()

    def _save_first_episode_filter_plots(self, ep_idx, raw_position, raw_ft, position, ft):
        if int(ep_idx) != 0:
            return
        try:
            save_plot_1_lin_kinematics(self.merged_dir, self.dt, raw_position, position)
            save_plot_2_rotvec_kinematics(self.merged_dir, self.dt, raw_position, position)
            save_plot_3_forces(self.merged_dir, self.dt, raw_ft, ft)
            self.get_logger().info(f"[VIZ] Saved first-episode trajectory plots to: {self.merged_dir}")
        except Exception as e:
            self.get_logger().error(f"[VIZ] Failed to save first-episode trajectory plots: {e}")

    def _compression_kwargs(self) -> Dict[str, object]:
        mode = str(self.image_compression).lower()
        if mode == "gzip":
            return dict(compression="gzip", compression_opts=int(self.image_gzip_level), shuffle=True)
        if mode == "lzf":
            return dict(compression="lzf", shuffle=True)
        return {}

    def _save_episode_to_hdf5(
        self,
        ep_idx,
        position,
        ft,
        images0,
        images1,
        sample_times,
        reason,
        raw_position=None,
        raw_ft=None,
        raw_sample_times=None,
        source_index=None,
        filter_meta=None,
    ):
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
                g.attrs["raw_len"] = int(raw_position.shape[0]) if raw_position is not None else int(position.shape[0])
                g.attrs["out_len"] = int(position.shape[0])
                g.attrs["record_hz"] = float(self.sample_hz)
                g.attrs["dt"] = float(self.dt)
                g.attrs["recording_mode"] = str(self.recording_mode)
                g.attrs["obs_mode"] = str(self.obs_mode)
                g.attrs["pose_topic"] = str(self.pose_topic)
                g.attrs["force_topic"] = str(self.force_topic)
                g.attrs["force_msg_type"] = str(self.force_msg_type)
                g.attrs["cam0_topic"] = str(self.image_topic)
                g.attrs["cam1_topic"] = str(self.global_image_topic)
                g.attrs["schema_version"] = "multimodal_stage1_filtered_v2"
                if filter_meta:
                    for key, value in filter_meta.items():
                        if isinstance(value, (str, int, float, np.integer, np.floating)):
                            g.attrs[str(key)] = value

                g.create_dataset("position", data=position.astype(np.float32), compression="gzip", compression_opts=4, shuffle=True)
                g.create_dataset("ft", data=ft.astype(np.float32), compression="gzip", compression_opts=4, shuffle=True)
                g.create_dataset("sample_time_unix", data=sample_times.astype(np.float64), compression="gzip", compression_opts=4, shuffle=True)
                if source_index is not None:
                    g.create_dataset("source_index", data=np.asarray(source_index, dtype=np.float32), compression="gzip", compression_opts=4, shuffle=True)
                if raw_position is not None:
                    g.create_dataset("raw_position", data=np.asarray(raw_position, dtype=np.float32), compression="gzip", compression_opts=4, shuffle=True)
                if raw_ft is not None:
                    g.create_dataset("raw_ft", data=np.asarray(raw_ft, dtype=np.float32), compression="gzip", compression_opts=4, shuffle=True)
                if raw_sample_times is not None:
                    g.create_dataset("raw_sample_time_unix", data=np.asarray(raw_sample_times, dtype=np.float64), compression="gzip", compression_opts=4, shuffle=True)

                g_img = g.create_group("images")
                g_img.create_dataset(self.image_dataset_name, data=images0.astype(np.uint8), **self._compression_kwargs())
                if images1 is not None:
                    g_img.create_dataset(self.global_image_dataset_name, data=images1.astype(np.uint8), **self._compression_kwargs())

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

    def _print_status(self, tag: str):
        with self.state_lock:
            pose_ok = self.latest_pose is not None
            force_ok = self.latest_force is not None
            img0_ok = self.latest_image is not None
            img1_ok = self.latest_global_image is not None
        self.get_logger().info(status(tag, [
            ("ep", self._ep_name(self.current_ep_idx)),
            ("active", int(self.episode_active)),
            ("saving", int(self.finishing)),
            ("samples", len(self.P_buf)),
            ("saved", sorted(self.saved_indices)),
            ("pose", int(pose_ok)),
            ("force", int(force_ok)),
            ("cam0", int(img0_ok)),
            ("cam1", int(img1_ok)),
        ]))

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


def spin_recorder(node_name: str, fixed_defaults: Optional[Dict[str, object]] = None, args=None):
    rclpy.init(args=args)
    node = HDF5Recorder(node_name=node_name, fixed_defaults=fixed_defaults)
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
    spin_recorder("hdf5_recorder_base")
