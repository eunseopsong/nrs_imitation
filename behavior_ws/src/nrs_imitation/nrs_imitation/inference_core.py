#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_core.py
(shared implementation for single/dual camera inference)

Current training-side architecture:
  - position encoder for [x y z wx wy wz]
  - force encoder (GRU) for force history [fx fy fz]_{t-L+1:t}
  - fusion encoder
  - image encoder + ACT

  
This node keeps the ROS topic interface mostly unchanged, but updates inference-side
preprocessing to match the new single-camera training structure:

  qpos current  : [x y z wx wy wz fx fy fz]
  force_history : recent L-step force history (online buffer), normalized using
                  the same qpos force statistics as training dataset.py

Stages / safety logic are kept the same as the previous version. Only the image input is changed from two cameras to one camera (cam0).

Usage:

Default recommended Flow Matching inference:

    cd ~/nrs_imitation/behavior_ws
    source install/setup.bash
    ros2 run nrs_imitation inference_single_cam

This default run is equivalent to the recommended Flow baseline. If ckpt_dir is
not provided, the node automatically selects the newest timestamped checkpoint
folder under:

    ~/nrs_imitation/checkpoints/flow/polishing/single_cam/

You can still override any parameter with --ros-args -p name:=value.

"""


import os
import sys
import time
import csv
import math
import pickle
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Deque, List, Tuple
from enum import Enum

import numpy as np
import torch

try:
    import cv2
except Exception:
    cv2 = None

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import Wrench
from std_msgs.msg import Float32, Float64MultiArray, Int32
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from y2_rob_motion_interfaces.srv import SingleArmCommand

try:
    # Optional: only needed for stain-relative-frame checkpoints
    # (use_relative_position=True). Absolute-frame runs never touch it.
    from stain_relative_frame.inference_adapter import StainOriginClient
except Exception:  # noqa: BLE001 -- package may not be on the path
    StainOriginClient = None


DEFAULT_ACT_ROOT = os.path.expanduser("~/nrs_imitation")


# ============================================================
# Helpers (QoS / time / math)
# ============================================================

class _NumpyCompatUnpickler(pickle.Unpickler):
    """Load NumPy-2 pickles on NumPy-1 ROS environments."""

    def find_class(self, module, name):
        if module == "numpy._core":
            module = "numpy.core"
        elif str(module).startswith("numpy._core."):
            module = "numpy.core." + str(module)[len("numpy._core."):]
        return super().find_class(module, name)


def _pickle_load_compat(path: str):
    with open(path, "rb") as f:
        return _NumpyCompatUnpickler(f).load()


def _monotonic() -> float:
    return time.monotonic()


def _reliability_from_str(s: str) -> ReliabilityPolicy:
    s = (s or "").strip().lower()
    if s in ["reliable", "rel"]:
        return ReliabilityPolicy.RELIABLE
    if s in ["best_effort", "besteffort", "best"]:
        return ReliabilityPolicy.BEST_EFFORT
    return ReliabilityPolicy.BEST_EFFORT


def _qos(depth: int, reliability: ReliabilityPolicy) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=reliability,
        durability=DurabilityPolicy.VOLATILE,
    )


def _exp_decay_weight(age_steps: int, tau_steps: float) -> float:
    if tau_steps <= 1e-9:
        return 1.0
    age_steps = max(0, int(age_steps))
    return float(math.exp(-float(age_steps) / float(tau_steps)))


def _beta_from_tau(dt: float, tau: float) -> float:
    if tau <= 1e-9:
        return 1.0
    return float(1.0 - math.exp(-float(dt) / float(tau)))


def _start_pose_envelope_violation(
    cmd_xyz: np.ndarray,
    start_xyz: np.ndarray,
    *,
    max_xy_mm: float,
    max_z_down_mm: float,
    max_z_up_mm: float,
) -> Tuple[str, np.ndarray, float]:
    """Return violation reason plus start-relative delta/XY radius."""
    cmd = np.asarray(cmd_xyz, dtype=np.float32).reshape(-1)
    start = np.asarray(start_xyz, dtype=np.float32).reshape(-1)
    if cmd.size < 3 or start.size < 3:
        return "invalid start-envelope xyz", np.zeros(3, dtype=np.float32), 0.0

    delta = (cmd[:3] - start[:3]).astype(np.float32)
    xy_radius = float(np.linalg.norm(delta[:2]))
    dz = float(delta[2])

    if max_xy_mm > 0.0 and xy_radius > float(max_xy_mm):
        return (
            f"start-relative XY radius {xy_radius:.3f}mm exceeds {float(max_xy_mm):.3f}mm",
            delta,
            xy_radius,
        )
    if max_z_down_mm > 0.0 and dz < -float(max_z_down_mm):
        return (
            f"start-relative Z down {-dz:.3f}mm exceeds {float(max_z_down_mm):.3f}mm",
            delta,
            xy_radius,
        )
    if max_z_up_mm > 0.0 and dz > float(max_z_up_mm):
        return (
            f"start-relative Z up {dz:.3f}mm exceeds {float(max_z_up_mm):.3f}mm",
            delta,
            xy_radius,
        )
    return "", delta, xy_radius


# ============================================================
# Helpers (checkpoint auto-discovery)
# ============================================================

def _policy_to_ckpt_subdir(policy_class: str) -> str:
    p = str(policy_class or "FLOW").strip().upper()
    if p == "ACT":
        return "act"
    if p == "DIFFUSION":
        return "diffusion"
    if p == "BSPLINE":
        return "bspline"
    return "flow"


def _is_timestamp_like_dirname(name: str) -> bool:
    s = str(name).strip()
    compact = s.replace("_", "")
    return compact.isdigit() and len(compact) in (8, 12, 14)


def _find_latest_checkpoint_dir(root_dir: str) -> Optional[str]:
    """
    Return the newest child directory that contains policy_best.ckpt.

    Priority:
      1) timestamp-like directory name, lexicographically latest
      2) directory modification time

    Expected Flow layout:
      checkpoints/flow/polishing/<single_cam|dual_cam>/YYYYMMDD_HHMM/policy_best.ckpt
    """
    root_dir = os.path.expanduser(str(root_dir))
    if not os.path.isdir(root_dir):
        return None

    candidates = []
    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        if not os.path.isdir(path):
            continue
        ckpt = os.path.join(path, "policy_best.ckpt")
        if not os.path.exists(ckpt):
            continue
        timestamp_bonus = 1 if _is_timestamp_like_dirname(name) else 0
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            mtime = 0.0
        candidates.append((timestamp_bonus, name, mtime, path))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return candidates[0][3]


def _resolve_checkpoint_dir(ckpt_dir: str, act_root: str, policy_class: str, ckpt_auto_subdir: str = "polishing") -> str:
    """
    Resolve the checkpoint directory used by inference.

    Cases:
      - ckpt_dir points directly to a checkpoint leaf containing policy_best.ckpt
      - ckpt_dir points to a checkpoint root containing timestamp folders
      - ckpt_dir is empty: auto-select latest folder under
        <act_root>/checkpoints/<policy_subdir>/<ckpt_auto_subdir>
    """
    ckpt_dir = os.path.expanduser(str(ckpt_dir or "").strip())
    act_root = os.path.expanduser(str(act_root or "").strip())

    if ckpt_dir:
        if os.path.isdir(ckpt_dir) and os.path.exists(os.path.join(ckpt_dir, "policy_best.ckpt")):
            return ckpt_dir
        latest = _find_latest_checkpoint_dir(ckpt_dir)
        if latest is not None:
            return latest
        return ckpt_dir

    subdir = _policy_to_ckpt_subdir(policy_class)
    leaf = str(ckpt_auto_subdir or "polishing").strip() or "polishing"
    root = os.path.join(act_root, "checkpoints", subdir, leaf)
    latest = _find_latest_checkpoint_dir(root)
    if latest is None:
        raise RuntimeError(
            "ckpt_dir was not provided and no usable checkpoint folder was found under: "
            f"{root} (expected */policy_best.ckpt)"
        )
    return latest


# ============================================================
# Helpers (Image decode)
# ============================================================

def _img_to_rgb_numpy(msg: Image) -> np.ndarray:
    """
    Convert sensor_msgs/Image -> np.uint8 (H,W,3) RGB
    Supports: rgb8, bgr8, rgba8, bgra8
    """
    h, w = int(msg.height), int(msg.width)
    enc = (msg.encoding or "").lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc == "rgb8":
        return buf.reshape((h, w, 3))
    if enc == "bgr8":
        img = buf.reshape((h, w, 3))
        return img[..., ::-1].copy()
    if enc == "rgba8":
        return buf.reshape((h, w, 4))[..., :3]
    if enc == "bgra8":
        img = buf.reshape((h, w, 4))[..., :3]
        return img[..., ::-1].copy()

    try:
        return buf.reshape((h, w, 3))
    except Exception as e:
        raise RuntimeError(f"Unsupported image encoding={msg.encoding}, size=({h},{w}), err={e}")


def _to_tensor_image_stack(
    images_rgb: List[np.ndarray],
    device: torch.device,
    resize_hw: int = 0,
    camera_names: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    [(H,W,3), ...] -> (1,K,3,H,W) float in [0,1]
    """
    camera_names = list(camera_names or [])
    if not images_rgb:
        raise RuntimeError("image stack is empty")

    chw_images = []
    for idx, img_rgb in enumerate(images_rgb):
        cam = camera_names[idx] if idx < len(camera_names) else f"cam{idx}"
        if img_rgb is None:
            raise RuntimeError(f"{cam} image is None")
        if resize_hw and resize_hw > 0:
            try:
                import cv2
                img_rgb = cv2.resize(img_rgb, (resize_hw, resize_hw), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                raise RuntimeError(f"cv2 resize failed for {cam} (resize_hw={resize_hw}): {e}")
        chw_images.append(np.transpose(img_rgb, (2, 0, 1)))

    img = np.stack(chw_images, axis=0).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1,K,3,H,W)
    return img_t


def _mask_msg_to_float_numpy(msg: Image) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    enc = (msg.encoding or "").lower()
    if enc in ("mono16", "16uc1"):
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape((h, w)).astype(np.float32)
        mx = float(arr.max()) if arr.size else 0.0
        return arr / max(mx, 1.0) if mx > 1.5 else arr
    if enc in ("32fc1",):
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape((h, w)).astype(np.float32)
        return np.clip(arr, 0.0, 1.0)
    if enc in ("rgb8", "bgr8", "rgba8", "bgra8"):
        rgb = _img_to_rgb_numpy(msg)
        arr = rgb[..., 0].astype(np.float32)
    else:
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        try:
            arr = arr.reshape((h, w)).astype(np.float32)
        except Exception:
            arr = arr.reshape((h, w, -1))[..., 0].astype(np.float32)
    if float(arr.max()) > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _to_tensor_stain_mask(mask: np.ndarray, device: torch.device, resize_hw: int = 0) -> torch.Tensor:
    if mask is None:
        raise RuntimeError("stain_mask is None")
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    if m.ndim != 2:
        raise RuntimeError(f"stain_mask must be 2D, got {m.shape}")
    if resize_hw and resize_hw > 0:
        try:
            import cv2
            m = cv2.resize(m, (resize_hw, resize_hw), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            raise RuntimeError(f"cv2 resize failed for stain_mask (resize_hw={resize_hw}): {e}")
    m = np.clip(m, 0.0, 1.0)[None, None, ...]
    return torch.from_numpy(m).to(device=device, dtype=torch.float32)


def _parse_camera_names(value, obs_mode: str) -> List[str]:
    obs = str(obs_mode or "single_cam").strip().lower()
    default = ["cam0", "cam1"] if obs == "dual_cam" else ["cam0"]
    if value is None:
        return default
    raw = []
    if isinstance(value, str):
        s = value.strip()
        if s == "" or s.lower() == "auto":
            return default
        s = s.strip("[]")
        raw = [p.strip().strip("'\"") for p in s.split(",")]
    else:
        try:
            for item in list(value):
                for part in str(item).split(","):
                    raw.append(part.strip().strip("'\""))
        except Exception:
            return default
    out = [x for x in raw if x]
    return out if out else default




# ============================================================
# Helpers (online camera stabilization / jitter diagnostics)
# ============================================================
def _estimate_pair_transform(prev_gray: np.ndarray, curr_gray: np.ndarray):
    if cv2 is None:
        return 0.0, 0.0, 0.0

    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=20,
        blockSize=3,
    )
    if prev_pts is None or len(prev_pts) < 8:
        return 0.0, 0.0, 0.0

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    if curr_pts is None or status is None:
        return 0.0, 0.0, 0.0

    good_prev = prev_pts[status.flatten() == 1]
    good_curr = curr_pts[status.flatten() == 1]
    if len(good_prev) < 8 or len(good_curr) < 8:
        return 0.0, 0.0, 0.0

    m, _ = cv2.estimateAffinePartial2D(good_prev, good_curr, method=cv2.RANSAC)
    if m is None:
        return 0.0, 0.0, 0.0

    dx = float(m[0, 2])
    dy = float(m[1, 2])
    da = float(np.arctan2(m[1, 0], m[0, 0]))
    return dx, dy, da


def _warp_rgb_affine(rgb: np.ndarray, dx: float, dy: float, da: float, border_mode: str = "reflect") -> np.ndarray:
    if cv2 is None:
        return rgb.copy()
    H, W = int(rgb.shape[0]), int(rgb.shape[1])
    c = float(np.cos(da))
    s = float(np.sin(da))
    m = np.array([[c, -s, dx], [s, c, dy]], dtype=np.float32)
    b = str(border_mode).strip().lower()
    if b == "constant":
        border_flag = cv2.BORDER_CONSTANT
    elif b == "replicate":
        border_flag = cv2.BORDER_REPLICATE
    else:
        border_flag = cv2.BORDER_REFLECT
    return cv2.warpAffine(
        rgb,
        m,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=border_flag,
    )

# ============================================================
# Helpers (Grad-CAM debug visualization)
# ============================================================

def _normalize_heatmap_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.nanmin(x))
    den = float(np.nanmax(x)) + float(eps)
    return np.clip(x / den, 0.0, 1.0).astype(np.float32)


def _rgb_numpy_to_image_msg(rgb: np.ndarray, stamp=None, frame_id: str = "") -> Image:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
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


def _make_gradcam_overlay_rgb(rgb: np.ndarray, heatmap01: np.ndarray, alpha: float = 0.45, colormap: str = "jet") -> np.ndarray:
    rgb_u8 = np.asarray(rgb)
    if rgb_u8.dtype != np.uint8:
        rgb_u8 = np.clip(rgb_u8, 0, 255).astype(np.uint8)
    H, W = int(rgb_u8.shape[0]), int(rgb_u8.shape[1])
    hm = _normalize_heatmap_np(heatmap01)
    if hm.shape[0] != H or hm.shape[1] != W:
        if cv2 is not None:
            hm = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            yy = (np.linspace(0, hm.shape[0] - 1, H)).astype(np.int32)
            xx = (np.linspace(0, hm.shape[1] - 1, W)).astype(np.int32)
            hm = hm[yy[:, None], xx[None, :]]
    a = float(np.clip(alpha, 0.0, 1.0))
    hm_u8 = np.clip(255.0 * hm, 0, 255).astype(np.uint8)
    if cv2 is not None:
        cm_name = str(colormap or "jet").strip().lower()
        cmap = cv2.COLORMAP_JET
        if cm_name in ("turbo",):
            cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
        elif cm_name in ("hot",):
            cmap = cv2.COLORMAP_HOT
        elif cm_name in ("viridis",):
            cmap = getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET)
        heat_bgr = cv2.applyColorMap(hm_u8, cmap)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    else:
        heat_rgb = np.zeros_like(rgb_u8, dtype=np.uint8)
        heat_rgb[..., 0] = hm_u8
    overlay = ((1.0 - a) * rgb_u8.astype(np.float32) + a * heat_rgb.astype(np.float32))
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _find_module_by_name(root: torch.nn.Module, name: str) -> Optional[torch.nn.Module]:
    name = str(name or "").strip()
    if not name:
        return None
    for n, m in root.named_modules():
        if n == name:
            return m
    return None


def _find_last_conv2d(root: torch.nn.Module):
    last_name = None
    last_module = None
    for n, m in root.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last_name = n
            last_module = m
    return last_name, last_module


# ============================================================
# Helpers (Stats)
# ============================================================

@dataclass
class StatsPack:
    qpos_mode: str   # "minmax_01", "minmax_m11", or "zscore"
    act_mode: str    # "minmax_01", "minmax_m11", or "zscore"
    qpos_a: np.ndarray   # min or mean
    qpos_b: np.ndarray   # max or std
    act_a: np.ndarray    # min or mean
    act_b: np.ndarray    # max or std
    xyz_scale: float = 1.0
    gripper_mode: str = "minmax_01"
    gripper_position_a: Optional[np.ndarray] = None
    gripper_position_b: Optional[np.ndarray] = None
    gripper_current_a: Optional[np.ndarray] = None
    gripper_current_b: Optional[np.ndarray] = None


XYZ_STATS_ABS_MAX_MM = 10000.0


def _infer_xyz_stats_scale(*arrays: np.ndarray) -> float:
    """
    Inference commands and /ur10skku/currentP use mm. A UR10 workspace should
    not produce 10m+ xyz values; those stats are almost certainly um-like data
    produced by applying an extra x1000 during recording.
    """
    xyz = []
    for arr in arrays:
        a = np.asarray(arr, dtype=np.float32).reshape(-1)
        if a.size >= 3:
            xyz.append(a[:3])
    if not xyz:
        return 1.0

    vals = np.concatenate(xyz, axis=0)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return 1.0

    max_abs = float(np.max(np.abs(finite)))
    if max_abs > XYZ_STATS_ABS_MAX_MM:
        return 0.001
    return 1.0


def _scale_xyz_prefix(arr: np.ndarray, scale: float) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    if out.size >= 3 and abs(float(scale) - 1.0) > 1e-12:
        out[:3] *= np.float32(scale)
    return out


def _canonical_norm_mode(mode: str) -> str:
    if mode is None:
        return "minmax_01"
    m = str(mode).strip().lower()
    if m in ["minmax", "minmax_01", "01", "0_1", "[0,1]", "zero_one"]:
        return "minmax_01"
    if m in ["minmax_m11", "m11", "-1_1", "[-1,1]", "minus1_1", "neg1_pos1"]:
        return "minmax_m11"
    if m in ["zscore", "standard", "meanstd", "mean_std"]:
        return "zscore"
    return m


def _sanitize_std(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return np.maximum(x, eps)


def _sanitize_range_minmax(vmin: np.ndarray, vmax: np.ndarray, eps: float = 1e-6, expected_size: Optional[int] = 9):
    vmin = np.asarray(vmin, dtype=np.float32).reshape(-1)
    vmax = np.asarray(vmax, dtype=np.float32).reshape(-1)
    if expected_size is not None and (vmin.size != expected_size or vmax.size != expected_size):
        raise ValueError(f"min/max size must be {expected_size}. got {vmin.size}, {vmax.size}")
    if vmin.size != vmax.size:
        raise ValueError(f"min/max size mismatch. got {vmin.size}, {vmax.size}")
    rng = np.maximum(vmax - vmin, eps)
    vmax_fix = vmin + rng
    return vmin.astype(np.float32), vmax_fix.astype(np.float32)



def _load_demo_start_pose_from_stats(ckpt_dir: str, xyz_scale: float = 1.0) -> Optional[np.ndarray]:
    """
    Load demo_start_pose_mean from ckpt_dir/dataset_stats.pkl.

    Expected key added by the Flow training entrypoints:
      demo_start_pose_mean = [x, y, z, wx, wy, wz]

    Fallback:
      demo_start_qpos_mean = [x, y, z, wx, wy, wz, fx, fy, fz]

    Return:
      np.ndarray shape (6,), or None if unavailable.
    """
    p = os.path.join(ckpt_dir, "dataset_stats.pkl")
    if not os.path.exists(p):
        return None

    try:
        st = _pickle_load_compat(p)
    except Exception:
        return None

    for key in ("demo_start_pose_mean", "demo_start_qpos_mean"):
        if key not in st:
            continue
        arr = np.asarray(st[key], dtype=np.float32).reshape(-1)
        if arr.size >= 6 and np.all(np.isfinite(arr[:6])):
            scale = float(xyz_scale)
            if scale <= 0.0 or not np.isfinite(scale):
                scale = _infer_xyz_stats_scale(arr[:6])
            elif abs(scale - 1.0) <= 1e-12:
                scale = _infer_xyz_stats_scale(arr[:6])
            return _scale_xyz_prefix(arr[:6], scale)

    return None


def _load_relative_frame_flags(ckpt_dir: str, act_root: str = "") -> Tuple[bool, str, bool]:
    """Return (use_relative_position, relative_transform_version, obs_force_xy_zeroed).

    Written by stain_relative_frame/dataset_relativize.py into the converted
    dataset's dataset_stats.pkl and (for checkpoints trained after that change)
    carried forward into the checkpoint's own stats by
    flow_train_core.carry_forward_relative_frame_stats.

    For a checkpoint trained before the carry-forward existed, fall back to the
    source dataset's dataset_stats.pkl named in `dataset_dir`. Absolute-frame
    checkpoints carry none of these keys -> (False, "", False), the historical
    behaviour.
    """
    def _read(pkl_path: str) -> Optional[dict]:
        if not pkl_path or not os.path.exists(pkl_path):
            return None
        try:
            return _pickle_load_compat(pkl_path)
        except Exception:
            return None

    st = _read(os.path.join(ckpt_dir, "dataset_stats.pkl")) or {}
    src = st
    if "use_relative_position" not in st:
        ds_dir = str(st.get("dataset_dir", "") or "")
        candidates = [os.path.join(ds_dir, "dataset_stats.pkl")] if ds_dir else []
        if ds_dir and act_root and not os.path.isabs(ds_dir):
            candidates.append(os.path.join(act_root, ds_dir, "dataset_stats.pkl"))
        for cand in candidates:
            ds_stats = _read(cand)
            if ds_stats is not None and "use_relative_position" in ds_stats:
                src = ds_stats
                break

    use_relative = bool(src.get("use_relative_position", False))
    version = str(src.get("relative_transform_version", "") or "")
    obs_fxy_zeroed = bool(src.get("observation_force_xy_zeroed", False))
    return use_relative, version, obs_fxy_zeroed


def _wrap_axis_delta(theta: float) -> float:
    """Smallest rotation (rad) that aligns one line direction onto another.
    A stain strip is a line -> direction is mod pi, so the result is in
    (-pi/2, pi/2]."""
    return ((float(theta) + math.pi / 2.0) % math.pi) - math.pi / 2.0


def _load_dataset_stats(ckpt_dir: str) -> Optional[StatsPack]:
    """
    Priority:
      1) qpos_min/qpos_max/action_min/action_max with explicit norm mode
      2) qpos_mean/qpos_std/action_mean/action_std legacy zscore

    Backward compatibility:
      - old dataset_stats.pkl without qpos_norm_mode/action_norm_mode is treated as [0,1].
      - old mode name "minmax" is treated as "minmax_01".
    """
    p = os.path.join(ckpt_dir, "dataset_stats.pkl")
    if not os.path.exists(p):
        return None

    st = _pickle_load_compat(p)

    if all(k in st for k in ["qpos_min", "qpos_max", "action_min", "action_max"]):
        qmin = np.asarray(st["qpos_min"], dtype=np.float32).reshape(9)
        qmax = np.asarray(st["qpos_max"], dtype=np.float32).reshape(9)
        amin = np.asarray(st["action_min"], dtype=np.float32).reshape(-1)
        amax = np.asarray(st["action_max"], dtype=np.float32).reshape(-1)
        if amin.size not in (9, 10, 11) or amax.size != amin.size:
            raise ValueError(
                f"action min/max size must be 9, 10, or 11. got {amin.size}, {amax.size}"
            )

        xyz_scale = _infer_xyz_stats_scale(qmin, qmax, amin, amax)
        if abs(xyz_scale - 1.0) > 1e-12:
            qmin = _scale_xyz_prefix(qmin, xyz_scale)
            qmax = _scale_xyz_prefix(qmax, xyz_scale)
            amin = _scale_xyz_prefix(amin, xyz_scale)
            amax = _scale_xyz_prefix(amax, xyz_scale)

        qmin, qmax = _sanitize_range_minmax(qmin, qmax, expected_size=9)
        amin, amax = _sanitize_range_minmax(amin, amax, expected_size=amin.size)

        qmode = _canonical_norm_mode(
            st.get("qpos_norm_mode", st.get("qpos_mode", "minmax_01"))
        )
        amode = _canonical_norm_mode(
            st.get("action_norm_mode", st.get("act_mode", "minmax_01"))
        )

        # Legacy stats may store "minmax"; force it to [0,1].
        if qmode == "minmax":
            qmode = "minmax_01"
        if amode == "minmax":
            amode = "minmax_01"

        gripper_mode = _canonical_norm_mode(st.get("gripper_norm_mode", qmode))
        gpmin = None
        gpmax = None
        if "gripper_position_min" in st and "gripper_position_max" in st:
            gpmin, gpmax = _sanitize_range_minmax(
                np.asarray(st["gripper_position_min"], dtype=np.float32).reshape(1),
                np.asarray(st["gripper_position_max"], dtype=np.float32).reshape(1),
                expected_size=1,
            )

        gcmin = None
        gcmax = None
        if "gripper_current_min" in st and "gripper_current_max" in st:
            gcmin, gcmax = _sanitize_range_minmax(
                np.asarray(st["gripper_current_min"], dtype=np.float32).reshape(1),
                np.asarray(st["gripper_current_max"], dtype=np.float32).reshape(1),
                expected_size=1,
            )

        return StatsPack(
            qpos_mode=qmode,
            act_mode=amode,
            qpos_a=qmin,
            qpos_b=qmax,
            act_a=amin,
            act_b=amax,
            xyz_scale=xyz_scale,
            gripper_mode=gripper_mode,
            gripper_position_a=gpmin,
            gripper_position_b=gpmax,
            gripper_current_a=gcmin,
            gripper_current_b=gcmax,
        )

    if all(k in st for k in ["qpos_mean", "qpos_std", "action_mean", "action_std"]):
        qm = np.asarray(st["qpos_mean"], dtype=np.float32).reshape(9)
        qs = _sanitize_std(np.asarray(st["qpos_std"], dtype=np.float32).reshape(9))
        am = np.asarray(st["action_mean"], dtype=np.float32).reshape(-1)
        astd = _sanitize_std(np.asarray(st["action_std"], dtype=np.float32).reshape(-1))
        if am.size not in (9, 10, 11) or astd.size != am.size:
            raise ValueError(
                f"action mean/std size must be 9, 10, or 11. got {am.size}, {astd.size}"
            )

        xyz_scale = _infer_xyz_stats_scale(qm, am)
        if abs(xyz_scale - 1.0) > 1e-12:
            qm = _scale_xyz_prefix(qm, xyz_scale)
            qs = _scale_xyz_prefix(qs, xyz_scale)
            am = _scale_xyz_prefix(am, xyz_scale)
            astd = _scale_xyz_prefix(astd, xyz_scale)

        return StatsPack(
            qpos_mode="zscore",
            act_mode="zscore",
            qpos_a=qm,
            qpos_b=qs,
            act_a=am,
            act_b=astd,
            xyz_scale=xyz_scale,
        )

    return None


def _normalize_qpos(q: torch.Tensor, stats: StatsPack) -> torch.Tensor:
    qa = torch.tensor(stats.qpos_a, dtype=torch.float32, device=q.device).view(1, 9)
    qb = torch.tensor(stats.qpos_b, dtype=torch.float32, device=q.device).view(1, 9)

    if stats.qpos_mode in ["minmax_01", "minmax_m11"]:
        den = torch.clamp(qb - qa, min=1e-6)
        q01 = (q - qa) / den
        if stats.qpos_mode == "minmax_m11":
            return torch.clamp(2.0 * q01 - 1.0, -1.0, 1.0)
        return torch.clamp(q01, 0.0, 1.0)

    return (q - qa) / torch.clamp(qb, min=1e-6)


def _normalize_force_history(force_hist: torch.Tensor, stats: StatsPack) -> torch.Tensor:
    """
    force_hist: (1,L,3)
    Must use the same qpos force statistics as training dataset.py.
    force dims in qpos/action = indices [6:9].
    """
    if force_hist.dim() != 3 or force_hist.shape[-1] != 3:
        raise RuntimeError(f"force_hist must be (B,L,3), got {tuple(force_hist.shape)}")

    if stats.qpos_mode in ["minmax_01", "minmax_m11"]:
        fmin = torch.tensor(stats.qpos_a[6:9], dtype=torch.float32, device=force_hist.device).view(1, 1, 3)
        fmax = torch.tensor(stats.qpos_b[6:9], dtype=torch.float32, device=force_hist.device).view(1, 1, 3)
        den = torch.clamp(fmax - fmin, min=1e-6)
        f01 = (force_hist - fmin) / den
        if stats.qpos_mode == "minmax_m11":
            return torch.clamp(2.0 * f01 - 1.0, -1.0, 1.0)
        return torch.clamp(f01, 0.0, 1.0)

    fmean = torch.tensor(stats.qpos_a[6:9], dtype=torch.float32, device=force_hist.device).view(1, 1, 3)
    fstd = torch.tensor(stats.qpos_b[6:9], dtype=torch.float32, device=force_hist.device).view(1, 1, 3)
    return (force_hist - fmean) / torch.clamp(fstd, min=1e-6)


def _normalize_gripper_scalar(
    value: torch.Tensor,
    value_a: Optional[np.ndarray],
    value_b: Optional[np.ndarray],
    mode: str,
    stats_name: str,
) -> torch.Tensor:
    if value_a is None or value_b is None:
        raise RuntimeError(f"dataset_stats.pkl missing {stats_name}_min/{stats_name}_max")
    va = torch.tensor(value_a, dtype=torch.float32, device=value.device).view(1, 1)
    vb = torch.tensor(value_b, dtype=torch.float32, device=value.device).view(1, 1)
    den = torch.clamp(vb - va, min=1e-6)
    value01 = (value - va) / den
    if mode == "minmax_m11":
        return torch.clamp(2.0 * value01 - 1.0, -1.0, 1.0)
    if mode == "zscore":
        return (value - va) / den
    return torch.clamp(value01, 0.0, 1.0)


def _normalize_gripper_position(position: torch.Tensor, stats: StatsPack) -> torch.Tensor:
    return _normalize_gripper_scalar(
        position,
        stats.gripper_position_a,
        stats.gripper_position_b,
        stats.gripper_mode,
        "gripper_position",
    )


def _normalize_gripper_current(current: torch.Tensor, stats: StatsPack) -> torch.Tensor:
    return _normalize_gripper_scalar(
        current,
        stats.gripper_current_a,
        stats.gripper_current_b,
        stats.gripper_mode,
        "gripper_current",
    )


def _normalize_gripper_history(history: torch.Tensor, stats: StatsPack) -> torch.Tensor:
    if history.dim() != 3 or history.shape[-1] != 2:
        raise RuntimeError(
            f"gripper_history must be (B,L,2), got {tuple(history.shape)}"
        )
    if (
        stats.gripper_position_a is None
        or stats.gripper_position_b is None
        or stats.gripper_current_a is None
        or stats.gripper_current_b is None
    ):
        raise RuntimeError(
            "gripper history requires position/current min/max in dataset_stats.pkl"
        )
    value_min = torch.tensor(
        [stats.gripper_position_a[0], stats.gripper_current_a[0]],
        dtype=torch.float32,
        device=history.device,
    ).view(1, 1, 2)
    value_max = torch.tensor(
        [stats.gripper_position_b[0], stats.gripper_current_b[0]],
        dtype=torch.float32,
        device=history.device,
    ).view(1, 1, 2)
    value01 = (history - value_min) / torch.clamp(value_max - value_min, min=1e-6)
    if stats.gripper_mode == "minmax_m11":
        return torch.clamp(2.0 * value01 - 1.0, -1.0, 1.0)
    return torch.clamp(value01, 0.0, 1.0)


def _denorm_action_seq(seq: torch.Tensor, stats: StatsPack) -> torch.Tensor:
    action_dim = int(stats.act_a.size)
    if seq.shape[-1] != action_dim:
        raise RuntimeError(f"policy output action dim={seq.shape[-1]} does not match stats action_dim={action_dim}")
    if seq.dim() == 2:
        aa = torch.tensor(stats.act_a, dtype=torch.float32, device=seq.device).view(1, action_dim)
        ab = torch.tensor(stats.act_b, dtype=torch.float32, device=seq.device).view(1, action_dim)
    elif seq.dim() == 3:
        aa = torch.tensor(stats.act_a, dtype=torch.float32, device=seq.device).view(1, 1, action_dim)
        ab = torch.tensor(stats.act_b, dtype=torch.float32, device=seq.device).view(1, 1, action_dim)
    else:
        raise RuntimeError(f"unexpected seq dim: {seq.shape}")

    if stats.act_mode in ["minmax_01", "minmax_m11"]:
        den = torch.clamp(ab - aa, min=1e-6)
        if stats.act_mode == "minmax_m11":
            seq01 = 0.5 * (seq + 1.0)
            return seq01 * den + aa
        return seq * den + aa

    return seq * torch.clamp(ab, min=1e-6) + aa

# ============================================================
# Helpers (Policy output shape)
# ============================================================

def _fix_a_hat_shape(a_hat: torch.Tensor, chunk_size: int, action_dim: int = 9) -> torch.Tensor:
    """
    Standardize output to (T,D)
    Handles:
      - (1,T,D)
      - (T,1,D)
      - (T,D)
    """
    if a_hat.dim() == 2:
        if a_hat.shape[-1] != action_dim:
            raise RuntimeError(f"Unexpected 2D a_hat last dim (need {action_dim}): {a_hat.shape}")
        return a_hat
    if a_hat.dim() != 3:
        raise RuntimeError(f"Unexpected a_hat dim: {a_hat.shape}")

    B0, B1, B2 = a_hat.shape
    if B2 != action_dim:
        raise RuntimeError(f"Unexpected last dim (need {action_dim}): {a_hat.shape}")

    if B0 == 1 and B1 == chunk_size:
        return a_hat[0]
    if B0 == chunk_size and B1 == 1:
        return a_hat[:, 0, :]
    if B1 == chunk_size:
        return a_hat[0]
    raise RuntimeError(f"Cannot interpret a_hat shape={a_hat.shape} with chunk_size={chunk_size}")


def _fix_policy_output_seq(seq: torch.Tensor, chunk_size: int, policy_class: str, action_dim: int = 9) -> torch.Tensor:
    """
    Standardize ACT / DIFFUSION / FLOW policy output to (T,D).
    ACT:
      usually (1,T,D) or (T,D)
    DIFFUSION:
      usually (B,T,D) or (T,D)
    """
    if seq.dim() == 2:
        if seq.shape[-1] != action_dim:
            raise RuntimeError(f"Unexpected 2D seq shape: {tuple(seq.shape)}")
        return seq

    if seq.dim() != 3:
        raise RuntimeError(f"Unexpected policy output dim: {tuple(seq.shape)}")

    if seq.shape[-1] != action_dim:
        raise RuntimeError(f"Unexpected last dim in policy output: {tuple(seq.shape)}")

    policy_class = str(policy_class).upper()
    if policy_class == "DIFFUSION":
        if seq.shape[0] == 1:
            return seq[0]
        if seq.shape[1] == chunk_size:
            return seq[0]
        raise RuntimeError(f"Cannot interpret diffusion output shape={tuple(seq.shape)} with chunk_size={chunk_size}")

    return _fix_a_hat_shape(seq, chunk_size, action_dim=action_dim)


# ============================================================
# Plan buffer entry
# ============================================================

@dataclass
class Plan:
    t0: float
    seq_den: np.ndarray  # (T,9) polishing or (T,10) gripper, denorm
    local_anchor_applied: bool = False


# ============================================================
# Stage machine
# ============================================================

class Stage(Enum):
    APPROACH = 0
    PRELOAD = 1
    TRACK = 2
    RELEASE = 3
    RECOVER = 4  # deprecated / unused


# ============================================================
# State dict compatibility loader
# ============================================================

def _strip_prefix_from_state_dict(sd: dict, prefixes: List[str]) -> dict:
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def _try_load_state_dict_compat(target: torch.nn.Module, state_dict: dict):
    """
    Try several key transforms and pick the best (min missing+unexpected).
    """
    candidates = []
    candidates.append(("orig", state_dict))
    candidates.append(("strip_model.", _strip_prefix_from_state_dict(state_dict, ["model."])))
    candidates.append(("strip_module.", _strip_prefix_from_state_dict(state_dict, ["module."])))
    candidates.append(("strip_policy.", _strip_prefix_from_state_dict(state_dict, ["policy."])))
    candidates.append(("strip_model+module", _strip_prefix_from_state_dict(state_dict, ["module.", "model."])))
    candidates.append(("strip_policy+module", _strip_prefix_from_state_dict(state_dict, ["module.", "policy."])))

    best_missing = None
    best_unexpected = None
    best_score = None

    for _, sd in candidates:
        try:
            missing, unexpected = target.load_state_dict(sd, strict=False)
            score = len(missing) + len(unexpected)
            if (best_score is None) or (score < best_score):
                best_score = score
                best_missing = missing
                best_unexpected = unexpected
        except Exception:
            continue

    if best_missing is None:
        missing, unexpected = target.load_state_dict(state_dict, strict=False)
        return missing, unexpected

    return best_missing, best_unexpected


def _load_state_dict_strict_compat(target: torch.nn.Module, state_dict: dict) -> str:
    """Load an exact architecture while tolerating known wrapper prefixes."""
    candidates = [
        ("orig", state_dict),
        ("strip_model.", _strip_prefix_from_state_dict(state_dict, ["model."])),
        ("strip_module.", _strip_prefix_from_state_dict(state_dict, ["module."])),
        ("strip_policy.", _strip_prefix_from_state_dict(state_dict, ["policy."])),
        (
            "strip_model+module",
            _strip_prefix_from_state_dict(state_dict, ["module.", "model."]),
        ),
        (
            "strip_policy+module",
            _strip_prefix_from_state_dict(state_dict, ["module.", "policy."]),
        ),
    ]
    errors = []
    for name, candidate in candidates:
        try:
            target.load_state_dict(candidate, strict=True)
            return name
        except RuntimeError as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError(
        "checkpoint architecture does not exactly match the constructed policy; "
        "refusing strict=False fallback. " + " | ".join(errors[:2])
    )


# ============================================================
# ROS2 Node
# ============================================================

class NodeCmdMotionInfer(Node):
    def __init__(self, node_name: str = "inference_core"):
        super().__init__(node_name)

        # -----------------------------
        # Parameters (paths / IO)
        # -----------------------------
        self.declare_parameter("ckpt_dir", "")  # empty -> auto latest checkpoint
        self.declare_parameter("act_root", DEFAULT_ACT_ROOT)
        self.declare_parameter("policy_class", "FLOW")  # ACT | DIFFUSION | FLOW | BSPLINE
        self.declare_parameter("ckpt_auto_subdir", "polishing")

        # Optional per-run CSV metrics log, so runs on different policy_class
        # values (e.g. FLOW vs BSPLINE) can be compared offline afterwards.
        self.declare_parameter("metrics_log_enable", False)
        self.declare_parameter("metrics_log_dir", "")  # empty -> <act_root>/logs/inference_metrics
        self.declare_parameter("metrics_run_tag", "")
        self.declare_parameter("obs_mode", "single_cam")  # single_cam | dual_cam
        self.declare_parameter("camera_names", "auto")    # auto | "cam0" | "cam0,cam1"
        self.declare_parameter("phase_mode", "pure")  # kept for recommended Flow command compatibility
        self.declare_parameter("chunk_size", 200)

        self.declare_parameter("pose_topic", "/ur10skku/currentP")
        self.declare_parameter("force_topic", "/ur10skku/currentF")
        self.declare_parameter("force_msg_type", "array")  # array | wrench
        self.declare_parameter("image_topic", "/realsense/vr/color/image_raw")
        self.declare_parameter("global_image_topic", "/realsense/global/color/image_raw")
        self.declare_parameter("stain_mask_topic", "")
        self.declare_parameter("cmd_topic", "/ur10skku/cmdMotion")
        # Latched stain-origin topic published once per episode by
        # stain_relative_frame/stain_origin_node. Only consumed when the
        # checkpoint's dataset_stats.pkl says use_relative_position=True;
        # ignored (never subscribed) for an absolute-frame checkpoint.
        self.declare_parameter("stain_origin_topic", "/stain_relative_frame/stain_origin")
        # auto = follow the checkpoint's observation_force_xy_zeroed stat;
        # true/false override it (ablation: e.g. zero fx,fy for an
        # absolute-frame checkpoint that was trained with them).
        self.declare_parameter("force_obs_xy_zero", "auto")
        # -------- inference-side rotation canonicalization (experimental) ----
        # Rotate the observation (image + qpos xy + tool yaw) so the live
        # stain always presents at the training direction, then rotate the
        # predicted trajectory back. Lets a single-direction policy follow a
        # stain drawn at an arbitrary angle -- WITHOUT retraining. Needs a
        # stain-relative checkpoint and a stain_origin_node that publishes the
        # strip angle (dark method). Approximate: an in-plane image rotation is
        # not a true rotated-camera view (tool/lighting/perspective drift),
        # worse the farther from the trained angle.
        self.declare_parameter("stain_canon_enable", False)
        self.declare_parameter("stain_canon_train_angle_deg", 132.0)
        # Override the live stain angle instead of using the (noisy, ~+/-15deg)
        # dark-cloud principal axis -- for a controlled test where you draw the
        # stain at a deliberate, known angle. <0 => use the detected value.
        self.declare_parameter("stain_canon_live_angle_deg", -1.0)
        self.declare_parameter("stain_canon_rotate_image", True)
        # Flip if the image rotates the wrong way vs the pose frame (pixel-Y
        # is down; the sign depends on the camera mount).
        self.declare_parameter("stain_canon_image_angle_sign", 1.0)
        # Refuse to canonicalize beyond this |alpha| (deg) -- image artifacts
        # make far rotations unreliable; 0 disables the guard.
        self.declare_parameter("stain_canon_max_angle_deg", 65.0)
        # Read-only model diagnostics: load the checkpoint and process live
        # observations, but do not create a robot-command publisher or control
        # timer. Only visualization publishers and the inference timer remain.
        self.declare_parameter("visualization_only", False)
        # Minimal FLOW executor: keep model trajectory replay, command-rate
        # smoothing, contact measurement, and command-envelope safety, while
        # bypassing the legacy ACT-era stall/kick/dither/recovery machinery.
        self.declare_parameter("clean_flow_execution", False)

        self.declare_parameter("image_qos", "best_effort")

        # Camera preprocessing for online inference.
        # Default ON: real-time stabilization of incoming RGB before policy observation.
        self.declare_parameter("camera_preprocess_mode", "stabilize")  # off | stabilize
        self.declare_parameter("camera_stabilize_alpha", 0.92)          # cumulative trajectory EMA
        self.declare_parameter("camera_stabilize_border_mode", "reflect")
        self.declare_parameter("camera_jitter_report_enable", True)
        self.declare_parameter("camera_jitter_log_every_n", 100)

        # -----------------------------
        # Grad-CAM debug visualization
        # -----------------------------
        # Default OFF: normal inference behavior is unchanged unless enabled.
        self.declare_parameter("gradcam_enable", False)
        self.declare_parameter("gradcam_every_n_infer", 5)
        self.declare_parameter("gradcam_target", "z")
        self.declare_parameter("gradcam_target_step", 0)
        self.declare_parameter("gradcam_target_horizon", 1)
        self.declare_parameter("gradcam_layer_name", "")
        self.declare_parameter("gradcam_alpha", 0.45)
        self.declare_parameter("gradcam_colormap", "jet")
        self.declare_parameter("gradcam_publish", True)
        self.declare_parameter("gradcam_overlay_topic", "/inference_core/gradcam_overlay")
        self.declare_parameter("gradcam_global_overlay_topic", "/inference_core/gradcam_overlay_global")
        self.declare_parameter("gradcam_save", False)
        self.declare_parameter("gradcam_save_dir", "~/nrs_imitation/gradcam")
        self.declare_parameter("gradcam_log_every_n", 5)

        # Modality attribution is intentionally separate from Grad-CAM. It
        # compares normalized policy-output changes after independently
        # ablating position, force, and RGB, then publishes a dashboard image.
        self.declare_parameter("modality_importance_enable", True)
        self.declare_parameter("modality_importance_every_n_infer", 5)
        self.declare_parameter("modality_importance_target", "action_norm")
        self.declare_parameter("modality_importance_target_step", 0)
        self.declare_parameter("modality_importance_target_horizon", 16)
        self.declare_parameter("modality_importance_ema_alpha", 0.80)
        self.declare_parameter("modality_importance_history_len", 120)
        self.declare_parameter(
            "modality_importance_topic", "/inference_core/modality_importance"
        )
        self.declare_parameter("modality_importance_log_every_n", 5)

        # FLOW trajectory/vector diagnostic. This is deliberately independent
        # from Grad-CAM: it draws the raw, denormalized absolute XYZ trajectory
        # inferred from the current observation over the local-camera image.
        # With flow_diagnostic_only enabled, the periodic control loop publishes
        # no robot commands. A bounded move is possible only through the explicit
        # Trigger service below.
        self.declare_parameter("flow_vector_overlay_enable", False)
        self.declare_parameter(
            "flow_vector_overlay_topic", "/inference_core/flow_vector_overlay"
        )
        self.declare_parameter("flow_vector_overlay_horizons", "1,5,15,30,60,127")
        self.declare_parameter("flow_vector_overlay_selected_horizon", 30)
        self.declare_parameter("flow_vector_overlay_tcp_center_x", 253)
        self.declare_parameter("flow_vector_overlay_tcp_center_y", 120)
        self.declare_parameter("flow_vector_overlay_pixels_per_mm", 2.0)
        # Eye-in-hand camera: the tool tip barely moves in-frame when the tool
        # moves (camera moves with it), so a single isotropic px/mm scalar
        # aligned to image X/Y is wrong in general. This 2x2 matrix maps
        # predicted tool-frame (dx_mm, dy_mm) to image (du_px, dv_px):
        #   du = m_du_dx * dx_mm + m_du_dy * dy_mm
        #   dv = m_dv_dx * dx_mm + m_dv_dy * dy_mm
        # Defaults below were empirically calibrated on 2026-08-08 for the
        # 250mm/45deg mount by jogging the robot +10mm along tool X then tool
        # Y and template-matching background (workpiece) patches between
        # frames to measure how far the scene panned -- not by tracking the
        # tool tip, which stays ~fixed in-frame on this eye-in-hand rig.
        # Re-calibrate the same way whenever the mount geometry changes.
        self.declare_parameter("flow_vector_overlay_m_du_dx", 0.0)
        self.declare_parameter("flow_vector_overlay_m_du_dy", 0.65)
        self.declare_parameter("flow_vector_overlay_m_dv_dx", 0.45)
        self.declare_parameter("flow_vector_overlay_m_dv_dy", 0.0)
        self.declare_parameter("flow_diagnostic_only", False)
        self.declare_parameter("flow_step_service_enable", False)
        self.declare_parameter("flow_step_service", "/inference_core/flow_step")
        self.declare_parameter("flow_step_max_xyz_mm", 0.5)
        self.declare_parameter("flow_step_max_rot_rad", 0.001)
        self.declare_parameter("flow_step_stats_margin_mm", 10.0)
        self.declare_parameter("flow_step_block_down_on_contact", True)

        self.declare_parameter("control_hz", 125.0)
        self.declare_parameter("infer_hz", 5.0)

        # FLOW is a trajectory generator, so consume the newest trajectory on
        # its training time axis. "auto" selects this for FLOW and keeps legacy
        # temporal aggregation for ACT/Diffusion checkpoints.
        self.declare_parameter("action_selection_mode", "auto")  # auto | trajectory_interp | temporal_agg
        self.declare_parameter("trajectory_hz", 30.0)
        # FLOW executor controls. The legacy path regenerated at infer_hz and
        # translated every absolute trajectory so step zero matched the current
        # pose. At 5 Hz inference / 30 Hz replay, that repeatedly consumed only
        # the first ~6 steps. Preserve the learned absolute frame and keep each
        # plan alive for a meaningful horizon instead.
        self.declare_parameter("flow_local_anchor_enable", False)
        self.declare_parameter("flow_replan_interval_steps", 30)

        # -----------------------------
        # New observation encoder / force history
        # -----------------------------
        self.declare_parameter("use_force_history", True)
        self.declare_parameter("force_history_len", 10)

        self.declare_parameter("position_dim", 6)
        self.declare_parameter("force_dim", 3)
        self.declare_parameter("position_encoder_hidden_dim", 128)
        self.declare_parameter("force_encoder_hidden_dim", 64)
        self.declare_parameter("force_encoder_num_layers", 1)
        self.declare_parameter("force_encoder_dropout", 0.0)
        self.declare_parameter("observation_encoder_activation", "gelu")

        # -----------------------------
        # Baseline safety (QP-safe)
        # -----------------------------
        self.declare_parameter("tau_sec", 0.8)
        self.declare_parameter("startup_ramp_sec", 3.0)
        self.declare_parameter("step_cap_pos_mm", 0.05)
        self.declare_parameter("step_cap_ang_rad", 0.0001)
        self.declare_parameter("step_cap_fz", 0.05)
        # cmd_target[8] (fz) is kept at 0 through TRACK until real contact --
        # letting the raw policy fz leak through pre-contact used to make
        # force_control.cpp treat any |Fd|>0.01N as "desired_force_active"
        # and drop z stiffness to 0 well before the sensor ever saw force,
        # driving with a fixed 15N push independent of cmd_z entirely. But
        # the model's raw (suppressed-before-publish) fz prediction rising is
        # itself a signal that it believes contact is close, so use it only
        # to slow the z-axis position rate -- not to command real force --
        # for a controlled final approach instead of hitting the surface at
        # full free-space speed.
        self.declare_parameter("approach_slow_fz_thr", 0.5)
        self.declare_parameter("approach_slow_step_cap_pos_mm", 0.015)

        self.declare_parameter("use_temporal_agg", True)
        self.declare_parameter("temporal_agg_mode", "exp")
        self.declare_parameter("temporal_agg_tau_steps", 20.0)
        self.declare_parameter("pred_step_offset", 1)
        self.declare_parameter("max_plans", 6)

        # contact gating
        self.declare_parameter("contact_on_thr", 3.0)
        self.declare_parameter("contact_off_thr", 1.2)
        self.declare_parameter("clear_plans_on_contact_change", False)
        # For the current flat-surface task, stop pose-Z descent at first
        # contact. Normal loading remains available through action Fz.
        self.declare_parameter("contact_z_descent_block_enable", True)
        self.declare_parameter("contact_z_descent_margin_mm", 0.2)

        # touch detection
        self.declare_parameter("touch_fz_thr", 0.5)
        self.declare_parameter("touch_ok_count", 3)
        self.declare_parameter("touch_min_after_start_sec", 1.0)
        self.declare_parameter("touch_baseline_tau_sec", 0.5)
        self.declare_parameter("touch_use_delta", True)

        # preload
        self.declare_parameter("preload_target_source", "stats_mean")  # stats_mean | fixed
        self.declare_parameter("preload_fixed_N", 10.0)
        self.declare_parameter("preload_target_scale", 1.0)
        self.declare_parameter("preload_min_N", 10.0)
        self.declare_parameter("preload_timeout_sec", 5.0)
        self.declare_parameter("preload_ok_count", 10)
        self.declare_parameter("preload_kp_mm_per_N", 0.02)
        self.declare_parameter("preload_dz_max_mm", 0.08)
        self.declare_parameter("preload_tol_N", 0.2)
        self.declare_parameter("preload_max_descent_mm", 8.0)
        # A PRELOAD exit right at the contact boundary can re-satisfy the
        # touch debounce again within tens of ms (robot still in contact),
        # repeatedly re-entering PRELOAD before force_control.cpp's mode
        # transition (~0.2s time constants) ever finishes settling -- each
        # interrupted transition stacks a new transient on the last, and the
        # accumulating oscillation can blow up (20260812 02:29 FLOW: PRELOAD
        # re-entered after 24ms/0.45s gaps, force grew 70->200N then spiked
        # to -777N). Block re-entry for this long after any PRELOAD exit.
        self.declare_parameter("preload_reentry_cooldown_sec", 0.8)

        # TRACK executed as discrete blocking PTP9D (pose+force) service
        # calls to singleArm_cmd instead of continuous 125Hz cmdMotion
        # publish. Each call is a bounded, checked move -- no open-loop
        # reference can drift ahead of the live pose between calls the way
        # the continuous topic-interp path could, so this replaces PRELOAD/
        # contact_z_descent_block/hold-anchor/approach-slowdown for TRACK
        # entirely (all of that stays intact and unused as a fallback when
        # this is off).
        self.declare_parameter("track_use_ptp9d_service", True)
        # Each PTP9D call now carries a short lookahead segment of
        # consecutive predicted waypoints (not one point) so the robot-side
        # blender (Y2RobMotion PTP9D_command_gen -> MotionBlender9D) can
        # ramp through them as one continuous motion instead of stopping
        # fully between every point.
        self.declare_parameter("ptp9d_segment_points", 15)
        self.declare_parameter("ptp9d_segment_stride", 1)
        # True: use the Y2RobMotion PTP9D_STREAM_{START,APPEND,STOP} queue
        # instead of the batched-segment PTP9D_STREAM_call-and-wait chain
        # above. The queue is a persistent robot-side thread that never
        # decelerates to a stop between waypoints (only when it actually
        # runs dry), so this is the only path with true call-to-call
        # continuity; the batched segments above still stop briefly at
        # every service call boundary.
        self.declare_parameter("ptp9d_use_stream", False)
        # Position lookahead can now be generous -- force no longer rides
        # along in the queue (see PTP9D_STREAM_SET_FORCE below), so a deep
        # buffer no longer risks delaying a contact-state change. Deepened
        # from 20pts/1.0s to roughly match the batch/service_call path's
        # implicit smoothing scale: each batched call there blends ~15-20mm
        # of predicted path into one confident MotionBlender9D curve, which
        # is what let it push through the ambiguous pre-contact "is this
        # really contact yet" prediction wobble instead of hesitating
        # through it point by point (20260815: service_stream kept letting
        # go right after light touches even after force was decoupled).
        self.declare_parameter("ptp9d_stream_topup_points", 40)
        self.declare_parameter("ptp9d_stream_min_lookahead_sec", 2.5)
        # The robot-side stream thread rate-limits toward and pops every
        # single point it's given -- unlike the batch path, there is no
        # MotionBlender9D accel-decel profile to smooth over per-sample
        # prediction noise, so raw consecutive 30Hz waypoints get chased
        # exactly as given (20260815 first live test: visibly oscillated).
        # smooth_window applies a centered moving average to position/
        # orientation (not force) before each segment is chosen; stride
        # skips samples so fewer, less noisy points need to be visited.
        # Widened alongside topup_points/min_lookahead_sec above, same
        # reasoning: match the batch path's much coarser effective
        # smoothing scale through the pre-contact transition.
        self.declare_parameter("ptp9d_stream_smooth_window", 35)
        self.declare_parameter("ptp9d_stream_stride", 2)
        # Force is sent through a separate immediate channel
        # (PTP9D_STREAM_SET_FORCE), not bundled into queued waypoints --
        # otherwise a contact-state change sits behind however much
        # position lookahead is already queued (up to ~1s) before it takes
        # effect, which made the robot repeatedly let go right after
        # touching (20260815 live test: 9 separate contact on/off cycles
        # instead of one sustained hold like every training demonstration).
        # min_interval_sec rate-limits routine updates while in steady
        # contact; a contact on/off transition always sends immediately,
        # bypassing this interval.
        self.declare_parameter("ptp9d_stream_force_min_interval_sec", 0.1)
        self.declare_parameter("ptp9d_target_velocity_mm_s", 10.0)
        self.declare_parameter("ptp9d_service_name", "/singleArm_cmd/single_arm_command")

        self.declare_parameter("press_force_cmd_mode", "target")  # keep|zero|target
        self.declare_parameter("press_hold_xy", True)
        self.declare_parameter("press_hold_rpy", True)

        # optional release assist
        self.declare_parameter("release_assist_enable", False)
        self.declare_parameter("release_ramp_sec", 1.0)

        # I/O shaping
        self.declare_parameter("force_indices", [0, 1, 2])
        self.declare_parameter("first_cmd_fz", 0.0)
        self.declare_parameter("force_xy_cmd_enable", True)
        self.declare_parameter("force_xy_hard_limit", 10.0)
        self.declare_parameter("action_type", "absolute")  # absolute | delta
        self.declare_parameter("normalize_qpos", True)
        self.declare_parameter("denorm_action", True)
        self.declare_parameter("resize_hw", 0)
        self.declare_parameter("debug_every_n", 30)

        # Temporary ablation for calibration-corrupted orientation labels.
        # When enabled, measured/predicted orientation is ignored: policy qpos,
        # demo-start alignment, and final robot commands all use the fixed
        # world-Z 90-degree rotation vector below. XYZ, force, force history, and images
        # remain live.
        self.declare_parameter("orientation_lock_enable", False)
        self.declare_parameter("orientation_lock_wx", 0.0)
        self.declare_parameter("orientation_lock_wy", 0.0)
        self.declare_parameter("orientation_lock_wz", 1.5707963268)

        # Force-command upper limit. Values <= 0 disable only the upper limit;
        # the final command remains non-negative.
        self.declare_parameter("fz_hard_limit", 30.0)

        # Demo-start alignment.
        # Default False preserves the previous inference behavior exactly.
        self.declare_parameter("auto_move_to_demo_start", True)
        self.declare_parameter("demo_start_move_sec", 5.0)
        self.declare_parameter("demo_start_hold_sec", 2.0)
        # Lift the demo-start alignment target along world +Z before inference.
        # This prevents curved/convex-surface policies from starting while already in contact.
        self.declare_parameter("demo_start_z_offset_mm", 0.0)
        # Refuse automatic demo-start moves that are too far from the live robot pose.
        # Use <=0 only when an external safety layer already constrains this motion.
        self.declare_parameter("demo_start_max_align_dist_mm", 75.0)
        # The alignment trajectory uses smoothstep interpolation, whose peak
        # speed is 1.5 * distance / duration.  Stretch long moves so they do
        # not become faster just because the start pose is far away.
        self.declare_parameter("demo_start_max_xyz_speed_mm_s", 50.0)
        self.declare_parameter("demo_start_max_rot_speed_rad_s", 0.25)
        # Policy inference starts only after the measured TCP, not merely the
        # commanded trajectory, has settled near the demonstration start.
        self.declare_parameter("demo_start_position_tolerance_mm", 5.0)
        self.declare_parameter("demo_start_rotation_tolerance_rad", 0.05)
        # Drive the current-pose -> demo_start move through the controller's
        # native PTP (singleArm_cmd/single_arm_command service) instead of
        # this node's own smoothstep + admittance position servo. False
        # restores the old smoothstep path for rollback.
        self.declare_parameter("demo_start_use_ptp_service", True)
        self.declare_parameter("ptp_service_name", "/singleArm_cmd/single_arm_command")
        self.declare_parameter("ptp_target_velocity_mm_s", 20.0)
        self.declare_parameter("ptp_service_timeout_sec", 30.0)
        # PTP_command_gen switches the low-level controller to Position mode
        # internally. Switch it back to Force mode (same service) once PTP
        # completes, before policy inference starts publishing to cmdMotion.
        self.declare_parameter("ptp_switch_to_force_mode", True)

        # Optional policy-output Z offset.
        # This is applied to every denormalized absolute action z target.
        # Default 0.0 preserves the original learned trajectory.
        self.declare_parameter("policy_z_offset_mm", 0.0)

        # Last-line command guard. It holds the current pose instead of publishing
        # a command whose xyz target is implausibly far from /currentP.
        self.declare_parameter("cmd_safety_enable", True)
        self.declare_parameter("cmd_safety_max_xyz_from_current_mm", 200.0)
        # Optional policy-start envelope. Values <= 0 disable that axis limit.
        # The polishing launch enables limits derived from the filtered58 demos.
        self.declare_parameter("cmd_safety_max_xy_from_start_mm", 0.0)
        self.declare_parameter("cmd_safety_max_z_down_from_start_mm", 0.0)
        self.declare_parameter("cmd_safety_max_z_up_from_start_mm", 0.0)
        self.declare_parameter("cmd_safety_latch_on_start_limit", True)

        # policy config
        self.declare_parameter("kl_weight", 10.0)
        self.declare_parameter("hidden_dim", 512)
        self.declare_parameter("dim_feedforward", 3200)
        self.declare_parameter("lr_backbone", 1e-5)
        self.declare_parameter("backbone", "resnet18")
        self.declare_parameter("enc_layers", 4)
        self.declare_parameter("dec_layers", 7)
        self.declare_parameter("nheads", 8)
        self.declare_parameter("image_resize_hw", 256)
        self.declare_parameter("image_pool_hw", 4)
        self.declare_parameter("pretrained_backbone", True)
        self.declare_parameter("image_backbone", "dinov3")
        self.declare_parameter("dino_model_name", "vit_small_patch16_dinov3.lvd1689m")
        self.declare_parameter("dino_checkpoint_path", "")
        self.declare_parameter("freeze_image_backbone", True)
        self.declare_parameter("dino_roi_pooling", "attention")
        self.declare_parameter("use_tcp_roi", True)
        self.declare_parameter("tcp_roi_reference_width", 424)
        self.declare_parameter("tcp_roi_reference_height", 240)
        self.declare_parameter("tcp_roi_center_x", 253)
        self.declare_parameter("tcp_roi_center_y", 120)
        self.declare_parameter("tcp_roi_area_fraction", 0.10)

        # diffusion policy config
        self.declare_parameter("diffusion_train_steps", 100)
        self.declare_parameter("diffusion_infer_steps", 10)
        self.declare_parameter("diffusion_beta_start", 1e-4)
        self.declare_parameter("diffusion_beta_end", 2e-2)
        self.declare_parameter("diffusion_loss_type", "mse")

        # FLOW policy config
        self.declare_parameter("flow_infer_steps", 10)
        # When enabled, every online plan starts from the same seeded FLOW noise.
        # The same tensor is also reused by the Grad-CAM backward pass.
        self.declare_parameter("flow_deterministic_noise", False)
        self.declare_parameter("flow_noise_seed", 0)
        self.declare_parameter("flow_train_eps", 1e-4)
        self.declare_parameter("flow_loss_type", "mse")
        self.declare_parameter("flow_obs_hidden_dim", 256)
        self.declare_parameter("flow_image_feature_dim", 512)
        self.declare_parameter("flow_global_cond_dim", 256)
        self.declare_parameter("flow_time_embed_dim", 256)
        self.declare_parameter("flow_down_dims", "256,512,1024")
        self.declare_parameter("flow_kernel_size", 5)
        self.declare_parameter("flow_n_groups", 8)
        self.declare_parameter("flow_cond_predict_scale", False)

        self.declare_parameter("use_stain_mask", False)
        self.declare_parameter("stain_mask_key", "observations/images/stain_mask")
        self.declare_parameter("stain_pooling_type", "masked_mean")
        self.declare_parameter("empty_stain_feature_mode", "zero")
        self.declare_parameter("stain_mask_threshold", 0.5)
        self.declare_parameter("debug_stain_pooling", False)

        # Optional gripper extension. When enabled, the policy uses gripper state
        # observations. action[9] is published to /gripper/command as position
        # tick, and action[10] is published as goal current in mA. The robot
        # motion command path still uses action[0:9] and the same polishing safety loop.
        self.declare_parameter("use_gripper", False)
        self.declare_parameter("gripper_position_topic", "/gripper/present_position")
        self.declare_parameter("gripper_current_topic", "/gripper/present_current_mA")
        self.declare_parameter("use_gripper_history", False)
        self.declare_parameter("gripper_history_len", 15)
        self.declare_parameter("gripper_history_hz", 30.0)
        self.declare_parameter("gripper_history_sync_slop_sec", 0.020)
        self.declare_parameter("gripper_history_max_age_sec", 0.20)
        self.declare_parameter("gripper_history_debug_every_n", 30)
        self.declare_parameter("gripper_command_topic", "/gripper/command")
        self.declare_parameter("gripper_goal_current_topic", "/gripper/goal_current_mA")
        self.declare_parameter("gripper_command_min_tick", -653)
        self.declare_parameter("gripper_command_max_tick", 733)
        self.declare_parameter("gripper_command_deadband_tick", 2)
        self.declare_parameter("gripper_command_slew_per_sec", 1000.0)
        self.declare_parameter("gripper_command_step_cap_tick", 200.0)
        self.declare_parameter("gripper_cmd_safety_enable", True)
        self.declare_parameter("gripper_cmd_safety_max_tick_from_present", 1500.0)
        self.declare_parameter("gripper_goal_current_min_mA", 0.0)
        self.declare_parameter("gripper_goal_current_max_mA", 1345.0)
        self.declare_parameter("gripper_goal_current_deadband_mA", 5.0)

        # stall + recover
        self.declare_parameter("stall_sec", 1.2)
        self.declare_parameter("stall_min_after_start_sec", 1.0)
        self.declare_parameter("stall_lpf_tau_sec", 0.40)
        self.declare_parameter("stall_window_net_pos_eps_mm", 0.25)
        self.declare_parameter("stall_window_net_ang_eps_rad", 0.0006)

        self.declare_parameter("fz_kick_N", 1.5)
        self.declare_parameter("fz_kick_dur_sec", 0.35)
        self.declare_parameter("fz_kick_cooldown_sec", 0.8)

        self.declare_parameter("recover_enable", True)
        self.declare_parameter("recover_cooldown_sec", 2.0)
        self.declare_parameter("recover_timeout_sec", 6.0)
        self.declare_parameter("recover_pos_tol_mm", 0.35)
        self.declare_parameter("recover_ang_tol_rad", 0.0008)
        self.declare_parameter("recover_ok_count", 10)

        # dither + improved recover
        self.declare_parameter("dither_enable", False)
        self.declare_parameter("dither_only_track", True)
        self.declare_parameter("dither_min_after_start_sec", 2.0)
        self.declare_parameter("dither_win_sec", 1.0)
        self.declare_parameter("dither_sec", 1.0)
        self.declare_parameter("dither_net_pos_thr_mm", 0.8)
        self.declare_parameter("dither_net_ang_thr_rad", 0.0015)
        self.declare_parameter("dither_path_ratio_thr", 6.0)
        self.declare_parameter("dither_rms_pos_thr_mm", 0.10)
        self.declare_parameter("dither_rms_ang_thr_rad", 0.00025)
        self.declare_parameter("dither_decay", 0.5)

        self.declare_parameter("kick_max_before_recover", 2)
        self.declare_parameter("kick_reset_sec", 6.0)

        self.declare_parameter("recover_check_lpf_tau_sec", 0.25)

        self.declare_parameter("recover_use_overrides", True)
        self.declare_parameter("recover_tau_sec", 0.25)
        self.declare_parameter("recover_startup_ramp_sec", 0.6)
        self.declare_parameter("recover_step_cap_pos_mm", 1.0)
        self.declare_parameter("recover_step_cap_ang_rad", 0.0012)
        self.declare_parameter("recover_step_cap_fz", 0.30)

        self.declare_parameter("recover_timeout_min_margin_sec", 1.0)
        self.declare_parameter("recover_timeout_scale", 1.4)

        # -----------------------------
        # Read params
        # -----------------------------
        self.ckpt_dir = str(self.get_parameter("ckpt_dir").value)
        self.act_root = os.path.expanduser(str(self.get_parameter("act_root").value))
        self.policy_class = str(self.get_parameter("policy_class").value).strip().upper()
        self.ckpt_auto_subdir = str(self.get_parameter("ckpt_auto_subdir").value).strip()
        self.metrics_log_enable = bool(self.get_parameter("metrics_log_enable").value)
        self.metrics_log_dir = str(self.get_parameter("metrics_log_dir").value).strip()
        self.metrics_run_tag = str(self.get_parameter("metrics_run_tag").value).strip()
        self.obs_mode = str(self.get_parameter("obs_mode").value).strip().lower()
        if self.obs_mode == "dual":
            self.obs_mode = "dual_cam"
        if self.obs_mode in ("", "auto"):
            self.obs_mode = "single_cam"
        if self.obs_mode not in ("single_cam", "dual_cam"):
            raise RuntimeError(f"obs_mode must be single_cam or dual_cam, got: {self.obs_mode}")
        self.camera_names = _parse_camera_names(self.get_parameter("camera_names").value, self.obs_mode)
        self.use_global_image = len(self.camera_names) >= 2
        self.phase_mode = str(self.get_parameter("phase_mode").value).strip().lower()
        self.chunk_size = int(self.get_parameter("chunk_size").value)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.force_topic = str(self.get_parameter("force_topic").value)
        self.force_msg_type = str(self.get_parameter("force_msg_type").value).strip().lower()
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.global_image_topic = str(self.get_parameter("global_image_topic").value)
        self.stain_mask_topic = str(self.get_parameter("stain_mask_topic").value).strip()
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.stain_origin_topic = str(self.get_parameter("stain_origin_topic").value).strip()
        self.stain_canon_enable = bool(self.get_parameter("stain_canon_enable").value)
        self.stain_canon_train_angle_deg = float(self.get_parameter("stain_canon_train_angle_deg").value)
        self.stain_canon_live_angle_deg = float(self.get_parameter("stain_canon_live_angle_deg").value)
        self.stain_canon_rotate_image = bool(self.get_parameter("stain_canon_rotate_image").value)
        self.stain_canon_image_angle_sign = float(self.get_parameter("stain_canon_image_angle_sign").value)
        self.stain_canon_max_angle_deg = float(self.get_parameter("stain_canon_max_angle_deg").value)
        self._canon_alpha = 0.0          # rad; set when the origin+angle arrive
        self._canon_active = False
        self._canon_setup_done = False
        self.visualization_only = bool(
            self.get_parameter("visualization_only").value
        )
        self.clean_flow_execution = bool(
            self.get_parameter("clean_flow_execution").value
        )

        self.image_qos_str = str(self.get_parameter("image_qos").value)
        self.camera_preprocess_mode = str(self.get_parameter("camera_preprocess_mode").value).strip().lower()
        self.camera_stabilize_alpha = float(self.get_parameter("camera_stabilize_alpha").value)
        self.camera_stabilize_border_mode = str(self.get_parameter("camera_stabilize_border_mode").value).strip().lower()
        self.camera_jitter_report_enable = bool(self.get_parameter("camera_jitter_report_enable").value)
        self.camera_jitter_log_every_n = max(1, int(self.get_parameter("camera_jitter_log_every_n").value))
        if self.camera_preprocess_mode not in ("off", "none", "raw", "stabilize"):
            raise RuntimeError(f"camera_preprocess_mode must be off or stabilize, got: {self.camera_preprocess_mode}")

        self.gradcam_enable = bool(self.get_parameter("gradcam_enable").value)
        self.gradcam_every_n_infer = max(1, int(self.get_parameter("gradcam_every_n_infer").value))
        self.gradcam_target = str(self.get_parameter("gradcam_target").value).strip().lower()
        self.gradcam_target_step = max(0, int(self.get_parameter("gradcam_target_step").value))
        self.gradcam_target_horizon = max(1, int(self.get_parameter("gradcam_target_horizon").value))
        self.gradcam_layer_name = str(self.get_parameter("gradcam_layer_name").value).strip()
        self.gradcam_alpha = float(self.get_parameter("gradcam_alpha").value)
        self.gradcam_colormap = str(self.get_parameter("gradcam_colormap").value).strip().lower()
        self.gradcam_publish = bool(self.get_parameter("gradcam_publish").value)
        self.gradcam_overlay_topic = str(self.get_parameter("gradcam_overlay_topic").value)
        self.gradcam_global_overlay_topic = str(self.get_parameter("gradcam_global_overlay_topic").value)
        self.gradcam_save = bool(self.get_parameter("gradcam_save").value)
        self.gradcam_save_dir = os.path.expanduser(str(self.get_parameter("gradcam_save_dir").value))
        self.gradcam_log_every_n = max(1, int(self.get_parameter("gradcam_log_every_n").value))
        self._gradcam_pub_count = 0
        self._gradcam_fail_count = 0
        self._gradcam_activation = None
        self._gradcam_gradient = None
        self._gradcam_target_layer_name = ""
        # Cached cam0 heatmap, reused by the FLOW vector overlay so it can
        # show Grad-CAM without running its own backward pass. One replan
        # tick stale at most (both update on the same ~few-second cadence).
        self._latest_gradcam_heat = None
        self._gradcam_target_layer = None
        self._gradcam_fwd_handle = None
        self._gradcam_bwd_handle = None
        self._gradcam_last_log_t = 0.0
        if self.gradcam_save:
            try:
                os.makedirs(self.gradcam_save_dir, exist_ok=True)
            except Exception as e:
                raise RuntimeError(f"Failed to create gradcam_save_dir={self.gradcam_save_dir}: {e}")

        self.modality_importance_enable = bool(
            self.get_parameter("modality_importance_enable").value
        )
        self.modality_importance_every_n_infer = max(
            1, int(self.get_parameter("modality_importance_every_n_infer").value)
        )
        self.modality_importance_target = str(
            self.get_parameter("modality_importance_target").value
        ).strip().lower()
        self.modality_importance_target_step = max(
            0, int(self.get_parameter("modality_importance_target_step").value)
        )
        self.modality_importance_target_horizon = max(
            1, int(self.get_parameter("modality_importance_target_horizon").value)
        )
        self.modality_importance_ema_alpha = float(np.clip(
            float(self.get_parameter("modality_importance_ema_alpha").value), 0.0, 0.999
        ))
        self.modality_importance_history_len = max(
            10, int(self.get_parameter("modality_importance_history_len").value)
        )
        self.modality_importance_topic = str(
            self.get_parameter("modality_importance_topic").value
        ).strip()
        self.modality_importance_log_every_n = max(
            1, int(self.get_parameter("modality_importance_log_every_n").value)
        )
        if self.modality_importance_enable and not self.modality_importance_topic:
            raise RuntimeError("modality_importance_topic must be non-empty when enabled")
        self._modality_importance_count = 0
        self._modality_importance_fail_count = 0
        self._modality_importance_last_log_t = 0.0
        self._modality_importance_ema: Optional[np.ndarray] = None
        self._modality_importance_worker_lock = threading.Lock()
        self._modality_importance_worker_busy = False
        self._modality_importance_worker_thread: Optional[threading.Thread] = None
        self._modality_importance_history: Deque[np.ndarray] = deque(
            maxlen=self.modality_importance_history_len
        )

        self.flow_vector_overlay_enable = bool(
            self.get_parameter("flow_vector_overlay_enable").value
        )
        self.flow_vector_overlay_topic = str(
            self.get_parameter("flow_vector_overlay_topic").value
        ).strip()
        raw_horizons = str(
            self.get_parameter("flow_vector_overlay_horizons").value
        ).strip()
        try:
            self.flow_vector_overlay_horizons = sorted(
                set(max(0, int(x.strip())) for x in raw_horizons.split(",") if x.strip())
            )
        except ValueError as exc:
            raise RuntimeError(
                "flow_vector_overlay_horizons must be comma-separated integers"
            ) from exc
        if not self.flow_vector_overlay_horizons:
            self.flow_vector_overlay_horizons = [1, 5, 15, 30, 60, 127]
        self.flow_vector_overlay_selected_horizon = max(
            0, int(self.get_parameter("flow_vector_overlay_selected_horizon").value)
        )
        self.flow_vector_overlay_tcp_center_x = int(
            self.get_parameter("flow_vector_overlay_tcp_center_x").value
        )
        self.flow_vector_overlay_tcp_center_y = int(
            self.get_parameter("flow_vector_overlay_tcp_center_y").value
        )
        self.flow_vector_overlay_pixels_per_mm = max(
            0.01, float(self.get_parameter("flow_vector_overlay_pixels_per_mm").value)
        )
        self.flow_vector_overlay_m_du_dx = float(
            self.get_parameter("flow_vector_overlay_m_du_dx").value
        )
        self.flow_vector_overlay_m_du_dy = float(
            self.get_parameter("flow_vector_overlay_m_du_dy").value
        )
        self.flow_vector_overlay_m_dv_dx = float(
            self.get_parameter("flow_vector_overlay_m_dv_dx").value
        )
        self.flow_vector_overlay_m_dv_dy = float(
            self.get_parameter("flow_vector_overlay_m_dv_dy").value
        )
        self.flow_diagnostic_only = bool(
            self.get_parameter("flow_diagnostic_only").value
        )
        self.flow_step_service_enable = bool(
            self.get_parameter("flow_step_service_enable").value
        )
        self.flow_step_service = str(
            self.get_parameter("flow_step_service").value
        ).strip()
        self.flow_step_max_xyz_mm = max(
            0.0, float(self.get_parameter("flow_step_max_xyz_mm").value)
        )
        self.flow_step_max_rot_rad = max(
            0.0, float(self.get_parameter("flow_step_max_rot_rad").value)
        )
        self.flow_step_stats_margin_mm = max(
            0.0, float(self.get_parameter("flow_step_stats_margin_mm").value)
        )
        self.flow_step_block_down_on_contact = bool(
            self.get_parameter("flow_step_block_down_on_contact").value
        )
        if self.flow_vector_overlay_enable and not self.flow_vector_overlay_topic:
            raise RuntimeError("flow_vector_overlay_topic must be non-empty when enabled")
        if self.flow_step_service_enable and not self.flow_step_service:
            raise RuntimeError("flow_step_service must be non-empty when enabled")
        if self.flow_step_service_enable and not self.flow_diagnostic_only:
            raise RuntimeError(
                "flow_step_service_enable requires flow_diagnostic_only=true so automatic "
                "control and manual stepping cannot run together"
            )

        self.control_hz = float(self.get_parameter("control_hz").value)
        self.infer_hz = float(self.get_parameter("infer_hz").value)

        self.action_selection_mode = str(self.get_parameter("action_selection_mode").value).strip().lower()
        self.trajectory_hz = max(1e-6, float(self.get_parameter("trajectory_hz").value))
        if self.action_selection_mode == "auto":
            # FLOW and BSPLINE both emit a full time-indexed trajectory in one
            # forward pass, so replaying it via interpolation fits both.
            self.action_selection_mode = (
                "trajectory_interp" if self.policy_class in ("FLOW", "BSPLINE") else "temporal_agg"
            )
        if self.action_selection_mode not in ("trajectory_interp", "temporal_agg"):
            raise RuntimeError(
                "action_selection_mode must be auto, trajectory_interp, or temporal_agg, "
                f"got: {self.action_selection_mode}"
            )
        self.flow_local_anchor_enable = bool(
            self.get_parameter("flow_local_anchor_enable").value
        )
        self.flow_replan_interval_steps = max(
            0, int(self.get_parameter("flow_replan_interval_steps").value)
        )
        self.use_force_history = bool(self.get_parameter("use_force_history").value)
        self.force_history_len = int(self.get_parameter("force_history_len").value)

        self.position_dim = int(self.get_parameter("position_dim").value)
        self.force_dim = int(self.get_parameter("force_dim").value)
        self.position_encoder_hidden_dim = int(self.get_parameter("position_encoder_hidden_dim").value)
        self.force_encoder_hidden_dim = int(self.get_parameter("force_encoder_hidden_dim").value)
        self.force_encoder_num_layers = int(self.get_parameter("force_encoder_num_layers").value)
        self.force_encoder_dropout = float(self.get_parameter("force_encoder_dropout").value)
        self.observation_encoder_activation = str(self.get_parameter("observation_encoder_activation").value)

        self.tau_sec = float(self.get_parameter("tau_sec").value)
        self.startup_ramp_sec = float(self.get_parameter("startup_ramp_sec").value)
        self.step_cap_pos_mm = float(self.get_parameter("step_cap_pos_mm").value)
        self.step_cap_ang_rad = float(self.get_parameter("step_cap_ang_rad").value)
        self.step_cap_fz = float(self.get_parameter("step_cap_fz").value)
        self.approach_slow_fz_thr = float(self.get_parameter("approach_slow_fz_thr").value)
        self.approach_slow_step_cap_pos_mm = max(
            1e-9, float(self.get_parameter("approach_slow_step_cap_pos_mm").value)
        )

        self.use_temporal_agg = bool(self.get_parameter("use_temporal_agg").value)
        self.temporal_agg_mode = str(self.get_parameter("temporal_agg_mode").value).strip().lower()
        self.temporal_agg_tau_steps = float(self.get_parameter("temporal_agg_tau_steps").value)
        self.pred_step_offset = int(self.get_parameter("pred_step_offset").value)
        self.max_plans = int(self.get_parameter("max_plans").value)

        self.contact_on_thr = float(self.get_parameter("contact_on_thr").value)
        self.contact_off_thr = float(self.get_parameter("contact_off_thr").value)
        self.clear_plans_on_contact_change = bool(self.get_parameter("clear_plans_on_contact_change").value)
        self.contact_z_descent_block_enable = bool(
            self.get_parameter("contact_z_descent_block_enable").value
        )
        self.contact_z_descent_margin_mm = max(
            0.0, float(self.get_parameter("contact_z_descent_margin_mm").value)
        )

        self.touch_fz_thr = float(self.get_parameter("touch_fz_thr").value)
        self.touch_ok_count = int(self.get_parameter("touch_ok_count").value)
        self.touch_min_after_start_sec = float(self.get_parameter("touch_min_after_start_sec").value)
        self.touch_baseline_tau_sec = float(self.get_parameter("touch_baseline_tau_sec").value)
        self.touch_use_delta = bool(self.get_parameter("touch_use_delta").value)

        self.preload_target_source = str(self.get_parameter("preload_target_source").value).strip().lower()
        self.preload_fixed_N = float(self.get_parameter("preload_fixed_N").value)
        self.preload_target_scale = float(self.get_parameter("preload_target_scale").value)
        self.preload_min_N = float(self.get_parameter("preload_min_N").value)
        self.preload_timeout_sec = float(self.get_parameter("preload_timeout_sec").value)
        self.preload_ok_count = int(self.get_parameter("preload_ok_count").value)
        self.preload_kp_mm_per_N = float(self.get_parameter("preload_kp_mm_per_N").value)
        self.preload_dz_max_mm = float(self.get_parameter("preload_dz_max_mm").value)
        self.preload_tol_N = float(self.get_parameter("preload_tol_N").value)
        self.preload_max_descent_mm = max(
            0.0, float(self.get_parameter("preload_max_descent_mm").value)
        )
        self.preload_reentry_cooldown_sec = max(
            0.0, float(self.get_parameter("preload_reentry_cooldown_sec").value)
        )
        self.track_use_ptp9d_service = bool(
            self.get_parameter("track_use_ptp9d_service").value
        )
        self.ptp9d_segment_points = max(
            1, int(self.get_parameter("ptp9d_segment_points").value)
        )
        self.ptp9d_segment_stride = max(
            1, int(self.get_parameter("ptp9d_segment_stride").value)
        )
        self.ptp9d_use_stream = bool(self.get_parameter("ptp9d_use_stream").value)
        self.ptp9d_stream_topup_points = max(
            1, int(self.get_parameter("ptp9d_stream_topup_points").value)
        )
        self.ptp9d_stream_min_lookahead_sec = max(
            0.05, float(self.get_parameter("ptp9d_stream_min_lookahead_sec").value)
        )
        self.ptp9d_stream_smooth_window = max(
            1, int(self.get_parameter("ptp9d_stream_smooth_window").value)
        )
        self.ptp9d_stream_stride = max(
            1, int(self.get_parameter("ptp9d_stream_stride").value)
        )
        self.ptp9d_stream_force_min_interval_sec = max(
            0.0, float(self.get_parameter("ptp9d_stream_force_min_interval_sec").value)
        )
        self.ptp9d_target_velocity_mm_s = max(
            0.0, float(self.get_parameter("ptp9d_target_velocity_mm_s").value)
        )
        self.ptp9d_service_name = str(self.get_parameter("ptp9d_service_name").value)
        self.press_force_cmd_mode = str(self.get_parameter("press_force_cmd_mode").value).strip().lower()
        self.press_hold_xy = bool(self.get_parameter("press_hold_xy").value)
        self.press_hold_rpy = bool(self.get_parameter("press_hold_rpy").value)

        self.release_assist_enable = bool(self.get_parameter("release_assist_enable").value)
        self.release_ramp_sec = float(self.get_parameter("release_ramp_sec").value)

        self.force_indices = tuple(int(x) for x in self.get_parameter("force_indices").value)
        self.first_cmd_fz = float(self.get_parameter("first_cmd_fz").value)
        self.force_xy_cmd_enable = bool(self.get_parameter("force_xy_cmd_enable").value)
        self.force_xy_hard_limit = float(self.get_parameter("force_xy_hard_limit").value)
        self.action_type = str(self.get_parameter("action_type").value).strip().lower()

        self.normalize_qpos_enabled = bool(self.get_parameter("normalize_qpos").value)
        self.denorm_action_enabled = bool(self.get_parameter("denorm_action").value)

        self.resize_hw = int(self.get_parameter("resize_hw").value)
        self.debug_every_n = max(1, int(self.get_parameter("debug_every_n").value))

        self.orientation_lock_enable = bool(
            self.get_parameter("orientation_lock_enable").value
        )
        self.orientation_lock_rotvec = np.asarray(
            [
                float(self.get_parameter("orientation_lock_wx").value),
                float(self.get_parameter("orientation_lock_wy").value),
                float(self.get_parameter("orientation_lock_wz").value),
            ],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(self.orientation_lock_rotvec)):
            raise RuntimeError("orientation_lock_wx/wy/wz must all be finite")

        self.fz_hard_limit = float(self.get_parameter("fz_hard_limit").value)

        self.auto_move_to_demo_start = bool(self.get_parameter("auto_move_to_demo_start").value)
        self.demo_start_move_sec = float(self.get_parameter("demo_start_move_sec").value)
        self.demo_start_hold_sec = float(self.get_parameter("demo_start_hold_sec").value)
        self.demo_start_z_offset_mm = float(self.get_parameter("demo_start_z_offset_mm").value)
        self.demo_start_max_align_dist_mm = float(self.get_parameter("demo_start_max_align_dist_mm").value)
        self.demo_start_max_xyz_speed_mm_s = max(
            0.0, float(self.get_parameter("demo_start_max_xyz_speed_mm_s").value)
        )
        self.demo_start_max_rot_speed_rad_s = max(
            0.0, float(self.get_parameter("demo_start_max_rot_speed_rad_s").value)
        )
        self.demo_start_position_tolerance_mm = max(
            0.0, float(self.get_parameter("demo_start_position_tolerance_mm").value)
        )
        self.demo_start_rotation_tolerance_rad = max(
            0.0, float(self.get_parameter("demo_start_rotation_tolerance_rad").value)
        )
        self.demo_start_use_ptp_service = bool(
            self.get_parameter("demo_start_use_ptp_service").value
        )
        self.ptp_service_name = str(self.get_parameter("ptp_service_name").value)
        self.ptp_target_velocity_mm_s = max(
            0.0, float(self.get_parameter("ptp_target_velocity_mm_s").value)
        )
        self.ptp_service_timeout_sec = max(
            0.0, float(self.get_parameter("ptp_service_timeout_sec").value)
        )
        self.ptp_switch_to_force_mode = bool(
            self.get_parameter("ptp_switch_to_force_mode").value
        )
        self.policy_z_offset_mm = float(self.get_parameter("policy_z_offset_mm").value)
        self.cmd_safety_enable = bool(self.get_parameter("cmd_safety_enable").value)
        self.cmd_safety_max_xyz_from_current_mm = float(self.get_parameter("cmd_safety_max_xyz_from_current_mm").value)
        self.cmd_safety_max_xy_from_start_mm = max(
            0.0, float(self.get_parameter("cmd_safety_max_xy_from_start_mm").value)
        )
        self.cmd_safety_max_z_down_from_start_mm = max(
            0.0, float(self.get_parameter("cmd_safety_max_z_down_from_start_mm").value)
        )
        self.cmd_safety_max_z_up_from_start_mm = max(
            0.0, float(self.get_parameter("cmd_safety_max_z_up_from_start_mm").value)
        )
        self.cmd_safety_latch_on_start_limit = bool(
            self.get_parameter("cmd_safety_latch_on_start_limit").value
        )

        if abs(self.policy_z_offset_mm) > 1e-9 and self.action_type != "absolute":
            self.get_logger().warn(
                f"[POLICY-Z-OFFSET] policy_z_offset_mm={self.policy_z_offset_mm:.3f} was requested, "
                f"but action_type={self.action_type}. The offset is only applied for absolute action_type."
            )


        self.stall_sec = float(self.get_parameter("stall_sec").value)
        self.stall_min_after_start_sec = float(self.get_parameter("stall_min_after_start_sec").value)
        self.stall_lpf_tau_sec = float(self.get_parameter("stall_lpf_tau_sec").value)
        self.stall_window_net_pos_eps_mm = float(self.get_parameter("stall_window_net_pos_eps_mm").value)
        self.stall_window_net_ang_eps_rad = float(self.get_parameter("stall_window_net_ang_eps_rad").value)

        self.fz_kick_N = float(self.get_parameter("fz_kick_N").value)
        self.fz_kick_dur_sec = float(self.get_parameter("fz_kick_dur_sec").value)
        self.fz_kick_cooldown_sec = float(self.get_parameter("fz_kick_cooldown_sec").value)

        self.recover_enable = False  # RECOVER logic removed
        self.recover_cooldown_sec = float(self.get_parameter("recover_cooldown_sec").value)
        self.recover_timeout_sec = float(self.get_parameter("recover_timeout_sec").value)
        self.recover_pos_tol_mm = float(self.get_parameter("recover_pos_tol_mm").value)
        self.recover_ang_tol_rad = float(self.get_parameter("recover_ang_tol_rad").value)
        self.recover_ok_count = int(self.get_parameter("recover_ok_count").value)

        self.dither_enable = bool(self.get_parameter("dither_enable").value)
        self.dither_only_track = bool(self.get_parameter("dither_only_track").value)
        self.dither_min_after_start_sec = float(self.get_parameter("dither_min_after_start_sec").value)
        self.dither_win_sec = float(self.get_parameter("dither_win_sec").value)
        self.dither_sec = float(self.get_parameter("dither_sec").value)
        self.dither_net_pos_thr_mm = float(self.get_parameter("dither_net_pos_thr_mm").value)
        self.dither_net_ang_thr_rad = float(self.get_parameter("dither_net_ang_thr_rad").value)
        self.dither_path_ratio_thr = float(self.get_parameter("dither_path_ratio_thr").value)
        self.dither_rms_pos_thr_mm = float(self.get_parameter("dither_rms_pos_thr_mm").value)
        self.dither_rms_ang_thr_rad = float(self.get_parameter("dither_rms_ang_thr_rad").value)
        self.dither_decay = float(self.get_parameter("dither_decay").value)

        self.kick_max_before_recover = int(self.get_parameter("kick_max_before_recover").value)
        self.kick_reset_sec = float(self.get_parameter("kick_reset_sec").value)

        self.recover_check_lpf_tau_sec = float(self.get_parameter("recover_check_lpf_tau_sec").value)
        self.recover_use_overrides = bool(self.get_parameter("recover_use_overrides").value)
        self.recover_tau_sec = float(self.get_parameter("recover_tau_sec").value)
        self.recover_startup_ramp_sec = float(self.get_parameter("recover_startup_ramp_sec").value)
        self.recover_step_cap_pos_mm = float(self.get_parameter("recover_step_cap_pos_mm").value)
        self.recover_step_cap_ang_rad = float(self.get_parameter("recover_step_cap_ang_rad").value)
        self.recover_step_cap_fz = float(self.get_parameter("recover_step_cap_fz").value)
        self.recover_timeout_min_margin_sec = float(self.get_parameter("recover_timeout_min_margin_sec").value)
        self.recover_timeout_scale = float(self.get_parameter("recover_timeout_scale").value)

        # diffusion policy config
        self.diffusion_train_steps = int(self.get_parameter("diffusion_train_steps").value)
        self.diffusion_infer_steps = int(self.get_parameter("diffusion_infer_steps").value)
        self.diffusion_beta_start = float(self.get_parameter("diffusion_beta_start").value)
        self.diffusion_beta_end = float(self.get_parameter("diffusion_beta_end").value)
        self.diffusion_loss_type = str(self.get_parameter("diffusion_loss_type").value)

        # FLOW policy config
        self.flow_infer_steps = int(self.get_parameter("flow_infer_steps").value)
        self.flow_deterministic_noise = bool(self.get_parameter("flow_deterministic_noise").value)
        self.flow_noise_seed = int(self.get_parameter("flow_noise_seed").value)
        self.flow_train_eps = float(self.get_parameter("flow_train_eps").value)
        self.flow_loss_type = str(self.get_parameter("flow_loss_type").value)
        self.flow_obs_hidden_dim = int(self.get_parameter("flow_obs_hidden_dim").value)
        self.flow_image_feature_dim = int(self.get_parameter("flow_image_feature_dim").value)
        self.flow_global_cond_dim = int(self.get_parameter("flow_global_cond_dim").value)
        self.flow_time_embed_dim = int(self.get_parameter("flow_time_embed_dim").value)
        self.flow_down_dims = str(self.get_parameter("flow_down_dims").value)
        self.flow_kernel_size = int(self.get_parameter("flow_kernel_size").value)
        self.flow_n_groups = int(self.get_parameter("flow_n_groups").value)
        self.flow_cond_predict_scale = bool(self.get_parameter("flow_cond_predict_scale").value)
        self.image_backbone = str(self.get_parameter("image_backbone").value)
        self.dino_model_name = str(self.get_parameter("dino_model_name").value)
        self.dino_checkpoint_path = str(self.get_parameter("dino_checkpoint_path").value)
        self.freeze_image_backbone = bool(self.get_parameter("freeze_image_backbone").value)
        self.dino_roi_pooling = str(self.get_parameter("dino_roi_pooling").value)
        self.use_tcp_roi = bool(self.get_parameter("use_tcp_roi").value)
        self.tcp_roi_reference_width = int(self.get_parameter("tcp_roi_reference_width").value)
        self.tcp_roi_reference_height = int(self.get_parameter("tcp_roi_reference_height").value)
        self.tcp_roi_center_x = int(self.get_parameter("tcp_roi_center_x").value)
        self.tcp_roi_center_y = int(self.get_parameter("tcp_roi_center_y").value)
        self.tcp_roi_area_fraction = float(self.get_parameter("tcp_roi_area_fraction").value)

        self.use_stain_mask = bool(self.get_parameter("use_stain_mask").value)
        self.stain_mask_key = str(self.get_parameter("stain_mask_key").value)
        self.stain_pooling_type = str(self.get_parameter("stain_pooling_type").value)
        self.empty_stain_feature_mode = str(self.get_parameter("empty_stain_feature_mode").value)
        self.stain_mask_threshold = float(self.get_parameter("stain_mask_threshold").value)
        self.debug_stain_pooling = bool(self.get_parameter("debug_stain_pooling").value)
        if self.use_stain_mask and not self.stain_mask_topic:
            raise RuntimeError("use_stain_mask=True requires a non-empty stain_mask_topic")

        self.use_gripper = bool(self.get_parameter("use_gripper").value)
        self.gripper_position_topic = str(self.get_parameter("gripper_position_topic").value)
        self.gripper_current_topic = str(self.get_parameter("gripper_current_topic").value)
        self.use_gripper_history = bool(self.get_parameter("use_gripper_history").value)
        self.gripper_history_len = max(1, int(self.get_parameter("gripper_history_len").value))
        self.gripper_history_hz = max(
            1e-6,
            float(self.get_parameter("gripper_history_hz").value),
        )
        self.gripper_history_sync_slop_sec = max(
            0.0,
            float(self.get_parameter("gripper_history_sync_slop_sec").value),
        )
        self.gripper_history_max_age_sec = max(
            0.0,
            float(self.get_parameter("gripper_history_max_age_sec").value),
        )
        self.gripper_history_debug_every_n = max(
            1,
            int(self.get_parameter("gripper_history_debug_every_n").value),
        )
        self.gripper_command_topic = str(self.get_parameter("gripper_command_topic").value)
        self.gripper_goal_current_topic = str(self.get_parameter("gripper_goal_current_topic").value)
        self.gripper_command_min_tick = int(self.get_parameter("gripper_command_min_tick").value)
        self.gripper_command_max_tick = int(self.get_parameter("gripper_command_max_tick").value)
        self.gripper_command_deadband_tick = max(0, int(self.get_parameter("gripper_command_deadband_tick").value))
        self.gripper_command_slew_per_sec = max(0.0, float(self.get_parameter("gripper_command_slew_per_sec").value))
        self.gripper_command_step_cap_tick = max(0.0, float(self.get_parameter("gripper_command_step_cap_tick").value))
        self.gripper_cmd_safety_enable = bool(self.get_parameter("gripper_cmd_safety_enable").value)
        self.gripper_cmd_safety_max_tick_from_present = max(
            0.0,
            float(self.get_parameter("gripper_cmd_safety_max_tick_from_present").value),
        )
        self.gripper_goal_current_min_mA = max(0.0, float(self.get_parameter("gripper_goal_current_min_mA").value))
        self.gripper_goal_current_max_mA = max(
            self.gripper_goal_current_min_mA,
            float(self.get_parameter("gripper_goal_current_max_mA").value),
        )
        self.gripper_goal_current_deadband_mA = max(
            0.0,
            float(self.get_parameter("gripper_goal_current_deadband_mA").value),
        )
        if self.use_gripper and self.policy_class != "FLOW":
            raise RuntimeError("use_gripper=True currently requires policy_class=FLOW")
        if self.use_gripper_history and not self.use_gripper:
            raise RuntimeError("use_gripper_history=True requires use_gripper=True")
        if self.force_msg_type not in ("array", "wrench"):
            raise RuntimeError(f"force_msg_type must be array or wrench, got: {self.force_msg_type}")

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"[INFO] Using device: {self.device}")
        fz_limit_desc = (
            f"{self.fz_hard_limit:.3f}N"
            if self.fz_hard_limit > 0.0
            else "disabled"
        )
        self.get_logger().info(
            f"[CAM] preprocess_mode={self.camera_preprocess_mode}, "
            f"obs_mode={self.obs_mode}, camera_names={self.camera_names}, "
            f"alpha={self.camera_stabilize_alpha:.3f}, border={self.camera_stabilize_border_mode}, "
            f"jitter_report={self.camera_jitter_report_enable}, log_every={self.camera_jitter_log_every_n}"
        )
        self.get_logger().info(
            f"[STAIN] use_stain_mask={int(self.use_stain_mask)}, topic={self.stain_mask_topic or 'disabled'}, "
            f"pooling={self.stain_pooling_type}, empty={self.empty_stain_feature_mode}, "
            f"threshold={self.stain_mask_threshold:.3f}"
        )
        self.get_logger().info(
            f"[GRADCAM] enable={int(self.gradcam_enable)}, every_n_infer={self.gradcam_every_n_infer}, "
            f"target={self.gradcam_target}, step={self.gradcam_target_step}, horizon={self.gradcam_target_horizon}, "
            f"publish={int(self.gradcam_publish)}, save={int(self.gradcam_save)}"
        )
        self.get_logger().info(
            f"[FLOW-NOISE] deterministic={int(self.flow_deterministic_noise)}, "
            f"seed={self.flow_noise_seed}, control_gradcam_shared=1"
        )

        # validate paths / resolve checkpoint
        if not self.act_root or not os.path.isdir(self.act_root):
            raise RuntimeError(f"act_root invalid: {self.act_root}")
        if self.policy_class not in ("ACT", "DIFFUSION", "FLOW", "BSPLINE"):
            raise RuntimeError(f"policy_class must be ACT, DIFFUSION, FLOW, or BSPLINE, got: {self.policy_class}")
        if self.flow_deterministic_noise and self.policy_class != "FLOW":
            self.get_logger().warn(
                f"[FLOW-NOISE] deterministic noise requested for policy_class={self.policy_class}; ignored."
            )

        raw_ckpt_dir = self.ckpt_dir
        self.ckpt_dir = _resolve_checkpoint_dir(
            ckpt_dir=self.ckpt_dir,
            act_root=self.act_root,
            policy_class=self.policy_class,
            ckpt_auto_subdir=self.ckpt_auto_subdir,
        )
        if not os.path.isdir(self.ckpt_dir) or not os.path.exists(os.path.join(self.ckpt_dir, "policy_best.ckpt")):
            raise RuntimeError(
                f"ckpt_dir invalid: {self.ckpt_dir} "
                "(expected a directory containing policy_best.ckpt)"
            )
        if str(raw_ckpt_dir or "").strip():
            self.get_logger().info(f"[CKPT] resolved ckpt_dir: {raw_ckpt_dir} -> {self.ckpt_dir}")
        else:
            self.get_logger().info(f"[CKPT] ckpt_dir not provided -> auto latest: {self.ckpt_dir}")

        # stats
        self.stats = _load_dataset_stats(self.ckpt_dir)
        if self.stats is None:
            self.get_logger().warn("[STATS] dataset_stats.pkl missing/invalid -> disable normalize/denorm.")
            self.normalize_qpos_enabled = False
            self.denorm_action_enabled = False
        else:
            self.get_logger().info(
                f"[STATS] Loaded dataset_stats.pkl from {self.ckpt_dir} | "
                f"qpos_mode={self.stats.qpos_mode}, act_mode={self.stats.act_mode}"
            )
            if abs(float(self.stats.xyz_scale) - 1.0) > 1e-12:
                self.get_logger().warn(
                    f"[STATS] Applied xyz_unit_scale={float(self.stats.xyz_scale):.6g} "
                    "to qpos/action xyz stats for mm-compatible inference."
                )
            if self.stats.qpos_mode in ["minmax_01", "minmax_m11"]:
                self.get_logger().info(
                    f"[STATS] qpos_z_range=[{float(self.stats.qpos_a[2]):.3f},{float(self.stats.qpos_b[2]):.3f}] "
                    f"action_z_range=[{float(self.stats.act_a[2]):.3f},{float(self.stats.act_b[2]):.3f}] "
                    f"action_fz_range=[{float(self.stats.act_a[8]):.3f},{float(self.stats.act_b[8]):.3f}]"
                )

        self.action_dim = 11 if self.use_gripper else 9
        if self.use_gripper and self.stats is None:
            raise RuntimeError("use_gripper=True requires dataset_stats.pkl")
        if self.stats is not None:
            stats_action_dim = int(self.stats.act_a.size)
            if stats_action_dim != self.action_dim:
                raise RuntimeError(
                    f"checkpoint action_dim={stats_action_dim} does not match "
                    f"use_gripper={int(self.use_gripper)} expected action_dim={self.action_dim}"
                )
            if self.use_gripper and (
                self.stats.gripper_current_a is None or self.stats.gripper_current_b is None
            ):
                raise RuntimeError("use_gripper=True requires gripper_current_min/max in dataset_stats.pkl")
            if self.use_gripper and (
                self.stats.gripper_position_a is None or self.stats.gripper_position_b is None
            ):
                self.get_logger().warn(
                    "[STATS] dataset_stats.pkl has no gripper_position_min/max; "
                    "using raw gripper position for legacy checkpoint compatibility. "
                    "New checkpoints must include normalized gripper position stats."
                )
            elif self.use_gripper:
                self.get_logger().info(
                    "[STATS] gripper observations normalized with "
                    f"mode={self.stats.gripper_mode}, "
                    f"position_range=[{float(self.stats.gripper_position_a[0]):.3f},"
                    f"{float(self.stats.gripper_position_b[0]):.3f}], "
                    f"current_range=[{float(self.stats.gripper_current_a[0]):.3f},"
                    f"{float(self.stats.gripper_current_b[0]):.3f}]"
                )

        # demo-start pose for optional initial alignment
        self.demo_start_pose6: Optional[np.ndarray] = None
        if self.auto_move_to_demo_start:
            demo_xyz_scale = float(self.stats.xyz_scale) if self.stats is not None else 1.0
            self.demo_start_pose6 = _load_demo_start_pose_from_stats(self.ckpt_dir, xyz_scale=demo_xyz_scale)
            if self.demo_start_pose6 is None:
                self.get_logger().warn(
                    "[DEMO_START] auto_move_to_demo_start=True, but demo_start_pose_mean "
                    "was not found in dataset_stats.pkl. Alignment will be skipped."
                )
                self.auto_move_to_demo_start = False
            else:
                if self.orientation_lock_enable:
                    original_demo_rotvec = self.demo_start_pose6[3:6].copy()
                    self.demo_start_pose6[3:6] = self.orientation_lock_rotvec
                    self.get_logger().warn(
                        "[ORIENTATION-LOCK] enabled: measured orientation observation and "
                        "predicted orientation command will be ignored; "
                        f"fixed rotvec="
                        f"{np.array2string(self.orientation_lock_rotvec, precision=7, separator=', ')}; "
                        f"demo-start rotvec changed from "
                        f"{np.array2string(original_demo_rotvec, precision=7, separator=', ')}"
                    )
                align_target = self.demo_start_pose6.astype(np.float32).copy()
                align_target[2] += float(self.demo_start_z_offset_mm)
                self.get_logger().info(
                    "[DEMO_START] loaded demo_start_pose_mean "
                    f"[x y z wx wy wz]={np.array2string(self.demo_start_pose6, precision=4, separator=', ')}"
                )
                self.get_logger().info(
                    "[DEMO_START] alignment target = demo_start_pose_mean + optional world_Z_offset "
                    f"({self.demo_start_z_offset_mm:.3f} mm): "
                    f"{np.array2string(align_target, precision=4, separator=', ')}"
                )

        # ---------------------------------------------------------------
        # Stain-relative frame (stain_relative_frame package, step [5]).
        # The checkpoint's dataset_stats.pkl decides this, not a launch arg:
        # a policy trained on stain-relative coordinates cannot be driven
        # through the absolute path or vice versa.
        #   observation : absolute TCP pose -> stain-relative (x,y only)
        #   command     : stain-relative action -> absolute base command
        # use_relative_position=False -> identity transform, no subscription;
        # every existing checkpoint keeps its exact current behaviour.
        # ---------------------------------------------------------------
        self.rel_use_relative, self.rel_transform_version, self.obs_force_xy_zeroed = (
            _load_relative_frame_flags(self.ckpt_dir, act_root=self.act_root)
        )
        # Ablation override: force obs fx,fy to 0 (or keep them) regardless of
        # what the checkpoint stats say. "auto" (default) = follow the stats.
        _fxyz = str(self.get_parameter("force_obs_xy_zero").value).strip().lower()
        if _fxyz in ("true", "1", "yes", "on"):
            self.obs_force_xy_zeroed = True
        elif _fxyz in ("false", "0", "no", "off"):
            self.obs_force_xy_zeroed = False
        if _fxyz not in ("auto", ""):
            self.get_logger().warn(
                f"[FORCE-OBS] force_obs_xy_zero:={_fxyz} overrides the checkpoint's "
                f"observation_force_xy_zeroed -> obs_force_xy_zeroed={self.obs_force_xy_zeroed}"
            )
        self._srf = None
        self._srf_demo_start_absolutized = False
        if self.rel_use_relative:
            if StainOriginClient is None:
                raise RuntimeError(
                    "checkpoint dataset_stats.pkl has use_relative_position=True but "
                    "the stain_relative_frame package is not importable. Build/source "
                    "it, or run an absolute-frame checkpoint."
                )
            self._srf = StainOriginClient(
                self,
                use_relative=True,
                origin_topic=self.stain_origin_topic,
            )
            self.get_logger().warn(
                "[SRF] stain-relative checkpoint "
                f"(transform={self.rel_transform_version or 'unversioned'}, "
                f"obs_force_xy_zeroed={self.obs_force_xy_zeroed}). Waiting for a latched "
                f"origin on {self.stain_origin_topic} before demo-start alignment / inference."
            )
        # else: absolute-frame checkpoint -- self._srf stays None and every
        # _srf_* helper below short-circuits on `not self.rel_use_relative`,
        # so an absolute run is byte-for-byte unchanged (no import required,
        # no subscription, no extra log line).

        # policy
        self.policy = self._load_policy_and_ckpt_from_act_root()
        self._setup_gradcam_hooks()
        self._setup_metrics_logger()

        # -----------------------------
        # State buffers
        # -----------------------------
        self._lock = threading.Lock()
        self._pose6: Optional[np.ndarray] = None
        self._force: Optional[np.ndarray] = None
        self._gripper_position: Optional[int] = None
        self._gripper_current_mA: Optional[float] = None
        self._gripper_position_pending: Deque[Tuple[float, float]] = deque(maxlen=20)
        self._gripper_current_pending: Deque[Tuple[float, float]] = deque(maxlen=20)
        self._gripper_hist: Deque[np.ndarray] = deque(maxlen=self.gripper_history_len)
        self._gripper_last_pair_t: Optional[float] = None
        self._gripper_pair_count = 0
        self._gripper_pair_drop_count = 0
        self._gripper_last_pair_skew_sec = 0.0
        self._gripper_pair_period_ema: Optional[float] = None
        self._img_cam0: Optional[np.ndarray] = None
        self._img_cam1: Optional[np.ndarray] = None
        self._stain_mask: Optional[np.ndarray] = None

        # Online camera stabilization state. All jitter values are pixel units.
        self._cam_prev_raw_gray: Optional[np.ndarray] = None
        self._cam_prev_proc_gray: Optional[np.ndarray] = None
        self._cam_cum = np.zeros(3, dtype=np.float32)
        self._cam_smooth_cum = np.zeros(3, dtype=np.float32)
        self._cam_frame_count = 0
        self._cam_raw_jitter_ema = 0.0
        self._cam_proc_jitter_ema = 0.0

        self._force_hist: Deque[np.ndarray] = deque(maxlen=max(1, self.force_history_len))

        # Inference diagnostics: helps identify why no action plan is generated.
        self._infer_wait_last_log = 0.0
        self._infer_plan_count = 0
        self._flow_fixed_initial_noise: Optional[torch.Tensor] = None
        self._flow_noise_create_count = 0
        self._ctrl_no_plan_last_log = 0.0
        self._cmd_safety_last_log = 0.0
        self._cmd_safety_latched = False
        self._cmd_safety_latch_reason = ""
        self._cmd_safety_hold_pose6: Optional[np.ndarray] = None
        self._demo_start_safety_last_log = 0.0
        self._demo_start_wait_last_log = 0.0
        self._last_gripper_cmd: Optional[int] = None
        self._last_gripper_cmd_t: Optional[float] = None
        self._gripper_startup_position: Optional[float] = None
        self._gripper_cmd_safety_last_log = 0.0
        self._last_gripper_goal_current_mA: Optional[float] = None
        self._latest_flow_raw_seq: Optional[np.ndarray] = None
        self._latest_flow_pose6: Optional[np.ndarray] = None
        self._latest_flow_force3: Optional[np.ndarray] = None
        self._latest_flow_plan_t: float = 0.0
        self._flow_vector_overlay_count = 0
        self._flow_vector_overlay_fail_count = 0
        self._flow_vector_delta_log_count = 0
        self._flow_step_count = 0

        # baseline state
        self._sent_first_cmd = False
        self.prev_cmd: Optional[np.ndarray] = None
        self._t_start = _monotonic()
        self._t_first_pub = None

        self._start_pose6: Optional[np.ndarray] = None

        # Optional demo-start alignment state.
        # When auto_move_to_demo_start=False, these variables do not affect control.
        self._demo_start_align_done = not self.auto_move_to_demo_start
        self._demo_start_align_t0: Optional[float] = None
        self._demo_start_hold_t0: Optional[float] = None
        self._demo_start_from_pose6: Optional[np.ndarray] = None
        self._demo_start_effective_move_sec: Optional[float] = None
        self._ptp_alignment_requested = False

        # contact state
        self._contact = False
        self._last_contact = False
        self._contact_z_floor_mm: Optional[float] = None
        self._contact_z_block_count = 0
        self._preload_trigger_ok = 0
        self._preload_last_exit_t = -1e9
        self._approach_slow_active = False
        self._ptp9d_track_active = False
        self._ptp9d_inflight = False
        self._ptp9d_last_target9: Optional[np.ndarray] = None
        self._ptp9d_stream_started = False
        self._ptp9d_stream_tail9: Optional[np.ndarray] = None
        self._ptp9d_stream_committed_until_t = 0.0
        self._ptp9d_stream_topup_inflight = False
        self._ptp9d_stream_force_inflight = False
        self._ptp9d_stream_last_force_send_t = 0.0
        self._ptp9d_stream_last_sent_fz: Optional[float] = None
        self._ptp9d_stream_last_sent_contact: Optional[bool] = None

        self.stage = Stage.APPROACH

        # anchor
        self._anchor_ready = False
        self._anchor_offset6 = np.zeros(6, dtype=np.float32)

        # plan buffer
        self.plans: Deque[Plan] = deque(maxlen=max(1, self.max_plans))

        # touch baseline
        self._fz_base = 0.0
        self._fz_base_init = False
        self._touch_ok = 0

        # preload
        self._preload_t0 = 0.0
        self._preload_ok = 0
        self._preload_hold_pose6 = None
        self._preload_target_N = max(self.preload_min_N, 10.0)

        # release
        self._release_t0 = 0.0
        self._release_start_fz_cmd = 0.0

        # stall / kick / recover
        self._stall_pose6_lpf: Optional[np.ndarray] = None
        self._stall_win_pose6: Optional[np.ndarray] = None
        self._stall_win_t0: float = _monotonic()

        self._fz_kick_active: bool = False
        self._fz_kick_t0: float = 0.0
        self._fz_kick_last_end_t: float = -1e9

        self._recover_t0: float = 0.0
        self._recover_ok: int = 0
        self._recover_last_end_t: float = -1e9
        self._recover_timeout_eff: float = self.recover_timeout_sec

        self._recover_pose6_lpf: Optional[np.ndarray] = None

        # dither
        self.dt_control = 1.0 / max(1e-6, self.control_hz)
        self.dt_infer = 1.0 / max(1e-6, self.infer_hz)

        hist_len = max(4, int(max(0.2, self.dither_win_sec) * self.control_hz) + 2)
        self._pose_hist6 = deque(maxlen=hist_len)
        self._dither_score = 0.0
        self._kick_count = 0
        self._kick_count_t0 = _monotonic()

        # -----------------------------
        # ROS I/O
        # -----------------------------
        img_rel = _reliability_from_str(self.image_qos_str)
        img_qos = _qos(depth=1, reliability=img_rel)
        vec_qos = _qos(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.create_subscription(Float64MultiArray, self.pose_topic, self._on_pose, vec_qos)
        if self.force_msg_type == "wrench":
            self.create_subscription(Wrench, self.force_topic, self._on_force_wrench, vec_qos)
        else:
            self.create_subscription(Float64MultiArray, self.force_topic, self._on_force, vec_qos)
        self.create_subscription(Image, self.image_topic, self._on_img, img_qos)
        if self.use_global_image:
            self.create_subscription(Image, self.global_image_topic, self._on_global_img, img_qos)
        if self.use_stain_mask:
            self.create_subscription(Image, self.stain_mask_topic, self._on_stain_mask, img_qos)
        if self.use_gripper:
            self.create_subscription(Int32, self.gripper_position_topic, self._on_gripper_position, vec_qos)
            self.create_subscription(Float32, self.gripper_current_topic, self._on_gripper_current, vec_qos)

        self.pub_cmd = None
        self.pub_gripper_cmd = None
        self.pub_gripper_goal_current = None
        if not self.visualization_only:
            self.pub_cmd = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)

        self._ptp_client = None
        if (self.auto_move_to_demo_start and self.demo_start_use_ptp_service) or self.track_use_ptp9d_service:
            self._ptp_client = self.create_client(SingleArmCommand, self.ptp_service_name)
        self._ptp9d_client = self._ptp_client
        if self.track_use_ptp9d_service and self.ptp9d_service_name != self.ptp_service_name:
            self._ptp9d_client = self.create_client(SingleArmCommand, self.ptp9d_service_name)
        if self.use_gripper and not self.visualization_only:
            self.pub_gripper_cmd = self.create_publisher(Int32, self.gripper_command_topic, 10)
            self.pub_gripper_goal_current = self.create_publisher(
                Float32,
                self.gripper_goal_current_topic,
                10,
            )
        self.pub_gradcam_overlay = None
        self.pub_gradcam_global_overlay = None
        if self.gradcam_enable and self.gradcam_publish:
            self.pub_gradcam_overlay = self.create_publisher(Image, self.gradcam_overlay_topic, 1)
            self.get_logger().info(f"[GRADCAM] publishing local overlay image: {self.gradcam_overlay_topic}")
            if self.use_global_image:
                self.pub_gradcam_global_overlay = self.create_publisher(Image, self.gradcam_global_overlay_topic, 1)
                self.get_logger().info(f"[GRADCAM] publishing global overlay image: {self.gradcam_global_overlay_topic}")

        self.pub_modality_importance = None
        if self.modality_importance_enable:
            modality_qos = _qos(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.pub_modality_importance = self.create_publisher(
                Image, self.modality_importance_topic, modality_qos
            )
            self.get_logger().info(
                f"[MODALITY] publishing standalone dashboard: {self.modality_importance_topic}"
            )

        self.pub_flow_vector_overlay = None
        if self.flow_vector_overlay_enable:
            flow_overlay_qos = _qos(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.pub_flow_vector_overlay = self.create_publisher(
                Image, self.flow_vector_overlay_topic, flow_overlay_qos
            )
            self.get_logger().info(
                f"[FLOW-VECTOR] publishing local-camera overlay: "
                f"{self.flow_vector_overlay_topic}"
            )

        self.srv_flow_step = None
        if self.flow_step_service_enable:
            self.srv_flow_step = self.create_service(
                Trigger, self.flow_step_service, self._on_flow_step_service
            )
            self.get_logger().warn(
                f"[FLOW-STEP] manual service enabled: {self.flow_step_service}; "
                f"max_xyz={self.flow_step_max_xyz_mm:.3f}mm, "
                f"max_rot={self.flow_step_max_rot_rad:.6f}rad, automatic_control=OFF"
            )

        self.timer_control = None
        if not self.visualization_only:
            self.timer_control = self.create_timer(self.dt_control, self._on_control_timer)
        self.timer_infer = self.create_timer(self.dt_infer, self._on_infer_timer)

        if self.visualization_only:
            self.get_logger().warn(
                "[VISUALIZATION-ONLY] command publisher/control timer disabled; "
                "this node cannot move the robot"
            )
        elif self.clean_flow_execution:
            self.get_logger().warn(
                "[CLEAN-FLOW] legacy stall/kick/dither/recovery control is bypassed; "
                "trajectory replay, smoothing, contact monitoring, and command safety remain active"
            )

        self.get_logger().info(
            "[INFO] ✅ Ready.\n"
            f"  stage_start={self.stage.name}\n"
            f"  pose_topic={self.pose_topic}\n"
            f"  force_topic={self.force_topic} ({self.force_msg_type})\n"
            f"  obs_mode={self.obs_mode} camera_names={self.camera_names}\n"
            f"  image_topic={self.image_topic}\n"
            f"  global_image_topic={self.global_image_topic if self.use_global_image else '(disabled)'}\n"
            f"  cmd_topic={self.cmd_topic}\n"
            f"  stain_relative_frame={'ON (' + self.stain_origin_topic + ', obs_force_xy_zeroed=' + str(self.obs_force_xy_zeroed) + ')' if self.rel_use_relative else 'off (absolute-frame checkpoint)'}\n"
            f"  stain_canon={'enabled (train_angle=' + str(self.stain_canon_train_angle_deg) + 'deg, image_rotate=' + str(self.stain_canon_rotate_image) + ')' if self.stain_canon_enable else 'off'}\n"
            f"  gripper(enable={int(self.use_gripper)}, state=({self.gripper_position_topic}, {self.gripper_current_topic}), "
            f"position_cmd={self.gripper_command_topic if self.use_gripper else '(disabled)'}, "
            f"goal_current_cmd={self.gripper_goal_current_topic if self.use_gripper else '(disabled)'})\n"
            f"  image_qos={self.image_qos_str}\n"
            f"  policy_class={self.policy_class} phase_mode={self.phase_mode}\n"
            f"  control_hz={self.control_hz} infer_hz={self.infer_hz}\n"
            f"  use_force_history={int(self.use_force_history)} force_history_len={self.force_history_len}\n"
            f"  FLOW_NOISE(deterministic={int(self.flow_deterministic_noise)}, seed={self.flow_noise_seed}, control_gradcam_shared=1)\n"
            f"  gripper_history(enable={int(self.use_gripper_history)}, "
            f"hz={self.gripper_history_hz:.3f}, len={self.gripper_history_len}, "
            f"sync_slop={self.gripper_history_sync_slop_sec:.3f}s, "
            f"max_age={self.gripper_history_max_age_sec:.3f}s)\n"
            f"  diffusion_infer_steps={self.diffusion_infer_steps}\n"
            f"  tau_sec={self.tau_sec} startup_ramp_sec={self.startup_ramp_sec}\n"
            f"  step_caps(pos_mm={self.step_cap_pos_mm}, ang_rad={self.step_cap_ang_rad}, fz={self.step_cap_fz})\n"
            f"  action_selection={self.action_selection_mode} trajectory_hz={self.trajectory_hz:.3f} "
            f"temporal_agg={int(self.use_temporal_agg)} mode={self.temporal_agg_mode} "
            f"tau_steps={self.temporal_agg_tau_steps} max_plans={self.max_plans}\n"
            f"  FLOW_EXECUTOR(local_anchor={int(self.flow_local_anchor_enable)}, "
            f"replan_steps={self.flow_replan_interval_steps}, "
            f"replan_sec={self.flow_replan_interval_steps / self.trajectory_hz:.3f}, "
            f"clean={int(self.clean_flow_execution)})\n"
            f"  contact_gate(on={self.contact_on_thr}, off={self.contact_off_thr}) clear_on_change={int(self.clear_plans_on_contact_change)}\n"
            f"  CONTACT_Z_BLOCK(enable={int(self.contact_z_descent_block_enable)}, "
            f"margin_mm={self.contact_z_descent_margin_mm:.3f})\n"
            f"  force_xy_cmd(enable={int(self.force_xy_cmd_enable)}, hard_limit={self.force_xy_hard_limit}N)\n"
            f"  force_z_cmd(upper_limit={fz_limit_desc}, nonnegative=1)\n"
            f"  touch(delta={int(self.touch_use_delta)}, thr={self.touch_fz_thr}, ok={self.touch_ok_count}, min_after={self.touch_min_after_start_sec}s, base_tau={self.touch_baseline_tau_sec}s)\n"
            f"  PRELOAD(removed: bypass APPROACH -> TRACK, nominal_src={self.preload_target_source}, nominal_min={self.preload_min_N}N)\n"
            f"  STALL(win_sec={self.stall_sec}, min_after={self.stall_min_after_start_sec}s, lpf_tau={self.stall_lpf_tau_sec}s, net_eps_pos={self.stall_window_net_pos_eps_mm}mm, net_eps_ang={self.stall_window_net_ang_eps_rad}rad)\n"
            f"  KICK(fz={self.fz_kick_N}N/{self.fz_kick_dur_sec}s, cooldown={self.fz_kick_cooldown_sec}s)\n"
            f"  RECOVER(removed)\n"
            f"  DITHER(enable={int(self.dither_enable)}, only_track={int(self.dither_only_track)}, min_after={self.dither_min_after_start_sec}s, win={self.dither_win_sec}s, dur={self.dither_sec}s, net_pos_thr={self.dither_net_pos_thr_mm}mm, ratio_thr={self.dither_path_ratio_thr}, rms_pos_thr={self.dither_rms_pos_thr_mm}mm)\n"
            f"  RELEASE(enable={int(self.release_assist_enable)}, ramp_sec={self.release_ramp_sec})\n"
            f"  DEMO_START(auto={int(self.auto_move_to_demo_start)}, min_move_sec={self.demo_start_move_sec}, "
            f"hold_sec={self.demo_start_hold_sec}, z_offset_mm={self.demo_start_z_offset_mm}, "
            f"max_xyz_speed={self.demo_start_max_xyz_speed_mm_s}mm/s, "
            f"max_rot_speed={self.demo_start_max_rot_speed_rad_s}rad/s, "
            f"pos_tol={self.demo_start_position_tolerance_mm}mm, "
            f"rot_tol={self.demo_start_rotation_tolerance_rad}rad)\n"
            f"  DEMO_START_SAFETY(max_align_dist_mm={self.demo_start_max_align_dist_mm}; <=0 disables total-distance gate)\n"
            f"  POLICY_OUTPUT(z_offset_mm={self.policy_z_offset_mm})\n"
            f"  CMD_SAFETY(enable={int(self.cmd_safety_enable)}, max_xyz_from_current_mm={self.cmd_safety_max_xyz_from_current_mm}, "
            f"start_xy_mm={self.cmd_safety_max_xy_from_start_mm}, "
            f"start_z_down_mm={self.cmd_safety_max_z_down_from_start_mm}, "
            f"start_z_up_mm={self.cmd_safety_max_z_up_from_start_mm}, "
            f"latch={int(self.cmd_safety_latch_on_start_limit)})\n"
            f"  GRADCAM(enable={int(self.gradcam_enable)}, layer={self._gradcam_target_layer_name}, target={self.gradcam_target}, every_n_infer={self.gradcam_every_n_infer}, topic={self.gradcam_overlay_topic}, global_topic={self.gradcam_global_overlay_topic if self.use_global_image else '(disabled)'})\n"
            f"  MODALITY(enable={int(self.modality_importance_enable)}, target={self.modality_importance_target}, "
            f"step={self.modality_importance_target_step}, horizon={self.modality_importance_target_horizon}, "
            f"every_n_infer={self.modality_importance_every_n_infer}, topic={self.modality_importance_topic})\n"
            f"  FLOW_VECTOR(enable={int(self.flow_vector_overlay_enable)}, "
            f"topic={self.flow_vector_overlay_topic}, "
            f"horizons={self.flow_vector_overlay_horizons}, "
            f"selected={self.flow_vector_overlay_selected_horizon}, "
            f"projection=calibrated_linear(du_dx={self.flow_vector_overlay_m_du_dx:.3f},"
            f"du_dy={self.flow_vector_overlay_m_du_dy:.3f},"
            f"dv_dx={self.flow_vector_overlay_m_dv_dx:.3f},"
            f"dv_dy={self.flow_vector_overlay_m_dv_dy:.3f}))\n"
            f"  FLOW_DIAGNOSTIC_ONLY(enable={int(self.flow_diagnostic_only)}, "
            f"manual_step={int(self.flow_step_service_enable)}, "
            f"service={self.flow_step_service})\n"
            f"  ORIENTATION_LOCK(enable={int(self.orientation_lock_enable)}, "
            f"source=fixed_z90, "
            f"rotvec={np.array2string(self.orientation_lock_rotvec, precision=7, separator=', ')})\n"
        )

    def destroy_node(self):
        worker = getattr(self, "_modality_importance_worker_thread", None)
        if (
            worker is not None
            and worker.is_alive()
            and worker is not threading.current_thread()
        ):
            worker.join(timeout=2.0)
        metrics_file = getattr(self, "_metrics_csv_file", None)
        if metrics_file is not None:
            try:
                metrics_file.close()
            except Exception:
                pass
        return super().destroy_node()

    # ------------------------------------------------------------
    # Cross-policy comparison: per-tick CSV metrics log
    # ------------------------------------------------------------
    def _setup_metrics_logger(self):
        """Optionally record one CSV row per published command.

        Used to compare policy_class runs (e.g. FLOW vs BSPLINE) offline with
        scripts/compare_policy_runs.py: same task, same topics, one CSV per run.
        """
        self._metrics_csv_file = None
        self._metrics_csv_writer = None
        self._metrics_t0 = _monotonic()
        if not self.metrics_log_enable:
            return
        log_dir = self.metrics_log_dir or os.path.join(self.act_root, "logs", "inference_metrics")
        try:
            os.makedirs(log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = f"_{self.metrics_run_tag}" if self.metrics_run_tag else ""
            fname = f"{self.policy_class.lower()}{tag}_{ts}.csv"
            path = os.path.join(log_dir, fname)
            self._metrics_csv_file = open(path, "w", newline="")
            self._metrics_csv_writer = csv.writer(self._metrics_csv_file)
            self._metrics_csv_writer.writerow([
                "t_wall", "t_elapsed_sec", "stage", "policy_class", "ckpt_dir",
                "meas_x_mm", "meas_y_mm", "meas_z_mm", "meas_rx", "meas_ry", "meas_rz",
                "meas_fx_N", "meas_fy_N", "meas_fz_N",
                "cmd_x_mm", "cmd_y_mm", "cmd_z_mm", "cmd_rx", "cmd_ry", "cmd_rz",
                "cmd_fx_N", "cmd_fy_N", "cmd_fz_N",
                "contact", "cmd_safety_blocked",
            ])
            self._metrics_csv_file.flush()
            self.get_logger().info(f"[METRICS] logging per-tick comparison CSV -> {path}")
        except Exception as e:
            self.get_logger().error(f"[METRICS] failed to open metrics log: {e}")
            self._metrics_csv_file = None
            self._metrics_csv_writer = None

    def _log_metrics_row(self, cmd9: np.ndarray, blocked: bool):
        if self._metrics_csv_writer is None:
            return
        try:
            with self._lock:
                pose6 = None if self._pose6 is None else self._pose6.copy()
                force = None if self._force is None else self._force.copy()
            meas6 = pose6.astype(np.float32).tolist() if pose6 is not None else [float("nan")] * 6
            f3 = (
                self._extract_force3(force).astype(np.float32).tolist()
                if force is not None and force.size >= 3
                else [float("nan")] * 3
            )
            cmd = np.asarray(cmd9, dtype=np.float32).reshape(-1)[:9].tolist()
            self._metrics_csv_writer.writerow([
                time.time(),
                _monotonic() - self._metrics_t0,
                self.stage.name,
                self.policy_class,
                self.ckpt_dir,
                *meas6,
                *f3,
                *cmd,
                int(bool(self._contact)),
                int(bool(blocked)),
            ])
            self._metrics_csv_file.flush()
        except Exception as e:
            self.get_logger().error(f"[METRICS] failed to log row: {e}")
            self._metrics_csv_writer = None

    # ------------------------------------------------------------
    # Stain-relative frame helpers (no-op unless use_relative_position=True)
    # ------------------------------------------------------------
    def _srf_ready(self) -> bool:
        """True once a stain origin is available (always True in absolute mode)."""
        if not self.rel_use_relative:
            return True
        if self._srf is None:
            return False
        if self._srf.ready:
            if not self._canon_setup_done:
                self._canon_setup()
            if not self._srf_demo_start_absolutized:
                self._absolutize_demo_start_pose()
            return True
        now = _monotonic()
        if now - getattr(self, "_srf_wait_last_log", 0.0) >= 2.0:
            self._srf_wait_last_log = now
            self.get_logger().warn(
                f"[SRF] waiting for latched stain_origin on {self.stain_origin_topic} "
                "-- is stain_origin_node running with the arm at the home pose?"
            )
        return False

    def _canon_setup(self) -> None:
        """Resolve the rotation that maps the live stain axis onto the trained
        one, once, when the frozen origin+angle arrive."""
        self._canon_setup_done = True
        self._canon_active = False
        self._canon_alpha = 0.0
        if not (self.rel_use_relative and self.stain_canon_enable):
            return
        if self.stain_canon_live_angle_deg >= 0.0:
            theta = math.radians(self.stain_canon_live_angle_deg)
            self.get_logger().warn(
                f"[CANON] using operator-supplied live stain angle "
                f"{self.stain_canon_live_angle_deg:.1f}deg (detection bypassed)"
            )
        else:
            theta = getattr(self._srf, "stain_angle", None)
        if theta is None:
            self.get_logger().warn(
                "[CANON] stain_canon_enable=True but no strip angle available "
                "(dark method publishes one; or pass stain_canon_live_angle_deg). "
                "Rotation canonicalization OFF."
            )
            return
        alpha = _wrap_axis_delta(math.radians(self.stain_canon_train_angle_deg) - float(theta))
        lim = math.radians(self.stain_canon_max_angle_deg) if self.stain_canon_max_angle_deg > 0 else None
        if lim is not None and abs(alpha) > lim:
            self.get_logger().error(
                f"[CANON] live stain axis {math.degrees(theta):.1f}deg needs alpha="
                f"{math.degrees(alpha):+.1f}deg, beyond stain_canon_max_angle_deg="
                f"{self.stain_canon_max_angle_deg:.0f}. CANONICALIZATION OFF -- the policy "
                f"will now stroke in the TRAINED direction (~{self.stain_canon_train_angle_deg:.0f}"
                "deg), NOT along your stain. Ctrl-C now if that's wrong; else raise "
                "stain_canon_max_angle_deg (image-rotation artifacts grow with it) or draw "
                "the stain nearer the trained direction."
            )
            return
        self._canon_alpha = float(alpha)
        self._canon_active = True
        # Safety: canon rotates the demo-start x,y about the (possibly far)
        # stain origin -- a big alpha swings it tens of mm. The inference
        # launches ship demo_start_max_align_dist_mm=0 (gate off); force a
        # finite gate whenever canon is active so a mis-detected angle can't
        # drive a violent alignment PTP.
        if self.demo_start_max_align_dist_mm <= 0.0:
            self.demo_start_max_align_dist_mm = 120.0
            self.get_logger().warn(
                "[CANON] demo_start_max_align_dist_mm was 0 (gate off) -> forcing 120mm "
                "while canon is active"
            )
        self.get_logger().warn(
            f"[CANON] live stain axis={math.degrees(theta):.1f}deg, trained="
            f"{self.stain_canon_train_angle_deg:.1f}deg -> rotating obs x,y by "
            f"{math.degrees(alpha):+.1f}deg (image_rotate={self.stain_canon_rotate_image}, "
            f"sign={self.stain_canon_image_angle_sign:+.0f}); orientation NOT rotated"
        )

    def _canon_rotate_image(self, rgb: np.ndarray) -> np.ndarray:
        """In-plane rotation of the policy's camera frame about the TCP-ROI
        centre so the live stain presents at the trained direction."""
        if not self._canon_active or not self.stain_canon_rotate_image or cv2 is None:
            return rgb
        h, w = rgb.shape[:2]
        cx = float(np.clip(self.flow_vector_overlay_tcp_center_x, 0, w - 1))
        cy = float(np.clip(self.flow_vector_overlay_tcp_center_y, 0, h - 1))
        deg = self.stain_canon_image_angle_sign * math.degrees(self._canon_alpha)
        M = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
        return cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

    def _canon_obs_pose6(self, p6: np.ndarray) -> np.ndarray:
        """Rotate a stain-relative pose into the canonical (trained-direction)
        frame: x,y only (Rz(alpha)). The tool orientation channel is left
        untouched -- the polishing pad is axisymmetric, so its yaw carries no
        task signal, and composing it through rotvec<->matrix near the +/-pi
        wrap flipped the representation and commanded a violent wrist move
        (20260910 canon_rot45 test). z / rotation / force pass through."""
        if not self._canon_active:
            return p6
        out = np.asarray(p6, dtype=np.float64).reshape(-1).copy()
        c, s = math.cos(self._canon_alpha), math.sin(self._canon_alpha)
        x, y = out[0], out[1]
        out[0], out[1] = c * x - s * y, s * x + c * y
        return out.astype(np.float32)

    def _canon_cmd_seq(self, seq_canon: np.ndarray) -> np.ndarray:
        """Inverse of _canon_obs_pose6 on a whole (T,>=6) trajectory: x,y only
        (Rz(-alpha)). Orientation and force columns untouched."""
        if not self._canon_active:
            return seq_canon
        out = seq_canon
        c, s = math.cos(-self._canon_alpha), math.sin(-self._canon_alpha)
        x = out[:, 0].copy(); y = out[:, 1].copy()
        out[:, 0] = c * x - s * y
        out[:, 1] = s * x + c * y
        return out

    def _absolutize_demo_start_pose(self) -> None:
        """demo_start_pose_mean is stored in the training (canonical) stain
        frame; rotate its x,y back to the live stain direction, then translate
        to absolute base coordinates so auto_move_to_demo_start PTPs correctly.
        Orientation is NOT rotated (see _canon_obs_pose6)."""
        if self._srf_demo_start_absolutized or self.demo_start_pose6 is None:
            self._srf_demo_start_absolutized = True
            return
        canon = self.demo_start_pose6.astype(np.float64).copy()[:6].reshape(1, 6)
        rel = self._canon_cmd_seq(canon)                       # x,y: Rz(-alpha)
        absolute = np.asarray(self._srf.command(rel[0]), dtype=np.float64).reshape(-1)
        self.demo_start_pose6[:2] = absolute[:2].astype(self.demo_start_pose6.dtype)
        self._srf_demo_start_absolutized = True
        self.get_logger().warn(
            "[SRF] demo_start_pose_mean -> absolute via stain_origin "
            f"{np.array2string(self._srf.stain_origin, precision=3, separator=', ')}"
            + (f" + canon x,y {math.degrees(self._canon_alpha):+.1f}deg" if self._canon_active else "")
            + f": {np.array2string(self.demo_start_pose6, precision=4, separator=', ')}"
        )

    def _srf_observation_pose6(self, pose6: np.ndarray) -> np.ndarray:
        """Absolute measured TCP pose -> what the policy was trained on
        (stain-relative, then rotation-canonical if enabled)."""
        if not self.rel_use_relative:
            return pose6
        rel = np.asarray(self._srf.observation(pose6.astype(np.float64)), dtype=np.float32)
        return self._canon_obs_pose6(rel.reshape(-1)[:6])

    def _srf_command_seq(self, seq_den: np.ndarray) -> np.ndarray:
        """Policy trajectory (canonical stain frame) -> absolute base commands:
        Rz(-alpha) into the stain-relative frame, then + origin. z/force pass
        through untouched (relative_frame.to_absolute contract)."""
        if not self.rel_use_relative:
            return seq_den
        seq_den = self._canon_cmd_seq(seq_den)                 # canon -> relative
        absolute_xy = np.asarray(
            self._srf.command(seq_den[:, :6].astype(np.float64)), dtype=np.float32
        )
        seq_den[:, 0] = absolute_xy[:, 0]
        seq_den[:, 1] = absolute_xy[:, 1]
        return seq_den

    # ------------------------------------------------------------
    # Small helpers (force extraction / history)
    # ------------------------------------------------------------
    def _extract_force3(self, raw_force: np.ndarray) -> np.ndarray:
        idx = list(self.force_indices)
        f3 = np.zeros(3, dtype=np.float32)
        for i, k in enumerate(idx):
            if k < raw_force.size:
                f3[i] = float(raw_force[k])
        return f3

    def _build_live_force_history(self, hist_list: List[np.ndarray], current_force3: np.ndarray) -> np.ndarray:
        """
        Returns (L,3), padded like dataset.py:
        if insufficient history, repeat the first available force on the left.
        """
        L = max(1, self.force_history_len)

        if len(hist_list) <= 0:
            hist = current_force3.reshape(1, 3).astype(np.float32)
        else:
            hist = np.stack(hist_list, axis=0).astype(np.float32)

        hist = hist[-L:]  # keep most recent L

        if hist.shape[0] < L:
            pad_count = L - hist.shape[0]
            pad_value = hist[0:1] if hist.shape[0] > 0 else current_force3.reshape(1, 3).astype(np.float32)
            pad = np.repeat(pad_value, pad_count, axis=0)
            hist = np.concatenate([pad, hist], axis=0)

        return hist.astype(np.float32)

    def _build_live_gripper_history(
        self,
        hist_list: List[np.ndarray],
        current_pair: np.ndarray,
    ) -> np.ndarray:
        """Return the newest L synchronized pairs with left-edge repetition."""
        length = max(1, self.gripper_history_len)
        if hist_list:
            history = np.stack(hist_list, axis=0).astype(np.float32)[-length:]
        else:
            history = np.asarray(current_pair, dtype=np.float32).reshape(1, 2)
        if history.shape[0] < length:
            pad = np.repeat(history[0:1], length - history.shape[0], axis=0)
            history = np.concatenate([pad, history], axis=0)
        return history.astype(np.float32)

    # ------------------------------------------------------------
    # Grad-CAM debug helpers
    # ------------------------------------------------------------
    def _flow_initial_noise_for_plan(self, q_t: torch.Tensor) -> Optional[torch.Tensor]:
        """Create one FLOW initial-noise tensor shared by control and Grad-CAM."""
        if self.policy_class != "FLOW" or self.use_gripper:
            return None

        batch = int(q_t.shape[0])
        horizon = int(getattr(self.policy, "num_queries", self.chunk_size))
        action_dim = int(getattr(self.policy, "action_dim", self.action_dim))
        shape = (batch, horizon, action_dim)

        if self.flow_deterministic_noise:
            cached = self._flow_fixed_initial_noise
            if (
                cached is None
                or tuple(cached.shape) != shape
                or cached.device != q_t.device
                or cached.dtype != q_t.dtype
            ):
                generator = torch.Generator(device=q_t.device)
                generator.manual_seed(int(self.flow_noise_seed) % (2**63 - 1))
                cached = torch.randn(
                    shape,
                    device=q_t.device,
                    dtype=q_t.dtype,
                    generator=generator,
                )
                self._flow_fixed_initial_noise = cached.detach()
                self.get_logger().warn(
                    f"[FLOW-NOISE] initialized fixed noise seed={self.flow_noise_seed} "
                    f"shape={shape} device={q_t.device}"
                )
            noise = self._flow_fixed_initial_noise
        else:
            noise = torch.randn(shape, device=q_t.device, dtype=q_t.dtype)

        self._flow_noise_create_count += 1
        return noise

    def _setup_gradcam_hooks(self):
        if not self.gradcam_enable:
            return

        layer = _find_module_by_name(self.policy, self.gradcam_layer_name)
        layer_name = self.gradcam_layer_name
        if layer is None:
            layer_name, layer = _find_last_conv2d(self.policy)

        if layer is None:
            self.get_logger().warn("[GRADCAM] no Conv2d layer found in policy. Grad-CAM disabled.")
            self.gradcam_enable = False
            return

        self._gradcam_target_layer = layer
        self._gradcam_target_layer_name = str(layer_name)

        def _fwd_hook(_module, _inp, out):
            if torch.is_tensor(out):
                self._gradcam_activation = out.detach()
            elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                self._gradcam_activation = out[0].detach()
            else:
                self._gradcam_activation = None

        def _bwd_hook(_module, _grad_input, grad_output):
            if isinstance(grad_output, (tuple, list)) and len(grad_output) > 0 and torch.is_tensor(grad_output[0]):
                self._gradcam_gradient = grad_output[0].detach()
            elif torch.is_tensor(grad_output):
                self._gradcam_gradient = grad_output.detach()
            else:
                self._gradcam_gradient = None

        self._gradcam_fwd_handle = layer.register_forward_hook(_fwd_hook)
        try:
            self._gradcam_bwd_handle = layer.register_full_backward_hook(_bwd_hook)
        except Exception:
            self._gradcam_bwd_handle = layer.register_backward_hook(_bwd_hook)

        self.get_logger().warn(
            f"[GRADCAM] enabled. target_layer='{self._gradcam_target_layer_name}', "
            f"target={self.gradcam_target}, every_n_infer={self.gradcam_every_n_infer}"
        )

    def _select_action_scalar(
        self,
        seq: torch.Tensor,
        *,
        target: str,
        target_step: int,
        target_horizon: int,
    ) -> torch.Tensor:
        if seq.dim() != 2 or seq.shape[-1] not in (9, 10, 11):
            raise RuntimeError(
                f"attribution seq must be (T,9), (T,10), or (T,11), got {tuple(seq.shape)}"
            )

        T = int(seq.shape[0])
        s = min(max(0, int(target_step)), max(0, T - 1))
        e = min(T, s + max(1, int(target_horizon)))
        block = seq[s:e]
        target = str(target or "z").strip().lower()

        if target in ("x", "cmd_x"):
            return block[:, 0].mean()
        if target in ("y", "cmd_y"):
            return block[:, 1].mean()
        if target in ("z", "cmd_z"):
            return block[:, 2].mean()
        if target in ("wx", "rx", "roll"):
            return block[:, 3].mean()
        if target in ("wy", "ry", "pitch"):
            return block[:, 4].mean()
        if target in ("wz", "rz", "yaw"):
            return block[:, 5].mean()
        if target in ("fx", "cmd_fx"):
            return block[:, 6].mean()
        if target in ("fy", "cmd_fy"):
            return block[:, 7].mean()
        if target in ("fz", "cmd_fz"):
            return block[:, 8].mean()
        if target in ("gripper", "grip", "tick", "gripper_position"):
            if block.shape[-1] < 10:
                raise RuntimeError("gradcam_target=gripper requires a gripper policy output")
            return block[:, 9].mean()
        if target in ("gripper_current", "goal_current", "gripper_goal_current"):
            if block.shape[-1] < 11:
                raise RuntimeError("gradcam_target=gripper_current requires an 11D gripper policy output")
            return block[:, 10].mean()
        if target in ("abs_z", "z_abs"):
            return block[:, 2].abs().mean()
        if target in ("abs_fz", "fz_abs"):
            return block[:, 8].abs().mean()
        if target in ("xyz_norm", "pos_norm", "position_norm"):
            return torch.linalg.norm(block[:, 0:3], dim=-1).mean()
        if target in ("rot_norm", "ori_norm", "orientation_norm"):
            return torch.linalg.norm(block[:, 3:6], dim=-1).mean()
        if target in ("force_norm", "f_norm"):
            return torch.linalg.norm(block[:, 6:9], dim=-1).mean()
        if target in ("action_norm", "all_norm"):
            return torch.linalg.norm(block[:, 0:9], dim=-1).mean()
        return block[:, 2].mean()

    def _select_gradcam_scalar(self, seq_phys: torch.Tensor) -> torch.Tensor:
        return self._select_action_scalar(
            seq_phys,
            target=self.gradcam_target,
            target_step=self.gradcam_target_step,
            target_horizon=self.gradcam_target_horizon,
        )

    def _gradcam_policy_forward(
        self,
        q_gc: torch.Tensor,
        img_gc: torch.Tensor,
        fh_gc: Optional[torch.Tensor],
        stain_mask_gc: Optional[torch.Tensor],
        flow_initial_noise_gc: Optional[torch.Tensor],
        gripper_position_gc: Optional[torch.Tensor],
        gripper_current_gc: Optional[torch.Tensor],
        gripper_history_gc: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.policy_class == "FLOW":
            if self.use_gripper:
                if hasattr(self.policy, "sample_action_with_grad"):
                    return self.policy.sample_action_with_grad(
                        qpos=q_gc,
                        image=img_gc,
                        force_history=fh_gc,
                        gripper_position=gripper_position_gc,
                        gripper_current=gripper_current_gc,
                        gripper_history=gripper_history_gc,
                    )

                if hasattr(self.policy, "predict_velocity"):
                    steps = max(1, int(getattr(self.policy, "flow_infer_steps", self.flow_infer_steps)))
                    B = int(q_gc.shape[0])
                    T = int(getattr(self.policy, "num_queries", self.chunk_size))
                    Da = int(getattr(self.policy, "action_dim", self.action_dim))
                    z = (
                        flow_initial_noise_gc.to(device=q_gc.device, dtype=q_gc.dtype).clone()
                        if flow_initial_noise_gc is not None
                        else torch.randn(B, T, Da, device=q_gc.device, dtype=q_gc.dtype)
                    )
                    dt = 1.0 / float(steps)
                    for k in range(steps):
                        t = torch.full((B,), (k + 0.5) / float(steps), device=q_gc.device, dtype=q_gc.dtype)
                        v = self.policy.predict_velocity(
                            z_t=z,
                            t=t,
                            qpos=q_gc,
                            image=img_gc,
                            force_history=fh_gc,
                            gripper_position=gripper_position_gc,
                            gripper_current=gripper_current_gc,
                            gripper_history=gripper_history_gc,
                        )
                        z = z + dt * v
                    return z

            if hasattr(self.policy, "sample_action_with_grad"):
                if self.use_force_history:
                    return self.policy.sample_action_with_grad(
                        qpos=q_gc,
                        image=img_gc,
                        force_history=fh_gc,
                        stain_mask=stain_mask_gc,
                        initial_noise=flow_initial_noise_gc,
                    )
                return self.policy.sample_action_with_grad(
                    qpos=q_gc,
                    image=img_gc,
                    stain_mask=stain_mask_gc,
                    initial_noise=flow_initial_noise_gc,
                )

            if hasattr(self.policy, "predict_velocity"):
                steps = max(1, int(getattr(self.policy, "flow_infer_steps", self.flow_infer_steps)))
                B = int(q_gc.shape[0])
                T = int(getattr(self.policy, "num_queries", self.chunk_size))
                Da = int(getattr(self.policy, "action_dim", 9))
                z = (
                    flow_initial_noise_gc.to(device=q_gc.device, dtype=q_gc.dtype).clone()
                    if flow_initial_noise_gc is not None
                    else torch.randn(B, T, Da, device=q_gc.device, dtype=q_gc.dtype)
                )
                dt = 1.0 / float(steps)
                for k in range(steps):
                    t = torch.full((B,), (k + 0.5) / float(steps), device=q_gc.device, dtype=q_gc.dtype)
                    if self.use_force_history:
                        v = self.policy.predict_velocity(
                            z_t=z,
                            t=t,
                            qpos=q_gc,
                            image=img_gc,
                            force_history=fh_gc,
                            stain_mask=stain_mask_gc,
                        )
                    else:
                        v = self.policy.predict_velocity(
                            z_t=z,
                            t=t,
                            qpos=q_gc,
                            image=img_gc,
                            stain_mask=stain_mask_gc,
                        )
                    z = z + dt * v
                return z

        if self.policy_class == "BSPLINE" and hasattr(self.policy, "predict_trajectory"):
            # BSPLINE's forward(actions=None) dispatches to sample_action(),
            # which is @torch.no_grad()-decorated (inference-only fast path)
            # -- that inner decorator overrides this function's outer
            # enable_grad(), so Grad-CAM's backward pass never gets a graph.
            # predict_trajectory() itself carries no such decorator; call it
            # directly instead.
            if self.use_force_history:
                return self.policy.predict_trajectory(q_gc, img_gc, force_history=fh_gc, stain_mask=stain_mask_gc)
            return self.policy.predict_trajectory(q_gc, img_gc, stain_mask=stain_mask_gc)

        if self.use_force_history:
            return self.policy(q_gc, img_gc, force_history=fh_gc, stain_mask=stain_mask_gc)
        return self.policy(q_gc, img_gc, stain_mask=stain_mask_gc)

    def _run_gradcam_debug(
        self,
        images_rgb: List[np.ndarray],
        q_t: torch.Tensor,
        img_t: torch.Tensor,
        force_hist_t: Optional[torch.Tensor],
        stain_mask_t: Optional[torch.Tensor],
        flow_initial_noise_t: Optional[torch.Tensor],
        gripper_position_t: Optional[torch.Tensor] = None,
        gripper_current_t: Optional[torch.Tensor] = None,
        gripper_history_t: Optional[torch.Tensor] = None,
    ) -> bool:
        if not self.gradcam_enable:
            return False
        if self._gradcam_target_layer is None:
            return False
        if self._infer_plan_count <= 0 or (self._infer_plan_count % self.gradcam_every_n_infer) != 0:
            return False

        self._gradcam_activation = None
        self._gradcam_gradient = None
        gradcam_params = None
        gradcam_param_states = None
        backbone_module = None
        backbone_freeze_prev = None
        local_published = False

        try:
            was_training = self.policy.training
            self.policy.eval()
            self.policy.zero_grad(set_to_none=True)
            gradcam_params = list(self.policy.parameters())
            gradcam_param_states = [p.requires_grad for p in gradcam_params]
            for p in gradcam_params:
                p.requires_grad_(False)

            # A frozen vision backbone (e.g. DINOv3Backbone) runs its own
            # forward under torch.no_grad() internally for efficiency (see
            # dinov3_backbone.py) -- that inner no_grad breaks the graph back
            # to the input image regardless of this function's outer
            # enable_grad(), so Grad-CAM would otherwise always see
            # requires_grad=False at the target layer. Temporarily disable
            # just that internal no_grad for this one backward pass; params
            # already had requires_grad forced False above, so this doesn't
            # risk an accidental backbone weight update.
            backbone_attr_name = (self._gradcam_target_layer_name or "").split(".model.")[0]
            if backbone_attr_name:
                try:
                    backbone_module = self.policy.get_submodule(backbone_attr_name)
                except Exception:
                    backbone_module = None
            if backbone_module is not None and isinstance(getattr(backbone_module, "freeze", None), bool):
                backbone_freeze_prev = backbone_module.freeze
                backbone_module.freeze = False

            q_gc = q_t.detach().clone()
            img_gc = img_t.detach().clone()
            img_gc.requires_grad_(True)
            fh_gc = None if force_hist_t is None else force_hist_t.detach().clone()
            stain_mask_gc = None if stain_mask_t is None else stain_mask_t.detach().clone()
            flow_initial_noise_gc = (
                None if flow_initial_noise_t is None else flow_initial_noise_t.detach().clone()
            )
            gp_gc = None if gripper_position_t is None else gripper_position_t.detach().clone()
            gc_gc = None if gripper_current_t is None else gripper_current_t.detach().clone()
            gh_gc = None if gripper_history_t is None else gripper_history_t.detach().clone()

            with torch.enable_grad():
                out = self._gradcam_policy_forward(
                    q_gc=q_gc,
                    img_gc=img_gc,
                    fh_gc=fh_gc,
                    stain_mask_gc=stain_mask_gc,
                    flow_initial_noise_gc=flow_initial_noise_gc,
                    gripper_position_gc=gp_gc,
                    gripper_current_gc=gc_gc,
                    gripper_history_gc=gh_gc,
                )
                seq = _fix_policy_output_seq(out, self.chunk_size, self.policy_class, action_dim=self.action_dim)
                if self.denorm_action_enabled and self.stats is not None:
                    seq_phys = _denorm_action_seq(seq, self.stats)
                else:
                    seq_phys = seq

                scalar = self._select_gradcam_scalar(seq_phys)
                if not torch.is_tensor(scalar) or not scalar.requires_grad:
                    raise RuntimeError("selected Grad-CAM scalar does not require grad")
                scalar.backward(retain_graph=False)

            if was_training:
                self.policy.train()

            act = self._gradcam_activation
            grad = self._gradcam_gradient
            if act is None or grad is None:
                raise RuntimeError("activation/gradient was not captured from target layer")
            if act.dim() != 4 or grad.dim() != 4:
                raise RuntimeError(f"expected Conv2d activation/gradient (N,C,H,W), got act={tuple(act.shape)}, grad={tuple(grad.shape)}")

            self._gradcam_pub_count += 1
            num_cam_maps = min(len(images_rgb), int(act.shape[0]), int(grad.shape[0]))
            if num_cam_maps <= 0:
                raise RuntimeError("no camera Grad-CAM maps available")

            published_names = []
            heat_shapes = []
            stamp = self.get_clock().now().to_msg()

            for cam_i in range(num_cam_maps):
                weights = grad[cam_i:cam_i + 1].mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * act[cam_i:cam_i + 1]).sum(dim=1, keepdim=False))[0]
                heat = _normalize_heatmap_np(cam.detach().float().cpu().numpy())
                if cam_i == 0:
                    self._latest_gradcam_heat = heat
                overlay = _make_gradcam_overlay_rgb(
                    images_rgb[cam_i],
                    heat,
                    alpha=self.gradcam_alpha,
                    colormap=self.gradcam_colormap,
                )
                cam_name = self.camera_names[cam_i] if cam_i < len(self.camera_names) else f"cam{cam_i}"
                heat_shapes.append(f"{cam_name}:{tuple(heat.shape)}")

                if self.gradcam_publish:
                    if cam_i == 0 and self.pub_gradcam_overlay is not None:
                        msg = _rgb_numpy_to_image_msg(overlay, stamp=stamp, frame_id=f"gradcam_{cam_name}")
                        self.pub_gradcam_overlay.publish(msg)
                        published_names.append(cam_name)
                        local_published = True
                    elif cam_i == 1 and self.pub_gradcam_global_overlay is not None:
                        msg = _rgb_numpy_to_image_msg(overlay, stamp=stamp, frame_id=f"gradcam_{cam_name}")
                        self.pub_gradcam_global_overlay.publish(msg)
                        published_names.append(cam_name)

                if self.gradcam_save:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    fname = f"gradcam_{ts}_{self._gradcam_pub_count:06d}_{cam_name}_{self.gradcam_target}.png"
                    out_path = os.path.join(self.gradcam_save_dir, fname)
                    if cv2 is not None:
                        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    else:
                        from PIL import Image as PILImage
                        PILImage.fromarray(overlay).save(out_path)

            if self._gradcam_pub_count <= 3 or (self._gradcam_pub_count % self.gradcam_log_every_n == 0):
                self.get_logger().info(
                    f"[GRADCAM] #{self._gradcam_pub_count} target={self.gradcam_target} "
                    f"scalar={float(scalar.detach().cpu()):.6f} layer={self._gradcam_target_layer_name} "
                    f"heat_shape={';'.join(heat_shapes)} publish={int(self.gradcam_publish)} "
                    f"published={published_names} save={int(self.gradcam_save)}"
                )
            return local_published

        except Exception as e:
            self._gradcam_fail_count += 1
            now_t = _monotonic()
            if self._gradcam_fail_count <= 3 or (now_t - self._gradcam_last_log_t) > 2.0:
                self._gradcam_last_log_t = now_t
                self.get_logger().warn(f"[GRADCAM] failed #{self._gradcam_fail_count}: {e}")
            return False
        finally:
            try:
                self.policy.zero_grad(set_to_none=True)
            except Exception:
                pass
            if gradcam_params is not None and gradcam_param_states is not None:
                for p, req in zip(gradcam_params, gradcam_param_states):
                    p.requires_grad_(req)
            if backbone_module is not None and backbone_freeze_prev is not None:
                backbone_module.freeze = backbone_freeze_prev
            self._gradcam_activation = None
            self._gradcam_gradient = None

    def _render_modality_importance_rgb(
        self,
        *,
        raw_scores: np.ndarray,
        raw_percent: np.ndarray,
        smooth_percent: np.ndarray,
    ) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for modality-importance visualization")

        width, height = 920, 600
        canvas = np.full((height, width, 3), 248, dtype=np.uint8)
        names = ("POSITION", "FORCE", "IMAGE")
        colors = ((52, 152, 219), (230, 126, 34), (46, 180, 100))

        cv2.putText(
            canvas,
            "LIVE OBSERVATION MODALITY IMPORTANCE",
            (32, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.88,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        method_label = (
            "batched FLOW velocity change at t=0.5"
            if self.policy_class == "FLOW"
            else "batched BSPLINE trajectory change"
        )
        subtitle = (
            f"{method_label} | target={self.modality_importance_target} "
            f"steps={self.modality_importance_target_step}:"
            f"{self.modality_importance_target_step + self.modality_importance_target_horizon}"
        )
        cv2.putText(
            canvas, subtitle, (32, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (70, 70, 70), 1, cv2.LINE_AA,
        )

        bar_x0, bar_x1 = 205, 850
        for idx, (name, color) in enumerate(zip(names, colors)):
            y = 118 + idx * 82
            cv2.putText(
                canvas, name, (32, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.66,
                (25, 25, 25), 2, cv2.LINE_AA,
            )
            cv2.rectangle(canvas, (bar_x0, y), (bar_x1, y + 31), (220, 220, 220), -1)
            fill_x = bar_x0 + int(round((bar_x1 - bar_x0) * float(smooth_percent[idx])))
            cv2.rectangle(canvas, (bar_x0, y), (fill_x, y + 31), color, -1)
            label = (
                f"{100.0 * float(smooth_percent[idx]):5.1f}%  "
                f"instant={100.0 * float(raw_percent[idx]):5.1f}%  "
                f"score={float(raw_scores[idx]):.3e}"
            )
            cv2.putText(
                canvas, label, (bar_x0 + 8, y + 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (15, 15, 15), 1, cv2.LINE_AA,
            )

        plot_x0, plot_y0, plot_x1, plot_y1 = 70, 382, 850, 548
        cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x1, plot_y1), (35, 35, 35), 1)
        for frac in (0.25, 0.50, 0.75):
            y = int(round(plot_y1 - frac * (plot_y1 - plot_y0)))
            cv2.line(canvas, (plot_x0, y), (plot_x1, y), (215, 215, 215), 1)
            cv2.putText(
                canvas, f"{int(frac * 100)}%", (20, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1, cv2.LINE_AA,
            )

        hist = np.asarray(list(self._modality_importance_history), dtype=np.float32)
        if hist.ndim == 2 and hist.shape[0] >= 2:
            xs = np.linspace(plot_x0, plot_x1, hist.shape[0]).astype(np.int32)
            for idx, color in enumerate(colors):
                ys = np.rint(plot_y1 - hist[:, idx] * (plot_y1 - plot_y0)).astype(np.int32)
                pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)

        cv2.putText(
            canvas, "smoothed relative sensitivity history", (plot_x0, plot_y0 - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (45, 45, 45), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Counterfactual diagnostic, not causal proof. Force removes current force + history together.",
            (32, 582),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def _run_modality_importance_debug(
        self,
        *,
        q_t: torch.Tensor,
        img_t: torch.Tensor,
        force_hist_t: Optional[torch.Tensor],
        stain_mask_t: Optional[torch.Tensor],
        flow_initial_noise_t: Optional[torch.Tensor],
        gripper_position_t: Optional[torch.Tensor] = None,
        gripper_current_t: Optional[torch.Tensor] = None,
        gripper_history_t: Optional[torch.Tensor] = None,
    ) -> bool:
        if not self.modality_importance_enable or self.pub_modality_importance is None:
            return False
        if self._infer_plan_count <= 0:
            return False
        if (self._infer_plan_count % self.modality_importance_every_n_infer) != 0:
            return False

        with self._modality_importance_worker_lock:
            if self._modality_importance_worker_busy:
                return False
            self._modality_importance_worker_busy = True

        def _snapshot(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if value is None else value.detach().clone()

        worker_inputs = {
            "q_t": _snapshot(q_t),
            "img_t": _snapshot(img_t),
            "force_hist_t": _snapshot(force_hist_t),
            "stain_mask_t": _snapshot(stain_mask_t),
            "flow_initial_noise_t": _snapshot(flow_initial_noise_t),
            "gripper_position_t": _snapshot(gripper_position_t),
            "gripper_current_t": _snapshot(gripper_current_t),
            "gripper_history_t": _snapshot(gripper_history_t),
        }

        def _worker() -> None:
            try:
                self._compute_modality_importance_debug(**worker_inputs)
            except Exception as exc:
                self._modality_importance_fail_count += 1
                self.get_logger().warn(
                    f"[MODALITY] worker failed #{self._modality_importance_fail_count}: {exc}"
                )
            finally:
                with self._modality_importance_worker_lock:
                    self._modality_importance_worker_busy = False

        worker = threading.Thread(
            target=_worker,
            name="modality_importance_worker",
            daemon=True,
        )
        self._modality_importance_worker_thread = worker
        worker.start()
        return True

    def _compute_modality_importance_debug(
        self,
        *,
        q_t: torch.Tensor,
        img_t: torch.Tensor,
        force_hist_t: Optional[torch.Tensor],
        stain_mask_t: Optional[torch.Tensor],
        flow_initial_noise_t: Optional[torch.Tensor],
        gripper_position_t: Optional[torch.Tensor] = None,
        gripper_current_t: Optional[torch.Tensor] = None,
        gripper_history_t: Optional[torch.Tensor] = None,
    ) -> bool:
        if self.policy_class not in ("FLOW", "BSPLINE"):
            if self._modality_importance_fail_count == 0:
                self.get_logger().warn(
                    "[MODALITY] fast modality attribution currently supports FLOW/BSPLINE only"
                )
            self._modality_importance_fail_count += 1
            return False
        if self.policy_class == "FLOW" and not (
            hasattr(self.policy, "_condition") and hasattr(self.policy, "velocity_net")
        ):
            self.get_logger().warn(
                "[MODALITY] FLOW policy does not expose _condition/velocity_net"
            )
            return False
        if self.policy_class == "BSPLINE" and not (
            hasattr(self.policy, "_condition")
            and hasattr(self.policy, "control_head")
            and hasattr(self.policy, "trajectory_basis")
        ):
            self.get_logger().warn(
                "[MODALITY] BSPLINE policy does not expose "
                "_condition/control_head/trajectory_basis"
            )
            return False

        def _target_block(seq: torch.Tensor) -> torch.Tensor:
            if seq.dim() != 2 or seq.shape[-1] not in (9, 10, 11):
                raise RuntimeError(f"modality seq must be (T,D), got {tuple(seq.shape)}")
            total_steps = int(seq.shape[0])
            start = min(
                max(0, int(self.modality_importance_target_step)),
                max(0, total_steps - 1),
            )
            end = min(
                total_steps,
                start + max(1, int(self.modality_importance_target_horizon)),
            )
            block = seq[start:end]
            target = str(self.modality_importance_target or "action_norm").strip().lower()
            scalar_index = {
                "x": 0, "cmd_x": 0,
                "y": 1, "cmd_y": 1,
                "z": 2, "cmd_z": 2, "abs_z": 2, "z_abs": 2,
                "wx": 3, "rx": 3, "roll": 3,
                "wy": 4, "ry": 4, "pitch": 4,
                "wz": 5, "rz": 5, "yaw": 5,
                "fx": 6, "cmd_fx": 6,
                "fy": 7, "cmd_fy": 7,
                "fz": 8, "cmd_fz": 8, "abs_fz": 8, "fz_abs": 8,
                "gripper": 9, "grip": 9, "tick": 9, "gripper_position": 9,
                "gripper_current": 10, "goal_current": 10,
                "gripper_goal_current": 10,
            }
            if target in scalar_index:
                idx = scalar_index[target]
                if idx >= block.shape[-1]:
                    raise RuntimeError(
                        f"target={target} requires action dim>{idx}, got {block.shape[-1]}"
                    )
                return block[:, idx:idx + 1]
            if target in ("xyz_norm", "pos_norm", "position_norm"):
                return block[:, 0:3]
            if target in ("rot_norm", "ori_norm", "orientation_norm"):
                return block[:, 3:6]
            if target in ("force_norm", "f_norm"):
                return block[:, 6:9]
            return block[:, 0:9]

        if q_t.dim() != 2 or q_t.shape[0] != 1 or q_t.shape[-1] < 9:
            raise RuntimeError(f"expected qpos shape (1,D>=9), got {tuple(q_t.shape)}")
        if img_t.dim() != 5 or img_t.shape[0] != 1:
            raise RuntimeError(f"expected image shape (1,K,3,H,W), got {tuple(img_t.shape)}")

        was_training = self.policy.training
        try:
            self.policy.eval()
            q_base = q_t.detach()
            image_base = img_t.detach()
            force_base = None if force_hist_t is None else force_hist_t.detach()

            q_position_off = q_base.clone()
            q_position_off[..., :6] = 0.0
            q_force_off = q_base.clone()
            q_force_off[..., 6:9] = 0.0
            force_history_off = (
                None if force_base is None else torch.zeros_like(force_base)
            )

            # FlowRGBPolicy applies ImageNet normalization internally. Feeding
            # ImageNet mean RGB therefore represents a neutral zero CNN input.
            image_off = image_base.clone()
            image_mean = torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=image_off.dtype,
                device=image_off.device,
            ).view(1, 1, 3, 1, 1)
            image_off.copy_(image_mean.expand_as(image_off))

            # Batch order: full, position-off, force-off, image-off.
            q_batch = torch.cat(
                [q_base, q_position_off, q_force_off, q_base], dim=0
            )
            image_batch = torch.cat(
                [image_base, image_base, image_base, image_off], dim=0
            )
            force_batch = None
            if force_base is not None:
                force_batch = torch.cat(
                    [force_base, force_base, force_history_off, force_base], dim=0
                )
            mask_batch = None
            if stain_mask_t is not None:
                mask_base = stain_mask_t.detach()
                mask_batch = mask_base.repeat(4, *([1] * (mask_base.dim() - 1)))

            gp_batch = None
            gc_batch = None
            gh_batch = None
            if gripper_position_t is not None:
                gp_batch = gripper_position_t.detach().repeat(
                    4, *([1] * (gripper_position_t.dim() - 1))
                )
            if gripper_current_t is not None:
                gc_batch = gripper_current_t.detach().repeat(
                    4, *([1] * (gripper_current_t.dim() - 1))
                )
            if gripper_history_t is not None:
                gh_batch = gripper_history_t.detach().repeat(
                    4, *([1] * (gripper_history_t.dim() - 1))
                )

            if self.policy_class == "FLOW":
                if flow_initial_noise_t is None:
                    z_base = torch.zeros(
                        (1, self.chunk_size, self.action_dim),
                        dtype=q_base.dtype,
                        device=q_base.device,
                    )
                else:
                    z_base = flow_initial_noise_t.detach()
                z_batch = z_base.repeat(4, 1, 1)
                t_batch = torch.full(
                    (4,), 0.5, dtype=q_base.dtype, device=q_base.device
                )

            start_t = _monotonic()
            with torch.inference_mode():
                cond_batch = self.policy._condition(
                    qpos=q_batch,
                    image=image_batch,
                    force_history=force_batch,
                    gripper_position=gp_batch,
                    gripper_current=gc_batch,
                    gripper_history=gh_batch,
                ) if self.use_gripper else self.policy._condition(
                    qpos=q_batch,
                    image=image_batch,
                    force_history=force_batch,
                    stain_mask=mask_batch,
                )
                if self.policy_class == "FLOW":
                    # Velocity field at t=0.5 stands in for the action the
                    # policy would ultimately integrate to -- cheap proxy,
                    # no need to run the full flow-matching sampler.
                    output_batch = self.policy.velocity_net(
                        sample=z_batch,
                        t=t_batch,
                        global_cond=cond_batch,
                    )
                else:
                    # BSPLINE has no iterative sampler: control points ->
                    # trajectory is the actual (single-pass) action output.
                    control_points_batch = self.policy.control_head(cond_batch)
                    output_batch = self.policy.trajectory_basis(control_points_batch)
            compute_ms = 1000.0 * (_monotonic() - start_t)

            reference_block = _target_block(output_batch[0])
            raw_scores = []
            for idx in (1, 2, 3):
                delta = _target_block(output_batch[idx]) - reference_block
                raw_scores.append(
                    float(torch.sqrt(torch.mean(delta.float().square()) + 1e-24).cpu())
                )
            raw_scores = np.asarray(raw_scores, dtype=np.float64)
            if not np.all(np.isfinite(raw_scores)):
                raise RuntimeError(f"non-finite modality scores: {raw_scores}")
            total = float(raw_scores.sum())
            if total <= 1e-20:
                raise RuntimeError("all modality ablation changes are effectively zero")
            raw_percent = raw_scores / total

            if self._modality_importance_ema is None:
                smooth_percent = raw_percent.copy()
            else:
                alpha = self.modality_importance_ema_alpha
                smooth_percent = (
                    alpha * self._modality_importance_ema
                    + (1.0 - alpha) * raw_percent
                )
            smooth_percent = smooth_percent / max(float(smooth_percent.sum()), 1e-20)
            self._modality_importance_ema = smooth_percent
            self._modality_importance_history.append(smooth_percent.astype(np.float32))
            self._modality_importance_count += 1

            rgb = self._render_modality_importance_rgb(
                raw_scores=raw_scores,
                raw_percent=raw_percent,
                smooth_percent=smooth_percent,
            )
            msg = _rgb_numpy_to_image_msg(
                rgb,
                stamp=self.get_clock().now().to_msg(),
                frame_id="modality_importance",
            )
            self.pub_modality_importance.publish(msg)

            if (
                self._modality_importance_count <= 3
                or self._modality_importance_count
                % self.modality_importance_log_every_n == 0
            ):
                self.get_logger().info(
                    "[MODALITY] "
                    f"#{self._modality_importance_count} "
                    f"target={self.modality_importance_target} "
                    f"position={100.0 * smooth_percent[0]:.1f}% "
                    f"force={100.0 * smooth_percent[1]:.1f}% "
                    f"image={100.0 * smooth_percent[2]:.1f}% "
                    f"ablation_scores="
                    f"{np.array2string(raw_scores, precision=3, separator=',')} "
                    f"compute={compute_ms:.1f}ms"
                )
            return True

        except Exception as e:
            self._modality_importance_fail_count += 1
            now_t = _monotonic()
            if (
                self._modality_importance_fail_count <= 3
                or now_t - self._modality_importance_last_log_t > 2.0
            ):
                self._modality_importance_last_log_t = now_t
                self.get_logger().warn(
                    f"[MODALITY] failed #{self._modality_importance_fail_count}: {e}"
                )
            return False
        finally:
            if was_training:
                self.policy.train()


    def _render_flow_vector_overlay_rgb(self, *, rgb: np.ndarray) -> np.ndarray:
        """TCP-ROI camera view: Grad-CAM heatmap (if available) + red ROI box.

        No text/vector-arrow clutter -- this is meant to sit side-by-side
        with the modality-importance dashboard in the recorded video.
        """
        if cv2 is None:
            raise RuntimeError("OpenCV is required for FLOW vector overlay")

        image = np.asarray(rgb).copy()
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"FLOW overlay RGB must be (H,W,3), got {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        height, width = image.shape[:2]
        cx = int(np.clip(self.flow_vector_overlay_tcp_center_x, 0, width - 1))
        cy = int(np.clip(self.flow_vector_overlay_tcp_center_y, 0, height - 1))

        if self._latest_gradcam_heat is not None:
            image = _make_gradcam_overlay_rgb(
                image,
                self._latest_gradcam_heat,
                alpha=self.gradcam_alpha,
                colormap=self.gradcam_colormap,
            )

        # TCP-centered square ROI used by this checkpoint.
        roi_side = max(1, int(round(math.sqrt(self.tcp_roi_area_fraction * float(width * height)))))
        roi_x0 = int(np.clip(cx - roi_side // 2, 0, max(0, width - roi_side)))
        roi_y0 = int(np.clip(cy - roi_side // 2, 0, max(0, height - roi_side)))
        cv2.rectangle(
            image,
            (roi_x0, roi_y0),
            (min(width - 1, roi_x0 + roi_side), min(height - 1, roi_y0 + roi_side)),
            (255, 80, 80),
            1,
            cv2.LINE_AA,
        )
        # Live camera feed and the hdf5-recorded training images are 180
        # degrees apart in composition -- rotate the display-only copy here
        # (not the tensor fed to the policy) so this overlay matches what
        # the recorded episodes actually look like.
        image = cv2.rotate(image, cv2.ROTATE_180)
        return image

    def _publish_flow_vector_overlay_frame(self, rgb: np.ndarray) -> bool:
        """Render+publish the overlay for one camera frame.

        Called from _on_img() at the camera's own native rate, so the
        recorded video is as smooth as the camera feed itself -- the
        overlay's only genuinely slow-updating ingredient is the cached
        Grad-CAM heatmap (refreshed once per replan tick), everything else
        (camera pixels, red ROI box) is fresh on every call.
        """
        if not self.flow_vector_overlay_enable or self.pub_flow_vector_overlay is None:
            return False
        try:
            overlay = self._render_flow_vector_overlay_rgb(rgb=rgb)
            msg = _rgb_numpy_to_image_msg(
                overlay,
                stamp=self.get_clock().now().to_msg(),
                frame_id="flow_vector_tool_xy_proxy",
            )
            self.pub_flow_vector_overlay.publish(msg)
            self._flow_vector_overlay_count += 1
            return True
        except Exception as exc:
            self._flow_vector_overlay_fail_count += 1
            if self._flow_vector_overlay_fail_count <= 3:
                self.get_logger().warn(
                    f"[FLOW-VECTOR] overlay failed #{self._flow_vector_overlay_fail_count}: {exc}"
                )
            return False

    def _log_flow_vector_delta(self, *, pose6: np.ndarray, seq_raw: np.ndarray) -> None:
        """Per-replan diagnostic log only -- the overlay image itself now
        publishes independently at camera rate (see
        _publish_flow_vector_overlay_frame)."""
        self._flow_vector_delta_log_count += 1
        if self._flow_vector_delta_log_count <= 3 or self._flow_vector_delta_log_count % 20 == 0:
            selected = min(self.flow_vector_overlay_selected_horizon, seq_raw.shape[0] - 1)
            delta = seq_raw[selected, :3] - pose6[:3]
            self.get_logger().info(
                f"[FLOW-VECTOR] #{self._flow_vector_delta_log_count} h={selected} "
                f"delta_xyz=[{delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f}]mm "
                f"pred_fz={seq_raw[selected,8]:+.3f}N"
            )

    def _on_flow_step_service(self, request, response):
        del request
        if not self.flow_diagnostic_only:
            response.success = False
            response.message = "refused: flow_diagnostic_only is false"
            return response
        if self.pub_cmd is None:
            response.success = False
            response.message = "refused: motion-command publisher is unavailable"
            return response

        with self._lock:
            pose6 = None if self._pose6 is None else self._pose6.copy()
            force = None if self._force is None else self._force.copy()
        seq = None if self._latest_flow_raw_seq is None else self._latest_flow_raw_seq.copy()
        plan_age = _monotonic() - float(self._latest_flow_plan_t)
        if pose6 is None or force is None or seq is None:
            response.success = False
            response.message = "refused: waiting for pose, force, image, mask, and FLOW plan"
            return response
        if plan_age > max(1.0, 3.0 * self.dt_infer):
            response.success = False
            response.message = f"refused: latest FLOW plan is stale ({plan_age:.2f}s)"
            return response

        selected = min(
            max(0, int(self.flow_vector_overlay_selected_horizon)), seq.shape[0] - 1
        )
        raw_target6 = seq[selected, :6].astype(np.float32)
        if not np.all(np.isfinite(pose6[:6])) or not np.all(np.isfinite(raw_target6)):
            response.success = False
            response.message = "refused: current or predicted pose contains NaN/Inf"
            return response
        delta_xyz = raw_target6[:3] - pose6[:3]
        if self.orientation_lock_enable:
            delta_rot = self.orientation_lock_rotvec - pose6[3:6]
        else:
            delta_rot = raw_target6[3:6] - pose6[3:6]

        xyz_norm = float(np.linalg.norm(delta_xyz))
        if self.flow_step_max_xyz_mm <= 0.0:
            delta_xyz[:] = 0.0
        elif xyz_norm > self.flow_step_max_xyz_mm:
            delta_xyz = delta_xyz * np.float32(self.flow_step_max_xyz_mm / xyz_norm)
        rot_norm = float(np.linalg.norm(delta_rot))
        if not self.orientation_lock_enable:
            if self.flow_step_max_rot_rad <= 0.0:
                delta_rot[:] = 0.0
            elif rot_norm > self.flow_step_max_rot_rad:
                delta_rot = delta_rot * np.float32(self.flow_step_max_rot_rad / rot_norm)

        measured_fz = float(force[2]) if force.size >= 3 else 0.0
        if (
            self.flow_step_block_down_on_contact
            and measured_fz >= self.contact_on_thr
            and float(delta_xyz[2]) < 0.0
        ):
            response.success = False
            response.message = (
                f"blocked: contact Fz={measured_fz:.2f}N and requested dz={delta_xyz[2]:+.3f}mm"
            )
            self.get_logger().warn(f"[FLOW-STEP] {response.message}")
            return response

        target6 = pose6[:6].astype(np.float32).copy()
        target6[:3] += delta_xyz
        if self.orientation_lock_enable:
            target6[3:6] = self.orientation_lock_rotvec
        else:
            target6[3:6] += delta_rot

        if (
            self.stats is not None
            and self.stats.act_mode in ("minmax_01", "minmax_m11")
            and self.stats.act_a.size >= 3
        ):
            lo = np.asarray(self.stats.act_a[:3], dtype=np.float32) - self.flow_step_stats_margin_mm
            hi = np.asarray(self.stats.act_b[:3], dtype=np.float32) + self.flow_step_stats_margin_mm
            if np.any(pose6[:3] < lo) or np.any(pose6[:3] > hi):
                response.success = False
                response.message = (
                    "refused: current XYZ is outside training envelope + margin; "
                    f"current={np.array2string(pose6[:3], precision=2)}"
                )
                return response
            if np.any(target6[:3] < lo) or np.any(target6[:3] > hi):
                response.success = False
                response.message = (
                    "refused: bounded step would leave training envelope + margin; "
                    f"target={np.array2string(target6[:3], precision=2)}"
                )
                return response

        command = np.zeros(9, dtype=np.float32)
        command[:6] = target6
        # Manual vector diagnosis checks the learned pose field first. Never
        # inject force targets through the step service.
        command[6:9] = 0.0
        published = self._publish_cmd(command)
        if not np.allclose(published[:6], command[:6], atol=1e-6, rtol=0.0):
            response.success = False
            response.message = "blocked by command safety; current-pose hold was published"
            return response

        self._flow_step_count += 1
        response.success = True
        response.message = (
            f"step#{self._flow_step_count} h={selected} "
            f"dxyz=[{delta_xyz[0]:+.3f},{delta_xyz[1]:+.3f},{delta_xyz[2]:+.3f}]mm "
            f"drot_norm={np.linalg.norm(delta_rot):.6f}rad "
            f"orientation={'locked_z90' if self.orientation_lock_enable else 'policy'} "
            f"Fz_cmd=0"
        )
        self.get_logger().warn(f"[FLOW-STEP] {response.message}")
        return response


    # ------------------------------------------------------------
    # Load policy (nrs_imitation/source/models/policy.py) + ckpt
    # ------------------------------------------------------------
    def _load_policy_and_ckpt_from_act_root(self):
        act_source = os.path.join(self.act_root, "source")
        if self.act_root not in sys.path:
            sys.path.insert(0, self.act_root)
        if act_source not in sys.path:
            sys.path.insert(0, act_source)

        try:
            from models.policy import ACTPolicy, DiffusionPolicy
        except Exception as e:
            raise RuntimeError(
                f"Failed to import ACT/Diffusion policy classes from {act_source}/models/policy.py : {e}"
            )

        flow_module = "models.gri_flow_core" if self.use_gripper else "models.flow_core"
        try:
            if self.use_gripper:
                from models.gri_flow_core import FlowRGBPolicy
            else:
                from models.flow_core import FlowRGBPolicy
        except Exception as e:
            FlowRGBPolicy = None
            if str(self.policy_class).upper() == "FLOW":
                raise RuntimeError(
                    f"Failed to import FlowRGBPolicy from {act_source}/{flow_module.replace('.', '/')}.py : {e}"
                )

        try:
            from models.bspline_core import BSplinePolicy
        except Exception as e:
            BSplinePolicy = None
            if str(self.policy_class).upper() == "BSPLINE":
                raise RuntimeError(
                    f"Failed to import BSplinePolicy from {act_source}/models/bspline_core.py : {e}"
                )

        args_override = {
            "kl_weight": float(self.get_parameter("kl_weight").value),
            "num_queries": int(self.chunk_size),

            "lr": 1e-4,
            "hidden_dim": int(self.get_parameter("hidden_dim").value),
            "dim_feedforward": int(self.get_parameter("dim_feedforward").value),
            "lr_backbone": float(self.get_parameter("lr_backbone").value),
            "backbone": str(self.get_parameter("backbone").value),
            "enc_layers": int(self.get_parameter("enc_layers").value),
            "dec_layers": int(self.get_parameter("dec_layers").value),
            "nheads": int(self.get_parameter("nheads").value),

            "camera_names": list(self.camera_names),
            "obs_mode": self.obs_mode,
            "state_dim": 9,
            "action_dim": self.action_dim,

            "image_resize_hw": int(self.get_parameter("image_resize_hw").value),
            "image_pool_hw": int(self.get_parameter("image_pool_hw").value),
            "pretrained_backbone": bool(self.get_parameter("pretrained_backbone").value),
            "image_backbone": self.image_backbone,
            "dino_model_name": self.dino_model_name,
            "dino_checkpoint_path": self.dino_checkpoint_path,
            "freeze_image_backbone": self.freeze_image_backbone,
            "dino_roi_pooling": self.dino_roi_pooling,
            "use_tcp_roi": self.use_tcp_roi,
            "tcp_roi_reference_width": self.tcp_roi_reference_width,
            "tcp_roi_reference_height": self.tcp_roi_reference_height,
            "tcp_roi_center_x": self.tcp_roi_center_x,
            "tcp_roi_center_y": self.tcp_roi_center_y,
            "tcp_roi_area_fraction": self.tcp_roi_area_fraction,

            # observation encoder config
            "position_dim": self.position_dim,
            "force_dim": self.force_dim,
            "position_encoder_hidden_dim": self.position_encoder_hidden_dim,
            "force_encoder_hidden_dim": self.force_encoder_hidden_dim,
            "force_encoder_num_layers": self.force_encoder_num_layers,
            "force_encoder_dropout": self.force_encoder_dropout,
            "observation_encoder_activation": self.observation_encoder_activation,

            # diffusion config (ignored by ACTPolicy/FLOW)
            "diffusion_train_steps": self.diffusion_train_steps,
            "diffusion_infer_steps": self.diffusion_infer_steps,
            "diffusion_beta_start": self.diffusion_beta_start,
            "diffusion_beta_end": self.diffusion_beta_end,
            "diffusion_loss_type": self.diffusion_loss_type,

            # FLOW config
            "use_force_history": self.use_force_history,
            "force_history_len": self.force_history_len,
            "flow_infer_steps": self.flow_infer_steps,
            "flow_train_eps": self.flow_train_eps,
            "flow_loss_type": self.flow_loss_type,
            "flow_obs_hidden_dim": self.flow_obs_hidden_dim,
            "flow_image_feature_dim": self.flow_image_feature_dim,
            "flow_global_cond_dim": self.flow_global_cond_dim,
            "flow_time_embed_dim": self.flow_time_embed_dim,
            "flow_down_dims": self.flow_down_dims,
            "flow_kernel_size": self.flow_kernel_size,
            "flow_n_groups": self.flow_n_groups,
            "flow_cond_predict_scale": self.flow_cond_predict_scale,
            "use_stain_mask": self.use_stain_mask,
            "stain_mask_key": self.stain_mask_key,
            "stain_pooling_type": self.stain_pooling_type,
            "empty_stain_feature_mode": self.empty_stain_feature_mode,
            "stain_mask_threshold": self.stain_mask_threshold,
            "debug_stain_pooling": self.debug_stain_pooling,

            # BSPLINE action head (ignored by ACT/DIFFUSION/FLOW)
            "num_control_points": 16,
            "bspline_degree": 3,
            "bspline_hidden_dim": 256,
            "bspline_loss_type": "mse",
        }

        # FLOW/BSPLINE checkpoints store their complete image backbone state.
        # Rebuild the exact architecture from dataset metadata without
        # downloading the pretrained DINO/ResNet weights again during online
        # inference.
        if str(self.policy_class).upper() in ("FLOW", "BSPLINE") and not self.use_gripper:
            try:
                stats_obj = _pickle_load_compat(os.path.join(self.ckpt_dir, "dataset_stats.pkl"))
                ckpt_policy_cfg = dict(stats_obj.get("policy_config", {}))
            except Exception as exc:
                raise RuntimeError(f"failed to load {self.policy_class} checkpoint metadata: {exc}")
            # "num_control_points" only ever appears in a BSPLINE-trained
            # policy_config. Catch a ckpt_dir/policy_class mismatch here,
            # loudly, instead of best-effort-loading mismatched weights into
            # a policy that then drives the robot.
            ckpt_is_bspline = "num_control_points" in ckpt_policy_cfg
            if str(self.policy_class).upper() == "BSPLINE" and not ckpt_is_bspline:
                raise RuntimeError(
                    f"policy_class=BSPLINE but checkpoint at {self.ckpt_dir} has no "
                    "'num_control_points' in its saved policy_config -> this does not "
                    "look like a BSPLINE checkpoint. Pass a matching ckpt_dir."
                )
            if str(self.policy_class).upper() == "FLOW" and ckpt_is_bspline:
                raise RuntimeError(
                    f"policy_class=FLOW but checkpoint at {self.ckpt_dir} looks like a "
                    "BSPLINE checkpoint (has 'num_control_points' in policy_config). "
                    "Pass a matching ckpt_dir."
                )
            for key in (
                "image_backbone",
                "dino_model_name",
                "freeze_image_backbone",
                "dino_roi_pooling",
                "use_tcp_roi",
                "tcp_roi_reference_width",
                "tcp_roi_reference_height",
                "tcp_roi_center_x",
                "tcp_roi_center_y",
                "tcp_roi_area_fraction",
                "num_control_points",
                "bspline_degree",
                "bspline_hidden_dim",
                "bspline_loss_type",
            ):
                if key in ckpt_policy_cfg:
                    args_override[key] = ckpt_policy_cfg[key]
            if "use_tcp_roi" not in ckpt_policy_cfg:
                args_override["use_tcp_roi"] = bool(
                    ckpt_policy_cfg.get("use_stain_mask", False)
                )
            if "image_backbone" not in ckpt_policy_cfg:
                # Legacy FLOW checkpoints predate the selectable backbone and
                # were trained with ResNet18.
                args_override["image_backbone"] = "resnet18"
            args_override["pretrained_backbone"] = False
            args_override["dino_checkpoint_path"] = ""

            # The launch file only forwards tcp_roi_* to stain_mask_publisher,
            # not to this node -- self.tcp_roi_* would otherwise stay stuck at
            # the ROS parameter default (e.g. 0.10) instead of the checkpoint's
            # true training-time value used above to build the policy. Sync
            # them here so the flow-vector overlay box always matches what the
            # model actually attends to.
            if "use_tcp_roi" in args_override:
                self.use_tcp_roi = bool(args_override["use_tcp_roi"])
            if "tcp_roi_reference_width" in args_override:
                self.tcp_roi_reference_width = int(args_override["tcp_roi_reference_width"])
            if "tcp_roi_reference_height" in args_override:
                self.tcp_roi_reference_height = int(args_override["tcp_roi_reference_height"])
            if "tcp_roi_center_x" in args_override:
                self.tcp_roi_center_x = int(args_override["tcp_roi_center_x"])
            if "tcp_roi_center_y" in args_override:
                self.tcp_roi_center_y = int(args_override["tcp_roi_center_y"])
            if "tcp_roi_area_fraction" in args_override:
                self.tcp_roi_area_fraction = float(args_override["tcp_roi_area_fraction"])

        if self.use_gripper:
            try:
                stats_obj = _pickle_load_compat(os.path.join(self.ckpt_dir, "dataset_stats.pkl"))
                ckpt_policy_cfg = dict(stats_obj.get("policy_config", {}))
            except Exception as exc:
                raise RuntimeError(f"failed to load gripper checkpoint metadata: {exc}")

            ckpt_use_force_history = bool(
                ckpt_policy_cfg.get("use_force_history", False)
            )
            if ckpt_use_force_history != self.use_force_history:
                raise RuntimeError(
                    "use_force_history mismatch: "
                    f"checkpoint={ckpt_use_force_history}, "
                    f"inference_arg={self.use_force_history}"
                )
            if ckpt_use_force_history:
                ckpt_force_history_len = int(
                    ckpt_policy_cfg.get("force_history_len", 0)
                )
                if ckpt_force_history_len != self.force_history_len:
                    raise RuntimeError(
                        "force_history_len mismatch: "
                        f"checkpoint={ckpt_force_history_len}, "
                        f"inference_arg={self.force_history_len}"
                    )

            ckpt_use_gripper_history = bool(
                ckpt_policy_cfg.get(
                    "use_gripper_history",
                    stats_obj.get("use_gripper_history", False),
                )
            )
            if ckpt_use_gripper_history != self.use_gripper_history:
                raise RuntimeError(
                    "use_gripper_history mismatch: "
                    f"checkpoint={ckpt_use_gripper_history}, "
                    f"inference_arg={self.use_gripper_history}. "
                    "Use the checkpoint's exact observation schema."
                )
            if ckpt_use_gripper_history:
                ckpt_history_len = int(
                    ckpt_policy_cfg.get(
                        "gripper_history_len",
                        stats_obj.get("gripper_history_len", 0),
                    )
                )
                if ckpt_history_len != self.gripper_history_len:
                    raise RuntimeError(
                        "gripper_history_len mismatch: "
                        f"checkpoint={ckpt_history_len}, "
                        f"inference_arg={self.gripper_history_len}"
                    )
                ckpt_history_hz = float(
                    stats_obj.get(
                        "dataset_hz",
                        ckpt_policy_cfg.get("dataset_hz", 0.0),
                    )
                )
                if ckpt_history_hz <= 0.0 or not math.isclose(
                    ckpt_history_hz,
                    self.gripper_history_hz,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ):
                    raise RuntimeError(
                        "gripper history Hz mismatch: "
                        f"checkpoint_dataset_hz={ckpt_history_hz}, "
                        f"inference_arg={self.gripper_history_hz}"
                    )
                expected_channels = ["present_position", "present_current_mA"]
                checkpoint_channels = list(
                    stats_obj.get("gripper_history_channels", [])
                )
                if checkpoint_channels != expected_channels:
                    raise RuntimeError(
                        "gripper history channel schema mismatch: "
                        f"checkpoint={checkpoint_channels}, expected={expected_channels}"
                    )
                if (
                    self.stats is None
                    or self.stats.gripper_position_a is None
                    or self.stats.gripper_position_b is None
                ):
                    raise RuntimeError(
                        "history checkpoint requires normalized gripper position stats"
                    )

            for key in (
                "num_queries",
                "state_dim",
                "action_dim",
                "force_dim",
                "use_force_history",
                "force_history_len",
                "force_encoder_hidden_dim",
                "force_encoder_num_layers",
                "force_encoder_dropout",
                "use_gripper_history",
                "gripper_history_len",
                "gripper_history_sec",
                "gripper_history_input_dim",
                "gripper_history_hidden_dim",
                "gripper_history_num_layers",
                "gripper_history_dropout",
                "gripper_encoder_hidden_dim",
                "gripper_feature_dim",
                "flow_marker_feature_dim",
                "flow_obs_hidden_dim",
                "flow_image_feature_dim",
                "flow_global_cond_dim",
                "flow_time_embed_dim",
                "flow_down_dims",
                "flow_kernel_size",
                "flow_n_groups",
                "flow_cond_predict_scale",
            ):
                if key in ckpt_policy_cfg:
                    args_override[key] = ckpt_policy_cfg[key]
            if "num_queries" in ckpt_policy_cfg:
                self.chunk_size = int(ckpt_policy_cfg["num_queries"])
            args_override["action_dim"] = 11
            args_override["pretrained_backbone"] = False

        policy_class = str(self.policy_class).upper()
        if policy_class == "ACT":
            self.get_logger().info("[INFO] Loading ACTPolicy from nrs_imitation/source/models/policy.py ...")
            policy = ACTPolicy(args_override).to(self.device)
        elif policy_class == "DIFFUSION":
            self.get_logger().info("[INFO] Loading DiffusionPolicy from nrs_imitation/source/models/policy.py ...")
            policy = DiffusionPolicy(args_override).to(self.device)
        elif policy_class == "FLOW":
            self.get_logger().info(f"[INFO] Loading FlowRGBPolicy from nrs_imitation/source/{flow_module.replace('.', '/')}.py ...")
            if FlowRGBPolicy is None:
                raise RuntimeError("FlowRGBPolicy import failed.")
            policy = FlowRGBPolicy(args_override).to(self.device)
        elif policy_class == "BSPLINE":
            self.get_logger().info("[INFO] Loading BSplinePolicy from nrs_imitation/source/models/bspline_core.py ...")
            if BSplinePolicy is None:
                raise RuntimeError("BSplinePolicy import failed.")
            policy = BSplinePolicy(args_override).to(self.device)
        else:
            raise RuntimeError(f"Unsupported policy_class: {self.policy_class}")

        policy.eval()

        ckpt_path = os.path.join(self.ckpt_dir, "policy_best.ckpt")
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"policy_best.ckpt not found: {ckpt_path}")

        ckpt_obj = torch.load(ckpt_path, map_location=self.device)
        if isinstance(ckpt_obj, dict):
            ckpt_cfg = ckpt_obj.get("config", {}).get("policy_config", {})
            ckpt_use_stain = bool(ckpt_cfg.get("use_stain_mask", False))
            if ckpt_use_stain != bool(self.use_stain_mask):
                raise RuntimeError(
                    f"use_stain_mask mismatch: checkpoint={ckpt_use_stain}, inference_arg={bool(self.use_stain_mask)}. "
                    "Use the same stain-mask setting as training."
                )

        if isinstance(ckpt_obj, dict):
            if "model_state_dict" in ckpt_obj:
                state_dict = ckpt_obj["model_state_dict"]
            elif "state_dict" in ckpt_obj:
                state_dict = ckpt_obj["state_dict"]
            else:
                state_dict = ckpt_obj
        else:
            state_dict = ckpt_obj

        if self.use_gripper:
            loaded_transform = _load_state_dict_strict_compat(policy, state_dict)
            missing, unexpected = [], []
        else:
            loaded_transform = "best-effort"
            missing, unexpected = _try_load_state_dict_compat(policy, state_dict)

        if (len(missing) + len(unexpected) > 0) and hasattr(policy, "model"):
            missing2, unexpected2 = _try_load_state_dict_compat(policy.model, state_dict)
            if (len(missing2) + len(unexpected2)) < (len(missing) + len(unexpected)):
                missing, unexpected = missing2, unexpected2

        self.get_logger().info(
            f"[INFO] Loaded ckpt from {ckpt_path}. transform={loaded_transform}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if len(missing) > 0:
            self.get_logger().warn(f"[INFO] missing sample: {list(missing)[:10]}")
        if len(unexpected) > 0:
            self.get_logger().warn(f"[INFO] unexpected sample: {list(unexpected)[:10]}")
        self.get_logger().info(
            f"[INFO] policy_class={policy_class}, obs_mode={self.obs_mode}, camera_names={self.camera_names}, "
            f"use_force_history={self.use_force_history}, force_history_len={self.force_history_len}, "
            f"use_gripper_history={self.use_gripper_history}, "
            f"gripper_history_len={self.gripper_history_len}"
        )
        return policy

    # ------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------
    def _on_pose(self, msg: Float64MultiArray):
        arr = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if arr.size >= 6:
            with self._lock:
                self._pose6 = arr[:6].copy()

    def _on_force(self, msg: Float64MultiArray):
        arr = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        with self._lock:
            self._force = arr.copy()
            if arr.size >= 3:
                self._force_hist.append(self._extract_force3(arr))

    def _on_force_wrench(self, msg: Wrench):
        arr = np.asarray([msg.force.x, msg.force.y, msg.force.z], dtype=np.float32)
        with self._lock:
            self._force = arr.copy()
            self._force_hist.append(arr.copy())

    def _match_gripper_pairs_locked(self) -> None:
        """Approximate-time synchronize headerless position/current messages."""
        while self._gripper_position_pending and self._gripper_current_pending:
            position_t, position = self._gripper_position_pending[0]
            current_t, current = self._gripper_current_pending[0]
            skew = float(position_t - current_t)
            if abs(skew) <= self.gripper_history_sync_slop_sec:
                self._gripper_position_pending.popleft()
                self._gripper_current_pending.popleft()
                pair_t = max(position_t, current_t)
                self._gripper_hist.append(
                    np.asarray([position, current], dtype=np.float32)
                )
                if self._gripper_last_pair_t is not None:
                    pair_period = pair_t - self._gripper_last_pair_t
                    if pair_period > 1e-6:
                        if self._gripper_pair_period_ema is None:
                            self._gripper_pair_period_ema = pair_period
                        else:
                            self._gripper_pair_period_ema = (
                                0.9 * self._gripper_pair_period_ema
                                + 0.1 * pair_period
                            )
                self._gripper_last_pair_t = pair_t
                self._gripper_last_pair_skew_sec = abs(skew)
                self._gripper_pair_count += 1
                if self._gripper_pair_count % self.gripper_history_debug_every_n == 0:
                    fill = len(self._gripper_hist) / float(max(1, self.gripper_history_len))
                    observed_hz = (
                        0.0
                        if self._gripper_pair_period_ema is None
                        else 1.0 / max(1e-6, self._gripper_pair_period_ema)
                    )
                    self.get_logger().info(
                        "[GRIPPER-HISTORY] "
                        f"pairs={self._gripper_pair_count}, "
                        f"fill={len(self._gripper_hist)}/{self.gripper_history_len} "
                        f"({fill:.0%}), skew_ms={1000.0 * abs(skew):.2f}, "
                        f"observed_hz={observed_hz:.2f}, "
                        f"target_hz={self.gripper_history_hz:.2f}, "
                        f"drops={self._gripper_pair_drop_count}"
                    )
            elif position_t < current_t:
                self._gripper_position_pending.popleft()
                self._gripper_pair_drop_count += 1
            else:
                self._gripper_current_pending.popleft()
                self._gripper_pair_drop_count += 1

    def _on_gripper_position(self, msg: Int32):
        with self._lock:
            self._gripper_position = int(msg.data)
            self._gripper_position_pending.append(
                (_monotonic(), float(msg.data))
            )
            self._match_gripper_pairs_locked()

    def _on_gripper_current(self, msg: Float32):
        with self._lock:
            self._gripper_current_mA = float(msg.data)
            self._gripper_current_pending.append(
                (_monotonic(), float(msg.data))
            )
            self._match_gripper_pairs_locked()

    def _preprocess_live_image(self, rgb: np.ndarray) -> np.ndarray:
        """
        Causal online stabilization for inference-time camera observation.

        This does not crop or resize. Resizing remains handled by _to_tensor_image_stack().
        It estimates frame-to-frame global translation/rotation, smooths the cumulative
        camera trajectory by EMA, and applies the correction to the current RGB frame.
        """
        if self.camera_preprocess_mode in ("off", "none", "raw") or cv2 is None:
            return rgb.copy()

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        if self._cam_prev_raw_gray is None:
            self._cam_prev_raw_gray = gray
            self._cam_prev_proc_gray = gray
            return rgb.copy()

        dx, dy, da = _estimate_pair_transform(self._cam_prev_raw_gray, gray)
        raw_norm = float(math.sqrt(dx * dx + dy * dy))

        delta = np.array([dx, dy, da], dtype=np.float32)
        self._cam_cum = (self._cam_cum + delta).astype(np.float32)

        alpha = float(np.clip(self.camera_stabilize_alpha, 0.0, 0.999))
        self._cam_smooth_cum = (
            alpha * self._cam_smooth_cum + (1.0 - alpha) * self._cam_cum
        ).astype(np.float32)
        correction = self._cam_smooth_cum - self._cam_cum

        proc = _warp_rgb_affine(
            rgb,
            dx=float(correction[0]),
            dy=float(correction[1]),
            da=float(correction[2]),
            border_mode=self.camera_stabilize_border_mode,
        )
        proc_gray = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)

        if self._cam_prev_proc_gray is not None:
            pdx, pdy, _ = _estimate_pair_transform(self._cam_prev_proc_gray, proc_gray)
            proc_norm = float(math.sqrt(pdx * pdx + pdy * pdy))
        else:
            proc_norm = raw_norm

        beta = 0.05
        if self._cam_frame_count <= 1:
            self._cam_raw_jitter_ema = raw_norm
            self._cam_proc_jitter_ema = proc_norm
        else:
            self._cam_raw_jitter_ema = (1.0 - beta) * self._cam_raw_jitter_ema + beta * raw_norm
            self._cam_proc_jitter_ema = (1.0 - beta) * self._cam_proc_jitter_ema + beta * proc_norm

        self._cam_frame_count += 1
        self._cam_prev_raw_gray = gray
        self._cam_prev_proc_gray = proc_gray

        if self.camera_jitter_report_enable and (self._cam_frame_count % self.camera_jitter_log_every_n == 0):
            reduction = 0.0
            if self._cam_raw_jitter_ema > 1e-9:
                reduction = 100.0 * (self._cam_raw_jitter_ema - self._cam_proc_jitter_ema) / self._cam_raw_jitter_ema
            self.get_logger().info(
                f"[CAM-JITTER] online EMA before={self._cam_raw_jitter_ema:.4f}px, "
                f"after={self._cam_proc_jitter_ema:.4f}px, "
                f"RMS_like_reduction={reduction:.2f}%, mode={self.camera_preprocess_mode}"
            )

        return proc

    def _on_img(self, msg: Image):
        try:
            rgb_raw = _img_to_rgb_numpy(msg)
            rgb = self._preprocess_live_image(rgb_raw)
            with self._lock:
                self._img_cam0 = rgb
        except Exception as e:
            self.get_logger().error(f"[CAM0 IMG] decode/preprocess failed: {e}")
            return
        # Publish the overlay at the camera's own native rate (not tied to
        # the ~few-second replan cadence) so the recorded video is as smooth
        # as the camera feed -- only the blended Grad-CAM heatmap is allowed
        # to lag (it's cached from the last replan tick).
        self._publish_flow_vector_overlay_frame(rgb)

    def _on_global_img(self, msg: Image):
        try:
            rgb = _img_to_rgb_numpy(msg)
            with self._lock:
                self._img_cam1 = rgb.copy()
        except Exception as e:
            self.get_logger().error(f"[CAM1 IMG] decode failed: {e}")

    def _on_stain_mask(self, msg: Image):
        try:
            mask = _mask_msg_to_float_numpy(msg)
            with self._lock:
                self._stain_mask = mask.copy()
        except Exception as e:
            self.get_logger().error(f"[STAIN MASK] decode failed: {e}")

    # ------------------------------------------------------------
    # Contact update
    # ------------------------------------------------------------
    def _update_contact(self, meas_fz: float) -> bool:
        prev = self._contact
        if (not self._contact) and (meas_fz >= self.contact_on_thr):
            self._contact = True
        elif self._contact and (meas_fz <= self.contact_off_thr):
            self._contact = False
        return (prev != self._contact)

    def _compute_preload_target(self) -> float:
        tgt = self.preload_fixed_N

        if (self.preload_target_source == "stats_mean") and (self.stats is not None):
            if getattr(self.stats, "qpos_mode", "zscore") == "zscore":
                mean_fz = float(self.stats.qpos_a[8])
                tgt = abs(mean_fz) * float(self.preload_target_scale)
            elif getattr(self.stats, "qpos_mode", "zscore") in ["minmax_01", "minmax_m11"]:
                qmin_fz = float(self.stats.qpos_a[8])
                qmax_fz = float(self.stats.qpos_b[8])
                mid_fz = 0.5 * (qmin_fz + qmax_fz)
                tgt = abs(mid_fz) * float(self.preload_target_scale)

        tgt = max(float(self.preload_min_N), float(tgt))
        return float(tgt)

    # ------------------------------------------------------------
    # Small helpers: reset dither / kick count
    # ------------------------------------------------------------
    def _reset_dither(self):
        self._pose_hist6.clear()
        self._dither_score = 0.0

    def _reset_kick_count(self):
        self._kick_count = 0
        self._kick_count_t0 = _monotonic()

    # ------------------------------------------------------------
    # Stage transitions
    # ------------------------------------------------------------
    def _enter_preload(self, pose6_now: np.ndarray):
        """
        Enter PRELOAD on first touch instead of handing control straight back
        to the raw FLOW/BSPLINE trajectory. TRACK's cmd is either a stale
        replan (un-anchored absolute target, possibly still mm away from the
        live pose) or a policy prediction with no notion of "we just made
        contact" -- letting it keep driving is what produced the violent
        force spikes at touchdown (20260811). PRELOAD instead closes a slow,
        measured-Fz-driven loop (preload_dz_max_mm per tick) from the live
        pose up to preload_target_N before TRACK resumes.
        """
        self._preload_t0 = _monotonic()
        self._preload_ok = 0
        self._preload_hold_pose6 = pose6_now.astype(np.float32).copy()
        self._preload_target_N = self._compute_preload_target()

        self.stage = Stage.PRELOAD
        self.plans.clear()
        self._anchor_ready = False

        self.get_logger().warn(
            f"[STAGE] -> PRELOAD (touch confirmed, target={self._preload_target_N:.2f}N)"
        )

    def _enter_track(self):
        self.stage = Stage.TRACK
        self.plans.clear()
        self._anchor_ready = False
        self._preload_last_exit_t = _monotonic()

        self._reset_dither()
        self._reset_kick_count()

        self.get_logger().warn("[STAGE] -> TRACK (resume inference)")

    def _enter_release(self, fz_cmd_start: float):
        self.stage = Stage.RELEASE
        self._release_t0 = _monotonic()
        self._release_start_fz_cmd = float(max(0.0, fz_cmd_start))

        self._reset_dither()

        self.get_logger().warn(
            f"[STAGE] -> RELEASE (fz ramp {self._release_start_fz_cmd:.3f} -> 0 in {self.release_ramp_sec:.2f}s)"
        )

    def _soft_reset_to_approach(self, reason: str):
        self.stage = Stage.APPROACH
        self.plans.clear()
        self._anchor_ready = False
        self._touch_ok = 0
        self._fz_kick_active = False

        self._stall_win_pose6 = None
        self._stall_win_t0 = _monotonic()
        self._stall_pose6_lpf = None
        self._recover_pose6_lpf = None

        self._reset_dither()
        self._reset_kick_count()

        self.get_logger().warn(f"[APPROACH-RESET] {reason} -> clear plans/anchor and continue APPROACH (RECOVER removed)")

    # ------------------------------------------------------------
    # Infer timer
    # ------------------------------------------------------------
    def _on_infer_timer(self):
        if not self._srf_ready():
            return
        if self.auto_move_to_demo_start and not self._demo_start_align_done:
            return

        if self.stage == Stage.PRELOAD:
            return

        # FLOW/BSPLINE produce a time-indexed trajectory, not a one-step action.
        # Do not replace it every inference tick before the controller has
        # reached the meaningful part of the plan. A value of zero restores
        # immediate replanning for comparison.
        if (
            not self.visualization_only
            and self.policy_class in ("FLOW", "BSPLINE")
            and self.action_selection_mode == "trajectory_interp"
            and self.flow_replan_interval_steps > 0
            and self.plans
        ):
            plan_age_sec = _monotonic() - float(self.plans[-1].t0)
            min_plan_age_sec = self.flow_replan_interval_steps / self.trajectory_hz
            if plan_age_sec < min_plan_age_sec:
                return

        with self._lock:
            pose6 = None if self._pose6 is None else self._pose6.copy()
            force = None if self._force is None else self._force.copy()
            gripper_position = self._gripper_position
            gripper_current_mA = self._gripper_current_mA
            gripper_hist_list = list(self._gripper_hist)
            gripper_last_pair_t = self._gripper_last_pair_t
            if self.use_gripper_history and gripper_hist_list:
                gripper_position = float(gripper_hist_list[-1][0])
                gripper_current_mA = float(gripper_hist_list[-1][1])
            cam0 = None if self._img_cam0 is None else self._img_cam0.copy()
            cam1 = None if self._img_cam1 is None else self._img_cam1.copy()
            stain_mask_np = None if self._stain_mask is None else self._stain_mask.copy()
            force_hist_list = list(self._force_hist)

        cam1_missing = self.use_global_image and cam1 is None
        stain_missing = self.use_stain_mask and stain_mask_np is None
        gripper_position_ok = gripper_position is not None
        gripper_current_ok = gripper_current_mA is not None
        gripper_history_missing = self.use_gripper_history and not gripper_hist_list
        gripper_history_age = (
            float("inf")
            if gripper_last_pair_t is None
            else max(0.0, _monotonic() - gripper_last_pair_t)
        )
        gripper_history_stale = (
            self.use_gripper_history
            and self.gripper_history_max_age_sec > 0.0
            and gripper_history_age > self.gripper_history_max_age_sec
        )
        gripper_missing = self.use_gripper and (
            not gripper_position_ok
            or not gripper_current_ok
            or gripper_history_missing
            or gripper_history_stale
        )
        if pose6 is None or force is None or cam0 is None or cam1_missing or stain_missing or gripper_missing:
            now_dbg = _monotonic()
            if now_dbg - self._infer_wait_last_log >= 1.0:
                self._infer_wait_last_log = now_dbg
                self.get_logger().warn(
                    "[INFER-WAIT] missing live input -> "
                    f"pose={pose6 is not None}, force={force is not None}, "
                    f"cam0={cam0 is not None}, cam1={cam1 is not None if self.use_global_image else 'disabled'}, "
                    f"stain_mask={stain_mask_np is not None if self.use_stain_mask else 'disabled'}, "
                    f"gripper={not gripper_missing if self.use_gripper else 'disabled'}"
                    f"(pos={gripper_position_ok}, current={gripper_current_ok}, "
                    f"history={len(gripper_hist_list)}/{self.gripper_history_len}, "
                    f"age_ms={1000.0 * gripper_history_age:.1f}, "
                    f"stale={gripper_history_stale}). "
                    "No policy plan will be generated until all are available."
                )
            return
        if force.size < 3:
            now_dbg = _monotonic()
            if now_dbg - self._infer_wait_last_log >= 1.0:
                self._infer_wait_last_log = now_dbg
                self.get_logger().warn(f"[INFER-WAIT] force vector too short: size={force.size}")
            return

        f3 = self._extract_force3(force)
        if self.obs_force_xy_zeroed:
            # This checkpoint was trained with observations/force fx,fy set to
            # 0 (the "fz-only observation" dataset). Their normalization range
            # is degenerate, so a live fx/fy would blow up after normalize --
            # zero them here to match training exactly. action/force is NOT
            # affected (it keeps a real range), and force_xy_cmd_enable still
            # governs whether predicted command fx,fy are used.
            f3 = f3.copy()
            f3[0] = 0.0
            f3[1] = 0.0

        policy_pose6 = pose6[:6].astype(np.float32).copy()
        # Stain-relative checkpoint: absolute measured pose -> relative frame
        # (x,y only) before it becomes policy input. No-op in absolute mode.
        policy_pose6 = self._srf_observation_pose6(policy_pose6).astype(np.float32).copy()
        if self.orientation_lock_enable:
            # Preserve the checkpoint's expected 9-D qpos shape while making
            # orientation a constant, non-informative channel. Only live XYZ,
            # force/force history, and images vary during this ablation.
            policy_pose6[3:6] = self.orientation_lock_rotvec
        q_np = np.concatenate([policy_pose6, f3], axis=0).astype(np.float32)
        q_t = torch.from_numpy(q_np).unsqueeze(0).to(self.device, dtype=torch.float32)

        if self.normalize_qpos_enabled and self.stats is not None:
            q_t = _normalize_qpos(q_t, self.stats)

        force_hist_t = None
        if self.use_force_history:
            hist_np = self._build_live_force_history(force_hist_list, f3)  # (L,3)
            if self.obs_force_xy_zeroed:
                hist_np = hist_np.copy()
                hist_np[:, 0:2] = 0.0
            force_hist_t = torch.from_numpy(hist_np).unsqueeze(0).to(self.device, dtype=torch.float32)  # (1,L,3)
            if self.normalize_qpos_enabled and self.stats is not None:
                force_hist_t = _normalize_force_history(force_hist_t, self.stats)

        try:
            # Rotation canonicalization: rotate the policy's camera frame so
            # the live stain presents at the trained direction. Only the tensor
            # fed to the policy is rotated -- overlays/recordings keep the true
            # view. No-op unless stain_canon_enable and the origin+angle arrived.
            images = [self._canon_rotate_image(cam0)]
            if self.use_global_image:
                images.append(cam1)
            img_t = _to_tensor_image_stack(
                images,
                device=self.device,
                resize_hw=self.resize_hw,
                camera_names=self.camera_names,
            )
            stain_mask_t = None
            if self.use_stain_mask:
                stain_mask_t = _to_tensor_stain_mask(
                    stain_mask_np,
                    device=self.device,
                    resize_hw=self.resize_hw,
                )
            gripper_position_t = None
            gripper_current_t = None
            gripper_history_t = None
            if self.use_gripper:
                gripper_position_t = torch.tensor(
                    [[float(gripper_position)]],
                    dtype=torch.float32,
                    device=self.device,
                )
                gripper_current_t = torch.tensor(
                    [[float(gripper_current_mA)]],
                    dtype=torch.float32,
                    device=self.device,
                )
                if self.stats is None:
                    raise RuntimeError("use_gripper=True requires dataset_stats.pkl")
                if (
                    self.stats.gripper_position_a is not None
                    and self.stats.gripper_position_b is not None
                ):
                    gripper_position_t = _normalize_gripper_position(gripper_position_t, self.stats)
                gripper_current_t = _normalize_gripper_current(gripper_current_t, self.stats)
                if self.use_gripper_history:
                    current_pair = np.asarray(
                        [float(gripper_position), float(gripper_current_mA)],
                        dtype=np.float32,
                    )
                    gripper_history_np = self._build_live_gripper_history(
                        gripper_hist_list,
                        current_pair,
                    )
                    gripper_history_t = torch.from_numpy(gripper_history_np).unsqueeze(0).to(
                        self.device,
                        dtype=torch.float32,
                    )
                    gripper_history_t = _normalize_gripper_history(
                        gripper_history_t,
                        self.stats,
                    )
        except Exception as e:
            self.get_logger().error(f"[INFER] image stack failed: {e}")
            return

        flow_initial_noise_t = None
        try:
            flow_initial_noise_t = self._flow_initial_noise_for_plan(q_t)
            with torch.inference_mode():
                if self.use_gripper:
                    out = self.policy(
                        q_t,
                        img_t,
                        force_history=force_hist_t,
                        gripper_position=gripper_position_t,
                        gripper_current=gripper_current_t,
                        gripper_history=gripper_history_t,
                    )
                elif self.policy_class == "FLOW" and hasattr(self.policy, "sample_action"):
                    out = self.policy.sample_action(
                        qpos=q_t,
                        image=img_t,
                        force_history=force_hist_t if self.use_force_history else None,
                        stain_mask=stain_mask_t,
                        num_steps=self.flow_infer_steps,
                        initial_noise=flow_initial_noise_t,
                    )
                elif self.use_force_history:
                    out = self.policy(q_t, img_t, force_history=force_hist_t, stain_mask=stain_mask_t)
                else:
                    out = self.policy(q_t, img_t, stain_mask=stain_mask_t)

            seq_model = _fix_policy_output_seq(
                out, self.chunk_size, self.policy_class, action_dim=self.action_dim
            )
            seq = seq_model

            if self.denorm_action_enabled and self.stats is not None:
                seq = _denorm_action_seq(seq, self.stats)

            seq_den = seq.detach().cpu().numpy().astype(np.float32)

            # Stain-relative checkpoint: the policy emits x,y in the stain
            # frame. Convert the whole trajectory back to absolute base
            # coordinates HERE, before anything downstream (z offset, safety
            # clips, anchoring, PTP9D, overlays) touches it -- from this point
            # on seq_den is absolute, exactly as an absolute-frame checkpoint's
            # output already is. z / rotation / force pass through unchanged.
            seq_den = self._srf_command_seq(seq_den)

            if abs(self.policy_z_offset_mm) > 1e-9:
                if self.action_type == "absolute":
                    seq_den[:, 2] += np.float32(self.policy_z_offset_mm)

            if self.force_xy_cmd_enable:
                lim_xy = abs(float(self.force_xy_hard_limit))
                seq_den[:, 6:8] = np.clip(seq_den[:, 6:8], -lim_xy, lim_xy)
            else:
                seq_den[:, 6:8] = 0.0
            if self.fz_hard_limit > 0.0:
                seq_den[:, 8] = np.clip(seq_den[:, 8], -self.fz_hard_limit, self.fz_hard_limit)

        except Exception as e:
            self.get_logger().error(f"[INFER] policy forward failed: {e}")
            return

        # Preserve the unanchored, denormalized absolute FLOW output for
        # vector-field diagnosis and explicit one-step motion. This must happen
        # before the legacy local trajectory anchor modifies seq_den in-place.
        self._latest_flow_raw_seq = seq_den.copy()
        self._latest_flow_pose6 = pose6[:6].astype(np.float32).copy()
        self._latest_flow_force3 = f3.astype(np.float32).copy()
        self._latest_flow_plan_t = _monotonic()
        self._log_flow_vector_delta(pose6=pose6[:6], seq_raw=self._latest_flow_raw_seq)

        local_anchor_applied = False
        if (
            self.flow_local_anchor_enable
            and self.action_selection_mode == "trajectory_interp"
            and self.action_type == "absolute"
        ):
            # Receding-horizon local anchoring: every newly generated FLOW
            # trajectory starts at the measured EE pose, while preserving all
            # learned relative motion over its 128-step training time axis.
            local_offset6 = pose6[:6].astype(np.float32) - seq_den[0, :6]
            seq_den[:, :6] = seq_den[:, :6] + local_offset6[None, :]
            local_anchor_applied = True

        self.plans.append(
            Plan(t0=_monotonic(), seq_den=seq_den, local_anchor_applied=local_anchor_applied)
        )
        self._infer_plan_count += 1

        if (
            self._ptp9d_track_active
            and not self.ptp9d_use_stream
            and self.stage == Stage.TRACK
            and not self._ptp9d_inflight
        ):
            self._ptp9d_advance()

        # Optional Grad-CAM debug visualization. This performs a separate backward pass
        # at a low rate, so the control loop and command generation remain unchanged.
        self._run_gradcam_debug(
            images_rgb=images,
            q_t=q_t,
            img_t=img_t,
            force_hist_t=force_hist_t,
            stain_mask_t=stain_mask_t,
            flow_initial_noise_t=flow_initial_noise_t,
            gripper_position_t=gripper_position_t,
            gripper_current_t=gripper_current_t,
            gripper_history_t=gripper_history_t,
        )
        # Standalone position/force/image attribution dashboard. This uses one
        # low-rate batched FLOW velocity probe and never changes the generated plan.
        self._run_modality_importance_debug(
            q_t=q_t,
            img_t=img_t,
            force_hist_t=force_hist_t,
            stain_mask_t=stain_mask_t,
            flow_initial_noise_t=flow_initial_noise_t,
            gripper_position_t=gripper_position_t,
            gripper_current_t=gripper_current_t,
            gripper_history_t=gripper_history_t,
        )

        if self._infer_plan_count <= 3 or (self._infer_plan_count % 20 == 0):
            gripper_dbg = ""
            if self.use_gripper and seq_den.shape[-1] >= 10:
                gripper_dbg = f" gripper_target={seq_den[0,9]:.1f}"
                if seq_den.shape[-1] >= 11:
                    gripper_dbg += f" gripper_goal_current={seq_den[0,10]:.1f}mA"
            self.get_logger().info(
                f"[INFER] plan appended #{self._infer_plan_count} | "
                f"seq_shape={tuple(seq_den.shape)} first_xyz=[{seq_den[0,0]:.3f},{seq_den[0,1]:.3f},{seq_den[0,2]:.3f}] "
                f"first_fxy=[{seq_den[0,6]:.3f},{seq_den[0,7]:.3f}] first_fz={seq_den[0,8]:.3f} "
                f"z_offset={self.policy_z_offset_mm:.3f}{gripper_dbg} plans={len(self.plans)} stage={self.stage.name}"
            )

    # ------------------------------------------------------------
    # Temporal aggregation
    # ------------------------------------------------------------
    def _trajectory_interp_cmd(self, now_t: float) -> Optional[np.ndarray]:
        """Linearly replay the newest plan on the dataset/training time axis."""
        if not self.plans:
            return None
        plan = self.plans[-1]
        seq = plan.seq_den
        if seq.shape[0] == 0:
            return None

        sample_pos = max(0.0, (now_t - plan.t0) * self.trajectory_hz + float(self.pred_step_offset))
        lo = int(math.floor(sample_pos))
        if lo >= seq.shape[0] - 1:
            return seq[-1].astype(np.float32).copy()
        hi = lo + 1
        alpha = float(sample_pos - lo)
        return ((1.0 - alpha) * seq[lo] + alpha * seq[hi]).astype(np.float32)

    def _temporal_agg_cmd(self, now_t: float) -> Optional[np.ndarray]:
        if not self.plans:
            return None

        vals: List[np.ndarray] = []
        wts: List[float] = []

        for p in list(self.plans):
            age_steps = int((now_t - p.t0) * self.control_hz)
            k = age_steps + int(self.pred_step_offset)
            if 0 <= k < p.seq_den.shape[0]:
                v = p.seq_den[k]
                if self.use_temporal_agg and self.temporal_agg_mode == "exp":
                    w = _exp_decay_weight(age_steps, self.temporal_agg_tau_steps)
                else:
                    w = 1.0
                vals.append(v.astype(np.float32))
                wts.append(float(w))

        if len(vals) == 0:
            p = self.plans[-1]
            age_steps = int((now_t - p.t0) * self.control_hz)
            k = int(np.clip(age_steps + int(self.pred_step_offset), 0, p.seq_den.shape[0] - 1))
            return p.seq_den[k].astype(np.float32)

        W = float(np.sum(wts))
        if W <= 1e-9:
            return vals[-1].astype(np.float32)

        dim = int(vals[-1].shape[0])
        acc = np.zeros(dim, dtype=np.float32)
        for v, w in zip(vals, wts):
            acc += (w / W) * v
        return acc.astype(np.float32)

    # ------------------------------------------------------------
    # Publish helpers
    # ------------------------------------------------------------
    def _current_pose6_snapshot(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._pose6 is None else self._pose6.copy()

    def _current_gripper_position_snapshot(self) -> Optional[float]:
        with self._lock:
            return None if self._gripper_position is None else float(self._gripper_position)

    def _hold_cmd_from_pose(self, pose6: np.ndarray) -> np.ndarray:
        hold = np.zeros(9, dtype=np.float32)
        hold[0:6] = np.asarray(pose6, dtype=np.float32).reshape(-1)[:6]
        hold[6:9] = 0.0
        return hold

    def _publish_cmd(self, cmd9: np.ndarray):
        cmd = np.asarray(cmd9, dtype=np.float32).reshape(-1)
        if cmd.size < 9:
            now = _monotonic()
            if now - self._cmd_safety_last_log >= 1.0:
                self._cmd_safety_last_log = now
                self.get_logger().error(f"[CMD-SAFETY] malformed cmd size={cmd.size}; publish skipped")
            return self.prev_cmd.copy() if self.prev_cmd is not None else np.zeros(9, dtype=np.float32)

        cmd = cmd[:9].astype(np.float32).copy()
        # Final publication boundary: when tangential force commands are
        # disabled, no upstream plan/stage path can leak measured-friction
        # targets into the robot command.
        if not self.force_xy_cmd_enable:
            cmd[6:8] = 0.0
        published = cmd
        blocked = False

        if self.cmd_safety_enable:
            reason = ""
            start_envelope_violation = False
            start_delta = np.zeros(3, dtype=np.float32)
            start_xy_radius = 0.0
            pose6 = self._current_pose6_snapshot()
            if self._cmd_safety_latched:
                reason = f"latched start-envelope violation: {self._cmd_safety_latch_reason}"
            elif not np.all(np.isfinite(cmd)):
                reason = "non-finite command"
            elif (
                pose6 is not None
                and self.cmd_safety_max_xyz_from_current_mm > 0.0
                and np.all(np.isfinite(pose6[:6]))
            ):
                dist = float(np.linalg.norm(cmd[0:3] - pose6[:3].astype(np.float32)))
                if dist > float(self.cmd_safety_max_xyz_from_current_mm):
                    reason = (
                        f"xyz target {dist:.3f}mm from current pose "
                        f"(limit={self.cmd_safety_max_xyz_from_current_mm:.3f}mm)"
                    )

            # Apply this only after optional demo-start alignment. At that point
            # _start_pose6 is the policy execution start pose, not the pre-align pose.
            if (
                not reason
                and self._demo_start_align_done
                and self._start_pose6 is not None
                and np.all(np.isfinite(self._start_pose6[:3]))
            ):
                reason, start_delta, start_xy_radius = _start_pose_envelope_violation(
                    cmd[0:3],
                    self._start_pose6[0:3],
                    max_xy_mm=self.cmd_safety_max_xy_from_start_mm,
                    max_z_down_mm=self.cmd_safety_max_z_down_from_start_mm,
                    max_z_up_mm=self.cmd_safety_max_z_up_from_start_mm,
                )
                start_envelope_violation = bool(reason)
                if start_envelope_violation and self.cmd_safety_latch_on_start_limit:
                    self._cmd_safety_latched = True
                    self._cmd_safety_latch_reason = reason

            if reason:
                # Anchor the hold to a single fixed pose the first tick a
                # violation fires, and keep publishing that same frozen pose
                # every tick after -- not a freshly re-sampled pose6 each
                # time. Re-sampling made "hold" chase wherever the robot
                # currently measures itself to be, so if it still had
                # residual velocity/momentum the admittance spring never saw
                # a position error and applied zero braking force: the robot
                # coasted straight through the "hold" into contact instead of
                # being arrested by it (20260811 17:19 FLOW -- ~2s of
                # unresolved ~20mm tracking error, hold engaged, robot fell
                # another ~9mm into a 147N impact anyway).
                if self._cmd_safety_hold_pose6 is None:
                    if pose6 is not None and np.all(np.isfinite(pose6[:6])):
                        self._cmd_safety_hold_pose6 = pose6.astype(np.float32).copy()
                    elif self.prev_cmd is not None:
                        self._cmd_safety_hold_pose6 = self.prev_cmd[:6].astype(np.float32).copy()

                if self._cmd_safety_hold_pose6 is not None:
                    published = self._hold_cmd_from_pose(self._cmd_safety_hold_pose6)
                elif self.prev_cmd is not None:
                    published = self.prev_cmd.astype(np.float32).copy()
                else:
                    published = np.zeros(9, dtype=np.float32)

                self.plans.clear()
                self._anchor_ready = False
                now = _monotonic()
                if now - self._cmd_safety_last_log >= 1.0:
                    self._cmd_safety_last_log = now
                    self.get_logger().error(
                        "[CMD-SAFETY] blocked unsafe command: "
                        f"{reason}. "
                        f"start_delta_xyz=[{start_delta[0]:.3f},{start_delta[1]:.3f},{start_delta[2]:.3f}]mm "
                        f"start_xy_radius={start_xy_radius:.3f}mm "
                        f"latched={int(self._cmd_safety_latched)}. "
                        "Publishing fixed-pose hold anchored at violation onset."
                    )
            else:
                self._cmd_safety_hold_pose6 = None
            blocked = bool(reason)

        m = Float64MultiArray()
        m.data = [float(x) for x in published.reshape(-1).tolist()]
        self.pub_cmd.publish(m)
        self._log_metrics_row(published, blocked)
        return published

    def _publish_gripper_command(self, target_tick: float, now_t: float) -> bool:
        if not self.use_gripper or self.pub_gripper_cmd is None:
            return False

        now = float(now_t)
        present = self._current_gripper_position_snapshot()
        target = float(target_tick)
        if not np.isfinite(target):
            if now - self._gripper_cmd_safety_last_log >= 1.0:
                self._gripper_cmd_safety_last_log = now
                self.get_logger().error("[GRIPPER-CMD-SAFETY] blocked non-finite gripper command")
            return False

        target = float(np.clip(target, self.gripper_command_min_tick, self.gripper_command_max_tick))

        if self._gripper_startup_position is None:
            if present is not None and np.isfinite(present):
                self._gripper_startup_position = float(
                    np.clip(present, self.gripper_command_min_tick, self.gripper_command_max_tick)
                )
            elif self._last_gripper_cmd is not None:
                self._gripper_startup_position = float(self._last_gripper_cmd)
            else:
                self._gripper_startup_position = target

        ramp = self._startup_ramp()
        target = float(self._gripper_startup_position + ramp * (target - self._gripper_startup_position))

        dt = self.dt_control
        base = None
        if self._last_gripper_cmd is not None:
            base = float(self._last_gripper_cmd)
        elif present is not None and np.isfinite(present):
            base = float(np.clip(present, self.gripper_command_min_tick, self.gripper_command_max_tick))

        if base is not None:
            beta = _beta_from_tau(dt, self.tau_sec)
            target = float(base + beta * (target - base))

        if self._last_gripper_cmd is not None:
            caps: List[float] = []
            if self.gripper_command_step_cap_tick > 0.0:
                caps.append(max(1.0, float(self.gripper_command_step_cap_tick) * ramp))
            if self.gripper_command_slew_per_sec > 0.0 and self._last_gripper_cmd_t is not None:
                caps.append(max(1.0, self.gripper_command_slew_per_sec * max(0.0, now - self._last_gripper_cmd_t)))
            if caps:
                max_delta = float(min(caps))
                target = float(np.clip(target, self._last_gripper_cmd - max_delta, self._last_gripper_cmd + max_delta))

        target = float(np.clip(target, self.gripper_command_min_tick, self.gripper_command_max_tick))

        if (
            self.gripper_cmd_safety_enable
            and present is not None
            and np.isfinite(present)
            and self.gripper_cmd_safety_max_tick_from_present > 0.0
        ):
            dist = abs(target - present)
            if dist > self.gripper_cmd_safety_max_tick_from_present:
                target = float(np.clip(present, self.gripper_command_min_tick, self.gripper_command_max_tick))
                self.plans.clear()
                self._anchor_ready = False
                if now - self._gripper_cmd_safety_last_log >= 1.0:
                    self._gripper_cmd_safety_last_log = now
                    self.get_logger().error(
                        "[GRIPPER-CMD-SAFETY] blocked unsafe gripper command: "
                        f"target {dist:.1f} tick from present gripper position "
                        f"(limit={self.gripper_cmd_safety_max_tick_from_present:.1f}). "
                        "Publishing present-position hold."
                    )

        target_i = int(round(float(np.clip(target, self.gripper_command_min_tick, self.gripper_command_max_tick))))
        if (
            self._last_gripper_cmd is not None
            and abs(target_i - self._last_gripper_cmd) < self.gripper_command_deadband_tick
        ):
            return False

        msg = Int32()
        msg.data = target_i
        self.pub_gripper_cmd.publish(msg)
        self._last_gripper_cmd = target_i
        self._last_gripper_cmd_t = now
        return True

    def _publish_gripper_goal_current(self, goal_current_mA: float) -> bool:
        if not self.use_gripper or self.pub_gripper_goal_current is None:
            return False

        value = float(goal_current_mA)
        if not np.isfinite(value):
            self.get_logger().error("[GRIPPER-CURRENT-SAFETY] blocked non-finite goal current command")
            return False

        value = float(np.clip(value, self.gripper_goal_current_min_mA, self.gripper_goal_current_max_mA))
        if (
            self._last_gripper_goal_current_mA is not None
            and abs(value - self._last_gripper_goal_current_mA) < self.gripper_goal_current_deadband_mA
        ):
            return False

        msg = Float32()
        msg.data = value
        self.pub_gripper_goal_current.publish(msg)
        self._last_gripper_goal_current_mA = value
        return True

    def _ramp_from(self, t0: float, ramp_sec: float) -> float:
        if ramp_sec <= 1e-6:
            return 1.0
        t = _monotonic() - float(t0)
        return float(np.clip(t / float(ramp_sec), 0.0, 1.0))

    def _startup_ramp(self) -> float:
        return self._ramp_from(self._t_start, self.startup_ramp_sec)

    # ------------------------------------------------------------
    # PRELOAD control
    # ------------------------------------------------------------
    def _preload_control_step(self, pose6_now: np.ndarray, meas_fz: float) -> np.ndarray:
        hold = self._preload_hold_pose6 if self._preload_hold_pose6 is not None else pose6_now.astype(np.float32)

        cmd = np.zeros(9, dtype=np.float32)
        cmd[0:6] = pose6_now.astype(np.float32)
        cmd[6] = 0.0
        cmd[7] = 0.0
        cmd[8] = 0.0

        if self.press_hold_xy:
            cmd[0] = hold[0]
            cmd[1] = hold[1]
        if self.press_hold_rpy:
            cmd[3] = hold[3]
            cmd[4] = hold[4]
            cmd[5] = hold[5]

        target = float(self._preload_target_N)
        err = float(target - meas_fz)

        dz = self.preload_kp_mm_per_N * max(0.0, err)
        dz = float(np.clip(dz, 0.0, self.preload_dz_max_mm))
        cmd[2] = float(cmd[2] - dz)
        # A false trigger (meas_fz never reaches target) makes err stay
        # large for the whole preload_timeout_sec window, so dz saturates
        # at its per-tick cap every tick -- bound the total blind descent
        # from the entry pose regardless of how long that keeps happening.
        if self.preload_max_descent_mm > 0.0:
            floor_z = float(hold[2]) - self.preload_max_descent_mm
            cmd[2] = float(max(cmd[2], floor_z))

        mode = self.press_force_cmd_mode
        if mode == "zero":
            cmd[8] = 0.0
        elif mode == "target":
            cmd[8] = float(target)
        else:
            prev_fz = float(self.prev_cmd[8]) if (self.prev_cmd is not None) else 0.0
            cmd[8] = float(prev_fz)

        cmd[8] = float(max(0.0, cmd[8]))
        if self.fz_hard_limit > 0.0:
            cmd[8] = float(min(cmd[8], self.fz_hard_limit))
        return cmd

    # ------------------------------------------------------------
    # RELEASE force shaping
    # ------------------------------------------------------------
    def _release_force(self, cmd_target: np.ndarray) -> np.ndarray:
        cmd = cmd_target.astype(np.float32).copy()
        t = _monotonic() - self._release_t0
        if self.release_ramp_sec <= 1e-6:
            s = 1.0
        else:
            s = float(np.clip(t / self.release_ramp_sec, 0.0, 1.0))
        fz = (1.0 - s) * float(self._release_start_fz_cmd)
        cmd[6] = 0.0
        cmd[7] = 0.0
        cmd[8] = float(max(0.0, fz))
        return cmd

    # ------------------------------------------------------------
    # Stall LPF + Window update
    # ------------------------------------------------------------
    def _stall_update(self, pose6_now: np.ndarray) -> float:
        dt = self.dt_control
        beta = _beta_from_tau(dt, self.stall_lpf_tau_sec)

        if self._stall_pose6_lpf is None:
            self._stall_pose6_lpf = pose6_now.astype(np.float32).copy()
        else:
            self._stall_pose6_lpf = (
                self._stall_pose6_lpf + beta * (pose6_now.astype(np.float32) - self._stall_pose6_lpf)
            ).astype(np.float32)

        lp = self._stall_pose6_lpf

        if self._stall_win_pose6 is None:
            self._stall_win_pose6 = lp.copy()
            self._stall_win_t0 = _monotonic()
            return 0.0

        net_dp = float(np.linalg.norm(lp[:3] - self._stall_win_pose6[:3]))
        net_da = float(np.linalg.norm(lp[3:6] - self._stall_win_pose6[3:6]))

        if (net_dp >= self.stall_window_net_pos_eps_mm) or (net_da >= self.stall_window_net_ang_eps_rad):
            self._stall_win_pose6 = lp.copy()
            self._stall_win_t0 = _monotonic()
            return 0.0

        return float(_monotonic() - self._stall_win_t0)

    # ------------------------------------------------------------
    # DITHER update
    # ------------------------------------------------------------
    def _dither_update(self, pose6_now: np.ndarray) -> float:
        self._pose_hist6.append(pose6_now.astype(np.float32).copy())
        if len(self._pose_hist6) < 4:
            return 0.0

        arr = np.stack(self._pose_hist6, axis=0)
        P = arr[:, :3]
        A = arr[:, 3:6]

        net_p = float(np.linalg.norm(P[-1] - P[0]))
        net_a = float(np.linalg.norm(A[-1] - A[0]))

        dP = P[1:] - P[:-1]
        dA = A[1:] - A[:-1]
        path_p = float(np.sum(np.linalg.norm(dP, axis=1)))
        path_a = float(np.sum(np.linalg.norm(dA, axis=1)))

        ratio_p = path_p / max(net_p, 1e-9)
        ratio_a = path_a / max(net_a, 1e-9)

        Pm = np.mean(P, axis=0)
        Am = np.mean(A, axis=0)
        rms_p = float(np.sqrt(np.mean(np.sum((P - Pm) ** 2, axis=1))))
        rms_a = float(np.sqrt(np.mean(np.sum((A - Am) ** 2, axis=1))))

        small_net = (net_p <= self.dither_net_pos_thr_mm) and (net_a <= self.dither_net_ang_thr_rad)
        oscill = (
            (ratio_p >= self.dither_path_ratio_thr) or (ratio_a >= self.dither_path_ratio_thr) or
            (rms_p >= self.dither_rms_pos_thr_mm) or (rms_a >= self.dither_rms_ang_thr_rad)
        )

        inside = bool(small_net and oscill)

        if inside:
            self._dither_score += self.dt_control
        else:
            self._dither_score = max(0.0, self._dither_score - self.dt_control * float(self.dither_decay))

        return float(self._dither_score)

    def _dither_allowed(self, elapsed_since_start: float) -> bool:
        if not self.dither_enable:
            return False
        if elapsed_since_start < self.dither_min_after_start_sec:
            return False
        if self.stage in (Stage.PRELOAD, Stage.RELEASE):
            return False
        if self.dither_only_track and (self.stage != Stage.TRACK):
            return False
        return True

    # ------------------------------------------------------------
    # Kick helper
    # ------------------------------------------------------------
    def _try_start_kick(self, now_t: float, reason: str, age_sec: float):
        if self._fz_kick_active:
            return False
        if (now_t - self._fz_kick_last_end_t) < self.fz_kick_cooldown_sec:
            return False

        self._fz_kick_active = True
        self._fz_kick_t0 = now_t
        self._kick_count += 1
        self._kick_count_t0 = now_t

        self.get_logger().warn(
            f"[{reason}] (contact=1) age={age_sec:.2f}s -> FZ KICK start "
            f"(#{self._kick_count}/{self.kick_max_before_recover}, fz={self.fz_kick_N:.2f}N, dur={self.fz_kick_dur_sec:.2f}s)"
        )

        self._stall_win_pose6 = None
        self._stall_win_t0 = now_t
        return True

    # ------------------------------------------------------------
    # Optional demo-start alignment
    # ------------------------------------------------------------
    def _reset_after_demo_start_alignment(self, pose6_now: np.ndarray, cmd9: np.ndarray, now_t: float):
        """
        Reset only the buffers that can contaminate the policy start after the
        initial move. This function is called only when auto_move_to_demo_start=True.
        """
        self.prev_cmd = cmd9.astype(np.float32).copy()
        self._t_first_pub = now_t
        self._t_start = now_t

        # After auto demo-start alignment, start normal policy tracking directly.
        # The old behavior reset to APPROACH and waited for the touch detector again.
        # That can deadlock when the robot is already in contact at the demo-start pose:
        # the force baseline is re-initialized near the measured contact force, so
        # delta-touch becomes almost zero and the node never enters TRACK.
        self.stage = Stage.TRACK
        self._start_pose6 = pose6_now.astype(np.float32).copy()

        self.plans.clear()
        self._anchor_ready = False
        self._anchor_offset6[:] = 0.0

        self._contact = False
        self._last_contact = False
        self._contact_z_floor_mm = None
        self._touch_ok = 0
        self._preload_trigger_ok = 0
        self._preload_last_exit_t = -1e9

        self._fz_base = 0.0
        self._fz_base_init = False

        self._stall_pose6_lpf = None
        self._stall_win_pose6 = None
        self._stall_win_t0 = now_t

        self._fz_kick_active = False
        self._fz_kick_last_end_t = -1e9
        self._recover_last_end_t = -1e9
        self._recover_pose6_lpf = None

        self._reset_dither()
        self._reset_kick_count()

        # Start policy force history from a neutral value. This prevents the
        # auto-alignment motion from being treated as part of the demonstration.
        self._force_hist.clear()
        for _ in range(max(1, self.force_history_len)):
            self._force_hist.append(np.zeros(3, dtype=np.float32))

        self._infer_wait_last_log = 0.0
        self._ctrl_no_plan_last_log = 0.0
        self._infer_plan_count = 0

        self._ptp9d_track_active = self.track_use_ptp9d_service
        self._ptp9d_inflight = False
        if self._ptp9d_track_active and self.ptp9d_use_stream:
            self._ptp9d_stream_stop()  # defensive: clears any stale queue/thread
            self._ptp9d_stream_start()

    def _start_ptp_alignment(self, pose6_now: np.ndarray):
        """
        Drive current pose -> demo_start_pose_mean through the controller's
        native PTP path (singleArm_cmd/single_arm_command service) instead of
        this node's own smoothstep + admittance position servo.
        """
        self._ptp_alignment_requested = True

        if self.demo_start_pose6 is None:
            self.get_logger().warn("[DEMO_START] no demo_start_pose6. Skip PTP alignment.")
            self._demo_start_align_done = True
            return

        target = self.demo_start_pose6.astype(np.float32).copy()
        target[2] += float(self.demo_start_z_offset_mm)

        align_dist = float(np.linalg.norm(target[0:3] - pose6_now[0:3]))
        if self.demo_start_max_align_dist_mm > 0.0 and align_dist > float(self.demo_start_max_align_dist_mm):
            self.get_logger().error(
                "[DEMO_START-SAFETY] PTP target is too far from current pose: "
                f"dist={align_dist:.3f}mm > limit={self.demo_start_max_align_dist_mm:.3f}mm. "
                "Refusing to PTP -- move the robot near demo_start manually or raise "
                "demo_start_max_align_dist_mm deliberately, then restart."
            )
            return

        if self._ptp_client is None or not self._ptp_client.service_is_ready():
            self.get_logger().error(
                f"[DEMO_START] PTP service '{self.ptp_service_name}' is not available "
                "-- is singleArm_cmd running (dev_ws robot bringup)? Will retry."
            )
            self._ptp_alignment_requested = False
            return

        req = SingleArmCommand.Request()
        req.command_mode = "PTP"
        # singleArm_cmd's PTP_command_gen converts target_pose[3:6] with
        # DegreeToRadian() -- this service wants degrees, unlike every other
        # pose field in this node which is radians.
        req.target_pose = [
            float(target[0]), float(target[1]), float(target[2]),
            float(np.degrees(target[3])),
            float(np.degrees(target[4])),
            float(np.degrees(target[5])),
        ]
        req.target_velocity = float(self.ptp_target_velocity_mm_s)

        self.get_logger().warn(
            "[DEMO_START] requesting PTP to demo_start_pose_mean + world_Z_offset "
            f"({self.demo_start_z_offset_mm:.3f}mm), dist={align_dist:.3f}mm, "
            f"target_velocity={req.target_velocity:.1f}mm/s via {self.ptp_service_name}"
        )
        future = self._ptp_client.call_async(req)
        future.add_done_callback(self._on_ptp_alignment_done)

    def _on_ptp_alignment_done(self, future):
        try:
            resp = future.result()
        except Exception as exc:
            self.get_logger().error(
                f"[DEMO_START] PTP service call raised: {exc}. "
                "Node will stay idle -- fix and restart."
            )
            return

        if not resp.success:
            self.get_logger().error(
                f"[DEMO_START] PTP motion failed: {resp.message}. "
                "Node will stay idle -- fix and restart."
            )
            return

        self.get_logger().warn(f"[DEMO_START] PTP complete: {resp.message}")

        if not self.ptp_switch_to_force_mode:
            self._finish_ptp_alignment()
            return

        # PTP_command_gen switches the low-level controller to Position mode
        # internally -- switch it back to Force before policy inference
        # starts publishing 9D pose+force commands to cmdMotion.
        force_req = SingleArmCommand.Request()
        force_req.command_mode = "Force"
        future2 = self._ptp_client.call_async(force_req)
        future2.add_done_callback(self._on_ptp_force_mode_done)

    def _on_ptp_force_mode_done(self, future):
        try:
            resp = future.result()
        except Exception as exc:
            self.get_logger().error(
                f"[DEMO_START] Force-mode switch raised: {exc}. "
                "Node will stay idle -- fix and restart."
            )
            return

        if not resp.success:
            self.get_logger().error(
                f"[DEMO_START] Force-mode switch failed: {resp.message}. "
                "Node will stay idle -- fix and restart."
            )
            return

        self.get_logger().warn(f"[DEMO_START] Force mode active: {resp.message}.")
        self._finish_ptp_alignment()

    def _finish_ptp_alignment(self):
        """
        PTP_command_gen streams to cmdMotion directly and this node publishes
        nothing while waiting on the PTP/Force-mode service calls, so
        self.prev_cmd, self._start_pose6, self.stage etc. are all still
        whatever they were before the (possibly 100+mm) PTP move -- the same
        staleness _run_demo_start_alignment's own completion always cleared
        via _reset_after_demo_start_alignment(). Skipping that here is what
        sent a stale ~130mm-old target back into TRACK/PRELOAD right after a
        real PTP move landed (20260811 17:14 FLOW run: dangerous motion right
        after arrival). Re-anchor everything to the live pose before letting
        control resume.
        """
        with self._lock:
            pose6_now = None if self._pose6 is None else self._pose6.copy()

        if pose6_now is None:
            self.get_logger().error(
                "[DEMO_START] no live pose available after PTP alignment -- "
                "cannot reset control state. Node will stay idle -- restart."
            )
            return

        pose6_now = pose6_now.astype(np.float32)
        cmd9 = self._hold_cmd_from_pose(pose6_now)
        self._reset_after_demo_start_alignment(
            pose6_now=pose6_now, cmd9=cmd9, now_t=_monotonic()
        )
        self._demo_start_align_done = True
        self.get_logger().warn(
            "[DEMO_START] state reset at live pose "
            f"{np.array2string(pose6_now, precision=3, separator=', ')} -- "
            "stage=TRACK, resuming policy control."
        )

    def _ptp9d_advance(self):
        """
        Drive TRACK as a chain of discrete, blocking PTP9D (pose+force)
        service calls instead of continuous 125Hz cmdMotion streaming. Each
        call carries a short lookahead segment of ptp9d_segment_points
        consecutive predicted waypoints (not just one), which the robot-side
        blender (Y2RobMotion PTP9D_command_gen -> MotionBlender9D) ramps
        through as a single continuous motion instead of decelerating to a
        full stop at every point -- fixes the "stop-start" choppy feel of
        one-point-per-call while keeping each call bounded/verifiable (no
        open-loop reference can run arbitrarily far ahead of the live pose
        between calls the way the continuous topic-interp path could). This
        replaces PRELOAD/hold-anchor/approach-slowdown for TRACK entirely
        rather than layering on top.
        Re-entrant-safe: called from both the infer timer (new plan arrived)
        and the previous call's own done-callback (chain to the next segment).
        """
        if not (self._ptp9d_track_active and self.stage == Stage.TRACK):
            return
        if self.ptp9d_use_stream:
            # Defense in depth: the streaming queue (_ptp9d_stream_topup)
            # owns TRACK in this mode. Both this and the stream client would
            # otherwise publish PTP9D-family commands concurrently -- they
            # did, in an earlier version of this gating, and it made the
            # robot visibly oscillate (20260815 first service_stream test).
            return
        if self._ptp9d_inflight:
            return
        if not self.plans:
            return  # no inference result yet; retried once one arrives

        with self._lock:
            pose6 = None if self._pose6 is None else self._pose6.copy()
            force = None if self._force is None else self._force.copy()
        if pose6 is None:
            return

        # _on_control_timer returns before reaching _update_contact() for
        # this whole stage (guard added above it), so self._contact was
        # frozen at whatever it was on TRACK entry -- permanently False,
        # forcing fz=0 below even after real contact. Update it here so
        # contact is actually recognized once it happens (20260812 04:36
        # BSPLINE: descended to ~178mm, real contact height, then oscillated
        # up/down forever because fz stayed force-suppressed the whole time).
        meas_fz = float(force[2]) if (force is not None and force.size >= 3) else 0.0
        self._update_contact(meas_fz)

        seq = self.plans[-1].seq_den
        if seq.shape[0] == 0:
            return

        pos_now = pose6[:3].astype(np.float64)
        dists_to_now = np.linalg.norm(seq[:, :3].astype(np.float64) - pos_now[None, :], axis=1)
        start_idx = int(np.argmin(dists_to_now))

        # Consecutive (not distance-walked) indices, spaced by
        # ptp9d_segment_stride raw samples -- a fixed lookahead window into
        # the near-term predicted trajectory, so this can never skip ahead
        # to a stale, low-confidence point near the far end of the chunk the
        # way distance-accumulation could in slow-motion (e.g. near-contact
        # hover) regions.
        last_idx = min(
            seq.shape[0] - 1,
            start_idx + self.ptp9d_segment_stride * self.ptp9d_segment_points,
        )
        segment_idx = list(range(start_idx + self.ptp9d_segment_stride, last_idx + 1, self.ptp9d_segment_stride))
        if not segment_idx:
            segment_idx = [min(start_idx + 1, seq.shape[0] - 1)]

        segment = seq[segment_idx].astype(np.float64)
        # Same premature force-mode hijack as the continuous topic path:
        # PTP9D always switches to Force mode, and force_control.cpp treats
        # any |Fd|>0.01N as desired_force_active -- with no real contact yet
        # (actual_force_active false) that drops stiffness to 0 and drives
        # with a fixed +-15N precontact hold, ignoring the requested xyz
        # entirely. Keep fz at 0 until real contact is confirmed so PTP9D
        # calls stay pure position moves (Fd=0 -> full stiffness) pre-contact
        # (20260812 04:15 PTP9D run: -3.13N on the very first waypoint,
        # robot never reached the requested z, "flailing" in the air).
        if self._contact:
            fz_col = segment[:, 8].copy()
            if self.fz_hard_limit > 0.0:
                fz_col = np.clip(fz_col, -self.fz_hard_limit, self.fz_hard_limit)
        else:
            fz_col = np.zeros(segment.shape[0], dtype=np.float64)
        segment[:, 8] = fz_col

        if self._ptp9d_client is None or not self._ptp9d_client.service_is_ready():
            self.get_logger().error(
                f"[PTP9D] service '{self.ptp9d_service_name}' not available -- "
                "will retry once the next plan arrives."
            )
            return

        flat_pose: list = []
        for row in segment:
            flat_pose.extend([
                float(row[0]), float(row[1]), float(row[2]),
                float(np.degrees(row[3])),
                float(np.degrees(row[4])),
                float(np.degrees(row[5])),
                float(row[6]), float(row[7]), float(row[8]),
            ])

        req = SingleArmCommand.Request()
        req.command_mode = "PTP9D"
        req.target_pose = flat_pose
        req.target_velocity = float(self.ptp9d_target_velocity_mm_s)

        self._ptp9d_inflight = True
        target = segment[-1]
        self.get_logger().info(
            f"[PTP9D] -> idx=[{segment_idx[0]}..{segment_idx[-1]}]/{seq.shape[0]-1} "
            f"pts={len(segment_idx)} "
            f"end_xyz=[{target[0]:.2f},{target[1]:.2f},{target[2]:.2f}]mm fz={target[8]:.2f}N"
        )
        # cmd9 logged here is the segment's final requested waypoint, not
        # what _publish_cmd would have sent (this path never calls it) --
        # still gives the same meas-vs-cmd trail in the CSV, just one row
        # per PTP9D call instead of one per 8ms control tick.
        target9 = target.copy()
        self._log_metrics_row(target9.astype(np.float32), False)
        self._ptp9d_last_target9 = target9

        future = self._ptp9d_client.call_async(req)
        future.add_done_callback(self._on_ptp9d_step_done)

    def _on_ptp9d_step_done(self, future):
        self._ptp9d_inflight = False

        try:
            resp = future.result()
        except Exception as exc:
            self.get_logger().error(f"[PTP9D] service call raised: {exc}")
            return

        if not resp.success:
            self.get_logger().error(f"[PTP9D] step failed: {resp.message}")
            return

        if self._ptp9d_last_target9 is not None:
            self._log_metrics_row(self._ptp9d_last_target9.astype(np.float32), False)

        self._ptp9d_advance()

    # ------------------------------------------------------------
    # PTP9D streaming (queued, non-stop-start) -- see PTP9D_command_gen
    # design note in Y2RobMotion robot_command.hpp for the C++ side.
    # Unlike _ptp9d_advance's call-wait-call chain, topup is fire-and-keep-
    # topping-up: it estimates how much queued motion is still buffered on
    # the robot side (_ptp9d_stream_committed_until_t, purely from elapsed
    # wall time -- there is no queue-depth feedback from the service) and
    # only appends more once that buffer runs low, instead of after every
    # single call finishes.
    # ------------------------------------------------------------

    @staticmethod
    def _smooth_columns_ma(arr: np.ndarray, window: int) -> np.ndarray:
        """Centered moving average per column, edge-padded to keep arr's length."""
        if window <= 1 or arr.shape[0] < 2:
            return arr
        window = min(window, arr.shape[0])
        half = window // 2
        padded = np.pad(arr, ((half, window - 1 - half), (0, 0)), mode="edge")
        kernel = np.ones(window, dtype=np.float64) / window
        out = np.empty_like(arr)
        for c in range(arr.shape[1]):
            out[:, c] = np.convolve(padded[:, c], kernel, mode="valid")
        return out

    def _ptp9d_stream_start(self):
        if self._ptp9d_client is None:
            self.get_logger().error(
                f"[PTP9D-STREAM] service '{self.ptp9d_service_name}' client not built."
            )
            return
        req = SingleArmCommand.Request()
        req.command_mode = "PTP9D_STREAM_START"
        req.target_pose = []
        req.target_velocity = float(self.ptp9d_target_velocity_mm_s)

        def _on_start_done(future):
            try:
                resp = future.result()
            except Exception as exc:
                self.get_logger().error(f"[PTP9D-STREAM] start call raised: {exc}")
                return
            if not resp.success:
                self.get_logger().error(f"[PTP9D-STREAM] start failed: {resp.message}")
                return
            self.get_logger().info(f"[PTP9D-STREAM] {resp.message}")

        self._ptp9d_stream_started = True
        self._ptp9d_stream_tail9 = None
        self._ptp9d_stream_committed_until_t = _monotonic()
        self._ptp9d_stream_last_sent_fz = None
        self._ptp9d_stream_last_sent_contact = None
        self._ptp9d_stream_last_force_send_t = 0.0
        future = self._ptp9d_client.call_async(req)
        future.add_done_callback(_on_start_done)

    def _ptp9d_stream_stop(self):
        if self._ptp9d_client is None or not self._ptp9d_client.service_is_ready():
            self._ptp9d_stream_started = False
            return
        req = SingleArmCommand.Request()
        req.command_mode = "PTP9D_STREAM_STOP"
        req.target_pose = []
        req.target_velocity = 0.0

        def _on_stop_done(future):
            try:
                future.result()
            except Exception as exc:
                self.get_logger().error(f"[PTP9D-STREAM] stop call raised: {exc}")

        self._ptp9d_stream_started = False
        future = self._ptp9d_client.call_async(req)
        future.add_done_callback(_on_stop_done)

    def _ptp9d_stream_topup(self):
        if not self._ptp9d_stream_started:
            return
        if self._ptp9d_stream_topup_inflight:
            return
        now_t = _monotonic()
        if (self._ptp9d_stream_committed_until_t - now_t) >= self.ptp9d_stream_min_lookahead_sec:
            return  # still comfortably buffered, nothing to do this tick
        if not self.plans:
            return

        seq_raw = self.plans[-1].seq_den
        if seq_raw.shape[0] == 0:
            return

        # Smooth position/orientation only (not force -- that's handled by
        # the contact-gated suppression right below) so the stream doesn't
        # chase raw per-sample prediction noise point by point the way the
        # first live test did; the batch/blended PTP9D path never had this
        # problem because MotionBlender9D fits one smooth accel-decel curve
        # across a whole segment instead of visiting every raw sample.
        seq = seq_raw.astype(np.float64).copy()
        seq[:, :6] = self._smooth_columns_ma(seq[:, :6], self.ptp9d_stream_smooth_window)

        if self._ptp9d_stream_tail9 is None:
            with self._lock:
                pose6 = None if self._pose6 is None else self._pose6.copy()
            if pose6 is None:
                return
            anchor_xyz = pose6[:3].astype(np.float64)
        else:
            anchor_xyz = self._ptp9d_stream_tail9[:3].astype(np.float64)

        dists = np.linalg.norm(seq[:, :3] - anchor_xyz[None, :], axis=1)
        anchor_idx = int(np.argmin(dists))
        stride = self.ptp9d_stream_stride
        lo = min(anchor_idx + stride, seq.shape[0] - 1)
        hi = min(lo + stride * self.ptp9d_stream_topup_points, seq.shape[0])
        segment_idx = list(range(lo, hi, stride))
        if not segment_idx:
            # Already at the end of this plan's horizon -- nothing new to
            # append until the next inference plan arrives.
            return

        segment = seq[segment_idx]
        # Force is NOT sent through the queue -- see _ptp9d_stream_update_force.
        # It used to be bundled per-point here, but that meant a contact-state
        # change could sit behind up to ~1s of already-queued pre-contact
        # fz=0 points before taking effect, repeatedly making the robot let
        # go right after touching (20260815 live test: contact toggled on/
        # off 9 times instead of one sustained multi-second hold like every
        # training demonstration). These force columns are unused by the
        # robot-side stream thread now; zeroed only to keep the wire payload
        # unambiguous.
        segment[:, 6:9] = 0.0

        if self._ptp9d_client is None or not self._ptp9d_client.service_is_ready():
            return

        flat_pose: list = []
        for row in segment:
            flat_pose.extend([
                float(row[0]), float(row[1]), float(row[2]),
                float(np.degrees(row[3])),
                float(np.degrees(row[4])),
                float(np.degrees(row[5])),
                float(row[6]), float(row[7]), float(row[8]),
            ])

        req = SingleArmCommand.Request()
        req.command_mode = "PTP9D_STREAM_APPEND"
        req.target_pose = flat_pose
        req.target_velocity = float(self.ptp9d_target_velocity_mm_s)

        # meas6/force here are the live pose/force at append time, not a
        # continuous trace of what the C++ stream thread actually executes
        # tick-by-tick (Python has no visibility into that) -- still useful
        # as a coarse "what was requested vs where the robot actually was"
        # series for offline oscillation diagnosis.
        self._log_metrics_row(segment[-1].astype(np.float32), False)

        self._ptp9d_stream_topup_inflight = True
        self._ptp9d_stream_tail9 = segment[-1].copy()
        # Estimate how much wall-clock motion this batch represents at the
        # trajectory's native sample rate -- a rough proxy (real robot-side
        # consumption depends on ptp9d_target_velocity_mm_s and per-point
        # spacing, not this rate), deliberately conservative: an
        # underestimate just triggers extra, harmless top-up calls, while
        # relying on the buffer running dry would delay contact response.
        self._ptp9d_stream_committed_until_t = max(
            self._ptp9d_stream_committed_until_t, now_t
        ) + (len(segment_idx) * stride) / max(1e-6, self.trajectory_hz)

        def _on_append_done(future):
            self._ptp9d_stream_topup_inflight = False
            try:
                resp = future.result()
            except Exception as exc:
                self.get_logger().error(f"[PTP9D-STREAM] append call raised: {exc}")
                return
            if not resp.success:
                self.get_logger().warn(f"[PTP9D-STREAM] append failed: {resp.message}")

        future = self._ptp9d_client.call_async(req)
        future.add_done_callback(_on_append_done)

    def _ptp9d_stream_update_force(self):
        """
        Push the current fz decision straight to the stream thread via
        PTP9D_STREAM_SET_FORCE, independent of _ptp9d_stream_topup's queue --
        see the ptp9d_stream_force_min_interval_sec declare_parameter comment
        for why force can't ride along with the queued waypoints.
        """
        if not self._ptp9d_stream_started:
            return
        if self._ptp9d_stream_force_inflight:
            return
        if not self.plans:
            return

        seq = self.plans[-1].seq_den
        if seq.shape[0] == 0:
            return

        with self._lock:
            pose6 = None if self._pose6 is None else self._pose6.copy()
        if pose6 is None:
            return

        dists = np.linalg.norm(
            seq[:, :3].astype(np.float64) - pose6[:3].astype(np.float64)[None, :], axis=1
        )
        idx = int(np.argmin(dists))

        if self._contact:
            fz = float(seq[idx, 8])
            if self.fz_hard_limit > 0.0:
                fz = float(np.clip(fz, -self.fz_hard_limit, self.fz_hard_limit))
        else:
            fz = 0.0

        contact_changed = self._contact != self._ptp9d_stream_last_sent_contact
        now_t = _monotonic()
        due = (now_t - self._ptp9d_stream_last_force_send_t) >= self.ptp9d_stream_force_min_interval_sec
        fz_changed = (
            self._ptp9d_stream_last_sent_fz is None
            or abs(fz - self._ptp9d_stream_last_sent_fz) >= 0.3
        )
        if not contact_changed and not (due and fz_changed):
            return

        if self._ptp9d_client is None or not self._ptp9d_client.service_is_ready():
            return

        req = SingleArmCommand.Request()
        req.command_mode = "PTP9D_STREAM_SET_FORCE"
        req.target_pose = [0.0, 0.0, fz]
        req.target_velocity = 0.0

        # _ptp9d_stream_topup zeroes the force columns it logs (force no
        # longer rides in the queue), so this is the only place that logs
        # the fz actually being commanded -- without this, metrics CSVs
        # would show cmd_fz_N as always 0 in service_stream runs.
        cmd9 = np.zeros(9, dtype=np.float32)
        cmd9[:6] = pose6[:6].astype(np.float32)
        cmd9[8] = fz
        self._log_metrics_row(cmd9, False)

        self._ptp9d_stream_force_inflight = True
        self._ptp9d_stream_last_sent_fz = fz
        self._ptp9d_stream_last_sent_contact = self._contact
        self._ptp9d_stream_last_force_send_t = now_t

        def _on_set_force_done(future):
            self._ptp9d_stream_force_inflight = False
            try:
                resp = future.result()
            except Exception as exc:
                self.get_logger().error(f"[PTP9D-STREAM] set_force call raised: {exc}")
                return
            if not resp.success:
                self.get_logger().warn(f"[PTP9D-STREAM] set_force failed: {resp.message}")

        future = self._ptp9d_client.call_async(req)
        future.add_done_callback(_on_set_force_done)

    def _run_demo_start_alignment(self, pose6: np.ndarray, now_t: float):
        """
        Move current robot pose to demo_start_pose_mean before policy inference.

        This path is active only when auto_move_to_demo_start=True.
        If auto_move_to_demo_start=False, the original control path is untouched.
        """
        if self.demo_start_pose6 is None:
            self.get_logger().warn("[DEMO_START] no demo_start_pose6. Skip alignment.")
            self._demo_start_align_done = True
            return

        if self.prev_cmd is None:
            return

        demo_align_target_check = self.demo_start_pose6.astype(np.float32).copy()
        demo_align_target_check[2] += float(self.demo_start_z_offset_mm)
        align_dist = float(np.linalg.norm(demo_align_target_check[0:3] - pose6[0:3].astype(np.float32)))
        if self.demo_start_max_align_dist_mm > 0.0 and align_dist > float(self.demo_start_max_align_dist_mm):
            hold = self._hold_cmd_from_pose(pose6)
            published = self._publish_cmd(hold)
            self.prev_cmd = published.copy()

            now_dbg = _monotonic()
            if now_dbg - self._demo_start_safety_last_log >= 1.0:
                self._demo_start_safety_last_log = now_dbg
                self.get_logger().error(
                    "[DEMO_START-SAFETY] alignment target is too far from current pose: "
                    f"dist={align_dist:.3f}mm > limit={self.demo_start_max_align_dist_mm:.3f}mm. "
                    "Holding current pose; move robot near demo_start or raise demo_start_max_align_dist_mm deliberately."
                )
            return

        if self._demo_start_align_t0 is None:
            self._demo_start_align_t0 = now_t
            self._demo_start_from_pose6 = pose6.astype(np.float32).copy()
            self._demo_start_hold_t0 = None
            self.plans.clear()
            self._anchor_ready = False

            target_for_timing = self.demo_start_pose6.astype(np.float32).copy()
            target_for_timing[2] += float(self.demo_start_z_offset_mm)
            pos_dist = float(np.linalg.norm(target_for_timing[0:3] - self._demo_start_from_pose6[0:3]))
            rot_dist = float(np.linalg.norm(target_for_timing[3:6] - self._demo_start_from_pose6[3:6]))
            effective_move_sec = max(1e-6, float(self.demo_start_move_sec))
            # smoothstep ds/dt peaks at 1.5 / duration
            if self.demo_start_max_xyz_speed_mm_s > 0.0:
                effective_move_sec = max(
                    effective_move_sec,
                    1.5 * pos_dist / self.demo_start_max_xyz_speed_mm_s,
                )
            if self.demo_start_max_rot_speed_rad_s > 0.0:
                effective_move_sec = max(
                    effective_move_sec,
                    1.5 * rot_dist / self.demo_start_max_rot_speed_rad_s,
                )
            self._demo_start_effective_move_sec = effective_move_sec
            self.get_logger().warn(
                "[DEMO_START] auto alignment start: current pose -> "
                "demo_start_pose_mean + world_Z_offset "
                f"({self.demo_start_z_offset_mm:.3f} mm), "
                f"distance={pos_dist:.3f}mm/{rot_dist:.4f}rad, "
                f"duration={effective_move_sec:.2f}s"
            )
            self.get_logger().info(
                f"[DEMO_START] from={np.array2string(self._demo_start_from_pose6, precision=4, separator=', ')}"
            )
            demo_align_target = self.demo_start_pose6.astype(np.float32).copy()
            demo_align_target[2] += float(self.demo_start_z_offset_mm)
            self.get_logger().info(
                f"[DEMO_START] to_raw  ={np.array2string(self.demo_start_pose6, precision=4, separator=', ')}"
            )
            self.get_logger().info(
                f"[DEMO_START] to_lift ={np.array2string(demo_align_target, precision=4, separator=', ')}"
            )

        T = self._demo_start_effective_move_sec
        if T is None:
            T = max(1e-6, float(self.demo_start_move_sec))
        elapsed = max(0.0, now_t - float(self._demo_start_align_t0))
        tau = float(np.clip(elapsed / T, 0.0, 1.0))
        smooth = float(3.0 * tau * tau - 2.0 * tau * tau * tau)

        start_pose = self._demo_start_from_pose6
        if start_pose is None:
            start_pose = pose6.astype(np.float32).copy()
            self._demo_start_from_pose6 = start_pose

        target_pose = self.demo_start_pose6.astype(np.float32).copy()
        target_pose[2] += float(self.demo_start_z_offset_mm)
        pose_cmd = ((1.0 - smooth) * start_pose + smooth * target_pose).astype(np.float32)

        cmd = np.zeros(9, dtype=np.float32)
        cmd[0:6] = pose_cmd
        cmd[6:9] = 0.0

        published = self._publish_cmd(cmd)
        self.prev_cmd = published.copy()

        if tau < 1.0:
            if (int(now_t * self.control_hz) % self.debug_every_n) == 0:
                pos_err_cmd = float(np.linalg.norm(target_pose[0:3] - pose_cmd[0:3]))
                rot_err_cmd = float(np.linalg.norm(target_pose[3:6] - pose_cmd[3:6]))
                self.get_logger().info(
                    f"[DEMO_START] moving tau={tau:.3f} "
                    f"cmd_xyz=[{cmd[0]:.3f},{cmd[1]:.3f},{cmd[2]:.3f}] "
                    f"pos_err_cmd={pos_err_cmd:.3f}mm rot_err_cmd={rot_err_cmd:.4f}rad"
                )
            return

        # Do not start policy inference based only on elapsed command time.
        # Wait until the measured TCP is inside the target tolerance and stays
        # there for demo_start_hold_sec.
        pos_err_now = float(np.linalg.norm(pose6[0:3].astype(np.float32) - target_pose[0:3]))
        rot_err_now = float(np.linalg.norm(pose6[3:6].astype(np.float32) - target_pose[3:6]))
        target_reached = (
            pos_err_now <= self.demo_start_position_tolerance_mm
            and rot_err_now <= self.demo_start_rotation_tolerance_rad
        )
        if not target_reached:
            self._demo_start_hold_t0 = None
            now_dbg = _monotonic()
            if now_dbg - self._demo_start_wait_last_log >= 1.0:
                self._demo_start_wait_last_log = now_dbg
                delta_xyz = target_pose[0:3] - pose6[0:3].astype(np.float32)
                # Previous pos_err/rot_err-only log couldn't distinguish "still
                # slowly converging" from "genuinely stuck": both print a
                # similar-looking scalar each second. Per-axis delta plus the
                # last published command makes that distinguishable at a glance.
                self.get_logger().warn(
                    "[DEMO_START] command trajectory finished; waiting for measured TCP: "
                    f"pos_err={pos_err_now:.3f}mm "
                    f"(tol={self.demo_start_position_tolerance_mm:.3f}), "
                    f"rot_err={rot_err_now:.4f}rad "
                    f"(tol={self.demo_start_rotation_tolerance_rad:.4f}) | "
                    f"delta_xyz=[{delta_xyz[0]:+.3f},{delta_xyz[1]:+.3f},{delta_xyz[2]:+.3f}]mm "
                    f"meas_xyz=[{pose6[0]:.3f},{pose6[1]:.3f},{pose6[2]:.3f}] "
                    f"last_cmd_xyz=[{cmd[0]:.3f},{cmd[1]:.3f},{cmd[2]:.3f}]"
                )
            return

        if self._demo_start_hold_t0 is None:
            self._demo_start_hold_t0 = now_t
            self.get_logger().warn(
                f"[DEMO_START] measured TCP reached target tolerance. "
                f"hold {self.demo_start_hold_sec:.2f}s "
                f"(pos_err={pos_err_now:.3f}mm, rot_err={rot_err_now:.4f}rad)"
            )

        if (now_t - float(self._demo_start_hold_t0)) < max(0.0, float(self.demo_start_hold_sec)):
            return

        self._reset_after_demo_start_alignment(pose6_now=pose6, cmd9=cmd, now_t=now_t)
        self._demo_start_align_done = True
        if self.flow_diagnostic_only:
            self.get_logger().warn(
                "[DEMO_START] alignment done -> diagnostic inference only; "
                "automatic policy commands remain OFF"
            )
        else:
            self.get_logger().warn(
                "[DEMO_START] alignment done -> TRACK directly and start normal policy inference"
            )

    # ------------------------------------------------------------
    # Control timer
    # ------------------------------------------------------------
    def _on_control_timer(self):
        now_t = _monotonic()

        with self._lock:
            pose6 = None if self._pose6 is None else self._pose6.copy()
            force = None if self._force is None else self._force.copy()

        if pose6 is None:
            return

        meas_fz = 0.0
        if force is not None and force.size >= 3:
            meas_fz = float(force[2])

        # (1) FIRST publish = current pose hold
        if not self._sent_first_cmd:
            cmd0 = np.zeros(9, dtype=np.float32)
            cmd0[0:6] = pose6.astype(np.float32)
            cmd0[6] = 0.0
            cmd0[7] = 0.0
            cmd0[8] = float(self.first_cmd_fz)

            self._sent_first_cmd = True
            self._t_first_pub = now_t
            self._t_start = now_t

            self.stage = Stage.APPROACH
            self._start_pose6 = pose6.astype(np.float32).copy()

            self._fz_base = max(0.0, meas_fz)
            self._fz_base_init = True
            self._touch_ok = 0

            self._stall_pose6_lpf = None
            self._stall_win_pose6 = None
            self._stall_win_t0 = now_t

            self._fz_kick_active = False
            self._fz_kick_last_end_t = -1e9
            self._recover_last_end_t = -1e9
            self._recover_pose6_lpf = None

            self._reset_dither()
            self._reset_kick_count()

            published = self._publish_cmd(cmd0)
            self.prev_cmd = published.copy()
            self.get_logger().info("[START] First publish = current pose. stage=APPROACH")
            return

        if self.prev_cmd is None:
            return

        if self.auto_move_to_demo_start and not self._demo_start_align_done:
            # A stain-relative checkpoint's demo_start_pose_mean is only usable
            # once the origin is latched (it is absolutized inside _srf_ready).
            if not self._srf_ready():
                return
            if self.demo_start_use_ptp_service:
                # PTP_command_gen streams its own path straight to cmdMotion
                # from singleArm_cmd -- publishing anything from here at the
                # same time would race it on the same topic. Just wait for
                # the async service callback to flip _demo_start_align_done.
                if not self._ptp_alignment_requested:
                    self._start_ptp_alignment(pose6.astype(np.float32))
            else:
                self._run_demo_start_alignment(pose6.astype(np.float32), now_t)
            return

        # Diagnostic mode allows only the one-time, speed-limited alignment
        # above. Once the measured TCP has settled at demo_start_pose_mean,
        # periodic policy/force/trajectory commands are disabled. Subsequent
        # motion is possible only through the explicit /flow_step service.
        if self.flow_diagnostic_only:
            return

        if self._ptp9d_track_active and self.stage == Stage.TRACK:
            # PTP9D drives TRACK entirely via its own async service-call
            # chain (see _ptp9d_advance/_on_ptp9d_step_done), or via the
            # streaming queue (_ptp9d_stream_topup) -- publishing anything
            # from here would race PTP9D_command_gen's own stream to
            # cmdMotion on the same topic.
            if self.ptp9d_use_stream:
                self._update_contact(meas_fz)
                self._ptp9d_stream_topup()
                self._ptp9d_stream_update_force()
            return

        # Debounce PRELOAD entry from TRACK: a single noisy tick crossing
        # contact_on_thr used to be enough to commit to a multi-second
        # PRELOAD hunt. Because preload_dz_max_mm saturates to its max rate
        # whenever meas_fz sits well below target -- exactly what a false
        # trigger looks like -- that turned each false positive into an
        # unbounded ~10mm/s blind descent for up to preload_timeout_sec,
        # repeatedly plunging the tool tens of mm in free space before it
        # finally hit the real surface still moving fast (20260811 v5 FLOW,
        # 225N peak). Require touch_ok_count consecutive ticks above
        # contact_on_thr before actually entering PRELOAD.
        if self.stage == Stage.TRACK:
            if meas_fz >= self.contact_on_thr:
                self._preload_trigger_ok += 1
            else:
                self._preload_trigger_ok = 0
            if self._preload_trigger_ok >= self.touch_ok_count:
                self._preload_trigger_ok = 0
                since_exit = now_t - self._preload_last_exit_t
                if since_exit >= self.preload_reentry_cooldown_sec:
                    self._enter_preload(pose6.astype(np.float32))
                else:
                    self.get_logger().warn(
                        f"[PRELOAD] re-entry suppressed ({since_exit:.3f}s < "
                        f"{self.preload_reentry_cooldown_sec:.2f}s cooldown since last exit)"
                    )

        changed = self._update_contact(meas_fz)
        if changed:
            if self._contact:
                self._contact_z_floor_mm = float(pose6[2])
                self._contact_z_block_count = 0
            else:
                self._contact_z_floor_mm = None
            if self.clear_plans_on_contact_change:
                self.plans.clear()
                self._anchor_ready = False
            z_floor_text = (
                "none"
                if self._contact_z_floor_mm is None
                else f"{self._contact_z_floor_mm:.3f}mm"
            )
            self.get_logger().warn(
                f"[CONTACT] changed -> {int(self._contact)} | meas_fz={meas_fz:.3f} | "
                f"z_floor={z_floor_text} | stage={self.stage.name}"
            )

            self._reset_dither()

            if self.release_assist_enable:
                if (not self._contact) and self._last_contact and (self.stage == Stage.TRACK):
                    fz_start = float(self.prev_cmd[8]) if self.prev_cmd is not None else 0.0
                    self._enter_release(fz_start)

        self._last_contact = self._contact

        if (now_t - self._kick_count_t0) >= self.kick_reset_sec:
            self._reset_kick_count()

        # -----------------------------
        # Stage-dependent cmd_target
        # -----------------------------
        cmd_target = None
        gripper_target_tick = None
        gripper_goal_current_mA = None

        if self.stage == Stage.PRELOAD:
            cmd_target = self._preload_control_step(pose6.astype(np.float32), meas_fz)

            if abs(meas_fz - self._preload_target_N) <= self.preload_tol_N:
                self._preload_ok += 1
            else:
                self._preload_ok = 0

            if self._preload_ok >= self.preload_ok_count:
                self.get_logger().warn(f"[PRELOAD] OK (meas_fz~{self._preload_target_N:.2f}N) for {self.preload_ok_count} ticks -> TRACK")
                self._enter_track()
            else:
                if (_monotonic() - self._preload_t0) >= self.preload_timeout_sec:
                    self.get_logger().warn(f"[PRELOAD] TIMEOUT {self.preload_timeout_sec:.2f}s (meas_fz={meas_fz:.2f}) -> TRACK anyway")
                    self._enter_track()

        else:
            if self.action_selection_mode == "trajectory_interp":
                cmd_pred_full = self._trajectory_interp_cmd(now_t)
            else:
                cmd_pred_full = self._temporal_agg_cmd(now_t)

            if cmd_pred_full is None:
                now_dbg = _monotonic()
                if now_dbg - self._ctrl_no_plan_last_log >= 1.0:
                    self._ctrl_no_plan_last_log = now_dbg
                    with self._lock:
                        has_img_dbg = self._img_cam0 is not None
                        has_pose_dbg = self._pose6 is not None
                        has_force_dbg = self._force is not None
                    self.get_logger().warn(
                        f"[CTRL-HOLD] no policy plan yet -> hold prev_cmd. "
                        f"stage={self.stage.name}, plans={len(self.plans)}, "
                        f"pose={has_pose_dbg}, force={has_force_dbg}, image={has_img_dbg}"
                    )
                published = self._publish_cmd(self.prev_cmd)
                self.prev_cmd = published.copy()
                return

            if self.use_gripper and cmd_pred_full.size >= 10:
                gripper_target_tick = float(cmd_pred_full[9])
            if self.use_gripper and cmd_pred_full.size >= 11:
                gripper_goal_current_mA = float(cmd_pred_full[10])
            cmd_target = cmd_pred_full[:9].astype(np.float32).copy()

            if self.action_type == "delta":
                cmd_target = (self.prev_cmd + cmd_target).astype(np.float32)

            if self.action_selection_mode != "trajectory_interp":
                if not self._anchor_ready:
                    self._anchor_offset6 = (pose6.astype(np.float32) - cmd_target[0:6]).astype(np.float32)
                    self._anchor_ready = True
                    self.get_logger().info("[ANCHOR] initialized")

                cmd_target[0:6] = (cmd_target[0:6] + self._anchor_offset6).astype(np.float32)

            if self.stage == Stage.APPROACH:
                cmd_target[6] = 0.0
                cmd_target[7] = 0.0
                cmd_target[8] = 0.0

            if self.stage == Stage.TRACK and not self._contact:
                # Keep fz at 0 (no premature force-mode hijack) but use the
                # model's own raw fz prediction -- which rises as it expects
                # contact soon -- to slow just the z position rate for a
                # controlled final approach.
                self._approach_slow_active = float(cmd_target[8]) >= self.approach_slow_fz_thr
                cmd_target[8] = 0.0
            else:
                self._approach_slow_active = False

            if self.stage == Stage.RELEASE:
                cmd_target = self._release_force(cmd_target)
                if (_monotonic() - self._release_t0) >= max(1e-6, self.release_ramp_sec):
                    self.stage = Stage.APPROACH
                    self.plans.clear()
                    self._anchor_ready = False
                    self._touch_ok = 0
                    self._reset_dither()
                    self.get_logger().warn("[STAGE] RELEASE done -> APPROACH")

        if self.force_xy_cmd_enable and self.stage == Stage.TRACK:
            lim_xy = abs(float(self.force_xy_hard_limit))
            cmd_target[6] = float(np.clip(cmd_target[6], -lim_xy, lim_xy))
            cmd_target[7] = float(np.clip(cmd_target[7], -lim_xy, lim_xy))
        else:
            cmd_target[6] = 0.0
            cmd_target[7] = 0.0

        cmd_target[8] = float(max(0.0, cmd_target[8]))
        if self.fz_hard_limit > 0.0:
            cmd_target[8] = float(min(cmd_target[8], self.fz_hard_limit))

        # -----------------------------
        # STALL check
        # -----------------------------
        stall_win_age = 0.0
        elapsed_since_start = (now_t - self._t_first_pub) if (self._t_first_pub is not None) else 0.0

        if self._t_first_pub is not None and not self.clean_flow_execution:
            stall_win_age = self._stall_update(pose6.astype(np.float32))

            can_check_stall = (elapsed_since_start >= self.stall_min_after_start_sec)
            stalled = can_check_stall and (stall_win_age >= self.stall_sec)

            if stalled and (self.stage not in (Stage.PRELOAD, Stage.RELEASE)):
                if self._contact:
                    if self.recover_enable and (self._kick_count >= self.kick_max_before_recover) and ((now_t - self._recover_last_end_t) >= self.recover_cooldown_sec):
                        self.get_logger().warn(f"[STALL] contact=1 but kick_count={self._kick_count} >= {self.kick_max_before_recover} -> APPROACH reset (RECOVER removed)")
                        self._soft_reset_to_approach("STALL contact=1 kick limit")
                    else:
                        self._try_start_kick(now_t, reason="STALL", age_sec=stall_win_age)
                else:
                    if self.recover_enable and ((now_t - self._recover_last_end_t) >= self.recover_cooldown_sec):
                        self.get_logger().warn(f"[STALL] (contact=0) window_age={stall_win_age:.2f}s -> APPROACH reset (RECOVER removed)")
                        self._soft_reset_to_approach("STALL contact=0")

            if self._fz_kick_active and ((now_t - self._fz_kick_t0) >= self.fz_kick_dur_sec):
                self._fz_kick_active = False
                self._fz_kick_last_end_t = now_t
                self.plans.clear()
                self._anchor_ready = False
                self.get_logger().warn("[STALL] FZ KICK end -> replan requested")

        # -----------------------------
        # DITHER check
        # -----------------------------
        dither_age = 0.0
        if (
            not self.clean_flow_execution
            and self._t_first_pub is not None
            and self._dither_allowed(elapsed_since_start)
        ):
            dither_age = self._dither_update(pose6.astype(np.float32))

            if dither_age >= self.dither_sec:
                if self._contact:
                    if self.recover_enable and (self._kick_count >= self.kick_max_before_recover) and ((now_t - self._recover_last_end_t) >= self.recover_cooldown_sec):
                        self.get_logger().warn(f"[DITHER] contact=1 and kick_count={self._kick_count} >= {self.kick_max_before_recover} -> APPROACH reset (RECOVER removed)")
                        self._soft_reset_to_approach("DITHER contact=1 kick limit")
                    else:
                        started = self._try_start_kick(now_t, reason="DITHER", age_sec=dither_age)
                        if not started and self.recover_enable and ((now_t - self._recover_last_end_t) >= self.recover_cooldown_sec) and (self._kick_count >= self.kick_max_before_recover):
                            self.get_logger().warn("[DITHER] kick cooldown but kick limit reached -> APPROACH reset (RECOVER removed)")
                            self._soft_reset_to_approach("DITHER cooldown + kick limit")
                else:
                    if self.recover_enable and ((now_t - self._recover_last_end_t) >= self.recover_cooldown_sec):
                        self.get_logger().warn(f"[DITHER] contact=0 age={dither_age:.2f}s -> APPROACH reset (RECOVER removed)")
                        self._soft_reset_to_approach("DITHER contact=0")

                self._reset_dither()

        # -----------------------------
        # Touch detector
        # -----------------------------
        if self.stage == Stage.APPROACH and (not self._fz_kick_active):
            if not self._fz_base_init:
                self._fz_base = max(0.0, meas_fz)
                self._fz_base_init = True
            else:
                beta_base = _beta_from_tau(self.dt_control, self.touch_baseline_tau_sec)
                self._fz_base = float((1.0 - beta_base) * self._fz_base + beta_base * max(0.0, meas_fz))

            if self.touch_use_delta:
                touch_sig = max(0.0, meas_fz - self._fz_base)
            else:
                touch_sig = max(0.0, meas_fz)

            elapsed = elapsed_since_start
            allow_touch = (elapsed >= self.touch_min_after_start_sec)

            if allow_touch and (touch_sig >= self.touch_fz_thr):
                self._touch_ok += 1
            else:
                self._touch_ok = 0

            if self._touch_ok >= self.touch_ok_count:
                self._touch_ok = 0
                self._enter_preload(pose6.astype(np.float32))

        # Do not use pose Z to push farther into the surface after contact.
        # Force loading is controlled independently by action Fz. This guard is
        # deliberately applied after stage selection so no learned/legacy pose
        # path can bypass it.
        if self.contact_z_descent_block_enable and self._contact:
            if self._contact_z_floor_mm is None:
                self._contact_z_floor_mm = float(pose6[2])
            min_safe_z = float(self._contact_z_floor_mm - self.contact_z_descent_margin_mm)
            if float(cmd_target[2]) < min_safe_z:
                requested_z = float(cmd_target[2])
                cmd_target[2] = min_safe_z
                self._contact_z_block_count += 1
                if self._contact_z_block_count <= 3 or self._contact_z_block_count % 100 == 0:
                    self.get_logger().warn(
                        f"[CONTACT-Z-BLOCK] #{self._contact_z_block_count} "
                        f"requested_z={requested_z:.3f} -> {min_safe_z:.3f}mm "
                        f"(contact_floor={self._contact_z_floor_mm:.3f}mm)"
                    )

        # -----------------------------
        # Apply kick
        # -----------------------------
        if self._fz_kick_active and not self.clean_flow_execution:
            cmd_target[8] = float(max(cmd_target[8], self.fz_kick_N))

        cmd_target[8] = float(max(0.0, cmd_target[8]))
        if self.fz_hard_limit > 0.0:
            cmd_target[8] = float(min(cmd_target[8], self.fz_hard_limit))

        if self.orientation_lock_enable:
            # Final action boundary for the orientation ablation: no stage,
            # planner, local anchor, or model prediction may change wx/wy/wz.
            cmd_target[3:6] = self.orientation_lock_rotvec

        # -----------------------------
        # QP-safe slow-follow
        # -----------------------------
        dt = self.dt_control

        if (self.stage == Stage.RECOVER) and self.recover_use_overrides:
            tau = float(self.recover_tau_sec)
            beta = _beta_from_tau(dt, tau)
            ramp = self._ramp_from(self._recover_t0, self.recover_startup_ramp_sec)
            cap_pos = max(1e-9, self.recover_step_cap_pos_mm * ramp)
            cap_ang = max(1e-12, self.recover_step_cap_ang_rad * ramp)
            cap_fz = max(1e-9, self.recover_step_cap_fz * ramp)
        else:
            beta = _beta_from_tau(dt, self.tau_sec)
            ramp = self._startup_ramp()
            cap_pos = max(1e-9, self.step_cap_pos_mm * ramp)
            cap_ang = max(1e-12, self.step_cap_ang_rad * ramp)
            cap_fz = max(1e-9, self.step_cap_fz * ramp)

        d = (cmd_target - self.prev_cmd).astype(np.float32)
        d = (beta * d).astype(np.float32)

        cap_pos_z = cap_pos
        if self._approach_slow_active and self.stage == Stage.TRACK:
            cap_pos_z = max(1e-9, self.approach_slow_step_cap_pos_mm * ramp)

        for i in range(3):
            di = float(d[i])
            this_cap = cap_pos_z if i == 2 else cap_pos
            if abs(di) > this_cap:
                d[i] = float(np.sign(di) * this_cap)
        for i in range(3, 6):
            di = float(d[i])
            if abs(di) > cap_ang:
                d[i] = float(np.sign(di) * cap_ang)
        for i in (6, 7):
            di = float(d[i])
            if abs(di) > cap_pos:
                d[i] = float(np.sign(di) * cap_pos)
        di = float(d[8])
        if abs(di) > cap_fz:
            d[8] = float(np.sign(di) * cap_fz)

        cmd_next = (self.prev_cmd + d).astype(np.float32)

        for i in range(9):
            a0 = float(self.prev_cmd[i])
            a1 = float(cmd_next[i])
            tg = float(cmd_target[i])
            if (a0 - tg) * (a1 - tg) < 0.0:
                cmd_next[i] = tg

        published = self._publish_cmd(cmd_next)
        self.prev_cmd = published.copy()
        if self.use_gripper:
            motion_safe = bool(np.allclose(published, cmd_next, atol=1e-6, rtol=0.0))
            if motion_safe and gripper_target_tick is not None:
                if gripper_goal_current_mA is not None:
                    self._publish_gripper_goal_current(float(gripper_goal_current_mA))
                self._publish_gripper_command(float(gripper_target_tick), now_t)
            else:
                present_grip = self._current_gripper_position_snapshot()
                if present_grip is not None:
                    self._publish_gripper_command(float(present_grip), now_t)

        if (int(now_t * self.control_hz) % self.debug_every_n) == 0:
            base = self._fz_base if self._fz_base_init else 0.0
            touch_sig = max(0.0, meas_fz - base) if self.touch_use_delta else max(0.0, meas_fz)
            self.get_logger().info(
                f"[CTRL] stage={self.stage.name} contact={int(self._contact)} meas_fz={meas_fz:.3f} "
                f"fz_base={base:.3f} touch_sig={touch_sig:.3f} touch_ok={self._touch_ok} | "
                f"stall_win={stall_win_age:.2f}s dither={dither_age:.2f}s kickN={int(self._fz_kick_active)} kickCnt={self._kick_count} | "
                f"beta={beta:.4f} ramp={ramp:.3f} cap(pos={cap_pos:.4f}, ang={cap_ang:.6f}, fz={cap_fz:.4f}) | "
                f"cmd_xyz=[{cmd_next[0]:.3f},{cmd_next[1]:.3f},{cmd_next[2]:.3f}] "
                f"cmd_fxy=[{cmd_next[6]:.3f},{cmd_next[7]:.3f}] cmd_fz={cmd_next[8]:.3f} "
                f"gripper_cmd={self._last_gripper_cmd if self.use_gripper else 'disabled'}"
            )


# ============================================================
# main
# ============================================================

def main(args=None, node_name: str = "inference_core"):
    rclpy.init(args=args)
    node = None
    try:
        node = NodeCmdMotionInfer(node_name=node_name)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
