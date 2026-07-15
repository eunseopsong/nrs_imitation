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
import math
import pickle
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, List
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


# ============================================================
# Helpers (checkpoint auto-discovery)
# ============================================================

def _policy_to_ckpt_subdir(policy_class: str) -> str:
    p = str(policy_class or "FLOW").strip().upper()
    if p == "ACT":
        return "act"
    if p == "DIFFUSION":
        return "diffusion"
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
        if amin.size not in (9, 10) or amax.size != amin.size:
            raise ValueError(f"action min/max size must be 9 or 10. got {amin.size}, {amax.size}")

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
            gripper_current_a=gcmin,
            gripper_current_b=gcmax,
        )

    if all(k in st for k in ["qpos_mean", "qpos_std", "action_mean", "action_std"]):
        qm = np.asarray(st["qpos_mean"], dtype=np.float32).reshape(9)
        qs = _sanitize_std(np.asarray(st["qpos_std"], dtype=np.float32).reshape(9))
        am = np.asarray(st["action_mean"], dtype=np.float32).reshape(-1)
        astd = _sanitize_std(np.asarray(st["action_std"], dtype=np.float32).reshape(-1))
        if am.size not in (9, 10) or astd.size != am.size:
            raise ValueError(f"action mean/std size must be 9 or 10. got {am.size}, {astd.size}")

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


def _normalize_gripper_current(current: torch.Tensor, stats: StatsPack) -> torch.Tensor:
    if stats.gripper_current_a is None or stats.gripper_current_b is None:
        raise RuntimeError("dataset_stats.pkl missing gripper_current_min/gripper_current_max")
    ca = torch.tensor(stats.gripper_current_a, dtype=torch.float32, device=current.device).view(1, 1)
    cb = torch.tensor(stats.gripper_current_b, dtype=torch.float32, device=current.device).view(1, 1)
    den = torch.clamp(cb - ca, min=1e-6)
    c01 = (current - ca) / den
    if stats.qpos_mode == "minmax_m11":
        return torch.clamp(2.0 * c01 - 1.0, -1.0, 1.0)
    if stats.qpos_mode == "zscore":
        return (current - ca) / den
    return torch.clamp(c01, 0.0, 1.0)


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
        self.declare_parameter("policy_class", "FLOW")  # ACT | DIFFUSION | FLOW
        self.declare_parameter("ckpt_auto_subdir", "polishing")
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

        self.declare_parameter("control_hz", 125.0)
        self.declare_parameter("infer_hz", 5.0)

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

        self.declare_parameter("use_temporal_agg", True)
        self.declare_parameter("temporal_agg_mode", "exp")
        self.declare_parameter("temporal_agg_tau_steps", 20.0)
        self.declare_parameter("pred_step_offset", 1)
        self.declare_parameter("max_plans", 6)

        # contact gating
        self.declare_parameter("contact_on_thr", 3.0)
        self.declare_parameter("contact_off_thr", 1.2)
        self.declare_parameter("clear_plans_on_contact_change", False)

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

        # force safety
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

        # Optional policy-output Z offset.
        # This is applied to every denormalized absolute action z target.
        # Default 0.0 preserves the original learned trajectory.
        self.declare_parameter("policy_z_offset_mm", 0.0)

        # Last-line command guard. It holds the current pose instead of publishing
        # a command whose xyz target is implausibly far from /currentP.
        self.declare_parameter("cmd_safety_enable", True)
        self.declare_parameter("cmd_safety_max_xyz_from_current_mm", 200.0)

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

        # diffusion policy config
        self.declare_parameter("diffusion_train_steps", 100)
        self.declare_parameter("diffusion_infer_steps", 10)
        self.declare_parameter("diffusion_beta_start", 1e-4)
        self.declare_parameter("diffusion_beta_end", 2e-2)
        self.declare_parameter("diffusion_loss_type", "mse")

        # FLOW policy config
        self.declare_parameter("flow_infer_steps", 10)
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
        # observations and action[9] is published to /gripper/command. The robot
        # motion command path still uses action[0:9] and the same polishing safety loop.
        self.declare_parameter("use_gripper", False)
        self.declare_parameter("gripper_position_topic", "/gripper/present_position")
        self.declare_parameter("gripper_current_topic", "/gripper/present_current_mA")
        self.declare_parameter("gripper_command_topic", "/gripper/command")
        self.declare_parameter("gripper_command_min_tick", -653)
        self.declare_parameter("gripper_command_max_tick", 733)
        self.declare_parameter("gripper_command_deadband_tick", 2)
        self.declare_parameter("gripper_command_slew_per_sec", 1000.0)
        self.declare_parameter("gripper_command_step_cap_tick", 200.0)
        self.declare_parameter("gripper_cmd_safety_enable", True)
        self.declare_parameter("gripper_cmd_safety_max_tick_from_present", 1500.0)

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
        self._gradcam_target_layer = None
        self._gradcam_fwd_handle = None
        self._gradcam_bwd_handle = None
        self._gradcam_last_log_t = 0.0
        if self.gradcam_save:
            try:
                os.makedirs(self.gradcam_save_dir, exist_ok=True)
            except Exception as e:
                raise RuntimeError(f"Failed to create gradcam_save_dir={self.gradcam_save_dir}: {e}")

        self.control_hz = float(self.get_parameter("control_hz").value)
        self.infer_hz = float(self.get_parameter("infer_hz").value)

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

        self.use_temporal_agg = bool(self.get_parameter("use_temporal_agg").value)
        self.temporal_agg_mode = str(self.get_parameter("temporal_agg_mode").value).strip().lower()
        self.temporal_agg_tau_steps = float(self.get_parameter("temporal_agg_tau_steps").value)
        self.pred_step_offset = int(self.get_parameter("pred_step_offset").value)
        self.max_plans = int(self.get_parameter("max_plans").value)

        self.contact_on_thr = float(self.get_parameter("contact_on_thr").value)
        self.contact_off_thr = float(self.get_parameter("contact_off_thr").value)
        self.clear_plans_on_contact_change = bool(self.get_parameter("clear_plans_on_contact_change").value)

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

        self.fz_hard_limit = float(self.get_parameter("fz_hard_limit").value)

        self.auto_move_to_demo_start = bool(self.get_parameter("auto_move_to_demo_start").value)
        self.demo_start_move_sec = float(self.get_parameter("demo_start_move_sec").value)
        self.demo_start_hold_sec = float(self.get_parameter("demo_start_hold_sec").value)
        self.demo_start_z_offset_mm = float(self.get_parameter("demo_start_z_offset_mm").value)
        self.demo_start_max_align_dist_mm = float(self.get_parameter("demo_start_max_align_dist_mm").value)
        self.policy_z_offset_mm = float(self.get_parameter("policy_z_offset_mm").value)
        self.cmd_safety_enable = bool(self.get_parameter("cmd_safety_enable").value)
        self.cmd_safety_max_xyz_from_current_mm = float(self.get_parameter("cmd_safety_max_xyz_from_current_mm").value)

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
        self.gripper_command_topic = str(self.get_parameter("gripper_command_topic").value)
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
        if self.use_gripper and self.policy_class != "FLOW":
            raise RuntimeError("use_gripper=True currently requires policy_class=FLOW")
        if self.force_msg_type not in ("array", "wrench"):
            raise RuntimeError(f"force_msg_type must be array or wrench, got: {self.force_msg_type}")

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"[INFO] Using device: {self.device}")
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

        # validate paths / resolve checkpoint
        if not self.act_root or not os.path.isdir(self.act_root):
            raise RuntimeError(f"act_root invalid: {self.act_root}")
        if self.policy_class not in ("ACT", "DIFFUSION", "FLOW"):
            raise RuntimeError(f"policy_class must be ACT, DIFFUSION, or FLOW, got: {self.policy_class}")

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

        self.action_dim = 10 if self.use_gripper else 9
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

        # policy
        self.policy = self._load_policy_and_ckpt_from_act_root()
        self._setup_gradcam_hooks()

        # -----------------------------
        # State buffers
        # -----------------------------
        self._lock = threading.Lock()
        self._pose6: Optional[np.ndarray] = None
        self._force: Optional[np.ndarray] = None
        self._gripper_position: Optional[int] = None
        self._gripper_current_mA: Optional[float] = None
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
        self._ctrl_no_plan_last_log = 0.0
        self._cmd_safety_last_log = 0.0
        self._demo_start_safety_last_log = 0.0
        self._last_gripper_cmd: Optional[int] = None
        self._last_gripper_cmd_t: Optional[float] = None
        self._gripper_startup_position: Optional[float] = None
        self._gripper_cmd_safety_last_log = 0.0

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

        # contact state
        self._contact = False
        self._last_contact = False

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

        self.pub_cmd = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)
        self.pub_gripper_cmd = None
        if self.use_gripper:
            self.pub_gripper_cmd = self.create_publisher(Int32, self.gripper_command_topic, 10)
        self.pub_gradcam_overlay = None
        self.pub_gradcam_global_overlay = None
        if self.gradcam_enable and self.gradcam_publish:
            self.pub_gradcam_overlay = self.create_publisher(Image, self.gradcam_overlay_topic, 1)
            self.get_logger().info(f"[GRADCAM] publishing local overlay image: {self.gradcam_overlay_topic}")
            if self.use_global_image:
                self.pub_gradcam_global_overlay = self.create_publisher(Image, self.gradcam_global_overlay_topic, 1)
                self.get_logger().info(f"[GRADCAM] publishing global overlay image: {self.gradcam_global_overlay_topic}")

        self.timer_control = self.create_timer(self.dt_control, self._on_control_timer)
        self.timer_infer = self.create_timer(self.dt_infer, self._on_infer_timer)

        self.get_logger().info(
            "[INFO] ✅ Ready.\n"
            f"  stage_start={self.stage.name}\n"
            f"  pose_topic={self.pose_topic}\n"
            f"  force_topic={self.force_topic} ({self.force_msg_type})\n"
            f"  obs_mode={self.obs_mode} camera_names={self.camera_names}\n"
            f"  image_topic={self.image_topic}\n"
            f"  global_image_topic={self.global_image_topic if self.use_global_image else '(disabled)'}\n"
            f"  cmd_topic={self.cmd_topic}\n"
            f"  gripper(enable={int(self.use_gripper)}, state=({self.gripper_position_topic}, {self.gripper_current_topic}), command={self.gripper_command_topic if self.use_gripper else '(disabled)'})\n"
            f"  image_qos={self.image_qos_str}\n"
            f"  policy_class={self.policy_class} phase_mode={self.phase_mode}\n"
            f"  control_hz={self.control_hz} infer_hz={self.infer_hz}\n"
            f"  use_force_history={int(self.use_force_history)} force_history_len={self.force_history_len}\n"
            f"  diffusion_infer_steps={self.diffusion_infer_steps}\n"
            f"  tau_sec={self.tau_sec} startup_ramp_sec={self.startup_ramp_sec}\n"
            f"  step_caps(pos_mm={self.step_cap_pos_mm}, ang_rad={self.step_cap_ang_rad}, fz={self.step_cap_fz})\n"
            f"  temporal_agg={int(self.use_temporal_agg)} mode={self.temporal_agg_mode} tau_steps={self.temporal_agg_tau_steps} max_plans={self.max_plans}\n"
            f"  contact_gate(on={self.contact_on_thr}, off={self.contact_off_thr}) clear_on_change={int(self.clear_plans_on_contact_change)}\n"
            f"  force_xy_cmd(enable={int(self.force_xy_cmd_enable)}, hard_limit={self.force_xy_hard_limit}N)\n"
            f"  touch(delta={int(self.touch_use_delta)}, thr={self.touch_fz_thr}, ok={self.touch_ok_count}, min_after={self.touch_min_after_start_sec}s, base_tau={self.touch_baseline_tau_sec}s)\n"
            f"  PRELOAD(removed: bypass APPROACH -> TRACK, nominal_src={self.preload_target_source}, nominal_min={self.preload_min_N}N)\n"
            f"  STALL(win_sec={self.stall_sec}, min_after={self.stall_min_after_start_sec}s, lpf_tau={self.stall_lpf_tau_sec}s, net_eps_pos={self.stall_window_net_pos_eps_mm}mm, net_eps_ang={self.stall_window_net_ang_eps_rad}rad)\n"
            f"  KICK(fz={self.fz_kick_N}N/{self.fz_kick_dur_sec}s, cooldown={self.fz_kick_cooldown_sec}s)\n"
            f"  RECOVER(removed)\n"
            f"  DITHER(enable={int(self.dither_enable)}, only_track={int(self.dither_only_track)}, min_after={self.dither_min_after_start_sec}s, win={self.dither_win_sec}s, dur={self.dither_sec}s, net_pos_thr={self.dither_net_pos_thr_mm}mm, ratio_thr={self.dither_path_ratio_thr}, rms_pos_thr={self.dither_rms_pos_thr_mm}mm)\n"
            f"  RELEASE(enable={int(self.release_assist_enable)}, ramp_sec={self.release_ramp_sec})\n"
            f"  DEMO_START(auto={int(self.auto_move_to_demo_start)}, move_sec={self.demo_start_move_sec}, hold_sec={self.demo_start_hold_sec}, z_offset_mm={self.demo_start_z_offset_mm})\n"
            f"  DEMO_START_SAFETY(max_align_dist_mm={self.demo_start_max_align_dist_mm})\n"
            f"  POLICY_OUTPUT(z_offset_mm={self.policy_z_offset_mm})\n"
            f"  CMD_SAFETY(enable={int(self.cmd_safety_enable)}, max_xyz_from_current_mm={self.cmd_safety_max_xyz_from_current_mm})\n"
            f"  GRADCAM(enable={int(self.gradcam_enable)}, layer={self._gradcam_target_layer_name}, target={self.gradcam_target}, every_n_infer={self.gradcam_every_n_infer}, topic={self.gradcam_overlay_topic}, global_topic={self.gradcam_global_overlay_topic if self.use_global_image else '(disabled)'})\n"
        )

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

    # ------------------------------------------------------------
    # Grad-CAM debug helpers
    # ------------------------------------------------------------
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

    def _select_gradcam_scalar(self, seq_phys: torch.Tensor) -> torch.Tensor:
        if seq_phys.dim() != 2 or seq_phys.shape[-1] not in (9, 10):
            raise RuntimeError(f"Grad-CAM seq must be (T,9) or (T,10), got {tuple(seq_phys.shape)}")

        T = int(seq_phys.shape[0])
        s = min(max(0, int(self.gradcam_target_step)), max(0, T - 1))
        e = min(T, s + max(1, int(self.gradcam_target_horizon)))
        block = seq_phys[s:e]
        target = str(self.gradcam_target or "z").strip().lower()

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
                raise RuntimeError("gradcam_target=gripper requires a 10D gripper policy output")
            return block[:, 9].mean()
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

    def _gradcam_policy_forward(
        self,
        q_gc: torch.Tensor,
        img_gc: torch.Tensor,
        fh_gc: Optional[torch.Tensor],
        stain_mask_gc: Optional[torch.Tensor],
        gripper_position_gc: Optional[torch.Tensor],
        gripper_current_gc: Optional[torch.Tensor],
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
                    )

                if hasattr(self.policy, "predict_velocity"):
                    steps = max(1, int(getattr(self.policy, "flow_infer_steps", self.flow_infer_steps)))
                    B = int(q_gc.shape[0])
                    T = int(getattr(self.policy, "num_queries", self.chunk_size))
                    Da = int(getattr(self.policy, "action_dim", self.action_dim))
                    z = torch.randn(B, T, Da, device=q_gc.device, dtype=q_gc.dtype)
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
                    )
                return self.policy.sample_action_with_grad(qpos=q_gc, image=img_gc, stain_mask=stain_mask_gc)

            if hasattr(self.policy, "predict_velocity"):
                steps = max(1, int(getattr(self.policy, "flow_infer_steps", self.flow_infer_steps)))
                B = int(q_gc.shape[0])
                T = int(getattr(self.policy, "num_queries", self.chunk_size))
                Da = int(getattr(self.policy, "action_dim", 9))
                z = torch.randn(B, T, Da, device=q_gc.device, dtype=q_gc.dtype)
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
        gripper_position_t: Optional[torch.Tensor] = None,
        gripper_current_t: Optional[torch.Tensor] = None,
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
        local_published = False

        try:
            was_training = self.policy.training
            self.policy.eval()
            self.policy.zero_grad(set_to_none=True)
            gradcam_params = list(self.policy.parameters())
            gradcam_param_states = [p.requires_grad for p in gradcam_params]
            for p in gradcam_params:
                p.requires_grad_(False)

            q_gc = q_t.detach().clone()
            img_gc = img_t.detach().clone()
            img_gc.requires_grad_(True)
            fh_gc = None if force_hist_t is None else force_hist_t.detach().clone()
            stain_mask_gc = None if stain_mask_t is None else stain_mask_t.detach().clone()
            gp_gc = None if gripper_position_t is None else gripper_position_t.detach().clone()
            gc_gc = None if gripper_current_t is None else gripper_current_t.detach().clone()

            with torch.enable_grad():
                out = self._gradcam_policy_forward(
                    q_gc=q_gc,
                    img_gc=img_gc,
                    fh_gc=fh_gc,
                    stain_mask_gc=stain_mask_gc,
                    gripper_position_gc=gp_gc,
                    gripper_current_gc=gc_gc,
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
            self._gradcam_activation = None
            self._gradcam_gradient = None

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
        }

        if self.use_gripper:
            try:
                stats_obj = _pickle_load_compat(os.path.join(self.ckpt_dir, "dataset_stats.pkl"))
                ckpt_policy_cfg = dict(stats_obj.get("policy_config", {}))
            except Exception:
                ckpt_policy_cfg = {}
            for key in (
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
            args_override["action_dim"] = 10
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

        missing, unexpected = _try_load_state_dict_compat(policy, state_dict)

        if (len(missing) + len(unexpected) > 0) and hasattr(policy, "model"):
            missing2, unexpected2 = _try_load_state_dict_compat(policy.model, state_dict)
            if (len(missing2) + len(unexpected2)) < (len(missing) + len(unexpected)):
                missing, unexpected = missing2, unexpected2

        self.get_logger().info(
            f"[INFO] Loaded ckpt from {ckpt_path}. missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if len(missing) > 0:
            self.get_logger().warn(f"[INFO] missing sample: {list(missing)[:10]}")
        if len(unexpected) > 0:
            self.get_logger().warn(f"[INFO] unexpected sample: {list(unexpected)[:10]}")
        self.get_logger().info(
            f"[INFO] policy_class={policy_class}, obs_mode={self.obs_mode}, camera_names={self.camera_names}, "
            f"use_force_history={self.use_force_history}, force_history_len={self.force_history_len}"
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

    def _on_gripper_position(self, msg: Int32):
        with self._lock:
            self._gripper_position = int(msg.data)

    def _on_gripper_current(self, msg: Float32):
        with self._lock:
            self._gripper_current_mA = float(msg.data)

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
        PRELOAD removed (bypassed).
        Keep this function name so the rest of the code path stays unchanged,
        but transition directly from APPROACH to TRACK once touch is confirmed.
        """
        self._preload_t0 = _monotonic()
        self._preload_ok = 0
        self._preload_hold_pose6 = pose6_now.astype(np.float32).copy()
        self._preload_target_N = self._compute_preload_target()

        self.get_logger().warn(
            f"[STAGE] PRELOAD bypassed -> TRACK directly "
            f"(touch confirmed, nominal_target={self._preload_target_N:.2f}N)"
        )
        self._enter_track()

    def _enter_track(self):
        self.stage = Stage.TRACK
        self.plans.clear()
        self._anchor_ready = False

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
        if self.auto_move_to_demo_start and not self._demo_start_align_done:
            return

        if self.stage == Stage.PRELOAD:
            return

        with self._lock:
            pose6 = None if self._pose6 is None else self._pose6.copy()
            force = None if self._force is None else self._force.copy()
            gripper_position = self._gripper_position
            gripper_current_mA = self._gripper_current_mA
            cam0 = None if self._img_cam0 is None else self._img_cam0.copy()
            cam1 = None if self._img_cam1 is None else self._img_cam1.copy()
            stain_mask_np = None if self._stain_mask is None else self._stain_mask.copy()
            force_hist_list = list(self._force_hist)

        cam1_missing = self.use_global_image and cam1 is None
        stain_missing = self.use_stain_mask and stain_mask_np is None
        gripper_position_ok = gripper_position is not None
        gripper_current_ok = gripper_current_mA is not None
        gripper_missing = self.use_gripper and (not gripper_position_ok or not gripper_current_ok)
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
                    f"(pos={gripper_position_ok}, current={gripper_current_ok}). "
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

        q_np = np.concatenate([pose6[:6], f3], axis=0).astype(np.float32)
        q_t = torch.from_numpy(q_np).unsqueeze(0).to(self.device, dtype=torch.float32)

        if self.normalize_qpos_enabled and self.stats is not None:
            q_t = _normalize_qpos(q_t, self.stats)

        force_hist_t = None
        if self.use_force_history:
            hist_np = self._build_live_force_history(force_hist_list, f3)  # (L,3)
            force_hist_t = torch.from_numpy(hist_np).unsqueeze(0).to(self.device, dtype=torch.float32)  # (1,L,3)
            if self.normalize_qpos_enabled and self.stats is not None:
                force_hist_t = _normalize_force_history(force_hist_t, self.stats)

        try:
            images = [cam0]
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
                gripper_current_t = _normalize_gripper_current(gripper_current_t, self.stats)
        except Exception as e:
            self.get_logger().error(f"[INFER] image stack failed: {e}")
            return

        try:
            with torch.inference_mode():
                if self.use_gripper:
                    out = self.policy(
                        q_t,
                        img_t,
                        force_history=force_hist_t,
                        gripper_position=gripper_position_t,
                        gripper_current=gripper_current_t,
                    )
                elif self.use_force_history:
                    out = self.policy(q_t, img_t, force_history=force_hist_t, stain_mask=stain_mask_t)
                else:
                    out = self.policy(q_t, img_t, stain_mask=stain_mask_t)

            seq = _fix_policy_output_seq(out, self.chunk_size, self.policy_class, action_dim=self.action_dim)

            if self.denorm_action_enabled and self.stats is not None:
                seq = _denorm_action_seq(seq, self.stats)

            seq_den = seq.detach().cpu().numpy().astype(np.float32)
            if abs(self.policy_z_offset_mm) > 1e-9:
                if self.action_type == "absolute":
                    seq_den[:, 2] += np.float32(self.policy_z_offset_mm)

            if self.force_xy_cmd_enable:
                lim_xy = abs(float(self.force_xy_hard_limit))
                seq_den[:, 6:8] = np.clip(seq_den[:, 6:8], -lim_xy, lim_xy)
            else:
                seq_den[:, 6:8] = 0.0
            seq_den[:, 8] = np.clip(seq_den[:, 8], -self.fz_hard_limit, self.fz_hard_limit)

        except Exception as e:
            self.get_logger().error(f"[INFER] policy forward failed: {e}")
            return

        self.plans.append(Plan(t0=_monotonic(), seq_den=seq_den))
        self._infer_plan_count += 1

        # Optional Grad-CAM debug visualization. This performs a separate backward pass
        # at a low rate, so the control loop and command generation remain unchanged.
        self._run_gradcam_debug(
            images_rgb=images,
            q_t=q_t,
            img_t=img_t,
            force_hist_t=force_hist_t,
            stain_mask_t=stain_mask_t,
            gripper_position_t=gripper_position_t,
            gripper_current_t=gripper_current_t,
        )

        if self._infer_plan_count <= 3 or (self._infer_plan_count % 20 == 0):
            gripper_dbg = ""
            if self.use_gripper and seq_den.shape[-1] >= 10:
                gripper_dbg = f" gripper_target={seq_den[0,9]:.1f}"
            self.get_logger().info(
                f"[INFER] plan appended #{self._infer_plan_count} | "
                f"seq_shape={tuple(seq_den.shape)} first_xyz=[{seq_den[0,0]:.3f},{seq_den[0,1]:.3f},{seq_den[0,2]:.3f}] "
                f"first_fxy=[{seq_den[0,6]:.3f},{seq_den[0,7]:.3f}] first_fz={seq_den[0,8]:.3f} "
                f"z_offset={self.policy_z_offset_mm:.3f}{gripper_dbg} plans={len(self.plans)} stage={self.stage.name}"
            )

    # ------------------------------------------------------------
    # Temporal aggregation
    # ------------------------------------------------------------
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
        published = cmd

        if self.cmd_safety_enable:
            reason = ""
            pose6 = self._current_pose6_snapshot()
            if not np.all(np.isfinite(cmd)):
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

            if reason:
                if pose6 is not None and np.all(np.isfinite(pose6[:6])):
                    published = self._hold_cmd_from_pose(pose6)
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
                        f"{reason}. Publishing current-pose hold/previous safe command."
                    )

        m = Float64MultiArray()
        m.data = [float(x) for x in published.reshape(-1).tolist()]
        self.pub_cmd.publish(m)
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

        mode = self.press_force_cmd_mode
        if mode == "zero":
            cmd[8] = 0.0
        elif mode == "target":
            cmd[8] = float(target)
        else:
            prev_fz = float(self.prev_cmd[8]) if (self.prev_cmd is not None) else 0.0
            cmd[8] = float(prev_fz)

        cmd[8] = float(np.clip(cmd[8], 0.0, self.fz_hard_limit))
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
        self._touch_ok = 0

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
            self.get_logger().warn(
                "[DEMO_START] auto alignment start: current pose -> "
                "demo_start_pose_mean + world_Z_offset "
                f"({self.demo_start_z_offset_mm:.3f} mm) over {self.demo_start_move_sec:.2f}s"
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

        # Hold the final target pose for a short time before policy starts.
        if self._demo_start_hold_t0 is None:
            self._demo_start_hold_t0 = now_t
            pos_err_now = float(np.linalg.norm(pose6[0:3].astype(np.float32) - target_pose[0:3]))
            rot_err_now = float(np.linalg.norm(pose6[3:6].astype(np.float32) - target_pose[3:6]))
            self.get_logger().warn(
                f"[DEMO_START] target command reached. hold {self.demo_start_hold_sec:.2f}s "
                f"(current pos_err={pos_err_now:.3f}mm, rot_err={rot_err_now:.4f}rad)"
            )

        if (now_t - float(self._demo_start_hold_t0)) < max(0.0, float(self.demo_start_hold_sec)):
            return

        self._reset_after_demo_start_alignment(pose6_now=pose6, cmd9=cmd, now_t=now_t)
        self._demo_start_align_done = True
        self.get_logger().warn("[DEMO_START] alignment done -> TRACK directly and start normal policy inference")

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
            self._run_demo_start_alignment(pose6.astype(np.float32), now_t)
            return

        changed = self._update_contact(meas_fz)
        if changed:
            if self.clear_plans_on_contact_change:
                self.plans.clear()
                self._anchor_ready = False
            self.get_logger().warn(f"[CONTACT] changed -> {int(self._contact)} | meas_fz={meas_fz:.3f} | stage={self.stage.name}")

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
            cmd_target = cmd_pred_full[:9].astype(np.float32).copy()

            if self.action_type == "delta":
                cmd_target = (self.prev_cmd + cmd_target).astype(np.float32)

            if not self._anchor_ready:
                self._anchor_offset6 = (pose6.astype(np.float32) - cmd_target[0:6]).astype(np.float32)
                self._anchor_ready = True
                self.get_logger().info("[ANCHOR] initialized")

            cmd_target[0:6] = (cmd_target[0:6] + self._anchor_offset6).astype(np.float32)

            if self.stage == Stage.APPROACH:
                cmd_target[6] = 0.0
                cmd_target[7] = 0.0
                cmd_target[8] = 0.0

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

        cmd_target[8] = float(np.clip(cmd_target[8], 0.0, self.fz_hard_limit))

        # -----------------------------
        # STALL check
        # -----------------------------
        stall_win_age = 0.0
        elapsed_since_start = (now_t - self._t_first_pub) if (self._t_first_pub is not None) else 0.0

        if self._t_first_pub is not None:
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
        if self._t_first_pub is not None and self._dither_allowed(elapsed_since_start):
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

        # -----------------------------
        # Apply kick
        # -----------------------------
        if self._fz_kick_active:
            cmd_target[8] = float(max(cmd_target[8], self.fz_kick_N))

        cmd_target[8] = float(np.clip(cmd_target[8], 0.0, self.fz_hard_limit))

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

        for i in range(3):
            di = float(d[i])
            if abs(di) > cap_pos:
                d[i] = float(np.sign(di) * cap_pos)
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
