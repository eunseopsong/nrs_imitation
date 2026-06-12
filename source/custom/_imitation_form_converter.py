#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared converter for hdf5_recorder_* merged HDF5 files.

Output episode layout is intentionally compact:

  episode_0.hdf5
  ├── action/
  │   ├── position
  │   └── force
  └── observations/
      ├── position
      ├── force
      └── images/
          ├── cam0
          ├── stain_mask  # optional, generated/copied from cam0 only
          └── cam1  # dual-camera only

When the input file marks episode_0 as a clean reference, this converter can
generate stain masks for the remaining episodes by comparing each frame to
nearby clean reference frames in pose space.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = PROJECT_ROOT / "datasets"


def _hdf5_files_under(path: Path, recursive: bool = False) -> List[Path]:
    patterns = ["**/*.hdf5", "**/*.h5"] if recursive else ["*.hdf5", "*.h5"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(path.glob(pattern))
    return sorted({p.resolve() for p in files if p.is_file()})


def _newest_file(files: Iterable[Path]) -> Path:
    candidates = list(files)
    if not candidates:
        raise FileNotFoundError("No HDF5 files found.")
    candidates.sort(key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
    return candidates[0]


def _has_completed_episodes(path: Path) -> bool:
    try:
        with h5py.File(str(path), "r") as f:
            if "episodes" not in f:
                return False
            return any(not str(name).endswith("__writing") for name in f["episodes"].keys())
    except Exception:
        return False


def _newest_usable_file(files: Iterable[Path]) -> Path:
    candidates = list(files)
    usable = [p for p in candidates if _has_completed_episodes(p)]
    if usable:
        return _newest_file(usable)
    return _newest_file(candidates)


def resolve_input_h5(input_h5: str, default_root: Path) -> Path:
    raw = str(input_h5 or "").strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.is_file():
            return p
        if p.is_dir():
            merged_dir = p / "merged_hdf5"
            if merged_dir.is_dir():
                files = _hdf5_files_under(merged_dir, recursive=False)
                if files:
                    return _newest_usable_file(files)
            files = _hdf5_files_under(p, recursive=False)
            if files:
                return _newest_usable_file(files)
            files = _hdf5_files_under(p, recursive=True)
            if files:
                return _newest_usable_file(files)
            raise FileNotFoundError(f"No .hdf5/.h5 file found under input directory: {p}")
        raise FileNotFoundError(f"input_h5 does not exist: {p}")

    root = default_root.expanduser().resolve()
    files = []
    for pattern in ("*/merged_hdf5/*.hdf5", "*/merged_hdf5/*.h5"):
        files.extend(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No merged HDF5 found under {root}/*/merged_hdf5/")
    usable = [p for p in files if _has_completed_episodes(p)]
    candidates = usable if usable else files
    candidates.sort(key=lambda p: (p.parent.parent.name, p.stat().st_mtime, str(p)), reverse=True)
    return candidates[0].resolve()


def infer_output_dir(input_h5: Path, output_dir: str, output_name: str = "imitation_form") -> Path:
    raw = str(output_dir or "").strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p
    if input_h5.parent.name == "merged_hdf5":
        return input_h5.parent.parent / output_name
    return input_h5.parent / output_name


def _read_optional_array(g: h5py.Group, paths: Sequence[str], dtype=None) -> Tuple[np.ndarray | None, str]:
    for path in paths:
        try:
            if path in g:
                arr = np.asarray(g[path])
                if dtype is not None:
                    arr = arr.astype(dtype)
                return arr, path
        except Exception:
            pass
    return None, ""


def _ensure_2d_min_dim(arr: np.ndarray, min_dim: int, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1 and arr.size >= min_dim:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < min_dim:
        raise ValueError(f"{name} must be (T,{min_dim}+) but got {arr.shape}")
    return arr[:, :min_dim]


def _ensure_image4(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must be (T,H,W,3) but got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _ensure_stain_mask4(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[:, :, :, None]
    elif arr.ndim == 4 and arr.shape[1] == 1 and arr.shape[-1] != 1:
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.ndim != 4 or arr.shape[-1] != 1:
        raise ValueError(f"stain_mask must be (T,H,W), (T,H,W,1), or (T,1,H,W), got {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _as_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"RGB frame must be (H,W,3), got {arr.shape}")
    if arr.dtype == np.uint8:
        return arr
    out = arr.astype(np.float32)
    if float(np.nanmax(out)) <= 1.5:
        out = out * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _rgb_to_value_saturation(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rgb_u8 = _as_uint8_rgb(rgb)
    if cv2 is not None:
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
        return hsv[:, :, 2].astype(np.float32), hsv[:, :, 1].astype(np.float32)

    x = rgb_u8.astype(np.float32)
    vmax = np.max(x, axis=2)
    vmin = np.min(x, axis=2)
    sat = np.zeros_like(vmax, dtype=np.float32)
    valid = vmax > 1e-6
    sat[valid] = (vmax[valid] - vmin[valid]) / vmax[valid] * 255.0
    return vmax, sat


def _resize_rgb_like(src: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    src = _as_uint8_rgb(src)
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if src.shape[:2] == (target_h, target_w):
        return src
    if cv2 is None:
        raise RuntimeError("cv2 is required to resize reference RGB frames")
    return cv2.resize(src, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _odd_kernel_size(value: int) -> int:
    k = int(value)
    if k <= 1:
        return 0
    return k if k % 2 == 1 else k + 1


def _lighting_normalized_value(value: np.ndarray, blur_kernel: int) -> np.ndarray:
    v = np.asarray(value, dtype=np.float32)
    k = _odd_kernel_size(blur_kernel)
    if k <= 1 or cv2 is None:
        return v
    local = cv2.GaussianBlur(v, (k, k), 0)
    scale = float(np.mean(local)) if local.size else 128.0
    return np.clip(v / np.maximum(local, 1.0) * scale, 0.0, 255.0).astype(np.float32)


def _local_darkness(value: np.ndarray, blur_kernel: int) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(value, dtype=np.float32)
    k = _odd_kernel_size(blur_kernel)
    if k <= 1 or cv2 is None:
        return np.zeros_like(v, dtype=np.float32), v
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    local_surface = cv2.morphologyEx(v, cv2.MORPH_CLOSE, kernel)
    local_dark = np.clip(local_surface - v, 0.0, 255.0).astype(np.float32)
    return local_dark, local_surface.astype(np.float32)


def _adaptive_v_floor(
    value: np.ndarray,
    requested_floor: float,
    percentile: float,
    ratio: float,
    min_floor: float,
) -> float:
    requested_floor = float(requested_floor)
    if requested_floor <= 0.0:
        return requested_floor
    if float(percentile) <= 0.0 or float(ratio) <= 0.0:
        return requested_floor
    arr = np.asarray(value, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return requested_floor
    p = float(np.percentile(finite, np.clip(float(percentile), 1.0, 99.0)))
    adaptive_floor = max(float(min_floor), p * float(ratio))
    return min(requested_floor, adaptive_floor)


def _estimate_alignment_warp(
    reference_rgb: np.ndarray,
    current_rgb: np.ndarray,
    mode: str,
    max_iters: int,
    eps: float,
) -> Tuple[str, Optional[np.ndarray], float]:
    mode = str(mode or "none").strip().lower()
    if cv2 is None or mode in ("", "none", "off"):
        return mode, None, 0.0

    ref = _as_uint8_rgb(reference_rgb)
    cur = _as_uint8_rgb(current_rgb)
    if ref.shape != cur.shape:
        return mode, None, 0.0

    if mode not in ("translation", "euclidean", "affine", "homography"):
        return mode, None, 0.0

    if mode == "homography":
        warp_mode = cv2.MOTION_HOMOGRAPHY
        warp = np.eye(3, 3, dtype=np.float32)
    else:
        warp_mode = {
            "translation": cv2.MOTION_TRANSLATION,
            "euclidean": cv2.MOTION_EUCLIDEAN,
            "affine": cv2.MOTION_AFFINE,
        }[mode]
        warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max(1, int(max_iters)),
        max(float(eps), 1e-9),
    )

    try:
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        cc, warp = cv2.findTransformECC(
            cur_gray,
            ref_gray,
            warp,
            warp_mode,
            criteria,
            None,
            1,
        )
        return mode, warp, float(cc)
    except Exception:
        return mode, None, 0.0


def _alignment_mode_candidates(mode: str) -> Tuple[str, ...]:
    mode = str(mode or "none").strip().lower()
    fallback_order = {
        "homography": ("homography", "affine", "euclidean", "translation"),
        "affine": ("affine", "euclidean", "translation"),
        "euclidean": ("euclidean", "translation"),
        "translation": ("translation",),
    }
    return fallback_order.get(mode, (mode,))


def _warp_array_to_current(
    arr: np.ndarray,
    alignment_mode: str,
    warp: Optional[np.ndarray],
    interpolation: int,
    border_mode: int,
    border_value: int = 0,
) -> np.ndarray:
    if cv2 is None or warp is None:
        return np.asarray(arr)
    src = np.asarray(arr)
    h, w = src.shape[:2]
    flags = int(interpolation) | cv2.WARP_INVERSE_MAP
    if alignment_mode == "homography":
        return cv2.warpPerspective(
            src,
            warp,
            (w, h),
            flags=flags,
            borderMode=border_mode,
            borderValue=border_value,
        )
    return cv2.warpAffine(
        src,
        warp,
        (w, h),
        flags=flags,
        borderMode=border_mode,
        borderValue=border_value,
    )


def _align_reference_to_current(
    reference_rgb: np.ndarray,
    current_rgb: np.ndarray,
    mode: str,
    max_iters: int,
    eps: float,
) -> np.ndarray:
    ref = _as_uint8_rgb(reference_rgb)
    for candidate_mode in _alignment_mode_candidates(mode):
        alignment_mode, warp, _ = _estimate_alignment_warp(
            reference_rgb=ref,
            current_rgb=current_rgb,
            mode=candidate_mode,
            max_iters=max_iters,
            eps=eps,
        )
        if warp is None:
            continue
        aligned = _warp_array_to_current(
            ref,
            alignment_mode=alignment_mode,
            warp=warp,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT,
        )
        return aligned.astype(np.uint8)
    return ref


def _align_mask_to_current(
    reference_mask_u8: np.ndarray,
    reference_rgb: np.ndarray,
    current_rgb: np.ndarray,
    mode: str,
    max_iters: int,
    eps: float,
    min_cc: float,
) -> Tuple[np.ndarray, bool, float]:
    mask = (np.asarray(reference_mask_u8) > 0).astype(np.uint8) * 255
    if cv2 is None:
        return np.zeros_like(mask, dtype=np.uint8), False, 0.0
    best_cc = 0.0
    for candidate_mode in _alignment_mode_candidates(mode):
        alignment_mode, warp, cc = _estimate_alignment_warp(
            reference_rgb=reference_rgb,
            current_rgb=current_rgb,
            mode=candidate_mode,
            max_iters=max_iters,
            eps=eps,
        )
        best_cc = max(best_cc, float(cc))
        if warp is None or float(cc) < float(min_cc):
            continue
        aligned = _warp_array_to_current(
            mask,
            alignment_mode=alignment_mode,
            warp=warp,
            interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            border_value=0,
        )
        return aligned.astype(np.uint8), True, float(cc)
    return np.zeros_like(mask, dtype=np.uint8), False, best_cc


def _constrain_to_near_mask(mask_u8: np.ndarray, core_bool: np.ndarray, kernel_size: int) -> np.ndarray:
    k = _odd_kernel_size(kernel_size)
    if cv2 is None or k <= 1:
        return mask_u8
    core = np.asarray(core_bool, dtype=np.uint8)
    kernel = np.ones((k, k), dtype=np.uint8)
    allowed = cv2.dilate(core, kernel) > 0
    return (np.asarray(mask_u8) * allowed.astype(np.uint8)).astype(np.uint8)


def _filter_components_by_support_ratio(
    mask_bool: np.ndarray,
    support_bool: np.ndarray,
    min_ratio: float,
) -> np.ndarray:
    min_ratio = max(0.0, float(min_ratio))
    mask = np.asarray(mask_bool, dtype=bool)
    if min_ratio <= 0.0 or cv2 is None:
        return mask

    support = np.asarray(support_bool, dtype=bool)
    if support.shape != mask.shape:
        raise ValueError(f"support_bool shape {support.shape} does not match mask shape {mask.shape}")

    binary = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(mask, dtype=bool)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        comp = labels == label
        ratio = float(np.count_nonzero(comp & support)) / float(area)
        if ratio >= min_ratio:
            out |= comp
    return out


def _filter_mask_components_by_signal_ratio(
    mask_u8: np.ndarray,
    signal_bool: np.ndarray,
    min_ratio: float,
) -> np.ndarray:
    min_ratio = max(0.0, float(min_ratio))
    mask = (np.asarray(mask_u8) > 0).astype(np.uint8)
    if min_ratio <= 0.0 or cv2 is None or int(np.count_nonzero(mask)) == 0:
        return (mask.astype(np.uint8) * 255)

    signal = np.asarray(signal_bool, dtype=bool)
    if signal.shape != mask.shape:
        raise ValueError(f"signal_bool shape {signal.shape} does not match mask shape {mask.shape}")

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        comp = labels == label
        ratio = float(np.count_nonzero(comp & signal)) / float(area)
        if ratio >= min_ratio:
            out[comp] = 255
    return out.astype(np.uint8)


def _postprocess_binary_mask(
    mask_bool: np.ndarray,
    min_area: int,
    max_area: int,
    morph_kernel: int,
    hole_close_kernel: int = 0,
    ignore_border_px: int = 0,
    component_context_mask: Optional[np.ndarray] = None,
    component_context_pad: int = 12,
    component_context_min_ratio: float = 0.0,
    component_max_width_frac: float = 0.0,
    component_max_height_frac: float = 0.0,
    component_max_area_frac: float = 0.0,
    component_max_aspect_ratio: float = 0.0,
    fill_holes_max_area: int = 0,
) -> np.ndarray:
    mask = np.asarray(mask_bool, dtype=np.uint8) * 255
    if cv2 is not None:
        k = _odd_kernel_size(morph_kernel)
        if k > 1:
            kernel = np.ones((k, k), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        hk = _odd_kernel_size(hole_close_kernel)
        if hk > 1:
            hole_kernel = np.ones((hk, hk), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, hole_kernel)

    if cv2 is None:
        return mask.astype(np.uint8)

    min_area = max(0, int(min_area))
    max_area = max(0, int(max_area))
    binary = (mask > 0).astype(np.uint8)
    H, W = binary.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    context = None if component_context_mask is None else np.asarray(component_context_mask).astype(bool)
    border = max(0, int(ignore_border_px))
    pad = max(1, int(component_context_pad))
    min_context_ratio = max(0.0, float(component_context_min_ratio))
    max_width_frac = max(0.0, float(component_max_width_frac))
    max_height_frac = max(0.0, float(component_max_height_frac))
    max_area_frac = max(0.0, float(component_max_area_frac))
    max_aspect_ratio = max(0.0, float(component_max_aspect_ratio))
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_area > 0 and area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        if max_width_frac > 0.0 and (float(w) / max(W, 1)) > max_width_frac:
            continue
        if max_height_frac > 0.0 and (float(h) / max(H, 1)) > max_height_frac:
            continue
        if max_area_frac > 0.0 and (float(area) / max(H * W, 1)) > max_area_frac:
            continue
        if max_aspect_ratio > 0.0:
            aspect = max(float(w) / max(float(h), 1.0), float(h) / max(float(w), 1.0))
            if aspect > max_aspect_ratio:
                continue
        if border > 0 and (x <= border or y <= border or x + w >= W - border or y + h >= H - border):
            continue
        if context is not None and min_context_ratio > 0.0 and context.shape == binary.shape:
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(W, x + w + pad)
            y1 = min(H, y + h + pad)
            comp_roi = labels[y0:y1, x0:x1] == label
            ring = ~comp_roi
            denom = int(np.count_nonzero(ring))
            if denom > 0:
                ratio = float(np.count_nonzero(context[y0:y1, x0:x1] & ring)) / float(denom)
                if ratio < min_context_ratio:
                    continue
        out[labels == label] = 255
    return _fill_mask_holes(out, fill_holes_max_area)


def _fill_mask_holes(mask_u8: np.ndarray, max_hole_area: int) -> np.ndarray:
    max_hole_area = int(max_hole_area)
    if max_hole_area == 0 or cv2 is None:
        return mask_u8.astype(np.uint8)

    binary = (np.asarray(mask_u8) > 0).astype(np.uint8)
    inv = (binary == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    H, W = binary.shape[:2]
    out = binary.copy()
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_border = x == 0 or y == 0 or (x + w) >= W or (y + h) >= H
        if touches_border:
            continue
        if max_hole_area < 0 or area <= max_hole_area:
            out[labels == label] = 1
    return (out.astype(np.uint8) * 255)


def _scaled_pose_array(position: np.ndarray, pos_scale: float, rot_scale: float, name: str) -> np.ndarray:
    pose = _ensure_2d_min_dim(position, 6, name).astype(np.float32)
    pos_scale = max(float(pos_scale), 1e-6)
    rot_scale = max(float(rot_scale), 1e-6)
    return np.concatenate([pose[:, :3] / pos_scale, pose[:, 3:6] / rot_scale], axis=1).astype(np.float32)


def _nearest_reference_index_bank_with_distances(
    current_position: np.ndarray,
    reference_position: np.ndarray,
    pos_scale: float,
    rot_scale: float,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    cur = _scaled_pose_array(current_position, pos_scale, rot_scale, "current_position")
    ref = _scaled_pose_array(reference_position, pos_scale, rot_scale, "reference_position")
    if ref.shape[0] <= 0:
        raise ValueError("reference_position must contain at least one frame")

    k = min(max(1, int(top_k)), int(ref.shape[0]))
    out = np.zeros((cur.shape[0], k), dtype=np.int64)
    out_dist = np.zeros((cur.shape[0], k), dtype=np.float32)
    for i in range(cur.shape[0]):
        dist = np.sum((ref - cur[i][None, :]) ** 2, axis=1)
        if k == 1:
            nearest_idx = int(np.argmin(dist))
            out[i, 0] = nearest_idx
            out_dist[i, 0] = float(dist[nearest_idx])
        else:
            nearest = np.argpartition(dist, k - 1)[:k]
            nearest = nearest[np.argsort(dist[nearest])].astype(np.int64)
            out[i] = nearest
            out_dist[i] = dist[nearest].astype(np.float32)
    return out, out_dist


def _monotonic_reference_center_indices(
    current_position: np.ndarray,
    reference_position: np.ndarray,
    pos_scale: float,
    rot_scale: float,
) -> np.ndarray:
    cur = _scaled_pose_array(current_position, pos_scale, rot_scale, "current_position")
    ref = _scaled_pose_array(reference_position, pos_scale, rot_scale, "reference_position")
    T, R = int(cur.shape[0]), int(ref.shape[0])
    if T <= 0 or R <= 0:
        raise ValueError("current_position and reference_position must be non-empty")

    cost = np.sum((cur[:, None, :] - ref[None, :, :]) ** 2, axis=2).astype(np.float32)
    dp = np.full((T, R), np.inf, dtype=np.float32)
    back = np.zeros((T, R), dtype=np.uint8)
    dp[0, 0] = cost[0, 0]

    for i in range(1, T):
        dp[i, 0] = cost[i, 0] + dp[i - 1, 0]
        back[i, 0] = 1
    for j in range(1, R):
        dp[0, j] = cost[0, j] + dp[0, j - 1]
        back[0, j] = 2

    for i in range(1, T):
        prev_row = dp[i - 1]
        row = dp[i]
        for j in range(1, R):
            diag = float(prev_row[j - 1])
            up = float(prev_row[j])
            left = float(row[j - 1])
            if diag <= up and diag <= left:
                row[j] = cost[i, j] + diag
                back[i, j] = 0
            elif up <= left:
                row[j] = cost[i, j] + up
                back[i, j] = 1
            else:
                row[j] = cost[i, j] + left
                back[i, j] = 2

    mapping = np.full((T,), -1, dtype=np.int64)
    mapping_cost = np.full((T,), np.inf, dtype=np.float32)
    i, j = T - 1, R - 1
    while True:
        c = cost[i, j]
        if mapping[i] < 0 or c < mapping_cost[i]:
            mapping[i] = j
            mapping_cost[i] = c
        if i == 0 and j == 0:
            break
        step = int(back[i, j])
        if step == 0:
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        else:
            j -= 1

    missing = np.where(mapping < 0)[0]
    if missing.size > 0:
        known = np.where(mapping >= 0)[0]
        if known.size == 0:
            return np.linspace(0, R - 1, T).round().astype(np.int64)
        mapping[missing] = np.interp(missing, known, mapping[known]).round().astype(np.int64)
    return np.clip(mapping, 0, R - 1).astype(np.int64)


def _reference_index_bank_with_distances(
    current_position: np.ndarray,
    reference_position: np.ndarray,
    pos_scale: float,
    rot_scale: float,
    top_k: int,
    match_mode: str,
    match_window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mode = str(match_mode or "nearest").strip().lower()
    if mode in ("nearest", "pose_nearest", "independent"):
        indices, distances = _nearest_reference_index_bank_with_distances(
            current_position=current_position,
            reference_position=reference_position,
            pos_scale=pos_scale,
            rot_scale=rot_scale,
            top_k=top_k,
        )
        return indices, distances, indices[:, 0].copy()

    cur = _scaled_pose_array(current_position, pos_scale, rot_scale, "current_position")
    ref = _scaled_pose_array(reference_position, pos_scale, rot_scale, "reference_position")
    if ref.shape[0] <= 0:
        raise ValueError("reference_position must contain at least one frame")
    centers = _monotonic_reference_center_indices(
        current_position=current_position,
        reference_position=reference_position,
        pos_scale=pos_scale,
        rot_scale=rot_scale,
    )

    k = min(max(1, int(top_k)), int(ref.shape[0]))
    window = max(0, int(match_window))
    out = np.zeros((cur.shape[0], k), dtype=np.int64)
    out_dist = np.zeros((cur.shape[0], k), dtype=np.float32)
    for i in range(cur.shape[0]):
        center = int(centers[i])
        lo = max(0, center - window)
        hi = min(int(ref.shape[0]), center + window + 1)
        candidates = np.arange(lo, hi, dtype=np.int64)
        if candidates.size == 0:
            candidates = np.asarray([center], dtype=np.int64)
        dist = np.sum((ref[candidates] - cur[i][None, :]) ** 2, axis=1)
        take = min(k, int(candidates.size))
        order = np.argsort(dist)[:take]
        chosen = candidates[order]
        chosen_dist = dist[order]
        if take < k:
            pad_count = k - take
            chosen = np.concatenate([chosen, np.repeat(chosen[-1], pad_count)])
            chosen_dist = np.concatenate([chosen_dist, np.repeat(chosen_dist[-1], pad_count)])
        out[i] = chosen.astype(np.int64)
        out_dist[i] = chosen_dist.astype(np.float32)
    return out, out_dist, centers.astype(np.int64)


def _nearest_reference_index_bank(
    current_position: np.ndarray,
    reference_position: np.ndarray,
    pos_scale: float,
    rot_scale: float,
    top_k: int,
) -> np.ndarray:
    indices, _ = _nearest_reference_index_bank_with_distances(
        current_position=current_position,
        reference_position=reference_position,
        pos_scale=pos_scale,
        rot_scale=rot_scale,
        top_k=top_k,
    )
    return indices


def _nearest_reference_indices(
    current_position: np.ndarray,
    reference_position: np.ndarray,
    pos_scale: float,
    rot_scale: float,
) -> np.ndarray:
    return _nearest_reference_index_bank(
        current_position=current_position,
        reference_position=reference_position,
        pos_scale=pos_scale,
        rot_scale=rot_scale,
        top_k=1,
    )[:, 0]


def _bool_consensus(candidates: Sequence[np.ndarray], min_support: float):
    if not candidates:
        return True
    stack = np.stack([np.asarray(c, dtype=bool) for c in candidates], axis=0)
    support = np.clip(float(min_support), 0.0, 1.0)
    if stack.shape[0] <= 1 or support <= 0.0:
        return np.any(stack, axis=0)
    return np.mean(stack, axis=0) >= support


def _find_nearest_strong_mask(
    counts: np.ndarray,
    index: int,
    direction: int,
    max_gap: int,
    min_pixels: int,
) -> int:
    T = int(counts.shape[0])
    step = 1 if int(direction) >= 0 else -1
    for gap in range(1, max(0, int(max_gap)) + 1):
        j = int(index) + step * gap
        if j < 0 or j >= T:
            break
        if int(counts[j]) >= int(min_pixels):
            return j
    return -1


def _temporal_fill_stain_mask_gaps(
    images_rgb: np.ndarray,
    masks: np.ndarray,
    rescue_gates: np.ndarray,
    min_pixels: int,
    max_gap: int,
    align_mode: str,
    align_max_iters: int,
    align_eps: float,
    min_align_cc: float,
    identity_fallback_max_gap: int,
    min_area: int,
    max_area: int,
    morph_kernel: int,
    hole_close_kernel: int,
    ignore_border_px: int,
    component_max_width_frac: float,
    component_max_height_frac: float,
    component_max_area_frac: float,
    component_max_aspect_ratio: float,
    fill_holes_max_area: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if cv2 is None or int(max_gap) <= 0:
        return masks, np.zeros((int(masks.shape[0]),), dtype=np.uint8)

    imgs = _ensure_image4(images_rgb, "cam0")
    out = np.asarray(masks, dtype=np.uint8).copy()
    if out.ndim != 4 or out.shape[-1] != 1:
        raise ValueError(f"masks must be (T,H,W,1), got {out.shape}")

    min_pixels = max(1, int(min_pixels))
    counts = np.count_nonzero(out[:, :, :, 0] > 0, axis=(1, 2)).astype(np.int64)
    filled = np.zeros((int(out.shape[0]),), dtype=np.uint8)
    gates = np.asarray(rescue_gates).astype(bool)
    if gates.shape != out.shape[:3]:
        gates = np.ones(out.shape[:3], dtype=bool)
    identity_fallback_max_gap = max(0, int(identity_fallback_max_gap))

    for i in range(int(out.shape[0])):
        if int(counts[i]) >= min_pixels:
            continue

        proposals: List[np.ndarray] = []
        for direction in (-1, 1):
            j = _find_nearest_strong_mask(
                counts=counts,
                index=i,
                direction=direction,
                max_gap=max_gap,
                min_pixels=min_pixels,
            )
            if j < 0:
                continue
            gap = abs(int(j) - int(i))
            warped, ok, _ = _align_mask_to_current(
                reference_mask_u8=out[j, :, :, 0],
                reference_rgb=imgs[j],
                current_rgb=imgs[i],
                mode=align_mode,
                max_iters=align_max_iters,
                eps=align_eps,
                min_cc=min_align_cc,
            )
            warped_pixels = int(np.count_nonzero(warped)) if ok else 0
            if (
                (not ok or warped_pixels < min_pixels)
                and identity_fallback_max_gap > 0
                and gap <= identity_fallback_max_gap
            ):
                warped = out[j, :, :, 0].copy()
                ok = True
            if ok and int(np.count_nonzero(warped)) >= min_pixels:
                proposals.append(warped > 0)

        if not proposals:
            continue

        candidate = np.any(np.stack(proposals, axis=0), axis=0)
        gated_candidate = candidate & gates[i]
        if int(np.count_nonzero(gated_candidate)) >= min_pixels:
            candidate = gated_candidate
        mask = _postprocess_binary_mask(
            candidate,
            min_area=min_area,
            max_area=max_area,
            morph_kernel=morph_kernel,
            hole_close_kernel=hole_close_kernel,
            ignore_border_px=ignore_border_px,
            component_context_mask=None,
            component_context_pad=0,
            component_context_min_ratio=0.0,
            component_max_width_frac=component_max_width_frac,
            component_max_height_frac=component_max_height_frac,
            component_max_area_frac=component_max_area_frac,
            component_max_aspect_ratio=component_max_aspect_ratio,
            fill_holes_max_area=fill_holes_max_area,
        )
        if int(np.count_nonzero(mask)) < min_pixels and len(proposals) > 1:
            proposal_masks: List[np.ndarray] = []
            for proposal in proposals:
                proposal_mask = _postprocess_binary_mask(
                    proposal,
                    min_area=min_area,
                    max_area=max_area,
                    morph_kernel=morph_kernel,
                    hole_close_kernel=hole_close_kernel,
                    ignore_border_px=ignore_border_px,
                    component_context_mask=None,
                    component_context_pad=0,
                    component_context_min_ratio=0.0,
                    component_max_width_frac=component_max_width_frac,
                    component_max_height_frac=component_max_height_frac,
                    component_max_area_frac=component_max_area_frac,
                    component_max_aspect_ratio=component_max_aspect_ratio,
                    fill_holes_max_area=fill_holes_max_area,
                )
                if int(np.count_nonzero(proposal_mask)) >= min_pixels:
                    proposal_masks.append(proposal_mask)
            if proposal_masks:
                mask = np.maximum.reduce(proposal_masks).astype(np.uint8)
        if int(np.count_nonzero(mask)) >= min_pixels:
            out[i, :, :, 0] = mask
            counts[i] = int(np.count_nonzero(mask))
            filled[i] = 1

    return out, filled


def _temporal_prune_inconsistent_components(
    images_rgb: np.ndarray,
    masks: np.ndarray,
    max_gap: int,
    align_mode: str,
    align_max_iters: int,
    align_eps: float,
    min_align_cc: float,
    identity_fallback_max_gap: int,
    support_dilate_kernel: int,
    min_overlap_ratio: float,
    min_area: int,
    max_area: int,
    morph_kernel: int,
    hole_close_kernel: int,
    ignore_border_px: int,
    component_max_width_frac: float,
    component_max_height_frac: float,
    component_max_area_frac: float,
    component_max_aspect_ratio: float,
    fill_holes_max_area: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cv2 is None or int(max_gap) <= 0:
        T = int(masks.shape[0])
        return masks, np.zeros((T,), dtype=np.uint16), np.zeros((T,), dtype=np.uint16)

    imgs = _ensure_image4(images_rgb, "cam0")
    src_masks = np.asarray(masks, dtype=np.uint8)
    out = src_masks.copy()
    if out.ndim != 4 or out.shape[-1] != 1:
        raise ValueError(f"masks must be (T,H,W,1), got {out.shape}")

    T = int(out.shape[0])
    counts = np.count_nonzero(src_masks[:, :, :, 0] > 0, axis=(1, 2)).astype(np.int64)
    comps_removed = np.zeros((T,), dtype=np.uint16)
    comps_kept = np.zeros((T,), dtype=np.uint16)
    min_overlap = max(0.0, float(min_overlap_ratio))
    identity_fallback_max_gap = max(0, int(identity_fallback_max_gap))
    support_kernel_size = _odd_kernel_size(support_dilate_kernel)
    support_kernel = (
        np.ones((support_kernel_size, support_kernel_size), dtype=np.uint8)
        if support_kernel_size > 1
        else None
    )

    for i in range(T):
        current = (src_masks[i, :, :, 0] > 0).astype(np.uint8)
        if int(np.count_nonzero(current)) == 0:
            continue

        support = np.zeros_like(current, dtype=np.uint8)
        for direction in (-1, 1):
            j = _find_nearest_strong_mask(
                counts=counts,
                index=i,
                direction=direction,
                max_gap=max_gap,
                min_pixels=max(1, int(min_area)),
            )
            if j < 0:
                continue
            gap = abs(int(j) - int(i))
            warped, ok, _ = _align_mask_to_current(
                reference_mask_u8=src_masks[j, :, :, 0],
                reference_rgb=imgs[j],
                current_rgb=imgs[i],
                mode=align_mode,
                max_iters=align_max_iters,
                eps=align_eps,
                min_cc=min_align_cc,
            )
            warped_pixels = int(np.count_nonzero(warped)) if ok else 0
            if (
                (not ok or warped_pixels == 0)
                and identity_fallback_max_gap > 0
                and gap <= identity_fallback_max_gap
            ):
                warped = src_masks[j, :, :, 0].copy()
                ok = True
            if ok:
                support |= (warped > 0).astype(np.uint8)

        if int(np.count_nonzero(support)) == 0:
            out[i, :, :, 0] = current.astype(np.uint8) * 255
            continue

        if support_kernel is not None:
            support = cv2.dilate(support, support_kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(current, connectivity=8)
        keep = np.zeros_like(current, dtype=bool)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            comp = labels == label
            overlap = float(np.count_nonzero(comp & (support > 0))) / float(area)
            if overlap >= min_overlap:
                keep |= comp
                comps_kept[i] += 1
            else:
                comps_removed[i] += 1

        pruned = _postprocess_binary_mask(
            keep,
            min_area=min_area,
            max_area=max_area,
            morph_kernel=morph_kernel,
            hole_close_kernel=hole_close_kernel,
            ignore_border_px=ignore_border_px,
            component_context_mask=None,
            component_context_pad=0,
            component_context_min_ratio=0.0,
            component_max_width_frac=component_max_width_frac,
            component_max_height_frac=component_max_height_frac,
            component_max_area_frac=component_max_area_frac,
            component_max_aspect_ratio=component_max_aspect_ratio,
            fill_holes_max_area=fill_holes_max_area,
        )
        out[i, :, :, 0] = pruned

    return out, comps_removed, comps_kept


def generate_reference_stain_artifacts(
    current_rgb: np.ndarray,
    current_position: np.ndarray,
    reference_rgb: np.ndarray,
    reference_position: np.ndarray,
    diff_thresh: float,
    dark_thresh: float,
    reflection_v_thresh: float,
    reflection_s_thresh: float,
    min_area: int,
    max_area: int,
    morph_kernel: int,
    hole_close_kernel: int,
    blur_kernel: int,
    ref_align_mode: str,
    ref_align_max_iters: int,
    ref_align_eps: float,
    ref_surface_v_min: float,
    context_v_min: float,
    context_kernel: int,
    dark_prior_enable: bool,
    dark_prior_component_min_diff_ratio: float,
    component_dark_v_max: float,
    component_dark_min_ratio: float,
    output_constraint_kernel: int,
    ignore_border_px: int,
    component_context_pad: int,
    component_context_min_ratio: float,
    component_max_width_frac: float,
    component_max_height_frac: float,
    component_max_area_frac: float,
    component_max_aspect_ratio: float,
    fill_holes_max_area: int,
    pose_pos_scale: float,
    pose_rot_scale: float,
    reference_max_pose_dist: float,
    reference_top_k: int,
    reference_match_mode: str,
    reference_match_window: int,
    reference_diff_percentile: float,
    reference_diff_min_support: float,
    ref_surface_min_support: float,
    local_dark_thresh: float,
    local_dark_ref_delta: float,
    local_dark_ref_percentile: float,
    local_dark_blur_kernel: int,
    adaptive_v_floor_percentile: float,
    adaptive_v_floor_ratio: float,
    adaptive_v_floor_min: float,
    temporal_fill_enable: bool,
    temporal_fill_max_gap: int,
    temporal_fill_min_pixels: int,
    temporal_fill_align_mode: str,
    temporal_fill_min_align_cc: float,
    temporal_fill_identity_fallback_max_gap: int,
    temporal_prune_enable: bool,
    temporal_prune_max_gap: int,
    temporal_prune_align_mode: str,
    temporal_prune_min_align_cc: float,
    temporal_prune_identity_fallback_max_gap: int,
    temporal_prune_support_dilate_kernel: int,
    temporal_prune_min_overlap_ratio: float,
) -> Dict[str, np.ndarray]:
    cur_rgb = _ensure_image4(current_rgb, "cam0")
    ref_rgb = _ensure_image4(reference_rgb, "reference_cam0")
    T, H, W, _ = cur_rgb.shape
    ref_index_bank, ref_distance_bank, ref_center_indices = _reference_index_bank_with_distances(
        current_position=current_position,
        reference_position=reference_position,
        pos_scale=pose_pos_scale,
        rot_scale=pose_rot_scale,
        top_k=reference_top_k,
        match_mode=reference_match_mode,
        match_window=reference_match_window,
    )

    masks = np.zeros((T, H, W, 1), dtype=np.uint8)
    nearest_pose_distances = np.sqrt(np.maximum(ref_distance_bank[:, 0], 0.0)).astype(np.float32)
    pose_rejected = np.zeros((T,), dtype=np.uint8)
    rescue_gates = np.zeros((T, H, W), dtype=bool)
    component_dark_signals = np.ones((T, H, W), dtype=bool)
    use_component_dark_filter = float(component_dark_v_max) > 0.0 and float(component_dark_min_ratio) > 0.0

    for i in range(T):
        cur_frame = _as_uint8_rgb(cur_rgb[i])
        cur_v, cur_s = _rgb_to_value_saturation(cur_frame)
        cur_norm = _lighting_normalized_value(cur_v, blur_kernel)
        local_dark, local_surface = _local_darkness(cur_v, local_dark_blur_kernel)
        context_floor = _adaptive_v_floor(
            local_surface,
            requested_floor=context_v_min,
            percentile=adaptive_v_floor_percentile,
            ratio=adaptive_v_floor_ratio,
            min_floor=adaptive_v_floor_min,
        )
        absolute_dark = (
            cur_v <= float(dark_thresh)
            if float(dark_thresh) > 0.0
            else np.zeros_like(cur_v, dtype=bool)
        )
        local_dark_current = (
            local_dark >= float(local_dark_thresh)
            if float(local_dark_thresh) > 0.0
            else np.zeros_like(cur_v, dtype=bool)
        )
        dark_current = absolute_dark | local_dark_current
        strict_dark_current = (
            cur_v <= float(component_dark_v_max)
            if use_component_dark_filter
            else np.ones_like(cur_v, dtype=bool)
        )
        component_dark_signals[i] = strict_dark_current
        reflection = (cur_v >= float(reflection_v_thresh)) & (cur_s <= float(reflection_s_thresh))
        bright_context = (
            (cur_v >= context_floor) | (local_surface >= context_floor)
            if context_floor > 0.0
            else None
        )
        if cv2 is not None and bright_context is not None:
            k = _odd_kernel_size(context_kernel)
            if k > 1:
                kernel = np.ones((k, k), dtype=np.uint8)
                surface_context = cv2.dilate(bright_context.astype(np.uint8), kernel) > 0
            else:
                surface_context = bright_context
        else:
            surface_context = True
        rescue_gates[i] = dark_current & surface_context & (~reflection)

        max_pose_dist = float(reference_max_pose_dist)
        nearest_pose_dist = float(nearest_pose_distances[i])
        if max_pose_dist > 0.0 and nearest_pose_dist > max_pose_dist:
            pose_rejected[i] = 1
            continue

        diff_candidates: List[np.ndarray] = []
        diff_support_candidates: List[np.ndarray] = []
        ref_surface_candidates: List[np.ndarray] = []
        ref_local_dark_candidates: List[np.ndarray] = []
        for ref_idx in ref_index_bank[i]:
            ref_frame = _resize_rgb_like(ref_rgb[int(ref_idx)], (H, W))
            ref_frame = _align_reference_to_current(
                reference_rgb=ref_frame,
                current_rgb=cur_frame,
                mode=ref_align_mode,
                max_iters=ref_align_max_iters,
                eps=ref_align_eps,
            )
            ref_v, _ = _rgb_to_value_saturation(ref_frame)
            ref_norm = _lighting_normalized_value(ref_v, blur_kernel)
            diff_candidate = (ref_norm - cur_norm).astype(np.float32)
            diff_candidates.append(diff_candidate)
            diff_support_candidates.append(diff_candidate >= float(diff_thresh))
            ref_local_dark, _ = _local_darkness(ref_v, local_dark_blur_kernel)
            ref_local_dark_candidates.append(ref_local_dark)
            if float(ref_surface_v_min) > 0.0:
                ref_floor = _adaptive_v_floor(
                    ref_v,
                    requested_floor=ref_surface_v_min,
                    percentile=adaptive_v_floor_percentile,
                    ratio=adaptive_v_floor_ratio,
                    min_floor=adaptive_v_floor_min,
                )
                ref_surface_candidates.append(ref_v >= ref_floor)

        diff_stack = np.stack(diff_candidates, axis=0)
        if diff_stack.shape[0] == 1:
            diff = diff_stack[0]
        else:
            diff_pct = np.clip(float(reference_diff_percentile), 0.0, 100.0)
            diff = np.percentile(diff_stack, diff_pct, axis=0).astype(np.float32)

        ref_local_dark_stack = np.stack(ref_local_dark_candidates, axis=0)
        if ref_local_dark_stack.shape[0] == 1:
            ref_local_dark = ref_local_dark_stack[0]
        else:
            ref_local_dark_pct = np.clip(float(local_dark_ref_percentile), 0.0, 100.0)
            ref_local_dark = np.percentile(
                ref_local_dark_stack,
                ref_local_dark_pct,
                axis=0,
            ).astype(np.float32)

        if ref_surface_candidates:
            ref_surface = _bool_consensus(ref_surface_candidates, ref_surface_min_support)
        else:
            ref_surface = True

        dark_change = (diff >= float(diff_thresh)) & _bool_consensus(
            diff_support_candidates,
            reference_diff_min_support,
        )
        local_dark_novel = local_dark_current & (
            (local_dark - ref_local_dark) >= float(local_dark_ref_delta)
        )
        dark_prior = bool(dark_prior_enable) & local_dark_novel
        common_gate = dark_current & ref_surface & surface_context & (~reflection)
        core_candidate = dark_change & common_gate
        mask = _postprocess_binary_mask(
            core_candidate,
            min_area=min_area,
            max_area=max_area,
            morph_kernel=morph_kernel,
            hole_close_kernel=hole_close_kernel,
            ignore_border_px=ignore_border_px,
            component_context_mask=bright_context,
            component_context_pad=component_context_pad,
            component_context_min_ratio=component_context_min_ratio,
            component_max_width_frac=component_max_width_frac,
            component_max_height_frac=component_max_height_frac,
            component_max_area_frac=component_max_area_frac,
            component_max_aspect_ratio=component_max_aspect_ratio,
            fill_holes_max_area=fill_holes_max_area,
        )
        mask = _constrain_to_near_mask(mask, core_candidate, output_constraint_kernel)
        if bool(dark_prior_enable):
            prior_candidate = dark_prior & common_gate
            prior_candidate &= _filter_components_by_support_ratio(
                core_candidate | prior_candidate,
                core_candidate,
                dark_prior_component_min_diff_ratio,
            )
            prior_mask = _postprocess_binary_mask(
                prior_candidate,
                min_area=min_area,
                max_area=max_area,
                morph_kernel=morph_kernel,
                hole_close_kernel=hole_close_kernel,
                ignore_border_px=ignore_border_px,
                component_context_mask=bright_context,
                component_context_pad=component_context_pad,
                component_context_min_ratio=component_context_min_ratio,
                component_max_width_frac=component_max_width_frac,
                component_max_height_frac=component_max_height_frac,
                component_max_area_frac=component_max_area_frac,
                component_max_aspect_ratio=component_max_aspect_ratio,
                fill_holes_max_area=fill_holes_max_area,
            )
            prior_mask = _constrain_to_near_mask(prior_mask, prior_candidate, output_constraint_kernel)
            mask = np.maximum(mask, prior_mask)
        if use_component_dark_filter:
            mask = _filter_mask_components_by_signal_ratio(
                mask,
                strict_dark_current,
                component_dark_min_ratio,
            )
        masks[i, :, :, 0] = mask

    temporal_filled = np.zeros((T,), dtype=np.uint8)
    if bool(temporal_fill_enable):
        min_pixels = int(temporal_fill_min_pixels)
        if min_pixels <= 0:
            min_pixels = max(int(min_area) * 2, 30)
        masks, temporal_filled = _temporal_fill_stain_mask_gaps(
            images_rgb=cur_rgb,
            masks=masks,
            rescue_gates=rescue_gates,
            min_pixels=min_pixels,
            max_gap=int(temporal_fill_max_gap),
            align_mode=str(temporal_fill_align_mode),
            align_max_iters=ref_align_max_iters,
            align_eps=ref_align_eps,
            min_align_cc=float(temporal_fill_min_align_cc),
            identity_fallback_max_gap=int(temporal_fill_identity_fallback_max_gap),
            min_area=min_area,
            max_area=max_area,
            morph_kernel=morph_kernel,
            hole_close_kernel=hole_close_kernel,
            ignore_border_px=ignore_border_px,
            component_max_width_frac=component_max_width_frac,
            component_max_height_frac=component_max_height_frac,
            component_max_area_frac=component_max_area_frac,
            component_max_aspect_ratio=component_max_aspect_ratio,
            fill_holes_max_area=fill_holes_max_area,
        )

    temporal_pruned_removed = np.zeros((T,), dtype=np.uint16)
    temporal_pruned_kept = np.zeros((T,), dtype=np.uint16)
    if bool(temporal_prune_enable):
        masks, temporal_pruned_removed, temporal_pruned_kept = _temporal_prune_inconsistent_components(
            images_rgb=cur_rgb,
            masks=masks,
            max_gap=int(temporal_prune_max_gap),
            align_mode=str(temporal_prune_align_mode),
            align_max_iters=ref_align_max_iters,
            align_eps=ref_align_eps,
            min_align_cc=float(temporal_prune_min_align_cc),
            identity_fallback_max_gap=int(temporal_prune_identity_fallback_max_gap),
            support_dilate_kernel=int(temporal_prune_support_dilate_kernel),
            min_overlap_ratio=float(temporal_prune_min_overlap_ratio),
            min_area=min_area,
            max_area=max_area,
            morph_kernel=morph_kernel,
            hole_close_kernel=hole_close_kernel,
            ignore_border_px=ignore_border_px,
            component_max_width_frac=component_max_width_frac,
            component_max_height_frac=component_max_height_frac,
            component_max_area_frac=component_max_area_frac,
            component_max_aspect_ratio=component_max_aspect_ratio,
            fill_holes_max_area=fill_holes_max_area,
        )

    if use_component_dark_filter:
        for i in range(T):
            masks[i, :, :, 0] = _filter_mask_components_by_signal_ratio(
                masks[i, :, :, 0],
                component_dark_signals[i],
                component_dark_min_ratio,
            )

    return {
        "stain_mask": masks,
        "_stain_reference_pose_dist": nearest_pose_distances,
        "_stain_reference_pose_rejected": pose_rejected,
        "_stain_reference_center_indices": ref_center_indices,
        "_stain_temporal_filled": temporal_filled,
        "_stain_temporal_pruned_removed": temporal_pruned_removed,
        "_stain_temporal_pruned_kept": temporal_pruned_kept,
    }


def _trim_to_min_len(items: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], int]:
    lengths = [int(v.shape[0]) for v in items.values()]
    if not lengths:
        raise ValueError("No arrays to trim.")
    T = min(lengths)
    return {k: v[:T] for k, v in items.items()}, T


def load_episode(g_ep: h5py.Group, camera_names: Sequence[str]) -> Dict[str, np.ndarray]:
    position, _ = _read_optional_array(
        g_ep,
        ["position", "observations/position"],
        dtype=np.float32,
    )
    if position is None:
        raise KeyError(f"Missing position in {g_ep.name}")

    force, _ = _read_optional_array(
        g_ep,
        ["ft", "force", "observations/force"],
        dtype=np.float32,
    )
    if force is None:
        raise KeyError(f"Missing force/ft in {g_ep.name}")

    data: Dict[str, np.ndarray] = {
        "position": _ensure_2d_min_dim(position, 6, "position").astype(np.float32),
        "force": _ensure_2d_min_dim(force, 3, "force").astype(np.float32),
    }

    for cam in camera_names:
        aliases = [
            f"images/{cam}",
            f"observations/images/{cam}",
        ]
        if cam == "cam0":
            aliases.extend(["image", "images/rgb", "observations/image"])
        if cam == "cam1":
            aliases.extend(["images/global", "observations/images/global"])

        image, _ = _read_optional_array(g_ep, aliases, dtype=None)
        if image is None:
            raise KeyError(f"Missing {cam} image in {g_ep.name}")
        data[cam] = _ensure_image4(image, cam)

    stain_mask, _ = _read_optional_array(
        g_ep,
        [
            "images/stain_mask",
            "observations/images/stain_mask",
            "stain_mask",
        ],
        dtype=None,
    )
    if stain_mask is not None:
        data["stain_mask"] = _ensure_stain_mask4(stain_mask)

    data, _ = _trim_to_min_len(data)
    return data


def truncate_episode(data: Dict[str, np.ndarray], max_len: int) -> Tuple[Dict[str, np.ndarray], bool, int]:
    T = int(data["position"].shape[0])
    if int(max_len) <= 0 or T <= int(max_len):
        return data, False, T
    return {k: v[: int(max_len)] for k, v in data.items()}, True, int(max_len)


def _compression_kwargs(mode: str, gzip_level: int) -> Dict:
    mode = str(mode).lower()
    if mode == "gzip":
        return {"compression": "gzip", "compression_opts": int(gzip_level), "shuffle": True}
    if mode == "lzf":
        return {"compression": "lzf", "shuffle": True}
    return {}


def _ordered_image_dataset_names(camera_names: Sequence[str], has_stain_mask: bool) -> List[str]:
    order: List[str] = []
    for cam in camera_names:
        order.append(str(cam))
        if has_stain_mask and str(cam) == "cam0":
            order.append("stain_mask")
    if has_stain_mask and "stain_mask" not in order:
        order.append("stain_mask")
    return order


def write_episode(
    out_path: Path,
    data: Dict[str, np.ndarray],
    camera_names: Sequence[str],
    source_h5: Path,
    source_episode: str,
    compression: str,
    gzip_level: int,
    orig_len: int,
    truncated: bool,
) -> None:
    if out_path.exists():
        out_path.unlink()

    kwargs = _compression_kwargs(compression, gzip_level)
    with h5py.File(str(out_path), "w") as f:
        f.attrs["schema_version"] = "imitation_form_compact_v1"
        f.attrs["source_h5"] = str(source_h5)
        f.attrs["source_episode"] = str(source_episode)
        f.attrs["camera_names_json"] = json.dumps(list(camera_names))
        f.attrs["orig_len"] = int(orig_len)
        f.attrs["truncated"] = int(bool(truncated))

        g_action = f.create_group("action")
        g_action.create_dataset("position", data=data["position"].astype(np.float32), **kwargs)
        g_action.create_dataset("force", data=data["force"].astype(np.float32), **kwargs)

        g_obs = f.create_group("observations", track_order=True)
        g_obs.create_dataset("position", data=data["position"].astype(np.float32), **kwargs)
        g_obs.create_dataset("force", data=data["force"].astype(np.float32), **kwargs)

        has_stain_mask = "stain_mask" in data
        g_images = g_obs.create_group("images", track_order=True)
        for name in _ordered_image_dataset_names(camera_names, has_stain_mask):
            if name == "stain_mask":
                ds = g_images.create_dataset(
                    "stain_mask",
                    data=data["stain_mask"].astype(np.uint8),
                    **kwargs,
                )
                ds.attrs["shape_convention"] = "T,H,W,1"
                ds.attrs["storage"] = "uint8_0_255"
                ds.attrs["model_value_range_after_div255"] = "float32_0_1"
                f.attrs["has_stain_mask"] = 1
            else:
                g_images.create_dataset(name, data=data[name].astype(np.uint8), **kwargs)


def _attr_to_bool(value, default: bool = False) -> bool:
    try:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(int(value))
    except Exception:
        return bool(default)


def _resolve_stain_mask_mode(f: h5py.File, requested_mode: str) -> str:
    mode = str(requested_mode or "auto").strip().lower()
    if mode not in ("auto", "copy", "reference_episode", "none"):
        raise ValueError(f"Unsupported stain_mask_mode={requested_mode}")
    if mode != "auto":
        return mode

    source = str(f.attrs.get("stain_mask_source", "")).strip().lower()
    reference_marked = _attr_to_bool(f.attrs.get("stain_reference_first_episode", 0))
    if source == "reference_episode" or reference_marked:
        return "reference_episode"
    return "copy"


def _resolve_reference_episode_name(ep_names: Sequence[str], requested: str) -> str:
    if not ep_names:
        raise RuntimeError("No episode names available")
    raw = str(requested or "ep_0000").strip()
    if raw in ("", "first", "episode_0", "0"):
        return ep_names[0]
    if raw in ep_names:
        return raw
    if raw.startswith("episode_"):
        suffix = raw.split("episode_", 1)[1]
        if suffix.isdigit():
            candidate = f"ep_{int(suffix):04d}"
            if candidate in ep_names:
                return candidate
    raise KeyError(f"Reference episode {requested!r} not found. Available: {list(ep_names)[:8]}...")


def convert_merged_h5(
    input_h5: Path,
    output_dir: Path,
    camera_names: Sequence[str],
    min_len: int,
    max_len: int,
    compression: str,
    gzip_level: int,
    overwrite: bool,
    write_summary: bool,
    stain_mask_mode: str = "auto",
    stain_reference_episode: str = "ep_0000",
    stain_exclude_reference_episode: bool = True,
    stain_diff_thresh: float = 18.0,
    stain_dark_thresh: float = 165.0,
    reflection_v_thresh: float = 235.0,
    reflection_s_thresh: float = 60.0,
    stain_min_area: int = 15,
    stain_max_area: int = 0,
    stain_morph_kernel: int = 3,
    stain_hole_close_kernel: int = 7,
    stain_ref_blur_kernel: int = 31,
    stain_ref_align_mode: str = "homography",
    stain_ref_align_max_iters: int = 40,
    stain_ref_align_eps: float = 1e-4,
    stain_ref_surface_v_min: float = 120.0,
    stain_context_v_min: float = 120.0,
    stain_context_kernel: int = 31,
    stain_dark_prior_enable: bool = True,
    stain_dark_prior_component_min_diff_ratio: float = 0.35,
    stain_component_dark_v_max: float = 120.0,
    stain_component_dark_min_ratio: float = 0.45,
    stain_output_constraint_kernel: int = 11,
    stain_ignore_border_px: int = 2,
    stain_component_context_pad: int = 12,
    stain_component_context_min_ratio: float = 0.08,
    stain_component_max_width_frac: float = 0.75,
    stain_component_max_height_frac: float = 0.75,
    stain_component_max_area_frac: float = 0.12,
    stain_component_max_aspect_ratio: float = 8.0,
    stain_fill_holes_max_area: int = 400,
    stain_ref_pose_pos_scale: float = 50.0,
    stain_ref_pose_rot_scale: float = 0.35,
    stain_reference_max_pose_dist: float = 1.0,
    stain_reference_top_k: int = 3,
    stain_reference_match_mode: str = "monotonic",
    stain_reference_match_window: int = 8,
    stain_reference_diff_percentile: float = 50.0,
    stain_reference_diff_min_support: float = 0.5,
    stain_ref_surface_min_support: float = 0.5,
    stain_local_dark_thresh: float = 14.0,
    stain_local_dark_ref_delta: float = 10.0,
    stain_local_dark_ref_percentile: float = 75.0,
    stain_local_dark_blur_kernel: int = 41,
    stain_adaptive_v_floor_percentile: float = 75.0,
    stain_adaptive_v_floor_ratio: float = 0.85,
    stain_adaptive_v_floor_min: float = 60.0,
    stain_temporal_fill_enable: bool = True,
    stain_temporal_fill_max_gap: int = 24,
    stain_temporal_fill_min_pixels: int = 0,
    stain_temporal_fill_align_mode: str = "homography",
    stain_temporal_fill_min_align_cc: float = 0.45,
    stain_temporal_fill_identity_fallback_max_gap: int = 3,
    stain_temporal_prune_enable: bool = True,
    stain_temporal_prune_max_gap: int = 12,
    stain_temporal_prune_align_mode: str = "homography",
    stain_temporal_prune_min_align_cc: float = 0.45,
    stain_temporal_prune_identity_fallback_max_gap: int = 3,
    stain_temporal_prune_support_dilate_kernel: int = 15,
    stain_temporal_prune_min_overlap_ratio: float = 0.12,
) -> List[Path]:
    camera_names = [str(c).strip() for c in camera_names if str(c).strip()]
    if not camera_names:
        raise ValueError("camera_names must not be empty.")

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif list(output_dir.glob("episode_*.hdf5")):
            raise RuntimeError(f"Output dir already contains episode files: {output_dir}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    failed: List[Tuple[str, str]] = []

    with h5py.File(str(input_h5), "r") as f:
        if "episodes" not in f:
            raise KeyError(f"{input_h5} does not contain /episodes group")

        ep_names = sorted(name for name in f["episodes"].keys() if not str(name).endswith("__writing"))
        if not ep_names:
            raise RuntimeError(f"No episodes found under {input_h5}/episodes")

        effective_stain_mode = _resolve_stain_mask_mode(f, stain_mask_mode)
        reference_ep_name = ""
        reference_data = None
        if effective_stain_mode == "reference_episode":
            reference_ep_name = _resolve_reference_episode_name(ep_names, stain_reference_episode)
            reference_data = load_episode(f["episodes"][reference_ep_name], camera_names)

        print(f"[INFO] input_h5       = {input_h5}")
        print(f"[INFO] output_dir     = {output_dir}")
        print(f"[INFO] camera_names   = {camera_names}")
        print(f"[INFO] episodes found = {len(ep_names)}")
        print(f"[INFO] stain_mode     = {effective_stain_mode}")
        if reference_ep_name:
            print(f"[INFO] stain_ref_ep   = {reference_ep_name} (exclude={int(bool(stain_exclude_reference_episode))})")

        out_idx = 0
        for ep_name in ep_names:
            try:
                if (
                    effective_stain_mode == "reference_episode"
                    and bool(stain_exclude_reference_episode)
                    and ep_name == reference_ep_name
                ):
                    print(f"[SKIP] {ep_name}: clean stain reference episode")
                    continue

                data = load_episode(f["episodes"][ep_name], camera_names)
                if effective_stain_mode == "none" and "stain_mask" in data:
                    del data["stain_mask"]
                elif effective_stain_mode == "reference_episode":
                    if reference_data is None:
                        raise RuntimeError("reference_data is not loaded")
                    artifacts = generate_reference_stain_artifacts(
                        current_rgb=data["cam0"],
                        current_position=data["position"],
                        reference_rgb=reference_data["cam0"],
                        reference_position=reference_data["position"],
                        diff_thresh=stain_diff_thresh,
                        dark_thresh=stain_dark_thresh,
                        reflection_v_thresh=reflection_v_thresh,
                        reflection_s_thresh=reflection_s_thresh,
                        min_area=stain_min_area,
                        max_area=stain_max_area,
                        morph_kernel=stain_morph_kernel,
                        hole_close_kernel=stain_hole_close_kernel,
                        blur_kernel=stain_ref_blur_kernel,
                        ref_align_mode=stain_ref_align_mode,
                        ref_align_max_iters=stain_ref_align_max_iters,
                        ref_align_eps=stain_ref_align_eps,
                        ref_surface_v_min=stain_ref_surface_v_min,
                        context_v_min=stain_context_v_min,
                        context_kernel=stain_context_kernel,
                        dark_prior_enable=stain_dark_prior_enable,
                        dark_prior_component_min_diff_ratio=stain_dark_prior_component_min_diff_ratio,
                        component_dark_v_max=stain_component_dark_v_max,
                        component_dark_min_ratio=stain_component_dark_min_ratio,
                        output_constraint_kernel=stain_output_constraint_kernel,
                        ignore_border_px=stain_ignore_border_px,
                        component_context_pad=stain_component_context_pad,
                        component_context_min_ratio=stain_component_context_min_ratio,
                        component_max_width_frac=stain_component_max_width_frac,
                        component_max_height_frac=stain_component_max_height_frac,
                        component_max_area_frac=stain_component_max_area_frac,
                        component_max_aspect_ratio=stain_component_max_aspect_ratio,
                        fill_holes_max_area=stain_fill_holes_max_area,
                        pose_pos_scale=stain_ref_pose_pos_scale,
                        pose_rot_scale=stain_ref_pose_rot_scale,
                        reference_max_pose_dist=stain_reference_max_pose_dist,
                        reference_top_k=stain_reference_top_k,
                        reference_match_mode=stain_reference_match_mode,
                        reference_match_window=stain_reference_match_window,
                        reference_diff_percentile=stain_reference_diff_percentile,
                        reference_diff_min_support=stain_reference_diff_min_support,
                        ref_surface_min_support=stain_ref_surface_min_support,
                        local_dark_thresh=stain_local_dark_thresh,
                        local_dark_ref_delta=stain_local_dark_ref_delta,
                        local_dark_ref_percentile=stain_local_dark_ref_percentile,
                        local_dark_blur_kernel=stain_local_dark_blur_kernel,
                        adaptive_v_floor_percentile=stain_adaptive_v_floor_percentile,
                        adaptive_v_floor_ratio=stain_adaptive_v_floor_ratio,
                        adaptive_v_floor_min=stain_adaptive_v_floor_min,
                        temporal_fill_enable=stain_temporal_fill_enable,
                        temporal_fill_max_gap=stain_temporal_fill_max_gap,
                        temporal_fill_min_pixels=stain_temporal_fill_min_pixels,
                        temporal_fill_align_mode=stain_temporal_fill_align_mode,
                        temporal_fill_min_align_cc=stain_temporal_fill_min_align_cc,
                        temporal_fill_identity_fallback_max_gap=stain_temporal_fill_identity_fallback_max_gap,
                        temporal_prune_enable=stain_temporal_prune_enable,
                        temporal_prune_max_gap=stain_temporal_prune_max_gap,
                        temporal_prune_align_mode=stain_temporal_prune_align_mode,
                        temporal_prune_min_align_cc=stain_temporal_prune_min_align_cc,
                        temporal_prune_identity_fallback_max_gap=stain_temporal_prune_identity_fallback_max_gap,
                        temporal_prune_support_dilate_kernel=stain_temporal_prune_support_dilate_kernel,
                        temporal_prune_min_overlap_ratio=stain_temporal_prune_min_overlap_ratio,
                    )
                    pose_rejected = artifacts.pop("_stain_reference_pose_rejected", None)
                    pose_dist = artifacts.pop("_stain_reference_pose_dist", None)
                    artifacts.pop("_stain_reference_center_indices", None)
                    temporal_filled = artifacts.pop("_stain_temporal_filled", None)
                    temporal_pruned_removed = artifacts.pop("_stain_temporal_pruned_removed", None)
                    temporal_pruned_kept = artifacts.pop("_stain_temporal_pruned_kept", None)
                    if pose_rejected is not None:
                        rejected_count = int(np.count_nonzero(np.asarray(pose_rejected)))
                        if rejected_count > 0:
                            max_dist = float(np.max(np.asarray(pose_dist))) if pose_dist is not None else 0.0
                            print(
                                f"[WARN] {ep_name}: {rejected_count}/{len(pose_rejected)} frames had no close "
                                f"clean reference (max_scaled_pose_dist={max_dist:.3f}, "
                                f"limit={float(stain_reference_max_pose_dist):.3f}); wrote empty masks for them."
                            )
                    if temporal_filled is not None:
                        filled_count = int(np.count_nonzero(np.asarray(temporal_filled)))
                        if filled_count > 0:
                            print(
                                f"[INFO] {ep_name}: temporal-filled {filled_count}/{len(temporal_filled)} "
                                "weak stain-mask frames from neighboring frames."
                            )
                    if temporal_pruned_removed is not None:
                        removed_count = int(np.sum(np.asarray(temporal_pruned_removed, dtype=np.int64)))
                        kept_count = int(np.sum(np.asarray(temporal_pruned_kept, dtype=np.int64))) if temporal_pruned_kept is not None else 0
                        if removed_count > 0:
                            print(
                                f"[INFO] {ep_name}: temporal-pruned {removed_count} inconsistent "
                                f"mask components (kept={kept_count})."
                            )
                    data.update(artifacts)

                orig_len = int(data["position"].shape[0])
                if orig_len < int(min_len):
                    raise RuntimeError(f"too short: T={orig_len} < min_len={min_len}")

                data, truncated, T_out = truncate_episode(data, max_len=max_len)
                out_path = output_dir / f"episode_{out_idx}.hdf5"
                write_episode(
                    out_path=out_path,
                    data=data,
                    camera_names=camera_names,
                    source_h5=input_h5,
                    source_episode=ep_name,
                    compression=compression,
                    gzip_level=gzip_level,
                    orig_len=orig_len,
                    truncated=truncated,
                )

                image_items = [f"{cam}={data[cam].shape}" for cam in camera_names]
                if "stain_mask" in data:
                    image_items.append(f"stain_mask={data['stain_mask'].shape}")
                image_shapes = ", ".join(image_items)
                print(
                    f"[OK] {ep_name} -> episode_{out_idx}.hdf5 | "
                    f"T={T_out}, position={data['position'].shape}, force={data['force'].shape}, {image_shapes}"
                )
                written.append(out_path)
                out_idx += 1

            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                failed.append((ep_name, msg))
                print(f"[FAIL] {ep_name}: {msg}")

    if write_summary:
        summary_path = output_dir / "conversion_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "input_h5": str(input_h5),
                    "output_dir": str(output_dir),
                    "camera_names": list(camera_names),
                    "num_written": len(written),
                    "written": [str(p) for p in written],
                    "num_failed": len(failed),
                    "failed": [{"episode": ep, "error": err} for ep, err in failed],
                    "schema_version": "imitation_form_compact_v1",
                    "stain_mask_mode": effective_stain_mode,
                    "stain_reference_episode": reference_ep_name,
                    "stain_reference_max_pose_dist": float(stain_reference_max_pose_dist),
                    "stain_reference_top_k": int(stain_reference_top_k),
                    "stain_reference_match_mode": str(stain_reference_match_mode),
                    "stain_reference_match_window": int(stain_reference_match_window),
                    "stain_reference_diff_percentile": float(stain_reference_diff_percentile),
                    "stain_reference_diff_min_support": float(stain_reference_diff_min_support),
                    "stain_ref_surface_min_support": float(stain_ref_surface_min_support),
                    "stain_local_dark_thresh": float(stain_local_dark_thresh),
                    "stain_local_dark_ref_delta": float(stain_local_dark_ref_delta),
                    "stain_local_dark_ref_percentile": float(stain_local_dark_ref_percentile),
                    "stain_ref_align_mode": str(stain_ref_align_mode),
                    "stain_dark_prior_enable": bool(stain_dark_prior_enable),
                    "stain_dark_prior_component_min_diff_ratio": float(stain_dark_prior_component_min_diff_ratio),
                    "stain_component_dark_v_max": float(stain_component_dark_v_max),
                    "stain_component_dark_min_ratio": float(stain_component_dark_min_ratio),
                    "stain_component_max_width_frac": float(stain_component_max_width_frac),
                    "stain_component_max_height_frac": float(stain_component_max_height_frac),
                    "stain_component_max_aspect_ratio": float(stain_component_max_aspect_ratio),
                    "stain_temporal_fill_enable": bool(stain_temporal_fill_enable),
                    "stain_temporal_fill_max_gap": int(stain_temporal_fill_max_gap),
                    "stain_temporal_fill_min_pixels": int(stain_temporal_fill_min_pixels),
                    "stain_temporal_fill_align_mode": str(stain_temporal_fill_align_mode),
                    "stain_temporal_fill_min_align_cc": float(stain_temporal_fill_min_align_cc),
                    "stain_temporal_fill_identity_fallback_max_gap": int(
                        stain_temporal_fill_identity_fallback_max_gap
                    ),
                    "stain_temporal_prune_enable": bool(stain_temporal_prune_enable),
                    "stain_temporal_prune_max_gap": int(stain_temporal_prune_max_gap),
                    "stain_temporal_prune_align_mode": str(stain_temporal_prune_align_mode),
                    "stain_temporal_prune_min_align_cc": float(stain_temporal_prune_min_align_cc),
                    "stain_temporal_prune_identity_fallback_max_gap": int(
                        stain_temporal_prune_identity_fallback_max_gap
                    ),
                    "stain_temporal_prune_support_dilate_kernel": int(stain_temporal_prune_support_dilate_kernel),
                    "stain_temporal_prune_min_overlap_ratio": float(stain_temporal_prune_min_overlap_ratio),
                },
                fp,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[INFO] wrote summary: {summary_path}")

    print(f"[DONE] converted episodes: {len(written)} / {len(written) + len(failed)}")
    if not written:
        raise RuntimeError("No episodes were converted successfully.")
    return written


def build_parser(description: str, default_root: Path, camera_names: Sequence[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input_h5",
        "--input",
        type=str,
        default="",
        help="Merged HDF5 file, merged_hdf5 directory, or run directory. If omitted, latest is selected.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(default_root),
        help="Dataset root used for auto-latest search.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory. If omitted, use <run_dir>/imitation_form.",
    )
    parser.add_argument("--min_len", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=0, help="0 means no truncation.")
    parser.add_argument("--compression", type=str, default="gzip", choices=["gzip", "lzf", "none"])
    parser.add_argument("--gzip_level", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="Replace output_dir if it contains old episodes.")
    parser.add_argument("--write_summary", action="store_true", help="Write conversion_summary.json into output_dir.")
    parser.add_argument(
        "--stain_mask_mode",
        type=str,
        default="auto",
        choices=["auto", "copy", "reference_episode", "none"],
        help=(
            "auto uses recorder metadata; reference_episode uses a clean episode as a pose-indexed "
            "reference bank and regenerates observations/images/stain_mask."
        ),
    )
    parser.add_argument("--stain_reference_episode", type=str, default="ep_0000")
    parser.add_argument(
        "--keep_stain_reference_episode",
        action="store_true",
        help="Do not skip the clean reference episode when stain_mask_mode=reference_episode.",
    )
    parser.add_argument("--stain_diff_thresh", type=float, default=18.0)
    parser.add_argument("--stain_dark_thresh", type=float, default=165.0)
    parser.add_argument("--reflection_v_thresh", type=float, default=235.0)
    parser.add_argument("--reflection_s_thresh", type=float, default=60.0)
    parser.add_argument("--stain_min_area", type=int, default=15)
    parser.add_argument("--stain_max_area", type=int, default=0, help="0 disables max-area filtering.")
    parser.add_argument("--stain_morph_kernel", type=int, default=3)
    parser.add_argument(
        "--stain_hole_close_kernel",
        type=int,
        default=7,
        help="Extra close kernel for filling small holes inside stain blobs. 0 disables.",
    )
    parser.add_argument("--stain_ref_blur_kernel", type=int, default=31)
    parser.add_argument(
        "--stain_ref_align_mode",
        type=str,
        default="homography",
        choices=["none", "translation", "euclidean", "affine", "homography"],
        help="Align the selected clean reference frame to the current frame before differencing.",
    )
    parser.add_argument("--stain_ref_align_max_iters", type=int, default=40)
    parser.add_argument("--stain_ref_align_eps", type=float, default=1e-4)
    parser.add_argument(
        "--stain_ref_surface_v_min",
        type=float,
        default=120.0,
        help="Require the clean reference pixel to be this bright; suppresses dark background/tool regions.",
    )
    parser.add_argument(
        "--stain_context_v_min",
        type=float,
        default=120.0,
        help="Require bright current-frame surface context near each stain candidate. 0 disables this gate.",
    )
    parser.add_argument("--stain_context_kernel", type=int, default=31)
    parser.add_argument(
        "--stain_dark_prior_enable",
        action="store_true",
        default=True,
        help="Allow local-dark evidence even when reference-difference consensus is weak.",
    )
    parser.add_argument(
        "--no_stain_dark_prior",
        dest="stain_dark_prior_enable",
        action="store_false",
        help="Require reference-difference consensus for every stain pixel.",
    )
    parser.add_argument(
        "--stain_dark_prior_component_min_diff_ratio",
        type=float,
        default=0.35,
        help=(
            "When dark-prior rescue is enabled, require each raw component to contain at least this "
            "fraction of reference-diff-supported pixels. 0 restores the old loose behavior."
        ),
    )
    parser.add_argument(
        "--stain_component_dark_v_max",
        type=float,
        default=120.0,
        help=(
            "Current-frame V threshold for strict dark evidence at component level. "
            "0 disables strict-dark component filtering."
        ),
    )
    parser.add_argument(
        "--stain_component_dark_min_ratio",
        type=float,
        default=0.45,
        help=(
            "Keep each mask component only if at least this fraction of its pixels have "
            "V <= stain_component_dark_v_max. 0 disables."
        ),
    )
    parser.add_argument(
        "--stain_output_constraint_kernel",
        type=int,
        default=11,
        help="Constrain the final postprocessed mask to this neighborhood around raw current-frame candidates.",
    )
    parser.add_argument(
        "--stain_ignore_border_px",
        type=int,
        default=2,
        help="Drop stain components touching the image border within this many pixels. 0 disables.",
    )
    parser.add_argument("--stain_component_context_pad", type=int, default=12)
    parser.add_argument(
        "--stain_component_context_min_ratio",
        type=float,
        default=0.08,
        help="Require this ratio of bright surface pixels around each component. 0 disables.",
    )
    parser.add_argument("--stain_component_max_width_frac", type=float, default=0.75)
    parser.add_argument("--stain_component_max_height_frac", type=float, default=0.75)
    parser.add_argument(
        "--stain_component_max_area_frac",
        type=float,
        default=0.12,
        help="Drop components covering more than this image fraction. 0 disables.",
    )
    parser.add_argument(
        "--stain_component_max_aspect_ratio",
        type=float,
        default=8.0,
        help="Drop very elongated components such as table edges. 0 disables.",
    )
    parser.add_argument(
        "--stain_fill_holes_max_area",
        type=int,
        default=400,
        help="Fill enclosed holes in stain components up to this area. -1 fills all holes, 0 disables.",
    )
    parser.add_argument("--stain_ref_pose_pos_scale", type=float, default=50.0)
    parser.add_argument("--stain_ref_pose_rot_scale", type=float, default=0.35)
    parser.add_argument(
        "--stain_reference_max_pose_dist",
        type=float,
        default=1.0,
        help=(
            "Maximum scaled pose distance to a clean reference frame. Frames farther than this get "
            "an empty mask. 0 disables."
        ),
    )
    parser.add_argument(
        "--stain_reference_top_k",
        type=int,
        default=3,
        help="Use this many nearest clean reference frames and fuse their dark-change evidence.",
    )
    parser.add_argument(
        "--stain_reference_match_mode",
        type=str,
        default="monotonic",
        choices=["nearest", "monotonic"],
        help=(
            "nearest matches each frame independently by pose; monotonic uses DTW over the full pose "
            "sequence, then samples references near the matched frame."
        ),
    )
    parser.add_argument(
        "--stain_reference_match_window",
        type=int,
        default=8,
        help="When match_mode=monotonic, restrict top-k reference candidates to +/- this many reference frames.",
    )
    parser.add_argument(
        "--stain_reference_diff_percentile",
        type=float,
        default=50.0,
        help="Percentile used to fuse top-k reference difference maps. 50=median, 100=max.",
    )
    parser.add_argument(
        "--stain_reference_diff_min_support",
        type=float,
        default=0.5,
        help="Minimum fraction of top-k references that must agree on dark-change evidence.",
    )
    parser.add_argument(
        "--stain_ref_surface_min_support",
        type=float,
        default=0.5,
        help="Minimum fraction of top-k references that must classify the pixel as clean bright surface.",
    )
    parser.add_argument(
        "--stain_local_dark_thresh",
        type=float,
        default=14.0,
        help="Local contrast threshold for dark stains, independent of absolute image brightness.",
    )
    parser.add_argument(
        "--stain_local_dark_ref_delta",
        type=float,
        default=10.0,
        help="Require current local darkness to exceed clean-reference local darkness by this margin.",
    )
    parser.add_argument(
        "--stain_local_dark_ref_percentile",
        type=float,
        default=75.0,
        help="Percentile used to fuse reference local-dark maps for fixed-structure suppression.",
    )
    parser.add_argument("--stain_local_dark_blur_kernel", type=int, default=41)
    parser.add_argument("--stain_adaptive_v_floor_percentile", type=float, default=75.0)
    parser.add_argument("--stain_adaptive_v_floor_ratio", type=float, default=0.85)
    parser.add_argument("--stain_adaptive_v_floor_min", type=float, default=60.0)
    parser.add_argument(
        "--stain_temporal_fill_enable",
        action="store_true",
        default=True,
        help="Fill weak/empty stain-mask frames by warping nearby confident masks.",
    )
    parser.add_argument(
        "--no_stain_temporal_fill",
        dest="stain_temporal_fill_enable",
        action="store_false",
        help="Disable temporal stain-mask gap filling.",
    )
    parser.add_argument(
        "--stain_temporal_fill_max_gap",
        type=int,
        default=24,
        help="Search this many frames forward/backward for confident masks when filling a weak frame.",
    )
    parser.add_argument(
        "--stain_temporal_fill_min_pixels",
        type=int,
        default=0,
        help="Frames below this mask-pixel count are fill candidates. 0 uses max(2*min_area, 30).",
    )
    parser.add_argument(
        "--stain_temporal_fill_align_mode",
        type=str,
        default="homography",
        choices=["translation", "euclidean", "affine", "homography"],
        help="Image alignment model used when warping neighboring masks for temporal fill.",
    )
    parser.add_argument(
        "--stain_temporal_fill_min_align_cc",
        type=float,
        default=0.45,
        help="Minimum ECC alignment score required to accept a warped neighboring mask.",
    )
    parser.add_argument(
        "--stain_temporal_fill_identity_fallback_max_gap",
        type=int,
        default=3,
        help=(
            "If temporal alignment fails, directly reuse a neighboring mask for weak frames within "
            "this many frames. 0 disables."
        ),
    )
    parser.add_argument(
        "--stain_temporal_prune_enable",
        action="store_true",
        default=True,
        help="Remove mask components that are not supported by nearby aligned frames.",
    )
    parser.add_argument(
        "--no_stain_temporal_prune",
        dest="stain_temporal_prune_enable",
        action="store_false",
        help="Disable temporal consistency pruning.",
    )
    parser.add_argument(
        "--stain_temporal_prune_max_gap",
        type=int,
        default=12,
        help="Search this many frames forward/backward for support when pruning components.",
    )
    parser.add_argument(
        "--stain_temporal_prune_align_mode",
        type=str,
        default="homography",
        choices=["translation", "euclidean", "affine", "homography"],
        help="Image alignment model used when checking temporal support.",
    )
    parser.add_argument(
        "--stain_temporal_prune_min_align_cc",
        type=float,
        default=0.45,
        help="Minimum ECC alignment score required to use a neighbor for pruning support.",
    )
    parser.add_argument(
        "--stain_temporal_prune_identity_fallback_max_gap",
        type=int,
        default=3,
        help=(
            "If temporal prune alignment fails, directly use neighboring mask support within this many "
            "frames. 0 disables."
        ),
    )
    parser.add_argument(
        "--stain_temporal_prune_support_dilate_kernel",
        type=int,
        default=15,
        help="Dilate aligned neighbor masks by this kernel before overlap testing.",
    )
    parser.add_argument(
        "--stain_temporal_prune_min_overlap_ratio",
        type=float,
        default=0.12,
        help="Minimum component overlap with aligned temporal support required to keep it.",
    )
    parser.set_defaults(camera_names=list(camera_names))
    return parser


def run_cli(description: str, default_root: Path, camera_names: Sequence[str]) -> None:
    parser = build_parser(description, default_root, camera_names)
    args = parser.parse_args()

    input_h5 = resolve_input_h5(args.input_h5, Path(args.dataset_root))
    output_dir = infer_output_dir(input_h5, args.output_dir)
    convert_merged_h5(
        input_h5=input_h5,
        output_dir=output_dir,
        camera_names=args.camera_names,
        min_len=int(args.min_len),
        max_len=int(args.max_len),
        compression=str(args.compression).lower(),
        gzip_level=int(args.gzip_level),
        overwrite=bool(args.overwrite),
        write_summary=bool(args.write_summary),
        stain_mask_mode=str(args.stain_mask_mode),
        stain_reference_episode=str(args.stain_reference_episode),
        stain_exclude_reference_episode=not bool(args.keep_stain_reference_episode),
        stain_diff_thresh=float(args.stain_diff_thresh),
        stain_dark_thresh=float(args.stain_dark_thresh),
        reflection_v_thresh=float(args.reflection_v_thresh),
        reflection_s_thresh=float(args.reflection_s_thresh),
        stain_min_area=int(args.stain_min_area),
        stain_max_area=int(args.stain_max_area),
        stain_morph_kernel=int(args.stain_morph_kernel),
        stain_hole_close_kernel=int(args.stain_hole_close_kernel),
        stain_ref_blur_kernel=int(args.stain_ref_blur_kernel),
        stain_ref_align_mode=str(args.stain_ref_align_mode),
        stain_ref_align_max_iters=int(args.stain_ref_align_max_iters),
        stain_ref_align_eps=float(args.stain_ref_align_eps),
        stain_ref_surface_v_min=float(args.stain_ref_surface_v_min),
        stain_context_v_min=float(args.stain_context_v_min),
        stain_context_kernel=int(args.stain_context_kernel),
        stain_dark_prior_enable=bool(args.stain_dark_prior_enable),
        stain_dark_prior_component_min_diff_ratio=float(args.stain_dark_prior_component_min_diff_ratio),
        stain_component_dark_v_max=float(args.stain_component_dark_v_max),
        stain_component_dark_min_ratio=float(args.stain_component_dark_min_ratio),
        stain_output_constraint_kernel=int(args.stain_output_constraint_kernel),
        stain_ignore_border_px=int(args.stain_ignore_border_px),
        stain_component_context_pad=int(args.stain_component_context_pad),
        stain_component_context_min_ratio=float(args.stain_component_context_min_ratio),
        stain_component_max_width_frac=float(args.stain_component_max_width_frac),
        stain_component_max_height_frac=float(args.stain_component_max_height_frac),
        stain_component_max_area_frac=float(args.stain_component_max_area_frac),
        stain_component_max_aspect_ratio=float(args.stain_component_max_aspect_ratio),
        stain_fill_holes_max_area=int(args.stain_fill_holes_max_area),
        stain_ref_pose_pos_scale=float(args.stain_ref_pose_pos_scale),
        stain_ref_pose_rot_scale=float(args.stain_ref_pose_rot_scale),
        stain_reference_max_pose_dist=float(args.stain_reference_max_pose_dist),
        stain_reference_top_k=int(args.stain_reference_top_k),
        stain_reference_match_mode=str(args.stain_reference_match_mode),
        stain_reference_match_window=int(args.stain_reference_match_window),
        stain_reference_diff_percentile=float(args.stain_reference_diff_percentile),
        stain_reference_diff_min_support=float(args.stain_reference_diff_min_support),
        stain_ref_surface_min_support=float(args.stain_ref_surface_min_support),
        stain_local_dark_thresh=float(args.stain_local_dark_thresh),
        stain_local_dark_ref_delta=float(args.stain_local_dark_ref_delta),
        stain_local_dark_ref_percentile=float(args.stain_local_dark_ref_percentile),
        stain_local_dark_blur_kernel=int(args.stain_local_dark_blur_kernel),
        stain_adaptive_v_floor_percentile=float(args.stain_adaptive_v_floor_percentile),
        stain_adaptive_v_floor_ratio=float(args.stain_adaptive_v_floor_ratio),
        stain_adaptive_v_floor_min=float(args.stain_adaptive_v_floor_min),
        stain_temporal_fill_enable=bool(args.stain_temporal_fill_enable),
        stain_temporal_fill_max_gap=int(args.stain_temporal_fill_max_gap),
        stain_temporal_fill_min_pixels=int(args.stain_temporal_fill_min_pixels),
        stain_temporal_fill_align_mode=str(args.stain_temporal_fill_align_mode),
        stain_temporal_fill_min_align_cc=float(args.stain_temporal_fill_min_align_cc),
        stain_temporal_fill_identity_fallback_max_gap=int(args.stain_temporal_fill_identity_fallback_max_gap),
        stain_temporal_prune_enable=bool(args.stain_temporal_prune_enable),
        stain_temporal_prune_max_gap=int(args.stain_temporal_prune_max_gap),
        stain_temporal_prune_align_mode=str(args.stain_temporal_prune_align_mode),
        stain_temporal_prune_min_align_cc=float(args.stain_temporal_prune_min_align_cc),
        stain_temporal_prune_identity_fallback_max_gap=int(args.stain_temporal_prune_identity_fallback_max_gap),
        stain_temporal_prune_support_dilate_kernel=int(args.stain_temporal_prune_support_dilate_kernel),
        stain_temporal_prune_min_overlap_ratio=float(args.stain_temporal_prune_min_overlap_ratio),
    )
