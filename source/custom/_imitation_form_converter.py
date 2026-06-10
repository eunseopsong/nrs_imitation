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
          ├── cam1  # dual-camera only
          └── stain_mask  # optional, copied when present in merged HDF5

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


def _align_reference_to_current(
    reference_rgb: np.ndarray,
    current_rgb: np.ndarray,
    mode: str,
    max_iters: int,
    eps: float,
) -> np.ndarray:
    mode = str(mode or "none").strip().lower()
    if cv2 is None or mode in ("", "none", "off"):
        return reference_rgb

    ref = _as_uint8_rgb(reference_rgb)
    cur = _as_uint8_rgb(current_rgb)
    if ref.shape != cur.shape:
        return ref

    if mode not in ("translation", "euclidean", "affine"):
        return ref

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
        _, warp = cv2.findTransformECC(
            cur_gray,
            ref_gray,
            warp,
            warp_mode,
            criteria,
            None,
            1,
        )
        aligned = cv2.warpAffine(
            ref,
            warp,
            (cur.shape[1], cur.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned.astype(np.uint8)
    except Exception:
        return ref


def _constrain_to_near_mask(mask_u8: np.ndarray, core_bool: np.ndarray, kernel_size: int) -> np.ndarray:
    k = _odd_kernel_size(kernel_size)
    if cv2 is None or k <= 1:
        return mask_u8
    core = np.asarray(core_bool, dtype=np.uint8)
    kernel = np.ones((k, k), dtype=np.uint8)
    allowed = cv2.dilate(core, kernel) > 0
    return (np.asarray(mask_u8) * allowed.astype(np.uint8)).astype(np.uint8)


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


def _extract_blob_proposals(
    mask_u8: np.ndarray,
    score_map: Optional[np.ndarray],
    max_blobs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_blobs = max(0, int(max_blobs))
    bboxes = np.zeros((max_blobs, 4), dtype=np.float32)
    features = np.zeros((max_blobs, 6), dtype=np.float32)
    is_pad = np.ones((max_blobs,), dtype=np.bool_)
    if max_blobs <= 0 or cv2 is None:
        return bboxes, features, is_pad

    binary = (np.asarray(mask_u8) > 0).astype(np.uint8)
    h, w = binary.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    proposals = []
    score_arr = None if score_map is None else np.asarray(score_map, dtype=np.float32)
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        if score_arr is not None:
            comp = labels == label
            score = float(np.mean(score_arr[comp])) * float(area)
        else:
            score = float(area)
        proposals.append((score, x, y, x + bw, y + bh, area))

    proposals.sort(key=lambda item: item[0], reverse=True)
    for out_i, (score, x1, y1, x2, y2, area) in enumerate(proposals[:max_blobs]):
        bboxes[out_i] = np.asarray([x1, y1, x2, y2], dtype=np.float32)
        features[out_i] = np.asarray(
            [
                ((x1 + x2) * 0.5) / max(w, 1),
                ((y1 + y2) * 0.5) / max(h, 1),
                (x2 - x1) / max(w, 1),
                (y2 - y1) / max(h, 1),
                area / max(h * w, 1),
                score / max(h * w * 255.0, 1.0),
            ],
            dtype=np.float32,
        )
        is_pad[out_i] = False
    return bboxes, features, is_pad


def _nearest_reference_index_bank(
    current_position: np.ndarray,
    reference_position: np.ndarray,
    pos_scale: float,
    rot_scale: float,
    top_k: int,
) -> np.ndarray:
    cur = _ensure_2d_min_dim(current_position, 6, "current_position").astype(np.float32)
    ref = _ensure_2d_min_dim(reference_position, 6, "reference_position").astype(np.float32)
    if ref.shape[0] <= 0:
        raise ValueError("reference_position must contain at least one frame")
    pos_scale = max(float(pos_scale), 1e-6)
    rot_scale = max(float(rot_scale), 1e-6)
    ref_pos = ref[:, :3] / pos_scale
    ref_rot = ref[:, 3:6] / rot_scale
    ref_scaled = np.concatenate([ref_pos, ref_rot], axis=1)

    k = min(max(1, int(top_k)), int(ref_scaled.shape[0]))
    out = np.zeros((cur.shape[0], k), dtype=np.int64)
    for i in range(cur.shape[0]):
        cur_scaled = np.concatenate([cur[i, :3] / pos_scale, cur[i, 3:6] / rot_scale], axis=0)
        dist = np.sum((ref_scaled - cur_scaled[None, :]) ** 2, axis=1)
        if k == 1:
            out[i, 0] = int(np.argmin(dist))
        else:
            nearest = np.argpartition(dist, k - 1)[:k]
            out[i] = nearest[np.argsort(dist[nearest])].astype(np.int64)
    return out


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
    output_constraint_kernel: int,
    ignore_border_px: int,
    component_context_pad: int,
    component_context_min_ratio: float,
    component_max_width_frac: float,
    component_max_height_frac: float,
    component_max_area_frac: float,
    fill_holes_max_area: int,
    max_blobs: int,
    pose_pos_scale: float,
    pose_rot_scale: float,
    reference_top_k: int,
    reference_diff_percentile: float,
    local_dark_thresh: float,
    local_dark_blur_kernel: int,
    adaptive_v_floor_percentile: float,
    adaptive_v_floor_ratio: float,
    adaptive_v_floor_min: float,
) -> Dict[str, np.ndarray]:
    cur_rgb = _ensure_image4(current_rgb, "cam0")
    ref_rgb = _ensure_image4(reference_rgb, "reference_cam0")
    T, H, W, _ = cur_rgb.shape
    ref_index_bank = _nearest_reference_index_bank(
        current_position=current_position,
        reference_position=reference_position,
        pos_scale=pose_pos_scale,
        rot_scale=pose_rot_scale,
        top_k=reference_top_k,
    )
    ref_indices = ref_index_bank[:, 0]

    masks = np.zeros((T, H, W, 1), dtype=np.uint8)
    bboxes = np.zeros((T, max(0, int(max_blobs)), 4), dtype=np.float32)
    features = np.zeros((T, max(0, int(max_blobs)), 6), dtype=np.float32)
    is_pad = np.ones((T, max(0, int(max_blobs))), dtype=np.bool_)

    for i in range(T):
        cur_frame = _as_uint8_rgb(cur_rgb[i])
        cur_v, cur_s = _rgb_to_value_saturation(cur_frame)
        cur_norm = _lighting_normalized_value(cur_v, blur_kernel)
        local_dark, local_surface = _local_darkness(cur_v, local_dark_blur_kernel)

        diff_candidates: List[np.ndarray] = []
        ref_surface_candidates: List[np.ndarray] = []
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
            diff_candidates.append((ref_norm - cur_norm).astype(np.float32))
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

        if ref_surface_candidates:
            ref_surface = np.any(np.stack(ref_surface_candidates, axis=0), axis=0)
        else:
            ref_surface = True

        context_floor = _adaptive_v_floor(
            local_surface,
            requested_floor=context_v_min,
            percentile=adaptive_v_floor_percentile,
            ratio=adaptive_v_floor_ratio,
            min_floor=adaptive_v_floor_min,
        )
        dark_change = diff >= float(diff_thresh)
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
        dark_prior = bool(dark_prior_enable) & local_dark_current
        stain_evidence = dark_change | dark_prior
        raw_candidate = stain_evidence & dark_current & ref_surface & surface_context & (~reflection)
        mask = _postprocess_binary_mask(
            raw_candidate,
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
            fill_holes_max_area=fill_holes_max_area,
        )
        mask = _constrain_to_near_mask(mask, raw_candidate, output_constraint_kernel)
        masks[i, :, :, 0] = mask
        frame_bboxes, frame_features, frame_is_pad = _extract_blob_proposals(
            mask,
            score_map=np.maximum(
                np.clip(diff, 0.0, 255.0),
                np.clip(local_dark, 0.0, 255.0),
            ),
            max_blobs=max_blobs,
        )
        if max_blobs > 0:
            bboxes[i] = frame_bboxes
            features[i] = frame_features
            is_pad[i] = frame_is_pad

    return {
        "stain_mask": masks,
        "stain_blob_bboxes": bboxes,
        "stain_blob_features": features,
        "stain_blob_is_pad": is_pad,
        "stain_reference_indices": ref_indices.astype(np.int64),
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

        g_obs = f.create_group("observations")
        g_obs.create_dataset("position", data=data["position"].astype(np.float32), **kwargs)
        g_obs.create_dataset("force", data=data["force"].astype(np.float32), **kwargs)

        g_images = g_obs.create_group("images")
        for cam in camera_names:
            g_images.create_dataset(cam, data=data[cam].astype(np.uint8), **kwargs)
        if "stain_mask" in data:
            ds = g_images.create_dataset("stain_mask", data=data["stain_mask"].astype(np.uint8), **kwargs)
            ds.attrs["shape_convention"] = "T,H,W,1"
            ds.attrs["storage"] = "uint8_0_255"
            ds.attrs["model_value_range_after_div255"] = "float32_0_1"
            f.attrs["has_stain_mask"] = 1
        if "stain_blob_bboxes" in data:
            g_obs.create_dataset("stain_blob_bboxes", data=data["stain_blob_bboxes"].astype(np.float32), **kwargs)
        if "stain_blob_features" in data:
            g_obs.create_dataset("stain_blob_features", data=data["stain_blob_features"].astype(np.float32), **kwargs)
        if "stain_blob_is_pad" in data:
            g_obs.create_dataset("stain_blob_is_pad", data=data["stain_blob_is_pad"].astype(np.bool_), **kwargs)
        if "stain_reference_indices" in data:
            g_obs.create_dataset("stain_reference_indices", data=data["stain_reference_indices"].astype(np.int64), **kwargs)


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
    stain_morph_kernel: int = 5,
    stain_hole_close_kernel: int = 7,
    stain_ref_blur_kernel: int = 31,
    stain_ref_align_mode: str = "euclidean",
    stain_ref_align_max_iters: int = 40,
    stain_ref_align_eps: float = 1e-4,
    stain_ref_surface_v_min: float = 120.0,
    stain_context_v_min: float = 120.0,
    stain_context_kernel: int = 31,
    stain_dark_prior_enable: bool = True,
    stain_output_constraint_kernel: int = 11,
    stain_ignore_border_px: int = 0,
    stain_component_context_pad: int = 12,
    stain_component_context_min_ratio: float = 0.08,
    stain_component_max_width_frac: float = 0.0,
    stain_component_max_height_frac: float = 0.0,
    stain_component_max_area_frac: float = 0.12,
    stain_fill_holes_max_area: int = 400,
    stain_max_blobs: int = 8,
    stain_ref_pose_pos_scale: float = 50.0,
    stain_ref_pose_rot_scale: float = 0.35,
    stain_reference_top_k: int = 3,
    stain_reference_diff_percentile: float = 75.0,
    stain_local_dark_thresh: float = 14.0,
    stain_local_dark_blur_kernel: int = 41,
    stain_adaptive_v_floor_percentile: float = 75.0,
    stain_adaptive_v_floor_ratio: float = 0.85,
    stain_adaptive_v_floor_min: float = 60.0,
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
                        output_constraint_kernel=stain_output_constraint_kernel,
                        ignore_border_px=stain_ignore_border_px,
                        component_context_pad=stain_component_context_pad,
                        component_context_min_ratio=stain_component_context_min_ratio,
                        component_max_width_frac=stain_component_max_width_frac,
                        component_max_height_frac=stain_component_max_height_frac,
                        component_max_area_frac=stain_component_max_area_frac,
                        fill_holes_max_area=stain_fill_holes_max_area,
                        max_blobs=stain_max_blobs,
                        pose_pos_scale=stain_ref_pose_pos_scale,
                        pose_rot_scale=stain_ref_pose_rot_scale,
                        reference_top_k=stain_reference_top_k,
                        reference_diff_percentile=stain_reference_diff_percentile,
                        local_dark_thresh=stain_local_dark_thresh,
                        local_dark_blur_kernel=stain_local_dark_blur_kernel,
                        adaptive_v_floor_percentile=stain_adaptive_v_floor_percentile,
                        adaptive_v_floor_ratio=stain_adaptive_v_floor_ratio,
                        adaptive_v_floor_min=stain_adaptive_v_floor_min,
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
                if "stain_blob_bboxes" in data:
                    image_items.append(f"stain_blobs={data['stain_blob_bboxes'].shape}")
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
                    "stain_reference_top_k": int(stain_reference_top_k),
                    "stain_reference_diff_percentile": float(stain_reference_diff_percentile),
                    "stain_local_dark_thresh": float(stain_local_dark_thresh),
                    "stain_ref_align_mode": str(stain_ref_align_mode),
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
    parser.add_argument("--stain_morph_kernel", type=int, default=5)
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
        default="euclidean",
        choices=["none", "translation", "euclidean", "affine"],
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
        help="Allow dark-on-bright-surface evidence even when reference difference is weak.",
    )
    parser.add_argument(
        "--no_stain_dark_prior",
        dest="stain_dark_prior_enable",
        action="store_false",
        help="Require reference difference evidence for every stain pixel.",
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
        default=0,
        help="Drop stain components touching the image border within this many pixels. 0 disables.",
    )
    parser.add_argument("--stain_component_context_pad", type=int, default=12)
    parser.add_argument(
        "--stain_component_context_min_ratio",
        type=float,
        default=0.08,
        help="Require this ratio of bright surface pixels around each component. 0 disables.",
    )
    parser.add_argument("--stain_component_max_width_frac", type=float, default=0.0)
    parser.add_argument("--stain_component_max_height_frac", type=float, default=0.0)
    parser.add_argument(
        "--stain_component_max_area_frac",
        type=float,
        default=0.12,
        help="Drop components covering more than this image fraction. 0 disables.",
    )
    parser.add_argument(
        "--stain_fill_holes_max_area",
        type=int,
        default=400,
        help="Fill enclosed holes in stain components up to this area. -1 fills all holes, 0 disables.",
    )
    parser.add_argument("--stain_max_blobs", type=int, default=8)
    parser.add_argument("--stain_ref_pose_pos_scale", type=float, default=50.0)
    parser.add_argument("--stain_ref_pose_rot_scale", type=float, default=0.35)
    parser.add_argument(
        "--stain_reference_top_k",
        type=int,
        default=3,
        help="Use this many nearest clean reference frames and fuse their dark-change evidence.",
    )
    parser.add_argument(
        "--stain_reference_diff_percentile",
        type=float,
        default=75.0,
        help="Percentile used to fuse top-k reference difference maps. 50=median, 100=max.",
    )
    parser.add_argument(
        "--stain_local_dark_thresh",
        type=float,
        default=14.0,
        help="Local contrast threshold for dark stains, independent of absolute image brightness.",
    )
    parser.add_argument("--stain_local_dark_blur_kernel", type=int, default=41)
    parser.add_argument("--stain_adaptive_v_floor_percentile", type=float, default=75.0)
    parser.add_argument("--stain_adaptive_v_floor_ratio", type=float, default=0.85)
    parser.add_argument("--stain_adaptive_v_floor_min", type=float, default=60.0)
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
        stain_output_constraint_kernel=int(args.stain_output_constraint_kernel),
        stain_ignore_border_px=int(args.stain_ignore_border_px),
        stain_component_context_pad=int(args.stain_component_context_pad),
        stain_component_context_min_ratio=float(args.stain_component_context_min_ratio),
        stain_component_max_width_frac=float(args.stain_component_max_width_frac),
        stain_component_max_height_frac=float(args.stain_component_max_height_frac),
        stain_component_max_area_frac=float(args.stain_component_max_area_frac),
        stain_fill_holes_max_area=int(args.stain_fill_holes_max_area),
        stain_max_blobs=int(args.stain_max_blobs),
        stain_ref_pose_pos_scale=float(args.stain_ref_pose_pos_scale),
        stain_ref_pose_rot_scale=float(args.stain_ref_pose_rot_scale),
        stain_reference_top_k=int(args.stain_reference_top_k),
        stain_reference_diff_percentile=float(args.stain_reference_diff_percentile),
        stain_local_dark_thresh=float(args.stain_local_dark_thresh),
        stain_local_dark_blur_kernel=int(args.stain_local_dark_blur_kernel),
        stain_adaptive_v_floor_percentile=float(args.stain_adaptive_v_floor_percentile),
        stain_adaptive_v_floor_ratio=float(args.stain_adaptive_v_floor_ratio),
        stain_adaptive_v_floor_min=float(args.stain_adaptive_v_floor_min),
    )
