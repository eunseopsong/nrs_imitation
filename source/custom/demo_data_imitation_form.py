#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_data_imitation_form.py

Convert one merged demonstration HDF5 file into episode_*.hdf5 files for
nrs_imitation Flow Matching training.

Supported final observation layouts:
  1) single_cam
     observations/images/cam0
     observations/position, observations/force

  2) dual_cam
     observations/images/cam0, observations/images/cam1
     observations/position, observations/force

  3) dual_cam_marker
     observations/images/cam0, observations/images/cam1
     observations/marker
     observations/position, observations/force

The converter is intentionally tolerant to several merged-HDF5 source layouts.
It searches common key names and writes a canonical ACT/Flow-compatible layout.

Example:
  cd ~/nrs_imitation/source/custom
  python3 demo_data_imitation_form.py \
    --input_h5 /home/eunseop/nrs_imitation/datasets/ACT/20260513_1402/merged_hdf5/vr_demo_merged_20260513_1402.hdf5 \
    --output_dir /home/eunseop/nrs_imitation/datasets/ACT/20260513_1402/episodes_multimodal \
    --require_cam1 \
    --require_marker

If --input_h5 is omitted, the latest merged_hdf5/*.hdf5 under datasets/ACT is used.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np


# =============================================================================
# Search helpers
# =============================================================================

def _as_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def _timestamp_like(name: str) -> bool:
    for fmt in ("%Y%m%d_%H%M", "%Y%m%d%H%M", "%m%d_%H%M"):
        try:
            datetime.strptime(name, fmt)
            return True
        except ValueError:
            pass
    return False


def find_latest_merged_h5(root: str = "/home/eunseop/nrs_imitation/datasets/ACT") -> Path:
    root_p = _as_path(root)
    if not root_p.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root_p}")

    candidates: List[Tuple[int, str, float, Path]] = []
    for h5 in root_p.rglob("*.hdf5"):
        if "merged_hdf5" not in str(h5.parent):
            continue
        run = h5.parent.parent.name if h5.parent.parent else h5.parent.name
        candidates.append((1 if _timestamp_like(run) else 0, run, h5.stat().st_mtime, h5))
    for h5 in root_p.rglob("*.h5"):
        if "merged_hdf5" not in str(h5.parent):
            continue
        run = h5.parent.parent.name if h5.parent.parent else h5.parent.name
        candidates.append((1 if _timestamp_like(run) else 0, run, h5.stat().st_mtime, h5))

    if not candidates:
        raise FileNotFoundError(f"no merged_hdf5/*.h5 or *.hdf5 found under {root_p}")
    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return candidates[0][3]


def infer_default_output_dir(input_h5: Path) -> Path:
    # Typical input:
    # datasets/ACT/YYYYMMDD_HHMM/merged_hdf5/vr_demo_merged_YYYYMMDD_HHMM.hdf5
    if input_h5.parent.name == "merged_hdf5":
        return input_h5.parent.parent / "episodes_multimodal"
    return input_h5.parent / "episodes_multimodal"


def _contains_dataset(g: h5py.Group, key: str) -> bool:
    try:
        obj = g[key]
        return isinstance(obj, h5py.Dataset)
    except Exception:
        return False


def _get_first_dataset(g: h5py.Group, keys: Iterable[str]) -> Optional[np.ndarray]:
    for k in keys:
        if _contains_dataset(g, k):
            return np.asarray(g[k])
    return None


def _all_episode_groups(f: h5py.File) -> List[Tuple[str, h5py.Group]]:
    """Return episode groups from common merged-HDF5 layouts."""
    if "episodes" in f and isinstance(f["episodes"], h5py.Group):
        eps = []
        for name in sorted(f["episodes"].keys()):
            obj = f["episodes"][name]
            if isinstance(obj, h5py.Group):
                eps.append((name, obj))
        if eps:
            return eps

    # Some files may have ep_0000 directly at root.
    eps = []
    for name in sorted(f.keys()):
        obj = f[name]
        if isinstance(obj, h5py.Group) and name.startswith(("ep_", "episode_")):
            eps.append((name, obj))
    if eps:
        return eps

    # Fallback: root itself behaves like one episode.
    return [("ep_0000", f)]


# =============================================================================
# Dataset key mapping
# =============================================================================

POSITION_KEYS = [
    "observations/position",
    "position",
    "pose",
    "ee_pose",
    "currentP",
    "robot_position",
]

FORCE_KEYS = [
    "observations/force",
    "force",
    "ft",
    "wrench",
    "currentF",
    "measured_force",
]

MARKER_KEYS = [
    "observations/marker",
    "marker",
    "aruco",
    "aruco_marker",
    "marker_pose",
    "aruco_pose",
]

CAMERA_KEY_CANDIDATES: Dict[str, List[str]] = {
    "cam0": [
        "observations/images/cam0",
        "images/cam0",
        "cam0",
        "image",
        "images/image",
        "rgb",
        "color",
        "vr_image",
        "local_image",
        "robot_image",
    ],
    "cam1": [
        "observations/images/cam1",
        "images/cam1",
        "cam1",
        "image2",
        "global_image",
        "scene_image",
        "top_image",
    ],
}


def _normalize_position(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] < 6:
        raise ValueError(f"position must have >=6 dims, got shape={arr.shape}")
    return arr[:, :6].astype(np.float32)


def _normalize_force(arr: np.ndarray, T: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] < 3:
        raise ValueError(f"force must have >=3 dims, got shape={arr.shape}")
    arr = arr[:, :3].astype(np.float32)
    if arr.shape[0] == 1 and T > 1:
        arr = np.repeat(arr, T, axis=0)
    return arr


def _normalize_marker(arr: Optional[np.ndarray], T: int, marker_dim: int) -> np.ndarray:
    """
    Canonical marker shape: (T, marker_dim), default marker_dim=7.
    Suggested convention: [mx,my,mz,mrx,mry,mrz,valid].
    If source marker has 6 dims, valid=1 is appended.
    If source marker is missing, all zeros are written.
    """
    marker_dim = int(marker_dim)
    if arr is None:
        return np.zeros((T, marker_dim), dtype=np.float32)

    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.shape[0] == 1 and T > 1:
        a = np.repeat(a, T, axis=0)

    out = np.zeros((a.shape[0], marker_dim), dtype=np.float32)
    copy_dim = min(marker_dim, a.shape[-1])
    out[:, :copy_dim] = a[:, :copy_dim]

    # If source provides pose6 only and marker_dim allows valid flag, set valid=1.
    if a.shape[-1] == 6 and marker_dim >= 7:
        out[:, 6] = 1.0

    return out


def _normalize_image_array(arr: np.ndarray, T: int) -> np.ndarray:
    """
    Preserve uint8 HDF5 storage if possible.
    Supports (T,H,W,3), (T,3,H,W), (H,W,3), (3,H,W).
    Canonical output: (T,H,W,3), uint8.
    """
    a = np.asarray(arr)
    if a.ndim == 3:
        # Single frame either HWC or CHW.
        if a.shape[0] == 3 and a.shape[-1] != 3:
            a = np.transpose(a, (1, 2, 0))
        a = a[None, ...]

    if a.ndim != 4:
        raise ValueError(f"image must be 3D/4D, got shape={a.shape}")

    if a.shape[1] == 3 and a.shape[-1] != 3:
        # T,C,H,W -> T,H,W,C
        a = np.transpose(a, (0, 2, 3, 1))

    if a.shape[-1] != 3:
        raise ValueError(f"image last dim must be 3 after normalization, got shape={a.shape}")

    if a.shape[0] == 1 and T > 1:
        a = np.repeat(a, T, axis=0)

    if a.dtype != np.uint8:
        # Allow [0,1] float or [0,255] float.
        af = a.astype(np.float32)
        if af.max(initial=0.0) <= 1.5:
            af = af * 255.0
        a = np.clip(af, 0, 255).astype(np.uint8)
    return a


def _align_length(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    lens = [int(a.shape[0]) for a in arrays if a is not None]
    if not lens:
        raise ValueError("no arrays for length alignment")
    T = min(lens)
    return tuple(a[:T] for a in arrays)


def _read_episode(ep: h5py.Group, require_cam1: bool, require_marker: bool, marker_dim: int):
    pos = _get_first_dataset(ep, POSITION_KEYS)
    if pos is None:
        raise KeyError(f"missing position dataset. tried: {POSITION_KEYS}")
    pos = _normalize_position(pos)
    T = int(pos.shape[0])

    force = _get_first_dataset(ep, FORCE_KEYS)
    if force is None:
        raise KeyError(f"missing force dataset. tried: {FORCE_KEYS}")
    force = _normalize_force(force, T=T)

    cam0_raw = _get_first_dataset(ep, CAMERA_KEY_CANDIDATES["cam0"])
    if cam0_raw is None:
        raise KeyError(f"missing cam0 image dataset. tried: {CAMERA_KEY_CANDIDATES['cam0']}")
    cam0 = _normalize_image_array(cam0_raw, T=T)

    cam1_raw = _get_first_dataset(ep, CAMERA_KEY_CANDIDATES["cam1"])
    if cam1_raw is None:
        if require_cam1:
            raise KeyError(f"missing cam1 image dataset. tried: {CAMERA_KEY_CANDIDATES['cam1']}")
        cam1 = None
    else:
        cam1 = _normalize_image_array(cam1_raw, T=T)

    marker_raw = _get_first_dataset(ep, MARKER_KEYS)
    if marker_raw is None and require_marker:
        raise KeyError(f"missing marker dataset. tried: {MARKER_KEYS}")
    marker = _normalize_marker(marker_raw, T=T, marker_dim=marker_dim)

    arrays = [pos, force, cam0, marker]
    if cam1 is not None:
        arrays.append(cam1)
    aligned = _align_length(*arrays)

    pos = aligned[0]
    force = aligned[1]
    cam0 = aligned[2]
    marker = aligned[3]
    if cam1 is not None:
        cam1 = aligned[4]
    T = pos.shape[0]

    return pos, force, cam0, cam1, marker, T


# =============================================================================
# Writer
# =============================================================================

def write_episode(
    out_path: Path,
    position: np.ndarray,
    force: np.ndarray,
    cam0: np.ndarray,
    cam1: Optional[np.ndarray],
    marker: np.ndarray,
    source_name: str,
    compression: str = "gzip",
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T = int(position.shape[0])
    is_pad = np.zeros((T,), dtype=np.bool_)

    with h5py.File(str(out_path), "w") as f:
        obs = f.create_group("observations")
        act = f.create_group("action")
        imgs = obs.create_group("images")
        meta = f.create_group("meta")

        obs.create_dataset("position", data=position.astype(np.float32), compression=compression)
        obs.create_dataset("force", data=force.astype(np.float32), compression=compression)
        obs.create_dataset("marker", data=marker.astype(np.float32), compression=compression)
        obs.create_dataset("is_pad", data=is_pad, compression=compression)

        imgs.create_dataset("cam0", data=cam0, compression=compression)
        camera_names = ["cam0"]
        if cam1 is not None:
            imgs.create_dataset("cam1", data=cam1, compression=compression)
            camera_names.append("cam1")

        # ACT/Flow-compatible action = next target chunk in the same 9D space.
        act.create_dataset("position", data=position.astype(np.float32), compression=compression)
        act.create_dataset("force", data=force.astype(np.float32), compression=compression)

        meta.attrs["orig_len"] = T
        meta.attrs["T_pad"] = T
        meta.attrs["pad_starts_at"] = T
        meta.attrs["truncated"] = False
        meta.attrs["source_episode"] = source_name
        meta.attrs["camera_names"] = json.dumps(camera_names)
        meta.attrs["marker_dim"] = int(marker.shape[-1])
        meta.attrs["schema"] = "nrs_imitation_multimodal_v1"


# =============================================================================
# Main
# =============================================================================

def main(args):
    input_h5 = _as_path(args.input_h5) if args.input_h5 else find_latest_merged_h5(args.dataset_root)
    output_dir = _as_path(args.output_dir) if args.output_dir else infer_default_output_dir(input_h5)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] input_h5      = {input_h5}")
    print(f"[INFO] output_dir    = {output_dir}")
    print(f"[INFO] require_cam1  = {args.require_cam1}")
    print(f"[INFO] require_marker= {args.require_marker}")
    print(f"[INFO] marker_dim    = {args.marker_dim}")

    written = 0
    skipped: List[Tuple[str, str]] = []

    with h5py.File(str(input_h5), "r") as f:
        episode_groups = _all_episode_groups(f)
        print(f"[INFO] found source episodes = {len(episode_groups)}")

        for ep_idx, (name, ep) in enumerate(episode_groups):
            try:
                pos, force, cam0, cam1, marker, T = _read_episode(
                    ep,
                    require_cam1=bool(args.require_cam1),
                    require_marker=bool(args.require_marker),
                    marker_dim=int(args.marker_dim),
                )
                if T < int(args.min_len):
                    skipped.append((name, f"too short: T={T}"))
                    continue
                out_path = output_dir / f"episode_{written}.hdf5"
                write_episode(
                    out_path=out_path,
                    position=pos,
                    force=force,
                    cam0=cam0,
                    cam1=cam1,
                    marker=marker,
                    source_name=name,
                    compression=args.compression,
                )
                print(f"[WRITE] {out_path.name} | T={T} cam1={cam1 is not None} marker_dim={marker.shape[-1]}")
                written += 1
            except Exception as e:
                skipped.append((name, repr(e)))
                print(f"[SKIP] {name}: {e}")

    print(f"\n[DONE] written episodes = {written}")
    if skipped:
        print(f"[WARN] skipped episodes = {len(skipped)}")
        for name, reason in skipped[:20]:
            print(f"  - {name}: {reason}")

    if written <= 0:
        raise RuntimeError("No episodes were written. Check source key names or disable require flags.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_h5", type=str, default=None)
    parser.add_argument("--dataset_root", type=str, default="/home/eunseop/nrs_imitation/datasets/ACT")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--require_cam1", action="store_true", default=False)
    parser.add_argument("--require_marker", action="store_true", default=False)
    parser.add_argument("--marker_dim", type=int, default=7)
    parser.add_argument("--min_len", type=int, default=2)
    parser.add_argument("--compression", type=str, default="gzip", choices=["gzip", "lzf", "none"])
    args = parser.parse_args()
    if args.compression == "none":
        args.compression = None
    main(args)
