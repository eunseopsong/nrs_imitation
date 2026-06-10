#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
source/data/dataset.py

Multimodal ACT/Flow-compatible HDF5 dataloader for nrs_imitation.

Supported observation modes:
  - single_cam       : cam0 + qpos + optional force_history
  - dual_cam         : cam0/cam1 + qpos + optional force_history
  - single_cam_marker: cam0 + marker + qpos + optional force_history

Canonical episode layout:
  episode_0.hdf5
  ├── action/position        (T,6)
  ├── action/force           (T,3)
  ├── action/gripper_present_position (T,), optional
  ├── observations/position  (T,6)
  ├── observations/force     (T,3)
  ├── observations/marker    (T,M), optional
  ├── observations/gripper/present_position, optional
  ├── observations/gripper/present_current_mA, optional
  ├── observations/images/cam0
  ├── observations/images/stain_mask, optional
  ├── observations/images/cam1, optional
  └── observations/is_pad, optional

Return tuple:
  default without marker:
    image, qpos, action, is_pad, force_history
  default with marker:
    image, qpos, action, is_pad, force_history, marker
  include_gripper=True without marker:
    image, qpos, action, is_pad, force_history, gripper_position, gripper_current
  include_gripper=True with marker:
    image, qpos, action, is_pad, force_history, marker, gripper_position, gripper_current

Shapes:
  image         : (K,3,H,W), float32 in [0,1]
  stain_mask    : (1,H,W), float32 in [0,1], appended when use_stain_mask=True
  qpos          : (9,), normalized
  action        : (seq_len,9) or (seq_len,10), normalized
  is_pad        : (seq_len,), bool
  force_history : (L,3), normalized, if requested
  marker        : (M,), normalized, only if obs_mode is a marker mode
  gripper_position : (1,), float32, raw position value cast from int when include_gripper=True
  gripper_current  : (1,), float32, normalized when include_gripper=True
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Small helpers
# =============================================================================

def _episode_files(dataset_dir: str | Path, num_episodes: int = 0) -> List[Path]:
    d = Path(dataset_dir).expanduser()
    if not d.is_dir():
        raise FileNotFoundError(f"dataset_dir does not exist: {d}")
    files = sorted(d.glob("episode_*.hdf5"))
    if not files:
        files = sorted(d.glob("episode_*.h5"))
    if not files:
        raise FileNotFoundError(f"no episode_*.hdf5 files found in {d}")
    if num_episodes is not None and int(num_episodes) > 0:
        files = files[: int(num_episodes)]
    return files


def _read_dataset(f: h5py.File, keys: Sequence[str], required: bool = True):
    ds = _find_dataset(f, keys, required=required)
    if ds is None:
        return None
    return np.asarray(ds)


def _find_dataset(f: h5py.File, keys: Sequence[str], required: bool = True):
    for k in keys:
        try:
            if k in f:
                return f[k]
        except Exception:
            pass
    if required:
        raise KeyError(f"missing dataset; tried keys={list(keys)}")
    return None


def _read_position(f: h5py.File) -> np.ndarray:
    arr = _read_dataset(f, ["observations/position", "position", "pose"], required=True)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] < 6:
        raise ValueError(f"position must have >=6 dims, got {arr.shape}")
    return arr[:, :6]


def _read_force(f: h5py.File, T: int) -> np.ndarray:
    arr = _read_dataset(f, ["observations/force", "force", "ft"], required=True)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] < 3:
        raise ValueError(f"force must have >=3 dims, got {arr.shape}")
    arr = arr[:, :3]
    if arr.shape[0] == 1 and T > 1:
        arr = np.repeat(arr, T, axis=0)
    return arr


def _read_marker(f: h5py.File, T: int, marker_dim: int) -> np.ndarray:
    arr = _read_dataset(f, ["observations/marker", "marker", "aruco", "aruco_pose"], required=False)
    if arr is None:
        return np.zeros((T, marker_dim), dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 1 and T > 1:
        arr = np.repeat(arr, T, axis=0)
    out = np.zeros((arr.shape[0], marker_dim), dtype=np.float32)
    d = min(marker_dim, arr.shape[-1])
    out[:, :d] = arr[:, :d]
    if arr.shape[-1] == 6 and marker_dim >= 7:
        out[:, 6] = 1.0
    return out


def _read_gripper_state(f: h5py.File, T: int) -> Tuple[np.ndarray, np.ndarray]:
    position = _read_dataset(
        f,
        [
            "observations/gripper/present_position",
            "gripper/present_position",
        ],
        required=True,
    )
    current = _read_dataset(
        f,
        [
            "observations/gripper/present_current_mA",
            "gripper/present_current_mA",
        ],
        required=True,
    )
    position = np.asarray(position, dtype=np.float32).reshape(-1)
    current = np.asarray(current, dtype=np.float32).reshape(-1)
    if position.shape[0] == 1 and T > 1:
        position = np.repeat(position, T, axis=0)
    if current.shape[0] == 1 and T > 1:
        current = np.repeat(current, T, axis=0)
    return position, current


def _read_action(
    f: h5py.File,
    fallback_pos: np.ndarray,
    fallback_force: np.ndarray,
    fallback_gripper_position: Optional[np.ndarray] = None,
    include_gripper: bool = False,
) -> np.ndarray:
    pos = _read_dataset(f, ["action/position", "actions/position"], required=False)
    force = _read_dataset(f, ["action/force", "actions/force"], required=False)
    gripper_pos = _read_dataset(
        f,
        ["action/gripper_present_position", "actions/gripper_present_position"],
        required=False,
    )
    if pos is None:
        pos = fallback_pos
    if force is None:
        force = fallback_force
    if include_gripper and gripper_pos is None:
        if fallback_gripper_position is None:
            raise KeyError("Missing action/gripper_present_position and no fallback provided")
        gripper_pos = fallback_gripper_position
    pos = np.asarray(pos, dtype=np.float32)
    force = np.asarray(force, dtype=np.float32)
    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    if force.ndim == 1:
        force = force.reshape(1, -1)
    out = [pos[:, :6], force[:, :3]]
    if include_gripper and gripper_pos is not None:
        gripper_pos = np.asarray(gripper_pos, dtype=np.float32).reshape(-1, 1)
        if gripper_pos.shape[0] == 1 and pos.shape[0] > 1:
            gripper_pos = np.repeat(gripper_pos, pos.shape[0], axis=0)
        out.append(gripper_pos[:, :1])
    return np.concatenate(out, axis=-1).astype(np.float32)


def _read_image(f: h5py.File, camera_name: str) -> np.ndarray:
    keys = [
        f"observations/images/{camera_name}",
        f"images/{camera_name}",
        camera_name,
    ]
    if camera_name == "cam0":
        keys += ["observations/image", "image", "rgb", "color"]
    arr = _read_dataset(f, keys, required=True)
    return np.asarray(arr)


def _read_image_frame(f: h5py.File, camera_name: str, frame_idx: int) -> np.ndarray:
    keys = [
        f"observations/images/{camera_name}",
        f"images/{camera_name}",
        camera_name,
    ]
    if camera_name == "cam0":
        keys += ["observations/image", "image", "rgb", "color"]
    ds = _find_dataset(f, keys, required=True)
    if ds.ndim == 3:
        return np.asarray(ds)
    if ds.ndim != 4:
        raise ValueError(f"image dataset must be 3D or 4D, got {ds.shape}")
    idx = int(np.clip(frame_idx, 0, max(ds.shape[0] - 1, 0)))
    return np.asarray(ds[idx])


def _read_stain_mask_frame(f: h5py.File, stain_mask_key: str, frame_idx: int) -> np.ndarray:
    key = str(stain_mask_key or "observations/images/stain_mask").strip()
    keys = [key]
    for fallback in [
        "observations/images/stain_mask",
        "images/stain_mask",
        "stain_mask",
    ]:
        if fallback not in keys:
            keys.append(fallback)
    ds = _find_dataset(f, keys, required=True)
    if ds.ndim == 2:
        return np.asarray(ds)
    if ds.ndim == 3:
        # Either (T,H,W) or a single-channel image already shaped (H,W,1)/(1,H,W).
        if ds.shape[-1] == 1 or ds.shape[0] == 1:
            return np.asarray(ds)
        idx = int(np.clip(frame_idx, 0, max(ds.shape[0] - 1, 0)))
        return np.asarray(ds[idx])
    if ds.ndim == 4:
        idx = int(np.clip(frame_idx, 0, max(ds.shape[0] - 1, 0)))
        return np.asarray(ds[idx])
    raise ValueError(f"stain_mask dataset must be 2D, 3D, or 4D, got {ds.shape}")


def _image_frame_to_chw_float(frame: np.ndarray) -> torch.Tensor:
    a = np.asarray(frame)
    if a.ndim != 3:
        raise ValueError(f"image frame must be 3D, got {a.shape}")
    if a.shape[0] == 3 and a.shape[-1] != 3:
        chw = a
    elif a.shape[-1] == 3:
        chw = np.transpose(a, (2, 0, 1))
    else:
        raise ValueError(f"cannot interpret image frame shape={a.shape}")
    chw = chw.astype(np.float32)
    if chw.max(initial=0.0) > 1.5:
        chw = chw / 255.0
    return torch.from_numpy(np.clip(chw, 0.0, 1.0))


def _stain_mask_frame_to_chw_float(frame: np.ndarray) -> torch.Tensor:
    a = np.asarray(frame)
    if a.ndim == 2:
        chw = a[None, ...]
    elif a.ndim == 3:
        if a.shape[0] == 1:
            chw = a
        elif a.shape[-1] == 1:
            chw = np.transpose(a, (2, 0, 1))
        else:
            raise ValueError(f"cannot interpret stain_mask frame shape={a.shape}")
    else:
        raise ValueError(f"stain_mask frame must be 2D or 3D, got {a.shape}")

    chw = chw.astype(np.float32)
    if chw.max(initial=0.0) > 1.5:
        chw = chw / 255.0
    return torch.from_numpy(np.clip(chw, 0.0, 1.0))


def _slice_pad(arr: np.ndarray, start: int, length: int, pad_value: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    T = arr.shape[0]
    start = int(np.clip(start, 0, max(T - 1, 0)))
    end = min(start + length, T)
    chunk = arr[start:end]
    pad_n = length - chunk.shape[0]
    is_pad = np.zeros((length,), dtype=np.bool_)
    if pad_n > 0:
        if pad_value is None:
            pad_value = chunk[-1:] if chunk.shape[0] > 0 else np.zeros((1, arr.shape[-1]), dtype=arr.dtype)
        pad = np.repeat(pad_value.reshape(1, -1), pad_n, axis=0).astype(arr.dtype)
        chunk = np.concatenate([chunk, pad], axis=0)
        is_pad[-pad_n:] = True
    return chunk, is_pad


def _force_history(force: np.ndarray, start: int, L: int) -> np.ndarray:
    L = max(1, int(L))
    T = force.shape[0]
    start = int(np.clip(start, 0, max(T - 1, 0)))
    lo = max(0, start - L + 1)
    hist = force[lo:start + 1]
    if hist.shape[0] < L:
        pad = np.repeat(hist[0:1], L - hist.shape[0], axis=0)
        hist = np.concatenate([pad, hist], axis=0)
    return hist.astype(np.float32)


def _sanitize_minmax(vmin: np.ndarray, vmax: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    vmin = np.asarray(vmin, dtype=np.float32)
    vmax = np.asarray(vmax, dtype=np.float32)
    return vmin, np.maximum(vmax, vmin + eps).astype(np.float32)


def normalize_minmax(x: np.ndarray, vmin: np.ndarray, vmax: np.ndarray, mode: str) -> np.ndarray:
    vmin, vmax = _sanitize_minmax(vmin, vmax)
    y = (x - vmin) / np.maximum(vmax - vmin, 1e-6)
    if mode == "minmax_m11":
        y = 2.0 * y - 1.0
        return np.clip(y, -1.0, 1.0).astype(np.float32)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


# =============================================================================
# Stats
# =============================================================================

def compute_dataset_stats(
    episode_paths: Sequence[Path],
    marker_dim: int = 7,
    qpos_norm_mode: str = "minmax_m11",
    action_norm_mode: str = "minmax_m11",
    marker_norm_mode: str = "minmax_m11",
    include_gripper: bool = False,
) -> Dict[str, np.ndarray | str | int]:
    qpos_all = []
    action_all = []
    marker_all = []
    gripper_current_all = []

    for p in episode_paths:
        with h5py.File(str(p), "r") as f:
            pos = _read_position(f)
            force = _read_force(f, T=pos.shape[0])
            if include_gripper:
                gripper_position, gripper_current = _read_gripper_state(f, T=pos.shape[0])
                T = min(pos.shape[0], force.shape[0], gripper_position.shape[0], gripper_current.shape[0])
                gripper_position = gripper_position[:T]
                gripper_current = gripper_current[:T]
            else:
                gripper_position = None
                gripper_current = None
                T = min(pos.shape[0], force.shape[0])
            pos = pos[:T]
            force = force[:T]
            qpos = np.concatenate([pos[:, :6], force[:, :3]], axis=-1).astype(np.float32)
            action = _read_action(
                f,
                fallback_pos=pos,
                fallback_force=force,
                fallback_gripper_position=gripper_position,
                include_gripper=include_gripper,
            )[:T]
            marker = _read_marker(f, T=T, marker_dim=marker_dim)[:T]
            qpos_all.append(qpos)
            action_all.append(action)
            marker_all.append(marker)
            if include_gripper and gripper_current is not None:
                gripper_current_all.append(gripper_current.reshape(-1, 1).astype(np.float32))

    q = np.concatenate(qpos_all, axis=0)
    a = np.concatenate(action_all, axis=0)
    m = np.concatenate(marker_all, axis=0)
    qmin, qmax = _sanitize_minmax(q.min(axis=0), q.max(axis=0))
    amin, amax = _sanitize_minmax(a.min(axis=0), a.max(axis=0))
    mmin, mmax = _sanitize_minmax(m.min(axis=0), m.max(axis=0))

    stats = {
        "qpos_min": qmin.astype(np.float32),
        "qpos_max": qmax.astype(np.float32),
        "action_min": amin.astype(np.float32),
        "action_max": amax.astype(np.float32),
        "marker_min": mmin.astype(np.float32),
        "marker_max": mmax.astype(np.float32),
        "qpos_norm_mode": qpos_norm_mode,
        "action_norm_mode": action_norm_mode,
        "marker_norm_mode": marker_norm_mode,
        "marker_dim": int(marker_dim),
        "include_gripper": bool(include_gripper),
        "num_total_timesteps": int(q.shape[0]),
    }
    if include_gripper and gripper_current_all:
        gc = np.concatenate(gripper_current_all, axis=0)
        gcmin, gcmax = _sanitize_minmax(gc.min(axis=0), gc.max(axis=0))
        stats["gripper_current_min"] = gcmin.astype(np.float32)
        stats["gripper_current_max"] = gcmax.astype(np.float32)
    return stats


# =============================================================================
# Dataset
# =============================================================================

class ImitationEpisodeDataset(Dataset):
    def __init__(
        self,
        episode_paths: Sequence[Path],
        stats: Dict,
        camera_names: Sequence[str],
        obs_mode: str = "single_cam",
        seq_len: int = 200,
        samples_per_episode: int = 50,
        seed: int = 0,
        return_force_history: bool = True,
        force_history_len: int = 10,
        marker_dim: int = 7,
        qpos_norm_mode: str = "minmax_m11",
        action_norm_mode: str = "minmax_m11",
        marker_norm_mode: str = "minmax_m11",
        include_gripper: bool = False,
        use_stain_mask: bool = False,
        stain_mask_key: str = "observations/images/stain_mask",
    ):
        super().__init__()
        self.episode_paths = list(episode_paths)
        self.stats = stats
        self.camera_names = list(camera_names)
        self.obs_mode = str(obs_mode)
        self.seq_len = int(seq_len)
        self.samples_per_episode = int(samples_per_episode)
        self.seed = int(seed)
        self.return_force_history = bool(return_force_history)
        self.force_history_len = int(force_history_len)
        self.marker_dim = int(marker_dim)
        self.qpos_norm_mode = str(qpos_norm_mode)
        self.action_norm_mode = str(action_norm_mode)
        self.marker_norm_mode = str(marker_norm_mode)
        self.include_gripper = bool(include_gripper)
        self.use_stain_mask = bool(use_stain_mask)
        self.stain_mask_key = str(stain_mask_key or "observations/images/stain_mask")

        self.return_marker = self.obs_mode in ("dual_cam_marker", "single_cam_marker")
        self.index = []
        for ep_i in range(len(self.episode_paths)):
            for s_i in range(max(1, self.samples_per_episode)):
                self.index.append((ep_i, s_i))

    def __len__(self) -> int:
        return len(self.index)

    def _choose_start(self, T: int, global_idx: int) -> int:
        if T <= 1:
            return 0
        rng = np.random.default_rng(self.seed + 1000003 * int(global_idx))
        max_start = max(0, T - 1)
        return int(rng.integers(0, max_start + 1))

    def __getitem__(self, idx: int):
        ep_i, _ = self.index[idx]
        path = self.episode_paths[ep_i]
        with h5py.File(str(path), "r") as f:
            pos = _read_position(f)
            force = _read_force(f, T=pos.shape[0])
            if self.include_gripper:
                gripper_position, gripper_current = _read_gripper_state(f, T=pos.shape[0])
            else:
                gripper_position = None
                gripper_current = None
            action = _read_action(
                f,
                fallback_pos=pos,
                fallback_force=force,
                fallback_gripper_position=gripper_position,
                include_gripper=self.include_gripper,
            )
            marker = _read_marker(f, T=pos.shape[0], marker_dim=self.marker_dim)

            if self.include_gripper:
                T = min(
                    pos.shape[0],
                    force.shape[0],
                    action.shape[0],
                    marker.shape[0],
                    gripper_position.shape[0],
                    gripper_current.shape[0],
                )
            else:
                T = min(pos.shape[0], force.shape[0], action.shape[0], marker.shape[0])
            pos = pos[:T]
            force = force[:T]
            action = action[:T]
            marker = marker[:T]
            if self.include_gripper:
                gripper_position = gripper_position[:T]
                gripper_current = gripper_current[:T]
            start = self._choose_start(T, idx)

            # Images use the frame at the current qpos time.
            imgs = []
            for cam in self.camera_names:
                frame = _read_image_frame(f, cam, start)
                imgs.append(_image_frame_to_chw_float(frame))
            image = torch.stack(imgs, dim=0)  # (K,3,H,W)
            if self.use_stain_mask:
                stain_frame = _read_stain_mask_frame(f, self.stain_mask_key, start)
                stain_mask = _stain_mask_frame_to_chw_float(stain_frame)
            else:
                stain_mask = None

        qpos_raw = np.concatenate([pos[start, :6], force[start, :3]], axis=0).astype(np.float32)
        action_chunk, is_pad = _slice_pad(action, start, self.seq_len)
        marker_raw = marker[start].astype(np.float32)
        fh_raw = _force_history(force, start, self.force_history_len)

        qpos = normalize_minmax(qpos_raw, self.stats["qpos_min"], self.stats["qpos_max"], self.qpos_norm_mode)
        action_norm = normalize_minmax(action_chunk, self.stats["action_min"], self.stats["action_max"], self.action_norm_mode)
        marker_norm = normalize_minmax(marker_raw, self.stats["marker_min"], self.stats["marker_max"], self.marker_norm_mode)
        fh_norm = normalize_minmax(fh_raw, self.stats["qpos_min"][6:9], self.stats["qpos_max"][6:9], self.qpos_norm_mode)

        image_t = image.float()
        qpos_t = torch.from_numpy(qpos).float()
        action_t = torch.from_numpy(action_norm).float()
        is_pad_t = torch.from_numpy(is_pad).bool()
        fh_t = torch.from_numpy(fh_norm).float()
        marker_t = torch.from_numpy(marker_norm).float()
        if self.include_gripper:
            gripper_position_raw = np.asarray([gripper_position[start]], dtype=np.float32)
            gripper_current_raw = np.asarray([gripper_current[start]], dtype=np.float32)
            gripper_current_norm = normalize_minmax(
                gripper_current_raw,
                self.stats["gripper_current_min"],
                self.stats["gripper_current_max"],
                self.qpos_norm_mode,
            )
            gripper_position_t = torch.from_numpy(gripper_position_raw).float()
            gripper_current_t = torch.from_numpy(gripper_current_norm).float()
        else:
            gripper_position_t = None
            gripper_current_t = None

        extra = (stain_mask.float(),) if self.use_stain_mask else ()
        if self.include_gripper and self.return_marker:
            return (image_t, qpos_t, action_t, is_pad_t, fh_t, marker_t, gripper_position_t, gripper_current_t) + extra
        if self.include_gripper:
            return (image_t, qpos_t, action_t, is_pad_t, fh_t, gripper_position_t, gripper_current_t) + extra
        if self.return_marker:
            return (image_t, qpos_t, action_t, is_pad_t, fh_t, marker_t) + extra
        return (image_t, qpos_t, action_t, is_pad_t, fh_t) + extra


# =============================================================================
# Loader factory
# =============================================================================

def make_loaders(
    dataset_dir: str,
    num_episodes: int = 0,
    camera_names: Sequence[str] = ("cam0",),
    obs_mode: str = "single_cam",
    batch_size_train: int = 12,
    batch_size_val: int = 12,
    seq_len_train: int = 200,
    seq_len_val: int = 200,
    seed: int = 0,
    samples_per_episode: int = 50,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    return_force_history: bool = True,
    use_force_history: bool = True,
    force_history_len: int = 10,
    qpos_norm_mode: str = "minmax_m11",
    action_norm_mode: str = "minmax_m11",
    marker_norm_mode: str = "minmax_m11",
    marker_dim: int = 7,
    include_gripper: bool = False,
    use_stain_mask: bool = False,
    stain_mask_key: str = "observations/images/stain_mask",
    stain_mask_threshold: float = 0.5,
):
    paths = _episode_files(dataset_dir, num_episodes=num_episodes)
    n = len(paths)
    if n == 1:
        train_paths = paths
        val_paths = paths
    else:
        rng = np.random.default_rng(seed)
        order = np.arange(n)
        rng.shuffle(order)
        split = max(1, int(round(0.9 * n)))
        split = min(split, n - 1)
        train_paths = [paths[i] for i in order[:split]]
        val_paths = [paths[i] for i in order[split:]]

    stats = compute_dataset_stats(
        train_paths,
        marker_dim=marker_dim,
        qpos_norm_mode=qpos_norm_mode,
        action_norm_mode=action_norm_mode,
        marker_norm_mode=marker_norm_mode,
        include_gripper=include_gripper,
    )
    stats["dataset_dir"] = str(Path(dataset_dir).expanduser())
    stats["camera_names"] = list(camera_names)
    stats["obs_mode"] = str(obs_mode)
    stats["use_stain_mask"] = bool(use_stain_mask)
    stats["stain_mask_key"] = str(stain_mask_key or "observations/images/stain_mask")
    stats["stain_mask_threshold"] = float(stain_mask_threshold)

    common = dict(
        stats=stats,
        camera_names=list(camera_names),
        obs_mode=obs_mode,
        samples_per_episode=samples_per_episode,
        return_force_history=return_force_history and use_force_history,
        force_history_len=force_history_len,
        marker_dim=marker_dim,
        qpos_norm_mode=qpos_norm_mode,
        action_norm_mode=action_norm_mode,
        marker_norm_mode=marker_norm_mode,
        include_gripper=include_gripper,
        use_stain_mask=use_stain_mask,
        stain_mask_key=stain_mask_key,
    )
    train_ds = ImitationEpisodeDataset(train_paths, seq_len=seq_len_train, seed=seed, **common)
    val_ds = ImitationEpisodeDataset(val_paths, seq_len=seq_len_val, seed=seed + 12345, **common)

    loader_kwargs = dict(
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers) if int(num_workers) > 0 else False,
    )
    if int(num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size_val, shuffle=False, drop_last=False, **loader_kwargs)

    meta = {
        "num_episodes_total": n,
        "num_train_episodes": len(train_paths),
        "num_val_episodes": len(val_paths),
        "num_train_samples": len(train_ds),
        "num_val_samples": len(val_ds),
        "camera_names": list(camera_names),
        "obs_mode": obs_mode,
        "marker_dim": int(marker_dim),
        "include_gripper": bool(include_gripper),
        "action_dim": int(stats["action_min"].shape[0]),
        "use_stain_mask": bool(use_stain_mask),
        "stain_mask_key": str(stain_mask_key or "observations/images/stain_mask"),
        "stain_mask_threshold": float(stain_mask_threshold),
    }
    return train_loader, val_loader, stats, meta
