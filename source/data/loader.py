# -*- coding: utf-8 -*-
"""
source/data/loader.py

DataLoader factory for ACT / FLOW / Diffusion branches.

This version adds normalization mode support:
    --norm_mode minmax_01   -> [0, 1]
    --norm_mode minmax_m11  -> [-1, 1]

The raw HDF5 dataset is unchanged. Only dataset-side qpos/action normalization
and inference-side denormalization depend on dataset_stats.pkl.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from torch.utils.data import DataLoader

from .dataset import EpisodicDataset, read_qpos_action_arrays, natural_key
from .normalization import canonical_norm_mode, norm_range_for_mode


def _episode_files(dataset_dir: str, num_episodes: int = 0) -> List[str]:
    d = Path(dataset_dir).expanduser()
    if not d.is_dir():
        raise FileNotFoundError(f"dataset_dir does not exist: {d}")

    files = sorted(d.glob("episode_*.hdf5"), key=natural_key)
    if num_episodes and int(num_episodes) > 0:
        files = files[: int(num_episodes)]

    if len(files) == 0:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {d}")

    return [str(p) for p in files]


def compute_dataset_stats(
    episode_files: Sequence[str],
    qpos_norm_mode: str = "minmax_01",
    action_norm_mode: str = "minmax_01",
) -> Dict:
    qpos_list = []
    action_list = []

    for p in episode_files:
        qpos, action = read_qpos_action_arrays(p)
        qpos_list.append(qpos.astype(np.float32))
        action_list.append(action.astype(np.float32))

    qpos_all = np.concatenate(qpos_list, axis=0)
    action_all = np.concatenate(action_list, axis=0)

    qpos_min = qpos_all.min(axis=0).astype(np.float32)
    qpos_max = qpos_all.max(axis=0).astype(np.float32)
    action_min = action_all.min(axis=0).astype(np.float32)
    action_max = action_all.max(axis=0).astype(np.float32)

    # protect constant dims, especially fx/fy sometimes intentionally zeroed
    eps = np.float32(1e-6)
    qpos_max = np.maximum(qpos_max, qpos_min + eps).astype(np.float32)
    action_max = np.maximum(action_max, action_min + eps).astype(np.float32)

    qpos_norm_mode = canonical_norm_mode(qpos_norm_mode)
    action_norm_mode = canonical_norm_mode(action_norm_mode)

    stats = {
        # current min-max stats
        "qpos_min": qpos_min,
        "qpos_max": qpos_max,
        "action_min": action_min,
        "action_max": action_max,

        # also keep mean/std for debug/backward compatibility
        "qpos_mean": qpos_all.mean(axis=0).astype(np.float32),
        "qpos_std": np.maximum(qpos_all.std(axis=0).astype(np.float32), eps),
        "action_mean": action_all.mean(axis=0).astype(np.float32),
        "action_std": np.maximum(action_all.std(axis=0).astype(np.float32), eps),

        # explicit modes for inference
        "qpos_norm_mode": qpos_norm_mode,
        "action_norm_mode": action_norm_mode,
        "qpos_mode": qpos_norm_mode,
        "act_mode": action_norm_mode,
        "norm_range": norm_range_for_mode(action_norm_mode),

        # useful metadata
        "state_dim": 9,
        "action_dim": 9,
    }
    return stats


def _make_loader(dataset, batch_size, shuffle, num_workers, pin_memory, persistent_workers, prefetch_factor):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
    )
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**kwargs)


def load_data(
    dataset_dir: str,
    num_episodes: int,
    camera_names: Sequence[str],
    batch_size_train: int,
    batch_size_val: int,
    seq_len_train: int,
    seq_len_val: int,
    seed: int = 0,
    samples_per_episode: int = 50,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,

    # force history
    return_force_history: bool = False,
    use_force_history: bool = False,
    force_history_len: int = 10,

    # normalization
    qpos_norm_mode: str = "minmax_01",
    action_norm_mode: str = "minmax_01",
):
    files = _episode_files(dataset_dir, num_episodes=num_episodes)

    rng = np.random.default_rng(seed)
    perm = np.arange(len(files))
    rng.shuffle(perm)

    n_val = max(1, int(round(0.2 * len(files)))) if len(files) > 1 else 1
    n_train = max(1, len(files) - n_val)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    if len(val_idx) == 0:
        val_idx = perm[-1:]

    train_files = [files[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]

    stats = compute_dataset_stats(
        files,
        qpos_norm_mode=qpos_norm_mode,
        action_norm_mode=action_norm_mode,
    )

    train_dataset = EpisodicDataset(
        episode_files=train_files,
        camera_names=camera_names,
        stats=stats,
        seq_len=seq_len_train,
        samples_per_episode=samples_per_episode,
        seed=seed,
        return_force_history=return_force_history,
        use_force_history=use_force_history,
        force_history_len=force_history_len,
    )

    val_dataset = EpisodicDataset(
        episode_files=val_files,
        camera_names=camera_names,
        stats=stats,
        seq_len=seq_len_val,
        samples_per_episode=samples_per_episode,
        seed=seed + 10000,
        return_force_history=return_force_history,
        use_force_history=use_force_history,
        force_history_len=force_history_len,
    )

    train_loader = _make_loader(
        train_dataset,
        batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    meta = {
        "N": len(files),
        "is_sim": False,
        "train_files": len(train_files),
        "val_files": len(val_files),
        "camera_names": list(camera_names),
        "samples_per_episode": int(samples_per_episode),
        "seq_len_train": int(seq_len_train),
        "seq_len_val": int(seq_len_val),
        "return_force_history": bool(return_force_history or use_force_history),
        "force_history_len": int(force_history_len),
        "qpos_norm_mode": canonical_norm_mode(qpos_norm_mode),
        "action_norm_mode": canonical_norm_mode(action_norm_mode),
    }

    return train_loader, val_loader, stats, meta