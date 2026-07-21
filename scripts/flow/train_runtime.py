#!/usr/bin/env python3
"""Runtime helpers shared by polishing and gripper Flow training."""

from __future__ import annotations

import math
from typing import Optional

import torch


def resolve_temporal_parameters(args) -> None:
    """Resolve second-based settings into dataset row counts in-place."""
    dataset_hz = float(args.dataset_hz)
    if dataset_hz <= 0.0:
        raise ValueError(f"dataset_hz must be positive, got {dataset_hz}")

    force_history_sec = float(args.force_history_sec)
    if force_history_sec > 0.0:
        args.force_history_len = max(1, int(round(dataset_hz * force_history_sec)))

    chunk_sec = float(args.chunk_sec)
    if chunk_sec > 0.0:
        args.chunk_size = max(4, int(round(dataset_hz * chunk_sec)))

    if int(args.chunk_size) % 4 != 0:
        raise ValueError(
            f"chunk_size={args.chunk_size} must be divisible by 4 for the current Flow U-Net"
        )


def build_epoch_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    num_epochs: int,
    warmup_epochs: int,
    min_lr: float,
    base_lr: float,
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    """Build a linear-warmup, epoch-level cosine learning-rate scheduler."""
    name = str(scheduler_name or "none").strip().lower()
    if name in ("", "none", "off"):
        return None
    if name != "cosine":
        raise ValueError(f"Unsupported lr_scheduler={scheduler_name!r}")

    total = max(1, int(num_epochs))
    warmup = min(max(0, int(warmup_epochs)), total - 1)
    base = max(float(base_lr), 1e-12)
    min_ratio = min(1.0, max(0.0, float(min_lr) / base))

    def lr_factor(epoch: int) -> float:
        epoch = max(0, int(epoch))
        if warmup > 0 and epoch < warmup:
            return max(min_ratio, float(epoch + 1) / float(warmup))
        progress = float(epoch - warmup) / float(max(1, total - warmup - 1))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)


def set_train_dataset_epoch(train_loader, epoch: int) -> None:
    """Notify a dataset that a new epoch is starting, when supported."""
    setter = getattr(train_loader.dataset, "set_epoch", None)
    if callable(setter):
        setter(int(epoch))
