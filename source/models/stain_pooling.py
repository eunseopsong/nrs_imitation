#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def prepare_stain_mask(
    stain_mask: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Return a float binary mask with shape (B,1,H,W)."""
    if stain_mask is None:
        raise RuntimeError("stain_mask is required when use_stain_mask=True")

    mask = stain_mask.float()
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() == 5 and mask.size(1) == 1:
        mask = mask[:, 0]

    if mask.dim() != 4:
        raise RuntimeError(f"stain_mask must be (B,1,H,W) or (B,H,W), got {tuple(stain_mask.shape)}")
    if mask.size(1) != 1:
        raise RuntimeError(f"stain_mask channel dim must be 1, got {tuple(mask.shape)}")

    if torch.isfinite(mask).all() and float(mask.detach().max().item()) > 1.5:
        mask = mask / 255.0
    mask = mask.clamp(0.0, 1.0)
    return (mask >= float(threshold)).to(dtype=mask.dtype)


def masked_mean_pool_feature_map(
    feature_map: torch.Tensor,
    stain_mask: torch.Tensor,
    *,
    threshold: float = 0.5,
    empty_mode: str = "zero",
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mask-weighted mean pooling over a spatial RGB feature map.

    feature_map: (B,C,Hf,Wf)
    stain_mask : (B,1,H,W) or (B,H,W)
    returns    : pooled (B,C), resized mask (B,1,Hf,Wf), mask_sum (B,1)
    """
    if feature_map.dim() != 4:
        raise RuntimeError(f"feature_map must be (B,C,H,W), got {tuple(feature_map.shape)}")

    mode = str(empty_mode).strip().lower()
    if mode not in ("zero", "global"):
        raise ValueError(f"empty_stain_feature_mode must be zero or global, got: {empty_mode}")

    mask = prepare_stain_mask(stain_mask, threshold=threshold).to(device=feature_map.device, dtype=feature_map.dtype)
    mask_small = F.interpolate(mask, size=feature_map.shape[-2:], mode="nearest")
    mask_sum = mask_small.sum(dim=(2, 3))

    pooled = (feature_map * mask_small).sum(dim=(2, 3)) / mask_sum.clamp_min(float(eps))
    empty = mask_sum < float(eps)
    if empty.any():
        if mode == "global":
            fallback = feature_map.mean(dim=(2, 3))
            pooled = torch.where(empty, fallback, pooled)
        else:
            pooled = torch.where(empty, torch.zeros_like(pooled), pooled)

    return pooled, mask_small, mask_sum


def stain_pooling_debug_stats(
    *,
    rgb: torch.Tensor,
    stain_mask: torch.Tensor,
    feature_map: torch.Tensor,
    resized_mask: torch.Tensor,
    global_feature: torch.Tensor,
    stain_feature: torch.Tensor,
    image_feature: torch.Tensor,
    mask_sum: torch.Tensor,
) -> Dict[str, object]:
    mask = stain_mask.detach().float()
    ms = mask_sum.detach().float()
    return {
        "rgb_shape": tuple(rgb.shape),
        "stain_mask_shape": tuple(stain_mask.shape),
        "feature_map_shape": tuple(feature_map.shape),
        "resized_mask_shape": tuple(resized_mask.shape),
        "global_feature_shape": tuple(global_feature.shape),
        "stain_feature_shape": tuple(stain_feature.shape),
        "final_image_feature_shape": tuple(image_feature.shape),
        "stain_mask_min": float(mask.min().item()),
        "stain_mask_max": float(mask.max().item()),
        "stain_mask_mean": float(mask.mean().item()),
        "mask_sum_min": float(ms.min().item()),
        "mask_sum_max": float(ms.max().item()),
        "mask_sum_mean": float(ms.mean().item()),
    }


def save_stain_pooling_debug_images(
    *,
    rgb: torch.Tensor,
    stain_mask: torch.Tensor,
    out_dir: str = "debug/stain_pooling",
    prefix: str = "stain",
) -> None:
    try:
        from PIL import Image
    except Exception:
        return

    os.makedirs(out_dir, exist_ok=True)

    r = rgb.detach().float().cpu()
    if r.dim() == 4:
        r = r[0]
    if r.dim() != 3:
        return
    if r.shape[0] == 3:
        r = r.permute(1, 2, 0)
    if r.shape[-1] != 3:
        return

    # Encoders receive ImageNet-normalized RGB. Undo it when values look normalized.
    if float(r.min().item()) < -0.05 or float(r.max().item()) > 1.5:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=r.dtype).view(1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=r.dtype).view(1, 1, 3)
        r = r * std + mean
    rgb_np = (r.clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)

    m = stain_mask.detach().float().cpu()
    if m.dim() == 4:
        m = m[0]
    if m.dim() == 3:
        m = m[0]
    if m.dim() != 2:
        return
    if float(m.max().item()) > 1.5:
        m = m / 255.0
    mask_np = (m.clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)

    if mask_np.shape[:2] != rgb_np.shape[:2]:
        mask_t = torch.from_numpy(mask_np).float().view(1, 1, *mask_np.shape) / 255.0
        mask_t = F.interpolate(mask_t, size=rgb_np.shape[:2], mode="nearest")
        mask_np = (mask_t[0, 0].numpy() * 255.0).astype(np.uint8)

    tint = np.zeros_like(rgb_np)
    tint[..., 0] = 255
    tint[..., 1] = 220
    alpha = (mask_np.astype(np.float32) / 255.0)[..., None] * 0.45
    overlay = (rgb_np.astype(np.float32) * (1.0 - alpha) + tint.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)

    Image.fromarray(rgb_np).save(os.path.join(out_dir, f"{prefix}_rgb_000.png"))
    Image.fromarray(mask_np).save(os.path.join(out_dir, f"{prefix}_stain_mask_000.png"))
    Image.fromarray(overlay).save(os.path.join(out_dir, f"{prefix}_stain_overlay_000.png"))
