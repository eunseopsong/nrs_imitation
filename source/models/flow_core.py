#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
source/models/flow_core.py

Flow Matching policy core for nrs_imitation.

Supported observation modes:
  - single_cam       : cam0 + qpos + optional force_history
  - dual_cam         : cam0/cam1 + qpos + optional force_history
  - dual_cam_marker  : cam0/cam1 + marker + qpos + optional force_history

Action remains the original 9D ACT-compatible command chunk:
  [x, y, z, wx, wy, wz, fx, fy, fz]

The class name FlowRGBPolicy is preserved for backward compatibility with the
existing inference/training code.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from .stain_pooling import (
    masked_mean_pool_feature_map,
    save_stain_pooling_debug_images,
    stain_pooling_debug_stats,
)

try:
    from torchvision.models import resnet18, ResNet18_Weights
except Exception:
    from torchvision.models import resnet18  # type: ignore
    ResNet18_Weights = None  # type: ignore


# =============================================================================
# Utilities
# =============================================================================

def _parse_down_dims(v) -> Tuple[int, ...]:
    if isinstance(v, str):
        return tuple(int(x.strip()) for x in v.split(",") if x.strip())
    if isinstance(v, Iterable):
        return tuple(int(x) for x in v)
    raise TypeError(f"Unsupported down_dims type: {type(v)}")


def _make_group_count(channels: int, requested_groups: int) -> int:
    g = min(int(requested_groups), int(channels))
    while g > 1 and channels % g != 0:
        g -= 1
    return max(1, g)


def _mish_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.Mish(),
        nn.Linear(hidden_dim, out_dim),
        nn.Mish(),
    )


# =============================================================================
# 1D conditional U-Net blocks
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        if half <= 0:
            return t[:, None]
        scale = math.log(10000.0) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -scale)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class Conv1dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        g = _make_group_count(out_ch, n_groups)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(g, out_ch),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Downsample1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
        ])
        self.cond_predict_scale = bool(cond_predict_scale)
        self.out_channels = int(out_channels)
        cond_out = out_channels * 2 if self.cond_predict_scale else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_out))
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        emb = self.cond_encoder(cond)
        if self.cond_predict_scale:
            emb = emb.view(emb.shape[0], 2, self.out_channels, 1)
            scale = emb[:, 0]
            bias = emb[:, 1]
            out = scale * out + bias
        else:
            out = out + emb[:, :, None]
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


# =============================================================================
# Observation encoder
# =============================================================================

class FlowRGBObservationEncoder(nn.Module):
    """
    Encodes qpos + image(s) + optional force history + optional marker into
    one global condition vector.

    qpos          : (B, state_dim)
    image         : (B, K, 3, H, W)
    force_history : optional (B, L, 3)
    marker        : optional (B, marker_dim)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)
        self.obs_mode = str(cfg.get("obs_mode", "single_cam"))
        self.camera_names = list(cfg.get("camera_names", ["cam0"]))
        self.num_cameras = max(1, len(self.camera_names))
        self.use_force_history = bool(cfg.get("use_force_history", True))
        self.use_marker = bool(cfg.get("use_marker", self.obs_mode == "dual_cam_marker"))

        state_dim = int(cfg.get("state_dim", 9))
        force_dim = int(cfg.get("force_dim", 3))
        marker_dim = int(cfg.get("marker_dim", 7))

        obs_hidden_dim = int(cfg.get("flow_obs_hidden_dim", 256))
        image_feature_dim = int(cfg.get("flow_image_feature_dim", 512))
        marker_feature_dim = int(cfg.get("flow_marker_feature_dim", 128))
        global_cond_dim = int(cfg.get("flow_global_cond_dim", 256))
        pretrained_backbone = bool(cfg.get("pretrained_backbone", True))
        self.use_stain_mask = bool(cfg.get("use_stain_mask", False))
        self.stain_pooling_type = str(cfg.get("stain_pooling_type", "masked_mean"))
        self.empty_stain_feature_mode = str(cfg.get("empty_stain_feature_mode", "zero"))
        self.stain_mask_threshold = float(cfg.get("stain_mask_threshold", 0.5))
        self.debug_stain_pooling = bool(cfg.get("debug_stain_pooling", False))
        self._debug_stain_pooling_printed = False
        if self.stain_pooling_type != "masked_mean":
            raise ValueError(f"Unsupported stain_pooling_type: {self.stain_pooling_type}")

        self.qpos_encoder = _mish_mlp(state_dim, obs_hidden_dim, obs_hidden_dim)

        if self.use_force_history:
            force_hidden_dim = int(cfg.get("force_encoder_hidden_dim", 64))
            force_num_layers = int(cfg.get("force_encoder_num_layers", 1))
            force_dropout = float(cfg.get("force_encoder_dropout", 0.0))
            self.force_gru = nn.GRU(
                input_size=force_dim,
                hidden_size=force_hidden_dim,
                num_layers=force_num_layers,
                dropout=force_dropout if force_num_layers > 1 else 0.0,
                batch_first=True,
            )
            force_out_dim = force_hidden_dim
        else:
            self.force_gru = None
            force_out_dim = 0

        if ResNet18_Weights is not None:
            weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
            backbone = resnet18(weights=weights)
        else:
            backbone = resnet18(pretrained=pretrained_backbone)
        backbone.fc = nn.Identity()
        self.image_backbone = backbone
        self.image_proj = nn.Sequential(
            nn.Linear(512, image_feature_dim),
            nn.LayerNorm(image_feature_dim),
            nn.Mish(),
        )
        image_out_dim = (self.num_cameras + (1 if self.use_stain_mask else 0)) * image_feature_dim

        if self.use_marker:
            self.marker_encoder = _mish_mlp(marker_dim, marker_feature_dim, marker_feature_dim)
            marker_out_dim = marker_feature_dim
        else:
            self.marker_encoder = None
            marker_out_dim = 0

        fuse_in = obs_hidden_dim + image_out_dim + force_out_dim + marker_out_dim
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, global_cond_dim),
            nn.LayerNorm(global_cond_dim),
            nn.Mish(),
            nn.Linear(global_cond_dim, global_cond_dim),
        )
        self.global_cond_dim = global_cond_dim
        self.marker_dim = marker_dim

    def _forward_image_features(self, x: torch.Tensor) -> torch.Tensor:
        b = self.image_backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        return x

    def _global_pool_image_features(self, feature_map: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.image_backbone.avgpool(feature_map), 1)

    def forward(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        force_history: Optional[torch.Tensor] = None,
        marker: Optional[torch.Tensor] = None,
        stain_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if qpos.dim() == 3:
            qpos = qpos[:, 0, :]
        if qpos.dim() != 2:
            raise RuntimeError(f"qpos must be (B,D), got {tuple(qpos.shape)}")

        if image.dim() != 5:
            raise RuntimeError(f"image must be (B,K,3,H,W), got {tuple(image.shape)}")
        B, K, C, H, W = image.shape
        if C != 3:
            raise RuntimeError(f"image channel dim must be 3, got {C}")
        if K != self.num_cameras:
            raise RuntimeError(f"model expected K={self.num_cameras} cameras {self.camera_names}, got image K={K}")

        q_feat = self.qpos_encoder(qpos)

        img_flat = image.reshape(B * K, C, H, W)
        feature_map = self._forward_image_features(img_flat)
        global_raw = self._global_pool_image_features(feature_map)
        global_proj = self.image_proj(global_raw).reshape(B, K, -1)
        image_feature_parts = [global_proj[:, cam_i] for cam_i in range(K)]

        if self.use_stain_mask:
            if stain_mask is None:
                raise RuntimeError("use_stain_mask=True but stain_mask was not provided to FlowRGBObservationEncoder")
            feature_map_by_cam = feature_map.reshape(B, K, feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])
            cam0_feature_map = feature_map_by_cam[:, 0]
            stain_raw, mask_small, mask_sum = masked_mean_pool_feature_map(
                cam0_feature_map,
                stain_mask,
                threshold=self.stain_mask_threshold,
                empty_mode=self.empty_stain_feature_mode,
            )
            stain_proj = self.image_proj(stain_raw)
            image_feature_parts.insert(1, stain_proj)
            if self.debug_stain_pooling and not self._debug_stain_pooling_printed:
                image_feature_debug = torch.cat([global_proj[:, 0], stain_proj], dim=-1)
                stats = stain_pooling_debug_stats(
                    rgb=image[:, 0],
                    stain_mask=stain_mask,
                    feature_map=cam0_feature_map,
                    resized_mask=mask_small,
                    global_feature=global_raw.reshape(B, K, -1)[:, 0],
                    stain_feature=stain_raw,
                    image_feature=image_feature_debug,
                    mask_sum=mask_sum,
                )
                print(f"[STAIN_POOLING][FLOW] {stats}")
                save_stain_pooling_debug_images(
                    rgb=image[:, 0],
                    stain_mask=stain_mask,
                    prefix="flow",
                )
                self._debug_stain_pooling_printed = True

        img_feat = torch.cat(image_feature_parts, dim=-1)

        feats = [q_feat, img_feat]

        if self.use_force_history:
            if force_history is None:
                # Fallback: use current force from qpos if no history is supplied.
                force_history = qpos[:, -3:].unsqueeze(1)
            if force_history.dim() == 4 and force_history.size(1) == 1:
                force_history = force_history[:, 0]
            _, h = self.force_gru(force_history)
            feats.append(h[-1])

        if self.use_marker:
            if marker is None:
                marker = torch.zeros((B, self.marker_dim), dtype=qpos.dtype, device=qpos.device)
            if marker.dim() == 3:
                marker = marker[:, 0, :]
            feats.append(self.marker_encoder(marker))

        return self.fuse(torch.cat(feats, dim=-1))


# =============================================================================
# Conditional U-Net velocity field
# =============================================================================

class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        time_embed_dim: int = 256,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()
        down_dims = list(down_dims)
        if len(down_dims) < 2:
            raise ValueError("down_dims must contain at least two levels")
        cond_dim = time_embed_dim + global_cond_dim

        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        start_dim = down_dims[0]
        self.input_proj = Conv1dBlock(input_dim, start_dim, kernel_size, n_groups)

        self.down_modules = nn.ModuleList()
        prev = start_dim
        for i, dim_out in enumerate(down_dims):
            is_last = i == len(down_dims) - 1
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(prev, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))
            prev = dim_out

        mid = down_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid, mid, cond_dim, kernel_size, n_groups, cond_predict_scale),
            ConditionalResidualBlock1D(mid, mid, cond_dim, kernel_size, n_groups, cond_predict_scale),
        ])

        up_pairs = list(reversed(list(zip(down_dims[:-1], down_dims[1:]))))
        self.up_modules = nn.ModuleList()
        for dim_out, dim_in in up_pairs:
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in * 2, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                Upsample1d(dim_out),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size, n_groups),
            nn.Conv1d(start_dim, input_dim, kernel_size=1),
        )

    def forward(self, sample: torch.Tensor, t: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        x = sample.moveaxis(-1, -2)  # (B,C,T)
        x = self.input_proj(x)
        cond = torch.cat([self.time_encoder(t), global_cond], dim=-1)

        skips = []
        for res1, res2, down in self.down_modules:
            x = res1(x, cond)
            x = res2(x, cond)
            skips.append(x)
            x = down(x)

        for mid in self.mid_modules:
            x = mid(x, cond)

        for res1, res2, up in self.up_modules:
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                m = min(x.shape[-1], skip.shape[-1])
                x = x[..., :m]
                skip = skip[..., :m]
            x = torch.cat([x, skip], dim=1)
            x = res1(x, cond)
            x = res2(x, cond)
            x = up(x)

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)


# =============================================================================
# Policy
# =============================================================================

class FlowRGBPolicy(nn.Module):
    """Backward-compatible Flow policy class name."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)
        self.num_queries = int(cfg.get("num_queries", 200))
        self.action_dim = int(cfg.get("action_dim", 9))
        self.flow_train_eps = float(cfg.get("flow_train_eps", 1e-4))
        self.flow_infer_steps = int(cfg.get("flow_infer_steps", 10))
        self.flow_loss_type = str(cfg.get("flow_loss_type", "mse")).lower()

        self.obs_encoder = FlowRGBObservationEncoder(cfg)
        self.velocity_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_encoder.global_cond_dim,
            time_embed_dim=int(cfg.get("flow_time_embed_dim", 256)),
            down_dims=_parse_down_dims(cfg.get("flow_down_dims", "256,512,1024")),
            kernel_size=int(cfg.get("flow_kernel_size", 5)),
            n_groups=int(cfg.get("flow_n_groups", 8)),
            cond_predict_scale=bool(cfg.get("flow_cond_predict_scale", False)),
        )
        self.image_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        B, K, C, H, W = image.shape
        flat = image.reshape(B * K, C, H, W)
        flat = self.image_normalize(flat)
        return flat.reshape(B, K, C, H, W)

    def _condition(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        force_history: Optional[torch.Tensor] = None,
        marker: Optional[torch.Tensor] = None,
        stain_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.obs_encoder(
            qpos=qpos,
            image=self._normalize_image(image),
            force_history=force_history,
            marker=marker,
            stain_mask=stain_mask,
        )

    def predict_velocity(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        qpos: torch.Tensor,
        image: torch.Tensor,
        force_history: Optional[torch.Tensor] = None,
        marker: Optional[torch.Tensor] = None,
        stain_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond = self._condition(
            qpos=qpos,
            image=image,
            force_history=force_history,
            marker=marker,
            stain_mask=stain_mask,
        )
        return self.velocity_net(sample=z_t, t=t, global_cond=cond)

    def _masked_loss(self, pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
        per = torch.abs(pred - target) if self.flow_loss_type == "l1" else (pred - target) ** 2
        valid = (~is_pad).unsqueeze(-1).float()
        denom = valid.sum().clamp_min(1.0) * pred.shape[-1]
        return (per * valid).sum() / denom

    def forward(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
        force_history: Optional[torch.Tensor] = None,
        marker: Optional[torch.Tensor] = None,
        stain_mask: Optional[torch.Tensor] = None,
    ):
        if actions is not None:
            assert is_pad is not None, "is_pad is required for training"
            z1 = actions[:, : self.num_queries]
            is_pad = is_pad[:, : self.num_queries]
            B = z1.shape[0]
            z0 = torch.randn_like(z1)
            eps = self.flow_train_eps
            t = torch.rand(B, device=z1.device, dtype=z1.dtype) * (1.0 - 2.0 * eps) + eps
            z_t = (1.0 - t.view(B, 1, 1)) * z0 + t.view(B, 1, 1) * z1
            target_v = z1 - z0
            pred_v = self.predict_velocity(z_t, t, qpos, image, force_history, marker, stain_mask)
            loss = self._masked_loss(pred_v, target_v, is_pad)
            return {"flow": loss, "loss": loss}

        return self.sample_action(qpos=qpos, image=image, force_history=force_history, marker=marker, stain_mask=stain_mask)

    def _sample_action_impl(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        force_history: Optional[torch.Tensor] = None,
        marker: Optional[torch.Tensor] = None,
        stain_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        steps = max(1, int(num_steps or self.flow_infer_steps))
        B = qpos.shape[0]
        z = torch.randn(B, self.num_queries, self.action_dim, device=qpos.device, dtype=qpos.dtype)
        dt = 1.0 / float(steps)
        for k in range(steps):
            t = torch.full((B,), (k + 0.5) / float(steps), device=qpos.device, dtype=qpos.dtype)
            v = self.predict_velocity(z, t, qpos, image, force_history, marker, stain_mask)
            z = z + dt * v
        return z

    @torch.no_grad()
    def sample_action(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        force_history: Optional[torch.Tensor] = None,
        marker: Optional[torch.Tensor] = None,
        stain_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        return self._sample_action_impl(
            qpos=qpos,
            image=image,
            force_history=force_history,
            marker=marker,
            stain_mask=stain_mask,
            num_steps=num_steps,
        )

    def sample_action_with_grad(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        force_history: Optional[torch.Tensor] = None,
        marker: Optional[torch.Tensor] = None,
        stain_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        return self._sample_action_impl(
            qpos=qpos,
            image=image,
            force_history=force_history,
            marker=marker,
            stain_mask=stain_mask,
            num_steps=num_steps,
        )


def build_flow_rgb_policy_and_optimizer(cfg: dict):
    model = FlowRGBPolicy(cfg)
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-6))
    beta1 = float(cfg.get("beta1", 0.95))
    beta2 = float(cfg.get("beta2", 0.999))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    return model, optimizer
