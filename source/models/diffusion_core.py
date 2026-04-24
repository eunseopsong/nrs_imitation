#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
diffusion_core.py

Diffusion-policy style visuomotor policy core for the current nrs_act codebase.

Design goals:
- Keep existing dataset format:
    observation = position(6) + force(3) + image(s)
    action      = position(6) + force(3)
- Reuse existing modular components when possible:
    * backbone.py
    * encoder.py
- Keep train/eval entrypoint logic thin and delegated to source/

This implementation follows the core idea of action diffusion:
  noisy action chunk + observation conditioning -> predict noise
and uses iterative denoising at inference time.

It is intentionally lightweight and easy to integrate into the current ACT-based codebase.
"""

from types import SimpleNamespace
from typing import Optional

import math
import torch
from torch import nn

from .backbone import build_backbone
from .encoder import PositionForceObservationEncoder


def _build_args(args_override: dict) -> SimpleNamespace:
    defaults = dict(
        # optimizer
        lr=1e-4,
        lr_backbone=1e-5,
        weight_decay=1e-4,
        # backbone
        backbone="resnet18",
        dilation=False,
        position_embedding="sine",
        camera_names=["cam0"],
        pretrained_backbone=True,
        # general dims
        hidden_dim=512,
        dim_feedforward=3200,
        dropout=0.1,
        nheads=8,
        enc_layers=4,
        dec_layers=7,
        pre_norm=False,
        masks=False,
        state_dim=9,
        action_dim=9,
        position_dim=6,
        force_dim=3,
        # observation encoder
        position_encoder_hidden_dim=128,
        force_encoder_hidden_dim=64,
        force_encoder_num_layers=1,
        force_encoder_dropout=0.0,
        observation_encoder_activation="gelu",
        # diffusion-specific
        num_queries=200,
        diffusion_train_steps=100,
        diffusion_infer_steps=10,
        diffusion_beta_start=1e-4,
        diffusion_beta_end=2e-2,
        diffusion_loss_type="mse",
        image_resize_hw=256,
        image_pool_hw=4,
    )
    defaults.update(args_override)
    return SimpleNamespace(**defaults)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        timesteps: (B,) int or float
        returns:   (B, dim)
        """
        if timesteps.dim() == 0:
            timesteps = timesteps[None]
        timesteps = timesteps.float()
        half = self.dim // 2
        device = timesteps.device

        emb_scale = math.log(10000.0) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * (-emb_scale))
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class ImageConditionEncoder(nn.Module):
    """
    Reuses backbone.py and produces one global image-conditioning vector per batch item.
    """

    def __init__(self, backbones, camera_names, hidden_dim: int):
        super().__init__()
        self.camera_names = list(camera_names)
        self.backbones = nn.ModuleList(backbones)

        shared_num_channels = backbones[0].num_channels
        self.cam_proj = nn.ModuleList([
            nn.Linear(shared_num_channels, hidden_dim) for _ in self.camera_names
        ])
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * len(self.camera_names), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: (B,K,3,H,W)
        returns: (B, hidden_dim)
        """
        if image.dim() == 6:
            image = image[:, 0, ...]
        if image.dim() != 5:
            raise ValueError(f"image must be (B,K,3,H,W), got {tuple(image.shape)}")

        B = image.size(0)
        feats = []
        shared = len(self.backbones) == 1

        for cam_id, _ in enumerate(self.camera_names):
            backbone = self.backbones[0] if shared else self.backbones[cam_id]
            xs, _ = backbone(image[:, cam_id])
            x = xs[0]                            # (B,C,H,W)
            x = x.mean(dim=(-2, -1))            # GAP -> (B,C)
            x = self.cam_proj[cam_id](x)        # (B,H)
            feats.append(x)

        fused = torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]
        if fused.dim() == 1:
            fused = fused.view(B, -1)
        return self.fuse(fused)


class DiffusionTransformerCore(nn.Module):
    """
    Diffusion-policy style conditional transformer denoiser.

    Conditioning:
      - PositionForceObservationEncoder for qpos / force_history
      - ImageConditionEncoder for camera input
      - timestep embedding
    Target:
      - predict Gaussian noise on future action chunk
    """

    def __init__(
        self,
        backbones,
        camera_names,
        hidden_dim: int,
        dim_feedforward: int,
        nheads: int,
        num_layers: int,
        action_dim: int,
        horizon: int,
        position_dim: int = 6,
        force_dim: int = 3,
        position_encoder_hidden_dim: int = 128,
        force_encoder_hidden_dim: int = 64,
        force_encoder_num_layers: int = 1,
        force_encoder_dropout: float = 0.0,
        observation_encoder_activation: str = "gelu",
        diffusion_train_steps: int = 100,
        diffusion_infer_steps: int = 10,
        diffusion_beta_start: float = 1e-4,
        diffusion_beta_end: float = 2e-2,
    ):
        super().__init__()
        self.camera_names = list(camera_names)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.diffusion_train_steps = int(diffusion_train_steps)
        self.diffusion_infer_steps = int(diffusion_infer_steps)

        self.observation_encoder = PositionForceObservationEncoder(
            position_dim=position_dim,
            force_dim=force_dim,
            position_hidden_dim=position_encoder_hidden_dim,
            force_gru_hidden_dim=force_encoder_hidden_dim,
            force_gru_num_layers=force_encoder_num_layers,
            force_gru_dropout=force_encoder_dropout,
            output_dim=hidden_dim,
            activation=observation_encoder_activation,
        )
        self.image_encoder = ImageConditionEncoder(
            backbones=backbones,
            camera_names=camera_names,
            hidden_dim=hidden_dim,
        )
        self.cond_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.action_in = nn.Linear(action_dim, hidden_dim)
        self.action_out = nn.Linear(hidden_dim, action_dim)
        self.seq_pos = nn.Embedding(horizon, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_dim)

        betas = torch.linspace(
            float(diffusion_beta_start),
            float(diffusion_beta_end),
            self.diffusion_train_steps,
            dtype=torch.float32,
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def _global_condition(self, qpos, image, force_history=None) -> torch.Tensor:
        obs_cond = self.observation_encoder(qpos=qpos, force_history=force_history)
        img_cond = self.image_encoder(image)
        return self.cond_fuse(torch.cat([obs_cond, img_cond], dim=-1))

    def denoise(self, noisy_actions, timesteps, qpos, image, force_history=None):
        """
        noisy_actions: (B,T,A)
        timesteps:     (B,)
        returns:       predicted noise (B,T,A)
        """
        B, T, A = noisy_actions.shape
        if A != self.action_dim:
            raise ValueError(f"action dim mismatch: {A} vs expected {self.action_dim}")
        if T > self.horizon:
            raise ValueError(f"sequence length {T} exceeds configured horizon {self.horizon}")

        cond = self._global_condition(qpos, image, force_history=force_history)  # (B,H)
        t_embed = self.time_embed(timesteps)                                     # (B,H)

        x = self.action_in(noisy_actions)                                        # (B,T,H)

        pos_ids = torch.arange(T, device=noisy_actions.device)
        pos = self.seq_pos(pos_ids).unsqueeze(0).expand(B, T, self.hidden_dim)   # (B,T,H)

        cond_tok = cond.unsqueeze(1).expand(B, T, self.hidden_dim)
        time_tok = t_embed.unsqueeze(1).expand(B, T, self.hidden_dim)

        x = x + pos + cond_tok + time_tok
        x = self.transformer(x)
        x = self.final_norm(x)
        return self.action_out(x)

    def diffusion_loss(self, qpos, image, actions, force_history=None, is_pad=None):
        """
        actions: (B,T,A)
        is_pad:  (B,T)
        """
        B, T, A = actions.shape
        device = actions.device

        t = torch.randint(
            low=0,
            high=self.diffusion_train_steps,
            size=(B,),
            device=device,
            dtype=torch.long,
        )

        noise = torch.randn_like(actions)
        sqrt_ab = self.sqrt_alpha_bars[t].view(B, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(B, 1, 1)
        noisy_actions = sqrt_ab * actions + sqrt_1mab * noise

        pred_noise = self.denoise(
            noisy_actions=noisy_actions,
            timesteps=t,
            qpos=qpos,
            image=image,
            force_history=force_history,
        )

        if is_pad is None:
            valid_mask = torch.ones((B, T, 1), dtype=torch.float32, device=device)
        else:
            valid_mask = (~is_pad).unsqueeze(-1).float()

        return pred_noise, noise, valid_mask

    @torch.no_grad()
    def sample_actions(
        self,
        qpos,
        image,
        force_history=None,
        horizon: Optional[int] = None,
        inference_steps: Optional[int] = None,
    ):
        """
        Deterministic DDIM-style sampling.
        returns: (B,T,A)
        """
        device = qpos.device
        B = qpos.shape[0]
        T = int(horizon if horizon is not None else self.horizon)
        inference_steps = int(inference_steps if inference_steps is not None else self.diffusion_infer_steps)
        inference_steps = max(1, min(inference_steps, self.diffusion_train_steps))

        x = torch.randn(B, T, self.action_dim, device=device, dtype=torch.float32)

        ts = torch.linspace(
            self.diffusion_train_steps - 1,
            0,
            inference_steps,
            device=device,
            dtype=torch.long,
        )

        prev_t = None
        for idx, t_scalar in enumerate(ts):
            t = torch.full((B,), int(t_scalar.item()), device=device, dtype=torch.long)
            eps = self.denoise(
                noisy_actions=x,
                timesteps=t,
                qpos=qpos,
                image=image,
                force_history=force_history,
            )

            alpha_bar_t = self.alpha_bars[t].view(B, 1, 1)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

            x0 = (x - sqrt_one_minus_alpha_bar_t * eps) / torch.clamp(sqrt_alpha_bar_t, min=1e-6)

            if idx == len(ts) - 1:
                x = x0
                break

            next_t_scalar = ts[idx + 1]
            next_t = torch.full((B,), int(next_t_scalar.item()), device=device, dtype=torch.long)
            alpha_bar_next = self.alpha_bars[next_t].view(B, 1, 1)

            # deterministic DDIM update
            x = torch.sqrt(alpha_bar_next) * x0 + torch.sqrt(1.0 - alpha_bar_next) * eps
            prev_t = next_t

        return x


def build_diffusion_model(args):
    backbones = [build_backbone(args) for _ in args.camera_names]

    model = DiffusionTransformerCore(
        backbones=backbones,
        camera_names=args.camera_names,
        hidden_dim=args.hidden_dim,
        dim_feedforward=args.dim_feedforward,
        nheads=args.nheads,
        num_layers=args.enc_layers,
        action_dim=getattr(args, "action_dim", 9),
        horizon=getattr(args, "num_queries", 200),
        position_dim=getattr(args, "position_dim", 6),
        force_dim=getattr(args, "force_dim", 3),
        position_encoder_hidden_dim=getattr(args, "position_encoder_hidden_dim", 128),
        force_encoder_hidden_dim=getattr(args, "force_encoder_hidden_dim", 64),
        force_encoder_num_layers=getattr(args, "force_encoder_num_layers", 1),
        force_encoder_dropout=getattr(args, "force_encoder_dropout", 0.0),
        observation_encoder_activation=getattr(args, "observation_encoder_activation", "gelu"),
        diffusion_train_steps=getattr(args, "diffusion_train_steps", 100),
        diffusion_infer_steps=getattr(args, "diffusion_infer_steps", 10),
        diffusion_beta_start=getattr(args, "diffusion_beta_start", 1e-4),
        diffusion_beta_end=getattr(args, "diffusion_beta_end", 2e-2),
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_params / 1e6,))
    return model


def build_Diffusion_model_and_optimizer(args_override):
    args = _build_args(args_override)
    model = build_diffusion_model(args)

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbones" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbones" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer
