#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
source/models/bspline_core.py

B-spline trajectory policy for nrs_imitation, added in parallel to FLOW.

This reuses FlowRGBObservationEncoder as-is (DINOv3/ResNet backbone, TCP-ROI
patch-attention pooling, qpos/force/marker fusion) so the observation and
feature-token pipeline is identical to the FLOW policy. Only the action head
differs: instead of a flow-matching velocity field over the full action
chunk, a small MLP regresses a handful of B-spline control points, which are
then evaluated (via a fixed, non-learned basis matrix) into the full
chunk_size trajectory.

Action remains the original 9D ACT-compatible command chunk:
  [x, y, z, wx, wy, wz, fx, fy, fz]
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from .flow_core import FlowRGBObservationEncoder


# =============================================================================
# Fixed (non-learned) clamped uniform B-spline basis
# =============================================================================

def _clamped_uniform_knots(num_control_points: int, degree: int) -> np.ndarray:
    n = int(num_control_points)
    k = int(degree)
    num_interior = n - k - 1
    interior = np.linspace(0.0, 1.0, num_interior + 2)[1:-1] if num_interior > 0 else np.zeros((0,))
    return np.concatenate([np.zeros(k + 1), interior, np.ones(k + 1)]).astype(np.float64)


def _bspline_basis_matrix(num_control_points: int, degree: int, num_samples: int) -> np.ndarray:
    """Cox-de Boor recursion. Returns a (num_samples, num_control_points) basis matrix
    such that `trajectory = basis @ control_points`."""
    n, k = int(num_control_points), int(degree)
    knots = _clamped_uniform_knots(n, k)
    us = np.linspace(0.0, 1.0, int(num_samples))

    basis = np.zeros((k + 1, n + k, us.shape[0]), dtype=np.float64)
    for i in range(n + k):
        lo, hi = knots[i], knots[i + 1]
        if i == n + k - 1:
            basis[0, i] = (us >= lo) & (us <= hi)
        else:
            basis[0, i] = (us >= lo) & (us < hi)

    for d in range(1, k + 1):
        for i in range(n + k - d):
            term_a = np.zeros_like(us)
            denom_a = knots[i + d] - knots[i]
            if denom_a > 1e-12:
                term_a = (us - knots[i]) / denom_a * basis[d - 1, i]
            term_b = np.zeros_like(us)
            denom_b = knots[i + d + 1] - knots[i + 1]
            if denom_b > 1e-12:
                term_b = (knots[i + d + 1] - us) / denom_b * basis[d - 1, i + 1]
            basis[d, i] = term_a + term_b

    result = basis[k, :n].T.copy()  # (num_samples, n)
    # Force exact clamped endpoints regardless of floating-point recursion noise:
    # trajectory[0] == control_points[0], trajectory[-1] == control_points[-1].
    result[0, :] = 0.0
    result[0, 0] = 1.0
    result[-1, :] = 0.0
    result[-1, -1] = 1.0
    return result


class BSplineTrajectoryBasis(nn.Module):
    """Maps (B, num_control_points, D) control points to (B, num_samples, D)
    trajectory samples via a fixed clamped uniform B-spline basis matrix."""

    def __init__(self, num_control_points: int, num_samples: int, degree: int = 3):
        super().__init__()
        if int(num_control_points) < int(degree) + 1:
            raise ValueError(
                f"num_control_points must be >= degree+1={int(degree) + 1}, got {num_control_points}"
            )
        basis_np = _bspline_basis_matrix(num_control_points, degree, num_samples)
        self.register_buffer("basis", torch.from_numpy(basis_np).float(), persistent=False)
        self.num_control_points = int(num_control_points)
        self.num_samples = int(num_samples)
        self.degree = int(degree)

    def forward(self, control_points: torch.Tensor) -> torch.Tensor:
        if control_points.dim() != 3 or control_points.shape[1] != self.num_control_points:
            raise RuntimeError(
                f"control_points must be (B,{self.num_control_points},D), got {tuple(control_points.shape)}"
            )
        basis = self.basis.to(dtype=control_points.dtype, device=control_points.device)
        return torch.einsum("tc,bcd->btd", basis, control_points)


# =============================================================================
# Control-point regression head
# =============================================================================

class BSplineControlHead(nn.Module):
    """Regresses B-spline control points from the shared observation condition vector."""

    def __init__(self, cond_dim: int, num_control_points: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.num_control_points = int(num_control_points)
        self.action_dim = int(action_dim)
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, self.num_control_points * self.action_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        B = cond.shape[0]
        return self.net(cond).view(B, self.num_control_points, self.action_dim)


# =============================================================================
# Policy
# =============================================================================

class BSplinePolicy(nn.Module):
    """
    Deterministic B-spline trajectory policy.

    Shares FlowRGBObservationEncoder (and therefore the DINOv3/ResNet image
    backbone + TCP-ROI patch-token pooling) with FlowRGBPolicy verbatim. Only
    the action head differs: a small MLP regresses `num_control_points`
    control points, evaluated into the full `num_queries`-step trajectory via
    a fixed clamped uniform B-spline basis. No iterative sampling is needed
    at inference time (single forward pass).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)
        self.num_queries = int(cfg.get("num_queries", 200))
        self.action_dim = int(cfg.get("action_dim", 9))
        self.num_control_points = int(cfg.get("num_control_points", 16))
        self.bspline_degree = int(cfg.get("bspline_degree", 3))
        self.bspline_loss_type = str(cfg.get("bspline_loss_type", "mse")).lower()
        if self.bspline_loss_type not in ("mse", "l1"):
            raise ValueError(f"bspline_loss_type must be mse or l1, got {self.bspline_loss_type}")
        # Modality dropout: zero out qpos for a random fraction of training
        # samples so the easy qpos shortcut can't fully explain the target
        # trajectory, forcing gradient through the image pathway instead.
        # Inference (sample_action) never applies this -- qpos is always
        # passed through normally there.
        self.qpos_dropout_prob = float(cfg.get("qpos_dropout_prob", 0.0))
        if not (0.0 <= self.qpos_dropout_prob < 1.0):
            raise ValueError(f"qpos_dropout_prob must be in [0,1), got {self.qpos_dropout_prob}")
        # Additive Gaussian noise on (already minmax-normalized) qpos, applied
        # every training sample regardless of dropout -- targets the case
        # where qpos isn't just an easy shortcut but an almost noiseless one
        # (e.g. per-direction position clusters corrected to ~1-2mm std), so
        # dropout alone still leaves a near-perfect signal on the kept
        # samples. Jitter blurs that signal on every sample instead of only
        # removing it sometimes.
        self.qpos_jitter_std = float(cfg.get("qpos_jitter_std", 0.0))
        if self.qpos_jitter_std < 0.0:
            raise ValueError(f"qpos_jitter_std must be >= 0, got {self.qpos_jitter_std}")

        self.obs_encoder = FlowRGBObservationEncoder(cfg)
        self.control_head = BSplineControlHead(
            cond_dim=self.obs_encoder.global_cond_dim,
            num_control_points=self.num_control_points,
            action_dim=self.action_dim,
            hidden_dim=int(cfg.get("bspline_hidden_dim", 256)),
        )
        self.trajectory_basis = BSplineTrajectoryBasis(
            num_control_points=self.num_control_points,
            num_samples=self.num_queries,
            degree=self.bspline_degree,
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

    def predict_control_points(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        force_history: Optional[torch.Tensor] = None,
        marker: Optional[torch.Tensor] = None,
        stain_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond = self._condition(qpos, image, force_history, marker, stain_mask)
        return self.control_head(cond)

    def predict_trajectory(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        force_history: Optional[torch.Tensor] = None,
        marker: Optional[torch.Tensor] = None,
        stain_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        control_points = self.predict_control_points(qpos, image, force_history, marker, stain_mask)
        return self.trajectory_basis(control_points)

    def _masked_loss(self, pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
        per = torch.abs(pred - target) if self.bspline_loss_type == "l1" else (pred - target) ** 2
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
            target = actions[:, : self.num_queries]
            is_pad = is_pad[:, : self.num_queries]
            if self.training and self.qpos_dropout_prob > 0.0:
                keep = (torch.rand(qpos.shape[0], 1, device=qpos.device, dtype=qpos.dtype)
                        >= self.qpos_dropout_prob).to(qpos.dtype)
                qpos = qpos * keep
            pred = self.predict_trajectory(qpos, image, force_history, marker, stain_mask)
            loss = self._masked_loss(pred, target, is_pad)
            return {"bspline": loss, "loss": loss}

        return self.sample_action(qpos=qpos, image=image, force_history=force_history, marker=marker, stain_mask=stain_mask)

    @torch.no_grad()
    def sample_action(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        force_history: Optional[torch.Tensor] = None,
        marker: Optional[torch.Tensor] = None,
        stain_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.predict_trajectory(qpos, image, force_history, marker, stain_mask)


def build_bspline_policy_and_optimizer(cfg: dict):
    model = BSplinePolicy(cfg)
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-6))
    beta1 = float(cfg.get("beta1", 0.95))
    beta2 = float(cfg.get("beta2", 0.999))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    return model, optimizer
