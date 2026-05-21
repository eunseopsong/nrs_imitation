#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
from torch import nn


def split_position_and_force_from_qpos(
    qpos: torch.Tensor,
    position_dim: int = 6,
    force_dim: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    qpos: (B, D) or (B, 1, D)
    returns:
        position_state: (B, position_dim)
        current_force:  (B, force_dim)
    """
    if qpos.dim() == 3:
        qpos = qpos[:, 0, :]

    if qpos.dim() != 2:
        raise ValueError(f"qpos must be 2D or 3D, got shape={tuple(qpos.shape)}")

    required_dim = position_dim + force_dim
    if qpos.size(-1) < required_dim:
        raise ValueError(
            f"qpos last dim must be >= {required_dim}, got {qpos.size(-1)}"
        )

    position_state = qpos[:, :position_dim]
    current_force = qpos[:, position_dim : position_dim + force_dim]
    return position_state, current_force


def prepare_force_history(
    force_history: Optional[torch.Tensor],
    current_force: torch.Tensor,
) -> torch.Tensor:
    """
    force_history:
        None             -> current_force.unsqueeze(1)      => (B,1,3)
        (B,3)            -> unsqueeze(1)                    => (B,1,3)
        (B,T,3)          -> 그대로 사용
        (B,1,T,3)        -> squeeze                         => (B,T,3)
    """
    if force_history is None:
        return current_force.unsqueeze(1)

    if force_history.dim() == 4:
        if force_history.size(1) != 1:
            raise ValueError(
                f"force_history 4D case expects shape (B,1,T,3), got {tuple(force_history.shape)}"
            )
        force_history = force_history[:, 0, :, :]

    if force_history.dim() == 2:
        force_history = force_history.unsqueeze(1)

    if force_history.dim() != 3:
        raise ValueError(
            f"force_history must be None, 2D, 3D, or 4D, got shape={tuple(force_history.shape)}"
        )

    if force_history.size(0) != current_force.size(0):
        raise ValueError(
            f"force_history batch mismatch: {force_history.size(0)} vs {current_force.size(0)}"
        )

    if force_history.size(-1) != current_force.size(-1):
        raise ValueError(
            f"force_history feature dim mismatch: {force_history.size(-1)} vs {current_force.size(-1)}"
        )

    return force_history


def prepare_gripper_scalar(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Accepts gripper scalars as:
      (B,), (B,1), or (B,1,1)
    Returns:
      (B,1) float tensor
    """
    if x.dim() == 3:
        if x.size(1) != 1:
            raise ValueError(f"{name} 3D case expects shape (B,1,1), got {tuple(x.shape)}")
        x = x[:, 0, :]
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    if x.dim() != 2 or x.size(-1) != 1:
        raise ValueError(f"{name} must be (B,), (B,1), or (B,1,1), got {tuple(x.shape)}")
    return x.float()


def _make_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "mish":
        return nn.Mish()
    raise ValueError(f"Unsupported activation: {name}")


def _make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    activation: str = "mish",
) -> nn.Sequential:
    act = _make_activation(activation)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        act,
        nn.Linear(hidden_dim, output_dim),
        _make_activation(activation),
    )


class PositionStateEncoder(nn.Module):
    """
    position encoder
    input : (B, 6)   -> [x, y, z, wx, wy, wz]
    output: (B, output_dim)
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 256,
        activation: str = "gelu",
    ):
        super().__init__()
        act = _make_activation(activation)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, position_state: torch.Tensor) -> torch.Tensor:
        return self.network(position_state)


class ForceHistoryGRUEncoder(nn.Module):
    """
    force encoder (GRU)
    input : (B, T, 3) -> force history
    output: (B, output_dim)
    """

    def __init__(
        self,
        input_dim: int = 3,
        gru_hidden_dim: int = 64,
        gru_num_layers: int = 1,
        gru_dropout: float = 0.0,
        output_dim: int = 256,
        activation: str = "gelu",
    ):
        super().__init__()
        act = _make_activation(activation)

        effective_dropout = gru_dropout if gru_num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(gru_hidden_dim, output_dim),
            act,
        )

    def forward(self, force_history: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(force_history)
        last_hidden = hidden[-1]
        return self.output_proj(last_hidden)


class PositionForceFusionEncoder(nn.Module):
    """
    fusion encoder
    input :
        position_embedding: (B, Dp)
        force_embedding   : (B, Df)
    output:
        fused_embedding   : (B, output_dim)
    """

    def __init__(
        self,
        position_embed_dim: int = 256,
        force_embed_dim: int = 256,
        output_dim: int = 256,
        activation: str = "gelu",
    ):
        super().__init__()
        act = _make_activation(activation)
        self.network = nn.Sequential(
            nn.Linear(position_embed_dim + force_embed_dim, output_dim),
            act,
        )

    def forward(
        self,
        position_embedding: torch.Tensor,
        force_embedding: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat([position_embedding, force_embedding], dim=-1)
        return self.network(fused)


class PositionForceObservationEncoder(nn.Module):
    """
    combined observation encoder
    - internally splits qpos into position / force
    - if force_history is None, current force is used as length-1 history

    input:
        qpos         : (B, 9) or (B, 1, 9)
        force_history: None or (B, T, 3)

    output:
        obs_embedding: (B, output_dim)
    """

    def __init__(
        self,
        position_dim: int = 6,
        force_dim: int = 3,
        position_hidden_dim: int = 128,
        force_gru_hidden_dim: int = 64,
        force_gru_num_layers: int = 1,
        force_gru_dropout: float = 0.0,
        output_dim: int = 256,
        activation: str = "gelu",
    ):
        super().__init__()
        self.position_dim = position_dim
        self.force_dim = force_dim

        self.position_state_encoder = PositionStateEncoder(
            input_dim=position_dim,
            hidden_dim=position_hidden_dim,
            output_dim=output_dim,
            activation=activation,
        )
        self.force_history_encoder = ForceHistoryGRUEncoder(
            input_dim=force_dim,
            gru_hidden_dim=force_gru_hidden_dim,
            gru_num_layers=force_gru_num_layers,
            gru_dropout=force_gru_dropout,
            output_dim=output_dim,
            activation=activation,
        )
        self.position_force_fusion_encoder = PositionForceFusionEncoder(
            position_embed_dim=output_dim,
            force_embed_dim=output_dim,
            output_dim=output_dim,
            activation=activation,
        )

    def forward(
        self,
        qpos: torch.Tensor,
        force_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        position_state, current_force = split_position_and_force_from_qpos(
            qpos,
            position_dim=self.position_dim,
            force_dim=self.force_dim,
        )
        force_history = prepare_force_history(force_history, current_force)

        position_embedding = self.position_state_encoder(position_state)
        force_embedding = self.force_history_encoder(force_history)
        fused_embedding = self.position_force_fusion_encoder(
            position_embedding,
            force_embedding,
        )
        return fused_embedding


class GripperPositionEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        output_dim: int = 64,
        activation: str = "mish",
    ):
        super().__init__()
        self.network = _make_mlp(1, hidden_dim, output_dim, activation=activation)

    def forward(self, gripper_position: torch.Tensor) -> torch.Tensor:
        return self.network(prepare_gripper_scalar(gripper_position, "gripper_position"))


class GripperCurrentEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        output_dim: int = 64,
        activation: str = "mish",
    ):
        super().__init__()
        self.network = _make_mlp(1, hidden_dim, output_dim, activation=activation)

    def forward(self, gripper_current: torch.Tensor) -> torch.Tensor:
        return self.network(prepare_gripper_scalar(gripper_current, "gripper_current"))


class GripperStateFusionEncoder(nn.Module):
    def __init__(
        self,
        position_embed_dim: int = 64,
        current_embed_dim: int = 64,
        output_dim: int = 64,
        activation: str = "mish",
    ):
        super().__init__()
        act = _make_activation(activation)
        self.network = nn.Sequential(
            nn.Linear(position_embed_dim + current_embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            act,
        )

    def forward(
        self,
        position_embedding: torch.Tensor,
        current_embedding: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat([position_embedding, current_embedding], dim=-1)
        return self.network(fused)


class GripperObservationEncoder(nn.Module):
    """
    input:
        gripper_position : (B,), (B,1), or (B,1,1)
        gripper_current  : (B,), (B,1), or (B,1,1)
    output:
        gripper_embedding: (B, output_dim)
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        output_dim: int = 64,
        activation: str = "mish",
    ):
        super().__init__()
        self.position_encoder = GripperPositionEncoder(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation=activation,
        )
        self.current_encoder = GripperCurrentEncoder(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation=activation,
        )
        self.fusion_encoder = GripperStateFusionEncoder(
            position_embed_dim=output_dim,
            current_embed_dim=output_dim,
            output_dim=output_dim,
            activation=activation,
        )

    def forward(
        self,
        gripper_position: torch.Tensor,
        gripper_current: torch.Tensor,
    ) -> torch.Tensor:
        position_embedding = self.position_encoder(gripper_position)
        current_embedding = self.current_encoder(gripper_current)
        return self.fusion_encoder(position_embedding, current_embedding)


class ImageObservationEncoder(nn.Module):
    """
    image encoder for DETR-style ACT model
    - extracts image features + positional encodings
    - shared backbone across cameras when len(backbones) == 1
    """

    def __init__(self, backbones, hidden_dim: int, camera_names):
        super().__init__()
        self.camera_names = list(camera_names)
        self.backbones = nn.ModuleList(backbones)
        self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)

    def forward(self, image: torch.Tensor):
        if image.dim() == 6:
            image = image[:, 0, ...]

        all_cam_features = []
        all_cam_pos = []

        shared_backbone = len(self.backbones) == 1

        for cam_id, _ in enumerate(self.camera_names):
            backbone = self.backbones[0] if shared_backbone else self.backbones[cam_id]
            cam_img = image[:, cam_id]
            features, pos = backbone(cam_img)

            features = features[0]
            pos = pos[0]

            proj_feat = self.input_proj(features)
            all_cam_features.append(proj_feat)
            all_cam_pos.append(pos)

        src = torch.cat(all_cam_features, dim=3)
        pos = torch.cat(all_cam_pos, dim=3)
        return src, pos


class CNNMLPImageEncoder(nn.Module):
    """
    image encoder for CNNMLP policy
    """

    def __init__(self, backbones, camera_names):
        super().__init__()
        self.camera_names = list(camera_names)
        self.backbones = nn.ModuleList(backbones)

        backbone_down_projs = []
        for backbone in backbones:
            down_proj = nn.Sequential(
                nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                nn.Conv2d(128, 64, kernel_size=5),
                nn.Conv2d(64, 32, kernel_size=5),
            )
            backbone_down_projs.append(down_proj)
        self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

        self.output_dim = 768 * len(backbones)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 6:
            image = image[:, 0, ...]

        bs = image.size(0)
        all_cam_features = []

        for cam_id, _ in enumerate(self.camera_names):
            features, _ = self.backbones[cam_id](image[:, cam_id])
            features = features[0]
            cam_feat = self.backbone_down_projs[cam_id](features)
            all_cam_features.append(cam_feat.reshape([bs, -1]))

        return torch.cat(all_cam_features, dim=1)
