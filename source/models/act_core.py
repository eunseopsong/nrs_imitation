#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import SimpleNamespace

import torch
from torch import nn
from torch.autograd import Variable

from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
from .encoder import (
    PositionForceObservationEncoder,
    ImageObservationEncoder,
    CNNMLPImageEncoder,
)



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
        camera_names=[],
        pretrained_backbone=True,
        # transformer
        enc_layers=4,
        dec_layers=6,
        dim_feedforward=2048,
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        num_queries=400,
        pre_norm=False,
        # misc
        masks=False,
        # dimensions
        state_dim=9,
        action_dim=9,
        position_dim=6,
        force_dim=3,
        # new encoder configs
        position_encoder_hidden_dim=128,
        force_encoder_hidden_dim=64,
        force_encoder_num_layers=1,
        force_encoder_dropout=0.0,
        observation_encoder_activation="gelu",
        cnnmlp_observation_embed_dim=256,
    )
    defaults.update(args_override)
    return SimpleNamespace(**defaults)


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    sinusoid_table = torch.tensor(
        [
            [position / (10000 ** (2 * (hid_j // 2) / d_hid)) for hid_j in range(d_hid)]
            for position in range(n_position)
        ],
        dtype=torch.float32,
    )
    sinusoid_table[:, 0::2] = sinusoid_table[:, 0::2].sin()
    sinusoid_table[:, 1::2] = sinusoid_table[:, 1::2].cos()
    return sinusoid_table.unsqueeze(0)


class DETRVAE(nn.Module):
    """
    Default:
      obs dim    = 9  (pos6 + force3)
      action dim = 9  (pos6 + force3)

    Backward compatibility:
    - qpos only                -> uses current force as length-1 GRU history
    - qpos + force_history     -> uses provided force history
    """

    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        obs_dim,
        action_dim,
        num_queries,
        camera_names,
        position_dim=6,
        force_dim=3,
        position_encoder_hidden_dim=128,
        force_encoder_hidden_dim=64,
        force_encoder_num_layers=1,
        force_encoder_dropout=0.0,
        observation_encoder_activation="gelu",
    ):
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder

        hidden_dim = transformer.d_model

        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if backbones is not None:
            self.image_encoder = ImageObservationEncoder(
                backbones=backbones,
                hidden_dim=hidden_dim,
                camera_names=camera_names,
            )
        else:
            self.image_encoder = None

        self.transformer_observation_encoder = PositionForceObservationEncoder(
            position_dim=position_dim,
            force_dim=force_dim,
            position_hidden_dim=position_encoder_hidden_dim,
            force_gru_hidden_dim=force_encoder_hidden_dim,
            force_gru_num_layers=force_encoder_num_layers,
            force_gru_dropout=force_encoder_dropout,
            output_dim=hidden_dim,
            activation=observation_encoder_activation,
        )

        self.latent_observation_encoder = PositionForceObservationEncoder(
            position_dim=position_dim,
            force_dim=force_dim,
            position_hidden_dim=position_encoder_hidden_dim,
            force_gru_hidden_dim=force_encoder_hidden_dim,
            force_gru_num_layers=force_encoder_num_layers,
            force_gru_dropout=force_encoder_dropout,
            output_dim=hidden_dim,
            activation=observation_encoder_activation,
        )

        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)

        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)

        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer("pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim))

        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

        _ = obs_dim  # kept for backward compatibility

    def forward(
        self,
        qpos,
        image,
        env_state=None,
        actions=None,
        is_pad=None,
        force_history=None,
    ):
        if qpos.dim() == 3:
            qpos = qpos[:, 0, :]
        if image.dim() == 6:
            image = image[:, 0, ...]

        is_training = actions is not None
        bs, _ = qpos.shape

        if is_training:
            action_embed = self.encoder_action_proj(actions)

            observation_embed = self.latent_observation_encoder(
                qpos=qpos,
                force_history=force_history,
            ).unsqueeze(1)

            cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

            encoder_input = torch.cat([cls_embed, observation_embed, action_embed], dim=1)
            encoder_input = encoder_input.permute(1, 0, 2)

            cls_joint_is_pad = torch.full((bs, 2), False, device=qpos.device)
            is_pad = torch.cat([cls_joint_is_pad, is_pad], dim=1)

            pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)

            encoder_output = self.encoder(
                encoder_input,
                pos=pos_embed,
                src_key_padding_mask=is_pad,
            )
            encoder_output = encoder_output[0, :, :]

            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]

            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32, device=qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.image_encoder is not None:
            src, pos = self.image_encoder(image)

            proprio_input = self.transformer_observation_encoder(
                qpos=qpos,
                force_history=force_history,
            )

            hs = self.transformer(
                src,
                None,
                self.query_embed.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
            )[0]
        else:
            observation_embed = self.transformer_observation_encoder(
                qpos=qpos,
                force_history=force_history,
            ).unsqueeze(1)

            hs = self.transformer(
                observation_embed,
                None,
                self.query_embed.weight,
                self.additional_pos_embed.weight[:1],
            )[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]


class CNNMLP(nn.Module):
    def __init__(
        self,
        backbones,
        obs_dim,
        action_dim,
        camera_names,
        position_dim=6,
        force_dim=3,
        position_encoder_hidden_dim=128,
        force_encoder_hidden_dim=64,
        force_encoder_num_layers=1,
        force_encoder_dropout=0.0,
        observation_encoder_activation="gelu",
        observation_embed_dim=256,
    ):
        super().__init__()
        self.camera_names = camera_names

        self.image_encoder = CNNMLPImageEncoder(
            backbones=backbones,
            camera_names=camera_names,
        )

        self.observation_encoder = PositionForceObservationEncoder(
            position_dim=position_dim,
            force_dim=force_dim,
            position_hidden_dim=position_encoder_hidden_dim,
            force_gru_hidden_dim=force_encoder_hidden_dim,
            force_gru_num_layers=force_encoder_num_layers,
            force_gru_dropout=force_encoder_dropout,
            output_dim=observation_embed_dim,
            activation=observation_encoder_activation,
        )

        mlp_in_dim = self.image_encoder.output_dim + observation_embed_dim
        self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=512, output_dim=action_dim, hidden_depth=2)

        _ = obs_dim  # kept for backward compatibility

    def forward(
        self,
        qpos,
        image,
        env_state=None,
        actions=None,
        force_history=None,
    ):
        if qpos.dim() == 3:
            qpos = qpos[:, 0, :]
        if image.dim() == 6:
            image = image[:, 0, ...]

        image_embedding = self.image_encoder(image)
        observation_embedding = self.observation_encoder(
            qpos=qpos,
            force_history=force_history,
        )

        features = torch.cat([image_embedding, observation_embedding], dim=1)
        return self.mlp(features)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
    for _ in range(hidden_depth - 1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
    mods.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*mods)


def build_encoder(args):
    d_model = args.hidden_dim
    dropout = getattr(args, "dropout", 0.1)
    nhead = args.nheads
    dim_feedforward = args.dim_feedforward
    num_encoder_layers = args.enc_layers
    normalize_before = getattr(args, "pre_norm", False)
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        activation,
        normalize_before,
    )
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    return TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)


def build_ACT_model(args):
    obs_dim = getattr(args, "state_dim", 9)
    action_dim = getattr(args, "action_dim", 9)

    backbones = [build_backbone(args)]
    transformer = build_transformer(args)
    encoder = build_encoder(args)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        position_dim=getattr(args, "position_dim", 6),
        force_dim=getattr(args, "force_dim", 3),
        position_encoder_hidden_dim=getattr(args, "position_encoder_hidden_dim", 128),
        force_encoder_hidden_dim=getattr(args, "force_encoder_hidden_dim", 64),
        force_encoder_num_layers=getattr(args, "force_encoder_num_layers", 1),
        force_encoder_dropout=getattr(args, "force_encoder_dropout", 0.0),
        observation_encoder_activation=getattr(args, "observation_encoder_activation", "gelu"),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_params / 1e6,))
    return model


def build_CNNMLP_model(args):
    obs_dim = getattr(args, "state_dim", 9)
    action_dim = getattr(args, "action_dim", 9)

    backbones = [build_backbone(args) for _ in args.camera_names]
    model = CNNMLP(
        backbones,
        obs_dim=obs_dim,
        action_dim=action_dim,
        camera_names=args.camera_names,
        position_dim=getattr(args, "position_dim", 6),
        force_dim=getattr(args, "force_dim", 3),
        position_encoder_hidden_dim=getattr(args, "position_encoder_hidden_dim", 128),
        force_encoder_hidden_dim=getattr(args, "force_encoder_hidden_dim", 64),
        force_encoder_num_layers=getattr(args, "force_encoder_num_layers", 1),
        force_encoder_dropout=getattr(args, "force_encoder_dropout", 0.0),
        observation_encoder_activation=getattr(args, "observation_encoder_activation", "gelu"),
        observation_embed_dim=getattr(args, "cnnmlp_observation_embed_dim", 256),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_params / 1e6,))
    return model


def build_ACT_model_and_optimizer(args_override):
    args = _build_args(args_override)
    model = build_ACT_model(args)

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    args = _build_args(args_override)
    model = build_CNNMLP_model(args)

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer