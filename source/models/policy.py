#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from .act_core import (
    build_ACT_model_and_optimizer,
    build_CNNMLP_model_and_optimizer,
)
from .diffusion_core import build_Diffusion_model_and_optimizer


class ACTPolicy(nn.Module):
    """ACT 기반 정책 네트워크"""

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]

        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        print(f"[ACTPolicy] KL Weight = {self.kl_weight}")

    def forward(self, qpos, image, actions=None, is_pad=None, force_history=None):
        env_state = None
        image = self._normalize(image)

        if actions is not None:
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos=qpos,
                image=image,
                env_state=env_state,
                actions=actions,
                is_pad=is_pad,
                force_history=force_history,
            )

            total_kld, _, _ = kl_divergence(mu, logvar)

            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            valid_mask = (~is_pad).unsqueeze(-1).float()

            valid_count = valid_mask.sum().clamp_min(1.0) * actions.shape[-1]
            l1 = (all_l1 * valid_mask).sum() / valid_count

            loss_dict = {
                "l1": l1,
                "kl": total_kld[0],
            }
            loss_dict["loss"] = loss_dict["l1"] + self.kl_weight * loss_dict["kl"]
            return loss_dict

        a_hat, _, _ = self.model(
            qpos=qpos,
            image=image,
            env_state=env_state,
            force_history=force_history,
        )
        return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    """단순 CNN+MLP 정책"""

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer

        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, qpos, image, actions=None, is_pad=None, force_history=None):
        env_state = None
        image = self._normalize(image)

        if actions is not None:
            actions = actions[:, 0]
            a_hat = self.model(
                qpos=qpos,
                image=image,
                env_state=env_state,
                actions=actions,
                force_history=force_history,
            )
            mse = F.mse_loss(actions, a_hat)
            return {"mse": mse, "loss": mse}

        a_hat = self.model(
            qpos=qpos,
            image=image,
            env_state=env_state,
            force_history=force_history,
        )
        return a_hat

    def configure_optimizers(self):
        return self.optimizer


class DiffusionPolicy(nn.Module):
    """
    Diffusion-policy style visuomotor policy.

    Training:
      noisy action chunk -> predict Gaussian noise
    Inference:
      iterative denoising -> action chunk sample
    """

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_Diffusion_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer

        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.loss_type = str(args_override.get("diffusion_loss_type", "mse")).lower()
        self.inference_steps = int(args_override.get("diffusion_infer_steps", 10))
        print(f"[DiffusionPolicy] loss_type={self.loss_type} infer_steps={self.inference_steps}")

    def forward(self, qpos, image, actions=None, is_pad=None, force_history=None):
        image = self._normalize(image)

        if actions is not None:
            actions = actions[:, : self.model.horizon]
            if is_pad is not None:
                is_pad = is_pad[:, : self.model.horizon]

            pred_noise, target_noise, valid_mask = self.model.diffusion_loss(
                qpos=qpos,
                image=image,
                actions=actions,
                force_history=force_history,
                is_pad=is_pad,
            )

            if self.loss_type == "l1":
                per_elem = F.l1_loss(pred_noise, target_noise, reduction="none")
            else:
                per_elem = F.mse_loss(pred_noise, target_noise, reduction="none")

            valid_count = valid_mask.sum().clamp_min(1.0) * actions.shape[-1]
            diff_loss = (per_elem * valid_mask).sum() / valid_count

            return {
                "diffusion": diff_loss,
                "loss": diff_loss,
            }

        a_hat = self.model.sample_actions(
            qpos=qpos,
            image=image,
            force_history=force_history,
            horizon=self.model.horizon,
            inference_steps=self.inference_steps,
        )
        return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    """ACT에서 사용하는 KL 계산"""
    batch_size = mu.size(0)
    assert batch_size != 0

    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld