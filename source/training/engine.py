#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict, List

import torch
from tqdm import tqdm

from common.utils import set_seed
from models.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from training.debug import debug_norm_once, make_grad_scaler, autocast_context

try:
    from training.plotting import plot_history
except Exception:
    plot_history = None


def _scalarize_loss_dict(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out = {}
    for k, v in loss_dict.items():
        out[k] = float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v)
    return out


def _mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if len(dicts) == 0:
        return {}
    keys = dicts[0].keys()
    return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}


def _format_scalars(scalars: Dict[str, float]) -> str:
    preferred = ["loss", "l1", "kl", "mse", "diffusion"]
    ordered = [k for k in preferred if k in scalars]
    ordered.extend(k for k in sorted(scalars.keys()) if k not in ordered)
    return " | ".join(f"{k}:{float(scalars[k]):.6f}" for k in ordered)


def _loss_postfix(scalars: Dict[str, float]) -> Dict[str, str]:
    preferred = ["loss", "l1", "kl", "mse", "diffusion"]
    out = {}
    for key in preferred:
        if key in scalars:
            out[key] = f"{float(scalars[key]):.4f}"
    return out


def make_policy(policy_class: str, policy_config: Dict):
    policy_class = str(policy_class).upper()
    if policy_class == "ACT":
        return ACTPolicy(policy_config)
    if policy_class == "CNNMLP":
        return CNNMLPPolicy(policy_config)
    if policy_class == "DIFFUSION":
        return DiffusionPolicy(policy_config)
    raise ValueError(f"Unsupported policy_class: {policy_class}")


def _unpack_batch(batch, device: torch.device, use_stain_mask: bool = False):
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"batch must be tuple/list, got {type(batch)}")
    items = list(batch)
    stain_mask = None
    if use_stain_mask:
        if len(items) < 5:
            raise ValueError("use_stain_mask=True but batch does not include stain_mask")
        stain_mask = items.pop(-1)

    if len(items) == 4:
        image, qpos, action, is_pad = items
        force_history = None
    elif len(items) == 5:
        image, qpos, action, is_pad, force_history = items
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")

    image = image.to(device, non_blocking=True)
    qpos = qpos.to(device, non_blocking=True)
    action = action.to(device, non_blocking=True)
    is_pad = is_pad.to(device, non_blocking=True)
    if force_history is not None:
        force_history = force_history.to(device, non_blocking=True)
    if stain_mask is not None:
        stain_mask = stain_mask.to(device, non_blocking=True)
    return image, qpos, action, is_pad, force_history, stain_mask


def forward_pass(batch, policy, device, use_stain_mask: bool = False):
    image, qpos, action, is_pad, force_history, stain_mask = _unpack_batch(
        batch,
        device,
        use_stain_mask=use_stain_mask,
    )
    kwargs = {}
    if force_history is not None:
        kwargs["force_history"] = force_history
    if use_stain_mask:
        kwargs["stain_mask"] = stain_mask
    return policy(qpos, image, action, is_pad, **kwargs)


@torch.no_grad()
def _run_validation(val_loader, policy, device, amp_enabled: bool = False, use_stain_mask: bool = False) -> Dict[str, float]:
    policy.eval()
    val_dicts = []
    val_iter = tqdm(val_loader, desc="Val", leave=False)
    for batch in val_iter:
        with autocast_context(amp_enabled, device):
            out = forward_pass(batch, policy, device, use_stain_mask=use_stain_mask)
        scalars = _scalarize_loss_dict(out)
        val_dicts.append(scalars)
        postfix = _loss_postfix(scalars)
        if postfix:
            val_iter.set_postfix(**postfix)
    return _mean_dict(val_dicts)


def _save_checkpoint(ckpt_path: str, epoch: int, policy, optimizer, train_summary, val_summary, config):
    payload = {
        "epoch": int(epoch),
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_summary": train_summary,
        "val_summary": val_summary,
        "config": config,
    }
    torch.save(payload, ckpt_path)


def _plot_history_if_possible(history, ckpt_dir: str, policy_class: str, seed: int):
    if plot_history is None:
        return
    try:
        plot_history(history, ckpt_dir=ckpt_dir, policy_class=policy_class, seed=seed)
    except Exception:
        pass


def train_bc(train_loader, val_loader, config):
    device = config["device"]
    seed = int(config.get("seed", 0))
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    use_stain_mask = bool(policy_config.get("use_stain_mask", False))
    num_epochs = int(config["num_epochs"])
    ckpt_dir = config["ckpt_dir"]

    amp_enabled = bool(config.get("amp", False))
    debug_norm = bool(config.get("debug_norm", False))
    debug_norm_batches = int(config.get("debug_norm_batches", 1))
    debug_batches = int(config.get("debug_batches", 3))
    save_every = int(config.get("save_every", 0))

    os.makedirs(ckpt_dir, exist_ok=True)
    set_seed(seed)

    if debug_norm:
        print("[INFO] debug_norm enabled: printing post-normalization stats for TRAIN and VAL.")
        debug_norm_once(train_loader, tag="TRAIN", max_batches=debug_norm_batches)
        debug_norm_once(val_loader, tag="VAL", max_batches=debug_norm_batches)

    policy = make_policy(policy_class, policy_config).to(device)
    optimizer = policy.configure_optimizers()
    scaler = make_grad_scaler(amp_enabled, device)

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[DEBUG] Policy class = {policy_class}, trainable params = {n_params / 1e6:.2f}M")

    best_val_loss = float("inf")
    best_epoch = -1
    best_ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    last_ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
    history = {"train": [], "val": []}
    epoch_pbar = tqdm(range(num_epochs))

    for epoch in epoch_pbar:
        print(f"Epoch {epoch}")
        val_summary = _run_validation(val_loader, policy, device, amp_enabled, use_stain_mask=use_stain_mask)
        history["val"].append({"epoch": epoch, **val_summary})

        if len(val_summary) > 0:
            val_msg = " | ".join(f"{k}:{v:.6f}" for k, v in sorted(val_summary.items()))
            print(f"Val: {val_msg}")
            current_val_loss = val_summary.get("loss", None)
            if current_val_loss is not None and current_val_loss < best_val_loss:
                best_val_loss = float(current_val_loss)
                best_epoch = int(epoch)
                _save_checkpoint(best_ckpt_path, epoch, policy, optimizer, None, val_summary, config)

        policy.train()
        train_dicts = []
        train_iter = tqdm(train_loader, desc=f"Train {epoch}", leave=False)
        train_total = len(train_loader) if hasattr(train_loader, "__len__") else None
        for batch_idx, batch in enumerate(train_iter):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp_enabled, device):
                out = forward_pass(batch, policy, device, use_stain_mask=use_stain_mask)
                loss = out["loss"]
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scalars = _scalarize_loss_dict(out)
            train_dicts.append(scalars)
            postfix = _loss_postfix(scalars)
            if postfix:
                train_iter.set_postfix(**postfix)
            if debug_batches < 0 or batch_idx < debug_batches:
                total_msg = "?" if train_total is None else str(train_total)
                lr = float(optimizer.param_groups[0].get("lr", 0.0))
                tqdm.write(
                    f"[DEBUG] Epoch {epoch}, batch {batch_idx + 1}/{total_msg}, "
                    f"{_format_scalars(scalars)} | lr:{lr:.2e}"
                )

        train_summary = _mean_dict(train_dicts)
        history["train"].append({"epoch": epoch, **train_summary})

        train_loss = train_summary.get("loss", float("nan"))
        val_loss = val_summary.get("loss", float("nan")) if len(val_summary) > 0 else float("nan")
        epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

        if save_every > 0 and (epoch % save_every == 0):
            periodic_ckpt = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            _save_checkpoint(periodic_ckpt, epoch, policy, optimizer, train_summary, val_summary, config)

        _save_checkpoint(last_ckpt_path, epoch, policy, optimizer, train_summary, val_summary, config)

    _plot_history_if_possible(history, ckpt_dir, policy_class, seed)
    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_ckpt_path": best_ckpt_path,
        "last_ckpt_path": last_ckpt_path,
        "history": history,
    }
