#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B-spline training implementation used by the single/dual camera entrypoints.

Added in parallel to scripts/flow/flow_train_core.py: the dataset loading,
observation-encoder config (DINOv3/ResNet backbone, TCP-ROI pooling, force
history, etc.), checkpoint format, and every policy-agnostic training-loop
helper are reused verbatim from flow_train_core.py. Only the policy family
differs (BSplinePolicy instead of FlowRGBPolicy), so a FLOW run and a
B-spline run on the same dataset are directly comparable.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_SOURCE_DIR = os.path.join(_PROJECT_ROOT, "source")
_FLOW_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts", "flow")
for p in [_PROJECT_ROOT, _SOURCE_DIR, _FLOW_SCRIPTS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from tqdm import tqdm

from data.loader import load_data
from models.bspline_core import build_bspline_policy_and_optimizer
from train_runtime import (
    build_epoch_scheduler,
    resolve_temporal_parameters,
    set_train_dataset_epoch,
)

# Every policy-agnostic helper below (dataset resolution, batch unpacking,
# checkpoint saving, debug printing, ...) is reused as-is from FLOW's script
# rather than duplicated.
import flow_train_core as _flow

set_seed = _flow.set_seed
resolve_dataset_dir = _flow.resolve_dataset_dir
_count_episodes = _flow._count_episodes
obs_mode_to_camera_names = _flow.obs_mode_to_camera_names
find_latest_timestamped_subdir = _flow.find_latest_timestamped_subdir
mode_to_ckpt_base = _flow.mode_to_ckpt_base
collect_demo_start_pose_stats = _flow.collect_demo_start_pose_stats
_unpack_batch = _flow._unpack_batch
_scalar_dict = _flow._scalar_dict
_mean_dict = _flow._mean_dict
validate = _flow.validate
save_checkpoint = _flow.save_checkpoint
_debug_one_batch = _flow._debug_one_batch
_print_stats_debug = _flow._print_stats_debug
_tensor_debug_line = _flow._tensor_debug_line

CHECKPOINTS_BSPLINE_ROOT = Path(_PROJECT_ROOT) / "checkpoints" / "bspline" / "polishing"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--train_all_obs_modes", action="store_true")
    parser.add_argument("--shared_timestamp", action="store_true", default=True)
    parser.add_argument("--obs_mode", type=str, default="single_cam", choices=["single_cam", "dual_cam"])

    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=0)
    parser.add_argument("--camera_names", nargs="+", default=None)

    parser.add_argument("--ckpt_root", type=str, default=str(CHECKPOINTS_BSPLINE_ROOT))
    parser.add_argument("--ckpt_dir", type=str, default=None)

    parser.add_argument("--norm_mode", type=str, default="minmax_m11", choices=["minmax_01", "minmax_m11"])
    parser.add_argument("--marker_dim", type=int, default=14)

    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--beta1", type=float, default=0.95)
    parser.add_argument("--beta2", type=float, default=0.999)

    parser.add_argument("--dataset_hz", type=float, default=30.0)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--chunk_sec", type=float, default=4.27)
    parser.add_argument("--train_seq_len", type=int, default=None)
    parser.add_argument("--val_seq_len", type=int, default=None)
    parser.add_argument("--samples_per_episode", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=50)

    parser.add_argument("--state_dim", type=int, default=9)
    parser.add_argument("--action_dim", type=int, default=9)
    parser.add_argument("--force_dim", type=int, default=3)

    parser.add_argument("--use_force_history", dest="use_force_history", action="store_true", default=True)
    parser.add_argument("--no_force_history", dest="use_force_history", action="store_false")
    parser.add_argument("--force_history_len", type=int, default=30)
    parser.add_argument("--force_history_sec", type=float, default=1.0)
    parser.add_argument("--force_encoder_hidden_dim", type=int, default=64)
    parser.add_argument("--force_encoder_num_layers", type=int, default=1)
    parser.add_argument("--force_encoder_dropout", type=float, default=0.0)

    # --- Observation encoder: identical knobs/defaults to FLOW (shared class) ---
    parser.add_argument("--no_pretrained", action="store_true", default=False)
    parser.add_argument(
        "--image_backbone",
        type=str,
        default="dinov3",
        choices=["resnet18", "dinov3", "dinov3_vits16"],
        help="Image observation backbone. dinov3_vits16 is an alias for dinov3.",
    )
    parser.add_argument("--dino_model_name", type=str, default="vit_small_patch16_dinov3.lvd1689m")
    parser.add_argument("--dino_checkpoint_path", type=str, default="")
    parser.add_argument("--freeze_image_backbone", dest="freeze_image_backbone", action="store_true", default=True)
    parser.add_argument("--train_image_backbone", dest="freeze_image_backbone", action="store_false")
    parser.add_argument("--dino_roi_pooling", type=str, default="attention", choices=["attention", "masked_mean"])
    parser.add_argument("--flow_obs_hidden_dim", type=int, default=256)
    parser.add_argument("--flow_image_feature_dim", type=int, default=512)
    parser.add_argument("--flow_marker_feature_dim", type=int, default=128)
    parser.add_argument("--flow_global_cond_dim", type=int, default=256)

    parser.add_argument("--use_tcp_roi", dest="use_tcp_roi", action="store_true", default=True)
    parser.add_argument("--no_tcp_roi", dest="use_tcp_roi", action="store_false")
    parser.add_argument("--tcp_roi_reference_width", type=int, default=424)
    parser.add_argument("--tcp_roi_reference_height", type=int, default=240)
    parser.add_argument("--tcp_roi_center_x", type=int, default=253)
    parser.add_argument("--tcp_roi_center_y", type=int, default=120)
    parser.add_argument("--tcp_roi_area_fraction", type=float, default=0.10)
    parser.add_argument("--empty_stain_feature_mode", type=str, default="zero", choices=["zero", "global"])
    parser.add_argument("--debug_stain_pooling", action="store_true", default=False)

    # --- B-spline action head (replaces FLOW's velocity-field args) ---
    parser.add_argument("--num_control_points", type=int, default=16)
    parser.add_argument("--bspline_degree", type=int, default=3)
    parser.add_argument("--bspline_hidden_dim", type=int, default=256)
    parser.add_argument("--bspline_loss_type", type=str, default="mse", choices=["mse", "l1"])
    # Modality dropout: zeroes qpos for this fraction of training samples so
    # the model can't shortcut on position alone and has to use the image.
    parser.add_argument("--qpos_dropout_prob", type=float, default=0.0)

    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"])
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--resample_each_epoch", dest="resample_each_epoch", action="store_true", default=True)
    parser.add_argument("--no_resample_each_epoch", dest="resample_each_epoch", action="store_false")

    # FACTR2/FIRST (arXiv:2606.12406): oversample pre-contact/contact chunk
    # start points instead of uniform. Applied to the train split only.
    parser.add_argument("--phase_resample_enable", action="store_true", default=False)
    parser.add_argument("--phase_contact_on_thr", type=float, default=3.0)
    parser.add_argument("--phase_contact_off_thr", type=float, default=1.2)
    parser.add_argument("--phase_precontact_sec", type=float, default=1.0)
    parser.add_argument("--phase_weight_free", type=float, default=1.0)
    parser.add_argument("--phase_weight_precontact", type=float, default=5.0)
    parser.add_argument("--phase_weight_contact", type=float, default=1.0)

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=True)
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent_workers", dest="persistent_workers", action="store_true", default=True)
    parser.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument(
        "--debug_batches",
        type=int,
        default=0,
        help="Number of initial train batches to print per epoch. Use -1 to print every train batch.",
    )
    return parser


def default_policy_config(args, obs_mode: str, camera_names: Sequence[str]) -> Dict:
    use_marker = obs_mode == "single_cam_marker"
    return {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "num_queries": args.chunk_size,
        "state_dim": args.state_dim,
        "action_dim": args.action_dim,
        "force_dim": args.force_dim,
        "marker_dim": args.marker_dim,
        "camera_names": list(camera_names),
        "obs_mode": obs_mode,
        "use_marker": use_marker,
        "pretrained_backbone": not args.no_pretrained,
        "image_backbone": args.image_backbone,
        "dino_model_name": args.dino_model_name,
        "dino_checkpoint_path": args.dino_checkpoint_path,
        "freeze_image_backbone": bool(args.freeze_image_backbone),
        "dino_roi_pooling": args.dino_roi_pooling,
        "use_force_history": args.use_force_history,
        "force_history_len": args.force_history_len,
        "force_history_sec": args.force_history_sec,
        "dataset_hz": args.dataset_hz,
        "chunk_sec": args.chunk_sec,
        "force_encoder_hidden_dim": args.force_encoder_hidden_dim,
        "force_encoder_num_layers": args.force_encoder_num_layers,
        "force_encoder_dropout": args.force_encoder_dropout,
        "flow_obs_hidden_dim": args.flow_obs_hidden_dim,
        "flow_image_feature_dim": args.flow_image_feature_dim,
        "flow_marker_feature_dim": args.flow_marker_feature_dim,
        "flow_global_cond_dim": args.flow_global_cond_dim,
        "num_control_points": args.num_control_points,
        "bspline_degree": args.bspline_degree,
        "bspline_hidden_dim": args.bspline_hidden_dim,
        "bspline_loss_type": args.bspline_loss_type,
        # qpos dropout is applied phase-aware in the dataset (only for
        # not-yet-in-contact chunk starts), not uniformly in the model --
        # see load_data(qpos_dropout_prob=...) below. Leave the model's own
        # (uniform, phase-blind) dropout at 0 so it doesn't double-apply.
        "norm_mode": args.norm_mode,
        "use_tcp_roi": bool(args.use_tcp_roi),
        "tcp_roi_reference_width": int(args.tcp_roi_reference_width),
        "tcp_roi_reference_height": int(args.tcp_roi_reference_height),
        "tcp_roi_center_x": int(args.tcp_roi_center_x),
        "tcp_roi_center_y": int(args.tcp_roi_center_y),
        "tcp_roi_area_fraction": float(args.tcp_roi_area_fraction),
        "use_stain_mask": False,
        "stain_pooling_type": "masked_mean",
        "empty_stain_feature_mode": args.empty_stain_feature_mode,
        "stain_mask_threshold": 0.5,
        "debug_stain_pooling": bool(args.debug_stain_pooling),
    }


def train_bspline(train_loader, val_loader, config):
    device = config["device"]
    seed = int(config.get("seed", 0))
    num_epochs = int(config["num_epochs"])
    ckpt_dir = str(config["ckpt_dir"])
    save_every = int(config.get("save_every", 0))
    debug_batches = int(config.get("debug_batches", 0))
    grad_clip_norm = float(config.get("grad_clip_norm", 0.0))
    early_stopping_patience = int(config.get("early_stopping_patience", 0))
    policy_config = config["policy_config"]
    os.makedirs(ckpt_dir, exist_ok=True)
    set_seed(seed)

    policy, optimizer = build_bspline_policy_and_optimizer(policy_config)
    policy = policy.to(device)
    scheduler = build_epoch_scheduler(
        optimizer=optimizer,
        scheduler_name=config.get("lr_scheduler", "none"),
        num_epochs=num_epochs,
        warmup_epochs=int(config.get("warmup_epochs", 0)),
        min_lr=float(config.get("min_lr", 0.0)),
        base_lr=float(policy_config.get("lr", 1e-4)),
    )

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[MODEL] params = {n_params / 1e6:.2f}M")

    best_val = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history = {"train": [], "val": []}
    last_epoch = -1
    last_train_summary = {}
    last_val_summary = {}

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        set_train_dataset_epoch(train_loader, epoch)
        current_lr = float(optimizer.param_groups[0]["lr"])
        print(f"Epoch {epoch} | lr={current_lr:.8g}")
        policy.train()
        train_outs = []
        train_iter = tqdm(train_loader, desc=f"Train {epoch}", leave=False)
        for bi, batch in enumerate(train_iter):
            image, qpos, action, is_pad, force_history, marker = _unpack_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = policy(
                qpos,
                image,
                actions=action,
                is_pad=is_pad,
                force_history=force_history,
                marker=marker,
            )
            loss = out["loss"]
            loss.backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            optimizer.step()
            scalars = _scalar_dict(out)
            train_outs.append(scalars)
            if "loss" in scalars:
                train_iter.set_postfix(loss=f"{scalars['loss']:.4f}")
            if debug_batches < 0 or bi < debug_batches:
                print(f"[DEBUG] Epoch {epoch}, batch {bi}, train loss = {float(loss.detach().cpu().item()):.6f}")

        train_summary = _mean_dict(train_outs)
        train_summary["lr"] = current_lr
        val_summary = validate(policy, val_loader, device)
        print("Val: " + " | ".join([f"{k}:{v:.6f}" for k, v in val_summary.items()]))
        val_loss = float(val_summary.get("loss", val_summary.get("bspline", float("inf"))))

        history["train"].append(train_summary)
        history["val"].append(val_summary)
        last_epoch = epoch
        last_train_summary = train_summary
        last_val_summary = val_summary

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                os.path.join(ckpt_dir, "policy_best.ckpt"),
                epoch, policy, optimizer, train_summary, val_summary, config, scheduler=scheduler,
            )
        else:
            epochs_without_improvement += 1

        if save_every > 0 and ((epoch + 1) % save_every == 0):
            save_checkpoint(
                os.path.join(ckpt_dir, f"policy_epoch_{epoch + 1}_seed_{seed}.ckpt"),
                epoch, policy, optimizer, train_summary, val_summary, config, scheduler=scheduler,
            )

        if scheduler is not None:
            scheduler.step()

        pbar.set_postfix(train_loss=train_summary.get("loss", 0.0), val_loss=val_loss)

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                f"[EARLY STOP] no validation improvement for "
                f"{early_stopping_patience} epochs; best_epoch={best_epoch}"
            )
            break

    last_path = os.path.join(ckpt_dir, "policy_last.ckpt")
    save_checkpoint(
        last_path, last_epoch, policy, optimizer, last_train_summary, last_val_summary, config, scheduler=scheduler,
    )

    print("[INFO] Training finished.")
    print(f"[INFO] Best epoch     = {best_epoch}")
    print(f"[INFO] Best val loss  = {best_val:.6f}")
    print(f"[INFO] Best ckpt path = {os.path.join(ckpt_dir, 'policy_best.ckpt')}")
    print(f"[INFO] Last ckpt path = {last_path}")


def run_one(args, obs_mode: str, timestamp: Optional[str] = None):
    resolve_temporal_parameters(args)
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    num_episodes = _count_episodes(dataset_dir)
    if args.num_episodes and args.num_episodes > 0:
        num_episodes = min(num_episodes, int(args.num_episodes))

    camera_names = obs_mode_to_camera_names(obs_mode, args.camera_names)
    train_seq_len = args.train_seq_len or args.chunk_size
    val_seq_len = args.val_seq_len or args.chunk_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print(f"[RUN] policy=BSPLINE obs_mode={obs_mode}")
    print(f"[INFO] device             = {device}")
    print(f"[INFO] dataset_dir        = {dataset_dir}")
    print(f"[INFO] num_episodes       = {num_episodes}")
    print(f"[INFO] camera_names       = {camera_names}")
    print(f"[INFO] marker_dim         = {args.marker_dim}")
    print(f"[INFO] norm_mode          = {args.norm_mode}")
    print(
        f"[INFO] image_backbone     = {args.image_backbone}, "
        f"pretrained={not args.no_pretrained}, frozen={bool(args.freeze_image_backbone)}"
    )
    if str(args.image_backbone).startswith("dinov3"):
        print(
            f"[INFO] dino               = model={args.dino_model_name}, "
            f"roi_pooling={args.dino_roi_pooling}, checkpoint={args.dino_checkpoint_path or 'timm'}"
        )
    print(f"[INFO] batch_size         = {args.batch_size}")
    print(f"[INFO] chunk_size         = {args.chunk_size}")
    print(f"[INFO] dataset_hz         = {args.dataset_hz}")
    print(f"[INFO] chunk_sec          = {args.chunk_sec} -> L={args.chunk_size}")
    print(
        f"[INFO] force_history      = {args.use_force_history}, "
        f"sec={args.force_history_sec}, L={args.force_history_len}"
    )
    print(
        f"[INFO] tcp_roi            = enabled={bool(args.use_tcp_roi)}, "
        f"ref={args.tcp_roi_reference_width}x{args.tcp_roi_reference_height}, "
        f"center=({args.tcp_roi_center_x},{args.tcp_roi_center_y}), "
        f"area_fraction={args.tcp_roi_area_fraction:.4f}"
    )
    print(
        f"[INFO] bspline            = control_points={args.num_control_points}, "
        f"degree={args.bspline_degree}, hidden_dim={args.bspline_hidden_dim}, "
        f"loss_type={args.bspline_loss_type}"
    )

    policy_config = default_policy_config(args, obs_mode, camera_names)

    if args.eval:
        ckpt_base = mode_to_ckpt_base(args, obs_mode)
        ckpt_dir = ckpt_base
        best_ckpt = os.path.join(ckpt_dir, "policy_best.ckpt")
        if not os.path.exists(best_ckpt):
            latest = find_latest_timestamped_subdir(ckpt_base)
            if latest is None:
                raise FileNotFoundError(f"No policy_best.ckpt found in {ckpt_base}")
            ckpt_dir = latest
            best_ckpt = os.path.join(ckpt_dir, "policy_best.ckpt")

        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"dataset_stats.pkl not found: {stats_path}")

        ckpt = torch.load(best_ckpt, map_location=device)
        if isinstance(ckpt, dict):
            ckpt_cfg = ckpt.get("config", {}).get("policy_config", {})
            for key in (
                "image_backbone",
                "dino_model_name",
                "freeze_image_backbone",
                "dino_roi_pooling",
                "use_tcp_roi",
                "tcp_roi_reference_width",
                "tcp_roi_reference_height",
                "tcp_roi_center_x",
                "tcp_roi_center_y",
                "tcp_roi_area_fraction",
                "num_control_points",
                "bspline_degree",
                "bspline_hidden_dim",
                "bspline_loss_type",
            ):
                if key in ckpt_cfg:
                    policy_config[key] = ckpt_cfg[key]
            # The complete backbone state is already stored in the checkpoint.
            # Avoid a redundant pretrained-weight download while reconstructing it.
            policy_config["pretrained_backbone"] = False
            policy_config["dino_checkpoint_path"] = ""
        policy, _ = build_bspline_policy_and_optimizer(policy_config)
        policy = policy.to(device)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        missing, unexpected = policy.load_state_dict(sd, strict=False)
        policy.eval()
        print(f"[EVAL] ckpt_dir={ckpt_dir}")
        print(f"[EVAL] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        print(f"[EVAL] stats loaded: obs_mode={stats.get('obs_mode')}, camera_names={stats.get('camera_names')}")
        print("\n✅ B-spline model ready for inference wrapper.\n")
        return

    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_root_for_mode = mode_to_ckpt_base(args, obs_mode)
    ckpt_dir = os.path.join(ckpt_root_for_mode, ts)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[TRAIN] Checkpoints will be saved under: {ckpt_dir}")

    train_loader, val_loader, stats, meta = load_data(
        dataset_dir=dataset_dir,
        num_episodes=num_episodes,
        camera_names=camera_names,
        obs_mode=obs_mode,
        batch_size_train=args.batch_size,
        batch_size_val=args.batch_size,
        seq_len_train=train_seq_len,
        seq_len_val=val_seq_len,
        seed=args.seed,
        samples_per_episode=args.samples_per_episode,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        return_force_history=args.use_force_history,
        use_force_history=args.use_force_history,
        force_history_len=args.force_history_len,
        qpos_norm_mode=args.norm_mode,
        action_norm_mode=args.norm_mode,
        marker_norm_mode=args.norm_mode,
        marker_dim=args.marker_dim,
        include_gripper=False,
        use_stain_mask=False,
        resample_each_epoch=args.resample_each_epoch,
        phase_resample_enable=args.phase_resample_enable,
        phase_contact_on_thr=args.phase_contact_on_thr,
        phase_contact_off_thr=args.phase_contact_off_thr,
        phase_precontact_sec=args.phase_precontact_sec,
        dataset_hz=args.dataset_hz,
        phase_weight_free=args.phase_weight_free,
        phase_weight_precontact=args.phase_weight_precontact,
        phase_weight_contact=args.phase_weight_contact,
        qpos_dropout_prob=args.qpos_dropout_prob,
    )
    if args.phase_resample_enable:
        print(
            "[INFO] phase_resample     = enabled, on/off="
            f"{args.phase_contact_on_thr}/{args.phase_contact_off_thr}N, "
            f"precontact={args.phase_precontact_sec}s, "
            f"weights(free,precontact,contact)="
            f"({args.phase_weight_free},{args.phase_weight_precontact},{args.phase_weight_contact})"
        )
    print(f"[INFO] data meta: {meta}")

    demo_start_stats = collect_demo_start_pose_stats(dataset_dir=dataset_dir, num_episodes=num_episodes)
    if demo_start_stats:
        stats.update(demo_start_stats)
    stats["policy_config"] = dict(policy_config)
    stats["data_meta"] = dict(meta)
    stats["dataset_hz"] = float(args.dataset_hz)
    stats["force_history_sec"] = float(args.force_history_sec)
    stats["force_history_len"] = int(args.force_history_len)
    stats["chunk_sec"] = float(args.chunk_sec)
    stats["chunk_size"] = int(args.chunk_size)
    _flow.carry_forward_relative_frame_stats(stats, dataset_dir)

    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    print(f"[INFO] saved dataset stats -> {stats_path}")

    config = {
        "device": device,
        "seed": args.seed,
        "num_epochs": args.num_epochs,
        "ckpt_dir": ckpt_dir,
        "save_every": args.save_every,
        "debug_batches": args.debug_batches,
        "lr_scheduler": args.lr_scheduler,
        "warmup_epochs": args.warmup_epochs,
        "min_lr": args.min_lr,
        "grad_clip_norm": args.grad_clip_norm,
        "early_stopping_patience": args.early_stopping_patience,
        "resample_each_epoch": args.resample_each_epoch,
        "policy_class": "BSPLINE",
        "obs_mode": obs_mode,
        "policy_config": policy_config,
    }
    train_bspline(train_loader, val_loader, config)


def main(args):
    if args.train_all_obs_modes:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M") if args.shared_timestamp else None
        modes = ["single_cam", "dual_cam"]
        print(f"[SEQ] train_all_obs_modes=True | modes={modes} | shared_timestamp={timestamp}")
        for mode in modes:
            run_one(args, obs_mode=mode, timestamp=timestamp)
        print("\n[SEQ] All observation-mode training runs finished.\n")
    else:
        run_one(args, obs_mode=args.obs_mode, timestamp=None)


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
