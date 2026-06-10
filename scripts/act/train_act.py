#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACT training & evaluation script for nrs_imitation.

This version supports normalization mode selection:

    --norm_mode minmax_01
        qpos/action -> [0, 1]

    --norm_mode minmax_m11
        qpos/action -> [-1, 1]

Raw HDF5 files are not modified. The selected mode is stored inside
dataset_stats.pkl and must be used by the inference node for denormalization.

Observation/action convention:
    qpos/action = [x, y, z, wx, wy, wz, fx, fy, fz]
    image       = cam0 or cam0/cam1 RGB, float [0,1], ImageNet normalization happens in policy.
"""

from __future__ import annotations

import os
import sys
import pickle
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_SOURCE_DIR = os.path.join(_PROJECT_ROOT, "source")

for p in [_PROJECT_ROOT, _SOURCE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from training.engine import train_bc, make_policy
from common.fs import CHECKPOINTS_ROOT, find_latest_timestamped_subdir
from data.loader import load_data


DATASET_ROOT_BY_MODE = {
    "single_cam": Path(_PROJECT_ROOT) / "datasets" / "single_cam",
    "dual_cam": Path(_PROJECT_ROOT) / "datasets" / "multi_cam",
}


def _is_probably_timestamp_dir(name: str) -> bool:
    for fmt in ("%Y%m%d_%H%M", "%Y%m%d%H%M", "%m%d_%H%M"):
        try:
            datetime.strptime(name, fmt)
            return True
        except ValueError:
            pass
    return False


def _episode_files(dataset_dir: str) -> List[Path]:
    d = Path(dataset_dir).expanduser()
    if not d.is_dir():
        return []
    files = sorted(d.glob("episode_*.hdf5"))
    if not files:
        files = sorted(d.glob("episode_*.h5"))
    return files


def _count_episodes(dataset_dir: str) -> int:
    return len(_episode_files(dataset_dir))


def find_latest_imitation_form(root_dir: str) -> str:
    root = Path(root_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    candidates = []
    for ep_dir in root.rglob("imitation_form"):
        if not ep_dir.is_dir():
            continue
        n = _count_episodes(str(ep_dir))
        if n <= 0:
            continue
        run_name = ep_dir.parent.name
        stat = ep_dir.stat()
        timestamp_bonus = 1 if _is_probably_timestamp_dir(run_name) else 0
        candidates.append((timestamp_bonus, run_name, stat.st_mtime, ep_dir, n))

    if not candidates:
        raise FileNotFoundError(
            f"No usable imitation_form dataset found. Expected episode_*.hdf5 under {root}/<RUN_ID>/imitation_form/"
        )

    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return str(candidates[0][3])


def resolve_dataset_dir(dataset_dir: Optional[str], obs_mode: str, dataset_root: Optional[str]) -> str:
    if dataset_dir is not None and str(dataset_dir).strip() != "":
        resolved = os.path.expanduser(dataset_dir)
        if not os.path.isdir(resolved):
            raise FileNotFoundError(f"dataset_dir does not exist: {resolved}")
        return resolved

    root = Path(dataset_root).expanduser() if dataset_root else DATASET_ROOT_BY_MODE[obs_mode]
    latest = find_latest_imitation_form(str(root))
    print(f"[AUTO] dataset_dir not provided -> using latest {obs_mode} imitation_form: {latest}")
    return latest


def parse_camera_names(camera_names_arg, obs_mode: str) -> List[str]:
    out = []
    if camera_names_arg is not None:
        raw = [camera_names_arg] if isinstance(camera_names_arg, str) else list(camera_names_arg)
        for item in raw:
            for part in str(item).split(","):
                s = part.strip()
                if s:
                    out.append(s)
    if out:
        return out
    if obs_mode == "dual_cam":
        return ["cam0", "cam1"]
    return ["cam0"]


def mode_to_ckpt_base(ckpt_dir: Optional[str], ckpt_root: str, obs_mode: str) -> str:
    if ckpt_dir is not None and str(ckpt_dir).strip():
        return os.path.expanduser(str(ckpt_dir))
    return os.path.join(os.path.expanduser(str(ckpt_root)), obs_mode)


def _norm_log(mode: str) -> str:
    if mode == "minmax_m11":
        return "min-max per-dim -> [-1,1] (qpos/action), image -> [0,1]"
    return "min-max per-dim -> [0,1] (qpos/action), image -> [0,1]"


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] using device = {device}")

    policy_class = args.policy_class.upper()
    if policy_class not in ("ACT", "CNNMLP"):
        raise NotImplementedError(policy_class)

    obs_mode = str(args.obs_mode).strip().lower()
    if obs_mode == "multi_cam":
        obs_mode = "dual_cam"
    if obs_mode not in ("single_cam", "dual_cam"):
        raise RuntimeError(f"obs_mode must be single_cam or dual_cam, got: {args.obs_mode}")

    dataset_dir = resolve_dataset_dir(args.dataset_dir, obs_mode, args.dataset_root)
    num_episodes = int(args.num_episodes)
    if num_episodes <= 0:
        num_episodes = _count_episodes(dataset_dir)

    camera_names = parse_camera_names(args.camera_names, obs_mode)

    if args.train_seq_len is None:
        args.train_seq_len = int(args.chunk_size)
    if args.val_seq_len is None:
        args.val_seq_len = int(args.chunk_size)

    print(f"[INFO] task_name         = {args.task_name}")
    print(f"[INFO] obs_mode          = {obs_mode}")
    print(f"[INFO] dataset_dir       = {dataset_dir}")
    print(f"[INFO] norm_mode         = {args.norm_mode}")
    print(f"[INFO] num_episodes      = {num_episodes}")
    print(f"[INFO] camera_names      = {camera_names}")
    print(f"[INFO] chunk_size        = {args.chunk_size}")
    print(f"[INFO] train_seq_len     = {args.train_seq_len}")
    print(f"[INFO] val_seq_len       = {args.val_seq_len}")
    print(f"[INFO] samples/ep        = {args.samples_per_episode}")
    print(f"[INFO] batch_size        = {args.batch_size}")
    print(f"[INFO] AMP               = {args.amp}")
    print(f"[INFO] use_force_history = {args.use_force_history}")
    print(f"[INFO] force_history_len = {args.force_history_len}")
    print(f"[INFO] use_stain_mask     = {bool(args.use_stain_mask)}")
    if args.use_stain_mask:
        print(
            f"[INFO] stain_mask_key    = {args.stain_mask_key}, "
            f"pooling={args.stain_pooling_type}, empty={args.empty_stain_feature_mode}, "
            f"threshold={args.stain_mask_threshold}"
        )
    if args.norm_mode == "minmax_m11":
        print("[INFO] qpos/action norm  = min-max per-dim -> [-1,1]")
    else:
        print("[INFO] qpos/action norm  = min-max per-dim -> [0,1]")
    print("[INFO] image norm        = raw RGB -> [0,1], then ImageNet normalization inside policy")

    if policy_class == "ACT":
        policy_config = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "num_queries": args.chunk_size,
            "kl_weight": args.kl_weight,
            "hidden_dim": args.hidden_dim,
            "dim_feedforward": args.dim_feedforward,
            "lr_backbone": args.lr_backbone,
            "backbone": args.backbone,
            "enc_layers": args.enc_layers,
            "dec_layers": args.dec_layers,
            "nheads": args.nheads,
            "camera_names": camera_names,
            "obs_mode": obs_mode,
            "state_dim": args.state_dim,
            "action_dim": args.action_dim,
            "image_resize_hw": args.image_resize_hw,
            "image_pool_hw": args.image_pool_hw,
            "temporal_agg": args.temporal_agg,
            "pretrained_backbone": (not args.no_pretrained),

            "position_dim": args.position_dim,
            "force_dim": args.force_dim,
            "position_encoder_hidden_dim": args.position_encoder_hidden_dim,
            "force_encoder_hidden_dim": args.force_encoder_hidden_dim,
            "force_encoder_num_layers": args.force_encoder_num_layers,
            "force_encoder_dropout": args.force_encoder_dropout,
            "observation_encoder_activation": args.observation_encoder_activation,
            "norm_mode": args.norm_mode,
            "use_stain_mask": bool(args.use_stain_mask),
            "stain_mask_key": args.stain_mask_key,
            "stain_pooling_type": args.stain_pooling_type,
            "empty_stain_feature_mode": args.empty_stain_feature_mode,
            "stain_mask_threshold": float(args.stain_mask_threshold),
            "debug_stain_pooling": bool(args.debug_stain_pooling),
        }
    else:
        policy_config = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "lr_backbone": args.lr_backbone,
            "backbone": args.backbone,
            "num_queries": 1,
            "camera_names": camera_names,
            "obs_mode": obs_mode,
            "state_dim": args.state_dim,
            "action_dim": args.action_dim,
            "image_resize_hw": args.image_resize_hw,
            "image_pool_hw": args.image_pool_hw,
            "temporal_agg": args.temporal_agg,
            "pretrained_backbone": (not args.no_pretrained),

            "position_dim": args.position_dim,
            "force_dim": args.force_dim,
            "position_encoder_hidden_dim": args.position_encoder_hidden_dim,
            "force_encoder_hidden_dim": args.force_encoder_hidden_dim,
            "force_encoder_num_layers": args.force_encoder_num_layers,
            "force_encoder_dropout": args.force_encoder_dropout,
            "observation_encoder_activation": args.observation_encoder_activation,
            "cnnmlp_observation_embed_dim": args.cnnmlp_observation_embed_dim,
            "norm_mode": args.norm_mode,
            "use_stain_mask": bool(args.use_stain_mask),
            "stain_mask_key": args.stain_mask_key,
            "stain_pooling_type": args.stain_pooling_type,
            "empty_stain_feature_mode": args.empty_stain_feature_mode,
            "stain_mask_threshold": float(args.stain_mask_threshold),
            "debug_stain_pooling": bool(args.debug_stain_pooling),
        }

    if args.eval:
        ckpt_dir = mode_to_ckpt_base(args.ckpt_dir, args.ckpt_root, obs_mode)
        best_ckpt = os.path.join(ckpt_dir, "policy_best.ckpt")
        if not os.path.exists(best_ckpt):
            latest_sub = find_latest_timestamped_subdir(ckpt_dir)
            if latest_sub is None:
                raise FileNotFoundError(
                    f"[EVAL] No policy_best.ckpt in {ckpt_dir} and no timestamped subdirectories were found."
                )
            ckpt_dir = latest_sub
            best_ckpt = os.path.join(ckpt_dir, "policy_best.ckpt")

        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        if not os.path.exists(best_ckpt):
            raise FileNotFoundError(f"[EVAL] policy_best.ckpt not found: {best_ckpt}")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"[EVAL] dataset_stats.pkl not found: {stats_path}")

        print(f"[EVAL] Using checkpoint dir: {ckpt_dir}")
        print(f"[INFO] Loading checkpoint from {best_ckpt}")

        policy = make_policy(policy_class, policy_config).to(device)
        ckpt = torch.load(best_ckpt, map_location=device)
        if isinstance(ckpt, dict):
            ckpt_cfg = ckpt.get("config", {}).get("policy_config", {})
            ckpt_use_stain = bool(ckpt_cfg.get("use_stain_mask", False))
            if ckpt_use_stain != bool(policy_config.get("use_stain_mask", False)):
                raise RuntimeError(
                    f"use_stain_mask mismatch: checkpoint={ckpt_use_stain}, current_arg={bool(policy_config.get('use_stain_mask', False))}"
                )
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        missing, unexpected = policy.load_state_dict(state_dict, strict=False)
        policy.eval()

        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        print(f"[INFO] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        print(f"[INFO] Loaded dataset stats from {stats_path}")
        print(f"[INFO] qpos_norm_mode={stats.get('qpos_norm_mode', 'minmax_01')} action_norm_mode={stats.get('action_norm_mode', 'minmax_01')}")
        print("\n✅ ACT model ready for inference!\n")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_base = mode_to_ckpt_base(args.ckpt_dir, args.ckpt_root, obs_mode)
    ckpt_dir = os.path.join(ckpt_base, timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[TRAIN] Checkpoints will be saved under: {ckpt_dir}")

    train_loader, val_loader, stats, meta = load_data(
        dataset_dir=dataset_dir,
        num_episodes=num_episodes,
        camera_names=camera_names,
        obs_mode=obs_mode,
        batch_size_train=args.batch_size,
        batch_size_val=args.batch_size,
        seq_len_train=args.train_seq_len,
        seq_len_val=args.val_seq_len,
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
        use_stain_mask=args.use_stain_mask,
        stain_mask_key=args.stain_mask_key,
        stain_mask_threshold=args.stain_mask_threshold,
    )
    print(f"[INFO] data meta: {meta}")

    with open(os.path.join(ckpt_dir, "dataset_stats.pkl"), "wb") as f:
        pickle.dump(stats, f)
    print(f"[INFO] saved dataset stats -> {ckpt_dir}/dataset_stats.pkl")

    config = {
        "num_epochs": args.num_epochs,
        "ckpt_dir": ckpt_dir,
        "policy_class": policy_class,
        "obs_mode": obs_mode,
        "policy_config": policy_config,
        "seed": args.seed,
        "device": device,
        "amp": args.amp,
        "debug_norm": args.debug_norm,
        "debug_norm_batches": 1,
        "debug_batches": args.debug_batches,
        "save_every": args.save_every,
        "temporal_agg": args.temporal_agg,
        "use_force_history": args.use_force_history,
        "force_history_len": args.force_history_len,
    }

    best_ckpt_info = train_bc(train_loader, val_loader, config)

    print("[INFO] Training finished.")
    print(f"[INFO] Best epoch     = {best_ckpt_info['best_epoch']}")
    print(f"[INFO] Best val loss  = {best_ckpt_info['best_val_loss']:.6f}")
    print(f"[INFO] Best ckpt path = {best_ckpt_info['best_ckpt_path']}")
    print(f"[INFO] Last ckpt path = {best_ckpt_info['last_ckpt_path']}")

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--eval", action="store_true")
    p.add_argument("--ckpt_root", type=str, default=str(CHECKPOINTS_ROOT / "act" / "polishing"))
    p.add_argument("--ckpt_dir", type=str, default=None)
    p.add_argument("--policy_class", type=str, default="ACT", choices=["ACT", "CNNMLP"])
    p.add_argument("--task_name", type=str, default="polishing")

    p.add_argument("--obs_mode", type=str, default="single_cam", choices=["single_cam", "dual_cam"])
    p.add_argument("--dataset_root", type=str, default=None)
    p.add_argument("--dataset_dir", type=str, default=None)
    p.add_argument("--norm_mode", type=str, default="minmax_m11", choices=["minmax_01", "minmax_m11"])
    p.add_argument("--num_episodes", type=int, default=0)
    p.add_argument("--camera_names", nargs="+", default=None)

    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--beta1", type=float, default=0.95)
    p.add_argument("--beta2", type=float, default=0.999)

    p.add_argument("--chunk_size", type=int, default=200)
    p.add_argument("--train_seq_len", type=int, default=None)
    p.add_argument("--val_seq_len", type=int, default=None)
    p.add_argument("--samples_per_episode", type=int, default=50)
    p.add_argument("--save_every", type=int, default=100)

    p.add_argument("--state_dim", type=int, default=9)
    p.add_argument("--action_dim", type=int, default=9)

    p.add_argument("--kl_weight", type=float, default=10)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dim_feedforward", type=int, default=3200)

    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--enc_layers", type=int, default=4)
    p.add_argument("--dec_layers", type=int, default=7)

    p.add_argument("--backbone", type=str, default="resnet18")
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--no_pretrained", action="store_true", default=False)

    p.add_argument("--image_resize_hw", type=int, default=256)
    p.add_argument("--image_pool_hw", type=int, default=4)

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument(
        "--debug_batches",
        type=int,
        default=3,
        help="Number of initial train batches to print per epoch. Use -1 to print every train batch.",
    )

    p.add_argument("--amp", action="store_true", default=False)

    p.add_argument("--temporal_agg", action="store_true", default=True)
    p.add_argument("--no_temporal_agg", dest="temporal_agg", action="store_false")

    p.add_argument("--debug_norm", action="store_true")

    p.add_argument("--use_force_history", action="store_true", default=True)
    p.add_argument("--no_force_history", dest="use_force_history", action="store_false")
    p.add_argument("--force_history_len", type=int, default=10)

    p.add_argument("--position_dim", type=int, default=6)
    p.add_argument("--force_dim", type=int, default=3)
    p.add_argument("--position_encoder_hidden_dim", type=int, default=128)
    p.add_argument("--force_encoder_hidden_dim", type=int, default=64)
    p.add_argument("--force_encoder_num_layers", type=int, default=1)
    p.add_argument("--force_encoder_dropout", type=float, default=0.0)
    p.add_argument("--observation_encoder_activation", type=str, default="gelu", choices=["relu", "gelu", "silu"])

    p.add_argument("--cnnmlp_observation_embed_dim", type=int, default=256)

    p.add_argument("--use_stain_mask", dest="use_stain_mask", action="store_true", default=True)
    p.add_argument("--no_stain_mask", dest="use_stain_mask", action="store_false")
    p.add_argument("--stain_mask_key", type=str, default="observations/images/stain_mask")
    p.add_argument("--stain_pooling_type", type=str, default="masked_mean", choices=["masked_mean"])
    p.add_argument("--empty_stain_feature_mode", type=str, default="zero", choices=["zero", "global"])
    p.add_argument("--stain_mask_threshold", type=float, default=0.5)
    p.add_argument("--debug_stain_pooling", action="store_true", default=False)
    return p


if __name__ == "__main__":
    p = build_arg_parser()
    args = p.parse_args()

    if args.train_seq_len is None:
        args.train_seq_len = int(args.chunk_size)
    if args.val_seq_len is None:
        args.val_seq_len = int(args.chunk_size)

    # Some DETR/ACT modules parse sys.argv again internally.
    # Remove custom arguments that are not known to those modules.
    def _strip_flag_with_optional_value(argv, flag):
        out = []
        skip = False
        for i, a in enumerate(argv):
            if skip:
                skip = False
                continue
            if a == flag:
                # value style: --flag value
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    skip = True
                continue
            if a.startswith(flag + "="):
                continue
            out.append(a)
        return out

    for flag in [
        "--debug_norm",
        "--temporal_agg",
        "--no_temporal_agg",
        "--norm_mode",
        "--obs_mode",
        "--dataset_root",
        "--ckpt_root",
        "--weight_decay",
        "--beta1",
        "--beta2",
        "--save_every",
        "--state_dim",
        "--action_dim",
        "--debug_batches",
        "--no_force_history",
        "--use_stain_mask",
        "--no_stain_mask",
        "--stain_mask_key",
        "--stain_pooling_type",
        "--empty_stain_feature_mode",
        "--stain_mask_threshold",
        "--debug_stain_pooling",
    ]:
        sys.argv = _strip_flag_with_optional_value(sys.argv, flag)

    main(args)
