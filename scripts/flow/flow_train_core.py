#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Flow training implementation used by the single/dual camera entrypoints."""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_SOURCE_DIR = os.path.join(_PROJECT_ROOT, "source")
for p in [_PROJECT_ROOT, _SOURCE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import h5py
import numpy as np
import torch
from tqdm import tqdm

from data.loader import load_data
from models.flow_core import build_flow_rgb_policy_and_optimizer
from train_runtime import (
    build_epoch_scheduler,
    resolve_temporal_parameters,
    set_train_dataset_epoch,
)

CHECKPOINTS_FLOW_ROOT = Path(_PROJECT_ROOT) / "checkpoints" / "flow" / "polishing"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--train_all_obs_modes", action="store_true")
    parser.add_argument("--shared_timestamp", action="store_true", default=True)
    parser.add_argument("--obs_mode", type=str, default="single_cam", choices=["single_cam", "dual_cam"])

    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=0)
    parser.add_argument("--camera_names", nargs="+", default=None)

    parser.add_argument("--ckpt_root", type=str, default=str(CHECKPOINTS_FLOW_ROOT))
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

    parser.add_argument("--no_pretrained", action="store_true", default=False)
    parser.add_argument(
        "--image_backbone",
        type=str,
        default="dinov3",
        choices=["resnet18", "dinov3", "dinov3_vits16"],
        help="Image observation backbone. dinov3_vits16 is an alias for dinov3.",
    )
    parser.add_argument(
        "--dino_model_name",
        type=str,
        default="vit_small_patch16_dinov3.lvd1689m",
    )
    parser.add_argument("--dino_checkpoint_path", type=str, default="")
    parser.add_argument(
        "--freeze_image_backbone",
        dest="freeze_image_backbone",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--train_image_backbone",
        dest="freeze_image_backbone",
        action="store_false",
    )
    parser.add_argument(
        "--dino_roi_pooling",
        type=str,
        default="attention",
        choices=["attention", "masked_mean"],
    )
    parser.add_argument("--flow_obs_hidden_dim", type=int, default=256)
    parser.add_argument("--flow_image_feature_dim", type=int, default=512)
    parser.add_argument("--flow_marker_feature_dim", type=int, default=128)
    parser.add_argument("--flow_global_cond_dim", type=int, default=256)
    parser.add_argument("--flow_time_embed_dim", type=int, default=256)
    parser.add_argument("--flow_down_dims", type=str, default="256,512,1024")
    parser.add_argument("--flow_kernel_size", type=int, default=5)
    parser.add_argument("--flow_n_groups", type=int, default=8)
    parser.add_argument("--flow_cond_predict_scale", action="store_true", default=False)
    parser.add_argument("--flow_train_eps", type=float, default=1e-4)
    parser.add_argument("--flow_loss_type", type=str, default="mse", choices=["mse", "l1"])
    parser.add_argument("--qpos_dropout_prob", type=float, default=0.0)
    # Shortcut-breaking augmentation: swap qpos (pre-contact chunk starts
    # only) with a start pose sampled from an episode of the OPPOSITE
    # stain_direction_deg. Image + true action target are left untouched,
    # so the loss can only still be minimized by reading direction off the
    # image. Requires episodes tagged with stain_direction_deg.
    parser.add_argument("--qpos_swap_prob", type=float, default=0.0)
    parser.add_argument("--flow_infer_steps", type=int, default=10)

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

    parser.add_argument("--use_tcp_roi", dest="use_tcp_roi", action="store_true", default=True)
    parser.add_argument("--no_tcp_roi", dest="use_tcp_roi", action="store_false")
    parser.add_argument("--tcp_roi_reference_width", type=int, default=424)
    parser.add_argument("--tcp_roi_reference_height", type=int, default=240)
    parser.add_argument("--tcp_roi_center_x", type=int, default=253)
    parser.add_argument("--tcp_roi_center_y", type=int, default=120)
    parser.add_argument("--tcp_roi_area_fraction", type=float, default=0.10)
    parser.add_argument("--empty_stain_feature_mode", type=str, default="zero", choices=["zero", "global"])
    parser.add_argument("--debug_stain_pooling", action="store_true", default=False)

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


# =============================================================================
# Utils
# =============================================================================

def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _timestamp_like(name: str) -> bool:
    for fmt in ("%Y%m%d_%H%M", "%Y%m%d%H%M", "%m%d_%H%M"):
        try:
            datetime.strptime(name, fmt)
            return True
        except ValueError:
            pass
    return False


def find_latest_timestamped_subdir(root: str | Path) -> Optional[str]:
    root = Path(root).expanduser()
    if not root.is_dir():
        return None
    candidates = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if not (d / "policy_best.ckpt").exists():
            continue
        candidates.append((1 if _timestamp_like(d.name) else 0, d.name, d.stat().st_mtime, d))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return str(candidates[0][3])


def _episode_files(dataset_dir: str | Path) -> List[Path]:
    d = Path(dataset_dir).expanduser()
    files = sorted(d.glob("episode_*.hdf5"))
    if not files:
        files = sorted(d.glob("episode_*.h5"))
    return files


def _count_episodes(dataset_dir: str | Path) -> int:
    return len(_episode_files(dataset_dir))


def find_latest_episode_dir(
    root_dir: str = str(Path(_PROJECT_ROOT) / "datasets"),
    subdir_preference: Sequence[str] = ("imitation_form",),
) -> str:
    root = Path(root_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    candidates = []
    for subdir_name in subdir_preference:
        for ep_dir in root.rglob(subdir_name):
            if not ep_dir.is_dir():
                continue
            n = _count_episodes(ep_dir)
            if n <= 0:
                continue
            run_name = ep_dir.parent.name
            candidates.append((1 if _timestamp_like(run_name) else 0, run_name, ep_dir.stat().st_mtime, ep_dir, n))
        if candidates:
            break

    if not candidates:
        raise FileNotFoundError(f"No usable episode dataset found under {root}")
    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return str(candidates[0][3])


def resolve_dataset_dir(dataset_dir: Optional[str]) -> str:
    if dataset_dir and str(dataset_dir).strip():
        resolved = os.path.expanduser(dataset_dir)
        if not os.path.isdir(resolved):
            raise FileNotFoundError(f"dataset_dir does not exist: {resolved}")
        return resolved
    latest = find_latest_episode_dir()
    print(f"[AUTO] dataset_dir not provided -> using latest episode dir: {latest}")
    return latest


def obs_mode_to_camera_names(obs_mode: str, camera_names_arg: Optional[Sequence[str]]) -> List[str]:
    if camera_names_arg:
        raw = []
        for item in camera_names_arg:
            raw.extend([p.strip() for p in str(item).split(",") if p.strip()])
        if raw:
            return raw

    if obs_mode == "single_cam":
        return ["cam0"]
    if obs_mode == "dual_cam":
        return ["cam0", "cam1"]
    raise ValueError(f"Unsupported obs_mode={obs_mode}")


def mode_to_ckpt_base(args, obs_mode: str) -> str:
    # --ckpt_dir explicitly points either to a timestamp dir for eval or to a root for training.
    if args.ckpt_dir:
        return os.path.expanduser(args.ckpt_dir)
    return os.path.join(os.path.expanduser(args.ckpt_root), obs_mode)


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
        "flow_time_embed_dim": args.flow_time_embed_dim,
        "flow_down_dims": args.flow_down_dims,
        "flow_kernel_size": args.flow_kernel_size,
        "flow_n_groups": args.flow_n_groups,
        "flow_cond_predict_scale": args.flow_cond_predict_scale,
        "flow_infer_steps": args.flow_infer_steps,
        "flow_train_eps": args.flow_train_eps,
        "flow_loss_type": args.flow_loss_type,
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




# =============================================================================
# Dataset / normalization debug
# =============================================================================

def _tensor_debug_line(name: str, x):
    if x is None:
        print(f"[DBG] {name:<16}: None")
        return

    if torch.is_tensor(x):
        t = x.detach().cpu()
        arr = t.float()
        finite = bool(torch.isfinite(arr).all().item())
        mn = float(arr.min().item()) if arr.numel() > 0 else float("nan")
        mx = float(arr.max().item()) if arr.numel() > 0 else float("nan")
        mean = float(arr.mean().item()) if arr.numel() > 0 else float("nan")
        print(
            f"[DBG] {name:<16}: shape={tuple(t.shape)}, dtype={t.dtype}, "
            f"min={mn:.4f}, max={mx:.4f}, mean={mean:.4f}, finite={finite}"
        )
    else:
        a = np.asarray(x)
        finite = bool(np.isfinite(a).all()) if a.size > 0 else True
        mn = float(np.min(a)) if a.size > 0 else float("nan")
        mx = float(np.max(a)) if a.size > 0 else float("nan")
        mean = float(np.mean(a)) if a.size > 0 else float("nan")
        print(
            f"[DBG] {name:<16}: shape={a.shape}, dtype={a.dtype}, "
            f"min={mn:.4f}, max={mx:.4f}, mean={mean:.4f}, finite={finite}"
        )


def _print_stats_debug(stats: Dict[str, object], obs_mode: str, camera_names: Sequence[str]):
    print("\n" + "-" * 80)
    print("[DBG] Dataset stats / normalization check")
    print(f"[DBG] obs_mode        = {obs_mode}")
    print(f"[DBG] camera_names    = {list(camera_names)}")
    print(f"[DBG] qpos_norm_mode  = {stats.get('qpos_norm_mode')}")
    print(f"[DBG] action_norm_mode= {stats.get('action_norm_mode')}")
    print(f"[DBG] marker_norm_mode= {stats.get('marker_norm_mode')}")
    print(f"[DBG] marker_dim      = {stats.get('marker_dim')}")

    for key in ["qpos_min", "qpos_max", "action_min", "action_max", "marker_min", "marker_max"]:
        if key in stats:
            a = np.asarray(stats[key], dtype=np.float32).reshape(-1)
            head = np.array2string(a[: min(6, a.size)], precision=4, separator=", ")
            tail = "" if a.size <= 6 else " ..."
            print(f"[DBG] {key:<12}: shape={a.shape}, head={head}{tail}")

    print("[DBG] Expected normalized ranges:")
    print("[DBG]   image          : [0, 1] before ImageNet normalization inside policy")
    print("[DBG]   qpos/action    : [-1, 1] when norm_mode=minmax_m11")
    print("[DBG]   force_history  : [-1, 1] when norm_mode=minmax_m11")
    print("[DBG]   marker         : [-1, 1] when marker_norm_mode=minmax_m11")
    print("-" * 80 + "\n")


def _debug_one_batch(train_loader, obs_mode: str, camera_names: Sequence[str]):
    print("\n" + "-" * 80)
    print("[DBG] First train batch check")
    batch = next(iter(train_loader))
    image, qpos, action, is_pad, force_history, marker = _unpack_batch(
        batch, torch.device("cpu")
    )

    _tensor_debug_line("image", image)
    _tensor_debug_line("qpos", qpos)
    _tensor_debug_line("action", action)
    _tensor_debug_line("is_pad", is_pad.float())
    _tensor_debug_line("force_history", force_history)
    _tensor_debug_line("marker", marker)

    expected_k = len(list(camera_names))
    actual_k = int(image.shape[1]) if torch.is_tensor(image) and image.dim() >= 2 else -1
    print(f"[DBG] camera count    : expected={expected_k}, actual={actual_k}, names={list(camera_names)}")

    if obs_mode == "single_cam":
        print("[DBG] obs check       : cam0 only expected; marker should be None")
    elif obs_mode == "dual_cam":
        print("[DBG] obs check       : cam0 + cam1 expected; marker should be None")

    print("-" * 80 + "\n")

# =============================================================================
# Demo-start stats
# =============================================================================

def _read_first_dataset_row(f: h5py.File, keys: Sequence[str]) -> Optional[np.ndarray]:
    for key in keys:
        if key in f:
            arr = np.asarray(f[key])
            if arr.shape[0] > 0:
                return np.asarray(arr[0], dtype=np.float32).reshape(-1).copy()
    return None


def _density_mode_index(points_xyz: np.ndarray, k: int = 5) -> int:
    """Index of the real sample sitting in the locally densest neighborhood.

    Demo-start XYZ can be multimodal (e.g. the same task recorded with
    varying reach/length, so recording sometimes started further along the
    approach than other times). Neither the arithmetic mean nor the global
    medoid handle that well: the mean lands on a synthetic point no episode
    ever occupied, and the medoid (min total distance to all others) still
    gets pulled toward the midpoint *between* clusters rather than landing
    inside the most-repeated one.

    This instead measures local density via distance to the k-th nearest
    neighbor (smaller = denser) and picks the point in the tightest,
    most-repeated neighborhood. It requires no prior knowledge of cluster
    count or an absolute distance scale, so it degrades gracefully to
    "medoid-like" behavior on a unimodal/tight dataset while correctly
    favoring the dominant cluster on a multimodal one.
    """
    n = points_xyz.shape[0]
    k_eff = max(1, min(int(k), n - 1))
    diffs = points_xyz[:, None, :] - points_xyz[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)
    knn_dist = np.sort(dist, axis=1)[:, k_eff]
    return int(np.argmin(knn_dist))


def collect_demo_start_pose_stats(dataset_dir: str, num_episodes: int = 0) -> Dict[str, object]:
    files = _episode_files(dataset_dir)
    if num_episodes is not None and int(num_episodes) > 0:
        files = files[: int(num_episodes)]
    poses = []
    qposes = []
    used_files = []
    for path in files:
        try:
            with h5py.File(str(path), "r") as f:
                p0 = _read_first_dataset_row(f, ["observations/position", "position", "pose"])
                if p0 is None or p0.size < 6:
                    continue
                f0 = _read_first_dataset_row(f, ["observations/force", "force", "ft"])
                if f0 is None or f0.size < 3:
                    f0 = np.zeros(3, dtype=np.float32)
                pose0 = p0[:6].astype(np.float32)
                force0 = f0[:3].astype(np.float32)
                poses.append(pose0)
                qposes.append(np.concatenate([pose0, force0], axis=0).astype(np.float32))
                used_files.append(str(path))
        except Exception:
            pass
    if not poses:
        print("[WARN] demo-start stats: no valid initial poses found.")
        return {}
    pose_all = np.stack(poses, axis=0).astype(np.float32)
    qpos_all = np.stack(qposes, axis=0).astype(np.float32)

    # "demo_start_pose_mean" drives the ROS node's auto_move_to_demo_start
    # alignment target, so it must be a pose that is actually safe to move
    # to. Select it via k-NN density mode (real recorded pose in the
    # tightest, most-repeated neighborhood) instead of the arithmetic mean,
    # which can land on a synthetic point far outside -- or between -- the
    # demonstrated envelope when start poses are spread out or multimodal
    # (e.g. varying task reach/length episode to episode). The key name is
    # kept for backward compatibility with existing checkpoint consumers.
    mode_i = _density_mode_index(pose_all[:, :3])
    arithmetic_mean_pose = pose_all.mean(axis=0).astype(np.float32)

    out = {
        "demo_start_pose_mean": pose_all[mode_i].copy(),
        "demo_start_pose_arithmetic_mean": arithmetic_mean_pose,
        "demo_start_pose_stat_method": "knn_density_mode",
        "demo_start_pose_std": pose_all.std(axis=0).astype(np.float32),
        "demo_start_pose_min": pose_all.min(axis=0).astype(np.float32),
        "demo_start_pose_max": pose_all.max(axis=0).astype(np.float32),
        "demo_start_pose_all": pose_all,
        "demo_start_qpos_mean": qpos_all[mode_i].copy(),
        "demo_start_qpos_arithmetic_mean": qpos_all.mean(axis=0).astype(np.float32),
        "demo_start_qpos_std": qpos_all.std(axis=0).astype(np.float32),
        "demo_start_qpos_min": qpos_all.min(axis=0).astype(np.float32),
        "demo_start_qpos_max": qpos_all.max(axis=0).astype(np.float32),
        "demo_start_qpos_all": qpos_all,
        "demo_start_num_episodes": int(pose_all.shape[0]),
        "demo_start_source_dataset_dir": str(Path(dataset_dir).expanduser()),
        "demo_start_episode_files": used_files,
        "demo_start_mode_episode_file": used_files[mode_i],
    }
    print(
        "[DEMO_START] pose_mean(knn_density_mode) = "
        + np.array2string(out["demo_start_pose_mean"], precision=4, separator=", ")
        + f"  <- {used_files[mode_i]}"
    )
    print(
        "[DEMO_START] pose_mean(arithmetic, for reference only) = "
        + np.array2string(arithmetic_mean_pose, precision=4, separator=", ")
    )
    return out


# Keys written by stain_relative_frame/dataset_relativize.py into the converted
# dataset's dataset_stats.pkl. The training stats are recomputed from scratch
# (compute_dataset_stats), so without this carry-forward they would be lost and
# the inference node could not tell a stain-relative checkpoint from an
# absolute-frame one. Names must stay in sync with
# stain_relative_frame.relative_frame.{USE_RELATIVE_ATTR,TRANSFORM_VERSION_ATTR}.
RELATIVE_FRAME_STAT_KEYS = (
    "use_relative_position",
    "relative_transform_version",
    "rotation_aligned",
    "stain_origin_report",
    "source_dataset_dir",
    "observation_force_xy_zeroed",
)


def carry_forward_relative_frame_stats(stats: Dict[str, object], dataset_dir: str) -> None:
    """Copy the stain-relative-frame markers from the dataset's own
    dataset_stats.pkl into the checkpoint stats, in place.

    No-op for a normal (absolute-frame) dataset, which carries none of these
    keys. Never overwrites a key the training pipeline set itself.
    """
    src = Path(dataset_dir).expanduser() / "dataset_stats.pkl"
    if not src.is_file():
        return
    try:
        with open(src, "rb") as f:
            ds_stats = pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] could not read {src} for relative-frame carry-forward: {exc}")
        return
    copied = {}
    for key in RELATIVE_FRAME_STAT_KEYS:
        if key in ds_stats and key not in stats:
            stats[key] = ds_stats[key]
            copied[key] = ds_stats[key]
    if copied:
        print(f"[INFO] carried forward relative-frame stats from dataset: {copied}")


# =============================================================================
# Training helpers
# =============================================================================

def _unpack_batch(batch, device: torch.device):
    items = list(batch)
    if len(items) == 4:
        image, qpos, action, is_pad = items
        force_history = None
        marker = None
    elif len(items) == 5:
        image, qpos, action, is_pad, force_history = items
        marker = None
    elif len(items) == 6:
        image, qpos, action, is_pad, force_history, marker = items
    else:
        raise RuntimeError(f"Unexpected batch length: {len(batch)}")
    image = image.to(device, non_blocking=True)
    qpos = qpos.to(device, non_blocking=True)
    action = action.to(device, non_blocking=True)
    is_pad = is_pad.to(device, non_blocking=True)
    if force_history is not None:
        force_history = force_history.to(device, non_blocking=True)
    if marker is not None:
        marker = marker.to(device, non_blocking=True)
    return image, qpos, action, is_pad, force_history, marker


def _scalar_dict(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v) for k, v in loss_dict.items()}


def _mean_dict(items: List[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    return {k: sum(d[k] for d in items) / len(items) for k in keys}


@torch.no_grad()
def validate(policy, val_loader, device):
    policy.eval()
    outs = []
    val_iter = tqdm(val_loader, desc="Val", leave=False)
    for batch in val_iter:
        image, qpos, action, is_pad, force_history, marker = _unpack_batch(batch, device)
        out = policy(
            qpos,
            image,
            actions=action,
            is_pad=is_pad,
            force_history=force_history,
            marker=marker,
        )
        scalars = _scalar_dict(out)
        outs.append(scalars)
        if "loss" in scalars:
            val_iter.set_postfix(loss=f"{scalars['loss']:.4f}")
    return _mean_dict(outs)


def save_checkpoint(path: str, epoch: int, policy, optimizer, train_summary, val_summary, config, scheduler=None):
    payload = {
        "epoch": int(epoch),
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_summary": train_summary,
        "val_summary": val_summary,
        "config": config,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def train_flow(train_loader, val_loader, config):
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

    policy, optimizer = build_flow_rgb_policy_and_optimizer(policy_config)
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
        val_loss = float(val_summary.get("loss", val_summary.get("flow", float("inf"))))

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
                epoch,
                policy,
                optimizer,
                train_summary,
                val_summary,
                config,
                scheduler=scheduler,
            )
        else:
            epochs_without_improvement += 1

        if save_every > 0 and ((epoch + 1) % save_every == 0):
            save_checkpoint(
                os.path.join(ckpt_dir, f"policy_epoch_{epoch + 1}_seed_{seed}.ckpt"),
                epoch,
                policy,
                optimizer,
                train_summary,
                val_summary,
                config,
                scheduler=scheduler,
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
        last_path,
        last_epoch,
        policy,
        optimizer,
        last_train_summary,
        last_val_summary,
        config,
        scheduler=scheduler,
    )

    print("[INFO] Training finished.")
    print(f"[INFO] Best epoch     = {best_epoch}")
    print(f"[INFO] Best val loss  = {best_val:.6f}")
    print(f"[INFO] Best ckpt path = {os.path.join(ckpt_dir, 'policy_best.ckpt')}")
    print(f"[INFO] Last ckpt path = {last_path}")


# =============================================================================
# One run / sequential run
# =============================================================================

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
    print(f"[RUN] obs_mode={obs_mode}")
    print(f"[INFO] device             = {device}")
    print(f"[INFO] dataset_dir        = {dataset_dir}")
    print(f"[INFO] num_episodes       = {num_episodes}")
    print(f"[INFO] camera_names       = {camera_names}")
    if obs_mode == "single_cam":
        print("[INFO] policy_obs         = cam0 RGB only")
    elif obs_mode == "dual_cam":
        print("[INFO] policy_obs         = cam0 RGB + cam1/global RGB")
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
            ):
                if key in ckpt_cfg:
                    policy_config[key] = ckpt_cfg[key]
            if "use_tcp_roi" not in ckpt_cfg:
                policy_config["use_tcp_roi"] = bool(ckpt_cfg.get("use_stain_mask", False))
                policy_config["use_stain_mask"] = bool(ckpt_cfg.get("use_stain_mask", False))
            if "image_backbone" not in ckpt_cfg:
                # Checkpoints created before DINOv3 support always used ResNet18.
                policy_config["image_backbone"] = "resnet18"
            # The complete backbone state is already stored in the FLOW checkpoint.
            # Avoid a redundant pretrained-weight download while reconstructing it.
            policy_config["pretrained_backbone"] = False
            policy_config["dino_checkpoint_path"] = ""
        policy, _ = build_flow_rgb_policy_and_optimizer(policy_config)
        policy = policy.to(device)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        missing, unexpected = policy.load_state_dict(sd, strict=False)
        policy.eval()
        print(f"[EVAL] ckpt_dir={ckpt_dir}")
        print(f"[EVAL] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        print(f"[EVAL] stats loaded: obs_mode={stats.get('obs_mode')}, camera_names={stats.get('camera_names')}")
        print("\n✅ FLOW model ready for inference wrapper.\n")
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
        qpos_swap_prob=args.qpos_swap_prob,
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
    carry_forward_relative_frame_stats(stats, dataset_dir)

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
        "policy_class": "FLOW",
        "obs_mode": obs_mode,
        "policy_config": policy_config,
    }
    train_flow(train_loader, val_loader, config)


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
