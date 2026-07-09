#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/flow/train_flow_gripper.py

Flow Matching training entrypoint with three observation modes:
  1) single_cam         : cam0(local RGB) + qpos + gripper + force_history
  2) dual_cam           : cam0(local RGB) + cam1(global RGB) + qpos + gripper + force_history
  3) single_cam_marker  : cam0(local RGB) + ArUco marker(id0,id1) + qpos + gripper + force_history

Sequential 3-model training:
  python3 scripts/flow/train_flow_gripper.py --train_all_obs_modes

Checkpoint layout:
  <repo>/checkpoints/flow/single_cam/YYYYMMDD_HHMM/
  <repo>/checkpoints/flow/dual_cam/YYYYMMDD_HHMM/
  <repo>/checkpoints/flow/single_cam_marker/YYYYMMDD_HHMM/

Single-model examples:
  python3 scripts/flow/train_flow_gripper.py --obs_mode single_cam
  python3 scripts/flow/train_flow_gripper.py --obs_mode dual_cam
  python3 scripts/flow/train_flow_gripper.py --obs_mode single_cam_marker

Eval-load check:
  python3 scripts/flow/train_flow_gripper.py --eval --obs_mode single_cam_marker \
    --ckpt_dir <repo>/checkpoints/flow/single_cam_marker/YYYYMMDD_HHMM
"""

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
from common.fs import CHECKPOINTS_ROOT, DATASETS_ACT_ROOT
from models.gri_flow_core import build_flow_rgb_policy_and_optimizer


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
    root_dir: str = str(DATASETS_ACT_ROOT),
    subdir_preference: Sequence[str] = ("episodes_multimodal", "episodes_ft_camproc", "episodes_ft"),
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
    if obs_mode == "single_cam_marker":
        return ["cam0"]

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
        "use_force_history": args.use_force_history,
        "force_history_len": args.force_history_len,
        "force_encoder_hidden_dim": args.force_encoder_hidden_dim,
        "force_encoder_num_layers": args.force_encoder_num_layers,
        "force_encoder_dropout": args.force_encoder_dropout,
        "flow_obs_hidden_dim": args.flow_obs_hidden_dim,
        "flow_image_feature_dim": args.flow_image_feature_dim,
        "flow_marker_feature_dim": args.flow_marker_feature_dim,
        "flow_global_cond_dim": args.flow_global_cond_dim,
        "gripper_encoder_hidden_dim": args.gripper_encoder_hidden_dim,
        "gripper_feature_dim": args.gripper_feature_dim,
        "flow_time_embed_dim": args.flow_time_embed_dim,
        "flow_down_dims": args.flow_down_dims,
        "flow_kernel_size": args.flow_kernel_size,
        "flow_n_groups": args.flow_n_groups,
        "flow_cond_predict_scale": args.flow_cond_predict_scale,
        "flow_infer_steps": args.flow_infer_steps,
        "flow_train_eps": args.flow_train_eps,
        "flow_loss_type": args.flow_loss_type,
        "norm_mode": args.norm_mode,
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

    for key in ["qpos_min", "qpos_max", "action_min", "action_max", "marker_min", "marker_max", "gripper_current_min", "gripper_current_max"]:
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
    image, qpos, action, is_pad, force_history, marker, gripper_position, gripper_current = _unpack_batch(batch, torch.device("cpu"))

    _tensor_debug_line("image", image)
    _tensor_debug_line("qpos", qpos)
    _tensor_debug_line("action", action)
    _tensor_debug_line("is_pad", is_pad.float())
    _tensor_debug_line("force_history", force_history)
    _tensor_debug_line("marker", marker)
    _tensor_debug_line("gripper_position", gripper_position)
    _tensor_debug_line("gripper_current", gripper_current)

    expected_k = len(list(camera_names))
    actual_k = int(image.shape[1]) if torch.is_tensor(image) and image.dim() >= 2 else -1
    print(f"[DBG] camera count    : expected={expected_k}, actual={actual_k}, names={list(camera_names)}")

    if obs_mode == "single_cam":
        print("[DBG] obs check       : cam0 only expected; marker should be None")
    elif obs_mode == "dual_cam":
        print("[DBG] obs check       : cam0 + cam1 expected; marker should be None")
    elif obs_mode == "single_cam_marker":
        print("[DBG] obs check       : cam0 + marker expected; camera count should be 1")
        if marker is not None and marker.numel() > 0:
            marker_np = marker.detach().cpu().numpy()
            id0_valid = marker_np[:, 6] if marker_np.shape[-1] >= 7 else None
            id1_valid = marker_np[:, 13] if marker_np.shape[-1] >= 14 else None
            if id0_valid is not None:
                print(
                    f"[DBG] marker id0 valid(normed) range: "
                    f"min={float(np.min(id0_valid)):.4f}, max={float(np.max(id0_valid)):.4f}"
                )
            if id1_valid is not None:
                print(
                    f"[DBG] marker id1 valid(normed) range: "
                    f"min={float(np.min(id1_valid)):.4f}, max={float(np.max(id1_valid)):.4f}"
                )

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
    out = {
        "demo_start_pose_mean": pose_all.mean(axis=0).astype(np.float32),
        "demo_start_pose_std": pose_all.std(axis=0).astype(np.float32),
        "demo_start_pose_min": pose_all.min(axis=0).astype(np.float32),
        "demo_start_pose_max": pose_all.max(axis=0).astype(np.float32),
        "demo_start_pose_all": pose_all,
        "demo_start_qpos_mean": qpos_all.mean(axis=0).astype(np.float32),
        "demo_start_qpos_std": qpos_all.std(axis=0).astype(np.float32),
        "demo_start_qpos_min": qpos_all.min(axis=0).astype(np.float32),
        "demo_start_qpos_max": qpos_all.max(axis=0).astype(np.float32),
        "demo_start_qpos_all": qpos_all,
        "demo_start_num_episodes": int(pose_all.shape[0]),
        "demo_start_source_dataset_dir": str(Path(dataset_dir).expanduser()),
        "demo_start_episode_files": used_files,
    }
    print("[DEMO_START] pose_mean = " + np.array2string(out["demo_start_pose_mean"], precision=4, separator=", "))
    return out


# =============================================================================
# Training helpers
# =============================================================================

def _unpack_batch(batch, device: torch.device):
    if len(batch) == 7:
        image, qpos, action, is_pad, force_history, gripper_position, gripper_current = batch
        marker = None
    elif len(batch) == 8:
        image, qpos, action, is_pad, force_history, marker, gripper_position, gripper_current = batch
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
    gripper_position = gripper_position.to(device, non_blocking=True)
    gripper_current = gripper_current.to(device, non_blocking=True)
    return image, qpos, action, is_pad, force_history, marker, gripper_position, gripper_current


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
        image, qpos, action, is_pad, force_history, marker, gripper_position, gripper_current = _unpack_batch(batch, device)
        out = policy(
            qpos,
            image,
            actions=action,
            is_pad=is_pad,
            force_history=force_history,
            marker=marker,
            gripper_position=gripper_position,
            gripper_current=gripper_current,
        )
        scalars = _scalar_dict(out)
        outs.append(scalars)
        if "loss" in scalars:
            val_iter.set_postfix(loss=f"{scalars['loss']:.4f}")
    return _mean_dict(outs)


def save_checkpoint(path: str, epoch: int, policy, optimizer, train_summary, val_summary, config):
    torch.save({
        "epoch": int(epoch),
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_summary": train_summary,
        "val_summary": val_summary,
        "config": config,
    }, path)


def train_flow_gripper(train_loader, val_loader, config):
    device = config["device"]
    seed = int(config.get("seed", 0))
    num_epochs = int(config["num_epochs"])
    ckpt_dir = str(config["ckpt_dir"])
    save_every = int(config.get("save_every", 0))
    debug_batches = int(config.get("debug_batches", 3))
    policy_config = config["policy_config"]

    os.makedirs(ckpt_dir, exist_ok=True)
    set_seed(seed)

    policy, optimizer = build_flow_rgb_policy_and_optimizer(policy_config)
    policy = policy.to(device)

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[MODEL] params = {n_params / 1e6:.2f}M")

    best_val = float("inf")
    best_epoch = -1
    history = {"train": [], "val": []}

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        print(f"Epoch {epoch}")
        val_summary = validate(policy, val_loader, device)
        print("Val: " + " | ".join([f"{k}:{v:.6f}" for k, v in val_summary.items()]))

        val_loss = float(val_summary.get("loss", val_summary.get("flow", float("inf"))))
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            save_checkpoint(os.path.join(ckpt_dir, "policy_best.ckpt"), epoch, policy, optimizer, {}, val_summary, config)

        policy.train()
        train_outs = []
        train_iter = tqdm(train_loader, desc=f"Train {epoch}", leave=False)
        for bi, batch in enumerate(train_iter):
            image, qpos, action, is_pad, force_history, marker, gripper_position, gripper_current = _unpack_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = policy(
                qpos,
                image,
                actions=action,
                is_pad=is_pad,
                force_history=force_history,
                marker=marker,
                gripper_position=gripper_position,
                gripper_current=gripper_current,
            )
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            scalars = _scalar_dict(out)
            train_outs.append(scalars)
            if "loss" in scalars:
                train_iter.set_postfix(loss=f"{scalars['loss']:.4f}")
            if bi < debug_batches:
                print(f"[DEBUG] Epoch {epoch}, batch {bi}, train loss = {float(loss.detach().cpu().item()):.6f}")

        train_summary = _mean_dict(train_outs)
        history["train"].append(train_summary)
        history["val"].append(val_summary)

        if save_every > 0 and (epoch % save_every == 0):
            save_checkpoint(os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt"), epoch, policy, optimizer, train_summary, val_summary, config)

        pbar.set_postfix(train_loss=train_summary.get("loss", 0.0), val_loss=val_loss)

    last_path = os.path.join(ckpt_dir, "policy_last.ckpt")
    save_checkpoint(last_path, num_epochs - 1, policy, optimizer, history["train"][-1], history["val"][-1], config)

    print("[INFO] Training finished.")
    print(f"[INFO] Best epoch     = {best_epoch}")
    print(f"[INFO] Best val loss  = {best_val:.6f}")
    print(f"[INFO] Best ckpt path = {os.path.join(ckpt_dir, 'policy_best.ckpt')}")
    print(f"[INFO] Last ckpt path = {last_path}")


# =============================================================================
# One run / sequential run
# =============================================================================

def run_one(args, obs_mode: str, timestamp: Optional[str] = None):
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
        print("[INFO] policy_obs         = cam0 RGB + gripper state")
    elif obs_mode == "dual_cam":
        print("[INFO] policy_obs         = cam0 RGB + cam1/global RGB + gripper state")
    elif obs_mode == "single_cam_marker":
        print("[INFO] policy_obs         = cam0 RGB + ArUco marker(id0,id1) + gripper state")
    print(f"[INFO] marker_dim         = {args.marker_dim}")
    print(f"[INFO] norm_mode          = {args.norm_mode}")
    print(f"[INFO] batch_size         = {args.batch_size}")
    print(f"[INFO] chunk_size         = {args.chunk_size}")
    print(f"[INFO] force_history      = {args.use_force_history}, L={args.force_history_len}")

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

        policy, _ = build_flow_rgb_policy_and_optimizer(policy_config)
        policy = policy.to(device)
        ckpt = torch.load(best_ckpt, map_location=device)
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
        include_gripper=True,
    )
    print(f"[INFO] data meta: {meta}")

    demo_start_stats = collect_demo_start_pose_stats(dataset_dir=dataset_dir, num_episodes=num_episodes)
    if demo_start_stats:
        stats.update(demo_start_stats)
    stats["policy_config"] = dict(policy_config)
    stats["data_meta"] = dict(meta)

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
        "policy_class": "FLOW",
        "obs_mode": obs_mode,
        "policy_config": policy_config,
    }
    train_flow_gripper(train_loader, val_loader, config)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--train_all_obs_modes", action="store_true")
    parser.add_argument("--shared_timestamp", action="store_true", default=True)
    parser.add_argument("--obs_mode", type=str, default="single_cam", choices=["single_cam", "dual_cam", "single_cam_marker"])

    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=0)
    parser.add_argument("--camera_names", nargs="+", default=None)

    parser.add_argument("--ckpt_root", type=str, default=str(CHECKPOINTS_ROOT / "flow"))
    parser.add_argument("--ckpt_dir", type=str, default=None)

    parser.add_argument("--norm_mode", type=str, default="minmax_m11", choices=["minmax_01", "minmax_m11"])
    parser.add_argument("--marker_dim", type=int, default=14)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--beta1", type=float, default=0.95)
    parser.add_argument("--beta2", type=float, default=0.999)

    parser.add_argument("--chunk_size", type=int, default=200)
    parser.add_argument("--train_seq_len", type=int, default=None)
    parser.add_argument("--val_seq_len", type=int, default=None)
    parser.add_argument("--samples_per_episode", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=100)

    parser.add_argument("--state_dim", type=int, default=9)
    parser.add_argument("--action_dim", type=int, default=10)
    parser.add_argument("--force_dim", type=int, default=3)

    parser.add_argument("--use_force_history", dest="use_force_history", action="store_true", default=True)
    parser.add_argument("--no_force_history", dest="use_force_history", action="store_false")
    parser.add_argument("--force_history_len", type=int, default=10)
    parser.add_argument("--force_encoder_hidden_dim", type=int, default=64)
    parser.add_argument("--force_encoder_num_layers", type=int, default=1)
    parser.add_argument("--force_encoder_dropout", type=float, default=0.0)

    parser.add_argument("--no_pretrained", action="store_true", default=False)
    parser.add_argument("--flow_obs_hidden_dim", type=int, default=256)
    parser.add_argument("--flow_image_feature_dim", type=int, default=512)
    parser.add_argument("--flow_marker_feature_dim", type=int, default=128)
    parser.add_argument("--gripper_encoder_hidden_dim", type=int, default=32)
    parser.add_argument("--gripper_feature_dim", type=int, default=64)
    parser.add_argument("--flow_global_cond_dim", type=int, default=256)
    parser.add_argument("--flow_time_embed_dim", type=int, default=256)
    parser.add_argument("--flow_down_dims", type=str, default="256,512,1024")
    parser.add_argument("--flow_kernel_size", type=int, default=5)
    parser.add_argument("--flow_n_groups", type=int, default=8)
    parser.add_argument("--flow_cond_predict_scale", action="store_true", default=False)
    parser.add_argument("--flow_train_eps", type=float, default=1e-4)
    parser.add_argument("--flow_loss_type", type=str, default="mse", choices=["mse", "l1"])
    parser.add_argument("--flow_infer_steps", type=int, default=10)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--debug_batches", type=int, default=3)
    return parser


def main(args):
    if args.train_all_obs_modes:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M") if args.shared_timestamp else None
        modes = ["single_cam", "dual_cam", "single_cam_marker"]
        print(f"[SEQ] train_all_obs_modes=True | modes={modes} | shared_timestamp={timestamp}")
        for mode in modes:
            run_one(args, obs_mode=mode, timestamp=timestamp)
        print("\n[SEQ] All observation-mode training runs finished.\n")
    else:
        run_one(args, obs_mode=args.obs_mode, timestamp=None)


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
