#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline comparison of the 2-dot vs 3-dot FLOW models.

There is no on-robot rollout here -- that stays the definitive test. This
computes the flow-matching validation loss (same quantity trained on) for
each model's policy_best AND policy_last checkpoint, on:
  - its OWN held-out val episodes  (reproduces the training best-val number)
  - the OTHER model's held-out val episodes (cross-generalisation)
Each cell is averaged over several fixed noise seeds (flow matching samples
noise per batch). The eval dataset is normalised with the EVALUATED MODEL's
own dataset_stats, so the model sees inputs in the range it was trained on.

Usage:
  python3 scripts/flow/eval_compare_2dot_3dot.py \
    --run2dot checkpoints/flow/polishing/single_cam/20260901_1652_2dot_calibpatch/20260901_1652 \
    --run3dot checkpoints/flow/polishing/single_cam/20260901_2335_3dot_calibpatch/20260901_2335 \
    --data2dot datasets/polishing/single_cam/20260901_1505_calibpatch/imitation_form \
    --data3dot datasets/polishing/single_cam/20260901_2141_calibpatch/imitation_form
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
for p in (_ROOT, os.path.join(_ROOT, "source"), _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.dataset import ImitationEpisodeDataset, _episode_files  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from flow_train_core import (  # noqa: E402
    build_flow_rgb_policy_and_optimizer,
    validate,
)

CHUNK = 128
FORCE_HIST_LEN = 30
MARKER_DIM = 14
NORM = "minmax_m11"
DATASET_HZ = 30.0
SAMPLES_PER_EP = 50


def val_paths_for(dataset_dir: str, seed: int = 0):
    """Reproduce make_loaders' 90/10 split so we hit the same held-out eps."""
    paths = _episode_files(dataset_dir)
    n = len(paths)
    if n == 1:
        return paths
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    rng.shuffle(order)
    split = min(max(1, int(round(0.9 * n))), n - 1)
    return [paths[i] for i in order[split:]]


def build_val_loader(dataset_dir, stats, seed=12345, batch_size=12):
    vp = val_paths_for(dataset_dir)
    ds = ImitationEpisodeDataset(
        vp, stats=stats, camera_names=["cam0"], obs_mode="single_cam",
        seq_len=CHUNK, samples_per_episode=SAMPLES_PER_EP, seed=seed,
        return_force_history=True, force_history_len=FORCE_HIST_LEN,
        marker_dim=MARKER_DIM, qpos_norm_mode=NORM, action_norm_mode=NORM,
        marker_norm_mode=NORM, resample_each_epoch=False,
        phase_resample_enable=False, dataset_hz=DATASET_HZ,
        qpos_dropout_prob=0.0, qpos_swap_prob=0.0,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2), len(vp), len(ds)


def load_policy(run_dir, which, device):
    ckpt = torch.load(os.path.join(run_dir, f"policy_{which}.ckpt"), map_location=device, weights_only=False)
    pc = dict(ckpt["config"]["policy_config"])
    pc["pretrained_backbone"] = False
    pc["dino_checkpoint_path"] = ""
    policy, _ = build_flow_rgb_policy_and_optimizer(pc)
    policy = policy.to(device)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    policy.load_state_dict(sd, strict=False)
    policy.eval()
    return policy, int(ckpt.get("epoch", -1)), ckpt.get("val_summary", {})


def eval_cell(policy, dataset_dir, stats, device, n_seeds=4):
    losses = []
    for s in range(n_seeds):
        torch.manual_seed(1000 + s)
        np.random.seed(1000 + s)
        loader, n_ep, n_samp = build_val_loader(dataset_dir, stats, seed=12345 + s)
        out = validate(policy, loader, device)
        losses.append(float(out.get("loss", out.get("flow", float("nan")))))
    return float(np.mean(losses)), float(np.std(losses)), n_ep, n_samp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run2dot", required=True)
    ap.add_argument("--run3dot", required=True)
    ap.add_argument("--data2dot", required=True)
    ap.add_argument("--data3dot", required=True)
    ap.add_argument("--n_seeds", type=int, default=4)
    ap.add_argument("--out", default="logs/flow_eval_compare_2dot_3dot.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs = {
        "2dot": {"dir": args.run2dot, "stats": pickle.load(open(os.path.join(args.run2dot, "dataset_stats.pkl"), "rb"))},
        "3dot": {"dir": args.run3dot, "stats": pickle.load(open(os.path.join(args.run3dot, "dataset_stats.pkl"), "rb"))},
    }
    data = {"2dot": args.data2dot, "3dot": args.data3dot}

    report = {"n_seeds": args.n_seeds, "cells": {}, "checkpoints": {}}
    for mtag, r in runs.items():
        for which in ("best", "last"):
            policy, epoch, vs = load_policy(r["dir"], which, device)
            report["checkpoints"][f"{mtag}/{which}"] = {"epoch": epoch, "logged_val": vs}
            for dtag, ddir in data.items():
                mean, std, n_ep, n_samp = eval_cell(policy, ddir, r["stats"], device, args.n_seeds)
                key = f"model={mtag}/{which}  eval_on={dtag}"
                report["cells"][key] = {"flow_loss_mean": round(mean, 4), "flow_loss_std": round(std, 4),
                                        "val_episodes": n_ep, "val_samples": n_samp}
                print(f"{key:44s} -> {mean:.4f} ± {std:.4f}   (val {n_ep} eps / {n_samp} samp)")
            del policy
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")

    # tidy matrix
    print("\n=== flow-matching val loss (lower = better) ===")
    print(f"{'':22s} {'eval:2dot':>12s} {'eval:3dot':>12s}")
    for mtag in ("2dot", "3dot"):
        for which in ("best", "last"):
            c2 = report["cells"][f"model={mtag}/{which}  eval_on=2dot"]["flow_loss_mean"]
            c3 = report["cells"][f"model={mtag}/{which}  eval_on=3dot"]["flow_loss_mean"]
            print(f"model {mtag}/{which:4s}        {c2:>12.4f} {c3:>12.4f}")


if __name__ == "__main__":
    main()
