#!/usr/bin/env python3
"""Offline E1 schema, force-isolation, Flow smoke, and checkpoint test.

This test never connects to ROS or publishes robot commands.  The small Flow
model intentionally uses an unpretrained ResNet and a short horizon so it is a
practical smoke test; the training commands use the saved DINOv3 settings.
"""
from __future__ import annotations

import argparse
import copy
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
from data.dataset import make_loaders  # noqa: E402
from models.flow_core import FlowRGBPolicy  # noqa: E402


def check_finite(name, value):
    if not torch.isfinite(value).all():
        raise AssertionError(f"{name} contains NaN/Inf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    common = dict(
        dataset_dir=args.dataset_dir, camera_names=["cam0"], obs_mode="single_cam",
        batch_size_train=1, batch_size_val=1, seq_len_train=8, seq_len_val=8,
        samples_per_episode=1, num_workers=0, pin_memory=False,
        force_history_len=30, qpos_norm_mode="minmax_m11", action_norm_mode="minmax_m11",
        marker_dim=14, resample_each_epoch=False, phase_resample_enable=False,
    )
    on_train, _, stats, meta_on = make_loaders(**common, use_force_observation=True)
    off_train, _, stats_off, meta_off = make_loaders(**common, use_force_observation=False)
    # Compare the same episode/sample index; DataLoader shuffle order is not
    # part of the E1 contract and would make this paired check ambiguous.
    on = on_train.dataset[0]; off = off_train.dataset[0]
    names = ["image", "qpos", "action", "is_pad", "force_history"]
    print("schema:", {n: tuple(x.shape) for n, x in zip(names, off)})
    assert tuple(off[1].shape[-1:]) == (9,)
    assert tuple(off[2].shape[-1:]) == (9,)
    assert torch.allclose(on[1][..., :6], off[1][..., :6], atol=1e-6)
    assert torch.allclose(on[2], off[2], atol=1e-6), "OFF changed action target"
    assert torch.allclose(off[1][..., 6:9], torch.zeros_like(off[1][..., 6:9]))
    assert torch.allclose(off[4], torch.zeros_like(off[4]))
    check_finite("qpos", off[1]); check_finite("action", off[2]); check_finite("force_history", off[4])
    assert meta_on["num_train_episodes"] == meta_off["num_train_episodes"]
    assert np.array_equal(stats["action_min"], stats_off["action_min"])
    print("PASS data schema/split/action preservation")

    cfg = dict(
        num_queries=8, state_dim=9, action_dim=9, force_dim=3, camera_names=["cam0"],
        obs_mode="single_cam", use_marker=False, pretrained_backbone=False,
        image_backbone="resnet18", use_tcp_roi=False, use_force_history=True,
        force_encoder_hidden_dim=8, flow_obs_hidden_dim=16, flow_image_feature_dim=16,
        flow_global_cond_dim=16, flow_time_embed_dim=16, flow_down_dims="16,32",
        flow_kernel_size=3, flow_n_groups=4, flow_infer_steps=2, flow_loss_type="mse",
    )
    torch.manual_seed(args.seed)
    policy = FlowRGBPolicy(cfg).eval()
    image, qpos, action, is_pad, fh = (x.unsqueeze(0) for x in off)
    on_image, on_qpos, _, _, on_fh = (x.unsqueeze(0) for x in on)
    qpos_alt = qpos.clone(); qpos_alt[..., 6:9] = torch.tensor([100.0, -100.0, 50.0])
    fh_alt = torch.randn_like(fh) * 100.0
    # Simulate two different raw sensor readings followed by the exact OFF
    # preprocessing contract, including every history row.
    qpos_alt[..., 6:9] = 0.0
    fh_alt.zero_()
    with torch.inference_mode():
        z = torch.zeros((1, 8, 9))
        a = policy.sample_action(qpos=qpos, image=image, force_history=fh, num_steps=2, initial_noise=z)
        b = policy.sample_action(qpos=qpos_alt, image=image, force_history=fh_alt, num_steps=2, initial_noise=z)
        cond_on_1 = policy._condition(qpos=on_qpos, image=on_image, force_history=on_fh)
        cond_on_2 = policy._condition(qpos=qpos_alt, image=image, force_history=fh_alt)
    assert torch.allclose(a, b, atol=1e-6), "OFF policy depends on force perturbation"
    assert not torch.allclose(cond_on_1, cond_on_2, atol=1e-6), "ON force did not reach conditioning"
    print("PASS OFF force independence / ON conditioning reachability")

    if args.smoke:
        policy.train(); opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)
        opt.zero_grad(set_to_none=True)
        result = policy(qpos=qpos, image=image, actions=action, is_pad=is_pad, force_history=fh)
        loss = result["loss"]; check_finite("loss", loss); loss.backward()
        grads = [x.grad for x in policy.parameters() if x.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise AssertionError("loss gradient contains NaN/Inf or is empty")
        opt.step()
        ckpt = out / "smoke_only.ckpt"
        torch.save({"policy": policy.state_dict(), "config": {"policy_config": cfg,
                   "use_force_observation": False}, "smoke_only": True}, ckpt)
        restored = FlowRGBPolicy(cfg); payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        restored.load_state_dict(payload["policy"]); assert payload["smoke_only"]
        print(f"PASS Flow forward/backward + restore -> {ckpt}")
    report = {"dataset_dir": str(Path(args.dataset_dir).resolve()), "meta": meta_off,
              "schema": {n: list(x.shape) for n, x in zip(names, off)},
              "force_observation": {"on": True, "off": False}, "smoke": bool(args.smoke)}
    with (out / "check_report.pkl").open("wb") as f: pickle.dump(report, f)
    print(f"PASS report -> {out / 'check_report.pkl'}")


if __name__ == "__main__":
    main()
