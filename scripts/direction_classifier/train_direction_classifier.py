#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small binary direction classifier (stain_direction_deg: 0 vs 90), used only
to pick which of two known demo_start_pose targets to align to before the
main policy takes over -- NOT part of the main trajectory policy. Frozen
DINOv3 global feature (same backbone as the main pipeline) + a small linear
head, trained on frames sampled across every labeled episode.
"""
from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "models"))
from dinov3_backbone import DINOv3PatchBackbone  # noqa: E402


class DirectionFrameDataset(Dataset):
    def __init__(self, episode_paths, frames_per_episode: int = 8):
        self.samples = []  # (path, frame_idx, label)
        self.labels_seen = set()
        for p in episode_paths:
            with h5py.File(p, "r") as f:
                direction = f.attrs.get("stain_direction_deg", None)
                if direction is None:
                    continue
                direction = int(direction)
                n = f["observations/images/cam0"].shape[0]
                idxs = np.linspace(0, n - 1, frames_per_episode).astype(int)
                for i in idxs:
                    self.samples.append((str(p), int(i), direction))
                self.labels_seen.add(direction)
        if len(self.labels_seen) < 2:
            raise ValueError(f"need >=2 distinct directions, found {self.labels_seen}")
        self.classes = sorted(self.labels_seen)  # e.g. [0, 90]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, frame_idx, direction = self.samples[idx]
        with h5py.File(path, "r") as f:
            frame = np.asarray(f["observations/images/cam0"][frame_idx])
        img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        label = self.class_to_idx[direction]
        return img, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--num_epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--frames_per_episode", type=int, default=8)
    ap.add_argument(
        "--out_ckpt", type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "direction_classifier" / "classifier.pt"),
    )
    args = ap.parse_args()

    episode_paths = sorted(glob.glob(str(Path(args.dataset_dir) / "episode_*.hdf5")))
    if not episode_paths:
        raise FileNotFoundError(f"no episode_*.hdf5 under {args.dataset_dir}")

    # Stratified split: hold out ~15% of EACH direction's episodes (>=1),
    # so val always covers both classes even with only a handful of small
    # captures (DirectionFrameDataset requires >=2 directions per split).
    random.seed(0)
    by_dir = {}
    for p in episode_paths:
        with h5py.File(p, "r") as f:
            d = f.attrs.get("stain_direction_deg", None)
        if d is None:
            continue
        by_dir.setdefault(int(d), []).append(p)
    train_paths, val_paths = [], []
    for d, paths in by_dir.items():
        random.shuffle(paths)
        n_val = max(1, round(0.15 * len(paths))) if len(paths) > 1 else 0
        val_paths += paths[:n_val]
        train_paths += paths[n_val:]
    if not val_paths:  # degenerate (1 episode/label) -- evaluate on train
        val_paths = list(train_paths)
    random.shuffle(train_paths)

    train_ds = DirectionFrameDataset(train_paths, frames_per_episode=args.frames_per_episode)
    val_ds = DirectionFrameDataset(val_paths, frames_per_episode=args.frames_per_episode)
    print(f"[INFO] classes={train_ds.classes} train_frames={len(train_ds)} val_frames={len(val_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = DINOv3PatchBackbone(
        model_name="vit_small_patch16_dinov3.lvd1689m", pretrained=True, freeze=True,
    ).to(device)
    backbone.eval()
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    head = nn.Sequential(
        nn.LayerNorm(backbone.feature_dim),
        nn.Linear(backbone.feature_dim, 128),
        nn.Mish(),
        nn.Linear(128, len(train_ds.classes)),
    ).to(device)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    best_val_acc = -1.0
    for epoch in range(args.num_epochs):
        head.train()
        total, correct, loss_sum = 0, 0, 0.0
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            with torch.no_grad():
                global_feat, _, _ = backbone(normalize(img))
            logits = head(global_feat)
            loss = loss_fn(logits, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss) * img.shape[0]
            correct += int((logits.argmax(-1) == label).sum())
            total += img.shape[0]
        train_acc = correct / max(1, total)
        train_loss = loss_sum / max(1, total)

        head.eval()
        vtotal, vcorrect = 0, 0
        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(device), label.to(device)
                global_feat, _, _ = backbone(normalize(img))
                logits = head(global_feat)
                vcorrect += int((logits.argmax(-1) == label).sum())
                vtotal += img.shape[0]
        val_acc = vcorrect / max(1, vtotal)
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            out_path = Path(args.out_ckpt)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "head_state_dict": head.state_dict(),
                "classes": train_ds.classes,
                "dino_model_name": "vit_small_patch16_dinov3.lvd1689m",
                "epoch": epoch,
                "val_acc": val_acc,
            }, out_path)

    print(f"[INFO] best val_acc={best_val_acc:.3f}, saved -> {args.out_ckpt}")


if __name__ == "__main__":
    main()
