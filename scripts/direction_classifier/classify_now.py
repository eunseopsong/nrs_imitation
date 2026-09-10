#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the EXISTING direction classifier on live camera frames -- right now,
at whatever pose the robot is in, no move, no training.

Grabs N frames from the camera topic, runs checkpoints/direction_classifier/
classifier.pt, and prints per-frame softmax + the overall vote so you can
see not just the prediction but how confident / consistent it is.

  # put the robot at home pose, draw a 0deg stain:
  python3 scripts/direction_classifier/classify_now.py
  # wipe, draw a 90deg stain:
  python3 scripts/direction_classifier/classify_now.py

Pass --label 0 (or 90) to ALSO save the grabbed frames as an
episode_<N>.hdf5 under --out_dir, so a diagnostic capture doubles as
training data if you later retrain.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from pick_start_and_launch import grab_frames

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(PROJECT_ROOT / "source" / "models"))
from dinov3_backbone import DINOv3PatchBackbone  # noqa: E402

DEFAULT_CKPT = str(PROJECT_ROOT / "checkpoints" / "direction_classifier" / "classifier.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_topic", type=str, default="/realsense/vr/color/image_raw")
    ap.add_argument("--num_frames", type=int, default=10)
    ap.add_argument("--grab_timeout_sec", type=float, default=8.0)
    ap.add_argument("--classifier_ckpt", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--reference_npz", type=str,
                    default=str(PROJECT_ROOT / "checkpoints" / "direction_classifier" / "home_reference.npz"),
                    help="if present, use the reference-difference classifier instead of the DINOv3 head")
    ap.add_argument("--label", type=int, default=None,
                    help="if set, also save the frames as an hdf5 episode "
                         "(0 / 90 = directions, -1 = novel non-direction type)")
    ap.add_argument("--out_dir", type=str,
                    default=str(PROJECT_ROOT / "datasets" / "direction_classifier" / "diag"))
    args = ap.parse_args()

    print(f"[classify_now] grabbing up to {args.num_frames} frames from {args.image_topic} ...")
    frames = grab_frames(args.image_topic, args.num_frames, args.grab_timeout_sec)
    if not frames:
        raise RuntimeError(f"no frames from {args.image_topic} within {args.grab_timeout_sec}s")
    print(f"[classify_now] got {len(frames)} frame(s)")

    if args.reference_npz and Path(args.reference_npz).is_file():
        from pick_start_and_launch import classify_stain_by_reference
        print(f"[classify_now] using reference-difference classifier: {args.reference_npz}")
        label, info = classify_stain_by_reference(frames, args.reference_npz)
        print("-" * 60)
        print(f"PREDICTION : label={label} ({info.get('label_name')})   {info}")
        _maybe_save(frames, args)
        return

    ckpt = torch.load(args.classifier_ckpt, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = DINOv3PatchBackbone(model_name=ckpt["dino_model_name"], pretrained=True, freeze=True).to(device)
    backbone.eval()
    head = nn.Sequential(
        nn.LayerNorm(backbone.feature_dim),
        nn.Linear(backbone.feature_dim, 128),
        nn.Mish(),
        nn.Linear(128, len(classes)),
    ).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    print(f"[classify_now] classifier: classes={classes} trained_val_acc={ckpt.get('val_acc')}")
    votes = np.zeros(len(classes), dtype=np.int64)
    probs_all = []
    with torch.no_grad():
        for i, frame in enumerate(frames):
            img = torch.from_numpy(np.asarray(frame)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            global_feat, _, _ = backbone(normalize(img))
            logits = head(global_feat)
            p = torch.softmax(logits, dim=-1).cpu().numpy().ravel()
            probs_all.append(p)
            pred = int(np.argmax(p))
            votes[pred] += 1
            pstr = "  ".join(f"{c}:{p[j]:.3f}" for j, c in enumerate(classes))
            print(f"  frame {i:2d}  ->  {classes[pred]:>3}deg   [{pstr}]")

    mean_p = np.mean(probs_all, axis=0)
    winner = int(np.argmax(votes))
    print("-" * 60)
    print(f"votes      : {dict(zip(classes, votes.tolist()))}")
    print(f"mean softmax: {dict(zip(classes, np.round(mean_p, 3).tolist()))}")
    print(f"PREDICTION : {classes[winner]}deg   (margin {abs(mean_p[0] - mean_p[1]):.3f})")

    _maybe_save(frames, args)


def _maybe_save(frames, args) -> None:
    if args.label is None:
        return
    import h5py
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = glob.glob(str(out_dir / "episode_*.hdf5"))
    idx = 1 + max([int(Path(p).stem.split("_")[1]) for p in existing], default=-1)
    path = out_dir / f"episode_{idx}.hdf5"
    arr = np.stack([np.asarray(f) for f in frames]).astype(np.uint8)
    with h5py.File(path, "w") as f:
        f.create_group("observations/images").create_dataset(
            "cam0", data=arr, compression="gzip", compression_opts=4)
        f.attrs["stain_direction_deg"] = int(args.label)
        f.attrs["source"] = "classify_now.py"
    print(f"[classify_now] saved {arr.shape[0]} frames -> {path} (label={args.label}deg)")


if __name__ == "__main__":
    main()
