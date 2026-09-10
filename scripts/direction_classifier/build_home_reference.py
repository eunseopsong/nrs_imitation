#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the reference-difference stain template from home-pose captures.

At autodir inference the classification pose is FIXED (home pose) and the
background is identical for every stain type -- so we don't need a trained
classifier, just one reference image per stain type (drawn at that pose) and
the pixel region where the references differ from each other (= where any
stain leaves a mark). A live frame is then whichever reference it is closest
to inside that region; if it is close to NONE of them it is a stain type we
have never templated. See classify_stain_by_reference().

Known labels: 0 / 90 are the polishing directions; -1 is a templated
non-direction stain (the "3-dot row" novelty). Any int label works.

Input: one or more dirs of episode_*.hdf5 from snapshot_pose_and_camera.py /
classify_now.py (each has observations/images/cam0 and an attr
`stain_direction_deg`; -1 = novel type).

  python3 scripts/direction_classifier/build_home_reference.py \
    --dataset_dir datasets/direction_classifier/home_20260830 \
    --dataset_dir datasets/direction_classifier/home_20260901_3dot

Output: checkpoints/direction_classifier/home_reference.npz
  refs         (K, H, W) float32  -- mean grayscale per label
  labels       (K,)      int      -- e.g. [0, 90, -1]
  label_names  (K,)      str      -- e.g. ["0deg", "90deg", "3dot_row"]
  stain_mask   (H, W)    bool     -- top-`mask_percentile` of max pairwise |ref_i-ref_j|
  diff_mask    (H, W)    bool     -- alias of stain_mask (back-compat)
  ref_gray_0, ref_gray_90 (H, W)  -- back-compat copies when those labels exist
  reject_dist  float              -- best-match SSD above this => "unmatched"
  pose_mean    (6,)      float64  -- the home pose these were shot at
  image_topic, pose_topic, stamp, mask_percentile, blur_sigma, n_ep
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = str(PROJECT_ROOT / "checkpoints" / "direction_classifier" / "home_reference.npz")

# int label -> human name; anything not listed falls back to "type_<label>"
LABEL_NAMES = {0: "0deg", 90: "90deg", -1: "3dot_row",
               2: "2dot_row", 3: "3dot_row", 4: "4dot_row"}


def _frames_of(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.asarray(f["observations/images/cam0"]).astype(np.float32).mean(axis=-1)  # (N,H,W)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, action="append", required=True,
                    help="dir of episode_*.hdf5; repeat for more (labels read from attrs)")
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--mask_percentile", type=float, default=95.0,
                    help="pixels above this max-pairwise |ref_i-ref_j| percentile form the stain mask")
    ap.add_argument("--blur_sigma", type=float, default=3.0)
    ap.add_argument("--reject_scale", type=float, default=8.0,
                    help="reject_dist = reject_scale * (95th-pct leave-out best-match SSD)")
    ap.add_argument("--reject_floor", type=float, default=60.0,
                    help="min reject_dist; within-session leave-out underestimates "
                         "cross-session drift (observed live best_d ~6 vs leave-out ~2)")
    args = ap.parse_args()

    paths = []
    for d in args.dataset_dir:
        paths += sorted(glob.glob(str(Path(d) / "episode_*.hdf5")))
    if not paths:
        raise FileNotFoundError(f"no episode_*.hdf5 under {args.dataset_dir}")

    by_label, poses = {}, []
    topic_i = topic_p = stamp = ""
    for p in paths:
        with h5py.File(p, "r") as f:
            if "stain_direction_deg" not in f.attrs:
                print(f"[build] skip (no stain_direction_deg attr): {p}")
                continue
            lab = int(f.attrs["stain_direction_deg"])
            if "pose_mean" in f.attrs:
                poses.append(np.asarray(f.attrs["pose_mean"], dtype=np.float64))
            topic_i = f.attrs.get("image_topic", topic_i or "/realsense/vr/color/image_raw")
            topic_p = f.attrs.get("pose_topic", topic_p or "/ur10skku/currentP")
            stamp = f.attrs.get("stamp", stamp)
        by_label.setdefault(lab, []).append(p)

    if len(by_label) < 2:
        raise ValueError(f"need >=2 labels, found {sorted(by_label)}")
    labels = sorted(by_label)
    names = [LABEL_NAMES.get(l, f"type_{l}") for l in labels]
    print(f"[build] labels: " + ", ".join(f"{l}({n}, {len(by_label[l])}ep)"
                                          for l, n in zip(labels, names)))

    # per-label mean reference, and per-label per-frame stack (for leave-out)
    frame_stacks = {l: np.concatenate([_frames_of(p) for p in by_label[l]], axis=0) for l in labels}
    refs = np.stack([frame_stacks[l].mean(axis=0) for l in labels]).astype(np.float32)  # (K,H,W)

    # stain mask = where the references most disagree with each other
    pair_max = np.zeros(refs.shape[1:], dtype=np.float32)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pair_max = np.maximum(pair_max, np.abs(refs[i] - refs[j]))
    pair_max = cv2.GaussianBlur(pair_max, (0, 0), args.blur_sigma)
    thr = float(np.percentile(pair_max, args.mask_percentile))
    mask = pair_max > thr
    m_sum = float(mask.sum())
    ref_mean_in_mask = float(refs.mean(axis=0)[mask].mean())

    # leave-out calibration: score every frame against the leave-one-out mean
    # of its OWN class -> that best-match SSD is the in-distribution noise
    # floor; a never-templated stain sits far above it.
    best_ds, inter_ratios = [], []
    for li, l in enumerate(labels):
        st = frame_stacks[l]
        loo_mean = st.mean(axis=0)  # ~= class mean (n>=20); fine for a floor estimate
        for fr in st:
            g = fr - float(fr[mask].mean()) + ref_mean_in_mask
            d = np.array([float(np.sum((g[mask] - refs[k][mask]) ** 2) / m_sum)
                          for k in range(len(labels))])
            best_ds.append(float(d[li]))
            srt = np.sort(d)
            inter_ratios.append(srt[1] / max(srt[0], 1e-6))
    best_ds = np.asarray(best_ds)
    reject_dist = max(args.reject_floor, args.reject_scale * float(np.percentile(best_ds, 95)))

    pose_mean = np.mean(poses, axis=0) if poses else np.zeros(6)
    save_kw = dict(
        refs=refs,
        labels=np.asarray(labels, dtype=np.int64),
        label_names=np.asarray(names),
        stain_mask=mask,
        diff_mask=mask,  # back-compat
        reject_dist=float(reject_dist),
        pose_mean=pose_mean,
        image_topic=str(topic_i),
        pose_topic=str(topic_p),
        stamp=str(stamp),
        mask_percentile=args.mask_percentile,
        blur_sigma=args.blur_sigma,
        n_ep={l: len(by_label[l]) for l in labels}.__repr__(),
    )
    for li, l in enumerate(labels):  # back-compat single-image copies
        if l in (0, 90):
            save_kw[f"ref_gray_{l}"] = refs[li]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **save_kw)

    print(f"[build] stain mask: {int(m_sum)} px ({100 * mask.mean():.2f}% of frame), "
          f"pairwise |ref-ref| max={pair_max.max():.1f}")
    print(f"[build] leave-out best-match SSD: mean={best_ds.mean():.2f} "
          f"p95={np.percentile(best_ds, 95):.2f} max={best_ds.max():.2f}  "
          f"| min inter/intra ratio={min(inter_ratios):.1f}")
    print(f"[build] reject_dist = {reject_dist:.1f}  (best_d above this => UNMATCHED)")
    print(f"[build] home pose_mean = {np.round(pose_mean, 4).tolist()}")
    print(f"[build] saved -> {out}")

    # leave-out sanity check through the real classifier
    from pick_start_and_launch import classify_stain_by_reference
    for l in labels:
        with h5py.File(by_label[l][-1], "r") as f:
            imgs = [np.asarray(x) for x in f["observations/images/cam0"][-10:]]
        pred, info = classify_stain_by_reference(imgs, str(out), verbose=False)
        ok = "OK" if pred == l else "MISMATCH"
        print(f"[build] check {ok}: GT={l}({LABEL_NAMES.get(l, l)}) -> pred={pred} "
              f"best_d={info['best_d']} margin={info['margin']} votes={info['votes']}")


if __name__ == "__main__":
    main()
