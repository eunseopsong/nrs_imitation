#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step [2]: per-episode `stain_origin`, plus the detector-stability report.

For each episode HDF5 the first `--stability_frames` (default 30) frames of
cam0 are each run through the diff-mask detector INDEPENDENTLY. That is a
measurement of detector noise only -- production uses ONE value per episode
(the median over the frames that succeeded), frozen for the whole episode.

Reported per episode: n ok / n failed, the origin, the per-axis spread and the
scalar spread. Episodes whose spread exceeds `stain_origin_std_tol_mm` (3 mm)
are listed explicitly, as are the failure frames (empty mask, or more than 5
connected components).

  ros2 run stain_relative_frame stain_origin_offline -- \
      --dataset_dir datasets/polishing/single_cam/<run>/imitation_form

Output: <artifact_dir>/stain_origin_stability.json
  origins: {episode_stem: [x, y]}   <- what step [3] consumes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from .clean_reference_capture import load_clean_reference
from .config import load_config
from .homography import (
    extrinsic_at_pose, load_homography, plane_homography,
)
from .report import table
from .stain_detect import dark_cloud_origin, measure_stability


def _read_frames(path: Path, n: int, key: str = "observations/images/cam0") -> np.ndarray:
    with h5py.File(str(path), "r") as f:
        if key not in f:
            raise KeyError(f"{path.name}: missing {key}")
        return np.asarray(f[key][: int(n)])


def _read_poses(path: Path, n: int, key: str = "observations/position") -> np.ndarray:
    with h5py.File(str(path), "r") as f:
        if key not in f:
            raise KeyError(f"{path.name}: missing {key}")
        return np.asarray(f[key][: int(n), :6], dtype=np.float64)


def _read_label(path: Path) -> Optional[int]:
    with h5py.File(str(path), "r") as f:
        v = f.attrs.get("stain_direction_deg", None)
    return None if v is None else int(v)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="[2] stain origin + stability report")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--clean_reference", type=str, default=None)
    ap.add_argument("--homography", type=str, default=None)
    ap.add_argument("--stability_frames", type=int, default=None)
    ap.add_argument("--std_tol_mm", type=float, default=None)
    ap.add_argument("--image_key", type=str, default="observations/images/cam0")
    ap.add_argument("--pose_key", type=str, default="observations/position")
    ap.add_argument("--method", choices=["auto", "diff", "dark"], default="auto",
                    help="'diff' = clean-reference difference (needs a contemporaneous "
                         "reference; frames must be at the calibration home pose). "
                         "'dark' = reference-free black-strip threshold + a per-frame "
                         "homography rebuilt from the depth extrinsic + each frame's "
                         "TCP pose. 'auto' picks 'dark' when homography.json is "
                         "method=depth_extrinsic.")
    ap.add_argument("--plate_roi", type=int, nargs=4, default=[120, 20, 340, 200],
                    metavar=("U0", "V0", "U1", "V1"))
    ap.add_argument("--tool_box", type=int, nargs=4, default=[228, 90, 300, 240],
                    metavar=("U0", "V0", "U1", "V1"))
    ap.add_argument("--dark_thresh", type=int, default=60)
    ap.add_argument("--max_episodes", type=int, default=0)
    ap.add_argument("--keep_unstable", action="store_true",
                    help="do not fail when most episodes are unstable (you have "
                         "inspected the spread and accept it)")
    ap.add_argument("--allow_failed_homography", action="store_true",
                    help="debug only: read an H that did not pass the [1] gate")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    n_frames = int(args.stability_frames or cfg.stain_stability_frames)
    out_path = Path(args.out) if args.out else cfg.path("stain_origin_report_file")

    H, h_meta = load_homography(
        args.homography or cfg.path("homography_file"),
        require_pass=not args.allow_failed_homography,
    )
    hm = h_meta.get("meta", {}) or {}
    method = args.method
    if method == "auto":
        method = "dark" if hm.get("method") == "depth_extrinsic" else "diff"
    if method == "dark":
        if hm.get("method") != "depth_extrinsic":
            print("[stain] --method dark needs homography.json written by "
                  "homography_depth_calibrate (needs R_cb / t_cb / home pose).",
                  file=sys.stderr)
            return 1
        K = tuple(hm["K_fxfycxcy"])
        R_cb_home = np.asarray(hm["R_cb"], float)
        t_cb_home = np.asarray(hm["t_cb_base_mm"], float)
        home_pose6 = np.asarray(hm["home_pose6"], float)
        z0 = float(hm["z0_mm"])
        plate_roi = tuple(int(x) for x in args.plate_roi)
        tool_box = tuple(int(x) for x in args.tool_box)
        print(f"[stain] method=dark (reference-free), per-frame homography from the "
              f"depth extrinsic; Z0={z0}mm")
        ref = None
    else:
        ref, ref_meta = load_clean_reference(
            args.clean_reference or cfg.path("clean_reference_file"))
        print(f"[stain] method=diff, clean reference {ref.shape} "
              f"lighting={ref_meta.get('lighting')!r}")
    tol = float(args.std_tol_mm if args.std_tol_mm is not None
                else (6.0 if method == "dark" else cfg.stain_origin_std_tol_mm))
    print(f"[stain] H held-out max error = {h_meta.get('max_heldout_error_mm')}mm")

    d = Path(args.dataset_dir).expanduser()
    files = sorted(d.glob("episode_*.hdf5")) or sorted(d.glob("episode_*.h5"))
    if not files:
        print(f"[stain] no episode_*.hdf5 under {d}", file=sys.stderr)
        return 1
    if args.max_episodes:
        files = files[: int(args.max_episodes)]
    print(f"[stain] {len(files)} episodes, first {n_frames} frames each, tol={tol}mm\n")

    detect_kw = dict(
        diff_thresh=cfg.stain_diff_thresh,
        blur_sigma=cfg.stain_blur_sigma,
        min_area=cfg.stain_min_area,
        morph_kernel=cfg.stain_morph_kernel,
        max_components=cfg.stain_max_components,
    )

    rows, origins, unstable, failures_all, errors = [], {}, [], {}, {}
    for p in files:
        try:
            frames = _read_frames(p, n_frames, args.image_key)
            poses = _read_poses(p, n_frames, args.pose_key) if method == "dark" else None
        except (KeyError, OSError) as exc:
            errors[p.stem] = str(exc)
            print(f"[stain] {p.stem}: READ FAILED -- {exc}", file=sys.stderr)
            continue

        if method == "dark":
            Hs = np.stack([
                plane_homography(K, *extrinsic_at_pose(R_cb_home, t_cb_home,
                                                       home_pose6, poses[i]), z0)
                for i in range(len(frames))
            ])
            rep = dark_cloud_origin(frames, Hs, plate_roi, tool_box, episode=p.stem,
                                    std_tol_mm=tol, dark_thresh=args.dark_thresh)
        else:
            rep = measure_stability(frames, ref, H, episode=p.stem, std_tol_mm=tol,
                                    **detect_kw)
        row = rep.row()
        row["label"] = _read_label(p)
        rows.append(row)
        if rep.origin_mm is not None:
            origins[p.stem] = rep.origin_mm.tolist()
        if rep.failures:
            failures_all[p.stem] = [{"frame": i, "status": s} for i, s in rep.failures]
        if rep.unstable:
            unstable.append(p.stem)

    if not rows:
        print("[stain] every episode failed to read", file=sys.stderr)
        return 1

    cols = ["episode", "label", "frames", "ok", "failed",
            "origin_x_mm", "origin_y_mm", "std_x_mm", "std_y_mm", "std_mm", "unstable"]
    print(table(rows, columns=cols, title="[2] stain origin stability (per episode)"))

    stds = np.asarray([r["std_mm"] for r in rows if r["std_mm"] is not None
                       and np.isfinite(r["std_mm"])], dtype=np.float64)
    n_no_origin = sum(1 for r in rows if r["origin_x_mm"] is None)
    print()
    print(f"[stain] episodes                  : {len(rows)}")
    print(f"[stain] with a usable origin      : {len(origins)}")
    print(f"[stain] no origin at all          : {n_no_origin}")
    if stds.size:
        print(f"[stain] spread over first {n_frames} frames: "
              f"median={np.median(stds):.3f}mm p95={np.percentile(stds, 95):.3f}mm "
              f"max={stds.max():.3f}mm")
    print(f"[stain] UNSTABLE (>{tol}mm or any failed frame): {len(unstable)}")
    for name in unstable:
        r = next(x for x in rows if x["episode"] == name)
        fails = failures_all.get(name, [])
        detail = ""
        if fails:
            kinds = sorted({f["status"] for f in fails})
            detail = f", {len(fails)} failed frames {kinds}"
        print(f"[stain]   - {name}: std={r['std_mm']}mm{detail}")

    payload = {
        "step": "2_stain_origin",
        "stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_dir": str(d),
        "stability_frames": n_frames,
        "std_tol_mm": tol,
        "detect_params": detect_kw if method != "dark" else {
            "method": "dark", "dark_thresh": args.dark_thresh,
            "plate_roi": list(args.plate_roi), "tool_box": list(args.tool_box)},
        "homography_file": str(args.homography or cfg.path("homography_file")),
        "homography_max_heldout_error_mm": h_meta.get("max_heldout_error_mm"),
        "stain_method": method,
        "clean_reference_lighting": (ref_meta.get("lighting") if method != "dark" else None),
        "origins": origins,
        "unstable_episodes": unstable,
        "failure_frames": failures_all,
        "read_errors": errors,
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[stain] report -> {out_path}")

    if not origins:
        print("[stain] no episode produced an origin -- check the clean reference "
              "lighting and stain_diff_thresh", file=sys.stderr)
        return 1

    n_total = len(rows)
    n_stable = n_total - len(unstable)
    if n_stable < 0.6 * n_total:
        print(f"[stain] GATE FAILED: only {n_stable}/{n_total} episodes gave a stable "
              f"origin ({len(unstable)} unstable, mostly FAIL_TOO_MANY_COMPONENTS).\n"
              "        The clean reference does not match these episodes -- it must be "
              "captured in the SAME session / lighting as the recordings. A fresh "
              "reference against archived data will not work.\n"
              "        Pass --keep_unstable only if you have inspected the spread and "
              "accept it.", file=sys.stderr)
        if not args.keep_unstable:
            return 1
    if unstable:
        print(f"[stain] {len(unstable)}/{n_total} unstable episodes listed above. Step [3] "
              f"will exclude them unless --keep_unstable is passed.")
    print("[stain] next: [3] dataset_relativize")
    return 0


if __name__ == "__main__":
    sys.exit(main())
