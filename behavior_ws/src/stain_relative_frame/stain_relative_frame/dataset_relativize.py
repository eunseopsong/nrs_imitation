#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step [3]: rewrite a dataset in the stain-relative frame.

  use_relative_position = False  (default)
      Numerically identical to the existing path. Every dataset in every
      episode is copied verbatim and then VERIFIED with np.array_equal
      against the source; a single mismatch aborts the run. The guarantee is
      checked, not asserted.

  use_relative_position = True
      obs_position_xy = obs_position_xy - stain_origin
      act_position_xy = act_position_xy - stain_origin
      z, rx, ry, rz and force are copied through untouched (also verified).
      The untouched absolute trajectories are preserved at
        analysis/absolute/observations_position
        analysis/absolute/action_position
      for analysis and logging. They are NOT policy inputs -- the dataloader
      reads observations/position and action/position only.

★ There is no rotation alignment, and no flag that could enable one. Rotating
  the frame onto the stain's principal axis would make 0/30/60/90 deg the same
  input and delete the task. Translation only.

dataset_stats are recomputed on the relative coordinates -- reusing absolute
statistics would put the normaliser off by the whole workpiece offset.

  ros2 run stain_relative_frame dataset_relativize -- \
      --dataset_dir  datasets/polishing/single_cam/<run>/imitation_form \
      --out_dir      datasets/polishing/single_cam/<run>/imitation_form_rel \
      --use_relative_position
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from .config import load_config
from .relative_frame import (
    ABS_ACT_POSITION_KEY,
    ABS_OBS_POSITION_KEY,
    STAIN_ORIGIN_ATTR,
    TRANSFORM_VERSION,
    TRANSFORM_VERSION_ATTR,
    USE_RELATIVE_ATTR,
    StainOrigin,
    to_relative,
)
from .report import table

OBS_POS = "observations/position"
ACT_POS = "action/position"


# ---------------------------------------------------------------------------
# dataset_stats
# ---------------------------------------------------------------------------

def _sanitize_minmax(vmin, vmax, eps: float = 1e-6):
    vmin = np.asarray(vmin, dtype=np.float32)
    vmax = np.asarray(vmax, dtype=np.float32)
    flat = (vmax - vmin) < eps
    vmax = np.where(flat, vmin + eps, vmax)
    return vmin, vmax


def compute_stats(paths: List[Path], qpos_norm_mode: str, action_norm_mode: str) -> Dict:
    """qpos = [position(6), force(3)]; action = [action/position(6), action/force(3)].

    Mirrors source/data/dataset.py::compute_dataset_stats for the single-cam,
    no-gripper case; `--verify_stats` cross-checks against that function
    directly when the training tree is importable.
    """
    q_all, a_all = [], []
    for p in paths:
        with h5py.File(str(p), "r") as f:
            pos = np.asarray(f[OBS_POS], dtype=np.float32)[:, :6]
            frc = np.asarray(f["observations/force"], dtype=np.float32)[:, :3]
            T = min(pos.shape[0], frc.shape[0])
            apos = np.asarray(f[ACT_POS], dtype=np.float32)[:T, :6]
            afrc = np.asarray(f["action/force"], dtype=np.float32)[:T, :3]
            q_all.append(np.concatenate([pos[:T], frc[:T]], axis=-1))
            a_all.append(np.concatenate([apos, afrc], axis=-1))
    q = np.concatenate(q_all, axis=0)
    a = np.concatenate(a_all, axis=0)
    qmin, qmax = _sanitize_minmax(q.min(axis=0), q.max(axis=0))
    amin, amax = _sanitize_minmax(a.min(axis=0), a.max(axis=0))
    return {
        "qpos_min": qmin, "qpos_max": qmax,
        "action_min": amin, "action_max": amax,
        "qpos_norm_mode": qpos_norm_mode,
        "action_norm_mode": action_norm_mode,
        "num_total_timesteps": int(q.shape[0]),
    }


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------

def _copy_tree(src: h5py.Group, dst: h5py.Group, skip: Tuple[str, ...] = ()) -> None:
    for key, item in src.items():
        path = f"{src.name}/{key}".lstrip("/")
        if path in skip:
            continue
        if isinstance(item, h5py.Group):
            g = dst.require_group(key)
            for ak, av in item.attrs.items():
                g.attrs[ak] = av
            _copy_tree(item, g, skip)
        else:
            d = dst.create_dataset(
                key, data=item[()], dtype=item.dtype,
                compression=item.compression, compression_opts=item.compression_opts,
            )
            for ak, av in item.attrs.items():
                d.attrs[ak] = av


def convert_episode(src_path: Path, dst_path: Path, origin: Optional[StainOrigin],
                    use_relative: bool) -> Dict:
    """Write one converted episode. Returns a per-episode audit record."""
    with h5py.File(str(src_path), "r") as fi, h5py.File(str(dst_path), "w") as fo:
        _copy_tree(fi, fo, skip=(OBS_POS, ACT_POS))

        obs_abs = np.asarray(fi[OBS_POS])
        act_abs = np.asarray(fi[ACT_POS])
        obs_new = to_relative(obs_abs, origin, use_relative=use_relative)
        act_new = to_relative(act_abs, origin, use_relative=use_relative)

        for key, arr, src_ds in ((OBS_POS, obs_new, fi[OBS_POS]), (ACT_POS, act_new, fi[ACT_POS])):
            d = fo.create_dataset(
                key, data=arr, dtype=src_ds.dtype,
                compression=src_ds.compression, compression_opts=src_ds.compression_opts,
            )
            for ak, av in src_ds.attrs.items():
                d.attrs[ak] = av

        for ak, av in fi.attrs.items():
            fo.attrs[ak] = av

        if use_relative:
            # Absolute trajectories kept for analysis/logging only. The
            # dataloader never reads under analysis/.
            fo.create_dataset(ABS_OBS_POSITION_KEY, data=obs_abs, dtype=obs_abs.dtype,
                              compression="gzip", compression_opts=4)
            fo.create_dataset(ABS_ACT_POSITION_KEY, data=act_abs, dtype=act_abs.dtype,
                              compression="gzip", compression_opts=4)
            fo.attrs[STAIN_ORIGIN_ATTR] = np.asarray(origin.xy, dtype=np.float64)
        fo.attrs[USE_RELATIVE_ATTR] = int(bool(use_relative))
        fo.attrs[TRANSFORM_VERSION_ATTR] = TRANSFORM_VERSION
        fo.attrs["rotation_aligned"] = 0  # ★ never; translation only

    return {
        "episode": src_path.stem,
        "origin_x_mm": None if origin is None else round(float(origin.xy[0]), 3),
        "origin_y_mm": None if origin is None else round(float(origin.xy[1]), 3),
        "obs_xy_shift_mm": round(float(np.abs(obs_abs[:, :2] - obs_new[:, :2]).max()), 4),
        "T": int(obs_abs.shape[0]),
    }


def verify_episode(src_path: Path, dst_path: Path, use_relative: bool,
                   origin: Optional[StainOrigin]) -> List[str]:
    """Re-open both files and check the guarantees rather than trusting them."""
    problems: List[str] = []
    with h5py.File(str(src_path), "r") as fi, h5py.File(str(dst_path), "r") as fo:
        keys: List[str] = []
        fi.visititems(lambda n, o: keys.append(n) if isinstance(o, h5py.Dataset) else None)
        for k in keys:
            a, b = np.asarray(fi[k]), np.asarray(fo[k])
            if k in (OBS_POS, ACT_POS) and use_relative:
                if not np.array_equal(a[:, 2:], b[:, 2:]):
                    problems.append(f"{k}: z/rotation columns changed")
                shift = a[:, :2] - b[:, :2]
                exp = np.asarray(origin.xy, dtype=shift.dtype)
                if not np.allclose(shift, exp, atol=1e-4):
                    problems.append(
                        f"{k}: xy shift {shift.mean(axis=0)} != stain_origin {exp}"
                    )
                continue
            if not np.array_equal(a, b):
                problems.append(f"{k}: not identical (use_relative={use_relative})")
        if use_relative:
            for k, ref in ((ABS_OBS_POSITION_KEY, OBS_POS), (ABS_ACT_POSITION_KEY, ACT_POS)):
                if k not in fo:
                    problems.append(f"{k}: missing absolute copy")
                elif not np.array_equal(np.asarray(fo[k]), np.asarray(fi[ref])):
                    problems.append(f"{k}: absolute copy does not match the source")
    return problems


# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="[3] rewrite a dataset in the stain-relative frame")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--origins", type=str, default=None,
                    help="stain_origin_stability.json from step [2]")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--use_relative_position", dest="use_relative", action="store_true")
    g.add_argument("--no_relative_position", dest="use_relative", action="store_false")
    ap.set_defaults(use_relative=None)
    ap.add_argument("--keep_unstable", action="store_true",
                    help="include episodes step [2] flagged as unstable")
    ap.add_argument("--qpos_norm_mode", type=str, default="minmax_m11")
    ap.add_argument("--action_norm_mode", type=str, default="minmax_m11")
    ap.add_argument("--verify_stats", action="store_true",
                    help="cross-check the stats against source/data/dataset.py")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    use_relative = cfg.use_relative_position if args.use_relative is None else bool(args.use_relative)

    src_dir = Path(args.dataset_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    files = sorted(src_dir.glob("episode_*.hdf5")) or sorted(src_dir.glob("episode_*.h5"))
    if not files:
        print(f"[relativize] no episode_*.hdf5 under {src_dir}", file=sys.stderr)
        return 1
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            print(f"[relativize] {out_dir} exists and is not empty (use --overwrite)",
                  file=sys.stderr)
            return 1
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    origins: Dict[str, StainOrigin] = {}
    unstable: List[str] = []
    origins_path = None
    if use_relative:
        origins_path = Path(args.origins) if args.origins else cfg.path("stain_origin_report_file")
        if not origins_path.is_file():
            print(f"[relativize] step [2] report not found: {origins_path}\n"
                  f"             Run stain_origin_offline first, or pass "
                  f"--no_relative_position.", file=sys.stderr)
            return 1
        rep = json.loads(origins_path.read_text())
        unstable = list(rep.get("unstable_episodes", []))
        for stem, xy in rep.get("origins", {}).items():
            origins[stem] = StainOrigin(xy, source=f"step2:{origins_path.name}").freeze()
        print(f"[relativize] {len(origins)} origins from {origins_path}"
              f" ({len(unstable)} flagged unstable)")

    print(f"[relativize] use_relative_position = {use_relative}"
          + ("  (identity copy; the flag-off path)" if not use_relative else ""))
    print(f"[relativize] {src_dir}  ->  {out_dir}")

    rows, skipped, all_problems = [], [], {}
    for p in files:
        origin = origins.get(p.stem) if use_relative else None
        if use_relative:
            if origin is None:
                skipped.append((p.stem, "no stain_origin from step [2]"))
                continue
            if p.stem in unstable and not args.keep_unstable:
                skipped.append((p.stem, "flagged unstable in step [2]"))
                continue
        dst = out_dir / p.name
        rows.append(convert_episode(p, dst, origin, use_relative))
        probs = verify_episode(p, dst, use_relative, origin)
        if probs:
            all_problems[p.stem] = probs

    if not rows:
        print("[relativize] nothing converted", file=sys.stderr)
        return 1

    print()
    print(table(rows, title="[3] converted episodes", max_rows=12))
    if skipped:
        print()
        print(table([{"episode": s, "reason": r} for s, r in skipped],
                    title=f"[3] skipped ({len(skipped)})", max_rows=12))
    if all_problems:
        print("\n[relativize] VERIFICATION FAILED:", file=sys.stderr)
        for stem, probs in all_problems.items():
            for msg in probs:
                print(f"[relativize]   {stem}: {msg}", file=sys.stderr)
        return 1
    print(f"\n[relativize] verification: {len(rows)} episodes re-opened and checked, "
          f"0 mismatches"
          + ("" if use_relative else " (bit-identical to the source)"))

    # ---- dataset_stats, recomputed on the CONVERTED coordinates ----------
    out_files = sorted(out_dir.glob("episode_*.hdf5"))
    stats = compute_stats(out_files, args.qpos_norm_mode, args.action_norm_mode)
    stats.update({
        "dataset_dir": str(out_dir),
        USE_RELATIVE_ATTR: bool(use_relative),
        TRANSFORM_VERSION_ATTR: TRANSFORM_VERSION,
        "rotation_aligned": False,
        "source_dataset_dir": str(src_dir),
        "stain_origin_report": None if origins_path is None else str(origins_path),
        "camera_names": ["cam0"],
        "obs_mode": "single_cam",
    })
    stats_path = out_dir / "dataset_stats.pkl"
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    abs_stats = compute_stats(files, args.qpos_norm_mode, args.action_norm_mode)
    srows = []
    names = ["x", "y", "z", "rx", "ry", "rz", "fx", "fy", "fz"]
    for i, nm in enumerate(names):
        srows.append({
            "dim": nm,
            "abs_min": f"{abs_stats['qpos_min'][i]:.3f}",
            "abs_max": f"{abs_stats['qpos_max'][i]:.3f}",
            "new_min": f"{stats['qpos_min'][i]:.3f}",
            "new_max": f"{stats['qpos_max'][i]:.3f}",
            "shifted": "yes" if abs(float(abs_stats["qpos_min"][i] - stats["qpos_min"][i])) > 1e-3 else "no",
        })
    print()
    print(table(srows, title="[3] qpos normalisation range: source vs converted"))
    print(f"\n[relativize] dataset_stats -> {stats_path}")

    if args.verify_stats:
        ok = _cross_check_stats(out_files, stats, args)
        if not ok:
            return 1

    manifest = {
        "step": "3_dataset_relativize",
        "stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        USE_RELATIVE_ATTR: bool(use_relative),
        TRANSFORM_VERSION_ATTR: TRANSFORM_VERSION,
        "rotation_aligned": False,
        "source_dataset_dir": str(src_dir),
        "out_dir": str(out_dir),
        "n_converted": len(rows),
        "skipped": [{"episode": s, "reason": r} for s, r in skipped],
        "episodes": rows,
        "stain_origin_report": None if origins_path is None else str(origins_path),
        "qpos_min": stats["qpos_min"].tolist(),
        "qpos_max": stats["qpos_max"].tolist(),
        "action_min": stats["action_min"].tolist(),
        "action_max": stats["action_max"].tolist(),
    }
    (out_dir / "relativize_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[relativize] manifest -> {out_dir / 'relativize_manifest.json'}")
    print("[relativize] next: [4] validate_shortcut")
    return 0


def _cross_check_stats(out_files, stats, args) -> bool:
    """Compare against source/data/dataset.py so the two cannot drift apart."""
    from .config import PROJECT_ROOT

    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from source.data.dataset import compute_dataset_stats  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        print(f"[relativize] --verify_stats: cannot import the training stats "
              f"function ({exc}); skipping cross-check", file=sys.stderr)
        return True
    ref = compute_dataset_stats(
        out_files, qpos_norm_mode=args.qpos_norm_mode,
        action_norm_mode=args.action_norm_mode, include_gripper=False,
    )
    bad = []
    for k in ("qpos_min", "qpos_max", "action_min", "action_max"):
        if not np.allclose(np.asarray(stats[k]), np.asarray(ref[k]), atol=1e-5):
            bad.append(f"{k}: {np.asarray(stats[k])} vs {np.asarray(ref[k])}")
    if bad:
        print("[relativize] --verify_stats MISMATCH vs source/data/dataset.py:",
              file=sys.stderr)
        for b in bad:
            print(f"[relativize]   {b}", file=sys.stderr)
        return False
    print("[relativize] --verify_stats: matches source/data/dataset.py")
    return True


if __name__ == "__main__":
    sys.exit(main())
