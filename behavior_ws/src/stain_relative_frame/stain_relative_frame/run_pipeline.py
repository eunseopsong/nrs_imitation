#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Driver: run the offline half of the pipeline and print the step-by-step table.

Covers the stages that need no operator in the loop -- [2] stain origin,
[3] dataset conversion, [4] the gate, [5] the path audit -- and reads the
[0]/[1] reports that the interactive stages wrote earlier. Ends with a single
READY_TO_TRAIN / BLOCKED verdict.

  ros2 run stain_relative_frame run_pipeline -- \
      --dataset_dir datasets/polishing/single_cam/<run>/imitation_form \
      --out_dir     datasets/polishing/single_cam/<run>/imitation_form_rel \
      --use_relative_position

Stages [0] and [1] are interactive (the arm has to move, points have to be
touched) and are NOT run from here; this reports their saved verdicts and
refuses to continue if either is missing or failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .config import load_config
from .report import table


def _read(path: Path) -> Optional[Dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="offline pipeline driver + summary table")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--use_relative_position", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip_prereq_check", action="store_true",
                    help="report [0]/[1] but do not stop when they are missing")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    stages: List[Dict] = []

    # ---- [0] / [1]: read the saved verdicts -----------------------------
    home = _read(cfg.path("home_pose_report_file"))
    homog = _read(cfg.path("homography_file"))
    clean = cfg.path("clean_reference_file")

    stages.append(_stage("[0a] home pose repeatability", home, "passed",
                         lambda r: f"xy std {r['std_xyz_mm'][0]:.3f}/{r['std_xyz_mm'][1]:.3f}mm, "
                                   f"rot std max {max(r['std_rot_deg']):.3f}deg"))
    stages.append({
        "step": "[0b] clean reference",
        "verdict": "PASS" if clean.is_file() else "MISSING",
        "detail": str(clean) if clean.is_file() else "run clean_reference_capture",
    })
    stages.append(_stage("[1] homography", homog, "passed",
                         lambda r: f"held-out max {r['max_heldout_error_mm']:.3f}mm "
                                   f"(tol {r['tol_mm']}mm)"))

    blocked = [s for s in stages if s["verdict"] != "PASS"]
    if blocked and not args.skip_prereq_check:
        print(table(stages, title="pipeline status"))
        print("\nBLOCKED: the preconditions above must pass before [2]-[4] mean anything.",
              file=sys.stderr)
        for s in blocked:
            print(f"  {s['step']}: {s['verdict']} -- {s['detail']}", file=sys.stderr)
        return 1

    # ---- [2] -------------------------------------------------------------
    from . import dataset_relativize, stain_origin_offline, validate_shortcut, audit_inference_path

    print("\n" + "=" * 78 + "\n[2] stain origin\n" + "=" * 78)
    rc2 = stain_origin_offline.main(["--dataset_dir", args.dataset_dir]
                                    + (["--config", args.config] if args.config else []))
    rep2 = _read(cfg.path("stain_origin_report_file")) or {}
    stages.append({
        "step": "[2] stain origin detection",
        "verdict": "PASS" if rc2 == 0 else "FAIL",
        "detail": f"{len(rep2.get('origins', {}))} origins, "
                  f"{len(rep2.get('unstable_episodes', []))} unstable",
    })
    if rc2 != 0:
        return _finish(stages, False)

    # ---- [3] -------------------------------------------------------------
    print("\n" + "=" * 78 + "\n[3] dataset conversion\n" + "=" * 78)
    argv3 = ["--dataset_dir", args.dataset_dir, "--out_dir", args.out_dir]
    argv3 += ["--use_relative_position"] if args.use_relative_position else ["--no_relative_position"]
    if args.overwrite:
        argv3.append("--overwrite")
    if args.config:
        argv3 += ["--config", args.config]
    rc3 = dataset_relativize.main(argv3)
    man = _read(Path(args.out_dir) / "relativize_manifest.json") or {}
    stages.append({
        "step": "[3] dataset relativize",
        "verdict": "PASS" if rc3 == 0 else "FAIL",
        "detail": f"use_relative={args.use_relative_position}, "
                  f"{man.get('n_converted', 0)} episodes, "
                  f"{len(man.get('skipped', []))} skipped",
    })
    if rc3 != 0:
        return _finish(stages, False)

    # ---- [4] -------------------------------------------------------------
    print("\n" + "=" * 78 + "\n[4] validation gate\n" + "=" * 78)
    rc4 = validate_shortcut.main(["--dataset_dir", args.out_dir]
                                 + (["--config", args.config] if args.config else []))
    rep4 = _read(cfg.path("validation_report_file")) or {}
    stages.append({
        "step": "[4] shortcut gate",
        "verdict": "PASS" if rc4 == 0 else "FAIL",
        "detail": f"(a) {_acc(rep4, 'a_relative_position')} "
                  f"(b) {_acc(rep4, 'b_force')} "
                  f"(c) {_acc(rep4, 'c_label_shuffle')}",
    })

    # ---- [5] -------------------------------------------------------------
    print("\n" + "=" * 78 + "\n[5] inference path audit\n" + "=" * 78)
    rc5 = audit_inference_path.main(["--config", args.config] if args.config else [])
    stages.append({
        "step": "[5] shared-preprocessing audit",
        "verdict": "PASS" if rc5 == 0 else "FAIL",
        "detail": "training and inference route through relative_frame; "
                  "no per-step recomputation",
    })

    return _finish(stages, rc4 == 0 and rc5 == 0)


def _acc(rep: Dict, key: str) -> str:
    r = rep.get(key) or {}
    a = r.get("accuracy_mean")
    return "n/a" if a is None else f"{a:.3f}"


def _stage(name: str, report: Optional[Dict], key: str, detail) -> Dict:
    if report is None:
        return {"step": name, "verdict": "MISSING", "detail": "report not found"}
    try:
        d = detail(report)
    except Exception:  # noqa: BLE001
        d = ""
    return {"step": name, "verdict": "PASS" if report.get(key) else "FAIL", "detail": d}


def _finish(stages: List[Dict], ok: bool) -> int:
    print("\n" + "=" * 78)
    print(table(stages, title="pipeline summary"))
    print("=" * 78)
    print("READY_TO_TRAIN" if ok else "BLOCKED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
