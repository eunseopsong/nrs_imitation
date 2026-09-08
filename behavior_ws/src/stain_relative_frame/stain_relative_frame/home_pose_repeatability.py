#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step [0a] GATE: home-pose repeatability.

Drives the arm away from home and back `trials` times and measures the spread
of the settled TCP pose. The whole pipeline rests on the camera pose being the
same every time the home-pose frame is grabbed; if the arm does not return to
the same place, the homography from step [1] describes a scene that no longer
exists and every stain_origin downstream is wrong by an unknown amount.

  ros2 run stain_relative_frame home_pose_repeatability -- \
      --home_pose 420.3 345.5 211.9 -0.009 0.164 2.444 --trials 10

Gate: per-axis std <= 1.0 mm for x and y, and <= 0.5 deg for each rotation.
Exit code 0 = PASS, 1 = FAIL. FAIL stops the pipeline -- steps [1]+ are not
meaningful and must not be run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import load_config
from .report import table
from .ros_utils import collect, make_grabber, pose_stats, ptp_to


def _load_home_pose(args, cfg) -> np.ndarray:
    if args.home_pose:
        return np.asarray(args.home_pose, dtype=np.float64)
    if args.home_pose_file:
        p = Path(args.home_pose_file).expanduser()
        data = json.loads(p.read_text())
        return np.asarray(data["home_pose"], dtype=np.float64)
    raise SystemExit(
        "no home pose given: pass --home_pose x y z rx ry rz (mm, radians) or "
        "--home_pose_file, or --measure_only to just sample where the arm is now"
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="[0a] home pose repeatability gate")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--home_pose", type=float, nargs=6, default=None,
                    help="x y z rx ry rz -- xyz in mm, rotations in RADIANS")
    ap.add_argument("--home_pose_file", type=str, default=None)
    ap.add_argument("--away_pose", type=float, nargs=6, default=None,
                    help="pose to retreat to between trials (default: home + --away_offset in z)")
    ap.add_argument("--away_offset_mm", type=float, default=60.0)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--samples", type=int, default=None, help="pose samples per settled trial")
    ap.add_argument("--settle_sec", type=float, default=None)
    ap.add_argument("--velocity_mm_s", type=float, default=None)
    ap.add_argument("--pose_topic", type=str, default=None)
    ap.add_argument("--manual", action="store_true",
                    help="no PTP: prompt the operator to return the arm to home each trial")
    ap.add_argument("--measure_only", action="store_true",
                    help="sample the current pose once and print it (to seed --home_pose)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    trials = int(args.trials or cfg.home_repeat_trials)
    samples = int(args.samples or cfg.home_pose_samples)
    settle = float(args.settle_sec if args.settle_sec is not None else cfg.home_settle_sec)
    velocity = float(args.velocity_mm_s or cfg.ptp_velocity_mm_s)
    pose_topic = args.pose_topic or cfg.pose_topic
    out_path = Path(args.out) if args.out else cfg.path("home_pose_report_file")

    import rclpy

    rclpy.init(args=None)
    try:
        node = make_grabber(pose_topic=pose_topic, node_name="srf_home_pose_repeatability")

        if args.measure_only:
            node.clear()
            _, poses, ok = collect(node, n_poses=samples, timeout_sec=15.0)
            if poses.shape[0] < 1:
                print(f"[home] no pose samples on {pose_topic}", file=sys.stderr)
                return 1
            st = pose_stats(poses)
            print(f"[home] current pose mean (mm, rad) = {np.round(st['mean'], 4).tolist()}")
            print(f"[home] current pose mean (mm, deg) = {np.round(st['mean_deg'], 4).tolist()}")
            print("[home] pass this to --home_pose: " +
                  " ".join(f"{v:.4f}" for v in st["mean"]))
            return 0

        home = _load_home_pose(args, cfg)
        away = (np.asarray(args.away_pose, dtype=np.float64) if args.away_pose
                else home + np.array([0.0, 0.0, float(args.away_offset_mm), 0.0, 0.0, 0.0]))

        print(f"[home] home  = {np.round(home, 4).tolist()}  (mm, rad)")
        print(f"[home] away  = {np.round(away, 4).tolist()}")
        print(f"[home] trials={trials} samples/trial={samples} settle={settle}s "
              f"mode={'manual' if args.manual else 'PTP'}")

        means = []
        for t in range(trials):
            if args.manual:
                input(f"[home] trial {t + 1}/{trials}: move the arm AWAY, return it to "
                      f"home, then press ENTER... ")
            else:
                ptp_to(away, velocity, cfg.arm_service, logger=None)
                time.sleep(0.4)
                ptp_to(home, velocity, cfg.arm_service, logger=None)
            time.sleep(settle)

            node.clear()
            _, poses, ok = collect(node, n_poses=samples, timeout_sec=max(10.0, samples * 0.2))
            if poses.shape[0] < 3:
                print(f"[home] trial {t + 1}: only {poses.shape[0]} pose samples "
                      f"from {pose_topic}", file=sys.stderr)
                return 1
            m = poses.mean(axis=0)
            means.append(m)
            print(f"[home] trial {t + 1:2d}/{trials}: "
                  f"x={m[0]:8.3f} y={m[1]:8.3f} z={m[2]:8.3f} mm  "
                  f"rx={np.degrees(m[3]):7.3f} ry={np.degrees(m[4]):7.3f} "
                  f"rz={np.degrees(m[5]):7.3f} deg  (n={poses.shape[0]})")
    finally:
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        rclpy.shutdown()

    M = np.stack(means)                       # (trials, 6)
    std_lin = M[:, :3].std(axis=0)            # mm
    std_rot = np.degrees(_wrapped_std(M[:, 3:]))  # deg

    xy_ok = bool(std_lin[0] <= cfg.home_pose_xy_std_tol_mm and
                 std_lin[1] <= cfg.home_pose_xy_std_tol_mm)
    rot_ok = bool(np.all(std_rot <= cfg.home_pose_rot_std_tol_deg))
    passed = xy_ok and rot_ok

    rows = [
        {"axis": "x", "unit": "mm", "mean": f"{M[:, 0].mean():.3f}", "std": f"{std_lin[0]:.4f}",
         "tol": f"{cfg.home_pose_xy_std_tol_mm:.2f}",
         "verdict": "PASS" if std_lin[0] <= cfg.home_pose_xy_std_tol_mm else "FAIL"},
        {"axis": "y", "unit": "mm", "mean": f"{M[:, 1].mean():.3f}", "std": f"{std_lin[1]:.4f}",
         "tol": f"{cfg.home_pose_xy_std_tol_mm:.2f}",
         "verdict": "PASS" if std_lin[1] <= cfg.home_pose_xy_std_tol_mm else "FAIL"},
        {"axis": "z", "unit": "mm", "mean": f"{M[:, 2].mean():.3f}", "std": f"{std_lin[2]:.4f}",
         "tol": "-", "verdict": "info"},
    ]
    for i, name in enumerate(("rx", "ry", "rz")):
        rows.append({
            "axis": name, "unit": "deg",
            "mean": f"{np.degrees(M[:, 3 + i].mean()):.3f}", "std": f"{std_rot[i]:.4f}",
            "tol": f"{cfg.home_pose_rot_std_tol_deg:.2f}",
            "verdict": "PASS" if std_rot[i] <= cfg.home_pose_rot_std_tol_deg else "FAIL",
        })
    print()
    print(table(rows, title="[0a] home pose repeatability"))

    payload = {
        "step": "0a_home_pose_repeatability",
        "stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trials": trials,
        "samples_per_trial": samples,
        "pose_topic": pose_topic,
        "home_pose": home.tolist(),
        "trial_means": M.tolist(),
        "std_xyz_mm": std_lin.tolist(),
        "std_rot_deg": std_rot.tolist(),
        "tol_xy_mm": cfg.home_pose_xy_std_tol_mm,
        "tol_rot_deg": cfg.home_pose_rot_std_tol_deg,
        "passed": passed,
        "lighting_note": cfg.lighting_note,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[home] report -> {out_path}")

    if not passed:
        print("\n[home] GATE FAILED. The camera pose is not reproducible at home, so the "
              "step [1] homography would not describe the scene it is applied to.\n"
              "       STOP: do not run steps [1]-[5]. Fix the return-to-home first "
              "(controller settling, backlash, fixture).", file=sys.stderr)
        return 1
    print("\n[home] GATE PASSED -> proceed to [0b] clean_reference_capture")
    return 0


def _wrapped_std(angles_rad: np.ndarray) -> np.ndarray:
    """Circular std, so a sample straddling +/-pi does not read as huge spread."""
    a = np.asarray(angles_rad, dtype=np.float64)
    s = np.sin(a).mean(axis=0)
    c = np.cos(a).mean(axis=0)
    R = np.clip(np.hypot(s, c), 1e-12, 1.0)
    return np.sqrt(-2.0 * np.log(R))


if __name__ == "__main__":
    sys.exit(main())
