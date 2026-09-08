#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step [1c] GATE: fit H on 6 points, hold out 2, gate on <= 2 mm.

  ros2 run stain_relative_frame homography_fit
  ros2 run stain_relative_frame homography_fit -- --undistort   # retry path

On FAIL the tool says which retry to take, in the order the protocol
specifies: widen the point layout and re-measure, then lens undistortion.
Exit 0 = PASS (H written), 1 = FAIL (H written with passed=false; every
downstream loader refuses it).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import load_config
from .homography import apply_homography, fit_homography, save_homography
from .report import table


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="[1c] fit + gate the pixel->robot homography")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--points_file", type=str, default=None)
    ap.add_argument("--n_fit", type=int, default=None)
    ap.add_argument("--tol_mm", type=float, default=None)
    ap.add_argument("--ransac_thresh_px", type=float, default=None)
    ap.add_argument("--heldout", type=int, nargs="*", default=None,
                    help="pin the held-out point indices (default: keep the fit set spread out)")
    ap.add_argument("--undistort", action="store_true",
                    help="apply cv2 lens undistortion to the pixel points before fitting "
                         "(needs camera_matrix/dist_coeffs in the config)")
    ap.add_argument("--all_splits", action="store_true",
                    help="also report every possible hold-out split, as a robustness check")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    pts_path = Path(args.points_file) if args.points_file else cfg.artifacts / "homography_points.npz"
    out_path = Path(args.out) if args.out else cfg.path("homography_file")
    tol = float(args.tol_mm if args.tol_mm is not None else cfg.homography_reproj_tol_mm)
    n_fit = int(args.n_fit or cfg.homography_fit_points)
    ransac = float(args.ransac_thresh_px if args.ransac_thresh_px is not None
                   else cfg.homography_ransac_thresh_px)
    undistort = bool(args.undistort or cfg.undistort)

    if not pts_path.is_file():
        print(f"[fit] not found: {pts_path}\nRun [1a] homography_collect and "
              f"[1b] homography_pick_pixels first.", file=sys.stderr)
        return 1
    with np.load(pts_path, allow_pickle=False) as z:
        if "pixel_pts" not in z.files:
            print(f"[fit] {pts_path} has no pixel_pts -- run [1b] homography_pick_pixels",
                  file=sys.stderr)
            return 1
        pixels = np.asarray(z["pixel_pts"], dtype=np.float64)
        robot = np.asarray(z["robot_pts"], dtype=np.float64)
        method = str(z["pixel_method"]) if "pixel_method" in z.files else "unknown"

    res = fit_homography(
        pixels, robot, n_fit=n_fit, tol_mm=tol, ransac_thresh_px=ransac,
        heldout_indices=args.heldout if args.heldout else None,
        camera_matrix=cfg.camera_matrix, dist_coeffs=cfg.dist_coeffs, undistort=undistort,
    )

    rows = []
    pred = apply_homography(res.H, pixels)
    for i in range(pixels.shape[0]):
        role = "held-out" if i in res.heldout_indices else "fit"
        err = float(np.linalg.norm(pred[i] - robot[i]))
        rows.append({
            "i": i, "role": role,
            "u": f"{pixels[i, 0]:.2f}", "v": f"{pixels[i, 1]:.2f}",
            "robot_x": f"{robot[i, 0]:.2f}", "robot_y": f"{robot[i, 1]:.2f}",
            "pred_x": f"{pred[i, 0]:.2f}", "pred_y": f"{pred[i, 1]:.2f}",
            "err_mm": f"{err:.3f}",
            "verdict": ("PASS" if err <= tol else "FAIL") if role == "held-out" else "",
        })
    print(table(rows, title=f"[1c] homography reprojection (pixel method: {method}"
                            f"{', undistorted' if undistort else ''})"))
    print()
    print(f"[fit] fit points     : {res.fit_indices}")
    print(f"[fit] held-out points: {res.heldout_indices}")
    print(f"[fit] fit error      : max={max(res.fit_errors_mm):.3f}mm "
          f"mean={np.mean(res.fit_errors_mm):.3f}mm")
    print(f"[fit] held-out error : {res.summary()}")

    if args.all_splits:
        from itertools import combinations
        n = pixels.shape[0]
        alt = []
        for held in combinations(range(n), n - n_fit):
            try:
                r = fit_homography(pixels, robot, n_fit=n_fit, tol_mm=tol,
                                   ransac_thresh_px=ransac, heldout_indices=held,
                                   camera_matrix=cfg.camera_matrix,
                                   dist_coeffs=cfg.dist_coeffs, undistort=undistort)
            except Exception:  # noqa: BLE001 -- degenerate fit set
                continue
            alt.append({"heldout": str(list(held)),
                        "max_err_mm": f"{r.max_heldout_error_mm:.3f}",
                        "verdict": "PASS" if r.passed else "FAIL"})
        alt.sort(key=lambda r: -float(r["max_err_mm"]))
        print()
        print(table(alt, title=f"[1c] all {len(alt)} hold-out splits (worst first)", max_rows=15))

    meta = {
        "points_file": str(pts_path),
        "pixel_method": method,
        "n_points": int(pixels.shape[0]),
        "n_fit": n_fit,
        "ransac_thresh_px": ransac,
        "pixel_pts": pixels.tolist(),
        "robot_pts": robot.tolist(),
    }
    saved = save_homography(out_path, res, meta=meta)
    print(f"[fit] saved -> {saved}")

    if not res.passed:
        print(f"\n[fit] GATE FAILED: held-out error {res.max_heldout_error_mm:.3f}mm > {tol}mm.\n"
              "      Retry in this order:\n"
              "      1. Spread the 8 points further apart across the work area and\n"
              "         re-run [1a] homography_collect (a tight cluster is the usual cause).\n"
              "      2. If it still fails, calibrate the lens, put camera_matrix and\n"
              "         dist_coeffs in the config, and re-run with --undistort.\n"
              "      Downstream loaders will refuse this H until it passes.", file=sys.stderr)
        return 1
    print("\n[fit] GATE PASSED -> proceed to [2] stain_origin_offline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
