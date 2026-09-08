#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step [1a], marker-free variant: build the pixel<->robot correspondences by
JOGGING the robot instead of touching physical points.

Why this exists
---------------
The rig is eye-in-hand (camera on the tool, 250mm/45deg mount). Two facts
that variant relies on:

  * The tool tip is rigidly co-mounted with the camera, so it projects to a
    FIXED pixel `tip_px` in every frame -- and `/ur10skku/currentP` always
    reports that tip's position in robot base mm. That pair
    (tip_px  <->  currentP_xy) is one free, marker-free correspondence.
  * The workpiece surface is (locally) planar and the camera translates
    parallel to it, so a small base-XY jog pans the workpiece image by a pure
    translation -- measurable by phase correlation on the surface texture,
    no marker in view.

Procedure
---------
  1. Arm at the HOME pose. Grab and average the home image. Fix `tip_px`.
  2. PTP the arm across a grid of base-XY offsets at constant Z and constant
     orientation, staying inside the demonstrated episode envelope.
  3. At each grid pose: read the true base XY from the pose topic, grab the
     image, and phase-correlate it against the home image over several
     surface patches -> scene pan (su, sv) in pixels.
  4. A world point that sits at the tool tip after jog i (base
     `currentP_home_xy + (dx_i, dy_i)`) appears at `tip_px` in that jogged
     frame, hence at `tip_px - (su_i, sv_i)` in the HOME frame. That is the
     correspondence: home-image pixel  <->  robot base XY.
  5. Write `homography_points.npz` in the exact schema `homography_fit`
     expects (`home_image`, `robot_pts`, `pixel_pts`, `pixel_method`), then
     optionally run the [1c] gate.

  ros2 run stain_relative_frame homography_jog_calibrate -- --run_fit

NO ROTATION, NO SURFACE CONTACT. The arm only translates in base XY at a Z
that the episodes themselves reach; nothing is touched or added to the scene.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import load_config
from .report import table
from .ros_utils import collect, make_grabber, ptp_to

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# The tool tip's fixed image location on this eye-in-hand rig. Matches
# inference_core.py's flow_vector_overlay_tcp_center_{x,y} (2026-08-08 mount).
DEFAULT_TIP_PX = (253.0, 120.0)

# Demonstrated episode envelope (union of the 0deg / 90deg polishing runs in
# datasets/polishing/single_cam/20260821_*). The grid is clamped to this so
# every calibration pose is inside space the arm has already worked in.
DEFAULT_SAFE_BOX = (378.0, 500.0, 314.0, 424.0)   # x_min, x_max, y_min, y_max


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb)
    if a.ndim == 3:
        return a.astype(np.float32).mean(axis=-1)
    return a.astype(np.float32)


def _feature_mask(shape, feat_roi, tool_box) -> np.ndarray:
    m = np.zeros(shape, dtype=np.uint8)
    u0, v0, u1, v1 = (int(x) for x in feat_roi)
    m[v0:v1, u0:u1] = 255
    tu0, tv0, tu1, tv1 = (int(x) for x in tool_box)
    m[tv0:tv1, tu0:tu1] = 0            # drop the co-mounted tool / splash guard
    return m


def surface_warp(
    ref_gray: np.ndarray,
    cur_gray: np.ndarray,
    feat_roi: Tuple[int, int, int, int],
    tool_box: Tuple[int, int, int, int],
    max_features: int = 600,
    ransac_px: float = 2.0,
) -> Tuple[Optional[np.ndarray], int, float, float]:
    """Homography G that warps `ref_gray` onto `cur_gray` for the dominant
    world-fixed plane (the workpiece surface).

    Sparse LK optical flow on corner features, then `cv2.findHomography` with
    RANSAC -- so the tool (co-mounted, near-zero flow) and any second plane
    (the bracket riser) fall out as outliers. A planar surface between two
    views of a translating camera is related *exactly* by a homography, so
    this captures the real scale / perspective change, not just a shift.

    Returns (G 3x3 or None, n_inliers, inlier_rms_px, median_flow_px).
    """
    if cv2 is None:
        raise ImportError("cv2 is required")
    ref8 = ref_gray.astype(np.uint8)
    cur8 = cur_gray.astype(np.uint8)
    mask = _feature_mask(ref_gray.shape, feat_roi, tool_box)
    p0 = cv2.goodFeaturesToTrack(ref8, max_features, 0.01, 6, mask=mask)
    if p0 is None or len(p0) < 12:
        return None, 0, float("nan"), float("nan")
    p1, st, _ = cv2.calcOpticalFlowPyrLK(ref8, cur8, p0, None,
                                         winSize=(21, 21), maxLevel=3)
    p0b, st2, _ = cv2.calcOpticalFlowPyrLK(cur8, ref8, p1, None,
                                           winSize=(21, 21), maxLevel=3)
    fb = np.linalg.norm((p0 - p0b).reshape(-1, 2), axis=1)      # fwd-bwd check
    good = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < 1.0)
    a, b = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
    if len(a) < 12:
        return None, 0, float("nan"), float("nan")
    G, inl = cv2.findHomography(a, b, cv2.RANSAC, ransac_px)
    if G is None:
        return None, 0, float("nan"), float("nan")
    inl = inl.reshape(-1).astype(bool)
    proj = cv2.perspectiveTransform(a[inl].reshape(-1, 1, 2), G).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj - b[inl]) ** 2, axis=1))))
    med_flow = float(np.median(np.linalg.norm(b[inl] - a[inl], axis=1)))
    return np.asarray(G, dtype=np.float64), int(inl.sum()), rms, med_flow


def _click_tip(home_rgb: np.ndarray, default: Tuple[float, float]) -> Tuple[float, float]:
    if cv2 is None:
        return default
    try:
        bgr = cv2.cvtColor(home_rgb, cv2.COLOR_RGB2BGR)
        state = {"pt": default, "zoom": 3.0}
        win = "click the TOOL TIP (where currentP is) | ENTER=accept | q=default"

        def on_mouse(event, x, y, flags, _):
            if event == cv2.EVENT_LBUTTONDOWN:
                state["pt"] = (x / state["zoom"], y / state["zoom"])
            elif event == cv2.EVENT_MOUSEWHEEL:
                state["zoom"] = float(np.clip(state["zoom"] * (1.25 if flags > 0 else 0.8), 1.0, 12.0))

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, on_mouse)
        while True:
            z = state["zoom"]
            out = cv2.resize(bgr, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
            p = (int(round(state["pt"][0] * z)), int(round(state["pt"][1] * z)))
            cv2.drawMarker(out, p, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.imshow(win, out)
            k = cv2.waitKey(20) & 0xFF
            if k in (13, ord("q")):
                break
        cv2.destroyAllWindows()
        return tuple(float(v) for v in state["pt"])
    except Exception as exc:  # noqa: BLE001 -- headless / no display
        print(f"[jog] --click_tip unavailable ({exc}); using {default}", file=sys.stderr)
        return default


def _grid_offsets(safe_box, nx, ny, center_xy) -> List[Tuple[float, float]]:
    """Boustrophedon (snake) order, entered from the corner nearest home so
    every move is between adjacent grid nodes except the first."""
    x_min, x_max, y_min, y_max = safe_box
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    if abs(center_xy[0] - x_max) < abs(center_xy[0] - x_min):
        xs = xs[::-1]
    if abs(center_xy[1] - y_max) < abs(center_xy[1] - y_min):
        ys = ys[::-1]
    pts: List[Tuple[float, float]] = []
    for j, y in enumerate(ys):
        row = xs if j % 2 == 0 else xs[::-1]
        for x in row:
            pts.append((float(x), float(y)))
    return pts


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="[1a] marker-free homography correspondences via robot jog")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--nx", type=int, default=7)
    ap.add_argument("--ny", type=int, default=7)
    ap.add_argument("--safe_box", type=float, nargs=4, default=list(DEFAULT_SAFE_BOX),
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="clamp every grid pose into this base-XY box (episode envelope)")
    ap.add_argument("--grid_z", type=float, default=None,
                    help="Z (mm) for every grid pose; default = measured home Z")
    ap.add_argument("--velocity_mm_s", type=float, default=None)
    ap.add_argument("--settle_sec", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=30, help="pose samples averaged per grid pose")
    ap.add_argument("--frames", type=int, default=6, help="camera frames averaged per grid pose")
    ap.add_argument("--tip_px", type=float, nargs=2, default=None, metavar=("U", "V"))
    ap.add_argument("--click_tip", action="store_true", help="click the tool tip on the home image")
    ap.add_argument("--feat_roi", type=int, nargs=4, default=[125, 55, 420, 236],
                    metavar=("U0", "V0", "U1", "V1"),
                    help="image region the workpiece surface occupies (corner features taken here)")
    ap.add_argument("--tool_box", type=int, nargs=4, default=[205, 12, 348, 216],
                    metavar=("U0", "V0", "U1", "V1"),
                    help="image box of the co-mounted tool / splash guard, excluded from tracking")
    ap.add_argument("--ransac_px", type=float, default=2.0)
    ap.add_argument("--min_inliers", type=int, default=40)
    ap.add_argument("--warp_rms_warn_px", type=float, default=1.0)
    ap.add_argument("--min_ok_frac", type=float, default=0.7,
                    help="abort if fewer than this fraction of grid poses yield a usable warp")
    ap.add_argument("--image_topic", type=str, default=None)
    ap.add_argument("--pose_topic", type=str, default=None)
    ap.add_argument("--home_xy_tol_mm", type=float, default=15.0)
    ap.add_argument("--skip_home_check", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="print the grid and exit, no motion")
    ap.add_argument("--save_frames", type=str, default=None,
                    help="also dump every grid frame + pose to this .npz (offline re-tuning)")
    ap.add_argument("--run_fit", action="store_true", help="run the [1c] homography_fit gate afterwards")
    ap.add_argument("--n_holdout", type=int, default=6)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    if cv2 is None:
        print("[jog] OpenCV is required", file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    image_topic = args.image_topic or cfg.image_topic
    pose_topic = args.pose_topic or cfg.pose_topic
    velocity = float(args.velocity_mm_s or cfg.ptp_velocity_mm_s)
    out_path = Path(args.out) if args.out else cfg.artifacts / "homography_points.npz"
    feat_roi = tuple(int(v) for v in args.feat_roi)
    tool_box = tuple(int(v) for v in args.tool_box)
    safe_box = tuple(float(v) for v in args.safe_box)

    import rclpy

    rclpy.init(args=None)
    home6 = None
    robot_pts: List[np.ndarray] = []
    pixel_pts: List[np.ndarray] = []
    poses6: List[np.ndarray] = []
    pose_stds: List[np.ndarray] = []
    pan_rows: List[dict] = []
    try:
        node = make_grabber(image_topic=image_topic, pose_topic=pose_topic,
                            node_name="srf_homography_jog_calibrate")

        # ---- home pose + home image ------------------------------------
        node.clear()
        _, hp, _ = collect(node, n_poses=max(args.samples, 20), timeout_sec=15.0)
        if hp.shape[0] < 5:
            print(f"[jog] no pose samples on {pose_topic}", file=sys.stderr)
            return 1
        home6 = hp.mean(axis=0)
        grid_z = float(args.grid_z if args.grid_z is not None else home6[2])
        print(f"[jog] home pose (mm, rad) = {np.round(home6, 4).tolist()}")
        print(f"[jog] grid Z = {grid_z:.2f} mm   orientation held at home rpy")

        if not args.skip_home_check:
            box_cx, box_cy = (safe_box[0] + safe_box[1]) / 2, (safe_box[2] + safe_box[3]) / 2
            if abs(home6[0] - box_cx) > 90 or abs(home6[1] - box_cy) > 90:
                print(f"[jog] home XY ({home6[0]:.1f},{home6[1]:.1f}) is far from the "
                      f"safe box centre ({box_cx:.1f},{box_cy:.1f}); is the arm homed? "
                      f"pass --skip_home_check to override.", file=sys.stderr)
                return 1

        node.clear()
        frames, _, _ = collect(node, n_frames=max(args.frames, 4), timeout_sec=15.0)
        if len(frames) < 3:
            print(f"[jog] no frames from {image_topic}", file=sys.stderr)
            return 1
        home_rgb = np.clip(np.stack(frames[:max(args.frames, 4)]).astype(np.float32).mean(0),
                           0, 255).astype(np.uint8)
        home_gray = _to_gray(home_rgb)
        H_img, W_img = home_gray.shape
        if not (0 <= feat_roi[0] < feat_roi[2] <= W_img and 0 <= feat_roi[1] < feat_roi[3] <= H_img):
            print(f"[jog] feat_roi {feat_roi} outside image {W_img}x{H_img}", file=sys.stderr)
            return 1

        tip_px = tuple(args.tip_px) if args.tip_px else (
            _click_tip(home_rgb, DEFAULT_TIP_PX) if args.click_tip else DEFAULT_TIP_PX)
        tip_px = np.asarray(tip_px, dtype=np.float64)
        tip_hom = tip_px.reshape(1, 1, 2).astype(np.float64)
        print(f"[jog] tool-tip pixel (anchor) = ({tip_px[0]:.1f}, {tip_px[1]:.1f})")

        # self-check: warp(home, home) must be ~identity
        G0, ni0, rms0, _ = surface_warp(home_gray, home_gray, feat_roi, tool_box,
                                        ransac_px=args.ransac_px)
        off0 = 0.0 if G0 is None else float(np.linalg.norm(
            cv2.perspectiveTransform(tip_hom, G0).reshape(2) - tip_px))
        print(f"[jog] warp self-check: {ni0} inliers, rms {rms0:.3f}px, "
              f"tip maps {off0:.3f}px from itself (expect ~0)")

        grid = _grid_offsets(safe_box, args.nx, args.ny, (home6[0], home6[1]))
        print(f"\n[jog] {len(grid)} grid poses over box x[{safe_box[0]:.0f},{safe_box[1]:.0f}] "
              f"y[{safe_box[2]:.0f},{safe_box[3]:.0f}] at Z={grid_z:.1f}")
        if args.dry_run:
            for i, (x, y) in enumerate(grid):
                print(f"  {i:2d}  x={x:7.2f}  y={y:7.2f}")
            return 0

        # ---- drive the grid ------------------------------------------
        n_skipped = 0
        raw_frames: List[np.ndarray] = []
        raw_poses: List[np.ndarray] = []
        for i, (gx, gy) in enumerate(grid):
            target = [gx, gy, grid_z, home6[3], home6[4], home6[5]]
            ptp_to(target, velocity, cfg.arm_service, logger=None)
            time.sleep(args.settle_sec)

            node.clear()
            fr, pp, _ = collect(node, n_frames=args.frames, n_poses=args.samples,
                                timeout_sec=max(12.0, args.samples * 0.2))
            if pp.shape[0] < 5 or len(fr) < 3:
                print(f"[jog] pose {i}: only {pp.shape[0]} poses / {len(fr)} frames",
                      file=sys.stderr)
                return 1
            m, s = pp.mean(axis=0), pp.std(axis=0)
            cur_rgb = np.clip(np.stack(fr[:args.frames]).astype(np.float32).mean(0),
                              0, 255).astype(np.uint8)
            cur_gray = _to_gray(cur_rgb)
            if args.save_frames:
                raw_frames.append(cur_rgb)
                raw_poses.append(m)

            G, n_inl, rms, med_flow = surface_warp(
                home_gray, cur_gray, feat_roi, tool_box, ransac_px=args.ransac_px)
            bad = G is None or n_inl < args.min_inliers or not np.isfinite(rms)
            if bad or rms > 3.0:
                n_skipped += 1
                print(f"[jog] {i:2d}/{len(grid) - 1}: SKIP  inliers={n_inl} rms={rms:.2f} "
                      f"(x={m[0]:.1f} y={m[1]:.1f})", file=sys.stderr)
                continue

            # world point currently under the tip -> where it sits in the HOME image
            px_home = cv2.perspectiveTransform(tip_hom, np.linalg.inv(G)).reshape(2)

            robot_pts.append(m[:2])
            pixel_pts.append(px_home)
            poses6.append(m)
            pose_stds.append(s)
            flag = "  <-- warp rms high" if rms > args.warp_rms_warn_px else ""
            flagm = "  <-- MOVING" if max(s[0], s[1]) > 0.5 else ""
            pan_rows.append({
                "i": i, "x_mm": f"{m[0]:.2f}", "y_mm": f"{m[1]:.2f}",
                "flow_px": f"{med_flow:.1f}", "inliers": n_inl,
                "rms_px": f"{rms:.2f}",
                "px_u": f"{px_home[0]:.1f}", "px_v": f"{px_home[1]:.1f}",
            })
            print(f"[jog] {i:2d}/{len(grid) - 1}: x={m[0]:7.2f} y={m[1]:7.2f}  "
                  f"flow={med_flow:5.1f}px  inl={n_inl:3d}  rms={rms:.2f}  "
                  f"px=({px_home[0]:6.1f},{px_home[1]:6.1f}){flag}{flagm}")

        # ---- return home --------------------------------------------
        ptp_to(list(home6), velocity, cfg.arm_service, logger=None)

        if args.save_frames and raw_frames:
            fp = Path(args.save_frames)
            fp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(fp, home_image=home_rgb, frames=np.stack(raw_frames),
                                poses6=np.stack(raw_poses), home_pose6=home6,
                                tip_px=tip_px, grid_z=np.float64(grid_z))
            print(f"[jog] raw frames -> {fp}")

        if len(robot_pts) < max(8, args.min_ok_frac * len(grid)):
            print(f"[jog] only {len(robot_pts)}/{len(grid)} poses gave a usable warp "
                  f"({n_skipped} skipped). The workpiece surface is not the dominant "
                  f"plane in --feat_roi, or --tool_box is letting the tool in.",
                  file=sys.stderr)
            return 1
    finally:
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        rclpy.shutdown()

    R = np.stack(robot_pts)
    P = np.stack(pixel_pts)
    span = R.max(axis=0) - R.min(axis=0)
    u_span, v_span = float(np.ptp(P[:, 0])), float(np.ptp(P[:, 1]))
    print()
    print(table(pan_rows, title="[1a-jog] per-pose warp"))
    print(f"\n[jog] robot span x={span[0]:.1f}mm  y={span[1]:.1f}mm   "
          f"pixel span u={u_span:.1f}  v={v_span:.1f}")

    # crude linear consistency: fit P ~ A R + b, report residual (a sanity
    # number only; homography_fit does the real projective fit + gate).
    A, res, *_ = np.linalg.lstsq(
        np.c_[R - R.mean(0), np.ones(len(R))],
        P - P.mean(0), rcond=None)
    pred = np.c_[R - R.mean(0), np.ones(len(R))] @ A + P.mean(0)
    lin_px = float(np.sqrt(np.mean(np.sum((pred - P) ** 2, axis=1))))
    print(f"[jog] affine pixel<->mm residual RMS = {lin_px:.2f}px "
          f"(~{lin_px * span[0] / max(u_span, 1e-6):.2f}mm)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        home_image=home_rgb,
        robot_pts=R,
        pixel_pts=P,
        pixel_method="jog_flow_homography",
        robot_pose6=np.stack(poses6),
        pose_std=np.stack(pose_stds),
        home_pose6=home6,
        tip_px=tip_px,
        feat_roi=np.asarray(feat_roi, dtype=np.int64),
        tool_box=np.asarray(tool_box, dtype=np.int64),
        grid_z=np.float64(grid_z),
        image_topic=image_topic,
        pose_topic=pose_topic,
        stamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    print(f"[jog] saved -> {out_path}  ({len(R)} correspondences)")

    if not args.run_fit:
        n_fit = max(4, len(R) - args.n_holdout)
        print(f"[jog] next: ros2 run stain_relative_frame homography_fit -- "
              f"--n_fit {n_fit}")
        return 0

    from . import homography_fit
    n_fit = max(4, len(R) - args.n_holdout)
    print("\n" + "=" * 78 + f"\n[1c] homography_fit  (fit {n_fit}, hold out {len(R) - n_fit})\n"
          + "=" * 78)
    rc = homography_fit.main(
        ["--points_file", str(out_path), "--n_fit", str(n_fit)]
        + (["--config", args.config] if args.config else []))
    return rc


if __name__ == "__main__":
    sys.exit(main())
