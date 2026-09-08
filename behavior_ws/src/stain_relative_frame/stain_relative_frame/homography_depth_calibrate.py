#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step [1], marker-free + touch-free variant using the RealSense DEPTH stream.

Only works when the VR camera is launched with depth on (``rsv`` on this PC,
or ``rs_vr_pcd``): needs ``/realsense/vr/aligned_depth_to_color/image_raw``.

Idea
----
The workpiece surface is brushed aluminium -- no visual texture, so the RGB
optical-flow variant (`homography_jog_calibrate`) locks onto the fixture
below it. Depth does not need texture: the plate is a clean plane in the
depth map (sub-mm fit). So:

  1. Grab aligned depth at the HOME pose and at a small ring of base-XY jogs
     (constant Z, constant orientation, inside the episode envelope).
  2. Each jog translates the eye-in-hand camera by a KNOWN base vector. ICP
     on the depth clouds measures the same translation in the CAMERA frame.
     `t_i^cam = -R_bc . jog_i^base` over the ring solves the camera->base
     rotation `R_bc` (Procrustes).
  3. The plate is horizontal in base at a known Z0 (the polishing-contact
     height). The tool tip projects to a fixed pixel and `/ur10skku/currentP`
     gives its base XY. Tip ray  ->  plane-in-depth intersection  +  tip
     anchor  +  Z0  closes the last translation d.o.f.  ->  full camera->base
     transform `T_cb` at the home pose.
  4. Re-project every home-image pixel through `T_cb` onto the Z=Z0 plane:
     that map is a homography. Save it as the usual `homography.json`, so
     stages [2]-[5] are unchanged.

Nothing is touched, nothing is added to the scene. Rotation is never
applied to the trajectory -- this only produces the pixel->base-XY map.

  ros2 run stain_relative_frame homography_depth_calibrate -- --z0 167 --run_fit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import load_config
from .homography import fit_homography, plane_homography, save_homography
from .report import table

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

DEFAULT_TIP_PX = (253.0, 120.0)
DEFAULT_SAFE_BOX = (378.0, 500.0, 314.0, 424.0)


# --------------------------------------------------------------------- geometry

def backproject(u, v, d, K) -> np.ndarray:
    """(u, v) pixels + depth d (mm) -> (N, 3) camera-frame points (mm)."""
    fx, fy, cx, cy = K
    u = np.asarray(u, float); v = np.asarray(v, float); d = np.asarray(d, float)
    return np.stack([(u - cx) / fx * d, (v - cy) / fy * d, d], axis=-1)


def project(X, K) -> np.ndarray:
    fx, fy, cx, cy = K
    X = np.asarray(X, float).reshape(-1, 3)
    z = np.where(np.abs(X[:, 2]) < 1e-6, 1e-6, X[:, 2])
    return np.stack([fx * X[:, 0] / z + cx, fy * X[:, 1] / z + cy], axis=-1)


def fit_plane(pts: np.ndarray, thresh_mm: float = 2.0, iters: int = 6
              ) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Robust total-least-squares plane fit. Returns (n, d, inlier_mask, rms).

    Plane is  n . X = d,  ||n|| = 1,  n oriented toward the camera (n_z < 0).
    """
    P = np.asarray(pts, float).reshape(-1, 3)
    keep = np.ones(len(P), bool)
    n = np.array([0.0, 0.0, -1.0]); d = 0.0
    for _ in range(iters):
        Q = P[keep]
        c = Q.mean(0)
        _, _, Vt = np.linalg.svd(Q - c, full_matrices=False)
        n = Vt[-1]
        d = float(n @ c)
        r = P @ n - d
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        keep = np.abs(r) < max(thresh_mm, 2.5 * s)
    if n[2] > 0:
        n, d = -n, -d
    rms = float(np.sqrt(np.mean((P[keep] @ n - d) ** 2)))
    return n, d, keep, rms


def _depth_mask(D, tool_box, roi, max_depth_mm: float = 560.0) -> np.ndarray:
    m = (D > 100.0) & (D < max_depth_mm)          # drop background / dropouts
    u0, v0, u1, v1 = (int(x) for x in roi)
    box = np.zeros(D.shape, bool); box[v0:v1, u0:u1] = True
    m &= box
    tu0, tv0, tu1, tv1 = (int(x) for x in tool_box)
    m[tv0:tv1, tu0:tu1] = False
    return m


def _smooth_depth(D):
    """Kill the ~1 mm depth quantisation banding before differentiating, so
    surface normals reflect real geometry and not stair-steps."""
    if cv2 is None:
        return D
    Df = D.astype(np.float32).copy()
    Df[Df <= 0] = np.nan
    filled = np.nan_to_num(Df, nan=float(np.nanmedian(Df)))
    sm = cv2.bilateralFilter(filled, 7, 30.0, 6.0)
    sm[np.isnan(Df)] = 0.0
    return sm


def _depth_normals(Ds, K):
    """`Ds` must already be smoothed (see `_smooth_depth`)."""
    H, W = Ds.shape
    gu, gv = np.meshgrid(np.arange(W), np.arange(H))
    X1full = backproject(gu.ravel(), gv.ravel(), Ds.ravel(), K).reshape(H, W, 3)
    du = np.zeros_like(X1full); dv = np.zeros_like(X1full)
    du[:, 1:-1] = X1full[:, 2:] - X1full[:, :-2]
    dv[1:-1, :] = X1full[2:, :] - X1full[:-2, :]
    nrm = np.cross(du, dv)
    valid = Ds > 0
    bad = ~(valid & np.roll(valid, 1, 1) & np.roll(valid, -1, 1)
            & np.roll(valid, 1, 0) & np.roll(valid, -1, 0))[..., None]
    nrm = np.where(bad, np.array([0.0, 0.0, -1.0]), nrm)
    nrm /= (np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9)
    nrm[nrm[..., 2] > 0] *= -1
    grad = np.hypot(*np.gradient(Ds))
    return X1full, nrm, grad


def icp_translation(D0, D1, K, mask, iters=30, huber_mm=3.0
                    ) -> Tuple[np.ndarray, float, int]:
    """Projective point-to-plane ICP for a pure camera translation:
    scene(D1) ~= scene(D0) + t  (camera mm).

    The big flat plate is blind to translation parallel to it, so its
    point-to-plane rows (all sharing one normal) only pin one d.o.f. The
    remaining two come from "structured" pixels -- edges, holes, the fixture,
    where the local normal tilts away from the dominant plane. Those are a
    minority, so they are up-weighted (planar pixels held at `flat_w`) to
    keep the fit from sliding along the plate.
    """
    H, W = D0.shape
    D0s, D1 = _smooth_depth(D0), _smooth_depth(D1)
    vs, us = np.nonzero(mask)
    X0 = backproject(us, vs, D0s[vs, us], K)
    _, nrm, grad = _depth_normals(D1, K)

    dom = np.median(nrm[mask], axis=0)
    dom /= np.linalg.norm(dom) + 1e-9
    flat_img = (np.abs(nrm @ dom) > np.cos(np.deg2rad(10.0))) & (grad < 3.0)
    flat_w = 0.05

    t = np.zeros(3)
    rms = np.nan
    n_in = 0
    for _ in range(iters):
        Xt = X0 + t
        uv = project(Xt, K)
        ui = np.round(uv[:, 0]).astype(int); vi = np.round(uv[:, 1]).astype(int)
        ok = (ui >= 1) & (ui < W - 1) & (vi >= 1) & (vi < H - 1)
        ui, vi = ui[ok], vi[ok]
        d1 = D1[vi, ui]
        good = d1 > 0
        ui, vi, d1 = ui[good], vi[good], d1[good]
        Xt_ok = Xt[ok][good]
        X1 = backproject(ui, vi, d1, K)
        nk = nrm[vi, ui]
        rp = np.einsum("ij,ij->i", nk, X1 - Xt_ok)             # point-to-plane
        if len(rp) < 30:
            return t, np.nan, len(rp)

        w = np.minimum(1.0, huber_mm / np.maximum(np.abs(rp), 1e-6))    # Huber
        w = w * np.where(flat_img[vi, ui], flat_w, 1.0)                 # up-weight structure
        sw = np.sqrt(w)
        dt, *_ = np.linalg.lstsq(nk * sw[:, None], rp * sw, rcond=None)
        t = t + dt
        inl = np.abs(rp) < huber_mm
        rms = float(np.sqrt(np.mean(rp[inl] ** 2))) if inl.any() else float(np.sqrt(np.mean(rp ** 2)))
        n_in = int(inl.sum())
        if np.linalg.norm(dt) < 5e-4:
            break
    return t, rms, n_in


def solve_base_rotation(dxdy: np.ndarray, T: np.ndarray
                        ) -> Tuple[np.ndarray, float, float, float]:
    """t_i^cam = dx_i * a + dy_i * b  with  a = -R_bc e_x, b = -R_bc e_y.

    Returns (R_bc, |a|, |b|, ortho_dev_deg) where ortho_dev_deg is how far the
    recovered a, b are from perpendicular. |a|,|b| ~ 1 validates the depth
    scale.
    """
    M = np.asarray(dxdy, float)                # (n, 2)
    ab, *_ = np.linalg.lstsq(M, np.asarray(T, float), rcond=None)   # (2, 3)
    a, b = ab[0], ab[1]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    xc, yc = -a / na, -b / nb
    ortho = float(abs(90.0 - np.degrees(np.arccos(np.clip(abs(xc @ yc), -1, 1)))))
    zc = np.cross(xc, yc); zc /= np.linalg.norm(zc)
    yc = np.cross(zc, xc)
    R_bc = np.stack([xc, yc, zc], axis=1)      # columns = base axes in cam frame
    U, _, Vt = np.linalg.svd(R_bc)
    R_bc = U @ Vt
    if np.linalg.det(R_bc) < 0:
        U[:, -1] *= -1
        R_bc = U @ Vt
    return R_bc, float(na), float(nb), ortho


def solve_base_rotation_planar(n_cam, dxdy, T, ) -> Tuple[np.ndarray, float, float, float]:
    """Both base X and base Y lie in the (horizontal) plate plane, so ICP's
    plate points give NO rotation constraint -- only the fixture / edges do,
    and they favour one image axis. Instead: take the vertical (2 d.o.f.)
    from the rock-solid plate normal, and fit only the remaining YAW + scale
    as a 2-D similarity between the base jogs and their in-plane camera
    translations.

    Returns (R_bc, scale, yaw_residual_mm, plate_normal_component_rms_mm).
    """
    zc = -np.asarray(n_cam, float)
    zc /= np.linalg.norm(zc)                       # base +Z expressed in cam frame
    e1 = np.cross(zc, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-3:
        e1 = np.cross(zc, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(zc, e1)

    T = np.asarray(T, float)
    P = np.c_[T @ e1, T @ e2]                      # camera translation, in-plane 2-D
    normal_comp = T @ zc                           # should be ~0 for horizontal plate
    S = np.asarray(dxdy, float)                    # base jogs (dx, dy)

    M = P.T @ S
    U, D, Vt = np.linalg.svd(M)
    R2 = U @ Vt
    if np.linalg.det(R2) < 0:
        U[:, -1] *= -1
        R2 = U @ Vt
    scale = float(D.sum() / (S ** 2).sum())
    resid = float(np.sqrt(np.mean(np.sum((P - scale * (S @ R2.T)) ** 2, axis=1))))

    # base axes in cam frame:  jog (1,0)->cam -scale*(R2 col0 in e1/e2 basis)
    xc = -(R2[0, 0] * e1 + R2[1, 0] * e2)
    yc = -(R2[0, 1] * e1 + R2[1, 1] * e2)
    xc /= np.linalg.norm(xc); yc /= np.linalg.norm(yc)
    zc2 = np.cross(xc, yc); zc2 /= np.linalg.norm(zc2)
    yc = np.cross(zc2, xc)
    R_bc = np.stack([xc, yc, zc2], axis=1)
    Uu, _, Vv = np.linalg.svd(R_bc)
    R_bc = Uu @ Vv
    if np.linalg.det(R_bc) < 0:
        Uu[:, -1] *= -1
        R_bc = Uu @ Vv
    return R_bc, scale, resid, float(np.sqrt(np.mean(normal_comp ** 2)))


def solve_tcb(R_cb, K, tip_px, n_cam, d_cam, home_xyz, z0
              ) -> Tuple[np.ndarray, float]:
    fx, fy, cx, cy = K
    d_tip = np.array([(tip_px[0] - cx) / fx, (tip_px[1] - cy) / fy, 1.0])
    d_tip /= np.linalg.norm(d_tip)
    denom = float(n_cam @ d_tip)
    if abs(denom) < 1e-6:
        raise RuntimeError("tip ray is parallel to the plate plane")
    s_hit = d_cam / denom                       # tip line-of-sight hits plate here
    u_base = R_cb @ d_tip                        # tip ray direction in base frame
    if abs(u_base[2]) < 1e-6:
        raise RuntimeError("tip ray has no vertical component in base frame")
    z_tip = s_hit - (z0 - home_xyz[2]) / u_base[2]
    t_cb = np.asarray(home_xyz, float) - z_tip * u_base
    return t_cb, float(z_tip)


def pixels_to_plane(px: np.ndarray, K, R_cb, t_cb, z0) -> np.ndarray:
    """Home-image pixels -> base XY where their camera ray meets Z = z0."""
    fx, fy, cx, cy = K
    px = np.asarray(px, float).reshape(-1, 2)
    r_cam = np.stack([(px[:, 0] - cx) / fx, (px[:, 1] - cy) / fy,
                      np.ones(len(px))], axis=-1)
    r_base = r_cam @ R_cb.T
    s = (z0 - t_cb[2]) / r_base[:, 2]
    return t_cb[None, :] + s[:, None] * r_base


# ------------------------------------------------------------------------ ROS

def _grab(image_topic, depth_topic, info_topic, pose_topic, n_frames, timeout):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import Float64MultiArray

    class G(Node):
        def __init__(self):
            super().__init__("srf_depth_grab")
            self.rgb: List[np.ndarray] = []
            self.depth: List[np.ndarray] = []
            self.pose: List[np.ndarray] = []
            self.K = None
            self.create_subscription(Image, image_topic, self._rgb, qos_profile_sensor_data)
            self.create_subscription(Image, depth_topic, self._d, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, info_topic, self._i, qos_profile_sensor_data)
            self.create_subscription(Float64MultiArray, pose_topic, self._p, qos_profile_sensor_data)

        def _rgb(self, m):
            a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
            self.rgb.append(a[:, :, ::-1].copy() if m.encoding == "bgr8" else a.copy())

        def _d(self, m):
            a = np.frombuffer(m.data, np.uint16).reshape(m.height, m.width).astype(np.float32)
            self.depth.append(a)

        def _i(self, m):
            self.K = (m.k[0], m.k[4], m.k[2], m.k[5])

        def _p(self, m):
            v = np.asarray(m.data, float)
            if v.shape[0] >= 6:
                self.pose.append(v[:6])

    import rclpy as _r
    node = G()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _r.spin_once(node, timeout_sec=0.05)
        if (len(node.rgb) >= n_frames and len(node.depth) >= n_frames
                and len(node.pose) >= 10 and node.K is not None):
            break
    out = (list(node.rgb), list(node.depth), np.asarray(node.pose), node.K)
    node.destroy_node()
    return out


# ----------------------------------------------------------------------- main

def _ring(cx, cy, r, safe_box, n=14):
    """Home pose + one ring of `n` jogs at radius `r`. A single moderate
    radius keeps every jogged view well-overlapped with home -- the wide
    outer poses were where the depth ICP fell apart."""
    x0, x1, y0, y1 = safe_box
    pts = [(cx, cy)]
    for k in range(n):
        a = 2 * np.pi * k / n
        x = min(max(cx + r * np.cos(a), x0), x1)
        y = min(max(cy + r * np.sin(a), y0), y1)
        pts.append((float(x), float(y)))
    return pts


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="[1] depth-based marker/touch-free homography")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--z0", type=float, required=True,
                    help="polishing-surface height in robot base mm (plate is horizontal)")
    ap.add_argument("--jog_mm", type=float, default=30.0)
    ap.add_argument("--safe_box", type=float, nargs=4, default=list(DEFAULT_SAFE_BOX))
    ap.add_argument("--grid_z", type=float, default=None)
    ap.add_argument("--velocity_mm_s", type=float, default=None)
    ap.add_argument("--settle_sec", type=float, default=1.2)
    ap.add_argument("--frames", type=int, default=20, help="depth frames median-ed per pose")
    ap.add_argument("--tip_px", type=float, nargs=2, default=list(DEFAULT_TIP_PX))
    ap.add_argument("--plate_roi", type=int, nargs=4, default=[135, 45, 300, 185],
                    metavar=("U0", "V0", "U1", "V1"))
    ap.add_argument("--tool_box", type=int, nargs=4, default=[224, 12, 306, 240],
                    metavar=("U0", "V0", "U1", "V1"))
    ap.add_argument("--icp_roi", type=int, nargs=4, default=[8, 8, 416, 232],
                    metavar=("U0", "V0", "U1", "V1"))
    ap.add_argument("--tol_mm", type=float, default=None)
    ap.add_argument("--gate_mm", type=float, default=5.0,
                    help="max rotation-LOO / plate-Z residual to PASS. 5mm is well "
                         "inside the ~62mm defect-class position separation this "
                         "calibration exists to remove.")
    ap.add_argument("--image_topic", type=str, default=None)
    ap.add_argument("--depth_topic", type=str,
                    default="/realsense/vr/aligned_depth_to_color/image_raw")
    ap.add_argument("--info_topic", type=str,
                    default="/realsense/vr/aligned_depth_to_color/camera_info")
    ap.add_argument("--pose_topic", type=str, default=None)
    ap.add_argument("--skip_home_check", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--save_raw", type=str, default=None)
    ap.add_argument("--offline", type=str, default=None,
                    help="skip the robot; re-run the analysis on a --save_raw .npz")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    if cv2 is None:
        print("[depth] OpenCV required", file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    image_topic = args.image_topic or cfg.image_topic
    pose_topic = args.pose_topic or cfg.pose_topic
    velocity = float(args.velocity_mm_s or cfg.ptp_velocity_mm_s)
    tol = float(args.tol_mm if args.tol_mm is not None else cfg.homography_reproj_tol_mm)
    out_path = Path(args.out) if args.out else cfg.path("homography_file")
    tip_px = np.asarray(args.tip_px, float)
    plate_roi = tuple(int(x) for x in args.plate_roi)
    tool_box = tuple(int(x) for x in args.tool_box)
    icp_roi = tuple(int(x) for x in args.icp_roi)
    z0 = float(args.z0)

    poses6, depths, rgbs, jog_xy = [], [], [], []

    if args.offline:
        z = np.load(args.offline)
        depths = [d for d in z["depths"]]
        rgbs = [r for r in z["rgbs"]]
        poses6 = [p for p in z["poses6"]]
        jog_xy = [j.tolist() for j in z["jog_xy"]]
        K = tuple(float(x) for x in z["K"])
        home6 = z["home_pose6"]
        z0 = float(z["z0"]) if "z0" in z.files else z0
        tip_px = np.asarray(z["tip_px"], float) if "tip_px" in z.files else tip_px
        print(f"[depth] OFFLINE from {args.offline}: {len(depths)} poses, K={np.round(K, 1).tolist()}")
        return _analyse(depths, rgbs, poses6, jog_xy, K, home6, z0, tip_px,
                        plate_roi, tool_box, icp_roi, tol, out_path, args)

    from .ros_utils import ptp_to
    import rclpy

    rclpy.init(args=None)
    try:
        rgb0, d0, pp, K = _grab(image_topic, args.depth_topic, args.info_topic,
                                pose_topic, args.frames, 20.0)
        if pp.shape[0] < 5 or not d0 or K is None:
            print(f"[depth] missing stream (poses={pp.shape[0]} depth={len(d0)} K={K}). "
                  f"Is the VR camera up WITH depth? (rsv / rs_vr_pcd)", file=sys.stderr)
            return 1
        home6 = pp.mean(axis=0)
        grid_z = float(args.grid_z if args.grid_z is not None else home6[2])
        print(f"[depth] K = fx{K[0]:.1f} fy{K[1]:.1f} cx{K[2]:.1f} cy{K[3]:.1f}")
        print(f"[depth] home = {np.round(home6, 3).tolist()}   Z0 = {z0}   grid_z = {grid_z:.1f}")

        cx0, cy0 = home6[0], home6[1]
        ring = _ring(cx0, cy0, args.jog_mm, tuple(args.safe_box))
        print(f"[depth] {len(ring)} poses, jog {args.jog_mm}mm")
        if args.dry_run:
            for i, (x, y) in enumerate(ring):
                print(f"  {i:2d}  x={x:7.2f} y={y:7.2f}")
            return 0

        for i, (gx, gy) in enumerate(ring):
            reached = False
            for attempt in range(2):
                ptp_to([gx, gy, grid_z, home6[3], home6[4], home6[5]], velocity,
                       cfg.arm_service, logger=None)
                time.sleep(args.settle_sec)
                rgb, dpth, pw, _ = _grab(image_topic, args.depth_topic, args.info_topic,
                                         pose_topic, args.frames, 15.0)
                if pw.shape[0] < 5:
                    continue
                m = pw.mean(axis=0)
                err = float(np.hypot(m[0] - gx, m[1] - gy))
                if err < 5.0:
                    reached = True
                    break
                print(f"[depth] pose {i}: arm at ({m[0]:.1f},{m[1]:.1f}) but commanded "
                      f"({gx:.1f},{gy:.1f}) -- {err:.1f}mm off (attempt {attempt + 1})",
                      file=sys.stderr)
            if not reached:
                print(f"[depth] ABORT at pose {i}: the arm is not executing PTP commands. "
                      f"Check that the UR external-control program is running "
                      f"(/io_and_status_controller/robot_program_running). "
                      f"{i} poses captured{' (saved to --save_raw)' if args.save_raw else ''}.",
                      file=sys.stderr)
                if args.save_raw and depths:
                    np.savez_compressed(args.save_raw, depths=np.stack(depths),
                                        rgbs=np.stack(rgbs), poses6=np.stack(poses6),
                                        jog_xy=np.asarray(jog_xy), K=np.asarray(K),
                                        z0=z0, tip_px=tip_px, home_pose6=home6)
                ptp_to(list(home6), velocity, cfg.arm_service, logger=None)
                return 1
            if len(dpth) < max(5, args.frames // 2):
                print(f"[depth] pose {i}: depth={len(dpth)}", file=sys.stderr)
                return 1
            Dstack = np.stack(dpth[:args.frames]); Dstack[Dstack == 0] = np.nan
            Dmed = np.nan_to_num(np.nanmedian(Dstack, axis=0))
            m = pw.mean(axis=0)
            poses6.append(m); depths.append(Dmed)
            rgbs.append(np.clip(np.stack(rgb[:8]).astype(np.float32).mean(0), 0, 255).astype(np.uint8))
            jog_xy.append([m[0] - cx0, m[1] - cy0])
            print(f"[depth] {i:2d}/{len(ring) - 1}: x={m[0]:7.2f} y={m[1]:7.2f}  "
                  f"depth valid {100 * np.mean(Dmed > 0):.0f}%")

        ptp_to(list(home6), velocity, cfg.arm_service, logger=None)
    finally:
        rclpy.shutdown()

    if args.save_raw:
        np.savez_compressed(args.save_raw, depths=np.stack(depths), rgbs=np.stack(rgbs),
                            poses6=np.stack(poses6), jog_xy=np.asarray(jog_xy),
                            K=np.asarray(K), z0=z0, tip_px=tip_px, home_pose6=home6)
        print(f"[depth] raw captures -> {args.save_raw}")

    return _analyse(depths, rgbs, poses6, jog_xy, K, home6, z0, tip_px,
                    plate_roi, tool_box, icp_roi, tol, out_path, args)


def _analyse(depths, rgbs, poses6, jog_xy, K, home6, z0, tip_px,
             plate_roi, tool_box, icp_roi, tol, out_path, args) -> int:
    home6 = np.asarray(home6, float)
    tip_px = np.asarray(tip_px, float)
    D_home = depths[0]

    # ---- plate plane at home (camera frame) -----------------------------
    pm = _depth_mask(D_home, tool_box, plate_roi)
    vs, us = np.nonzero(pm)
    P_home = backproject(us, vs, D_home[vs, us], K)
    n_cam, d_cam, inl, plate_rms = fit_plane(P_home, thresh_mm=2.0)
    tilt = float(np.degrees(np.arccos(abs(n_cam[2]))))
    print(f"\n[depth] plate plane: {inl.sum()} px, RMS {plate_rms:.2f}mm, "
          f"n_cam={np.round(n_cam, 3).tolist()}, tilt-from-optical-axis {tilt:.1f}deg")

    # ---- ICP: camera-frame translation per jog ------------------------
    icp_mask0 = _depth_mask(D_home, tool_box, icp_roi)
    T_all, rms_all, rows = [], [], []
    for i in range(1, len(depths)):
        t, rms, nu = icp_translation(D_home, depths[i], K, icp_mask0)
        T_all.append(t); rms_all.append(rms)
        jn = float(np.hypot(jog_xy[i][0], jog_xy[i][1]))
        ratio = float(np.linalg.norm(t) / jn) if jn > 1 else 0.0
        rows.append({"i": i, "jog_x": f"{jog_xy[i][0]:.1f}", "jog_y": f"{jog_xy[i][1]:.1f}",
                     "t_cam": np.round(t, 1).tolist(), "icp_rms": f"{rms:.2f}",
                     "|t|/|jog|": f"{ratio:.2f}"})
    T_all = np.asarray(T_all)
    rms_all = np.asarray(rms_all)
    dxdy_all = np.asarray(jog_xy[1:])
    jn_all = np.hypot(dxdy_all[:, 0], dxdy_all[:, 1])
    ratio_all = np.linalg.norm(T_all, axis=1) / np.maximum(jn_all, 1e-6)

    # reject poses whose ICP is unreliable, then robustly reject the rest
    ok = (rms_all < 1.6) & (ratio_all > 0.8) & (ratio_all < 1.25)
    for row, o in zip(rows, ok):
        row["use"] = "yes" if o else "NO"
    print(table(rows, title="[depth] per-jog ICP (camera-frame translation, mm)"))
    if ok.sum() < 4:
        print(f"[depth] only {ok.sum()} usable jog poses -- need >=4", file=sys.stderr)
        return 1

    idxs = np.where(ok)[0]
    for _ in range(6):
        R_bc, sa, sb, ortho = solve_base_rotation(dxdy_all[idxs], T_all[idxs])
        pred = -(dxdy_all[idxs] @ np.array([R_bc[:, 0], R_bc[:, 1]]))
        res = np.linalg.norm(pred - T_all[idxs], axis=1)
        thr = max(2.5, 2.5 * np.median(res))
        good = res <= thr
        if good.all() or good.sum() < 4:
            break
        idxs = idxs[good]
    kept = sorted(int(i + 1) for i in idxs)
    R_cb = R_bc.T
    horiz_err = float(np.degrees(np.arccos(np.clip(abs(R_bc[:, 2] @ (-n_cam)), -1, 1))))
    print(f"\n[depth] rotation on {len(idxs)}/{len(T_all)} jog poses {kept}")
    print(f"[depth]   |a|={sa:.3f} |b|={sb:.3f} (expect ~1.0), a^b err {ortho:.2f}deg, "
          f"horizontal-plate check {horiz_err:.2f}deg")

    loo = []
    for k in range(len(idxs)):
        sub = np.delete(idxs, k)
        Rbc_k, *_ = solve_base_rotation(dxdy_all[sub], T_all[sub])
        pred = -(Rbc_k @ np.array([dxdy_all[idxs[k]][0], dxdy_all[idxs[k]][1], 0.0]))
        loo.append(np.linalg.norm(pred - T_all[idxs[k]]))
    loo = np.asarray(loo)
    print(f"[depth]   rotation leave-one-out residual: max {loo.max():.2f}mm  mean {loo.mean():.2f}mm")
    T_cam, dxdy = T_all[idxs], dxdy_all[idxs]

    # ---- translation from the tip anchor + Z0 -------------------------
    t_cb, z_tip = solve_tcb(R_cb, K, tip_px, n_cam, d_cam, home6[:3], z0)
    print(f"[depth] tip depth z_tip = {z_tip:.1f}mm   t_cb (cam origin in base) = "
          f"{np.round(t_cb, 1).tolist()}")

    # ---- Z0 consistency: map every plate pixel of every pose to base --
    # (independent of the ICP -- tests R_cb + t_cb end to end)
    z_dev = []
    for i in range(len(depths)):
        mm = _depth_mask(depths[i], tool_box, plate_roi)
        vv, uu = np.nonzero(mm)
        Xc = backproject(uu, vv, depths[i][vv, uu], K)
        Xb = Xc @ R_cb.T + t_cb + np.array([jog_xy[i][0], jog_xy[i][1], 0.0])
        d = Xb[:, 2] - z0
        z_dev.append(d[np.abs(d - np.median(d)) < 20.0])       # drop depth dropouts
    z_dev = np.concatenate(z_dev)
    z95 = float(np.percentile(np.abs(z_dev), 95))
    print(f"[depth] plate->base Z vs Z0={z0}: bias {z_dev.mean():+.2f}mm  "
          f"p95 |dev| {z95:.2f}mm  max {np.abs(z_dev).max():.2f}mm")

    # ---- synthesise the pixel->base-XY homography --------------------
    u0, v0, u1, v1 = plate_roi
    gu, gv = np.meshgrid(np.linspace(u0, u1, 12), np.linspace(v0, v1, 12))
    samp = np.c_[gu.ravel(), gv.ravel()]
    base_xy = pixels_to_plane(samp, K, R_cb, t_cb, z0)[:, :2]
    # synthetic points lie exactly on one homography -> any split is fine;
    # pin it so fit_homography skips its O(C(n,k)) spread search.
    res = fit_homography(samp, base_xy, n_fit=len(samp) - 6,
                         heldout_indices=list(range(6)), tol_mm=tol)
    res.H = plane_homography(K, R_cb, t_cb, z0)     # exact, not the RANSAC fit

    gate_rms = max(float(loo.max()), z95)
    passed = bool(gate_rms <= args.gate_mm and plate_rms <= 2.0
                  and horiz_err <= 2.0 and ortho <= 5.0 and 150 < z_tip < 600)

    meta = {
        "method": "depth_extrinsic",
        "z0_mm": z0, "K_fxfycxcy": list(K),
        "R_cb": R_cb.tolist(), "t_cb_base_mm": t_cb.tolist(), "z_tip_mm": z_tip,
        "n_cam_plate_normal": n_cam.tolist(), "plate_plane_rms_mm": plate_rms,
        "plate_tilt_deg": tilt,
        "rotation_scale_a_b": [sa, sb], "rotation_ortho_deg": ortho,
        "horizontal_plate_check_deg": horiz_err,
        "rotation_loo_max_mm": float(loo.max()), "rotation_loo_mean_mm": float(loo.mean()),
        "plate_to_base_z_bias_mm": float(z_dev.mean()),
        "plate_to_base_z_p95_mm": z95,
        "n_jog_poses": len(depths), "jog_mm": args.jog_mm,
        "tip_px": tip_px.tolist(),
        "home_pose6": home6.tolist(),
        "gate_value_mm": gate_rms, "tol_mm": tol,
    }
    res.passed = passed
    res.tol_mm = tol
    saved = save_homography(out_path, res, meta=meta)
    print(f"\n[depth] homography saved -> {saved}")

    verdict = [
        {"check": "plate planar in depth", "value": f"{plate_rms:.2f}mm", "tol": "<=2.0",
         "verdict": "PASS" if plate_rms <= 2.0 else "FAIL"},
        {"check": "rotation scale |a|,|b|", "value": f"{sa:.3f}/{sb:.3f}", "tol": "~1.0",
         "verdict": "PASS" if 0.95 < sa < 1.05 and 0.95 < sb < 1.05 else "WARN"},
        {"check": "horizontal-plate check", "value": f"{horiz_err:.2f}deg", "tol": "<=2.0",
         "verdict": "PASS" if horiz_err <= 2.0 else "FAIL"},
        {"check": "rotation leave-one-out", "value": f"{loo.max():.2f}mm", "tol": f"<={args.gate_mm}",
         "verdict": "PASS" if loo.max() <= args.gate_mm else "FAIL"},
        {"check": "plate->base Z vs Z0 (p95)", "value": f"{z95:.2f}mm", "tol": f"<={args.gate_mm}",
         "verdict": "PASS" if z95 <= args.gate_mm else "FAIL"},
    ]
    print()
    print(table(verdict, title="[1-depth] acceptance"))
    print("\n" + ("GATE PASSED -> proceed to [2] stain_origin_offline" if passed
                   else "GATE FAILED -- see checks above; H written with passed=false"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
