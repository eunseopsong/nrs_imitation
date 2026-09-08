#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step [1]: pixel (u, v) -> robot base (x, y) mm homography.

The mapping is only valid because the camera pose is fixed at the home pose;
step [0]'s repeatability gate is what licenses that assumption. Every later
stage reads H back from the saved JSON -- nothing recomputes it.

Fit protocol (as specified):
  * 8 well-separated points on the workpiece plane
  * fit on 6 with cv2.findHomography(..., cv2.RANSAC)
  * hold out 2, report their reprojection error in mm
  * gate: held-out error <= 2 mm
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class HomographyResult:
    H: np.ndarray                    # (3,3) pixel -> robot mm
    fit_indices: List[int]
    heldout_indices: List[int]
    heldout_errors_mm: List[float]
    fit_errors_mm: List[float]
    max_heldout_error_mm: float
    mean_heldout_error_mm: float
    inlier_mask: List[int]
    undistorted: bool
    passed: bool
    tol_mm: float

    def summary(self) -> str:
        return (
            f"held-out max={self.max_heldout_error_mm:.3f}mm "
            f"mean={self.mean_heldout_error_mm:.3f}mm "
            f"tol={self.tol_mm:.2f}mm -> {'PASS' if self.passed else 'FAIL'}"
        )


def apply_homography(H: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Map (N,2) pixel coords through H to (N,2) robot mm."""
    pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)  # (N,3)
    out = hom @ np.asarray(H, dtype=np.float64).T                    # (N,3)
    w = out[:, 2:3]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("homography produced a point at infinity")
    return out[:, :2] / w


# ---- depth-extrinsic path: build H from the camera pose, per frame --------

def _K_inv(K_fxfycxcy) -> np.ndarray:
    fx, fy, cx, cy = K_fxfycxcy
    return np.array([[1.0 / fx, 0.0, -cx / fx],
                     [0.0, 1.0 / fy, -cy / fy],
                     [0.0, 0.0, 1.0]])


def plane_homography(K_fxfycxcy, R_cb, t_cb, z0: float) -> np.ndarray:
    """Closed-form pixel -> base-XY homography for a camera at (R_cb, t_cb)
    looking at the horizontal plane Z = z0.

    A camera ray K^-1 [u,v,1] becomes, in base, r = R_cb K^-1 [u,v,1]; it
    meets Z=z0 at  C + (z0 - C_z)/r_z * r  with  C = t_cb. Writing that as a
    projective map of [u,v,1] gives this 3x3 directly (no sampling / fit).
    """
    M = np.asarray(R_cb, float) @ _K_inv(K_fxfycxcy)      # (3,3), rows M0,M1,M2
    C = np.asarray(t_cb, float)
    dz = float(z0 - C[2])
    H = np.stack([C[0] * M[2] + dz * M[0],
                  C[1] * M[2] + dz * M[1],
                  M[2]])
    return H / H[2, 2]


def pose6_to_matrix(pose6) -> np.ndarray:
    """UR TCP pose [x,y,z, rx,ry,rz] (mm + rotation VECTOR rad) -> 4x4."""
    from scipy.spatial.transform import Rotation
    p = np.asarray(pose6, float).reshape(-1)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(p[3:6]).as_matrix()
    T[:3, 3] = p[:3]
    return T


def extrinsic_at_pose(R_cb_home, t_cb_home, home_pose6, pose6):
    """Given the camera->base extrinsic at the calibration home TCP pose,
    return it at any other TCP pose (eye-in-hand: camera rigid to the tool).

    tool_T_cam = inv(base_T_tool_home) @ base_T_cam_home   is constant;
    base_T_cam_i = base_T_tool_i @ tool_T_cam.
    """
    base_T_tool_home = pose6_to_matrix(home_pose6)
    base_T_cam_home = np.eye(4)
    base_T_cam_home[:3, :3] = np.asarray(R_cb_home, float)
    base_T_cam_home[:3, 3] = np.asarray(t_cb_home, float)
    tool_T_cam = np.linalg.inv(base_T_tool_home) @ base_T_cam_home
    base_T_cam_i = pose6_to_matrix(pose6) @ tool_T_cam
    return base_T_cam_i[:3, :3], base_T_cam_i[:3, 3]


def undistort_pixels(pixels: np.ndarray, camera_matrix, dist_coeffs) -> np.ndarray:
    """Lens-distortion correction for the calibration points (the [1] retry)."""
    if cv2 is None:
        raise ImportError("cv2 is required for undistortion")
    K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    d = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, K, d, P=K)
    return np.asarray(out).reshape(-1, 2)


def fit_homography(
    pixel_pts: np.ndarray,
    robot_pts: np.ndarray,
    n_fit: int = 6,
    tol_mm: float = 2.0,
    ransac_thresh_px: float = 3.0,
    heldout_indices: Optional[Sequence[int]] = None,
    camera_matrix=None,
    dist_coeffs=None,
    undistort: bool = False,
) -> HomographyResult:
    """Fit on `n_fit` points, measure reprojection on the rest.

    `heldout_indices` pins the split (reproducibility); when omitted the split
    that maximises the spread of the fit points is used, so the held-out pair
    is genuinely interpolated rather than sitting inside a tight cluster.
    """
    if cv2 is None:
        raise ImportError("cv2 is required to fit a homography")

    px = np.asarray(pixel_pts, dtype=np.float64).reshape(-1, 2)
    rb = np.asarray(robot_pts, dtype=np.float64).reshape(-1, 2)
    if px.shape[0] != rb.shape[0]:
        raise ValueError(f"point count mismatch: {px.shape[0]} pixel vs {rb.shape[0]} robot")
    n = px.shape[0]
    if n < 5:
        raise ValueError(f"need >=5 correspondences to fit and hold out, got {n}")
    n_fit = int(min(max(4, n_fit), n - 1))

    if undistort:
        if camera_matrix is None or dist_coeffs is None:
            raise ValueError("undistort=True requires camera_matrix and dist_coeffs")
        px = undistort_pixels(px, camera_matrix, dist_coeffs)

    if heldout_indices is not None:
        held = sorted(int(i) for i in heldout_indices)
        fit_idx = [i for i in range(n) if i not in held]
    else:
        fit_idx, held = _best_spread_split(px, n_fit)

    H, mask = cv2.findHomography(
        px[fit_idx].reshape(-1, 1, 2),
        rb[fit_idx].reshape(-1, 1, 2),
        cv2.RANSAC,
        float(ransac_thresh_px),
    )
    if H is None:
        raise RuntimeError(
            "cv2.findHomography failed -- the fit points are probably collinear "
            "or duplicated; spread them further across the workpiece"
        )
    inliers = [] if mask is None else np.asarray(mask).reshape(-1).astype(int).tolist()

    fit_err = np.linalg.norm(apply_homography(H, px[fit_idx]) - rb[fit_idx], axis=1)
    held_err = np.linalg.norm(apply_homography(H, px[held]) - rb[held], axis=1)

    return HomographyResult(
        H=np.asarray(H, dtype=np.float64),
        fit_indices=list(map(int, fit_idx)),
        heldout_indices=list(map(int, held)),
        heldout_errors_mm=held_err.tolist(),
        fit_errors_mm=fit_err.tolist(),
        max_heldout_error_mm=float(held_err.max()),
        mean_heldout_error_mm=float(held_err.mean()),
        inlier_mask=inliers,
        undistorted=bool(undistort),
        passed=bool(held_err.max() <= float(tol_mm)),
        tol_mm=float(tol_mm),
    )


def _best_spread_split(px: np.ndarray, n_fit: int) -> Tuple[List[int], List[int]]:
    """Choose the hold-out set so the fit set still covers the workpiece.

    Scored by the minimum pairwise distance among fit points -- a fit set with
    two near-coincident points is degenerate no matter how many points it has.
    """
    n = px.shape[0]
    n_held = n - n_fit
    best, best_score = None, -np.inf
    for held in combinations(range(n), n_held):
        fit = [i for i in range(n) if i not in held]
        pts = px[fit]
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        score = float(d[np.triu_indices(len(fit), k=1)].min())
        if score > best_score:
            best, best_score = (fit, list(held)), score
    return best[0], best[1]


def save_homography(path: str | Path, result: HomographyResult, meta: Optional[Dict] = None) -> Path:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "H_pixel_to_robot_mm": result.H.tolist(),
        "fit_indices": result.fit_indices,
        "heldout_indices": result.heldout_indices,
        "heldout_errors_mm": result.heldout_errors_mm,
        "fit_errors_mm": result.fit_errors_mm,
        "max_heldout_error_mm": result.max_heldout_error_mm,
        "mean_heldout_error_mm": result.mean_heldout_error_mm,
        "inlier_mask": result.inlier_mask,
        "undistorted": result.undistorted,
        "tol_mm": result.tol_mm,
        "passed": result.passed,
        "stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if meta:
        payload["meta"] = meta
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return p


def load_homography(path: str | Path, require_pass: bool = True) -> Tuple[np.ndarray, Dict]:
    """Read H back. Every downstream stage goes through here."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(
            f"homography not found: {p}\nRun step [1] first "
            f"(ros2 run stain_relative_frame homography_fit)."
        )
    payload = json.loads(p.read_text())
    if require_pass and not payload.get("passed", False):
        raise RuntimeError(
            f"homography at {p} did NOT pass the {payload.get('tol_mm')}mm held-out "
            f"gate (max={payload.get('max_heldout_error_mm')}mm). Step [1] must pass "
            "before anything downstream is meaningful."
        )
    return np.asarray(payload["H_pixel_to_robot_mm"], dtype=np.float64), payload
