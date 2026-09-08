#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step [1b]: read the pixel coordinates of the calibration points.

Two reproducible methods, as required:

  --method click   OpenCV window on the SAVED home image from [1a]; click the
                   points in the same order they were touched. Zoom with the
                   mouse wheel, 'u' undoes, 'q' finishes. The clicks are
                   recorded at full-resolution sub-pixel coordinates.
  --method aruco   Detect ArUco markers and use their centres, ordered by
                   marker id. Fully repeatable -- no operator in the loop.

Output: adds `pixel_pts` (N,2) and `pixel_method` to homography_points.npz.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .config import load_config


def detect_aruco(image: np.ndarray, dictionary: str = "DICT_4X4_50"):
    """Marker centres ordered by id. Returns (pts (N,2), ids (N,))."""
    if cv2 is None:
        raise ImportError("cv2 required")
    if not hasattr(cv2, "aruco"):
        raise ImportError("this OpenCV build has no aruco module; use --method click")
    d = getattr(cv2.aruco, dictionary)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    try:  # OpenCV >= 4.7
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(d),
                                      cv2.aruco.DetectorParameters())
        corners, ids, _ = det.detectMarkers(gray)
    except AttributeError:  # OpenCV < 4.7
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, cv2.aruco.Dictionary_get(d), parameters=cv2.aruco.DetectorParameters_create()
        )
    if ids is None or len(ids) == 0:
        raise RuntimeError("no ArUco markers detected in the home image")
    ids = np.asarray(ids).reshape(-1)
    cent = np.stack([np.asarray(c).reshape(4, 2).mean(axis=0) for c in corners])
    order = np.argsort(ids)
    return cent[order].astype(np.float64), ids[order]


def pick_by_click(image: np.ndarray, n_points: int) -> np.ndarray:
    if cv2 is None:
        raise ImportError("cv2 required for --method click")
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image.copy()
    pts: List[List[float]] = []
    state = {"zoom": 3.0, "cx": image.shape[1] / 2.0, "cy": image.shape[0] / 2.0, "pan": False}
    win = "pick calibration points (click in touch order | u=undo | q=done)"

    def view():
        z = state["zoom"]
        out = cv2.resize(bgr, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
        for i, (u, v) in enumerate(pts):
            p = (int(round(u * z)), int(round(v * z)))
            cv2.drawMarker(out, p, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(out, str(i), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(out, f"{len(pts)}/{n_points}  zoom={z:.1f}x",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return out

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n_points:
            pts.append([x / state["zoom"], y / state["zoom"]])
        elif event == cv2.EVENT_MOUSEWHEEL:
            state["zoom"] = float(np.clip(
                state["zoom"] * (1.25 if flags > 0 else 0.8), 1.0, 12.0))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    print(f"[pixels] click the {n_points} points IN THE SAME ORDER they were touched")
    while True:
        cv2.imshow(win, view())
        k = cv2.waitKey(20) & 0xFF
        if k == ord("u") and pts:
            popped = pts.pop()
            print(f"[pixels] undo -> {np.round(popped, 2).tolist()}")
        elif k in (ord("q"), 13) and len(pts) == n_points:
            break
        elif k == 27:
            cv2.destroyAllWindows()
            raise SystemExit("[pixels] aborted")
    cv2.destroyAllWindows()
    return np.asarray(pts, dtype=np.float64)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="[1b] pixel coordinates of the calibration points")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--points_file", type=str, default=None)
    ap.add_argument("--method", choices=["click", "aruco"], default="click")
    ap.add_argument("--aruco_dict", type=str, default="DICT_4X4_50")
    ap.add_argument("--save_preview", action="store_true",
                    help="write a PNG with the picked points drawn on it")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    pts_path = Path(args.points_file) if args.points_file else cfg.artifacts / "homography_points.npz"
    if not pts_path.is_file():
        print(f"[pixels] not found: {pts_path}\nRun step [1a] homography_collect first.",
              file=sys.stderr)
        return 1

    with np.load(pts_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    image = np.asarray(data["home_image"])
    robot = np.asarray(data["robot_pts"], dtype=np.float64)
    n = robot.shape[0]

    if args.method == "aruco":
        pixels, ids = detect_aruco(image, args.aruco_dict)
        print(f"[pixels] aruco ids {ids.tolist()}")
        if pixels.shape[0] != n:
            print(f"[pixels] detected {pixels.shape[0]} markers but {n} points were "
                  f"touched -- counts must match", file=sys.stderr)
            return 1
    else:
        pixels = pick_by_click(image, n)

    for i in range(n):
        print(f"[pixels] {i}: px=({pixels[i, 0]:7.2f}, {pixels[i, 1]:7.2f})  "
              f"robot=({robot[i, 0]:8.3f}, {robot[i, 1]:8.3f}) mm")

    data["pixel_pts"] = pixels
    data["pixel_method"] = args.method
    np.savez_compressed(pts_path, **data)
    print(f"[pixels] saved -> {pts_path}")

    if args.save_preview and cv2 is not None:
        prev = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
        for i, (u, v) in enumerate(pixels):
            cv2.drawMarker(prev, (int(round(u)), int(round(v))), (0, 0, 255),
                           cv2.MARKER_CROSS, 14, 2)
            cv2.putText(prev, str(i), (int(u) + 6, int(v) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        out = pts_path.with_name("homography_points_preview.png")
        cv2.imwrite(str(out), prev)
        print(f"[pixels] preview -> {out}")

    print("[pixels] next: ros2 run stain_relative_frame homography_fit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
