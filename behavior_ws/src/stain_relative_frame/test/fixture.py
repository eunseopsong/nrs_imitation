#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic dataset generator for exercising steps [1]-[4] without hardware.

Builds a workpiece scene with a KNOWN pixel->robot homography, a clean
reference, and episodes whose stains sit in a type-dependent region while the
TCP approaches each stain with a type-INDEPENDENT offset. So by construction:

  absolute start position -> type   : separable   (the shortcut)
  relative start position -> type   : chance      (what [3] should produce)
  first-frame force       -> type   : chance      (no contact confound)

That is the fixture the gate is supposed to pass, which makes it a real test
of the gate rather than of the data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np

H_TRUE = np.array([
    [0.62, 0.015, 180.0],
    [0.010, 0.60, 250.0],
    [0.0, 0.0, 1.0],
])
IMG_H, IMG_W = 240, 424
BG_GRAY = 140.0


def _to_robot(px: np.ndarray) -> np.ndarray:
    p = np.asarray(px, dtype=np.float64).reshape(-1, 2)
    hom = np.concatenate([p, np.ones((p.shape[0], 1))], axis=1)
    out = hom @ H_TRUE.T
    return out[:, :2] / out[:, 2:3]


def _to_pixel(robot: np.ndarray) -> np.ndarray:
    Hinv = np.linalg.inv(H_TRUE)
    p = np.asarray(robot, dtype=np.float64).reshape(-1, 2)
    hom = np.concatenate([p, np.ones((p.shape[0], 1))], axis=1)
    out = hom @ Hinv.T
    return out[:, :2] / out[:, 2:3]


def clean_frame(rng: np.random.Generator) -> np.ndarray:
    img = np.full((IMG_H, IMG_W), BG_GRAY, dtype=np.float32)
    img += rng.normal(0.0, 1.5, img.shape).astype(np.float32)
    return np.clip(np.stack([img] * 3, axis=-1), 0, 255).astype(np.uint8)


def draw_stain(frame: np.ndarray, center_px, angle_deg: float,
               length: float = 22.0, width: float = 7.0) -> np.ndarray:
    """An oriented dark ellipse. Orientation encodes the type; the CENTROID
    does not depend on it, which is the whole point of translation-only."""
    out = frame.astype(np.float32).copy()
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    dx = xx - float(center_px[0])
    dy = yy - float(center_px[1])
    t = np.radians(angle_deg)
    u = dx * np.cos(t) + dy * np.sin(t)
    v = -dx * np.sin(t) + dy * np.cos(t)
    inside = (u / length) ** 2 + (v / width) ** 2 <= 1.0
    out[inside] -= 90.0
    return np.clip(out, 0, 255).astype(np.uint8)


def make_dataset(out_dir: Path, n_per_class: int = 30, T: int = 12,
                 seed: int = 0, force_confound: bool = False,
                 position_confound: bool = False) -> Dict:
    """Write episode_*.hdf5 in the imitation_form layout.

    force_confound / position_confound inject the failures the gate must
    catch, so the tests can check that it does not just always say PASS.
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_rgb = clean_frame(np.random.default_rng(seed + 999))

    # Type-dependent stain REGIONS -> absolute position is separable.
    regions = {0: (130.0, 90.0), 90: (300.0, 160.0)}
    truth = {}
    idx = 0
    for label, (cx0, cy0) in regions.items():
        for _ in range(n_per_class):
            center_px = np.array([cx0 + rng.normal(0, 12.0), cy0 + rng.normal(0, 10.0)])
            origin_mm = _to_robot(center_px)[0]

            # Approach offset: SAME distribution for both types, so the
            # stain-relative start position carries no label information.
            offset = rng.normal(0.0, 6.0, size=2)
            if position_confound:
                offset += np.array([25.0, 0.0]) if label == 0 else np.array([-25.0, 0.0])
            start_xy = origin_mm + offset

            frames = np.stack([
                draw_stain(clean_frame(rng), center_px, float(label)) for _ in range(T)
            ])

            pos = np.zeros((T, 6), dtype=np.float32)
            pos[:, 0] = start_xy[0] + np.linspace(0, 4, T)
            pos[:, 1] = start_xy[1] + np.linspace(0, 3, T)
            pos[:, 2] = 210.0 + rng.normal(0, 0.2, T)
            pos[:, 3] = -0.009
            pos[:, 4] = 0.164
            pos[:, 5] = 2.444

            base = np.array([0.5, -0.4, 1.2]) if not force_confound else (
                np.array([0.5, -0.4, 1.2]) if label == 0 else np.array([-4.0, -4.5, -2.0])
            )
            force = (base + rng.normal(0, 0.8, size=(T, 3))).astype(np.float32)

            path = out_dir / f"episode_{idx}.hdf5"
            with h5py.File(path, "w") as f:
                f.create_dataset("observations/images/cam0", data=frames,
                                 compression="gzip", compression_opts=1)
                f.create_dataset("observations/position", data=pos)
                f.create_dataset("observations/force", data=force)
                f.create_dataset("action/position", data=pos.copy())
                f.create_dataset("action/force", data=force.copy())
                f.attrs["stain_direction_deg"] = int(label)
                f.attrs["schema_version"] = "imitation_form_compact_v1"
            truth[path.stem] = {"origin_mm": origin_mm.tolist(),
                                "center_px": center_px.tolist(), "label": int(label)}
            idx += 1

    return {"reference_rgb": ref_rgb, "truth": truth, "H_true": H_TRUE}


def write_clean_reference(path: Path, ref_rgb: np.ndarray, lighting: str = "synthetic") -> Path:
    from stain_relative_frame.stain_detect import build_clean_reference

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        reference=build_clean_reference(ref_rgb[None, ...]).astype(np.float32),
        n_frames=1, lighting=lighting, stamp="synthetic",
        image_topic="synthetic", pose_topic="synthetic",
        image_shape=np.asarray(ref_rgb.shape, dtype=np.int64),
    )
    return path


def calibration_points(n: int = 8, seed: int = 3, noise_px: float = 0.3):
    """Well-spread pixel points and their true robot coordinates."""
    rng = np.random.default_rng(seed)
    gx = np.linspace(40, IMG_W - 40, 4)
    gy = np.linspace(30, IMG_H - 30, 2)
    px = np.array([[x, y] for y in gy for x in gx], dtype=np.float64)[:n]
    px += rng.normal(0, noise_px, px.shape)
    return px, _to_robot(px)
