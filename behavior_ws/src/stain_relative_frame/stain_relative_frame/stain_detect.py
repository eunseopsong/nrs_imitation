#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step [2]: stain centroid at the home pose -> `stain_origin` in robot mm.

Reuses the existing reference-difference rule
(`scripts/direction_classifier/build_home_reference.py`, the converter's
`reference_diff` path): the clean-specimen reference captured at the home
pose in step [0] is subtracted from the home-pose frame, and what survives
thresholding + morphology is the stain.

Failure rules (recorded, never silently skipped):
  * empty mask after filtering              -> FAIL_EMPTY_MASK
  * more than `max_components` blobs        -> FAIL_TOO_MANY_COMPONENTS

The centroid is the AREA-WEIGHTED centre of the surviving blobs, mapped
through H. It is computed once per episode and frozen -- see relative_frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .homography import apply_homography

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


FAIL_EMPTY_MASK = "FAIL_EMPTY_MASK"
FAIL_TOO_MANY_COMPONENTS = "FAIL_TOO_MANY_COMPONENTS"
FAIL_SHAPE_MISMATCH = "FAIL_SHAPE_MISMATCH"
OK = "OK"


@dataclass
class StainDetection:
    status: str
    centroid_px: Optional[np.ndarray] = None     # (2,) u, v
    origin_mm: Optional[np.ndarray] = None       # (2,) robot x, y
    num_components: int = 0
    area_px: int = 0
    mask: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == OK


def to_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr.astype(np.float32).mean(axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0].astype(np.float32)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    raise ValueError(f"cannot grayscale an array of shape {arr.shape}")


def build_clean_reference(frames: np.ndarray) -> np.ndarray:
    """Mean grayscale of N clean-specimen frames -- step [0]'s reference."""
    arr = np.asarray(frames)
    if arr.ndim == 4:
        return np.stack([to_gray(f) for f in arr]).mean(axis=0).astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return to_gray(arr)
    if arr.ndim == 3:
        return arr.astype(np.float32).mean(axis=0)
    raise ValueError(f"expected (N,H,W,3) or (N,H,W), got {arr.shape}")


def _morphology(mask_u8: np.ndarray, kernel_size: int) -> np.ndarray:
    k = int(kernel_size)
    if k <= 1 or cv2 is None:
        return mask_u8
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), dtype=np.uint8)
    opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def diff_mask(
    frame: np.ndarray,
    clean_reference: np.ndarray,
    diff_thresh: int = 25,
    blur_sigma: float = 2.0,
    min_area: int = 20,
    morph_kernel: int = 3,
) -> Tuple[np.ndarray, int, List[Tuple[float, float, int]]]:
    """|frame - clean_reference| -> binary stain mask.

    Returns (mask_u8, num_components, [(cx, cy, area), ...]).

    The frame is DC-matched to the reference inside the frame first, so a
    global illumination shift between the reference capture and the episode
    does not light up the whole plate.
    """
    g = to_gray(frame)
    ref = np.asarray(clean_reference, dtype=np.float32)
    if g.shape != ref.shape:
        raise ValueError(f"frame {g.shape} does not match clean reference {ref.shape}")

    g = g - float(g.mean()) + float(ref.mean())
    d = np.abs(g - ref)
    if cv2 is not None and blur_sigma > 0:
        d = cv2.GaussianBlur(d, (0, 0), float(blur_sigma))

    mask = (d > float(diff_thresh)).astype(np.uint8) * 255
    mask = _morphology(mask, morph_kernel)

    comps: List[Tuple[float, float, int]] = []
    if cv2 is None:
        area = int(np.count_nonzero(mask))
        if area >= int(min_area):
            ys, xs = np.nonzero(mask)
            comps.append((float(xs.mean()), float(ys.mean()), area))
        return mask, len(comps), comps

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    kept = np.zeros_like(mask)
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        kept[labels == label] = 255
        comps.append((float(centroids[label][0]), float(centroids[label][1]), area))
    return kept, len(comps), comps


def detect_stain_origin(
    frame: np.ndarray,
    clean_reference: np.ndarray,
    H: np.ndarray,
    diff_thresh: int = 25,
    blur_sigma: float = 2.0,
    min_area: int = 20,
    morph_kernel: int = 3,
    max_components: int = 5,
    keep_mask: bool = False,
) -> StainDetection:
    """One frame -> one `stain_origin` candidate in robot mm."""
    try:
        mask, n_comp, comps = diff_mask(
            frame, clean_reference,
            diff_thresh=diff_thresh, blur_sigma=blur_sigma,
            min_area=min_area, morph_kernel=morph_kernel,
        )
    except ValueError:
        return StainDetection(status=FAIL_SHAPE_MISMATCH)

    held = mask if keep_mask else None
    if n_comp == 0:
        return StainDetection(status=FAIL_EMPTY_MASK, num_components=0, mask=held)
    if n_comp > int(max_components):
        return StainDetection(
            status=FAIL_TOO_MANY_COMPONENTS, num_components=n_comp,
            area_px=int(sum(c[2] for c in comps)), mask=held,
        )

    areas = np.asarray([c[2] for c in comps], dtype=np.float64)
    cx = float(np.average([c[0] for c in comps], weights=areas))
    cy = float(np.average([c[1] for c in comps], weights=areas))
    origin = apply_homography(H, np.array([[cx, cy]]))[0]
    return StainDetection(
        status=OK,
        centroid_px=np.array([cx, cy], dtype=np.float64),
        origin_mm=np.asarray(origin, dtype=np.float64),
        num_components=int(n_comp),
        area_px=int(areas.sum()),
        mask=held,
    )


def dark_blob(
    frame: np.ndarray,
    plate_roi: Tuple[int, int, int, int],
    tool_box: Tuple[int, int, int, int],
    dark_thresh: int = 60,
    min_area: int = 40,
    max_components: int = 8,
    morph_kernel: int = 3,
) -> Tuple[np.ndarray, int, List[Tuple[float, float, int]]]:
    """Reference-free: the defect is a black strip on bright brushed metal.

    Threshold very-dark pixels inside the plate ROI (minus the co-mounted
    tool), keep blobs >= min_area. No clean reference, so lighting drift
    between recording sessions does not matter.
    """
    g = to_gray(frame)
    m = np.zeros(g.shape, np.uint8)
    u0, v0, u1, v1 = (int(x) for x in plate_roi)
    m[v0:v1, u0:u1] = 1
    tu0, tv0, tu1, tv1 = (int(x) for x in tool_box)
    m[tv0:tv1, tu0:tu1] = 0
    dark = ((g < float(dark_thresh)) & (m > 0)).astype(np.uint8)
    dark = _morphology(dark * 255, morph_kernel)

    comps: List[Tuple[float, float, int]] = []
    if cv2 is None:
        area = int(np.count_nonzero(dark))
        if area >= int(min_area):
            ys, xs = np.nonzero(dark)
            comps.append((float(xs.mean()), float(ys.mean()), area))
        return dark, len(comps), comps

    n_labels, labels, stats, cents = cv2.connectedComponentsWithStats(
        (dark > 0).astype(np.uint8), connectivity=8)
    kept = np.zeros_like(dark)
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        kept[labels == label] = 255
        comps.append((float(cents[label][0]), float(cents[label][1]), area))
    return kept, len(comps), comps


def _strip_midpoint(mask: np.ndarray) -> Tuple[float, float]:
    """Midpoint of the dark region along its principal axis -- more stable
    than the centroid when the tool occludes one part of the strip."""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    c = pts.mean(0)
    d = pts - c
    _, _, Vt = np.linalg.svd(d, full_matrices=False)
    axis = Vt[0]
    proj = d @ axis
    mid = c + 0.5 * (proj.min() + proj.max()) * axis
    return float(mid[0]), float(mid[1])


def detect_stain_dark(
    frame: np.ndarray,
    H: np.ndarray,
    plate_roi: Tuple[int, int, int, int],
    tool_box: Tuple[int, int, int, int],
    dark_thresh: int = 60,
    min_area: int = 40,
    max_components: int = 8,
    morph_kernel: int = 3,
    keep_mask: bool = False,
) -> StainDetection:
    """One frame -> one `stain_origin` candidate (robot mm), reference-free,
    using the per-frame homography `H`."""
    mask, n, comps = dark_blob(frame, plate_roi, tool_box, dark_thresh=dark_thresh,
                               min_area=min_area, max_components=max_components,
                               morph_kernel=morph_kernel)
    held = mask if keep_mask else None
    if n == 0:
        return StainDetection(status=FAIL_EMPTY_MASK, num_components=0, mask=held)
    if n > int(max_components):
        return StainDetection(status=FAIL_TOO_MANY_COMPONENTS, num_components=n,
                              area_px=int(sum(c[2] for c in comps)), mask=held)
    cx, cy = _strip_midpoint(mask)
    origin = apply_homography(H, np.array([[cx, cy]]))[0]
    return StainDetection(
        status=OK, centroid_px=np.array([cx, cy], float),
        origin_mm=np.asarray(origin, float), num_components=int(n),
        area_px=int(sum(c[2] for c in comps)), mask=held,
    )


def dark_cloud_origin(
    frames: np.ndarray,
    per_frame_H: np.ndarray,
    plate_roi: Tuple[int, int, int, int],
    tool_box: Tuple[int, int, int, int],
    episode: str = "",
    std_tol_mm: float = 6.0,
    dark_thresh: int = 60,
    min_area: int = 40,
    max_components: int = 8,
) -> "StabilityReport":
    """Reference-free origin for an occluded strip.

    Each frame's dark pixels are mapped to robot mm through that frame's own
    homography and POOLED. The tool hides a different part of the strip in
    every frame, so the union recovers the whole strip; its robust centre is
    the origin. Spread = scatter of the per-frame centres (measurement
    noise), reported but not the frozen value.
    """
    arr = np.asarray(frames)
    Hs = np.asarray(per_frame_H, float)
    pooled: List[np.ndarray] = []
    per_frame_c: List[np.ndarray] = []
    failures: List[Tuple[int, str]] = []
    for i, fr in enumerate(arr):
        mask, n, comps = dark_blob(fr, plate_roi, tool_box, dark_thresh=dark_thresh,
                                   min_area=min_area, max_components=max_components)
        if n == 0:
            failures.append((i, FAIL_EMPTY_MASK)); continue
        if n > max_components:
            failures.append((i, FAIL_TOO_MANY_COMPONENTS)); continue
        ys, xs = np.nonzero(mask)
        b = apply_homography(Hs[i], np.stack([xs, ys], axis=1).astype(np.float64))
        pooled.append(b)
        per_frame_c.append(np.median(b, axis=0))

    if len(pooled) < max(3, 0.3 * len(arr)):
        return StabilityReport(episode=episode, n_frames=int(arr.shape[0]),
                               n_ok=len(pooled), failures=failures, origin_mm=None,
                               std_mm=None, std_norm_mm=float("nan"), unstable=True)

    allpts = np.concatenate(pooled, axis=0)
    c = np.median(allpts, axis=0)
    for _ in range(4):                                   # robust trimmed centre
        d = np.linalg.norm(allpts - c, axis=1)
        c = allpts[d <= np.quantile(d, 0.85)].mean(axis=0)

    # Principal axis of the pooled cloud -> strip direction in base mm.
    # Trim to the inlier core first so a stray blob doesn't tilt it.
    core = allpts[np.linalg.norm(allpts - c, axis=1) <= np.quantile(
        np.linalg.norm(allpts - c, axis=1), 0.85)]
    _, _, Vt = np.linalg.svd(core - core.mean(axis=0), full_matrices=False)
    angle = float(np.arctan2(Vt[0][1], Vt[0][0]) % np.pi)

    pf = np.stack(per_frame_c)
    med = np.median(pf, axis=0)
    std = 1.4826 * np.median(np.abs(pf - med), axis=0)
    std_norm = float(np.sqrt(float((std ** 2).sum())))
    return StabilityReport(
        episode=episode, n_frames=int(arr.shape[0]), n_ok=len(pooled), failures=failures,
        origin_mm=c, std_mm=std, std_norm_mm=std_norm,
        unstable=bool(std_norm > float(std_tol_mm)
                      or len(failures) > 0.4 * arr.shape[0]),
        angle_rad=angle,
    )


def measure_stability_perframe(
    frames: np.ndarray,
    per_frame_H: np.ndarray,
    detect_fn,
    episode: str = "",
    std_tol_mm: float = 8.0,
) -> "StabilityReport":
    """Like `measure_stability` but each frame carries its own homography
    (`per_frame_H` is (T,3,3)) -- used when the camera pose varies frame to
    frame and the map is rebuilt from the depth extrinsic. `detect_fn(frame,
    H)` returns a `StainDetection`."""
    arr = np.asarray(frames)
    Hs = np.asarray(per_frame_H, float)
    origins, failures = [], []
    for i, fr in enumerate(arr):
        det = detect_fn(fr, Hs[i])
        if det.ok:
            origins.append(det.origin_mm)
        else:
            failures.append((int(i), det.status))
    if not origins:
        return StabilityReport(episode=episode, n_frames=int(arr.shape[0]), n_ok=0,
                               failures=failures, origin_mm=None, std_mm=None,
                               std_norm_mm=float("nan"), unstable=True)
    o = np.stack(origins)
    # robust centre + spread (a single bad frame should not flip the verdict)
    med = np.median(o, axis=0)
    std = 1.4826 * np.median(np.abs(o - med), axis=0)
    std_norm = float(np.sqrt(float((std ** 2).sum())))
    return StabilityReport(
        episode=episode, n_frames=int(arr.shape[0]), n_ok=len(origins), failures=failures,
        origin_mm=med, std_mm=std, std_norm_mm=std_norm,
        unstable=bool(std_norm > float(std_tol_mm) or len(failures) > 0.4 * arr.shape[0]),
    )


@dataclass
class StabilityReport:
    """Step [2]'s per-episode stability measurement over the first N frames."""

    episode: str
    n_frames: int
    n_ok: int
    failures: List[Tuple[int, str]]
    origin_mm: Optional[np.ndarray]       # median over the OK frames
    std_mm: Optional[np.ndarray]          # (2,) per-axis std
    std_norm_mm: float                    # sqrt(var_x + var_y), the reported scalar
    unstable: bool
    # Principal-axis angle of the pooled dark cloud, base-frame radians in
    # [0, pi) (a strip is a line, so direction is mod pi). None unless the
    # detector computed it (dark_cloud_origin does). NOT used by the
    # translation-only relative frame -- only the optional inference-side
    # rotation-canonicalization consumes it.
    angle_rad: Optional[float] = None

    def row(self) -> dict:
        return {
            "episode": self.episode,
            "frames": self.n_frames,
            "ok": self.n_ok,
            "failed": self.n_frames - self.n_ok,
            "origin_x_mm": None if self.origin_mm is None else round(float(self.origin_mm[0]), 3),
            "origin_y_mm": None if self.origin_mm is None else round(float(self.origin_mm[1]), 3),
            "std_x_mm": None if self.std_mm is None else round(float(self.std_mm[0]), 3),
            "std_y_mm": None if self.std_mm is None else round(float(self.std_mm[1]), 3),
            "std_mm": round(float(self.std_norm_mm), 3),
            "unstable": bool(self.unstable),
        }


def measure_stability(
    frames: np.ndarray,
    clean_reference: np.ndarray,
    H: np.ndarray,
    episode: str = "",
    std_tol_mm: float = 3.0,
    **detect_kw,
) -> StabilityReport:
    """Run detection independently on each of the first N frames.

    This is a MEASUREMENT of detector noise, not the production path -- the
    production path detects once. A large spread here means the origin the
    episode was frozen at is itself uncertain by that much.
    """
    arr = np.asarray(frames)
    origins, failures = [], []
    for i, fr in enumerate(arr):
        det = detect_stain_origin(fr, clean_reference, H, **detect_kw)
        if det.ok:
            origins.append(det.origin_mm)
        else:
            failures.append((int(i), det.status))

    if not origins:
        return StabilityReport(
            episode=episode, n_frames=int(arr.shape[0]), n_ok=0, failures=failures,
            origin_mm=None, std_mm=None, std_norm_mm=float("nan"), unstable=True,
        )

    o = np.stack(origins)
    std = o.std(axis=0)
    std_norm = float(np.sqrt(float((std ** 2).sum())))
    return StabilityReport(
        episode=episode,
        n_frames=int(arr.shape[0]),
        n_ok=len(origins),
        failures=failures,
        origin_mm=np.median(o, axis=0),
        std_mm=std,
        std_norm_mm=std_norm,
        unstable=bool(std_norm > float(std_tol_mm) or failures),
    )
