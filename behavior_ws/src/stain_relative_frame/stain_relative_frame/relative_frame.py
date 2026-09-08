#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single source of truth for the stain-relative position transform.

★ Both the dataset converter (step [3], `dataset_relativize.py`) and the
  inference path (step [5], `stain_origin_node.py` / `RelativeFrameAdapter`)
  MUST call the functions in this module. Nothing else may subtract or add
  `stain_origin` by hand -- `audit_inference_path.py` fails the build if it
  finds an open-coded subtraction elsewhere.

Contract
--------
* `stain_origin` is a 2-vector (x, y) in ROBOT BASE mm.
* It is an EPISODE CONSTANT. It is computed once, at the home pose, before
  the episode starts, and never recomputed per frame. `StainOrigin.freeze()`
  makes that structural: a frozen origin raises on any further write.
* Only x and y move. z, rx, ry, rz and force pass through untouched --
  they are copied through this module verbatim so that the "unchanged"
  guarantee is testable rather than a comment.

★ NO ROTATION ALIGNMENT.
  Aligning the frame to the stain's principal axis would map 0/30/60/90 deg
  onto the same input and delete the learning target itself. This module
  only ever removes a TRANSLATION. There is deliberately no rotation
  argument, no `theta`, and no place to add one without touching the
  signature everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Bumped whenever the numerical meaning of the transform changes. Written into
# converted datasets and read back by the inference adapter; a mismatch is an
# error, not a warning.
TRANSFORM_VERSION = "stain_relative_v1"

# Keys used to preserve the untouched absolute trajectory in converted
# datasets. Analysis and logging read these; the policy never sees them.
ABS_OBS_POSITION_KEY = "analysis/absolute/observations_position"
ABS_ACT_POSITION_KEY = "analysis/absolute/action_position"
STAIN_ORIGIN_ATTR = "stain_origin_xy_mm"
USE_RELATIVE_ATTR = "use_relative_position"
TRANSFORM_VERSION_ATTR = "relative_transform_version"


class StainOriginError(RuntimeError):
    """Raised on any attempt to mutate a frozen (episode-constant) origin."""


@dataclass
class StainOrigin:
    """An episode-constant stain origin in robot base mm.

    Freeze it as soon as it is determined. A frozen origin cannot be
    reassigned, which is what makes "computed once per episode" a property of
    the code rather than of the caller's discipline.
    """

    xy: np.ndarray
    source: str = "unset"
    frame_index: int = -1
    _frozen: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        arr = np.asarray(self.xy, dtype=np.float64).reshape(-1)
        if arr.shape[0] != 2:
            raise ValueError(f"stain_origin must be (x, y), got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"stain_origin must be finite, got {arr}")
        object.__setattr__(self, "xy", arr)

    def freeze(self) -> "StainOrigin":
        object.__setattr__(self, "_frozen", True)
        return self

    @property
    def frozen(self) -> bool:
        return bool(self._frozen)

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False) and name != "_frozen":
            raise StainOriginError(
                "stain_origin is an EPISODE CONSTANT and is already frozen; "
                "recomputing it per step is exactly the bug this class exists "
                f"to prevent (attempted to set {name!r})"
            )
        object.__setattr__(self, name, value)


def _as_xy(stain_origin) -> np.ndarray:
    if isinstance(stain_origin, StainOrigin):
        return stain_origin.xy
    arr = np.asarray(stain_origin, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 2:
        raise ValueError(f"stain_origin must be (x, y), got shape {arr.shape}")
    return arr


def _check_pose(pose: np.ndarray) -> np.ndarray:
    arr = np.asarray(pose, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[-1] < 2:
        raise ValueError(f"pose array must be (T, >=2), got {arr.shape}")
    return arr


def to_relative(pose_xyzrpy: np.ndarray, stain_origin, use_relative: bool = True) -> np.ndarray:
    """Absolute robot pose -> stain-relative pose.

    Args:
        pose_xyzrpy: (T, D) or (D,) with D >= 2. Columns 0,1 are x,y in mm.
                     Every other column is passed through untouched.
        stain_origin: StainOrigin or (x, y) in mm.
        use_relative: when False this is the identity -- bit-for-bit the same
                      array values as the input (the [3] "flag off == old
                      path" guarantee).

    Returns:
        Array with the same shape and dtype as the input.
    """
    arr = _check_pose(pose_xyzrpy)
    out = arr.copy()
    if use_relative:
        origin = _as_xy(stain_origin)
        out[:, 0] -= origin[0]
        out[:, 1] -= origin[1]
    if np.ndim(pose_xyzrpy) == 1:
        out = out.reshape(-1)
    return out.astype(np.asarray(pose_xyzrpy).dtype, copy=False)


def to_absolute(pose_xyzrpy: np.ndarray, stain_origin, use_relative: bool = True) -> np.ndarray:
    """Stain-relative pose -> absolute robot pose. Inverse of `to_relative`.

    Called on the policy output immediately before the robot command, so the
    controller always receives absolute base coordinates.
    """
    arr = _check_pose(pose_xyzrpy)
    out = arr.copy()
    if use_relative:
        origin = _as_xy(stain_origin)
        out[:, 0] += origin[0]
        out[:, 1] += origin[1]
    if np.ndim(pose_xyzrpy) == 1:
        out = out.reshape(-1)
    return out.astype(np.asarray(pose_xyzrpy).dtype, copy=False)


def passthrough_columns(dim: int) -> list:
    """Columns `to_relative` leaves untouched -- z, rx, ry, rz (and beyond)."""
    return list(range(2, int(dim)))


class RelativeFrameAdapter:
    """Inference-side wrapper: one frozen origin, one transform, both ways.

    Constructed once when the home-pose capture has resolved the origin. The
    per-step calls are `observation()` and `command()`; neither can change
    the origin, because there is no setter and the origin is frozen.

        adapter = RelativeFrameAdapter(origin, use_relative=True)
        qpos_rel  = adapter.observation(qpos_abs)     # policy input
        cmd_abs   = adapter.command(action_rel)       # robot command
    """

    def __init__(self, stain_origin, use_relative: bool = False,
                 transform_version: str = TRANSFORM_VERSION):
        if transform_version != TRANSFORM_VERSION:
            raise ValueError(
                f"transform version mismatch: checkpoint={transform_version!r} "
                f"code={TRANSFORM_VERSION!r}"
            )
        self._use_relative = bool(use_relative)
        if isinstance(stain_origin, StainOrigin):
            self._origin = stain_origin.freeze()
        else:
            self._origin = StainOrigin(_as_xy(stain_origin), source="explicit").freeze()
        self._obs_calls = 0
        self._cmd_calls = 0

    @property
    def use_relative(self) -> bool:
        return self._use_relative

    @property
    def stain_origin(self) -> np.ndarray:
        return self._origin.xy.copy()

    def observation(self, pose_abs: np.ndarray) -> np.ndarray:
        self._obs_calls += 1
        return to_relative(pose_abs, self._origin, use_relative=self._use_relative)

    def command(self, pose_rel: np.ndarray) -> np.ndarray:
        self._cmd_calls += 1
        return to_absolute(pose_rel, self._origin, use_relative=self._use_relative)

    def stats(self) -> dict:
        return {
            "use_relative_position": self._use_relative,
            "stain_origin_xy_mm": self._origin.xy.tolist(),
            "origin_source": self._origin.source,
            "origin_frozen": self._origin.frozen,
            "observation_calls": self._obs_calls,
            "command_calls": self._cmd_calls,
        }
