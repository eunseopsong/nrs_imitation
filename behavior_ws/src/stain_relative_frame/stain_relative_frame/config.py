#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Config for the stain-relative frame pipeline.

One YAML holds every threshold that the acceptance gates in steps [0]-[4] are
judged against, so a run can be reproduced from the file alone. Defaults here
match config/stain_relative_frame.yaml; the YAML wins when present.

`use_relative_position` defaults to False -- with it False the whole pipeline
is a no-op copy of the existing path (see `dataset_relativize.py`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

def _find_project_root() -> Path:
    """Locate the nrs_imitation checkout.

    $NRS_IMITATION_ROOT wins. Otherwise walk up from this file looking for the
    repo markers -- this file lives at
    <root>/behavior_ws/src/stain_relative_frame/stain_relative_frame/, but the
    node also runs from an ament install/ tree, where that relative depth no
    longer holds.
    """
    env = os.environ.get("NRS_IMITATION_ROOT")
    if env:
        return Path(env).expanduser()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "checkpoints").is_dir() and (parent / "datasets").is_dir():
            return parent
    return here.parents[4] if len(here.parents) > 4 else here.parent


PROJECT_ROOT = _find_project_root()


def _default_artifact_dir() -> str:
    return str(PROJECT_ROOT / "checkpoints" / "stain_relative_frame")


@dataclass
class StainRelativeConfig:
    # ---- step [3] master flag -------------------------------------------
    # False -> the converter emits a numerically identical dataset.
    use_relative_position: bool = False

    # ---- topics ----------------------------------------------------------
    image_topic: str = "/realsense/vr/color/image_raw"
    pose_topic: str = "/ur10skku/currentP"
    arm_service: str = "/singleArm_cmd/single_arm_command"

    # ---- artifacts -------------------------------------------------------
    artifact_dir: str = field(default_factory=_default_artifact_dir)
    homography_file: str = "homography.json"
    clean_reference_file: str = "clean_reference.npz"
    home_pose_report_file: str = "home_pose_repeatability.json"
    stain_origin_report_file: str = "stain_origin_stability.json"
    validation_report_file: str = "validation_gate.json"

    # ---- [0] home pose repeatability gate --------------------------------
    home_repeat_trials: int = 10
    home_pose_xy_std_tol_mm: float = 1.0
    home_pose_rot_std_tol_deg: float = 0.5
    home_settle_sec: float = 1.5
    home_pose_samples: int = 40
    ptp_velocity_mm_s: float = 20.0

    # ---- [0] clean reference --------------------------------------------
    clean_reference_frames: int = 40

    # ---- [1] homography gate --------------------------------------------
    homography_num_points: int = 8
    homography_fit_points: int = 6
    homography_reproj_tol_mm: float = 2.0
    homography_ransac_thresh_px: float = 3.0
    undistort: bool = False
    camera_matrix: Optional[List[List[float]]] = None
    dist_coeffs: Optional[List[float]] = None

    # ---- [2] stain detection --------------------------------------------
    # Reuses the existing diff-mask rule: |home frame - clean reference|.
    stain_diff_thresh: int = 25
    stain_blur_sigma: float = 2.0
    stain_min_area: int = 20
    stain_morph_kernel: int = 3
    stain_max_components: int = 5
    stain_stability_frames: int = 30
    stain_origin_std_tol_mm: float = 3.0

    # ---- [4] validation gate --------------------------------------------
    val_test_fraction: float = 0.2
    val_seed: int = 0
    val_repeats: int = 20            # repeated episode-level splits
    val_chance_margin: float = 0.10  # accuracy above chance + margin -> BLOCKED
    val_shuffle_repeats: int = 20

    # ---- provenance ------------------------------------------------------
    lighting_note: str = ""

    # ------------------------------------------------------------------
    @property
    def artifacts(self) -> Path:
        return Path(self.artifact_dir).expanduser()

    def path(self, key: str) -> Path:
        return self.artifacts / getattr(self, key)

    @classmethod
    def load(cls, path: Optional[str] = None, **overrides) -> "StainRelativeConfig":
        data: Dict[str, Any] = {}
        if path:
            p = Path(path).expanduser()
            if not p.is_file():
                raise FileNotFoundError(f"config not found: {p}")
            data = _read_yaml_or_json(p)
            # tolerate a ros2 param-file wrapper: {node: {ros__parameters: {...}}}
            if len(data) == 1:
                inner = next(iter(data.values()))
                if isinstance(inner, dict) and "ros__parameters" in inner:
                    data = inner["ros__parameters"]
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))


def _read_yaml_or_json(p: Path) -> Dict[str, Any]:
    text = p.read_text()
    if p.suffix.lower() in (".json",):
        return json.loads(text)
    try:
        import yaml  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required to read a .yaml config") from exc
    return yaml.safe_load(text) or {}


def default_config_path() -> Path:
    """The installed config/stain_relative_frame.yaml, or the source copy."""
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("stain_relative_frame"))
        cand = share / "config" / "stain_relative_frame.yaml"
        if cand.is_file():
            return cand
    except Exception:  # noqa: BLE001 -- not built / not sourced yet
        pass
    return Path(__file__).resolve().parents[1] / "config" / "stain_relative_frame.yaml"


def load_config(path: Optional[str] = None, **overrides) -> StainRelativeConfig:
    if path is None:
        p = default_config_path()
        path = str(p) if p.is_file() else None
    return StainRelativeConfig.load(path, **overrides)
