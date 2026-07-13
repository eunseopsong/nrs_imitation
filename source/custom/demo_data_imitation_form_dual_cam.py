#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert hdf5_recorder_dual_cam.py merged HDF5 output into compact imitation form.

Output HDF5 layout:
  action/position
  action/force
  observations/position
  observations/force
  observations/images/cam0
  observations/images/stain_mask, optional
  observations/images/cam1

Stain masks are generated/copied for cam0 only; cam1 is stored as RGB context
without its own stain mask.
"""

from __future__ import annotations

from _imitation_form_converter import DATASETS_ROOT, run_cli


def main() -> None:
    run_cli(
        description="Convert dual-camera merged HDF5 into compact imitation-form episodes.",
        default_root=DATASETS_ROOT / "polishing" / "dual_cam",
        camera_names=("cam0", "cam1"),
    )


if __name__ == "__main__":
    main()
