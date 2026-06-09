#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert hdf5_recorder_single_cam.py merged HDF5 output into compact imitation form.

Output HDF5 layout:
  action/position
  action/force
  observations/position
  observations/force
  observations/images/cam0
  observations/images/stain_mask, optional

If the merged HDF5 was recorded with hdf5_recorder_single_cam_stain_mask,
episode_0 is treated as a clean surface reference by default and is used to
generate stain_mask/blob proposals for the remaining episodes.
"""

from __future__ import annotations

from _imitation_form_converter import DATASETS_ROOT, run_cli


def main() -> None:
    run_cli(
        description="Convert single-camera merged HDF5 into compact imitation-form episodes.",
        default_root=DATASETS_ROOT / "single_cam",
        camera_names=("cam0",),
    )


if __name__ == "__main__":
    main()
