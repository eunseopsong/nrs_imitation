#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert gripper_hdf5_recorder_single_cam.py merged HDF5 output into compact
single-camera imitation form.

Output HDF5 layout extends demo_data_imitation_form_single_cam.py with:
  action/gripper_present_current_mA
  action/gripper_present_position
  observations/gripper/present_current_mA
  observations/gripper/present_position
"""

from __future__ import annotations

from _imitation_form_converter import DATASETS_ROOT, run_cli


def main() -> None:
    run_cli(
        description="Convert single-camera gripper merged HDF5 into compact imitation-form episodes.",
        default_root=DATASETS_ROOT / "single_cam",
        camera_names=("cam0",),
        include_gripper=True,
    )


if __name__ == "__main__":
    main()
