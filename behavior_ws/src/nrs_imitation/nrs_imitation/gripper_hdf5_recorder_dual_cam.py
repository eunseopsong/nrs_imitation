#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nrs_imitation.gripper_hdf5_recorder_base import spin_recorder


FIXED_DEFAULTS = {
    "recording_mode": "robot",
    "pose_topic": "/ur10skku/currentP",
    "force_topic": "/ur10skku/currentF",
    "force_msg_type": "array",
    "image_topic": "/realsense/robot/color/image_raw",
    "enable_global_cam": True,
    "global_image_topic": "/realsense/global/color/image_raw",
    "gripper_position_topic": "/gripper/present_position",
    "gripper_current_topic": "/gripper/present_current_mA",
    "pose_xyz_scale": 1.0,
    "file_prefix": "gripper_hdf5_recorder_dual_cam",
}


def main(args=None):
    spin_recorder("gripper_hdf5_recorder_dual_cam", fixed_defaults=FIXED_DEFAULTS, args=args)


if __name__ == "__main__":
    main()
