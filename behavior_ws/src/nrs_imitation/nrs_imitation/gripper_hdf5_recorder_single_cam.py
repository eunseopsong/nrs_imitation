#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nrs_imitation.gripper_hdf5_recorder_base import spin_recorder


FIXED_DEFAULTS = {
    "recording_mode": "tracker",
    "pose_topic": "/calibrated_pose",
    "force_topic": "/ftsensor/measured_Cvalue",
    "force_msg_type": "wrench",
    "image_topic": "/realsense/vr/color/image_raw",
    "enable_global_cam": False,
    "enable_gripper_state": True,
    "gripper_position_topic": "/gripper/present_position",
    "gripper_current_topic": "/gripper/present_current_mA",
    "file_prefix": "gripper_hdf5_recorder_single_cam",
}


def main(args=None):
    spin_recorder("gripper_hdf5_recorder_single_cam", fixed_defaults=FIXED_DEFAULTS, args=args)


if __name__ == "__main__":
    main()
