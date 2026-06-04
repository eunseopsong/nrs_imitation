#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nrs_imitation.hdf5_recorder_base import spin_recorder


FIXED_DEFAULTS = {
    "recording_mode": "robot",
    "pose_topic": "/ur10skku/currentP",
    "force_topic": "/ur10skku/currentF",
    "force_msg_type": "array",
    "image_topic": "/realsense/robot/color/image_raw",
    "enable_global_cam": True,
    "global_image_topic": "/realsense/global/color/image_raw",
    "file_prefix": "hdf5_recorder_dual_cam",
}


def main(args=None):
    spin_recorder("hdf5_recorder_dual_cam", fixed_defaults=FIXED_DEFAULTS, args=args)


if __name__ == "__main__":
    main()
