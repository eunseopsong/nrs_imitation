#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nrs_imitation.hdf5_recorder_base import spin_recorder


FIXED_DEFAULTS = {
    "recording_mode": "tracker",
    "pose_topic": "/calibrated_pose",
    "force_topic": "/ftsensor/measured_Cvalue",
    "force_msg_type": "wrench",
    "image_topic": "/realsense/vr/color/image_raw",
    "enable_global_cam": False,
    "file_prefix": "hdf5_recorder_single_cam",
}


def main(args=None):
    spin_recorder("hdf5_recorder_single_cam", fixed_defaults=FIXED_DEFAULTS, args=args)


if __name__ == "__main__":
    main()
