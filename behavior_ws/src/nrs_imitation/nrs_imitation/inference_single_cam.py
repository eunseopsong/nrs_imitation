#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from nrs_imitation.inference_core import main as _main


DEFAULT_ROS_ARGS = [
    "--ros-args",
    "-p", "obs_mode:=single_cam",
    "-p", "ckpt_auto_subdir:=polishing/single_cam",
    "-p", "image_topic:=/realsense/robot/color/image_raw",
    "-p", "gradcam_overlay_topic:=/inference_single_cam/gradcam_overlay",
]


def main(args=None):
    user_args = list(sys.argv[1:] if args is None else args)
    _main(args=[sys.argv[0]] + DEFAULT_ROS_ARGS + user_args, node_name="inference_single_cam")


if __name__ == "__main__":
    main()
