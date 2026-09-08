#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bring up the online stain-origin node ([2]+[5] inference side).

The node resolves stain_origin ONCE from the home-pose view and latches it on
/stain_relative_frame/stain_origin (transient local), then releases its camera
subscription. Start it BEFORE the inference stack so the latched value is
already there when the policy node subscribes.

  ros2 launch stain_relative_frame stain_origin_online.launch.py \
      config:=/path/to/stain_relative_frame.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = "stain_relative_frame"


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory(PKG), "config", "stain_relative_frame.yaml"
    )
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_config),
        DeclareLaunchArgument("image_topic", default_value=""),
        DeclareLaunchArgument("frames", default_value="10"),
        DeclareLaunchArgument("homography", default_value=""),
        DeclareLaunchArgument("clean_reference", default_value=""),
        Node(
            package=PKG,
            executable="stain_origin_node",
            name="stain_origin_node",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "config": LaunchConfiguration("config"),
                "image_topic": LaunchConfiguration("image_topic"),
                "frames": LaunchConfiguration("frames"),
                "homography": LaunchConfiguration("homography"),
                "clean_reference": LaunchConfiguration("clean_reference"),
            }],
        ),
    ])
