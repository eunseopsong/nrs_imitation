#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step [0a] as a launch file, for the PTP-driven (non-interactive) run.

  ros2 launch stain_relative_frame home_pose_check.launch.py \
      x:=420.34 y:=345.54 z:=211.86 rx:=-0.009 ry:=0.164 rz:=2.444 trials:=10

xyz in mm, the rotation triple in RADIANS -- the same convention as
/ur10skku/currentP. To find the current values first:

  ros2 run stain_relative_frame home_pose_repeatability -- --measure_only

--manual mode needs stdin, so run that one with `ros2 run`, not from here.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

PKG = "stain_relative_frame"


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory(PKG), "config", "stain_relative_frame.yaml"
    )
    args = [
        DeclareLaunchArgument("config", default_value=default_config),
        DeclareLaunchArgument("trials", default_value="10"),
        DeclareLaunchArgument("away_offset_mm", default_value="60.0"),
    ]
    args += [DeclareLaunchArgument(n, description=f"home pose {n}")
             for n in ("x", "y", "z", "rx", "ry", "rz")]

    return LaunchDescription(args + [
        ExecuteProcess(
            cmd=[
                "ros2", "run", PKG, "home_pose_repeatability", "--",
                "--config", LaunchConfiguration("config"),
                "--trials", LaunchConfiguration("trials"),
                "--away_offset_mm", LaunchConfiguration("away_offset_mm"),
                "--home_pose",
                LaunchConfiguration("x"), LaunchConfiguration("y"), LaunchConfiguration("z"),
                LaunchConfiguration("rx"), LaunchConfiguration("ry"), LaunchConfiguration("rz"),
            ],
            output="screen",
        ),
    ])
