#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    joy_topic = LaunchConfiguration("joy_topic")
    command_topic = LaunchConfiguration("command_topic")
    dpad_left_positive = LaunchConfiguration("dpad_left_positive")
    device_id = LaunchConfiguration("device_id")
    deadzone = LaunchConfiguration("deadzone")
    autorepeat_rate = LaunchConfiguration("autorepeat_rate")

    return LaunchDescription([
        DeclareLaunchArgument("joy_topic", default_value="/joy_il"),
        DeclareLaunchArgument("command_topic", default_value="/vr_demo_recorder/command"),
        DeclareLaunchArgument("dpad_left_positive", default_value="true"),
        DeclareLaunchArgument("device_id", default_value="0"),
        DeclareLaunchArgument("deadzone", default_value="0.05"),
        DeclareLaunchArgument("autorepeat_rate", default_value="20.0"),

        Node(
            package="joy",
            executable="joy_node",
            name="joy_node_il",
            output="screen",
            parameters=[{
                "device_id": device_id,
                "deadzone": deadzone,
                "autorepeat_rate": autorepeat_rate,
            }],
            remappings=[
                ("/joy", joy_topic),
            ],
        ),

        Node(
            package="nrs_imitation",
            executable="vr_demo_joy_controller",
            name="vr_demo_joy_controller",
            output="screen",
            parameters=[{
                "joy_topic": joy_topic,
                "command_topic": command_topic,
                "dpad_left_positive": dpad_left_positive,
            }],
        ),
    ])