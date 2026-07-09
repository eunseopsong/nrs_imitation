#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2026 eunseop

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("dynamixel_joy_controller"), "config", "f710_gripper_joy.yaml"]
    )

    config_file = LaunchConfiguration("config_file")
    joy_topic = LaunchConfiguration("joy_topic")
    command_topic = LaunchConfiguration("command_topic")
    present_position_topic = LaunchConfiguration("present_position_topic")
    device_id = LaunchConfiguration("device_id")
    deadzone = LaunchConfiguration("deadzone")
    autorepeat_rate = LaunchConfiguration("autorepeat_rate")
    min_tick = LaunchConfiguration("min_tick")
    max_tick = LaunchConfiguration("max_tick")
    step_tick = LaunchConfiguration("step_tick")
    button_open = LaunchConfiguration("button_open")
    button_close = LaunchConfiguration("button_close")
    button_home = LaunchConfiguration("button_home")
    axis_control_enabled = LaunchConfiguration("axis_control_enabled")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument("joy_topic", default_value="/joy_f710"),
            DeclareLaunchArgument("command_topic", default_value="/gripper/command"),
            DeclareLaunchArgument(
                "present_position_topic",
                default_value="/gripper/present_position",
            ),
            DeclareLaunchArgument("device_id", default_value="0"),
            DeclareLaunchArgument("deadzone", default_value="0.05"),
            DeclareLaunchArgument("autorepeat_rate", default_value="5.0"),
            DeclareLaunchArgument("min_tick", default_value="590"),
            DeclareLaunchArgument("max_tick", default_value="2500"),
            DeclareLaunchArgument("step_tick", default_value="50"),
            DeclareLaunchArgument("button_open", default_value="1"),
            DeclareLaunchArgument("button_close", default_value="0"),
            DeclareLaunchArgument("button_home", default_value="-1"),
            DeclareLaunchArgument("axis_control_enabled", default_value="false"),
            Node(
                package="joy",
                executable="joy_node",
                name="joy_node_f710",
                output="screen",
                parameters=[
                    {
                        "device_id": ParameterValue(device_id, value_type=int),
                        "deadzone": ParameterValue(deadzone, value_type=float),
                        "autorepeat_rate": ParameterValue(
                            autorepeat_rate,
                            value_type=float,
                        ),
                    }
                ],
                remappings=[
                    ("/joy", joy_topic),
                ],
            ),
            Node(
                package="dynamixel_joy_controller",
                executable="f710_gripper_joy_controller",
                name="f710_gripper_joy_controller",
                output="screen",
                parameters=[
                    config_file,
                    {
                        "joy_topic": joy_topic,
                        "command_topic": command_topic,
                        "present_position_topic": present_position_topic,
                        "min_tick": ParameterValue(min_tick, value_type=int),
                        "max_tick": ParameterValue(max_tick, value_type=int),
                        "step_tick": ParameterValue(step_tick, value_type=int),
                        "button_open": ParameterValue(button_open, value_type=int),
                        "button_close": ParameterValue(button_close, value_type=int),
                        "button_home": ParameterValue(button_home, value_type=int),
                        "axis_control.enabled": ParameterValue(
                            axis_control_enabled,
                            value_type=bool,
                        ),
                    },
                ],
            ),
        ]
    )
