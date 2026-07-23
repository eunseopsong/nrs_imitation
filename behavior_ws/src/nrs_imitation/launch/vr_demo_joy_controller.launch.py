#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    joy_topic = LaunchConfiguration("joy_topic")
    command_topic = LaunchConfiguration("command_topic")
    gripper_command_topic = LaunchConfiguration("gripper_command_topic")
    gripper_present_position_topic = LaunchConfiguration(
        "gripper_present_position_topic"
    )
    button_start = LaunchConfiguration("button_start")
    button_end = LaunchConfiguration("button_end")
    button_gripper_close = LaunchConfiguration("button_gripper_close")
    button_gripper_open = LaunchConfiguration("button_gripper_open")
    gripper_close_tick = LaunchConfiguration("gripper_close_tick")
    gripper_open_tick = LaunchConfiguration("gripper_open_tick")
    gripper_step_tick = LaunchConfiguration("gripper_step_tick")
    fine_close_axis = LaunchConfiguration("fine_close_axis")
    fine_open_axis = LaunchConfiguration("fine_open_axis")
    fine_trigger_threshold = LaunchConfiguration("fine_trigger_threshold")
    fine_repeat_sec = LaunchConfiguration("fine_repeat_sec")
    dpad_axis = LaunchConfiguration("dpad_axis")
    dpad_threshold = LaunchConfiguration("dpad_threshold")
    dpad_invert = LaunchConfiguration("dpad_invert")
    dpad_repeat_sec = LaunchConfiguration("dpad_repeat_sec")
    device_id = LaunchConfiguration("device_id")
    deadzone = LaunchConfiguration("deadzone")
    autorepeat_rate = LaunchConfiguration("autorepeat_rate")

    return LaunchDescription([
        DeclareLaunchArgument("joy_topic", default_value="/joy_il"),
        DeclareLaunchArgument("command_topic", default_value="/vr_demo_recorder/command"),
        DeclareLaunchArgument("gripper_command_topic", default_value="/gripper/command"),
        DeclareLaunchArgument(
            "gripper_present_position_topic",
            default_value="/gripper/present_position",
        ),
        DeclareLaunchArgument("button_start", default_value="5"),
        DeclareLaunchArgument("button_end", default_value="4"),
        DeclareLaunchArgument("button_gripper_close", default_value="7"),
        DeclareLaunchArgument("button_gripper_open", default_value="6"),
        DeclareLaunchArgument("gripper_close_tick", default_value="-653"),
        DeclareLaunchArgument("gripper_open_tick", default_value="733"),
        DeclareLaunchArgument("gripper_step_tick", default_value="50"),
        DeclareLaunchArgument("fine_close_axis", default_value="5"),
        DeclareLaunchArgument("fine_open_axis", default_value="2"),
        DeclareLaunchArgument("fine_trigger_threshold", default_value="0.50"),
        DeclareLaunchArgument("fine_repeat_sec", default_value="0.15"),
        DeclareLaunchArgument("dpad_axis", default_value="-1"),
        DeclareLaunchArgument("dpad_threshold", default_value="0.50"),
        DeclareLaunchArgument("dpad_invert", default_value="false"),
        DeclareLaunchArgument("dpad_repeat_sec", default_value="0.15"),
        DeclareLaunchArgument("device_id", default_value="0"),
        DeclareLaunchArgument("deadzone", default_value="0.05"),
        DeclareLaunchArgument("autorepeat_rate", default_value="20.0"),

        Node(
            package="joy",
            executable="joy_node",
            name="joy_node_il",
            output="screen",
            parameters=[{
                "device_id": ParameterValue(device_id, value_type=int),
                "deadzone": ParameterValue(deadzone, value_type=float),
                "autorepeat_rate": ParameterValue(autorepeat_rate, value_type=float),
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
                "gripper_command_topic": gripper_command_topic,
                "gripper_present_position_topic": gripper_present_position_topic,
                "button_start": ParameterValue(button_start, value_type=int),
                "button_end": ParameterValue(button_end, value_type=int),
                "button_gripper_close": ParameterValue(
                    button_gripper_close,
                    value_type=int,
                ),
                "button_gripper_open": ParameterValue(
                    button_gripper_open,
                    value_type=int,
                ),
                "gripper_close_tick": ParameterValue(
                    gripper_close_tick,
                    value_type=int,
                ),
                "gripper_open_tick": ParameterValue(
                    gripper_open_tick,
                    value_type=int,
                ),
                "gripper_step_tick": ParameterValue(
                    gripper_step_tick,
                    value_type=int,
                ),
                "fine_close_axis": ParameterValue(
                    fine_close_axis,
                    value_type=int,
                ),
                "fine_open_axis": ParameterValue(
                    fine_open_axis,
                    value_type=int,
                ),
                "fine_trigger_threshold": ParameterValue(
                    fine_trigger_threshold,
                    value_type=float,
                ),
                "fine_repeat_sec": ParameterValue(
                    fine_repeat_sec,
                    value_type=float,
                ),
                "dpad_axis": ParameterValue(dpad_axis, value_type=int),
                "dpad_threshold": ParameterValue(dpad_threshold, value_type=float),
                "dpad_invert": ParameterValue(dpad_invert, value_type=bool),
                "dpad_repeat_sec": ParameterValue(
                    dpad_repeat_sec,
                    value_type=float,
                ),
            }],
        ),
    ])
