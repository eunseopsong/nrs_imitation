#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    ckpt_dir = LaunchConfiguration("ckpt_dir")
    act_root = LaunchConfiguration("act_root")
    pose_topic = LaunchConfiguration("pose_topic")
    force_topic = LaunchConfiguration("force_topic")
    image_topic = LaunchConfiguration("image_topic")
    cmd_topic = LaunchConfiguration("cmd_topic")

    gradcam_enable = LaunchConfiguration("gradcam_enable")
    gradcam_publish = LaunchConfiguration("gradcam_publish")
    gradcam_every_n_infer = LaunchConfiguration("gradcam_every_n_infer")
    gradcam_target = LaunchConfiguration("gradcam_target")
    gradcam_target_step = LaunchConfiguration("gradcam_target_step")
    gradcam_target_horizon = LaunchConfiguration("gradcam_target_horizon")
    gradcam_layer_name = LaunchConfiguration("gradcam_layer_name")
    gradcam_overlay_topic = LaunchConfiguration("gradcam_overlay_topic")
    gradcam_save = LaunchConfiguration("gradcam_save")
    gradcam_save_dir = LaunchConfiguration("gradcam_save_dir")
    visualize = LaunchConfiguration("visualize")

    return LaunchDescription([
        DeclareLaunchArgument("ckpt_dir", default_value=""),
        DeclareLaunchArgument("act_root", default_value="~/nrs_imitation"),
        DeclareLaunchArgument("pose_topic", default_value="/ur10skku/currentP"),
        DeclareLaunchArgument("force_topic", default_value="/ur10skku/currentF"),
        DeclareLaunchArgument("image_topic", default_value="/realsense/robot/color/image_raw"),
        DeclareLaunchArgument("cmd_topic", default_value="/ur10skku/cmdMotion"),

        DeclareLaunchArgument("gradcam_enable", default_value="true"),
        DeclareLaunchArgument("gradcam_publish", default_value="true"),
        DeclareLaunchArgument("gradcam_every_n_infer", default_value="1"),
        DeclareLaunchArgument("gradcam_target", default_value="z"),
        DeclareLaunchArgument("gradcam_target_step", default_value="0"),
        DeclareLaunchArgument("gradcam_target_horizon", default_value="1"),
        DeclareLaunchArgument("gradcam_layer_name", default_value=""),
        DeclareLaunchArgument("gradcam_overlay_topic", default_value="/inference_single_cam/gradcam_overlay"),
        DeclareLaunchArgument("gradcam_save", default_value="false"),
        DeclareLaunchArgument("gradcam_save_dir", default_value="~/nrs_imitation/gradcam"),
        DeclareLaunchArgument("visualize", default_value="true"),

        Node(
            package="nrs_imitation",
            executable="inference_single_cam",
            name="inference_single_cam",
            output="screen",
            parameters=[{
                "ckpt_dir": ckpt_dir,
                "act_root": act_root,
                "pose_topic": pose_topic,
                "force_topic": force_topic,
                "image_topic": image_topic,
                "cmd_topic": cmd_topic,
                "gradcam_enable": ParameterValue(gradcam_enable, value_type=bool),
                "gradcam_publish": ParameterValue(gradcam_publish, value_type=bool),
                "gradcam_every_n_infer": ParameterValue(gradcam_every_n_infer, value_type=int),
                "gradcam_target": gradcam_target,
                "gradcam_target_step": ParameterValue(gradcam_target_step, value_type=int),
                "gradcam_target_horizon": ParameterValue(gradcam_target_horizon, value_type=int),
                "gradcam_layer_name": gradcam_layer_name,
                "gradcam_overlay_topic": gradcam_overlay_topic,
                "gradcam_save": ParameterValue(gradcam_save, value_type=bool),
                "gradcam_save_dir": gradcam_save_dir,
            }],
        ),

        Node(
            package="rqt_image_view",
            executable="rqt_image_view",
            name="gradcam_viewer",
            output="screen",
            arguments=[gradcam_overlay_topic],
            condition=IfCondition(visualize),
        ),
    ])
