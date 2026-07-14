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
    policy_class = LaunchConfiguration("policy_class")
    ckpt_auto_subdir = LaunchConfiguration("ckpt_auto_subdir")
    pose_topic = LaunchConfiguration("pose_topic")
    force_topic = LaunchConfiguration("force_topic")
    image_topic = LaunchConfiguration("image_topic")
    use_stain_mask = LaunchConfiguration("use_stain_mask")
    stain_mask_topic = LaunchConfiguration("stain_mask_topic")
    auto_stain_mask = LaunchConfiguration("auto_stain_mask")
    stain_mask_overlay_topic = LaunchConfiguration("stain_mask_overlay_topic")
    publish_stain_mask_overlay = LaunchConfiguration("publish_stain_mask_overlay")
    stain_dark_thresh = LaunchConfiguration("stain_dark_thresh")
    reflection_v_thresh = LaunchConfiguration("reflection_v_thresh")
    reflection_s_thresh = LaunchConfiguration("reflection_s_thresh")
    stain_min_area = LaunchConfiguration("stain_min_area")
    stain_morph_kernel = LaunchConfiguration("stain_morph_kernel")
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
        DeclareLaunchArgument("policy_class", default_value="FLOW"),
        DeclareLaunchArgument("ckpt_auto_subdir", default_value="polishing/single_cam"),
        DeclareLaunchArgument("pose_topic", default_value="/ur10skku/currentP"),
        DeclareLaunchArgument("force_topic", default_value="/ur10skku/currentF"),
        DeclareLaunchArgument("image_topic", default_value="/realsense/vr/color/image_raw"),
        DeclareLaunchArgument("use_stain_mask", default_value="false"),
        DeclareLaunchArgument("stain_mask_topic", default_value="/inference_single_cam/stain_mask"),
        DeclareLaunchArgument("auto_stain_mask", default_value="false"),
        DeclareLaunchArgument("stain_mask_overlay_topic", default_value="/inference_single_cam/stain_mask_overlay"),
        DeclareLaunchArgument("publish_stain_mask_overlay", default_value="true"),
        DeclareLaunchArgument("stain_dark_thresh", default_value="80"),
        DeclareLaunchArgument("reflection_v_thresh", default_value="235"),
        DeclareLaunchArgument("reflection_s_thresh", default_value="60"),
        DeclareLaunchArgument("stain_min_area", default_value="20"),
        DeclareLaunchArgument("stain_morph_kernel", default_value="3"),
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
            executable="stain_mask_publisher",
            name="stain_mask_publisher",
            output="screen",
            parameters=[{
                "image_topic": image_topic,
                "mask_topic": stain_mask_topic,
                "overlay_topic": stain_mask_overlay_topic,
                "publish_overlay": ParameterValue(publish_stain_mask_overlay, value_type=bool),
                "stain_dark_thresh": ParameterValue(stain_dark_thresh, value_type=int),
                "reflection_v_thresh": ParameterValue(reflection_v_thresh, value_type=int),
                "reflection_s_thresh": ParameterValue(reflection_s_thresh, value_type=int),
                "stain_min_area": ParameterValue(stain_min_area, value_type=int),
                "stain_morph_kernel": ParameterValue(stain_morph_kernel, value_type=int),
            }],
            condition=IfCondition(auto_stain_mask),
        ),

        Node(
            package="nrs_imitation",
            executable="inference_single_cam",
            name="inference_single_cam",
            output="screen",
            parameters=[{
                "ckpt_dir": ckpt_dir,
                "act_root": act_root,
                "policy_class": policy_class,
                "ckpt_auto_subdir": ckpt_auto_subdir,
                "pose_topic": pose_topic,
                "force_topic": force_topic,
                "image_topic": image_topic,
                "use_stain_mask": ParameterValue(use_stain_mask, value_type=bool),
                "stain_mask_topic": stain_mask_topic,
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
