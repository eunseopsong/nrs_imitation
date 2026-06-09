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
    global_image_topic = LaunchConfiguration("global_image_topic")
    cmd_topic = LaunchConfiguration("cmd_topic")

    gradcam_enable = LaunchConfiguration("gradcam_enable")
    gradcam_publish = LaunchConfiguration("gradcam_publish")
    gradcam_every_n_infer = LaunchConfiguration("gradcam_every_n_infer")
    gradcam_target = LaunchConfiguration("gradcam_target")
    gradcam_target_step = LaunchConfiguration("gradcam_target_step")
    gradcam_target_horizon = LaunchConfiguration("gradcam_target_horizon")
    gradcam_layer_name = LaunchConfiguration("gradcam_layer_name")
    gradcam_overlay_topic = LaunchConfiguration("gradcam_overlay_topic")
    gradcam_global_overlay_topic = LaunchConfiguration("gradcam_global_overlay_topic")
    gradcam_save = LaunchConfiguration("gradcam_save")
    gradcam_save_dir = LaunchConfiguration("gradcam_save_dir")
    trajectory_viz_enable = LaunchConfiguration("trajectory_viz_enable")
    trajectory_viz_topic = LaunchConfiguration("trajectory_viz_topic")
    trajectory_viz_frame_id = LaunchConfiguration("trajectory_viz_frame_id")
    trajectory_viz_xyz_scale = LaunchConfiguration("trajectory_viz_xyz_scale")
    trajectory_viz_line_width_m = LaunchConfiguration("trajectory_viz_line_width_m")
    trajectory_viz_max_points = LaunchConfiguration("trajectory_viz_max_points")
    trajectory_viz_lifetime_sec = LaunchConfiguration("trajectory_viz_lifetime_sec")
    trajectory_overlay_enable = LaunchConfiguration("trajectory_overlay_enable")
    trajectory_overlay_max_points = LaunchConfiguration("trajectory_overlay_max_points")
    trajectory_overlay_origin = LaunchConfiguration("trajectory_overlay_origin")
    trajectory_overlay_pixels_per_mm = LaunchConfiguration("trajectory_overlay_pixels_per_mm")
    trajectory_overlay_center_u_ratio = LaunchConfiguration("trajectory_overlay_center_u_ratio")
    trajectory_overlay_center_v_ratio = LaunchConfiguration("trajectory_overlay_center_v_ratio")
    trajectory_overlay_line_width_px = LaunchConfiguration("trajectory_overlay_line_width_px")
    trajectory_overlay_point_radius_px = LaunchConfiguration("trajectory_overlay_point_radius_px")
    visualize = LaunchConfiguration("visualize")
    visualize_global = LaunchConfiguration("visualize_global")

    return LaunchDescription([
        DeclareLaunchArgument("ckpt_dir", default_value=""),
        DeclareLaunchArgument("act_root", default_value="~/nrs_imitation"),
        DeclareLaunchArgument("pose_topic", default_value="/ur10skku/currentP"),
        DeclareLaunchArgument("force_topic", default_value="/ur10skku/currentF"),
        DeclareLaunchArgument("image_topic", default_value="/realsense/robot/color/image_raw"),
        DeclareLaunchArgument("global_image_topic", default_value="/realsense/global/color/image_raw"),
        DeclareLaunchArgument("cmd_topic", default_value="/ur10skku/cmdMotion"),

        DeclareLaunchArgument("gradcam_enable", default_value="true"),
        DeclareLaunchArgument("gradcam_publish", default_value="true"),
        DeclareLaunchArgument("gradcam_every_n_infer", default_value="1"),
        DeclareLaunchArgument("gradcam_target", default_value="z"),
        DeclareLaunchArgument("gradcam_target_step", default_value="0"),
        DeclareLaunchArgument("gradcam_target_horizon", default_value="1"),
        DeclareLaunchArgument("gradcam_layer_name", default_value=""),
        DeclareLaunchArgument("gradcam_overlay_topic", default_value="/inference_dual_cam/gradcam_overlay"),
        DeclareLaunchArgument("gradcam_global_overlay_topic", default_value="/inference_dual_cam/gradcam_overlay_global"),
        DeclareLaunchArgument("gradcam_save", default_value="false"),
        DeclareLaunchArgument("gradcam_save_dir", default_value="~/nrs_imitation/gradcam"),
        DeclareLaunchArgument("trajectory_viz_enable", default_value="false"),
        DeclareLaunchArgument("trajectory_viz_topic", default_value="/inference_dual_cam/predicted_xyz_trajectory"),
        DeclareLaunchArgument("trajectory_viz_frame_id", default_value="base_link"),
        DeclareLaunchArgument("trajectory_viz_xyz_scale", default_value="0.001"),
        DeclareLaunchArgument("trajectory_viz_line_width_m", default_value="0.003"),
        DeclareLaunchArgument("trajectory_viz_max_points", default_value="200"),
        DeclareLaunchArgument("trajectory_viz_lifetime_sec", default_value="1.0"),
        DeclareLaunchArgument("trajectory_overlay_enable", default_value="true"),
        DeclareLaunchArgument("trajectory_overlay_max_points", default_value="80"),
        DeclareLaunchArgument("trajectory_overlay_origin", default_value="current"),
        DeclareLaunchArgument("trajectory_overlay_pixels_per_mm", default_value="8.0"),
        DeclareLaunchArgument("trajectory_overlay_center_u_ratio", default_value="0.50"),
        DeclareLaunchArgument("trajectory_overlay_center_v_ratio", default_value="0.55"),
        DeclareLaunchArgument("trajectory_overlay_line_width_px", default_value="3"),
        DeclareLaunchArgument("trajectory_overlay_point_radius_px", default_value="4"),
        DeclareLaunchArgument("visualize", default_value="true"),
        DeclareLaunchArgument("visualize_global", default_value="true"),

        Node(
            package="nrs_imitation",
            executable="inference_dual_cam",
            name="inference_dual_cam",
            output="screen",
            parameters=[{
                "ckpt_dir": ckpt_dir,
                "act_root": act_root,
                "pose_topic": pose_topic,
                "force_topic": force_topic,
                "image_topic": image_topic,
                "global_image_topic": global_image_topic,
                "cmd_topic": cmd_topic,
                "gradcam_enable": ParameterValue(gradcam_enable, value_type=bool),
                "gradcam_publish": ParameterValue(gradcam_publish, value_type=bool),
                "gradcam_every_n_infer": ParameterValue(gradcam_every_n_infer, value_type=int),
                "gradcam_target": gradcam_target,
                "gradcam_target_step": ParameterValue(gradcam_target_step, value_type=int),
                "gradcam_target_horizon": ParameterValue(gradcam_target_horizon, value_type=int),
                "gradcam_layer_name": gradcam_layer_name,
                "gradcam_overlay_topic": gradcam_overlay_topic,
                "gradcam_global_overlay_topic": gradcam_global_overlay_topic,
                "gradcam_save": ParameterValue(gradcam_save, value_type=bool),
                "gradcam_save_dir": gradcam_save_dir,
                "trajectory_viz_enable": ParameterValue(trajectory_viz_enable, value_type=bool),
                "trajectory_viz_topic": trajectory_viz_topic,
                "trajectory_viz_frame_id": trajectory_viz_frame_id,
                "trajectory_viz_xyz_scale": ParameterValue(trajectory_viz_xyz_scale, value_type=float),
                "trajectory_viz_line_width_m": ParameterValue(trajectory_viz_line_width_m, value_type=float),
                "trajectory_viz_max_points": ParameterValue(trajectory_viz_max_points, value_type=int),
                "trajectory_viz_lifetime_sec": ParameterValue(trajectory_viz_lifetime_sec, value_type=float),
                "trajectory_overlay_enable": ParameterValue(trajectory_overlay_enable, value_type=bool),
                "trajectory_overlay_max_points": ParameterValue(trajectory_overlay_max_points, value_type=int),
                "trajectory_overlay_origin": trajectory_overlay_origin,
                "trajectory_overlay_pixels_per_mm": ParameterValue(trajectory_overlay_pixels_per_mm, value_type=float),
                "trajectory_overlay_center_u_ratio": ParameterValue(trajectory_overlay_center_u_ratio, value_type=float),
                "trajectory_overlay_center_v_ratio": ParameterValue(trajectory_overlay_center_v_ratio, value_type=float),
                "trajectory_overlay_line_width_px": ParameterValue(trajectory_overlay_line_width_px, value_type=int),
                "trajectory_overlay_point_radius_px": ParameterValue(trajectory_overlay_point_radius_px, value_type=int),
            }],
        ),

        Node(
            package="rqt_image_view",
            executable="rqt_image_view",
            name="gradcam_viewer_local",
            output="screen",
            arguments=[gradcam_overlay_topic],
            condition=IfCondition(visualize),
        ),

        Node(
            package="rqt_image_view",
            executable="rqt_image_view",
            name="gradcam_viewer_global",
            output="screen",
            arguments=[gradcam_global_overlay_topic],
            condition=IfCondition(visualize_global),
        ),
    ])
