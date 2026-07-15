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
    force_msg_type = LaunchConfiguration("force_msg_type")
    image_topic = LaunchConfiguration("image_topic")
    cmd_topic = LaunchConfiguration("cmd_topic")
    gripper_position_topic = LaunchConfiguration("gripper_position_topic")
    gripper_current_topic = LaunchConfiguration("gripper_current_topic")
    gripper_command_topic = LaunchConfiguration("gripper_command_topic")
    control_hz = LaunchConfiguration("control_hz")
    infer_hz = LaunchConfiguration("infer_hz")
    use_temporal_agg = LaunchConfiguration("use_temporal_agg")
    temporal_agg_mode = LaunchConfiguration("temporal_agg_mode")
    temporal_agg_tau_steps = LaunchConfiguration("temporal_agg_tau_steps")
    pred_step_offset = LaunchConfiguration("pred_step_offset")
    max_plans = LaunchConfiguration("max_plans")
    gripper_command_min_tick = LaunchConfiguration("gripper_command_min_tick")
    gripper_command_max_tick = LaunchConfiguration("gripper_command_max_tick")
    gripper_command_deadband_tick = LaunchConfiguration("gripper_command_deadband_tick")
    gripper_command_slew_per_sec = LaunchConfiguration("gripper_command_slew_per_sec")
    tau_sec = LaunchConfiguration("tau_sec")
    startup_ramp_sec = LaunchConfiguration("startup_ramp_sec")
    gripper_command_step_cap_tick = LaunchConfiguration("gripper_command_step_cap_tick")
    gripper_cmd_safety_enable = LaunchConfiguration("gripper_cmd_safety_enable")
    gripper_cmd_safety_max_tick_from_present = LaunchConfiguration("gripper_cmd_safety_max_tick_from_present")

    gradcam_enable = LaunchConfiguration("gradcam_enable")
    gradcam_publish = LaunchConfiguration("gradcam_publish")
    gradcam_every_n_infer = LaunchConfiguration("gradcam_every_n_infer")
    gradcam_target = LaunchConfiguration("gradcam_target")
    gradcam_overlay_topic = LaunchConfiguration("gradcam_overlay_topic")
    visualize = LaunchConfiguration("visualize")

    return LaunchDescription([
        DeclareLaunchArgument("ckpt_dir", default_value=""),
        DeclareLaunchArgument("act_root", default_value="~/nrs_imitation"),
        DeclareLaunchArgument("policy_class", default_value="FLOW"),
        DeclareLaunchArgument("ckpt_auto_subdir", default_value="gripper/single_cam"),
        DeclareLaunchArgument("pose_topic", default_value="/ur10skku/currentP"),
        DeclareLaunchArgument("force_topic", default_value="/ur10skku/currentF"),
        DeclareLaunchArgument("force_msg_type", default_value="array"),
        DeclareLaunchArgument("image_topic", default_value="/realsense/vr/color/image_raw"),
        DeclareLaunchArgument("cmd_topic", default_value="/ur10skku/cmdMotion"),
        DeclareLaunchArgument("gripper_position_topic", default_value="/gripper/present_position"),
        DeclareLaunchArgument("gripper_current_topic", default_value="/gripper/present_current_mA"),
        DeclareLaunchArgument("gripper_command_topic", default_value="/gripper/command"),
        DeclareLaunchArgument("control_hz", default_value="125.0"),
        DeclareLaunchArgument("infer_hz", default_value="5.0"),
        DeclareLaunchArgument("use_temporal_agg", default_value="true"),
        DeclareLaunchArgument("temporal_agg_mode", default_value="exp"),
        DeclareLaunchArgument("temporal_agg_tau_steps", default_value="20.0"),
        DeclareLaunchArgument("pred_step_offset", default_value="1"),
        DeclareLaunchArgument("max_plans", default_value="6"),
        DeclareLaunchArgument("gripper_command_min_tick", default_value="-653"),
        DeclareLaunchArgument("gripper_command_max_tick", default_value="733"),
        DeclareLaunchArgument("gripper_command_deadband_tick", default_value="2"),
        DeclareLaunchArgument("gripper_command_slew_per_sec", default_value="1000.0"),
        DeclareLaunchArgument("tau_sec", default_value="0.8"),
        DeclareLaunchArgument("startup_ramp_sec", default_value="3.0"),
        DeclareLaunchArgument("gripper_command_step_cap_tick", default_value="200.0"),
        DeclareLaunchArgument("gripper_cmd_safety_enable", default_value="true"),
        DeclareLaunchArgument("gripper_cmd_safety_max_tick_from_present", default_value="1500.0"),

        DeclareLaunchArgument("gradcam_enable", default_value="true"),
        DeclareLaunchArgument("gradcam_publish", default_value="true"),
        DeclareLaunchArgument("gradcam_every_n_infer", default_value="1"),
        DeclareLaunchArgument("gradcam_target", default_value="gripper"),
        DeclareLaunchArgument("gradcam_overlay_topic", default_value="/inference_gripper_single_cam/gradcam_overlay"),
        DeclareLaunchArgument("visualize", default_value="true"),

        Node(
            package="nrs_imitation",
            executable="inference_gripper_single_cam",
            name="inference_gripper_single_cam",
            output="screen",
            parameters=[{
                "ckpt_dir": ckpt_dir,
                "act_root": act_root,
                "policy_class": policy_class,
                "ckpt_auto_subdir": ckpt_auto_subdir,
                "use_gripper": True,
                "use_stain_mask": False,
                "obs_mode": "single_cam",
                "pose_topic": pose_topic,
                "force_topic": force_topic,
                "force_msg_type": force_msg_type,
                "image_topic": image_topic,
                "cmd_topic": cmd_topic,
                "gripper_position_topic": gripper_position_topic,
                "gripper_current_topic": gripper_current_topic,
                "gripper_command_topic": gripper_command_topic,
                "control_hz": ParameterValue(control_hz, value_type=float),
                "infer_hz": ParameterValue(infer_hz, value_type=float),
                "use_temporal_agg": ParameterValue(use_temporal_agg, value_type=bool),
                "temporal_agg_mode": temporal_agg_mode,
                "temporal_agg_tau_steps": ParameterValue(temporal_agg_tau_steps, value_type=float),
                "pred_step_offset": ParameterValue(pred_step_offset, value_type=int),
                "max_plans": ParameterValue(max_plans, value_type=int),
                "gripper_command_min_tick": ParameterValue(gripper_command_min_tick, value_type=int),
                "gripper_command_max_tick": ParameterValue(gripper_command_max_tick, value_type=int),
                "gripper_command_deadband_tick": ParameterValue(gripper_command_deadband_tick, value_type=int),
                "gripper_command_slew_per_sec": ParameterValue(gripper_command_slew_per_sec, value_type=float),
                "tau_sec": ParameterValue(tau_sec, value_type=float),
                "startup_ramp_sec": ParameterValue(startup_ramp_sec, value_type=float),
                "gripper_command_step_cap_tick": ParameterValue(gripper_command_step_cap_tick, value_type=float),
                "gripper_cmd_safety_enable": ParameterValue(gripper_cmd_safety_enable, value_type=bool),
                "gripper_cmd_safety_max_tick_from_present": ParameterValue(gripper_cmd_safety_max_tick_from_present, value_type=float),
                "gradcam_enable": ParameterValue(gradcam_enable, value_type=bool),
                "gradcam_publish": ParameterValue(gradcam_publish, value_type=bool),
                "gradcam_every_n_infer": ParameterValue(gradcam_every_n_infer, value_type=int),
                "gradcam_target": gradcam_target,
                "gradcam_overlay_topic": gradcam_overlay_topic,
            }],
        ),

        Node(
            package="rqt_image_view",
            executable="rqt_image_view",
            name="gripper_gradcam_viewer",
            output="screen",
            arguments=[gradcam_overlay_topic],
            condition=IfCondition(visualize),
        ),
    ])
