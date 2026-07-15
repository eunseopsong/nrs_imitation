#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from nrs_imitation.inference_core import main as _main


DEFAULT_ROS_ARGS = [
    "--ros-args",
    "-p", "policy_class:=FLOW",
    "-p", "phase_mode:=pure",
    "-p", "obs_mode:=single_cam",
    "-p", "ckpt_auto_subdir:=gripper/single_cam",
    "-p", "use_gripper:=true",
    "-p", "use_stain_mask:=false",
    "-p", "camera_preprocess_mode:=stabilize",
    "-p", "pose_topic:=/ur10skku/currentP",
    "-p", "force_topic:=/ur10skku/currentF",
    "-p", "force_msg_type:=array",
    "-p", "image_topic:=/realsense/vr/color/image_raw",
    "-p", "cmd_topic:=/ur10skku/cmdMotion",
    "-p", "chunk_size:=200",
    "-p", "gripper_position_topic:=/gripper/present_position",
    "-p", "gripper_current_topic:=/gripper/present_current_mA",
    "-p", "gripper_command_topic:=/gripper/command",
    "-p", "control_hz:=125.0",
    "-p", "infer_hz:=5.0",
    "-p", "use_temporal_agg:=true",
    "-p", "temporal_agg_mode:=exp",
    "-p", "temporal_agg_tau_steps:=20.0",
    "-p", "pred_step_offset:=1",
    "-p", "max_plans:=6",
    "-p", "use_force_history:=true",
    "-p", "force_history_len:=10",
    "-p", "flow_infer_steps:=10",
    "-p", "auto_move_to_demo_start:=true",
    "-p", "demo_start_move_sec:=5.0",
    "-p", "demo_start_hold_sec:=2.0",
    "-p", "tau_sec:=0.8",
    "-p", "startup_ramp_sec:=3.0",
    "-p", "step_cap_pos_mm:=0.05",
    "-p", "step_cap_ang_rad:=0.0001",
    "-p", "step_cap_fz:=0.05",
    "-p", "fz_hard_limit:=30.0",
    "-p", "contact_on_thr:=3.0",
    "-p", "contact_off_thr:=1.2",
    "-p", "clear_plans_on_contact_change:=false",
    "-p", "dither_enable:=false",
    "-p", "pretrained_backbone:=false",
    "-p", "gripper_command_min_tick:=-653",
    "-p", "gripper_command_max_tick:=733",
    "-p", "gripper_command_deadband_tick:=2",
    "-p", "gripper_command_slew_per_sec:=1000.0",
    "-p", "gripper_command_step_cap_tick:=200.0",
    "-p", "gripper_cmd_safety_enable:=true",
    "-p", "gripper_cmd_safety_max_tick_from_present:=1500.0",
    "-p", "gradcam_enable:=true",
    "-p", "gradcam_publish:=true",
    "-p", "gradcam_every_n_infer:=1",
    "-p", "gradcam_target:=gripper",
    "-p", "gradcam_overlay_topic:=/inference_gripper_single_cam/gradcam_overlay",
]


def main(args=None):
    user_args = list(sys.argv[1:] if args is None else args)
    _main(args=[sys.argv[0]] + DEFAULT_ROS_ARGS + user_args, node_name="inference_gripper_single_cam")


if __name__ == "__main__":
    main()
