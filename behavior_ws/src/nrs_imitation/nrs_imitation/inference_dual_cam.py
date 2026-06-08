#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from nrs_imitation.inference_core import main as _main


# Mirrors the old README "Recommended Flow inference baseline" control defaults.
# Checkpoint selection stays on the current auto-latest dual-cam path.
DEFAULT_ROS_ARGS = [
    "--ros-args",
    "-p", "policy_class:=FLOW",
    "-p", "phase_mode:=pure",
    "-p", "obs_mode:=dual_cam",
    "-p", "ckpt_auto_subdir:=polishing/dual_cam",
    "-p", "camera_preprocess_mode:=stabilize",
    "-p", "image_topic:=/realsense/robot/color/image_raw",
    "-p", "global_image_topic:=/realsense/global/color/image_raw",
    "-p", "chunk_size:=200",
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
    "-p", "infer_hz:=5.0",
    "-p", "control_hz:=125.0",
    "-p", "temporal_agg_tau_steps:=20.0",
    "-p", "max_plans:=6",
    "-p", "contact_on_thr:=3.0",
    "-p", "contact_off_thr:=1.2",
    "-p", "clear_plans_on_contact_change:=false",
    "-p", "dither_enable:=false",
    "-p", "gradcam_enable:=true",
    "-p", "gradcam_publish:=true",
    "-p", "gradcam_every_n_infer:=1",
    "-p", "gradcam_target:=z",
    "-p", "gradcam_overlay_topic:=/inference_dual_cam/gradcam_overlay",
    "-p", "gradcam_global_overlay_topic:=/inference_dual_cam/gradcam_overlay_global",
]


def main(args=None):
    user_args = list(sys.argv[1:] if args is None else args)
    _main(args=[sys.argv[0]] + DEFAULT_ROS_ARGS + user_args, node_name="inference_dual_cam")


if __name__ == "__main__":
    main()
