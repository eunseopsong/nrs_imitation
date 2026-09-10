#!/usr/bin/env python3
"""Direction-auto-detecting inference launch.

Two-phase launch, both driven from a single `ros2 launch` command:

  Phase 1 (neutral): PTPs the arm straight to the neutral viewing pose
  through the SingleArmCommand service (move_to_pose_and_grab_frames) -- no
  inference stack, no Force mode, no PTP9D stream. The neutral pose is the
  base checkpoint's demo_start_pose_arithmetic_mean, a genuine 0deg/90deg
  midpoint that gives the wrist camera a usable, direction-unbiased view of
  the workpiece. We wait for /ur10skku/currentP to settle there, THEN grab
  classifier frames -- grabbing them before the arm has moved off its
  arbitrary idle pose was tried first and always misclassified.

  An earlier version ran a whole base-checkpoint inference stack for phase 1
  and SIGINT'd it; that entered TRACK + Force mode and left the arm
  force-loaded / mid-trajectory, so phase 2's alignment PTP crashed ~40mm
  into the workpiece. The lightweight PTP avoids all of that.

  Phase 2 (final): classifies the settled frames (0deg vs 90deg stain) and
  launches inference_clean_single_cam.launch.py with whichever per-direction
  FLOW policy (ckpt_dir_0deg / ckpt_dir_90deg) matches -- each is a
  standalone policy trained on that direction only (42 EP each) and carries
  its own demo_start_pose_mean, so the real run starts aligned to the
  direction actually in front of it.

Usage (same style as the normal inference launch):
  ros2 launch nrs_imitation inference_autodir_single_cam.launch.py \\
    inference_mode:=service_stream use_stain_mask:=false \\
    metrics_log_enable:=true
"""
import os
import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

PROJECT_ROOT = Path("/home/eunseop/nrs_imitation")
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "direction_classifier"))


def _pick_ckpt_and_include(context, *args, **kwargs):
    # Imported here (not at module top) so `torch` loads before `rclpy` --
    # see the same note in pick_start_and_launch.py.
    import numpy as np

    from pick_start_and_launch import (
        LABEL_NOVEL_TYPE,
        LABEL_UNMATCHED,
        classify_direction,
        classify_stain_by_reference,
        load_demo_start_pose_mean,
        move_to_pose_and_grab_frames,
    )

    image_topic = LaunchConfiguration("image_topic").perform(context)
    pose_topic = LaunchConfiguration("pose_topic").perform(context)
    num_frames = int(LaunchConfiguration("classifier_num_frames").perform(context))
    settle_pos_tol_mm = float(LaunchConfiguration("settle_pos_tol_mm").perform(context))
    settle_rot_tol_rad = float(LaunchConfiguration("settle_rot_tol_rad").perform(context))
    settle_hold_sec = float(LaunchConfiguration("settle_hold_sec").perform(context))
    settle_timeout_sec = float(LaunchConfiguration("settle_timeout_sec").perform(context))
    classifier_ckpt = LaunchConfiguration("classifier_ckpt").perform(context)
    reference_npz = LaunchConfiguration("reference_npz").perform(context)
    base_ckpt_dir = LaunchConfiguration("base_ckpt_dir").perform(context)
    ckpt_dir_0deg = LaunchConfiguration("ckpt_dir_0deg").perform(context)
    ckpt_dir_90deg = LaunchConfiguration("ckpt_dir_90deg").perform(context)
    ckpt_dir_other = LaunchConfiguration("ckpt_dir_other").perform(context)
    inference_mode = LaunchConfiguration("inference_mode").perform(context)
    use_stain_mask = LaunchConfiguration("use_stain_mask").perform(context)
    metrics_log_enable = LaunchConfiguration("metrics_log_enable").perform(context)

    use_reference = bool(reference_npz) and os.path.isfile(reference_npz)

    # Neutral / classification pose. With a reference template we classify
    # from the exact home pose the template was captured at (stored in the
    # npz). Without one, fall back to the base ckpt's 0deg/90deg midpoint
    # (demo_start_pose_arithmetic_mean) -- NOT the KNN pick, which for
    # balanced42 lands on the 90deg start pose and biases every prediction.
    if use_reference:
        neutral_target_pose6 = np.asarray(
            np.load(reference_npz, allow_pickle=True)["pose_mean"], dtype=float)
        print(f"[autodir] phase 1: classification pose from reference = {neutral_target_pose6.tolist()}")
    else:
        neutral_target_pose6 = load_demo_start_pose_mean(base_ckpt_dir, prefer_arithmetic=True)
        print(f"[autodir] phase 1: neutral pose (0deg/90deg midpoint) = {neutral_target_pose6.tolist()}")

    print(f"[autodir] phase 1: move to neutral pose + grab {num_frames} frames "
          f"(tol={settle_pos_tol_mm}mm/{settle_rot_tol_rad}rad, hold={settle_hold_sec}s, "
          f"timeout={settle_timeout_sec}s) ...")
    frames = move_to_pose_and_grab_frames(
        target_pose6=neutral_target_pose6,
        image_topic=image_topic,
        pose_topic=pose_topic,
        num_frames=num_frames,
        pos_tol_mm=settle_pos_tol_mm,
        rot_tol_rad=settle_rot_tol_rad,
        hold_sec=settle_hold_sec,
        timeout_sec=settle_timeout_sec,
    )

    if not frames:
        raise RuntimeError(
            f"never settled at neutral pose / no frames grabbed within {settle_timeout_sec}s "
            f"(image_topic={image_topic}, pose_topic={pose_topic})"
        )
    print(f"[autodir] got {len(frames)} frame(s) at neutral pose")

    if use_reference:
        label, info = classify_stain_by_reference(frames, reference_npz)
    else:
        print(f"[autodir] WARN: reference_npz '{reference_npz}' not found -- "
              "falling back to the DINOv3 classifier (known to background-bias at a fixed pose)")
        label, info = classify_direction(frames, classifier_ckpt), {}

    # The reference classifier can now say "this is not a 0/90 direction
    # stain": a templated non-direction type (e.g. the 2/3/4-dot rows, labels
    # 2/3/4 or the legacy LABEL_NOVEL_TYPE=-1) or LABEL_UNMATCHED (matched no
    # template at all). Running a direction policy on any of those is wrong,
    # so stop here unless the operator has wired a policy via ckpt_dir_other.
    name = info.get("label_name", "unknown") if isinstance(info, dict) else "unknown"
    is_novel_type = label not in (0, 90) and label != LABEL_UNMATCHED
    if label == 0 or label == 90:
        direction = label
        ckpt_dir = ckpt_dir_0deg if label == 0 else ckpt_dir_90deg
        print(f"[autodir] predicted {label}deg -> ckpt_dir={ckpt_dir}")
    elif is_novel_type and ckpt_dir_other:
        direction = label
        ckpt_dir = ckpt_dir_other
        print(f"[autodir] classified non-direction stain type '{name}' (label={label}) "
              f"-> ckpt_dir_other={ckpt_dir}")
    else:
        raise RuntimeError(
            f"[autodir] stain classified as '{name}' (label={label}), not a known "
            f"polishing direction (0/90). Refusing to run a direction policy on an "
            f"unrecognised stain. classifier info: {info}. "
            f"Draw a 0deg or 90deg stain, or pass ckpt_dir_other:=<policy> if you "
            f"have one trained for this type."
        )

    base_launch = PathJoinSubstitution(
        [FindPackageShare("nrs_imitation"), "launch", "inference_clean_single_cam.launch.py"]
    )
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(base_launch),
            launch_arguments={
                "policy_class": "FLOW",
                "ckpt_dir": ckpt_dir,
                "inference_mode": LaunchConfiguration("inference_mode"),
                "use_stain_mask": LaunchConfiguration("use_stain_mask"),
                "metrics_log_enable": LaunchConfiguration("metrics_log_enable"),
                "metrics_run_tag": f"autodir_{direction}deg",
                "image_topic": LaunchConfiguration("image_topic"),
                "pose_topic": LaunchConfiguration("pose_topic"),
                "stain_origin_topic": LaunchConfiguration("stain_origin_topic"),
                # inference_clean defaults gradcam_enable=true, which adds a
                # backward pass through DINOv3 every replan tick. On the
                # phase-2 operational run that stalled replanning to ~5s
                # (MODALITY compute 46ms -> 3500ms) and the robot held a
                # stale force-mode target between plans -> ESTOP. Keep the
                # heavy diagnostic off here, same as phase 1 does.
                "gradcam_enable": LaunchConfiguration("gradcam_enable"),
                "overlay_record_enable": LaunchConfiguration("overlay_record_enable"),
                "removal_viz_enable": LaunchConfiguration("removal_viz_enable"),
                "removal_heatmap_tail_sec": LaunchConfiguration("removal_heatmap_tail_sec"),
                "removal_view_margin_mm": LaunchConfiguration("removal_view_margin_mm"),
            }.items(),
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("image_topic", default_value="/realsense/vr/color/image_raw"),
            DeclareLaunchArgument("pose_topic", default_value="/ur10skku/currentP"),
            DeclareLaunchArgument(
                "stain_origin_topic", default_value="/stain_relative_frame/stain_origin"
            ),
            DeclareLaunchArgument("classifier_num_frames", default_value="5"),
            # "Arrived at the neutral viewing pose" tolerance. Looser than
            # inference_core's own demo-start tol (5mm/0.05rad): a raw PTP
            # lands a little coarser than the inference stack's alignment
            # loop, and the classifier only needs the camera roughly at the
            # viewing pose, not mm precision.
            DeclareLaunchArgument("settle_pos_tol_mm", default_value="12.0"),
            DeclareLaunchArgument("settle_rot_tol_rad", default_value="0.10"),
            DeclareLaunchArgument("settle_hold_sec", default_value="1.0"),
            DeclareLaunchArgument("settle_timeout_sec", default_value="40.0"),
            # Primary stain check: reference-difference against the home-pose
            # templates (0deg / 90deg / any templated novel type, each drawn at
            # the pose stored inside the npz). No network, no training -- see
            # build_home_reference.py and classify_stain_by_reference(). Can
            # also return "not a 0/90 stain" (label -1 or -2), in which case
            # autodir aborts unless ckpt_dir_other is set. If this file is
            # missing, autodir falls back to classifier_ckpt (the DINOv3 head,
            # which background-biases at a fixed pose -- keep the npz current).
            DeclareLaunchArgument(
                "reference_npz",
                default_value=str(PROJECT_ROOT / "checkpoints" / "direction_classifier" / "home_reference.npz"),
            ),
            DeclareLaunchArgument(
                "classifier_ckpt",
                default_value=str(PROJECT_ROOT / "checkpoints" / "direction_classifier" / "classifier.pt"),
            ),
            # Direction-agnostic checkpoint used ONLY for the phase-1 neutral
            # settling pose (its demo_start_pose_mean is a 0deg/90deg midpoint
            # that gives the wrist camera a usable view of the workpiece).
            DeclareLaunchArgument(
                "base_ckpt_dir",
                default_value="/home/eunseop/nrs_imitation/checkpoints/flow/polishing/single_cam/20260826_1242",
            ),
            # Per-direction FLOW policies (DINOv3 + tcp_roi, 42 EP each,
            # 500 epoch, trained 20260827-28). 0deg was variance-edited to
            # match the 90deg inter-episode spread (dataset
            # 20260821_0deg_phasematch90); 90deg is the untouched split.
            DeclareLaunchArgument(
                "ckpt_dir_0deg",
                default_value="/home/eunseop/nrs_imitation/checkpoints/flow/polishing/single_cam/20260827_2128_phasematch90_0deg/20260827_2129",
            ),
            DeclareLaunchArgument(
                "ckpt_dir_90deg",
                default_value="/home/eunseop/nrs_imitation/checkpoints/flow/polishing/single_cam/20260827_2128_only90deg/20260828_0341",
            ),
            # Optional policy for a templated NON-direction stain type
            # (classifier label -1, e.g. the 3-dot row). Empty => autodir
            # aborts with a clear message when it sees such a stain instead
            # of running a 0/90 policy on it.
            DeclareLaunchArgument("ckpt_dir_other", default_value=""),
            DeclareLaunchArgument("inference_mode", default_value="service_stream"),
            DeclareLaunchArgument("use_stain_mask", default_value="false"),
            DeclareLaunchArgument("metrics_log_enable", default_value="true"),
            # Operational defaults for the phase-2 policy run. GradCam off:
            # it puts an extra DINOv3 backward pass on the replan path and
            # was the cause of the ~5s replan stall / force-mode ESTOP.
            # Overlay video stays on (separate process, off the control path).
            DeclareLaunchArgument("gradcam_enable", default_value="false"),
            DeclareLaunchArgument("overlay_record_enable", default_value="true"),
            # Polishing-removal heatmap of the phase-2 (policy) trajectory,
            # written on shutdown. See polishing_removal_recorder.
            DeclareLaunchArgument("removal_viz_enable", default_value="true"),
            DeclareLaunchArgument("removal_heatmap_tail_sec", default_value="15.0"),
            DeclareLaunchArgument("removal_view_margin_mm", default_value="40.0"),
            OpaqueFunction(function=_pick_ckpt_and_include),
        ]
    )
