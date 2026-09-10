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
    metrics_log_enable = LaunchConfiguration("metrics_log_enable")
    metrics_log_dir = LaunchConfiguration("metrics_log_dir")
    metrics_run_tag = LaunchConfiguration("metrics_run_tag")
    overlay_record_enable = LaunchConfiguration("overlay_record_enable")
    removal_viz_enable = LaunchConfiguration("removal_viz_enable")
    removal_viz_open_on_exit = LaunchConfiguration("removal_viz_open_on_exit")
    removal_heatmap_tail_sec = LaunchConfiguration("removal_heatmap_tail_sec")
    removal_view_margin_mm = LaunchConfiguration("removal_view_margin_mm")
    pose_topic = LaunchConfiguration("pose_topic")
    force_topic = LaunchConfiguration("force_topic")
    image_topic = LaunchConfiguration("image_topic")
    use_stain_mask = LaunchConfiguration("use_stain_mask")
    stain_mask_topic = LaunchConfiguration("stain_mask_topic")
    stain_origin_topic = LaunchConfiguration("stain_origin_topic")
    force_obs_xy_zero = LaunchConfiguration("force_obs_xy_zero")
    stain_canon_enable = LaunchConfiguration("stain_canon_enable")
    stain_canon_train_angle_deg = LaunchConfiguration("stain_canon_train_angle_deg")
    stain_canon_live_angle_deg = LaunchConfiguration("stain_canon_live_angle_deg")
    stain_canon_rotate_image = LaunchConfiguration("stain_canon_rotate_image")
    stain_canon_image_angle_sign = LaunchConfiguration("stain_canon_image_angle_sign")
    stain_canon_max_angle_deg = LaunchConfiguration("stain_canon_max_angle_deg")
    auto_stain_mask = LaunchConfiguration("auto_stain_mask")
    stain_mask_overlay_topic = LaunchConfiguration("stain_mask_overlay_topic")
    publish_stain_mask_overlay = LaunchConfiguration("publish_stain_mask_overlay")
    stain_mask_mode = LaunchConfiguration("stain_mask_mode")
    task_roi_center_x = LaunchConfiguration("task_roi_center_x")
    task_roi_y_end = LaunchConfiguration("task_roi_y_end")
    task_roi_half_width = LaunchConfiguration("task_roi_half_width")
    tcp_roi_reference_width = LaunchConfiguration("tcp_roi_reference_width")
    tcp_roi_reference_height = LaunchConfiguration("tcp_roi_reference_height")
    tcp_roi_center_x = LaunchConfiguration("tcp_roi_center_x")
    tcp_roi_center_y = LaunchConfiguration("tcp_roi_center_y")
    tcp_roi_area_fraction = LaunchConfiguration("tcp_roi_area_fraction")
    stain_dark_thresh = LaunchConfiguration("stain_dark_thresh")
    reflection_v_thresh = LaunchConfiguration("reflection_v_thresh")
    reflection_s_thresh = LaunchConfiguration("reflection_s_thresh")
    stain_min_area = LaunchConfiguration("stain_min_area")
    stain_morph_kernel = LaunchConfiguration("stain_morph_kernel")
    cmd_topic = LaunchConfiguration("cmd_topic")
    visualization_only = LaunchConfiguration("visualization_only")
    clean_flow_execution = LaunchConfiguration("clean_flow_execution")
    camera_preprocess_mode = LaunchConfiguration("camera_preprocess_mode")
    chunk_size = LaunchConfiguration("chunk_size")
    use_force_history = LaunchConfiguration("use_force_history")
    force_history_len = LaunchConfiguration("force_history_len")
    flow_infer_steps = LaunchConfiguration("flow_infer_steps")
    flow_deterministic_noise = LaunchConfiguration("flow_deterministic_noise")
    flow_noise_seed = LaunchConfiguration("flow_noise_seed")
    action_selection_mode = LaunchConfiguration("action_selection_mode")
    trajectory_hz = LaunchConfiguration("trajectory_hz")
    flow_local_anchor_enable = LaunchConfiguration("flow_local_anchor_enable")
    flow_replan_interval_steps = LaunchConfiguration("flow_replan_interval_steps")
    contact_z_descent_block_enable = LaunchConfiguration(
        "contact_z_descent_block_enable"
    )
    contact_z_descent_margin_mm = LaunchConfiguration("contact_z_descent_margin_mm")
    policy_z_offset_mm = LaunchConfiguration("policy_z_offset_mm")
    force_xy_cmd_enable = LaunchConfiguration("force_xy_cmd_enable")
    orientation_lock_enable = LaunchConfiguration("orientation_lock_enable")
    orientation_lock_wx = LaunchConfiguration("orientation_lock_wx")
    orientation_lock_wy = LaunchConfiguration("orientation_lock_wy")
    orientation_lock_wz = LaunchConfiguration("orientation_lock_wz")
    track_use_ptp9d_service = LaunchConfiguration("track_use_ptp9d_service")
    ptp9d_segment_points = LaunchConfiguration("ptp9d_segment_points")
    ptp9d_segment_stride = LaunchConfiguration("ptp9d_segment_stride")
    ptp9d_use_stream = LaunchConfiguration("ptp9d_use_stream")
    ptp9d_stream_topup_points = LaunchConfiguration("ptp9d_stream_topup_points")
    ptp9d_stream_min_lookahead_sec = LaunchConfiguration("ptp9d_stream_min_lookahead_sec")
    ptp9d_stream_smooth_window = LaunchConfiguration("ptp9d_stream_smooth_window")
    ptp9d_stream_stride = LaunchConfiguration("ptp9d_stream_stride")
    ptp9d_stream_force_min_interval_sec = LaunchConfiguration("ptp9d_stream_force_min_interval_sec")
    ptp9d_target_velocity_mm_s = LaunchConfiguration("ptp9d_target_velocity_mm_s")
    auto_move_to_demo_start = LaunchConfiguration("auto_move_to_demo_start")
    demo_start_move_sec = LaunchConfiguration("demo_start_move_sec")
    demo_start_hold_sec = LaunchConfiguration("demo_start_hold_sec")
    demo_start_max_align_dist_mm = LaunchConfiguration("demo_start_max_align_dist_mm")
    demo_start_max_xyz_speed_mm_s = LaunchConfiguration("demo_start_max_xyz_speed_mm_s")
    demo_start_max_rot_speed_rad_s = LaunchConfiguration("demo_start_max_rot_speed_rad_s")
    demo_start_position_tolerance_mm = LaunchConfiguration("demo_start_position_tolerance_mm")
    demo_start_rotation_tolerance_rad = LaunchConfiguration("demo_start_rotation_tolerance_rad")
    cmd_safety_enable = LaunchConfiguration("cmd_safety_enable")
    cmd_safety_max_xyz_from_current_mm = LaunchConfiguration("cmd_safety_max_xyz_from_current_mm")
    cmd_safety_max_xy_from_start_mm = LaunchConfiguration("cmd_safety_max_xy_from_start_mm")
    cmd_safety_max_z_down_from_start_mm = LaunchConfiguration("cmd_safety_max_z_down_from_start_mm")
    cmd_safety_max_z_up_from_start_mm = LaunchConfiguration("cmd_safety_max_z_up_from_start_mm")
    cmd_safety_latch_on_start_limit = LaunchConfiguration("cmd_safety_latch_on_start_limit")

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
    modality_importance_enable = LaunchConfiguration("modality_importance_enable")
    modality_importance_every_n_infer = LaunchConfiguration("modality_importance_every_n_infer")
    modality_importance_target = LaunchConfiguration("modality_importance_target")
    modality_importance_target_step = LaunchConfiguration("modality_importance_target_step")
    modality_importance_target_horizon = LaunchConfiguration("modality_importance_target_horizon")
    modality_importance_ema_alpha = LaunchConfiguration("modality_importance_ema_alpha")
    modality_importance_topic = LaunchConfiguration("modality_importance_topic")
    visualize_modality_importance = LaunchConfiguration("visualize_modality_importance")
    flow_vector_overlay_enable = LaunchConfiguration("flow_vector_overlay_enable")
    flow_vector_overlay_topic = LaunchConfiguration("flow_vector_overlay_topic")
    flow_vector_overlay_horizons = LaunchConfiguration("flow_vector_overlay_horizons")
    flow_vector_overlay_selected_horizon = LaunchConfiguration(
        "flow_vector_overlay_selected_horizon"
    )
    flow_vector_overlay_tcp_center_x = LaunchConfiguration(
        "flow_vector_overlay_tcp_center_x"
    )
    flow_vector_overlay_tcp_center_y = LaunchConfiguration(
        "flow_vector_overlay_tcp_center_y"
    )
    flow_vector_overlay_pixels_per_mm = LaunchConfiguration(
        "flow_vector_overlay_pixels_per_mm"
    )
    flow_vector_overlay_m_du_dx = LaunchConfiguration("flow_vector_overlay_m_du_dx")
    flow_vector_overlay_m_du_dy = LaunchConfiguration("flow_vector_overlay_m_du_dy")
    flow_vector_overlay_m_dv_dx = LaunchConfiguration("flow_vector_overlay_m_dv_dx")
    flow_vector_overlay_m_dv_dy = LaunchConfiguration("flow_vector_overlay_m_dv_dy")
    flow_diagnostic_only = LaunchConfiguration("flow_diagnostic_only")
    flow_step_service_enable = LaunchConfiguration("flow_step_service_enable")
    flow_step_service = LaunchConfiguration("flow_step_service")
    flow_step_max_xyz_mm = LaunchConfiguration("flow_step_max_xyz_mm")
    flow_step_max_rot_rad = LaunchConfiguration("flow_step_max_rot_rad")
    flow_step_stats_margin_mm = LaunchConfiguration("flow_step_stats_margin_mm")
    flow_step_block_down_on_contact = LaunchConfiguration(
        "flow_step_block_down_on_contact"
    )
    visualize_flow_vector = LaunchConfiguration("visualize_flow_vector")

    return LaunchDescription([
        DeclareLaunchArgument("ckpt_dir", default_value=""),
        DeclareLaunchArgument("act_root", default_value="~/nrs_imitation"),
        DeclareLaunchArgument("policy_class", default_value="FLOW"),  # FLOW | BSPLINE | ACT | DIFFUSION
        DeclareLaunchArgument("ckpt_auto_subdir", default_value="polishing/single_cam"),
        # Optional per-run CSV metrics log for offline FLOW-vs-BSPLINE comparison
        # (see scripts/compare_policy_runs.py).
        DeclareLaunchArgument("metrics_log_enable", default_value="false"),
        DeclareLaunchArgument("metrics_log_dir", default_value=""),
        DeclareLaunchArgument("metrics_run_tag", default_value=""),
        # Auto screen-recording of the run (ffmpeg x11grab -> ~/Videos/Screencasts).
        # Starts with this launch, stops (and finalizes the webm) on shutdown.
        DeclareLaunchArgument("overlay_record_enable", default_value="true"),
        # Polishing-removal visualization: records currentP/currentF for the
        # whole launch and, on shutdown, writes a Preston removal heatmap +
        # 3 companion plots for the traversed trajectory (ported from
        # ~/nrspath_ws polishing_removal). Start/end = launch start/end.
        DeclareLaunchArgument("removal_viz_enable", default_value="true"),
        DeclareLaunchArgument("removal_viz_open_on_exit", default_value="false"),
        # How long the heatmap is held at the end of the overlay webm, and
        # how much workpiece margin (mm) to show around the polished area.
        DeclareLaunchArgument("removal_heatmap_tail_sec", default_value="15.0"),
        DeclareLaunchArgument("removal_view_margin_mm", default_value="40.0"),
        DeclareLaunchArgument("pose_topic", default_value="/ur10skku/currentP"),
        DeclareLaunchArgument("force_topic", default_value="/ur10skku/currentF"),
        DeclareLaunchArgument("image_topic", default_value="/realsense/vr/color/image_raw"),
        DeclareLaunchArgument("use_stain_mask", default_value="false"),
        DeclareLaunchArgument("stain_mask_topic", default_value="/inference_single_cam/stain_mask"),
        # Latched stain-origin topic (stain_relative_frame/stain_origin_node).
        # Only consumed when the checkpoint is a stain-relative one.
        DeclareLaunchArgument(
            "stain_origin_topic", default_value="/stain_relative_frame/stain_origin"
        ),
        # auto | true | false -- override observation fx,fy zeroing (ablation).
        DeclareLaunchArgument("force_obs_xy_zero", default_value="auto"),
        # Inference-side rotation canonicalization (experimental) -- let a
        # single-direction policy follow a stain drawn at an arbitrary angle.
        DeclareLaunchArgument("stain_canon_enable", default_value="false"),
        DeclareLaunchArgument("stain_canon_train_angle_deg", default_value="132.0"),
        DeclareLaunchArgument("stain_canon_live_angle_deg", default_value="-1.0"),
        DeclareLaunchArgument("stain_canon_rotate_image", default_value="true"),
        DeclareLaunchArgument("stain_canon_image_angle_sign", default_value="1.0"),
        DeclareLaunchArgument("stain_canon_max_angle_deg", default_value="65.0"),
        DeclareLaunchArgument("auto_stain_mask", default_value="false"),
        DeclareLaunchArgument("stain_mask_overlay_topic", default_value="/inference_single_cam/stain_mask_overlay"),
        DeclareLaunchArgument("publish_stain_mask_overlay", default_value="true"),
        DeclareLaunchArgument("stain_mask_mode", default_value="rgb_threshold"),
        DeclareLaunchArgument("task_roi_center_x", default_value="253"),
        DeclareLaunchArgument("task_roi_y_end", default_value="110"),
        DeclareLaunchArgument("task_roi_half_width", default_value="12"),
        DeclareLaunchArgument("tcp_roi_reference_width", default_value="424"),
        DeclareLaunchArgument("tcp_roi_reference_height", default_value="240"),
        DeclareLaunchArgument("tcp_roi_center_x", default_value="253"),
        DeclareLaunchArgument("tcp_roi_center_y", default_value="120"),
        DeclareLaunchArgument("tcp_roi_area_fraction", default_value="0.10"),
        DeclareLaunchArgument("stain_dark_thresh", default_value="80"),
        DeclareLaunchArgument("reflection_v_thresh", default_value="235"),
        DeclareLaunchArgument("reflection_s_thresh", default_value="60"),
        DeclareLaunchArgument("stain_min_area", default_value="20"),
        DeclareLaunchArgument("stain_morph_kernel", default_value="3"),
        DeclareLaunchArgument("cmd_topic", default_value="/ur10skku/cmdMotion"),
        DeclareLaunchArgument("visualization_only", default_value="false"),
        DeclareLaunchArgument("clean_flow_execution", default_value="false"),
        DeclareLaunchArgument("camera_preprocess_mode", default_value="stabilize"),
        DeclareLaunchArgument("chunk_size", default_value="200"),
        DeclareLaunchArgument("use_force_history", default_value="true"),
        DeclareLaunchArgument("force_history_len", default_value="10"),
        DeclareLaunchArgument("flow_infer_steps", default_value="10"),
        DeclareLaunchArgument("flow_deterministic_noise", default_value="true"),
        DeclareLaunchArgument("flow_noise_seed", default_value="0"),
        DeclareLaunchArgument("action_selection_mode", default_value="trajectory_interp"),
        DeclareLaunchArgument("trajectory_hz", default_value="30.0"),
        DeclareLaunchArgument("flow_local_anchor_enable", default_value="false"),
        DeclareLaunchArgument("flow_replan_interval_steps", default_value="30"),
        DeclareLaunchArgument("contact_z_descent_block_enable", default_value="true"),
        DeclareLaunchArgument("contact_z_descent_margin_mm", default_value="0.2"),
        DeclareLaunchArgument("policy_z_offset_mm", default_value="0.0"),
        DeclareLaunchArgument("force_xy_cmd_enable", default_value="false"),
        DeclareLaunchArgument("orientation_lock_enable", default_value="false"),
        DeclareLaunchArgument("orientation_lock_wx", default_value="0.0"),
        DeclareLaunchArgument("orientation_lock_wy", default_value="0.0"),
        DeclareLaunchArgument("orientation_lock_wz", default_value="1.5707963268"),
        # true  = TRACK stage drives the robot via discrete PTP9D service
        #         calls (safe, default -- see _ptp9d_advance/_on_ptp9d_step_done).
        # false = TRACK stage streams continuous 9D commands directly onto
        #         cmd_topic at control_hz (legacy topic-publish path).
        DeclareLaunchArgument("track_use_ptp9d_service", default_value="true"),
        # Each PTP9D call carries this many consecutive lookahead waypoints
        # (spaced ptp9d_segment_stride raw samples apart) blended into one
        # continuous robot-side motion instead of stopping at every point --
        # more points/call means fewer calls and smoother motion, at the
        # cost of coarser per-call contact/safety re-evaluation.
        DeclareLaunchArgument("ptp9d_segment_points", default_value="15"),
        DeclareLaunchArgument("ptp9d_segment_stride", default_value="1"),
        # True: use the persistent PTP9D_STREAM queue (see
        # _ptp9d_stream_topup in inference_core.py) instead of the batched
        # call-and-wait segments above -- never stops between calls, only
        # when the queue actually runs dry.
        DeclareLaunchArgument("ptp9d_use_stream", default_value="false"),
        DeclareLaunchArgument("ptp9d_stream_topup_points", default_value="40"),
        DeclareLaunchArgument("ptp9d_stream_min_lookahead_sec", default_value="2.5"),
        # Smooths position/orientation (not force) before each streamed
        # segment is chosen -- without this the stream thread chases raw
        # per-sample prediction noise point by point (visibly oscillated in
        # the first live test). stride skips raw samples so fewer, less
        # noisy points need to be visited.
        DeclareLaunchArgument("ptp9d_stream_smooth_window", default_value="35"),
        DeclareLaunchArgument("ptp9d_stream_stride", default_value="2"),
        # Force is pushed via a separate immediate channel
        # (PTP9D_STREAM_SET_FORCE) instead of riding along with queued
        # waypoints -- a contact on/off transition always sends right away;
        # this only rate-limits routine updates during steady contact.
        DeclareLaunchArgument("ptp9d_stream_force_min_interval_sec", default_value="0.1"),
        DeclareLaunchArgument("ptp9d_target_velocity_mm_s", default_value="10.0"),
        DeclareLaunchArgument("auto_move_to_demo_start", default_value="true"),
        DeclareLaunchArgument("demo_start_move_sec", default_value="5.0"),
        DeclareLaunchArgument("demo_start_hold_sec", default_value="2.0"),
        # Disable only the old total-distance rejection. The speed-limited
        # alignment and current-pose command guard remain active.
        DeclareLaunchArgument("demo_start_max_align_dist_mm", default_value="0.0"),
        DeclareLaunchArgument("demo_start_max_xyz_speed_mm_s", default_value="50.0"),
        DeclareLaunchArgument("demo_start_max_rot_speed_rad_s", default_value="0.25"),
        DeclareLaunchArgument("demo_start_position_tolerance_mm", default_value="5.0"),
        DeclareLaunchArgument("demo_start_rotation_tolerance_rad", default_value="0.05"),

        # filtered58 demo envelope maxima were approximately XY=134.9 mm,
        # Z-down=79.3 mm, Z-up=86.1 mm. These defaults retain a small margin.
        DeclareLaunchArgument("cmd_safety_enable", default_value="true"),
        DeclareLaunchArgument("cmd_safety_max_xyz_from_current_mm", default_value="200.0"),
        DeclareLaunchArgument("cmd_safety_max_xy_from_start_mm", default_value="140.0"),
        DeclareLaunchArgument("cmd_safety_max_z_down_from_start_mm", default_value="85.0"),
        DeclareLaunchArgument("cmd_safety_max_z_up_from_start_mm", default_value="95.0"),
        DeclareLaunchArgument("cmd_safety_latch_on_start_limit", default_value="true"),

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
        DeclareLaunchArgument("modality_importance_enable", default_value="true"),
        DeclareLaunchArgument("modality_importance_every_n_infer", default_value="5"),
        DeclareLaunchArgument("modality_importance_target", default_value="action_norm"),
        DeclareLaunchArgument("modality_importance_target_step", default_value="0"),
        DeclareLaunchArgument("modality_importance_target_horizon", default_value="16"),
        DeclareLaunchArgument("modality_importance_ema_alpha", default_value="0.80"),
        DeclareLaunchArgument(
            "modality_importance_topic",
            default_value="/inference_single_cam/modality_importance",
        ),
        DeclareLaunchArgument("visualize_modality_importance", default_value="true"),
        DeclareLaunchArgument("flow_vector_overlay_enable", default_value="false"),
        DeclareLaunchArgument(
            "flow_vector_overlay_topic",
            default_value="/inference_single_cam/flow_vector_overlay",
        ),
        DeclareLaunchArgument(
            "flow_vector_overlay_horizons", default_value="1,5,15,30,60,127"
        ),
        DeclareLaunchArgument("flow_vector_overlay_selected_horizon", default_value="30"),
        DeclareLaunchArgument("flow_vector_overlay_tcp_center_x", default_value="253"),
        DeclareLaunchArgument("flow_vector_overlay_tcp_center_y", default_value="120"),
        DeclareLaunchArgument("flow_vector_overlay_pixels_per_mm", default_value="2.0"),
        # Eye-in-hand camera: 2x2 tool-mm -> image-px matrix. Defaults below
        # are the 2026-08-08 empirical calibration for the 250mm/45deg mount
        # (measured via background-patch tracking, see inference_core.py
        # _render_flow_vector_overlay_rgb docstring). Re-measure and update
        # these whenever the mount geometry changes.
        DeclareLaunchArgument("flow_vector_overlay_m_du_dx", default_value="0.0"),
        DeclareLaunchArgument("flow_vector_overlay_m_du_dy", default_value="0.65"),
        DeclareLaunchArgument("flow_vector_overlay_m_dv_dx", default_value="0.45"),
        DeclareLaunchArgument("flow_vector_overlay_m_dv_dy", default_value="0.0"),
        DeclareLaunchArgument("flow_diagnostic_only", default_value="false"),
        DeclareLaunchArgument("flow_step_service_enable", default_value="false"),
        DeclareLaunchArgument(
            "flow_step_service", default_value="/inference_single_cam/flow_step"
        ),
        DeclareLaunchArgument("flow_step_max_xyz_mm", default_value="0.5"),
        DeclareLaunchArgument("flow_step_max_rot_rad", default_value="0.001"),
        DeclareLaunchArgument("flow_step_stats_margin_mm", default_value="10.0"),
        DeclareLaunchArgument("flow_step_block_down_on_contact", default_value="true"),
        DeclareLaunchArgument("visualize_flow_vector", default_value="false"),

        Node(
            package="nrs_imitation",
            executable="overlay_video_recorder",
            name="overlay_video_recorder",
            output="screen",
            parameters=[{
                "enable": ParameterValue(overlay_record_enable, value_type=bool),
                "policy_class": policy_class,
                "run_tag": metrics_run_tag,
                "modality_importance_topic": modality_importance_topic,
                "flow_vector_overlay_topic": flow_vector_overlay_topic,
                "record_modality_importance": ParameterValue(modality_importance_enable, value_type=bool),
                "record_flow_vector_overlay": ParameterValue(flow_vector_overlay_enable, value_type=bool),
                # tack the polishing-removal heatmap onto the end of the webm
                "append_removal_heatmap": ParameterValue(removal_viz_enable, value_type=bool),
                "removal_heatmap_tail_sec": ParameterValue(removal_heatmap_tail_sec, value_type=float),
            }],
        ),

        Node(
            package="nrs_imitation",
            executable="polishing_removal_recorder",
            name="polishing_removal_recorder",
            output="screen",
            parameters=[{
                "enable": ParameterValue(removal_viz_enable, value_type=bool),
                "open_on_exit": ParameterValue(removal_viz_open_on_exit, value_type=bool),
                "position_topic": pose_topic,
                "force_topic": force_topic,
                "run_tag": metrics_run_tag,
                "view_margin_mm": ParameterValue(removal_view_margin_mm, value_type=float),
            }],
        ),

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
                "mask_mode": stain_mask_mode,
                "task_roi_center_x": ParameterValue(task_roi_center_x, value_type=int),
                "task_roi_y_end": ParameterValue(task_roi_y_end, value_type=int),
                "task_roi_half_width": ParameterValue(task_roi_half_width, value_type=int),
                "tcp_roi_reference_width": ParameterValue(tcp_roi_reference_width, value_type=int),
                "tcp_roi_reference_height": ParameterValue(tcp_roi_reference_height, value_type=int),
                "tcp_roi_center_x": ParameterValue(tcp_roi_center_x, value_type=int),
                "tcp_roi_center_y": ParameterValue(tcp_roi_center_y, value_type=int),
                "tcp_roi_area_fraction": ParameterValue(tcp_roi_area_fraction, value_type=float),
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
                "metrics_log_enable": ParameterValue(metrics_log_enable, value_type=bool),
                "metrics_log_dir": metrics_log_dir,
                "metrics_run_tag": metrics_run_tag,
                "pose_topic": pose_topic,
                "force_topic": force_topic,
                "image_topic": image_topic,
                "stain_origin_topic": stain_origin_topic,
                # ROS/YAML turns the bare token "true"/"false" into a Boolean;
                # force it back to str so the node's string param accepts it.
                "force_obs_xy_zero": ParameterValue(force_obs_xy_zero, value_type=str),
                "stain_canon_enable": ParameterValue(stain_canon_enable, value_type=bool),
                "stain_canon_train_angle_deg": ParameterValue(stain_canon_train_angle_deg, value_type=float),
                "stain_canon_live_angle_deg": ParameterValue(stain_canon_live_angle_deg, value_type=float),
                "stain_canon_rotate_image": ParameterValue(stain_canon_rotate_image, value_type=bool),
                "stain_canon_image_angle_sign": ParameterValue(stain_canon_image_angle_sign, value_type=float),
                "stain_canon_max_angle_deg": ParameterValue(stain_canon_max_angle_deg, value_type=float),
                "visualization_only": ParameterValue(
                    visualization_only, value_type=bool
                ),
                "clean_flow_execution": ParameterValue(
                    clean_flow_execution, value_type=bool
                ),
                # ROS/YAML interprets the unquoted token "off" as Boolean False.
                # Force a string so camera_preprocess_mode:=off reaches the node as intended.
                "camera_preprocess_mode": ParameterValue(camera_preprocess_mode, value_type=str),
                "chunk_size": ParameterValue(chunk_size, value_type=int),
                "use_force_history": ParameterValue(use_force_history, value_type=bool),
                "force_history_len": ParameterValue(force_history_len, value_type=int),
                "flow_infer_steps": ParameterValue(flow_infer_steps, value_type=int),
                "flow_deterministic_noise": ParameterValue(flow_deterministic_noise, value_type=bool),
                "flow_noise_seed": ParameterValue(flow_noise_seed, value_type=int),
                "action_selection_mode": action_selection_mode,
                "trajectory_hz": ParameterValue(trajectory_hz, value_type=float),
                "flow_local_anchor_enable": ParameterValue(
                    flow_local_anchor_enable, value_type=bool
                ),
                "flow_replan_interval_steps": ParameterValue(
                    flow_replan_interval_steps, value_type=int
                ),
                "contact_z_descent_block_enable": ParameterValue(
                    contact_z_descent_block_enable, value_type=bool
                ),
                "contact_z_descent_margin_mm": ParameterValue(
                    contact_z_descent_margin_mm, value_type=float
                ),
                "policy_z_offset_mm": ParameterValue(policy_z_offset_mm, value_type=float),
                "force_xy_cmd_enable": ParameterValue(force_xy_cmd_enable, value_type=bool),
                "orientation_lock_enable": ParameterValue(
                    orientation_lock_enable, value_type=bool
                ),
                "orientation_lock_wx": ParameterValue(
                    orientation_lock_wx, value_type=float
                ),
                "orientation_lock_wy": ParameterValue(
                    orientation_lock_wy, value_type=float
                ),
                "orientation_lock_wz": ParameterValue(
                    orientation_lock_wz, value_type=float
                ),
                "track_use_ptp9d_service": ParameterValue(
                    track_use_ptp9d_service, value_type=bool
                ),
                "ptp9d_segment_points": ParameterValue(
                    ptp9d_segment_points, value_type=int
                ),
                "ptp9d_segment_stride": ParameterValue(
                    ptp9d_segment_stride, value_type=int
                ),
                "ptp9d_use_stream": ParameterValue(
                    ptp9d_use_stream, value_type=bool
                ),
                "ptp9d_stream_topup_points": ParameterValue(
                    ptp9d_stream_topup_points, value_type=int
                ),
                "ptp9d_stream_min_lookahead_sec": ParameterValue(
                    ptp9d_stream_min_lookahead_sec, value_type=float
                ),
                "ptp9d_stream_smooth_window": ParameterValue(
                    ptp9d_stream_smooth_window, value_type=int
                ),
                "ptp9d_stream_stride": ParameterValue(
                    ptp9d_stream_stride, value_type=int
                ),
                "ptp9d_stream_force_min_interval_sec": ParameterValue(
                    ptp9d_stream_force_min_interval_sec, value_type=float
                ),
                "ptp9d_target_velocity_mm_s": ParameterValue(
                    ptp9d_target_velocity_mm_s, value_type=float
                ),
                "auto_move_to_demo_start": ParameterValue(
                    auto_move_to_demo_start, value_type=bool
                ),
                "demo_start_move_sec": ParameterValue(demo_start_move_sec, value_type=float),
                "demo_start_hold_sec": ParameterValue(demo_start_hold_sec, value_type=float),
                "demo_start_max_align_dist_mm": ParameterValue(
                    demo_start_max_align_dist_mm, value_type=float
                ),
                "demo_start_max_xyz_speed_mm_s": ParameterValue(
                    demo_start_max_xyz_speed_mm_s, value_type=float
                ),
                "demo_start_max_rot_speed_rad_s": ParameterValue(
                    demo_start_max_rot_speed_rad_s, value_type=float
                ),
                "demo_start_position_tolerance_mm": ParameterValue(
                    demo_start_position_tolerance_mm, value_type=float
                ),
                "demo_start_rotation_tolerance_rad": ParameterValue(
                    demo_start_rotation_tolerance_rad, value_type=float
                ),
                "cmd_safety_enable": ParameterValue(cmd_safety_enable, value_type=bool),
                "cmd_safety_max_xyz_from_current_mm": ParameterValue(
                    cmd_safety_max_xyz_from_current_mm, value_type=float
                ),
                "cmd_safety_max_xy_from_start_mm": ParameterValue(
                    cmd_safety_max_xy_from_start_mm, value_type=float
                ),
                "cmd_safety_max_z_down_from_start_mm": ParameterValue(
                    cmd_safety_max_z_down_from_start_mm, value_type=float
                ),
                "cmd_safety_max_z_up_from_start_mm": ParameterValue(
                    cmd_safety_max_z_up_from_start_mm, value_type=float
                ),
                "cmd_safety_latch_on_start_limit": ParameterValue(
                    cmd_safety_latch_on_start_limit, value_type=bool
                ),
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
                "modality_importance_enable": ParameterValue(
                    modality_importance_enable, value_type=bool
                ),
                "modality_importance_every_n_infer": ParameterValue(
                    modality_importance_every_n_infer, value_type=int
                ),
                "modality_importance_target": modality_importance_target,
                "modality_importance_target_step": ParameterValue(
                    modality_importance_target_step, value_type=int
                ),
                "modality_importance_target_horizon": ParameterValue(
                    modality_importance_target_horizon, value_type=int
                ),
                "modality_importance_ema_alpha": ParameterValue(
                    modality_importance_ema_alpha, value_type=float
                ),
                "modality_importance_topic": modality_importance_topic,
                "flow_vector_overlay_enable": ParameterValue(
                    flow_vector_overlay_enable, value_type=bool
                ),
                "flow_vector_overlay_topic": flow_vector_overlay_topic,
                "flow_vector_overlay_horizons": flow_vector_overlay_horizons,
                "flow_vector_overlay_selected_horizon": ParameterValue(
                    flow_vector_overlay_selected_horizon, value_type=int
                ),
                "flow_vector_overlay_tcp_center_x": ParameterValue(
                    flow_vector_overlay_tcp_center_x, value_type=int
                ),
                "flow_vector_overlay_tcp_center_y": ParameterValue(
                    flow_vector_overlay_tcp_center_y, value_type=int
                ),
                "flow_vector_overlay_pixels_per_mm": ParameterValue(
                    flow_vector_overlay_pixels_per_mm, value_type=float
                ),
                "flow_vector_overlay_m_du_dx": ParameterValue(
                    flow_vector_overlay_m_du_dx, value_type=float
                ),
                "flow_vector_overlay_m_du_dy": ParameterValue(
                    flow_vector_overlay_m_du_dy, value_type=float
                ),
                "flow_vector_overlay_m_dv_dx": ParameterValue(
                    flow_vector_overlay_m_dv_dx, value_type=float
                ),
                "flow_vector_overlay_m_dv_dy": ParameterValue(
                    flow_vector_overlay_m_dv_dy, value_type=float
                ),
                "flow_diagnostic_only": ParameterValue(
                    flow_diagnostic_only, value_type=bool
                ),
                "flow_step_service_enable": ParameterValue(
                    flow_step_service_enable, value_type=bool
                ),
                "flow_step_service": flow_step_service,
                "flow_step_max_xyz_mm": ParameterValue(
                    flow_step_max_xyz_mm, value_type=float
                ),
                "flow_step_max_rot_rad": ParameterValue(
                    flow_step_max_rot_rad, value_type=float
                ),
                "flow_step_stats_margin_mm": ParameterValue(
                    flow_step_stats_margin_mm, value_type=float
                ),
                "flow_step_block_down_on_contact": ParameterValue(
                    flow_step_block_down_on_contact, value_type=bool
                ),
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

        Node(
            package="rqt_image_view",
            executable="rqt_image_view",
            name="modality_importance_viewer",
            output="screen",
            arguments=[modality_importance_topic],
            condition=IfCondition(visualize_modality_importance),
        ),

        Node(
            package="rqt_image_view",
            executable="rqt_image_view",
            name="flow_vector_overlay_viewer",
            output="screen",
            arguments=[flow_vector_overlay_topic],
            condition=IfCondition(visualize_flow_vector),
        ),
    ])
