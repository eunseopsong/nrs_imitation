#!/usr/bin/env python3
"""Minimal operational FLOW inference with the two requested diagnostics."""

import os
import pickle

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.substitutions import FindPackageShare


class _NumpyCompatUnpickler(pickle.Unpickler):
    """`ros2 launch` runs under the system python (ROS numpy 1.x); the
    dataset_stats.pkl is pickled under the conda env (numpy 2.x). Remap the
    moved numpy module so the flag is still readable here. Mirrors
    inference_core._NumpyCompatUnpickler."""

    def find_class(self, module, name):
        if module == "numpy._core":
            module = "numpy.core"
        elif str(module).startswith("numpy._core."):
            module = "numpy.core." + str(module)[len("numpy._core."):]
        return super().find_class(module, name)


def _read_relative_flags(ckpt_dir: str):
    """(use_relative_position, stain_origin_report) from a checkpoint's stats,
    falling back to the converted dataset's stats named in `dataset_dir`."""
    def _load(p, warn):
        if not os.path.exists(p):
            if warn:
                print(f"[inference_clean] checkpoint stats not found: {p}")
            return None
        try:
            with open(p, "rb") as f:
                return _NumpyCompatUnpickler(f).load()
        except Exception as exc:  # noqa: BLE001
            print(f"[inference_clean] could not read {p}: {exc!r}")
            return None

    st = _load(os.path.join(ckpt_dir, "dataset_stats.pkl"), warn=True) or {}
    src = st
    if "use_relative_position" not in st:
        # Pre-carry-forward checkpoint: try the converted dataset's own stats.
        # Missing is normal for an absolute-frame checkpoint -> don't warn.
        ds_dir = str(st.get("dataset_dir", "") or "")
        for cand in (ds_dir, os.path.join("/home/eunseop/nrs_imitation", ds_dir)):
            if not ds_dir:
                break
            ds = _load(os.path.join(cand, "dataset_stats.pkl"), warn=False)
            if ds and "use_relative_position" in ds:
                src = ds
                break
    return bool(src.get("use_relative_position", False)), str(src.get("stain_origin_report", "") or "")


def _maybe_stain_origin(context, *_a, **_kw):
    autostart = LaunchConfiguration("stain_origin_autostart").perform(context).strip().lower()
    if autostart in ("false", "0", "no", "off"):
        return []
    ckpt_dir = LaunchConfiguration("ckpt_dir").perform(context)
    use_relative, report = _read_relative_flags(ckpt_dir)
    if autostart in ("auto", "") and not use_relative:
        return []                       # absolute-frame checkpoint -> nothing to launch
    if not use_relative:
        print("[inference_clean] stain_origin_autostart forced on for a checkpoint whose "
              "stats do not say use_relative_position=True -- launching it anyway.")

    detect_params = LaunchConfiguration("stain_origin_detect_params").perform(context) or report
    # stain_origin_node runs from its own cwd -- a relative report path (as
    # stored by dataset_relativize) would not resolve there. Anchor it to the
    # project root and drop it if it still isn't a real file.
    if detect_params and not os.path.isabs(detect_params):
        cand = os.path.join("/home/eunseop/nrs_imitation", detect_params)
        detect_params = cand if os.path.exists(cand) else detect_params
    if detect_params and not os.path.exists(detect_params):
        print(f"[inference_clean] detect_params not found ({detect_params}) -- "
              "stain_origin_node will fall back to config defaults")
        detect_params = ""
    stain_launch = PathJoinSubstitution(
        [FindPackageShare("stain_relative_frame"), "launch", "stain_origin_online.launch.py"]
    )
    print(f"[inference_clean] stain-relative checkpoint -> auto-launching stain_origin_node "
          f"(detect_params={detect_params or '(config defaults)'})")
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(stain_launch),
            launch_arguments={
                "image_topic": LaunchConfiguration("image_topic"),
                "pose_topic": LaunchConfiguration("pose_topic"),
                "frames": LaunchConfiguration("stain_origin_frames"),
                "method": "auto",
                "detect_params": detect_params,
            }.items(),
        )
    ]


def generate_launch_description():
    ckpt_dir = LaunchConfiguration("ckpt_dir")
    act_root = LaunchConfiguration("act_root")
    policy_class = LaunchConfiguration("policy_class")
    ckpt_auto_subdir = LaunchConfiguration("ckpt_auto_subdir")
    pose_topic = LaunchConfiguration("pose_topic")
    force_topic = LaunchConfiguration("force_topic")
    image_topic = LaunchConfiguration("image_topic")
    stain_origin_topic = LaunchConfiguration("stain_origin_topic")
    force_obs_xy_zero = LaunchConfiguration("force_obs_xy_zero")
    stain_canon_enable = LaunchConfiguration("stain_canon_enable")
    stain_canon_train_angle_deg = LaunchConfiguration("stain_canon_train_angle_deg")
    stain_canon_live_angle_deg = LaunchConfiguration("stain_canon_live_angle_deg")
    stain_canon_rotate_image = LaunchConfiguration("stain_canon_rotate_image")
    stain_canon_image_angle_sign = LaunchConfiguration("stain_canon_image_angle_sign")
    stain_canon_max_angle_deg = LaunchConfiguration("stain_canon_max_angle_deg")
    modality_every_n = LaunchConfiguration("modality_every_n")
    vector_horizon = LaunchConfiguration("vector_horizon")
    metrics_log_enable = LaunchConfiguration("metrics_log_enable")
    metrics_log_dir = LaunchConfiguration("metrics_log_dir")
    metrics_run_tag = LaunchConfiguration("metrics_run_tag")
    policy_z_offset_mm = LaunchConfiguration("policy_z_offset_mm")
    overlay_record_enable = LaunchConfiguration("overlay_record_enable")
    removal_viz_enable = LaunchConfiguration("removal_viz_enable")
    removal_viz_open_on_exit = LaunchConfiguration("removal_viz_open_on_exit")
    removal_heatmap_tail_sec = LaunchConfiguration("removal_heatmap_tail_sec")
    removal_view_margin_mm = LaunchConfiguration("removal_view_margin_mm")
    gradcam_enable = LaunchConfiguration("gradcam_enable")
    use_stain_mask = LaunchConfiguration("use_stain_mask")
    ptp9d_segment_points = LaunchConfiguration("ptp9d_segment_points")
    ptp9d_segment_stride = LaunchConfiguration("ptp9d_segment_stride")
    ptp9d_target_velocity_mm_s = LaunchConfiguration("ptp9d_target_velocity_mm_s")
    inference_mode = LaunchConfiguration("inference_mode")
    track_use_ptp9d_service = PythonExpression(
        ["'true' if '", inference_mode, "' != 'topic_publish' else 'false'"]
    )
    ptp9d_use_stream = PythonExpression(
        ["'true' if '", inference_mode, "' == 'service_stream' else 'false'"]
    )

    base_launch = PathJoinSubstitution(
        [FindPackageShare("nrs_imitation"), "launch", "inference_gradcam_single_cam.launch.py"]
    )
    return LaunchDescription(
        [
            # NOTE: ckpt_dir defaults to a pinned FLOW checkpoint. Switching
            # policy_class to BSPLINE requires passing a matching BSPLINE
            # ckpt_dir too (or "" to auto-select the latest one) -- inference_core
            # raises a clear error instead of silently loading mismatched weights.
            DeclareLaunchArgument(
                "ckpt_dir",
                default_value=(
                    "/home/eunseop/nrs_imitation/checkpoints/flow/polishing/"
                    "single_cam/20260802_1549"
                ),
            ),
            DeclareLaunchArgument("act_root", default_value="/home/eunseop/nrs_imitation"),
            DeclareLaunchArgument("policy_class", default_value="FLOW"),  # FLOW | BSPLINE
            DeclareLaunchArgument("ckpt_auto_subdir", default_value="polishing/single_cam"),
            DeclareLaunchArgument("pose_topic", default_value="/ur10skku/currentP"),
            DeclareLaunchArgument("force_topic", default_value="/ur10skku/currentF"),
            DeclareLaunchArgument(
                "image_topic", default_value="/realsense/vr/color/image_raw"
            ),
            # Stain-relative checkpoints only: latched origin from
            # stain_relative_frame/stain_origin_node. Harmless for
            # absolute-frame checkpoints (never subscribed).
            DeclareLaunchArgument(
                "stain_origin_topic", default_value="/stain_relative_frame/stain_origin"
            ),
            # auto  -> launch stain_origin_node iff the checkpoint is stain-relative
            # true  -> always launch it   |   false -> never (run it yourself)
            # Arm must already be at the home/viewing pose when this launches.
            DeclareLaunchArgument("stain_origin_autostart", default_value="auto"),
            DeclareLaunchArgument("stain_origin_frames", default_value="10"),
            # step-[2] origin report whose detect_params built this checkpoint's
            # training origins. Empty -> taken from the checkpoint stats'
            # stain_origin_report, else config defaults.
            DeclareLaunchArgument("stain_origin_detect_params", default_value=""),
            # Ablation: auto (follow checkpoint) | true | false. Set true to
            # zero obs fx,fy for an absolute-frame checkpoint trained with them.
            DeclareLaunchArgument("force_obs_xy_zero", default_value="auto"),
            # Rotation canonicalization (experimental): follow a stain drawn at
            # an arbitrary angle with a single-direction policy, no retraining.
            DeclareLaunchArgument("stain_canon_enable", default_value="false"),
            DeclareLaunchArgument("stain_canon_train_angle_deg", default_value="132.0"),
            DeclareLaunchArgument("stain_canon_live_angle_deg", default_value="-1.0"),
            DeclareLaunchArgument("stain_canon_rotate_image", default_value="true"),
            DeclareLaunchArgument("stain_canon_image_angle_sign", default_value="1.0"),
            DeclareLaunchArgument("stain_canon_max_angle_deg", default_value="65.0"),
            DeclareLaunchArgument("modality_every_n", default_value="1"),
            DeclareLaunchArgument("vector_horizon", default_value="30"),
            # Optional per-run CSV metrics log for offline FLOW-vs-BSPLINE
            # comparison (see scripts/compare_policy_runs.py).
            DeclareLaunchArgument("metrics_log_enable", default_value="false"),
            DeclareLaunchArgument("metrics_log_dir", default_value=""),
            DeclareLaunchArgument("metrics_run_tag", default_value=""),
            DeclareLaunchArgument("policy_z_offset_mm", default_value="0.0"),
            DeclareLaunchArgument("overlay_record_enable", default_value="true"),
            # Polishing-removal visualization: heatmap of the traversed
            # trajectory's material removal, written on shutdown. Start/end
            # tied to this launch's lifecycle.
            DeclareLaunchArgument("removal_viz_enable", default_value="true"),
            DeclareLaunchArgument("removal_viz_open_on_exit", default_value="false"),
            DeclareLaunchArgument("removal_heatmap_tail_sec", default_value="15.0"),
            DeclareLaunchArgument("removal_view_margin_mm", default_value="40.0"),
            # Feeds the flow-vector-overlay panel's heatmap (see
            # inference_core.py _render_flow_vector_overlay_rgb). Adds one
            # extra backward pass per replan tick -- pass :=false to skip it.
            DeclareLaunchArgument("gradcam_enable", default_value="true"),
            # Default "true" matches the pinned legacy FLOW ckpt_dir default
            # above. Newer dinov3/use_tcp_roi checkpoints (both FLOW and
            # BSPLINE) were trained with use_stain_mask=False -- pass
            # use_stain_mask:=false for those or inference_core refuses to
            # start (use_stain_mask mismatch: checkpoint vs inference_arg).
            DeclareLaunchArgument("use_stain_mask", default_value="true"),
            # service_call (default) = TRACK stage drives the robot via
            #   discrete, batched PTP9D service calls -- each call blends
            #   ptp9d_segment_points waypoints smoothly, but still stops
            #   briefly at every call boundary (see _ptp9d_advance in
            #   inference_core.py).
            # service_stream = TRACK stage keeps a persistent PTP9D queue
            #   topped up (see _ptp9d_stream_topup); the robot never stops
            #   between calls, only if the queue actually runs dry. This is
            #   the only mode with true call-to-call continuity.
            # topic_publish = legacy continuous 9D command streaming
            #   directly onto cmd_topic at control_hz.
            DeclareLaunchArgument(
                "inference_mode",
                default_value="service_call",
                choices=["service_call", "service_stream", "topic_publish"],
            ),
            # Each PTP9D call now carries ptp9d_segment_points consecutive
            # lookahead waypoints (ptp9d_segment_stride raw samples apart),
            # blended into one continuous robot-side motion instead of
            # stopping fully at every point. Raise segment_points for
            # smoother/less "stop-start" motion at the cost of coarser
            # per-call contact/safety re-evaluation. Only used when
            # inference_mode=service_call.
            DeclareLaunchArgument("ptp9d_segment_points", default_value="15"),
            DeclareLaunchArgument("ptp9d_segment_stride", default_value="1"),
            # Only used when inference_mode=service_stream.
            DeclareLaunchArgument("ptp9d_stream_topup_points", default_value="40"),
            DeclareLaunchArgument("ptp9d_stream_min_lookahead_sec", default_value="2.5"),
            # Smooths position/orientation before each streamed segment is
            # chosen and skips raw samples -- without this the stream
            # thread chases raw per-sample prediction noise point by point.
            DeclareLaunchArgument("ptp9d_stream_smooth_window", default_value="35"),
            DeclareLaunchArgument("ptp9d_stream_stride", default_value="2"),
            # Force is pushed via a separate immediate channel instead of
            # riding along with queued waypoints -- a contact on/off
            # transition always sends right away; this only rate-limits
            # routine updates during steady contact.
            DeclareLaunchArgument("ptp9d_stream_force_min_interval_sec", default_value="0.1"),
            DeclareLaunchArgument("ptp9d_target_velocity_mm_s", default_value="10.0"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(base_launch),
                launch_arguments={
                    "ckpt_dir": ckpt_dir,
                    "act_root": act_root,
                    "policy_class": policy_class,
                    "ckpt_auto_subdir": ckpt_auto_subdir,
                    "metrics_log_enable": metrics_log_enable,
                    "metrics_log_dir": metrics_log_dir,
                    "metrics_run_tag": metrics_run_tag,
                    "policy_z_offset_mm": policy_z_offset_mm,
                    "overlay_record_enable": overlay_record_enable,
                    "removal_viz_enable": removal_viz_enable,
                    "removal_viz_open_on_exit": removal_viz_open_on_exit,
                    "removal_heatmap_tail_sec": removal_heatmap_tail_sec,
                    "removal_view_margin_mm": removal_view_margin_mm,
                    "pose_topic": pose_topic,
                    "force_topic": force_topic,
                    "image_topic": image_topic,
                    "stain_origin_topic": stain_origin_topic,
                    "force_obs_xy_zero": force_obs_xy_zero,
                    "stain_canon_enable": stain_canon_enable,
                    "stain_canon_train_angle_deg": stain_canon_train_angle_deg,
                    "stain_canon_live_angle_deg": stain_canon_live_angle_deg,
                    "stain_canon_rotate_image": stain_canon_rotate_image,
                    "stain_canon_image_angle_sign": stain_canon_image_angle_sign,
                    "stain_canon_max_angle_deg": stain_canon_max_angle_deg,
                    # Operational mode. Legacy ACT-era recovery heuristics are
                    # bypassed, but command smoothing and envelope safety stay on.
                    "visualization_only": "false",
                    "clean_flow_execution": "true",
                    "flow_diagnostic_only": "false",
                    "flow_step_service_enable": "false",
                    "auto_move_to_demo_start": "true",
                    "orientation_lock_enable": "false",
                    "contact_z_descent_block_enable": "true",
                    "force_xy_cmd_enable": "false",
                    "cmd_safety_enable": "true",
                    # Match the checkpoint's observation construction.
                    "use_stain_mask": use_stain_mask,
                    "track_use_ptp9d_service": track_use_ptp9d_service,
                    "ptp9d_use_stream": ptp9d_use_stream,
                    "ptp9d_segment_points": ptp9d_segment_points,
                    "ptp9d_segment_stride": ptp9d_segment_stride,
                    "ptp9d_stream_topup_points": LaunchConfiguration("ptp9d_stream_topup_points"),
                    "ptp9d_stream_min_lookahead_sec": LaunchConfiguration("ptp9d_stream_min_lookahead_sec"),
                    "ptp9d_stream_smooth_window": LaunchConfiguration("ptp9d_stream_smooth_window"),
                    "ptp9d_stream_stride": LaunchConfiguration("ptp9d_stream_stride"),
                    "ptp9d_stream_force_min_interval_sec": LaunchConfiguration("ptp9d_stream_force_min_interval_sec"),
                    "ptp9d_target_velocity_mm_s": ptp9d_target_velocity_mm_s,
                    "auto_stain_mask": "true",
                    "stain_mask_mode": "tcp_roi",
                    "tcp_roi_reference_width": "424",
                    "tcp_roi_reference_height": "240",
                    "tcp_roi_center_x": "253",
                    "tcp_roi_center_y": "120",
                    # Must match the checkpoint's training-time tcp_roi_area_fraction
                    # (inference_core.py overrides the model's own value from
                    # dataset_stats.pkl, but stain_mask_publisher only sees this
                    # launch arg -- keep them in sync or the overlay box drifts
                    # from what the model actually attends to).
                    "tcp_roi_area_fraction": "0.25",
                    "camera_preprocess_mode": "stabilize",
                    "chunk_size": "128",
                    "use_force_history": "true",
                    "force_history_len": "30",
                    "flow_infer_steps": "10",
                    "flow_deterministic_noise": "true",
                    "flow_noise_seed": "0",
                    # Replay the learned FLOW trajectory directly. Let each
                    # absolute-referenced plan run close to its full
                    # chunk_size=128 horizon (~4.3s @ 30Hz) before replanning,
                    # instead of cutting it off after ~1s. local_anchor is off:
                    # anchoring plans to the live pose removed the self-
                    # correction against the model's absolute target, so a
                    # small per-step z bias accumulated into unbounded ascent
                    # (20260811 FLOW anchorfix runs, ESTOP both times).
                    "action_selection_mode": "trajectory_interp",
                    "trajectory_hz": "30.0",
                    "flow_local_anchor_enable": "false",
                    "flow_replan_interval_steps": "120",
                    # Only the requested diagnostic windows are shown.
                    "gradcam_enable": gradcam_enable,
                    "visualize": "false",
                    "modality_importance_enable": "true",
                    "modality_importance_every_n_infer": modality_every_n,
                    "modality_importance_target": "action_norm",
                    "modality_importance_target_step": "0",
                    "modality_importance_target_horizon": "16",
                    "visualize_modality_importance": "true",
                    "flow_vector_overlay_enable": "true",
                    "flow_vector_overlay_horizons": "1,5,15,30,60,127",
                    "flow_vector_overlay_selected_horizon": vector_horizon,
                    "flow_vector_overlay_tcp_center_x": "253",
                    "flow_vector_overlay_tcp_center_y": "120",
                    "flow_vector_overlay_pixels_per_mm": "2.0",
                    "visualize_flow_vector": "true",
                }.items(),
            ),
            OpaqueFunction(function=_maybe_stain_origin),
        ]
    )
