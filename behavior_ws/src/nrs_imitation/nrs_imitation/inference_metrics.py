"""Read-only ROS adapter for execution_metrics. Never supplies policy/control input."""
import json
import math
from pathlib import Path
import threading
import time
import numpy as np

from .execution_metrics import ExecutionRecorder, stamp, pose_fields, numeric_valid

PARAMETERS = {
    "metrics_specimen_id": "default", "metrics_repeat_id": "",
    "metrics_rpm_setpoint": "", "metrics_rpm_assumed_constant": False,
    "metrics_context_file": "", "metrics_queue_size": 8192,
    "metrics_snapshot_enable": True,
    "metrics_extra_telemetry_enable": False,
    "metrics_sample_hz": 20.0,  # pose/force/legacy only; 0 keeps every sample
}


class InferenceMetrics:
    def __init__(self, node):
        self.node = node
        self.last_pose_stamp = self.last_force_stamp = None
        self.last_image = None
        self.initial_image_saved = self.mask_saved = self.reference_saved = False
        self.last_stage = None
        self.inference_id = 0
        self.command_id = 0
        self.first_sent = False
        self.plan_ids = {}
        self.observer = self.executor = self.observer_thread = None
        self.snapshot_enabled = node.get_parameter("metrics_snapshot_enable").value
        params = {name: parameter.value for name, parameter in node.get_parameters_by_prefix("").items()}
        self.extra_telemetry_enabled = bool(params["metrics_extra_telemetry_enable"])
        sample_hz = float(params["metrics_sample_hz"])
        if not math.isfinite(sample_hz) or sample_hz < 0:
            raise ValueError("metrics_sample_hz must be finite and nonnegative")
        self.sample_period_ns = int(1e9 / sample_hz) if sample_hz > 0 else 0
        self._sample_last_ns = {}
        self._sample_skipped = {}
        # JSON is logging-only; operator assertions are preserved, not used as transforms.
        context_path = params["metrics_context_file"]
        context = {}
        if context_path:
            with Path(context_path).expanduser().open() as f:
                context = json.load(f)
        rpm_text = str(params["metrics_rpm_setpoint"]).strip()
        rpm = float(rpm_text) if rpm_text else None
        if rpm is not None and (not math.isfinite(rpm) or rpm < 0):
            raise ValueError("metrics_rpm_setpoint must be finite and nonnegative, or empty")
        root = Path(node.act_root)
        dev = Path.home() / "dev_ws/src/y2_ur10skku_control"
        artifacts = [Path(node.ckpt_dir) / "dataset_stats.pkl", Path(__file__),
            Path(__file__).with_name("execution_metrics.py"), Path(__file__).with_name("inference_core.py"),
            root / "behavior_ws/src/nrs_imitation/launch/inference_clean_single_cam.launch.py",
            root / "behavior_ws/src/nrs_imitation/launch/inference_gradcam_single_cam.launch.py",
            root / "source/models/flow_core.py", root / "source/data/dataset.py",
            root / "checkpoints/stain_relative_frame/homography.json",
            root / "checkpoints/stain_relative_frame/stain_origin_90deg_single.json",
            root / "behavior_ws/src/stain_relative_frame/config/stain_relative_frame.yaml",
            dev / "Y2RobMotion/config/setup_parameters.yaml",
            dev / "Y2RobMotion/src/robot_motion.cpp", dev / "Y2RobMotion/src/force_control.cpp",
            dev / "Y2FT_AQ/src/FTGetMain.cpp", dev / "Y2FT_AQ/config/spindle_gravity.yaml",
            dev / "Y2Matrix/src/RotationTransform.cpp"]
        if context_path:
            artifacts.append(Path(context_path).expanduser())
        ckpt = Path(node.ckpt_dir) / "policy_best.ckpt"
        ckpt_stat = ckpt.stat()
        resolved_names = ["use_force_observation", "use_force_history", "force_history_len", "chunk_size",
            "action_dim", "flow_infer_steps", "flow_deterministic_noise", "flow_noise_seed",
            "obs_force_xy_zeroed", "rel_use_relative", "rel_transform_version", "denorm_action_enabled",
            "normalize_qpos_enabled", "use_stain_mask", "stain_canon_enable", "force_xy_cmd_enable",
            "device", "camera_names", "action_type", "policy_class"]
        metadata = dict(code_root=str(root), task="polishing", run_tag=node.metrics_run_tag,
            specimen_id=params["metrics_specimen_id"] or None, repeat_id=params["metrics_repeat_id"] or None,
            force_observation="ON" if node.use_force_observation else "OFF",
            force_observation_method="measured" if node.use_force_observation else "constant_zero_after_channel_normalization",
            runtime_parameters=params, resolved_runtime={k: getattr(node, k, None) for k in resolved_names},
            checkpoint=dict(path=str(ckpt), bytes=ckpt_stat.st_size, mtime_ns=ckpt_stat.st_mtime_ns,
                            sha256=None, identity_note="path+size+mtime; no large weight read in logging startup"),
            checkpoint_config=getattr(node, "_metrics_checkpoint_config", None),
            normalizer=dict(path=str(Path(node.ckpt_dir) / "dataset_stats.pkl"), hash_in="artifacts"),
            artifact_paths=[str(p) for p in artifacts], operator_context=context,
            clocks=dict(receipt_ros="node.get_clock().now()", use_sim_time=params.get("use_sim_time", False),
                        receipt_monotonic="time.monotonic_ns on this host only", source="publisher ROS (sync unverified)",
                        source_age_computed=False, no_header="receipt timestamps only; acquisition time unknown"),
            tool_motion_type="rotary", rpm_setpoint=rpm,
            rpm_setpoint_source="manual:metrics_rpm_setpoint" if rpm is not None else None,
            rpm_measurement_available=False, measured_rpm=None,
            rpm_assumed_constant=bool(params["metrics_rpm_assumed_constant"]) if rpm is not None else False,
            rotation_state=None, tool_outer_diameter_mm=None, effective_contact_radius_mm=None,
            effective_contact_area_mm2=None, preston_coefficient=None,
            surface_normal=None, compression_sign=None, contact_point_offset=None,
            calibration_status="source configuration archived, deployed EE2TCP/gravity parameters unverified",
            actual_velocity_available=False, physical_completion_available=False,
            action_schema=["x_mm", "y_mm", "z_mm", "rotvec_x_rad", "rotvec_y_rad", "rotvec_z_rad", "fx_N", "fy_N", "fz_N"],
            observation_schema="pose6 + measured force3; force history Lx3; images (checkpoint config authoritative)",
            controller_applied_final_force=None,
            controller_applied_note="targetF = FC_AC_desX; internal commanded_Fd can differ (precontact hold/Q frame); NOT published",
            raw_sensor_available=False,
            raw_sensor_note="unfiltered/unzeroed sensor-frame FT not published; ftdata_tcp_raw is zeroed+MOV filtered+axis-rotated",
            sources={
                node.pose_topic: "headerless joint-feedback FK * EE2TCP; base mm, rotation vector rad (verified default only)",
                node.force_topic: "headerless republished ftdata; may repeat same acquisition; NOT new sensor samples",
                "/ur10skku/ftdata": "WrenchStamped base; MOV + conditional gravity compensation; nominal source loop 2000Hz",
                "/ur10skku/ftdata_tcp_raw": "WrenchStamped tcp axes BEFORE gravity, AFTER zero/MOV; nominal 100Hz",
                "/ur10skku/ftdata_tcp": "WrenchStamped tcp axes AFTER conditional gravity; nominal 100Hz",
                "/ur10skku/targetP": "controller-reported target_pose; base mm/rad; not actual feedback",
                "/ur10skku/targetF": "controller-reported FC_AC_desX[6:9], unused torques zero; frame runtime-dependent",
                "/ur10skku/ctlMode": "controller state; NOT spindle state"},
            extra_telemetry_enabled=self.extra_telemetry_enabled,
            sampling=dict(max_hz_per_source=sample_hz, zero_means_unlimited=True,
                          streams=["tcp_pose", "wrench", "controller_targets", "legacy"],
                          commands_and_events="node_sent, policy_prediction and events are not downsampled",
                          note="receipt-time sampling; skipped samples are not middleware loss; cannot resolve high-frequency force peaks"),
            subscriber_qos="optional extra telemetry: BEST_EFFORT KEEP_LAST 1 VOLATILE",
            middleware_loss_count=None, stale_threshold_ns=None,
            warnings=["No hardware verification", "No surface normal/contact footprint/Preston coefficient",
                      "Clock synchronization and active external calibration unverified",
                      "No physical task/point-transition completion signal; use manual intervals"])
        if rpm is None:
            metadata["warnings"].append("RPM UNKNOWN: supply metrics_rpm_setpoint (logging only)")
            node.get_logger().warn("[METRICS] RPM unknown: recorded null; not a spindle command")
        self.recorder = ExecutionRecorder(node.metrics_log_dir or root / "logs/inference_metrics",
            node.metrics_run_tag, metadata, int(params["metrics_queue_size"]), warn=node.get_logger().warn)
        self.event("run_start", task_phase="unknown", physical_task_started=False)

    def timing(self, msg=None, source="inference_node"):
        try:
            ros_ns = self.node.get_clock().now().nanoseconds
        except Exception:
            ros_ns = None  # shutdown clock unavailable; don't invent time zero
        return stamp(ros_ns, msg, source)

    def event(self, name, **details):
        if name.startswith("inference_"):
            details.setdefault("inference_id", self.inference_id)
        self.recorder.event(name, self.timing(), **details)

    def start_observer(self):
        # Avoid the additional DDS traffic entirely by default, including 2kHz FT.
        # Throttling inside its callback alone would not reduce publisher/network load.
        if not self.extra_telemetry_enabled:
            self.event("telemetry_disabled", reason="metrics_extra_telemetry_enable=false; existing policy pose/force inputs still recorded")
            return
        # Opt-in telemetry uses a separate executor, but still shares Python's GIL.
        # Only checked default robot has known topic schemas. Overrides stay unknown.
        if self.node.pose_topic != "/ur10skku/currentP" or self.node.force_topic != "/ur10skku/currentF":
            self.event("telemetry_unavailable", reason="non-default robot topics; extra topic schema not verified")
            return
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from geometry_msgs.msg import WrenchStamped
        from std_msgs.msg import Float64MultiArray, String
        self.observer = Node("e1_metrics_observer", context=self.node.context, use_global_arguments=False,
            parameter_overrides=[Parameter("use_sim_time", value=self.node.get_parameter("use_sim_time").value)])
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        for suffix, representation, frame in [
            ("ftdata", "base_filtered_conditional_gravity", "robot_base"),
            ("ftdata_tcp_raw", "tcp_zeroed_filtered_before_gravity", "tcp_axes"),
            ("ftdata_tcp", "tcp_filtered_conditional_gravity", "tcp_axes")]:
            topic = "/ur10skku/" + suffix
            self.observer.create_subscription(WrenchStamped, topic,
                lambda msg, t=topic, r=representation, f=frame: self.wrench(msg, t, r, f), qos)
        for suffix in ("targetP", "targetF"):
            topic = "/ur10skku/" + suffix
            self.observer.create_subscription(Float64MultiArray, topic,
                lambda msg, t=topic: self.controller_target(msg, t), qos)
        self.observer.create_subscription(String, "/ur10skku/ctlMode", self.controller_mode, qos)
        self.executor = SingleThreadedExecutor(context=self.node.context)
        self.executor.add_node(self.observer)
        def spin():
            try:
                self.executor.spin()
            except Exception as exc:
                self.event("telemetry_executor_stopped", error=repr(exc))
        self.observer_thread = threading.Thread(target=spin, name="metrics-ros-observer", daemon=True)
        self.observer_thread.start()

    def stage(self):
        stage = getattr(getattr(self.node, "stage", None), "name", "unknown")
        if stage != self.last_stage:
            self.event("stage_observed", previous=self.last_stage, stage=stage,
                       machining_interval="unknown; stage not a force-threshold segmentation")
            self.last_stage = stage

    def _sample_allowed(self, source, now_ns=None):
        """Gate before row construction/queueing; never gates control observations."""
        if not self.sample_period_ns:
            return True
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        previous = self._sample_last_ns.get(source)
        if previous is not None and now_ns - previous < self.sample_period_ns:
            self._sample_skipped[source] = self._sample_skipped.get(source, 0) + 1
            return False
        self._sample_last_ns[source] = now_ns
        return True

    def pose(self, msg):
        timing = self.timing(msg, self.node.pose_topic)
        self.last_pose_stamp = timing
        if not self._sample_allowed(self.node.pose_topic, timing["receipt_monotonic_ns"]):
            return
        vals = list(msg.data)
        row = dict(timing, **pose_fields(vals, self.node.pose_topic == "/ur10skku/currentP"))
        row["details"] = dict(source_acquisition_stamp=None, calibration="deployed EE2TCP unverified",
                              layout=str(msg.layout))
        if numeric_valid(vals, 6) and self.node.rel_use_relative and self.node._srf is not None and self.node._srf.ready:
            row["work_pose"] = dict(
                values=self.node._srf_observation_pose6(np.asarray(vals[:6], dtype=np.float32)).tolist(),
                schema="x_mm,y_mm,z_mm,rotation_vector_rad[3]",
                frame="policy_stain_relative_xy_base_z", transform=self.node.rel_transform_version,
                note="existing policy XY transform only; not a full rotated rigid TCP frame")
        self.recorder.emit("tcp_pose", row)

    def force(self, msg):
        timing = self.timing(msg, self.node.force_topic)
        self.last_force_stamp = timing
        self.wrench(msg, self.node.force_topic, "policy_source_unmasked_republished",
                    "robot_base" if self.node.force_topic == "/ur10skku/currentF" else None, timing)

    def wrench(self, msg, source, representation, frame, timing=None):
        if not self._sample_allowed(source, timing["receipt_monotonic_ns"] if timing else None):
            return
        timing = timing or self.timing(msg, source)
        if hasattr(msg, "data"):
            vals = list(msg.data)
        else:
            w = getattr(msg, "wrench", msg)
            vals = [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z]
        row = dict(timing, representation=representation, raw_values=vals,
                   force_unit="N", torque_unit="N*m", semantic_frame=frame,
            correction_status=representation + "; active gravity flag/bias epoch unknown",
                   transform_validity="source transform; pose stamp/torque reference unknown",
                   details=dict(surface_normal_unknown=True, no_new_correction=True,
                                unit_provenance="ROS Wrench SI convention; hardware sensor scaling unverified",
                                repeated_acquisition_possible=representation == "policy_source_unmasked_republished"))
        row["validity"] = "valid" if numeric_valid(vals, 6) else "partial_or_invalid"
        row.update(zip(["fx", "fy", "fz", "tx", "ty", "tz"], vals[:6]))
        self.recorder.emit("wrench", row)

    def controller_mode(self, msg):
        mode = msg.data
        if mode != getattr(self, "last_controller_mode", None):
            self.recorder.event("controller_mode", self.timing(msg, "/ur10skku/ctlMode"), mode=mode)
            self.last_controller_mode = mode

    def controller_target(self, msg, source):
        if not self._sample_allowed(source):
            return
        vals = list(msg.data)
        pose = source.endswith("targetP")
        row = dict(self.timing(msg, source), command_stage="controller_reported_target",
            command_mode="targetP" if pose else "targetF", raw_values=vals,
            semantic_frame="robot_base" if pose else "runtime_force_coordinate_unknown",
            orientation_unit="rotvec_rad" if pose else None, position_unit="mm" if pose else None,
            force_unit=None if pose else "N", execution_status="not_physical_completion",
            details=dict(command_id_unavailable=True, final_internal_force_setpoint_verified=False))
        row.update(zip(["x", "y", "z", "rx", "ry", "rz"] if pose else ["fx", "fy", "fz"], vals))
        row["validity"] = "valid" if numeric_valid(vals, 6) else "invalid"
        self.recorder.emit("commands", row)

    def reference(self):
        if self.reference_saved:
            return
        n = self.node
        if n.rel_use_relative and (n._srf is None or not n._srf.ready):
            return
        xy = n._srf.stain_origin.tolist() if n.rel_use_relative else [0.0, 0.0]
        if not numeric_valid(xy, 2):
            if not getattr(self, "reference_invalid_reported", False):
                self.event("reference_invalid", raw_center_xy=xy)
                self.reference_invalid_reported = True
            return
        row = dict(self.timing(source=n.stain_origin_topic if n.rel_use_relative else "absolute_identity"),
            center_xy_mm=xy, direction_rad=n._srf.stain_angle if n.rel_use_relative else None,
            semantic_frame="robot_base", source_stamp_ns=None,
            reference_transform=dict(version=n.rel_transform_version, use_relative=n.rel_use_relative,
                translation_xy_mm=xy, rotation_canonicalization=getattr(n, "_canon_active", False),
                canon_alpha_rad=getattr(n, "_canon_alpha", None)),
            source_time_note="time first observed ready; origin message is headerless; not detector acquisition",
            surface_normal=None, contact_offset=None)
        if self.recorder.emit("roi", row):
            self.reference_saved = True
            self.event("reference_fixed", reference=row)

    def image(self, msg, rgb):
        if not self.snapshot_enabled:
            return
        timing = self.timing(msg, self.node.image_topic)
        # rgb_raw is an owned frame from message decoding; retained only, not mutated.
        self.last_image = (timing, rgb)
        if not self.initial_image_saved:
            self.initial_image_saved = self.recorder.emit("snapshot", dict(timing, kind="initial_image", rgb=rgb.copy()))

    def mask(self, msg, mask):
        if self.snapshot_enabled and not self.mask_saved:
            self.mask_saved = self.recorder.emit("snapshot", dict(self.timing(msg, self.node.stain_mask_topic),
                kind="mask", rgb=(mask.clip(0, 1) * 255).astype("uint8")))

    def infer_start(self, force_history):
        self.inference_id += 1
        self.stage()
        vals = [float(x) for row in force_history for x in row]
        self.event("inference_start", inference_id=self.inference_id,
            force_observation=bool(self.node.use_force_observation),
            force_observation_method="measured" if self.node.use_force_observation else "constant_zero",
            raw_history_norm=math.sqrt(sum(x*x for x in vals)), raw_history_shape=[len(force_history), 3],
            conditioned_force_norm=None if self.node.use_force_observation else 0.0,
            norm_note="raw host history before padding/normalization; ON normalized norm not copied from GPU")

    def prediction(self, seq):
        self.predicted_force = seq[:, 6:9].copy()
        frame = "stain_relative_xy_base_z" if self.node.rel_use_relative else "robot_base"
        timing = self.timing(source="denormalized_policy_prediction")
        for i, vals in enumerate(seq.tolist()):
            row = dict(timing, inference_id=self.inference_id, plan_id=self.inference_id, action_index=i,
                command_stage="policy_prediction", semantic_frame=frame, orientation_unit="rotvec_rad",
                position_unit="mm", force_unit="N", execution_status="prediction_only_not_sent",
                raw_values=vals, validity="valid" if numeric_valid(vals, 9) else "invalid")
            row.update(zip(["x", "y", "z", "rx", "ry", "rz", "fx", "fy", "fz"], vals))
            self.recorder.emit("commands", row)

    def postprocess(self, seq):
        # Existing CPU denormalized arrays only; no GPU reads or new control transforms.
        changed = (self.predicted_force != seq[:, 6:9]).any(axis=1)
        self.event("policy_force_postprocess", inference_id=self.inference_id,
            changed_action_indices=changed.nonzero()[0].tolist(),
            configured_rules=dict(force_xy_cmd_enable=self.node.force_xy_cmd_enable,
                                  fz_hard_limit_N=self.node.fz_hard_limit),
            reason="existing XY command disable/limit and Fz hard limit; prediction preserved in commands.csv")

    def plan(self, plan):
        self.plan_ids[id(plan)] = self.inference_id
        if len(self.plan_ids) > 256:
            del self.plan_ids[next(iter(self.plan_ids))]
        self.event("inference_end", inference_id=self.inference_id, plan_id=self.inference_id, status="plan_appended")

    def sent(self, values, source, mode="cmdMotion", indices=None, future=None, blocked=None, reason=None, velocity=None):
        self.command_id += 1
        cid = self.command_id
        plans = getattr(self.node, "plans", [])
        pid = self.plan_ids.get(id(plans[-1])) if plans else None
        timing = self.timing(source=source)
        if not self.first_sent:
            self.first_sent = True
            self.event("first_command_sent", command_id=cid, physical_task_started="unknown")
        self.stage()
        width = 9 if mode in ("cmdMotion", "PTP9D", "PTP9D_STREAM_APPEND") else 6 if mode == "PTP" else 3 if mode == "PTP9D_STREAM_SET_FORCE" else 0
        chunks = [values[i:i+width] for i in range(0, len(values), width)] if width else [values]
        for i, vals in enumerate(chunks):
            row = dict(timing, inference_id=pid, plan_id=pid, command_id=cid,
                action_index=indices[i] if indices is not None and i < len(indices) else None,
                command_stage="node_sent", command_mode=mode,
                semantic_frame="pose:robot_base; force:controller_config_dependent",
                orientation_unit=("rotvec_rad" if mode == "cmdMotion" else "rotvec_components_deg") if width in (6, 9) else None,
                position_unit="mm" if width in (6, 9) else None, force_unit="N", raw_values=list(vals), limited=blocked,
                limiting_reason=reason, execution_status="sent_not_verified_executed",
                details=dict(target_velocity_mm_s=velocity, force_xy_cmd_enable=self.node.force_xy_cmd_enable,
                    limiting_scope="this send boundary; earlier force changes have policy_force_postprocess events",
                    contact_gate=getattr(self.node, "_contact", None),
                    action_index_note="unknown for interpolation/aggregation/non-policy commands" if indices is None else None,
                    final_controller_setpoint=None))
            keys = ["fx", "fy", "fz"] if mode == "PTP9D_STREAM_SET_FORCE" else ["x", "y", "z", "rx", "ry", "rz", "fx", "fy", "fz"]
            row.update(zip(keys, vals))
            self.recorder.emit("commands", row)
        self.event("command_sent", command_id=cid, plan_id=pid, command_mode=mode)
        if blocked:
            self.event("safety_limit", command_id=cid, reason=reason)
        if future is not None:
            def response(f):
                try:
                    r = f.result()
                    self.event("service_response", command_id=cid, success=r.success, message=r.message,
                               physical_completion="unknown")
                except Exception as exc:
                    self.event("service_error", command_id=cid, error=repr(exc), physical_completion="unknown")
            future.add_done_callback(response)

    def legacy(self, cmd, blocked):
        if not blocked and not self._sample_allowed("legacy"):
            return
        n = self.node
        with n._lock:
            p = None if n._pose6 is None else n._pose6.tolist()
            f = None if n._force is None else [float(n._force[k]) if k < n._force.size else None for k in n.force_indices]
        row = self.timing(source="legacy_latest_snapshot_not_measurement")
        row.update(t_wall=time.time(), t_elapsed_sec=time.monotonic()-n._metrics_t0,
                   stage=n.stage.name, policy_class=n.policy_class, ckpt_dir=n.ckpt_dir,
                   contact=n._contact, cmd_safety_blocked=blocked,
                   measurement_semantics="latest cached values; NOT independent sensor samples; includes service-response snapshots")
        row.update(zip(["meas_x_mm", "meas_y_mm", "meas_z_mm", "meas_rx", "meas_ry", "meas_rz"], p or []))
        row.update(zip(["meas_fx_N", "meas_fy_N", "meas_fz_N"], f or []))
        row.update(zip(["cmd_x_mm", "cmd_y_mm", "cmd_z_mm", "cmd_rx", "cmd_ry", "cmd_rz", "cmd_fx_N", "cmd_fy_N", "cmd_fz_N"], cmd.tolist()))
        for key, st in [("pose", self.last_pose_stamp), ("force", self.last_force_stamp)]:
            row[key+"_receipt_monotonic_ns"] = st["receipt_monotonic_ns"] if st else None
            row[key+"_age_ns"] = row["receipt_monotonic_ns"]-st["receipt_monotonic_ns"] if st else None
        row["stale"] = None  # no unverified age threshold
        self.recorder.emit("legacy", row)

    def close(self):
        try:
            if self.executor is not None:
                self.executor.shutdown(timeout_sec=1.0)
            if self.observer_thread is not None:
                self.observer_thread.join(timeout=1.0)
            if self.observer is not None:
                self.observer.destroy_node()
        except Exception as exc:
            self.event("telemetry_shutdown_error", error=repr(exc))
        if self.last_image is not None:
            timing, rgb = self.last_image
            self.recorder.emit("snapshot", dict(timing, kind="final_image", rgb=rgb.copy()))
        self.event("sampling_summary", skipped_by_source=dict(self._sample_skipped),
                   note="intentional logging downsampling; policy observations unchanged")
        self.event("run_end", reason="node_destroy", physical_task_completion="unknown")
        self.recorder.close(timeout=3.0)
