"""Offline only: real inference methods/Flow sampler, fake ROS transport, no rclpy.init."""
import ast
from collections import deque
import csv
import json
from pathlib import Path
import random
import threading
from types import SimpleNamespace as NS, MethodType

import numpy as np
import pytest
import torch

from nrs_imitation.execution_metrics import ExecutionRecorder, stamp, pose_fields
from nrs_imitation.inference_metrics import InferenceMetrics, PARAMETERS
from nrs_imitation import inference_core as core
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Image

ROOT = Path(__file__).resolve().parents[4]
CKPT = ROOT / "checkpoints/flow/polishing/single_cam/e1_force_observation_20260916/off/20260916_1531"


def rows(path, name):
    with (path / name).open() as f:
        return list(csv.DictReader(f))


class FakeNode:
    """Bind repository methods, not a reimplementation of normalization/control."""
    @property
    def _parameters(self):
        # Let the installed rclpy Node's parameter enumeration run unchanged.
        # A fake list_parameters method previously hid an API absent in Humble.
        from rclpy.parameter import Parameter
        return {name: Parameter(name, value=value) for name, value in self.params.items()}

    def __getattr__(self, name):
        method = getattr(core.NodeCmdMotionInfer, name, None)
        if callable(method):
            return MethodType(method, self)
        raise AttributeError(name)

    def __init__(self, root, policy=None, force_on=False):
        defaults = {}
        tree = ast.parse(Path(core.__file__).read_text())
        for item in ast.walk(tree):
            if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and item.func.attr == "declare_parameter" and len(item.args) == 2:
                try:
                    defaults[ast.literal_eval(item.args[0])] = ast.literal_eval(item.args[1])
                except (ValueError, TypeError):
                    pass
        self.__dict__.update(defaults)
        self.__dict__.update(dict(act_root=str(ROOT), ckpt_dir=str(CKPT), policy_class="FLOW", policy=policy,
            metrics_log_dir=str(root), metrics_run_tag="offline_OFF" if not force_on else "offline_ON",
            metrics_log_enable=True, use_force_observation=force_on, force_indices=[0, 1, 2],
            use_force_history=True, force_history_len=30, chunk_size=8, action_dim=9,
            use_gripper=False, use_gripper_history=False, use_global_image=False, use_stain_mask=False,
            rel_use_relative=True, rel_transform_version="stain_relative_v1", obs_force_xy_zeroed=False,
            _canon_setup_done=True, _canon_active=False, _canon_alpha=0., _srf_demo_start_absolutized=True,
            auto_move_to_demo_start=False, _demo_start_align_done=True, stage=core.Stage.TRACK,
            flow_replan_interval_steps=0, flow_local_anchor_enable=False, flow_infer_steps=2,
            device=torch.device("cpu"), camera_names=["cam0"], resize_hw=32,
            normalize_qpos_enabled=True, denorm_action_enabled=True, orientation_lock_enable=False,
            force_xy_cmd_enable=False, action_type="absolute", policy_z_offset_mm=0., fz_hard_limit=0.,
            _lock=threading.Lock(), _force_hist=deque(maxlen=30), _gripper_hist=deque(),
            _gripper_position=None, _gripper_current_mA=None, _gripper_last_pair_t=None,
            _pose6=None, _force=None, _img_cam0=np.full((32, 32, 3), 120, np.uint8), _img_cam1=None,
            _stain_mask=None, plans=deque(), _infer_plan_count=0, _flow_fixed_initial_noise=None,
            _flow_noise_create_count=0, flow_deterministic_noise=True, flow_noise_seed=0,
            _ptp9d_track_active=False, _ptp9d_inflight=False, _metrics=None,
            _contact=True, _cmd_safety_latched=False, _cmd_safety_hold_pose6=None,
            _start_pose6=None, prev_cmd=None, cmd_safety_enable=True, cmd_safety_max_xyz_from_current_mm=0.,
            _cmd_safety_last_log=0., _metrics_t0=core._monotonic(), errors=[]))
        self._cam_prev_raw_gray = self._cam_prev_proc_gray = None
        self._offline_mock_test = True
        if force_on:
            self.ckpt_dir = str(CKPT.parent.parent / "on/20260916_1531")
        from stain_relative_frame.relative_frame import RelativeFrameAdapter
        adapter = RelativeFrameAdapter([400., 500.], use_relative=True)
        self._srf = NS(ready=True, stain_origin=np.array([400., 500.]), stain_angle=1.57,
                       observation=adapter.observation, command=adapter.command)
        self.stats = core._load_dataset_stats(self.ckpt_dir)
        self.params = dict(PARAMETERS, use_sim_time=False, **{k: v for k,v in self.__dict__.items() if isinstance(v, (str,bool,int,float))})
        self.get_parameter = lambda key: NS(value=self.params[key])
        self.get_clock = lambda: NS(now=lambda: NS(nanoseconds=123456789000))
        self.get_logger = lambda: NS(warn=lambda m: None, info=lambda m: None, error=self.errors.append)
        self.sent_messages = []
        self.pub_cmd = NS(publish=lambda m: self.sent_messages.append(list(m.data)))
        for name in ("_log_flow_vector_delta", "_run_gradcam_debug", "_run_modality_importance_debug", "_publish_flow_vector_overlay_frame"):
            setattr(self, name, lambda *a, **kw: None)  # debug-only; policy path is real


@pytest.fixture(scope="module")
def policy():
    import sys
    sys.path.insert(0, str(ROOT / "source"))
    from models.flow_core import FlowRGBPolicy
    torch.set_num_threads(1)
    torch.manual_seed(10)
    cfg = dict(num_queries=8, state_dim=9, action_dim=9, force_dim=3, camera_names=["cam0"],
        obs_mode="single_cam", pretrained_backbone=False, image_backbone="resnet18", use_tcp_roi=False,
        use_force_history=True, force_encoder_hidden_dim=8, flow_obs_hidden_dim=16,
        flow_image_feature_dim=16, flow_global_cond_dim=16, flow_time_embed_dim=16,
        flow_down_dims="16,32", flow_kernel_size=3, flow_n_groups=4, flow_infer_steps=2)
    return FlowRGBPolicy(cfg).eval()


def execute(root, policy, force_on, logging, force_scale, sample_hz=0.0):
    n = FakeNode(root, policy, force_on)
    n.params["metrics_sample_hz"] = sample_hz
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    py_state = random.getstate()
    if logging:
        n._metrics = InferenceMetrics(n)  # NOT start_observer: no ROS nodes/executors
    p = Float64MultiArray(data=[420., 530., 200., 0.01, 0.02, 0.03])
    n._on_pose(p)
    for i in range(30):
        n._on_force(Float64MultiArray(data=[force_scale*(i+1), -force_scale*2, force_scale*3, .1, .2, .3]))
    n._on_img(Image(height=32, width=32, encoding="rgb8", step=96, data=bytes([120])*3072))
    before = n._force.copy()
    n._on_infer_timer()
    assert len(n.plans) == 1, n.errors
    prediction = n.plans[-1].seq_den.copy()
    command = n._publish_cmd(prediction[0])
    assert np.array_equal(before, n._force)
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    assert py_state == random.getstate()
    assert not n.errors, n.errors
    path = None
    if logging:
        n._metrics.close()
        path = n._metrics.recorder.path
        assert not n._metrics.recorder.write_errors
    return prediction, command, path


def test_off_isolation_real_preprocess_flow_and_logging_invariance(tmp_path, policy):
    for force_on in (False, True):
        a, cmd_a, _ = execute(tmp_path, policy, force_on, False, 1.)
        b, cmd_b, path = execute(tmp_path, policy, force_on, True, 1.)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(cmd_a, cmd_b)
        assert np.any(a[:, 8] != 0), "target force output must survive OFF"
        raw = rows(path, "wrench.csv")
        assert len(raw) == 30 and float(raw[-1]["fx"]) == 30
        assert float(raw[-1]["fz"]) == 3 and float(raw[-1]["tz"]) == .3
        actual = rows(path, "tcp_pose.csv")[0]
        sent = [r for r in rows(path, "commands.csv") if r["command_stage"] == "node_sent"][0]
        assert float(actual["x"]) == 420 and float(actual["x"]) != float(sent["x"])
        assert float(sent["fx"]) == 0 and float(sent["fy"]) == 0
        assert float(sent["fz"]) == cmd_b[8]
        events = [json.loads(l) for l in (path / "events.jsonl").read_text().splitlines()]
        start = next(e for e in events if e["event"] == "inference_start")
        assert start["details"]["force_observation_method"] == ("measured" if force_on else "constant_zero")
        if not force_on:
            c, cmd_c, _ = execute(tmp_path, policy, False, True, -73.)
            np.testing.assert_allclose(b, c, atol=1e-6, rtol=0)
            np.testing.assert_allclose(cmd_b, cmd_c, atol=1e-6, rtol=0)
        # ON reachability: observe actual conditioning inputs, not action divergence requirement.
        if force_on:
            captured = []
            original = policy._condition
            def capture(*args, **kw):
                captured.append((kw["qpos"].detach().clone(), kw["force_history"].detach().clone()))
                return original(*args, **kw)
            policy._condition = capture
            try:
                execute(tmp_path, policy, True, False, 1.)
                execute(tmp_path, policy, True, True, 5.)
            finally:
                policy._condition = original
            assert not torch.equal(captured[0][0][:, 6:9], captured[-1][0][:, 6:9])
            assert not torch.equal(captured[0][1], captured[-1][1])


def test_runtime_parameter_snapshot_uses_installed_rclpy_api(tmp_path):
    n = FakeNode(tmp_path)
    n.params["diagnostics.sample"] = 3
    n.params["metrics_rpm_setpoint"] = ""
    m = InferenceMetrics(n)
    m.close()
    meta = json.loads((m.recorder.path / "metadata.json").read_text())
    assert meta["runtime_parameters"] == n.params
    assert meta["rpm_setpoint"] is None
    assert not m.recorder.write_errors


def alignment_node(tmp_path, monkeypatch):
    n = FakeNode(tmp_path)
    clock = [100.0]
    monkeypatch.setattr(core, "_monotonic", lambda: clock[0])
    n.auto_move_to_demo_start = True
    n._demo_start_align_done = False
    n._ptp_alignment_requested = True
    n._ptp_alignment_failed = False
    n._ptp_feedback_pending = False
    n._ptp_feedback_deadline = 0.0
    n._ptp_alignment_target6 = np.array(
        [468.150, 357.508, 214.157, -0.16069, 0.014245, 2.3556], dtype=np.float32)
    n._pose6 = n._ptp_alignment_target6.copy()
    n._pose_receipt_t = clock[0]
    n.service_requests = []
    n.reset_calls = []
    from concurrent.futures import Future
    n.force_future = Future()
    def call(req):
        n.service_requests.append(req)
        return n.force_future
    n._ptp_client = NS(call_async=call, service_is_ready=lambda: True)
    n._reset_after_demo_start_alignment = lambda **kw: n.reset_calls.append(kw)
    return n, clock, NS(result=lambda: NS(success=True, message="PTP motion generated"))


def test_ptp_success_with_recorded_frozen_pose_blocks_force_and_inference(tmp_path, monkeypatch):
    n, clock, response = alignment_node(tmp_path, monkeypatch)
    # Recorded 2026-09-17 failure: success response but TCP stayed at home.
    n._on_pose(Float64MultiArray(data=[445.0, 394.5, 220.0, -0.0001195, -0.0001174, 2.356]))
    n._on_ptp_alignment_done(response)
    assert n._ptp_feedback_pending and not n.service_requests
    clock[0] += n.ptp_feedback_timeout_sec + .01
    n._on_pose(Float64MultiArray(data=n._pose6.tolist()))  # fresh, still unmoved
    n._on_control_timer()
    n._on_infer_timer()
    assert n._ptp_alignment_failed and not n._demo_start_align_done
    assert not n.service_requests and not n.reset_calls and not n.plans
    assert any("alignment NOT verified" in e for e in n.errors)
    # No automatic recovery/retry after the failure, even if feedback changes.
    n._on_pose(Float64MultiArray(data=n._ptp_alignment_target6.tolist()))
    n._on_control_timer()
    assert not n.service_requests and not n._demo_start_align_done


@pytest.mark.parametrize("fault", ["stale", "nan", "orientation", "missing"])
def test_ptp_rejects_invalid_or_unreached_feedback(tmp_path, monkeypatch, fault):
    n, clock, response = alignment_node(tmp_path, monkeypatch)
    if fault == "stale":
        n._pose_receipt_t -= n.ptp_feedback_max_age_sec + 1
    elif fault == "nan":
        n._pose6[0] = float("nan")
    elif fault == "orientation":
        n._pose6[3] += n.demo_start_rotation_tolerance_rad + .1
    else:
        n._pose6 = None
    n._on_ptp_alignment_done(response)
    clock[0] += n.ptp_feedback_timeout_sec + .01
    n._on_control_timer()
    assert n._ptp_alignment_failed and not n._demo_start_align_done
    assert not n.service_requests and not n.reset_calls


def test_ptp_waits_for_feedback_then_switches_force_and_completes(tmp_path, monkeypatch):
    n, clock, response = alignment_node(tmp_path, monkeypatch)
    n._pose6[0] += 20.0
    n._on_ptp_alignment_done(response)
    assert not n.service_requests
    clock[0] += .1
    n._on_pose(Float64MultiArray(data=n._ptp_alignment_target6.tolist()))
    n._on_control_timer()
    assert [r.command_mode for r in n.service_requests] == ["Force"]
    assert not n._demo_start_align_done
    n.force_future.set_result(NS(success=True, message="Force mode"))
    assert n._demo_start_align_done and len(n.reset_calls) == 1
    assert not n._ptp_alignment_failed


def test_ptp_rechecks_feedback_after_force_response(tmp_path, monkeypatch):
    n, clock, response = alignment_node(tmp_path, monkeypatch)
    n._on_ptp_alignment_done(response)
    assert [r.command_mode for r in n.service_requests] == ["Force"]
    clock[0] += n.ptp_feedback_max_age_sec + .01
    n.force_future.set_result(NS(success=True, message="Force mode"))
    assert n._ptp_alignment_failed and not n._demo_start_align_done
    assert not n.reset_calls


def test_rates_stamps_ids_roi_and_reload(tmp_path):
    n = FakeNode(tmp_path)
    n.params["metrics_sample_hz"] = 0.0  # this test checks full-rate provenance
    m = InferenceMetrics(n)
    n._metrics = m
    for i in range(10):
        msg = WrenchStamped()
        msg.header.stamp.sec, msg.header.stamp.nanosec = 5, i*1_000_000
        msg.header.frame_id = "sensor_test"
        msg.wrench.force.z = -7.0
        m.wrench(msg, "/ur10skku/ftdata", "synthetic", "test_frame")
        if i in (0, 5):
            n._on_pose(Float64MultiArray(data=[float(i), 2., 3., 0., 0., 0.]))
    m.reference()
    n._srf.stain_origin[:] = 99
    m.reference()  # must not replace frozen reference
    img = NS(header=NS(stamp=NS(sec=10, nanosec=20), frame_id="camera", seq=None))
    m.image(img, np.zeros((4, 4, 3), np.uint8))
    m.image(img, np.ones((4, 4, 3), np.uint8))
    m.close()
    path = m.recorder.path
    assert not m.recorder.write_errors
    assert len(rows(path, "tcp_pose.csv")) == 2
    wr = rows(path, "wrench.csv")
    assert [int(r["source_stamp_ns"]) for r in wr] == [5_000_000_000+i*1_000_000 for i in range(10)]
    assert all(float(r["fz"]) == -7 for r in wr)
    assert all(r["normal_force_signed"] == "" for r in wr)
    assert all(r["source_stamp_ns"] == "" for r in rows(path, "tcp_pose.csv"))
    roi = json.loads((path / "roi.json").read_text())
    assert roi["reference"]["center_xy_mm"] == [400., 500.]
    assert roi["initial_image"] and roi["final_image"]
    for file in path.glob("*.json"):
        assert json.loads(file.read_text())["run_id"] == path.name
    for name in ("wrench.csv", "tcp_pose.csv", "commands.csv"):
        assert all(r["run_id"] == path.name for r in rows(path, name))
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_inference_log import inspect
    report = inspect(path)
    assert report["run_id_mismatches"] == 0 and report["issues"] == []
    assert report["rpm"]["rpm_setpoint"] is None
    assert report["middleware_loss_count"] is None
    limited = inspect(path, max_age_ms=1., source_max_gap_ms={"/ur10skku/ftdata": 2.})
    assert limited["streams"]["wrench:/ur10skku/ftdata"]["explicit_gap_limit_ms"] == 2.
    assert limited["streams"]["wrench:/ur10skku/ftdata"]["stale"] is None
    # Crash/incomplete logs remain readable even without a final summary.
    (path / "summary.json").rename(path / "summary.saved.json")
    incomplete = inspect(path)
    assert incomplete["streams"] and any("summary.json" in x for x in incomplete["issues"])


def test_bounded_overflow_collision_flush(tmp_path):
    gate = threading.Event()
    class SlowRecorder(ExecutionRecorder):
        def _provenance(self):
            gate.wait(timeout=2)
    a = SlowRecorder(tmp_path, "same", {}, queue_size=1)
    assert a.ready.wait(2)
    for i in range(100):
        a.event("sample", stamp(i))
    gate.set()
    assert a.close()
    b = ExecutionRecorder(tmp_path, "same", {})
    b.event("sample", stamp(1))
    assert b.close()
    assert a.path != b.path
    s = json.loads((a.path / "summary.json").read_text())
    assert s["counts"]["events"]["dropped"] == 99
    assert s["counts"]["events"]["written"] == 1
    assert s["drained"]
    assert len((b.path / "events.jsonl").read_text().splitlines()) == 1
    with pytest.raises(ValueError):
        ExecutionRecorder(tmp_path, "unbounded", {}, queue_size=0)


def test_write_error_report_and_no_control_exception(tmp_path):
    class BadDisk(ExecutionRecorder):
        def _provenance(self):
            raise OSError("mock disk full")
    warnings = []
    rec = BadDisk(tmp_path, "bad_disk", {}, warn=warnings.append)
    rec.event("sample", stamp(1))
    assert rec.close()
    assert rec.write_errors and any("WRITE ERROR" in w for w in warnings)
    assert (rec.path / "logger_error.json").exists()
    assert not rec.emit("events", {})
    assert (rec.path / "metadata.json").exists()


def test_service_request_exact_rows_response_not_execution(tmp_path):
    from concurrent.futures import Future
    sent = []
    for logging in (False, True):
        n = FakeNode(tmp_path)
        if logging:
            n._metrics = InferenceMetrics(n)
        n._on_pose(Float64MultiArray(data=[1., 2., 3., 0., 0., 0.]))
        n._on_force(Float64MultiArray(data=[0., 0., 8., .1, .2, .3]))
        n._ptp9d_track_active = True
        n.ptp9d_use_stream = False
        future = Future()
        calls = []
        def call(req):
            calls.append(list(req.target_pose))
            return future
        n._ptp9d_client = NS(service_is_ready=lambda: True, call_async=call)
        seq = np.tile(np.array([1., 2., 3., .01, .02, .03, 0., 0., 9.], np.float32), (8, 1))
        seq[:, 0] += np.arange(8)
        plan = core.Plan(t0=core._monotonic(), seq_den=seq, local_anchor_applied=False)
        n.plans.append(plan)
        n._metrics_call("plan", plan)
        n._ptp9d_advance()
        assert len(calls) == 1 and not n.errors
        sent.append(calls[0])
        n.stage = core.Stage.RELEASE  # prevent chaining the next MOCK service
        future.set_result(NS(success=True, message="transport accepted"))
        if logging:
            n._metrics.close()
            path = n._metrics.recorder.path
            records = [r for r in rows(path, "commands.csv") if r["command_stage"] == "node_sent"]
            assert len(records)*9 == len(calls[0])
            assert sum([json.loads(r["raw_values"]) for r in records], []) == calls[0]
            assert all(r["action_index"] != "" for r in records)
            events = [json.loads(l) for l in (path / "events.jsonl").read_text().splitlines()]
            response = next(e for e in events if e["event"] == "service_response")
            assert response["details"]["physical_completion"] == "unknown"
    assert sent[0] == sent[1]


def test_writer_owns_file_io_and_image_encoding(tmp_path, monkeypatch):
    import cv2
    thread_names = []
    original_open, original_encode = Path.open, cv2.imwrite
    def guarded_open(path, mode="r", *args, **kwargs):
        if any(m in mode for m in ("w", "x", "a")):
            thread_names.append(threading.current_thread().name)
        return original_open(path, mode, *args, **kwargs)
    def guarded_encode(*args, **kwargs):
        thread_names.append(threading.current_thread().name)
        return original_encode(*args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(cv2, "imwrite", guarded_encode)
    rec = ExecutionRecorder(tmp_path, "io", {})
    rec.emit("snapshot", dict(stamp(10), kind="initial_image", rgb=np.zeros((2,2,3), np.uint8)))
    assert rec.close()
    assert not rec.write_errors
    assert thread_names and set(thread_names) == {"inference-metrics-writer"}


def test_observer_topics_qos_and_no_control_handles(tmp_path, monkeypatch):
    import rclpy.node
    import rclpy.executors
    from rclpy.qos import ReliabilityPolicy
    subscriptions = []
    stop = threading.Event()
    class Observer:
        def __init__(self, *args, **kw):
            assert kw["use_global_arguments"] is False
        def create_subscription(self, kind, topic, callback, qos):
            subscriptions.append((kind, topic, callback, qos))
        def destroy_node(self):
            pass
    class Executor:
        def __init__(self, **kw):
            pass
        def add_node(self, node):
            pass
        def spin(self):
            stop.wait(timeout=2)
        def shutdown(self, **kw):
            stop.set()
    monkeypatch.setattr(rclpy.node, "Node", Observer)
    monkeypatch.setattr(rclpy.executors, "SingleThreadedExecutor", Executor)
    n = FakeNode(tmp_path)
    n.params["metrics_extra_telemetry_enable"] = True
    n.context = object()
    m = InferenceMetrics(n)
    m.start_observer()
    assert {x[1] for x in subscriptions} == {"/ur10skku/ftdata", "/ur10skku/ftdata_tcp_raw",
        "/ur10skku/ftdata_tcp", "/ur10skku/targetP", "/ur10skku/targetF", "/ur10skku/ctlMode"}
    assert all(x[3].reliability == ReliabilityPolicy.BEST_EFFORT for x in subscriptions)
    assert all(x[3].depth == 1 for x in subscriptions)
    m.close()
    assert not m.recorder.write_errors


def test_default_metrics_do_not_create_extra_ros_observer(tmp_path, monkeypatch):
    import rclpy.node
    def forbidden(*args, **kwargs):
        pytest.fail("default metrics must not create a ROS node/subscriptions")
    monkeypatch.setattr(rclpy.node, "Node", forbidden)
    n = FakeNode(tmp_path)
    m = InferenceMetrics(n)
    m.start_observer()
    assert m.observer is None and m.executor is None and m.observer_thread is None
    m.close()
    meta = json.loads((m.recorder.path / "metadata.json").read_text())
    assert meta["extra_telemetry_enabled"] is False
    assert meta["sampling"]["max_hz_per_source"] == 20.0


def test_sampled_metrics_keep_latest_control_input_and_all_sent_commands(tmp_path):
    n = FakeNode(tmp_path)
    m = n._metrics = InferenceMetrics(n)
    receipt = [1_000_000_000]
    original_timing = m.timing
    def timing(*args, **kwargs):
        return dict(original_timing(*args, **kwargs), receipt_monotonic_ns=receipt[0])
    m.timing = timing
    for i in range(1000):
        receipt[0] = 1_000_000_000 + i * 1_000_000
        n._on_pose(Float64MultiArray(data=[float(i), 530., 200., 0., 0., 0.]))
        n._on_force(Float64MultiArray(data=[float(i), 2., 3., .1, .2, .3]))
    assert n._pose6[0] == n._force[0] == 999.
    assert len(n._force_hist) == 30 and n._force_hist[-1][0] == 999.
    assert m.last_pose_stamp["receipt_monotonic_ns"] == receipt[0]
    assert m.last_force_stamp["receipt_monotonic_ns"] == receipt[0]
    for i in range(5):
        m.sent([float(i)] * 9, n.cmd_topic)
    m.close()
    assert len(rows(m.recorder.path, "tcp_pose.csv")) == 20
    assert len(rows(m.recorder.path, "wrench.csv")) == 20
    sent = [r for r in rows(m.recorder.path, "commands.csv") if r["command_stage"] == "node_sent"]
    assert len(sent) == 5
    events = [json.loads(l) for l in (m.recorder.path / "events.jsonl").read_text().splitlines()]
    skipped = next(e["details"]["skipped_by_source"] for e in events if e["event"] == "sampling_summary")
    assert skipped[n.pose_topic] == skipped[n.force_topic] == 980
    assert not n.errors and not m.recorder.write_errors


@pytest.mark.parametrize("force_on", [False, True])
def test_sampled_logging_does_not_change_policy_or_commands(tmp_path, policy, force_on):
    a, cmd_a, _ = execute(tmp_path, policy, force_on, False, 1.)
    b, cmd_b, _ = execute(tmp_path, policy, force_on, True, 1., sample_hz=20.)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(cmd_a, cmd_b)


@pytest.mark.parametrize("max_hz", [0., 10.])
def test_overlay_rate_limit_preserves_all_camera_observations(tmp_path, monkeypatch, max_hz):
    n = FakeNode(tmp_path)
    n.flow_vector_overlay_enable = True
    n.flow_vector_overlay_max_hz = max_hz
    n._flow_vector_overlay_last_t = None
    n._flow_vector_overlay_count = n._flow_vector_overlay_fail_count = 0
    n._publish_flow_vector_overlay_frame = MethodType(core.NodeCmdMotionInfer._publish_flow_vector_overlay_frame, n)
    published = []
    clock = [0.]
    monkeypatch.setattr(core, "_monotonic", lambda: clock[0])
    monkeypatch.setattr(core, "_rgb_numpy_to_image_msg", lambda rgb, **kw: rgb)
    n.get_clock = lambda: NS(now=lambda: NS(to_msg=lambda: None))
    n.pub_flow_vector_overlay = NS(publish=lambda msg: published.append(clock[0]))
    n._render_flow_vector_overlay_rgb = lambda rgb: rgb
    n._preprocess_live_image = lambda rgb: rgb
    for i in range(100):
        clock[0] = i / 100.
        n._on_img(Image(height=2, width=2, encoding="rgb8", step=6, data=bytes([i]) * 12))
        assert np.all(n._img_cam0 == i)
    assert not n.errors
    if max_hz:
        assert 9 <= len(published) <= 10
        assert min(np.diff(published)) >= 0.1 - 1e-12
    else:
        assert len(published) == 100


def test_missing_nonfinite_quaternion_no_fabrication():
    short = pose_fields([1., 2.], verified=True)
    assert short["validity"] == "invalid" and "z" not in short and "qx" not in short
    bad = pose_fields([1., 2., float("nan"), 0., 0., 0.], verified=True)
    assert bad["validity"] == "invalid"
    p = pose_fields([1., 2., 3., 0., 0., np.pi], verified=True)
    assert abs(p["qz"]-1) < 1e-9 and abs(p["qw"]) < 1e-9
    assert "qx" not in pose_fields([1., 2., 3., 0., 0., 0.], verified=False)


def test_manual_rpm_is_not_a_measurement_or_command(tmp_path):
    n = FakeNode(tmp_path)
    n.params["metrics_rpm_setpoint"] = "1234"  # synthetic test value, not an experiment setting
    n.params["metrics_rpm_assumed_constant"] = True
    m = InferenceMetrics(n)
    m.close()
    meta = json.loads((m.recorder.path / "metadata.json").read_text())
    assert meta["rpm_setpoint"] == 1234 and meta["rpm_assumed_constant"] is True
    assert meta["measured_rpm"] is None and meta["rpm_measurement_available"] is False
    assert meta["rpm_setpoint_source"] == "manual:metrics_rpm_setpoint"
    assert n.sent_messages == []
    for invalid in ("nan", "inf", "-1", "not_a_number"):
        n.params["metrics_rpm_setpoint"] = invalid
        with pytest.raises(ValueError):
            InferenceMetrics(n)


@pytest.mark.parametrize("inference_mode", ["service_call", "service_stream"])
def test_clean_launch_forwards_logging_and_e1_arguments_without_executing_nodes(inference_mode):
    import importlib.util
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
    from launch.utilities import perform_substitutions, normalize_to_list_of_substitutions
    from launch_ros.actions import Node
    from launch_ros.utilities import evaluate_parameters
    path = ROOT / "behavior_ws/src/nrs_imitation/launch/inference_clean_single_cam.launch.py"
    spec = importlib.util.spec_from_file_location("clean_launch_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for use_force in ("true", "false"):
        context = LaunchContext()
        context.launch_configurations.update(dict(use_force_observation=use_force, use_stain_mask="false",
            stain_canon_enable="false", metrics_rpm_setpoint="1234", metrics_repeat_id="r01",
            metrics_log_enable="true", metrics_specimen_id="TEST", metrics_run_tag="TEST",
            inference_mode=inference_mode, gradcam_enable="false",
            removal_viz_enable="false"))
        desc = module.generate_launch_description()
        checked = False
        for entity in desc.entities:
            if isinstance(entity, DeclareLaunchArgument):
                entity.execute(context)  # declares in-memory defaults ONLY, no ROS node
            if not isinstance(entity, IncludeLaunchDescription):
                continue
            for k, v in entity.launch_arguments:
                context.launch_configurations[k] = perform_substitutions(context, normalize_to_list_of_substitutions(v))
            child = entity.launch_description_source.get_launch_description(context)
            for action in child.entities:
                if isinstance(action, DeclareLaunchArgument):
                    action.execute(context)
                if not isinstance(action, Node):
                    continue
                executable = perform_substitutions(context, normalize_to_list_of_substitutions(action.node_executable))
                if executable in ("rqt_image_view", "stain_mask_publisher"):
                    assert not action.condition.evaluate(context)
                for params in evaluate_parameters(context, action._Node__parameters):
                    if executable == "overlay_video_recorder" and isinstance(params, dict):
                        assert params["enable"] is True and params["framerate"] == 10.0
                    if isinstance(params, dict) and "metrics_log_enable" in params:
                        assert params["metrics_log_enable"] is True
                        assert params["metrics_extra_telemetry_enable"] is False
                        assert params["metrics_sample_hz"] == 20.0
                        assert params["flow_vector_overlay_max_hz"] == 10.0
                        assert params["metrics_rpm_setpoint"] == "1234"  # logging-only string, not spindle
                        assert params["metrics_repeat_id"] == "r01"
                        assert params["metrics_specimen_id"] == "TEST"
                        assert params["use_force_observation"] is (use_force == "true")
                        assert params["use_stain_mask"] is False and params["stain_canon_enable"] is False
                        assert params["track_use_ptp9d_service"] is True
                        assert params["ptp9d_use_stream"] is (inference_mode == "service_stream")
                        assert params["gradcam_enable"] is False
                        assert params["flow_infer_steps"] == 10 and params["chunk_size"] == 128
                        checked = True
        assert checked
