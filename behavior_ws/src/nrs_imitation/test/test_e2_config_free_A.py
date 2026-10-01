"""Config-free A integration: expand real launch actions without hardware I/O."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from launch import LaunchContext, LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, GroupAction,
    IncludeLaunchDescription, LogInfo, OpaqueFunction, PopEnvironment,
    PopLaunchConfigurations, PushEnvironment, PushLaunchConfigurations,
    RegisterEventHandler, SetEnvironmentVariable, SetLaunchConfiguration,
    UnsetLaunchConfiguration)
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters
from rclpy.parameter import Parameter

from nrs_imitation.e2_direct_a import (ROOT, DEFAULT_CHECKPOINT, PARAMETERS,
                                      build_runtime, object_hash)
from nrs_imitation.e2_timed_execution import executor_settings, TimedExecution, TimedPlan

SPEC = importlib.util.spec_from_file_location('config_free_A_launch',
    ROOT/'behavior_ws/src/nrs_imitation/launch/e2_a.launch.py')
launch_a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch_a)


def context(mode='run'):
    ctx = LaunchContext()
    for action in launch_a.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(ctx)
    ctx.launch_configurations['mode'] = mode
    return ctx


def test_check_reads_original_checkpoint_without_experiment_config(tmp_path, monkeypatch):
    from nrs_imitation import e2_providers
    def forbidden_config(*args):
        pytest.fail('Direct A must not read an experiment config')
    monkeypatch.setattr(e2_providers, 'load_config', forbidden_config)
    monkeypatch.setattr(launch_a, 'ROOT', tmp_path)
    actions = launch_a.configure(context('check'))
    assert all(isinstance(a, LogInfo) for a in actions)
    assert not list(tmp_path.iterdir())


def test_real_provider_startup_accepts_original_filename_and_pose6_stats(monkeypatch):
    from nrs_imitation import inference_core as core
    # Run the real startup/parameter/schema path. Stop before model creation;
    # actual weight loading is covered by the separate offline inference run.
    monkeypatch.setattr(core.Node, '__init__', lambda *args, **kwargs: None)
    monkeypatch.setattr(core, 'StainOriginClient', lambda *args, **kwargs: NS())
    overrides = dict(e2_direct_a=True, checkpoint=DEFAULT_CHECKPOINT,
        e2_session_id='offline_startup_test', use_force_observation=False,
        use_force_history=False, use_stain_mask=False, auto_move_to_demo_start=True,
        chunk_size=128, flow_replan_interval_steps=120, flow_deterministic_noise=True,
        flow_local_anchor_enable=False, force_history_len=30,
        force_xy_cmd_enable=False, fz_hard_limit=0.,
        gradcam_enable=False, gradcam_save=False)
    class StartupVerified(Exception):
        pass
    class StubProvider(core.NodeCmdMotionInfer):
        def declare_parameter(self, name, default, *args, **kwargs):
            self.values[name] = overrides.get(name, default)
        def get_parameter(self, name):
            return NS(value=self.values[name])
        def get_logger(self):
            return NS(info=lambda _: None, warn=lambda _: None, error=lambda _: None)
        def _load_policy_and_ckpt_from_act_root(self):
            assert self._e2_transport and self.motion_only and self.action_dim == 6
            assert self.checkpoint == DEFAULT_CHECKPOINT
            assert self.stats.qpos_a.shape == (6,)
            assert self._e2_context['il']['checkpoint'] == DEFAULT_CHECKPOINT
            raise StartupVerified()
    node = StubProvider.__new__(StubProvider)
    node.values = {}
    with pytest.raises(StartupVerified):
        node.__init__()


def test_launch_resolves_provider_and_executor_with_identical_typed_parameters(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Test must never start a ROS node or process')
    monkeypatch.setattr(Node, 'execute', forbidden)
    monkeypatch.setattr(ExecuteProcess, 'execute', forbidden)
    monkeypatch.setattr(launch_a, 'get_package_share_directory',
                        lambda _: str(ROOT/'behavior_ws/src/nrs_imitation'))
    ctx = context()
    nodes = {}
    allowed = (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction,
        PopEnvironment, PopLaunchConfigurations, PushEnvironment, PushLaunchConfigurations,
        RegisterEventHandler, SetEnvironmentVariable, SetLaunchConfiguration, UnsetLaunchConfiguration)

    def inspect(entities):
        for action in entities:
            if isinstance(action, LaunchDescription):
                inspect(action.entities)
                continue
            if action.condition and not action.condition.evaluate(ctx):
                continue
            if isinstance(action, Node):
                executable = perform_substitutions(ctx,
                    normalize_to_list_of_substitutions(action.node_executable))
                nodes[executable] = evaluate_parameters(ctx, action._Node__parameters)[0]
                continue
            if isinstance(action, LogInfo):
                continue
            assert isinstance(action, allowed), type(action)
            inspect(action.execute(ctx) or [])

    inspect(launch_a.configure(ctx))
    provider, executor = nodes['inference_single_cam'], nodes['e2_executor']
    for params in (provider, executor):
        assert params['e2_config'] == ''
        assert params['e2_direct_a'] is True
        assert params['checkpoint'] == DEFAULT_CHECKPOINT
        for key, default in PARAMETERS.items():
            assert Parameter(key, value=params[key]).type_ == Parameter(key, value=default).type_
        assert list(params['measured_force_abs_limits_N']) == [200.0, 200.0, 200.0]
    assert provider['use_force_observation'] is False
    assert provider['use_force_history'] is False
    assert provider['use_stain_mask'] is False
    pc = build_runtime({k: provider[k] for k in PARAMETERS})
    ec = build_runtime({k: executor[k] for k in PARAMETERS})
    assert object_hash(pc) == object_hash(ec)
    assert provider['ckpt_dir'] == str(Path(DEFAULT_CHECKPOINT).parent)
    # The 9-D transport padding is separate from the actual six learned outputs.
    engine = TimedExecution(executor_settings(pc, 'il'))
    pose = np.array(pc['common']['demo_start_pose6'])
    action = np.tile(np.r_[pose, 0., 0., 0.], (128, 1))
    engine.accept(TimedPlan(1, 10., np.arange(128)/30., action,
                           np.full(128, 'policy'), False), 10.)
    engine.start(10., pose)
    before = engine.tick(10., pose, np.array([0., 0., 7.]), 0., 0., external_force_fz=0.)
    working = engine.tick(10.008, pose, np.array([0., 0., 7.]), 0., 0., external_force_fz=23.)
    assert before['requested'][8] == 0.
    assert working['requested'][8] == 23.
    assert working['sent'][8] <= 30.*0.008+1e-8


def test_wrong_checkpoint_and_nonfinite_force_rejected():
    with pytest.raises(ValueError, match='checkpoint file missing'):
        build_runtime(dict(checkpoint='/tmp/no-such-A-checkpoint.ckpt'))
    with pytest.raises(ValueError, match='constant_force_N'):
        build_runtime(dict(checkpoint=DEFAULT_CHECKPOINT, constant_force_N=float('nan')))
    with pytest.raises(ValueError, match='positive finite limits'):
        build_runtime(dict(checkpoint=DEFAULT_CHECKPOINT,
                           measured_force_abs_limits_N=[0., 200., 200.]))


def test_offline_launch_is_only_an_explicit_checkpoint_process():
    actions = launch_a.configure(context('offline'))
    assert len(actions) == 1 and isinstance(actions[0], ExecuteProcess)


@pytest.mark.parametrize('axis', range(6))
@pytest.mark.parametrize('sign', [-1., 1.])
def test_direct_executor_selects_external_force_and_stops_on_each_measured_limit(monkeypatch, axis, sign):
    from nrs_imitation.e2_protection import FeedbackProtectionMonitor
    import test_e2_timed_execution as mock_runtime
    config = build_runtime(dict(checkpoint=DEFAULT_CHECKPOINT))
    monkeypatch.setattr(mock_runtime, 'config', lambda: config)
    node, clock, future = mock_runtime.mock_bridge(monkeypatch)
    node.direct_a = True
    node.config = config
    node.method = 'il'
    node.a_processing_force = 23.
    node.force = np.array([0., 0., 7., 0., 0., 0.])
    node.sample_last = {}
    node.recorder.emit = lambda *args: None
    node.last_console_status = None
    node.protection_wait_reason = None
    node.protection = FeedbackProtectionMonitor(config['protection'], node.settings)
    node.get_clock = lambda: NS(now=lambda: NS(nanoseconds=int(clock[0]*1e9)))
    node.get_logger = lambda: NS(info=lambda _: None, error=lambda _: None)
    node.state = 'waiting_feedback'
    assert node.protection_check(clock[0])
    assert not node.processing and not node.sent
    node.engine = TimedExecution(node.settings)
    node.engine.accept(mock_runtime.plan(), clock[0])
    node.state = 'arming'
    node.arm_at = clock[0]
    node.tick()
    assert node.state == 'arming' and not node.processing  # need a new Force ack
    clock[0] += .008
    node.pose_at = node.force_at = node.mode_at = clock[0]
    node.tick()
    assert node.state == 'running' and node.processing
    starts = [event for event in node.events if event['event'] == 'processing_start']
    assert len(starts) == 1 and starts[0]['source'] == 'execution_start'
    assert starts[0]['automatic'] is True and starts[0]['physical_processing_verified'] is False
    assert not node.operator_event('processing_start', NS()).success
    clock[0] += .008
    node.tick()
    assert 0. < node.sent[-1][8] <= .24 + 1e-8  # common slew, no A-only ramp
    clock[0] += .008
    node.pose_at = node.force_at = node.mode_at = clock[0]
    node.tick()
    assert 0. < node.sent[-1][8] <= .48 + 1e-8
    assert node.force_selection_status()['external_force_fz_N'] == 23.
    assert sum(event['event'] == 'processing_start' for event in node.events) == 1
    first = next(event for event in node.events if event['event'] == 'first_command_sent')
    assert first['processing_started'] is True
    phase = next(event for event in node.events if event['event'] == 'provider_phase')
    assert phase['processing_interval'] == 'automatic_execution_start'
    # Exercise the actual currentF callback, with no provenance/raw packet.
    force = node.force.copy()
    force[axis] = sign * 200.
    clock[0] += .008
    sent = len(node.sent)
    node.on_force(NS(data=force.tolist()))
    node.tick()
    assert len(node.sent) == sent
    assert node.state == 'stopping' and node.stop_reason == 'safety_stop'
    assert node.modes[-1] == 'Idling'
    assert node.requests[-1].command_mode == 'PTP9D_STREAM_STOP'


def test_direct_executor_subscribes_currentF_without_source_producers(monkeypatch):
    from nrs_imitation import e2_executor_node as bridge
    monkeypatch.setattr(bridge.Node, '__init__', lambda *args, **kwargs: None)
    monkeypatch.setattr(bridge, 'ExecutionRecorder',
        lambda *args, **kwargs: NS(path=None, event=lambda *a, **k: None))
    overrides = dict(e2_direct_a=True, checkpoint=DEFAULT_CHECKPOINT,
                     e2_session_id='test_currentF_subscriptions')
    class StubExecutor(bridge.E2Executor):
        def declare_parameter(self, name, default):
            self.values[name] = overrides.get(name, default)
        def get_parameter(self, name):
            return NS(value=self.values[name])
        def get_logger(self):
            return NS(info=lambda _: None, warn=lambda _: None)
        def get_clock(self):
            return NS(now=lambda: NS(nanoseconds=10_000_000_000))
        def create_subscription(self, kind, topic, callback, qos):
            self.recorded_subscriptions[topic] = callback
        def create_publisher(self, *args):
            return NS(publish=lambda _: None)
        def create_client(self, *args):
            return NS()
        def create_service(self, *args):
            return NS()
        def create_timer(self, *args):
            return NS()
    node = StubExecutor.__new__(StubExecutor)
    node.values, node.recorded_subscriptions = {}, {}
    node.__init__()
    try:
        assert node.recorded_subscriptions['/ur10skku/currentF'] == node.on_force
        assert '/ur10skku/currentF_provenance' not in node.recorded_subscriptions
        assert '/ur10skku/ft_acquisition' not in node.recorded_subscriptions
        assert not node.protection.requires_source
    finally:
        node.lock_file.close()


@pytest.mark.parametrize('bad_force', [[0., 0., 7.], [0., 0., float('nan'), 0., 0., 0.]])
def test_currentF_invalid_six_axis_feedback_stops(monkeypatch, bad_force):
    node, clock, _ = currentF_bridge(monkeypatch)
    node.on_force(NS(data=bad_force))
    node.tick()
    assert not node.sent and node.state == 'stopping'
    assert node.stop_reason == 'safety_stop'


def currentF_bridge(monkeypatch):
    from nrs_imitation.e2_protection import FeedbackProtectionMonitor
    import test_e2_timed_execution as mock_runtime
    config = build_runtime(dict(checkpoint=DEFAULT_CHECKPOINT))
    monkeypatch.setattr(mock_runtime, 'config', lambda: config)
    node, clock, future = mock_runtime.mock_bridge(monkeypatch)
    node.force = np.array([0., 0., 7., 0., 0., 0.])
    node.protection = FeedbackProtectionMonitor(config['protection'], node.settings)
    node.protection_wait_reason = None
    node.sample_last = {'force': clock[0]}
    node.recorder.emit = lambda *args: None
    node.get_clock = lambda: NS(now=lambda: NS(nanoseconds=int(clock[0]*1e9)))
    assert node.protection_check(clock[0])
    return node, clock, future


def test_currentF_stops_on_receipt_timeout_despite_fresh_pose_and_mode(monkeypatch):
    node, clock, _ = currentF_bridge(monkeypatch)
    clock[0] += .201
    node.pose_at = node.mode_at = clock[0]
    node.tick()
    assert not node.sent and node.state == 'stopping'
    assert node.stop_reason == 'safety_stop'
    assert node.modes[-1] == 'Idling'


def test_currentF_workspace_anchor_covers_alignment_and_execution(monkeypatch):
    node, clock, _ = currentF_bridge(monkeypatch)
    node.pose[0] = 141.
    node.tick()
    assert not node.sent and node.state == 'stopping'
    assert node.protection.fault == 'actual TCP outside common workspace from startup anchor'


@pytest.mark.parametrize('duration', [-1., float('nan'), float('inf'), True])
def test_invalid_force_ramp_duration_rejected(duration):
    with pytest.raises(ValueError, match='force_ramp_sec'):
        build_runtime(dict(checkpoint=DEFAULT_CHECKPOINT, force_ramp_sec=duration))


def test_force_ramp_starts_at_contact_and_survives_replan_and_contact_loss():
    config = build_runtime(dict(checkpoint=DEFAULT_CHECKPOINT))
    engine = TimedExecution(executor_settings(config, 'il'))
    action = np.zeros((128, 9))
    times = np.arange(128)/30.
    engine.accept(TimedPlan(1, 10., times, action, np.full(128, 'policy'), False), 10.)
    engine.start(10., np.zeros(6))
    first_contact = 11.2
    previous_sent = 0.
    starts = 0
    for step in range(551):
        now = 10. + step*.008
        if step == 250:
            engine.accept(TimedPlan(2, now, times, action, np.full(128, 'policy'), False), now)
        contact = now >= first_contact and not 11.9 <= now < 12.0
        result = engine.tick(now, np.zeros(6), np.array([0., 0., 7. if contact else 0.]),
            0., 0., external_force_fz=23., external_force_ramp_sec=3.)
        expected = 23.*np.clip((now-first_contact)/3., 0., 1.)
        assert result['requested'][8] == pytest.approx(expected)
        assert abs(result['sent'][8]-previous_sent) <= 30.*.008 + 1e-8
        if not contact:
            assert result['gated'][8] == 0.
        starts += result['force_ramp_started_now']
        previous_sent = result['sent'][8]
    assert starts == 1
    assert engine.force_ramp_start == pytest.approx(first_contact)
    assert result['requested'][8] == 23.


def test_force_ramp_cannot_finish_during_a_long_approach():
    config = build_runtime(dict(checkpoint=DEFAULT_CHECKPOINT))
    engine = TimedExecution(executor_settings(config, 'il'))
    action = np.zeros((256, 9))
    engine.accept(TimedPlan(1, 10., np.arange(256)/30., action,
                           np.full(256, 'policy'), False), 10.)
    engine.start(10., np.zeros(6))
    for step in range(751):
        now = 10. + step*.008
        result = engine.tick(now, np.zeros(6), np.zeros(3), 0., 0.,
            external_force_fz=23., external_force_ramp_sec=3.)
        assert result['requested'][8] == result['sent'][8] == 0.
    result = engine.tick(16.008, np.zeros(6), np.array([0., 0., 7.]), 0., 0.,
        external_force_fz=23., external_force_ramp_sec=3.)
    assert result['force_ramp_started_now'] and result['requested'][8] == 0.


def test_legacy_executor_still_requires_explicit_processing_start(monkeypatch):
    import test_e2_timed_execution as mock_runtime
    config = build_runtime(dict(checkpoint=DEFAULT_CHECKPOINT))
    monkeypatch.setattr(mock_runtime, 'config', lambda: config)
    node, clock, _ = mock_runtime.mock_bridge(monkeypatch)
    node.engine = TimedExecution(node.settings)
    node.engine.accept(mock_runtime.plan(), clock[0])
    node.state = 'arming'
    node.arm_at = clock[0] - .008
    node.tick()
    assert node.state == 'running' and not node.processing
    assert not any(event['event'] == 'processing_start' for event in node.events)


def test_automatic_start_does_not_restart_after_processing_end(monkeypatch):
    import test_e2_timed_execution as mock_runtime
    config = build_runtime(dict(checkpoint=DEFAULT_CHECKPOINT))
    monkeypatch.setattr(mock_runtime, 'config', lambda: config)
    node, clock, _ = mock_runtime.mock_bridge(monkeypatch)
    node.direct_a = True
    node.config = config
    node.a_processing_force = 23.
    node.protection = None
    node.protection_wait_reason = None
    node.last_console_status = None
    node.get_logger = lambda: NS(info=lambda _: None)
    node.processing = True
    assert node.operator_event('processing_end', NS()).success
    node.tick()
    assert not node.processing and node.force_selection_status()['external_force_fz_N'] == 0.
    assert not node.operator_event('processing_start', NS()).success
    assert not any(event['event'] == 'processing_start' for event in node.events)
