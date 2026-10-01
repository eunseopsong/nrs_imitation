"""No hardware: matched launch, policy startup, force selection and protection."""
import importlib.util
import json
import sys
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

from nrs_imitation.e2_direct_abc import ROOT, CHECKPOINTS, PARAMETERS, build_runtime, object_hash
from nrs_imitation.e2_timed_execution import TimedExecution, TimedPlan, executor_settings
from nrs_imitation.e2_protection import FeedbackProtectionMonitor


def load_launch(name):
    spec = importlib.util.spec_from_file_location(name.replace('.', '_'),
        ROOT/'behavior_ws/src/nrs_imitation/launch'/name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch_common = load_launch('e2_abc_common.launch.py')


def context(condition='A', mode='check', **values):
    ctx = LaunchContext()
    for action in launch_common.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(ctx)
    ctx.launch_configurations.update(condition=condition, mode=mode, session='matched_test', **values)
    return ctx


def inspect_launch(actions, ctx, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Offline test must never start a ROS node or process')
    monkeypatch.setattr(Node, 'execute', forbidden)
    monkeypatch.setattr(ExecuteProcess, 'execute', forbidden)
    nodes = {}
    allowed = (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction,
        PopEnvironment, PopLaunchConfigurations, PushEnvironment, PushLaunchConfigurations,
        RegisterEventHandler, SetEnvironmentVariable, SetLaunchConfiguration, UnsetLaunchConfiguration)

    def walk(entities):
        for action in entities:
            if isinstance(action, LaunchDescription):
                walk(action.entities)
                continue
            if action.condition and not action.condition.evaluate(ctx):
                continue
            if isinstance(action, Node):
                executable = perform_substitutions(ctx, normalize_to_list_of_substitutions(action.node_executable))
                nodes[executable] = evaluate_parameters(ctx, action._Node__parameters)[0]
            elif not isinstance(action, LogInfo):
                assert isinstance(action, allowed), type(action)
                walk(action.execute(ctx) or [])
    walk(actions)
    return nodes


def test_runtime_hash_covers_identical_execution_and_safety_for_all_conditions():
    configs = {c: build_runtime(dict(e2_condition=c)) for c in 'ABC'}
    assert len({cfg['comparison']['common_settings_sha256'] for cfg in configs.values()}) == 1
    for c, cfg in configs.items():
        assert cfg['il']['checkpoint'] == CHECKPOINTS[c]
        assert cfg['force_ramp_sec'] == 0.
        assert cfg['external_force_N'] == (23. if c == 'A' else None)
        assert cfg['il']['use_force_observation'] == (c == 'C')
        assert cfg['il']['force_action'] == (c != 'A')
    changed = build_runtime(dict(measured_force_abs_limits_N=[100., 100., 100.]))
    assert changed['comparison']['common_settings_sha256'] != configs['A']['comparison']['common_settings_sha256']


@pytest.mark.parametrize('condition', list('ABC'))
def test_check_is_readonly(condition, tmp_path, monkeypatch):
    monkeypatch.setattr(launch_common, 'ROOT', tmp_path)
    actions = launch_common.configure(context(condition))
    assert all(isinstance(a, LogInfo) for a in actions)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('condition', list('ABC'))
def test_real_launch_has_matching_provider_executor_parameters(condition, tmp_path, monkeypatch):
    monkeypatch.setattr(launch_common, 'ROOT', tmp_path)
    monkeypatch.setattr(launch_common, 'get_package_share_directory',
                        lambda _: str(ROOT/'behavior_ws/src/nrs_imitation'))
    ctx = context(condition, mode='run', repeat='5')
    nodes = inspect_launch(launch_common.configure(ctx), ctx, monkeypatch)
    provider, executor = nodes['inference_single_cam'], nodes['e2_executor']
    pc, ec = [build_runtime({k: params[k] for k in PARAMETERS}) for params in (provider, executor)]
    assert object_hash(pc) == object_hash(ec)
    assert provider['use_force_observation'] == (condition == 'C')
    assert provider['use_force_history'] == (condition != 'A')
    assert provider['e2_direct_abc'] and not provider['e2_direct_a']
    assert provider['metrics_repeat_id'] == '5'
    assert provider['metrics_extra_telemetry_enable'] is False
    assert provider['metrics_sample_hz'] == 20.
    assert executor['force_ramp_sec'] == 0.
    assert executor['e2_trial_index'] == 5
    saved = json.loads((Path(pc['run']['directory'])/'runtime.json').read_text())
    assert object_hash(saved) == object_hash(pc)


def test_repeated_attempts_are_preserved_and_cohort_drift_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(launch_common, 'ROOT', tmp_path)
    for _ in range(2):
        launch_common.configure(context('A', mode='run'))
    saved = list(tmp_path.rglob('runtime.json'))
    assert len(saved) == 2 and saved[0].parent != saved[1].parent
    with pytest.raises(ValueError, match='Common E2 settings changed'):
        launch_common.configure(context('B', mode='run', constant_force_N='24.0'))
    assert len(list(tmp_path.rglob('runtime.json'))) == 2
    manifest_path = next(tmp_path.rglob('comparison.json'))
    manifest = json.loads(manifest_path.read_text())
    manifest['model_artifacts']['B']['checkpoint_sha256'] = 'changed'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='checkpoint/normalizer changed'):
        launch_common.configure(context('B', mode='run'))


@pytest.mark.parametrize('condition', list('ABC'))
def test_public_entry_point_expands_to_the_matched_runtime(condition, tmp_path, monkeypatch):
    from nrs_imitation import e2_direct_abc
    # The included launch is loaded as a fresh module, so set its imported ROOT
    # before expansion. Checkpoint paths remain pinned to the real training files.
    monkeypatch.setattr(e2_direct_abc, 'ROOT', tmp_path)
    public = load_launch('e2_abc.launch.py')
    monkeypatch.setattr(public, 'get_package_share_directory',
                        lambda _: str(ROOT/'behavior_ws/src/nrs_imitation'))
    ctx = LaunchContext()
    ctx.launch_configurations.update(condition=condition, mode='run', session='public_test')
    for action in public.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(ctx)
    nodes = inspect_launch(public.configure(ctx), ctx, monkeypatch)
    for executable in ('inference_single_cam', 'e2_executor'):
        assert nodes[executable]['e2_direct_abc'] is True
        assert nodes[executable]['e2_condition'] == condition
        assert nodes[executable]['e2_trial_index'] == 1
    assert len(list(tmp_path.rglob('runtime.json'))) == 1


def test_C_five_B_five_A_five_are_numbered_automatically(tmp_path, monkeypatch):
    monkeypatch.setattr(launch_common, 'ROOT', tmp_path)
    indices = []
    for condition in 'CBA':
        for repeat in range(1, 6):
            launch_common.configure(context(condition, mode='run'))
            records = [json.loads(path.read_text()) for path in tmp_path.rglob('runtime.json')]
            matching = [record for record in records if record['condition'] == condition]
            assert sorted(record['run']['repeat_index'] for record in matching) == list(range(1, repeat+1))
            indices.append((condition, repeat))
        with pytest.raises(ValueError, match=f'All five planned {condition} attempts'):
            launch_common.configure(context(condition, mode='run'))
    assert len(list(tmp_path.rglob('runtime.json'))) == 15
    manifest = json.loads(next(tmp_path.rglob('comparison.json')).read_text())
    assert manifest['planned_order'] == [['C']*5, ['B']*5, ['A']*5]
    assert len({record['comparison']['common_settings_sha256'] for record in records}) == 1


def test_checks_and_failed_preflight_do_not_consume_auto_indices(tmp_path, monkeypatch):
    monkeypatch.setattr(launch_common, 'ROOT', tmp_path)
    launch_common.configure(context('C', mode='run'))
    before = list(tmp_path.rglob('runtime.json'))
    # A recorded abort remains an attempt; status is never used to hide it.
    (before[0].parent/'outcome.json').write_text(json.dumps(dict(termination='manual_abort')))
    launch_common.configure(context('C'))
    with pytest.raises(ValueError, match='Common E2 settings changed'):
        launch_common.configure(context('C', mode='run', constant_force_N='24.0'))
    assert list(tmp_path.rglob('runtime.json')) == before
    launch_common.configure(context('C', mode='run'))
    assert sorted(json.loads(path.read_text())['run']['repeat_index']
                  for path in tmp_path.rglob('runtime.json')) == [1, 2]


@pytest.mark.parametrize('values', [dict(force_ramp_sec=3.), dict(e2_trial_index=0),
    dict(e2_trial_index=6), dict(e2_trial_index=True), dict(e2_condition='D'),
    dict(checkpoint=CHECKPOINTS['B'], e2_condition='A')])
def test_incomparable_overrides_rejected(values):
    with pytest.raises(ValueError):
        build_runtime(values)


@pytest.mark.parametrize('condition', list('ABC'))
def test_real_provider_startup_matches_native_checkpoint_schema(condition, monkeypatch):
    from nrs_imitation import inference_core as core
    monkeypatch.setattr(core.Node, '__init__', lambda *args, **kwargs: None)
    monkeypatch.setattr(core, 'StainOriginClient', lambda *args, **kwargs: NS())
    overrides = dict(e2_direct_abc=True, e2_condition=condition, checkpoint=CHECKPOINTS[condition],
        ckpt_dir=str(Path(CHECKPOINTS[condition]).parent), e2_session_id='startup_'+condition,
        use_force_observation=condition == 'C', use_force_history=condition != 'A',
        use_stain_mask=False, auto_move_to_demo_start=True, chunk_size=128,
        flow_replan_interval_steps=120, flow_deterministic_noise=True,
        flow_local_anchor_enable=False, force_history_len=30, force_xy_cmd_enable=False,
        fz_hard_limit=0., gradcam_enable=False, gradcam_save=False)
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
            assert self._e2_transport and self.motion_only == (condition == 'A')
            assert self.action_dim == (6 if condition == 'A' else 9)
            assert self.stats.qpos_a.shape == (self.action_dim,)
            assert self._e2_context['condition'] == condition
            raise StartupVerified()
    node = StubProvider.__new__(StubProvider)
    node.values = {}
    with pytest.raises(StartupVerified):
        node.__init__()


def running_bridge(condition, monkeypatch):
    import test_e2_timed_execution as mock_runtime
    cfg = build_runtime(dict(e2_condition=condition))
    monkeypatch.setattr(mock_runtime, 'config', lambda: cfg)
    node, clock, future = mock_runtime.mock_bridge(monkeypatch)
    node.direct_abc = True
    node.direct_a = False
    node.config = cfg
    node.method = 'il'
    node.a_processing_force = cfg['external_force_N']
    node.protection = FeedbackProtectionMonitor(cfg['protection'], node.settings)
    node.protection_wait_reason = None
    node.force = np.array([0., 0., 7., 0., 0., 0.])
    node.sample_last = {}
    node.recorder.emit = lambda *args: None
    node.last_console_status = None
    node.get_clock = lambda: NS(now=lambda: NS(nanoseconds=int(clock[0]*1e9)))
    node.get_logger = lambda: NS(info=lambda _: None, error=lambda _: None)
    node.engine = TimedExecution(node.settings)
    actions = np.tile(np.r_[np.zeros(6), 0., 0., 9.], (128, 1))
    node.engine.accept(TimedPlan(1, clock[0], np.arange(128)/30., actions, np.full(128, 'policy'), False), clock[0])
    node.state = 'arming'
    node.arm_at = clock[0] - .008
    node.tick()
    assert node.state == 'running' and node.processing
    assert sum(e['event'] == 'processing_start' for e in node.events) == 1
    assert not node.operator_event('processing_start', NS()).success
    assert not node.operator_event('processing_end', NS()).success
    return node, clock, future


@pytest.mark.parametrize('condition', list('ABC'))
def test_external_force_only_A_and_common_slew_from_first_tick(condition, monkeypatch):
    node, clock, _ = running_bridge(condition, monkeypatch)
    assert node.uses_external_force() == (condition == 'A')
    for _ in range(8):
        clock[0] += .008
        node.pose_at = node.force_at = node.mode_at = clock[0]
        node.tick()
    assert node.sent[0][8] == pytest.approx(.24)
    assert node.sent[-1][8] == pytest.approx(8*.24)
    assert not any(e['event'] == 'force_ramp_started' for e in node.events)
    assert node.force_selection_status()['external_force_fz_N'] == (23. if condition == 'A' else None)


@pytest.mark.parametrize('condition', list('ABC'))
@pytest.mark.parametrize('axis', range(6))
@pytest.mark.parametrize('sign', [-1., 1.])
def test_same_measured_wrench_stop_for_every_condition(condition, axis, sign, monkeypatch):
    node, clock, _ = running_bridge(condition, monkeypatch)
    wrench = node.force.copy()
    wrench[axis] = sign*200.
    clock[0] += .008
    node.on_force(NS(data=wrench.tolist()))
    node.tick()
    assert not node.sent and node.state == 'stopping'
    assert node.stop_reason == 'safety_stop'
    assert node.modes[-1] == 'Idling'
    assert node.requests[-1].command_mode == 'PTP9D_STREAM_STOP'


@pytest.mark.parametrize('condition', list('ABC'))
def test_timed_commands_match_archived_BC_without_an_extra_ramp(condition):
    archive_dir = ROOT/'results/20260926/E2/E2_BC_01/B/E2_B_20260926T205903_fc196a96a836'
    source = next((archive_dir/'executor').glob('*/artifacts/1_e2_timed_execution.py'))
    spec = importlib.util.spec_from_file_location('archived_timed_execution', source)
    archive = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = archive
    spec.loader.exec_module(archive)
    archived_cfg = json.loads((archive_dir/'config.json').read_text())
    current_cfg = build_runtime(dict(e2_condition=condition))
    current = TimedExecution(executor_settings(current_cfg, 'il'))
    previous = archive.TimedExecution(archive.executor_settings(archived_cfg, 'il'))
    times = np.arange(128)/30.
    actions = np.zeros((128, 9))
    actions[:, 0] = 10.*np.sin(times)
    actions[:, 2] = -8.*np.sin(times/2.)
    actions[:, 5] = .1*np.sin(times)
    actions[:, 8] = 23. if condition == 'A' else 9.+8.*np.sin(times)
    feedback = np.zeros(6)
    phase = np.full(128, 'policy')
    for engine, plan_type in ((current, TimedPlan), (previous, archive.TimedPlan)):
        engine.accept(plan_type(1, 10., times, actions, phase, False), 10.)
        engine.start(10., feedback)
    for step in range(600):
        now = 10.+step*.008
        if step in (150, 350):
            replanned = actions.copy()
            replanned[:, 0] -= 7.
            for engine, plan_type in ((current, TimedPlan), (previous, archive.TimedPlan)):
                engine.accept(plan_type(step, now, times, replanned, phase, False), now)
        force = np.array([0., 0., 0. if 100 <= step < 140 or 270 <= step < 300 else 7.])
        old = previous.tick(now, feedback, force, 0., 0.)
        result = current.tick(now, feedback, force, 0., 0.,
            external_force_fz=23. if condition == 'A' else None, external_force_ramp_sec=0.)
        for key in ('requested', 'conditioned', 'gated', 'sent'):
            np.testing.assert_allclose(result[key], old[key], atol=1e-12, rtol=0.)
        assert result['contact'] == old['contact']
        assert result['force_ramp_started_now'] is False
        feedback = result['sent'][:6]
