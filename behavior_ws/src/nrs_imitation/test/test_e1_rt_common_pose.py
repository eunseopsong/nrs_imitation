"""E1 R/T stabilization parity with archived C; no ROS or hardware execution."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nrs_imitation.e2_timed_execution import (
    TimedExecution, TimedPlan, ExecutionFault, executor_settings, validate_settings,
)
from nrs_imitation.e2_providers import rule_actions, TimedActions

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / 'experiments/e1_rt_common_20260927/config.json'


def config():
    return json.loads(CONFIG.read_text())


def source_plan():
    t = np.arange(0., 13.001, .008)
    a = np.zeros((len(t), 9))
    a[:, 0] = 10*np.sin(t*.8)
    a[:, 1] = 3*np.cos(t*5)
    a[:, 5] = np.deg2rad(np.where(np.sin(t) >= 0, 179., -179.))
    a[:, 8] = 12*np.sin(t*3)
    return TimedPlan(1, 10., t, a, np.full(len(t), 'work'), True)


@pytest.mark.parametrize('method', ['rule', 'replay'])
def test_finite_matches_c_pose_chunks_and_preserves_original_force_clock(method):
    c = config(); p = source_plan(); original = p.action.copy()
    e = TimedExecution(executor_settings(c, method))
    old_cfg = copy.deepcopy(c); old_cfg.pop('rtc_pose_conditioning')
    unconditioned = TimedExecution(executor_settings(old_cfg, method))
    reference = TimedExecution(executor_settings(c, 'il'))
    initial = np.array([0., 0., 0., 0., 0., np.pi])
    for engine in [e, unconditioned]:
        engine.accept(p, 10.); engine.start(10., initial)

    def c_chunk(index):
        times = np.arange(128)/30.
        values = [p.sample(min(t+4*index, p.time[-1])) for t in times]
        return TimedPlan(index+1, 10.+4*index, times,
            np.array([v[0] for v in values]), np.full(128, 'work'), False)

    reference.accept(c_chunk(0), 10.); reference.start(10., initial)
    previous = e.command.copy(); velocity = np.zeros(6); chunk = 0
    for i in range(1, len(p.time)):
        now = 10.+i*.008
        due = int((now-10.)/4.)
        if due > chunk:
            chunk = due; reference.accept(c_chunk(chunk), 10.+4*chunk)
        f = [0., 0., 6. if i % 53 < 30 else .5]
        result = e.tick(now, previous[:6], f, 0., 0.)
        before = unconditioned.tick(now, previous[:6], f, 0., 0.)
        expected = reference.tick(now, previous[:6], f, 0., 0.)
        for stage in ['conditioned', 'gated', 'sent']:
            np.testing.assert_allclose(result[stage][:6], expected[stage][:6], atol=2e-8, rtol=0)
        np.testing.assert_array_equal(result['requested'], before['requested'])
        for stage in ['requested', 'gated', 'sent']:
            np.testing.assert_array_equal(result[stage][6:], before[stage][6:])
        assert result['contact'] == before['contact']
        assert result['elapsed_s'] == before['elapsed_s']
        delta = np.r_[result['sent'][:3]-previous[:3],
            (Rotation.from_rotvec(result['sent'][3:6])*Rotation.from_rotvec(previous[3:6]).inv()).as_rotvec()]
        dt = min(result['dt_s'], .008)
        current = delta/dt
        assert np.max(np.abs(current[:3]-velocity[:3])/dt) <= 25.+1e-6
        assert np.max(np.abs(current[3:]-velocity[3:])/dt) <= np.deg2rad(100.)+1e-6
        previous, velocity = result['sent'], current
    assert e.tick(10.+p.time[-1]+.016, previous[:6], f, 0., 0.)['end'] == 'trajectory_end'
    np.testing.assert_array_equal(p.action, original)
    np.testing.assert_array_equal(e.plan.time, p.time)
    with pytest.raises(ExecutionFault, match='cannot be replaced'):
        e.accept(TimedPlan(2, 23., p.time, p.action, p.phase, True), 23.)


@pytest.mark.parametrize('method', ['rule', 'replay'])
def test_actual_selected_r_t_plans_finish_on_original_clock(method):
    c = config()
    actions = rule_actions(c['task'], c['recipe']) if method == 'rule' else TimedActions.load(c['replay']['template'])
    values = actions.action.astype(float); values[:, :2] += [400., 400.]
    p = TimedPlan(1, 10., actions.time, values, actions.phase, True)
    e = TimedExecution(executor_settings(c, method)); e.accept(p, 10.); e.start(10., values[0, :6])
    times = np.r_[np.arange(.008, p.time[-1], .008), p.time[-1]]
    for t in times:
        result = e.tick(10.+t, e.command[:6], [0., 0., 6.], 0., 0.)
        assert 'end' not in result
    # Kinematic command test only: no claim about real measured TCP tracking.
    if method == 'rule':
        assert np.linalg.norm(result['sent'][:3]-values[-1, :3]) < c['executor']['completion_position_tolerance_mm']
    # T's recorded source returns faster than the common 10 mm/s cap near
    # the end. Original T also stopped short. Keep its clock; do not silently
    # extend it until the endpoint is reached or relabel that as success.
    assert e.tick(10.+p.time[-1]+.008, e.command[:6], [0., 0., 6.], 0., 0.)['end'] == 'trajectory_end'


@pytest.mark.parametrize('key,value', [('pose_sampling_hz', 125.), ('horizon_points', 64),
                                     ('stride_points', 128), ('profile', {})])
def test_mismatched_c_filter_contract_is_rejected(key, value):
    c = config(); c['rtc_pose_conditioning'][key] = value
    assert any('rtc_pose_conditioning' in message for message in validate_settings(c))


@pytest.mark.parametrize('method', ['R', 'T', 'C'])
def test_shared_launch_profile_is_resolved_without_starting_nodes(method, monkeypatch, tmp_path):
    from test_rtc_launch import rtc, context
    from launch.actions import LogInfo
    c = config(); ctx = context(method)
    ctx.launch_configurations.update(config=str(CONFIG), checkpoint=c['il']['checkpoint'], episode=c['replay']['episode_id'])
    monkeypatch.setattr(rtc, 'ROOT', tmp_path)
    actions = rtc.configure(ctx)
    assert all(isinstance(a, LogInfo) for a in actions)
    assert list(tmp_path.iterdir()) == []
    for original in c['rtc_pose_conditioning']['reference_C_runs']:
        archived = json.loads(Path(original['config']).read_text())
        assert executor_settings(c, 'il') == executor_settings(archived, 'il')
