"""C motion regression tests; no ROS context or hardware calls."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nrs_imitation.e2_timed_execution import (
    CPoseConditioner, ExecutionFault, TimedExecution, TimedPlan,
    executor_settings, validate_settings,
)

ROOT = Path(__file__).resolve().parents[4]


def config():
    return json.loads((ROOT/'experiments/e2_rule_replay_20260920/config.json').read_text())


def plan(plan_id=1, generated_at=10., shift=0.):
    t = np.arange(128)/30.
    a = np.zeros((128, 9))
    a[:, 0] = 5.*t + 4.*np.cos(np.arange(128)*np.pi) + shift
    a[:, 1] = -3.*t + shift
    a[:, 5] = np.deg2rad(np.where(np.arange(128) % 2, -179., 179.))
    a[:, 8] = 5.*np.sin(t*3.)
    return TimedPlan(plan_id, generated_at, t, a, np.full(128, 'unknown'), False)


def engine(p=None):
    p = plan() if p is None else p
    e = TimedExecution(executor_settings(config(), 'il'))
    e.accept(p, 10.)
    e.start(10., np.array([0., 0., 0., 0., 0., np.pi]))
    return e


def test_profile_is_only_injected_for_c_and_config_is_not_mutated():
    c = config(); before = copy.deepcopy(c)
    for method in [None, 'rule', 'replay']:
        settings = executor_settings(c, method)
        assert 'c_pose_conditioning' not in settings
        assert TimedExecution(settings).conditioner is None
    assert TimedExecution(executor_settings(c, 'il')).conditioner is not None
    assert c == before


def test_smoothing_does_not_change_input_times_forces_or_quaternion_branch():
    p = plan(); original = p.action.copy(); times = p.time.copy()
    conditioner = CPoseConditioner(config()['il']['pose_conditioning'])
    conditioner.accept(p, 10.)
    smoothed = conditioner.plan
    np.testing.assert_array_equal(p.action, original)
    np.testing.assert_array_equal(smoothed.time, times)
    np.testing.assert_array_equal(smoothed.action[:, 6:], original[:, 6:])
    np.testing.assert_array_equal(smoothed.phase, p.phase)
    assert smoothed.generated_at == p.generated_at
    assert np.max(np.abs(np.diff(smoothed.action[:, 0]))) < 1.
    angles = (Rotation.from_rotvec(smoothed.action[:, 3:6])*
              Rotation.from_rotvec([0., 0., np.pi]).inv()).magnitude()
    assert np.max(angles) < np.deg2rad(1.01)


def test_start_and_replacement_have_continuous_position_and_velocity():
    e = engine(); conditioner = e.conditioner
    np.testing.assert_allclose(conditioner.sample(10.), e.start_pose)
    for now in [10.2, 10.3, 12., 14.]:
        h = 1e-6
        before = conditioner.sample(now)
        velocity_before = (before[:3]-conditioner.sample(now-h)[:3])/h
        e.accept(plan(e.last_plan_id+1, now, 30.), now)
        after = conditioner.sample(now)
        velocity_after = (conditioner.sample(now+h)[:3]-after[:3])/h
        np.testing.assert_allclose(after[:3], before[:3], atol=1e-10)
        assert (Rotation.from_rotvec(after[3:])*Rotation.from_rotvec(before[3:]).inv()).magnitude() < 1e-10
        np.testing.assert_allclose(velocity_after, velocity_before, atol=.02)


def test_command_acceleration_and_speed_remain_bounded_across_replan():
    e = engine(); previous = e.command.copy(); velocity = np.zeros(6)
    cfg = e.cfg; period = cfg['control_period_s']
    accel = np.r_[np.full(3, 25.), np.full(3, np.deg2rad(100.))]
    speed = np.r_[np.full(3, cfg['linear_speed_mm_s']), np.full(3, cfg['angular_speed_rad_s'])]
    for i in range(1, 801):
        now = 10.+i*period
        if i == 450:
            e.accept(plan(2, now, -30.), now)
        r = e.tick(now, previous[:6], [0., 0., 7.], 0., 0.)
        step = np.r_[r['sent'][:3]-previous[:3],
                     (Rotation.from_rotvec(r['sent'][3:6])*
                      Rotation.from_rotvec(previous[3:6]).inv()).as_rotvec()]
        current = step/period
        assert np.all(np.abs(current) <= speed+1e-8)
        assert np.all(np.abs(current-velocity)/period <= accel+1e-6)
        assert abs(r['sent'][8]-previous[8]) <= cfg['force_rate_N_s']*period+1e-10
        previous, velocity = r['sent'], current


def test_stationary_target_settles_after_acceleration_limited_motion():
    p = plan(); p.action[:, :3] = [5., -5., 3.]
    p.action[:, 3:6] = [0., 0., np.pi]; p.action[:, 8] = 0.
    p = TimedPlan(1, 10., p.time, p.action, p.phase, False)
    e = engine(p); positions = []
    for i in range(1, 501):
        r = e.tick(10.+i*.008, e.command[:6], [0., 0., 0.], 0., 0.)
        positions.append(r['sent'][:3])
    np.testing.assert_allclose(positions[-1], [5., -5., 3.], atol=.001)
    assert np.max(np.ptp(np.asarray(positions)[-50:], axis=0)) < .001


def test_force_commands_and_gate_are_identical_to_unpatched_c():
    p = plan(); old = TimedExecution(executor_settings(config()))
    old.accept(p, 10.); old.start(10., np.array([0., 0., 0., 0., 0., np.pi]))
    patched = engine(p)
    for i in range(500):
        now = 10.+i*.008
        force = [0., 0., 7. if i % 37 < 21 else .5]
        before = old.tick(now, old.command[:6], force, 0., 0.)
        after = patched.tick(now, old.command[:6], force, 0., 0.)
        for stage in ['requested', 'gated', 'sent']:
            np.testing.assert_array_equal(before[stage][6:], after[stage][6:])
        assert before['contact'] == after['contact']
        assert before['elapsed_s'] == after['elapsed_s']
        np.testing.assert_array_equal(after['requested'], p.sample(now-10.)[0])


def test_finite_replay_is_rejected_if_misrouted_to_c_conditioner():
    e = TimedExecution(executor_settings(config(), 'il'))
    p = plan(); p.final = True
    with pytest.raises(ExecutionFault, match='finite R/T'):
        e.accept(p, 10.)


def test_raw_workspace_guard_cannot_be_hidden_by_smoothing():
    p = plan(); p.action[1, 0] = 500.
    e = engine(p)
    e.tick(10.008, e.command[:6], [0., 0., 7.], 0., 0.)
    with pytest.raises(ExecutionFault, match='workspace'):
        e.tick(10.016, e.command[:6], [0., 0., 7.], 0., 0.)


def test_pose_conditioning_does_not_extend_plan_horizon_or_watchdogs():
    e = engine()
    for i in range(530):
        r = e.tick(10.+i*.008, e.command[:6], [0., 0., 7.], 0., 0.)
    assert e.tick(14.24, e.command[:6], [0., 0., 7.], 0., 0.)['end'] == 'plan_underrun'
    e = engine()
    with pytest.raises(ExecutionFault, match='stale'):
        e.tick(10.008, e.command[:6], [0., 0., 7.], 0., .3)


def test_acceleration_limited_braking_cannot_leave_common_workspace():
    p = plan(); p.action[:, :3] = 0.
    e = engine(TimedPlan(1, 10., p.time, p.action, p.phase, False))
    e.cfg['workspace_limits'] = dict(e.cfg['workspace_limits'], max_xy_from_start_mm=1.)
    e.command[0] = .999
    e.conditioner.velocity[0] = 10.
    with pytest.raises(ExecutionFault, match='workspace'):
        e.tick(10.008, e.command[:6].copy(), [0., 0., 7.], 0., 0.)


def test_bridge_logs_c_conditioned_stage_and_preserves_r_t_stages(monkeypatch):
    from test_e2_timed_execution import mock_bridge
    for method in ['il', 'rule', 'replay']:
        n, clock, _ = mock_bridge(monkeypatch)
        n.method = method
        if method == 'il':
            n.settings = executor_settings(config(), 'il')
            n.engine = engine()
        stages = []
        n.log_command = lambda stage, *args: stages.append(stage)
        clock[0] = 10.008
        n.tick()
        expected = ['time_sampled', 'contact_gated', 'node_sent']
        if method == 'il': expected.insert(1, 'pose_conditioned')
        assert stages == expected
        assert len(n.sent) == 1


@pytest.mark.parametrize('key,value', [('pose_smooth_window', 2), ('pose_smooth_window', True),
    ('handover_s', 0.), ('linear_acceleration_mm_s2', float('nan')),
    ('angular_acceleration_rad_s2', -1.), ('profile', 'unknown')])
def test_invalid_conditioning_settings_fail_preflight(key, value):
    c = config(); c['il']['pose_conditioning'][key] = value
    assert any('pose_conditioning' in message for message in validate_settings(c))
