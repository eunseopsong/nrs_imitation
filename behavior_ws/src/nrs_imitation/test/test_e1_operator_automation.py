"""E1 operator workflow with synthetic feedback; never start ROS or hardware."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
from launch import LaunchContext
from launch.actions import EmitEvent, LogInfo
from launch_ros.actions import Node

from test_e2_timed_execution import config, mock_bridge, settle
from test_e2_return_home import spec
from test_rtc_launch import rtc


def automated_bridge(monkeypatch):
    n, clock, future = mock_bridge(monkeypatch)
    n.e1_operator_automation = True
    n.auto_processing_interval = 0
    n.last_console_status = None
    n.console = []
    n.get_logger = lambda: NS(info=n.console.append, error=n.console.append)
    return n, clock, future


def advance(n, clock, elapsed, fz):
    clock[0] = 10. + elapsed
    n.pose_at = n.force_at = n.mode_at = n.provider_at = clock[0]
    n.force = np.array([0., 0., fz])
    n.tick()
    if n.sent:
        n.pose = np.array(n.sent[-1][:6])


@pytest.mark.parametrize('method', ['rule', 'replay', 'il'])
def test_contact_markers_do_not_change_any_motion_force_or_stop_command(monkeypatch, method):
    executions = []
    for automatic in (False, True):
        with monkeypatch.context() as patch:
            n, clock, future = automated_bridge(patch)
            n.method = method
            n.e1_operator_automation = automatic
            for i in range(1, 221):
                # Existing hysteresis: 2 N retains contact, 1 N releases it.
                fz = 0. if i < 20 else 7. if i < 60 else 2. if i < 100 else 1. if i < 150 else 7.
                advance(n, clock, i * .008, fz)
            assert n.operator_event('finish', NS()).success
            assert not n.may_exit()  # A service response is not a physical stop.
            future.cb(future)
            assert not n.may_exit()  # Neither is the queue-cancel ACK.
            settle(n, clock)
            executions.append(n)
    old, new = executions
    np.testing.assert_array_equal(new.sent, old.sent)
    assert new.modes == old.modes
    assert [r.command_mode for r in new.requests] == [r.command_mode for r in old.requests]
    assert new.state == old.state == 'stopped'
    assert new.may_exit() and not old.may_exit()
    markers = [e for e in new.events if e['event'] in ('processing_start', 'processing_end')]
    assert [e['event'] for e in markers] == [
        'processing_start', 'processing_end', 'processing_start', 'processing_end']
    assert [e['interval_id'] for e in markers] == [1, 1, 2, 2]
    assert [e['measured_fz_N'] for e in markers] == [7., 1., 7., 7.]
    assert all(e['source'] == 'automatic_contact_proxy' and
               e['physical_processing_verified'] is False and
               e['evaluation_interval_verified'] is False for e in markers)
    assert not markers[1]['interval_truncated']
    assert markers[-1]['termination'] == 'operator_finish' and markers[-1]['interval_truncated']


def test_no_contact_leaves_processing_unknown_and_status_is_shown_on_change(monkeypatch):
    n, clock, future = automated_bridge(monkeypatch)
    for i in range(1, 5):
        advance(n, clock, i * .008, 0.)
        n.status()
    assert len(n.console) == 1
    assert not [e for e in n.events if e['event'].startswith('processing_')]
    assert n.statuses[-1]['physical_processing_verified'] is False
    assert not n.operator_event('processing_start', NS()).success
    assert not n.processing
    assert n.operator_event('finish', NS()).success
    future.cb(future)
    settle(n, clock)
    assert n.may_exit()
    assert n.statuses[-1]['physical_stop_verified'] is True
    assert 'physical_stop_verified=True' in n.console[-1]
    assert not [e for e in n.events if e['event'].startswith('processing_')]


@pytest.mark.parametrize('termination', ['abort', 'stale_feedback'])
def test_abort_and_fault_close_estimate_and_never_exit_before_verified_stop(monkeypatch, termination):
    n, clock, future = automated_bridge(monkeypatch)
    advance(n, clock, .008, 7.)
    if termination == 'abort':
        assert n.operator_event('abort', NS()).success
    else:
        clock[0] += .008
        n.pose_at = 0.
        n.tick()
    assert n.state == 'stopping' and not n.processing and not n.may_exit()
    marker = next(e for e in reversed(n.events) if e['event'] == 'processing_end')
    assert marker['interval_truncated']
    assert marker['physical_contact_release'] == 'unknown'
    assert marker['termination'] == ('manual_abort' if termination == 'abort' else 'safety_stop')
    # Missing ACK/feedback still latches the existing failure, never auto-closes.
    clock[0] += 16.
    n.tick()
    assert n.state == 'stop_failed' and not n.may_exit()
    assert not any(e['event'] == 'normal_completion' for e in n.events)


def test_r_does_not_close_at_intermediate_hold_before_automatic_retract(monkeypatch):
    n, clock, future = automated_bridge(monkeypatch)
    n.method = 'rule'
    n.config = config()
    n.config['common']['demo_start_pose6'] = [10., 0., 10., 0., 0., 0.]
    n.return_spec = spec()
    n.return_home = None
    n.reference = [0., 0.]
    n.pose = np.array([0., 0., 3., 0., 0., 0.])
    n.final_target = np.zeros(6)
    n.request_stop('trajectory_end')
    future.cb(future)
    for _ in range(50):
        clock[0] += .01
        n.pose_at = n.force_at = n.mode_at = n.provider_at = clock[0]
        n.mode = 'Position'
        n.tick()
        assert not n.may_exit()
        if n.state == 'returning':
            break
    assert n.state == 'returning'


def test_automatic_markers_cannot_drive_e2_ablation_force_schedule(monkeypatch):
    n, clock, future = automated_bridge(monkeypatch)
    n.ablation = True
    n.automatic_processing_marker(True, result={'phase': 'processing'})
    assert not n.processing and not n.events


@pytest.mark.parametrize('code', [0, 1])
def test_launch_closes_recorders_only_on_its_executor_exit(code):
    context = LaunchContext()
    unrelated = NS(action=Node(package='nrs_imitation', executable='overlay_video_recorder'), returncode=code)
    assert rtc.executor_exited(unrelated, context) == []
    executor = NS(action=Node(package='nrs_imitation', executable='e2_executor'), returncode=code)
    actions = rtc.executor_exited(executor, context)
    assert len(actions) == 2 and isinstance(actions[0], LogInfo) and isinstance(actions[1], EmitEvent)
