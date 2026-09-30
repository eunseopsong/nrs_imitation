"""No ROS context; commissioning evidence below is TEMPORARY MOCK DATA."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from nrs_imitation import e2_pilot as pilot
from nrs_imitation.e2_ablation import (ROOT, select_condition, cohort_id, digest,
    prepare_A_force, execution_contract, object_hash)
from nrs_imitation.e2_providers import hardware_blockers
from nrs_imitation.e2_run_context import create_attempt
from nrs_imitation.e2_timed_execution import TimedExecution, TimedPlan, executor_settings
from test_e2_timed_execution import mock_bridge, settle

PILOT = ROOT/'experiments/e2_A_pilot_20260927/config.json'
MASTER = ROOT/'experiments/e2_force_ablation_20260926/config.json'


def config():
    return select_condition(json.loads(PILOT.read_text()), 'A')


def seal(c):
    c['execution_contract_hash'] = object_hash(execution_contract(c))
    c['cohort_id'] = cohort_id(c)
    return c


def write_record(c, record, path):
    path.write_text(json.dumps(record))
    c['pilot']['commissioning_record'] = dict(path=str(path), sha256=digest(path))
    seal(c)


@pytest.fixture
def approved_mock(tmp_path, monkeypatch):
    # Mock commissioning evidence is not an approval for the real equipment.
    before = {p: digest(p) for p in (PILOT, MASTER)}
    c = config()
    proof = tmp_path/'MOCK_NOT_HARDWARE.txt'; proof.write_text('Synthetic fixture only')
    ref = dict(path=str(proof), sha256=digest(proof))
    record = dict(schema='E2_A_commissioning_v1', status='approved',
        approved_by='MOCK ONLY', approved_at='MOCK TIME', scope_sha256=pilot.commissioning_scope(c),
        command_frame='controller_tcp', command_axis='Fz', command_unit='N',
        processing_command_fz_range_N=[-2., 24.], measured_force_abs_limits_N=[71., 72., 73.],
        measured_torque_abs_limits_Nm=[4., 5., 6.], measured_wrench_frame='controller_tcp',
        sensor_raw_force_abs_limits_N=[91., 92., 93.],
        sensor_raw_torque_abs_limits_Nm=[7., 8., 9.],
        acquisition_max_age_s=.123)
    for key in ('applicability_evidence', 'command_direction_evidence', 'limits_evidence',
                'sensor_overload_evidence', 'workspace_evidence', 'automatic_stop_evidence',
                'approach_exit_evidence', 'sensor_driver_deployment_evidence'):
        record[key] = ref.copy()
    write_record(c, record, tmp_path/'MOCK_record.json')
    yield c, record
    assert all(digest(p) == h for p, h in before.items())


def test_real_pilot_is_blocked_by_actual_protection_gaps_not_quality_freeze():
    errors = hardware_blockers(config(), 'il', True)
    for field in ('measured_force_abs_limits_N', 'measured_torque_abs_limits_Nm',
                  'acquisition_max_age_s', 'sensor_overload_evidence',
                  'workspace_evidence', 'sensor_driver_deployment_evidence'):
        assert any(field in e for e in errors)
    assert not any('calibrated and frozen' in e for e in errors)
    with pytest.raises(ValueError, match='actual approval required'):
        prepare_A_force(config())


def test_protected_mock_allows_only_pilot_and_does_not_promote(approved_mock):
    c, record = approved_mock; before = copy.deepcopy(c)
    assert hardware_blockers(c, 'il', True) == []
    assert prepare_A_force(c) == 23.
    assert pilot.pilot_force(c, False) == 0.
    assert c == before and not c['pilot']['F0_frozen']
    assert c['pilot']['quality_validation'] == 'pending'
    c['trial_stage'] = 'main'; seal(c)
    errors = hardware_blockers(c, 'il', True)
    assert any('calibrated and frozen' in e for e in errors)
    assert any('pilot_result_evidence' in e for e in errors)
    # Setting only freeze metadata cannot promote an unreviewed pilot to main.
    c['f0'].update(status='frozen', freeze_evidence='MOCK ONLY',
                   pilot_result_evidence=c['pilot']['commissioning_record'].copy())
    seal(c)
    assert any('review of actual pilot results' in e for e in hardware_blockers(c, 'il', True))


@pytest.mark.parametrize('field', ['processing_command_fz_range_N', 'measured_force_abs_limits_N',
    'measured_torque_abs_limits_Nm', 'measured_wrench_frame', 'acquisition_max_age_s',
    'sensor_raw_force_abs_limits_N', 'sensor_raw_torque_abs_limits_Nm',
    'applicability_evidence', 'command_direction_evidence', 'limits_evidence',
    'sensor_overload_evidence', 'workspace_evidence', 'automatic_stop_evidence', 'approach_exit_evidence',
    'sensor_driver_deployment_evidence'])
def test_missing_requirement_blocks_even_mock_backend(approved_mock, tmp_path, field):
    c, record = approved_mock; record[field] = None
    write_record(c, record, tmp_path/'missing.json')
    assert any(field in e for e in hardware_blockers(c, 'il', True))


def test_F0_and_abort_thresholds_never_copy_each_other(approved_mock, tmp_path):
    c, record = approved_mock; original = copy.deepcopy(record)
    c['f0']['command_fz_N'] = 22.; seal(c)
    assert prepare_A_force(c) == 22.
    assert record == original and c['executor']['fz_hard_limit_N'] == 0.
    c['f0']['command_fz_N'] = 25.; seal(c)
    assert any('range_N' in e for e in hardware_blockers(c, 'il', True))
    with pytest.raises(ValueError): prepare_A_force(c)


def test_approval_cannot_install_missing_protection(approved_mock, monkeypatch):
    c, _ = approved_mock
    from nrs_imitation import e2_protection
    monkeypatch.setattr(e2_protection, 'CAPABILITIES', frozenset())
    assert any('protection implementation unavailable' in e for e in hardware_blockers(c, 'il', True))


def test_evidence_identity_scope_and_real_activation_are_required(approved_mock, tmp_path):
    c, record = approved_mock
    c['common']['workspace_limits']['max_xy_from_start_mm'] += 1.; seal(c)
    assert any('scope' in e for e in hardware_blockers(c, 'il', True))
    c['pilot']['commissioning_record']['sha256'] = 'wrong'; seal(c)
    assert any('SHA' in e for e in hardware_blockers(c, 'il', True))


@pytest.mark.parametrize('fault', ['sensor_nan', 'sensor_stale', 'mode_lost', 'manual_abort'])
def test_A_stop_contract_no_more_commands_or_queued_work(monkeypatch, approved_mock, fault):
    c, _ = approved_mock
    n, time, future = mock_bridge(monkeypatch)
    n.ablation = True; n.config = c; n.method = 'il'; n.a_processing_force = prepare_A_force(c)
    n.processing = True
    time[0] += .008
    if fault == 'sensor_nan':
        n.on_force(NS(data=[0., 0., float('nan'), 0., 0., 0.]))
    elif fault == 'sensor_stale': n.force_at = 0.
    elif fault == 'mode_lost': n.mode = 'Position'
    else: n.operator_event('abort', NS())
    n.tick()
    assert n.state == 'stopping' and n.engine.closed and n.engine.plan is None
    assert n.modes == ['Idling'] and not n.sent
    assert n.requests[0].command_mode == 'PTP9D_STREAM_STOP'
    future.cb(future)
    assert n.state == 'stopping' and n.modes == ['Idling', 'Position']
    settle(n, time)
    assert n.state == 'stopped' and not n.sent
    assert any(e['event'] == 'physical_hold_verified' for e in n.events)
    assert not any(e['event'] == 'normal_completion' for e in n.events)


def test_attempt_keeps_pending_state_and_snapshots_provenance(approved_mock, tmp_path):
    c, _ = approved_mock
    attempt, folder = create_attempt(c, 'A', 'mock_pilot', session_path=tmp_path/'attempts')
    assert hardware_blockers(attempt, 'il', True) == []
    stored = json.loads((folder/'attempt.json').read_text())
    assert stored['trial_stage'] == 'pilot' and stored['F0_frozen'] is False
    assert stored['quality_validation'] == 'pending' and stored['executed'] is False
    assert (folder/'artifacts/commissioning.json').exists()
    assert (folder/'artifacts/f0.json').exists()


def test_real_blocked_launch_retains_attempt_without_starting_nodes(tmp_path, monkeypatch):
    from test_e2_direct_launch import launch_fixture
    module, context, callbacks = launch_fixture(tmp_path, monkeypatch, condition='A')
    context.launch_configurations['config'] = str(PILOT)
    with pytest.raises(RuntimeError, match='preflight blocked'):
        module.configure(context)
    for close in callbacks: close()
    attempts = list(tmp_path.glob('test_session/A/*/attempt.json'))
    assert len(attempts) == 1
    record = json.loads(attempts[0].read_text())
    assert record['status'] == 'preflight_blocked' and record['executed'] is False
    assert record['trial_stage'] == 'pilot' and record['quality_validation'] == 'pending'
    assert not (attempts[0].parent/'launch_claim.json').exists()
    assert (attempts[0].parent/'archive_manifest.json').is_file()


def test_approved_A_pilot_resolves_real_launch_and_pinned_checkpoint(approved_mock, tmp_path, monkeypatch):
    """Resolve the actual run launch with temporary evidence; never execute actions."""
    from launch.actions import GroupAction, IncludeLaunchDescription
    from test_e2_direct_launch import (launch_fixture, resolved_inference_parameters,
                                      assert_inference_parameter_types)
    c, _ = approved_mock
    assert c['pilot']['quality_validation'] == 'pending'
    assert c['pilot']['F0_frozen'] is False and c['f0']['freeze_evidence'] is None
    assert c['reference']['status'] != 'frozen'
    path = tmp_path/'MOCK_pilot_config.json'; path.write_text(json.dumps(c))
    module, context, callbacks = launch_fixture(tmp_path, monkeypatch, condition='A')
    context.launch_configurations['config'] = str(path)
    # Keep the real preflight. Only commissioning evidence is synthetic.
    actions = module.configure(context)
    group = next(a for a in actions if isinstance(a, GroupAction))
    include = next(a for a in group.get_sub_entities() if isinstance(a, IncludeLaunchDescription))
    parameters = resolved_inference_parameters(include, context)
    assert_inference_parameter_types(parameters)
    assert parameters['ckpt_dir'] == str(Path(c['models']['A']['checkpoint']).parent)
    assert parameters['use_force_observation'] is False
    assert parameters['use_force_history'] is False
    assert dict(include.launch_arguments)['inference_mode'] == 'timed_topic'
    prepared = json.loads(Path(parameters['e2_config']).read_text())
    assert prepared['il']['motion_only'] is True and prepared['il']['force_action'] is False
    assert prepare_A_force(prepared) == 23.
    assert prepared['pilot']['F0_frozen'] is False
    assert prepared['pilot']['quality_validation'] == 'pending'
    for close in callbacks: close()


@pytest.mark.parametrize('condition', ['B', 'C'])
def test_archived_BC_common_engine_same_outputs(condition):
    folder = next((ROOT/'results/20260926/E2'/condition).glob('*/launch_context'))
    cfg = json.loads((folder/'config.json').read_text())
    src = folder/'artifacts/code/behavior_ws/src/nrs_imitation/nrs_imitation/e2_timed_execution.py'
    spec = importlib.util.spec_from_file_location('archived_e2_'+condition, src)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    current = select_condition(json.loads(MASTER.read_text()), condition)
    assert cfg['executor'] == current['executor']
    assert cfg['motion_postprocessor'] == current['motion_postprocessor']
    engines = [old.TimedExecution(old.executor_settings(cfg, 'il')),
               TimedExecution(executor_settings(current, 'il'))]
    t = np.arange(128)/30.; action = np.zeros((128, 9))
    action[:, 0] = np.sin(t)*12.; action[:, 2] = -t; action[:, 8] = 17.+np.sin(2*t)
    for e, cls in zip(engines, (old.TimedPlan, TimedPlan)):
        e.accept(cls(1, 10., t, action, ['test']*128, False), 10.)
        e.start(10., np.zeros(6))
    pose = np.zeros(6)
    for i in range(525):
        now = 10.+i*.008
        force = [0., 0., 0. if 250 <= i < 275 else 7.]
        if i == 500:
            for e, cls in zip(engines, (old.TimedPlan, TimedPlan)):
                e.accept(cls(2, now, t, action, ['test']*128, False), now)
        results = [e.tick(now, pose, force, 0., 0.) for e in engines]
        for key in ('requested', 'conditioned', 'gated', 'sent'):
            np.testing.assert_array_equal(results[0][key], results[1][key])
        pose = results[0]['sent'][:6]
