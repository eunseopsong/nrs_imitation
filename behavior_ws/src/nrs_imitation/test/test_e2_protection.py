"""Pure guards and the real executor callbacks, with no ROS context or hardware.

All limits/approval records here are synthetic fixtures, never equipment values.
"""
import copy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from nrs_imitation.e2_protection import (ProtectionMonitor, ProtectionFault,
    RAW_SCHEMA, BASE_SCHEMA, LIMIT_KEYS)
from test_e2_A_pilot import approved_mock, write_record
from test_e2_timed_execution import mock_bridge


def record():
    return dict(measured_wrench_frame='robot_base',
        measured_force_abs_limits_N=[31., 32., 33.],
        measured_torque_abs_limits_Nm=[4., 5., 6.],
        sensor_raw_force_abs_limits_N=[61., 62., 63.],
        sensor_raw_torque_abs_limits_Nm=[7., 8., 9.], acquisition_max_age_s=.1)


def settings():
    return dict(feedback_max_age_s=.2, workspace_limits=dict(
        max_xy_from_start_mm=140., max_z_down_from_start_mm=85., max_z_up_from_start_mm=95.))


def packets(t=10., seq=1, raw=None, base=None):
    stamp = int(t*1e9)-1_000_000
    return (dict(schema=RAW_SCHEMA,boot_id='MOCK_BOOT',sequence=seq,
                 source_ros_ns=stamp,valid=True,initialized=True,
                 wrench_frame='sensor',force_unit='N',torque_unit='Nm',
                 timestamp_origin='host_udp_receive_estimate',
                 sensor_wrench=raw or [0.,0.,7.,0.,0.,0.]),
            dict(schema=BASE_SCHEMA,source_contract=True,source_ros_ns=stamp,
                 wrench_frame='robot_base',force_unit='N',torque_unit='Nm',
                 timestamp_origin='host_udp_receive_estimate',
                 valid=True,base_wrench=base or [0.,0.,7.,0.,0.,0.]))


def feed(g, t=10., seq=1, **kwargs):
    raw, base = packets(t, seq, **kwargs)
    g.ingest_raw(raw,t,int(t*1e9)); g.ingest_base(base,t,int(t*1e9))


def check(g, t=10., pose=None, **ages):
    return g.check(t,int(t*1e9),np.zeros(6) if pose is None else pose,
                   ages.get('pose',0.),ages.get('force',0.),ages.get('mode',0.))


@pytest.mark.parametrize('key', LIMIT_KEYS+('acquisition_max_age_s',))
@pytest.mark.parametrize('value', [None, False, float('nan'), 0.])
def test_no_implicit_limits_or_timeout(key, value):
    r=record(); r[key]=value
    with pytest.raises(ProtectionFault): ProtectionMonitor(r,settings())


def test_missing_source_never_arms_and_cached_updates_never_refresh():
    g=ProtectionMonitor(record(),settings())
    with pytest.raises(ProtectionFault,match='waiting'): check(g)
    feed(g); check(g)
    raw,base=packets()
    assert not g.ingest_raw(raw,10.08,10_080_000_000)
    g.ingest_base(base,10.08,10_080_000_000)
    assert g.raw_at==g.base_at==10.
    with pytest.raises(ProtectionFault,match='stopped advancing'): check(g,10.11)


@pytest.mark.parametrize('side', ['sensor','controller'])
def test_one_live_producer_cannot_hide_other_source_stall(side):
    g=ProtectionMonitor(record(),settings()); feed(g); check(g)
    raw,base=packets(10.11,2)
    if side=='sensor': g.ingest_base(base,10.11,10_110_000_000)
    else: g.ingest_raw(raw,10.11,10_110_000_000)
    with pytest.raises(ProtectionFault,match=side+' acquisition'): check(g,10.11)


@pytest.mark.parametrize('change,expected', [
    ({'boot_id':'NEW_BOOT'},'restarted'), ({'sequence':0},'sequence'),
    ({'valid':False},'invalid Ethernet'), ({'source_ros_ns':10_001_000_000},'clocks'),
    ({'source_ros_ns':9_000_000_000},'stale'), ({'source_ros_ns':9_999_100_000},'restamped'),
    ({'sequence':2},'failed to advance'), ({'schema':'old_driver'},'contract')])
def test_sensor_protocol_faults_latch(change,expected):
    g=ProtectionMonitor(record(),settings()); feed(g); check(g)
    raw,_=packets(); raw.update(change)
    with pytest.raises(ProtectionFault,match=expected): g.ingest_raw(raw,10.,10_000_000_000)
    with pytest.raises(ProtectionFault): feed(g,10.01,3)


@pytest.mark.parametrize('key,value', [('valid',False),('source_contract',False),
    ('base_wrench',[0.,0.,float('nan'),0.,0.,0.]), ('base_wrench',[0.,0.,0.])])
def test_bad_controller_contract_rejected(key,value):
    g=ProtectionMonitor(record(),settings()); _,base=packets(); base[key]=value
    with pytest.raises(ProtectionFault): g.ingest_base(base,10.,10_000_000_000)


@pytest.mark.parametrize('axis', range(6))
@pytest.mark.parametrize('sign', [-1.,1.])
def test_all_measured_axes_both_signs_stop_at_bound(axis,sign):
    r=record(); g=ProtectionMonitor(r,settings()); v=[0.]*6
    v[axis]=sign*(r['measured_force_abs_limits_N']+r['measured_torque_abs_limits_Nm'])[axis]
    feed(g,base=v)
    with pytest.raises(ProtectionFault,match='abort threshold'): check(g)


@pytest.mark.parametrize('axis', range(6))
def test_raw_envelope_independent_of_zeroed_filtered_value(axis):
    r=record(); g=ProtectionMonitor(r,settings()); v=[0.]*6
    v[axis]=(r['sensor_raw_force_abs_limits_N']+r['sensor_raw_torque_abs_limits_Nm'])[axis]
    with pytest.raises(ProtectionFault,match='before zero'): feed(g,raw=v,base=[0.]*6)


def test_rotation_uses_actual_pose_and_torque_axes():
    r=record();r.update(measured_wrench_frame='controller_tcp',
                       measured_force_abs_limits_N=[40.,20.,40.])
    g=ProtectionMonitor(r,settings());feed(g,base=[25.,0.,0.,0.,0.,0.])
    with pytest.raises(ProtectionFault,match='abort threshold'):
        check(g,pose=[0.,0.,0.,0.,0.,np.pi/2])


@pytest.mark.parametrize('pose', [[141.,0.,0.,0.,0.,0.], [0.,0.,-86.,0.,0.,0.],
                                [0.,0.,96.,0.,0.,0.], [100.,100.,0.,0.,0.,0.]])
def test_actual_workspace_from_startup_not_changing_target(pose):
    g=ProtectionMonitor(record(),settings());feed(g);check(g)
    with pytest.raises(ProtectionFault,match='actual TCP'):check(g,pose=pose)
    np.testing.assert_array_equal(g.anchor,np.zeros(3))


def test_initialization_and_feedback_age_required():
    g=ProtectionMonitor(record(),settings());raw,base=packets();raw['initialized']=False
    g.ingest_raw(raw,10.,10_000_000_000);g.ingest_base(base,10.,10_000_000_000)
    with pytest.raises(ProtectionFault,match='initialization'):check(g)
    feed(g,10.01,2);check(g,10.01)
    with pytest.raises(ProtectionFault,match='feedback stale'):check(g,10.01,force=.21)
    raw,_=packets(10.02,3);raw['initialized']=False
    with pytest.raises(ProtectionFault,match='initialization lost'):
        g.ingest_raw(raw,10.02,10_020_000_000)


def protected_bridge(monkeypatch,c,r):
    n,t,f=mock_bridge(monkeypatch)
    n.ablation=True;n.config=c;n.method='il';n.a_processing_force=23.
    n.force=np.array([0.,0.,7.,0.,0.,0.]);n.sample_last={}
    n.protection=ProtectionMonitor(r,n.settings);n.protection_wait_reason=None
    n.get_clock=lambda:NS(now=lambda:NS(nanoseconds=int(t[0]*1e9)))
    return n,t,f


def source_callbacks(n,t,seq=1):
    raw,base=packets(t[0],seq)
    n.on_acquisition(NS(data=json.dumps(raw)))
    n.on_wrench_provenance(NS(data=json.dumps(base)))


def test_real_executor_waits_for_both_new_producers_before_any_start_command(monkeypatch,approved_mock):
    c,r=approved_mock;n,t,f=protected_bridge(monkeypatch,c,r)
    n.state='waiting_feedback';n.started_at=None
    n.tick()
    assert n.state=='waiting_feedback' and not n.sent and not n.modes and not n.requests
    raw,base=packets(t[0]);n.on_acquisition(NS(data=json.dumps(raw)));n.tick()
    assert n.state=='waiting_feedback' and not n.modes
    n.on_wrench_provenance(NS(data=json.dumps(base)));n.tick()
    assert n.state=='stopping' and n.stop_reason=='startup_reset' and n.modes==['Idling']
    f.cb(f)
    for i in range(60):
        t[0]+=.01;n.pose_at=n.force_at=n.mode_at=t[0];n.mode='Position'
        source_callbacks(n,t,i+2);n.tick()
    assert n.state=='ready' and not n.sent
    assert c['pilot']['F0_frozen'] is False and c['pilot']['quality_validation']=='pending'


@pytest.mark.parametrize('phase',['ready','arming','running','stopping'])
@pytest.mark.parametrize('fault',['force','torque','raw_envelope','source_stale','workspace','sensor_invalid'])
def test_protection_stops_all_phases_and_preserves_queue_hold_contract(monkeypatch,approved_mock,phase,fault):
    c,r=approved_mock;n,t,f=protected_bridge(monkeypatch,c,r)
    # Initialize at waiting state so asynchronous arrival of the first two streams is permitted.
    n.state='waiting_feedback';source_callbacks(n,t);assert n.protection_check(t[0])
    n.state=phase
    if phase=='stopping':
        n.state='ready';n.request_stop('startup_reset',initial=True)
    t[0]+=.008;n.pose_at=n.force_at=n.mode_at=t[0]
    raw,base=packets(t[0],2)
    if fault=='force':base['base_wrench'][2]=r['measured_force_abs_limits_N'][2]
    if fault=='torque':base['base_wrench'][4]=r['measured_torque_abs_limits_Nm'][1]
    if fault=='raw_envelope':raw['sensor_wrench'][0]=r['sensor_raw_force_abs_limits_N'][0]
    if fault=='sensor_invalid':raw['valid']=False
    if fault=='workspace':n.pose[0]=141.
    if fault=='source_stale':
        t[0]+=.13;n.pose_at=n.force_at=n.mode_at=t[0]
    else:
        n.on_acquisition(NS(data=json.dumps(raw)))
        n.on_wrench_provenance(NS(data=json.dumps(base)))
    n.tick()
    assert n.state=='stopping' and n.engine.closed and n.engine.plan is None
    assert n.modes and set(n.modes)=={'Idling'} and not n.sent and n.stop_failed_latched
    assert n.requests[0].command_mode=='PTP9D_STREAM_STOP' and len(n.requests)==1
    assert any(e['event']=='protection_fault' for e in n.events)
    f.cb(f)
    for _ in range(60):
        t[0]+=.01;n.pose_at=n.force_at=n.mode_at=t[0];n.mode='Position';n.tick()
    assert n.state=='stopped' and n.modes[-1]=='Position' and not n.sent
    assert not any(e['event'] in ('normal_completion','executor_ready') for e in n.events)


def test_processing_F0_does_not_change_guard_limits(monkeypatch,approved_mock):
    c,r=approved_mock;n,t,f=protected_bridge(monkeypatch,c,r);original=copy.deepcopy(r)
    n.state='waiting_feedback';source_callbacks(n,t);assert n.protection_check(t[0])
    n.state='running';n.processing=True;t[0]+=.008;n.tick()
    assert n.sent and 0.<n.sent[-1][-1]<=23.
    assert r==original
    for key in LIMIT_KEYS:np.testing.assert_array_equal(n.protection.limits[key],r[key])


@pytest.mark.parametrize('termination', ['finish', 'abort'])
def test_A_F0_selection_range_allows_common_phase_commands(
        monkeypatch, approved_mock, tmp_path, record_property, termination):
    """Exercise real callbacks/engine/guards; all feedback and publishers are mocks."""
    from nrs_imitation.e2_ablation import motion_to_contract, prepare_A_force
    from nrs_imitation.e2_timed_execution import TimedExecution, TimedPlan, executor_settings
    c, r = approved_mock
    r['processing_command_fz_range_N'] = [23., 23.]
    write_record(c, r, tmp_path/'MOCK_single_setpoint.json')
    original = copy.deepcopy(r)
    n, t, future = protected_bridge(monkeypatch, c, r)
    n.settings = executor_settings(c, 'il')
    n.protection = ProtectionMonitor(r, n.settings)
    n.a_processing_force = prepare_A_force(c)
    n.engine = TimedExecution(n.settings)
    times = np.arange(128)/30.
    motion = np.zeros((128, 6)); motion[:, 0] = times; motion[:, 2] = -.2*times
    actions = motion_to_contract(motion)
    n.engine.accept(TimedPlan(1, t[0], times, actions, ['unknown']*128, False), t[0])
    n.engine.start(t[0], n.pose)
    n.state = 'waiting_feedback'; source_callbacks(n, t)
    assert n.protection_check(t[0])
    n.state = 'running'
    rows = []; stage_rows = {}; sequence = 1
    n.log_command = lambda stage, values, result, now: stage_rows.update({stage: values.copy()})

    def step(phase, measured_fz):
        nonlocal sequence
        t[0] += n.settings['control_period_s']; sequence += 1
        n.pose_at = n.force_at = n.mode_at = n.provider_at = t[0]
        n.force = np.array([0., 0., measured_fz, 0., 0., 0.])
        raw, base = packets(t[0], sequence, raw=n.force.tolist(), base=n.force.tolist())
        n.on_acquisition(NS(data=json.dumps(raw)))
        n.on_wrench_provenance(NS(data=json.dumps(base)))
        count = len(n.sent); n.tick()
        assert n.state == 'running' and len(n.sent) == count+1
        sent = np.asarray(n.sent[-1])
        requested = stage_rows['time_sampled'][8]
        assert requested == (23. if n.processing else 0.)
        assert np.all(sent[6:8] == 0.)
        previous = np.asarray(n.sent[-2]) if len(n.sent) > 1 else np.zeros(9)
        assert abs(sent[8]-previous[8]) <= 30.*.008+1e-10
        assert np.max(np.abs(sent[:3]-previous[:3])) <= 10.*.008+1e-10
        rows.append(dict(phase=phase, measured_fz_N=measured_fz, requested_fz_N=requested,
                         gated_fz_N=stage_rows['contact_gated'][8], sent_fz_N=sent[8]))
        n.pose = sent[:6].copy()
        return rows[-1]

    for _ in range(12): assert step('approach', 0.)['sent_fz_N'] == 0.
    # Contact alone must not apply F0 before the processing marker.
    for _ in range(4): assert step('approach_contact', 7.)['sent_fz_N'] == 0.
    assert n.operator_event('processing_start', NS()).success
    for _ in range(160): row = step('processing', 7.)
    assert 22.9 < row['sent_fz_N'] <= 23.
    assert any(0. < r['sent_fz_N'] < 23. for r in rows)
    # The scalar selection range is not a gate or a measured-force envelope.
    for measured, contact in [(1.2, False), (2., False), (3., True), (2., True), (1.2, False), (29., True)]:
        row = step('contact_gate', measured)
        assert row['requested_fz_N'] == 23.
        assert row['gated_fz_N'] == (23. if contact else 0.)
    assert n.operator_event('processing_end', NS()).success
    first_retract = step('retract', 7.)
    assert first_retract['gated_fz_N'] == 0. and first_retract['sent_fz_N'] > 0.
    for _ in range(219): step('retract', 7.)
    for _ in range(12): row = step('release', -2.)
    assert row['gated_fz_N'] == 0. and abs(row['sent_fz_N']) < 1e-6
    assert not n.requests and not n.modes and not n.protection.fault
    assert r == original
    for key in LIMIT_KEYS: np.testing.assert_array_equal(n.protection.limits[key], r[key])
    assert c['pilot']['quality_validation'] == 'pending' and c['pilot']['F0_frozen'] is False

    count = len(n.sent)
    assert n.operator_event(termination, NS()).success
    assert n.state == 'stopping' and n.engine.closed and n.engine.plan is None
    assert n.modes == ['Idling'] and n.requests[0].command_mode == 'PTP9D_STREAM_STOP'
    future.cb(future)
    for _ in range(60):
        t[0] += .01; sequence += 1
        n.pose_at = n.force_at = n.mode_at = t[0]; n.mode = 'Position'
        raw, base = packets(t[0], sequence, raw=[0.]*6, base=[0.]*6)
        n.on_acquisition(NS(data=json.dumps(raw))); n.on_wrench_provenance(NS(data=json.dumps(base)))
        n.tick()
    assert n.state == 'stopped' and len(n.sent) == count and not n.protection.fault
    assert any(e['event'] == 'physical_hold_verified' for e in n.events)
    if termination == 'abort': assert not any(e['event'] == 'normal_completion' for e in n.events)
    summary = []
    for phase in dict.fromkeys(row['phase'] for row in rows):
        group = [r for r in rows if r['phase'] == phase]
        summary.append(dict(phase=phase, ticks=len(group),
            requested_fz_N=sorted(set(r['requested_fz_N'] for r in group)),
            sent_fz_min_N=min(r['sent_fz_N'] for r in group),
            sent_fz_max_N=max(r['sent_fz_N'] for r in group)))
    record_property('mock_only_no_hardware', True)
    record_property('phase_summary', json.dumps(summary))


def test_protection_stop_never_claims_hold_without_queue_ack(monkeypatch,approved_mock):
    c,r=approved_mock;n,t,f=protected_bridge(monkeypatch,c,r)
    n.state='waiting_feedback';source_callbacks(n,t);assert n.protection_check(t[0])
    n.state='running';n.protection_failure('MOCK sensor fault')
    t[0]+=n.settings['stop_timeout_s']+.01;n.pose_at=n.force_at=n.mode_at=t[0];n.tick()
    assert n.state=='stop_failed' and not n.sent and n.modes==['Idling','Idling']
    assert any(e['event']=='stop_not_verified' for e in n.events)
    assert not any(e['event']=='physical_hold_verified' for e in n.events)
