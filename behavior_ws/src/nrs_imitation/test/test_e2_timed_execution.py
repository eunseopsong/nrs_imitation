"""No ROS context: exercise time sampling, watchdogs and real executor callbacks."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
from types import MethodType

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from nrs_imitation.e2_timed_execution import TimedExecution, TimedPlan, ExecutionFault, StopVerifier, executor_settings
from nrs_imitation.e2_providers import hardware_blockers
from nrs_imitation.e2_executor_node import E2Executor
import nrs_imitation.e2_executor_node as bridge

ROOT=Path(__file__).resolve().parents[4]
CONFIG=ROOT/'experiments/e2_rule_replay_20260920/config.json'


def config():
    c=json.loads(CONFIG.read_text());c['common']['task_time_budget_s']=30.
    return c


def plan(final=True):
    a=np.zeros((3,9));a[:,0]=[0.,1.,0.];a[:,8]=[0.,8.,-4.]
    return TimedPlan(1,10.,[0.,1.,2.],a,['approach','processing','end'],final)


def engine(final=True):
    e=TimedExecution(executor_settings(config()));e.accept(plan(final),10.);e.start(10.,np.zeros(6));return e


def test_same_xyz_uses_time_for_force_and_exact_final_phase():
    p=plan()
    assert p.sample(0.)[0][0]==p.sample(2.)[0][0]
    assert p.sample(0.)[0][8]==0. and p.sample(2.)[0][8]==-4.
    assert p.sample(1.5)[0][8]==2. and p.sample(2.)[1]=='end'
    a=np.zeros((2,9));a[:,5]=np.deg2rad([179.,-179.])
    p=TimedPlan(1,0.,[0.,1.],a,['x','x'],True)
    assert abs(abs(p.sample(.5)[0][5])-np.pi)<1e-6


def test_gate_rate_limits_and_single_clock():
    e=engine();previous=np.zeros(9)
    for i in range(251):
        t=10.+i*.008
        r=e.tick(t,previous[:6],np.array([0.,0.,7.]),0.,0.)
        np.testing.assert_allclose(r['requested'],plan().sample(t-10.)[0],atol=1e-12)
        assert np.max(np.abs(r['sent'][:3]-previous[:3]))<=.08000001
        assert abs(r['sent'][8]-previous[8])<=.24000001
        assert np.all(r['sent'][6:8]==0.)
        previous=r['sent']
    assert e.tick(12.008,previous[:6],[0.,0.,7.],0.,0.)['end']=='trajectory_end'
    # Common contact gate clears requested force; rate limiting is separate.
    e=engine();r=e.tick(10.008,np.zeros(6),[0.,0.,0.],0.,0.)
    assert r['requested'][8]>0 and r['gated'][8]==r['sent'][8]==0


@pytest.mark.parametrize('fault', ['stale','gap','envelope','nan'])
def test_watchdogs(fault):
    e=engine();args=[10.008,np.zeros(6),[0.,0.,7.],0.,0.]
    if fault=='stale':args[-1]=.3
    if fault=='gap':args[0]=10.2
    if fault=='envelope':args[1]=np.ones(6)*1000
    if fault=='nan':args[2]=[0.,0.,np.nan]
    with pytest.raises(ExecutionFault):e.tick(*args)


def test_no_repeat_no_stale_plan_and_explicit_underrun():
    e=engine()
    with pytest.raises(ExecutionFault,match='duplicate'):e.accept(plan(),10.)
    p=plan();p.plan_id=2
    with pytest.raises(ExecutionFault,match='finite'):e.accept(p,10.)
    e.stop()
    with pytest.raises(ExecutionFault,match='terminal'):e.accept(p,10.)
    e=engine(False)
    for i in range(252):r=e.tick(10.+i*.008,np.zeros(6),[0.,0.,7.],0.,0.)
    assert r['end']=='plan_underrun'
    e=TimedExecution(executor_settings(config()))
    with pytest.raises(ExecutionFault,match='stale'):e.accept(plan(),11.)


def test_stationary_verifier_needs_new_mode_and_distinct_fresh_samples():
    v=StopVerifier(1.)
    for t in np.arange(1.01,1.51,.01):
        assert not v.observe(t,np.zeros(6),'Position',.99,t)
    for t in np.arange(2.01,2.51,.01):
        done=v.observe(t,np.zeros(6),'Position',t,t)
    assert done
    assert not v.observe(3.,np.zeros(6),'Position',2.5,2.5)
    v=StopVerifier(1.)
    for t in np.arange(1.01,1.6,.01):
        assert not v.observe(t,np.array([t*10,0,0,0,0,0]),'Position',t,t)


def mock_bridge(monkeypatch):
    clock=[10.];monkeypatch.setattr(bridge.time,'monotonic',lambda:clock[0])
    n=NS(ablation=False,phase_index=-1,feedback_received={'pose':0,'force':0})
    for name,fn in E2Executor.__dict__.items():
        if callable(fn) and name not in ('__init__','destroy_node'):setattr(n,name,MethodType(fn,n))
    n.settings=executor_settings(config());n.engine=engine();n.state='running'
    n.stop_failed_latched=False;n.shutdown_requested=False;n.shutdown_at=None;n.started_at=10.
    n.stop_reason=None;n.final_target=None;n.processing=False;n.pending=None
    n.pose=np.zeros(6);n.force=np.array([0.,0.,7.]);n.mode='Force'
    n.pose_at=n.force_at=n.mode_at=n.provider_at=10.;n.reference=None;n.last_phase=None;n.command_id=0
    n.session='test';n.identity='config';n.clock_identity='host';n.method='replay'
    n.events=[];n.modes=[];n.sent=[];n.statuses=[];n.requests=[]
    n.event=lambda name,**d:n.events.append(dict(event=name,**d))
    n.pub_mode=NS(publish=lambda m:n.modes.append(m.data))
    n.pub_command=NS(publish=lambda m:n.sent.append(m.data))
    n.pub_status=NS(publish=lambda m:n.statuses.append(json.loads(m.data)))
    n.recorder=NS(path=None,write_errors=[])
    n.get_logger=lambda:NS(error=lambda m:None)
    n.log_command=lambda *a:None
    class Future:
        def add_done_callback(self,cb):self.cb=cb
        def result(self):return NS(success=True,message='MOCK ACK')
    f=Future()
    n.client=NS(service_is_ready=lambda:True,call_async=lambda req:(n.requests.append(req) or f))
    return n,clock,f


def settle(n,clock,seconds=.6):
    for i in range(int(seconds/.01)):
        clock[0]+=.01;n.pose_at=n.force_at=n.mode_at=clock[0];n.mode='Position';n.tick()


def test_stop_ack_is_not_stop_verification(monkeypatch):
    n,t,f=mock_bridge(monkeypatch)
    n.request_stop('manual_abort')
    assert n.modes==['Idling'] and n.engine.closed and not n.engine.plan
    assert n.requests[0].command_mode=='PTP9D_STREAM_STOP'
    f.cb(f)
    assert n.modes==['Idling','Position'] and n.state=='stopping'
    settle(n,t)
    assert n.state=='stopped' and any(e['event']=='manual_abort' and e.get('controller_hold_verified') for e in n.events)
    assert not any(e['event']=='normal_completion' for e in n.events)
    assert not n.sent


def test_failed_initial_stop_never_auto_resumes_after_late_ack(monkeypatch):
    n,t,f=mock_bridge(monkeypatch);n.state='waiting_feedback';n.started_at=None
    n.request_stop('startup_reset',initial=True)
    t[0]+=16.;n.tick();assert n.state=='stop_failed'
    f.cb(f);settle(n,t)
    assert n.state=='stopped' and not any(e['event']=='executor_ready' for e in n.events)


def test_operator_finish_and_processing_are_separate(monkeypatch):
    n,t,f=mock_bridge(monkeypatch)
    assert n.operator_event('processing_start',NS()).success
    assert not n.operator_event('processing_start',NS()).success
    assert n.operator_event('finish',NS()).success
    assert not any(e['event']=='normal_completion' for e in n.events)
    f.cb(f);settle(n,t)
    assert any(e['event']=='normal_completion' and not e['retract_complete'] for e in n.events)
    assert any(e['event']=='processing_end' for e in n.events)


@pytest.mark.parametrize('reason',['heartbeat','mode','logger'])
def test_bridge_fault_stops_before_publishing(monkeypatch,reason):
    n,t,f=mock_bridge(monkeypatch);t[0]=10.008
    if reason=='heartbeat':n.provider_at=0.
    if reason=='mode':n.mode='Position'
    if reason=='logger':n.recorder.write_errors=['disk full']
    n.tick();assert n.state=='stopping' and n.modes==['Idling'] and not n.sent


def test_config_no_unconditional_capability_blocks():
    c=config()
    c['common'].update(rpm_setpoint=1.,spindle_control_procedure='OFFLINE TEST ONLY',surface_normal_base=[0,0,1],
        tcp_calibration_id='mock',force_frame_sign_verification='mock',
        approach_retract_protocol='mock',controller_deployment_id='mock')
    assert hardware_blockers(c,'il',True)==[]  # synthetic config is NEVER used to launch
    c['executor']['transport']='service_stream'
    assert any('transport' in s for s in hardware_blockers(c,'il',True))


def test_executed_orientation_crosses_pi_on_short_arc():
    e=TimedExecution(executor_settings(config()))
    a=np.zeros((2,9));a[:,5]=np.deg2rad([179.,-179.])
    e.accept(TimedPlan(1,10.,[0.,1.],a,['x','x'],True),10.)
    pose=a[0,:6].copy();e.start(10.,pose)
    for i in range(126):
        result=e.tick(10.+i*.008,pose,[0.,0.,7.],0.,0.)
        pose=result['sent'][:6]
        change=(Rotation.from_rotvec(pose[3:])*Rotation.from_rotvec(a[0,3:6]).inv()).magnitude()
        assert change<=np.deg2rad(2.00001)
    assert change>np.deg2rad(1.8)


def test_plan_identity_and_frozen_reference_are_enforced(monkeypatch):
    n,t,f=mock_bridge(monkeypatch)
    p=plan();d=dict(session_id='unrelated',clock_id='host',config_sha256='config',method='replay',frame='robot_base',
        plan_id=2,generated_at=10.,time=p.time.tolist(),action=p.action.tolist(),phase=p.phase.tolist(),
        final=True,reference_xy_mm=[400.,500.])
    n.on_plan(NS(data=json.dumps(d)));assert n.state=='running' and not n.modes
    d['session_id']='test';d['clock_id']='wrong_host'
    n.on_plan(NS(data=json.dumps(d)));assert n.state=='stopping' and n.modes==['Idling']


def test_startup_cancel_and_observed_hold_precede_ready(monkeypatch):
    n,t,f=mock_bridge(monkeypatch);n.state='waiting_feedback';n.started_at=None
    n.tick();assert n.state=='stopping'
    assert not any(e['event']=='executor_ready' for e in n.events)
    f.cb(f);settle(n,t)
    assert n.state=='ready' and any(e['event']=='executor_ready' for e in n.events)


def test_controller_stop_no_ack_cannot_claim_completion(monkeypatch):
    n,t,f=mock_bridge(monkeypatch);n.request_stop('operator_finish')
    settle(n,t,1.)
    assert n.state=='stopping' and not any(e['event']=='normal_completion' for e in n.events)
    n.request_shutdown();t[0]+=18.;n.tick()
    assert n.state=='stop_failed' and n.may_exit()
    assert any(e['event']=='shutdown_without_verified_stop' for e in n.events)


@pytest.mark.parametrize('method',['rule','replay','il'])
def test_fully_specified_config_allows_each_method_without_bypass(tmp_path,method):
    from nrs_imitation.e2_providers import TimedActions
    c=config();c['common'].update(rpm_setpoint=1.,spindle_control_procedure='offline-only mock',
        force_frame_sign_verification='offline-only mock',controller_deployment_id='offline-only mock')
    fixture=json.loads((CONFIG.parent/'rule_synthetic_fixture.json').read_text())
    c['task'].update(fixture['task']);c['recipe']=fixture['recipe'];c['recipe']['simulation_only']=False
    c['common']['demo_start_pose6']=[-1.,0.,1.,0.,0.,0.]
    c['recipe']['approach']=dict(waypoints_pose6=[[-1.,0.,1.,0.,0.,0.],[-1.,0.,0.,0.,0.,0.]],
        peak_speed_mm_s=10.,peak_angular_speed_rad_s=.5)
    t=TimedActions.load(CONFIG.parent/'templates/episode_29_candidate.npz')
    t.metadata.update(selected_success=True,selection_evidence='UNIT TEST ONLY; not an operator confirmation')
    target=tmp_path/'mock_selected.npz';t.save(target)
    c['replay']['template']=str(target)
    assert hardware_blockers(c,method,True)==[]
    assert any('enable_hardware' in s for s in hardware_blockers(c,method,False))


def test_nonuniform_final_interval_consumes_endpoint_once_without_time_scaling():
    e=TimedExecution(executor_settings(config()));p=plan();p.time=np.array([0.,1.,2.003])
    p=TimedPlan(1,10.,p.time,p.action,p.phase,True);e.accept(p,10.);e.start(10.,np.zeros(6))
    for i in range(252):r=e.tick(10.+i*.008,np.zeros(6),[0.,0.,7.],0.,0.)
    np.testing.assert_array_equal(r['requested'],p.action[-1])
    assert r['phase']=='end' and np.isclose(r['endpoint_lateness_s'],.005)
    assert e.tick(12.016,np.zeros(6),[0.,0.,7.],0.,0.)['end']=='trajectory_end'


def test_stop_verifier_does_not_bridge_an_unobserved_feedback_gap():
    v=StopVerifier(1.)
    for t in [1.01,1.02,1.03,1.6,1.61,1.62]:
        assert not v.observe(t,np.zeros(6),'Position',t,t)
    for t in np.arange(1.63,2.1,.01):done=v.observe(t,np.zeros(6),'Position',t,t)
    assert done
