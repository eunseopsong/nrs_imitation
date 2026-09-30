"""Hardware-free E2 invariants, including the pre-E2 inference method as oracle."""
import ast
from collections import deque
import csv
import json
from pathlib import Path
import subprocess
from types import MethodType, SimpleNamespace as NS

import numpy as np
import pytest
import torch
from std_msgs.msg import Float64MultiArray
from nrs_imitation import inference_core as core
from nrs_imitation.e2_providers import TimedActions, export_episode, hardware_blockers, rule_actions
from nrs_imitation.inference_metrics import InferenceMetrics
from test_execution_metrics import FakeNode, policy  # actual small FLOW fixture, no ROS init

ROOT = Path(__file__).resolve().parents[4]
E2 = ROOT/'experiments/e2_rule_replay_20260920'


def fixture():
    return json.loads((E2/'rule_synthetic_fixture.json').read_text())


def test_rule_geometry_endpoints_speed_phase_and_time_budget():
    f=fixture();a=rule_actions(f['task'],f['recipe'])
    assert np.all(np.diff(a.time)>0)
    assert np.all(np.abs(a.action[:,0])<=1) and np.all(a.action[:,1:3]==0)
    assert a.time[-1]==8.5 and a.action[0,8]==a.action[-1,8]==0
    assert a.action[:,0].max()==1 and a.action[-1,0]==-1
    assert np.max(np.abs(np.diff(a.action[:,0])/np.diff(a.time)))<=1.0001
    assert set(a.phase)=={'force_ramp_in','processing','force_ramp_out'}
    for boundary in [.5,4.25,8.]:
        i=np.where(a.time==boundary)[0][0]
        assert abs((a.action[i+1,0]-a.action[i,0])/(a.time[i+1]-a.time[i]))<.01
    with pytest.raises(ValueError,match='budget'):
        rule_actions(f['task'],dict(f['recipe'],task_time_budget_s=8.))
    with pytest.raises(ValueError,match='line'):
        rule_actions(dict(f['task'],task_type='two_point'),f['recipe'])


def test_rule_explicit_approach_precedes_force_and_respects_peak_rates():
    from scipy.spatial.transform import Rotation
    f=fixture()
    f['recipe']['approach']=dict(waypoints_pose6=[[-1.,0.,4.,0.,0.,.1],[-1.,0.,0.,0.,0.,0.]],
        peak_speed_mm_s=2.,peak_angular_speed_rad_s=.1)
    f['recipe']['task_time_budget_s']=20.
    a=rule_actions(f['task'],f['recipe'])
    assert a.metadata['approach_duration_s']==3.75
    np.testing.assert_array_equal(a.action[0,:3],[-1.,0.,4.])
    assert np.all(a.action[a.phase=='approach',6:]==0)
    assert a.time[-1]==12.25 and np.all(np.diff(a.time)>0)
    at=np.where(a.time==3.75)[0][0]
    assert a.phase[at]=='force_ramp_in' and a.action[at,8]==0
    np.testing.assert_array_equal(a.action[at,:6],[-1.,0.,0.,0.,0.,0.])
    dt=np.diff(a.time);linear=np.linalg.norm(np.diff(a.action[:,:3],axis=0),axis=1)/dt
    rot=Rotation.from_rotvec(a.action[:,3:6]);angular=(rot[:-1].inv()*rot[1:]).magnitude()/dt
    assert max(linear)<=2.001 and max(angular)<=.101
    with pytest.raises(ValueError,match='budget'):
        rule_actions(f['task'],dict(f['recipe'],task_time_budget_s=10.))
    f['recipe']['approach']['waypoints_pose6'][-1][2]=1.
    with pytest.raises(ValueError,match='first rule working pose'):
        rule_actions(f['task'],f['recipe'])


def test_real_rule_candidate_has_no_initial_jump_and_is_still_offline_only():
    candidate=E2/'rule_validation_candidate/config.R_validation_candidate.json'
    c=json.loads(candidate.read_text());a=rule_actions(c['task'],c['recipe'])
    np.testing.assert_allclose(a.action[0,:6],c['common']['demo_start_pose6'],atol=1e-5)
    assert a.metadata['approach_duration_s']>0 and a.time[-1]<c['common']['task_time_budget_s']
    assert c['recipe']['simulation_only'] and not c['recipe']['main_experiment_ready']
    reasons=hardware_blockers(c,'rule',True)
    assert len(reasons)==1 and 'offline-only' in reasons[0]
    # In-memory TEST fixture only: ensure no hidden blockers after explicit
    # operator authorization. The saved config remains blocked.
    c['recipe']['simulation_only']=False
    assert hardware_blockers(c,'rule',True)==[]
    c['recipe']['approach']['waypoints_pose6'][0][2]+=10.
    c['recipe']['task_time_budget_s']=c['common']['task_time_budget_s']=100.
    assert any('demo_start' in r for r in hardware_blockers(c,'rule',True))


def test_long_finite_plan_is_saved_exactly_without_csv_queue_burst(tmp_path):
    c=json.loads((E2/'rule_validation_candidate/config.R_validation_candidate.json').read_text())
    n=FakeNode(tmp_path,force_on=True);n.execution_method='rule';n._e2_context={'offline_test':True}
    n._e2_actions=rule_actions(c['task'],c['recipe'])
    n.normalize_qpos_enabled=n.denorm_action_enabled=False
    n.params.update(metrics_queue_size=128,metrics_snapshot_enable=False)
    n._metrics=InferenceMetrics(n)
    assert n._metrics.recorder.ready.wait(10.)
    n._on_infer_timer();assert not n.errors
    expected=n.plans[-1].seq_den.copy()
    n._metrics.close()
    recorder=n._metrics.recorder
    assert not recorder.write_errors and not any(v['dropped'] for v in recorder.counts.values())
    assert recorder.counts['plan_snapshot']['written']==2
    assert recorder.counts.get('commands',{}).get('received',0)==0
    for kind,array in [('provider_prediction',n._e2_actions.action),('postprocessed',expected)]:
        with np.load(recorder.path/'prepared_plans'/('000001_'+kind+'.npz'),allow_pickle=False) as data:
            np.testing.assert_array_equal(data['action'],array)
            np.testing.assert_array_equal(data['time'],n._e2_actions.time)
            np.testing.assert_array_equal(data['phase'],n._e2_actions.phase)


def test_slerp_short_arc_and_exact_knots():
    a=np.zeros((2,9));a[:,5]=np.deg2rad([179.,-179.]);a[:,8]=[2.,8.]
    t=TimedActions([0.,1.],a,['unknown']*2,{'frame':'stain_relative_v1'})
    mid,_=t.sample([.5])
    assert abs(abs(mid[0,5])-np.pi)<1e-6 and mid[0,8]==5
    np.testing.assert_array_equal(t.sample(t.time)[0],t.action)
    with pytest.raises(ValueError):t.sample([1.1])
    with pytest.raises(ValueError):TimedActions([0.,0.],a,['unknown']*2,{'frame':'stain_relative_v1'})


def test_all_real_templates_unique_time_force_and_no_normalization():
    import h5py
    inv=json.loads((E2/'inventory.json').read_text())
    paths=list((E2/'templates').glob('*_candidate.npz'))
    assert len(paths)==38
    for path in paths:
        t=TimedActions.load(path)
        assert Path(t.metadata['source_episode']).name in inv['train']
        assert not t.metadata['selected_success'] and t.metadata['smoothing']=='none'
        with h5py.File(t.metadata['source_episode']) as f:
            expected=np.column_stack([f['action/position'][:],f['action/force'][:]])
        np.testing.assert_array_equal(t.action,expected)
        np.testing.assert_array_equal(t.sample(t.time)[0],expected)
        replay=t.resample(125)
        assert t.time[-1]==replay.time[-1]
        assert np.all(np.diff(replay.time)>0)
    with pytest.raises(ValueError,match='split'):
        export_episode(ROOT/inv['dataset']/'episode_35.hdf5',inv['train'])


def baseline_infer():
    source=subprocess.check_output(['git','show','a556c58333f53c265a1939a1a8e88b48242652ed:behavior_ws/src/nrs_imitation/nrs_imitation/inference_core.py'],cwd=ROOT,text=True)
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='NodeCmdMotionInfer')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_on_infer_timer')
    namespace=dict(vars(core));exec(compile(ast.Module(body=[method],type_ignores=[]),'pre_e2_inference','exec'),namespace)
    return namespace['_on_infer_timer']


def test_c_identical_to_pre_e2_method_with_same_flow_rng_and_sent_values(tmp_path,policy):
    outputs=[]
    for old in [True,False]:
        n=FakeNode(tmp_path,policy,True)
        n._pose6=np.array([420.,530.,200.,.01,.02,.03],np.float32)
        n._force=np.array([2.,-3.,7.],np.float32)
        n._force_hist=deque([n._force.copy() for _ in range(30)])
        before=torch.random.get_rng_state().clone()
        if old:baseline_infer()(n)
        else:
            n._e2_context={'offline_regression':True}
            n._metrics=InferenceMetrics(n)
            n._on_infer_timer()
        assert not n.errors
        assert torch.equal(before,torch.random.get_rng_state())
        outputs.append((n.plans[-1].seq_den.copy(),n._publish_cmd(n.plans[-1].seq_den[0])))
        if n._metrics:n._metrics.close()
    for a,b in zip(*outputs):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('method',['rule','replay','il'])
@pytest.mark.parametrize('termination',['manual_abort','timeout','safety_stop','error'])
def test_shared_transform_force_gate_limit_logging_and_cancel(tmp_path,method,termination):
    n=FakeNode(tmp_path,force_on=True);n.execution_method=method;n._e2_context={'test':True}
    n.params['metrics_sample_hz']=0.;n.params['metrics_snapshot_enable']=False
    n.normalize_qpos_enabled=n.denorm_action_enabled=False;n.fz_hard_limit=11.
    a=np.tile(np.array([20.,30.,200.,0.,0.,.1,3.,4.,17.],np.float32),(8,1))
    n._e2_actions=TimedActions(np.arange(8)/30.,a,['unknown']*8,{'frame':'stain_relative_v1'}) if method!='il' else None
    if method=='il':
        n.policy=NS(sample_action=lambda **kw:torch.from_numpy(a[None]))
    n._metrics=InferenceMetrics(n)
    n._on_pose(Float64MultiArray(data=[420.,530.,200.,0.,0.,.1]))
    n._on_force(Float64MultiArray(data=[2.,-3.,7.,.1,.2,.3]))
    n._on_infer_timer()
    p=n.plans[-1].seq_den
    np.testing.assert_array_equal(p[:,:2],np.tile([420.,530.],(8,1)))
    np.testing.assert_array_equal(p[:,3:6],a[:,3:6])
    assert np.all(p[:,6:8]==0) and np.all(p[:,8]==11.)
    requests=[]
    class Future:
        def result(self):return NS(success=True,message='MOCK queue cancelled; physical stop unknown')
        def add_done_callback(self,callback):callback(self)
    def call(req):requests.append(req);return Future()
    n._ptp9d_client=NS(service_is_ready=lambda:True,call_async=call)
    n._ptp9d_stream_started=True;n._ptp9d_stream_force_inflight=False
    n._ptp9d_stream_last_sent_contact=None;n._ptp9d_stream_last_sent_fz=None;n._ptp9d_stream_last_force_send_t=0.
    for contact in [False,True]:
        n._contact=contact;n._ptp9d_stream_update_force()
    assert [r.target_pose[2] for r in requests]==[0.,11.]
    with pytest.raises(RuntimeError,match='completion'):n._e2_finish('normal_completion')
    n._e2_finish(termination)
    n._on_infer_timer();n._on_control_timer()
    assert not n.plans and not n._force_hist and requests[-1].command_mode=='PTP9D_STREAM_STOP'
    n._metrics.close();path=n._metrics.recorder.path
    with (path/'commands.csv').open() as f:rows=list(csv.DictReader(f))
    predicted=[r for r in rows if r['command_stage'] in ('provider_prediction','policy_prediction')]
    assert all(float(r['fz'])==17. for r in predicted)
    with (path/'wrench.csv').open() as f:measured=list(csv.DictReader(f))
    assert float(measured[0]['fz'])==7. and float(measured[0]['tz'])==.3
    assert not n._metrics.recorder.write_errors
    events=[json.loads(l) for l in (path/'events.jsonl').read_text().splitlines()]
    assert any(e['event']==termination for e in events)
    assert not any(e['event']=='normal_completion' for e in events)


def test_hardware_fail_closed_even_with_explicit_enable():
    config=json.loads((E2/'config.json').read_text())
    config['common'].update(force_frame_sign_verification=None,task_time_budget_s=None)
    for method in ['il','rule','replay']:
        reasons=hardware_blockers(config,method,True)
        assert not any('service_stream has no' in r for r in reasons)
        assert any('force_frame_sign_verification' in r for r in reasons)
        assert any('task_time_budget_s' in r for r in reasons)
    config['common']['stain_canon_enable']=True
    assert any('stain_canon' in r for r in hardware_blockers(config,'replay',True))


def test_launch_preflight_before_any_node_and_legacy_default_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path/'launch_logs'))
    import importlib.util
    from launch import LaunchContext
    path=ROOT/'behavior_ws/src/nrs_imitation/launch/inference_clean_single_cam.launch.py'
    spec=importlib.util.spec_from_file_location('e2_launch_check',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    context=LaunchContext()
    context.launch_configurations.update(execution_method='il',e2_config='',e2_enable_hardware='false')
    assert module._e2_preflight(context)==[]
    context.launch_configurations.update(execution_method='replay',e2_config=str(E2/'config.json'),e2_enable_hardware='true')
    with pytest.raises(RuntimeError,match='before launching nodes'):
        module._e2_preflight(context)
    # No LaunchService / ROS context / executors were instantiated.


@pytest.mark.parametrize('method',['rule','replay','il'])
def test_new_plan_wire_contract_transforms_once_and_core_never_tracks(tmp_path,method):
    n=FakeNode(tmp_path,force_on=True);n.execution_method=method
    n._e2_transport=True;n._e2_stopped=False;n._e2_executor_state='ready'
    n._e2_executor_receipt=core._monotonic();n.e2_session_id='mock-session'
    n._e2_config_hash='mock-config';n._e2_clock_id='mock-clock'
    n.normalize_qpos_enabled=n.denorm_action_enabled=False
    a=np.tile(np.array([20.,30.,200.,0.,0.,.1,0.,0.,7.],np.float32),(8,1))
    n._e2_actions=TimedActions(np.arange(8)/30.,a,['phase']*8,{'frame':'stain_relative_v1'}) if method!='il' else None
    if method=='il':n.policy=NS(sample_action=lambda **kw:torch.from_numpy(a[None]))
    wire=[];n._e2_plan_pub=NS(publish=lambda msg:wire.append(json.loads(msg.data)))
    n._on_pose(Float64MultiArray(data=[420.,530.,200.,0.,0.,.1]))
    n._on_force(Float64MultiArray(data=[2.,-3.,7.,.1,.2,.3]))
    n._on_infer_timer()
    assert len(wire)==1 and wire[0]['session_id']=='mock-session' and wire[0]['frame']=='robot_base'
    np.testing.assert_array_equal(np.asarray(wire[0]['action'])[:,:2],np.tile([420.,530.],(8,1)))
    assert wire[0]['final']==(method!='il') and wire[0]['reference_xy_mm']==[400.,500.]
    # Tracking belongs exclusively to the other process once alignment finished.
    before=list(n.sent_messages);n._on_control_timer();assert n.sent_messages==before
