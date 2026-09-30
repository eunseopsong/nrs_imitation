"""E2 tests are all CPU/filesystem only; no ROS context, service or publisher."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nrs_imitation.e2_ablation import (EXPERIMENT, select_condition, execution_contract, object_hash,
    motion_to_contract, scheduled_force, model_errors, cohort_id, digest)
from nrs_imitation.e2_timed_execution import TimedExecution, TimedPlan, executor_settings, ExecutionFault
from nrs_imitation.e2_force_analysis import (Unavailable, align, normal_force, evaluate_profile,
    weighted_rmse, metrology, f0_candidate)
from nrs_imitation.execution_metrics import ExecutionRecorder, stamp

ROOT=EXPERIMENT.parents[1]


def config():return json.loads((EXPERIMENT/'config.json').read_text())


def load_script(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def test_contract_same_for_all_conditions():
    cfg=config();values=[]
    for c in 'ABC':
        selected=select_condition(cfg,c)
        values.append(object_hash(execution_contract(selected)))
        assert executor_settings(selected,'il')['c_pose_conditioning']==cfg['motion_postprocessor']
    assert len(set(values))==1
    assert cfg==config()


def test_same_synthetic_commands_after_adapter_for_abc():
    cfg=config();engines=[TimedExecution(executor_settings(select_condition(cfg,c),'il')) for c in 'ABC']
    t=np.arange(128)/30.;a=np.zeros((128,9));a[:,0]=np.sin(t)*12;a[:,8]=11.
    for engine in engines:
        engine.accept(TimedPlan(1,10.,t,a,np.full(128,'unknown'),False),10.)
        engine.start(10.,np.zeros(6))
    pose=np.zeros(6)
    for i in range(1,450):
        now=10+i*.008
        if i==400:
            for e in engines:e.accept(TimedPlan(2,now,t,a,np.full(128,'unknown'),False),now)
        results=[e.tick(now,pose,[0,0,7],0,0) for e in engines]
        for r in results[1:]:np.testing.assert_array_equal(r['sent'],results[0]['sent'])
        pose=results[0]['sent'][:6]
    for e in engines:
        e.stop();assert e.tick(now+.008,pose,[0,0,7],0,0) is None
        with pytest.raises(ExecutionFault):e.accept(TimedPlan(3,now,t,a,np.full(128,'unknown'),False),now)


def test_A_scheduler_and_prediction_separation():
    a=motion_to_contract(np.zeros((128,6)))
    assert a.shape==(128,9) and np.all(a[:,6:]==0)
    assert scheduled_force({},False)==0.
    with pytest.raises(ValueError):scheduled_force({},True)
    f0={'status':'frozen','command_fz_N':13.}
    assert scheduled_force(f0,True)==13. and scheduled_force(f0,False)==0.
    with pytest.raises(ValueError):motion_to_contract(np.zeros((128,9)))


def calibration():
    return dict(normal_force_verified=True,calibration_id='synthetic-test-only',gravity_zero_filter_evidence='synthetic',
        surface_normal_base=[0,1,0],compression_sign=1,rotation_basis='actual_feedback_tcp',max_pose_age_s=.2)


def test_rotation_90deg_physical_equivalence_and_no_double_rotation():
    c=calibration();pose=dict(clock_id='test',pose_kind='actual_feedback',time=[0.,1.],
                            rotvec=[[0,0,np.pi/2]]*2)
    # Supply exact sample times so no interpolation across a long gap is needed.
    f=normal_force([[2,0,0]]*2,'controller_tcp',c,time=[0,1],pose=pose,clock='test')
    base=normal_force([[0,2,0]]*2,'robot_base',c)
    np.testing.assert_allclose(f,base,atol=1e-15)
    with pytest.raises(Unavailable):normal_force([[2],[2]],'robot_base',c)
    with pytest.raises(Unavailable):normal_force([[2,0,0]],'controller_tcp',c,time=[.5],pose=pose,clock='test')
    pose['pose_kind']='command_target'
    with pytest.raises(Unavailable):normal_force([[2,0,0]],'controller_tcp',c,time=[0],pose=pose,clock='test')


def profile_case(variable=True):
    t=np.linspace(0,1,101);f=10+8*np.sin(t*2*np.pi) if variable else np.ones(101)*10
    ref=dict(calibration_hash='test',phases=[dict(phase_id='pass_1',progress=t.tolist(),force_normal_N=f.tolist())])
    signals={k:dict(clock_id='test',time=t.tolist(),values=f.tolist()) for k in ['F_meas','F_tar','F_sent','F_cmd_applied']}
    run=dict(normal_force_verified=True,calibration_hash='test',clock_id='test',
             phases=[dict(phase_id='pass_1',start=0,end=1,complete=True)],signals=signals)
    analysis=dict(max_gap_s=.2,grid_points_per_phase=101)
    return ref,run,analysis


def test_variable_force_exact_profile_zero_despite_raw_variation():
    ref,r,a=profile_case();out=evaluate_profile(ref,r,a)
    assert np.std(r['signals']['F_meas']['values'])>5
    assert out['E_profile_N']==out['E_target_N']==out['E_track_applied_N']==0
    r['signals']['F_meas']['values']=[10.]*101
    assert evaluate_profile(ref,r,a)['E_profile_N']>5


def test_gate_zero_tracking_does_not_imply_skill_success():
    ref,r,a=profile_case(False)
    ref['phases'][0]['force_normal_N']=[20.]*101
    r['signals']['F_tar']['values']=[20.]*101
    for k in ['F_meas','F_sent','F_cmd_applied']:r['signals'][k]['values']=[0.]*101
    out=evaluate_profile(ref,r,a)
    assert out['E_track_applied_N']==0 and out['E_target_N']==0 and out['E_profile_N']==20


@pytest.mark.parametrize('problem',['incomplete','missing','gap','clock','reverse','calibration'])
def test_incomplete_invalid_profiles_are_NA(problem):
    ref,r,a=profile_case()
    if problem=='incomplete':r['phases'][0]['complete']=False
    if problem=='missing':r['phases']=[]
    if problem=='gap':
        r['signals']['F_meas']['time']=[0,1];r['signals']['F_meas']['values']=[10,10]
    if problem=='clock':r['signals']['F_tar']['clock_id']='another_host'
    if problem=='reverse':r['signals']['F_meas']['time'].reverse()
    if problem=='calibration':r['normal_force_verified']=False
    out=evaluate_profile(ref,r,a)
    assert out['E_profile_N'] is None and out['n_profile_valid']==0 and out['reasons']


def test_absent_applied_is_NA_and_sent_has_own_metric():
    ref,r,a=profile_case();del r['signals']['F_cmd_applied']
    out=evaluate_profile(ref,r,a)
    assert out['E_profile_N']==0 and out['E_track_applied_N'] is None and out['E_discrepancy_sent_N']==0


def test_common_clock_no_extrapolation_and_rate_invariance():
    for n in [21,101,501]:
        t=np.linspace(0,1,n);out,valid=align(t,2*t,np.linspace(0,1,51),.2)
        assert valid.all();np.testing.assert_allclose(out,np.linspace(0,2,51),atol=1e-15)
    _,valid=align([0,1],[0,1],[-.1,.5,1.1],.2)
    assert not valid.any()


def test_metrology_negative_reduction_and_limits():
    r=dict(measurement_valid='true',before_depth_um='10',after_depth_um='12',quantification_limit_um='1',
           instrument_settings_id='test',reference_region='test')
    assert metrology(r)['depth_reduction_pct']==-20
    r['after_depth_um']='.5';assert metrology(r)['depth_reduction_pct'] is None
    assert metrology({})['depth_reduction_pct'] is None


def test_f0_never_defaults_to_R_recipe():
    assert config()['f0']['command_fz_N'] is None
    with pytest.raises(Unavailable):f0_candidate({'episodes':[]},calibration(),ROOT,{'train':[]})


def test_logger_collision_flush_and_partial_attempt(tmp_path):
    a=ExecutionRecorder(tmp_path,'same',{},64);b=ExecutionRecorder(tmp_path,'same',{},64)
    for r in (a,b):
        assert r.ready.wait(10)
        r.event('manual_abort',stamp(None),reason='test only');r.close()
    assert a.path!=b.path
    for r in (a,b):
        summary=json.loads((r.path/'summary.json').read_text())
        assert not summary['write_errors']
        assert any(json.loads(line)['event']=='manual_abort' for line in (r.path/'events.jsonl').read_text().splitlines())


def test_attempt_collision_and_preservation_no_launch(tmp_path,monkeypatch):
    cli=load_script(ROOT/'scripts/e2_ablation.py','e2_cli_test')
    monkeypatch.setattr(cli,'session_root',lambda session:tmp_path/session)
    _,a=cli.create_attempt(config(),'B','test',1,'spec1','region1','new_surface')
    _,b=cli.create_attempt(config(),'B','test',1,'spec1','region1','new_surface')
    assert a!=b and (a/'attempt.json').is_file() and (b/'attempt.json').is_file()
    before=(a/'attempt.json').read_bytes();cli.seal_archive(a)
    assert before==(a/'attempt.json').read_bytes()


def test_recipe_model_revision_changes_cohort_but_local_copy_does_not():
    cfg=config();original=cohort_id(cfg)
    cfg['models']['B']['checkpoint_sha256']='different-trained-model'
    assert cohort_id(cfg)!=original
    cfg=config();cfg['f0']['command_fz_N']=12.
    assert cohort_id(cfg)!=original
    cfg=config();cfg['reference']['artifact']='/run/artifacts/reference.json'
    assert cohort_id(cfg)==original


def test_attempt_copies_reference_and_calibration_before_original_changes(tmp_path,monkeypatch):
    cli=load_script(ROOT/'scripts/e2_ablation.py','e2_cli_snapshot_test')
    monkeypatch.setattr(cli,'session_root',lambda session:tmp_path/session)
    ref=tmp_path/'selected_reference.json';ref.write_text('{"test": 1}')
    cfg=config();cfg['reference'].update(artifact=str(ref),sha256=digest(ref))
    selected,folder=cli.create_attempt(cfg,'B','test',1,'spec1','region1','new_surface')
    local=Path(selected['reference']['artifact']);ref.write_text('{"test": 2}')
    assert local.is_relative_to(folder) and digest(local)==cfg['reference']['sha256']
    assert json.loads((folder/'artifacts/calibration.json').read_text())==cfg['calibration']
    assert (folder/'artifacts/code/source/models/flow_core.py').is_file()
    assert json.loads((folder/'artifacts/index.json').read_text())['files']


def test_processing_marker_duration_survives_unavailable_force_reference(tmp_path,monkeypatch):
    cli=load_script(ROOT/'scripts/e2_ablation.py','e2_cli_duration_test')
    monkeypatch.setattr(cli,'session_root',lambda session:tmp_path/session)
    _,folder=cli.create_attempt(config(),'B','test',1,'spec1','region1','new_surface')
    log=folder/'executor/test';log.mkdir(parents=True)
    events=[dict(event=name,receipt_monotonic_ns=int(at*1e9),details=detail) for name,at,detail in [
        ('execution_start',1.,{}),('processing_start',2.,{'source':'operator'}),
        ('processing_end',4.,{'source':'operator'}),('stop_requested',5.,{'reason':'operator_finish'})]]
    (log/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    (log/'wrench.csv').write_text('fz\n1\n')
    out=cli.analyze_run(folder)
    assert out['processing_time_s']==2. and out['elapsed_until_stop_s']==4.
    assert out['E_profile_N'] is None and out['reasons']
    cli.aggregate(folder.parent.parent)
    excluded=json.loads((folder.parent.parent/'exclusion_manifest.json').read_text())
    assert excluded['before']==excluded['after']==1 and excluded['excluded']==0
    assert not excluded['originals_deleted']


def test_tracking_integrates_commands_between_sensor_samples():
    ref,r,a=profile_case(False)
    mt=np.linspace(0,1,11);ct=np.linspace(0,1,21)
    r['signals']['F_meas']=dict(clock_id='test',time=mt.tolist(),values=[0.]*len(mt))
    # 2 N command in the second half of each 0.1 s sensor interval.
    r['signals']['F_cmd_applied']=dict(clock_id='test',time=ct.tolist(),values=[0. if i%2==0 else 2. for i in range(len(ct))])
    assert evaluate_profile(ref,r,a)['E_track_applied_N']==pytest.approx(np.sqrt(2))


def test_raw_transforms_keep_uncovered_edge_commands_in_original_only(tmp_path):
    cli=load_script(ROOT/'scripts/e2_ablation.py','e2_cli_transform_test')
    cfg=config();cfg['calibration']=dict(calibration(),controller_target_frame='controller_tcp')
    log=tmp_path/'executor/test';log.mkdir(parents=True)
    (log/'events.jsonl').write_text('')
    (log/'metadata.json').write_text(json.dumps({'clock_id':'test'}))
    fields=['receipt_monotonic_ns','fx','fy','fz']
    cli.write_csv(log/'wrench.csv',[dict(zip(fields,[int(t*1e9),0,2,0])) for t in [.05,.1,.15]],fields)
    fields=['receipt_monotonic_ns','rx','ry','rz']
    cli.write_csv(log/'tcp_pose.csv',[dict(zip(fields,[int(t*1e9),0,0,0])) for t in [.05,.15]],fields)
    commands=[dict(receipt_monotonic_ns=int(t*1e9),fx=0,fy=2,fz=0,command_stage=stage,details='{"contact": true}')
              for t in [0,.05,.1,.15,.2] for stage in ['time_sampled','node_sent']]
    cli.write_csv(log/'commands.csv',commands)
    result=cli.raw_run(tmp_path,cfg)
    assert result['signals']['F_tar']['values']==[2.]*3
    assert result['transform_limits']['F_sent']['edge_samples_without_pose']==2
    assert len(cli.csv_rows(log/'commands.csv'))==10


def test_partial_csv_is_explicitly_unavailable(tmp_path):
    cli=load_script(ROOT/'scripts/e2_ablation.py','e2_cli_partial_csv_test')
    csv_path=tmp_path/'wrench.csv';csv_path.write_text('t,fx,fy,fz\n0,0,0,1\n1,0,')
    with pytest.raises(Unavailable,match='partial/malformed CSV'):cli.csv_rows(csv_path)


@pytest.mark.parametrize('tag',['184357','184521','184622','184724','184820'])
def test_C_recorded_plans_command_regression(tag):
    old=load_script(EXPERIMENT/'before/behavior_ws/src/nrs_imitation/nrs_imitation/e2_timed_execution.py',
                    'nrs_imitation.e2_before_timed_'+tag)
    base=ROOT/f'results/20260926/E1/C/RTC_C_20260926T{tag}'
    with (base/'provider/commands.csv').open() as f:rows=list(csv.DictReader(f))
    plans={}
    for r in rows:
        if r['command_stage']=='postprocessed':plans.setdefault(int(r['plan_id']),[]).append(r)
    cfg=json.loads((base/'launch_context/config.json').read_text());settings=executor_settings(cfg,'il')
    engines=[old.TimedExecution(settings),TimedExecution(executor_settings(select_condition(config(),'C'),'il'))]
    # Actual archived postprocessed policy chunks, identical feedback and clock
    # under both implementations. This is offline replay, not a new C trial.
    pose=None;count=0
    for seq,(pid,rr) in enumerate(sorted(plans.items())):
        a=np.array([[float(r[k]) for k in ['x','y','z','rx','ry','rz','fx','fy','fz']] for r in rr])
        t=np.arange(len(a))/30.;now=10.+seq*4.
        if pose is None:pose=a[0,:6].copy()
        for e in engines:
            e.accept(TimedPlan(pid,now,t,a,np.full(len(a),'unknown'),False),now)
            if seq==0:e.start(now,pose)
        for i in range(1,501):
            at=now+i*.008;feedback=[0,0,8 if i%70<50 else 0]
            results=[e.tick(at,pose,feedback,0,0) for e in engines]
            for key in ('requested','conditioned','gated','sent'):np.testing.assert_array_equal(results[0][key],results[1][key])
            pose=results[0]['sent'][:6];count+=1
    assert count>=2000
