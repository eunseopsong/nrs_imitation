#!/usr/bin/env python3
"""E2 inventory, template export, preview, preflight and hardware-free dry runs."""
import argparse
import csv
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'behavior_ws/src/nrs_imitation'), str(ROOT/'behavior_ws/src/stain_relative_frame'), str(ROOT/'source')]
import numpy as np
from nrs_imitation.e2_providers import (TimedActions, export_episode, file_hash,
    hardware_blockers, load_config, rule_actions)

OUT = ROOT/'experiments/e2_rule_replay_20260920'
DATA = ROOT/'datasets/polishing/single_cam/20260910_90deg_rel_single/imitation_form'
CKPT = ROOT/'checkpoints/flow/polishing/single_cam/e1_force_observation_20260916/on/20260916_1531/policy_best.ckpt'
BASELINE_COMMIT = 'a556c58333f53c265a1939a1a8e88b48242652ed'


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module.replace('numpy._core', 'numpy.core'), name)


def stats():
    with CKPT.with_name('dataset_stats.pkl').open('rb') as f:
        return CompatUnpickler(f).load()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False,
        default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


def train_split():
    # Same lexicographic file ordering and local Generator as make_loaders.
    files = sorted(DATA.glob('episode_*.hdf5'))
    order = np.arange(len(files)); np.random.default_rng(0).shuffle(order)
    split = min(max(1, round(.9*len(files))), len(files)-1)
    return files, [files[i].name for i in order[:split]], [files[i].name for i in order[split:]]


def audit():
    import h5py
    from PIL import Image, ImageDraw
    files, train, val = train_split(); st = stats()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, images, train_actions = [], [], []
    for p in files:
        with h5py.File(p, 'r') as f:
            n = len(f['action/position'])
            source, source_ep = str(f.attrs['source_h5']), str(f.attrs['source_episode'])
            with h5py.File(source, 'r') as raw:
                g = raw['episodes/'+source_ep]; t = g['sample_time_unix'][:]
                dt = np.diff(t)
            rows.append(dict(episode_id=p.stem, split='train' if p.name in train else 'validation',
                timestep_count=n, original_duration_s=float(t[-1]-t[0]),
                timestamp_key='source sample_time_unix', timestamp_monotonic=bool(np.all(dt > 0)),
                max_dt_s=float(dt.max()), success_label='unknown', source_episode=source_ep,
                source_h5=source, sha256=file_hash(p)))
            # All candidates, no automatic successful-episode selection.
            tile = Image.new('RGB', (424, 270), 'white')
            tile.paste(Image.fromarray(f['observations/images/cam0'][0]), (0, 30))
            ImageDraw.Draw(tile).text((8, 8), f'{p.stem} | {rows[-1]["split"]} | success UNKNOWN', fill='black')
            images.append(tile)
            if p.name in train:
                train_actions.append(np.column_stack([f['action/position'][:], f['action/force'][:]]))
    all_a = np.concatenate(train_actions)
    extrema_match = np.array_equal(all_a.min(0), st['action_min']) and np.array_equal(all_a.max(0), st['action_max'])
    if not extrema_match or len(all_a) != st['num_total_timesteps']:
        raise RuntimeError('Reconstructed C train split does not match saved normalizer/count')
    provenance = dict(dataset=str(DATA), checkpoint=str(CKPT), checkpoint_sha256=file_hash(CKPT),
        normalizer=str(CKPT.with_name('dataset_stats.pkl')), normalizer_sha256=file_hash(CKPT.with_name('dataset_stats.pkl')),
        seed=0, train=train, validation=val, train_timesteps=len(all_a),
        split_status='reconstructed from exact loader order/seed; extrema and count match; historical explicit IDs absent',
        normalizer_extrema_match=extrema_match, policy_config=st['policy_config'],
        relative_transform_version=st['relative_transform_version'],
        task_type='line', task_evidence='42 episode first-frame contact sheets; long single black line in inspected frames',
        episodes=rows, git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_json(OUT/'inventory.json', provenance)
    with (OUT/'episodes.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for page in range(3):
        sheet=Image.new('RGB',(424*4,270*4),(220,220,220))
        for i,tile in enumerate(images[page*16:(page+1)*16]): sheet.paste(tile,((i%4)*424,(i//4)*270))
        sheet.save(OUT/f'episodes_preview_{page+1}.jpg',quality=85)
    print(json.dumps(dict(total=len(files),train=len(train),validation=val,extrema_match=extrema_match,output=str(OUT)),ensure_ascii=False))


def export(args):
    inventory=load_config(OUT/'inventory.json')
    names = inventory['train'] if args.all_candidates else [args.episode+'.hdf5']
    exported=[]
    for name in names:
        p=DATA/name
        actions=export_episode(p,inventory['train'], selected_success=bool(args.success_evidence), evidence=args.success_evidence)
        actions.metadata.update(normalizer_sha256=inventory['normalizer_sha256'],
            calibration_id='source xyz_correction_mm + stain_relative_v1; live deployment unverified',
            export_code_sha256=file_hash(ROOT/'behavior_ws/src/nrs_imitation/nrs_imitation/e2_providers.py'))
        suffix='_selected' if args.success_evidence else '_candidate'
        path=OUT/'templates'/(p.stem+suffix+'.npz')
        actions.save(path);exported.append(str(path))
    print(json.dumps(dict(exported=exported,selected_success=bool(args.success_evidence))))


def preview(args):
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=TimedActions.load(args.template);b=a.resample(125.)
    fig,axes=plt.subplots(3,2,figsize=(12,10))
    axes[0,0].plot(a.action[:,0],a.action[:,1],label='source/export')
    axes[0,0].plot(b.action[:,0],b.action[:,1],'--',label='replay 125Hz')
    axes[0,0].set(xlabel='relative base X (mm)',ylabel='relative base Y (mm)',aspect='equal')
    for ax,idx,label in [(axes[0,1],2,'base Z (mm)'),(axes[1,0],8,'training target Fz (N; live frame unverified)'),(axes[1,1],5,'rotvec Z (rad)')]:
        ax.plot(a.time,a.action[:,idx],label='source/export');ax.plot(b.time,b.action[:,idx],'--',label='replay 125Hz')
        ax.set(xlabel='original elapsed time (s)',ylabel=label)
    for ax in axes[:2].flat: ax.legend();ax.grid(alpha=.3)
    with h5py.File(a.metadata['source_episode'], 'r') as f:
        for ax, index, title in [(axes[2,0],0,'source first frame'),(axes[2,1],-1,'source last frame')]:
            ax.imshow(f['observations/images/cam0'][index]);ax.set_title(title+' (camera pose differs)');ax.axis('off')
    fig.suptitle(f'{Path(args.template).stem}: success unconfirmed; no time scaling / smoothing')
    fig.tight_layout(); path=OUT/(Path(args.template).stem+'_preview.png');fig.savefig(path,dpi=140);plt.close(fig)
    write_json(path.with_suffix('.json'),dict(source_count=len(a.time),resampled_count=len(b.time),
        source_duration_s=float(a.time[-1]),resampled_duration_s=float(b.time[-1]),
        knot_reconstruction_max_abs=float(np.max(np.abs(a.sample(a.time)[0]-a.action))),
        smoothing='none',scaling=1.,orientation_interpolation='SO(3) Slerp',source=a.metadata))
    print(path)


def make_configs():
    if (OUT/'config.json').exists() or (OUT/'trial_manifest.csv').exists():
        raise FileExistsError('Preserve prepared/operator settings; make-configs is initial setup only')
    st=stats();inv=load_config(OUT/'inventory.json')
    config=dict(schema='E2_preparation_v1',task=dict(task_type='line',direction_label_deg=90,
        endpoints_relative_mm=None,orientation_rotvec_rad=None,line_width_mm=None,effective_contact_width_mm=None),
        common=dict(specimen_id='default',region_id=None,block_id=None,repeat_id=None,
            use_stain_mask=False,stain_canon_enable=False,frame='stain_relative_v1',
            reference_detection='existing stain_origin_online frozen translation only',
            rpm_setpoint=None,rpm_measured=None,spindle_control_procedure=None,
            surface_normal_base=None,tcp_calibration_id=None,force_frame_sign_verification=None,
            frozen_roi=None,workspace_limits=None,task_time_budget_s=None,approach_retract_protocol=None,
            controller_deployment_id=None,demo_start_pose6=st['demo_start_pose_mean'].tolist(),
            metrics_sample_hz=20,metrics_extra_telemetry_enable=False,video_auto_record=True,viewers=False),
        recipe=dict(simulation_only=True,force_reference_N=None,feed_speed_mm_s=None,pass_count=None,
            offset_spacing_mm=0.,force_ramp_s=None,task_time_budget_s=None),
        replay=dict(template=None,source_dataset=str(DATA),episode_id=None,success_evidence=None,
            time_axis='source sample_time_unix',resample_hz=125.,smoothing='none',scaling=1.),
        il=dict(checkpoint=str(CKPT),checkpoint_sha256=inv['checkpoint_sha256'],
            normalizer_sha256=inv['normalizer_sha256'],use_force_observation=True,force_action=True,
            seed=0,flow_infer_steps=10,chunk_size=128,replan_interval_steps=120,
            trajectory_hz=30.,force_history_len=30,inference_mode='service_stream'))
    # Synthetic-only fixtures deliberately detached from any robot recipe.
    fixture=dict(task_type='line',endpoints_relative_mm=[[-1.,0.,0.],[1.,0.,0.]],orientation_rotvec_rad=[0.,0.,0.])
    recipe=dict(simulation_only=True,force_reference_N=1.,feed_speed_mm_s=1.,pass_count=2,
        force_ramp_s=.5,offset_spacing_mm=0.,task_time_budget_s=9.,
        source='dimensioned unit-test fixture ONLY; not an experimental recipe or tuned force')
    write_json(OUT/'config.json',config);write_json(OUT/'rule_synthetic_fixture.json',dict(task=fixture,recipe=recipe))
    rng=random.Random(20260920);rows=[]
    for block in range(1,4):
        methods=['R','T','C'];rng.shuffle(methods)
        for method in methods:
            rows.append(dict(planned_order=len(rows)+1,method=method,block=block,repeat=block,task='line_90deg',
                specimen_id='default',region_id='',config_sha256=file_hash(OUT/'config.json'),
                order_seed=20260920,actual_run_id='',status='planned',termination_reason='',
                stain_reset_confirmed='',rpm_confirmed='',operator_notes=''))
    with (OUT/'trial_manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(OUT/'config.json')


def load_c_policy(node):
    import torch
    from models.flow_core import FlowRGBPolicy
    from nrs_imitation import inference_core as core
    cfg=dict(stats()['policy_config']);cfg['pretrained_backbone']=False;cfg['dino_checkpoint_path']=''
    torch.set_num_threads(2)
    policy=FlowRGBPolicy(cfg).eval()
    obj=torch.load(CKPT,map_location='cpu',weights_only=False)
    sd=obj.get('model_state_dict',obj.get('state_dict',obj))
    core._load_state_dict_strict_compat(policy,sd)
    node.policy=policy;node.stats=core._load_dataset_stats(str(CKPT.parent));node.ckpt_dir=str(CKPT.parent)
    node.chunk_size=128;node.resize_hw=0;node.flow_infer_steps=10;node.force_history_len=30
    node._metrics_checkpoint_config=cfg


def dry_run(args):
    """Actual C sampler / native R,T, actual shared postprocess + timed engine,
    in-memory ROS messages/services ONLY. No ROS context/executor/driver launch.
    """
    import h5py
    import torch
    from types import SimpleNamespace as NS
    from std_msgs.msg import Float64MultiArray
    from nrs_imitation.e2_offline import OfflineNode
    from nrs_imitation.inference_metrics import InferenceMetrics
    from nrs_imitation import inference_core as core
    started=time.monotonic()
    n=OfflineNode(OUT/'dry_runs',force_on=True)
    n.execution_method=args.method;n._e2_context=load_config(args.config)
    n.metrics_run_tag='E2_TIMED_OFFLINE_'+{'rule':'R','replay':'T','il':'C'}[args.method]
    n.params.update(execution_method=args.method,metrics_sample_hz=0.,metrics_snapshot_enable=False)
    n._e2_actions=None
    if args.method=='rule':
        # A configured R dry-run must exercise its real geometry and approach.
        # The tiny synthetic fixture is available only when explicitly selected.
        rule_config = (load_config(OUT/'rule_synthetic_fixture.json')
                       if args.synthetic_rule_fixture else n._e2_context)
        n._e2_actions=rule_actions(rule_config['task'],rule_config['recipe'])
        n.stats=None;n.ckpt_dir='';n.normalize_qpos_enabled=n.denorm_action_enabled=False
    elif args.method=='replay':
        if not args.template: raise ValueError('Explicit --template required; no default episode selection')
        n._e2_actions=TimedActions.load(args.template).resample(125.)
        n.stats=None;n.ckpt_dir='';n.normalize_qpos_enabled=n.denorm_action_enabled=False
    else:
        load_c_policy(n)
        # Stored real observation, not an arbitrary blank image, for C preview.
        with h5py.File(DATA/'episode_29.hdf5','r') as f:
            n._img_cam0=f['observations/images/cam0'][0]
            pose=f['analysis/absolute/observations_position'][0]
            force=f['observations/force'][0]
        from stain_relative_frame.relative_frame import RelativeFrameAdapter
        adapter=RelativeFrameAdapter([457.4,375.],use_relative=True)
        n._srf=NS(ready=True,stain_origin=np.array([457.4,375.]),stain_angle=np.pi/2,
            observation=adapter.observation,command=adapter.command)
    if args.method!='il':
        pose=n._e2_actions.action[0,:6].copy();pose[:2]+=np.array([400.,500.])
        force=np.array([2.,-3.,7.])  # synthetic measurement != requested force
    # Reflect the resolved offline settings, especially C's actual 128-step
    # horizon rather than the small fixture's initial 8-step defaults.
    for key in list(n.params):
        if hasattr(n, key) and isinstance(getattr(n, key), (str, bool, int, float)):
            n.params[key] = getattr(n, key)
    n.params.update(execution_method=args.method,metrics_run_tag=n.metrics_run_tag,
        metrics_sample_hz=0.,metrics_snapshot_enable=False)
    n._metrics=InferenceMetrics(n)
    if not n._metrics.recorder.ready.wait(10.) or n._metrics.recorder.path is None:
        raise RuntimeError('Offline logger failed to initialize')
    n._metrics.event('offline_initialization_ready',elapsed_s=time.monotonic()-started,
        model_preloaded=args.method=='il',synthetic_transport=True,physical_robot_used=False)
    n._on_pose(Float64MultiArray(data=[float(x) for x in pose]))
    for _ in range(30):n._on_force(Float64MultiArray(data=[float(x) for x in force]+[.1,.2,.3]))
    old = None
    if args.method == 'il':
        import ast
        import copy
        from collections import deque
        source = subprocess.check_output(['git','show',BASELINE_COMMIT+':behavior_ws/src/nrs_imitation/nrs_imitation/inference_core.py'],cwd=ROOT,text=True)
        cls = next(x for x in ast.parse(source).body if isinstance(x,ast.ClassDef) and x.name=='NodeCmdMotionInfer')
        method = next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_on_infer_timer')
        namespace = dict(vars(core))
        exec(compile(ast.Module(body=[method],type_ignores=[]),'git_HEAD_infer','exec'),namespace)
        old = copy.copy(n);old.plans=deque();old._metrics=None
        state = torch.random.get_rng_state().clone()
        namespace['_on_infer_timer'](old)
        assert torch.equal(state,torch.random.get_rng_state())
    n._on_infer_timer()
    if not n.plans or n.errors: raise RuntimeError(n.errors or 'No plan generated')
    plan=n.plans[-1]
    if old is not None:
        np.testing.assert_array_equal(old.plans[-1].seq_den, plan.seq_den)
        np.testing.assert_array_equal(old._publish_cmd(old.plans[-1].seq_den[0]), n._publish_cmd(plan.seq_den[0]))
        n._metrics.event('C_regression_pass',reference=BASELINE_COMMIT,
            checkpoint=str(CKPT),seed=0,flow_steps=10,plan_bitwise_equal=True,pre_send_command_bitwise_equal=True)
    np.savez_compressed(n._metrics.recorder.path/'prepared_plan.npz',action=plan.seq_den,
        time=getattr(plan,'elapsed_time_s',np.arange(len(plan.seq_den))/30.))
    # The SAME ROS-free timed engine as the separate hardware executor.
    # Virtual feedback follows the previous command exactly; no physical model.
    from nrs_imitation.e2_timed_execution import TimedExecution, TimedPlan, executor_settings
    from nrs_imitation.execution_metrics import stamp, pose_fields
    import copy
    cfg=copy.deepcopy(n._e2_context)
    times=getattr(plan,'elapsed_time_s',np.arange(len(plan.seq_den))/30.)
    if args.method=='il' or (args.method=='rule' and args.synthetic_rule_fixture):
        cfg['common']['task_time_budget_s']=float(times[-1])+1.  # explicit preview only
    engine=TimedExecution(executor_settings(cfg))
    engine.accept(TimedPlan(1,10.,times,plan.seq_den,
        getattr(plan,'provider_phase',np.full(len(times),'unknown')),args.method!='il'),10.)
    engine.start(10.,pose)
    feedback=np.asarray(pose,float).copy()
    samples=0;timings=[];end=None
    for i in range(int(np.ceil(times[-1]/.008))+2):
        now=10.+i*.008
        before=time.perf_counter()
        result=engine.tick(now,feedback,np.array([2.,-3.,7.]),0.,0.)
        if 'end' in result:
            end=result['end'];break
        samples+=1
        for stage,key in [('time_sampled','requested'),('contact_gated','gated'),('node_sent','sent')]:
            row=dict(stamp(None,source='offline_virtual_executor'),command_stage=stage,
                command_mode='cmdMotion',command_id=i+1,inference_id=1,plan_id=1,
                action_index=result['action_index'],semantic_frame='robot_base',
                position_unit='mm',orientation_unit='rotvec_rad',force_unit='N',
                execution_status='OFFLINE_ONLY_no_publication',raw_values=result[key].tolist(),
                details=dict(virtual_elapsed_s=now-10.,phase=result['phase'],contact=result['contact']))
            row.update(zip(['x','y','z','rx','ry','rz','fx','fy','fz'],result[key].tolist()))
            n._metrics.recorder.emit('commands',row)
        feedback=result['sent'][:6].copy()
        timings.append(time.perf_counter()-before)
        if i%6==0:
            n._on_pose(Float64MultiArray(data=feedback.tolist()))
            n._on_force(Float64MultiArray(data=[2.,-3.,7.,.1,.2,.3]))
        # Offline loop runs faster than wall time: keep a bounded writer queue
        # without changing real executor's nonblocking behavior.
        if n._metrics.recorder.queue.qsize()>6000:
            time.sleep(.01)
    engine.stop();n.plans.clear();n._force_hist.clear()
    n._metrics.event('offline_timed_execution_end',engine_end=end,virtual_samples=samples,
        processing_started=False,physical_robot_used=False,physical_stop_verified=False)
    n._metrics.close()
    if n.errors or n._metrics.recorder.write_errors:raise RuntimeError(n.errors or n._metrics.recorder.write_errors)
    path=n._metrics.recorder.path
    write_json(path/'offline_result.json',dict(method=args.method,synthetic_transport=True,
        real_checkpoint_loaded=args.method=='il',transport='e2_timed_topic_v1',
        engine_end=end,virtual_samples=samples,provider_duration_s=float(times[-1]),
        buffer_empty=not n.plans,physical_stop_verified=False,
        sample_and_enqueue_time_ms=dict(p50=float(np.percentile(timings,50)*1000),
            p99=float(np.percentile(timings,99)*1000),max=float(max(timings)*1000)),
        warning='Virtual clock/feedback, no ROS I/O; CPU timing is not a hardware deadline guarantee'))
    print(path)


def check_logs(args):
    root=Path(args.path);report={};errors=[]
    meta=load_config(root/'metadata.json')
    for name in ['tcp_pose','wrench','commands']:
        with (root/(name+'.csv')).open() as f: rows=list(csv.DictReader(f))
        times=[int(r['receipt_monotonic_ns']) for r in rows if r['receipt_monotonic_ns']]
        by_source={}
        for r in rows:
            by_source.setdefault(r['source'],[]).append(int(r['receipt_monotonic_ns']))
        report[name]=dict(rows=len(rows),invalid=sum(r['validity']!='valid' for r in rows),
            monotonic_by_source=all(np.all(np.diff(t)>=0) for t in by_source.values()),
            max_receipt_gap_s_by_source={k:float(max(np.diff(v),default=0)/1e9) for k,v in by_source.items()},
            sources=list(by_source))
        if not rows or not report[name]['monotonic_by_source']: errors.append(name+' empty or time reversed')
    summary=load_config(root/'summary.json')
    if summary.get('write_error_count') or not summary.get('drained'):
        errors.append('Logger write error or incomplete drain')
    if any(v['dropped'] for v in summary.get('counts',{}).values()):
        errors.append('Logger queue dropped records')
    if any(v['invalid'] for v in report.values()):
        errors.append('Invalid numeric rows')
    report.update(method=meta.get('method'),offline_mock=meta.get('offline_mock'),summary=summary,
        errors=errors,source_clock_sync='unverified',controller_applied_force='unknown',
        surface_normal='unknown',rpm='unknown',processing_interval='unknown unless explicit operator events')
    write_json(root/'quality_check.json',report);print(json.dumps(report,ensure_ascii=False))
    if errors:raise SystemExit(1)


def configure(args):
    """Apply explicitly supplied JSON values only; keep a dated backup."""
    import datetime
    target=Path(args.config)
    cfg=load_config(target)
    for key, raw in args.set:
        parts=key.split('.')
        obj=cfg
        for part in parts[:-1]:
            if part not in obj or not isinstance(obj[part],dict):raise ValueError('Unknown config key: '+key)
            obj=obj[part]
        if parts[-1] not in obj:raise ValueError('Unknown config key: '+key)
        obj[parts[-1]]=json.loads(raw)
    backup=target.with_name(target.name+'.'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.bak')
    backup.write_bytes(target.read_bytes())
    write_json(target,cfg)
    print(json.dumps(dict(saved=str(target),backup=str(backup),hardware_started=False,
        remaining={m:hardware_blockers(cfg,m,True) for m in ['rule','replay','il']}),ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('audit');sub.add_parser('make-configs')
    c=sub.add_parser('configure');c.add_argument('--config',default=str(OUT/'config.json'))
    c.add_argument('--set',nargs=2,action='append',required=True,metavar=('KEY','JSON'))
    e=sub.add_parser('export');sel=e.add_mutually_exclusive_group(required=True)
    sel.add_argument('--episode');sel.add_argument('--all-candidates',action='store_true')
    e.add_argument('--success-evidence')
    v=sub.add_parser('preview');v.add_argument('--template',required=True)
    for name in ('dry-run','preflight'):
        d=sub.add_parser(name);d.add_argument('--method',choices=['rule','replay','il'],required=True)
        d.add_argument('--config',default=str(OUT/'config.json'));d.add_argument('--enable-hardware',action='store_true')
        if name=='dry-run':
            d.add_argument('--template')
            d.add_argument('--synthetic-rule-fixture', action='store_true',
                help='Explicitly test the small synthetic rule instead of --config geometry')
    l=sub.add_parser('check-logs');l.add_argument('path')
    args=p.parse_args()
    if args.command=='audit':audit()
    elif args.command=='make-configs':make_configs()
    elif args.command=='configure':configure(args)
    elif args.command=='export':
        if args.all_candidates and args.success_evidence:p.error('Select a specific episode before certifying success')
        export(args)
    elif args.command=='preview':preview(args)
    elif args.command=='dry-run':
        if args.enable_hardware:p.error('dry-run never accepts --enable-hardware')
        dry_run(args)
    elif args.command=='check-logs':check_logs(args)
    elif args.command=='preflight':
        errors=hardware_blockers(load_config(args.config),args.method,args.enable_hardware)
        print(json.dumps(dict(hardware_allowed=not errors,blockers=errors),ensure_ascii=False,indent=2))
        if errors:raise SystemExit(2)


if __name__=='__main__':main()
