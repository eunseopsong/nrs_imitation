#!/usr/bin/env python3
"""Offline-first E2 preparation, attempt preservation, and operator run entry."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'behavior_ws/src/nrs_imitation'))
from nrs_imitation.e2_ablation import (EXPERIMENT, digest, object_hash, select_condition,
    execution_contract, readiness, read_stats, model_errors, cohort_id)
from nrs_imitation.e2_force_analysis import (Unavailable, episode_signal, f0_candidate,
    build_reference, normal_force, evaluate_profile, metrology)
import numpy as np
from nrs_imitation.e2_run_context import read, write, identifier, session_root, seal_archive
from nrs_imitation.e2_run_context import create_attempt as _create_attempt


def create_attempt(cfg,condition,session,block=None,specimen=None,region=None,reuse=None):
    return _create_attempt(cfg,condition,session,block,specimen,region,reuse,
                           session_path=session_root(session))

DATASET=ROOT/'datasets/polishing/single_cam/20260910_90deg_rel_single/imitation_form'


def split():
    names=sorted(p.name for p in DATASET.glob('episode_*.hdf5'))
    order=np.arange(len(names));np.random.default_rng(0).shuffle(order);n=round(.9*len(names))
    return dict(train=[names[i] for i in order[:n]],validation=[names[i] for i in order[n:]])


def write_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(rows[0])
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def preview(cfg):
    s=split();rows=[];preview=EXPERIMENT/'teacher_preview';preview.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(preview/'teacher_force_preview_UNCALIBRATED.pdf') as pdf:
        for names in [s['validation'],s['train']]:
            for offset in range(0,len(names),6):
                fig,axs=plt.subplots(3,2,figsize=(10,10));axs=axs.ravel()
                for ax,name in zip(axs,names[offset:offset+6]):
                    source=episode_signal(DATASET/name);t,f=source['time'],source['force']
                    istrain=name in s['train']
                    ax.plot(t,f[:,2]);ax.set(title=f'{name} | '+('train' if istrain else 'validation'),xlabel='Original time (s)',ylabel='Teacher recorded Fz (N)')
                    rows.append(dict(episode=name,split='train' if istrain else 'validation',samples=len(t),duration_s=float(t[-1]),
                        source_frame='teacher_calibrated; live equivalence unverified',success_confirmed=None,
                        processing_annotation=None,**{k:v for k,v in source['provenance'].items() if k!='episode'}))
                for ax in axs[len(names[offset:offset+6]):]:ax.axis('off')
                fig.suptitle('Recorded teacher forces: preview only, no automatic processing/success selection')
                fig.tight_layout(rect=(0,0,1,.96));pdf.savefig(fig);plt.close(fig)
    write_csv(preview/'episodes.csv',rows);write(preview/'split.json',s)
    example={'purpose':'F0_calibration OR evaluation_reference (separate selections)',
        'episodes':[], 'exclusions':[],
        'entry_schema':{'episode':'episode_ID.hdf5','success_confirmed':False,'selection_evidence':None,
            'phases':[{'phase_id':'pass_1','start_s':None,'end_s':None,'evidence':'video/operator annotation; not force threshold'}]}}
    if not (EXPERIMENT/'annotations.template.json').exists():write(EXPERIMENT/'annotations.template.json',example)
    write(EXPERIMENT/'f0_candidate_status.json',{'status':'blocked','normal_force_N':None,'command_fz_N':None,
        'reasons':['No confirmed processing annotation','Teacher/live frame and compression calibration unavailable'],
        'validated_operating_range':None,'E1_R_18N_reused':False})
    print('Preview:',preview,'\nHeld-out episodes:',', '.join(s['validation']))


def seal(cfg,path,evidence):
    if not evidence.strip():raise ValueError('Concrete protocol review evidence required')
    if cfg['protocol']['termination']!='operator_finish' or not cfg['protocol']['phase_ids']:
        raise ValueError('Review common termination and phase IDs first')
    cfg['protocol'].update(reviewed=True,review_evidence=evidence)
    cfg['execution_contract_hash']=object_hash(execution_contract(cfg))
    cfg['cohort_id']=cohort_id(cfg)
    write(path,cfg)


def schedule(cfg,session):
    target=session_root(session);target.mkdir(parents=True,exist_ok=True)
    rng=random.Random(cfg['protocol']['schedule_seed']);items=[]
    for block in range(1,6):
        order=list('ABC');rng.shuffle(order)
        for condition in order:items.append(dict(scheduled_trial=len(items)+1,block_id=block,condition=condition,
            specimen_id=None,region_id=None,surface_reuse=None,status='scheduled_not_attempted'))
    write(target/'schedule.json',dict(experiment_id='E2',session_id=session,schedule_seed=cfg['protocol']['schedule_seed'],
        sampling_seed=cfg['sampling']['seed'],execution_contract_hash=cfg['execution_contract_hash'],trials=items),exclusive=True)
    write_csv(target/'schedule.csv',items);print(target/'schedule.csv')


def reject_recipe_change_after_execution(cfg):
    for p in (ROOT/'results').glob('*/E2/*/[ABC]/*/attempt.json'):
        attempt=read(p)
        if attempt.get('cohort_id')!=cfg['cohort_id']:continue
        # launch_requested/unknown is conservative: logs may not have closed yet.
        if attempt.get('executed') is not False:
            raise ValueError('This cohort has launched attempts. F0 must be fixed before B/C results; use a separately reviewed cohort.')


def run(cfg,args):
    # This branch is called only by an explicit operator run command.
    selected,folder=create_attempt(cfg,args.condition,args.session,args.block,args.specimen,args.region,args.surface_reuse)
    from nrs_imitation.e2_providers import hardware_blockers
    blockers=hardware_blockers(selected,'il',True)
    record=read(folder/'attempt.json')
    if blockers:
        record.update(status='preflight_blocked',executed=False,blockers=blockers)
        write(folder/'attempt.json',record);seal_archive(folder)
        raise SystemExit('Attempt preserved at '+str(folder)+'\n- '+'\n- '.join(blockers))
    record.update(status='launch_requested',executed=None);write(folder/'attempt.json',record)
    env=os.environ.copy();env['ROS_LOG_DIR']=str(folder/'ros_logs')
    command=['ros2','launch','nrs_imitation','e2_abc.launch.py','mode:=run',
        'condition:='+args.condition,'config:='+str(folder/'config.json')]
    write(folder/'launch_command.json',dict(argv=command,ROS_LOG_DIR=env['ROS_LOG_DIR']))
    try:
        process=subprocess.Popen(command,env=env)
        # The terminal delivers SIGINT to ROS launch as well. Keep the wrapper
        # alive while the existing ROS stop/hold/flush path finishes; never use
        # subprocess.call's KeyboardInterrupt kill fallback or a new robot stop.
        while True:
            try:
                status=process.wait();break
            except KeyboardInterrupt:
                print('Waiting for existing ROS stop/hold and log flush...',flush=True)
        record.update(status='launch_exited',launch_exit_code=status,
                      completed=None,closed_at=datetime.now().astimezone().isoformat())
    except BaseException as exc:
        record.update(status='launch_interrupted_or_failed',error=repr(exc));raise
    finally:
        write(folder/'attempt.json',record);seal_archive(folder)
    print('Preserved attempt:',folder,'; use analyze to determine execution/hold/completion from logs')


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise Unavailable('partial/malformed CSV rows: '+str(path))
    return rows


def read_events(path):
    values=[];partial=False
    for line in Path(path).read_text().splitlines():
        try:values.append(json.loads(line))
        except json.JSONDecodeError:partial=True
    return values,partial


def raw_run(folder,cfg):
    dirs=list((folder/'executor').glob('*/events.jsonl'))
    if len(dirs)!=1:raise Unavailable('one executor log required; attempt may have failed before execution')
    ep=dirs[0].parent;events,partial=read_events(ep/'events.jsonl')
    meta=read(ep/'metadata.json');wr=csv_rows(ep/'wrench.csv');pr=csv_rows(ep/'tcp_pose.csv');cr=csv_rows(ep/'commands.csv')
    clock=meta['clock_id'];cal=cfg['calibration'];phases=[];active=None;processing=False
    for e in events:
        t=int(e['receipt_monotonic_ns'])/1e9;name=e['event'];detail=e['details']
        if name=='processing_start':processing=True
        if name=='phase_or_pass_marker' and processing:
            if active:active.update(end=t,complete=True)
            active=dict(phase_id=detail['phase_id'],start=t,end=None,complete=False);phases.append(active)
        if name=='processing_end':
            if active:active.update(end=t,complete=detail.get('source')=='operator')
            active=None;processing=False
    result=dict(clock_id=clock,phases=phases,signals={},normal_force_verified=False,
                calibration_hash=object_hash(cal),partial_event_file=partial,transform_limits={})
    sent=[r for r in cr if r['command_stage']=='node_sent']
    result['gate']=dict(time=[int(r['receipt_monotonic_ns'])/1e9 for r in sent],
                       contact=[json.loads(r['details']).get('contact') for r in sent])
    for rows,stage in [(wr,'F_meas'),([r for r in cr if r['command_stage']=='time_sampled'],'F_tar'),
                       ([r for r in cr if r['command_stage']=='node_sent'],'F_sent')]:
        if not rows:continue
        t=np.array([int(r['receipt_monotonic_ns'])/1e9 for r in rows]);f=np.array([[float(r[k]) for k in ['fx','fy','fz']] for r in rows])
        pose=dict(clock_id=clock,pose_kind='actual_feedback',time=[int(r['receipt_monotonic_ns'])/1e9 for r in pr],
                  rotvec=[[float(r[k]) for k in ['rx','ry','rz']] for r in pr])
        frame='robot_base' if stage=='F_meas' else cal.get('controller_target_frame')
        if frame=='controller_tcp' and len(pose['time'])>=2:
            # Edge commands may precede/follow the 20 Hz pose log. Preserve them
            # in raw CSV; never extrapolate a rotation. Phase coverage still
            # fails if a processing phase extends outside the retained range.
            valid=(t>=pose['time'][0])&(t<=pose['time'][-1])
            result['transform_limits'][stage]=dict(edge_samples_without_pose=int((~valid).sum()),
                                                  pose_start=pose['time'][0],pose_end=pose['time'][-1])
            t=t[valid];f=f[valid]
            if len(t)<2:raise Unavailable('insufficient command samples with observed TCP rotation')
        result['signals'][stage]=dict(time=t.tolist(),values=normal_force(f,frame,cal,time=t,pose=pose,clock=clock).tolist(),
                                     clock_id=clock,frame='calibrated_surface_normal',unit='N',source_frame=frame)
    result['normal_force_verified']=True
    return result


SUMMARY_FIELDS=['run_id','condition','cohort_id','n_attempted','n_executed','n_completed','n_profile_valid','excluded','exclusion_reason',
 'E_profile_N','E_target_N','E_track_applied_N','E_discrepancy_sent_N','phase_coverage','processing_time_s',
 'elapsed_until_stop_s','peak_logged_force','stop_status','hold_status','release_status','recording_notes','reasons']


def analyze_run(folder):
    folder=Path(folder);cfg=read(folder/'config.json');attempt=read(folder/'attempt.json')
    result={k:None for k in SUMMARY_FIELDS};result.update(run_id=attempt['run_id'],condition=attempt['condition'],
        cohort_id=attempt['cohort_id'],n_attempted=1,n_executed=0,n_completed=0,n_profile_valid=0,reasons=[],
        excluded=bool(attempt.get('excluded')),exclusion_reason=attempt.get('exclusion_reason'),canonical_signals_current=False)
    paths=list((folder/'executor').glob('*/events.jsonl'))
    if paths:
        ev,partial=read_events(paths[0]);by={}
        for e in ev:by.setdefault(e['event'],[]).append(e)
        start=by.get('execution_start',[]);stops=[e for e in by.get('stop_requested',[]) if not e['details'].get('initial_reset')]
        result['n_executed']=int(bool(start));result['n_completed']=int(bool(by.get('normal_completion')))
        if start and stops:result['elapsed_until_stop_s']=(stops[0]['receipt_monotonic_ns']-start[0]['receipt_monotonic_ns'])/1e9
        result['stop_status']=[e['details'].get('reason') for e in stops] or None
        result['hold_status']=True if by.get('physical_hold_verified') else None
        result['release_status']=True if by.get('contact_release_verified') else None
        processing_start=None;processing_intervals=[]
        for event in ev:
            at=int(event['receipt_monotonic_ns'])/1e9
            if event['event']=='processing_start':processing_start=at
            elif event['event']=='processing_end' and processing_start is not None:
                if event['details'].get('source')=='operator' and at>processing_start:
                    processing_intervals.append(at-processing_start)
                processing_start=None
        if processing_intervals and not partial:result['processing_time_s']=sum(processing_intervals)
        try:
            wr=csv_rows(paths[0].parent/'wrench.csv')
            if wr:result['peak_logged_force']=max(float(r['fz']) for r in wr)
        except (Unavailable,ValueError,KeyError,TypeError,FileNotFoundError) as exc:result['reasons'].append(str(exc))
        result['peak_force_semantics']='filtered robot-base Fz, whole recorded attempt, not calibrated normal force'
        result['recording_notes']={'partial_events':partial,'sensor_DDS_loss':None,
            'logger_summary':read(paths[0].parent/'summary.json') if (paths[0].parent/'summary.json').is_file() else None}
    try:
        ref_info=cfg['reference']
        if ref_info['status']!='frozen':raise Unavailable('evaluation reference not frozen')
        if digest(ref_info['artifact'])!=ref_info['sha256']:raise Unavailable('reference artifact hash mismatch')
        ref=read(ref_info['artifact'])
        if ref.get('analysis_hash')!=object_hash(cfg['analysis']):raise Unavailable('analysis alignment config changed from frozen reference')
        canonical=raw_run(folder,cfg)
        ann=folder/'phase_annotations.json'
        if ann.is_file():
            annotation=read(ann)
            if annotation['clock_id']!=canonical['clock_id'] or not annotation.get('evidence'):
                raise Unavailable('annotation clock/evidence invalid')
            canonical['phases']=annotation['phases']
        profile=evaluate_profile(ref,canonical,cfg['analysis'])
        # Marker duration is independent of force calibration/reference availability.
        marker_duration=result['processing_time_s'];recording_reasons=result['reasons'];result.update(profile)
        result['reasons']=recording_reasons+result['reasons']
        if marker_duration is not None:result['processing_time_s']=marker_duration
        write(folder/'analysis/canonical_signals.json',canonical)
        result['canonical_signals_current']=True
    except (Unavailable,ValueError,KeyError,TypeError,FileNotFoundError) as exc:result['reasons'].append(str(exc))
    write(folder/'analysis/summary.json',result);write_csv(folder/'analysis/summary.csv',[result],SUMMARY_FIELDS)
    return result


def plot_profiles(folder,run,reference,analysis,force_limits=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from nrs_imitation.e2_force_analysis import align
    fig,axs=plt.subplots(2,1,figsize=(10,8))
    origin=min(min(s['time']) for s in run['signals'].values())
    for name,s in run['signals'].items():
        axs[0].plot(np.asarray(s['time'])-origin,s['values'],label=name)
    for phase in run['phases']:
        axs[0].axvline(phase['start']-origin,color='gray',ls=':',lw=.7)
        if phase.get('end') is not None:axs[0].axvline(phase['end']-origin,color='gray',ls='--',lw=.7)
    axs[0].set(xlabel='Common monotonic time origin (s)',ylabel='Calibrated normal force (N)');axs[0].legend()
    gate=run.get('gate',{})
    if gate.get('time'):
        gax=axs[0].twinx()
        gax.step(np.asarray(gate['time'])-origin,[np.nan if v is None else float(v) for v in gate['contact']],
                 where='post',color='gray',alpha=.35)
        gax.set(ylim=(-.05,1.05),yticks=[0,1],ylabel='Contact gate (0/1)')
    for i,ref in enumerate(reference['phases']):
        axs[1].plot(i+np.asarray(ref['progress']),ref['force_normal_N'],label=ref['phase_id']+' F_ref')
        phase=next((p for p in run['phases'] if p['phase_id']==ref['phase_id'] and p.get('complete')),None)
        if phase:
            q=phase['start']+np.asarray(ref['progress'])*(phase['end']-phase['start'])
            f,valid=align(run['signals']['F_meas']['time'],run['signals']['F_meas']['values'],q,analysis['max_gap_s'])
            f[~valid]=np.nan;axs[1].plot(i+np.asarray(ref['progress']),f,label=ref['phase_id']+' F_meas')
    axs[1].set(xlabel='Predeclared phase + progress',ylabel='Normal force (N)');axs[1].legend()
    if force_limits is not None:
        for ax in axs:ax.set_ylim(force_limits)
    durations=['%s: %s'%(p['phase_id'],('%.2f s'%(p['end']-p['start'])) if p.get('complete') else 'incomplete') for p in run['phases']]
    fig.suptitle(' | '.join(durations) or 'No confirmed processing phases')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(Path(folder)/'analysis/force_profiles.png');plt.close(fig)


def aggregate(folder):
    folder=Path(folder);runs=[analyze_run(p.parent) for p in sorted(folder.glob('[ABC]/*/attempt.json'))]
    if not runs:
        print('No attempted trials. No measured/synthetic experiment results generated.');return
    write_csv(folder/'summary.csv',runs,SUMMARY_FIELDS);write(folder/'summary.json',runs)
    groups=[]
    for cohort in sorted(set(r['cohort_id'] for r in runs)):
        plots=[];all_force=[]
        for p in sorted(folder.glob('[ABC]/*/analysis/canonical_signals.json')):
            run_folder=p.parent.parent;cfg=read(run_folder/'config.json')
            if cfg['cohort_id']!=cohort:continue
            if not read(run_folder/'analysis/summary.json').get('canonical_signals_current'):continue
            canonical=read(p);ref=read(cfg['reference']['artifact'])
            plots.append((run_folder,canonical,ref,cfg['analysis']))
            all_force.extend(v for s in canonical['signals'].values() for v in s['values'])
            all_force.extend(v for phase in ref['phases'] for v in phase['force_normal_N'])
        if all_force:
            lo,hi=min(0.,min(all_force)),max(all_force);margin=max(1.,(hi-lo)*.05)
            for item in plots:plot_profiles(*item,force_limits=(lo-margin,hi+margin))
        for c in 'ABC':
            rr=[r for r in runs if r['cohort_id']==cohort and r['condition']==c]
            row=dict(condition=c,cohort_id=cohort,**{k:sum(r[k] or 0 for r in rr) for k in ['n_attempted','n_executed','n_completed','n_profile_valid']})
            row.update(n_excluded=sum(r['excluded'] for r in rr),n_after_exclusion=sum(not r['excluded'] for r in rr))
            for key in ['E_profile_N','E_target_N','E_track_applied_N','E_discrepancy_sent_N']:
                a=[r[key] for r in rr if not r['excluded'] and r.get(key) is not None]
                row[key+'_mean']=float(np.mean(a)) if a else None
                row[key+'_between_run_SD']=float(np.std(a,ddof=1)) if len(a)>1 else None
            groups.append(row)
    write_csv(folder/'aggregate.csv',groups);write(folder/'aggregate.json',groups)
    write(folder/'exclusion_manifest.json',dict(before=len(runs),excluded=sum(r['excluded'] for r in runs),
        after=sum(not r['excluded'] for r in runs),originals_deleted=False,
        trials=[{k:r[k] for k in ('run_id','condition','excluded','exclusion_reason')} for r in runs]))
    print(folder/'aggregate.csv')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=str(EXPERIMENT/'config.json'))
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('check');sub.add_parser('preview')
    for name in ['f0-candidate','reference-candidate']:
        q=sub.add_parser(name);q.add_argument('--annotations',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('freeze-f0');q.add_argument('--candidate',required=True);q.add_argument('--command-fz',required=True,type=float)
    q.add_argument('--operating-range',nargs=2,type=float,required=True);q.add_argument('--evidence',required=True)
    q=sub.add_parser('freeze-reference');q.add_argument('--candidate',required=True);q.add_argument('--evidence',required=True)
    q=sub.add_parser('seal');q.add_argument('--evidence',required=True)
    q=sub.add_parser('schedule');q.add_argument('--session',required=True)
    q=sub.add_parser('run');q.add_argument('--condition',required=True,choices=list('ABC'));q.add_argument('--session',required=True)
    q.add_argument('--block',type=int,help='Optional scheduled block; omitted for repeated common-state trials')
    q.add_argument('--specimen');q.add_argument('--region')
    q.add_argument('--surface-reuse',choices=['assumed_same_state','new_surface','repainted_same_surface','reused_surface'])
    q=sub.add_parser('analyze');q.add_argument('--session-dir',required=True)
    q=sub.add_parser('annotate');q.add_argument('--run-dir',required=True);q.add_argument('--annotations',required=True)
    q=sub.add_parser('archive');q.add_argument('--session-dir',required=True)
    q=sub.add_parser('register-model');q.add_argument('--condition',choices=list('ABC'),required=True);q.add_argument('--checkpoint',required=True)
    q.add_argument('--selection-evidence',required=True)
    q=sub.add_parser('metrology');q.add_argument('--csv',required=True);q.add_argument('--output',required=True)
    args=p.parse_args();cfg=read(args.config)
    if args.command=='check':print(json.dumps(readiness(cfg),ensure_ascii=False,indent=2))
    elif args.command=='preview':preview(cfg)
    elif args.command in ('f0-candidate','reference-candidate'):
        req=read(args.annotations)
        result=(f0_candidate(req,cfg['calibration'],DATASET,split()) if args.command=='f0-candidate' else
                build_reference(req,cfg['calibration'],DATASET,split(),cfg['analysis']))
        write(args.output,result,exclusive=True)
    elif args.command=='freeze-f0':
        reject_recipe_change_after_execution(cfg)
        candidate=read(args.candidate);lo,hi=args.operating_range
        if candidate['calibration_hash']!=object_hash(cfg['calibration']) or not cfg['calibration']['normal_force_verified']:
            raise Unavailable('Candidate/calibration mismatch')
        if not np.isfinite([lo,hi,args.command_fz]).all() or not lo<=args.command_fz<=hi or not args.evidence.strip():
            raise ValueError('Validated command-force operating range and conversion evidence required')
        cfg['f0']=dict(status='frozen',normal_force_N=candidate['normal_force_N'],command_fz_N=args.command_fz,
            candidate=str(Path(args.candidate).resolve()),candidate_sha256=digest(args.candidate),
            validated_command_range_N=[lo,hi],freeze_evidence=args.evidence,calibration_hash=object_hash(cfg['calibration']))
        cfg['cohort_id']=cohort_id(cfg)
        write(args.config,cfg)
    elif args.command=='freeze-reference':
        reject_recipe_change_after_execution(cfg)
        ref=read(args.candidate)
        if ref['calibration_hash']!=object_hash(cfg['calibration']) or ref['analysis_hash']!=object_hash(cfg['analysis']) or not args.evidence.strip():
            raise Unavailable('Reference/calibration/analysis evidence mismatch')
        if [r['phase_id'] for r in ref['phases']]!=cfg['protocol']['phase_ids']:raise Unavailable('Reference phases differ from protocol')
        cfg['reference']=dict(status='frozen',artifact=str(Path(args.candidate).resolve()),sha256=digest(args.candidate),selection_evidence=args.evidence)
        cfg['cohort_id']=cohort_id(cfg)
        write(args.config,cfg)
    elif args.command=='seal':seal(cfg,args.config,args.evidence)
    elif args.command=='schedule':schedule(cfg,args.session)
    elif args.command=='run':run(cfg,args)
    elif args.command=='analyze':aggregate(args.session_dir)
    elif args.command=='annotate':
        folder=Path(args.run_dir);events=list((folder/'executor').glob('*/metadata.json'))
        if len(events)!=1:raise Unavailable('executor metadata/clock required')
        annotation=read(args.annotations);metadata=read(events[0]);run_cfg=read(folder/'config.json')
        if annotation.get('clock_id')!=metadata['clock_id'] or not annotation.get('evidence'):
            raise Unavailable('Manual video/event evidence and matching monotonic clock required')
        allowed=run_cfg['protocol']['phase_ids'];previous=-float('inf');seen=[]
        for phase in annotation['phases']:
            if phase['phase_id'] not in allowed or phase['phase_id'] in seen or phase['start']<previous:
                raise Unavailable('annotation phase identity/order mismatch')
            seen.append(phase['phase_id'])
            if phase.get('complete'):
                if not np.isfinite([phase['start'],phase['end']]).all() or phase['end']<=phase['start']:
                    raise Unavailable('invalid complete phase boundaries')
                previous=phase['end']
        write(folder/'phase_annotations.json',annotation,exclusive=True)
    elif args.command=='archive':print('Preserved files:',seal_archive(args.session_dir))
    elif args.command=='register-model':
        source=Path(args.checkpoint).resolve();stats=source.with_name('dataset_stats.pkl')
        if not source.is_file() or not stats.is_file():raise ValueError('checkpoint + matching normalizer required')
        import torch
        ck=torch.load(source,map_location='cpu',weights_only=False)
        if ck.get('smoke_only') or ck.get('epoch',-1)<0:raise ValueError('Smoke/untrained checkpoint cannot be registered')
        pc=read_stats(stats).get('policy_config',{})
        saved_pc=ck.get('config',{}).get('policy_config',{})
        for key in ('state_dim','action_dim','use_force_observation','motion_only','force_action'):
            if pc.get(key)!=saved_pc.get(key):raise ValueError('Checkpoint and stats disagree on '+key)
        # Keep the original training files; the existing runtime consumes policy_best.ckpt by name.
        dest=EXPERIMENT/'registered_models'/(args.condition+'_'+digest(source)[:16]);dest.mkdir(parents=True,exist_ok=False)
        shutil.copy2(source,dest/'policy_best.ckpt');shutil.copy2(stats,dest/'dataset_stats.pkl')
        cfg['models'][args.condition]=dict(checkpoint=str(dest/'policy_best.ckpt'),checkpoint_sha256=digest(source),
            normalizer_sha256=digest(stats),source_checkpoint=str(source),selected_epoch_zero_based=ck['epoch'],selection_evidence=args.selection_evidence)
        errors=model_errors(select_condition(cfg,args.condition))
        if errors:raise ValueError('; '.join(errors))
        cfg['cohort_id']=cohort_id(cfg)
        write(args.config,cfg)
    elif args.command=='metrology':
        rows=csv_rows(args.csv)
        with Path(args.csv).open(encoding='utf-8-sig') as f:fields=list(csv.DictReader(f).fieldnames or [])
        fields=list(dict.fromkeys(fields+list(metrology({}))))
        write_csv(args.output,[dict(r,**metrology(r)) for r in rows],fields)


if __name__=='__main__':main()
