"""Read-only C5 audit and exact cleanup manifest preparation; never deletes."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import csv
import hashlib
import json
import math
import re
import subprocess

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

USER_ROOT = Path('/home/eunseop')
ROOT = USER_ROOT/'nrs_imitation'
LOG = ROOT/'logs'
ROS = USER_ROOT/'.ros/log'
VIDEO = USER_ROOT/'Videos/Screencasts'
OUT = Path(__file__).resolve().parent
PROFILE = 'c_pose_conditioning_20260926_v1'
EXPECTED_KEEP = {'RTC_C_20260926T'+t for t in ('184357','184521','184622','184724','184820')}
EXPECTED_REMOVE = {'RTC_C_20260926T'+t for t in ('175117','175131','184718')}


def js(p): return json.loads(p.read_text())
def events(p): return [json.loads(s) for s in (p/'events.jsonl').read_text().splitlines() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def timestamp(ns): return datetime.fromtimestamp(ns/1e9).astimezone().isoformat(timespec='milliseconds')


groups={}; context_tags={}; protected=[]
for p in sorted((LOG/'inference_metrics').glob('RTC_[RTC]_20260926T*')):
    match=re.match(r'(RTC_[RTC]_20260926T\d{6})_',p.name)
    assert match,p
    tag=match.group(1); m=js(p/'metadata.json')
    if tag[4] in 'RT': protected.append(str(p))
    for artifact in m.get('artifacts',[]):
        path=Path(artifact['path'])
        if path.name=='config.json' and path.parent.parent==LOG/'rtc_launch_context':
            context_tags[str(path)]=tag
            if tag[4] in 'RT': protected.append(str(path.parent))
    if tag[4]!='C': continue
    g=groups.setdefault(tag,dict(tag=tag,paths=[],metadata={}))
    role='executor' if '_executor_' in p.name else 'provider'
    assert role not in g['metadata'],tag
    g['metadata'][role]=m;g[role+'_dir']=str(p);g['paths'].append(str(p))
assert set(groups)==EXPECTED_KEEP|EXPECTED_REMOVE,set(groups)

for p in sorted(ROS.glob('2026-09-26-*/launch.log')):
    content=p.read_text();m=re.search(r'Run context: (\S+)',content)
    if not m or m.group(1) not in context_tags: continue
    tag=context_tags[m.group(1)]
    paths=[str(Path(m.group(1)).parent),str(p.parent)]
    start=float(content.splitlines()[0].split()[0])
    pids=re.findall(r'process started with pid \[(\d+)\]',content)
    for pid in pids:
        for node_log in ROS.glob('*_'+pid+'_*.log'):
            suffix=re.search(r'_(\d+)\.log$',node_log.name)
            if suffix and start-2 <= int(suffix.group(1))/1000 <= start+30:
                paths.append(str(node_log))
    if tag[4] in 'RT': protected.extend(paths);continue
    g=groups[tag]; assert 'launch_dir' not in g,tag
    g.update(context_file=m.group(1),launch_dir=str(p.parent),launch_pids=pids,
             launch_errors=[line for line in content.splitlines() if '[ERROR]' in line],
             clean_exit_pids=re.findall(r'process has finished cleanly \[pid (\d+)\]',content))
    g['paths'].extend(paths)

for p in (LOG/'polishing_removal').glob('RTC_[RT]_20260926T*'): protected.append(str(p))
for p in VIDEO.glob('*RTC_[RT]_20260926T*.webm'): protected.append(str(p))

patch_validation=js(ROOT/'reports/20260926_C_execution_patch/validation.json')
expected_code={Path(p).name:h for p,h in patch_validation['files_sha256'].items() if p.endswith('.py')}
baseline=js(ROOT/'reports/20260926_C_execution_patch/before/experiments/e2_rule_replay_20260920/config.json')

for tag,g in sorted(groups.items()):
    plot_dirs=sorted((LOG/'polishing_removal').glob(tag+'_*'))
    videos=sorted(VIDEO.glob('*'+tag+'.webm'))
    g.update(plot_dirs=list(map(str,plot_dirs)),videos=list(map(str,videos)))
    g['paths']=sorted(set(g['paths']+list(map(str,plot_dirs+videos))))
    ep=Path(g['executor_dir']); ee=events(ep); em=g['metadata']['executor']
    starts=[e for e in ee if e['event']=='execution_start']
    stops=[e for e in ee if e['event']=='stop_requested' and not e['details'].get('initial_reset')]
    completions=[e for e in ee if e['event']=='normal_completion']
    faults=[e for e in ee if e['event'] in ('safety_stop','error','stop_not_verified','shutdown_without_verified_stop')]
    g['normal_completion_events']=completions
    g['stop_reasons']=[dict(time=timestamp(e['receipt_ros_ns']),**e['details']) for e in stops]
    g['execution_started']=bool(starts)
    g['keep']=bool(starts and completions and not faults and
                   stops[-1]['details']['reason']=='operator_finish' and
                   (em.get('c_pose_conditioning') or {}).get('profile')==PROFILE)
    assert g['keep']==(tag in EXPECTED_KEEP),tag
    if not g['keep']:
        g['reason']='known_prepatch_vibration_manual_abort' if tag.endswith('175131') else 'manual_abort_before_execution'
        g.pop('metadata');continue
    assert len(starts)==len(completions)==len(stops)==1
    assert completions[0]['details']['controller_hold_verified'] and completions[0]['details']['queue_cancel_verified']
    assert not g['launch_errors'] and set(g['launch_pids'])==set(g['clean_exit_pids'])
    assert len(g['launch_pids'])==5
    assert len(plot_dirs)==len(videos)==1
    pp=Path(g['provider_dir']);pm=g['metadata']['provider'];pe=events(pp)
    assert not any(e['event'] in ('inference_error','error','safety_stop') for e in pe)
    assert em['session_id']==pm['e2_session_id']
    assert pm['runtime_parameters']['e2_config']==g['context_file']
    assert em['config_sha256']==sha(Path(g['context_file']))
    for key in ('common','executor','recipe','replay'): assert em['config'][key]==baseline[key],(tag,key)
    for artifact in em['artifacts']:
        name=Path(artifact['path']).name
        if name in expected_code: assert artifact['sha256']==expected_code[name],(tag,name)
        assert sha(ep/artifact['copy'])==artifact['sha256']
    g.update(reason='recording_complete_operator_finish_and_verified_hold',session_id=em['session_id'],
             config_sha256=em['config_sha256'],conditioning_profile=PROFILE,
             started_at=timestamp(starts[0]['receipt_ros_ns']),finish_requested_at=timestamp(stops[0]['receipt_ros_ns']),
             execution_duration_s=stops[0]['receipt_monotonic_ns']/1e9-starts[0]['details']['monotonic_start'],
             operator_processing_markers=sum(e['event'] in ('processing_start','processing_end') for e in ee))
    t0=starts[0]['details']['monotonic_start']; t1=stops[0]['receipt_monotonic_ns']/1e9
    g['streams']={}; all_rows={}
    for role in ('executor','provider'):
        folder=Path(g[role+'_dir']);s=js(folder/'summary.json')
        assert s['drained'] and s['queue_pending']==s['write_error_count']==0,(tag,role)
        assert all(c['received']==c['enqueued']==c['written'] and c['dropped']==0 for c in s['counts'].values())
        assert len(events(folder))>=s['counts']['events']['written']
        for name in ('commands','tcp_pose','wrench','legacy'):
            with (folder/(name+'.csv')).open() as f: rows=list(csv.DictReader(f))
            all_rows[role+'/'+name]=rows
            assert len(rows)==s['counts'].get(name,{}).get('written',0),(tag,role,name)
            assert all(None not in r and all(v is not None for v in r.values()) for r in rows)
            assert all(r.get('validity')=='valid' for r in rows)
            assert all(math.isfinite(float(r[k])) for r in rows for k in ('x','y','z','rx','ry','rz','fx','fy','fz') if r.get(k))
            ts=np.array([int(r['receipt_monotonic_ns'])/1e9 for r in rows])
            assert np.all(np.diff(ts)>=0)
            stream=dict(rows=len(rows),logger_drops=0,write_errors=0,finite_values=True,time_order=True)
            if name in ('tcp_pose','wrench'):
                crop=ts[(ts>=t0)&(ts<=t1)]
                assert len(crop)>150
                stream['execution_rows']=len(crop)
                stream['execution_max_gap_with_edges_ms']=float(np.diff(np.r_[t0,crop,t1]).max()*1000)
                assert stream['execution_max_gap_with_edges_ms']<200.
            g['streams'][role+'/'+name]=stream
    rows=all_rows['executor/commands']
    by_stage={stage:[r for r in rows if r['command_stage']==stage] for stage in
              ('time_sampled','pose_conditioned','contact_gated','node_sent')}
    sent=by_stage['node_sent']; assert len(rows)==4*len(sent)
    assert all(len(rr)==len(sent) for rr in by_stage.values())
    assert all([r['command_id'] for r in rr]==[r['command_id'] for r in sent] for rr in by_stage.values())
    details=[json.loads(r['details']) for r in sent]
    assert all(d['conditioning_profile']==PROFILE for d in details)
    period=np.minimum([d['dt_s'] for d in details],.008)
    assert np.all(period>0)
    a=np.array([[float(r[k]) for k in ('x','y','z','rx','ry','rz','fx','fy','fz')] for r in sent])
    velocity=np.diff(a[:,:3],axis=0)/period[1:,None]
    acceleration=np.diff(velocity,axis=0)/period[2:,None]
    rotations=Rotation.from_rotvec(a[:,3:6]);angular=(rotations[1:]*rotations[:-1].inv()).as_rotvec()/period[1:,None]
    angular_acc=np.diff(angular,axis=0)/period[2:,None]
    max_speed=float(np.abs(velocity).max());max_acc=float(np.abs(acceleration).max())
    assert max_speed<=10.0001 and max_acc<=25.0001
    assert np.abs(angular).max()<=em['config']['executor']['angular_speed_rad_s']+1e-6
    assert np.abs(angular_acc).max()<=em['c_pose_conditioning']['angular_acceleration_rad_s2']+1e-5
    assert np.max(np.abs(np.diff(a[:,8]))/period[1:])<=30.0001
    assert np.all(a[:,6:8]==0.)
    for raw,conditioned in zip(by_stage['time_sampled'],by_stage['pose_conditioned']):
        assert [raw[k] for k in ('fx','fy','fz')]==[conditioned[k] for k in ('fx','fy','fz')]
    ts=np.array([int(r['receipt_monotonic_ns'])/1e9-t0 for r in sent]);contacts=np.array([d['contact'] for d in details])
    frows=all_rows['executor/wrench'];ft=np.array([int(r['receipt_monotonic_ns'])/1e9 for r in frows]);fz=np.array([float(r['fz']) for r in frows])
    mask=(ft>=t0)&(ft<=t1)
    plan_events=[e for e in ee if e['event']=='plan_received']
    assert all(e['details']['c_pose_conditioning']['profile']==PROFILE for e in plan_events)
    provider_plans=all_rows['provider/commands']
    for event in plan_events:
        for stage in ('policy_prediction','postprocessed'):
            rr=[r for r in provider_plans if r['command_stage']==stage and int(r['plan_id'])==event['details']['plan_id']]
            assert len(rr)==128,(tag,event['details']['plan_id'],stage,len(rr))
            assert sorted(int(r['action_index']) for r in rr)==list(range(128))
    g['motion']=dict(node_sent_rows=len(sent),plan_count=len(plan_events),
        max_command_interval_ms=float(np.diff(ts).max()*1000),max_axis_speed_mm_s=max_speed,
        max_axis_acceleration_mm_s2=max_acc,max_angular_acceleration_rad_s2=float(np.abs(angular_acc).max()),
        contact_gate_transitions_first4s=int(np.count_nonzero(np.diff(contacts[ts<4.]))),
        contact_gate_transitions_total=int(np.count_nonzero(np.diff(contacts))),
        measured_base_fz_min_N=float(fz[mask].min()),measured_base_fz_max_N=float(fz[mask].max()),
        sent_fz_min_N=float(a[:,8].min()),sent_fz_max_N=float(a[:,8].max()))
    images=sorted(plot_dirs[0].glob('*.png'))+sorted((pp/'snapshots').glob('*.png'))
    assert len(images)==6
    for p in images:
        with Image.open(p) as im:im.verify()
    probe=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries',
        'stream=nb_read_frames,width,height:format=duration','-of','json',str(videos[0])],capture_output=True,text=True,check=True)
    assert not probe.stderr.strip(),probe.stderr
    video= json.loads(probe.stdout)
    decoded=subprocess.run(['ffmpeg','-v','error','-i',str(videos[0]),'-f','null','-'],capture_output=True,text=True,check=True)
    assert not decoded.stderr.strip(),decoded.stderr
    assert int(video['streams'][0]['nb_read_frames'])>100
    g['media']=dict(png_images_verified=len(images),video_probe=video,full_video_decode_errors=0)
    g.pop('metadata')
    print(json.dumps(dict(tag=tag,duration_s=g['execution_duration_s'],**g['motion']),ensure_ascii=False),flush=True)

assert len({g['session_id'] for g in groups.values() if g['keep']})==5
assert len({g['config_sha256'] for g in groups.values() if g['keep']})==1
keep=sorted(set(p for g in groups.values() if g['keep'] for p in g['paths']))
delete=sorted(set(p for g in groups.values() if not g['keep'] for p in g['paths']))
protected=sorted(set(protected))
assert not set(delete)&set(keep+protected)
allowed={LOG/'inference_metrics',LOG/'rtc_launch_context',LOG/'polishing_removal',ROS,VIDEO}
for p in map(Path,keep+delete+protected):
    assert p.parent in allowed and p.exists() and not p.is_symlink(),p


def inventory(paths):
    result={}
    for p in map(Path,paths):
        for f in sorted(p.rglob('*')) if p.is_dir() else [p]:
            assert not f.is_symlink(),f
            if f.is_dir():continue
            assert f.is_file() and not f.is_symlink(),f
            st=f.stat();result[str(f)]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino,sha256=sha(f))
    return result


manifest=dict(created_at=datetime.now().astimezone().isoformat(),status='prepared_not_deleted',
    scope='Today RTC C only, excluding the five patched operator-finished recordings; exact metrics/context/media/local ROS associations',
    authorization='User requested auditing five C runs and removal of abnormal logs.',
    runs=[groups[k] for k in sorted(groups)],retained_count=5,excluded_count=3,
    keep_paths=keep,delete_paths=delete,protected_R_T_paths=protected,
    retained_file_inventory=inventory(keep),delete_file_inventory=inventory(delete),protected_R_T_file_inventory=inventory(protected),
    physical_observation='User explicitly reported no unintended motion in the five patched C runs. Contact-gate chatter is retained as a model/execution behavior metric, not a deletion criterion.',
    operator_statement='의도하지 않은 동작은 없었어. 접촉 판정이 잦은 것이 있다면 그것은 그냥 모델 자체의 문제야',
    limitations=['normal_completion is operator_finish plus verified hold, not verified contact release/home arrival or independently measured polishing success.',
                 'No operator processing_start/end markers; whole execution time is not pure processing time.',
                 'Runs 2 and 3 retain frequent early contact-gate transitions; physical high-frequency vibration cannot be certified absent from 20 Hz feedback.',
                 'Old vibration and patch reports remain, but original prepatch C input logs are among explicitly removed failed runs. Their replay scripts will need those original files to reproduce historical results.'])
(OUT/'cleanup_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
csv_rows=[]
for g in groups.values():
    if not g['keep']:continue
    csv_rows.append(dict(run_tag=g['tag'],recording_status=g['reason'],started_at=g['started_at'],
        duration_s=g['execution_duration_s'],normal_completion=True,controller_hold_verified=True,
        tcp_rows=g['streams']['executor/tcp_pose']['rows'],force_rows=g['streams']['executor/wrench']['rows'],
        **g['motion'],executor_dir=g['executor_dir'],provider_dir=g['provider_dir'],video=g['videos'][0]))
with (OUT/'retained_runs.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
print(json.dumps(dict(retain_runs=5,exclude_runs=3,delete_top_level_paths=len(delete),
    delete_files=len(manifest['delete_file_inventory']),delete_bytes=sum(v['bytes'] for v in manifest['delete_file_inventory'].values()),
    retained_files=len(manifest['retained_file_inventory']),protected_RT_files=len(manifest['protected_R_T_file_inventory']),
    delete_path_categories=dict(Counter(str(Path(p).parent) for p in delete))),indent=2),flush=True)
