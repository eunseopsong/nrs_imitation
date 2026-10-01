"""Offline E1/E2 v2 reanalysis. Writes only beside this file; never uses ROS.

Original helpers and complete baseline analyses are reused from reused_baseline.py.
All phase/reference decisions added here are retrospective descriptive analyses.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
import argparse, csv, hashlib, json, pickle, sys, subprocess, platform
import numpy as np
import h5py
from scipy.signal import fftconvolve
import reused_baseline as legacy

ROOT = Path('/home/eunseop/nrs_imitation')
OUT = Path(__file__).resolve().parent
DATA = {'E1': ROOT/'results/20260927/E1', 'E2': ROOT/'results/20260930/E2'}
RECIPE_Z = 167.10179138183594
Z_THRESH = RECIPE_Z + 10.0
THRESHOLDS = [Z_THRESH-2, Z_THRESH, Z_THRESH+2]
MAX_GAP = .2
GRID_N = 201
COLORS = {'R':'#17836D','T':'#326AB8','C':'#CD7031','A':'#17836D','B':'#326AB8'}
ENDPOINTS = np.array([[-15.225713729858398,30.618316650390625],
                       [43.69847869873047,-39.007537841796875]])
sha, rows, events, vals, times = legacy.sha, legacy.rows, legacy.events, legacy.vals, legacy.times


def clean(v):
    if isinstance(v, Path): return str(v)
    if isinstance(v, dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [clean(x) for x in v]
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, (np.integer,np.bool_)): return v.item()
    if isinstance(v, (float,np.floating)): return float(v) if np.isfinite(v) else None
    return v


def write_json(name, data):
    p=OUT/name; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def write_csv(name, records):
    records=list(records); p=OUT/name; p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in records for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in records:
            rr=clean(r)
            w.writerow({k:('NA' if rr.get(k) is None else json.dumps(rr[k],ensure_ascii=False)
                           if isinstance(rr[k],(list,dict)) else rr[k]) for k in keys})


def read_json(p, default=None):
    try: return json.loads(p.read_text())
    except (OSError,ValueError): return {} if default is None else default


def interval_edges(t,a,b):
    return np.r_[a,t[(t>a)&(t<b)],b]


def zoh(t,v,q,max_gap=MAX_GAP):
    """Receipt-time hold; no extrapolation or unlimited gap filling."""
    q=np.asarray(q);idx=np.searchsorted(t,q,side='right')-1
    valid=(idx>=0)&(q<=t[-1]); idx=np.clip(idx,0,len(t)-1)
    nextidx=np.minimum(idx+1,len(t)-1)
    valid &= (t[nextidx]-t[idx]<=max_gap+1e-12)
    ans=np.array(v[idx],float,copy=True);ans[~valid]=np.nan
    return ans


def pose_interp(t,p,q):
    q=np.asarray(q);ans=np.stack([np.interp(q,t,p[:,j]) for j in range(p.shape[1])],axis=1)
    idx=np.searchsorted(t,q,side='right')-1;idx=np.clip(idx,0,len(t)-2)
    ok=(q>=t[0])&(q<=t[-1])&(t[idx+1]-t[idx]<=MAX_GAP)
    ans[~ok]=np.nan
    return ans


def force_stats(r,a,b):
    if b<=a or len(r['ft'])<2: return None
    try: return legacy.weighted_force(r['ft'],r['force'][:,2],a,b)
    except (AssertionError,ValueError): return None


def low_z_intervals(t,z,threshold):
    """Exact crossings of measured Z polyline; no force-gate dependence."""
    intervals=[]; active=float(t[0]) if z[0]<=threshold else None
    for i in range(len(t)-1):
        down=z[i]>threshold and z[i+1]<=threshold
        up=z[i]<=threshold and z[i+1]>threshold
        if down or up:
            cross=float(t[i]+(threshold-z[i])/(z[i+1]-z[i])*(t[i+1]-t[i]))
            if down: active=cross
            elif active is not None: intervals.append((active,cross));active=None
    if active is not None: intervals.append((active,float(t[-1])))
    return intervals


def stages(r,threshold):
    all_intervals=low_z_intervals(r['pt'],r['pose'][:,2],threshold)
    candidates=[x for x in all_intervals if x[1]-x[0]>=.5]
    if not candidates: return None
    a,b=candidates[0]
    total=sum(y-x for x,y in all_intervals)
    return dict(approach=[0,a],work_height=[a,b],withdrawal=[b,r['metric']['execution_duration_s']],
                later_low_z_intervals=[x for x in all_intervals if x[0]>=b+1e-9],
                all_low_z_intervals=all_intervals,threshold_mm=threshold,
                first_interval_share_of_low_z_time=(b-a)/total if total else None,
                status='geometric proxy, not verified physical processing')


def audit_csv(path,rr):
    numeric={'wrench.csv':['fx','fy','fz','tx','ty','tz'],
             'tcp_pose.csv':['x','y','z','rx','ry','rz'],
             'commands.csv':['x','y','z','rx','ry','rz','fx','fy','fz']}[path.name]
    nonfinite=0;blank=0
    for r in rr:
        for k in numeric:
            if not r.get(k): blank+=1;continue
            try: nonfinite+=int(not np.isfinite(float(r[k])))
            except ValueError: nonfinite+=1
    tt=times(rr) if rr else np.array([]);delta=np.diff(tt)
    return dict(path=path.relative_to(ROOT),sha256=sha(path),rows=len(rr),
                duplicate_rows=len(rr)-len({tuple(x.items()) for x in rr}),
                duplicate_receipt_timestamps=int(np.sum(delta==0)),
                reversed_receipt_timestamps=int(np.sum(delta<0)),nonfinite_numeric=nonfinite,
                blank_numeric_fields=blank,maximum_gap_ms=float(delta.max()*1000) if len(delta) else None,
                gaps_over_200ms=int(np.sum(delta>MAX_GAP)),
                source_acquisition_timestamp_present=sum(bool(x.get('source_stamp_ns')) for x in rr),
                frames=sorted({x.get('semantic_frame','') for x in rr}),
                position_units=sorted({x.get('position_unit','') for x in rr}),
                force_units=sorted({x.get('force_unit','') for x in rr}),
                orientation_units=sorted({x.get('orientation_unit','') for x in rr}),
                torque_units=sorted({x.get('torque_unit','') for x in rr}))


def add_run(r, ex, audit, boundaries, settings, unavailable):
    m=r['metric'];m['experiment']=ex;m['condition']=m.get('condition',m.get('method'))
    b=r['base']; em=read_json(b/'executor/metadata.json'); pm=read_json(b/'provider/metadata.json')
    r['executor_metadata']=em;r['provider_metadata']=pm
    r['pev']=events(b/'provider/events.jsonl')
    full={}
    for sub in ['executor','provider']:
        for name in ['wrench.csv','tcp_pose.csv','commands.csv']:
            p=b/sub/name
            try:
                rr=rows(p);full[(sub,name)]=rr;a=audit_csv(p,rr)
                summary=read_json(b/sub/'summary.json');stream=name[:-4]
                count=summary.get('counts',{}).get(stream,{})
                a.update(experiment=ex,label=m['label'],logger_written=count.get('written'),
                         rows_match_logger_written=len(rr)==count.get('written'),
                         logger_dropped=count.get('dropped'),write_errors=summary.get('write_error_count'),
                         middleware_loss_count=read_json(b/sub/'metadata.json').get('middleware_loss_count'))
                audit.append(a)
            except (OSError,ValueError) as e:
                audit.append(dict(path=p.relative_to(ROOT),experiment=ex,label=m['label'],read_error=str(e)))
        ep=b/sub/'events.jsonl'
        ev=events(ep); tt=np.array([e['receipt_monotonic_ns'] for e in ev],dtype=np.int64)
        audit.append(dict(path=ep.relative_to(ROOT),experiment=ex,label=m['label'],rows=len(ev),sha256=sha(ep),
                          duplicate_receipt_timestamps=int(np.sum(np.diff(tt)==0)),
                          reversed_receipt_timestamps=int(np.sum(np.diff(tt)<0)),event_types=dict(Counter(e['event'] for e in ev))))
    sent=full['executor','commands.csv']
    sent=[x for x in sent if x['command_stage']=='node_sent' and r['start']<=int(x['receipt_monotonic_ns'])/1e9<=r['end']]
    r['details']=[json.loads(x['details']) for x in sent]
    r['sent_rows']=sent
    req=[x for x in full['executor','commands.csv'] if x['command_stage']=='time_sampled' and r['start']<=int(x['receipt_monotonic_ns'])/1e9<=r['end']]
    r['rt']=times(req)-r['start'];r['target']=vals(req,['fx','fy','fz'])
    r['stage']=stages(r,Z_THRESH)
    stage_variants={str(t):stages(r,t) for t in THRESHOLDS}
    # Preserve the geometric crossings as ranges; ±2 mm is a sensitivity choice,
    # not a calibrated confidence interval on true contact.
    start_range=[s['work_height'][0] for s in stage_variants.values() if s]
    end_range=[s['work_height'][1] for s in stage_variants.values() if s]
    first=next(e for e in r['events'] if e['event']=='execution_start')
    stop=next(e for e in r['events'] if e['event']=='stop_requested' and not e['details'].get('initial_reset'))
    boundary=dict(experiment=ex,label=m['label'],run_id=m['archive_run_id'],start_receipt_monotonic_ns=first['receipt_monotonic_ns'],
                  end_receipt_monotonic_ns=stop['receipt_monotonic_ns'],start_event='execution_start',end_event='first noninitial stop_requested',
                  initial_reset_events=[e for e in r['events'] if e['event']=='stop_requested' and e['details'].get('initial_reset')],
                  whole_duration_s=m['execution_duration_s'],geometric_stages=r['stage'],threshold_variants=stage_variants,
                  work_start_sensitivity_range_s=[min(start_range),max(start_range)] if start_range else None,
                  work_end_sensitivity_range_s=[min(end_range),max(end_range)] if end_range else None,
                  provider_phases=[dict(time_s=e['receipt_monotonic_ns']/1e9-r['start'],details=e['details'])
                                   for e in r['events'] if e['event'] in ['provider_phase','return_phase_started','home_return_started']])
    boundaries.append(boundary)
    st=r['stage']
    for stage in ['approach','work_height','withdrawal']:
        a,z=st[stage] if st else (np.nan,np.nan)
        fm=force_stats(r,a,z) if st else None
        m[stage+'_duration_s']=z-a if st else None
        m[stage+'_mean_fz_N']=fm['mean_fz_N'] if fm else None
        m[stage+'_peak_fz_N']=fm['peak_fz_N'] if fm else None
    m['work_height_fraction']=m['work_height_duration_s']/m['execution_duration_s'] if st else None
    m['comparable_first_work_share']=st['first_interval_share_of_low_z_time'] if st else None
    m['later_low_z_candidate_s']=sum(b-a for a,b in st['later_low_z_intervals']) if st else None
    m['recorder_minus_executor_duration_s']=m['recorder_duration_s']-m['execution_duration_s']
    inf=np.array([e['receipt_monotonic_ns']/1e9 for e in r['pev'] if e['event']=='inference_start'])
    plan=np.array([e['receipt_monotonic_ns']/1e9 for e in r['events'] if e['event']=='plan_received'])
    for name,t in [('inference_update',inf),('plan_receipt',plan)]:
        gaps=np.diff(t)
        m[name+'_interval_median_s']=float(np.median(gaps)) if len(gaps) else None
        m[name+'_interval_min_s']=float(gaps.min()) if len(gaps) else None
        m[name+'_interval_max_s']=float(gaps.max()) if len(gaps) else None
    p_inf=[e['details'] for e in r['pev'] if e['event']=='inference_start']
    off=[x.get('conditioned_force_norm') for x in p_inf if x.get('force_observation') is False]
    m['off_conditioned_force_norm_zero_verified']=all(x==0 for x in off) if off else None
    r['plan_times']=plan-r['start'];r['inference_times']=inf-r['start']
    m['sent_mean_fz_controller_N']=float(np.average(r['action'][:,8],weights=np.diff(np.r_[r['ct'],m['execution_duration_s']])))
    m['sent_peak_fz_controller_N']=float(r['action'][:,8].max())
    # Test the actual configured tick cap, using logged dt_s rather than receipt jitter.
    change=np.abs(np.diff(r['action'][:,8])); logged_dt=np.array([d['dt_s'] for d in r['details'][1:]])
    cap=r['config']['executor']['force_rate_N_s']*np.minimum(np.maximum(logged_dt,0),r['config']['executor']['control_period_s'])
    m['sent_slew_cap_excess_max_N']=float(np.maximum(change-cap,0).max()) if len(change) else None
    m['sent_slew_cap_violation_count']=int(np.sum(change>cap+1e-5))
    m['nominal_control_hz']=1/r['config']['executor']['control_period_s']
    m['trajectory_grid_hz']=r['config']['il'].get('trajectory_hz',30)
    for k in ['E_profile_N','E_target_N','E_sent_N','E_applied_N']:
        m[k]=None
    reasons={'E_profile_N':'episode_29 measured force uses tracker topic /ftsensor/measured_Cvalue; transform/sign equivalence to robot_base currentF not archived. Geometric task correspondence alone cannot repair force coordinates.',
             'E_target_N':'Teaching target axis and deployed robot target axis/sign equivalence unverified; source target is not the measured teaching force.',
             'E_sent_N':'Archive specifies TCP target and robot_base measurement, but active deployed coordinate readback is unavailable. No verified common-axis sent-measured subtraction is made.',
             'E_applied_N':'controller_applied_force is null and no controller-applied force stream is recorded.',
             'physical_normal_force_N':'surface_normal/compression_sign not archived; base Fz is a common-axis component.',
             'physical_processing_duration_s':'Processing is an automatic execution/contact marker; geometric stages do not independently prove material processing.',
             'Preston_removal_or_rate':'No verified normal force, relative rotational speed, contact area or calibrated Preston coefficient; historical TCP-speed heatmap is a proxy.',
             'actual_rpm':'Same configured RPM is operator testimony; no numeric setpoint or tachometer stream.',
             'sensor_acquisition_latency_s':'currentF is headerless; source acquisition timestamp absent.'}
    for key,why in reasons.items():unavailable.append(dict(experiment=ex,label=m['label'],metric=key,value=None,reason=why))
    setting=dict(experiment=ex,label=m['label'],run_id=m['archive_run_id'],config_path=b/('runtime.json' if ex=='E2' else 'launch_context/config.json'),
                 config_sha256=sha(b/('runtime.json' if ex=='E2' else 'launch_context/config.json')),executor=r['config']['executor'],
                 checkpoint=r['config'].get('il',{}),force_observation=pm.get('force_observation'),
                 off_normalized_force_verified=m['off_conditioned_force_norm_zero_verified'],
                 currentF_frame='robot_base',command_frame='controller_config_dependent; archived Force_Con_Coordinate=1 (TCP)',
                 calibration_status=pm.get('calibration_status'),clock_id=em.get('clock_id'),
                 provider_git_commit=pm.get('git_commit'),executor_git_commit=em.get('git_commit'),
                 common_settings_sha256=r['config'].get('comparison',{}).get('common_settings_sha256'))
    settings.append(setting)
    write_csv('signals/'+m['archive_run_id']+'_measured.csv',
              [dict(receipt_elapsed_s=t,fx_base_N=f[0],fy_base_N=f[1],fz_base_N=f[2],signal='F_meas',
                    time_basis='host receipt; no acquisition timestamp')
               for t,f in zip(r['ft'],r['force']) if 0<=t<=m['execution_duration_s']])
    write_csv('signals/'+m['archive_run_id']+'_sent.csv',
              [dict(receipt_elapsed_s=t,F_sent_controller_fz_N=a[8],target_x_base_mm=a[0],target_y_base_mm=a[1],target_z_base_mm=a[2],
                    gate=d['contact'],processing_control=d.get('processing'),plan_id=row['plan_id'],
                    dt_control_s=d['dt_s'],force_source=d.get('force_source'),signal='F_sent',applied_force=None)
               for t,a,d,row in zip(r['ct'],r['action'],r['details'],sent)])
    return r
