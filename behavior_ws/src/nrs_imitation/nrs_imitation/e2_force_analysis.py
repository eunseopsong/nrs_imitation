"""Offline E2 force provenance, phase alignment, and metrology (no ROS)."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .e2_ablation import digest, object_hash


class Unavailable(ValueError):
    """Known missing/invalid evidence: report NA with this reason."""


def time_array(values):
    t=np.asarray(values,float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise Unavailable('insufficient/nonfinite/nonmonotonic source timestamps')
    return t


def align(t, values, query, max_gap, mode='linear'):
    """No extrapolation or filling missing data with zero; exact samples survive gaps."""
    t=time_array(t);v=np.asarray(values,float);q=np.asarray(query,float)
    if len(v)!=len(t) or not np.isfinite(v).all():raise Unavailable('invalid signal values')
    right=np.searchsorted(t,q,side='left');right=np.clip(right,0,len(t)-1)
    left=np.maximum(0,right-1);exact=np.isclose(t[right],q,rtol=0,atol=1e-9)
    valid=(q>=t[0])&(q<=t[-1])&(exact|((t[right]-t[left])<=max_gap))
    if mode=='zoh':
        index=np.maximum(0,np.searchsorted(t,q,side='right')-1)
        out=v[index].copy()
        valid &= (q-t[index])<=max_gap
    else:
        if v.ndim==1:out=np.interp(q,t,v)
        else:out=np.column_stack([np.interp(q,t,v[:,i]) for i in range(v.shape[1])])
    return out,valid


def normal_force(force, frame, calibration, *, time=None, pose=None, clock=None):
    if not calibration.get('normal_force_verified'):
        raise Unavailable('normal-force frame/sign/zero/filter calibration not verified')
    if not calibration.get('calibration_id') or not calibration.get('gravity_zero_filter_evidence'):
        raise Unavailable('calibration identity and correction/filter evidence missing')
    n=np.asarray(calibration.get('surface_normal_base'),float);sign=calibration.get('compression_sign')
    if n.shape!=(3,) or not np.isfinite(n).all() or not np.isclose(np.linalg.norm(n),1.) or sign not in (-1,1):
        raise Unavailable('fixed unit normal and compression sign required')
    f=np.asarray(force,float)
    if f.ndim!=2 or f.shape[1]!=3 or not np.isfinite(f).all():
        raise Unavailable('full finite 3D force required; scalar base Fz cannot be rotated')
    if frame=='robot_base':base=f
    elif frame=='teacher_calibrated':
        matrix=np.asarray(calibration.get('teacher_rotation_base'),float)
        if matrix.shape!=(3,3) or not np.allclose(matrix.T@matrix,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(matrix),1.):
            raise Unavailable('verified teacher-to-base rotation unavailable')
        base=Rotation.from_matrix(matrix).apply(f)
    elif frame=='controller_tcp':
        if calibration.get('rotation_basis')!='actual_feedback_tcp' or pose is None or time is None:
            raise Unavailable('controller actual-feedback TCP basis/pose unavailable')
        if pose.get('clock_id')!=clock or pose.get('pose_kind')!='actual_feedback':
            raise Unavailable('clock or actual/command pose mismatch')
        pt=time_array(pose['time']);t=np.asarray(time,float)
        _,valid=align(pt,pose['rotvec'],t,calibration.get('max_pose_age_s',.2))
        age=np.min(np.abs(pt[:,None]-t[None,:]),axis=0)
        valid &= age<=calibration.get('max_pose_age_s',.2)
        if not valid.all():raise Unavailable('stale/missing pose transform or long pose gap')
        r=Slerp(pt,Rotation.from_rotvec(pose['rotvec']))(t)
        base=r.apply(f)
    else:raise Unavailable('unverified force frame: '+str(frame))
    # Already-calibrated vector; never apply another gravity/sign correction upstream.
    return sign*(base@n)


def weighted_rmse(actual, reference, weights):
    a,b,w=map(lambda x:np.asarray(x,float),(actual,reference,weights))
    if not (a.shape==b.shape==w.shape) or not np.isfinite(np.r_[a,b,w]).all() or np.any(w<0) or w.sum()<=0:
        raise Unavailable('invalid common alignment weights/values')
    return float(np.sqrt(np.sum(w*(a-b)**2)/np.sum(w)))


def evaluate_profile(reference, run, analysis):
    """Run is canonical normal-force signals; caller must supply verified provenance."""
    result=dict(E_profile_N=None,E_target_N=None,E_track_applied_N=None,E_discrepancy_sent_N=None,
        phase_coverage=0.,sample_coverage=0.,processing_time_s=None,n_profile_valid=0,
        partial_profile_diagnostic_N=None,reasons=[],phase_details=[])
    if run.get('partial_event_file'):
        result['reasons'].append('partial event stream; phase completion cannot be certified');return result
    if not run.get('normal_force_verified') or run.get('calibration_hash')!=reference.get('calibration_hash'):
        result['reasons'].append('run/reference normal-force calibration missing or differs');return result
    if run.get('clock_id') is None or any(s.get('clock_id')!=run['clock_id'] for s in run.get('signals',{}).values()):
        result['reasons'].append('signal clocks differ or clock identity missing');return result
    measured=[];teacher=[];target=[];weights=[];time_total=0.;covered=0;valid_total=0;grid_total=0
    signals=run.get('signals',{});gap=analysis['max_gap_s'];grid_n=analysis['grid_points_per_phase']
    target_valid=True;complete_times=[]
    for ref in reference['phases']:
        grid=np.linspace(0,1,grid_n);w=np.ones(grid_n);w[[0,-1]]=.5
        grid_total+=grid_n;phase=next((p for p in run['phases'] if p['phase_id']==ref['phase_id']),None)
        detail=dict(phase_id=ref['phase_id'],complete=False,duration_s=None,valid_fraction=0.)
        result['phase_details'].append(detail)
        if not phase or phase.get('complete') is not True:
            result['reasons'].append('missing/incomplete phase '+ref['phase_id']);continue
        start,end=phase['start'],phase['end']
        if not np.isfinite([start,end]).all() or end<=start:
            result['reasons'].append('invalid phase time');continue
        if complete_times and start<complete_times[-1][1]:
            result['reasons'].append('overlapping/out-of-order phases');continue
        complete_times.append((start,end));q=start+grid*(end-start)
        try:
            meas,valid=align(signals['F_meas']['time'],signals['F_meas']['values'],q,gap)
            fref=np.interp(grid,ref['progress'],ref['force_normal_N'])
        except (Unavailable,KeyError) as exc:result['reasons'].append(str(exc));continue
        valid_total+=int(valid.sum());detail['valid_fraction']=float(valid.mean())
        if not valid.all():result['reasons'].append('missing/gap samples '+ref['phase_id'])
        detail.update(complete=True,duration_s=end-start);time_total+=end-start
        if valid.all():covered+=1
        measured.extend(meas[valid]);teacher.extend(fref[valid]);weights.extend(w[valid])
        try:
            tar,tvalid=align(signals['F_tar']['time'],signals['F_tar']['values'],q,gap,'zoh')
            target_valid &= bool(tvalid[valid].all());target.extend(tar[valid])
        except (Unavailable,KeyError):target_valid=False
    count=len(reference['phases']);result['phase_coverage']=covered/count if count else 0.
    result['sample_coverage']=valid_total/grid_total if grid_total else 0.
    result['processing_time_s']=time_total if complete_times else None
    if measured:
        result['partial_profile_diagnostic_N']=weighted_rmse(measured,teacher,weights)
    if count and covered==count and not result['reasons']:
        result['E_profile_N']=result['partial_profile_diagnostic_N'];result['n_profile_valid']=1
        if target_valid:result['E_target_N']=weighted_rmse(target,teacher,weights)
    # Tracking/discrepancy are computed on real time, not phase-warped time.
    for key,outkey in [('F_cmd_applied','E_track_applied_N'),('F_sent','E_discrepancy_sent_N')]:
        if key not in signals:
            result['reasons'].append(key+' unavailable');continue
        squared=[];dt_all=[]
        try:
            for start,end in complete_times:
                mt=time_array(signals['F_meas']['time'])
                ct=time_array(signals[key]['time'])
                # Integrate every ZOH change, including commands between sensor samples.
                edges=np.unique(np.r_[start,mt[(mt>start)&(mt<end)],ct[(ct>start)&(ct<end)],end])
                m,vm=align(mt,signals['F_meas']['values'],edges[:-1],gap,'zoh')
                cmd,vc=align(signals[key]['time'],signals[key]['values'],edges[:-1],gap,'zoh')
                if not (vm&vc).all() or np.any(np.diff(edges)>gap):raise Unavailable('gap in tracking interval')
                squared.extend((m-cmd)**2);dt_all.extend(np.diff(edges))
            if dt_all:result[outkey]=float(np.sqrt(np.average(squared,weights=dt_all)))
        except (Unavailable,KeyError) as exc:result['reasons'].append(key+': '+str(exc))
    return result


def episode_signal(path):
    """Original episode rows exactly once; verify timestamp/source mapping."""
    import h5py
    path=Path(path)
    with h5py.File(path,'r') as f:
        if f.attrs.get('truncated',0):raise Unavailable('truncated episode lacks original time mapping')
        force=f['action/force'][:];position=f['action/position'][:]
        source=str(f.attrs['source_h5']);group=str(f.attrs['source_episode'])
        pose=f['analysis/absolute/action_position'][:]
    with h5py.File(source,'r') as f:
        g=f['episodes/'+group];t=g['sample_time_unix'][:];original=g['ft'][:]
        if not np.array_equal(force,original) or not np.array_equal(g['source_index'][:],np.arange(len(t))):
            raise Unavailable('teacher force/time mapping differs from original source')
        clock=str(g.attrs.get('sync_clock','unknown'))
    time_array(t)
    if len(force)!=len(t):raise Unavailable('teacher timestamp length mismatch')
    return dict(time=t-t[0],force=force,pose=pose,position=position,
        provenance=dict(episode=str(path.resolve()),episode_sha256=digest(path),source_h5=source,
            source_sha256=digest(source),source_group=group,source_clock=clock,time_origin_unix=float(t[0]),
            force_frame='teacher_calibrated',force_semantics='filtered teacher ft copied unchanged to action/force; not independently measured robot applied target'))


def build_reference(request, calibration, dataset, split, analysis):
    if len(request.get('episodes',[]))!=1:
        raise Unavailable('Select one strategy-matched evaluation episode explicitly; no automatic averaging')
    e=request['episodes'][0]
    if not e.get('success_confirmed') or not e.get('selection_evidence'):
        raise Unavailable('evaluation episode success/strategy selection unconfirmed')
    source=episode_signal(Path(dataset)/e['episode'])
    normal=normal_force(source['force'],'teacher_calibrated',calibration)
    ref=dict(schema='E2_force_reference_v1',status='candidate',calibration_hash=object_hash(calibration),
        source=source['provenance'],in_sample=e['episode'] in split['train'],
        selection_evidence=e['selection_evidence'],analysis_hash=object_hash(analysis),phases=[])
    for phase in e['phases']:
        start,end=phase['start_s'],phase['end_s'];grid=np.linspace(0,1,analysis['grid_points_per_phase'])
        if end<=start:raise Unavailable('nonpositive reference phase')
        force,valid=align(source['time'],normal,start+grid*(end-start),analysis['max_gap_s'])
        if not valid.all():raise Unavailable('reference phase crosses missing/gap data')
        ref['phases'].append(dict(phase_id=phase['phase_id'],progress=grid.tolist(),force_normal_N=force.tolist(),
                                 duration_s=end-start,source_start_s=start,source_end_s=end))
    if not ref['phases']:raise Unavailable('No annotated reference phases')
    return ref


def f0_candidate(request, calibration, dataset, split):
    """Time weighted within episode, equal episode weights; no force threshold."""
    means=[];details=[]
    for e in request.get('episodes',[]):
        if e['episode'] not in split['train'] or not e.get('success_confirmed') or not e.get('selection_evidence'):
            raise Unavailable('F0 requires explicitly confirmed train/calibration episodes, not evaluation selection')
        source=episode_signal(Path(dataset)/e['episode']);f=normal_force(source['force'],'teacher_calibrated',calibration)
        total=0.;duration=0.;previous=-np.inf
        for phase in e['phases']:
            a,b=phase['start_s'],phase['end_s']
            if b<=a or a<previous:raise Unavailable('overlapping or invalid processing annotations')
            previous=b;t=source['time'];edges=np.unique(np.r_[a,t[(t>a)&(t<b)],b])
            values,valid=align(t,f,edges[:-1],.2,'zoh');weights=np.diff(edges)
            if not valid.all() or np.any(weights>.2):raise Unavailable('missing/gap F0 interval')
            total+=float(weights@values);duration+=float(weights.sum())
        if duration<=0:raise Unavailable('No explicitly annotated processing interval')
        means.append(total/duration);details.append(dict(**source['provenance'],phases=e['phases'],duration_s=duration,mean_N=total/duration))
    if not means:raise Unavailable('No confirmed processing annotations; F0 remains unresolved')
    return dict(status='candidate_not_frozen',normal_force_N=float(np.mean(means)),command_fz_N=None,
        calibration_hash=object_hash(calibration),episodes=details,
        weighting='time ZOH within episode; equal total weight per episode',
        exclusions=request.get('exclusions',[]),low_force_samples_removed=False,
        operating_range_candidates=None,reason='command conversion and validated operating range require calibration/operator evidence')


def metrology(row):
    result=dict(depth_reduction_pct=None,residual_depth_um=None,local_dishing_um=None,Ra_after=None,
                metrology_validity='NA: no valid stylus measurement')
    if str(row.get('measurement_valid','')).lower() not in ('true','1'):return result
    try:
        before,after=float(row['before_depth_um']),float(row['after_depth_um'])
        limit=float(row['quantification_limit_um'])
        if not np.isfinite([before,after,limit]).all() or limit<0 or before<=limit or after<limit:
            raise ValueError('below quantification limit or invalid baseline')
        if not row.get('instrument_settings_id') or not row.get('reference_region'):
            raise ValueError('instrument/reference provenance missing')
        result.update(depth_reduction_pct=100*(before-after)/before,residual_depth_um=after,
                      local_dishing_um=float(row['dishing_um']) if row.get('dishing_um') else None,
                      Ra_after=float(row['Ra_after']) if row.get('Ra_after') else None,metrology_validity='valid')
    except (KeyError,ValueError) as exc:result['metrology_validity']='NA: '+str(exc)
    return result
