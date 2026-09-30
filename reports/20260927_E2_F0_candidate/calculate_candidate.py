"""Compute a review-only teacher-channel constant. Never modify runtime F0."""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path('/home/eunseop/nrs_imitation')
OUT=Path(__file__).resolve().parent
EXP=ROOT/'experiments/e2_force_ablation_20260926'
DATA=ROOT/'datasets/polishing/single_cam/20260910_90deg_rel_single/imitation_form'


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def time_mean(t,f,a,b):
    assert t[0] <= a < b <= t[-1] and np.all(np.diff(t)>0)
    edges=np.unique(np.r_[a,t[(t>a)&(t<b)],b])
    indices=np.searchsorted(t,edges[:-1],side='right')-1
    assert np.max(np.diff(edges)) <= .2
    weights=np.diff(edges)
    # All values in the interval, including low and negative force, survive.
    return float(np.dot(weights,f[indices])/(b-a))


def main():
    phases=json.loads((OUT/'pose_phase_proposals.json').read_text())
    review=json.loads((OUT/'visual_review.json').read_text())
    assert review['phase_sha256']==digest(OUT/'pose_phase_proposals.json')
    split=json.loads((EXP/'teacher_preview/split.json').read_text())
    assert set(e['episode'] for e in phases['episodes'])==set(split['train'])
    cfg=json.loads((EXP/'config.json').read_text());cfg_sha=digest(EXP/'config.json')
    rows=[];signals={};variants={k:[] for k in ['base','expand_0.25s','shrink_0.25s',
        'expand_0.50s','shrink_0.50s','height_0.5mm','height_1.5mm','trapezoidal']}
    for e in phases['episodes']:
        with h5py.File(DATA/e['episode']) as f:
            force=f['action/force'][:].astype(float)
        with h5py.File(e['source_h5']) as f:
            g=f['episodes/'+e['source_group']];t=g['sample_time_unix'][:];t-=t[0]
            np.testing.assert_array_equal(force,g['ft'][:])
        assert force.shape==(len(t),3) and np.isfinite(force).all()
        a,b=e['start_s'],e['end_s'];means=[time_mean(t,force[:,k],a,b) for k in range(3)]
        variants['base'].append(means[2])
        for width in [.25,.50]:
            variants['expand_%.2fs'%width].append(time_mean(t,force[:,2],max(t[0],a-width),min(t[-1],b+width)))
            variants['shrink_%.2fs'%width].append(time_mean(t,force[:,2],a+width,b-width))
        for width in [.5,1.5]:
            band=e['geometric_boundary_sensitivity'][str(width)]
            variants['height_%.1fmm'%width].append(time_mean(t,force[:,2],band['start_s'],band['end_s']))
        edges=np.unique(np.r_[a,t[(t>a)&(t<b)],b]);v=np.interp(edges,t,force[:,2])
        trap=float(np.trapz(v,edges)/(b-a));variants['trapezoidal'].append(trap)
        row=dict(episode=e['episode'],source_group=e['source_group'],start_s=a,end_s=b,duration_s=b-a,
            Fx_time_mean_N=means[0],Fy_time_mean_N=means[1],Fz_time_mean_N=means[2],
            interval_samples=int(np.sum((t>=a)&(t<b))),success_confirmed=True,
            boundary_status='pose_and_boundary_image_reviewed_approximate',source_file_sha256=digest(DATA/e['episode']))
        rows.append(row);signals[e['episode']]=(t,force)
    means=np.asarray(variants['base']);mean=float(means.mean());rounded=float(np.round(mean))
    summaries={k:float(np.mean(v)) for k,v in variants.items()}
    out=dict(schema='E2_teacher_channel_F0_draft_v1',status='preliminary_not_calibrated_not_frozen',
        calculated_at=datetime.now().astimezone().isoformat(),signal='stored action/force[:,2] = filtered teacher ft[:,2]',
        quantity='recorded teacher Fz; candidate for the existing learned-force command channel, not certified surface-normal force',
        teacher_Fz_constant_candidate_N=mean,rounded_teacher_channel_candidate_N=rounded,
        proposed_rounding='nearest 1 N for a future reviewed recipe; no runtime parameter set',
        normal_force_N=None,command_fz_N=None,validated_operating_range_N=None,
        sample_count=len(rows),processing_samples=sum(r['interval_samples'] for r in rows),
        proposed_processing_duration_s=float(sum(r['duration_s'] for r in rows)),
        weighting='Actual-time ZOH mean within each proposed processing interval; equal weight across all 38 training episodes.',
        per_episode_mean_min_N=float(means.min()),per_episode_mean_max_N=float(means.max()),
        per_episode_mean_std_N=float(means.std(ddof=1)),boundary_sensitivity_N=summaries,
        low_force_samples_removed=False,absolute_value_applied=False,new_zero_or_gravity_correction_applied=False,
        excluded_successful_train_episodes=[],evaluation_episodes_used=False,E1_R_18N_reused=False,E2_BC_outcomes_used=False,
        operator_confirmation=str(EXP/'F0_operator_confirmation_20260927.json'),
        phase_proposals_sha256=digest(OUT/'pose_phase_proposals.json'),visual_review_sha256=digest(OUT/'visual_review.json'),
        runtime_config_sha256=cfg_sha,calibration=cfg['calibration'],
        remaining=['Approximate processing boundaries need acceptance for final recipe selection.',
          'Teacher force-axis/zero/calibration relation to current TCP target is not independently verified.',
          'No validated constant-force operating range was supplied; observed teacher statistics are not operating limits.',
          'Final numerical recipe/operator freeze confirmation has not been given.'],
        hardware_config_modified=False,hardware_executed=False)
    (OUT/'candidate.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'episode_means.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,ax=plt.subplots(figsize=(13,4.5));x=np.arange(len(rows))
    ax.bar(x,means,color='#3f789c');ax.axhline(mean,color='#b25033',label='Equal-episode mean %.3f N'%mean)
    ax.set_xticks(x,[r['episode'].replace('episode_','').replace('.hdf5','') for r in rows],fontsize=8)
    ax.set_xlabel('Training episode ID');ax.set_ylabel('Recorded teacher Fz mean [N]')
    ax.set_title('Preliminary constant from proposed work intervals — not a calibrated TCP recipe')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/'candidate_overview.png',dpi=150)
    with PdfPages(OUT/'F0_candidate_review.pdf') as pdf:
        pdf.savefig(fig);plt.close(fig)
        for off in range(0,len(rows),6):
            fig,axes=plt.subplots(3,2,figsize=(12,10))
            for ax,r in zip(axes.flat,rows[off:off+6]):
                t,f=signals[r['episode']];ax.plot(t,f[:,2],label='stored teacher Fz',lw=1)
                ax.axvspan(r['start_s'],r['end_s'],color='#3f789c',alpha=.15,label='pose-only proposed work')
                ax.axhline(mean,color='#b25033',ls='--',lw=.8)
                ax.set_title(r['episode']+' | mean %.2f N'%r['Fz_time_mean_N']);ax.set_xlabel('time [s]');ax.set_ylabel('N');ax.grid(alpha=.2)
            for ax in list(axes.flat)[len(rows[off:off+6]):]:ax.axis('off')
            fig.suptitle('All samples within each proposed interval are retained; boundaries were selected before viewing force means.')
            fig.tight_layout(rect=(0,0,1,.96));pdf.savefig(fig);plt.close(fig)
    assert digest(EXP/'config.json')==cfg_sha
    print(json.dumps({k:out[k] for k in ['status','teacher_Fz_constant_candidate_N','rounded_teacher_channel_candidate_N',
        'sample_count','processing_samples','proposed_processing_duration_s','per_episode_mean_min_N',
        'per_episode_mean_max_N','boundary_sensitivity_N']},indent=2))


if __name__=='__main__':main()
