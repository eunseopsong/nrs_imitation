"""Draft pose-only phase proposals for visual review; does not freeze a recipe."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import csv
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/eunseop/nrs_imitation')
OUT = Path(__file__).resolve().parent
DATA = ROOT/'datasets/polishing/single_cam/20260910_90deg_rel_single/imitation_form'
EXP = ROOT/'experiments/e2_force_ablation_20260926'


def pose_phase(t, z):
    """Three linear Z segments: approach / slow vertical work / retract.

    This is a force-blind proposal, not contact ground truth. Review images and
    keep boundary uncertainty. No target force is an input to the segmentation.
    """
    t = np.asarray(t, float); z = np.asarray(z, float); n = len(t)
    prefix = np.vstack([np.zeros(5), np.cumsum(np.c_[t, t*t, z, z*z, t*z], axis=0)])
    i = np.arange(n)[:, None]; j = np.arange(1, n+1)[None, :]
    size = j-i; good = size >= 2; count = np.maximum(size, 1)
    sums = prefix[j]-prefix[i]
    st, st2, sz, sz2, stz = np.moveaxis(sums, -1, 0)
    xx = st2-st*st/count; xy = stz-st*sz/count
    slope = np.divide(xy, xx, out=np.zeros_like(xy), where=good & (xx > 1e-12))
    cost = np.maximum(0, sz2-sz*sz/count-slope*xy)
    cost[~good] = np.inf
    candidates = []
    for a in range(2, n-3):
        if not 1.0 <= t[a] <= .5*t[-1]:
            continue
        b = np.arange(a+2, n-1)
        valid = ((t[b]-t[a] >= 2.0) & (t[-1]-t[b] >= 1.0) &
                 (np.abs(slope[a, b-1]) <= 2.0) &
                 (slope[0, a-1] < -2.0) & (slope[b, n-1] > 2.0))
        if not valid.any():
            continue
        b = b[valid]
        losses = cost[0, a-1]+cost[a, b-1]+cost[b, n-1]
        k = int(np.argmin(losses))
        candidates.append((float(losses[k]), a, int(b[k])))
    if not candidates:
        raise ValueError('No approach/work/retract pose proposal')
    loss, rough_a, rough_b = min(candidates)
    # Global straight-line fits are biased by the curved home/return paths.
    # Estimate the work-height trend from the interior, then locate approach
    # and retract at that height. Keep one contiguous interval, including any
    # internal variation; never drop low-force samples.
    interior = (t >= t[rough_a]+1.) & (t <= t[rough_b]-1.)
    m, c = np.polyfit(t[interior], z[interior], 1)
    residual = z-(m*t+c)
    search = (t >= t[rough_a]-1.5) & (t <= t[rough_b]+1.5)
    bands = {}
    for width in (.5, 1., 1.5):
        near = np.flatnonzero(search & (np.abs(residual) <= width))
        if len(near) < 2:
            raise ValueError('Work-height fit requires manual phase annotation')
        bands[str(width)] = dict(start_s=float(t[near[0]]),end_s=float(t[near[-1]]))
    a=int(np.searchsorted(t,bands['1.0']['start_s']))
    b=int(np.searchsorted(t,bands['1.0']['end_s']))
    return a, b, dict(z_fit_rmse_mm=float(np.sqrt(loss/n)),
        work_z_slope_mm_s=float(m),work_height_intercept_mm=float(c),
        initial_segment_start_s=float(t[rough_a]),initial_segment_end_s=float(t[rough_b]),
        work_height_residual_range_mm=[float(residual[a:b+1].min()),float(residual[a:b+1].max())],
        geometric_boundary_sensitivity=bands)


def load_episode(name):
    with h5py.File(DATA/name) as f:
        source = str(f.attrs['source_h5']); group = str(f.attrs['source_episode'])
        pose = f['analysis/absolute/action_position'][:].astype(float)
        with h5py.File(source) as original:
            g = original['episodes/'+group]
            t = g['sample_time_unix'][:]; t -= t[0]
    return t, pose, source, group


def main():
    split = json.loads((EXP/'teacher_preview/split.json').read_text())
    confirmation = json.loads((EXP/'F0_operator_confirmation_20260927.json').read_text())
    assert confirmation['scope']['all_42_normal_successful']
    names = sorted(split['train'], key=lambda s:int(s.split('_')[1].split('.')[0]))
    proposals = []
    for name in names:
        t, pose, source, group = load_episode(name)
        a, b, details = pose_phase(t, pose[:, 2])
        proposals.append(dict(episode=name,source_h5=source,source_group=group,
            success_confirmed=True,success_evidence=str(EXP/'F0_operator_confirmation_20260927.json'),
            start_s=float(t[a]),end_s=float(t[b]),start_index=a,end_index=b,
            boundary_status='pose_only_proposal_pending_visual_review',**details))
    document = dict(created_at=datetime.now().astimezone().isoformat(),
        status='proposals_not_frozen',force_values_used_for_selection=False,
        criterion='Initial minimum squared Z residual in three linear segments; work>=2 s with '
        '|dz/dt|<=2 mm/s, approach/retract slopes <-2/>2 mm/s. Fit work-height trend to the '
        'interior after removing 1 s from each rough edge; locate first/last pose within 1 mm '
        'of that trend, searching 1.5 s around the rough interval. Keep every sample between '
        'boundaries. Check 0.5 and 1.5 mm boundary sensitivity. These are offline proposal '
        'settings, not safety limits or force thresholds.',
        input='stored absolute teacher pose; images for review; no wrench input',
        held_out_episodes=split['validation'],episodes=proposals)
    (OUT/'pose_phase_proposals.json').write_text(json.dumps(document,ensure_ascii=False,indent=2)+'\n')
    for offset in range(0,len(proposals),6):
        page = proposals[offset:offset+6]
        fig, axes = plt.subplots(len(page),7,figsize=(21,2.1*len(page)),squeeze=False)
        for row,e in enumerate(page):
            t,pose,_,_ = load_episode(e['episode'])
            times=[e['start_s']-.25,e['start_s'],e['start_s']+.25,
                   e['end_s']-.25,e['end_s'],e['end_s']+.25]
            with h5py.File(DATA/e['episode']) as f:
                for ax,at in zip(axes[row,:6],times):
                    k=int(np.argmin(abs(t-at)))
                    ax.imshow(f['observations/images/cam0'][k]); ax.axis('off')
                    ax.set_title(e['episode']+' | %.2fs'%t[k],fontsize=9)
            ax=axes[row,-1];ax.plot(t,pose[:,2],lw=1)
            ax.axvspan(e['start_s'],e['end_s'],color='tab:blue',alpha=.18)
            ax.set_xlabel('time [s]');ax.set_ylabel('stored Z [mm]');ax.grid(alpha=.2)
            ax.set_title('Pose-only proposal',fontsize=9)
        fig.suptitle('Training teachers only: start [-0.25s / proposed / +0.25s] | end [-0.25s / proposed / +0.25s]',fontsize=13)
        fig.tight_layout(rect=(0,0,1,.975))
        fig.savefig(OUT/('boundary_review_%02d.png'%(offset//6+1)),dpi=130);plt.close(fig)
    print(json.dumps({'phase_proposals':len(proposals),'review_pages':(len(proposals)+5)//6,'force_used':False}))


if __name__=='__main__':
    main()
