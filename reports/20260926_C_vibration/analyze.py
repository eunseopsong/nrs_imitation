"""Read recorded CSVs only; no ROS, robot I/O, policy inference or runtime edits."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/eunseop/nrs_imitation')
LOG = ROOT / 'logs/inference_metrics'
OUT = Path(__file__).parent
KEYS = ['x','y','z','rx','ry','rz','fx','fy','fz']


def load_rows(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def arr(rows, keys):
    # Position-mode return commands intentionally have no force columns.
    return np.array([[float(r[k]) if r[k] else np.nan for k in keys] for r in rows])


def series(rows, t0):
    return np.array([int(r['receipt_monotonic_ns']) / 1e9 - t0 for r in rows])


def smooth35(a):
    p = np.pad(a, ((17,17),(0,0)), mode='edge')
    return np.stack([np.convolve(p[:,j], np.ones(35)/35, mode='valid') for j in range(a.shape[1])], axis=1)


def run(prefix):
    ep = next(LOG.glob(prefix + '_executor*'))
    pp = next(LOG.glob(prefix + '_2026*'))
    events = [json.loads(s) for s in (ep/'events.jsonl').read_text().splitlines() if s.strip()]
    start = next(e for e in events if e['event']=='execution_start')
    t0 = start['receipt_monotonic_ns']/1e9
    rows = load_rows(ep/'commands.csv')
    stages = {}
    for stage in ['time_sampled','contact_gated','node_sent']:
        rr = [r for r in rows if r['command_stage']==stage]
        stages[stage] = dict(rows=rr, t=series(rr,t0), a=arr(rr,KEYS),
            ids=np.array([int(r['plan_id']) for r in rr]),
            contact=np.array([json.loads(r['details'])['contact'] for r in rr]))
    pose = load_rows(ep/'tcp_pose.csv')
    force = load_rows(ep/'wrench.csv')
    return dict(prefix=prefix, ep=ep, pp=pp, events=events, t0=t0,
        stages=stages, pt=series(pose,t0), pose=arr(pose,KEYS[:6]),
        ft=series(force,t0), force=arr(force,KEYS[6:]),
        metadata=json.loads((pp/'metadata.json').read_text()),
        summary=json.loads((ep/'summary.json').read_text()))


runs = {k:run(p) for k,p in {'C':'RTC_C_20260926T175131',
    'T':'RTC_T_20260926T172827','R':'RTC_R_20260926T170642'}.items()}
c = runs['C']
evidence = {'generated_at':datetime.now().astimezone().isoformat(), 'runs':{},
    'interpretation_limits':[
        'Requested trajectory jumps are not instantaneous physical robot displacements.',
        'Measured force is base-frame republished Fz; commanded force depends on TCP controller settings. No frame-matched tracking error is claimed.',
        'About 20 Hz feedback cannot resolve all high-frequency mechanical vibration or sensor acquisition timing.',
        'The 35-point smoothing comparison is an offline mathematical comparison, not a validated hardware fix.',
        'Absence of a driver log event does not rule out every scheduling or network disturbance.',
    ]}
for label,r in runs.items():
    stages = r['stages']
    sent = stages['node_sent']
    raw = stages['time_sampled']
    dt = np.diff(sent['t'])
    v = np.diff(sent['a'][:,:3],axis=0)/dt[:,None]
    v4 = v[(sent['t'][1:]>.05)&(sent['t'][1:]<4)]
    flips = ((v4[1:]*v4[:-1]<0)&(np.abs(v4[1:])>1)&(np.abs(v4[:-1])>1)).sum(axis=0)
    force_mask = (r['ft']>=0)&(r['ft']<=14.22)
    changes = np.flatnonzero(np.diff(raw['ids']))+1
    record = dict(executor_dir=str(r['ep']), provider_dir=str(r['pp']),
        command_rows=r['summary']['counts']['commands']['written'], node_sent_rows=len(sent['t']),
        node_sent_max_gap_ms=float(dt.max()*1000),
        sent_XYZ_direction_reversals_first4s_over_1mm_s=flips.tolist(),
        contact_transitions_first1s=int(np.count_nonzero(np.diff(sent['contact'][sent['t']<=1]))),
        contact_transitions_first4s=int(np.count_nonzero(np.diff(sent['contact'][sent['t']<=4]))),
        contact_transitions_total=int(np.count_nonzero(np.diff(sent['contact']))),
        sent_fz_min_N=float(np.nanmin(sent['a'][:,8])), sent_fz_max_N=float(np.nanmax(sent['a'][:,8])),
        measured_base_fz_min_first14s_N=float(r['force'][force_mask,2].min()),
        measured_base_fz_max_first14s_N=float(r['force'][force_mask,2].max()),
        plan_switches=[dict(at_s=float(raw['t'][i]), old_plan=int(raw['ids'][i-1]),new_plan=int(raw['ids'][i]),
            requested_XYZ_jump_mm=float(np.linalg.norm(raw['a'][i,:3]-raw['a'][i-1,:3])),
            sent_XYZ_step_mm=float(np.linalg.norm(sent['a'][i,:3]-sent['a'][i-1,:3]))) for i in changes],
        logger_drops=sum(v['dropped'] for v in r['summary']['counts'].values()),
        logger_write_errors=r['summary']['write_error_count'], drained=r['summary']['drained'])
    evidence['runs'][label]=record

provider = load_rows(c['pp']/'commands.csv')
plans = {}
for plan_id in sorted({int(r['plan_id']) for r in provider if r['command_stage']=='postprocessed'}):
    rows = [r for r in provider if r['command_stage']=='postprocessed' and int(r['plan_id'])==plan_id]
    rows.sort(key=lambda r:int(r['action_index']))
    a=arr(rows,KEYS); sm=smooth35(a[:,:3])
    plans[plan_id]={'a':a,'smooth_xyz':sm}
evidence['C_plan_adjacency']=[dict(plan_id=i, points=len(p['a']),
    raw_XYZ_step_mm_p50_p95_max=np.percentile(np.linalg.norm(np.diff(p['a'][:,:3],axis=0),axis=1),[50,95,100]).tolist(),
    offline_MA35_XYZ_step_mm_p50_p95_max=np.percentile(np.linalg.norm(np.diff(p['smooth_xyz'],axis=0),axis=1),[50,95,100]).tolist()) for i,p in plans.items()]
peak=int(np.argmax(c['force'][:,2]))
evidence['C_peak_measured_base_fz']={'at_s':float(c['ft'][peak]),'fz_N':float(c['force'][peak,2])}
evidence['C_terminal_events']=[e for e in c['events'] if e['event'] in ('stop_requested','manual_abort','safety_stop','normal_completion')]
references={
    'legacy_service_C':'C_force_obs_ON_E1_20260916_20260920T171726_1789892246083511592_ua8reqys',
    'prior_vibrating_timed_C':'E2_line_C_r01_20260920T201422_1789902862705332180_75h5dukq'}
evidence['reference_comparisons']={}
for label,name in references.items():
    old=json.loads((LOG/name/'metadata.json').read_text())
    evidence['reference_comparisons'][label]={
        'run':name, **{k+'_same':old.get(k)==c['metadata'].get(k) for k in ['checkpoint','normalizer','resolved_runtime']},
        'executor_settings_same':old.get('e2_configuration',{}).get('executor')==c['metadata']['e2_configuration']['executor'],
        'artifacts_old':{Path(a['path']).name:a.get('sha256') for a in old['artifacts']},
        'artifacts_current':{Path(a['path']).name:a.get('sha256') for a in c['metadata']['artifacts']}}
files=[c[k]/name for k in ['ep','pp'] for name in ['metadata.json','events.jsonl','commands.csv','tcp_pose.csv','wrench.csv','summary.json']]
evidence['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')

plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
fig,axs=plt.subplots(3,2,figsize=(15,11),constrained_layout=True)
fig.suptitle('C vibration diagnosis — 2026-09-26 17:51:31 (recorded data only)',fontsize=15)
raw=c['stages']['time_sampled'];sent=c['stages']['node_sent']
ax=axs[0,0]
ax.plot(raw['t'],raw['a'][:,2],lw=.9,label='Time-sampled requested Z')
ax.plot(sent['t'],sent['a'][:,2],lw=1.5,label='Node-sent Z')
ax.plot(c['pt'],c['pose'][:,2],lw=1.2,label='Measured TCP Z')
ax.set(xlim=(0,14.3),ylabel='Z (mm)',title='Pose command / measured motion');ax.legend(fontsize=8)
ax=axs[0,1]
ax.plot(c['ft'],c['force'][:,2],color='firebrick',lw=1.1,label='Measured base-frame Fz')
ax.scatter([c['ft'][peak]],[c['force'][peak,2]],color='firebrick')
ax.annotate(f"{c['force'][peak,2]:.1f} N",(c['ft'][peak],c['force'][peak,2]),xytext=(7.5,110),arrowprops={'arrowstyle':'->'})
ax.set(xlim=(0,14.3),ylabel='Fz (N)',title='Force spike after contact');ax.legend(fontsize=8)
ax=axs[1,0]
for label,color in [('C','firebrick'),('T','steelblue'),('R','seagreen')]:
    s=runs[label]['stages']['node_sent'];v=np.diff(s['a'][:,0])/np.diff(s['t'])
    ax.plot(s['t'][1:],v,color=color,lw=.9,label=label)
ax.set(xlim=(0,4),ylim=(-13,13),ylabel='Sent X velocity (mm/s)',title='C reverses direction repeatedly; R/T do not in first 4 s');ax.legend(fontsize=8)
ax=axs[1,1]
ax.plot(raw['t'],raw['a'][:,8],lw=.8,alpha=.65,label='Requested controller Fz')
ax.plot(sent['t'],sent['a'][:,8],lw=1.2,label='Node-sent controller Fz')
ax.set(xlim=(0,4),ylabel='Controller-axis target (N)',title='Early force gate chatter / signed targets')
ax2=ax.twinx();ax2.step(sent['t'],sent['contact'].astype(int),where='post',color='gray',alpha=.35,lw=.7)
ax2.set(ylim=(-.1,1.2),yticks=[0,1],ylabel='Contact gate');ax.legend(fontsize=8)
ax=axs[2,0]
p=plans[1];tp=np.arange(len(p['a']))/30
ax.plot(tp,p['a'][:,0],lw=1,label='First predicted plan: raw X')
ax.plot(tp,p['smooth_xyz'][:,0],lw=2,label='Same plan: legacy 35-point MA (offline)')
ax.set(ylabel='Requested X (mm)',title='Removed smoothing: mathematical comparison only');ax.legend(fontsize=8)
ax=axs[2,1]
switches=evidence['runs']['C']['plan_switches']
ax.bar([f"{x['old_plan']} → {x['new_plan']}" for x in switches],[x['requested_XYZ_jump_mm'] for x in switches],color='darkorange')
for i,x in enumerate(switches):ax.text(i,x['requested_XYZ_jump_mm']+.6,f"{x['requested_XYZ_jump_mm']:.1f} mm",ha='center')
ax.set(ylim=(0,38),ylabel='Requested XYZ discontinuity (mm)',title='Plan replacement jumps (not physical jumps)')
for ax in axs.flat:
    if ax!=axs[2,1]:ax.set_xlabel('Seconds after executor start / plan time')
for ax in [axs[0,0],axs[0,1]]:
    for item in switches:ax.axvline(item['at_s'],color='gray',lw=.7,ls='--',alpha=.5)
fig.savefig(OUT/'diagnostic.png',dpi=160)
fig.savefig(OUT/'diagnostic.pdf')
plt.close(fig)
print(json.dumps({'C':evidence['runs']['C'],'C_plan_adjacency':evidence['C_plan_adjacency'],'peak':evidence['C_peak_measured_base_fz'],'plot':str(OUT/'diagnostic.png')},indent=2))
