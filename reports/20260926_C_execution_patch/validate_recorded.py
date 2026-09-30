"""Offline command replay only. Recorded contact states are inputs, not a plant model."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('/home/eunseop/nrs_imitation')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'behavior_ws/src/nrs_imitation'))
from nrs_imitation import e2_timed_execution as patched

old_path = OUT/'before/behavior_ws/src/nrs_imitation/nrs_imitation/e2_timed_execution.py'
spec = importlib.util.spec_from_file_location('prepatch_execution', old_path)
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
cfg_path = ROOT/'experiments/e2_rule_replay_20260920/config.json'
cfg = json.loads(cfg_path.read_text())
baseline = json.loads((OUT/'before/experiments/e2_rule_replay_20260920/config.json').read_text())
assert cfg['executor'] == baseline['executor']
assert cfg['common'] == baseline['common']
assert cfg['recipe'] == baseline['recipe']
assert cfg['replay'] == baseline['replay']
assert {k:v for k,v in cfg['il'].items() if k != 'pose_conditioning'} == baseline['il']
assert not patched.validate_settings(cfg)
KEYS = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'fx', 'fy', 'fz']
INPUTS = {}


def record(path):
    INPUTS[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def rows(path):
    with record(path).open() as f:
        return list(csv.DictReader(f))


def load(executor):
    prefix = executor.name.split('_executor')[0]
    provider = next(p for p in executor.parent.glob(prefix+'_*') if '_executor' not in p.name)
    meta = json.loads(record(executor/'metadata.json').read_text())
    assert meta['config']['executor'] == baseline['executor']
    assert meta['config']['common'] == baseline['common']
    events = [json.loads(s) for s in record(executor/'events.jsonl').read_text().splitlines()]
    start = next(e for e in events if e['event'] == 'execution_start')['details']['monotonic_start']
    receives = [e for e in events if e['event'] == 'plan_received']
    command = [r for r in rows(executor/'commands.csv') if r['command_stage'] == 'time_sampled']
    for r in command:
        r['detail'] = json.loads(r['details'])
    command = [r for r in command if r['detail'].get('controller_mode', 'Force') == 'Force']
    poses = rows(executor/'tcp_pose.csv')
    pose_times = np.array([int(r['receipt_monotonic_ns'])/1e9 for r in poses])
    pose_values = np.array([[float(r[k]) for k in KEYS[:6]] for r in poses])
    initial = pose_values[max(0, np.searchsorted(pose_times, start, side='right')-1)]
    method = {'R':'rule', 'T':'replay', 'C':'il'}[prefix[4]]
    plan_rows = rows(provider/'commands.csv')
    plans = {}
    for event in receives:
        d = event['details']; plan_id = d['plan_id']
        snapshot = provider/'prepared_plans'/('%06d_postprocessed.npz' % plan_id)
        if snapshot.exists():
            with np.load(record(snapshot), allow_pickle=False) as data:
                a, t, phase = data['action'].copy(), data['time'].copy(), data['phase'].copy()
        else:
            rr = sorted((r for r in plan_rows if r['command_stage'] == 'postprocessed' and
                         int(r['plan_id']) == plan_id), key=lambda r:int(r['action_index']))
            a = np.array([[float(r[k]) for k in KEYS] for r in rr])
            if method == 'replay':
                with np.load(record(Path(meta['config']['replay']['template'])), allow_pickle=False) as data:
                    t, phase = data['time'].copy(), data['phase'].copy()
            else:
                t = np.arange(len(a))/meta['config']['il']['trajectory_hz']
                phase = np.full(len(a), 'unknown')
        assert len(a) == len(t)
        plans[plan_id] = dict(plan_id=plan_id, generated_at=d['source_plan_time'], time=t,
                             action=a, phase=phase, final=method != 'il')
    return dict(prefix=prefix, method=method, start=start, initial=initial, plans=plans,
                receives=receives, rows=command, pose_times=pose_times, poses=pose_values)


def replay(data, module):
    settings = (module.executor_settings(cfg, data['method']) if module is patched
                else module.executor_settings(baseline))
    engine = module.TimedExecution(settings)
    events = iter(data['receives']); next_event = next(events, None)
    first = data['receives'][0]
    engine.accept(module.TimedPlan(**data['plans'][first['details']['plan_id']]),
                  first['receipt_monotonic_ns']/1e9)
    next_event = next(events, None)
    engine.start(data['start'], data['initial'])
    output = []; ticks = []; bridges = []; prepare_ms = []; tick_ms = []
    for row in data['rows']:
        plan_id = int(row['plan_id']); detail = row['detail']
        origin = data['start'] if plan_id == 1 else data['plans'][plan_id]['generated_at']
        now = origin+detail['elapsed_s']
        while next_event is not None and next_event['receipt_monotonic_ns']/1e9 <= now:
            at = next_event['receipt_monotonic_ns']/1e9
            previous = engine.conditioner.sample(at) if module is patched and engine.conditioner else None
            p = module.TimedPlan(**data['plans'][next_event['details']['plan_id']])
            started = time.perf_counter()
            engine.accept(p, at)
            prepare_ms.append(1000.*(time.perf_counter()-started))
            if previous is not None:
                after = engine.conditioner.sample(at)
                bridges.append(dict(at_s=at-data['start'],
                                    position_discontinuity_mm=float(np.linalg.norm(after[:3]-previous[:3])),
                                    rotation_discontinuity_rad=float((Rotation.from_rotvec(after[3:])*
                                                                     Rotation.from_rotvec(previous[3:]).inv()).magnitude())))
            next_event = next(events, None)
        index = max(0, np.searchsorted(data['pose_times'], now, side='right')-1)
        # Exact recorded gate decisions isolate the command generator. This
        # does not predict the new robot/force response or remove gate chatter.
        fz = cfg['executor']['contact_on_N']+1. if detail['contact'] else cfg['executor']['contact_off_N']-1.
        started = time.perf_counter()
        result = engine.tick(now, data['poses'][index], [0., 0., fz], 0., 0.)
        tick_ms.append(1000.*(time.perf_counter()-started))
        assert 'end' not in result, (data['prefix'], result)
        output.append(result); ticks.append(now-data['start'])
    return dict(result=output, time=np.array(ticks), bridges=bridges,
                prepare_ms=prepare_ms, tick_ms=tick_ms)


def stats(replay):
    sent = np.array([r['sent'] for r in replay['result']])
    dt = np.array([min(r['dt_s'], .008) for r in replay['result']])
    velocity = np.diff(sent[:, :3], axis=0)/dt[1:, None]
    first = velocity[(replay['time'][1:] > .05) & (replay['time'][1:] < 4.)]
    flips = ((first[1:]*first[:-1] < 0.) & (np.abs(first[1:]) > 1.) & (np.abs(first[:-1]) > 1.)).sum(axis=0)
    # Count full excursions through +/-1 mm/s as well. Adjacent-sample abrupt
    # flips alone necessarily disappear under the new acceleration bound.
    excursions = []
    for axis in range(3):
        signs = np.sign(first[np.abs(first[:, axis]) > 1., axis])
        excursions.append(int(np.count_nonzero(np.diff(signs))))
    acceleration = np.diff(velocity, axis=0)/dt[2:, None]
    return dict(first4s_abrupt_direction_reversals_XYZ_over_1mm_s=flips.tolist(),
                first4s_full_direction_reversals_with_1mm_s_deadband=excursions,
                max_abs_axis_speed_mm_s=np.max(np.abs(velocity), axis=0).tolist(),
                max_abs_axis_acceleration_mm_s2=np.max(np.abs(acceleration), axis=0).tolist(),
                max_plan_preparation_ms=max(replay['prepare_ms'], default=0.),
                tick_ms_p50_p95_p99_max=np.percentile(replay['tick_ms'], [50, 95, 99, 100]).tolist())


report = dict(scope='C-only offline pose command validation; R/T recordings retained',
              common_executor_settings_unchanged=True, common_config_unchanged=True,
              C_profile=cfg['il']['pose_conditioning'], R_T_regressions=[],
              robot_executed=False, physical_vibration_resolved=None,
              contact_replay='recorded per-tick gate states, not simulated/measured post-patch feedback',
              force_samples_gate_thresholds_and_force_rate_unchanged=True)
log = ROOT/'logs/inference_metrics'
for ep in sorted(log.glob('RTC_*_20260926T*_executor*')):
    prefix = ep.name.split('_executor')[0]
    if prefix[4] not in ('R', 'T') and prefix != 'RTC_C_20260926T175131':
        continue
    data = load(ep)
    before = replay(data, old); after = replay(data, patched)
    for left, right in zip(before['result'], after['result']):
        np.testing.assert_array_equal(left['requested'], right['requested'])
        for stage in ['gated', 'sent']:
            np.testing.assert_array_equal(left[stage][6:], right[stage][6:])
        assert left['contact'] == right['contact']
    if data['method'] != 'il':
        for left, right in zip(before['result'], after['result']):
            for stage in ['requested', 'gated', 'sent']:
                np.testing.assert_array_equal(left[stage], right[stage])
                assert left[stage].tobytes() == right[stage].tobytes()
        report['R_T_regressions'].append(dict(run=prefix, compared_ticks=len(after['result']),
            all_9d_commands_bitwise_equal=True, source_timestamps_preserved=True))
        print(prefix, 'unchanged', len(after['result']), flush=True)
    else:
        report['C'] = dict(run=prefix, before=stats(before), after=stats(after),
                           handovers=after['bridges'], compared_ticks=len(after['result']))
        assert all(b['position_discontinuity_mm'] < 1e-9 and b['rotation_discontinuity_rad'] < 1e-9
                   for b in after['bridges'])
        assert max(report['C']['after']['max_abs_axis_acceleration_mm_s2']) <= 25.0001
        assert max(report['C']['after']['max_abs_axis_speed_mm_s']) <= 10.0001
        assert report['C']['after']['first4s_abrupt_direction_reversals_XYZ_over_1mm_s'] == [0, 0, 0]
        assert all(a < b for a,b in zip(report['C']['after']['first4s_full_direction_reversals_with_1mm_s_deadband'][:2],
                                        report['C']['before']['first4s_full_direction_reversals_with_1mm_s_deadband'][:2]))
        C = (data, before, after)
        print(json.dumps(report['C'], indent=2), flush=True)
assert len(report['R_T_regressions']) == 10
for path, expected in INPUTS.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
report['source_sha256'] = INPUTS
(OUT/'recorded_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')

data, before, after = C
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .2})
fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
fig.suptitle('C pose patch — offline replay of 2026-09-26 17:51:31\nRecorded contact states held fixed; no hardware / force-response simulation')
for r, label, color in [(before, 'Before', 'firebrick'), (after, 'Patched C', 'steelblue')]:
    sent = np.array([x['sent'] for x in r['result']])
    dt = np.array([min(x['dt_s'], .008) for x in r['result']])
    v = np.diff(sent[:, :3], axis=0)/dt[1:, None]
    axes[0, 0].plot(r['time'][1:], v[:, 0], label=label, color=color, lw=1.)
    axes[0, 1].plot(r['time'][1:], v[:, 1], label=label, color=color, lw=1.)
    axes[1, 0].plot(r['time'], sent[:, 2], label=label, color=color, lw=1.2)
    axes[1, 1].plot(sent[:, 0], sent[:, 1], label=label, color=color, lw=1.2)
axes[0, 0].set(xlim=(0, 4), xlabel='Time (s)', ylabel='X command velocity (mm/s)')
axes[0, 1].set(xlim=(0, 4), xlabel='Time (s)', ylabel='Y command velocity (mm/s)')
axes[1, 0].set(xlabel='Time (s)', ylabel='Z command (mm)')
axes[1, 1].set(xlabel='X command (mm)', ylabel='Y command (mm)')
for ax in axes.flat: ax.legend()
fig.savefig(OUT/'command_comparison.png', dpi=170)
fig.savefig(OUT/'command_comparison.pdf')
print('Offline validation passed. Original logs unchanged; robot not run.', flush=True)
