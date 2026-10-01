"""Read-only reuse of archived report parsers and analysis functions.
Source: reports/20260927_E1_pdf_report/build_report.py and reports/20260930_E2_pdf_report/build_report.py.
AST-extracted definitions only; original module-level output mutations are not executed.
"""

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from types import SimpleNamespace
import csv, hashlib, json
import numpy as np
ROOT = Path("/home/eunseop/nrs_imitation")

def js(path):
    return json.loads(path.read_text())

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def events(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def vals(rr, keys):
    return np.array([[float(r[k]) for k in keys] for r in rr])

def times(rr):
    return np.array([int(r['receipt_monotonic_ns']) for r in rr], dtype=np.int64) / 1e9

def weighted_force(t, f, start, end):
    assert t[0] <= start < end <= t[-1]
    edges = np.r_[start, t[(t > start) & (t < end)], end]
    weights = np.diff(edges)
    obs = f[np.searchsorted(t, edges[:-1], side='right') - 1]
    mean = float(np.average(obs, weights=weights))
    observed = f[(t >= start) & (t <= end)]
    return dict(mean_fz_N=mean,
                temporal_sd_fz_N=float(np.sqrt(np.average((obs - mean) ** 2, weights=weights))),
                peak_fz_N=float(observed.max()), min_fz_N=float(observed.min()),
                time_fz_ge_3N_s=float(weights[obs >= 3].sum()),
                time_fz_ge_50N_s=float(weights[obs >= 50].sum()))

def analyze_e1():
    archive = js(DATA / 'manifest.json')
    integrity = []
    for item in archive['files']:
        p = DATA / item['path']
        ok = p.is_file() and sha(p) == item['sha256'] and p.stat().st_size == item['bytes']
        integrity.append({'path': item['path'], 'ok': ok})
    assert all(x['ok'] for x in integrity), [x for x in integrity if not x['ok']]
    index = rows(DATA / 'run_index.csv')
    assert len(index) == len(archive['runs']) == 15
    runs = []
    for rec in archive['runs']:
        base = DATA / rec['run_directory']
        ep, pp = base / 'executor', base / 'provider'
        ev, pev = events(ep / 'events.jsonl'), events(pp / 'events.jsonl')
        start_event = next(e for e in ev if e['event'] == 'execution_start')
        stops = [e for e in ev if e['event'] == 'stop_requested' and not e['details'].get('initial_reset')]
        # Same event receipt basis as run_index.csv; no source acquisition time exists.
        start, end = start_event['receipt_monotonic_ns'] / 1e9, stops[0]['receipt_monotonic_ns'] / 1e9
        assert abs(end - start - rec['execution_to_first_stop_s']) < 1e-6
        wr, pr, cr = rows(ep / 'wrench.csv'), rows(ep / 'tcp_pose.csv'), rows(ep / 'commands.csv')
        ft, pt = times(wr), times(pr)
        force, pose = vals(wr, ['fx', 'fy', 'fz']), vals(pr, ['x', 'y', 'z', 'rx', 'ry', 'rz'])
        assert all(r['semantic_frame'] == 'robot_base' for r in wr + pr)
        assert len(wr) == rec['force_rows'] and len(pr) == rec['tcp_rows']
        assert np.all(np.diff(ft) > 0) and np.all(np.diff(pt) > 0)
        assert np.isfinite(force).all() and np.isfinite(pose).all()
        assert pt[0] <= start < end <= pt[-1]
        window_t = np.r_[start, pt[(pt > start) & (pt < end)], end]
        window_pose = np.stack([np.interp(window_t, pt, pose[:, j]) for j in range(6)], axis=1)
        full_sent = [r for r in cr if r['command_stage'] == 'node_sent']
        assert len(full_sent) == rec['node_sent_rows']
        sent = [r for r in full_sent if start <= int(r['receipt_monotonic_ns']) / 1e9 <= end]
        ct = times(sent)
        action = vals(sent, ['x', 'y', 'z', 'rx', 'ry', 'rz', 'fx', 'fy', 'fz'])
        assert np.isfinite(action).all()
        details = [json.loads(r['details']) for r in sent]
        gate = np.array([d['contact'] for d in details], dtype=bool)
        gate_edges = np.r_[ct, end]
        gate_time = float(np.diff(gate_edges)[gate].sum())
        first_plan = next(e for e in ev if e['event'] == 'plan_received')
        reference = np.array(first_plan['details']['reference_xy_mm'])
        terminal = [e for e in ev if e['event'] in ['safety_stop', 'trajectory_end', 'normal_completion', 'manual_abort']][-1]
        assert terminal['event'] == rec['logged_terminal_event']
        config = js(base / 'launch_context/config.json')
        assert sha(base / 'launch_context/config.json') == rec['config_sha256']
        for sub, rrmap in [(ep, {'wrench': wr, 'tcp_pose': pr, 'commands': cr}), (pp, {})]:
            summary = js(sub / 'summary.json')
            assert summary['drained'] and summary['queue_pending'] == 0
            assert summary['write_error_count'] == 0
            assert sum(v['dropped'] for v in summary['counts'].values()) == 0
            for key, rr in rrmap.items():
                assert len(rr) == summary['counts'][key]['written']
        raw = {}
        for line in (base / 'plots/summary.txt').read_text().splitlines():
            k, v = line.split(':', 1)
            raw[k.strip()] = float(v.strip())
        beginnings, latencies = {}, []
        for e in pev:
            if e['event'] == 'inference_start':
                beginnings[e['details']['inference_id']] = e['receipt_monotonic_ns']
            elif e['event'] == 'inference_end':
                ident = e['details']['inference_id']
                if ident in beginnings:
                    latencies.append((e['receipt_monotonic_ns'] - beginnings[ident]) / 1e6)
        metric = dict(label=f"{rec['method']}{rec['repeat_index']}", method=rec['method'],
                      repeat_index=rec['repeat_index'], archive_run_id=rec['archive_run_id'],
                      execution_start_local=rec['execution_start_local'],
                      execution_duration_s=end-start, **weighted_force(ft, force[:, 2], start, end),
                      tcp_xy_path_mm=float(np.linalg.norm(np.diff(window_pose[:, :2], axis=0), axis=1).sum()),
                      tcp_xyz_path_mm=float(np.linalg.norm(np.diff(window_pose[:, :3], axis=0), axis=1).sum()),
                      command_rows_execution=len(sent), max_command_gap_ms=float(np.diff(ct).max()*1e3),
                      median_command_gap_ms=float(np.median(np.diff(ct))*1e3),
                      force_feedback_hz=float((len(ft)-1)/(ft[-1]-ft[0])),
                      max_force_feedback_gap_ms=float(np.diff(ft).max()*1e3),
                      max_force_gap_execution_ms=float(np.diff(ft)[(ft[:-1] < end) & (ft[1:] > start)].max()*1e3),
                      gate_on_s=gate_time, gate_transitions=int(np.count_nonzero(np.diff(gate))),
                      inference_count=len(latencies), first_pipeline_ms=latencies[0] if latencies else None,
                      warm_pipeline_median_ms=float(np.median(latencies[1:])) if len(latencies)>1 else None,
                      terminal_event=terminal['event'], terminal_after_first_stop_s=terminal['receipt_monotonic_ns']/1e9-end,
                      first_stop_reason=stops[0]['details']['reason'],
                      later_stop_details=[e['details'].get('details') for e in stops[1:]],
                      target_distance_at_first_stop_mm=float(np.linalg.norm(window_pose[-1,:3]-np.array(first_plan['details']['last_requested_pose'][:3]))) if rec['method'] in 'RT' else None,
                      raw_heatmap_mean=raw['mean_heatmap_removal'], raw_heatmap_sd=raw['std_heatmap_removal'],
                      recorder_duration_s=raw['duration_s'], raw_max_speed_mm_s=raw['max_speed_mm_s'],
                      raw_max_fn_N=raw['max_fn_N'], tcp_rows=len(pr), force_rows=len(wr),
                      node_sent_rows_total=len(full_sent), run_directory=rec['run_directory'])
        runs.append(dict(metric=metric, record=rec, base=base, config=config, start=start, end=end,
                         ft=ft-start, force=force, pt=window_t-start, pose=window_pose,
                         xy=window_pose[:,:2]-reference, ct=ct-start, action=action, gate=gate,
                         raw=raw, events=ev, latencies=latencies))
    assert len({json.dumps(r['config']['executor'], sort_keys=True) for r in runs}) == 1
    assert len({json.dumps(r['config']['rtc_pose_conditioning'], sort_keys=True) for r in runs}) == 1
    metrics = [r['metric'] for r in runs]
    groups = {m:[r for r in runs if r['metric']['method']==m] for m in 'RTC'}
    summaries = {}
    keys = ['execution_duration_s', 'mean_fz_N', 'peak_fz_N', 'temporal_sd_fz_N',
            'time_fz_ge_3N_s', 'time_fz_ge_50N_s', 'tcp_xy_path_mm', 'tcp_xyz_path_mm',
            'gate_on_s', 'gate_transitions', 'max_command_gap_ms', 'force_feedback_hz',
            'max_force_feedback_gap_ms', 'raw_heatmap_mean', 'recorder_duration_s', 'terminal_after_first_stop_s']
    for m, rr in groups.items():
        summaries[m] = {}
        for k in keys:
            a = np.array([r['metric'][k] for r in rr])
            summaries[m][k] = dict(mean=float(a.mean()), sample_sd=float(a.std(ddof=1)), min=float(a.min()), max=float(a.max()))
    result = dict(created_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(), source=str(DATA),
                  manifest_sha256=sha(DATA/'manifest.json'), verified_file_count=len(integrity),
                  analysis_window='execution_start receipt to first non-startup stop_requested receipt; includes approach; excludes subsequent R return',
                  force='filtered republished robot-base Fz; time-weighted zero-order hold; not calibrated normal force or command tracking error',
                  path='measured TCP polyline in mm with linear interpolation at window endpoints',
                  sd='sample SD across five run-level values, ddof=1',
                  runs=metrics, method_summary=summaries)
    (OUT/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    (OUT/'integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
    return archive, runs, groups, summaries, result

def analyze_e2():
    archive = js(DATA / 'manifest.json')
    integrity = []
    for item in archive['files']:
        p = DATA / item['path']
        integrity.append(dict(path=item['path'], ok=p.is_file() and
                              p.stat().st_size == item['bytes'] and sha(p) == item['sha256']))
    assert all(x['ok'] for x in integrity), [x for x in integrity if not x['ok']]
    index = rows(DATA / 'run_index.csv')
    assert len(index) == len(archive['runs']) == 15
    cohort = js(DATA / 'comparison.json')
    runs = []
    for rec in archive['runs']:
        base = DATA / rec['run_directory']
        ep, pp = base / 'executor', base / 'provider'
        config, emeta, pmeta = js(base / 'runtime.json'), js(ep / 'metadata.json'), js(pp / 'metadata.json')
        assert sha(base / 'runtime.json') == rec['runtime_sha256']
        assert config['condition'] == rec['condition']
        assert config['run']['repeat_index'] == rec['repeat_index']
        assert config['comparison']['common_settings_sha256'] == cohort['common_settings_sha256']
        pinned = cohort['model_artifacts'][rec['condition']]
        assert all(config['il'][k] == pinned[k] for k in ('checkpoint', 'checkpoint_sha256', 'normalizer_sha256'))
        assert pmeta['checkpoint']['path'] == config['il']['checkpoint']
        assert pmeta['force_observation'] == ('ON' if rec['condition'] == 'C' else 'OFF')
        ev, pev = events(ep / 'events.jsonl'), events(pp / 'events.jsonl')
        start_event = next(e for e in ev if e['event'] == 'execution_start')
        stops = [e for e in ev if e['event'] == 'stop_requested' and not e['details'].get('initial_reset')]
        start, end = start_event['receipt_monotonic_ns']/1e9, stops[0]['receipt_monotonic_ns']/1e9
        assert abs(end - start - rec['execution_to_first_stop_s']) < 1e-6
        wr, pr, cr = rows(ep / 'wrench.csv'), rows(ep / 'tcp_pose.csv'), rows(ep / 'commands.csv')
        ft, pt = times(wr), times(pr)
        wrench = vals(wr, ['fx', 'fy', 'fz', 'tx', 'ty', 'tz'])
        force, pose = wrench[:, :3], vals(pr, ['x', 'y', 'z', 'rx', 'ry', 'rz'])
        assert all(r['semantic_frame'] == 'robot_base' for r in wr + pr)
        assert all(r['force_unit'] == 'N' for r in wr)
        assert all(r['position_unit'] == 'mm' for r in pr)
        assert len(wr) == rec['executor_wrench_rows'] and len(pr) == rec['executor_tcp_pose_rows']
        assert np.all(np.diff(ft) > 0) and np.all(np.diff(pt) > 0)
        assert np.isfinite(wrench).all() and np.isfinite(pose).all()
        assert ft[0] <= start < end <= ft[-1] and pt[0] <= start < end <= pt[-1]
        window_t = np.r_[start, pt[(pt > start) & (pt < end)], end]
        window_pose = np.stack([np.interp(window_t, pt, pose[:, j]) for j in range(6)], axis=1)
        full_sent = [r for r in cr if r['command_stage'] == 'node_sent']
        sent = [r for r in full_sent if start <= int(r['receipt_monotonic_ns'])/1e9 <= end]
        ct, action = times(sent), vals(sent, ['x', 'y', 'z', 'rx', 'ry', 'rz', 'fx', 'fy', 'fz'])
        assert len(sent) > 1 and np.isfinite(action).all() and np.all(np.diff(ct) > 0)
        details = [json.loads(r['details']) for r in sent]
        gate = np.array([d['contact'] for d in details], dtype=bool)
        gate_time = float(np.diff(np.r_[ct, end])[gate].sum())
        first_plan = next(e for e in ev if e['event'] == 'plan_received')
        reference = np.array(first_plan['details']['reference_xy_mm'])
        terminal = next(e for e in reversed(ev) if e['event'] in
                        ('normal_completion', 'safety_stop', 'trajectory_end', 'manual_abort'))
        assert terminal['event'] == rec['terminal_event'] == 'normal_completion'
        assert terminal['details']['controller_hold_verified'] is True
        assert terminal['details']['queue_cancel_verified'] is True
        for log in (ep, pp):
            summary = js(log / 'summary.json')
            assert summary['drained'] and summary['queue_pending'] == 0
            assert summary['write_error_count'] == 0
            assert all(v['dropped'] == 0 for v in summary['counts'].values())
        raw = {}
        for line in (base / 'plots/summary.txt').read_text().splitlines():
            k, v = line.split(':', 1)
            raw[k.strip()] = float(v.strip())
        beginnings, latencies = {}, []
        for e in pev:
            if e['event'] == 'inference_start':
                beginnings[e['details']['inference_id']] = e['receipt_monotonic_ns']
            elif e['event'] == 'inference_end':
                ident = e['details']['inference_id']
                if ident in beginnings:
                    latencies.append((e['receipt_monotonic_ns'] - beginnings[ident])/1e6)
        assert len(latencies) >= 2
        fm = style.weighted_force(ft, force[:, 2], start, end)
        time_mask = (ft >= start) & (ft <= end)
        metric = dict(label=rec['label'], condition=rec['condition'], repeat_index=rec['repeat_index'],
            archive_run_id=rec['archive_run_id'], execution_start_local=rec['execution_start_local'],
            execution_duration_s=end-start, **fm,
            tcp_xy_path_mm=float(np.linalg.norm(np.diff(window_pose[:, :2], axis=0), axis=1).sum()),
            tcp_xyz_path_mm=float(np.linalg.norm(np.diff(window_pose[:, :3], axis=0), axis=1).sum()),
            command_rows_execution=len(sent), node_sent_rows_total=len(full_sent),
            max_command_gap_ms=float(np.diff(ct).max()*1e3), median_command_gap_ms=float(np.median(np.diff(ct))*1e3),
            force_feedback_hz=float((len(ft)-1)/(ft[-1]-ft[0])),
            max_force_feedback_gap_ms=float(np.diff(ft).max()*1e3),
            max_force_gap_execution_ms=float(np.diff(ft)[(ft[:-1] < end) & (ft[1:] > start)].max()*1e3),
            gate_on_s=gate_time, gate_transitions=int(np.count_nonzero(np.diff(gate))),
            gate_initial_unobserved_s=float(ct[0]-start),
            command_mean_fz_N=float(np.average(action[:, 8], weights=np.diff(np.r_[ct, end]))),
            command_peak_fz_N=float(action[:, 8].max()),
            measured_force_abs_peak_N=wrench[time_mask, :3].__abs__().max(axis=0).tolist(),
            measured_torque_abs_peak_Nm=wrench[time_mask, 3:].__abs__().max(axis=0).tolist(),
            inference_count=len(latencies), first_pipeline_ms=latencies[0],
            warm_pipeline_median_ms=float(np.median(latencies[1:])),
            terminal_event=terminal['event'], first_stop_reason=stops[0]['details']['reason'],
            terminal_after_first_stop_s=terminal['receipt_monotonic_ns']/1e9-end,
            controller_hold_verified=terminal['details']['controller_hold_verified'],
            queue_cancel_verified=terminal['details']['queue_cancel_verified'],
            raw_heatmap_mean=raw['mean_heatmap_removal'], raw_heatmap_sd=raw['std_heatmap_removal'],
            recorder_duration_s=raw['duration_s'], raw_max_speed_mm_s=raw['max_speed_mm_s'],
            raw_max_fn_N=raw['max_fn_N'], tcp_rows=len(pr), force_rows=len(wr),
            video_count=rec['video_count'], run_directory=rec['run_directory'])
        runs.append(dict(metric=metric, record=rec, base=base, config=config, start=start, end=end,
            ft=ft-start, force=force, pt=window_t-start, pose=window_pose, xy=window_pose[:, :2]-reference,
            ct=ct-start, action=action, gate=gate, raw=raw, events=ev, latencies=latencies,
            provider_metadata=pmeta, executor_metadata=emeta))
    assert len({json.dumps(r['config']['executor'], sort_keys=True) for r in runs}) == 1
    assert len({json.dumps(r['config']['il']['pose_conditioning'], sort_keys=True) for r in runs}) == 1
    assert all(r['raw'][key] == expected for r in runs for key, expected in
               [('k_preston', 1.0), ('cell_mm', 1.0), ('pad_radius_mm', 20.0),
                ('contact_threshold_N', .5), ('speed_threshold_mm_s', .1)])
    groups = {c: [r for r in runs if r['metric']['condition'] == c] for c in 'ABC'}
    assert all(len(g) == 5 for g in groups.values())
    keys = ['execution_duration_s', 'mean_fz_N', 'peak_fz_N', 'temporal_sd_fz_N',
        'time_fz_ge_3N_s', 'time_fz_ge_50N_s', 'tcp_xy_path_mm', 'tcp_xyz_path_mm', 'gate_on_s',
        'gate_transitions', 'max_command_gap_ms', 'force_feedback_hz', 'max_force_feedback_gap_ms',
        'raw_heatmap_mean', 'recorder_duration_s', 'terminal_after_first_stop_s', 'warm_pipeline_median_ms']
    summaries = {}
    for c, rr in groups.items():
        summaries[c] = {}
        for key in keys:
            a = np.array([r['metric'][key] for r in rr])
            summaries[c][key] = dict(mean=float(a.mean()), sample_sd=float(a.std(ddof=1)),
                                     min=float(a.min()), max=float(a.max()))
    metrics = [r['metric'] for r in runs]
    result = dict(created_at=datetime.now(TZ).isoformat(), source=str(DATA),
        manifest_sha256=sha(DATA / 'manifest.json'), verified_file_count=len(integrity),
        template_pdf=str(TEMPLATE_PDF), template_pdf_sha256=sha(TEMPLATE_PDF),
        analysis_window='execution_start receipt through first non-startup stop_requested receipt; same rule for A/B/C',
        force='filtered republished robot-base Fz; time-weighted zero-order hold, with full interval coverage; not calibrated surface-normal force',
        path='measured TCP polyline in mm; linear interpolation only at window boundaries',
        sd='sample SD across five run-level measurements, ddof=1',
        numbering='Original launch repeat_index, rather than chronological renumbering',
        selection=js(DATA / 'selection.json'), runs=metrics, condition_summary=summaries)
    (OUT / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (OUT / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    (OUT / 'integrity.json').write_text(json.dumps(integrity, indent=2)+'\n')
    return archive, runs, groups, summaries, result

style = SimpleNamespace(**{k:globals()[k] for k in ['js', 'sha', 'rows', 'events', 'vals', 'times', 'weighted_force']})
TZ=ZoneInfo("Asia/Seoul")
TEMPLATE_PDF=ROOT/"results/20260927/E1_report_20260927.pdf"

