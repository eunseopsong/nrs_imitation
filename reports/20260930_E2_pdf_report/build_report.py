"""Reproduce the E1 PDF layout using the archived 2026-09-30 E2 A/B/C logs."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import importlib.util
import json
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/nrs_e2_20260930_mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image

ROOT = Path('/home/eunseop/nrs_imitation')
DATA = ROOT / 'results/20260930/E2'
OUT = Path(__file__).resolve().parent
TARGET = ROOT / 'results/20260930/E2_report_20260930.pdf'
TEMPLATE_PDF = ROOT / 'results/20260927/E1_report_20260927.pdf'
TEMPLATE_CODE = ROOT / 'reports/20260927_E1_pdf_report/build_report.py'
TZ = ZoneInfo('Asia/Seoul')
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'previews').mkdir(exist_ok=True)

# Use the original report's fonts, table styling, margins and overflow checks.
spec = importlib.util.spec_from_file_location('e1_report_style', TEMPLATE_CODE)
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)
style.OUT = OUT
style.PAGE_TITLES.clear()
style.TEXT_CHECKS.clear()
table, body, heading, finish, image_on, pm = (
    style.table, style.body, style.heading, style.finish, style.image_on, style.pm)
NAVY, INK, GRAY = style.NAVY, style.INK, style.GRAY
COLORS = {'A': '#17836D', 'B': '#326AB8', 'C': '#CD7031'}
NAMES = {'A': '힘 관측 OFF · 고정 힘 23 N',
         'B': '힘 관측 OFF · 학습된 힘 출력',
         'C': '힘 관측 ON · 학습된 힘 출력'}


def js(path):
    return json.loads(path.read_text())


def sha(path):
    return style.sha(path)


def rows(path):
    return style.rows(path)


def events(path):
    return style.events(path)


def vals(rr, keys):
    return style.vals(rr, keys)


def times(rr):
    return style.times(rr)


def analyze():
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


def page(title, sub=''):
    style.PAGE_TITLES.append(title)
    fig = plt.figure(figsize=(8.27, 11.69), facecolor='white', dpi=110)
    fig.text(.065, .955, 'E2  /  EXPERIMENT REPORT', color=GRAY, size=9, weight='bold')
    fig.text(.94, .955, '2026.09.30', color=GRAY, size=9, ha='right')
    fig.text(.065, .908, title, color=NAVY, size=21, weight='bold')
    if sub:
        fig.text(.065, .879, sub, color=GRAY, size=8.7)
    fig.add_artist(plt.Line2D([.065, .94], [.861, .861], transform=fig.transFigure, color='#CBD8E1', lw=.9))
    fig.text(.065, .026, '2026-09-30 E2  ·  A/B/C 각 5회  ·  보관 로그 기반 분석', size=7.2, color=GRAY)
    fig.text(.94, .026, f'{len(style.PAGE_TITLES):02d} / 15', ha='right', size=8.5, color=GRAY)
    return fig


def dotplot(ax, groups, key, title, unit):
    for j, c in enumerate('ABC'):
        values = np.array([r['metric'][key] for r in groups[c]])
        ax.scatter(j+np.linspace(-.14, .14, len(values)), values, s=31, color=COLORS[c],
                   alpha=.78, zorder=3, edgecolor='white', lw=.4)
        ax.errorbar(j+.26, values.mean(), yerr=values.std(ddof=1), fmt='D', ms=4,
                    color=NAVY, capsize=3, lw=1, zorder=4)
    ax.set_xticks(range(3), ['A', 'B', 'C']); ax.set_xlim(-.5, 2.6)
    ax.set_ylabel(unit); ax.set_title(title, loc='left', weight='bold', pad=9)
    ax.set_ylim(bottom=0)


def render(archive, runs, groups, summary, result):
    temporary = TARGET.with_name(TARGET.stem + '.building.pdf')
    first = min(r['metric']['execution_start_local'] for r in runs)[11:16]
    last = max(r['metric']['execution_start_local'] for r in runs)[11:16]
    count = result['verified_file_count']
    duration_max = max(r['metric']['execution_duration_s'] for r in runs)
    x_max = float(np.ceil(duration_max / 5) * 5)
    observed_force = np.concatenate([r['force'][(r['ft'] >= 0) &
        (r['ft'] <= r['metric']['execution_duration_s']), 2] for r in runs])
    f_min = float(np.floor((observed_force.min() - 5) / 10) * 10)
    f_max = float(np.ceil((observed_force.max() + 5) / 10) * 10)
    with PdfPages(temporary) as pdf:
        pdf.infodict().update(Title='E2 실험 로그 분석 보고서 | 2026-09-30',
            Author='NRS experiment log analysis',
            Subject='A/B/C 각 5회 완료 기록의 힘·경로·종료 상태와 초기/최종 사진 및 히트맵',
            Keywords='E2 A B C 20260930 Korean experiment report')

        # 01 — Same executive-summary structure as the supplied E1 report.
        fig = page('E2 실험 로그 분석 보고서',
            'A: 고정 힘 23 N  ·  B: 힘 관측 OFF + 학습 힘 출력  ·  C: 힘 관측 ON + 학습 힘 출력')
        y = heading(fig, .826, '15회 완료 기록을 같은 시간 기준으로 비교')
        body(fig, y, f'2026년 9월 30일 {first}–{last}(KST)에 시작한 A·B·C 각 5회를 분석했다. '
            '실행 시간은 execution_start부터 첫 정지 요청까지다. A4는 작업자가 요청한 재시도를 사용했으며, '
            '이전 중단 기록은 별도로 보관했다.', size=10, line=.025)
        table(fig, ['분석 실행', '보관 파일 검증', '로거 드롭 / 쓰기 오류'],
            [['15회 (각 5회)', f'{count} / {count} SHA-256 일치', '0 / 0']],
            .637, .076, [.29, .38, .33], 10)
        heading(fig, .602, '주요 결과  |  실행별 지표의 평균 ± 표본 SD')
        table(fig, ['조건', '실행 시간\n(s)', '평균 base Fz\n(N)', '관측 peak Fz\n(N)', 'XY 이동 거리\n(mm)'],
            [[c, pm(summary, c, 'execution_duration_s', 3), pm(summary, c, 'mean_fz_N'),
              pm(summary, c, 'peak_fz_N'), pm(summary, c, 'tcp_xy_path_mm')] for c in 'ABC'],
            .408, .151, [.075, .245, .22, .23, .23], 8.3)
        y = heading(fig, .372, '읽어야 할 핵심')
        y = body(fig, y, '실행 구간 평균 base Fz는 '
            + ', '.join(f"{c} {summary[c]['mean_fz_N']['mean']:.2f} N" for c in 'ABC')
            + '이다. 실행별 관측 peak의 평균은 '
            + ', '.join(f"{c} {summary[c]['peak_fz_N']['mean']:.2f} N" for c in 'ABC')
            + '이었다.', size=10, line=.024)
        y = body(fig, y, '선택된 15회 모두 operator_finish 후 normal_completion으로 종료됐고, '
            '명령 큐 취소 및 controller_hold_verified=true가 확인됐다. A4의 이전 시도는 '
            '실행 시작 전에 manual_abort됐으며 분석 집계에서 제외했다.', size=9.8, line=.024)
        body(fig, y, '이 결과는 보관된 완료 기록의 기술 통계다. 실측 제거량·고정 평가 ROI·확인된 RPM이 '
            '없어 표면 품질이나 가공 생산성의 순위를 산출하지 않았다. 검증 파일 수에는 별도 보관한 '
            '중단 시도와 보관용 목록도 포함한다.', size=9.5, line=.023)
        finish(pdf, fig)

        # 02 — Conditions and methods, with the actual E2 ablation differences.
        fig = page('실험 조건과 분석 기준',
            '조건 출처: 각 실행의 runtime.json 및 executor/provider 메타데이터')
        table(fig, ['항목', '보관 설정 및 조건 차이'], [
            ['A 정책 / 힘 목표', '힘 관측·history·학습 힘 출력 OFF, pose6 정책 + 외부 고정 23 N'],
            ['B 정책 / 힘 목표', '힘 관측·history 0 마스킹, 학습된 힘을 포함한 정책 출력'],
            ['C 정책 / 힘 목표', '측정 힘·history 사용, 학습된 힘을 포함한 정책 출력'],
            ['정책 및 체크포인트', 'Flow / validation best / 추론 10 steps / 128점 / 120 steps 재계획'],
            ['공통 명령 및 궤적', '125 Hz (8 ms), gain 15/s / 정책 궤적 30 Hz'],
            ['공통 위치·자세 처리', '35점 평활 / 0.5 s 연결 / 위치·자세 normalizer 범위 동일'],
            ['공통 위치 제한', '속도 10 mm/s, 가속도 25 mm/s²'],
            ['공통 회전 제한', '속도 40 deg/s, 가속도 100 deg/s²'],
            ['공통 접촉 / 힘 변화율', 'base Fz ON 3 N / OFF 1.2 N, 힘 변화율 30 N/s, A 전용 ramp 0 s'],
            ['공통 보호 / 종료 상한', '각 축 200 N·200 Nm / receipt watchdog 0.2 s / 실행 상한 50 s'],
        ], .482, .344, [.27, .73], 8.1)
        y = heading(fig, .449, '숫자를 계산한 방법')
        y = body(fig, y, '실행 구간: execution_start의 receipt_monotonic_ns부터 초기 reset을 제외한 '
            '첫 stop_requested까지. 세 조건에 같은 규칙을 적용했으며 접근·이탈이 포함될 수 있다. '
            '자동 processing 이벤트는 실제 접촉·가공을 확인한 정답 시각이 아니다.', size=9.2, line=.022)
        y = body(fig, y, '힘: executor/wrench.csv의 robot_base Fz를 직전 샘플 유지(ZOH)로 시간 가중했다. '
            '평균·시간 표준편차·임계값 이상 시간을 계산하고, peak는 구간 안의 실제 기록 최댓값을 사용했다. '
            '힘 기록은 약 18 Hz이며 센서 취득 시각은 없다.', size=9.2, line=.022)
        y = body(fig, y, '경로: executor/tcp_pose.csv를 구간 경계에서 선형 보간하고 연속 XY 위치의 '
            '거리 합을 계산했다. 접촉 게이트 ON 시간은 발행 명령의 contact 상태를 누적했다. '
            '표의 ±는 조건별 5개 실행 지표의 표본 SD(ddof=1)다.', size=9.2, line=.022)
        body(fig, y, 'A의 외부 23 N 및 B/C의 예측 힘은 제어기 목표다. 측정 base Fz와 동일한 좌표·법선력으로 '
            '검증되지 않았고 controller_applied_force도 기록되지 않아 힘 추종 RMSE는 계산하지 않았다. '
            'RPM·시편 초기 상태·공구 상태의 일치는 저장 설정만으로 확인할 수 없다.', size=9, line=.021, color=GRAY)
        finish(pdf, fig)

        # 03 — Preserve original repeat labels, including the late retries.
        fig = page('15회 개별 실행 결과',
            '시작 시각은 2026-09-30 KST  ·  실행 구간은 시작 이벤트부터 첫 정지 요청까지')
        table(fig, ['실행', '시작 시각', '시간\n(s)', '평균 Fz\n(N)', 'peak Fz\n(N)', 'XY 거리\n(mm)', '게이트 ON\n(s)'],
            [[r['metric']['label'], r['metric']['execution_start_local'][11:19],
              f"{r['metric']['execution_duration_s']:.3f}", f"{r['metric']['mean_fz_N']:.2f}",
              f"{r['metric']['peak_fz_N']:.2f}", f"{r['metric']['tcp_xy_path_mm']:.2f}",
              f"{r['metric']['gate_on_s']:.2f}"] for r in runs],
            .319, .509, [.08, .15, .16, .15, .15, .16, .15], 9)
        y = heading(fig, .285, '표 해석')
        y = body(fig, y, '모든 실행의 종료 기준은 작업자 finish다. 기록된 시간·거리 차이는 정책 경로와 '
            '종료 시점에 함께 영향을 받으며, 동일 제거량에 대한 작업 주기나 효율 차이를 의미하지 않는다.',
            size=9.8, line=.024)
        y = body(fig, y, 'A1–A5, B1–B5, C1–C5는 원래 repeat_index를 유지했다. C1·C2는 추가 실행으로 '
            'B5 뒤에 기록됐고, A4 재시도는 A5 뒤에 기록됐다. 시각자료 부록에도 같은 번호를 사용한다.',
            size=9.8, line=.024)
        body(fig, y, '게이트 ON은 base Fz 히스테리시스로 추정한 상태다. 물리적 접촉 시간 또는 '
            '유효 가공 시간을 독립적으로 확인한 값으로 취급하지 않았다.', size=9.5, line=.023, color=GRAY)
        finish(pdf, fig)

        # 04 — Individual values plus mean and sample SD.
        fig = page('조건별 지표와 반복 편차',
            '색 점: 개별 실행  ·  남색 마름모와 오차막대: 5회 평균 ± 표본 SD')
        chart_defs = [('execution_duration_s', '실행 시간', 's'),
            ('tcp_xy_path_mm', '실측 XY 이동 거리', 'mm'), ('mean_fz_N', '실행 구간 평균 Fz', 'N'),
            ('peak_fz_N', '실행 구간 관측 peak Fz', 'N'), ('gate_on_s', '접촉 게이트 ON 시간', 's'),
            ('time_fz_ge_50N_s', 'Fz ≥ 50 N 누적 시간', 's')]
        for i, (key, label, unit) in enumerate(chart_defs):
            row, col = divmod(i, 2)
            ax = fig.add_axes([.115 + .46 * col, .646 - .257 * row, .335, .155])
            dotplot(ax, groups, key, label, unit)
        body(fig, .073, '50 N은 힘 분포를 요약하는 분석 기준이다. 실행 보호 한계는 모든 조건에서 각 축 200 N이었다.',
            size=8.5, line=.02, color=GRAY, units=115)
        finish(pdf, fig)

        # 05 — Common axes for all force traces.
        fig = page('실측 힘의 시간 변화',
            'executor/wrench.csv  ·  시작 정렬  ·  세 패널의 시간축과 힘 범위를 동일하게 표시')
        for j, c in enumerate('ABC'):
            ax = fig.add_axes([.115, .665 - .238 * j, .81, .145])
            for i, r in enumerate(groups[c]):
                mask = (r['ft'] >= 0) & (r['ft'] <= r['metric']['execution_duration_s'])
                ax.plot(r['ft'][mask], r['force'][mask, 2], color=COLORS[c], alpha=.38 + .14 * i,
                        lw=.8, label=r['metric']['label'])
            ax.set(xlim=(0, x_max), ylim=(f_min, f_max), ylabel='base Fz (N)', xlabel='실행 시작 후 시간 (s)')
            ax.set_title(f'{c}  |  {NAMES[c]}', loc='left', weight='bold', pad=8)
            ax.legend(loc='upper right', ncol=5, frameon=False, handlelength=1)
        y = heading(fig, .130, '관측된 차이')
        range_text = ', '.join(f"{c} {summary[c]['peak_fz_N']['min']:.2f}–{summary[c]['peak_fz_N']['max']:.2f} N"
                               for c in 'ABC')
        body(fig, y, '실행별 관측 peak 범위는 ' + range_text + '이었다. A의 고정 목표 23 N은 측정 '
            'base Fz의 제한값이 아니다. 약 18 Hz 기록이므로 샘플 사이의 더 빠른 힘 변동이나 연속 신호의 '
            '최댓값은 이 그래프에서 확인할 수 없다.', size=9.1, line=.021)
        finish(pdf, fig)

        # 06 — Measured path in a translated stain-relative frame and base Z.
        fig = page('실측 TCP 경로와 높이',
            '왼쪽: 얼룩 기준점의 XY 평행이동을 제거  ·  오른쪽: robot_base Z  ·  첫 정지 요청까지')
        all_xy, all_z = np.concatenate([r['xy'] for r in runs]), np.concatenate([r['pose'][:, 2] for r in runs])
        xy_lo, xy_hi = np.floor((all_xy.min(axis=0) - 5) / 10) * 10, np.ceil((all_xy.max(axis=0) + 5) / 10) * 10
        z_lo, z_hi = np.floor((all_z.min() - 3) / 10) * 10, np.ceil((all_z.max() + 3) / 10) * 10
        for j, c in enumerate('ABC'):
            y = .623 - .260 * j
            ax, az = fig.add_axes([.11, y, .315, .202]), fig.add_axes([.58, y, .345, .202])
            for i, r in enumerate(groups[c]):
                alpha = .38 + .14 * i
                ax.plot(r['xy'][:, 0], r['xy'][:, 1], color=COLORS[c], alpha=alpha, lw=.9)
                ax.scatter(*r['xy'][-1], color=COLORS[c], alpha=alpha, marker='s', s=10, zorder=3)
                az.plot(r['pt'], r['pose'][:, 2], color=COLORS[c], alpha=alpha, lw=.9, label=r['metric']['label'])
            ax.set(xlim=(xy_lo[0], xy_hi[0]), ylim=(xy_lo[1], xy_hi[1]), xlabel='상대 X (mm)', ylabel='상대 Y (mm)')
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(f'{c}  |  XY 경로', loc='left', weight='bold', pad=8)
            az.set(xlim=(0, x_max), ylim=(z_lo, z_hi), xlabel='실행 시작 후 시간 (s)', ylabel='Z (mm)')
            az.set_title(f'{c}  |  TCP 높이', loc='left', weight='bold', pad=8)
            az.legend(loc='lower right', ncol=3, frameon=False, handlelength=1)
        body(fig, .055, 'XY 사각형은 분석 종료 위치다. 회전·시편 변형 보정은 적용하지 않았으며 접근·이탈이 포함될 수 있다.',
            size=8.1, units=122, line=.018, color=GRAY)
        finish(pdf, fig)

        # 07 — Logged stop state and file quality, with missing video provenance.
        fig = page('기록 품질과 종료 상태',
            '로그 파일 완결성, 정지 확인, 물리 작업 완료를 각각 구분')
        table(fig, ['조건', '힘 기록률\n평균 (Hz)', '발행 명령 최대 간격\n범위 (ms)', '실행 중 힘 샘플\n최대 간격 범위 (ms)'],
            [[c, f"{summary[c]['force_feedback_hz']['mean']:.2f}",
              f"{min(r['metric']['max_command_gap_ms'] for r in groups[c]):.2f}–{max(r['metric']['max_command_gap_ms'] for r in groups[c]):.2f}",
              f"{min(r['metric']['max_force_gap_execution_ms'] for r in groups[c]):.2f}–{max(r['metric']['max_force_gap_execution_ms'] for r in groups[c]):.2f}"] for c in 'ABC'],
            .674, .151, [.1, .22, .34, .34], 8.4)
        body(fig, .648, '선택된 30개 logger(provider·executor)의 큐는 모두 비워졌고 내부 드롭·쓰기 오류는 0이다. '
            'CSV 행 수도 summary 집계와 일치했다. middleware_loss_count는 null이며, 125 Hz 명령 발행과 '
            '약 18 Hz 힘 기록률은 서로 다르다.', size=9.3, line=.023)
        table(fig, ['조건', '첫 정지 요청', '최종 이벤트', '정지 유지 확인'],
            [[f'{c} (5회)', 'operator_finish', 'normal_completion', '5 / 5'] for c in 'ABC'],
            .430, .133, [.13, .29, .34, .24], 8.5)
        y = body(fig, .406, '15회 모두 controller_hold_verified와 queue_cancel_verified가 true였다. '
            '이는 작업자 finish 뒤 정지 유지 및 명령 큐 취소의 기록이다. final_target_reached·retract_complete·'
            'home_pose_reached는 false, physical_contact_release는 unknown이었다.', size=9.4, line=.023)
        y = body(fig, y, '첫 회를 제외한 추론 이벤트 구간의 실행별 중앙값 평균은 '
            + ', '.join(f"{c} {summary[c]['warm_pipeline_median_ms']['mean']:.2f} ms" for c in 'ABC')
            + '다. inference_start→end의 전체 경과시간이며 GPU 연산만의 시간은 아니다.', size=9.4, line=.023)
        y = body(fig, y, '녹화 원본은 B·C 각 5개를 보관했고 ffprobe로 프레임·길이를 확인했다. '
            'A는 원본 실행 폴더와 Screencasts에서 해당 영상 파일을 찾지 못했다. '
            '초기·최종 사진은 모든 조건에 5쌍씩 있으며 이 PDF에 전부 포함했다.', size=9.3, line=.022)
        body(fig, y, '정지 확인과 로그 완결성은 표면 품질 또는 물리 작업 성공률의 정답 판정과 구분된다. '
            '실측 제거량·RPM 자료가 없어 그 판정은 보고서 범위에 포함하지 않았다.', size=9.1, line=.022, color=GRAY)
        finish(pdf, fig)

        # 08 — Original proxy values, preserving recorder windows and units.
        fig = page('기존 제거량 추정 지표',
            'plots/summary.txt의 값을 그대로 집계  ·  임의단위(a.u.)  ·  실측 제거 깊이/질량 자료 없음')
        ax = fig.add_axes([.14, .589, .76, .210])
        dotplot(ax, groups, 'raw_heatmap_mean', '원본 히트맵 평균값', 'a.u.')
        ax.set_ylim(0, np.ceil(max(r['metric']['raw_heatmap_mean'] for r in runs) * 12) / 10)
        table(fig, ['조건', '히트맵 평균값\n평균 ± SD (a.u.)', 'recorder 기록 길이\n평균 ± SD (s)', '실행 비교 구간\n평균 (s)'],
            [[c, pm(summary, c, 'raw_heatmap_mean', 4), pm(summary, c, 'recorder_duration_s', 2),
              f"{summary[c]['execution_duration_s']['mean']:.3f}"] for c in 'ABC'],
            .383, .142, [.1, .33, .33, .24], 8.6)
        y = heading(fig, .350, '히트맵을 해석하는 범위')
        y = body(fig, y, '모든 원본 summary에는 K=1.0, grid 1 mm, pad 반경 20 mm, 힘 기준 0.5 N, '
            '속도 기준 0.1 mm/s가 기록돼 있다. 재료 제거 깊이·질량으로 보정된 값은 아니다.', size=9.5, line=.023)
        y = body(fig, y, '히트맵 평균의 5회 평균은 '
            + ', '.join(f"{c} {summary[c]['raw_heatmap_mean']['mean']:.4f}" for c in 'ABC')
            + ' a.u.다. recorder 구간에는 실행 비교 구간 밖의 데이터도 포함되고, 경로·방문 셀 영역도 '
            '다르므로 이 차이를 제거율 또는 표면 품질의 순위로 해석하지 않는다.', size=9.5, line=.023)
        y = body(fig, y, '원본 max_speed_mm_s와 max_fn_N도 recorder 전체 구간의 값이다. '
            '제어기의 명령 제한 또는 이번 실행 비교 구간의 실측 최고값으로 대체하지 않았다.', size=9.3, line=.022)
        body(fig, y, '12–14쪽에 15개 원본 히트맵을 모두 실었다. 각 그림의 색·좌표 범위는 원본을 유지한다. '
            '그림 안 std는 공간적 편차이며 이 표의 반복 간 SD와 다르다.', size=9.2, line=.022, color=GRAY)
        finish(pdf, fig)

        # 09–11 — All before/after images, exactly five rows per condition.
        for c in 'ABC':
            fig = page(f'{c} | 기록 초기·최종 사진',
                f'{NAMES[c]}  ·  provider/snapshots의 저장 이미지  ·  순서와 전체 화면 유지')
            fig.text(.375, .828, '초기 저장 이미지', ha='center', size=9.5, color=NAVY, weight='bold')
            fig.text(.755, .828, '최종 저장 이미지', ha='center', size=9.5, color=NAVY, weight='bold')
            for i, r in enumerate(groups[c]):
                y, m = .677 - .145 * i, r['metric']
                fig.text(.075, y + .11, m['label'], size=12, weight='bold', color=COLORS[c])
                fig.text(.075, y + .085, m['execution_start_local'][11:19], size=7.3, color=GRAY)
                fig.text(.075, y + .05, f"{m['execution_duration_s']:.2f} s", size=7.6, color=INK)
                fig.text(.075, y + .026, f"Fz {m['mean_fz_N']:.2f} N", size=7.1, color=INK)
                image_on(fig, [.205, y, .34, .136], r['base'] / 'provider/snapshots/initial_image.png')
                image_on(fig, [.585, y, .34, .136], r['base'] / 'provider/snapshots/final_image.png')
            body(fig, .067, '사진 저장 시점은 분석 구간 경계와 다를 수 있다. 공구 가림·조명·시점 차이가 있어 정량 제거율은 산출하지 않았다.',
                size=8.1, units=122, line=.018, color=GRAY)
            finish(pdf, fig)

        # 12–14 — Five original heatmaps with their original colorbars.
        for c in 'ABC':
            fig = page(f'{c} | 원본 제거량 추정 히트맵',
                f'{NAMES[c]}  ·  각 그림의 색 범위와 XY 좌표는 원본 그대로  ·  단위: a.u.')
            for i, r in enumerate(groups[c]):
                row, col = divmod(i, 2)
                x, y = .060 + .45 * col, .595 - .254 * row
                fig.text(x + .02, y + .227,
                    f"{r['metric']['label']}  |  {r['metric']['execution_start_local'][11:19]}",
                    size=9, color=COLORS[c], weight='bold')
                image_on(fig, [x, y, .435, .223], r['base'] / 'plots/01_removal_heatmap.png')
            fig.text(.545, .299, f'{c} 5회 요약', size=12, color=NAVY, weight='bold', va='top')
            body(fig, .264, f"히트맵 평균의 5회 평균\n{summary[c]['raw_heatmap_mean']['mean']:.4f} a.u.\n"
                f"반복 간 표본 SD\n{summary[c]['raw_heatmap_mean']['sample_sd']:.4f} a.u.\n\n"
                '그림 안 std는 각 히트맵의\n공간적 편차이며 반복 간 SD와 다르다.\n색상은 그림별 colorbar를 참조한다.',
                size=8.6, units=43, line=.023, x=.545, color=INK)
            body(fig, .056, '출처: 각 실행의 plots/01_removal_heatmap.png. 가공량을 실측한 결과가 아닌 원본 추정 시각자료다.',
                size=8.2, units=122, line=.018, color=GRAY)
            finish(pdf, fig)

        # 15 — Traceability, including exclusions and already-deleted C records.
        fig = page('자료 범위와 재현 정보',
            '단일 PDF에 요약·정량 분석·15회 사진·15개 히트맵을 포함')
        y = heading(fig, .826, '분석 대상과 선택 이력')
        y = body(fig, y, '분석 대상은 results/20260930/E2/{A,B,C}의 각 5회다. '
            'A4의 재시도 결과를 포함했고 이전 중단 시도는 archive/excluded_attempts/A에 보관했다. '
            'C의 초기 r1·r2는 작업자 요청으로 삭제된 뒤 추가 실행됐으며, 삭제 이력을 selection.json에 남겼다.',
            size=9.7, line=.024)
        y = body(fig, y, '원본 세션 폴더는 유지하고 자료를 복사했다. 각 실행의 logger·plots 하위 폴더만 '
            '한 단계 정리했으며 원본 파일 내용은 보존했다. 이 선별 묶음은 전체 시도의 성공률 또는 '
            '무작위 반복 실험의 통계적 우월성을 평가하는 표본으로 사용하지 않았다.', size=9.7, line=.024)
        table(fig, ['자료 (각 실행 폴더 기준)', '보고서에서 사용한 내용'], [
            ['manifest.json / run_index.csv (E2 루트)', '표본 목록, 선택 이력, 상태, SHA-256'],
            ['executor/events.jsonl', '시작·첫 정지 시각, 종료 이벤트, 계획 수신'],
            ['executor/wrench.csv / tcp_pose.csv', '측정 힘, 시간 가중 통계, 실측 이동 경로'],
            ['executor/commands.csv', 'node_sent 발행 간격, 접촉 게이트 상태'],
            ['provider/events.jsonl / */summary.json', '추론 이벤트 경과시간, 로거 집계와 완결성'],
            ['runtime.json / */metadata.json', '정책·제어 조건, 모델 해시, 기록 방식'],
            ['plots/summary.txt / 01_removal_heatmap.png', '원본 추정 지표와 히트맵'],
            ['provider/snapshots/*.png', '15회 초기·최종 이미지 30장'],
        ], .365, .261, [.53, .47], 8.2)
        y = heading(fig, .331, '검증 및 재생성')
        y = body(fig, y, f'목록 내 {count}개 파일의 크기·SHA-256을 확인했다. 15회 CSV 행 수, '
            '이벤트 종료 상태, 설정 해시, 수치 유효성과 구간 경계도 대조했다. '
            '영상은 B·C 합계 10개이며 A는 원본 파일이 없다. 영상 자체는 PDF에 삽입하지 않았다.',
            size=9.4, line=.023)
        body(fig, y, '계산 지표와 생성 코드: reports/20260930_E2_pdf_report\n'
            '재생성: python3 reports/20260930_E2_pdf_report/build_report.py', size=8.9, line=.022, units=107)
        fig.text(.07, .132, '원본 manifest SHA-256', size=8.4, color=NAVY, weight='bold')
        digest = result['manifest_sha256']
        fig.text(.07, .110, digest[:32], size=8.4, color=GRAY)
        fig.text(.07, .091, digest[32:], size=8.4, color=GRAY)
        fig.text(.07, .059, '작성: ' + result['created_at'][:19].replace('T', ' ') + ' KST  |  측정 단위: mm, s, N',
            size=8, color=GRAY)
        finish(pdf, fig)
    assert len(style.PAGE_TITLES) == 15
    temporary.replace(TARGET)
    validation = dict(output=str(TARGET), pages=len(style.PAGE_TITLES), titles=style.PAGE_TITLES,
        sha256=sha(TARGET), bytes=TARGET.stat().st_size, manifest_sha256=result['manifest_sha256'],
        template_pdf_sha256=result['template_pdf_sha256'], template_code_sha256=sha(TEMPLATE_CODE),
        verified_archive_files=count, layout_checks=style.TEXT_CHECKS)
    (OUT / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2)+'\n')
    (DATA / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (DATA / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['runs'][0]))
        writer.writeheader(); writer.writerows(result['runs'])
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    data = analyze()
    if '--analysis-only' in sys.argv:
        print(json.dumps(dict(verified_files=data[-1]['verified_file_count'],
            summaries=data[-2], runs=[{k: r['metric'][k] for k in
              ('label', 'execution_duration_s', 'mean_fz_N', 'peak_fz_N', 'tcp_xy_path_mm',
               'gate_on_s', 'max_command_gap_ms', 'warm_pipeline_median_ms', 'raw_heatmap_mean')}
              for r in data[1]]), ensure_ascii=False, indent=2))
    else:
        render(*data)
