"""Reproducible Korean PDF report for the 2026-09-27 E1 archive."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
import sys
import unicodedata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import fontManager
import numpy as np
from PIL import Image

ROOT = Path('/home/eunseop/nrs_imitation')
DATA = ROOT / 'results/20260927/E1'
OUT = Path(__file__).resolve().parent
TARGET = ROOT / 'results/20260927/E1_report_20260927.pdf'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'previews').mkdir(exist_ok=True)


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


def analyze():
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


if __name__ == '__main__' and '--analysis-only' in sys.argv:
    a, runs, groups, summaries, result = analyze()
    print(json.dumps({'verified_files':result['verified_file_count'],'summaries':summaries,
                      'runs':[{k:r['metric'][k] for k in ['label','execution_duration_s','mean_fz_N','peak_fz_N','tcp_xy_path_mm','gate_on_s','gate_transitions','max_command_gap_ms','target_distance_at_first_stop_mm','warm_pipeline_median_ms','later_stop_details']} for r in runs]},ensure_ascii=False,indent=2))
    sys.exit(0)


REG = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
BOLD = '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
fontManager.addfont(REG)
fontManager.addfont(BOLD)
plt.rcParams.update({'font.family':'NanumGothic', 'pdf.fonttype':42, 'ps.fonttype':42,
    'axes.unicode_minus':False, 'font.size':9, 'axes.titlesize':10, 'axes.labelsize':8.5,
    'xtick.labelsize':8, 'ytick.labelsize':8, 'legend.fontsize':7.2,
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.grid':True,
    'grid.alpha':.2, 'grid.linewidth':.5, 'savefig.facecolor':'white'})
NAVY = '#15334D'
INK = '#293B48'
GRAY = '#637583'
PALE = '#EEF3F7'
COLORS = {'R':'#17836D', 'T':'#326AB8', 'C':'#CD7031'}
NAMES = {'R':'규칙 경로', 'T':'교시 재생', 'C':'힘 관측 학습 정책'}
PAGE_TITLES = []
TEXT_CHECKS = []


def wrap_cjk(s, units=99):
    lines = []
    for para in s.split('\n'):
        line, width = '', 0
        for word in para.split():
            word_width = sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in word)
            if line and width + 1 + word_width > units:
                lines.append(line)
                line, width = '', 0
            if line:
                line += ' '
                width += 1
            line += word
            width += word_width
        lines.append(line)
    return lines


def page(title, sub=''):
    PAGE_TITLES.append(title)
    fig = plt.figure(figsize=(8.27, 11.69), facecolor='white', dpi=110)
    fig.text(.065, .955, 'E1  /  EXPERIMENT REPORT', color=GRAY, size=9, weight='bold')
    fig.text(.94, .955, '2026.09.27', color=GRAY, size=9, ha='right')
    fig.text(.065, .908, title, color=NAVY, size=21, weight='bold')
    if sub:
        fig.text(.065, .879, sub, color=GRAY, size=8.7)
    fig.add_artist(plt.Line2D([.065,.94], [.861,.861], transform=fig.transFigure, color='#CBD8E1', lw=.9))
    fig.text(.065,.026,'2026-09-27 E1  ·  R/T/C 각 5회  ·  보관 로그 기반 분석',size=7.2,color=GRAY)
    fig.text(.94,.026,f'{len(PAGE_TITLES):02d} / 15',ha='right',size=8.5,color=GRAY)
    return fig


def body(fig, y, text, size=10, units=98, line=.023, x=.07, color=INK):
    for s in wrap_cjk(text, units):
        assert y > .048, ('Text overflow', len(PAGE_TITLES), s)
        fig.text(x,y,s,size=size,color=color,va='top')
        y -= line
    return y - .01


def heading(fig, y, text):
    fig.text(.07,y,text,size=12,color=NAVY,weight='bold',va='top')
    return y - .031


def table(fig, headers, data, bottom, height, widths=None, size=8.5):
    ax = fig.add_axes([.07,bottom,.87,height]); ax.axis('off')
    t = ax.table(cellText=data,colLabels=headers,colWidths=widths,cellLoc='center',bbox=[0,0,1,1])
    t.auto_set_font_size(False); t.set_fontsize(size)
    for (r,c), cell in t.get_celld().items():
        cell.set_edgecolor('#D4DFE6'); cell.set_linewidth(.45); cell.PAD=.035
        if r == 0:
            cell.set_facecolor(NAVY); cell.get_text().set_color('white'); cell.get_text().set_weight('bold')
        else:
            cell.set_facecolor(PALE if r%2 else 'white')
    return t


def finish(pdf, fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w,h = fig.canvas.get_width_height()
    for t in fig.texts:
        b=t.get_window_extent(renderer)
        assert b.x0>=0 and b.x1<=w+1 and b.y0>=0 and b.y1<=h+1, (len(PAGE_TITLES),t.get_text())
    for ax in fig.axes:
        for tb in ax.tables:
            for cell in tb.get_celld().values():
                cb=cell.get_window_extent(renderer); b=cell.get_text().get_window_extent(renderer)
                assert b.x0 >= cb.x0-1 and b.x1 <= cb.x1+1 and b.y0 >= cb.y0-1 and b.y1 <= cb.y1+1, (len(PAGE_TITLES),cell.get_text().get_text(),tuple(b.bounds),tuple(cb.bounds))
    TEXT_CHECKS.append({'page':len(PAGE_TITLES),'figure_text_and_table_cells_fit':True})
    pdf.savefig(fig)
    fig.savefig(OUT/'previews'/f'page_{len(PAGE_TITLES):02d}.png',dpi=110)
    plt.close(fig)


def pm(summary, method, key, digits=2):
    s=summary[method][key]
    if 0 < s['sample_sd'] < .001 and digits >= 3:
        return f"{s['mean']:.{digits}f} ± <0.001"
    return f"{s['mean']:.{digits}f} ± {s['sample_sd']:.{digits}f}"


def dotplot(ax, groups, key, label, unit, annotate=False):
    for j,m in enumerate('RTC'):
        y=np.array([r['metric'][key] for r in groups[m]])
        ax.scatter(j+np.linspace(-.14,.14,len(y)),y,s=31,color=COLORS[m],alpha=.78,zorder=3,edgecolor='white',lw=.4)
        ax.errorbar(j+.26,y.mean(),yerr=y.std(ddof=1),fmt='D',ms=4,color=NAVY,capsize=3,lw=1,zorder=4)
        if annotate:
            ax.text(j,y.max()+.04*(max(y.max(),.01)),f'{y.mean():.3f}',ha='center',size=8,color=COLORS[m])
    ax.set_xticks(range(3),['R','T','C']); ax.set_xlim(-.5,2.6)
    ax.set_ylabel(unit); ax.set_title(label,loc='left',weight='bold',pad=9)
    ax.set_ylim(bottom=0)


def image_on(fig, rect, path):
    ax=fig.add_axes(rect); ax.imshow(Image.open(path)); ax.axis('off')
    return ax


def render(archive, runs, groups, summary, result):
    temporary=TARGET.with_name(TARGET.stem+'.building.pdf')
    with PdfPages(temporary) as pdf:
        pdf.infodict().update(Title='E1 실험 로그 분석 보고서 | 2026-09-27',
                             Author='NRS experiment log analysis',
                             Subject='R/T/C 15회 실행의 힘·경로·종료 상태와 원본 시각자료',
                             Keywords='E1 R T C 20260927 Korean experiment report')
        # 01 — Readable executive summary.
        fig=page('E1 실험 로그 분석 보고서','R: 규칙 경로  ·  T: 교시 재생  ·  C: 힘 관측을 사용하는 학습 정책')
        y=heading(fig,.826,'15회 실행을 같은 시간 기준으로 비교')
        body(fig,y,'2026년 9월 27일 23:08–23:28(KST)에 시작한 R·T·C 각 5회를 분석했다. 실행 시간은 시작 이벤트부터 첫 정지 요청까지이며, 접근 동작을 포함한다. R의 이후 복귀 동작은 이 비교 구간에서 제외했다.',size=10,line=.025)
        table(fig,['보관 실행','검증 파일','로거 누락 / 쓰기 오류'],
              [['15회 (각 5회)','866 / 866 SHA-256 일치','0 / 0']],.637,.076,[.31,.36,.33],10)
        y=heading(fig,.602,'주요 결과  |  실행별 지표의 평균 ± 표본 SD')
        table(fig,['방법','실행 시간\n(s)','평균 base Fz\n(N)','관측 peak Fz\n(N)','XY 이동 거리\n(mm)'],
              [[m,pm(summary,m,'execution_duration_s',3),pm(summary,m,'mean_fz_N'),pm(summary,m,'peak_fz_N'),pm(summary,m,'tcp_xy_path_mm')] for m in 'RTC'],
              .408,.151,[.075,.245,.22,.23,.23],8.3)
        y=heading(fig,.372,'읽어야 할 핵심')
        y=body(fig,y,'R의 실행 구간 평균 Fz는 11.96 N으로 세 방법 중 낮았다. T와 C의 평균은 각각 15.61 N, 15.17 N이며, 실행별 peak의 평균은 T 117.84 N, C 91.99 N이었다.',size=10,line=.024)
        y=body(fig,y,'C 5회는 작업자 finish 후 normal_completion, T 5회는 trajectory_end로 종료됐다. R 5회는 첫 경로 종료 뒤 복귀 과정에서 safety_stop이 기록됐다. 모든 실행에서 controller_hold_verified=true가 확인됐다.',size=9.8,line=.024)
        body(fig,y,'이는 기록된 동작과 종료 상태의 요약이다. 종료 기준·가공 구간이 다르고 실측 제거량이 없어, 방법별 품질·성공률·가공 생산성의 순위를 확정하지 않는다.',size=9.8,line=.024)
        finish(pdf,fig)

        # 02 — Conditions and definitions.
        fig=page('실험 조건과 분석 기준','조건 출처: 각 실행의 launch_context/config.json 및 executor/provider 메타데이터')
        table(fig,['항목','보관 설정 및 방법 차이'],[
            ['R 경로','규칙 기반 편도 1회, recipe Fz +18 N, feed 10 mm/s'],
            ['T 경로','episode_29의 시간·힘을 재생, 시간 축 scaling 1.0'],
            ['C 정책','Flow 정책 / 힘 관측 ON / 추론 10 steps / 힘 이력 30개'],
            ['C 재계획','30 Hz, 128점 구간, 120 steps마다 다음 구간'],
            ['공통 명령 발행','125 Hz (8 ms), gain 15/s'],
            ['공통 위치·자세 처리','35점 평활 / 0.5 s 연결 / 30 Hz 위치·자세 격자'],
            ['공통 변화율 제한','축별 위치 10 mm/s, 회전 40 deg/s, 힘 30 N/s'],
            ['공통 가속도 제한','위치 25 mm/s², 회전 100 deg/s²'],
            ['공통 접촉 게이트','측정 base Fz 기준 ON 3 N / OFF 1.2 N'],
            ['힘·위치 피드백 기록','최대 20 Hz 설정; 수신 시각 기반, 센서 취득 시각 없음'],
        ],.482,.344,[.27,.73],8.4)
        y=heading(fig,.449,'숫자를 계산한 방법')
        y=body(fig,y,'실행 구간: execution_start의 receipt_monotonic_ns부터 초기 reset을 제외한 첫 stop_requested까지. 모든 방법에 같은 규칙을 적용했으며 순수 가공 구간을 의미하지 않는다.',size=9.4,line=.022)
        y=body(fig,y,'힘: executor/wrench.csv의 robot_base Fz를 사용했다. 평균·시간 표준편차는 직전 샘플 유지(ZOH) 시간 가중치로 계산하고, peak는 구간 안에 실제 기록된 값의 최댓값이다.',size=9.4,line=.022)
        y=body(fig,y,'경로: executor/tcp_pose.csv의 실측 TCP 위치를 구간 경계에서 선형 보간한 뒤, 연속 XY 점 사이 거리를 합산했다. 접촉 게이트 ON 시간은 발행 명령의 contact 상태를 누적했다.',size=9.4,line=.022)
        y=body(fig,y,'표의 ±는 각 방법 5개 실행별 지표의 표본 표준편차(ddof=1)다. 한 실행 안의 힘 변동과 구분한다. 통계 검정이나 표본 밖 일반화는 수행하지 않았다.',size=9.4,line=.022)
        body(fig,y,'base Fz는 보정된 표면 법선력과 동일하다고 검증되지 않았다. 명령 힘의 좌표 의미와 controller_applied_force도 확정되지 않아 힘 추종 RMSE를 계산하지 않았다.',size=9.1,line=.021,color=GRAY)
        finish(pdf,fig)

        # 03 — Every selected run.
        fig=page('15회 개별 실행 결과','시작 시각은 2026-09-27 KST  ·  모든 수치는 첫 정지 요청까지의 구간 기준')
        table(fig,['실행','시작 시각','시간\n(s)','평균 Fz\n(N)','peak Fz\n(N)','XY 거리\n(mm)','게이트 ON\n(s)'],
              [[r['metric']['label'],r['metric']['execution_start_local'][11:19],f"{r['metric']['execution_duration_s']:.3f}",f"{r['metric']['mean_fz_N']:.2f}",f"{r['metric']['peak_fz_N']:.2f}",f"{r['metric']['tcp_xy_path_mm']:.2f}",f"{r['metric']['gate_on_s']:.2f}"] for r in runs],
              .319,.509,[.08,.15,.16,.15,.15,.16,.15],9)
        y=heading(fig,.285,'표 해석')
        y=body(fig,y,'R과 T의 시간은 원본 경로 길이에 의해 거의 고정된다. C는 작업자가 종료를 요청하므로 시간의 실행 간 편차가 있다. 세 방법의 시간 차이를 완료 주기나 가공 효율 차이로 직접 환산하지 않는다.',size=9.8,line=.024)
        y=body(fig,y,'게이트 ON은 기존 접촉 추정 상태다. 자동 processing 이벤트도 같은 게이트에서 생성되므로 별도의 가공 정답 구간으로 취급하지 않았다.',size=9.8,line=.024)
        body(fig,y,'R1–R5, T1–T5, C1–C5는 각 방법의 보관 실행을 시간순으로 붙인 번호다. 사진과 히트맵 부록에서도 같은 번호를 사용한다.',size=9.5,line=.023,color=GRAY)
        finish(pdf,fig)

        # 04 — Distributions, no misleading bar-only summaries.
        fig=page('방법별 지표와 반복 편차','색 점: 개별 실행  ·  남색 마름모와 오차막대: 5회 평균 ± 표본 SD')
        chart_defs=[('execution_duration_s','실행 시간','s'),('tcp_xy_path_mm','실측 XY 이동 거리','mm'),
                    ('mean_fz_N','실행 구간 평균 Fz','N'),('peak_fz_N','실행 구간 관측 peak Fz','N'),
                    ('gate_on_s','접촉 게이트 ON 시간','s'),('time_fz_ge_50N_s','Fz ≥ 50 N 누적 시간','s')]
        for i,(key,label,unit) in enumerate(chart_defs):
            row,col=divmod(i,2)
            ax=fig.add_axes([.115+.46*col,.646-.257*row,.335,.155])
            dotplot(ax,groups,key,label,unit)
        body(fig,.073,'50 N은 힘 분포를 요약하기 위한 분석 기준이며, 장비의 판정 한계로 정의한 값이 아니다.',size=8.5,line=.02,color=GRAY,units=115)
        finish(pdf,fig)

        # 05 — Full selected force traces.
        fig=page('실측 힘의 시간 변화','executor/wrench.csv  ·  시작 정렬  ·  세 패널의 시간축과 힘 범위를 동일하게 표시')
        for j,m in enumerate('RTC'):
            ax=fig.add_axes([.115,.665-.238*j,.81,.145])
            for i,r in enumerate(groups[m]):
                mask=(r['ft']>=0)&(r['ft']<=r['end']-r['start'])
                ax.plot(r['ft'][mask],r['force'][mask,2],color=COLORS[m],alpha=.38+.14*i,lw=.8,label=r['metric']['label'])
            ax.set(xlim=(0,34),ylim=(-8,130),ylabel='base Fz (N)',xlabel='실행 시작 후 시간 (s)')
            ax.set_title(f'{m}  |  {NAMES[m]}',loc='left',weight='bold',pad=8)
            ax.legend(loc='upper right',ncol=5,frameon=False,handlelength=1)
        y=heading(fig,.130,'관측된 차이')
        body(fig,y,'T의 실행별 peak는 114.63–120.50 N, C는 73.14–101.39 N, R은 49.16–56.54 N이었다. 기록률은 약 18 Hz로, 기록 사이의 더 빠른 힘 변동이나 실제 연속 신호의 최대값까지 보장하지 않는다.',size=9.1,line=.021)
        finish(pdf,fig)

        # 06 — XY and Z show approach and actual motion.
        fig=page('실측 TCP 경로와 높이','왼쪽: 얼룩 기준점의 XY 평행이동을 제거  ·  오른쪽: robot_base Z  ·  첫 정지 요청까지')
        all_xy=np.concatenate([r['xy'] for r in runs]); lo=np.floor((all_xy.min(axis=0)-5)/10)*10; hi=np.ceil((all_xy.max(axis=0)+5)/10)*10
        for j,m in enumerate('RTC'):
            y=.623-.260*j
            ax=fig.add_axes([.11,y,.315,.202]); az=fig.add_axes([.58,y,.345,.202])
            for i,r in enumerate(groups[m]):
                alpha=.38+.14*i
                ax.plot(r['xy'][:,0],r['xy'][:,1],color=COLORS[m],alpha=alpha,lw=.9,label=r['metric']['label'])
                ax.scatter(*r['xy'][-1],color=COLORS[m],alpha=alpha,marker='s',s=10,zorder=3)
                az.plot(r['pt'],r['pose'][:,2],color=COLORS[m],alpha=alpha,lw=.9,label=r['metric']['label'])
            ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='상대 X (mm)',ylabel='상대 Y (mm)')
            ax.set_aspect('equal',adjustable='box')
            ax.set_title(f'{m}  |  XY 경로',loc='left',weight='bold',pad=8)
            az.set(xlim=(0,34),ylim=(160,220),xlabel='실행 시작 후 시간 (s)',ylabel='Z (mm)')
            az.set_title(f'{m}  |  TCP 높이',loc='left',weight='bold',pad=8)
            az.legend(loc='lower right',ncol=3,frameon=False,handlelength=1)
        body(fig,.055,'XY 사각형은 분석 종료 위치다. 회전·시편 변형 보정은 적용하지 않았으며 접근·복귀 일부가 포함될 수 있다.',size=8.1,units=122,line=.018,color=GRAY)
        finish(pdf,fig)

        # 07 — Actual data quality and differentiated completion.
        fig=page('기록 품질과 종료 상태','로그의 파일 완결성, 정지 확인, 목표 도달을 각각 구분')
        table(fig,['방법','힘 기록률\n평균 (Hz)','발행 명령 최대 간격\n범위 (ms)','실행 중 힘 샘플\n최대 간격 범위 (ms)'],
              [[m,f"{summary[m]['force_feedback_hz']['mean']:.2f}",
                f"{min(r['metric']['max_command_gap_ms'] for r in groups[m]):.2f}–{max(r['metric']['max_command_gap_ms'] for r in groups[m]):.2f}",
                f"{min(r['metric']['max_force_gap_execution_ms'] for r in groups[m]):.2f}–{max(r['metric']['max_force_gap_execution_ms'] for r in groups[m]):.2f}"] for m in 'RTC'],
              .674,.151,[.1,.22,.34,.34],8.4)
        body(fig,.648,'30개 logger(provider·executor)의 큐가 모두 비워졌고 누락·쓰기 오류는 0이다. 이는 로거 내부 집계이며 middleware_loss_count는 null이다. 125 Hz 명령 발행과 힘 피드백 기록률은 서로 다르다.',size=9.3,line=.023)
        table(fig,['방법','첫 정지 요청','최종 이벤트','정지 유지 확인'],[
            ['R (5회)','trajectory_end','safety_stop','5 / 5'],
            ['T (5회)','trajectory_end','trajectory_end','5 / 5'],
            ['C (5회)','operator_finish','normal_completion','5 / 5']],
            .430,.133,[.13,.29,.34,.24],8.5)
        y=body(fig,.406,'R1–R4는 후속 retract 도달/접촉 해제 확인 timeout, R5는 home 복귀 중 접촉 재검출이 기록됐다. 최종 이벤트는 첫 정지 요청 뒤 10.40–12.90초에 남았다. 보관 manifest는 R 5회를 작업자 확인·수용 상태로 포함한다.',size=9.4,line=.023)
        y=body(fig,y,'T는 시간 종료 시 마지막 요청 위치와 실측 TCP 사이의 3D 거리가 13.45–13.54 mm였다. 종료 위치를 보간해 계산한 단일 시점 거리이며 전체 궤적 추종 오차는 아니다.',size=9.4,line=.023)
        y=body(fig,y,'모든 최종 이벤트의 final_target_reached·retract_complete·home_pose_reached는 false, physical_contact_release는 unknown이다. C의 normal_completion은 작업자 종료 후 정지 확인을 뜻하며 표면 품질 완료를 판정한 것은 아니다.',size=9.4,line=.023)
        body(fig,y,'C는 실행마다 4개 계획을 수신했다. 첫 회를 제외한 추론 이벤트 구간의 실행별 중앙값은 58.08–58.54 ms이다. 이는 inference_start→end 전체 경과시간이며 GPU 연산만의 시간은 아니다.',size=9.1,line=.022,color=GRAY)
        finish(pdf,fig)

        # 08 — Original proxy, no spurious physical units or quality ranking.
        fig=page('기존 제거량 추정 지표','plots/summary.txt의 값을 그대로 집계  ·  임의단위(a.u.)  ·  실측 제거 깊이/질량 자료 없음')
        ax=fig.add_axes([.14,.589,.76,.210])
        dotplot(ax,groups,'raw_heatmap_mean','원본 히트맵 평균값','a.u.')
        ax.set_ylim(0,.34)
        table(fig,['방법','히트맵 평균값\n평균 ± SD (a.u.)','recorder 기록 길이\n평균 ± SD (s)','실행 비교 구간\n평균 (s)'],
              [[m,pm(summary,m,'raw_heatmap_mean',4),pm(summary,m,'recorder_duration_s',2),f"{summary[m]['execution_duration_s']['mean']:.3f}"] for m in 'RTC'],
              .383,.142,[.1,.33,.33,.24],8.6)
        y=heading(fig,.350,'히트맵을 해석하는 범위')
        y=body(fig,y,'모든 원본 summary에는 K=1.0, grid 1 mm, pad 반경 20 mm, 힘 기준 0.5 N, 속도 기준 0.1 mm/s가 기록돼 있다. 재료 제거 깊이·질량으로 보정된 값은 아니다.',size=9.5,line=.023)
        y=body(fig,y,'T와 C의 평균 지표는 0.2697, 0.2670으로 비슷하며 R은 0.1562다. 기록 구간, 경로, 방문 셀 영역이 달라 이 차이를 제거율 또는 표면 품질 우열로 해석하지 않는다.',size=9.5,line=.023)
        y=body(fig,y,'원본 recorder는 실행 구간 밖의 데이터도 포함하며 executor와 다른 기록 방식이다. 예를 들어 원본 최대 속도에 R1 140.8, R5 312.9, T4 362.8 mm/s가 나타난다. 이 값을 공통 명령 속도나 실행 중 실측 최고속도로 대체하지 않았다.',size=9.3,line=.022)
        body(fig,y,'12–14쪽에 15개 원본 히트맵을 모두 실었다. 각 그림의 색·좌표 범위는 원본을 유지하므로 색의 농도만으로 비교하지 않는다.',size=9.2,line=.022,color=GRAY)
        finish(pdf,fig)

        # 09–11 — All saved before/after images, no selection by appearance.
        for m in 'RTC':
            fig=page(f'{m} | 기록 초기·최종 사진',f'{NAMES[m]}  ·  provider/snapshots의 저장 이미지  ·  순서와 전체 화면 유지')
            fig.text(.375,.828,'초기 저장 이미지',ha='center',size=9.5,color=NAVY,weight='bold')
            fig.text(.755,.828,'최종 저장 이미지',ha='center',size=9.5,color=NAVY,weight='bold')
            for i,r in enumerate(groups[m]):
                y=.677-.145*i
                metric=r['metric']
                fig.text(.075,y+.11,metric['label'],size=12,weight='bold',color=COLORS[m])
                fig.text(.075,y+.085,metric['execution_start_local'][11:19],size=7.3,color=GRAY)
                fig.text(.075,y+.05,f"{metric['execution_duration_s']:.2f} s",size=7.6,color=INK)
                fig.text(.075,y+.026,f"Fz {metric['mean_fz_N']:.2f} N",size=7.1,color=INK)
                image_on(fig,[.205,y,.34,.136],r['base']/'provider/snapshots/initial_image.png')
                image_on(fig,[.585,y,.34,.136],r['base']/'provider/snapshots/final_image.png')
            body(fig,.067,'사진 저장 시점은 분석 구간 경계와 다를 수 있다. 공구 가림·조명·시점 차이가 있어 정량 제거율은 산출하지 않았다.',size=8.1,units=122,line=.018,color=GRAY)
            finish(pdf,fig)

        # 12–14 — All original heatmaps with readable axes and provenance.
        for m in 'RTC':
            fig=page(f'{m} | 원본 제거량 추정 히트맵',f'{NAMES[m]}  ·  각 그림의 색 범위와 XY 좌표는 원본 그대로  ·  단위: a.u.')
            for i,r in enumerate(groups[m]):
                row,col=divmod(i,2)
                x=.060+.45*col; y=.595-.254*row
                fig.text(x+.02,y+.227,f"{r['metric']['label']}  |  {r['record']['archive_run_id'][-6:]}",size=9,color=COLORS[m],weight='bold')
                image_on(fig,[x,y,.435,.223],r['base']/'plots/01_removal_heatmap.png')
            y=heading(fig,.299,f'{m} 5회 요약')
            # Put the note in the unused sixth panel instead of over the fifth plot.
            for t in fig.texts[-1:]:
                t.set_position((.545,.299))
            body(fig,.264,f"히트맵 평균의 5회 평균\n{summary[m]['raw_heatmap_mean']['mean']:.4f} a.u.\n반복 간 표본 SD\n{summary[m]['raw_heatmap_mean']['sample_sd']:.4f} a.u.\n\n그림 안 std는 각 히트맵의\n공간적 편차이며 반복 간 SD와 다르다.\n색상은 그림별 colorbar를 참조한다.",size=8.6,units=43,line=.023,x=.545,color=INK)
            body(fig,.056,'출처: 각 실행의 plots/01_removal_heatmap.png. 가공량을 실측한 결과가 아닌 원본 추정 시각자료다.',size=8.2,units=122,line=.018,color=GRAY)
            finish(pdf,fig)

        # 15 — Traceable scope and reproducibility.
        fig=page('자료 범위와 재현 정보','단일 PDF에 요약·정량 분석·15회 사진·15개 히트맵을 포함')
        y=heading(fig,.826,'분석 대상과 선택 이력')
        y=body(fig,y,'분석 대상은 results/20260927/E1 안에 보관된 15회다. 이전 날짜의 실행은 표본에 포함하지 않았다. manifest에는 실행 전 중단 R 1회·T 3회와, 정확히 5회 보관 요청에 따라 제외한 여섯 번째 T 실행의 이력이 남아 있다.',size=9.7,line=.024)
        y=body(fig,y,'이 보고서는 현재 선별된 묶음을 기술한다. 제외된 모든 시도까지 포함한 성공률이나 무작위 반복 실험의 통계적 우월성을 계산하지 않았다.',size=9.7,line=.024)
        table(fig,['자료 (각 실행 폴더 기준)','보고서에서 사용한 내용'],[
            ['manifest.json / run_index.csv (E1 루트)','표본 목록, 선택 이력, 상태, SHA-256'],
            ['executor/events.jsonl','시작·첫 정지 시각, 종료 이벤트, 계획 수신'],
            ['executor/wrench.csv / tcp_pose.csv','측정 힘, 시간 가중 통계, 실측 이동 경로'],
            ['executor/commands.csv','node_sent 발행 간격, 접촉 게이트 상태'],
            ['provider/events.jsonl / */summary.json','추론 이벤트 경과시간, 로거 집계와 완결성'],
            ['launch_context/config.json / */metadata.json','방법·제어 조건, 힘 좌표 의미, 기록 방식'],
            ['plots/summary.txt / 01_removal_heatmap.png','원본 추정 지표와 히트맵'],
            ['provider/snapshots/*.png','15회 초기·최종 이미지 30장'],
        ],.365,.261,[.53,.47],8.2)
        y=heading(fig,.331,'검증 및 재생성')
        y=body(fig,y,'목록 내 866개 파일(실행 자료 865개와 run_index.csv)의 크기·SHA-256을 확인했다. 15회 CSV 행 수, 이벤트 종료 상태, 설정 해시, 수치 유효성과 구간 경계도 대조했다. 영상 15개는 체크섬 검증 대상이며 이 PDF에 영상 자체를 삽입하지 않았다.',size=9.4,line=.023)
        y=body(fig,y,'계산 지표와 생성 코드: reports/20260927_E1_pdf_report\n재생성: python3 reports/20260927_E1_pdf_report/build_report.py',size=8.9,line=.022,units=107)
        fig.text(.07,.132,'원본 manifest SHA-256',size=8.4,color=NAVY,weight='bold')
        digest=result['manifest_sha256']
        fig.text(.07,.110,digest[:32],size=8.4,color=GRAY)
        fig.text(.07,.091,digest[32:],size=8.4,color=GRAY)
        fig.text(.07,.059,'작성: '+result['created_at'][:19].replace('T',' ')+' KST  |  측정 단위: mm, s, N',size=8,color=GRAY)
        finish(pdf,fig)
    assert len(PAGE_TITLES)==15
    temporary.replace(TARGET)
    validation=dict(output=str(TARGET),pages=len(PAGE_TITLES),titles=PAGE_TITLES,
                    sha256=sha(TARGET),bytes=TARGET.stat().st_size,
                    manifest_sha256=result['manifest_sha256'],verified_archive_files=result['verified_file_count'],
                    layout_checks=TEXT_CHECKS)
    (OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(validation,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    render(*analyze())
