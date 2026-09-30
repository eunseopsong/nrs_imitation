"""Generate one Korean PDF from the immutable, SHA-verified E1 archive."""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import textwrap
import unicodedata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np

ROOT=Path('/home/eunseop/nrs_imitation')
DATA=ROOT/'results/20260926/E1'
OUT=Path(__file__).resolve().parent
TARGET=ROOT/'results/20260926/E1_final_report_20260926.pdf'
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'previews').mkdir(exist_ok=True)
REG='/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
BOLD='/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
fontManager.addfont(REG);fontManager.addfont(BOLD)
plt.rcParams.update({'font.family':'NanumGothic','pdf.fonttype':42,'ps.fonttype':42,
    'axes.unicode_minus':False,'font.size':9,'axes.titlesize':10,'axes.labelsize':8,
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
NAVY='#173451';TEXT='#263746';MUTED='#647486';COLORS={'R':'#267a68','T':'#3268a8','C':'#c66b24'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p):return json.loads(p.read_text())
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def events(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def values(rr,keys):return np.array([[float(r[k]) if r.get(k) else np.nan for k in keys] for r in rr])
def times(rr):return np.array([int(r['receipt_monotonic_ns'])/1e9 for r in rr])


archive=js(DATA/'manifest.json')
for item in archive['files']:
    assert sha(DATA/item['path'])==item['sha256'],item['path']
r_audit=js(DATA/'audit/R/cleanup_manifest.json')
t_audit=js(DATA/'audit/T/audit.json')
c_audit=js(DATA/'audit/C/cleanup_manifest.json')
r_info={r['tag']:r for r in r_audit['runs'] if r['keep']}
t_info={r['run_tag']:r for r in t_audit['runs']}
c_info={r['tag']:r for r in c_audit['runs'] if r['keep']}


def force_stats(t,f,start,end):
    """Time weights use the latest recorded sample (zero-order hold)."""
    assert t[0]<=start<end<=t[-1]
    edges=np.r_[start,t[(t>start)&(t<end)],end]
    weights=np.diff(edges);index=np.searchsorted(t,edges[:-1],side='right')-1
    obs=f[index];mean=float(np.sum(weights*obs)/np.sum(weights))
    variance=float(np.sum(weights*(obs-mean)**2)/np.sum(weights))
    logged=f[(t>=start)&(t<=end)]
    return dict(mean_base_fz_N=mean,temporal_sd_base_fz_N=float(np.sqrt(variance)),
        max_logged_base_fz_N=float(logged.max()),min_logged_base_fz_N=float(logged.min()),
        time_fz_ge_3N_s=float(weights[obs>=3.].sum()),time_fz_ge_50N_s=float(weights[obs>=50.].sum()))


runs=[]
for rec in archive['runs']:
    base=DATA/rec['run_directory'];ep=base/'executor';pp=base/'provider'
    ev=events(ep/'events.jsonl');meta=js(ep/'metadata.json');summary=js(ep/'summary.json')
    started=next(e for e in ev if e['event']=='execution_start')
    stops=[e for e in ev if e['event']=='stop_requested' and not e['details'].get('initial_reset')]
    start=started['details']['monotonic_start'];end=stops[0]['receipt_monotonic_ns']/1e9
    wr=rows(ep/'wrench.csv');pr=rows(ep/'tcp_pose.csv')
    ft,pt=times(wr),times(pr);force=values(wr,['fx','fy','fz']);pose=values(pr,['x','y','z','rx','ry','rz'])
    pose_t=np.r_[start,pt[(pt>start)&(pt<end)],end]
    pose_window=np.stack([np.interp(pose_t,pt,pose[:,j]) for j in range(6)],axis=1)
    first_plan=next(e for e in ev if e['event']=='plan_received')
    ref=np.array(first_plan['details']['reference_xy_mm'])
    cr=[r for r in rows(ep/'commands.csv') if r['command_stage']=='node_sent' and
        start<=int(r['receipt_monotonic_ns'])/1e9<=end]
    ct=times(cr);detail=[json.loads(r['details']) for r in cr]
    assert all(d.get('controller_mode','Force')=='Force' for d in detail)
    action=values(cr,['x','y','z','rx','ry','rz','fx','fy','fz'])
    dt=np.minimum([d['dt_s'] for d in detail],.008)
    velocity=np.diff(action[:,:3],axis=0)/dt[1:,None]
    acceleration=np.diff(velocity,axis=0)/dt[2:,None]
    contact=np.array([d['contact'] for d in detail])
    raw_summary={}
    for line in (base/'plots/summary.txt').read_text().splitlines():
        key,value=line.split(':',1);raw_summary[key.strip()]=float(value.strip())
    inference=events(pp/'events.jsonl');begins={};latencies=[]
    for e in inference:
        if e['event']=='inference_start':begins[e['details']['inference_id']]=e['receipt_monotonic_ns']
        elif e['event']=='inference_end':
            seq=e['details']['inference_id']
            if seq in begins:latencies.append((e['receipt_monotonic_ns']-begins[seq])/1e6)
    metric=dict(method=rec['method'],repeat_index=rec['repeat_index'],label=rec['method']+str(rec['repeat_index']),
        run_tag=rec['run_tag'],run_tag_time=rec['run_tag'][-6:],execution_duration_s=end-start,
        **force_stats(ft,force[:,2],start,end),
        tcp_xy_path_mm=float(np.linalg.norm(np.diff(pose_window[:,:2],axis=0),axis=1).sum()),
        tcp_xyz_path_mm=float(np.linalg.norm(np.diff(pose_window[:,:3],axis=0),axis=1).sum()),
        command_rows_in_execution=len(cr),max_command_gap_ms=float(np.diff(ct).max()*1000),
        max_axis_command_speed_mm_s=float(np.abs(velocity).max()),
        max_axis_command_acceleration_mm_s2=float(np.abs(acceleration).max()),
        contact_gate_transitions_first4s=int(np.count_nonzero(np.diff(contact[ct-start<4.]))),
        contact_gate_transitions_total=int(np.count_nonzero(np.diff(contact))),
        normal_completion=rec['normal_completion'],termination_reasons=rec['termination_reasons'],
        record_status=rec['record_status'],tcp_rows_total=rec['tcp_rows'],force_rows_total=rec['force_rows'],
        removal_proxy_mean_raw=raw_summary['mean_heatmap_removal'],removal_recorder_duration_s=raw_summary['duration_s'],
        final_target_error_mm=t_info[rec['run_tag']]['final_tcp_target_error_mm'] if rec['method']=='T' else None,
        warm_pipeline_median_ms=float(np.median(latencies[1:])) if rec['method']=='C' and len(latencies)>1 else None,
        run_directory=rec['run_directory'],video=rec['video'])
    if rec['method']=='R':
        interval=r_info[rec['run_tag']]['processing_proxy']
        metric['R_processing_proxy_s']=interval['duration_s']
        work=force_stats(ft,force[:,2],interval['start_monotonic_ns']/1e9,interval['end_monotonic_ns']/1e9)
        metric['R_processing_base_fz_mean_N']=work['mean_base_fz_N']
    else:
        metric['R_processing_proxy_s']=None;metric['R_processing_base_fz_mean_N']=None
    assert np.isfinite(action).all() and np.isfinite(force).all() and np.isfinite(pose).all()
    runs.append(dict(metric=metric,base=base,meta=meta,start=start,end=end,
        ft=ft-start,force=force,pt=pose_t-start,pose=pose_window,xy=pose_window[:,:2]-ref,
        ct=ct-start,action=action,velocity=velocity,contact=contact,raw_summary=raw_summary))

metrics=[r['metric'] for r in runs]
groups={m:[r for r in runs if r['metric']['method']==m] for m in 'RTC'}
summary={}
for method,rr in groups.items():
    summary[method]={}
    for key in ('execution_duration_s','mean_base_fz_N','max_logged_base_fz_N','time_fz_ge_3N_s','time_fz_ge_50N_s','tcp_xy_path_mm'):
        a=np.array([r['metric'][key] for r in rr])
        summary[method][key]=dict(mean=float(a.mean()),sample_sd=float(a.std(ddof=1)),min=float(a.min()),max=float(a.max()))
stats=dict(created_at=datetime.now().astimezone().isoformat(),source=str(DATA),source_manifest_sha256=sha(DATA/'manifest.json'),
    analysis_window='engine monotonic_start to first non-startup stop_requested receipt; includes approach; excludes subsequent R return',
    force_axis='filtered/re-published robot-base Fz, not calibrated surface-normal force or TCP force-command error',
    force_time_weighting='zero-order hold of latest recorded sample; boundaries covered by available feedback',
    path_length='sum of planar measured TCP sample distances, linearly interpolated to interval endpoints',
    uncertainty='per-run temporal SD differs from sample SD across five run-level metrics; no hypothesis test or causal ranking',
    runs=metrics,method_summary=summary)
(OUT/'metrics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def wrap_cjk(s,units=92):
    out=[]
    for line in s.splitlines():
        current='';width=0
        for ch in line:
            inc=2 if unicodedata.east_asian_width(ch) in ('W','F') else 1
            if width+inc>units:
                out.append(current.rstrip());current='';width=0
            current+=ch;width+=inc
        out.append(current.rstrip())
    return out


page_number=0
page_titles=[]


def page(title,sub=''):
    global page_number
    page_number+=1;page_titles.append(title)
    fig=plt.figure(figsize=(8.27,11.69),facecolor='white',dpi=110)
    fig.text(.065,.957,'E1  /  2026.09.26',fontsize=10,color=MUTED,weight='bold')
    fig.text(.065,.914,title,fontsize=20,color=NAVY,weight='bold')
    if sub:fig.text(.065,.883,sub,fontsize=9.2,color=MUTED)
    fig.add_artist(plt.Line2D([.065,.94],[.867,.867],transform=fig.transFigure,color='#cfdae4',lw=1))
    fig.text(.065,.027,'R·T·C 각 5회 | 원본 데이터 기준 | 가공 품질의 독립 판정과 구분',fontsize=7.2,color=MUTED)
    fig.text(.94,.027,f'{page_number:02d}',ha='right',fontsize=9,color=MUTED)
    return fig


def body(fig,y,text,size=10.2,units=93,line=.023,color=TEXT,x=.07):
    for s in wrap_cjk(text,units):
        assert y>.055,('Text would overflow page',page_number,s)
        fig.text(x,y,s,fontsize=size,color=color,va='top')
        y-=line
    return y-.01


def heading(fig,y,text):
    fig.text(.07,y,text,fontsize=12,color=NAVY,weight='bold',va='top')
    return y-.032


def table(fig,headers,data,bottom,height,widths=None,fontsize=8.3):
    ax=fig.add_axes([.07,bottom,.87,height]);ax.axis('off')
    tab=ax.table(cellText=data,colLabels=headers,colWidths=widths,cellLoc='center',bbox=[0,0,1,1])
    tab.auto_set_font_size(False);tab.set_fontsize(fontsize)
    for (row,col),cell in tab.get_celld().items():
        cell.set_edgecolor('#d8e1e9');cell.set_linewidth(.45)
        if row==0:
            cell.set_facecolor(NAVY);cell.get_text().set_color('white');cell.get_text().set_weight('bold')
        else:cell.set_facecolor('#eef3f7' if row%2 else 'white')
    return tab


def finish(pdf,fig):
    fig.canvas.draw()
    # Check every figure-level label, including wrapped Korean prose, against
    # the physical page. Axes labels are visually reviewed in rendered pages.
    renderer=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height()
    for text in fig.texts:
        box=text.get_window_extent(renderer)
        assert box.x0>=0 and box.x1<=w+1 and box.y0>=0 and box.y1<=h+1,(page_number,text.get_text())
    pdf.savefig(fig)
    fig.savefig(OUT/'previews'/f'page_{page_number:02d}.png',dpi=95)
    plt.close(fig)


def pm(method,key,d=2):
    s=summary[method][key];return f"{s['mean']:.{d}f} ± {s['sample_sd']:.{d}f}"


partial=TARGET.with_name(TARGET.stem+'.building.pdf')
with PdfPages(partial) as pdf:
    pdf.infodict().update(Title='E1 실험 데이터 최종 보고서 — 2026-09-26',
        Author='E1 experiment data audit',Subject='R/T/C 15회 기록·실행 결과 및 보관 검증',Keywords='E1 R T C 20260926')
    fig=page('실험 데이터 최종 보고서','R: 규칙 경로  ·  T: 교시 재생  ·  C: 힘 관측 포함 학습 정책')
    y=heading(fig,.823,'15회 데이터·영상 보관 완료')
    y=body(fig,y,'오늘 보존한 R 5회, T 5회, C 5회의 실행 데이터와 원본 영상 15개를 묶었다. 원본 실험 파일 865개를 복사했고, 전부 SHA-256 일치를 확인했다. 기록 누락(drop)·쓰기 오류는 0이다.')
    table(fig,['방법','기록 완료','영상','종료 상태'],[
        ['R','5 / 5','5','가공 기록 완료, 이후 복귀 timeout'],
        ['T','5 / 5','5','재생 종료·정지, 최종 목표 미도달'],
        ['C','5 / 5','5','작업자 finish 후 정상 종료·정지 확인']],.545,.15,[.10,.15,.10,.65],8.7)
    y=heading(fig,.507,'실행 구간의 요약 지표')
    body(fig,y,'아래 ±는 5개 실행별 지표 사이의 표본 표준편차다. 접근을 포함한 실행 구간을 요약했으며, 순수 가공 시간이나 완료 주기의 우열을 뜻하지 않는다.',size=9.2,line=.021)
    table(fig,['방법','실행 시간 (s)','평균 base Fz (N)','실측 peak Fz (N)'],
        [[m,pm(m,'execution_duration_s'),pm(m,'mean_base_fz_N'),pm(m,'max_logged_base_fz_N')] for m in 'RTC'],
        .284,.13,[.10,.30,.30,.30],8.5)
    y=heading(fig,.245,'결과 해석')
    y=body(fig,y,'C 5회는 사용자가 의도하지 않은 동작이 없었다고 확인했다. 접촉 판정 전환은 원본 지표로 유지했다. R/T의 기존 기록·설정도 보존했다.',size=9.8,line=.022)
    body(fig,y,'C에는 추가 궤적 안정화 패치가 적용돼 있다. 종료 기준과 실행 길이도 방법별로 달라, 모델 자체의 우수성·가공 품질·성공률을 이 요약만으로 확정하지 않는다.',size=9.8,line=.022)
    finish(pdf,fig)

    fig=page('실행 조건과 지표 정의','오늘의 R/T/C 비교 묶음을 E1으로 표시한다. 과거 B/C 실험 자료는 포함하지 않았다.')
    table(fig,['항목','R','T','C'],[
        ['경로 공급','규칙 기반 편도 1회','episode_29 원본 재생','FLOW 정책 예측'],
        ['정책/계획','힘 목표 +18 N','원본 위치·힘·시간','힘 관측 ON, 128점 / 30 Hz'],
        ['진행 방식','유한 경로','14.9104초 유한 경로','약 4초마다 재계획'],
        ['종료 기준','경로 끝 → 복귀 시도','원본 시간축 끝','작업자 finish'],
        ['C 전용 처리','없음','없음','MA35 · 0.5초 연결 · 가속 제한']],.603,.225,[.18,.25,.25,.32],8.1)
    y=heading(fig,.565,'공통 실행 조건')
    y=body(fig,y,'timed_topic 전송, 125 Hz 발행, gain 15/s, 위치 속도 축별 10 mm/s, 회전 속도 축별 40 deg/s, 힘 변화율 30 N/s. 접촉 판정 ON 3.0 N / OFF 1.2 N과 기존 작업영역·피드백 감시를 공유한다.',size=9.7,line=.022)
    y=body(fig,y,'C만 위치·자세의 35점 평활과 0.5초 계획 연결, 축별 위치 가속도 25 mm/s² / 회전 가속도 100 deg/s²를 추가했다. 원본 예측 시간축·힘 목표·공통 접촉 판정은 유지한다.',size=9.7,line=.022)
    y=heading(fig,y-.012,'분석 구간과 단위')
    y=body(fig,y,'실행 구간: execution_start의 monotonic_start부터 최초 stop_requested까지. 초기 접근을 포함하고 R의 이후 복귀는 제외한다. 모든 방법에 processing_start/end 작업자 마커가 없어 순수 가공 시간은 별도로 확정하지 않는다.',size=9.4,line=.021)
    y=body(fig,y,'힘: 필터링·재발행된 robot-base Fz. 평균과 시간 표준편차는 최신 관측값 유지(ZOH) 방식으로 시간 가중했다. peak는 기록된 표본의 최대값이다. 목표 힘은 제어기 TCP 축이므로 base Fz와 빼서 추종 오차로 계산하지 않았다.',size=9.4,line=.021)
    y=body(fig,y,'τ50: base Fz가 50 N 이상인 누적 시간(초). 토크나 안전 한계가 아니다. XY 길이: 같은 구간의 실측 TCP 평면 경로 길이이며, 구간 양 끝은 선형 보간했다. 센서 기록은 최대 약 20 Hz라 고주파 진동을 완전히 복원할 수 없다.',size=9.4,line=.021)
    finish(pdf,fig)

    fig=page('15회 실행별 수치','모두 같은 실행 구간 정의로 계산. 평균 ± σt는 한 실행 안의 시간 변동이다.')
    data=[]
    for m in metrics:
        data.append([m['label']+'  '+m['run_tag_time'][:2]+':'+m['run_tag_time'][2:4]+':'+m['run_tag_time'][4:],
            f"{m['execution_duration_s']:.3f}",f"{m['mean_base_fz_N']:.2f} ± {m['temporal_sd_base_fz_N']:.2f}",
            f"{m['max_logged_base_fz_N']:.2f}",f"{m['time_fz_ge_50N_s']:.3f}",f"{m['tcp_xy_path_mm']:.2f}"])
    table(fig,['실행 / 태그 시각','시간\n(s)','base Fz 평균 ± σt\n(N)','peak Fz\n(N)','τ50\n(s)','XY 길이\n(mm)'],data,
        .275,.552,[.23,.11,.24,.13,.13,.16],8.2)
    y=heading(fig,.239,'표를 읽는 기준')
    y=body(fig,y,'시간은 가공만의 시간이 아니다. R은 원본 경로 진행을 마친 뒤 복귀가 별도로 실패했고, T는 재생 시간이 끝났지만 최종 목표 도달은 확인되지 않았다. C는 작업자가 종료했다.',size=9.6,line=.022)
    body(fig,y,'힘의 σt는 한 실행 안에서 변한 정도이며, 5회 반복 간 표준편차가 아니다. τ50과 XY 길이는 구간 길이와 경로에 영향을 받으므로 작은 값만으로 성능이 좋다고 판단하지 않는다.',size=9.6,line=.022)
    finish(pdf,fig)

    for method in 'RTC':
        subtitles={'R':'규칙 경로 +18 N · 원본 경로 종료까지의 센서 기록',
                    'T':'episode_29 원본 시간축 14.9104초 · 재생 종료까지의 센서 기록',
                    'C':'동일 체크포인트 + C 전용 안정화 패치 · 작업자 finish까지의 기록'}
        fig=page(method+' 실행 궤적과 힘',subtitles[method])
        rr=groups[method]; cmap=plt.get_cmap({'R':'Greens','T':'Blues','C':'Oranges'}[method])
        palette=[cmap(x) for x in np.linspace(.45,.9,5)]
        force_ax=fig.add_axes([.09,.548,.39,.27]);xy_ax=fig.add_axes([.58,.548,.35,.27])
        for r,color in zip(rr,palette):
            d=r['metric'];mask=(r['ft']>=0)&(r['ft']<=d['execution_duration_s'])
            force_ax.plot(r['ft'][mask],r['force'][mask,2],lw=1.,color=color,label=d['label'])
            xy_ax.plot(r['xy'][:,0],r['xy'][:,1],lw=1.15,color=color,label=d['label'])
        force_ax.set(title='실측 base Fz',xlabel='실행 경과 (s)',ylabel='Fz (N)',ylim=(-8,130))
        force_ax.legend(ncol=3,loc='upper right',fontsize=6.5)
        xy_ax.set(title='실측 TCP 경로',xlabel='오염 원점 상대 X (mm)',ylabel='상대 Y (mm)')
        xy_ax.set_aspect('equal',adjustable='datalim')
        if method!='C':
            z_ax=fig.add_axes([.09,.284,.84,.18])
            for r,color in zip(rr,palette):z_ax.plot(r['pt'],r['pose'][:,2],lw=1.,color=color,label=r['metric']['label'])
            z_ax.set(title='실측 TCP 높이',xlabel='실행 경과 (s)',ylabel='Base Z (mm)')
            z_ax.legend(ncol=5,fontsize=7,loc='upper right')
        else:
            vx=fig.add_axes([.09,.284,.39,.18]);gate=fig.add_axes([.59,.284,.34,.18])
            for r,color in zip(rr,palette):vx.plot(r['ct'][1:],r['velocity'][:,0],lw=1.,color=color,label=r['metric']['label'])
            vx.set(xlim=(0,4),ylim=(-11,11),title='첫 4초 발행 X 속도',xlabel='실행 경과 (s)',ylabel='mm/s')
            gate.bar([r['metric']['label'] for r in rr],[r['metric']['contact_gate_transitions_first4s'] for r in rr],color=palette)
            gate.set(title='첫 4초 접촉 판정 전환',ylabel='횟수',ylim=(0,70))
        if method=='R':
            note='R 5회는 processing 단계 약 17.104초의 가공 구간 기록을 모두 확보했다. 그 뒤 retract 도착/접촉 해제 확인 timeout으로 홈 복귀는 완료되지 않았다. 그림은 복귀 이전 실행 구간만 포함한다.'
        elif method=='T':
            errors=[r['metric']['final_target_error_mm'] for r in rr]
            note=f'T 5회 모두 재생과 정지 유지는 확인됐다. 최종 TCP 목표 오차는 {min(errors):.2f}~{max(errors):.2f} mm이며 final_target_reached=false였다. 원본 시간축은 유지됐지만 실제 목표 도달 완료와 같지는 않다.'
        else:
            note='C 5회 모두 정상 종료 이벤트와 정지 유지가 확인됐다. 실제 발행 위치 가속도는 축별 25 mm/s² 이내였다. 접촉 판정 전환은 0/58/40/4/8회이며, 사용자가 의도하지 않은 동작이 없었다고 확인해 5회 모두 보존했다.'
        body(fig,.204,note,size=9.7,line=.023)
        body(fig,.106,'XY는 실행별 동결 오염 원점을 뺀 좌표다. 힘 그래프는 접촉 법선 힘이나 가공 품질을 직접 측정한 값이 아니다.',size=8.6,line=.02,color=MUTED)
        finish(pdf,fig)

    for method in 'RTC':
        fig=page(method+' 초기·종료 이미지','모든 보존 실행을 순서대로 표시. 이미지로 제거율이나 가공 성공을 자동 판정하지 않았다.')
        fig.text(.295,.836,'초기 기록',ha='center',fontsize=10,color=NAVY,weight='bold')
        fig.text(.725,.836,'종료 기록',ha='center',fontsize=10,color=NAVY,weight='bold')
        for i,r in enumerate(groups[method]):
            top=.812-i*.145
            fig.text(.07,top,r['metric']['label']+'  '+r['metric']['run_tag_time'],fontsize=8.4,color=TEXT,weight='bold')
            for left,name in [(.09,'initial_image.png'),(.52,'final_image.png')]:
                ax=fig.add_axes([left,top-.128,.395,.12]);ax.axis('off')
                ax.imshow(plt.imread(r['base']/'provider/snapshots'/name))
        body(fig,.069,'원본 영상과 424 × 240 원본 이미지는 각 실행의 video/ 및 provider/snapshots/에 보관했다.',size=8.2,line=.018,color=MUTED)
        finish(pdf,fig)

    fig=page('검증·보관·해석 범위','전체 데이터의 출처와 선별 기준을 남겨 재검토할 수 있도록 구성했다.')
    table(fig,['방법','당일 시도','보존','제외','제외/보존 기준'],[
        ['R','20','5','15','가공 구간 기록 완전성; 복귀 실패는 보존'],
        ['T','5','5','0','원본 재생·정지·기록 완료'],
        ['C','8','5','3','시작 전 중단 2회, 패치 전 진동 1회 제외']],.653,.17,[.10,.12,.10,.10,.58],8.3)
    y=heading(fig,.611,'파일 검증')
    y=body(fig,y,'C 비정상 3회의 관련 109개 파일을 삭제했다. C 보존 파일 285개와 R/T 파일 580개의 해시가 그대로임을 확인한 뒤, 총 865개 원본 실험 파일을 결과 폴더로 복사했다. 영상 15개와 audit/index를 포함한 보관본은 882개 파일이다.',size=9.6,line=.022)
    y=body(fig,y,'기록 CSV의 유한값·시간 순서·행 수·종료 시 큐 완료를 검사했다. 영상 전체 디코딩과 이미지 읽기 검사를 통과했다. SHA-256 출처/복사본 대조도 완료했다. 모델과 실행 조건의 스냅샷은 실행별 metadata 및 artifacts에 남아 있다.',size=9.6,line=.022)
    y=heading(fig,y-.006,'보관 위치')
    y=body(fig,y,'results/20260926/E1/{R,T,C}/RTC_<방법>_<실행태그>/\n  executor/ · provider/ · launch_context/ · plots/ · video/ · ros_logs/\nE1/run_index.csv : 15회 목록, 종료 상태, 영상·데이터 상대 경로\nE1/manifest.json : 복사 파일별 원본 경로·크기·SHA-256\nE1/audit/ : R/T/C 기록 검사와 C 패치 설명',size=8.8,units=102,line=.021)
    y=heading(fig,y-.008,'최종 해석 범위')
    y=body(fig,y,'기록 완료와 물리 작업 성공은 구분한다. C의 추가 처리, 서로 다른 종료 기준, 동일 시편/영역의 독립성 미확인 때문에 인과적 모델 성능 순위나 통계적 우월성은 결론 내리지 않았다. RPM과 표면 법선 보정도 새로 측정하지 않았다.',size=9.4,line=.021)
    y=body(fig,y,'plots의 제거량 히트맵은 k=1 접촉·속도 기반 대용지표다. 센서/영상 전체 기록 구간이 방법별로 달라 절대 제거량이나 공정 간 품질 순위로 집계하지 않았다. 실제 가공량과 성공 여부에는 별도 판정이 필요하다.',size=9.4,line=.021)
    body(fig,y,'보고서 수치 재현 자료: reports/20260926_E1_final_report/metrics.csv 및 metrics.json. 입력은 보관된 E1 폴더이며 삭제된 비정상 로그를 사용하지 않는다.',size=8.4,units=106,line=.019,color=MUTED)
    finish(pdf,fig)

partial.replace(TARGET)
assert page_number==10
for entry in archive['files']:assert sha(DATA/entry['path'])==entry['sha256'],entry['path']
validation=dict(output=str(TARGET),pages=page_number,page_titles=page_titles,bytes=TARGET.stat().st_size,
    pdf_sha256=sha(TARGET),run_count=15,counts_by_method={m:len(groups[m]) for m in 'RTC'},
    all_archive_file_hashes_unchanged=True,source_manifest_sha256=sha(DATA/'manifest.json'),
    metrics_csv_sha256=sha(OUT/'metrics.csv'),metrics_json_sha256=sha(OUT/'metrics.json'),
    font='Embedded NanumGothic TrueType',text_bounds_checked=True,
    analysis_uses_archived_good_runs_only=True,raw_data_modified=False)
(OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(validation,ensure_ascii=False,indent=2),flush=True)
