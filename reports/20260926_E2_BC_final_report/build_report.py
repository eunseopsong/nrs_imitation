"""Korean E2 B/C report using the same descriptive metrics as the E1 report."""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import re
import unicodedata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import fontManager
import numpy as np

ROOT=Path('/home/eunseop/nrs_imitation')
DATA=ROOT/'results/20260926/E2'
OUT=Path(__file__).resolve().parent
TARGET=ROOT/'results/20260926/E2_BC_final_report_20260926.pdf'
OUT.mkdir(parents=True,exist_ok=True);(OUT/'previews').mkdir(exist_ok=True)
for f in ['NanumGothic.ttf','NanumGothicBold.ttf']:fontManager.addfont('/usr/share/fonts/truetype/nanum/'+f)
plt.rcParams.update({'font.family':'NanumGothic','pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':False,
    'font.size':9,'axes.titlesize':10,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
    'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
NAVY='#173451';TEXT='#263746';MUTED='#647486';COLORS={'B':'#3268a8','C':'#c66b24'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p):return json.loads(p.read_text())
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))
def events(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def vals(rr,keys):return np.array([[float(r[k]) for k in keys] for r in rr])
def times(rr):return np.array([int(r['receipt_monotonic_ns'])/1e9 for r in rr])


def force_stats(t,f,start,end):
    assert t[0]<=start<end<=t[-1]
    edges=np.r_[start,t[(t>start)&(t<end)],end];w=np.diff(edges)
    values=f[np.searchsorted(t,edges[:-1],side='right')-1];mean=float(np.average(values,weights=w))
    return dict(mean_base_fz_N=mean,temporal_sd_base_fz_N=float(np.sqrt(np.average((values-mean)**2,weights=w))),
        max_logged_base_fz_N=float(f[(t>=start)&(t<=end)].max()),min_logged_base_fz_N=float(f[(t>=start)&(t<=end)].min()),
        time_fz_ge_3N_s=float(w[values>=3].sum()),time_fz_ge_50N_s=float(w[values>=50].sum()))


archive=js(DATA/'manifest.json')
for item in archive['files']:assert sha(DATA/item['path'])==item['sha256'],item['path']
runs=[]
for rec in archive['runs']:
    base=DATA/rec['run_directory'];ep=base/'executor';pp=base/'provider'
    ev=events(ep/'events.jsonl');start_event=next(e for e in ev if e['event']=='execution_start')
    start=start_event['details']['monotonic_start']
    end=next(e['receipt_monotonic_ns']/1e9 for e in ev if e['event']=='stop_requested' and not e['details'].get('initial_reset'))
    wr=rows(ep/'wrench.csv');pr=rows(ep/'tcp_pose.csv');ft,pt=times(wr),times(pr)
    f=vals(wr,['fx','fy','fz']);p=vals(pr,['x','y','z','rx','ry','rz'])
    assert pt[0]<=start<end<=pt[-1]
    window=np.r_[start,pt[(pt>start)&(pt<end)],end]
    pose=np.column_stack([np.interp(window,pt,p[:,i]) for i in range(6)])
    cr=[r for r in rows(ep/'commands.csv') if r['command_stage']=='node_sent' and start<=int(r['receipt_monotonic_ns'])/1e9<=end]
    ct=times(cr);details=[json.loads(r['details']) for r in cr];a=vals(cr,['x','y','z','rx','ry','rz','fx','fy','fz'])
    dt=np.minimum([d['dt_s'] for d in details],.008);velocity=np.diff(a[:,:3],axis=0)/dt[1:,None]
    accel=np.diff(velocity,axis=0)/dt[2:,None];contact=np.array([d['contact'] for d in details])
    origin=np.array(next(e['details']['reference_xy_mm'] for e in ev if e['event']=='plan_received'))
    video_log=[]
    for log in (base/'ros_logs').glob('*.log'):
        for line in log.read_text().splitlines():
            if '[overlay_video_recorder]' in line and 'recording composite' in line:
                video_log.append(float(re.search(r'\[(\d+\.\d+)\]',line).group(1)))
    assert len(video_log)==1
    delay=video_log[0]-start_event['receipt_ros_ns']/1e9
    assert not any(e['event'] in ('processing_start','processing_end') for e in ev)
    raw={}
    for line in (base/'plots/summary.txt').read_text().splitlines():
        k,v=line.split(':',1);raw[k.strip()]=float(v.strip())
    seq={};latencies=[]
    for e in events(pp/'events.jsonl'):
        if e['event']=='inference_start':seq[e['details']['inference_id']]=e['receipt_monotonic_ns']
        if e['event']=='inference_end' and e['details']['inference_id'] in seq:
            latencies.append((e['receipt_monotonic_ns']-seq[e['details']['inference_id']])/1e6)
    m=dict(condition=rec['condition'],repeat_index=rec['repeat_index'],label=rec['condition']+str(rec['repeat_index']),
        run_id=rec['run_id'],tag_time=rec['run_id'].split('T')[1].split('_')[0],execution_duration_s=end-start,
        **force_stats(ft,f[:,2],start,end),tcp_xy_path_mm=float(np.linalg.norm(np.diff(pose[:,:2],axis=0),axis=1).sum()),
        tcp_xyz_path_mm=float(np.linalg.norm(np.diff(pose[:,:3],axis=0),axis=1).sum()),
        node_sent_rows=len(cr),max_command_gap_ms=float(np.diff(ct).max()*1000),
        max_axis_command_speed_mm_s=float(np.abs(velocity).max()),max_axis_command_acceleration_mm_s2=float(np.abs(accel).max()),
        contact_gate_transitions_total=int(np.count_nonzero(np.diff(contact))),
        contact_gate_transitions_first4s=int(np.count_nonzero(np.diff(contact[ct-start<4]))),
        video_duration_s=rec['video_duration_s'],video_frames=rec['video_frames'],video_start_delay_s=delay,
        video_initial_delay_over_1s=delay>1,video_full_decode_passed=rec['video_full_decode_passed'],
        warm_pipeline_median_ms=float(np.median(latencies[1:])),normal_completion=True,controller_hold_verified=True,
        removal_proxy_mean_raw=raw['mean_heatmap_removal'],removal_recorder_duration_s=raw['duration_s'],
        processing_time_s=None,E_profile_N=None,E_target_N=None,E_track_applied_N=None,E_discrepancy_sent_N=None,
        run_directory=rec['run_directory'],video=rec['video'])
    assert np.isfinite(a).all() and np.isfinite(f).all() and np.isfinite(pose).all()
    runs.append(dict(m=m,base=base,ft=ft-start,f=f,pt=window-start,pose=pose,xy=pose[:,:2]-origin,
        ct=ct-start,a=a,velocity=velocity,contact=contact))
metrics=[r['m'] for r in runs];groups={c:[r for r in runs if r['m']['condition']==c] for c in 'BC'}
keys=['execution_duration_s','mean_base_fz_N','max_logged_base_fz_N','time_fz_ge_3N_s','time_fz_ge_50N_s','tcp_xy_path_mm','contact_gate_transitions_total']
summary={c:{k:dict(mean=float(np.mean([r['m'][k] for r in rr])),sample_sd=float(np.std([r['m'][k] for r in rr],ddof=1))) for k in keys} for c,rr in groups.items()}
stats=dict(created_at=datetime.now().astimezone().isoformat(),source=str(DATA),source_manifest_sha256=sha(DATA/'manifest.json'),
    analysis_window='executor monotonic_start to first non-startup stop_requested receipt, including approach',
    force_axis='filtered/re-published robot-base Fz; not calibrated normal force or TCP command tracking error',
    weighting='latest recorded sensor value held (ZOH); sensor coverage includes both boundaries',
    path='sum of measured TCP sample distances with linearly interpolated boundaries',
    uncertainty='Within-run temporal SD and between-run sample SD are distinct. Descriptive comparison only.',
    no_processing_markers=True,reference_frozen=False,normal_force_calibration_verified=False,
    video_limitation='C5 recorder started 4.223 s after execution_start; other runs 0.139–0.201 s. Videos are decodable, not complete pre-roll records.',
    runs=metrics,condition_summary=summary)
save(OUT/'metrics.json',stats)
with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def wrap(s,units=92):
    result=[]
    for line in s.splitlines():
        current='';width=0
        for ch in line:
            n=2 if unicodedata.east_asian_width(ch) in ('W','F') else 1
            if width+n>units:result.append(current.rstrip());current='';width=0
            current+=ch;width+=n
        result.append(current.rstrip())
    return result
number=0;titles=[]
def page(title,sub=''):
    global number
    number+=1;titles.append(title);fig=plt.figure(figsize=(8.27,11.69),facecolor='white',dpi=110)
    fig.text(.065,.957,'E2  /  B·C  /  2026.09.26',fontsize=10,color=MUTED,weight='bold')
    fig.text(.065,.914,title,fontsize=20,color=NAVY,weight='bold')
    if sub:fig.text(.065,.883,sub,fontsize=9.2,color=MUTED)
    fig.add_artist(plt.Line2D([.065,.94],[.867,.867],transform=fig.transFigure,color='#cfdae4',lw=1))
    fig.text(.065,.027,'B·C 각 5회 | 원본 데이터 기준 | A 미포함 | 실행 기록과 가공 품질을 구분',fontsize=7.2,color=MUTED)
    fig.text(.94,.027,f'{number:02d}',ha='right',fontsize=9,color=MUTED);return fig
def body(fig,y,text,size=10,units=92,line=.023,color=TEXT):
    for s in wrap(text,units):
        assert y>.055,(number,s)
        fig.text(.07,y,s,fontsize=size,color=color,va='top');y-=line
    return y-.01
def heading(fig,y,text):
    fig.text(.07,y,text,fontsize=12,color=NAVY,weight='bold',va='top');return y-.032
def table(fig,headers,data,bottom,height,widths=None,size=8.3):
    ax=fig.add_axes([.07,bottom,.87,height]);ax.axis('off')
    tab=ax.table(cellText=data,colLabels=headers,colWidths=widths,cellLoc='center',bbox=[0,0,1,1])
    tab.auto_set_font_size(False);tab.set_fontsize(size)
    for (row,col),cell in tab.get_celld().items():
        cell.set_edgecolor('#d8e1e9');cell.set_linewidth(.45)
        if row==0:cell.set_facecolor(NAVY);cell.get_text().set_color('white');cell.get_text().set_weight('bold')
        else:cell.set_facecolor('#eef3f7' if row%2 else 'white')
    return tab
def finish(pdf,fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height()
    for t in fig.texts:
        b=t.get_window_extent(renderer);assert b.x0>=0 and b.x1<=w+1 and b.y0>=0 and b.y1<=h+1,(number,t.get_text())
    pdf.savefig(fig);fig.savefig(OUT/'previews'/f'page_{number:02d}.png',dpi=90);plt.close(fig)
def pm(c,k):
    s=summary[c][k];return f"{s['mean']:.2f} ± {s['sample_sd']:.2f}"


partial=TARGET.with_suffix('.building.pdf')
with PdfPages(partial) as pdf:
    pdf.infodict().update(Title='E2 B/C 실험 데이터 최종 보고서 — 2026-09-26',Author='E2 data audit',Subject='B/C 10회 실행 기록 및 데이터·영상 보관')
    fig=page('B/C 실험 데이터 최종 보고서','B: 힘 관측 OFF + 힘 출력  ·  C: 힘 관측 ON + 힘 출력')
    y=heading(fig,.823,'정상 실행 10회 보존 · 시작 실패 2회 삭제')
    body(fig,y,'B 5회와 C 5회 모두 작업자 finish, normal_completion 및 실제 정지 유지가 확인됐다. 명령·센서 기록의 logger drop과 쓰기 오류는 0이다. 원본 파일 810개를 SHA-256 검증 후 복사했다.')
    table(fig,['조건','정상 실행','영상 파일','정리 결과'],[
        ['B','5 / 5','5','시작 실패 2회 삭제'],['C','5 / 5','5','제외 없음 · C5 영상 시작 지연']],.555,.135,[.10,.17,.17,.56],8.7)
    y=heading(fig,.519,'실행 구간의 기술 통계')
    body(fig,y,'±는 실행별 지표 5개 사이의 표본 표준편차다. 분석 구간은 접근을 포함한 실행 시작부터 작업자 종료 요청까지다.',size=9.3,line=.022)
    table(fig,['조건','실행 시간 (s)','평균 base Fz (N)','peak base Fz (N)'],
        [[c,pm(c,'execution_duration_s'),pm(c,'mean_base_fz_N'),pm(c,'max_logged_base_fz_N')] for c in 'BC'],.326,.12,[.10,.30,.30,.30],8.6)
    y=heading(fig,.286,'함께 확인할 사항')
    y=body(fig,y,'C5 영상은 실행 시작 약 4.223초 뒤에 녹화가 시작됐다. 해당 실행의 명령·센서 데이터는 온전하므로 보존했고, 영상 초반 누락을 별도 표기했다. 나머지 영상의 시작 지연은 0.139~0.201초다.',size=9.5,line=.022)
    body(fig,y,'processing_start/end 마커와 검증된 법선 힘 기준 프로파일이 없어 순수 가공 시간 및 E_profile은 미산출(NA)이다. 실험의 물리적 제거 품질이나 모델 우열은 확정하지 않는다.',size=9.5,line=.022)
    finish(pdf,fig)

    fig=page('실행 조건과 비교 범위','동일 실행 계약 및 코드 스냅샷을 사용하는 B/C 묶음이다.')
    table(fig,['항목','B','C'],[
        ['관측 힘','정규화 후 0으로 고정','힘 관측 + history 사용'],
        ['동작 출력','위치·자세 6 + 힘 3','위치·자세 6 + 힘 3'],
        ['체크포인트','독립 학습 OFF / best','독립 학습 ON / best'],
        ['계획·전송','128점 / 30 Hz, timed_topic','동일'],
        ['실행 후처리','E1 C 기반 MA35·연결·가속 제한','동일'],
        ['종료','작업자 finish / 상한 50초','동일']],.566,.265,[.24,.40,.36],8.4)
    y=heading(fig,.529,'공통 실행 조건')
    y=body(fig,y,'125 Hz 발행, gain 15/s, 축별 위치 속도 10 mm/s, 회전 속도 40 deg/s, 힘 변화율 30 N/s. 접촉 판정 ON 3.0 N / OFF 1.2 N, MA35 위치·자세 평활, 0.5초 연결, 위치 가속도 25 mm/s²를 공유한다.',size=9.6,line=.022)
    y=body(fig,y,'시편은 매회 동일 상태라는 사용자 가정을 적용했다. 잉크 제거·한 방향 작업·E1 C 설정을 사용했다. B 5회 다음 C 5회를 실행했으므로 순서·환경 변화 효과를 분리할 수 없다.',size=9.6,line=.022)
    y=heading(fig,y-.006,'E1과 동일한 수치 정의')
    y=body(fig,y,'실행 시간: execution_start의 monotonic_start부터 첫 stop_requested까지. 힘: robot-base Fz의 시간 가중 평균·시간 표준편차(ZOH). peak는 기록 표본 최대값이다. XY 길이는 실제 TCP 표본 간 평면 거리 합이며 구간 양끝을 선형 보간했다.',size=9.3,line=.021)
    body(fig,y,'τ50은 base Fz ≥ 50 N 누적 시간이며 안전 기준이 아니다. TCP 힘 명령과 base 센서 Fz의 좌표계가 달라 둘을 그대로 빼서 추종 오차를 만들지 않았다. 기록은 최대 20 Hz로 고주파 진동을 전부 복원하지 못한다.',size=9.3,line=.021)
    finish(pdf,fig)

    fig=page('10회 실행별 수치','평균 ± σt는 한 실행 안의 시간 변동이며 반복 간 표준편차와 구분한다.')
    data=[]
    for m in metrics:
        t=m['tag_time'];data.append([m['label']+'  '+t[:2]+':'+t[2:4]+':'+t[4:],f"{m['execution_duration_s']:.3f}",
            f"{m['mean_base_fz_N']:.2f} ± {m['temporal_sd_base_fz_N']:.2f}",f"{m['max_logged_base_fz_N']:.2f}",
            f"{m['time_fz_ge_50N_s']:.3f}",f"{m['tcp_xy_path_mm']:.2f}"])
    table(fig,['실행 / 시작 태그','시간\n(s)','base Fz 평균 ± σt\n(N)','peak Fz\n(N)','τ50\n(s)','XY 길이\n(mm)'],data,.34,.49,[.23,.11,.24,.13,.13,.16],8.3)
    y=heading(fig,.293,'산출하지 않은 지표')
    y=body(fig,y,'순수 가공 시간, E_profile, E_target, E_track_applied, 보정된 법선 힘 기반 discrepancy는 모두 NA로 보존했다. 가공 구간 마커, 동결된 reference 및 검증된 좌표·부호 보정이 없기 때문이다.',size=9.7,line=.023)
    body(fig,y,'시간·힘·이동 길이는 작업자의 종료 시점에 영향을 받는다. 전체 실행 길이가 다르므로 누적량이나 평균 힘이 작다는 이유만으로 가공 효율이 좋다고 해석하지 않는다.',size=9.7,line=.023)
    finish(pdf,fig)

    max_duration=max(m['execution_duration_s'] for m in metrics)
    fmin=min(r['f'][(r['ft']>=0)&(r['ft']<=r['m']['execution_duration_s']),2].min() for r in runs)
    fmax=max(m['max_logged_base_fz_N'] for m in metrics)
    xyall=np.concatenate([r['xy'] for r in runs]);lo=xyall.min(axis=0);hi=xyall.max(axis=0)
    fig=page('실측 힘과 TCP 궤적','조건 간 동일 축 범위. XY는 실행별 동결 오염 원점을 뺀 좌표다.')
    for row,c in enumerate('BC'):
        bottom=.56-row*.345;fa=fig.add_axes([.09,bottom,.37,.245]);xa=fig.add_axes([.59,bottom,.34,.245])
        palette=plt.get_cmap('Blues' if c=='B' else 'Oranges')(np.linspace(.45,.9,5))
        for r,color in zip(groups[c],palette):
            m=r['m'];mask=(r['ft']>=0)&(r['ft']<=m['execution_duration_s'])
            fa.plot(r['ft'][mask],r['f'][mask,2],lw=1,color=color,label=m['label'])
            xa.plot(r['xy'][:,0],r['xy'][:,1],lw=1.1,color=color,label=m['label'])
        fa.set(title=c+' · robot-base Fz',xlabel='실행 경과 (s)',ylabel='Fz (N)',xlim=(0,max_duration+.5),ylim=(fmin-3,fmax+5));fa.legend(ncol=3,fontsize=6.5)
        xa.set(title=c+' · 실측 TCP 평면 궤적',xlabel='상대 X (mm)',ylabel='상대 Y (mm)',xlim=(lo[0]-1,hi[0]+1),ylim=(lo[1]-1,hi[1]+1));xa.set_aspect('equal',adjustable='box')
    body(fig,.126,'10회 모두 접근 구간을 포함한다. 시작 위치·종료 시점·시편 조건의 영향을 분리하지 않았고, 파형 또는 궤적 차이를 가공 품질 점수로 바꾸지 않았다.',size=9.4,line=.022)
    finish(pdf,fig)

    fig=page('명령 전송과 접촉 판정','접촉 전환을 제외 사유로 사용하지 않고 모든 실행의 지표로 유지했다.')
    force_lims=(min(r['a'][:,8].min() for r in runs)-2,max(r['a'][:,8].max() for r in runs)+2)
    for col,c in enumerate('BC'):
        ax=fig.add_axes([.09+col*.49,.574,.36,.235])
        palette=plt.get_cmap('Blues' if c=='B' else 'Oranges')(np.linspace(.45,.9,5))
        for r,color in zip(groups[c],palette):ax.plot(r['ct'],r['a'][:,8],color=color,lw=1,label=r['m']['label'])
        ax.set(title=c+' · 발행 Fz 목표',xlabel='실행 경과 (s)',ylabel='controller TCP Fz (N)',xlim=(0,max_duration+.5),ylim=force_lims);ax.legend(ncol=3,fontsize=6.5)
    ax=fig.add_axes([.10,.314,.82,.17]);ax.bar([m['label'] for m in metrics],[m['contact_gate_transitions_total'] for m in metrics],color=[COLORS[m['condition']] for m in metrics]);ax.set(ylabel='판정 전환 횟수',title='전체 실행의 접촉 판정 전환')
    y=heading(fig,.257,'전송 검증 결과')
    maxgap=max(m['max_command_gap_ms'] for m in metrics);maxacc=max(m['max_axis_command_acceleration_mm_s2'] for m in metrics)
    y=body(fig,y,f'명령 ID 연속성, 시간 순서와 유한값 검사를 통과했다. 실행 중 발행 명령의 최대 간격은 {maxgap:.3f} ms이며, 기록된 위치 명령의 최대 축별 가속도는 {maxacc:.3f} mm/s²다. logger drop과 쓰기 오류는 0이다.',size=9.6,line=.022)
    body(fig,y,'발행 힘 목표는 제어기가 실제 적용한 최종 힘 또는 센서 법선 힘을 뜻하지 않는다. 두 조건의 힘 명령은 동일한 실행 처리 과정을 거치며, 파형 차이는 원본 그대로 보존했다.',size=9.6,line=.022)
    finish(pdf,fig)

    for c in 'BC':
        fig=page(c+' 초기·종료 이미지','5회 모두 표시. 이미지로 제거율이나 물리 작업 성공을 자동 판정하지 않았다.')
        fig.text(.29,.836,'초기 기록',ha='center',fontsize=10,color=NAVY,weight='bold');fig.text(.725,.836,'종료 기록',ha='center',fontsize=10,color=NAVY,weight='bold')
        for i,r in enumerate(groups[c]):
            top=.812-i*.145;fig.text(.07,top,r['m']['label']+'  '+r['m']['tag_time'],fontsize=8.4,color=TEXT,weight='bold')
            for left,name in [(.09,'initial_image.png'),(.52,'final_image.png')]:
                ax=fig.add_axes([left,top-.128,.395,.12]);ax.axis('off');ax.imshow(plt.imread(r['base']/'provider/snapshots'/name))
        note='원본 이미지와 WebM 영상은 provider/snapshots/ 및 video/에 보관했다.' if c=='B' else 'C5 초기 이미지도 보존됐다. C5 동영상은 실행 시작 후 약 4.223초부터 기록됐다.'
        body(fig,.069,note,size=8.2,line=.018,color=MUTED);finish(pdf,fig)

    fig=page('기록·영상 검증과 보관 목록','E1과 같은 조건별 데이터·영상 보관 구조를 사용했다.')
    data=[[m['label'],str(m['node_sent_rows']),f"{m['video_duration_s']:.1f}",str(m['video_frames']),f"{m['video_start_delay_s']:.3f}",
           '초반 지연 / 데이터 정상' if m['video_initial_delay_over_1s'] else '기록 정상'] for m in metrics]
    table(fig,['실행','발행 행 수','영상 (s)','프레임','녹화 시작 지연 (s)','검사 결과'],data,.389,.44,[.09,.16,.12,.11,.22,.30],8.1)
    y=heading(fig,.35,'정리 결과와 출처')
    y=body(fig,y,'총 시도는 B 7회·C 5회다. B의 추론 시작 실패 2회는 실행 시작 이벤트와 명령이 없어 삭제했다. 해당 두 시도의 전용 ROS launch 로그를 포함한 105개 파일을 제거했다. 정상 실행을 성능 값으로 선별하지 않았다.',size=9.4,line=.022)
    y=body(fig,y,'원본 실험 파일 810개, WebM 10개, 이미지/그림 60개를 확인했다. 모든 영상은 전체 디코딩에 성공했으나 C5는 실행 초반 영상이 없다. 시각은 녹화 시작 로그 기준이며 프레임별 카메라 촬영 시각과 같지는 않다.',size=9.4,line=.022)
    body(fig,y,'보관 위치: results/20260926/E2/{B,C}/<실행 ID>/\n원본: E2/E2_BC_01/{B,C}/ · 실행 목록: E2/run_index.csv\n파일 해시: E2/manifest.json · 삭제 근거: E2/audit/cleanup_manifest.json\n재현 수치: E2/analysis/metrics.csv 및 metrics.json',size=8.6,units=105,line=.021)
    finish(pdf,fig)

    fig=page('해석 범위와 후속 A 실험','B/C 데이터 묶음은 완료됐으며 A 학습·실행 결과는 포함하지 않는다.')
    y=heading(fig,.823,'확인된 사실')
    y=body(fig,y,'B는 힘 관측 OFF, C는 ON이며 두 조건 모두 학습된 힘 출력을 사용했다. 체크포인트 SHA-256, 동일 실행 계약 및 실행별 설정·소스 스냅샷을 확인했다. 10회 모두 작업자 종료 후 명령 큐 취소와 실제 정지 유지가 기록됐다.',size=10,line=.025)
    y=heading(fig,y-.014,'측정 범위의 한계')
    y=body(fig,y,'processing_start/end 마커가 없다. finish의 processing_end_recorded=true는 현재 processing 상태가 비활성이라는 뜻이므로 실제 가공 시작·종료 마커가 있었다고 해석하지 않았다. 보고서의 실행 시간은 접근을 포함한다.',size=9.8,line=.024)
    y=body(fig,y,'교사 기준 프로파일이 동결되지 않았고 법선/부호 보정도 미검증이다. 원본 Fz 통계는 제공하지만 법선 힘 프로파일 오차는 NA로 남겼다. 최종 controller-applied force도 별도 측정되지 않았다.',size=9.8,line=.024)
    y=body(fig,y,'시편 상태는 매회 동일하다고 가정했다. B 이후 C를 연속 수행했으며 독립 시편 반복·무작위 순서 실험이 아니다. 유의성 검정이나 인과적 모델 우열은 주장하지 않는다.',size=9.8,line=.024)
    y=body(fig,y,'제거량 히트맵은 접촉·속도 기반 대용지표이고 실제 잉크 제거량의 독립 측정이 아니다. 접촉 판정 전환은 모델 동작 지표로 유지했다. 센서 DDS 단계의 손실은 확인할 수 없다.',size=9.8,line=.024)
    y=heading(fig,y-.014,'A 준비 상태')
    body(fig,y,'A 본학습은 사용자가 별도로 시작을 요청했다. A는 motion-only 6차원 출력이며 힘 관측·힘 정답 loss·힘 출력이 없다. 학습 완료 체크포인트와 실행용 일정 힘 F0 확정은 서로 다른 단계다. 이 PDF에는 A 결과를 합성하거나 대신 채워 넣지 않았다.',size=9.8,line=.024)
    finish(pdf,fig)
partial.replace(TARGET)
assert number==9
validation=dict(output=str(TARGET),pages=number,page_titles=titles,bytes=TARGET.stat().st_size,pdf_sha256=sha(TARGET),
    counts_by_condition={'B':5,'C':5},font='Embedded NanumGothic TrueType',text_bounds_checked=True,
    video_initial_delay_C5_s=next(m['video_start_delay_s'] for m in metrics if m['label']=='C5'),
    raw_data_modified=False,metrics_csv_sha256=sha(OUT/'metrics.csv'),metrics_json_sha256=sha(OUT/'metrics.json'))
save(OUT/'validation.json',validation)
for item in archive['files']:assert sha(DATA/item['path'])==item['sha256'],item['path']
print(json.dumps(validation,ensure_ascii=False,indent=2),flush=True)
