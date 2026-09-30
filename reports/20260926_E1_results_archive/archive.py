"""Copy the 15 audited E1 runs and every associated file; verify SHA-256."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import csv
import hashlib
import json
import re
import shutil
import tempfile

ROOT=Path('/home/eunseop/nrs_imitation')
USER_ROOT=ROOT.parent
REPORT=Path(__file__).resolve().parent
DEST=ROOT/'results/20260926/E1'
ROS=USER_ROOT/'.ros/log'


def js(p): return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def expand(p):
    if p.is_file():return [p]
    result=[]
    for f in sorted(p.rglob('*')):
        assert not f.is_symlink(),f
        if f.is_file():result.append(f)
    return result


audit=js(ROOT/'reports/20260926_C5_log_audit/cleanup_manifest.json')
assert audit['status']=='deleted_and_verified'
expected=dict(audit['retained_file_inventory'],**audit['protected_R_T_file_inventory'])
assert len(expected)==865
assert not DEST.exists(),f'Destination already exists; no overwrite: {DEST}'
assert {k:len(v) for k,v in audit['remaining_runs'].items()}=={'R':5,'T':5,'C':5}
r_audit=js(ROOT/'reports/20260926_R5_log_audit/cleanup_manifest.json')
r_info={g['tag']:g for g in r_audit['runs'] if g['keep']}
t_audit=js(ROOT/'reports/20260926_T5_log_audit/audit.json')
t_info={g['run_tag']:g for g in t_audit['runs']}
c_info={g['tag']:g for g in audit['runs'] if g['keep']}
sources={};records=[]
launches={}
for p in ROS.glob('2026-09-26-*/launch.log'):
    content=p.read_text();match=re.search(r'Run context: (\S+)',content)
    if match:launches[match.group(1)]=(p.parent,content)


def add(source, destination):
    assert source.exists() and not source.is_symlink(),source
    for f in expand(source):
        target=destination/f.relative_to(source) if source.is_dir() else destination
        assert str(f) not in sources,f
        sources[str(f)]=str(target)


for method in 'RTC':
    for number,tag in enumerate(audit['remaining_runs'][method],1):
        base=Path(method)/tag
        candidates=list((ROOT/'logs/inference_metrics').glob(tag+'_*'))
        executors=[p for p in candidates if '_executor_' in p.name]
        providers=[p for p in candidates if '_executor_' not in p.name]
        assert len(executors)==len(providers)==1
        ep,pp=executors[0],providers[0]
        em,pm=js(ep/'metadata.json'),js(pp/'metadata.json')
        context=Path(pm['runtime_parameters']['e2_config'])
        assert context.parent.parent==ROOT/'logs/rtc_launch_context'
        plot_dirs=list((ROOT/'logs/polishing_removal').glob(tag+'_*'))
        videos=list((USER_ROOT/'Videos/Screencasts').glob('*'+tag+'.webm'))
        assert len(plot_dirs)==len(videos)==1
        launch,content=launches[str(context)]
        add(ep,base/'executor');add(pp,base/'provider')
        add(context.parent,base/'launch_context')
        add(plot_dirs[0],base/'plots');add(videos[0],base/'video'/videos[0].name)
        add(launch,base/'ros_logs/launch')
        start=float(content.splitlines()[0].split()[0])
        for pid in re.findall(r'process started with pid \[(\d+)\]',content):
            for p in ROS.glob('*_'+pid+'_*.log'):
                match=re.search(r'_(\d+)\.log$',p.name)
                if match and start-2<=int(match.group(1))/1000<=start+30:
                    add(p,base/'ros_logs/nodes'/p.name)
        ev=[json.loads(line) for line in (ep/'events.jsonl').read_text().splitlines()]
        started=next(e for e in ev if e['event']=='execution_start')
        stops=[e for e in ev if e['event']=='stop_requested' and not e['details'].get('initial_reset')]
        ends=[e for e in ev if e['details'].get('controller_hold_verified') and e['event']!='queue_cancel_ack']
        summary=js(ep/'summary.json')
        node_sent=sum(1 for row in csv.DictReader((ep/'commands.csv').open()) if row['command_stage']=='node_sent')
        status={'R':'recording_complete_return_timeout','T':'recording_complete_endpoint_not_reached',
                'C':'normal_completion_operator_finish_verified_hold'}[method]
        records.append(dict(method=method,repeat_index=number,run_tag=tag,recording_complete=True,
            record_status=status,normal_completion=any(e['event']=='normal_completion' for e in ev),
            controller_hold_verified=bool(ends),termination_reasons=';'.join(e['details']['reason'] for e in stops),
            execution_start_local=datetime.fromtimestamp(started['receipt_ros_ns']/1e9).astimezone().isoformat(timespec='milliseconds'),
            execution_to_first_stop_s=(stops[0]['receipt_monotonic_ns']-started['receipt_monotonic_ns'])/1e9,
            processing_proxy_s=r_info[tag]['processing_proxy']['duration_s'] if method=='R' else None,
            template_duration_s=t_info[tag]['template_duration_s'] if method=='T' else None,
            tcp_rows=summary['counts']['tcp_pose']['written'],force_rows=summary['counts']['wrench']['written'],
            node_sent_rows=node_sent,logger_drops=0,write_errors=0,
            conditioning_profile=(em.get('c_pose_conditioning') or {}).get('profile'),
            config_sha256=em['config_sha256'],session_id=em['session_id'],
            run_directory=str(base),video=str(base/'video'/videos[0].name),
            executor_directory=str(base/'executor'),provider_directory=str(base/'provider')))

assert set(sources)==set(expected),(set(sources)-set(expected),set(expected)-set(sources))
assert len(set(sources.values()))==len(sources)
for name,info in expected.items():
    p=Path(name);assert p.stat().st_size==info['bytes'] and sha(p)==info['sha256'],p
DEST.parent.mkdir(parents=True,exist_ok=True)
staging=Path(tempfile.mkdtemp(prefix='.E1.staging_',dir=DEST.parent))
inventory=[]
for name,target in sorted(sources.items()):
    source=Path(name);dest=staging/target
    dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists(),dest
    shutil.copy2(source,dest)
    actual=sha(dest)
    assert actual==expected[name]['sha256'],dest
    inventory.append(dict(source=name,path=target,bytes=dest.stat().st_size,sha256=actual,kind='run_artifact'))

audit_files={
    'R':('20260926_R5_log_audit',['README.md','retained_runs.csv','cleanup_manifest.json']),
    'T':('20260926_T5_log_audit',['README.md','retained_runs.csv','audit.json']),
    'C':('20260926_C5_log_audit',['README.md','retained_runs.csv','cleanup_manifest.json']),
    'C_patch':('20260926_C_execution_patch',['README.md','validation.json','changes.patch','command_comparison.png','command_comparison.pdf']),
}
for method,(folder,names) in audit_files.items():
    for name in names:
        source=ROOT/'reports'/folder/name;dest=staging/'audit'/method/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
        h=sha(source);assert sha(dest)==h
        inventory.append(dict(source=str(source),path=str(dest.relative_to(staging)),bytes=dest.stat().st_size,sha256=h,kind='audit_reference'))

with (staging/'run_index.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
readme='''2026-09-26 E1 전체 결과
======================

오늘 보존한 R 5회, T 5회, C 5회(총 15회)의 데이터와 영상 전체 복사본이다.
원본 logs 및 Videos/Screencasts는 그대로 유지했다. 비정상 C 3회는 사용자 요청으로 삭제되어 이 묶음에 포함하지 않았다.

각 방법 폴더의 실행 태그별 구성:

```text
R 또는 T 또는 C/
  RTC_<방법>_20260926T<시각>/
    executor/        실행기 CSV·이벤트·요약·소스/설정 스냅샷
    provider/        예측/경로 CSV·계획 NPZ·센서·시작/종료 이미지·소스 스냅샷
    launch_context/ 실행 설정
    plots/          궤적·힘·제거량 그래프와 summary
    video/          원본 WebM 영상
    ros_logs/       해당 실행의 launch 및 노드 로그
```

[실행 목록 CSV](run_index.csv)에 각 실행의 상대 경로, 영상 경로, 종료 상태, 기록 수와 설정 해시를 기록했다.
[파일 목록·SHA-256](manifest.json)은 복사된 모든 원본 파일의 출처와 상대 경로를 연결한다.
감사 결과와 C 패치 설명은 `audit/`에 함께 저장했다. 원본 audit 문서와 원본 metadata 내부의 절대 경로는 출처 보존을 위해 수정하지 않았다.

| 방법 | 보존 횟수 | 기록/종료 상태 |
|---|---:|---|
| [R](R/) | 5 | 가공 구간 기록 완료; 이후 복귀 timeout. normal_completion은 아님 |
| [T](T/) | 5 | 재생과 정지 기록 완료; 최종 목표 도달 미확인 |
| [C](C/) | 5 | 작업자 finish 후 normal_completion 및 정지 유지 확인 |

C에는 35점 궤적 평활·재계획 연결·가속도 제한 패치가 적용됐다. R/T 실행 조건은 오늘 원본을 유지했다.
사용자는 C 5회에서 의도하지 않은 동작이 없었다고 확인했다. 접촉 판정 전환은 분석 지표로 보존했다.
물리 가공 품질·실험 영역의 독립성·순수 가공 시간은 이 파일 정리 작업에서 새로 판정하지 않았다.
영상과 기록은 성공 여부를 선택적으로 보정하거나 편집하지 않은 원본 복사본이다.
'''
(staging/'README.md').write_text(readme)
for name in ('README.md','run_index.csv'):
    p=staging/name
    inventory.append(dict(source=None,path=name,bytes=p.stat().st_size,sha256=sha(p),kind='generated_index'))
for name,info in expected.items():assert sha(Path(name))==info['sha256'],name
assert sum(x['path'].endswith('.webm') for x in inventory)==15
assert all(len(list((staging/m).iterdir()))==5 for m in 'RTC')
assert {str(p.relative_to(staging)) for p in staging.rglob('*') if p.is_file()}=={e['path'] for e in inventory}
manifest=dict(created_at=datetime.now().astimezone().isoformat(),destination=str(DEST),
    operation='copy; originals retained',experiment='E1',date='20260926',
    runs_by_method=dict(Counter(r['method'] for r in records)),runs=records,
    source_run_files=865,original_videos=15,source_and_destination_sha256_verified=True,
    bytes_copied=sum(e['bytes'] for e in inventory if e['source']),
    inventory_excludes_self=True,files=inventory)
(staging/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
assert not DEST.exists(),DEST
staging.rename(DEST)
for entry in inventory:
    p=DEST/entry['path'];assert sha(p)==entry['sha256'],p
validation=dict(completed_at=datetime.now().astimezone().isoformat(),destination=str(DEST),
    runs_by_method=dict(Counter(r['method'] for r in records)),run_artifact_files=865,videos=15,
    archive_files_including_manifest=len(inventory)+1,bytes_copied=manifest['bytes_copied'],
    all_source_files_unchanged=True,all_destination_hashes_verified=True,
    manifest_sha256=sha(DEST/'manifest.json'),invalid_C_runs_excluded=3)
(REPORT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(validation,ensure_ascii=False,indent=2))
