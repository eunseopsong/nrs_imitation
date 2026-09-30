"""Audit E2 B/C, remove the two authorized failed attempts, and copy good runs."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys

from PIL import Image

ROOT = Path('/home/eunseop/nrs_imitation')
OUT = Path(__file__).resolve().parent
DEST = ROOT / 'results/20260926/E2'
SESSION = DEST / 'E2_BC_01'
BAD = {'E2_B_20260926T205231_1963a9a57021', 'E2_B_20260926T205257_dfbae5c8ec66'}
E1 = (ROOT/'results/20260926/E1/manifest.json',
      ROOT/'results/20260926/E1_final_report_20260926.pdf')


def read(p): return json.loads(p.read_text())
def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()
def events(p): return [json.loads(x) for x in p.read_text().splitlines()]


def video_check(p):
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_read_frames:format=duration,size',
        '-of', 'json', str(p)], capture_output=True, text=True, check=True)
    decode = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(p), '-map', '0:v:0',
        '-f', 'null', '-'], capture_output=True, text=True, check=True)
    assert not decode.stderr, (p, decode.stderr)
    return dict(path=str(p), probe= json.loads(probe.stdout), full_decode_passed=True)


def audit_run(folder):
    a, cfg = read(folder/'attempt.json'), read(folder/'config.json')
    m = read(folder/'archive_manifest.json')
    inventory = {str(folder/x['path']):dict(bytes=x['bytes'], sha256=x['sha256']) for x in m['files']}
    inventory[str(folder/'archive_manifest.json')] = dict(bytes=(folder/'archive_manifest.json').stat().st_size,
                                                         sha256=sha(folder/'archive_manifest.json'))
    actual = {str(p) for p in folder.rglob('*') if p.is_file()}
    assert actual == set(inventory), folder
    for name, item in inventory.items():
        p=Path(name)
        assert not p.is_symlink() and p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], p
    assert a['closed_at'] and a['status']=='launch_exited' and not a['archive_errors'], a
    roles={}
    for role in ('executor', 'provider'):
        paths=list((folder/role).glob('*/events.jsonl'))
        if not paths: continue
        assert len(paths)==1
        ep=paths[0].parent; ev=events(paths[0]); meta=read(ep/'metadata.json'); s=read(ep/'summary.json')
        assert s['drained'] and s['queue_pending']==s['write_error_count']==0
        assert not s['write_errors'] and all(x['dropped']==0 for x in s['counts'].values())
        assert len(ev)==s['counts']['events']['written']
        tables={}; sent=[]
        for name, keys in [('commands', ['x','y','z','rx','ry','rz','fx','fy','fz']),
                           ('tcp_pose', ['x','y','z','rx','ry','rz']), ('wrench', ['fx','fy','fz','tx','ty','tz'])]:
            with (ep/(name+'.csv')).open() as f: rr=list(csv.DictReader(f))
            assert len(rr)==s['counts'].get(name,{}).get('written',0)
            last=-1
            for row in rr:
                assert None not in row and all(v is not None for v in row.values())
                assert row['run_id']==meta['run_id'] and row['validity']=='valid'
                assert all(math.isfinite(float(row[k])) for k in keys if row[k])
                t=int(row['receipt_monotonic_ns']); assert t>=last;last=t
            tables[name]=dict(rows=len(rr), first_ns=int(rr[0]['receipt_monotonic_ns']) if rr else None,
                             last_ns=int(rr[-1]['receipt_monotonic_ns']) if rr else None)
            if name=='commands':
                tables[name]['stages']=dict(Counter(x['command_stage'] for x in rr))
                sent=[x for x in rr if x['command_stage']=='node_sent']
        counts=Counter(x['event'] for x in ev)
        info=dict(directory=str(ep), event_counts=dict(counts), tables=tables,
                  logger_drops=0, write_errors=0, drained=True)
        if role=='executor':
            start=[e for e in ev if e['event']=='execution_start']
            stop=[e for e in ev if e['event']=='stop_requested' and not e['details'].get('initial_reset')]
            complete=[e for e in ev if e['event']=='normal_completion']
            info.update(executed=bool(start), completed=bool(complete), stop_reasons=[e['details']['reason'] for e in stop],
                normal_completion_details=[e['details'] for e in complete],
                elapsed_until_stop_s=(stop[0]['receipt_monotonic_ns']-start[0]['receipt_monotonic_ns'])/1e9 if start and stop else None)
            details=[json.loads(x['details']) for x in sent]
            assert [int(x['command_id']) for x in sent]==list(range(1,len(sent)+1))
            info['contact_gate_transitions']=sum(bool(d['gate_transition']) for d in details)
            info['max_pose_age_s']=max((d['pose_age_s'] for d in details),default=None)
            info['max_force_age_s']=max((d['force_age_s'] for d in details),default=None)
        else:
            enabled=a['condition']=='C'
            assert meta['force_observation']==('ON' if enabled else 'OFF')
            assert meta['runtime_parameters']['use_force_observation'] is enabled
            assert cfg['il']['use_force_observation'] is enabled and cfg['il']['force_action'] is True
            assert meta['predicted_force']=='policy' and len(meta['action_schema'])==9
            assert meta['checkpoint']['path']==cfg['il']['checkpoint']
            assert meta['execution_contract_hash']==cfg['execution_contract_hash']
            info.update(force_observation=meta['force_observation'], force_action=True,
                        checkpoint=meta['checkpoint']['path'], checkpoint_sha256=cfg['il']['checkpoint_sha256'])
        roles[role]=info
    normal=roles['executor']['completed']
    if normal:
        e=roles['executor'];assert e['executed'] and e['stop_reasons']==['operator_finish']
        assert e['event_counts']['physical_hold_verified']==1
        assert all(d['controller_hold_verified'] and d['queue_cancel_verified'] for d in e['normal_completion_details'])
        assert all(p['exit_code']==0 for p in a['launch_processes'])
        videos=list((folder/'video').glob('*.webm'));assert len(videos)==1
        pngs=list(folder.rglob('*.png'));assert len(pngs)==6
        for png in pngs:
            with Image.open(png) as im: im.verify()
    else:
        assert folder.name in BAD and not roles['executor']['executed']
        assert roles['executor']['tables']['commands']['rows']==0
        assert any(p['name'].startswith('inference_single_cam') and p['exit_code']==1 for p in a['launch_processes'])
        videos=[]
    return dict(run_id=folder.name, condition=a['condition'], normal=normal, created_at=a['created_at'],
        source=str(folder), cohort_id=cfg['cohort_id'], execution_contract_hash=cfg['execution_contract_hash'],
        parent_launch_log_source=a['parent_launch_log_source'], launch_processes=a['launch_processes'],
        roles=roles, videos=list(map(str,videos)), inventory=inventory)


def main():
    assert sys.argv[1:]==['--apply'], 'Explicit --apply required for the already-authorized deletion.'
    assert not (DEST/'B').exists() and not (DEST/'C').exists(), 'Archive exists; never overwrite.'
    # E1 checks compare the actual before/after hashes.
    e1_before={str(p):sha(p) for p in E1}
    runs=[audit_run(p.parent) for c in 'BC' for p in sorted((SESSION/c).glob('*/attempt.json'))]
    good=[r for r in runs if r['normal']];bad=[r for r in runs if not r['normal']]
    assert Counter(r['condition'] for r in good)=={'B':5,'C':5}
    assert {r['run_id'] for r in bad}==BAD
    assert len({r['cohort_id'] for r in good})==len({r['execution_contract_hash'] for r in good})==1
    model_hashes={}
    for c in 'BC':
        p=next(r for r in good if r['condition']==c)['roles']['provider']
        model_hashes[c]=sha(Path(p['checkpoint']));assert model_hashes[c]==p['checkpoint_sha256']
    with ThreadPoolExecutor(max_workers=3) as pool:
        videos=list(pool.map(video_check,[Path(r['videos'][0]) for r in good]))
    video_by_path={v['path']:v for v in videos}
    for r in good:r['video_validation']=video_by_path[r['videos'][0]]
    delete=[];delete_inventory={}
    for r in bad:
        folder=Path(r['source']);parent=Path(r['parent_launch_log_source'])
        assert folder.parent==SESSION/'B' and parent.parent==Path('/home/eunseop/.ros/log')
        assert not parent.is_symlink() and {x.name for x in parent.iterdir()}=={'launch.log'}
        assert sha(parent/'launch.log')==sha(folder/'ros_logs/launch_parent/launch.log')
        delete.extend([folder,parent]);delete_inventory.update(r['inventory'])
        delete_inventory[str(parent/'launch.log')]=dict(bytes=(parent/'launch.log').stat().st_size,sha256=sha(parent/'launch.log'))
    audit=dict(schema='E2_BC_cleanup_archive_v1', created_at=datetime.now().astimezone().isoformat(),
        status='audited_before_deletion', original_attempts={'B':7,'C':5}, retained={'B':5,'C':5},
        removed={'B':2,'C':0}, deletion_authorization=['B 5번 실행했어. 로그 확인해주고 비정상 로그는 제거해줘',
        'C 도 5번 실행 마쳤어. 마찬가지로 비정상 로그 제거해주고, 끝났으면 B/C 에 대해서 E1 과 같은 방식으로 정리 진행해줘'],
        removal_reason='Two B inference startup failures; no execution_start and zero executor commands. No completed trial discarded.',
        normal_definition='execution_start + operator_finish + normal_completion + queue cancellation + fresh stationary hold + intact data/video',
        notes=['Contact gate transitions retained as model behavior, not an exclusion criterion.',
               'Provider KeyboardInterrupt is launch shutdown; executor normal completion is verified separately.',
               'Processing markers absent; calibrated profile metrics remain NA. Physical removal quality not inferred.',
               'Logger drop counts are zero; upstream DDS loss is unknown.'],
        delete_paths=list(map(str,delete)), delete_file_inventory=delete_inventory, runs=runs,
        model_hashes_verified=model_hashes, e1_before=e1_before)
    write(OUT/'cleanup_manifest.json',audit)
    for p in delete:shutil.rmtree(p)
    assert all(not p.exists() for p in delete)
    assert all(len(list((SESSION/c).glob('*/attempt.json')))==5 for c in 'BC')
    audit.update(status='deleted_and_verified', completed_at=datetime.now().astimezone().isoformat(),
                 deleted_files=len(delete_inventory),deleted_bytes=sum(x['bytes'] for x in delete_inventory.values()))
    write(OUT/'cleanup_manifest.json',audit)
    old_b=SESSION/'B_log_audit_20260926.json'
    if old_b.exists():
        b=read(old_b);b.update(cleanup_status='deleted_and_verified',cleanup_manifest=str(OUT/'cleanup_manifest.json'))
        write(old_b,b)
    inventory=[];records=[]
    for c in 'BC':
        for i,r in enumerate([x for x in good if x['condition']==c],1):
            src=Path(r['source']);base=DEST/c/r['run_id']
            base.mkdir(parents=True,exist_ok=False)
            for name,item in r['inventory'].items():
                f=Path(name);rel=f.relative_to(src);parts=rel.parts
                if parts[0] in ('executor','provider','plots'):
                    dest_rel=Path(parts[0],*parts[2:])
                elif parts[0] in ('video','ros_logs','analysis'):dest_rel=rel
                else:dest_rel=Path('launch_context')/rel
                to=base/dest_rel;to.parent.mkdir(parents=True,exist_ok=True)
                assert not to.exists();assert sha(f)==item['sha256'];shutil.copy2(f,to);assert sha(to)==item['sha256']
                inventory.append(dict(source=str(f),path=str(to.relative_to(DEST)),**item))
            e=r['roles']['executor'];v=r['video_validation'];stream=v['probe']['streams'][0]
            records.append(dict(condition=c,repeat_index=i,run_id=r['run_id'],normal_completion=True,
                controller_hold_verified=True,termination='operator_finish',elapsed_until_stop_s=e['elapsed_until_stop_s'],
                tcp_rows=e['tables']['tcp_pose']['rows'],force_rows=e['tables']['wrench']['rows'],
                node_sent_rows=e['tables']['commands']['stages']['node_sent'],contact_gate_transitions=e['contact_gate_transitions'],
                logger_drops=0,write_errors=0,video_duration_s=float(v['probe']['format']['duration']),
                video_frames=int(stream['nb_read_frames']),video_full_decode_passed=True,
                cohort_id=r['cohort_id'],execution_contract_hash=r['execution_contract_hash'],
                run_directory=str(base.relative_to(DEST)),video=str((base/'video'/Path(r['videos'][0]).name).relative_to(DEST))))
    audit_dest=DEST/'audit';audit_dest.mkdir(exist_ok=True)
    shutil.copy2(OUT/'cleanup_manifest.json',audit_dest/'cleanup_manifest.json')
    with (DEST/'run_index.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (DEST/'README.md').write_text('''# 2026-09-26 E2 B/C 결과

B(힘 관측 OFF) 5회, C(힘 관측 ON) 5회, 총 10회의 정상 실행 기록과 원본 영상이다.
두 조건 모두 학습된 힘 출력을 사용하며 E1 C 기반의 동일한 실행·평활·정지 로직을 공유한다.
A는 아직 실행하지 않았으며 이 결과에 포함하지 않았다.

- `B/<실행 ID>/`, `C/<실행 ID>/`: E1과 동일한 조건별 보관 구조
- `executor/`, `provider/`: 명령·센서 CSV, 이벤트, 원본 메타데이터와 이미지
- `launch_context/`: 실행 설정, 시도 정보, 코드 스냅샷, 원본 archive_manifest
- `plots/`, `video/`, `ros_logs/`: 그래프·원본 WebM·관련 ROS 로그
- `run_index.csv`: 10회 목록, 종료 상태, 데이터/영상 경로
- `manifest.json`: 원본 파일 출처·크기·SHA-256 및 보관본 무결성
- `audit/cleanup_manifest.json`: 시도 12회 중 정상 10회 보존, 시작 실패 B 2회 삭제 근거
- `analysis/`: 실행 구간 수치 CSV/JSON 및 보고서 검증
- 상위 날짜 폴더의 `E2_BC_final_report_20260926.pdf`: 단일 통합 보고서

정상 원본은 `E2_BC_01/{B,C}/`에도 유지했다. 원본 파일 내부의 절대 경로는 출처 보존을 위해 수정하지 않았다.
비정상 B 두 시도의 원본 폴더와 해당 전용 ROS launch 로그 폴더는 사용자 요청에 따라 삭제했다.
삭제한 원시 로그는 보관하지 않았고 삭제 이유와 파일 해시 목록만 audit에 남겼다. C에는 비정상 시도가 없다.

10회 모두 operator_finish, normal_completion 및 실제 정지 유지 확인이 있다.
접촉 판정 전환 횟수만으로 실행을 제외하지 않았다. 실행 기록 정상과 물리 가공 품질은 별개다.
processing_start/end 마커가 없어 순수 가공 시간은 미확정이며, 실행 시간은 접근을 포함한다.
힘은 robot-base Fz의 기술 통계이며, 미검증 법선 힘·교사 기준 프로파일·제어기 적용 힘 오차는 NA다.
시편 상태는 사용자 지시에 따라 매회 동일하다고 가정했다. B 5회 뒤 C 5회가 수행되어 순서 효과는 분리할 수 없다.
''')
    for p in [audit_dest/'cleanup_manifest.json',DEST/'run_index.csv',DEST/'README.md']:
        inventory.append(dict(source=None,path=str(p.relative_to(DEST)),bytes=p.stat().st_size,sha256=sha(p)))
    manifest=dict(created_at=datetime.now().astimezone().isoformat(),experiment='E2',conditions=['B','C'],
        runs_by_condition={'B':5,'C':5},videos=10,runs=records,source_run_files=sum(len(r['inventory']) for r in good),
        operation='copy verified normal originals; delete two explicitly authorized failed B attempts',
        source_session=str(SESSION),files=inventory,inventory_excludes_self=True,
        inventory_scope='B/, C/, audit/, README.md, run_index.csv; source session not duplicated in this manifest')
    write(DEST/'manifest.json',manifest)
    assert {str(p):sha(p) for p in E1}==e1_before
    for r in good:
        for p,item in r['inventory'].items():assert sha(Path(p))==item['sha256']
    for item in inventory:assert sha(DEST/item['path'])==item['sha256']
    validation=dict(status='complete',retained={'B':5,'C':5},deleted={'B':2,'C':0},deleted_files=len(delete_inventory),
        source_run_files=manifest['source_run_files'],videos=10,images_verified=60,all_source_and_copy_hashes_match=True,
        all_normal_originals_unchanged=True,e1_unchanged=True,manifest_sha256=sha(DEST/'manifest.json'))
    write(OUT/'validation.json',validation);print(json.dumps(validation,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
