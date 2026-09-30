"""Apply the audited, explicitly authorized C-only removal manifest once."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import re
import shutil
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
USER_ROOT=ROOT.parent
MANIFEST=OUT/'cleanup_manifest.json'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify(inventory, exact_stat=False):
    for name,info in inventory.items():
        p=Path(name)
        assert p.is_file() and not p.is_symlink(),p
        s=p.stat()
        assert s.st_size==info['bytes'] and sha(p)==info['sha256'],p
        if exact_stat:assert s.st_ino==info['inode'] and s.st_mtime_ns==info['mtime_ns'],p


m=json.loads(MANIFEST.read_text())
assert m['status']=='prepared_not_deleted',m['status']
bad={g['tag'] for g in m['runs'] if not g['keep']}
assert bad=={'RTC_C_20260926T175117','RTC_C_20260926T175131','RTC_C_20260926T184718'}
good={g['tag'] for g in m['runs'] if g['keep']}
assert len(good)==5
targets=list(map(Path,m['delete_paths']))
assert len(targets)==29
assert set(m['delete_paths'])=={p for g in m['runs'] if not g['keep'] for p in g['paths']}
allowed={ROOT/'logs/inference_metrics',ROOT/'logs/rtc_launch_context',ROOT/'logs/polishing_removal',
         USER_ROOT/'.ros/log',USER_ROOT/'Videos/Screencasts'}
expanded=set()
for p in targets:
    assert p.parent in allowed and p.exists() and not p.is_symlink(),p
    for keep in map(Path,m['keep_paths']+m['protected_R_T_paths']):
        assert p!=keep and p not in keep.parents and keep not in p.parents,(p,keep)
    for f in p.rglob('*') if p.is_dir() else [p]:
        assert not f.is_symlink(),f
        if f.is_file():expanded.add(str(f))
assert expanded==set(m['delete_file_inventory'])
verify(m['delete_file_inventory'],exact_stat=True)
verify(m['retained_file_inventory'])
verify(m['protected_R_T_file_inventory'])
if sys.argv[1:]!=['--apply']:
    print('Verified manifest. Use --apply for its 3 failed runs / 29 exact paths / 109 files.')
    raise SystemExit(0)
for p in targets:
    if p.is_dir():shutil.rmtree(p)
    else:p.unlink()
assert all(not p.exists() for p in targets)
verify(m['retained_file_inventory'])
verify(m['protected_R_T_file_inventory'])
remaining={}
for method in 'RTC':
    ep=sorted((ROOT/'logs/inference_metrics').glob('RTC_'+method+'_20260926T*_executor*'))
    tags={re.match(r'(RTC_[RTC]_20260926T\d{6})_',p.name).group(1) for p in ep}
    assert len(tags)==5,(method,tags)
    if method=='C':assert tags==good
    remaining[method]=sorted(tags)
m.update(status='deleted_and_verified',completed_at=datetime.now().astimezone().isoformat(),
         deleted_files=len(m['delete_file_inventory']),deleted_bytes=sum(v['bytes'] for v in m['delete_file_inventory'].values()),
         retained_files_sha256_verified=len(m['retained_file_inventory']),
         protected_R_T_files_sha256_verified=len(m['protected_R_T_file_inventory']),remaining_runs=remaining)
MANIFEST.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:m[k] for k in ('status','deleted_files','deleted_bytes','retained_files_sha256_verified',
                                  'protected_R_T_files_sha256_verified','remaining_runs')},ensure_ascii=False,indent=2))
