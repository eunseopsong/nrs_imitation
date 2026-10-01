"""Archive the requested 2026-09-30 E2 cohort without modifying its source logs."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess
import sys

ROOT = Path('/home/eunseop/nrs_imitation')
DATE = '20260930'
SOURCE = ROOT / f'results/{DATE}/E2/E2_ABC_{DATE}'
DATA = ROOT / f'results/{DATE}/E2'
TZ = ZoneInfo('Asia/Seoul')
sys.path.insert(0, str(ROOT / 'scripts'))
from check_inference_log import inspect


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def json_file(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert read(path) == value, f'Existing generated file differs: {path}'
    else:
        path.write_text(text)


def logger_path(attempt, branch):
    matches = list((attempt / branch).glob('*/metadata.json'))
    assert len(matches) == 1, (attempt, branch, matches)
    return matches[0].parent


def event_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def inspect_attempt(attempt):
    config = read(attempt / 'runtime.json')
    events = event_rows(logger_path(attempt, 'executor') / 'events.jsonl')
    start = next((e for e in events if e['event'] == 'execution_start'), None)
    stop = next((e for e in events if e['event'] == 'stop_requested'
                 and not e['details'].get('initial_reset')), None)
    terminal = next((e for e in reversed(events) if e['event'] in
                     ('normal_completion', 'safety_stop', 'manual_abort', 'trajectory_end')), None)
    return config, start, stop, terminal


def main():
    assert SOURCE.is_dir()
    all_attempts = [inspect_attempt(d) + (d,) for c in 'ABC'
                    for d in sorted((SOURCE / c).iterdir()) if d.is_dir()]
    selected, excluded = [], []
    for condition in 'ABC':
        for repeat in range(1, 6):
            candidates = [r for r in all_attempts if r[0]['condition'] == condition
                          and r[0]['run']['repeat_index'] == repeat]
            assert candidates, (condition, repeat)
            # A4 was explicitly rejected by the operator and then repeated.
            # The archive preserves that earlier attempt outside the 15-run set.
            if len(candidates) > 1:
                assert condition == 'A' and repeat == 4
            choice = max(candidates, key=lambda r: r[-1].name)
            assert choice[1] and choice[2] and choice[3]
            assert choice[2]['details']['reason'] == 'operator_finish'
            assert choice[3]['event'] == 'normal_completion'
            assert choice[3]['details']['controller_hold_verified'] is True
            assert choice[3]['details']['queue_cancel_verified'] is True
            selected.append(choice)
            excluded.extend(r for r in candidates if r is not choice)
    assert len(selected) == 15 and len(excluded) == 1
    source_files, generated_files, records, excluded_records = [], [], [], []

    def record_generated(path):
        generated_files.append(dict(path=str(path.relative_to(DATA)), bytes=path.stat().st_size,
                                    sha256=sha(path), kind='derived_archive_record'))

    def copy_one(src, dst):
        assert src.is_file() and not src.is_symlink(), src
        digest = sha(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            assert sha(dst) == digest, f'Refusing to overwrite different data: {dst}'
        else:
            shutil.copy2(src, dst)
        assert dst.stat().st_size == src.stat().st_size and sha(dst) == digest
        source_files.append(dict(path=str(dst.relative_to(DATA)), source=str(src),
                                 bytes=dst.stat().st_size, sha256=digest, kind='source_copy'))

    for config, start, stop, terminal, original in selected + excluded:
        is_selected = any(original == r[-1] for r in selected)
        condition, repeat = config['condition'], config['run']['repeat_index']
        folder = DATA / condition / original.name if is_selected else (
            DATA / 'archive/excluded_attempts' / condition / original.name)
        branch_roots = {branch: logger_path(original, branch) for branch in ('provider', 'executor')}
        plot_summaries = list((original / 'plots').rglob('summary.txt'))
        assert len(plot_summaries) == 1
        branch_roots['plots'] = plot_summaries[0].parent
        for src in sorted(original.rglob('*')):
            if not src.is_file():
                continue
            dest_rel = src.relative_to(original)
            for branch, branch_root in branch_roots.items():
                if src.is_relative_to(branch_root):
                    dest_rel = Path(branch) / src.relative_to(branch_root)
                    break
            copy_one(src, folder / dest_rel)
        (folder / 'video').mkdir(exist_ok=True)
        audit_counts = {}
        for branch in ('executor', 'provider'):
            log = folder / branch
            audit = inspect(log)
            assert not (audit['issues'] or audit['run_id_mismatches'] or audit['malformed_event_lines'])
            assert audit['queue_pending'] == 0 and not audit['logger_write_errors']
            summary = read(log / 'summary.json')
            assert summary['drained'] and summary['write_error_count'] == 0
            assert all(v['dropped'] == 0 for v in summary['counts'].values())
            for stream in audit['streams'].values():
                assert not (stream['nonfinite'] or stream['missing_required_values']
                            or stream['monotonic_retrograde'] or stream['logger_seq_retrograde'])
            for name in ('commands', 'wrench', 'tcp_pose', 'legacy'):
                with (log / f'{name}.csv').open(newline='') as f:
                    actual = sum(1 for _ in csv.DictReader(f))
                assert actual == summary['counts'].get(name, {}).get('written', 0), (log, name)
                audit_counts[f'{branch}_{name}_rows'] = actual
            for artifact in read(log / 'metadata.json')['artifacts']:
                if artifact.get('copy'):
                    p = log / artifact['copy']
                    assert p.is_file() and sha(p) == artifact['sha256']
                    assert p.stat().st_size == artifact['bytes']
            json_file(log / 'log_audit.json', audit)
            record_generated(log / 'log_audit.json')
        videos = []
        for video in sorted((folder / 'video').glob('*.webm')):
            probe = subprocess.run(['ffprobe', '-v', 'warning', '-count_frames',
                '-show_entries', 'stream=nb_read_frames,r_frame_rate:format=duration,size',
                '-of', 'json', str(video)], capture_output=True, text=True)
            videos.append(dict(path=str(video.relative_to(folder)), returncode=probe.returncode,
                               warnings=probe.stderr.strip(), probe=json.loads(probe.stdout)))
        json_file(folder / 'video_checks.json', dict(videos=videos,
                  status='present' if videos else 'not_found_in_source'))
        record_generated(folder / 'video_checks.json')
        identity = dict(condition=condition, repeat_index=repeat, selected_for_analysis=is_selected,
            original_attempt_id=original.name, original_directory=str(original),
            archive_directory=str(folder), runtime_sha256=sha(folder / 'runtime.json'),
            original_logger_directories={b: str(branch_roots[b]) for b in ('executor', 'provider')},
            layout='single logger/plot subdirectory flattened; original file contents unchanged',
            selection_reason='operator-completed cohort; A4 uses requested retry' if is_selected else
                             'operator rejected original A4; manually aborted before execution_start')
        json_file(folder / 'archive_identity.json', identity)
        record_generated(folder / 'archive_identity.json')
        rec = dict(label=f'{condition}{repeat}', condition=condition, repeat_index=repeat,
            archive_run_id=original.name, run_directory=str(folder.relative_to(DATA)),
            execution_start_local=datetime.fromtimestamp(start['receipt_ros_ns']/1e9, TZ).isoformat() if start else None,
            execution_to_first_stop_s=(stop['receipt_monotonic_ns']-start['receipt_monotonic_ns'])/1e9 if start and stop else None,
            first_stop_reason=stop['details']['reason'] if stop else None,
            terminal_event=terminal['event'] if terminal else None,
            controller_hold_verified=terminal['details'].get('controller_hold_verified') if terminal else None,
            queue_cancel_verified=terminal['details'].get('queue_cancel_verified') if terminal else None,
            runtime_sha256=identity['runtime_sha256'], video_count=len(videos),
            video_probe_ok=all(v['returncode'] == 0 and not v['warnings'] for v in videos) if videos else None,
            **audit_counts)
        (records if is_selected else excluded_records).append(rec)
    copy_one(SOURCE / 'comparison.json', DATA / 'comparison.json')
    selection = dict(cohort=SOURCE.name, selected_runs=records, excluded_runs=excluded_records,
        previously_deleted_attempts=[
            dict(attempt_id='E2_ABC_20260930_C_r1_20260930T180515_46b2318333b3', status='deleted_at_operator_request'),
            dict(attempt_id='E2_ABC_20260930_C_r2_20260930T180603_bc10faa9c21f', status='deleted_at_operator_request')],
        labels='Original repeat_index preserved; C1/C2 were recorded after B; A4 retry after A5.',
        scope='Descriptive summary of 15 selected completed records; excluded/deleted attempts are not a success-rate denominator.')
    json_file(DATA / 'selection.json', selection)
    record_generated(DATA / 'selection.json')
    with (DATA / 'run_index.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    record_generated(DATA / 'run_index.csv')
    manifest_path = DATA / 'manifest.json'
    created = read(manifest_path)['created_at'] if manifest_path.exists() else datetime.now(TZ).isoformat()
    manifest = dict(schema='E2_result_archive_v1', created_at=created, experiment_id='E2', date=DATE,
        source_cohort=str(SOURCE), original_logs_preserved=True,
        selected_count=15, excluded_count=len(excluded), runs=records, excluded_runs=excluded_records,
        files=source_files + generated_files,
        source_copied_file_count=len(source_files), generated_file_count=len(generated_files))
    json_file(manifest_path, manifest)
    print(json.dumps(dict(output=str(DATA), selected=len(records), excluded=len(excluded_records),
        verified_files=len(manifest['files']), source_files=len(source_files),
        conditions={c: sum(r['condition'] == c for r in records) for c in 'ABC'},
        videos={c: sum(r['video_count'] for r in records if r['condition'] == c) for c in 'ABC'}),
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
