"""Run-local E2 artifacts shared by the CLI and direct ROS launch entry."""
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import uuid

from .e2_ablation import ROOT, digest, select_condition, execution_contract


def read(p):return json.loads(Path(p).read_text())


def write(p,obj,exclusive=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    else:
        tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',value):raise ValueError('Use a short identifier without path separators')
    return value


def session_root(session):
    return ROOT/'results'/datetime.now().strftime('%Y%m%d')/'E2'/identifier(session)


def create_attempt(cfg,condition,session,block=None,specimen=None,region=None,reuse=None,*,session_path=None):
    specimen=specimen or 'unspecified';region=region or 'unspecified'
    assumption=cfg['protocol'].get('specimen_state_assumption',{})
    reuse=reuse or ('assumed_same_state' if assumption.get('mode')=='same_state_each_run' else 'unknown')
    identifier(specimen);identifier(region)
    if block is not None and not 1<=block<=cfg['protocol']['scheduled_blocks']:
        raise ValueError('block outside predeclared schedule')
    run_id='E2_'+condition+'_'+datetime.now().strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:12]
    folder=(Path(session_path) if session_path is not None else session_root(session))/condition/run_id
    folder.mkdir(parents=True,exist_ok=False)
    selected=select_condition(cfg,condition)
    selected['run']=dict(run_id=run_id,session_id=session,condition=condition,task_id=cfg['protocol']['task_id'],
        specimen_id=specimen,region_id=region,block_id=block,repeat_id=run_id,attempt_id=run_id,
        surface_reuse=reuse,surface_process=cfg['protocol']['surface_process'],directory=str(folder),
        specimen_state_assumption=selected['protocol'].get('specimen_state_assumption'),
        created_at=datetime.now().astimezone().isoformat(),cohort_id=cfg['cohort_id'])
    if 'trial_stage' in selected:
        selected['run'].update(trial_stage=selected['trial_stage'],
            quality_validation=selected.get('pilot', {}).get('quality_validation'),
            F0_frozen=selected.get('f0', {}).get('status') == 'frozen')
    selected['common'].update(specimen_id=specimen,region_id=region,block_id=block,repeat_id=run_id,
                              specimen_state_assumption=selected['run']['specimen_state_assumption'])
    snapshot_artifacts(selected,folder)
    write(folder/'config.json',selected,exclusive=True)
    write(folder/'attempt.json',dict(**selected['run'],experiment_id='E2',status='created',executed=False,
                                   completed=None,excluded=False,exclusion_reason=None),exclusive=True)
    return selected,folder


def snapshot_artifacts(cfg,folder):
    """Small run-local evidence; checkpoints stay pinned by their original hash."""
    inventory=[]
    def copy_file(source,relative,expected=None):
        source=Path(source);target=folder/'artifacts'/relative
        if not source.is_file():
            inventory.append(dict(source=str(source),status='missing',expected_sha256=expected));return None
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        actual=digest(target)
        inventory.append(dict(source=str(source),path=str(target.relative_to(folder)),sha256=actual,
                              expected_sha256=expected,status='copied' if expected in (None,actual) else 'hash_mismatch'))
        return str(target)
    for name,sha in execution_contract(cfg)['code_sha256'].items():copy_file(ROOT/name,Path('code')/name,sha)
    for source,sha in cfg['common'].get('expected_controller_sources',{}).items():
        copy_file(source,Path('controller')/Path(source).name,sha)
    for source,sha in cfg.get('pilot',{}).get('required_driver_sources',{}).items():
        copy_file(source,Path('pilot_driver')/Path(source).name,sha)
    for key,pathkey,hashkey in [('reference','artifact','sha256'),('f0','candidate','candidate_sha256')]:
        info=cfg[key]
        if info.get(pathkey):
            local=copy_file(info[pathkey],key+'.json',info.get(hashkey))
            if local:info[pathkey]=local
    commissioning = cfg.get('pilot', {}).get('commissioning_record')
    if commissioning and commissioning.get('path'):
        local = copy_file(commissioning['path'], 'commissioning.json', commissioning.get('sha256'))
        if local:
            # Keep referenced evidence too, without rewriting the signed record.
            try:
                record = read(local)
                for key, ref in record.items():
                    if key.endswith('_evidence') and isinstance(ref, dict) and ref.get('path'):
                        copy_file(ref['path'], Path('commissioning_evidence')/key/Path(ref['path']).name,
                                  ref.get('sha256'))
            except (OSError, ValueError):
                pass  # Preflight reports invalid evidence; failed attempts survive.
            commissioning['path'] = local
    model=cfg['models'][cfg['condition']]
    if model.get('checkpoint'):
        copy_file(Path(model['checkpoint']).with_name('dataset_stats.pkl'),'dataset_stats.pkl',model.get('normalizer_sha256'))
    write(folder/'artifacts/calibration.json',cfg['calibration'],exclusive=True)
    write(folder/'artifacts/execution_contract.json',execution_contract(cfg),exclusive=True)
    write(folder/'artifacts/index.json',dict(files=inventory,checkpoint_identity=model),exclusive=True)


def seal_archive(folder):
    """Inventory only; never delete, rename, or overwrite source artifacts."""
    folder=Path(folder).resolve();files=[]
    for p in sorted(folder.rglob('*')):
        if p.is_file() and p.name not in ('archive_manifest.json','archive_manifest.json.tmp'):
            files.append(dict(path=str(p.relative_to(folder)),bytes=p.stat().st_size,sha256=digest(p)))
    write(folder/'archive_manifest.json',dict(schema='E2_all_attempt_files_v1',files=files,
          created_at=datetime.now().astimezone().isoformat(),originals_deleted=False))
    return len(files)


class LaunchRunRecord:
    """Launch-process bookkeeping only; never sends robot commands or stops."""
    def __init__(self, folder, launch_log_directory):
        self.folder=Path(folder)
        self.launch_logs=Path(launch_log_directory)
        self.closed=False

    def update(self, **changes):
        record=read(self.folder/'attempt.json');record.update(changes)
        write(self.folder/'attempt.json',record)

    def process_started(self, event, context):
        record=read(self.folder/'attempt.json')
        processes=record.get('launch_processes',[])
        processes.append(dict(pid=event.pid,name=event.process_name,exit_code=None))
        self.update(launch_processes=processes,status='launch_running',executed=None)

    def process_exited(self, event, context):
        record=read(self.folder/'attempt.json')
        processes=record.get('launch_processes',[])
        for process in processes:
            if process['pid']==event.pid:process['exit_code']=event.returncode
        self.update(launch_processes=processes)

    def shutdown_requested(self, event, context):
        self.update(launch_shutdown_reason=str(event.reason))

    def close(self):
        # Registered with atexit: ROS launch has finished its existing child
        # shutdown/flush before copying the parent launch log and hashing files.
        if self.closed:return
        self.closed=True
        record=read(self.folder/'attempt.json');errors=[]
        try:
            if self.launch_logs.is_dir() and not self.launch_logs.resolve().is_relative_to(self.folder.resolve()):
                for p in self.launch_logs.rglob('*'):
                    if p.is_file() and not p.is_symlink():
                        target=self.folder/'ros_logs/launch_parent'/p.relative_to(self.launch_logs)
                        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        except OSError as exc:errors.append('parent launch log copy: '+str(exc))
        if record.get('status')!='preflight_blocked':
            record.update(status='launch_exited',executed=None if record.get('launch_processes') else False,
                          completed=None)
        record.update(closed_at=datetime.now().astimezone().isoformat(),
                      parent_launch_log_source=str(self.launch_logs),archive_errors=errors)
        write(self.folder/'attempt.json',record)
        try:seal_archive(self.folder)
        except OSError as exc:
            record['archive_errors'].append(str(exc));write(self.folder/'attempt.json',record)
            print('E2 archive incomplete; original files retained at',self.folder,flush=True)
