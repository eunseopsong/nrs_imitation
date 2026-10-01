"""Matched E2 A/B/C trials; check mode never creates nodes or result files."""
import fcntl
import json
import uuid
from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
                            LogInfo, OpaqueFunction, UnsetLaunchConfiguration)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from nrs_imitation.e2_direct_abc import (CHECKPOINTS, PARAMETERS, ROOT, SCHEMA,
    build_runtime, launch_arguments, launch_values)


def repeat_index(cohort_dir, condition, repeat_text):
    """Count every saved attempt, including failures; check mode never reserves."""
    if repeat_text != 'auto':
        return int(repeat_text)
    indices = [json.loads(path.read_text())['run']['repeat_index']
               for path in (cohort_dir/condition).glob('*/runtime.json')]
    index = max([len(indices), *indices], default=0)+1
    if index > 5:
        raise ValueError(f'All five planned {condition} attempts have already been recorded for this session')
    return index


def prepare_attempt(values, session, repeat_text, cohort_dir):
    """Allocate the condition's next index and unique folder under one lock."""
    cohort_dir.mkdir(parents=True, exist_ok=True)
    with (cohort_dir/'.attempt_sequence.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        condition = values['e2_condition']
        index = repeat_index(cohort_dir, condition, repeat_text)
        values['e2_trial_index'] = index
        cfg = build_runtime(values)
        manifest_path = cohort_dir/'comparison.json'
        manifest = dict(schema=SCHEMA, session=session, **cfg['comparison'])
        # Reject drift before launching any ROS node, including checkpoint
        # bytes, normalizers, runtime code and the shared safety limits.
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous['common_settings_sha256'] != manifest['common_settings_sha256']:
                raise ValueError('Common E2 settings changed within this session; use the same settings for A/B/C')
        else:
            previous = manifest
            models = {c: build_runtime(dict(values, e2_condition=c, checkpoint='')) for c in CHECKPOINTS}
            if any(model['comparison']['common_settings_sha256'] != manifest['common_settings_sha256']
                   for model in models.values()):
                raise ValueError('A/B/C common settings or pose normalizers differ')
            previous['model_artifacts'] = {c: {key: model['il'][key] for key in
                ('checkpoint', 'checkpoint_sha256', 'normalizer_sha256')} for c, model in models.items()}
            with manifest_path.open('x') as stream:
                stream.write(json.dumps(previous, indent=2)+'\n')
        pinned = previous['model_artifacts'][condition]
        if any(cfg['il'][key] != pinned[key] for key in ('checkpoint_sha256', 'normalizer_sha256')):
            raise ValueError('E2 checkpoint/normalizer changed within this session')
        run_id = f'{session}_{condition}_r{index}_'+datetime.now().strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:12]
        folder = cohort_dir/condition/run_id
        values['e2_run_dir'] = str(folder)
        cfg = build_runtime(values)
        folder.mkdir(parents=True, exist_ok=False)
        (folder/'runtime.json').write_text(json.dumps(cfg, indent=2)+'\n')
    return cfg, folder, run_id


def configure(context):
    mode = LaunchConfiguration('mode').perform(context)
    condition = LaunchConfiguration('condition').perform(context)
    session = LaunchConfiguration('session').perform(context)
    repeat_text = LaunchConfiguration('repeat').perform(context)
    if repeat_text not in ('auto', '1', '2', '3', '4', '5'):
        raise ValueError('repeat must be auto, 1, 2, 3, 4 or 5')
    if (not session or session in ('.', '..') or len(session) > 80 or
            any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in session)):
        raise ValueError('session must be 1-80 letters, digits, underscores, dots or hyphens')
    values = launch_values(context)
    cohort_dir = ROOT/'results'/datetime.now().strftime('%Y%m%d')/'E2'/session
    # These identities are controlled by this entry point, not user overrides.
    values.update(e2_direct_a=False, e2_direct_abc=True, e2_condition=condition,
                  e2_cohort=session, e2_trial_index=1 if repeat_text == 'auto' else int(repeat_text), e2_run_dir='')
    cfg = build_runtime(values)
    if mode == 'check':
        if repeat_text == 'auto':
            values['e2_trial_index'] = repeat_index(cohort_dir, condition, repeat_text)
            cfg = build_runtime(values)
        return [LogInfo(msg=json.dumps(dict(status='ready', hardware_io=False,
            condition=condition, repeat=cfg['run']['repeat_index'], repeat_selection=repeat_text,
            checkpoint=cfg['il']['checkpoint'],
            force_source='external_F0' if condition == 'A' else 'policy',
            constant_force_N=cfg['external_force_N'], force_ramp_sec=cfg['force_ramp_sec'],
            common_settings_sha256=cfg['comparison']['common_settings_sha256'],
            protection=cfg['protection'], comparison=cfg['comparison'])))]
    if mode != 'run':
        raise ValueError('mode must be run or check')
    cfg, folder, run_id = prepare_attempt(values, session, repeat_text, cohort_dir)
    repeat_text = str(cfg['run']['repeat_index'])
    args = {key: (value if isinstance(value, str) else json.dumps(value)) for key, value in values.items()}
    args.update(checkpoint=cfg['il']['checkpoint'], ckpt_dir=str(Path(cfg['il']['checkpoint']).parent),
        e2_config='', e2_session_id=run_id, execution_method='il', act_root=str(ROOT),
        policy_class='FLOW', inference_mode='timed_topic',
        use_force_observation=json.dumps(condition == 'C'), use_force_history=json.dumps(condition != 'A'),
        use_stain_mask='false', stain_canon_enable='false', gradcam_enable='false',
        metrics_log_enable='true', metrics_log_dir=str(folder/'provider'), metrics_run_tag=run_id,
        metrics_extra_telemetry_enable='false', metrics_sample_hz='20.0',
        metrics_repeat_id=repeat_text, metrics_context_file=str(folder/'runtime.json'),
        overlay_record_enable='true', overlay_record_fps='10.0',
        overlay_record_output_dir=str(folder/'video'), removal_output_dir=str(folder/'plots'),
        visualize_flow_vector='false', visualize_modality_importance='false', removal_viz_open_on_exit='false')
    include = Path(get_package_share_directory('nrs_imitation'))/'launch/inference_clean_single_cam.launch.py'
    return [LogInfo(msg=f'E2 {condition} repeat {repeat_text}/5: {cfg["il"]["checkpoint"]}; logs {folder}'),
        GroupAction(actions=[UnsetLaunchConfiguration(key) for key in ('mode', 'session', 'condition', 'repeat')]+
            [IncludeLaunchDescription(PythonLaunchDescriptionSource(str(include)), launch_arguments=args.items())])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='check', choices=['run', 'check']),
        DeclareLaunchArgument('condition', default_value='A', choices=list(CHECKPOINTS)),
        DeclareLaunchArgument('session', default_value='E2_ABC_'+datetime.now().strftime('%Y%m%d')),
        DeclareLaunchArgument('repeat', default_value='auto', choices=['auto', '1', '2', '3', '4', '5'],
                             description='Omit to number attempts automatically per condition'),
        *launch_arguments(), OpaqueFunction(function=configure)])
