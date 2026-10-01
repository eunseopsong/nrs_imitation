"""Direct E2 A/B/C launch with unique attempt preservation; check starts no nodes."""
import atexit
import json
from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
                            OpaqueFunction, LogInfo, SetEnvironmentVariable, UnsetLaunchConfiguration,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessStart, OnProcessExit, OnShutdown
from launch.logging import launch_config
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from nrs_imitation.e2_ablation import ROOT, EXPERIMENT, select_condition, readiness
from nrs_imitation.e2_providers import hardware_blockers
from nrs_imitation.e2_run_context import create_attempt, read, write, seal_archive, LaunchRunRecord
from nrs_imitation.e2_direct_abc import launch_arguments as common_arguments, PARAMETERS


def configure(context):
    config_text = LaunchConfiguration('config').perform(context)
    if not config_text:
        source = Path(get_package_share_directory('nrs_imitation'))/'launch/e2_abc_common.launch.py'
        args = {key: LaunchConfiguration(key).perform(context) for key in
                (*PARAMETERS, 'condition', 'mode', 'session', 'repeat')}
        return [IncludeLaunchDescription(PythonLaunchDescriptionSource(str(source)),
                                         launch_arguments=args.items())]
    path = Path(config_text).expanduser().resolve()
    cfg = json.loads(path.read_text())
    condition = LaunchConfiguration('condition').perform(context)
    mode = LaunchConfiguration('mode').perform(context)
    selected = select_condition(cfg, condition)
    if mode == 'check':
        return [LogInfo(msg=json.dumps(readiness(cfg),ensure_ascii=False))]
    if mode != 'run':raise ValueError('mode must be check or run')
    direct=not cfg.get('run');handlers=[]
    if direct:
        block=LaunchConfiguration('block',default='').perform(context)
        cfg,folder=create_attempt(cfg,condition,LaunchConfiguration('session',default='E2_pilot_01').perform(context),
            block=int(block) if block else None,
            specimen=LaunchConfiguration('specimen',default='').perform(context) or None,
            region=LaunchConfiguration('region',default='').perform(context) or None,
            reuse=LaunchConfiguration('surface_reuse',default='').perform(context) or None)
        path=folder/'config.json'
        lifecycle=LaunchRunRecord(folder,launch_config.log_dir)
        atexit.register(lifecycle.close)
        handlers=[RegisterEventHandler(OnProcessStart(on_start=lifecycle.process_started)),
                  RegisterEventHandler(OnProcessExit(on_exit=lifecycle.process_exited)),
                  RegisterEventHandler(OnShutdown(on_shutdown=lifecycle.shutdown_requested))]
    elif cfg.get('condition')!=condition or cfg.get('il')!=selected['il']:
        raise ValueError('Prepared attempt does not match condition/model selection')
    run=cfg['run'];folder=Path(run['directory'])
    if folder.resolve()!=path.parent or not (folder/'attempt.json').is_file():
        raise ValueError('Attempt/config directory mismatch')
    blockers = hardware_blockers(cfg, 'il', enabled=True)
    if blockers:
        record=read(folder/'attempt.json');record.update(status='preflight_blocked',executed=False,blockers=blockers)
        write(folder/'attempt.json',record);seal_archive(folder)
        raise RuntimeError(f'E2 attempt preserved at {folder}; preflight blocked: ' + '; '.join(blockers))
    record=read(folder/'attempt.json')
    if record['status'] not in ('created','launch_requested'):
        raise ValueError('An existing completed/blocked attempt cannot be launched again; use the master config')
    write(folder/'launch_claim.json',dict(condition=condition,entry='ros2_launch'),exclusive=True)
    if direct:
        lifecycle.update(status='launch_requested',executed=None)
        write(folder/'launch_command.json',dict(entry='ros2 launch nrs_imitation e2_abc.launch.py',
              condition=condition,config=str(path),ROS_LOG_DIR=str(folder/'ros_logs')))
    args = dict(execution_method='il',e2_config=str(path),e2_enable_hardware='true',
        e2_session_id=run['run_id'],act_root=str(ROOT),policy_class='FLOW',
        ckpt_dir=str(Path(cfg['il']['checkpoint']).parent),ckpt_auto_subdir='polishing/single_cam',
        use_force_observation=str(condition=='C').lower(),use_force_history=str(condition!='A').lower(),
        use_stain_mask='false',stain_canon_enable='false',inference_mode='timed_topic',gradcam_enable='false',
        overlay_record_enable='true',overlay_record_fps='10.0',overlay_record_output_dir=str(folder/'video'),
        removal_output_dir=str(folder/'plots'),
        visualize_flow_vector='false',visualize_modality_importance='false',removal_viz_open_on_exit='false',
        metrics_log_enable='true',metrics_extra_telemetry_enable='false',metrics_sample_hz='20.0',
        metrics_log_dir=str(folder/'provider'),metrics_context_file=str(path),metrics_run_tag=run['run_id'],
        metrics_specimen_id=run['specimen_id'],metrics_repeat_id=str(run['repeat_id']),
        metrics_rpm_setpoint='' if cfg['common'].get('rpm_setpoint') is None else str(cfg['common']['rpm_setpoint']))
    source = Path(get_package_share_directory('nrs_imitation'))/'launch/inference_clean_single_cam.launch.py'
    return [*handlers,LogInfo(msg=f"E2 {condition}: {run['run_id']} | contract {cfg['execution_contract_hash']}"),
        GroupAction(actions=[*(UnsetLaunchConfiguration(k) for k in ('config','condition','mode','session','block','specimen','region','surface_reuse')),
            SetEnvironmentVariable('ROS_LOG_DIR',str(folder/'ros_logs')),
            SetEnvironmentVariable('OMP_NUM_THREADS','2'),SetEnvironmentVariable('OPENBLAS_NUM_THREADS','2'),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(str(source)),launch_arguments=args.items())])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config',default_value='',description='Empty: matched A/B/C. Explicit JSON: legacy E2 protocol'),
        DeclareLaunchArgument('condition',choices=['A','B','C']),
        DeclareLaunchArgument('mode',default_value='check',choices=['check','run']),
        DeclareLaunchArgument('session',default_value='E2_ABC_'+datetime.now().strftime('%Y%m%d')),
        DeclareLaunchArgument('repeat',default_value='auto',choices=['auto','1','2','3','4','5']),
        DeclareLaunchArgument('block',default_value='',description='Optional block; not required for repeated common-state trials'),
        DeclareLaunchArgument('specimen',default_value=''),
        DeclareLaunchArgument('region',default_value=''),
        DeclareLaunchArgument('surface_reuse',default_value=''),
        *common_arguments(),
        OpaqueFunction(function=configure)])
