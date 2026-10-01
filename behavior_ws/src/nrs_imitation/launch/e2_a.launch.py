"""Run A directly from its original checkpoint; no experiment config file."""
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            LogInfo, OpaqueFunction, GroupAction, UnsetLaunchConfiguration)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from nrs_imitation.e2_direct_a import (DEFAULT_CHECKPOINT, PARAMETERS, ROOT,
                                     build_runtime, launch_arguments, launch_values)


def configure(context):
    mode = LaunchConfiguration('mode').perform(context)
    values = launch_values(context)
    cfg = build_runtime(values)
    if mode == 'check':
        return [LogInfo(msg=json.dumps(dict(status='ready', hardware_io=False,
            checkpoint=cfg['il']['checkpoint'], constant_force_N=cfg['external_force_N'],
            force_ramp_sec=cfg['force_ramp_sec'],
            force_ramp_start='first_contact' if cfg['force_ramp_sec'] else None,
            protection=cfg['protection'], config_file_required=False)))]
    session = LaunchConfiguration('session').perform(context)
    # Always append an ID: repeated launches never reuse earlier result files.
    if not session or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in session):
        raise ValueError('session must contain letters, digits, underscores, dots or hyphens')
    run_id = session+'_'+datetime.now().strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:12]
    folder = ROOT/'results'/datetime.now().strftime('%Y%m%d')/'E2'/'A_direct'/run_id
    if mode == 'offline':
        python = '/home/eunseop/miniconda3/envs/nrs_imitation/bin/python3'
        pythonpath = os.pathsep.join((str(ROOT/'behavior_ws/src/nrs_imitation'),
            str(ROOT/'source'), os.environ.get('PYTHONPATH', '')))
        return [ExecuteProcess(cmd=[python, '-m', 'nrs_imitation.e2_direct_a',
            '--checkpoint', cfg['il']['checkpoint'], '--output', str(folder)],
            cwd=str(ROOT), output='screen', additional_env=dict(PYTHONPATH=pythonpath,
                HF_HUB_OFFLINE='1', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2'))]
    if mode != 'run':
        raise ValueError('mode must be run, check or offline')
    include = Path(get_package_share_directory('nrs_imitation'))/'launch/inference_clean_single_cam.launch.py'
    args = {key: LaunchConfiguration(key).perform(context) for key in PARAMETERS}
    args.update(checkpoint=cfg['il']['checkpoint'], ckpt_dir=str(Path(cfg['il']['checkpoint']).parent),
        e2_direct_a='true', e2_config='', e2_session_id=run_id, e2_run_dir=str(folder),
        execution_method='il', act_root=str(ROOT), policy_class='FLOW', inference_mode='timed_topic',
        use_force_observation='false', use_force_history='false', use_stain_mask='false',
        stain_canon_enable='false', gradcam_enable='false', metrics_log_enable='true',
        metrics_log_dir=str(folder/'provider'), metrics_run_tag=run_id,
        overlay_record_enable='true', overlay_record_fps='10.0',
        overlay_record_output_dir=str(folder/'video'), removal_output_dir=str(folder/'plots'),
        visualize_flow_vector='false', visualize_modality_importance='false',
        removal_viz_open_on_exit='false')
    return [LogInfo(msg=f'E2 A: original checkpoint {cfg["il"]["checkpoint"]}; logs {folder}'),
        GroupAction(actions=[UnsetLaunchConfiguration('mode'), UnsetLaunchConfiguration('session'),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(str(include)),
                                     launch_arguments=args.items())])]


def generate_launch_description():
    arguments = launch_arguments()
    # Expose checkpoint selection without copying or renaming training files.
    arguments = [a for a in arguments if a.name != 'checkpoint']
    return LaunchDescription([
        DeclareLaunchArgument('checkpoint', default_value=DEFAULT_CHECKPOINT),
        DeclareLaunchArgument('mode', default_value='run', choices=['run', 'check', 'offline']),
        DeclareLaunchArgument('session', default_value='E2_A'),
        *arguments, OpaqueFunction(function=configure)])
