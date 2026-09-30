"""Direct ROS launch entry for the existing R/T/C timed-topic executor.

This does not implement or certify E1's proposed common service transport.
The default check mode starts no ROS nodes. No checkpoint/episode is selected
implicitly for C/T; run mode requires the operator's matching explicit choice.
"""
import copy
import json
import pickle
import tempfile
from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, LogInfo, OpaqueFunction,
                            SetEnvironmentVariable, UnsetLaunchConfiguration)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from nrs_imitation.e2_providers import file_hash, hardware_blockers, load_config


ROOT = Path('/home/eunseop/nrs_imitation')
METHODS = {'R': 'rule', 'T': 'replay', 'C': 'il'}
ENTRY_ARGUMENTS = ('method', 'config', 'mode', 'checkpoint', 'episode', 'run_tag')


class StatsUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy._core' or module.startswith('numpy._core.'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def prepare(method, config_path, checkpoint='', episode=''):
    """Read-only validation, shared by check/run; no policy forward or ROS I/O."""
    if method not in METHODS:
        raise ValueError('method must be R, T, or C')
    path = Path(config_path).expanduser().resolve()
    cfg = load_config(path)
    il = cfg['il']
    if method == 'C':
        if not checkpoint:
            raise ValueError('C requires checkpoint:=/absolute/path/policy_best.ckpt; '
                             'no automatic selection. Config candidate: ' + il['checkpoint'])
        chosen = Path(checkpoint).expanduser().resolve()
        if chosen != Path(il['checkpoint']).expanduser().resolve():
            raise ValueError('checkpoint differs from config.il.checkpoint; '
                             'provide a reviewed matching config, no fallback')
        if chosen.name != 'policy_best.ckpt':
            raise ValueError('Existing inference entry requires policy_best.ckpt')
        with chosen.with_name('dataset_stats.pkl').open('rb') as handle:
            stats = StatsUnpickler(handle).load()
        policy = stats.get('policy_config', {})
        if policy.get('use_force_observation') is not True:
            raise ValueError('C normalizer policy_config must explicitly record '
                             'use_force_observation=true; OFF/legacy missing flag rejected')
        if (policy.get('state_dim') != 9 or policy.get('action_dim') != 9 or
                len(stats['qpos_min']) != 9 or len(stats['action_min']) != 9):
            raise ValueError('C requires pose6 + force3 observation/action schema')
    if method == 'T':
        if not episode:
            raise ValueError('T requires episode:=episode_<ID>; no automatic selection. '
                             'Config candidate: ' + str(cfg['replay'].get('episode_id')))
        if episode != cfg['replay'].get('episode_id'):
            raise ValueError('episode differs from config.replay.episode_id; '
                             'export/select a matching template first')
        from nrs_imitation.e2_providers import TimedActions
        metadata = TimedActions.load(cfg['replay']['template']).metadata
        if Path(metadata['source_episode']).stem != episode:
            raise ValueError('Replay template source episode differs from explicit selection')
    # inference_clean fixes these policy settings. Do not silently ignore a
    # different reviewed config when wrapping that existing entry point.
    for key, expected in {'seed': 0, 'flow_infer_steps': 10, 'chunk_size': 128,
                          'force_history_len': 30, 'trajectory_hz': 30.0,
                          'replan_interval_steps': 120}.items():
        if il.get(key) != expected:
            raise ValueError('Existing clean launch requires il.%s=%s' % (key, expected))
    blockers = hardware_blockers(cfg, METHODS[method], enabled=True)
    if blockers:
        raise ValueError('Existing timed executor preflight failed:\n- ' + '\n- '.join(blockers))
    return path, cfg


def configure(context):
    args = {name: LaunchConfiguration(name).perform(context) for name in ENTRY_ARGUMENTS}
    if args['mode'] not in ('check', 'run'):
        raise ValueError('mode must be check or run')
    path, cfg = prepare(args['method'], args['config'], args['checkpoint'], args['episode'])
    info = ('RTC %s: existing e2_timed_topic_v1; config=%s; '
            'execution_equivalence=unverified. This is not the proposed E1 service executor.'
            % (args['method'], path))
    profile = cfg.get('il', {}).get('pose_conditioning') if args['method'] == 'C' else None
    info += ' C pose conditioning: ' + (profile['profile'] if profile else 'inactive') + '.'
    if args['mode'] == 'check':
        return [LogInfo(msg=info), LogInfo(msg='Configuration check passed; no nodes started. '
                    'Hardware calibration and E1 protocol readiness are not certified.')]

    # Preserve the reviewed source config and record current paper numbering.
    # Only explicit run mode writes a context file or includes robot nodes.
    run_cfg = copy.deepcopy(cfg)
    run_cfg['paper_experiment'] = 'E1'
    run_cfg['execution_equivalence'] = 'unverified'
    run_cfg['launch_context'] = dict(entry='rtc_timed.launch.py', method=args['method'],
        legacy_config_path=str(path), legacy_config_sha256=file_hash(path),
        checkpoint_argument=args['checkpoint'] or None, episode_argument=args['episode'] or None,
        c_pose_conditioning=profile,
        scope='existing timed executor only; common service E1 preparation incomplete')
    log_root = ROOT / 'logs/rtc_launch_context'
    log_root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='RTC_%s_' % args['method'], dir=log_root))
    snapshot = folder / 'config.json'
    with snapshot.open('x') as handle:
        json.dump(run_cfg, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    tag = args['run_tag'] or 'RTC_%s_%s' % (args['method'], datetime.now().strftime('%Y%m%dT%H%M%S'))
    common = cfg['common']
    launch_args = dict(execution_method=METHODS[args['method']], e2_config=str(snapshot),
        e2_enable_hardware='true', act_root=str(ROOT), policy_class='FLOW',
        ckpt_dir=str(Path(cfg['il']['checkpoint']).expanduser().resolve().parent),
        use_force_observation='true', use_stain_mask='false', stain_canon_enable='false',
        inference_mode='timed_topic', gradcam_enable='false',
        overlay_record_enable='true', overlay_record_fps='10.0',
        visualize_flow_vector='false', visualize_modality_importance='false',
        removal_viz_open_on_exit='false', metrics_log_enable='true',
        metrics_extra_telemetry_enable='false', metrics_sample_hz='20.0',
        metrics_context_file=str(snapshot), metrics_specimen_id=str(common.get('specimen_id') or 'default'),
        metrics_repeat_id=str(common.get('repeat_id') or ''), metrics_run_tag=tag,
        metrics_rpm_setpoint='' if common.get('rpm_setpoint') is None else str(common['rpm_setpoint']))
    source = Path(get_package_share_directory('nrs_imitation')) / 'launch/inference_clean_single_cam.launch.py'
    return [LogInfo(msg=info), LogInfo(msg='Run context: ' + str(snapshot)), GroupAction(actions=[
        # Includes inherit launch configurations. Consume wrapper-only names
        # inside this scope: otherwise stain_origin_online's "config" default
        # is replaced by the experiment JSON, which its YAML loader rejects.
        # Keep the inherited ROS environment and unrelated launch settings.
        *(UnsetLaunchConfiguration(name) for name in ENTRY_ARGUMENTS),
        SetEnvironmentVariable('OMP_NUM_THREADS', '2'),
        SetEnvironmentVariable('OPENBLAS_NUM_THREADS', '2'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(source)),
                                 launch_arguments=launch_args.items()),
    ])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('method', choices=list(METHODS), description='R=rule, T=replay, C=IL'),
        DeclareLaunchArgument('config', description='Explicit reviewed experiment config JSON path'),
        DeclareLaunchArgument('mode', default_value='check', choices=['check', 'run'],
                              description='check: no nodes; run: starts existing robot execution nodes'),
        DeclareLaunchArgument('checkpoint', default_value='',
                              description='Required for C; exact policy_best.ckpt matching config'),
        DeclareLaunchArgument('episode', default_value='',
                              description='Required for T; exact episode_ID matching config/template'),
        DeclareLaunchArgument('run_tag', default_value='', description='Optional log tag; otherwise timestamped'),
        OpaqueFunction(function=configure),
    ])
