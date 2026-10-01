"""Matched A/B/C execution settings, built without legacy experiment files.

The policies differ in force inputs/outputs. Their transport, pose conditioning,
watchdogs, measured-wrench protection and logging protocol are shared.
"""
import json
import math
from pathlib import Path

import numpy as np

from .e2_ablation import ROOT, digest, object_hash, read_stats
from .e2_protection import LIMIT_KEYS, FeedbackProtectionMonitor
from .e2_timed_execution import executor_settings, validate_settings

SCHEMA = 'E2_direct_ABC_v1'
CHECKPOINTS = {
    'A': str(ROOT/'checkpoints/flow/polishing/single_cam/'
             'e2_force_ablation_20260926/A/20260926_2129/policy_best.ckpt'),
    'B': str(ROOT/'checkpoints/flow/polishing/single_cam/'
             'e1_force_observation_20260916/off/20260916_1531/policy_best.ckpt'),
    'C': str(ROOT/'checkpoints/flow/polishing/single_cam/'
             'e1_force_observation_20260916/on/20260916_1531/policy_best.ckpt'),
}
TRIAL_ORDER = [['C']*5, ['B']*5, ['A']*5]
PARAMETERS = {
    'e2_direct_a': False,  # Compatibility with e2_a.launch.py.
    'e2_direct_abc': False,
    'e2_condition': 'A',
    'e2_cohort': '',
    'e2_trial_index': 1,
    'checkpoint': '',
    'e2_run_dir': '',
    'constant_force_N': 23.0,
    'force_ramp_sec': 0.0,
    'measured_wrench_frame': 'robot_base',
    **{key: [200.0, 200.0, 200.0] for key in LIMIT_KEYS[:2]},
}


def build_runtime(parameters, require_protection=True, legacy_a=False):
    """Resolve a policy and pin the same native-unit execution for all conditions."""
    p = dict(PARAMETERS, **parameters)
    condition = 'A' if legacy_a else p['e2_condition']
    if condition not in CHECKPOINTS:
        raise ValueError('e2_condition must be A, B or C')
    repeat = p['e2_trial_index']
    if isinstance(repeat, bool) or not isinstance(repeat, int) or not 1 <= repeat <= 5:
        raise ValueError('e2_trial_index must be an integer from 1 to 5')
    if p['e2_direct_a'] and p['e2_direct_abc']:
        raise ValueError('Select only one direct E2 entry point')
    checkpoint = Path(p['checkpoint'] or CHECKPOINTS[condition]).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f'checkpoint file missing: {checkpoint}')
    if not legacy_a and checkpoint != Path(CHECKPOINTS[condition]):
        raise ValueError(f'Matched E2 {condition} requires its pinned policy_best.ckpt')
    stats_path = checkpoint.with_name('dataset_stats.pkl')
    stats = read_stats(stats_path)
    pc = stats.get('policy_config', {})
    motion_only = condition == 'A'
    action_dim = 6 if motion_only else 9
    expected = dict(state_dim=action_dim, action_dim=action_dim,
                    use_force_observation=condition == 'C', use_force_history=not motion_only,
                    motion_only=motion_only, force_action=not motion_only)
    for key, value in expected.items():
        # B/C predate explicit motion_only/force_action metadata; their 9-D
        # action schema includes learned force. A must declare both explicitly.
        default = {'motion_only': False, 'force_action': True}.get(key)
        if pc.get(key, default) != value:
            raise ValueError(f'E2 {condition} requires checkpoint {key}={value!r}')
    for key in ('qpos_min', 'qpos_max', 'action_min', 'action_max'):
        values = np.asarray(stats[key], float)
        if values.shape != (action_dim,) or not np.isfinite(values).all():
            raise ValueError(f'E2 {condition} requires finite {action_dim}-D normalizer: {key}')
    if (stats.get('relative_transform_version') != 'stain_relative_v1' or
            stats.get('rotation_aligned') is not False):
        raise ValueError('E2 requires translation-only stain_relative_v1')
    pose = np.asarray(stats['demo_start_pose_mean'], float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError('checkpoint demo_start_pose_mean must be finite pose6')
    f0 = p['constant_force_N']
    if isinstance(f0, bool) or not isinstance(f0, (float, int)) or not math.isfinite(f0):
        raise ValueError('constant_force_N must be finite')
    ramp = p['force_ramp_sec']
    if (isinstance(ramp, bool) or not isinstance(ramp, (float, int)) or
            not math.isfinite(ramp) or ramp < 0):
        raise ValueError('force_ramp_sec must be nonnegative and finite')
    if not legacy_a and ramp != 0:
        raise ValueError('Matched A/B/C requires force_ramp_sec:=0.0; shared force slew remains active')
    profile = dict(profile='c_pose_conditioning_20260926_v1', pose_smooth_window=35,
                   handover_s=0.5, linear_acceleration_mm_s2=25.0,
                   angular_acceleration_rad_s2=math.radians(100.0))
    config = dict(schema='E2_direct_A_v1' if legacy_a else SCHEMA, experiment_id='E2', condition=condition,
        run=dict(directory=p['e2_run_dir'] or None, cohort_id=p['e2_cohort'] or None,
                 condition=condition, repeat_index=repeat, planned_trials_per_condition=5),
        external_force_N=float(f0) if motion_only else None, force_ramp_sec=float(ramp),
        executor=dict(transport='e2_timed_topic_v1', control_period_s=0.008,
            gain_hz=15.0, linear_speed_mm_s=10.0, angular_speed_rad_s=math.radians(40.0),
            force_rate_N_s=30.0, fz_hard_limit_N=0.0, contact_on_N=3.0, contact_off_N=1.2,
            feedback_max_age_s=0.2, max_tick_gap_s=0.1, max_plan_age_s=0.5,
            mode_timeout_s=2.0, provider_timeout_s=2.0, stop_window_s=0.4, stop_timeout_s=15.0,
            completion_position_tolerance_mm=5.0, completion_rotation_tolerance_rad=0.05),
        common=dict(workspace_limits=dict(max_xy_from_start_mm=140.0,
            max_z_down_from_start_mm=85.0, max_z_up_from_start_mm=95.0,
            max_xyz_from_current_mm=200.0), task_time_budget_s=50.0,
            demo_start_pose6=pose.tolist(), rpm_setpoint=None, rpm_status='unknown'),
        il=dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
            normalizer_sha256=digest(stats_path), seed=0, flow_infer_steps=10,
            chunk_size=128, replan_interval_steps=120, trajectory_hz=30.0,
            force_history_len=30, use_force_observation=condition == 'C',
            use_force_history=not motion_only, motion_only=motion_only, force_action=not motion_only,
            pose_conditioning=profile),
        protocol=dict(phase_ids=['pass_1'], termination='operator_finish',
            processing_marker='automatic_execution_start', physical_processing_verified=False,
            failed_attempts='retain; do not replace failures with successful retries'),
        protection=dict(feedback_topic='/ur10skku/currentF',
            freshness='local ROS receipt; sensor acquisition timestamp unavailable',
            **{key: p[key] for key in (*LIMIT_KEYS[:2], 'measured_wrench_frame')}))
    errors = validate_settings(config)
    if errors:
        raise ValueError('; '.join(errors))
    if require_protection:
        FeedbackProtectionMonitor(config['protection'], executor_settings(config, 'il'))
    if not legacy_a:
        sampling = {key: config['il'][key] for key in ('seed', 'flow_infer_steps', 'chunk_size',
                    'replan_interval_steps', 'trajectory_hz', 'force_history_len', 'pose_conditioning')}
        contract = dict(profile=SCHEMA, executor=config['executor'], common=config['common'],
            sampling=sampling, protection=config['protection'], protocol=config['protocol'],
            checkpoint_selection='minimum validation loss (policy_best.ckpt)',
            checkpoints=CHECKPOINTS, A_constant_force_N=float(f0), force_ramp_sec=0.0,
            pose_normalizers={key: np.asarray(stats[key], float)[:6].tolist()
                              for key in ('qpos_min', 'qpos_max', 'action_min', 'action_max')},
            code_sha256={name: digest(Path(__file__).with_name(name)) for name in
                ('e2_direct_abc.py', 'e2_timed_execution.py', 'e2_protection.py',
                 'e2_executor_node.py', 'inference_core.py', 'inference_metrics.py')})
        config['comparison'] = dict(common_settings_sha256=object_hash(contract), contract=contract,
            planned_trials_per_condition=5, planned_order=TRIAL_ORDER,
            order_source='operator request: C x5, B x5, A x5',
            physical_setup_verified=False, A_force_calibration_verified=False,
            physical_setup='operator must match RPM, initial stain/surface, tool and force-controller settings')
    return config


def from_node(node):
    p = {key: node.get_parameter(key).value for key in PARAMETERS}
    return build_runtime(p, legacy_a=not p['e2_direct_abc'])


def launch_values(context):
    import yaml
    from launch.substitutions import LaunchConfiguration
    return {key: (LaunchConfiguration(key).perform(context) if isinstance(default, str)
                  else yaml.safe_load(LaunchConfiguration(key).perform(context)))
            for key, default in PARAMETERS.items()}


def launch_arguments():
    from launch.actions import DeclareLaunchArgument
    return [DeclareLaunchArgument(key, default_value=(default if isinstance(default, str)
            else json.dumps(default))) for key, default in PARAMETERS.items()]


def node_parameters():
    from launch.substitutions import LaunchConfiguration
    from launch_ros.parameter_descriptions import ParameterValue
    return {key: ParameterValue(LaunchConfiguration(key), value_type=(None if isinstance(default, list)
            else type(default))) for key, default in PARAMETERS.items()}
