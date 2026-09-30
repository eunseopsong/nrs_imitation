"""Paper E2 A/B/C contract and adapters. Importing this module has no ROS I/O."""
import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

SCHEMA = 'E2_ABC_force_ablation_v1'
ROOT = Path('/home/eunseop/nrs_imitation')
EXPERIMENT = ROOT / 'experiments/e2_force_ablation_20260926'
CONTRACT_SOURCES = (
    'behavior_ws/src/nrs_imitation/nrs_imitation/e2_timed_execution.py',
    'behavior_ws/src/nrs_imitation/nrs_imitation/e2_executor_node.py',
    'behavior_ws/src/nrs_imitation/nrs_imitation/e2_ablation.py',
    'behavior_ws/src/nrs_imitation/nrs_imitation/e2_pilot.py',
    'behavior_ws/src/nrs_imitation/nrs_imitation/e2_protection.py',
    'behavior_ws/src/nrs_imitation/nrs_imitation/e2_run_context.py',
    'behavior_ws/src/nrs_imitation/nrs_imitation/inference_core.py',
    'behavior_ws/src/nrs_imitation/nrs_imitation/e2_providers.py',
    'behavior_ws/src/nrs_imitation/nrs_imitation/inference_metrics.py',
    'behavior_ws/src/nrs_imitation/nrs_imitation/execution_metrics.py',
    'behavior_ws/src/nrs_imitation/launch/e2_abc.launch.py',
    'behavior_ws/src/nrs_imitation/launch/inference_clean_single_cam.launch.py',
    'behavior_ws/src/nrs_imitation/launch/inference_gradcam_single_cam.launch.py',
    'source/models/flow_core.py',
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def is_ablation(config):
    return config.get('schema') == SCHEMA and config.get('experiment_id') == 'E2'


class StatsUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy._core' or module.startswith('numpy._core.'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def read_stats(path):
    with Path(path).open('rb') as f:
        return StatsUnpickler(f).load()


def execution_contract(config):
    """Intentional model/condition/identifier differences are excluded."""
    return dict(version='E2_common_motion_v1', executor=config['executor'],
        pose_conditioning=config['motion_postprocessor'],
        task=config['task'],
        common={k: config['common'].get(k) for k in (
            'workspace_limits', 'task_time_budget_s', 'demo_start_pose6',
            'frozen_roi', 'frame', 'stain_canon_enable', 'use_stain_mask',
            'approach_retract_protocol', 'controller_deployment_id')},
        sampling=config['sampling'], protocol=config['protocol'],
        code_sha256={p: digest(ROOT / p) for p in CONTRACT_SOURCES})


def cohort_id(config):
    """Separate recipe/model/analysis revisions without depending on artifact paths."""
    identity = dict(execution_contract_hash=config.get('execution_contract_hash'),
        models={c: {k: m.get(k) for k in ('checkpoint_sha256', 'normalizer_sha256')}
                for c, m in config['models'].items()},
        calibration=config['calibration'], analysis=config['analysis'],
        f0={k: config['f0'].get(k) for k in ('status', 'command_fz_N', 'normal_force_N',
            'candidate_sha256', 'calibration_hash', 'validated_command_range_N')},
        reference={k: config['reference'].get(k) for k in ('status', 'sha256')})
    if 'trial_stage' in config or 'pilot' in config:
        identity['trial_stage'] = config.get('trial_stage', 'main')
        pilot = config.get('pilot', {})
        identity['pilot'] = {k: v for k, v in pilot.items() if k != 'commissioning_record'}
        identity['commissioning_sha256'] = (pilot.get('commissioning_record') or {}).get('sha256')
    return 'E2_' + object_hash(identity)[:16]


def select_condition(config, condition):
    if not is_ablation(config) or condition not in 'ABC' or len(condition) != 1:
        raise ValueError('Explicit E2 A/B/C config and condition required')
    result = copy.deepcopy(config)
    result['condition'] = condition
    model = result['models'][condition]
    result['il'] = dict(result['sampling'], checkpoint=model.get('checkpoint'),
        checkpoint_sha256=model.get('checkpoint_sha256'),
        normalizer_sha256=model.get('normalizer_sha256'),
        use_force_observation=condition == 'C', force_action=condition != 'A',
        motion_only=condition == 'A', use_force_history=condition != 'A',
        inference_mode='timed_topic', pose_conditioning=result['motion_postprocessor'])
    return result


def model_errors(config):
    c = config.get('condition')
    if c not in ('A', 'B', 'C'):
        return ['condition must be A/B/C']
    model = config['models'][c]
    path = model.get('checkpoint')
    if not path or not Path(path).is_file():
        return [c + ' trained checkpoint missing (smoke weights are not a trained policy)']
    errors = []
    if model.get('checkpoint_sha256') != digest(path):
        errors.append(c + ' checkpoint SHA mismatch')
    stats_path = Path(path).with_name('dataset_stats.pkl')
    if not stats_path.is_file():
        return errors + [c + ' normalizer missing']
    if model.get('normalizer_sha256') != digest(stats_path):
        errors.append(c + ' normalizer SHA mismatch')
    s = read_stats(stats_path); pc = s.get('policy_config', {})
    expected = dict(state_dim=6 if c == 'A' else 9, action_dim=6 if c == 'A' else 9,
                    use_force_observation=c == 'C', use_force_history=c != 'A')
    for key, value in expected.items():
        if pc.get(key) != value:
            errors.append(c + ' checkpoint schema mismatch: ' + key)
    if c == 'A' and (pc.get('motion_only') is not True or pc.get('force_action') is not False):
        errors.append('A must be trained motion-only, not a masked B/C output')
    if len(s['qpos_min']) != expected['state_dim'] or len(s['action_min']) != expected['action_dim']:
        errors.append('normalizer dimensions differ from condition')
    if s.get('relative_transform_version') != 'stain_relative_v1' or s.get('rotation_aligned') is not False:
        errors.append('model must preserve translation-only stain_relative_v1')
    return errors


def runtime_contract_errors(config):
    errors = model_errors(config)
    il = config['il']; c = config['condition']
    if il != select_condition(config,c)['il']:
        errors.append('runtime model/sampling adapter differs from frozen condition selection')
    for key, expected in dict(use_force_observation=c == 'C', force_action=c != 'A',
                              motion_only=c == 'A', use_force_history=c != 'A').items():
        if il.get(key) != expected:
            errors.append('condition adapter differs from checkpoint: ' + key)
    if il.get('pose_conditioning') != config['motion_postprocessor']:
        errors.append('motion postprocessor must be identical A/B/C')
    if config.get('execution_contract_hash') != object_hash(execution_contract(config)):
        errors.append('execution contract/settings/code hash changed; review and seal again')
    if config.get('cohort_id') != cohort_id(config):
        errors.append('model/recipe/calibration/analysis cohort identity changed; seal again')
    protocol = config.get('protocol', {})
    if protocol.get('termination') != 'operator_finish' or not protocol.get('reviewed') or not protocol.get('review_evidence'):
        errors.append('common operator-finish/phase protocol not reviewed/frozen')
    if not protocol.get('surface_process'):
        errors.append('surface process must be documented (ink or physical scratch)')
    if (not config.get('calibration', {}).get('execution_convention_verified') or
            not config.get('calibration', {}).get('execution_convention_evidence')):
        errors.append('current controller/tool force command convention not verified for E2')
    stage = config.get('trial_stage', 'main')
    if stage not in ('pilot', 'main'):
        errors.append('trial_stage must be pilot or main')
    if stage == 'pilot' and c != 'A':
        errors.append('this pilot contract supports motion-only A only')
    if c == 'A' and stage == 'pilot':
        from .e2_pilot import pilot_errors
        errors.extend(pilot_errors(config))
    elif c == 'A':
        f0 = config.get('f0', {})
        if f0.get('status') != 'frozen' or not isinstance(f0.get('command_fz_N'), (int, float)):
            errors.append('A F0 is not calibrated and frozen; real execution blocked')
        elif not np.isfinite(f0['command_fz_N']) or not f0.get('freeze_evidence'):
            errors.append('A F0 requires finite command and calibration evidence')
        elif f0.get('calibration_hash') != object_hash(config['calibration']):
            errors.append('A F0 calibration changed since freeze')
        elif not f0.get('candidate') or not Path(f0['candidate']).is_file() or digest(f0['candidate'])!=f0.get('candidate_sha256'):
            errors.append('A F0 source artifact identity changed/missing')
        from .e2_pilot import commissioning_errors, evidence_errors
        errors.extend(commissioning_errors(config))
        errors.extend(evidence_errors(f0.get('pilot_result_evidence'), 'f0.pilot_result_evidence'))
        if config.get('pilot', {}).get('quality_validation') != 'reviewed':
            errors.append('main A requires review of actual pilot results; pilot readiness is not main approval')
    return errors


def motion_to_contract(motion):
    """Placeholder force slots are transport-only, never policy predictions."""
    a = np.asarray(motion)
    if a.ndim != 2 or a.shape[1] != 6 or not np.isfinite(a).all():
        raise ValueError('A policy must output finite [horizon, pose6] only')
    return np.concatenate((a, np.zeros((len(a), 3), dtype=a.dtype)), axis=1)


def scheduled_force(f0, processing):
    if not processing:
        return 0.
    if f0.get('status') != 'frozen' or not np.isfinite(f0.get('command_fz_N', np.nan)):
        raise ValueError('processing A requires frozen external F0')
    return float(f0['command_fz_N'])


def prepare_A_force(config):
    """Preflight once, then reuse the approved scalar with the common phase gate."""
    if config.get('trial_stage', 'main') == 'pilot':
        from .e2_pilot import pilot_force
        return pilot_force(config, True)
    return scheduled_force(config['f0'], True)


def readiness(config):
    from .e2_providers import hardware_blockers
    from .e2_force_analysis import normal_force, Unavailable
    models = {c: not model_errors(select_condition(config, c)) for c in 'ABC'}
    errors = {c: hardware_blockers(select_condition(config, c), 'il', True) for c in 'ABC'}
    calibration=False;reference=False;analysis_reasons=[]
    try:
        cal=config['calibration']
        normal_force([[0.,0.,0.]],'robot_base',cal)
        normal_force([[0.,0.,0.]],'teacher_calibrated',cal)
        if cal.get('controller_target_frame')!='controller_tcp' or cal.get('rotation_basis')!='actual_feedback_tcp':
            raise Unavailable('controller target rotation basis is not verified')
        calibration=True
    except (Unavailable,ValueError,TypeError) as exc:analysis_reasons.append(str(exc))
    try:
        info=config['reference']
        if info.get('status')!='frozen' or not info.get('artifact'):raise Unavailable('evaluation reference not frozen')
        if digest(info['artifact'])!=info.get('sha256'):raise Unavailable('reference artifact hash mismatch')
        ref=json.loads(Path(info['artifact']).read_text())
        if ref.get('calibration_hash')!=object_hash(config['calibration']) or ref.get('analysis_hash')!=object_hash(config['analysis']):
            raise Unavailable('reference calibration/alignment identity changed')
        if [p['phase_id'] for p in ref['phases']]!=config['protocol']['phase_ids']:
            raise Unavailable('reference phases differ from frozen protocol')
        reference=True
    except (Unavailable,ValueError,KeyError,TypeError,OSError) as exc:analysis_reasons.append(str(exc))
    return dict(model_ready=models,
        execution_contract_ready=config.get('execution_contract_hash') == object_hash(execution_contract(config)),
        hardware_preflight_status={c: {'status': 'offline_pass' if not errors[c] else 'blocked',
                                      'reasons': errors[c]} for c in 'ABC'},
        force_frame_calibration_ready=bool(calibration), reference_and_alignment_ready=reference,
        force_profile_metrics_ready=bool(calibration and reference), stylus_metrology_ready=False,
        force_analysis_blockers=analysis_reasons,
        hardware_executed=False, physical_safety_validation=False, quality_verified=False)
