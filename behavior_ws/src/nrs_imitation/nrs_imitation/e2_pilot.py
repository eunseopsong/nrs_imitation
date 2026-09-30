"""A commissioning preflight. No ROS I/O, approval writer, or safety bypass.

Quality and F0 freezing belong to main experiments. Pilot permission instead
needs an applicable commissioning record AND installed protection support.
Runtime source provenance is additionally required before the executor is ready.
"""
import json
from pathlib import Path

import numpy as np


def installed_protection_errors():
    """Software support check; runtime never becomes ready without producer data.

    This replaces the old unconditional error list with the actual monitor.
    It does NOT claim that the staged robot-side patch has been deployed.
    """
    from .e2_protection import CAPABILITIES
    required = {'measured_wrench', 'source_freshness', 'raw_sensor_envelope',
                'actual_tcp_workspace', 'startup_alignment_coverage'}
    return ['protection implementation unavailable: ' + name for name in sorted(required-CAPABILITIES)]


def finite_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and np.isfinite(value))


def evidence_errors(ref, name):
    from .e2_ablation import digest
    if not isinstance(ref, dict) or not ref.get('path') or not ref.get('sha256'):
        return [name + ': pinned evidence path + sha256 required']
    path = Path(ref['path'])
    if not path.is_file() or digest(path) != ref['sha256']:
        return [name + ': evidence missing or SHA mismatch']
    return []


def commissioning_scope(config):
    """Pin applicability to this task, controller, tool/TCP and command settings."""
    from .e2_ablation import object_hash
    return object_hash(dict(common={k: config['common'].get(k) for k in (
        'controller_deployment_id', 'expected_controller_sources', 'tcp_calibration_id',
        'approach_retract_protocol', 'workspace_limits', 'spindle_control_procedure')},
        task=config['task'], executor=config['executor'],
        postprocessor=config['motion_postprocessor'], calibration=config['calibration'],
        protection_version='e2_source_wrench_workspace_v1',
        required_driver_sources=config.get('pilot', {}).get('required_driver_sources')))


def commissioning_errors(config):
    """A record is necessary, but never sufficient to claim installed protection."""
    errors = installed_protection_errors()
    from .e2_ablation import digest
    sources = config.get('pilot', {}).get('required_driver_sources')
    if not isinstance(sources, dict) or not sources:
        errors.append('pilot.required_driver_sources: pinned driver patch required')
    else:
        for path, sha in sources.items():
            if not Path(path).is_file() or digest(path) != sha:
                errors.append('pilot.required_driver_sources: missing or changed source: ' + path)
    ref = config.get('pilot', {}).get('commissioning_record')
    missing = evidence_errors(ref, 'pilot.commissioning_record')
    if missing:
        return errors + missing
    try:
        record = json.loads(Path(ref['path']).read_text())
    except (OSError, ValueError) as exc:
        return errors + ['pilot.commissioning_record: invalid JSON: ' + str(exc)]
    if not isinstance(record, dict):
        return errors + ['pilot.commissioning_record: JSON object required']
    if (record.get('schema') != 'E2_A_commissioning_v1' or
            record.get('scope_sha256') != commissioning_scope(config)):
        errors.append('commissioning scope does not match current tool/controller/task/settings')
    if (record.get('status') != 'approved' or not record.get('approved_by') or
            not record.get('approved_at')):
        errors.append('commissioning status/approved_by/approved_at: actual approval required')
    if (record.get('command_frame'), record.get('command_axis'), record.get('command_unit')) != (
            'controller_tcp', 'Fz', 'N'):
        errors.append('commissioning command convention must be controller_tcp Fz in N')
    bounds = record.get('processing_command_fz_range_N')
    fz = config.get('f0', {}).get('command_fz_N')
    # A closed interval may authorize just one processing setpoint. It does not
    # change approach/exit zero requests, slew limits, or measured-wrench trips.
    if (not isinstance(bounds, list) or len(bounds) != 2 or
            not all(finite_number(x) for x in bounds) or bounds[0] > bounds[1] or
            not finite_number(fz) or not bounds[0] <= fz <= bounds[1]):
        errors.append('processing_command_fz_range_N: approved signed range must contain candidate F0')
    # These are measured-wrench protection limits, never derived from F0,
    # sensor ratings, past peaks, or an analysis threshold.
    from .e2_protection import LIMIT_KEYS
    for key in LIMIT_KEYS:
        limits = record.get(key)
        if (not isinstance(limits, list) or len(limits) != 3 or
                not all(finite_number(v) and v > 0 for v in limits)):
            errors.append(key + ': three approved positive per-axis limits required')
    if record.get('measured_wrench_frame') not in ('robot_base', 'controller_tcp', 'sensor'):
        errors.append('measured_wrench_frame: explicit protection frame required')
    age = record.get('acquisition_max_age_s')
    if not finite_number(age) or age <= 0:
        errors.append('acquisition_max_age_s: approved source-acquisition timeout required')
    for key in ('applicability_evidence', 'command_direction_evidence', 'limits_evidence',
                'sensor_overload_evidence', 'workspace_evidence',
                'automatic_stop_evidence', 'approach_exit_evidence', 'sensor_driver_deployment_evidence'):
        errors.extend(evidence_errors(record.get(key), key))
    return errors


def pilot_errors(config):
    from .e2_ablation import digest
    errors = commissioning_errors(config)
    pilot = config.get('pilot', {})
    f0 = config.get('f0', {})
    if (pilot.get('F0_frozen') is not False or pilot.get('quality_validation') != 'pending' or
            pilot.get('start_mode') != 'operator_existing_procedure'):
        errors.append('pilot must retain F0_frozen=false, quality_validation=pending and operator start')
    if f0.get('status') != 'candidate' or not finite_number(f0.get('command_fz_N')):
        errors.append('pilot requires finite candidate command F0; no frozen/main recipe')
    path = f0.get('candidate')
    if not path or not Path(path).is_file() or digest(path) != f0.get('candidate_sha256'):
        errors.append('pilot F0 candidate source missing or SHA mismatch')
    return errors


def pilot_force(config, processing):
    """Compile at preflight; never read approval files in the 125 Hz loop."""
    errors = pilot_errors(config)
    if errors:
        raise ValueError('; '.join(errors))
    return float(config['f0']['command_fz_N']) if processing else 0.
