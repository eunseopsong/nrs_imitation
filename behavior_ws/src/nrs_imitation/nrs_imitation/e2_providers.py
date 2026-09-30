"""Hardware-free E2 action providers. Native contract: mm, rotation-vector rad, N.

The frame is stain_relative_v1 (relative XY, absolute base Z/orientation).
No normalizer, hidden scaling, contact decision, or ROS I/O belongs here.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@dataclass
class TimedActions:
    time: np.ndarray
    action: np.ndarray
    phase: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=np.float64)
        self.action = np.asarray(self.action, dtype=np.float32)
        self.phase = np.asarray(self.phase, dtype=str)
        if (self.time.ndim != 1 or len(self.time) < 2 or
                self.action.shape != (len(self.time), 9) or
                self.phase.shape != self.time.shape):
            raise ValueError('Expected unique time[N], action[N,9], phase[N], N >= 2')
        if (not np.isfinite(self.time).all() or not np.isfinite(self.action).all()
                or self.time[0] != 0 or np.any(np.diff(self.time) <= 0)):
            raise ValueError('Nonfinite actions or non-increasing/ nonzero-origin time')
        if self.metadata.get('frame') != 'stain_relative_v1':
            raise ValueError('Unsupported action frame; no automatic frame conversion')

    def sample(self, times):
        t = np.asarray(times, dtype=np.float64)
        if not np.isfinite(t).all() or np.any(t < 0) or np.any(t > self.time[-1]):
            raise ValueError('Replay outside original time span is prohibited')
        out = np.stack([np.interp(t, self.time, self.action[:, i]) for i in range(9)], axis=-1)
        out[:, 3:6] = Slerp(self.time, Rotation.from_rotvec(self.action[:, 3:6]))(t).as_rotvec()
        # Preserve the exact stored representation at original knots.
        indices = np.searchsorted(self.time, t)
        exact = self.time[indices] == t
        out[exact] = self.action[indices[exact]]
        phase = self.phase[np.maximum(0, np.searchsorted(self.time, t, side='right') - 1)]
        return out.astype(np.float32), phase

    def resample(self, hz):
        if not np.isfinite(hz) or hz <= 0:
            raise ValueError('hz must be positive')
        times = np.arange(int(np.floor(self.time[-1] * hz)) + 1, dtype=float) / hz
        if times[-1] < self.time[-1]:
            times = np.append(times, self.time[-1])
        action, phase = self.sample(times)
        return TimedActions(times, action, phase, dict(self.metadata, resample_hz=hz,
            original_duration_s=float(self.time[-1]), final_interval_may_be_short=True))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Never silently overwrite a selected template.
        with path.open('xb') as f:
            np.savez_compressed(f, time=self.time, action=self.action, phase=self.phase,
                metadata=np.array(json.dumps(self.metadata, ensure_ascii=False)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            return cls(f['time'], f['action'], f['phase'], json.loads(str(f['metadata'])))


def _rule_approach(task, recipe, hz):
    """Explicit, source-backed approach waypoints, at rest at each waypoint.

    No surface height/clearance is inferred here. The caller supplies all six
    pose coordinates, including the common demo start and first working pose.
    """
    spec = recipe.get('approach')
    if spec is None:
        return None
    poses = np.asarray(spec.get('waypoints_pose6'), float)
    if (poses.ndim != 2 or poses.shape[1] != 6 or len(poses) < 2 or
            not np.isfinite(poses).all()):
        raise ValueError('Approach requires finite waypoints_pose6[N,6], N >= 2')
    linear = float(spec['peak_speed_mm_s'])
    angular = float(spec['peak_angular_speed_rad_s'])
    if not np.isfinite([linear, angular]).all() or min(linear, angular) <= 0:
        raise ValueError('Approach peak speeds must be finite and positive')
    target = np.r_[task['endpoints_relative_mm'][0], task['orientation_rotvec_rad']]
    if (not np.allclose(poses[-1, :3], target[:3], atol=1e-5, rtol=0) or
            (Rotation.from_rotvec(poses[-1, 3:]).inv() *
             Rotation.from_rotvec(target[3:])).magnitude() > 1e-5):
        raise ValueError('Approach must end at the first rule working pose')
    return pose_waypoint_actions(poses, linear, angular, hz)


def pose_waypoint_actions(poses, linear, angular, hz=125.):
    """Zero-force quintic pose path through explicit native-unit waypoints."""
    poses = np.asarray(poses, float)
    if (poses.ndim != 2 or poses.shape[1] != 6 or len(poses) < 2 or
            not np.isfinite(poses).all() or not np.isfinite([linear, angular, hz]).all() or
            min(linear, angular, hz) <= 0):
        raise ValueError('Finite pose waypoints and positive speed/sample rates required')
    times, actions = [0.], [np.r_[poses[0], np.zeros(3)]]
    boundaries = [0.]
    for start, end in zip(poses[:-1], poses[1:]):
        delta = (Rotation.from_rotvec(start[3:]).inv() * Rotation.from_rotvec(end[3:])).magnitude()
        duration = 1.875 * max(np.linalg.norm(end[:3]-start[:3])/linear, delta/angular)
        if duration <= 1e-12:
            continue
        local = np.r_[np.arange(1./hz, duration, 1./hz), duration]
        u = local/duration
        s = u*u*u*(10.+u*(-15.+6.*u))
        a = np.zeros((len(local), 9))
        a[:, :3] = start[:3]+s[:, None]*(end[:3]-start[:3])
        a[:, 3:6] = Slerp([0., 1.], Rotation.from_rotvec([start[3:], end[3:]]))(s).as_rotvec()
        a[-1, :6] = end
        times.extend(times[-1]+local)
        actions.extend(a)
        boundaries.append(times[-1])
    if len(times) < 2:
        times.append(1./hz); actions.append(actions[0].copy())
    return np.asarray(times), np.asarray(actions), boundaries


def rule_actions(task, recipe, hz=125.):
    """Quintic rest-to-rest centerline strokes; geometry/force are explicit inputs.

    feed_speed_mm_s is the peak, not average, speed (quintic peak is 1.875).
    An explicit waypoint approach can precede the strokes, with zero requested
    force. No guessed lift, contact depth, or return-to-home is generated.
    """
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError('hz must be positive')
    if task['task_type'] != 'line':
        raise ValueError('Only the image-confirmed line task is implemented')
    if recipe.get('offset_spacing_mm', 0) != 0:
        raise ValueError('Parallel coverage requires a validated contact-width recipe')
    ends = np.asarray(task['endpoints_relative_mm'], float)
    rot = np.asarray(task['orientation_rotvec_rad'], float)
    if ends.shape != (2, 3) or rot.shape != (3,) or not np.isfinite(ends).all() or not np.isfinite(rot).all():
        raise ValueError('Explicit finite endpoints[2,3] and rotvec[3] are required')
    speed, ramp = float(recipe['feed_speed_mm_s']), float(recipe['force_ramp_s'])
    force = float(recipe['force_reference_N'])
    count = recipe['pass_count']
    if (not np.isfinite([speed, ramp, force]).all() or speed <= 0 or ramp <= 0 or
            isinstance(count, bool) or int(count) != count or count < 1):
        raise ValueError('Invalid fixed recipe')
    count = int(count)
    distance = np.linalg.norm(ends[1] - ends[0])
    if distance <= 0:
        raise ValueError('Zero line length')
    duration = 1.875 * distance / speed
    approach = _rule_approach(task, recipe, hz)
    approach_duration = 0. if approach is None else float(approach[0][-1])
    total = 2 * ramp + int(count) * duration
    budget = float(recipe['task_time_budget_s'])
    if not np.isfinite(budget) or total + approach_duration > budget:
        raise ValueError('Recipe exceeds the common time budget; no automatic truncation')
    # Include all phase boundaries and final time exactly.
    t = np.unique(np.r_[np.arange(0., total, 1./hz), 0., ramp,
                        ramp + np.arange(1, count + 1) * duration, total])
    a = np.zeros((len(t), 9), np.float32)
    a[:, 3:6] = rot
    phase = np.full(len(t), 'processing', dtype='<U24')
    smooth = lambda u: u*u*u*(10. + u*(-15. + 6.*u))
    for i, ti in enumerate(t):
        if ti < ramp:
            a[i, :3] = ends[0]; a[i, 8] = force * smooth(ti/ramp)
            phase[i] = 'force_ramp_in'
        elif ti < ramp + count*duration:
            k = min(count - 1, int((ti-ramp)/duration))
            u = np.clip((ti-ramp-k*duration)/duration, 0., 1.)
            a[i, :3] = ends[k % 2] + smooth(u)*(ends[(k+1) % 2] - ends[k % 2])
            a[i, 8] = force
        else:
            a[i, :3] = ends[count % 2]
            a[i, 8] = force * (1. - smooth(np.clip((ti-ramp-count*duration)/ramp, 0., 1.)))
            phase[i] = 'force_ramp_out'
    if approach is not None:
        at, aa, _ = approach
        # One shared knot at the approach/ramp boundary; never start the stroke
        # clock while the tool is still descending from the demo start pose.
        t = np.r_[at[:-1], t+approach_duration]
        a = np.vstack([aa[:-1], a])
        phase = np.r_[np.full(len(at)-1, 'approach'), phase]
    return TimedActions(t, a, phase, dict(method='R', frame='stain_relative_v1',
        units=['mm', 'rotvec_rad', 'N'], task_type='line', recipe=recipe, task=task,
        simulation_only=bool(recipe.get('simulation_only', False)), smoothing='none',
        approach_duration_s=approach_duration,
        approach_waypoint_times_s=[] if approach is None else approach[2],
        approach_retract=('explicit approach; verified Position lift and home return after work'
            if recipe.get('return_home', {}).get('enabled') else
            'explicit approach if supplied; common verified hold, manual retract'),
        pass_definition='one endpoint-to-endpoint stroke; two passes = one round trip'))


def export_episode(episode, train_names, *, selected_success=False, evidence=None):
    """Export each stored action exactly once; recover matching original clock.

    Success is never inferred from length or force. Candidate exports are safe
    for previews, but cannot be used as selected references without evidence.
    """
    import h5py
    episode = Path(episode).resolve()
    if episode.name not in train_names:
        raise ValueError('Reference must be in the reconstructed C training split')
    if selected_success and not evidence:
        raise ValueError('Successful-reference selection requires operator evidence')
    with h5py.File(episode, 'r') as f:
        if f.attrs.get('relative_transform_version') != 'stain_relative_v1' or f.attrs.get('rotation_aligned', 0):
            raise ValueError('Unsupported relative transform')
        if f.attrs.get('truncated', 0):
            raise ValueError('Truncated episode requires an explicit audited index mapping')
        pos, force = f['action/position'][:], f['action/force'][:]
        source_path, source_name = str(f.attrs['source_h5']), str(f.attrs['source_episode'])
        correction = np.asarray(f.attrs['xyz_correction_mm'], float)
        origin = np.asarray(f.attrs['stain_origin_xy_mm'], float)
        absolute = f['analysis/absolute/action_position'][:]
    with h5py.File(source_path, 'r') as f:
        g = f['episodes/' + source_name]
        t_unix = g['sample_time_unix'][:]
        source_index = g['source_index'][:]
        raw_pos, filtered_force = g['position'][:], g['ft'][:]
        # Verified converter uses filtered ft unchanged as BOTH obs and action.
        if not np.array_equal(force, filtered_force):
            raise ValueError('Source filtered force does not match training action; conversion unverified')
        if len(t_unix) != len(pos) or not np.array_equal(source_index, np.arange(len(pos))):
            raise ValueError('Original timestamp/index mapping is not one-to-one')
        expected = raw_pos.copy(); expected[:, :3] += correction
        if not np.allclose(expected, absolute, atol=5e-5, rtol=0):
            raise ValueError('Teacher-to-robot correction does not match stored absolute pose')
        from stain_relative_frame.relative_frame import to_relative
        if not np.allclose(to_relative(absolute, origin), pos, atol=5e-5, rtol=0):
            raise ValueError('Relative pose mapping mismatch')
        clock = str(g.attrs['sync_clock'])
        hz = float(g.attrs['record_hz'])
    return TimedActions(t_unix-t_unix[0], np.column_stack([pos, force]),
        np.full(len(pos), 'unknown', dtype='<U24'), dict(method='T', frame='stain_relative_v1',
        units=['mm', 'rotvec_rad', 'N'], task_type='line', source_episode=str(episode),
        episode_sha256=file_hash(episode), source_h5=source_path, source_group=source_name,
        source_timestamp_key='sample_time_unix', source_clock=clock,
        source_time_origin_unix=float(t_unix[0]), source_record_hz=hz,
        source_index_mapping='identity, verified all unique timesteps',
        source_action_sha256=hashlib.sha256(np.column_stack([pos, force]).tobytes()).hexdigest(),
        force_conversion='source ft (already EMA filtered) copied to action/force, unchanged sign; no new smoothing',
        teacher_force_topic='/ftsensor/measured_Cvalue', force_frame='teacher calibrated axes; live equivalence unverified',
        xyz_correction_mm=correction.tolist(), source_origin_xy_mm=origin.tolist(),
        rotation_aligned=False, scaling=1., smoothing='none', phase_source=None, tool_state=None,
        selected_success=bool(selected_success), selection_evidence=evidence,
        split_provenance='seed 0 loader split reconstructed and train extrema compared; no historical ID list saved'))


def load_config(path):
    with Path(path).open() as f:
        return json.load(f)


def hardware_blockers(config, method, enabled=False):
    """Fail closed. Do not represent an operator checkbox as transport support."""
    errors = []
    if method not in ('il', 'rule', 'replay'):
        return ['Unknown execution_method']
    if not enabled:
        errors.append('Explicit e2_enable_hardware:=true is required')
    common = config.get('common', {})
    if common.get('frame') != 'stain_relative_v1':
        errors.append('Common frame must match C stain_relative_v1')
    for key in ('spindle_control_procedure',
                'tcp_calibration_id', 'force_frame_sign_verification', 'frozen_roi',
                'workspace_limits', 'task_time_budget_s', 'approach_retract_protocol',
                'controller_deployment_id'):
        if common.get(key) is None:
            errors.append('Missing common.' + key)
    # RPM is logging metadata; the executor never drives the spindle. Explicit
    # operator-reported unknown is valid and must not become an invented value.
    if common.get('rpm_setpoint') is None and not (
            common.get('rpm_status') == 'unknown' and
            isinstance(common.get('rpm_unknown_reason'), str) and common['rpm_unknown_reason'].strip()):
        errors.append('Missing common.rpm_setpoint (or explicit rpm_status=unknown with reason)')
    for key in ('rpm_setpoint', 'task_time_budget_s'):
        value = common.get(key)
        if value is not None and (not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0):
            errors.append('Invalid positive finite common.' + key)
    if common.get('surface_normal_base') is not None:
        normal = np.asarray(common['surface_normal_base'], float)
        if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1.):
            errors.append('Surface normal must be a calibrated unit 3-vector')
    roi = common.get('frozen_roi')
    if isinstance(roi, dict):
        path = roi.get('detect_params_file')
        if not path or not Path(path).is_file() or file_hash(path) != roi.get('sha256'):
            errors.append('Frozen ROI detection file identity mismatch')
    else:
        errors.append('common.frozen_roi must pin the actual detection parameter file and hash')
    if config.get('task', {}).get('task_type') != 'line':
        errors.append('Only image-confirmed line task supported')
    if common.get('stain_canon_enable') is not False or common.get('use_stain_mask') is not False:
        errors.append('Preserve C stain_canon_enable=false and use_stain_mask=false')
    if method == 'rule':
        recipe = config.get('recipe', {})
        if recipe.get('simulation_only', True):
            errors.append('Rule recipe is offline-only: review the concrete recipe before authorizing a validation run')
        for key in ('endpoints_relative_mm', 'orientation_rotvec_rad'):
            if config.get('task', {}).get(key) is None:
                errors.append('Missing task.' + key)
        for key in ('force_reference_N', 'feed_speed_mm_s', 'pass_count',
                    'force_ramp_s', 'task_time_budget_s', 'approach'):
            if recipe.get(key) is None:
                errors.append('Missing recipe.' + key)
    if method == 'replay':
        template_path = config.get('replay', {}).get('template')
        if not template_path or not Path(template_path).is_file():
            errors.append('No explicitly selected successful replay template')
        else:
            meta = TimedActions.load(template_path).metadata
            if not meta.get('selected_success') or not meta.get('selection_evidence'):
                errors.append('Replay template is an unconfirmed preview candidate')
            if meta.get('task_type') != config['task']['task_type']:
                errors.append('Replay task differs from common task')
            if meta.get('normalizer_sha256') != config.get('il', {}).get('normalizer_sha256'):
                errors.append('Replay reference normalizer differs from C')
            source = meta.get('source_episode')
            if not source or not Path(source).is_file() or file_hash(source) != meta.get('episode_sha256'):
                errors.append('Replay source episode identity mismatch')
    from .e2_ablation import is_ablation, runtime_contract_errors
    if is_ablation(config):
        if method != 'il':
            errors.append('E2 A/B/C must all use learned IL providers')
        errors.extend(runtime_contract_errors(config))
    elif config.get('il', {}).get('use_force_observation') is not True or config.get('il', {}).get('force_action') is not True:
        errors.append('E2 C must include measured force observation and learned force action')
    ckpt = config.get('il', {}).get('checkpoint')
    model_label = config.get('condition', 'C') if is_ablation(config) else 'C'
    if not ckpt or not Path(ckpt).is_file():
        errors.append(f'Explicit {model_label} checkpoint missing; automatic fallback prohibited')
    elif config['il'].get('checkpoint_sha256') != file_hash(ckpt):
        errors.append(f'{model_label} checkpoint identity mismatch')
    from .e2_timed_execution import TRANSPORT_VERSION, validate_settings
    if config.get('executor', {}).get('transport') != TRANSPORT_VERSION:
        errors.append('Missing or unsupported executor.transport')
    errors.extend(validate_settings(config))
    if config.get('il', {}).get('inference_mode') != 'timed_topic':
        errors.append('All E2 methods require timed_topic transport')
    demo = np.asarray(common.get('demo_start_pose6'), float)
    if demo.shape != (6,) or not np.isfinite(demo).all():
        errors.append('Invalid common.demo_start_pose6')
    if ckpt and Path(ckpt).is_file():
        stats_path = Path(ckpt).with_name('dataset_stats.pkl')
        if not stats_path.is_file() or file_hash(stats_path) != config['il'].get('normalizer_sha256'):
            errors.append(f'{model_label} normalizer identity mismatch')
    if method == 'rule':
        from .e2_return_home import validate_return_home
        errors.extend(validate_return_home(config))
        try:
            actions = rule_actions(config['task'], config['recipe'])
            if actions.time[-1] > float(common['task_time_budget_s']):
                errors.append('Rule trajectory exceeds common time budget')
            if demo.shape == (6,) and not np.allclose(actions.action[0, :6], demo, atol=1e-4, rtol=0):
                errors.append('Rule approach must start at common.demo_start_pose6')
            approach = config['recipe'].get('approach')
            executor = config['executor']
            if approach is not None and (
                    approach['peak_speed_mm_s'] > executor['linear_speed_mm_s'] or
                    approach['peak_angular_speed_rad_s'] > executor['angular_speed_rad_s']):
                errors.append('Rule approach peak speed exceeds common executor rate limit')
            if config['recipe']['feed_speed_mm_s'] > executor['linear_speed_mm_s']:
                errors.append('Rule feed peak speed exceeds common executor rate limit')
            if 1.875*abs(config['recipe']['force_reference_N'])/config['recipe']['force_ramp_s'] > executor['force_rate_N_s']+1e-6:
                errors.append('Rule force ramp exceeds common executor force rate limit')
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            errors.append('Invalid rule recipe: ' + str(exc))
    if method == 'replay' and template_path and Path(template_path).is_file():
        replay = TimedActions.load(template_path)
        if common.get('task_time_budget_s') is not None and replay.time[-1] > common['task_time_budget_s']:
            errors.append('Replay exceeds common time budget; no time scaling allowed')
        if config['replay'].get('scaling') != 1. or config['replay'].get('smoothing') != 'none':
            errors.append('Replay scaling/smoothing must remain 1.0/none')
    return errors
