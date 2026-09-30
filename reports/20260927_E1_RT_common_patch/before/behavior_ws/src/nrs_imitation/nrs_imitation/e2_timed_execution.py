"""ROS-free E2 timed executor, shared by R/T/C. No policy, I/O or guessed recipe.

Plans arrive in base mm/rotvec rad/N AFTER the original shared postprocessor.
The requested pose and force use one elapsed-time sample, never nearest XYZ.
The existing C++ stream's gain and per-axis rate caps are retained explicitly.
"""
from dataclasses import dataclass
import math
import socket
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


TRANSPORT_VERSION = 'e2_timed_topic_v1'
C_POSE_PROFILE = 'c_pose_conditioning_20260926_v1'


def clock_id():
    return socket.gethostname() + ':' + Path('/proc/sys/kernel/random/boot_id').read_text().strip()


@dataclass
class TimedPlan:
    plan_id: int
    generated_at: float
    time: np.ndarray
    action: np.ndarray
    phase: np.ndarray
    final: bool

    def __post_init__(self):
        self.time = np.asarray(self.time, np.float64)
        self.action = np.asarray(self.action, np.float64)
        self.phase = np.asarray(self.phase, str)
        if (self.time.ndim != 1 or len(self.time) < 2 or self.time[0] != 0 or
                self.action.shape != (len(self.time), 9) or self.phase.shape != self.time.shape or
                not np.isfinite(self.time).all() or not np.isfinite(self.action).all() or
                not np.isfinite(self.generated_at) or np.any(np.diff(self.time) <= 0)):
            raise ValueError('Invalid timed plan: finite unique increasing time[N] and base action[N,9] required')
        self.rotation = Slerp(self.time, Rotation.from_rotvec(self.action[:, 3:6]))

    def sample(self, elapsed):
        t = float(np.clip(elapsed, 0., self.time[-1]))
        hi = min(int(np.searchsorted(self.time, t, side='right')), len(self.time)-1)
        lo = max(0, hi-1)
        u = (t-self.time[lo])/(self.time[hi]-self.time[lo]) if hi != lo else 0.
        a = self.action[lo] + u*(self.action[hi]-self.action[lo])
        a[3:6] = self.rotation([t]).as_rotvec()[0]
        phase_index = min(int(np.searchsorted(self.time, t, side='right'))-1, len(self.time)-1)
        if t == self.time[phase_index]:
            a = self.action[phase_index].copy()
        return a, str(self.phase[phase_index]), phase_index


class ExecutionFault(RuntimeError):
    pass


class CPoseConditioner:
    """C-only pose preparation; never changes plan clocks or force samples.

    Restore the legacy 35-point pose smoothing, averaging orientation on SO(3)
    rather than across rotation-vector branch cuts. Blend a replacement from
    the previous reference and its velocity, then bound commanded acceleration.
    R/T do not instantiate this class: their recorded execution remains intact.
    """
    def __init__(self, settings):
        self.cfg = dict(settings)
        self.plan = None
        self.origin = None
        self.bridge_at = None
        self.bridge_pose = None
        self.bridge_velocity = np.zeros(6)
        self.velocity = np.zeros(6)

    def accept(self, plan, now):
        if plan.final:
            raise ExecutionFault('C pose conditioning cannot modify a finite R/T plan')
        if self.plan is not None and self.bridge_at is not None:
            # Estimate the left derivative on the existing reference, including
            # an unfinished bridge when plans arrive unusually close together.
            previous = self.sample(now)
            h = 1e-5
            before = self.sample(now-h)
            self.bridge_velocity[:3] = (previous[:3]-before[:3])/h
            self.bridge_velocity[3:] = (
                Rotation.from_rotvec(previous[3:])*Rotation.from_rotvec(before[3:]).inv()
            ).as_rotvec()/h
            self.bridge_pose, self.bridge_at = previous, now
        action = plan.action.copy()
        window = self.cfg['pose_smooth_window']
        half = window//2
        weights = np.ones(window)/window
        padded = np.pad(action[:, :3], ((half, half), (0, 0)), mode='edge')
        for axis in range(3):
            action[:, axis] = np.convolve(padded[:, axis], weights, mode='valid')
        # Batched Markley quaternion means: q and -q have the same outer
        # product, so +179/-179 degree samples average near pi, never near zero.
        q = Rotation.from_rotvec(action[:, 3:6]).as_quat()
        outer = np.einsum('ni,nj->nij', q, q).reshape(-1, 16)
        padded = np.pad(outer, ((half, half), (0, 0)), mode='edge')
        means = np.stack([np.convolve(padded[:, j], weights, mode='valid')
                          for j in range(16)], axis=1).reshape(-1, 4, 4)
        _, vectors = np.linalg.eigh(means)
        action[:, 3:6] = Rotation.from_quat(vectors[:, :, -1]).as_rotvec()
        self.plan = TimedPlan(plan.plan_id, plan.generated_at, plan.time.copy(),
                              action, plan.phase.copy(), plan.final)
        self.origin = plan.generated_at

    def start(self, now, pose):
        self.origin = self.bridge_at = now
        self.bridge_pose = np.asarray(pose, float).copy()
        self.bridge_velocity[:] = 0.
        self.velocity[:] = 0.

    def sample(self, now):
        target = self.plan.sample(now-self.origin)[0][:6].copy()
        if self.bridge_at is None:
            return target
        elapsed = max(0., now-self.bridge_at)
        u = min(1., elapsed/self.cfg['handover_s'])
        if u >= 1.:
            return target
        weight = u*u*u*(10.+u*(-15.+6.*u))
        anchor = self.bridge_pose.copy()
        anchor[:3] += self.bridge_velocity[:3]*elapsed
        anchor_rot = (Rotation.from_rotvec(self.bridge_velocity[3:]*elapsed)*
                      Rotation.from_rotvec(anchor[3:]))
        target[:3] = anchor[:3] + weight*(target[:3]-anchor[:3])
        error = (Rotation.from_rotvec(target[3:])*anchor_rot.inv()).as_rotvec()
        target[3:] = (Rotation.from_rotvec(weight*error)*anchor_rot).as_rotvec()
        return target

    def motion_step(self, error, proposed, period):
        """Slew the already speed-limited motion, retaining state on replan.

        The stopping-speed bound anticipates braking for a stationary target.
        Acceleration is per base XYZ / spatial rotation-vector axis. Neither
        the existing force gate nor the signed force-rate limit is touched.
        """
        if period <= 0.:
            return np.zeros(6)
        acceleration = np.r_[np.full(3, self.cfg['linear_acceleration_mm_s2']),
                             np.full(3, self.cfg['angular_acceleration_rad_s2'])]
        desired = proposed/period
        braking = (np.sqrt((acceleration*period)**2 + 8.*acceleration*np.abs(error))
                   - acceleration*period)/2.
        desired = np.clip(desired, -braking, braking)
        self.velocity += np.clip(desired-self.velocity, -acceleration*period, acceleration*period)
        return self.velocity*period


class TimedExecution:
    def __init__(self, settings):
        self.cfg = dict(settings)
        self.plan = None
        self.last_plan_id = 0
        self.start_time = self.plan_start = self.last_tick = None
        self.command = self.start_pose = None
        self.contact = False
        self.closed = False
        self.endpoint_sent = False
        profile = self.cfg.get('c_pose_conditioning')
        self.conditioner = CPoseConditioner(profile) if profile is not None else None

    def accept(self, plan, now):
        if self.closed:
            raise ExecutionFault('executor is terminal; restart with a new session')
        if plan.plan_id <= self.last_plan_id:
            raise ExecutionFault('duplicate or out-of-order plan ID')
        age = now-plan.generated_at
        if age < -.001 or age > self.cfg['max_plan_age_s']:
            raise ExecutionFault('stale plan or mismatched monotonic clock')
        if self.plan is not None and self.plan.final:
            raise ExecutionFault('finite R/T trajectory cannot be replaced or repeated')
        if self.conditioner is not None:
            self.conditioner.accept(plan, now)
        self.plan, self.last_plan_id = plan, plan.plan_id
        self.endpoint_sent = False
        # First plan starts ONLY after observed Force mode. Subsequent C plans
        # retain their generation clock so middleware delay cannot stretch time.
        if self.start_time is not None:
            self.plan_start = plan.generated_at

    def start(self, now, pose):
        if self.plan is None or self.start_time is not None or self.closed:
            raise ExecutionFault('invalid executor start')
        self.start_time = self.plan_start = self.last_tick = float(now)
        self.command = np.r_[np.asarray(pose, float)[:6], np.zeros(3)]
        self.start_pose = self.command[:6].copy()
        self.contact = False
        if self.conditioner is not None:
            self.conditioner.start(now, pose)

    def stop(self):
        self.closed = True
        self.plan = None

    def _check_workspace(self, action, pose):
        delta = action[:3]-self.start_pose[:3]
        limits = self.cfg['workspace_limits']
        if (np.linalg.norm(delta[:2]) > limits['max_xy_from_start_mm'] or
                -delta[2] > limits['max_z_down_from_start_mm'] or
                delta[2] > limits['max_z_up_from_start_mm'] or
                np.linalg.norm(action[:3]-pose[:3]) > limits['max_xyz_from_current_mm']):
            raise ExecutionFault('requested action outside common workspace envelope')

    def tick(self, now, pose, force, pose_age, force_age, external_force_fz=None):
        if self.closed or self.start_time is None or self.plan is None:
            return None
        if (pose_age < 0 or force_age < 0 or max(pose_age, force_age) > self.cfg['feedback_max_age_s']):
            raise ExecutionFault('stale pose/force feedback')
        pose, force = np.asarray(pose, float), np.asarray(force, float)
        if pose.shape != (6,) or len(force) < 3 or not np.isfinite(pose).all() or not np.isfinite(force).all():
            raise ExecutionFault('invalid feedback')
        dt = now-self.last_tick
        if dt < 0 or dt > self.cfg['max_tick_gap_s']:
            raise ExecutionFault('control timer deadline missed')
        elapsed = now-self.plan_start
        if now-self.start_time > self.cfg['task_time_budget_s']:
            return {'end': 'timeout'}
        if elapsed > self.plan.time[-1] and (not self.plan.final or self.endpoint_sent):
            return {'end': 'trajectory_end' if self.plan.final else 'plan_underrun',
                    'last_target': self.plan.action[-1, :6].copy()}
        # Consume a finite trajectory's exact last knot once, even when the
        # original final interval is shorter than a control tick. Its timestamp
        # stays unchanged; record the actual scheduler lateness separately.
        if self.plan.final and elapsed >= self.plan.time[-1]:
            self.endpoint_sent = True
        raw, phase, index = self.plan.sample(elapsed)
        if external_force_fz is not None:
            if not np.isfinite(external_force_fz):
                raise ExecutionFault('invalid external force scheduler output')
            raw[6:9] = (0., 0., float(external_force_fz))
        conditioned = raw.copy()
        if self.conditioner is not None:
            conditioned[:6] = self.conditioner.sample(now)
        for action in (raw, conditioned) if self.conditioner is not None else (raw,):
            self._check_workspace(action, pose)
        previous_contact = self.contact
        if not self.contact and force[2] >= self.cfg['contact_on_N']:
            self.contact = True
        elif self.contact and force[2] <= self.cfg['contact_off_N']:
            self.contact = False
        target = conditioned.copy()
        target[6:8] = 0.  # same disabled tangential commands as the pinned C
        if not self.contact:
            target[8] = 0.
        fmax = self.cfg['fz_hard_limit_N']
        if fmax > 0:
            target[8] = np.clip(target[8], -fmax, fmax)
        # Explicit common filtering: exact legacy per-axis gain/rate caps.
        # Template export stays unsmoothed; requested and sent rows are distinct.
        period = min(max(dt, 0.), self.cfg['control_period_s'])
        change = (target-self.command)*min(1., self.cfg['gain_hz']*period)
        caps = np.r_[np.full(3,self.cfg['linear_speed_mm_s']),
                     np.full(3,self.cfg['angular_speed_rad_s']),np.full(3,self.cfg['force_rate_N_s'])]*period
        change = np.clip(change,-caps,caps)
        # Rotation vectors have a branch cut at pi. Apply a bounded physical
        # SO(3) increment, not a long component-wise trip through the branch.
        current_rot = Rotation.from_rotvec(self.command[3:6])
        angular_error = (Rotation.from_rotvec(target[3:6])*current_rot.inv()).as_rotvec()
        angular_step = np.clip(angular_error*min(1.,self.cfg['gain_hz']*period), -caps[3:6], caps[3:6])
        if self.conditioner is not None:
            motion = self.conditioner.motion_step(np.r_[target[:3]-self.command[:3], angular_error],
                                                 np.r_[change[:3], angular_step], period)
            change[:3], angular_step = motion[:3], motion[3:]
        self.command += change
        self.command[3:6] = (Rotation.from_rotvec(angular_step)*current_rot).as_rotvec()
        self.command[6:8] = 0.
        if self.conditioner is not None:
            # Slew-limited braking can carry momentum beyond a moving target.
            # Never publish that step outside the unchanged common envelope.
            self._check_workspace(self.command, pose)
        self.last_tick = now
        return dict(requested=raw, conditioned=conditioned, gated=target, sent=self.command.copy(), phase=phase,
                    plan_id=self.plan.plan_id, action_index=index, elapsed_s=elapsed,
                    endpoint_lateness_s=max(0., elapsed-self.plan.time[-1]),
                    contact=self.contact, limited=bool(np.any(self.command != raw)), dt_s=dt,
                    contact_transition=(self.contact != previous_contact),
                    gate_reason='base_fz_hysteresis', pose_age_s=pose_age, force_age_s=force_age,
                    force_source='external_F0' if external_force_fz is not None else 'provider',
                    conditioning_profile=C_POSE_PROFILE if self.conditioner is not None else None)


class StopVerifier:
    """Receipt-fresh observed controller mode + stationary TCP over a window.

    This verifies a controlled hold, NOT contact release or spindle/retraction.
    A service acknowledgement alone never satisfies it.
    """
    def __init__(self, since, window_s=.4, position_mm=.1, rotation_rad=.002, max_age_s=.2):
        self.since, self.window = since, window_s
        self.position_mm, self.rotation_rad, self.max_age = position_mm, rotation_rad, max_age_s
        self.samples = []

    def observe(self, now, pose, mode, mode_at, pose_at):
        if (mode not in ('Position','Idling') or mode_at <= self.since or pose_at <= self.since or
                not 0 <= now-mode_at <= self.max_age or not 0 <= now-pose_at <= self.max_age):
            self.samples.clear(); return False
        p = np.asarray(pose,float)
        if p.shape != (6,) or not np.isfinite(p).all():
            self.samples.clear(); return False
        if self.samples and pose_at <= self.samples[-1][0]:
            return False  # repeated callbacks are not distinct feedback samples
        if self.samples and pose_at-self.samples[-1][0] > self.max_age:
            self.samples.clear()  # a feedback outage is not stationary evidence
        self.samples.append((pose_at,p.copy()))
        while len(self.samples)>2 and self.samples[1][0] <= now-self.window:
            self.samples.pop(0)
        start = self.samples[0][1]
        for _, q in self.samples:
            angle=(Rotation.from_rotvec(q[3:])*Rotation.from_rotvec(start[3:]).inv()).magnitude()
            if np.linalg.norm(q[:3]-start[:3]) > self.position_mm or angle > self.rotation_rad:
                self.samples=[self.samples[-1]];return False
        return len(self.samples)>=3 and self.samples[-1][0]-self.samples[0][0] >= self.window


def executor_settings(config, method=None):
    """Only explicit runtime settings; defaults below are sourced control constants."""
    settings = dict(config['executor'], workspace_limits=config['common']['workspace_limits'],
                    task_time_budget_s=config['common']['task_time_budget_s'])
    if method == 'il' and 'pose_conditioning' in config.get('il', {}):
        settings['c_pose_conditioning'] = dict(config['il']['pose_conditioning'])
    return settings


def validate_settings(config):
    errors = []
    settings = config.get('executor', {})
    positive = ('control_period_s', 'gain_hz', 'linear_speed_mm_s', 'angular_speed_rad_s',
        'force_rate_N_s', 'feedback_max_age_s', 'max_tick_gap_s', 'max_plan_age_s',
        'mode_timeout_s', 'provider_timeout_s', 'stop_window_s', 'stop_timeout_s',
        'completion_position_tolerance_mm', 'completion_rotation_tolerance_rad')
    for key in positive:
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not np.isfinite(value) or value <= 0:
            errors.append('Invalid positive finite executor.' + key)
    for key in ('contact_on_N', 'contact_off_N', 'fz_hard_limit_N'):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not np.isfinite(value):
            errors.append('Invalid finite executor.' + key)
    if errors:
        return errors
    if not settings['contact_off_N'] < settings['contact_on_N']:
        errors.append('Invalid contact hysteresis')
    if not np.isclose(settings['control_period_s'], .008):
        errors.append('E2 controller period must match 125 Hz')
    if not settings['control_period_s'] < settings['max_tick_gap_s'] <= settings['feedback_max_age_s']:
        errors.append('Invalid deadline/feedback freshness bounds')
    if not settings['stop_window_s'] < settings['stop_timeout_s'] <= 15.:
        errors.append('Stop timeout must cover observation window and fit launch shutdown grace (15s max)')
    limits = config.get('common',{}).get('workspace_limits') or {}
    for key in ('max_xy_from_start_mm','max_z_down_from_start_mm','max_z_up_from_start_mm','max_xyz_from_current_mm'):
        value = limits.get(key)
        if not isinstance(value,(int,float)) or not np.isfinite(value) or value <= 0:
            errors.append('Invalid common.workspace_limits.'+key)
    profile = config.get('il', {}).get('pose_conditioning')
    if profile is not None:
        if not isinstance(profile, dict):
            return errors + ['Invalid il.pose_conditioning']
        if profile.get('profile') != C_POSE_PROFILE:
            errors.append('Unsupported il.pose_conditioning.profile')
        window = profile.get('pose_smooth_window')
        if isinstance(window, bool) or not isinstance(window, int) or not 3 <= window <= 127 or window % 2 != 1:
            errors.append('il.pose_conditioning.pose_smooth_window must be an odd integer in [3, 127]')
        for key in ('handover_s', 'linear_acceleration_mm_s2', 'angular_acceleration_rad_s2'):
            value = profile.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                errors.append('Invalid positive finite il.pose_conditioning.'+key)
    return errors
