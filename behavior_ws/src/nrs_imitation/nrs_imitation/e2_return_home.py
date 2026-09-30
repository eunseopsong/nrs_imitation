"""Feedback-verified Position-mode lift and home return; no ROS or force mode.

Construct ONLY after the task queue was cancelled and a Position hold verified.
Lift at measured XY to the explicit home height before any return XY motion.
The same timed sampler, gain and rate caps as task execution remain in use.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .e2_providers import pose_waypoint_actions
from .e2_timed_execution import TimedExecution, TimedPlan, StopVerifier, ExecutionFault


def validate_return_home(config):
    spec = config.get('recipe', {}).get('return_home')
    if not spec or spec.get('enabled') is False:
        return []
    errors = []
    if spec.get('enabled') is not True:
        errors.append('recipe.return_home.enabled must be boolean')
    if spec.get('home_source') != 'common.demo_start_pose6':
        errors.append('Return home must use the explicitly selected common demo-start pose')
    for key in ('timeout_s', 'verification_timeout_s', 'release_abs_fz_N'):
        v = spec.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0:
            errors.append('Invalid positive recipe.return_home.'+key)
    if errors:
        return errors
    home = np.asarray(config.get('common', {}).get('demo_start_pose6'), float)
    ends = np.asarray(config.get('task', {}).get('endpoints_relative_mm'), float)
    if home.shape != (6,) or ends.shape != (2, 3) or home[2] <= np.max(ends[:, 2]):
        errors.append('Return home clearance must be above the rule working endpoints')
    if spec['release_abs_fz_N'] > config['executor']['contact_off_N']:
        errors.append('Return release threshold must not relax the existing contact-OFF threshold')
    if spec['verification_timeout_s'] <= config['executor']['stop_window_s']:
        errors.append('Return verification timeout must cover the stationary observation window')
    return errors


class ReturnHome:
    def __init__(self, settings, spec, start_pose, home_pose, now):
        self.cfg, self.spec = dict(settings), dict(spec)
        self.home = np.asarray(home_pose, float).copy()
        start = np.asarray(start_pose, float)
        if (start.shape != (6,) or self.home.shape != (6,) or
                not np.isfinite(start).all() or not np.isfinite(self.home).all()):
            raise ExecutionFault('Invalid measured/home pose for return')
        if start[2] > self.home[2]+self.cfg['completion_position_tolerance_mm']:
            raise ExecutionFault('Return cannot descend before contact release')
        self.started_at = float(now)
        self.released = False
        self.complete = False
        self.events = []
        self.lift = start.copy()
        self.lift[2] = max(start[2], self.home[2])
        self._segment(start, self.lift, 'retract', now, 1)

    def _segment(self, start, target, phase, now, plan_id):
        times, actions, _ = pose_waypoint_actions([start, target],
            self.cfg['linear_speed_mm_s'], self.cfg['angular_speed_rad_s'],
            1./self.cfg['control_period_s'])
        self.motion_duration = float(times[-1])
        # Keep publishing the same target while feedback catches up and the
        # release/arrival window is verified; then fail if it never verifies.
        times = np.r_[times, times[-1]+self.spec['verification_timeout_s']]
        actions = np.vstack([actions, actions[-1]])
        self.target = np.asarray(target, float).copy()
        self.phase, self.phase_at = phase, float(now)
        cfg = dict(self.cfg, task_time_budget_s=self.spec['timeout_s'])
        self.engine = TimedExecution(cfg)
        self.engine.accept(TimedPlan(plan_id, now, times, actions,
            np.full(len(times), phase), True), now)
        self.engine.start(now, start)
        self.verifier = StopVerifier(now, window_s=self.cfg['stop_window_s'],
            max_age_s=self.cfg['feedback_max_age_s'])
        self.force_last_at = None
        self.events.append(dict(event='return_phase_started', phase=phase,
            start_pose=np.asarray(start).tolist(), target_pose=self.target.tolist(),
            nominal_motion_s=self.motion_duration, controller_mode='Position'))

    def tick(self, now, pose, force, pose_at, force_at, mode, mode_at):
        if self.complete:
            return dict(done=True, contact_release_verified=True, home_reached=True)
        if now-self.started_at > self.spec['timeout_s']:
            raise ExecutionFault('Return home total timeout')
        if mode != 'Position' or not 0 <= now-mode_at <= self.cfg['feedback_max_age_s']:
            raise ExecutionFault('Return requires fresh observed Position mode')
        pose, force = np.asarray(pose, float), np.asarray(force, float)
        if self.phase == 'return_home' and (len(force) < 3 or abs(force[2]) >= self.cfg['contact_on_N']):
            raise ExecutionFault('Contact detected again during home return')
        result = self.engine.tick(now, pose, force, now-pose_at, now-force_at)
        if result is None or 'end' in result:
            raise ExecutionFault(self.phase+' arrival/contact-release verification timeout')
        result['controller_mode'] = 'Position'
        result['done'] = False
        arrived = (np.linalg.norm(pose[:3]-self.target[:3]) <= self.cfg['completion_position_tolerance_mm'] and
            (Rotation.from_rotvec(pose[3:])*Rotation.from_rotvec(self.target[3:]).inv()).magnitude()
                <= self.cfg['completion_rotation_tolerance_rad'])
        # Fz=0 requested alone does not prove release. Require fresh low |Fz|
        # plus a stationary pose at clearance/at home for a full stop window.
        low_force = abs(force[2]) <= self.spec['release_abs_fz_N']
        if (not arrived or not low_force or now-self.phase_at < self.motion_duration or
                force_at <= self.phase_at):
            self.verifier.samples.clear()
        else:
            if self.force_last_at is not None and force_at-self.force_last_at > self.cfg['feedback_max_age_s']:
                self.verifier.samples.clear()
            fresh_force = self.force_last_at is None or force_at > self.force_last_at
            self.force_last_at = force_at
            if fresh_force and self.verifier.observe(now, pose, mode, mode_at, pose_at):
                if self.phase == 'retract':
                    self.released = True
                    self.events.append(dict(event='contact_release_verified',
                        pose=pose.tolist(), measured_fz_N=float(force[2]),
                        rule='at home clearance height, stationary and low absolute measured Fz'))
                    # Preserve the exact commanded endpoint as the next start;
                    # starting at noisy feedback would create a new command jump.
                    self._segment(result['sent'][:6], self.home, 'return_home', now, 2)
                else:
                    self.complete = True
                    result.update(done=True, contact_release_verified=True, home_reached=True)
                    self.events.append(dict(event='home_pose_reached', pose=pose.tolist(),
                        target=self.home.tolist(), contact_release_verified=True))
        return result
