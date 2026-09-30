"""Reproducible registered-A + candidate-F0 check. Hardware is never opened."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import traceback
import uuid

import numpy as np
import rclpy
from nrs_imitation.e2_ablation import ROOT, digest, select_condition
from nrs_imitation.e2_offline import registered_A_sample
from nrs_imitation.e2_providers import hardware_blockers
from nrs_imitation.e2_timed_execution import TimedExecution, TimedPlan, executor_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
        default=ROOT/'experiments/e2_A_pilot_20260927/config.json')
    args = parser.parse_args()
    # These hard failures catch accidental I/O introduced into this offline path.
    def no_hardware(*a, **kw):
        raise AssertionError('Offline command must not initialize ROS or open a node')
    rclpy.init = no_hardware
    rclpy.node.Node.__init__ = no_hardware
    cfg = json.loads(args.config.read_text())
    watched = [args.config, ROOT/'experiments/e2_force_ablation_20260926/config.json',
               ROOT/'experiments/e2_force_ablation_20260926/f0_candidate_status.json']
    before = {str(p): digest(p) for p in watched}
    out = ROOT/'results'/datetime.now().strftime('%Y%m%d')/'E2/A_offline'/(
        datetime.now().strftime('%H%M%S')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    def write(name, obj):
        (out/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    report = dict(status='started', offline_only=True, hardware_io=False,
                  F0_frozen=False, quality_validation='pending', source_config_sha256=before[str(args.config)])
    write('attempt.json', report); write('config_snapshot.json', cfg)
    try:
        action, pose, model = registered_A_sample(cfg, out)
        np.save(out/'policy_motion6.npy', action[:, :6])
        selected = select_condition(cfg, 'A')
        blockers = hardware_blockers(selected, 'il', True)
        settings = executor_settings(selected, 'il')
        f0 = cfg['f0']['command_fz_N']
        assert f0 == 23. and cfg['f0']['status'] == 'candidate'
        assert cfg['pilot']['F0_frozen'] is False and cfg['pilot']['quality_validation'] == 'pending'
        engine = TimedExecution(settings)
        t = np.arange(len(action))/cfg['sampling']['trajectory_hz']
        engine.accept(TimedPlan(1, 10., t, action, np.full(len(t), 'offline'), False), 10.)
        engine.start(10., pose)
        rows = []; previous = np.r_[pose, np.zeros(3)]
        # Candidate sign is an offline hypothesis. Fresh feedback here is mocked,
        # never evidence of contact dynamics, sensor health or permissible load.
        for i in range(500):
            elapsed = i*settings['control_period_s']; now = 10.+elapsed
            processing = .5 <= elapsed < 2.5
            force = np.array([0., 0., 0. if 1.6 <= elapsed < 1.8 else 7., 0., 0., 0.])
            result = engine.tick(now, pose, force, 0., 0., external_force_fz=f0 if processing else 0.)
            assert 'end' not in result
            assert result['requested'][8] == (f0 if processing else 0.)
            assert abs(result['sent'][8]-previous[8]) <= settings['force_rate_N_s']*.008+1e-9
            if not result['contact']:
                assert result['gated'][8] == 0.
            rows.append(dict(time_s=elapsed, processing=processing, feedback_synthetic=True,
                actual_pose_mock=pose.tolist(), actual_wrench_mock=force.tolist(),
                requested=result['requested'].tolist(), conditioned=result['conditioned'].tolist(),
                gated=result['gated'].tolist(), sent=result['sent'].tolist(), contact=result['contact']))
            previous = result['sent']; pose = previous[:6].copy()
        engine.stop()
        assert engine.tick(14., pose, force, 0., 0.) is None and engine.plan is None
        (out/'executor_mock.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        write('events_mock.json', [dict(time_s=.5, event='processing_start', source='synthetic'),
            dict(time_s=2.5, event='processing_end', source='synthetic'),
            dict(time_s=4., event='stop_requested', reason='offline_check_end', hardware_io=False)])
        report.update(status='offline_pass', model=model, candidate_F0_N=f0,
            synthetic_feedback=True, ticks=len(rows), gate_and_slew_limits_preserved=True,
            command_cap_N=settings['fz_hard_limit_N'], command_cap_enabled=settings['fz_hard_limit_N']>0,
            measured_force_abort_limit_N=None, real_preflight_blockers=blockers,
            physical_safety_validated=False, quality_validation='pending')
    except Exception as exc:
        report.update(status='offline_failed', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['real_configuration_unchanged'] = all(digest(Path(p)) == h for p, h in before.items())
        write('attempt.json', report)
        write('manifest.json', {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*'))
                               if p.is_file() and p.name != 'manifest.json'})
        print(json.dumps(dict(result=str(out), status=report['status'],
                             real_configuration_unchanged=report['real_configuration_unchanged']), ensure_ascii=False))


if __name__ == '__main__':
    main()
