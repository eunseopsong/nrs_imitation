"""Review a 23 N assumption offline; never freeze F0 or start a ROS context."""
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np

ROOT = Path('/home/eunseop/nrs_imitation')
OUT = Path(__file__).resolve().parent
EXP = ROOT/'experiments/e2_force_ablation_20260926'
sys.path.insert(0, str(ROOT/'behavior_ws/src/nrs_imitation'))
from nrs_imitation.e2_ablation import motion_to_contract, select_condition
from nrs_imitation.e2_timed_execution import TimedPlan, TimedExecution, executor_settings


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    before = (EXP/'config.json').read_bytes()
    cfg = json.loads(before)
    assert cfg['f0']['status'] != 'frozen'
    assert cfg['f0']['command_fz_N'] is None
    candidate = ROOT/'reports/20260927_E2_F0_candidate/candidate.json'
    assumed = float(json.loads(candidate.read_text())['rounded_teacher_channel_candidate_N'])
    draft = dict(schema='E2_offline_force_assumption_v1', status='unverified_offline_only',
        created_at=datetime.now().astimezone().isoformat(),
        operator_request='그냥 임의의 값으로 채워줘',
        assumed_command_Fz_N=assumed, source=str(candidate), source_sha256=digest(candidate),
        assumption='Reuse rounded teacher Fz candidate numerically as a controller target for software calculation only; physical teacher/TCP equivalence remains unverified.',
        physical_force_calibration_verified=False, validated_operating_range_N=None,
        physical_abort_limit_N=None, F0_frozen=False, hardware_authorized_by_this_file=False)
    # This separate schema is not an accepted E2 hardware experiment config.
    (OUT/'offline_assumption.json').write_text(json.dumps(draft,ensure_ascii=False,indent=2)+'\n')
    settings = executor_settings(select_condition(cfg,'A'),'il')
    pose = np.asarray(cfg['common']['demo_start_pose6'],float).copy()
    pose[:2] += [400.,500.]
    times = np.arange(128)/30.
    actions = motion_to_contract(np.tile(pose,(128,1)))
    engine = TimedExecution(settings)
    engine.accept(TimedPlan(1,10.,times,actions,np.full(128,'synthetic'),False),10.)
    engine.start(10.,pose)
    previous = 0.; rows=[]
    for i in range(1,451):
        t = i*.008
        processing = .4 <= t < 2.8
        # Synthetic contact loss/reacquisition exercises the existing gate.
        measured = 6. if (.6 <= t < 1. or 1.2 <= t < 3.) else 0.
        target = assumed if processing else 0.
        result=engine.tick(10.+t,pose,[0.,0.,measured],0.,0.,external_force_fz=target)
        assert 'end' not in result
        assert result['requested'][8]==target
        assert result['gated'][8]==(target if result['contact'] else 0.)
        assert abs(result['sent'][8]-previous)<=settings['force_rate_N_s']*.008+1e-8
        np.testing.assert_allclose(result['sent'][:6],pose,atol=1e-9,rtol=0)
        rows.append(dict(synthetic_time_s=t,synthetic_processing=processing,
            synthetic_measured_base_Fz_N=measured,assumed_target_N=target,
            contact=result['contact'],gated_N=float(result['gated'][8]),calculated_sent_N=float(result['sent'][8])))
        previous=result['sent'][8]
    assert (EXP/'config.json').read_bytes()==before
    with (OUT/'synthetic_force_calculation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(status='offline_calculation_pass',checked_at=datetime.now().astimezone().isoformat(),
        assumed_command_Fz_N=assumed,synthetic_ticks=len(rows),real_robot_samples_used=False,
        stationary_pose_only=True,trained_policy_inference=False,contact_gate_preserved=True,
        force_slew_limit_N_s=settings['force_rate_N_s'],runtime_config_unchanged=True,
        runtime_config_sha256=hashlib.sha256(before).hexdigest(),
        ROS_context_initialized=False,robot_commands_sent=0,physical_validation=False,F0_frozen=False)
    (OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
