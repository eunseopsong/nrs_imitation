"""Offline force-selection regression: durations/ack/gate/sign, not sensor peaks."""
import importlib.util
import json
from pathlib import Path

import pytest
from nrs_imitation.e2_providers import hardware_blockers

ROOT=Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('force_reference',ROOT/'scripts/e2_force_reference_from_e1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def commands(values):
    rows=[];events=[]
    for i,(t,f,gate) in enumerate(values):
        rows.append(dict(command_id=str(i+1),receipt_monotonic_ns=str(int(t*1e9)),
            command_stage='node_sent',command_mode='PTP9D_STREAM_SET_FORCE',validity='valid',
            fz=str(f),details=json.dumps(dict(contact_gate=gate))))
        events.append(dict(event='service_response',details=dict(command_id=i+1,success=True)))
    events.append(dict(event='run_interrupted',receipt_monotonic_ns=int((values[-1][0]+1)*1e9)))
    return rows,events


def test_duration_weighting_ignores_send_frequency_and_zero_gaps():
    # Many brief 100N target updates must not outweigh a long 18N target hold.
    values=[(0,0,False),(1,100,True),(1.1,100,True),(1.2,100,True),
            (1.3,18,True),(10,0,False),(10.1,30,True),(10.2,0,False)]
    summary,intervals=m.reconstruct(*commands(values))
    assert summary['time_weighted_median_N']==18.
    assert summary['interval_start_s']==1. and summary['interval_end_s']==10.
    assert not intervals[0]['selected_interval'] and not intervals[-2]['selected_interval']


def test_negative_sign_is_preserved_and_missing_ack_is_rejected():
    rows,events=commands([(0,0,False),(1,-18,True),(5,0,False)])
    assert m.reconstruct(rows,events)[0]['time_weighted_median_N']==-18.
    events[1]['details']['success']=False
    with pytest.raises(ValueError,match='response'):m.reconstruct(rows,events)


def test_unknown_rpm_requires_explicit_provenance_and_does_not_clear_other_guards():
    c=json.loads((ROOT/'experiments/e2_rule_replay_20260920/config.json').read_text())
    c['task']['endpoints_relative_mm']=None
    c['common']['force_frame_sign_verification']=None
    assert c['common']['rpm_setpoint'] is None and c['common']['rpm_status']=='unknown'
    errors=hardware_blockers(c,'rule',True)
    assert not any('rpm' in e.lower() for e in errors)
    assert any('endpoints_relative_mm' in e for e in errors)
    assert any('force_frame_sign_verification' in e for e in errors)
    c['common'].pop('rpm_unknown_reason')
    assert any('rpm' in e.lower() for e in hardware_blockers(c,'rule',True))
    c['common']['rpm_setpoint']=-1
    assert any('rpm' in e.lower() for e in hardware_blockers(c,'rule',True))


def test_recorded_e1_selection_is_reproducible(tmp_path):
    report=m.analyze(ROOT/'results/20260920',tmp_path)
    assert report['force_reference_N']==18.
    assert report['runs']['C']['time_weighted_median_N']==pytest.approx(17.694250106811523)
    c=json.loads((ROOT/'experiments/e2_rule_replay_20260920/config.json').read_text())
    assert c['recipe']['force_reference_N']==report['force_reference_N']
    assert report['runs']['C']['run_id']==c['recipe']['force_reference_source']['run_id']
