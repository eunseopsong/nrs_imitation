"""No robot/ROS initialization: retract before XY, verify feedback, abort safely."""
import copy
from types import SimpleNamespace as NS

import numpy as np
import pytest

from nrs_imitation.e2_return_home import ReturnHome, validate_return_home
from nrs_imitation.e2_timed_execution import ExecutionFault, executor_settings
from nrs_imitation.e2_providers import rule_actions
from test_e2_timed_execution import config, mock_bridge, settle


def spec():
    return dict(enabled=True, home_source='common.demo_start_pose6',
        timeout_s=50., verification_timeout_s=2., release_abs_fz_N=1.2)


def run_home(force_function=lambda r,p: 0. if p[2] >= 9. else 7.):
    c=config();r=ReturnHome(executor_settings(c),spec(),np.zeros(6),[10,0,10,0,0,.1],10.)
    pose=np.zeros(6);rows=[]
    for i in range(1,1600):
        t=10.+i*.008
        result=r.tick(t,pose,[0,0,force_function(r,pose)],t,t,'Position',t)
        if result['done']:return r,rows
        rows.append((r.released,result['phase'],result['sent'].copy()))
        pose=result['sent'][:6].copy()
    raise AssertionError('Return did not finish')


def test_lift_zero_force_before_xy_then_verified_home_at_common_rate():
    r,rows=run_home()
    positions=np.asarray([x[2] for x in rows])
    assert all(np.allclose(v[:2],0) for released,phase,v in rows if not released or phase=='retract')
    assert np.all(positions[:,6:]==0)
    assert np.max(np.abs(np.diff(positions[:,:3],axis=0))/.008)<=10.001
    assert r.complete and r.released
    np.testing.assert_allclose(positions[-1,:6],r.home,atol=.1)
    events=[x['event'] for x in r.events]
    assert events.index('contact_release_verified') < events.index('home_pose_reached')


def test_positive_or_negative_residual_force_cannot_claim_release_or_move_xy():
    for force in [7.,-7.]:
        r=ReturnHome(executor_settings(config()),spec(),np.zeros(6),[10,0,10,0,0,0],10.)
        pose=np.zeros(6)
        with pytest.raises(ExecutionFault,match='verification timeout'):
            for i in range(1,1000):
                t=10.+i*.008;result=r.tick(t,pose,[0,0,force],t,t,'Position',t)
                assert result['sent'][0]==0 and not r.released
                pose=result['sent'][:6].copy()


@pytest.mark.parametrize('fault',['stale_force','stale_pose','wrong_mode','stale_mode'])
def test_return_rejects_missing_feedback_and_wrong_mode(fault):
    r=ReturnHome(executor_settings(config()),spec(),np.zeros(6),[10,0,10,0,0,0],10.)
    with pytest.raises(ExecutionFault):
        r.tick(10.008,np.zeros(6),[0,0,0],9. if fault=='stale_pose' else 10.008,
            9. if fault=='stale_force' else 10.008,'Force' if fault=='wrong_mode' else 'Position',
            9. if fault=='stale_mode' else 10.008)


def test_new_contact_during_return_stops_instead_of_continuing_xy():
    with pytest.raises(ExecutionFault,match='Contact detected again'):
        run_home(lambda r,p:7. if r.phase=='return_home' else 0.)


def test_return_never_descends_to_nominal_work_depth():
    r=ReturnHome(executor_settings(config()),spec(),[0,0,8,0,0,0],[10,0,10,0,0,0],10.)
    result=r.tick(10.008,[0,0,8,0,0,0],[0,0,7],10.008,10.008,'Position',10.008)
    assert result['sent'][2]>=8 and r.lift[2]==10
    with pytest.raises(ExecutionFault,match='descend'):
        ReturnHome(executor_settings(config()),spec(),[0,0,20,0,0,0],[10,0,10,0,0,0],10.)


def test_real_single_pass_has_no_backward_processing_and_finish_force_zero():
    c=config();c['recipe']['pass_count']=1
    a=rule_actions(c['task'],c['recipe'])
    processing=a.action[a.phase=='processing',:3]
    direction=np.diff(np.asarray(c['task']['endpoints_relative_mm']),axis=0)[0]
    assert np.all(np.diff(processing,axis=0)@direction>=-1e-3)
    np.testing.assert_allclose(a.action[-1,:3],c['task']['endpoints_relative_mm'][1],atol=1e-5)
    assert a.action[-1,8]==0 and a.time[-1]<34.


def test_bridge_holds_before_lift_and_normal_completion_requires_home(monkeypatch):
    n,t,f=mock_bridge(monkeypatch)
    n.config=config();n.config['common']['demo_start_pose6']=[10.,0.,10.,0.,0.,0.]
    n.return_spec=spec();n.return_home=None;n.reference=[0.,0.];n.method='rule'
    n.pose=np.array([0.,0.,3.,0.,0.,0.]);n.final_target=np.zeros(6)
    n.request_stop('trajectory_end');assert not n.sent
    f.cb(f)
    # Exact hold window only; the new return phase has not published anything.
    for _ in range(45):
        t[0]+=.01;n.pose_at=n.force_at=n.mode_at=n.provider_at=t[0];n.mode='Position';n.tick()
        if n.state=='returning':break
    assert n.state=='returning' and not n.sent
    assert not any(e['event']=='normal_completion' for e in n.events)
    for _ in range(1500):
        t[0]+=.008;n.pose_at=n.force_at=n.mode_at=n.provider_at=t[0]
        n.force=np.array([0.,0.,0. if n.pose[2]>9. else 7.]);n.tick()
        if n.sent:n.pose=np.array(n.sent[-1])
        if n.state=='stopping':break
    assert n.stop_reason=='home_return_complete' and n.state=='stopping'
    assert all(len(row)==6 for row in n.sent)
    assert not any(e['event']=='normal_completion' for e in n.events)
    f.cb(f);settle(n,t)
    assert n.state=='stopped'
    assert any(e['event']=='normal_completion' and e['retract_complete'] and e['home_pose_reached'] for e in n.events)


def test_abort_does_not_trigger_automatic_return(monkeypatch):
    n,t,f=mock_bridge(monkeypatch);n.return_spec=spec();n.return_home=None
    n.request_stop('manual_abort');f.cb(f);settle(n,t)
    assert n.state=='stopped' and n.return_home is None and not n.sent


def test_return_preflight_rejects_changed_home_and_relaxed_release_threshold():
    c=config();c['recipe']['return_home']=spec()
    assert validate_return_home(c)==[]
    c['recipe']['return_home']['home_source']='arbitrary'
    assert validate_return_home(c)
    c['recipe']['return_home']=spec();c['recipe']['return_home']['release_abs_fz_N']=100.
    assert validate_return_home(c)
