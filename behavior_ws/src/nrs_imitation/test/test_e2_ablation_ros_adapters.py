"""Import ROS message types but never initialize a ROS context or real node."""
import copy
import csv
import json
from pathlib import Path
import pickle
from types import SimpleNamespace as NS

import h5py
import numpy as np
import pytest
import torch
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image

from nrs_imitation import inference_core as core
from nrs_imitation.inference_metrics import InferenceMetrics
from nrs_imitation.e2_ablation import EXPERIMENT,select_condition
from test_execution_metrics import FakeNode
from test_e2_timed_execution import mock_bridge,settle


@pytest.mark.parametrize('condition', ['A', 'B', 'C'])
def test_runtime_loader_restores_condition_schema_and_weights(tmp_path, condition):
    from models.flow_core import FlowRGBPolicy
    motion_only = condition == 'A'
    dim = 6 if motion_only else 9
    cfg = dict(num_queries=8, state_dim=dim, action_dim=dim, force_dim=3,
        camera_names=['cam0'], obs_mode='single_cam', pretrained_backbone=False,
        image_backbone='resnet18', use_tcp_roi=False,
        use_force_history=not motion_only, use_force_observation=condition == 'C',
        motion_only=motion_only, force_action=not motion_only,
        flow_obs_hidden_dim=16, flow_image_feature_dim=16, flow_global_cond_dim=16,
        flow_time_embed_dim=16, flow_down_dims='16,32', flow_kernel_size=3, flow_n_groups=4)
    original = FlowRGBPolicy(cfg).eval()
    torch.save({'model_state_dict': original.state_dict(), 'config': {'policy_config': cfg}},
               tmp_path/'policy_best.ckpt')
    with (tmp_path/'dataset_stats.pkl').open('wb') as f:
        pickle.dump({'policy_config': cfg}, f)
    node = FakeNode(tmp_path/'metrics', force_on=condition == 'C')
    node.ckpt_dir = str(tmp_path)
    for key, value in cfg.items():
        setattr(node, key, value)
        node.params[key] = value
    # Exercise the actual ROS startup loader, not a policy constructed by the test.
    loaded = node._load_policy_and_ckpt_from_act_root()
    assert loaded.cfg['state_dim'] == loaded.action_dim == dim
    assert loaded.cfg['use_force_observation'] == (condition == 'C')
    assert loaded.cfg['use_force_history'] == (condition != 'A')
    assert loaded.state_dict().keys() == original.state_dict().keys()
    for key, value in original.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key
    inputs = dict(qpos=torch.ones(1, dim), image=torch.ones(1, 1, 3, 32, 32),
                  force_history=torch.ones(1, 30, 3) if not motion_only else None,
                  initial_noise=torch.zeros(1, 8, dim), num_steps=2)
    torch.testing.assert_close(loaded.sample_action(**inputs), original.sample_action(**inputs),
                               rtol=0, atol=0)


def test_motion_only_inference_real_adapter_and_null_force_logging(tmp_path):
    from models.flow_core import FlowRGBPolicy
    cfg=dict(num_queries=8,state_dim=6,action_dim=6,force_dim=3,camera_names=['cam0'],
        obs_mode='single_cam',pretrained_backbone=False,image_backbone='resnet18',use_tcp_roi=False,
        use_force_history=False,use_force_observation=False,motion_only=True,force_action=False,
        flow_obs_hidden_dim=16,flow_image_feature_dim=16,flow_global_cond_dim=16,
        flow_time_embed_dim=16,flow_down_dims='16,32',flow_kernel_size=3,flow_n_groups=4)
    policy=FlowRGBPolicy(cfg).eval();outputs=[]
    for i,scale in enumerate([1.,-73.]):
        n=FakeNode(tmp_path/str(i),policy,False)
        n.motion_only=True;n.action_dim=6;n.use_force_history=False
        n.stats=copy.deepcopy(n.stats)
        for key in ['qpos_a','qpos_b','act_a','act_b']:setattr(n.stats,key,getattr(n.stats,key)[:6].copy())
        n._e2_context=select_condition(json.loads((EXPERIMENT/'config.json').read_text()),'A')
        n._metrics=InferenceMetrics(n)
        n._on_pose(Float64MultiArray(data=[420.,530.,200.,.01,.02,.03]))
        for _ in range(30):n._on_force(Float64MultiArray(data=[scale,2*scale,3*scale,0.,0.,0.]))
        n._on_img(Image(height=32,width=32,encoding='rgb8',step=96,data=bytes([120])*3072))
        n._on_infer_timer();assert len(n.plans)==1,n.errors
        outputs.append(n.plans[-1].seq_den.copy());assert not n.errors
        n._metrics.close();path=n._metrics.recorder.path
        with (path/'commands.csv').open() as f:rows=list(csv.DictReader(f))
        pred=[r for r in rows if r['command_stage']=='policy_prediction']
        assert pred and all(r['fz']=='' and json.loads(r['details'])['predicted_force'] is None for r in pred)
        assert json.loads((path/'metadata.json').read_text())['method']=='A'
    np.testing.assert_array_equal(outputs[0],outputs[1])


def test_A_dataset_ignores_force_target_supervision(tmp_path):
    from data.dataset import _read_action,compute_dataset_stats
    p=tmp_path/'episode_0.hdf5'
    pose=np.arange(60,dtype=np.float32).reshape(10,6)
    with h5py.File(p,'w') as f:
        f.create_dataset('observations/position',data=pose)
        f.create_dataset('observations/force',data=np.ones((10,3),np.float32))
        f.create_dataset('action/position',data=pose)
        f.create_dataset('action/force',data=np.full((10,3),np.nan,np.float32))
    with h5py.File(p) as f:
        np.testing.assert_array_equal(_read_action(f,pose,np.ones((10,3)),motion_only=True),pose)
    stats=compute_dataset_stats([p],motion_only=True)
    assert len(stats['qpos_min'])==len(stats['action_min'])==6
    assert np.isfinite(stats['action_min']).all()


def test_A_stats_loader_roundtrip(tmp_path):
    st=dict(qpos_min=np.zeros(6,np.float32),qpos_max=np.ones(6,np.float32)*100,
            action_min=np.zeros(6,np.float32),action_max=np.ones(6,np.float32)*100,
            qpos_norm_mode='minmax_m11',action_norm_mode='minmax_m11',policy_config={'motion_only':True})
    with (tmp_path/'dataset_stats.pkl').open('wb') as f:pickle.dump(st,f)
    stats=core._load_dataset_stats(str(tmp_path))
    assert core._normalize_qpos(torch.ones(1,6),stats).shape==(1,6)
    assert core._denorm_action_seq(torch.zeros(8,6),stats).shape==(8,6)


def test_operator_marker_common_state_and_stop_verification(monkeypatch):
    n,t,f=mock_bridge(monkeypatch);n.ablation=True
    n.config=select_condition(json.loads((EXPERIMENT/'config.json').read_text()),'B')
    n.config['protocol']['phase_ids']=['pass_1','pass_2'];n.phase_index=-1
    assert n.operator_event('processing_start',NS()).success
    assert n.operator_event('phase_next',NS()).success
    assert not n.operator_event('phase_next',NS()).success
    assert n.operator_event('processing_end',NS()).success
    assert n.operator_event('finish',NS()).success
    f.cb(f);settle(n,t)
    assert n.state=='stopped'
    assert any(e['event']=='physical_hold_verified' for e in n.events)
    assert any(e['event']=='finish_requested' for e in n.events)
