"""Validate the registered trained A policy with CPU and fake ROS I/O only."""
from datetime import datetime
from pathlib import Path
import copy
import csv
import importlib.util
import json
import os
import sys
import tempfile

ROOT=Path('/home/eunseop/nrs_imitation')
OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'behavior_ws/src/nrs_imitation'),
               str(ROOT/'behavior_ws/src/nrs_imitation/test')]
os.environ['HF_HUB_OFFLINE']='1'

import h5py
import numpy as np
import torch
from launch import LaunchContext
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from nrs_imitation import inference_core as core
from nrs_imitation.e2_ablation import EXPERIMENT,select_condition,model_errors,digest,motion_to_contract,read_stats
from nrs_imitation.inference_metrics import InferenceMetrics
from test_execution_metrics import FakeNode
from test_e2_direct_launch import resolved_inference_parameters,assert_inference_parameter_types

torch.set_num_threads(2)
cfg=json.loads((EXPERIMENT/'config.json').read_text());selected=select_condition(cfg,'A')
assert not model_errors(selected)
checkpoint=Path(cfg['models']['A']['checkpoint']);ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert ck['epoch']==499
# Exercise the actual startup loader with the same resolved launch parameters.
source=ROOT/'behavior_ws/src/nrs_imitation/launch/inference_clean_single_cam.launch.py'
include=IncludeLaunchDescription(PythonLaunchDescriptionSource(str(source)),launch_arguments={
    'execution_method':'il','e2_config':str(EXPERIMENT/'config.json'),
    'ckpt_dir':str(checkpoint.parent),'ckpt_auto_subdir':'polishing/single_cam',
    'use_force_observation':'false','use_force_history':'false','inference_mode':'timed_topic',
    'use_stain_mask':'false','stain_canon_enable':'false',
    'gradcam_enable':'false','removal_viz_enable':'false'}.items())
parameters=resolved_inference_parameters(include,LaunchContext());checked=assert_inference_parameter_types(parameters)
loader=FakeNode(OUT/'unused_loader_metrics')
loader.params.update(parameters)
for key,value in parameters.items():
    if key not in ('device','camera_names'):
        setattr(loader,key,value)
loader.motion_only=True;loader.action_dim=6
policy=loader._load_policy_and_ckpt_from_act_root()
assert policy.cfg['state_dim']==6
loaded_state=policy.state_dict();expected_state=ck['model_state_dict']
assert loaded_state.keys()==expected_state.keys()
assert all(torch.equal(loaded_state[k],v) for k,v in expected_state.items())
result=policy.load_state_dict(expected_state,strict=True)
assert not result.missing_keys and not result.unexpected_keys
assert policy.action_dim==6
del ck,expected_state,loaded_state
stats=read_stats(checkpoint.with_name('dataset_stats.pkl'))
split=json.loads((EXPERIMENT/'teacher_preview/split.json').read_text())
assert stats['split_episode_files']==split
for c in 'BC':
    other=read_stats(Path(cfg['models'][c]['checkpoint']).with_name('dataset_stats.pkl'))
    for key in ('qpos_min','qpos_max','action_min','action_max'):
        np.testing.assert_array_equal(stats[key],other[key][:6])
dataset=ROOT/'datasets/polishing/single_cam/20260910_90deg_rel_single/imitation_form'
episode=split['validation'][0]
with h5py.File(dataset/episode) as f:
    index=len(f['observations/position'])//2
    pose=np.asarray(f['observations/position'][index],np.float64)[:6]
    frame=np.asarray(f['observations/images/cam0'][index],np.uint8)
pose[:2]+=[400.,500.]
predictions=[];logging=[]
with tempfile.TemporaryDirectory(prefix='E2_A_offline_') as temp:
    for i,force in enumerate(([1.,2.,3.],[-73.,51.,-99.])):
        n=FakeNode(Path(temp)/str(i),policy,False)
        n.ckpt_dir=str(checkpoint.parent);n.motion_only=True;n.action_dim=6;n.chunk_size=128
        n.flow_infer_steps=10;n.resize_hw=0;n.use_force_history=False;n._e2_context=selected
        n.metrics_run_tag='offline_registered_A';n.stats=core._load_dataset_stats(n.ckpt_dir)
        n.params.update(ckpt_dir=n.ckpt_dir,chunk_size=128,flow_infer_steps=10,resize_hw=0,
                        use_force_history=False,use_force_observation=False)
        n._metrics=InferenceMetrics(n)
        try:
            n._on_pose(Float64MultiArray(data=pose.tolist()))
            for _ in range(30):n._on_force(Float64MultiArray(data=list(force)+[0.,0.,0.]))
            n._on_img(Image(height=frame.shape[0],width=frame.shape[1],encoding='rgb8',
                            step=frame.shape[1]*3,data=frame.tobytes()))
            n._on_infer_timer();assert len(n.plans)==1,n.errors
            transport=n.plans[-1].seq_den.copy();assert transport.shape==(128,9)
            assert np.isfinite(transport).all() and not n.errors,n.errors
            assert np.all(transport[:,6:]==0), 'Transport placeholders must not predict force'
            predictions.append(transport[:,:6])
        finally:n._metrics.close()
        path=n._metrics.recorder.path
        with (path/'commands.csv').open() as f:rows=list(csv.DictReader(f))
        pred=[r for r in rows if r['command_stage']=='policy_prediction']
        assert len(pred)==128 and all(r['fz']=='' and json.loads(r['details'])['predicted_force'] is None for r in pred)
        assert all(len(json.loads(r['raw_values']))==6 for r in pred)
        summary=json.loads((path/'summary.json').read_text())
        assert summary['drained'] and summary['write_error_count']==0
        assert all(v['dropped']==0 for v in summary['counts'].values())
        logging.append(dict(prediction_rows=len(pred),force_fields_null=True,logger_drop_count=0,write_errors=0))
np.testing.assert_array_equal(predictions[0],predictions[1])

assert parameters['use_force_observation'] is False and parameters['use_force_history'] is False
assert core._resolve_checkpoint_dir(parameters['ckpt_dir'],parameters['act_root'],parameters['policy_class'],
    parameters['ckpt_auto_subdir'])==str(checkpoint.parent)
report=dict(checked_at=datetime.now().astimezone().isoformat(),checkpoint=str(checkpoint),
    checkpoint_sha256=digest(checkpoint),selected_epoch_zero_based=499,strict_state_dict_load=True,
    actual_ROS_startup_loader=True,loaded_weights_exactly_match_checkpoint=True,state_dim=6,
    device='cpu',actual_checkpoint_inference=True,validation_episode=episode,sample_index=index,
    model_output_shape=list(predictions[0].shape),transport_shape=[128,9],transport_force_placeholders_zero=True,
    finite_predictions=True,force_observation_invariant=True,
    max_force_perturbation_prediction_difference=float(np.max(np.abs(predictions[0]-predictions[1]))),
    force_prediction_logged_as_null=True,common_motion_normalizers_match_BC=True,
    launch_parameter_types_checked=len(checked),exact_registered_checkpoint_resolution=True,
    sampling_steps=10,logging=logging,ros_context_initialized=False,hardware_io=False,
    F0_frozen=False,physical_quality_validated=False)
(OUT/'trained_A_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
