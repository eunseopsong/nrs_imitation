#!/usr/bin/env python3
"""E2 paired training configuration and bounded, real-data CPU smoke check."""
import argparse
import copy
import gc
import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'source'), str(ROOT/'behavior_ws/src/nrs_imitation')]
from flow_train_core import (build_arg_parser, default_policy_config, set_seed,
    _unpack_batch, run_one, carry_forward_relative_frame_stats, collect_demo_start_pose_stats)
from data.dataset import make_loaders
from models.flow_core import build_flow_rgb_policy_and_optimizer
from nrs_imitation.e2_ablation import EXPERIMENT, digest
import numpy as np
import torch


def arguments(condition):
    args = build_arg_parser().parse_args([])
    args.dataset_dir = str(ROOT/'datasets/polishing/single_cam/20260910_90deg_rel_single/imitation_form')
    args.ckpt_dir = str(ROOT/f'checkpoints/flow/polishing/single_cam/e2_force_ablation_20260926/{condition}')
    args.camera_names = ['cam0']; args.obs_mode = 'single_cam'
    args.tcp_roi_area_fraction = .25; args.phase_resample_enable = True
    args.motion_only = condition == 'A'
    args.use_force_observation = condition == 'C'
    args.use_force_history = condition != 'A'
    args.state_dim = args.action_dim = 6 if condition == 'A' else 9
    return args


def prepare():
    result = {}
    for c in 'ABC':
        args = arguments(c)
        result[c] = dict(args=vars(args),policy_config=default_policy_config(args,'single_cam',['cam0']),
            selection_rule='fixed epoch 500 / policy_last.ckpt; never compare loss magnitudes across dimensions',
            initial_weights='same public frozen DINOv3 backbone, fresh seed-0 remaining layers; no C fine-tuning',
            experiment_id='E2',condition=c)
    (EXPERIMENT/'training_configs.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def smoke():
    # Network is unnecessary: current B/C checkpoints contain the backbone;
    # A uses the locally cached public pretrained backbone via timm.
    os.environ['HF_HUB_OFFLINE'] = '1'
    torch.set_num_threads(2)
    configs = prepare(); audit = json.loads((EXPERIMENT/'models_audit.json').read_text())
    output = EXPERIMENT/'checks'; output.mkdir(exist_ok=True)
    results = {}; common_stats = None
    for c in 'ABC':
        print('SMOKE',c,flush=True)
        args=arguments(c);set_seed(0)
        loader, _, stats, meta = make_loaders(dataset_dir=args.dataset_dir,camera_names=['cam0'],
            obs_mode='single_cam',batch_size_train=1,batch_size_val=1,seq_len_train=128,seq_len_val=128,
            seed=0,samples_per_episode=1,num_workers=0,force_history_len=30,
            return_force_history=args.use_force_history,use_force_history=args.use_force_history,
            use_force_observation=args.use_force_observation,motion_only=args.motion_only,
            phase_resample_enable=True,resample_each_epoch=True,dataset_hz=30.)
        batch=_unpack_batch(next(iter(loader)),torch.device('cpu'))
        image,qpos,action,pad,history,marker=batch
        pc=copy.deepcopy(configs[c]['policy_config'])
        if c != 'A':pc['pretrained_backbone']=False
        policy,optimizer=build_flow_rgb_policy_and_optimizer(pc)
        baseline=None
        if c != 'A':
            checkpoint=torch.load(audit['models'][c]['checkpoint'],map_location='cpu',weights_only=False)
            policy.load_state_dict(checkpoint['model_state_dict'],strict=True)
            del checkpoint;gc.collect()
        policy.eval();noise=torch.randn(1,128,args.action_dim)
        with torch.no_grad():
            first=policy.sample_action(qpos,image,history,num_steps=10,initial_noise=noise)
            changed_q = qpos.clone()
            if c == 'A': changed_q=torch.cat([qpos,torch.ones(1,3)*999.],dim=-1)
            else: changed_q[:,6:9]=.79
            changed_h = torch.ones(1,30,3)*-.83
            second=policy.sample_action(changed_q,image,changed_h,num_steps=10,initial_noise=noise)
        if c in 'AB':assert torch.equal(first,second),c+' leaked measured force'
        assert torch.isfinite(first).all()
        # C regression against the actual pre-patch class, same checkpoint and inputs.
        regression=None
        if c=='C':
            import importlib.util
            old_path=EXPERIMENT/'before/source/models/flow_core.py'
            spec=importlib.util.spec_from_file_location('models.e2_before_flow_core',old_path)
            old_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_module)
            old_policy,_=old_module.build_flow_rgb_policy_and_optimizer(pc)
            old_policy.load_state_dict(policy.state_dict(),strict=True);old_policy.eval()
            with torch.no_grad():baseline=old_policy.sample_action(qpos,image,history,num_steps=10,initial_noise=noise)
            assert torch.equal(first,baseline)
            regression={'bitwise_equal':True,'max_abs_difference':float((first-baseline).abs().max())}
            del old_policy;gc.collect()
        policy.train();optimizer.zero_grad(set_to_none=True)
        loss=policy(qpos,image,actions=action,is_pad=pad,force_history=history)['loss']
        assert torch.isfinite(loss);loss.backward()
        gradients=[p.grad for p in policy.parameters() if p.requires_grad and p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert action.shape[-1] == (6 if c=='A' else 9)
        # No optimizer step: this is never an experiment-ready checkpoint.
        policy.eval()
        saved=output/(c+'_smoke_only.ckpt')
        torch.save({'model_state_dict':policy.state_dict(),'policy_config':pc,'stats':stats,
                    'smoke_only':True,'experiment_id':'E2','condition':c,'trained':False},saved)
        reload=torch.load(saved,map_location='cpu',weights_only=False)
        policy.load_state_dict(reload['model_state_dict'],strict=True)
        assert reload['stats']['split_episode_files']==stats['split_episode_files']
        with torch.no_grad():third=policy.sample_action(qpos,image,history,num_steps=10,initial_noise=noise)
        assert torch.equal(first,third)
        if common_stats is None: common_stats={k:stats[k][:6].copy() for k in ('qpos_min','qpos_max','action_min','action_max')}
        for k,v in common_stats.items():assert np.array_equal(v,stats[k][:6])
        results[c]=dict(actual_data=True,checkpoint='untrained public-backbone initialization' if c=='A' else audit['models'][c]['checkpoint'],
            checkpoint_ready=c!='A',action_dim=args.action_dim,state_dim=args.state_dim,
            measured_force_invariant=True if c in 'AB' else None,C_force_input_reaches_encoder=c=='C',
            force_action_supervision=c!='A',force_action_output=c!='A',loss=float(loss),
            finite_gradients=True,checkpoint_roundtrip_identical=True,C_output_regression=regression,
            parameters=sum(p.numel() for p in policy.parameters()),split=stats['split_episode_files'],
            common_motion_normalizers_equal=True,smoke_checkpoint_sha256=digest(saved),optimizer_steps=0)
        (output/'training_smoke.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps({c:{k:v for k,v in results[c].items() if k!='split'}},indent=2),flush=True)
        del policy,optimizer,reload,loss,gradients,first,second,third,loader;gc.collect()
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','smoke','train'])
    parser.add_argument('--condition',choices=list('ABC'))
    args=parser.parse_args()
    if args.command=='prepare':prepare();print(EXPERIMENT/'training_configs.json')
    elif args.command=='smoke':smoke()
    else:
        if not args.condition:parser.error('--condition required')
        if not torch.cuda.is_available():raise RuntimeError('Full training requires CUDA; no silent CPU fallback')
        train_args=arguments(args.condition)
        # The common public initialization is downloaded only by an explicit training invocation if needed.
        run_one(train_args,'single_cam')


if __name__=='__main__':main()
