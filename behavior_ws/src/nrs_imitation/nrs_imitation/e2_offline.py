"""Offline transport fixture. Binds actual inference methods; never calls rclpy.init.

The only publishers/services are in-memory Python objects. Telemetry and
operator events are explicitly synthetic; no controller-applied values exist.
"""
import ast
from collections import deque
from pathlib import Path
import threading
from types import SimpleNamespace as NS, MethodType
import numpy as np
import torch
from . import inference_core as core
from .inference_metrics import InferenceMetrics, PARAMETERS
ROOT = Path(__file__).resolve().parents[4]
CKPT = ROOT / "checkpoints/flow/polishing/single_cam/e1_force_observation_20260916/off/20260916_1531"

class OfflineNode:
    """Bind repository methods, not a reimplementation of normalization/control."""
    @property
    def _parameters(self):
        # Let the installed rclpy Node's parameter enumeration run unchanged.
        # A fake list_parameters method previously hid an API absent in Humble.
        from rclpy.parameter import Parameter
        return {name: Parameter(name, value=value) for name, value in self.params.items()}

    def __getattr__(self, name):
        descriptor = core.NodeCmdMotionInfer.__dict__.get(name)
        if isinstance(descriptor, staticmethod):
            return descriptor.__func__
        method = getattr(core.NodeCmdMotionInfer, name, None)
        if callable(method):
            return MethodType(method, self)
        raise AttributeError(name)

    def __init__(self, root, policy=None, force_on=True):
        defaults = {}
        tree = ast.parse(Path(core.__file__).read_text())
        for item in ast.walk(tree):
            if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and item.func.attr == "declare_parameter" and len(item.args) == 2:
                try:
                    defaults[ast.literal_eval(item.args[0])] = ast.literal_eval(item.args[1])
                except (ValueError, TypeError):
                    pass
        self.__dict__.update(defaults)
        self.__dict__.update(dict(act_root=str(ROOT), ckpt_dir=str(CKPT), policy_class="FLOW", policy=policy,
            metrics_log_dir=str(root), metrics_run_tag="offline_OFF" if not force_on else "offline_ON",
            metrics_log_enable=True, use_force_observation=force_on, force_indices=[0, 1, 2],
            use_force_history=True, force_history_len=30, chunk_size=8, action_dim=9,
            use_gripper=False, use_gripper_history=False, use_global_image=False, use_stain_mask=False,
            rel_use_relative=True, rel_transform_version="stain_relative_v1", obs_force_xy_zeroed=False,
            _canon_setup_done=True, _canon_active=False, _canon_alpha=0., _srf_demo_start_absolutized=True,
            auto_move_to_demo_start=False, _demo_start_align_done=True, stage=core.Stage.TRACK,
            flow_replan_interval_steps=0, flow_local_anchor_enable=False, flow_infer_steps=2,
            device=torch.device("cpu"), camera_names=["cam0"], resize_hw=32,
            normalize_qpos_enabled=True, denorm_action_enabled=True, orientation_lock_enable=False,
            force_xy_cmd_enable=False, action_type="absolute", policy_z_offset_mm=0., fz_hard_limit=0.,
            _lock=threading.Lock(), _force_hist=deque(maxlen=30), _gripper_hist=deque(),
            _gripper_position=None, _gripper_current_mA=None, _gripper_last_pair_t=None,
            _pose6=None, _force=None, _img_cam0=np.full((32, 32, 3), 120, np.uint8), _img_cam1=None,
            _stain_mask=None, plans=deque(), _infer_plan_count=0, _flow_fixed_initial_noise=None,
            _flow_noise_create_count=0, flow_deterministic_noise=True, flow_noise_seed=0,
            _ptp9d_track_active=False, _ptp9d_inflight=False, _metrics=None,
            _contact=True, _cmd_safety_latched=False, _cmd_safety_hold_pose6=None,
            _start_pose6=None, prev_cmd=None, cmd_safety_enable=True, cmd_safety_max_xyz_from_current_mm=0.,
            _cmd_safety_last_log=0., _metrics_t0=core._monotonic(), errors=[]))
        self._cam_prev_raw_gray = self._cam_prev_proc_gray = None
        self._offline_mock_test = True
        if force_on:
            self.ckpt_dir = str(CKPT.parent.parent / "on/20260916_1531")
        from stain_relative_frame.relative_frame import RelativeFrameAdapter
        adapter = RelativeFrameAdapter([400., 500.], use_relative=True)
        self._srf = NS(ready=True, stain_origin=np.array([400., 500.]), stain_angle=1.57,
                       observation=adapter.observation, command=adapter.command)
        self.stats = None  # caller explicitly loads C stats; R/T never need a checkpoint
        self.params = dict(PARAMETERS, use_sim_time=False, **{k: v for k,v in self.__dict__.items() if isinstance(v, (str,bool,int,float))})
        self.get_parameter = lambda key: NS(value=self.params[key])
        self.get_clock = lambda: NS(now=lambda: NS(nanoseconds=123456789000))
        self.get_logger = lambda: NS(warn=lambda m: None, info=lambda m: None, error=self.errors.append)
        self.sent_messages = []
        self.pub_cmd = NS(publish=lambda m: self.sent_messages.append(list(m.data)))
        for name in ("_log_flow_vector_delta", "_run_gradcam_debug", "_run_modality_importance_debug", "_publish_flow_vector_overlay_frame"):
            setattr(self, name, lambda *a, **kw: None)  # debug-only; policy path is real


def registered_A_sample(config, output):
    """Actual registered weights + inference adapter; synthetic transport only.

    Keep prediction/feedback logs, including a second inference with perturbed
    measured force. This does not approve F0 or initialize a ROS context.
    """
    import csv
    import json
    import h5py
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float64MultiArray
    from .e2_ablation import select_condition, model_errors, read_stats, digest

    from .e2_direct_a import is_direct_a
    direct = is_direct_a(config)
    cfg = config if direct else select_condition(config, 'A')
    errors = [] if direct else model_errors(cfg)
    if errors:
        raise ValueError('; '.join(errors))
    checkpoint = Path(cfg['il']['checkpoint'] if direct else cfg['models']['A']['checkpoint'])
    stats = read_stats(checkpoint.with_name('dataset_stats.pkl'))
    pc = stats['policy_config']
    for key, expected in dict(state_dim=6, action_dim=6, motion_only=True,
                              force_action=False, use_force_observation=False,
                              use_force_history=False).items():
        assert pc[key] == expected, key
    torch.set_num_threads(2)
    loader = OfflineNode(Path(output)/'loader', force_on=False)
    loader.ckpt_dir = str(checkpoint.parent)
    loader.checkpoint = str(checkpoint)
    loader.motion_only = True; loader.action_dim = 6; loader.chunk_size = 128
    loader.use_force_history = False; loader.flow_infer_steps = 10
    loader.params.update(ckpt_dir=loader.ckpt_dir, use_force_history=False,
                         use_force_observation=False, chunk_size=128)
    policy = loader._load_policy_and_ckpt_from_act_root()
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    selected_epoch = int(ck['epoch'])
    assert selected_epoch >= 0
    actual, saved = policy.state_dict(), ck['model_state_dict']
    assert actual.keys() == saved.keys()
    assert all(torch.equal(actual[k], v) for k, v in saved.items())
    del ck, actual, saved
    episode = stats['split_episode_files']['validation'][0]
    path = ROOT/'datasets/polishing/single_cam/20260910_90deg_rel_single/imitation_form'/episode
    with h5py.File(path) as f:
        index = len(f['observations/position'])//2
        pose = np.asarray(f['observations/position'][index], float)[:6]
        frame = np.asarray(f['observations/images/cam0'][index], np.uint8)
    pose[:2] += [400., 500.]  # explicitly synthetic frozen stain translation
    predictions = []
    for i, force in enumerate(([1., 2., 3.], [-73., 51., -99.])):
        node = OfflineNode(Path(output)/('provider_'+str(i)), policy, force_on=False)
        node.ckpt_dir = str(checkpoint.parent); node.motion_only = True
        node.checkpoint = str(checkpoint)
        node.action_dim = 6; node.chunk_size = 128; node.use_force_history = False
        node.flow_infer_steps = 10; node.resize_hw = 0; node._e2_context = cfg
        node.stats = core._load_dataset_stats(node.ckpt_dir)
        node.params.update(ckpt_dir=node.ckpt_dir, chunk_size=128, flow_infer_steps=10,
                           resize_hw=0, use_force_history=False, use_force_observation=False)
        node._metrics = InferenceMetrics(node)
        try:
            node._on_pose(Float64MultiArray(data=pose.tolist()))
            for _ in range(30):
                node._on_force(Float64MultiArray(data=list(force)+[0., 0., 0.]))
            node._on_img(Image(height=frame.shape[0], width=frame.shape[1], encoding='rgb8',
                               step=frame.shape[1]*3, data=frame.tobytes()))
            node._on_infer_timer()
            assert not node.errors and len(node.plans) == 1, node.errors
            action = node.plans[-1].seq_den.copy()
            assert action.shape == (128, 9) and np.isfinite(action).all()
            assert np.all(action[:, 6:] == 0.)
            predictions.append(action)
        finally:
            node._metrics.close()
        with (node._metrics.recorder.path/'commands.csv').open() as f:
            rows = [r for r in csv.DictReader(f) if r['command_stage'] == 'policy_prediction']
        assert len(rows) == 128
        assert all(len(json.loads(r['raw_values'])) == 6 and r['fz'] == '' and
                   json.loads(r['details'])['predicted_force'] is None for r in rows)
        summary = json.loads((node._metrics.recorder.path/'summary.json').read_text())
        assert summary['drained'] and summary['write_error_count'] == 0
        assert all(v['dropped'] == 0 for v in summary['counts'].values())
    np.testing.assert_array_equal(*predictions)
    return predictions[0], pose, dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
        selected_epoch_zero_based=selected_epoch, loaded_weights_exactly_match_checkpoint=True,
        state_dim=6, action_dim=6, force_observation=False, force_history=False,
        force_action=False, force_supervision_channels=0, force_prediction_logged_as_null=True,
        force_perturbation_invariant=True, dataset=str(path), sample_index=index,
        output_shape=[128, 6], transport_shape=[128, 9],
        ros_context_initialized=False, hardware_io=False)
