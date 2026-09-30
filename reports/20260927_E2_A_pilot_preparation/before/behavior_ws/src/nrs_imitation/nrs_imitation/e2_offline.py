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
