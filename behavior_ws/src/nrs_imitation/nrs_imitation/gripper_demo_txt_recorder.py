#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gripper_demo_txt_recorder.py

Records one txt trajectory with:
  position(6) + force(3) + gripper_present_position + gripper_present_current_mA

The robot trajectory uses the same Stage-1 filtering pipeline as
vr_demo_txt_recorder. Gripper state is resampled onto the filtered trajectory
timebase with nearest-source indexing.
"""

import os
import time
import subprocess
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String, Int32, Float32
from geometry_msgs.msg import Wrench

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nrs_imitation.pretty_print import block
from nrs_imitation.recorder_sync import TimedValueBuffer
from nrs_imitation.stage1_filtering import (
    apply_stage1_filter,
    stage1_config_from_recorder,
    take_nearest_by_source_index,
)
from nrs_imitation.vr_demo_txt_recorder import (
    Limits,
    eval_qp_proxy,
    print_eval,
    save_plot_1_lin_kinematics,
    save_plot_2_rotvec_kinematics,
    save_plot_3_forces,
)


REPO_ROOT = os.path.expanduser("~/nrs_imitation")
DEFAULT_SAVE_PATH = os.path.join(
    REPO_ROOT,
    "behavior_ws",
    "src",
    "nrs_imitation",
    "txtcmd",
    "gripper_cmd_continue11D.txt",
)
DEFAULT_VIZ_ROOT = os.path.join(REPO_ROOT, "behavior_ws", "src", "nrs_imitation", "log")


def _time_axes_time_aligned(dt: float, rawN: int, filN: int):
    if filN <= 0:
        t_after = np.zeros((0,), dtype=np.float64)
        T_after = 0.0
    elif filN == 1:
        t_after = np.array([0.0], dtype=np.float64)
        T_after = 0.0
    else:
        t_after = np.arange(filN, dtype=np.float64) * dt
        T_after = float(t_after[-1])

    if rawN <= 0:
        t_before = np.zeros((0,), dtype=np.float64)
    elif rawN == 1:
        t_before = np.array([0.0], dtype=np.float64)
    else:
        t_before = np.linspace(0.0, T_after, rawN, dtype=np.float64)

    return t_before, t_after


def save_plot_4_gripper_states(viz_dir: str, dt: float, rawG: np.ndarray, filtG: np.ndarray):
    rawN = rawG.shape[0]
    filN = filtG.shape[0]
    t_raw, t_fil = _time_axes_time_aligned(dt, rawN, filN)

    fig = plt.figure(figsize=(14, 6))
    specs = [
        ("present_position", "tick", 0),
        ("present_current_mA", "mA", 1),
    ]
    for i, (name, ylabel, col) in enumerate(specs, start=1):
        ax = plt.subplot(1, 2, i)
        ax.plot(t_raw, rawG[:, col], label="before")
        ax.plot(t_fil, filtG[:, col], label="after")
        ax.set_title(name)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("time [s]")
        ax.grid(True)
        if i == 1:
            ax.legend()

    fig.suptitle("Gripper states: present_position / present_current_mA (before vs after, AFTER-time-aligned)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    outpath = os.path.join(viz_dir, "plot_4_gripper_states.png")
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


class GripperDemoTxtRecorder(Node):
    def __init__(self):
        super().__init__("gripper_demo_txt_recorder")

        # topics
        self.declare_parameter("pose_topic", "/calibrated_pose")
        self.declare_parameter("force_topic", "/ftsensor/measured_Cvalue")
        self.declare_parameter("command_topic", "/vr_demo_recorder/command")
        self.declare_parameter("gripper_position_topic", "/gripper/present_position")
        self.declare_parameter("gripper_current_topic", "/gripper/present_current_mA")

        # timing
        self.declare_parameter("record_hz", 125.0)
        self.declare_parameter("require_fresh_sec", 0.2)
        self.declare_parameter("require_gripper_fresh_sec", 0.5)
        self.declare_parameter("sync_enable", True)
        self.declare_parameter("sync_delay_sec", 0.01)
        self.declare_parameter("sync_max_error_sec", 0.05)
        self.declare_parameter("sync_buffer_sec", 1.0)

        # save / viz
        self.declare_parameter("save_path", DEFAULT_SAVE_PATH)
        self.declare_parameter("viz_root", DEFAULT_VIZ_ROOT)

        # SCP transfer
        self.declare_parameter("transfer_enable", True)
        self.declare_parameter("remote_user", "nrs_forcecon")
        self.declare_parameter("remote_ip", "192.168.0.151")
        self.declare_parameter("remote_dir", "dev_ws/src/y2_ur10skku_control/Y2RobMotion/txtcmd/")

        # Stage-1-compatible force / pose trajectory filtering
        self.declare_parameter("force_filter_mode", "ema")  # ema | contact_cleanup
        self.declare_parameter("zero_xy_forces", False)
        self.declare_parameter("force_clamp_abs", 200.0)
        self.declare_parameter("force_ema_alpha", 0.2)
        self.declare_parameter("contact_thr_N", 5.0)
        self.declare_parameter("consec_on", 10)
        self.declare_parameter("consec_off", 10)
        self.declare_parameter("fz_contact_smooth_enable", True)
        self.declare_parameter("fz_contact_lam_d2", 4000.0)
        self.declare_parameter("hampel_enable", True)
        self.declare_parameter("hampel_win", 16)
        self.declare_parameter("hampel_sig", 2.0)
        self.declare_parameter("lam_pos_d2", 250000.0)
        self.declare_parameter("lam_ang_d2", 6000.0)
        self.declare_parameter("pose_ema_enable", True)
        self.declare_parameter("pose_ema_alpha", 0.10)
        self.declare_parameter("retime_k", 2)
        self.declare_parameter("approach_slowdown_enable", True)
        self.declare_parameter("approach_pre_sec", 5.0)
        self.declare_parameter("approach_post_sec", 0.3)
        self.declare_parameter("approach_scale_max", 30.0)
        self.declare_parameter("approach_use_fz_ramp", True)
        self.declare_parameter("approach_fz_full", 20.0)
        self.declare_parameter("post_enable", True)
        self.declare_parameter("lam_pos_d3", 2.0e7)
        self.declare_parameter("lam_ang_d3", 6.0e5)
        self.declare_parameter("qp_guard_enable", True)
        self.declare_parameter("qp_guard_safety", 0.75)
        self.declare_parameter("qp_guard_max_iter", 8)
        self.declare_parameter("qp_guard_growth", 2.2)
        self.declare_parameter("max_dev_pos_mm", 8.0)
        self.declare_parameter("max_dev_ang_rad", 0.06)
        self.declare_parameter("cg_iters", 400)
        self.declare_parameter("cg_tol", 1e-8)
        self.declare_parameter("pos_vmax", 30.0)
        self.declare_parameter("pos_amax", 120.0)
        self.declare_parameter("ang_vmax", 0.6)
        self.declare_parameter("ang_amax", 3.0)
        self.declare_parameter("pos_jmax", 5000.0)
        self.declare_parameter("ang_jmax", 80.0)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.force_topic = str(self.get_parameter("force_topic").value)
        self.command_topic = str(self.get_parameter("command_topic").value)
        self.gripper_position_topic = str(self.get_parameter("gripper_position_topic").value)
        self.gripper_current_topic = str(self.get_parameter("gripper_current_topic").value)

        self.record_hz = float(self.get_parameter("record_hz").value)
        self.dt = 1.0 / max(1e-9, self.record_hz)
        self.require_fresh_sec = float(self.get_parameter("require_fresh_sec").value)
        self.require_gripper_fresh_sec = float(self.get_parameter("require_gripper_fresh_sec").value)
        self.sync_enable = bool(self.get_parameter("sync_enable").value)
        self.sync_delay_sec = float(self.get_parameter("sync_delay_sec").value)
        self.sync_max_error_sec = float(self.get_parameter("sync_max_error_sec").value)
        self.sync_buffer_sec = float(self.get_parameter("sync_buffer_sec").value)

        self.save_path = os.path.expanduser(str(self.get_parameter("save_path").value))
        self.viz_root = os.path.expanduser(str(self.get_parameter("viz_root").value))

        self.transfer_enable = bool(self.get_parameter("transfer_enable").value)
        self.remote_user = str(self.get_parameter("remote_user").value)
        self.remote_ip = str(self.get_parameter("remote_ip").value)
        self.remote_dir = str(self.get_parameter("remote_dir").value)

        self.force_filter_mode = str(self.get_parameter("force_filter_mode").value)
        self.zero_xy_forces = bool(self.get_parameter("zero_xy_forces").value)
        self.force_clamp_abs = float(self.get_parameter("force_clamp_abs").value)
        self.force_ema_alpha = float(self.get_parameter("force_ema_alpha").value)
        self.contact_thr_N = float(self.get_parameter("contact_thr_N").value)
        self.consec_on = int(self.get_parameter("consec_on").value)
        self.consec_off = int(self.get_parameter("consec_off").value)
        self.fz_contact_smooth_enable = bool(self.get_parameter("fz_contact_smooth_enable").value)
        self.fz_contact_lam_d2 = float(self.get_parameter("fz_contact_lam_d2").value)
        self.hampel_enable = bool(self.get_parameter("hampel_enable").value)
        self.hampel_win = int(self.get_parameter("hampel_win").value)
        self.hampel_sig = float(self.get_parameter("hampel_sig").value)
        self.lam_pos_d2 = float(self.get_parameter("lam_pos_d2").value)
        self.lam_ang_d2 = float(self.get_parameter("lam_ang_d2").value)
        self.pose_ema_enable = bool(self.get_parameter("pose_ema_enable").value)
        self.pose_ema_alpha = float(self.get_parameter("pose_ema_alpha").value)
        self.retime_k = int(self.get_parameter("retime_k").value)
        self.approach_slowdown_enable = bool(self.get_parameter("approach_slowdown_enable").value)
        self.approach_pre_sec = float(self.get_parameter("approach_pre_sec").value)
        self.approach_post_sec = float(self.get_parameter("approach_post_sec").value)
        self.approach_scale_max = float(self.get_parameter("approach_scale_max").value)
        self.approach_use_fz_ramp = bool(self.get_parameter("approach_use_fz_ramp").value)
        self.approach_fz_full = float(self.get_parameter("approach_fz_full").value)
        self.post_enable = bool(self.get_parameter("post_enable").value)
        self.lam_pos_d3 = float(self.get_parameter("lam_pos_d3").value)
        self.lam_ang_d3 = float(self.get_parameter("lam_ang_d3").value)
        self.qp_guard_enable = bool(self.get_parameter("qp_guard_enable").value)
        self.qp_guard_safety = float(self.get_parameter("qp_guard_safety").value)
        self.qp_guard_max_iter = int(self.get_parameter("qp_guard_max_iter").value)
        self.qp_guard_growth = float(self.get_parameter("qp_guard_growth").value)
        self.max_dev_pos_mm = float(self.get_parameter("max_dev_pos_mm").value)
        self.max_dev_ang_rad = float(self.get_parameter("max_dev_ang_rad").value)
        self.cg_iters = int(self.get_parameter("cg_iters").value)
        self.cg_tol = float(self.get_parameter("cg_tol").value)
        self.pos_vmax = float(self.get_parameter("pos_vmax").value)
        self.pos_amax = float(self.get_parameter("pos_amax").value)
        self.ang_vmax = float(self.get_parameter("ang_vmax").value)
        self.ang_amax = float(self.get_parameter("ang_amax").value)
        self.pos_jmax = float(self.get_parameter("pos_jmax").value)
        self.ang_jmax = float(self.get_parameter("ang_jmax").value)
        self.lim = Limits(
            pos_vmax=self.pos_vmax,
            pos_amax=self.pos_amax,
            ang_vmax=self.ang_vmax,
            ang_amax=self.ang_amax,
            pos_jmax=self.pos_jmax,
            ang_jmax=self.ang_jmax,
        )

        self.latest_pose6_mm_rad: Optional[np.ndarray] = None
        self.latest_force3_N: Optional[np.ndarray] = None
        self.latest_gripper_position: Optional[int] = None
        self.latest_gripper_current_mA: Optional[float] = None
        self.latest_pose_t = 0.0
        self.latest_force_t = 0.0
        self.latest_gripper_position_t = 0.0
        self.latest_gripper_current_t = 0.0
        self.pose_sync_buffer = TimedValueBuffer(self.sync_buffer_sec)
        self.force_sync_buffer = TimedValueBuffer(self.sync_buffer_sec)
        self.gripper_position_sync_buffer = TimedValueBuffer(self.sync_buffer_sec)
        self.gripper_current_sync_buffer = TimedValueBuffer(self.sync_buffer_sec)

        self.episode_active = False
        self.finishing_ = False
        self.buf_pose = []
        self.buf_force = []
        self.buf_gripper = []

        self.sub_pose = self.create_subscription(Float64MultiArray, self.pose_topic, self.cb_pose, 50)
        self.sub_force = self.create_subscription(Wrench, self.force_topic, self.cb_force, 10)
        self.sub_gripper_position = self.create_subscription(Int32, self.gripper_position_topic, self.cb_gripper_position, 10)
        self.sub_gripper_current = self.create_subscription(Float32, self.gripper_current_topic, self.cb_gripper_current, 10)
        self.sub_command = self.create_subscription(String, self.command_topic, self.cb_command, 10)
        self.timer = self.create_timer(self.dt, self.cb_timer)

        self.get_logger().info(block("gripper_demo_txt_recorder READY", [
            ("save", self.save_path),
            ("pose", self.pose_topic),
            ("force", self.force_topic),
            ("gripper", f"pos={self.gripper_position_topic}, cur={self.gripper_current_topic}"),
            ("record_hz", self.record_hz),
            ("sync", f"timer delay={self.sync_delay_sec:.3f}s, max_error={self.sync_max_error_sec:.3f}s"),
            ("filter", f"stage1 retime=x{self.retime_k}, force={self.force_filter_mode}, approach={self.approach_slowdown_enable}, QP={self.qp_guard_enable}"),
            ("columns", "x y z rx ry rz fx fy fz gripper_present_position gripper_present_current_mA"),
            ("command", self.command_topic),
        ]))

    def cb_pose(self, msg: Float64MultiArray):
        if len(msg.data) < 6:
            return
        x, y, z, rx, ry, rz = msg.data[:6]
        value = np.array([1000.0 * x, 1000.0 * y, 1000.0 * z, rx, ry, rz], dtype=np.float64)
        stamp = time.time()
        self.latest_pose6_mm_rad = value
        self.latest_pose_t = stamp
        self.pose_sync_buffer.add(stamp, value)

    def cb_force(self, msg: Wrench):
        value = np.array([msg.force.x, msg.force.y, msg.force.z], dtype=np.float64)
        stamp = time.time()
        self.latest_force3_N = value
        self.latest_force_t = stamp
        self.force_sync_buffer.add(stamp, value)

    def cb_gripper_position(self, msg: Int32):
        value = int(msg.data)
        stamp = time.time()
        self.latest_gripper_position = value
        self.latest_gripper_position_t = stamp
        self.gripper_position_sync_buffer.add(stamp, np.asarray(value, dtype=np.int32))

    def cb_gripper_current(self, msg: Float32):
        value = float(msg.data)
        stamp = time.time()
        self.latest_gripper_current_mA = value
        self.latest_gripper_current_t = stamp
        self.gripper_current_sync_buffer.add(stamp, np.asarray(value, dtype=np.float32))

    def cb_command(self, msg: String):
        cmd = str(msg.data).strip().lower()
        if not cmd:
            return
        self.get_logger().warn(f"[COMMAND] {cmd}")
        if cmd == "start_recording":
            self.start_episode(reason="joystick_start")
        elif cmd == "end_recording":
            self.end_episode(reason="joystick_end")
        else:
            self.get_logger().warn(f"[COMMAND] unknown command ignored: {cmd}")

    def start_episode(self, reason: str = "start"):
        if self.finishing_:
            self.get_logger().warn("Cannot start episode: previous episode is still being saved.")
            return
        if self.episode_active:
            self.get_logger().warn("Episode already active.")
            return
        self.episode_active = True
        self.buf_pose.clear()
        self.buf_force.clear()
        self.buf_gripper.clear()
        self.get_logger().info(f"=== EPISODE STARTED ({reason}) ===")

    def end_episode(self, reason: str = "end"):
        if not self.episode_active:
            self.get_logger().warn("No active episode to end.")
            return
        if self.finishing_:
            self.get_logger().warn("Episode already finishing.")
            return
        self.get_logger().info(f"=== EPISODE ENDED ({reason}) ===")
        self.finish_episode()

    def cb_timer(self):
        if (not self.episode_active) or self.finishing_:
            return
        now = time.time()
        if self.sync_enable:
            target = now - max(0.0, self.sync_delay_sec)
            pose_result = self.pose_sync_buffer.sample(target, mode="linear")
            force_result = self.force_sync_buffer.sample(target, mode="linear")
            grip_pos_result = self.gripper_position_sync_buffer.sample(target, mode="nearest")
            grip_cur_result = self.gripper_current_sync_buffer.sample(target, mode="nearest")
            results = (pose_result, force_result, grip_pos_result, grip_cur_result)
            if any(result is None for result in results):
                return
            if any(result.error_sec > self.sync_max_error_sec for result in results):
                return
            self.buf_pose.append(pose_result.value.copy())
            self.buf_force.append(force_result.value.copy())
            self.buf_gripper.append(np.array([int(grip_pos_result.value), float(grip_cur_result.value)], dtype=np.float64))
            return
        if self.latest_pose6_mm_rad is None or (now - self.latest_pose_t) > self.require_fresh_sec:
            return
        if self.latest_force3_N is None or (now - self.latest_force_t) > self.require_fresh_sec:
            return
        if self.latest_gripper_position is None or (now - self.latest_gripper_position_t) > self.require_gripper_fresh_sec:
            return
        if self.latest_gripper_current_mA is None or (now - self.latest_gripper_current_t) > self.require_gripper_fresh_sec:
            return

        self.buf_pose.append(self.latest_pose6_mm_rad.copy())
        self.buf_force.append(self.latest_force3_N.copy())
        self.buf_gripper.append(np.array([self.latest_gripper_position, self.latest_gripper_current_mA], dtype=np.float64))

    def _make_viz_dir(self) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        out_dir = os.path.join(self.viz_root, ts)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _save_viz(self, out_dir: str, rawP: np.ndarray, rawF: np.ndarray, rawG: np.ndarray, filtP: np.ndarray, filtF: np.ndarray, filtG: np.ndarray):
        try:
            save_plot_1_lin_kinematics(out_dir, self.dt, rawP, filtP)
            save_plot_2_rotvec_kinematics(out_dir, self.dt, rawP, filtP)
            save_plot_3_forces(out_dir, self.dt, rawF, filtF)
            save_plot_4_gripper_states(out_dir, self.dt, rawG, filtG)
            self.get_logger().info(f"[VIZ] Saved plots to: {out_dir}")
        except Exception as e:
            self.get_logger().error(f"[VIZ] Failed to save plots: {e}")

    def finish_episode(self):
        if self.finishing_:
            return
        self.finishing_ = True
        self.episode_active = False

        if len(self.buf_pose) < 10:
            self.get_logger().warn("Episode too short. Discarding.")
            rclpy.shutdown()
            return

        rawP = np.asarray(self.buf_pose, dtype=np.float64)
        rawF = np.asarray(self.buf_force, dtype=np.float64)
        rawG = np.asarray(self.buf_gripper, dtype=np.float64).reshape(-1, 2)

        st0, _ = eval_qp_proxy(rawP, self.dt, self.lim, safety=1.0)
        print_eval(self.get_logger(), "RAW (before)", st0, self.lim, 1.0)

        filter_result = apply_stage1_filter(
            rawP,
            rawF,
            stage1_config_from_recorder(self, self.record_hz),
            logger=self.get_logger(),
        )
        Pf = filter_result.position
        Fr = filter_result.force
        Gf = take_nearest_by_source_index(rawG, filter_result.source_index).astype(np.float32)

        st2, _ = eval_qp_proxy(Pf, self.dt, self.lim, safety=1.0)
        print_eval(self.get_logger(), "FINAL pose (stage1-filtered)", st2, self.lim, 1.0)

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        out11 = np.hstack([Pf, Fr, Gf])
        with open(self.save_path, "w") as f:
            for row in out11:
                f.write("\t".join([f"{v:.6f}" for v in row.tolist()]) + "\n")
        self.get_logger().info(f"Saved: {self.save_path}  (rows={out11.shape[0]}, cols={out11.shape[1]})")

        viz_dir = self._make_viz_dir()
        self._save_viz(viz_dir, rawP, rawF, rawG, Pf, Fr, Gf)

        if self.transfer_enable:
            self._transfer_file()

        self.get_logger().info("Shutting down (end condition met).")
        rclpy.shutdown()

    def _transfer_file(self):
        try:
            self.get_logger().info(f"Sending file to Control PC ({self.remote_ip})...")
            dst = f"{self.remote_user}@{self.remote_ip}:{self.remote_dir}"
            subprocess.run(["scp", self.save_path, dst], check=True)
            self.get_logger().info(f"SUCCESS: transferred to {self.remote_dir}")
        except Exception as e:
            self.get_logger().error(f"FAILED: scp transfer error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = GripperDemoTxtRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
