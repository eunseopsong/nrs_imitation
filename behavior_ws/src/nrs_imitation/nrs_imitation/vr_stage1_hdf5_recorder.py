#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vr_stage1_hdf5_recorder.py

Stage-1 human demo recorder.

Records only:
  - VR tracker pose : /calibrated_pose              Float64MultiArray [x y z wx wy wz]
  - FT sensor       : /ftsensor/measured_Cvalue     geometry_msgs/Wrench [fx fy fz]

Saves one HDF5 file per episode:

  /home/eunseop/nrs_imitation/datasets/ACT/YYYYMMDD_HHMM/stage1_vr_episodes/
    episode_0000.hdf5
    episode_0001.hdf5
    ...

Each file contains:
  traj      (T,9) = [x_mm y_mm z_mm wx wy wz fx fy fz]
  position  (T,6)
  force     (T,3)

Episode trigger:
  - auto start: |Fx| >= start_abs_fx
  - auto end  : |Fy| >= stop_abs_fy

Optional command topic:
  /vr_stage1_recorder/command std_msgs/String
    start_recording
    end_recording
    discard_current_episode
    terminate_node
"""

from __future__ import annotations

import os
import time
import json
import atexit
import threading
import select
import termios
import tty
from datetime import datetime
from typing import Optional, List

import numpy as np
import h5py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import Wrench


def make_qos(depth: int = 10, best_effort: bool = False) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=int(depth),
        reliability=ReliabilityPolicy.BEST_EFFORT if best_effort else ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


class KeyboardQuitter:
    def __init__(self, quit_key: str = "q"):
        self.quit_key = (quit_key or "q").lower()
        self._stop_evt = threading.Event()
        self._hit_quit = threading.Event()
        self._thread = None
        self._fd = None
        self._old = None

    def start(self) -> bool:
        import sys
        if not sys.stdin.isatty():
            return False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self._restore()

    def hit(self) -> bool:
        return self._hit_quit.is_set()

    def _restore(self):
        try:
            if self._fd is not None and self._old is not None:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        except Exception:
            pass

    def _loop(self):
        import sys
        try:
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            while not self._stop_evt.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch and ch.lower() == self.quit_key:
                    self._hit_quit.set()
                    break
        except Exception:
            pass
        finally:
            self._restore()


class VRStage1HDF5Recorder(Node):
    def __init__(self):
        super().__init__("vr_stage1_hdf5_recorder")

        # Save path.
        self.declare_parameter("act_root_dir", "/home/eunseop/nrs_imitation/datasets/ACT")
        self.declare_parameter("output_subdir", "stage1_vr_episodes")
        self.declare_parameter("run_timestamp", "")  # empty -> now YYYYMMDD_HHMM
        self.declare_parameter("overwrite_episode", True)

        # Topics.
        self.declare_parameter("pose_topic", "/calibrated_pose")
        self.declare_parameter("force_topic", "/ftsensor/measured_Cvalue")
        self.declare_parameter("command_topic", "/vr_stage1_recorder/command")

        # Timing / freshness.
        self.declare_parameter("sample_hz", 125.0)
        self.declare_parameter("require_pose_fresh_sec", 0.20)
        self.declare_parameter("require_force_fresh_sec", 0.20)
        self.declare_parameter("status_period_sec", 1.0)

        # Episode trigger.
        self.declare_parameter("auto_trigger_enable", True)
        self.declare_parameter("start_abs_fx", 10.0)
        self.declare_parameter("stop_abs_fy", 10.0)
        self.declare_parameter("min_samples", 10)

        # Units.
        self.declare_parameter("pose_xyz_scale", 1000.0)  # m -> mm

        # Quit.
        self.declare_parameter("quit_key", "q")

        self.act_root_dir = str(self.get_parameter("act_root_dir").value)
        self.output_subdir = str(self.get_parameter("output_subdir").value)
        ts = str(self.get_parameter("run_timestamp").value).strip()
        self.run_timestamp = ts if ts else datetime.now().strftime("%Y%m%d_%H%M")
        self.overwrite_episode = bool(self.get_parameter("overwrite_episode").value)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.force_topic = str(self.get_parameter("force_topic").value)
        self.command_topic = str(self.get_parameter("command_topic").value)

        self.sample_hz = float(self.get_parameter("sample_hz").value)
        self.dt = 1.0 / max(self.sample_hz, 1e-9)
        self.require_pose_fresh_sec = float(self.get_parameter("require_pose_fresh_sec").value)
        self.require_force_fresh_sec = float(self.get_parameter("require_force_fresh_sec").value)
        self.status_period_sec = float(self.get_parameter("status_period_sec").value)

        self.auto_trigger_enable = bool(self.get_parameter("auto_trigger_enable").value)
        self.start_abs_fx = float(self.get_parameter("start_abs_fx").value)
        self.stop_abs_fy = float(self.get_parameter("stop_abs_fy").value)
        self.min_samples = int(self.get_parameter("min_samples").value)
        self.pose_xyz_scale = float(self.get_parameter("pose_xyz_scale").value)
        self.quit_key = str(self.get_parameter("quit_key").value)

        self.output_dir = os.path.join(self.act_root_dir, self.run_timestamp, self.output_subdir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.lock = threading.Lock()
        self.latest_pose: Optional[np.ndarray] = None
        self.latest_force: Optional[np.ndarray] = None
        self.latest_pose_t = 0.0
        self.latest_force_t = 0.0

        self.recording = False
        self.finishing = False
        self.ep_idx = self._next_episode_index()

        self.buf_pose: List[np.ndarray] = []
        self.buf_force: List[np.ndarray] = []
        self.buf_t: List[float] = []

        self.last_status_t = 0.0
        self.stop_requested = False

        qos = make_qos(50, best_effort=False)
        self.create_subscription(Float64MultiArray, self.pose_topic, self._pose_cb, qos)
        self.create_subscription(Wrench, self.force_topic, self._force_cb, qos)
        self.create_subscription(String, self.command_topic, self._cmd_cb, qos)

        self.create_timer(self.dt, self._sample_timer)
        self.create_timer(0.10, self._quit_timer)

        self.kb = KeyboardQuitter(self.quit_key)
        kb_ok = self.kb.start()
        atexit.register(self.kb.stop)

        self.get_logger().info(
            "[READY] VRStage1HDF5Recorder\n"
            f"  output_dir={self.output_dir}\n"
            f"  pose_topic={self.pose_topic}\n"
            f"  force_topic={self.force_topic}\n"
            f"  sample_hz={self.sample_hz}\n"
            f"  auto_trigger={int(self.auto_trigger_enable)} start=|Fx|>={self.start_abs_fx}, end=|Fy|>={self.stop_abs_fy}\n"
            f"  next_ep=episode_{self.ep_idx:04d}.hdf5\n"
            f"  command_topic={self.command_topic}\n"
            f"  keyboard_quit={'on' if kb_ok else 'off'}"
        )

    def _next_episode_index(self) -> int:
        ids = []
        if os.path.isdir(self.output_dir):
            for name in os.listdir(self.output_dir):
                if name.startswith("episode_") and name.endswith(".hdf5"):
                    try:
                        ids.append(int(name[len("episode_"):-len(".hdf5")]))
                    except Exception:
                        pass
        return max(ids) + 1 if ids else 0

    def _pose_cb(self, msg: Float64MultiArray):
        if len(msg.data) < 6:
            return
        pose = np.asarray(msg.data[:6], dtype=np.float32)
        pose[:3] *= np.float32(self.pose_xyz_scale)
        with self.lock:
            self.latest_pose = pose
            self.latest_pose_t = time.time()

    def _force_cb(self, msg: Wrench):
        f = np.asarray([msg.force.x, msg.force.y, msg.force.z], dtype=np.float32)
        with self.lock:
            self.latest_force = f
            self.latest_force_t = time.time()

        if self.auto_trigger_enable and not self.stop_requested:
            if (not self.recording) and (not self.finishing) and abs(float(f[0])) >= self.start_abs_fx:
                self.start_episode(reason="auto_fx_threshold")
            elif self.recording and abs(float(f[1])) >= self.stop_abs_fy:
                self.end_episode(reason="auto_fy_threshold")

    def _cmd_cb(self, msg: String):
        cmd = str(msg.data).strip().lower()
        self.get_logger().warn(f"[COMMAND] {cmd}")

        if cmd == "start_recording":
            self.start_episode(reason="command")
        elif cmd == "end_recording":
            self.end_episode(reason="command")
        elif cmd == "discard_current_episode":
            self.discard_current(reason="command")
        elif cmd == "terminate_node":
            self.stop_requested = True
            if self.recording:
                self.end_episode(reason="terminate")
            else:
                self._shutdown()

    def _quit_timer(self):
        if self.kb.hit() and not self.stop_requested:
            self.stop_requested = True
            self.get_logger().warn(f"[STOP] keyboard '{self.quit_key}'")
            if self.recording:
                self.end_episode(reason=f"keyboard_{self.quit_key}")
            elif not self.finishing:
                self._shutdown()

    def start_episode(self, reason: str):
        if self.recording or self.finishing:
            return
        self.buf_pose.clear()
        self.buf_force.clear()
        self.buf_t.clear()
        self.recording = True
        self.get_logger().warn(f"[START] episode_{self.ep_idx:04d}.hdf5 reason={reason}")

    def end_episode(self, reason: str):
        if not self.recording or self.finishing:
            return
        self.recording = False
        self.finishing = True

        pose = list(self.buf_pose)
        force = list(self.buf_force)
        ts = list(self.buf_t)
        ep_idx = int(self.ep_idx)

        th = threading.Thread(target=self._save_worker, args=(ep_idx, pose, force, ts, reason), daemon=True)
        th.start()

    def discard_current(self, reason: str):
        if self.recording:
            self.recording = False
            self.buf_pose.clear()
            self.buf_force.clear()
            self.buf_t.clear()
            self.get_logger().warn(f"[DISCARD] current recording reason={reason}")

    def _sample_timer(self):
        if not self.recording or self.finishing:
            return

        now = time.time()
        with self.lock:
            pose = None if self.latest_pose is None else self.latest_pose.copy()
            force = None if self.latest_force is None else self.latest_force.copy()
            pose_age = now - self.latest_pose_t if self.latest_pose_t > 0 else 1e9
            force_age = now - self.latest_force_t if self.latest_force_t > 0 else 1e9

        missing = []
        if pose is None or pose_age > self.require_pose_fresh_sec:
            missing.append(f"pose(age={pose_age:.3f})")
        if force is None or force_age > self.require_force_fresh_sec:
            missing.append(f"force(age={force_age:.3f})")

        if missing:
            if now - self.last_status_t >= self.status_period_sec:
                self.last_status_t = now
                self.get_logger().warn("[WAIT] " + ", ".join(missing))
            return

        self.buf_pose.append(pose.astype(np.float32))
        self.buf_force.append(force[:3].astype(np.float32))
        self.buf_t.append(float(now))

        if now - self.last_status_t >= self.status_period_sec:
            self.last_status_t = now
            self.get_logger().info(
                f"[REC] ep={self.ep_idx:04d} samples={len(self.buf_pose)} "
                f"force=[{force[0]:.2f},{force[1]:.2f},{force[2]:.2f}]"
            )

    def _save_worker(self, ep_idx: int, pose_list, force_list, t_list, reason: str):
        try:
            N = len(pose_list)
            if N < self.min_samples:
                self.get_logger().warn(f"[DROP] episode_{ep_idx:04d}: too short N={N} < {self.min_samples}")
                return

            P = np.asarray(pose_list, dtype=np.float32).reshape(N, 6)
            F = np.asarray(force_list, dtype=np.float32).reshape(N, 3)
            traj = np.concatenate([P, F], axis=1).astype(np.float32)

            out_path = os.path.join(self.output_dir, f"episode_{ep_idx:04d}.hdf5")
            if os.path.exists(out_path) and self.overwrite_episode:
                os.remove(out_path)

            with h5py.File(out_path, "w") as f:
                f.attrs["schema_version"] = "stage1_vr_episode_v1"
                f.attrs["saved_unix"] = float(time.time())
                f.attrs["reason"] = str(reason)
                f.attrs["sample_hz"] = float(self.sample_hz)
                f.attrs["pose_topic"] = str(self.pose_topic)
                f.attrs["force_topic"] = str(self.force_topic)
                f.attrs["columns"] = "x_mm,y_mm,z_mm,wx,wy,wz,fx,fy,fz"

                f.create_dataset("position", data=P, compression="gzip", compression_opts=4, shuffle=True)
                f.create_dataset("force", data=F, compression="gzip", compression_opts=4, shuffle=True)
                f.create_dataset("traj", data=traj, compression="gzip", compression_opts=4, shuffle=True)
                f.create_dataset("sample_time_unix", data=np.asarray(t_list, dtype=np.float64),
                                 compression="gzip", compression_opts=4, shuffle=True)

            self.get_logger().warn(f"[SAVE] {out_path} | traj={traj.shape}")
            if ep_idx == self.ep_idx:
                self.ep_idx += 1

        except Exception as e:
            self.get_logger().error(f"[SAVE-ERR] episode_{ep_idx:04d}: {repr(e)}")
        finally:
            self.finishing = False
            if self.stop_requested and not self.recording:
                self._shutdown()

    def _shutdown(self):
        try:
            self.kb.stop()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = VRStage1HDF5Recorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_requested = True
        if node.recording:
            node.end_episode(reason="KeyboardInterrupt")
        else:
            node._shutdown()


if __name__ == "__main__":
    main()
