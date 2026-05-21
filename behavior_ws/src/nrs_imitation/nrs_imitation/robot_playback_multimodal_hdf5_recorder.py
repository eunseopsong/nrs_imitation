#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robot_playback_multimodal_hdf5_recorder.py

Stage-2 robot playback observation recorder.

While the robot plays back a pushed Stage-1 episode, this node records:
  - robot current pose  : /ur10skku/currentP
  - robot current force : /ur10skku/currentF
  - cam0/local RGB     : /realsense/robot/color/image_raw
  - cam1/global RGB    : /realsense/global/color/image_raw
  - ArUco id0 pose     : /aruco/id_0/pose
  - ArUco id1 pose     : /aruco/id_1/pose

Strict sampling:
  If any one of pose/force/cam0/cam1/marker0/marker1 is missing or stale,
  the sample is NOT saved.

Keyboard:
  s : start recording current episode
  e : end episode and save
  d : discard current recording or delete last saved episode
  u : undo last delete
  q : quit

Command topic:
  /robot_playback_recorder/command std_msgs/String
    start_recording
    end_recording
    discard_current_episode
    terminate_node

Output merged HDF5:
  ~/nrs_imitation/datasets/ACT/YYYYMMDD_HHMM/merged_hdf5/
    robot_playback_merged_YYYYMMDD_HHMM.hdf5

Layout compatible with demo_data_imitation_form.py:
  episodes/ep_0000/{position, ft, images/cam0, images/cam1,
                    marker/id0, marker/id1, marker/combined}
"""

from __future__ import annotations

import os
import sys
import time
import json
import threading
import shutil
from datetime import datetime
from typing import Optional, List, Tuple, Dict

import numpy as np
import h5py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image


REPO_ROOT = os.path.expanduser("~/nrs_imitation")
DEFAULT_ACT_ROOT_DIR = os.path.join(REPO_ROOT, "datasets", "ACT")


def make_qos(depth: int = 10, best_effort: bool = False) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=int(depth),
        reliability=ReliabilityPolicy.BEST_EFFORT if best_effort else ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def image_to_rgb_numpy(msg: Image) -> Optional[np.ndarray]:
    enc = (msg.encoding or "").lower()
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    if h <= 0 or w <= 0 or step <= 0:
        return None

    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if buf.size < h * step:
        return None
    row = buf[: h * step].reshape(h, step)

    if enc in ("rgb8", "bgr8"):
        need = w * 3
        if step < need:
            return None
        img = row[:, :need].reshape(h, w, 3)
        if enc == "bgr8":
            img = img[:, :, ::-1]
        return img.copy()

    if enc in ("rgba8", "bgra8"):
        need = w * 4
        if step < need:
            return None
        img = row[:, :need].reshape(h, w, 4)[:, :, :3]
        if enc == "bgra8":
            img = img[:, :, ::-1]
        return img.copy()

    if enc == "mono8":
        need = w
        if step < need:
            return None
        gray = row[:, :need].reshape(h, w)
        return np.repeat(gray[:, :, None], 3, axis=2).copy()

    return None


def pick_image_shape(frames: List[Optional[np.ndarray]]) -> Tuple[int, int]:
    for im in frames:
        if im is not None and im.ndim == 3 and im.shape[-1] == 3:
            return int(im.shape[0]), int(im.shape[1])
    return 0, 0


def stack_images(frames: List[Optional[np.ndarray]], logger=None, tag="image") -> np.ndarray:
    H, W = pick_image_shape(frames)
    if H <= 0 or W <= 0:
        raise RuntimeError(f"No valid {tag} frames.")
    out = np.zeros((len(frames), H, W, 3), dtype=np.uint8)
    last = np.zeros((H, W, 3), dtype=np.uint8)
    valid = 0
    for i, im in enumerate(frames):
        if im is not None and im.shape == (H, W, 3):
            out[i] = im
            last = im
            valid += 1
        else:
            out[i] = last
    if logger:
        logger.info(f"[STACK] {tag}: shape={out.shape}, valid={valid}/{len(frames)}")
    return out


def quat_xyzw_to_rotvec(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.zeros(3, dtype=np.float32)
    q /= n
    qx, qy, qz, qw = [float(v) for v in q]
    qw = max(-1.0, min(1.0, qw))
    angle = 2.0 * np.arccos(qw)
    s = np.sqrt(max(0.0, 1.0 - qw * qw))
    if s < 1e-8:
        axis = np.asarray([qx, qy, qz], dtype=np.float64)
        an = np.linalg.norm(axis)
        if an < 1e-12:
            return np.zeros(3, dtype=np.float32)
        axis /= an
    else:
        axis = np.asarray([qx / s, qy / s, qz / s], dtype=np.float64)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return (axis * angle).astype(np.float32)


def pose_stamped_to_marker(msg: PoseStamped) -> Tuple[np.ndarray, np.ndarray]:
    p = msg.pose.position
    q = msg.pose.orientation
    rv = quat_xyzw_to_rotvec(q.x, q.y, q.z, q.w)
    marker7 = np.asarray([p.x, p.y, p.z, rv[0], rv[1], rv[2], 1.0], dtype=np.float32)
    marker8 = np.asarray([p.x, p.y, p.z, q.x, q.y, q.z, q.w, 1.0], dtype=np.float32)
    return marker7, marker8


class RobotPlaybackMultimodalHDF5Recorder(Node):
    def __init__(self):
        super().__init__("robot_playback_multimodal_hdf5_recorder")

        # Save.
        self.declare_parameter("act_root_dir", DEFAULT_ACT_ROOT_DIR)
        self.declare_parameter("merged_subdir", "merged_hdf5")
        self.declare_parameter("file_prefix", "robot_playback_merged")
        self.declare_parameter("run_timestamp", "")  # empty -> now
        self.declare_parameter("overwrite_file", False)

        # Topics.
        self.declare_parameter("position_topic", "/ur10skku/currentP")
        self.declare_parameter("force_topic", "/ur10skku/currentF")
        self.declare_parameter("cam0_topic", "/realsense/robot/color/image_raw")
        self.declare_parameter("cam1_topic", "/realsense/global/color/image_raw")
        self.declare_parameter("aruco_id0_topic", "/aruco/id_0/pose")
        self.declare_parameter("aruco_id1_topic", "/aruco/id_1/pose")
        self.declare_parameter("command_topic", "/robot_playback_recorder/command")

        # Sampling.
        self.declare_parameter("sample_hz", 20.0)
        self.declare_parameter("fresh_sec", 0.50)
        self.declare_parameter("marker_fresh_sec", 0.80)
        self.declare_parameter("status_period_sec", 1.0)
        self.declare_parameter("min_samples", 10)

        # Keyboard.
        self.declare_parameter("enable_keyboard", True)

        self.act_root_dir = os.path.expanduser(str(self.get_parameter("act_root_dir").value))
        self.merged_subdir = str(self.get_parameter("merged_subdir").value)
        self.file_prefix = str(self.get_parameter("file_prefix").value)
        ts = str(self.get_parameter("run_timestamp").value).strip()
        self.run_timestamp = ts if ts else datetime.now().strftime("%Y%m%d_%H%M")
        self.overwrite_file = bool(self.get_parameter("overwrite_file").value)

        self.position_topic = str(self.get_parameter("position_topic").value)
        self.force_topic = str(self.get_parameter("force_topic").value)
        self.cam0_topic = str(self.get_parameter("cam0_topic").value)
        self.cam1_topic = str(self.get_parameter("cam1_topic").value)
        self.aruco_id0_topic = str(self.get_parameter("aruco_id0_topic").value)
        self.aruco_id1_topic = str(self.get_parameter("aruco_id1_topic").value)
        self.command_topic = str(self.get_parameter("command_topic").value)

        self.sample_hz = float(self.get_parameter("sample_hz").value)
        self.dt = 1.0 / max(1e-9, self.sample_hz)
        self.fresh_sec = float(self.get_parameter("fresh_sec").value)
        self.marker_fresh_sec = float(self.get_parameter("marker_fresh_sec").value)
        self.status_period_sec = float(self.get_parameter("status_period_sec").value)
        self.min_samples = int(self.get_parameter("min_samples").value)
        self.enable_keyboard = bool(self.get_parameter("enable_keyboard").value)

        self.save_root = os.path.join(self.act_root_dir, self.run_timestamp, self.merged_subdir)
        os.makedirs(self.save_root, exist_ok=True)
        self.h5_path = os.path.join(self.save_root, f"{self.file_prefix}_{self.run_timestamp}.hdf5")
        if os.path.exists(self.h5_path) and self.overwrite_file:
            os.remove(self.h5_path)

        self.h5 = h5py.File(self.h5_path, "a")
        self.grp_eps = self.h5.require_group("episodes")
        self.ep_idx = self._next_ep_index()

        self.lock = threading.Lock()
        self.recording = False
        self.last_saved_ep: Optional[str] = None
        self.undo_buffer: Optional[Dict[str, np.ndarray]] = None
        self.cmd_queue: List[str] = []

        self.latest_pos = None
        self.latest_force = None
        self.latest_cam0 = None
        self.latest_cam1 = None
        self.latest_m0 = None
        self.latest_m1 = None
        self.latest_m0q = None
        self.latest_m1q = None

        self.t_pos = self.t_force = self.t_cam0 = self.t_cam1 = self.t_m0 = self.t_m1 = 0.0

        self.buf_pos: List[np.ndarray] = []
        self.buf_force: List[np.ndarray] = []
        self.buf_cam0: List[Optional[np.ndarray]] = []
        self.buf_cam1: List[Optional[np.ndarray]] = []
        self.buf_m0: List[np.ndarray] = []
        self.buf_m1: List[np.ndarray] = []
        self.buf_m0q: List[np.ndarray] = []
        self.buf_m1q: List[np.ndarray] = []
        self.buf_time: List[float] = []

        qos_rel = make_qos(20, best_effort=False)
        qos_img = make_qos(1, best_effort=True)

        self.create_subscription(Float64MultiArray, self.position_topic, self._pos_cb, qos_rel)
        self.create_subscription(Float64MultiArray, self.force_topic, self._force_cb, qos_rel)
        self.create_subscription(Image, self.cam0_topic, self._cam0_cb, qos_img)
        self.create_subscription(Image, self.cam1_topic, self._cam1_cb, qos_img)
        self.create_subscription(PoseStamped, self.aruco_id0_topic, self._m0_cb, qos_rel)
        self.create_subscription(PoseStamped, self.aruco_id1_topic, self._m1_cb, qos_rel)
        self.create_subscription(String, self.command_topic, self._cmd_cb, qos_rel)

        self.create_timer(self.dt, self._tick)
        self.create_timer(2.0, self._status)

        if self.enable_keyboard:
            threading.Thread(target=self._keyboard_loop, daemon=True).start()

        self.get_logger().info(
            "[READY] RobotPlaybackMultimodalHDF5Recorder\n"
            f"  h5_path={self.h5_path}\n"
            f"  position={self.position_topic}\n"
            f"  force={self.force_topic}\n"
            f"  cam0={self.cam0_topic}\n"
            f"  cam1={self.cam1_topic}\n"
            f"  aruco0={self.aruco_id0_topic}\n"
            f"  aruco1={self.aruco_id1_topic}\n"
            f"  strict_sampling=1 sample_hz={self.sample_hz}\n"
            f"  keyboard: s=start e=end d=discard u=undo q=quit"
        )

    def _next_ep_index(self) -> int:
        ids = []
        for k in self.grp_eps.keys():
            if k.startswith("ep_"):
                try:
                    ids.append(int(k.split("_")[1]))
                except Exception:
                    pass
        return max(ids) + 1 if ids else 0

    def _pos_cb(self, msg):
        if len(msg.data) < 6:
            return
        with self.lock:
            self.latest_pos = np.asarray(msg.data[:6], dtype=np.float32)
            self.t_pos = time.time()

    def _force_cb(self, msg):
        if len(msg.data) < 3:
            return
        with self.lock:
            self.latest_force = np.asarray(msg.data[:3], dtype=np.float32)
            self.t_force = time.time()

    def _cam0_cb(self, msg):
        im = image_to_rgb_numpy(msg)
        if im is not None:
            with self.lock:
                self.latest_cam0 = im
                self.t_cam0 = time.time()

    def _cam1_cb(self, msg):
        im = image_to_rgb_numpy(msg)
        if im is not None:
            with self.lock:
                self.latest_cam1 = im
                self.t_cam1 = time.time()

    def _m0_cb(self, msg):
        m, mq = pose_stamped_to_marker(msg)
        with self.lock:
            self.latest_m0 = m
            self.latest_m0q = mq
            self.t_m0 = time.time()

    def _m1_cb(self, msg):
        m, mq = pose_stamped_to_marker(msg)
        with self.lock:
            self.latest_m1 = m
            self.latest_m1q = mq
            self.t_m1 = time.time()

    def _cmd_cb(self, msg: String):
        cmd = str(msg.data).strip().lower()
        if cmd:
            with self.lock:
                self.cmd_queue.append(cmd)

    def _keyboard_loop(self):
        while rclpy.ok():
            c = sys.stdin.readline().strip().lower()
            if c:
                with self.lock:
                    self.cmd_queue.append(c[0])

    def _status(self):
        with self.lock:
            flags = {
                "pos": self.latest_pos is not None,
                "force": self.latest_force is not None,
                "cam0": self.latest_cam0 is not None,
                "cam1": self.latest_cam1 is not None,
                "m0": self.latest_m0 is not None,
                "m1": self.latest_m1 is not None,
            }
            samples = len(self.buf_pos)

        self.get_logger().info(
            f"[STATUS] recording={int(self.recording)} ep={self.ep_idx:04d} samples={samples} "
            f"streams={flags}"
        )

    def _process_cmds(self):
        with self.lock:
            cmds = self.cmd_queue[:]
            self.cmd_queue.clear()

        for c in cmds:
            if c in ("s", "start_recording"):
                self._start()
            elif c in ("e", "end_recording"):
                self._end(save=True)
            elif c in ("d", "discard_current_episode"):
                self._discard()
            elif c in ("u", "undo"):
                self._undo()
            elif c in ("q", "terminate_node"):
                if self.recording:
                    self._end(save=True)
                self._close()
                rclpy.shutdown()
                return
            else:
                self.get_logger().warn(f"Unknown command: {c}")

    def _tick(self):
        self._process_cmds()
        if not self.recording:
            return

        now = time.time()
        with self.lock:
            pos = None if self.latest_pos is None else self.latest_pos.copy()
            force = None if self.latest_force is None else self.latest_force.copy()
            cam0 = None if self.latest_cam0 is None else self.latest_cam0.copy()
            cam1 = None if self.latest_cam1 is None else self.latest_cam1.copy()
            m0 = None if self.latest_m0 is None else self.latest_m0.copy()
            m1 = None if self.latest_m1 is None else self.latest_m1.copy()
            m0q = None if self.latest_m0q is None else self.latest_m0q.copy()
            m1q = None if self.latest_m1q is None else self.latest_m1q.copy()

            ages = {
                "pos": now - self.t_pos if self.t_pos > 0 else 1e9,
                "force": now - self.t_force if self.t_force > 0 else 1e9,
                "cam0": now - self.t_cam0 if self.t_cam0 > 0 else 1e9,
                "cam1": now - self.t_cam1 if self.t_cam1 > 0 else 1e9,
                "aruco0": now - self.t_m0 if self.t_m0 > 0 else 1e9,
                "aruco1": now - self.t_m1 if self.t_m1 > 0 else 1e9,
            }

        missing = []
        if pos is None or ages["pos"] > self.fresh_sec:
            missing.append(f"pos(age={ages['pos']:.2f})")
        if force is None or ages["force"] > self.fresh_sec:
            missing.append(f"force(age={ages['force']:.2f})")
        if cam0 is None or ages["cam0"] > self.fresh_sec:
            missing.append(f"cam0(age={ages['cam0']:.2f})")
        if cam1 is None or ages["cam1"] > self.fresh_sec:
            missing.append(f"cam1(age={ages['cam1']:.2f})")
        if m0 is None or ages["aruco0"] > self.marker_fresh_sec:
            missing.append(f"aruco0(age={ages['aruco0']:.2f})")
        if m1 is None or ages["aruco1"] > self.marker_fresh_sec:
            missing.append(f"aruco1(age={ages['aruco1']:.2f})")

        if missing:
            if not hasattr(self, "_last_wait_log") or now - self._last_wait_log > 1.0:
                self._last_wait_log = now
                self.get_logger().warn("[WAIT] " + ", ".join(missing))
            return

        self.buf_pos.append(pos.astype(np.float32))
        self.buf_force.append(force[:3].astype(np.float32))
        self.buf_cam0.append(cam0)
        self.buf_cam1.append(cam1)
        self.buf_m0.append(m0)
        self.buf_m1.append(m1)
        self.buf_m0q.append(m0q)
        self.buf_m1q.append(m1q)
        self.buf_time.append(float(now))

    def _start(self):
        if self.recording:
            return
        self.recording = True
        self.buf_pos.clear()
        self.buf_force.clear()
        self.buf_cam0.clear()
        self.buf_cam1.clear()
        self.buf_m0.clear()
        self.buf_m1.clear()
        self.buf_m0q.clear()
        self.buf_m1q.clear()
        self.buf_time.clear()
        self.get_logger().warn(f"[START] ep_{self.ep_idx:04d}")

    def _end(self, save=True):
        if not self.recording:
            return
        self.recording = False
        if save:
            self._save()

    def _save(self):
        N = len(self.buf_pos)
        if N < self.min_samples:
            self.get_logger().warn(f"[DROP] too short N={N} < {self.min_samples}")
            return

        ep = f"ep_{self.ep_idx:04d}"
        if ep in self.grp_eps:
            del self.grp_eps[ep]

        pos = np.asarray(self.buf_pos, dtype=np.float32).reshape(N, 6)
        ft = np.asarray(self.buf_force, dtype=np.float32).reshape(N, 3)
        cam0 = stack_images(self.buf_cam0, self.get_logger(), "cam0")
        cam1 = stack_images(self.buf_cam1, self.get_logger(), "cam1")
        m0 = np.asarray(self.buf_m0, dtype=np.float32).reshape(N, 7)
        m1 = np.asarray(self.buf_m1, dtype=np.float32).reshape(N, 7)
        m0q = np.asarray(self.buf_m0q, dtype=np.float32).reshape(N, 8)
        m1q = np.asarray(self.buf_m1q, dtype=np.float32).reshape(N, 8)
        marker = np.concatenate([m0, m1], axis=1).astype(np.float32)

        g = self.grp_eps.create_group(ep)
        g.attrs["saved_unix"] = float(time.time())
        g.attrs["record_hz"] = float(self.sample_hz)
        g.attrs["schema_version"] = "robot_playback_multimodal_v1"
        g.attrs["position_topic"] = str(self.position_topic)
        g.attrs["force_topic"] = str(self.force_topic)
        g.attrs["cam0_topic"] = str(self.cam0_topic)
        g.attrs["cam1_topic"] = str(self.cam1_topic)
        g.attrs["aruco_id0_topic"] = str(self.aruco_id0_topic)
        g.attrs["aruco_id1_topic"] = str(self.aruco_id1_topic)
        g.attrs["marker_format"] = "[id0: x,y,z,rx,ry,rz,valid] + [id1: x,y,z,rx,ry,rz,valid]"

        g.create_dataset("position", data=pos, compression="gzip", compression_opts=4, shuffle=True)
        g.create_dataset("ft", data=ft, compression="gzip", compression_opts=4, shuffle=True)
        g.create_dataset("sample_time_unix", data=np.asarray(self.buf_time, dtype=np.float64),
                         compression="gzip", compression_opts=4, shuffle=True)

        ig = g.create_group("images")
        ig.create_dataset("cam0", data=cam0, compression="gzip", compression_opts=4, shuffle=True)
        ig.create_dataset("cam1", data=cam1, compression="gzip", compression_opts=4, shuffle=True)

        mg = g.create_group("marker")
        mg.create_dataset("id0", data=m0, compression="gzip", compression_opts=4, shuffle=True)
        mg.create_dataset("id1", data=m1, compression="gzip", compression_opts=4, shuffle=True)
        mg.create_dataset("combined", data=marker, compression="gzip", compression_opts=4, shuffle=True)
        mg.create_dataset("id0_quat", data=m0q, compression="gzip", compression_opts=4, shuffle=True)
        mg.create_dataset("id1_quat", data=m1q, compression="gzip", compression_opts=4, shuffle=True)

        self.h5.flush()

        self.last_saved_ep = ep
        self.undo_buffer = None
        self.ep_idx += 1

        self.get_logger().warn(
            f"[SAVE] {ep} N={N} pos={pos.shape} ft={ft.shape} "
            f"cam0={cam0.shape} cam1={cam1.shape} marker={marker.shape}"
        )

    def _discard(self):
        if self.recording:
            self.recording = False
            self.buf_pos.clear()
            self.buf_force.clear()
            self.buf_cam0.clear()
            self.buf_cam1.clear()
            self.buf_m0.clear()
            self.buf_m1.clear()
            self.buf_m0q.clear()
            self.buf_m1q.clear()
            self.buf_time.clear()
            self.get_logger().warn("[DISCARD] current recording not saved")
            return

        if self.last_saved_ep is None or self.last_saved_ep not in self.grp_eps:
            self.get_logger().warn("[DISCARD] no saved episode to delete")
            return

        ep = self.last_saved_ep
        tmp_path = os.path.join(REPO_ROOT, "tmp", f"{ep}_undo_copy.hdf5")
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        with h5py.File(tmp_path, "w") as tf:
            self.h5.copy(self.grp_eps[ep], tf, name=ep)
        del self.grp_eps[ep]
        self.h5.flush()
        self.undo_buffer = {"ep": ep, "tmp_path": tmp_path}
        self.last_saved_ep = None
        self.get_logger().warn(f"[DELETE] {ep} undo available")

    def _undo(self):
        if self.undo_buffer is None:
            self.get_logger().warn("[UNDO] nothing to undo")
            return
        ep = self.undo_buffer["ep"]
        tmp_path = self.undo_buffer["tmp_path"]
        if ep in self.grp_eps:
            del self.grp_eps[ep]
        with h5py.File(tmp_path, "r") as tf:
            tf.copy(tf[ep], self.grp_eps, name=ep)
        self.h5.flush()
        self.last_saved_ep = ep
        self.undo_buffer = None
        self.get_logger().warn(f"[UNDO] restored {ep}")

    def _close(self):
        try:
            self.h5.flush()
            self.h5.close()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = RobotPlaybackMultimodalHDF5Recorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
