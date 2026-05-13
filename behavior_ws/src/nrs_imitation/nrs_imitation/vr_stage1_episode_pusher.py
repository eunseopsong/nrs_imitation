#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vr_stage1_episode_pusher.py

Stage-1 per-episode HDF5 -> TXT -> send to robot playback PC.

Input:
  /home/eunseop/nrs_imitation/datasets/ACT/YYYYMMDD_HHMM/stage1_vr_episodes/
    episode_0000.hdf5
    episode_0001.hdf5
    ...

Each episode file must contain:
  traj (T,9) = [x_mm y_mm z_mm wx wy wz fx fy fz]

Keyboard:
  [Enter] : push current episode
  d       : idx + 1
  a       : idx - 1
  r       : re-push current episode
  g <idx> : go to idx
  l       : list
  s       : status
  h       : help
  q       : quit
"""

from __future__ import annotations

import os
import sys
import time
import shlex
import glob
import subprocess
import threading
from pathlib import Path
from typing import List

import numpy as np
import h5py

import rclpy
from rclpy.node import Node


def find_latest_stage1_dir(act_root_dir: str, output_subdir: str) -> str:
    pattern = os.path.join(act_root_dir, "*", output_subdir)
    dirs = sorted([d for d in glob.glob(pattern) if os.path.isdir(d)])
    if not dirs:
        raise FileNotFoundError(f"No stage1 episode dir found: {pattern}")
    return dirs[-1]


class VRStage1EpisodePusher(Node):
    def __init__(self):
        super().__init__("vr_stage1_episode_pusher")

        self.declare_parameter("act_root_dir", "/home/eunseop/nrs_imitation/datasets/ACT")
        self.declare_parameter("episode_dir", "")  # empty -> latest <act_root>/*/stage1_vr_episodes
        self.declare_parameter("output_subdir", "stage1_vr_episodes")
        self.declare_parameter("traj_dataset", "traj")

        self.declare_parameter("local_txt_path", "/tmp/cmd_continue9D.txt")
        self.declare_parameter("remote_user", "nrs_forcecon")
        self.declare_parameter("remote_ip", "192.168.0.151")
        self.declare_parameter(
            "remote_txt_path",
            "/home/nrs_forcecon/dev_ws/src/y2_ur10skku_control/Y2RobMotion/txtcmd/cmd_continue9D.txt",
        )
        self.declare_parameter("txt_fmt", "%.10f")
        self.declare_parameter("use_atomic_remote_replace", True)

        self.act_root_dir = str(self.get_parameter("act_root_dir").value)
        self.output_subdir = str(self.get_parameter("output_subdir").value)
        episode_dir = str(self.get_parameter("episode_dir").value).strip()
        self.episode_dir = episode_dir if episode_dir else find_latest_stage1_dir(self.act_root_dir, self.output_subdir)

        self.traj_dataset = str(self.get_parameter("traj_dataset").value)
        self.local_txt_path = str(self.get_parameter("local_txt_path").value)
        self.remote_user = str(self.get_parameter("remote_user").value)
        self.remote_ip = str(self.get_parameter("remote_ip").value)
        self.remote_txt_path = str(self.get_parameter("remote_txt_path").value)
        self.txt_fmt = str(self.get_parameter("txt_fmt").value)
        self.use_atomic_remote_replace = bool(self.get_parameter("use_atomic_remote_replace").value)

        self.episodes = sorted(glob.glob(os.path.join(self.episode_dir, "episode_*.hdf5")))
        if not self.episodes:
            raise RuntimeError(f"No episode_*.hdf5 found in {self.episode_dir}")

        self.cur_idx = 0
        self.stop_evt = threading.Event()
        self.kbd_thread = threading.Thread(target=self.keyboard_loop, daemon=True)

        self.get_logger().info("============================================================")
        self.get_logger().info("VRStage1EpisodePusher initialized")
        self.get_logger().info(f"  episode_dir={self.episode_dir}")
        self.get_logger().info(f"  episodes={len(self.episodes)}")
        self.get_logger().info(f"  local_txt={self.local_txt_path}")
        self.get_logger().info(f"  remote={self.remote_user}@{self.remote_ip}:{self.remote_txt_path}")
        self.get_logger().info("============================================================")
        self.print_help()
        self.get_logger().info(self.status_line())

        self.kbd_thread.start()

    def ep_path(self, idx: int) -> str:
        idx = int(np.clip(idx, 0, len(self.episodes) - 1))
        return self.episodes[idx]

    def ep_name(self, idx: int) -> str:
        return os.path.basename(self.ep_path(idx))

    def status_line(self) -> str:
        return f"[STATUS] total={len(self.episodes)} | cur_idx={self.cur_idx} | cur_ep={self.ep_name(self.cur_idx)}"

    def print_help(self):
        self.get_logger().info("")
        self.get_logger().info("================= Keyboard Commands =================")
        self.get_logger().info("  [Enter] : push CURRENT idx episode")
        self.get_logger().info("  d       : idx + 1")
        self.get_logger().info("  a       : idx - 1")
        self.get_logger().info("  r       : re-push current")
        self.get_logger().info("  g <idx> : go to idx")
        self.get_logger().info("  l       : list episodes")
        self.get_logger().info("  s       : status")
        self.get_logger().info("  h       : help")
        self.get_logger().info("  q       : quit")
        self.get_logger().info("=====================================================")

    def list_episodes(self, max_show: int = 80):
        self.get_logger().info(f"[LIST] total={len(self.episodes)}")
        show = self.episodes if len(self.episodes) <= max_show else self.episodes[:max_show]
        for i, p in enumerate(show):
            mark = "<--" if i == self.cur_idx else ""
            self.get_logger().info(f"  {i:4d}: {os.path.basename(p)} {mark}")
        if len(self.episodes) > max_show:
            self.get_logger().info("  ...")

    def load_traj(self, path: str) -> np.ndarray:
        with h5py.File(path, "r") as f:
            if self.traj_dataset not in f:
                raise KeyError(f"Dataset '{self.traj_dataset}' not found in {path}")
            traj = np.asarray(f[self.traj_dataset], dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] != 9:
            raise ValueError(f"traj must be (T,9), got {traj.shape} from {path}")
        return traj

    def push_current(self):
        idx = int(self.cur_idx)
        path = self.ep_path(idx)
        t0 = time.time()

        self.get_logger().info(f"[PUSH] idx={idx} loading {path}")
        traj = self.load_traj(path)

        os.makedirs(os.path.dirname(self.local_txt_path) or ".", exist_ok=True)
        np.savetxt(self.local_txt_path, traj, fmt=self.txt_fmt)
        self.get_logger().info(f"[PUSH] saved local TXT: {self.local_txt_path} shape={traj.shape}")

        if self.remote_ip.strip():
            self.get_logger().info("[PUSH] sending to robot PC...")
            if self.use_atomic_remote_replace:
                remote_tmp = self.remote_txt_path + ".tmp"
                self._scp(self.local_txt_path, remote_tmp)
                self._ssh(f"mv -f {shlex.quote(remote_tmp)} {shlex.quote(self.remote_txt_path)}")
            else:
                self._scp(self.local_txt_path, self.remote_txt_path)

        self.get_logger().warn(f"[DONE] pushed idx={idx} {self.ep_name(idx)} in {time.time() - t0:.2f}s")
        self.get_logger().info(self.status_line())

    def _scp(self, local_path: str, remote_path: str):
        dst = f"{self.remote_user}@{self.remote_ip}:{remote_path}"
        result = subprocess.run(["scp", local_path, dst], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("SCP failed:\n" + (result.stderr or "").strip())

    def _ssh(self, remote_cmd: str):
        host = f"{self.remote_user}@{self.remote_ip}"
        result = subprocess.run(["ssh", host, remote_cmd], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("SSH failed:\n" + (result.stderr or "").strip())

    def keyboard_loop(self):
        while not self.stop_evt.is_set():
            line = sys.stdin.readline()
            if line == "":
                time.sleep(0.05)
                continue

            cmd = line.strip()

            if cmd == "":
                try:
                    self.push_current()
                except Exception as e:
                    self.get_logger().error(f"[ERR] push failed: {e}")
                continue

            if cmd == "q":
                self.stop_evt.set()
                rclpy.shutdown()
                return
            if cmd == "h":
                self.print_help()
            elif cmd == "s":
                pass
            elif cmd == "l":
                self.list_episodes()
            elif cmd == "d":
                self.cur_idx = min(len(self.episodes) - 1, self.cur_idx + 1)
            elif cmd == "a":
                self.cur_idx = max(0, self.cur_idx - 1)
            elif cmd == "r":
                try:
                    self.push_current()
                except Exception as e:
                    self.get_logger().error(f"[ERR] re-push failed: {e}")
            elif cmd.startswith("g "):
                try:
                    self.cur_idx = int(np.clip(int(cmd.split()[1]), 0, len(self.episodes) - 1))
                except Exception:
                    self.get_logger().warn("Usage: g <idx>")
            else:
                self.get_logger().warn(f"Unknown command: {cmd}")

            self.get_logger().info(self.status_line())


def main(args=None):
    rclpy.init(args=args)
    node = VRStage1EpisodePusher()
    try:
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
