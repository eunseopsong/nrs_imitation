#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vr_stage1_episode_pusher.py

Stage-1 per-episode HDF5 -> optional extra contact slow-down -> TXT -> send to robot playback PC.

Purpose
-------
Convex-surface polishing playback can become unstable around the initial transition:

    position control approach -> first surface contact -> force-control region

This pusher therefore applies an extra time-warp ONLY at push time.
The saved Stage-1 HDF5 episode is not modified.

Contact detection rule requested by user:
    first contact index = first index where fz > 0

Input episode file:
  ~/nrs_imitation/datasets/ACT/YYYYMMDD_HHMM/stage1_vr_episodes/
    episode_0000.hdf5
    episode_0001.hdf5
    ...

Required dataset:
  traj (T,9) = [x_mm y_mm z_mm wx wy wz fx fy fz]

Output:
  ~/nrs_imitation/tmp/cmd_continue9D.txt
  optionally scp to robot playback PC

Keyboard:
  [Enter] : push current episode
  d       : idx + 1
  a       : idx - 1
  r       : re-push current episode
  g <idx> : go to idx
  l       : list episodes
  s       : status
  h       : help
  q       : quit

Key new parameters
------------------
  extra_contact_slowdown_enable : default True
  contact_fz_threshold          : default 0.0
  contact_pre_sec               : default 6.0
  contact_post_sec              : default 2.0
  contact_scale_max             : default 15.0
  contact_consec                : default 1

How it works
------------
Given original samples at dt, a per-segment time scale is built:

  scale[k] = 1 outside window
  scale[k] > 1 around first contact

Then the trajectory is re-sampled at uniform dt.
More samples are inserted around contact, so the robot passes the initial
surface-contact region more slowly while following the same geometric path.

The slowdown window is:
  [contact_idx - contact_pre_sec, contact_idx + contact_post_sec]

The default profile is conservative:
  - long pre-contact slow approach
  - slowest at first contact
  - smooth cosine taper to avoid discontinuity
"""

from __future__ import annotations

import os
import sys
import time
import shlex
import glob
import subprocess
import threading
from typing import List, Optional, Tuple, Dict

import numpy as np
import h5py

import rclpy
from rclpy.node import Node

from nrs_imitation.pretty_print import block, status


REPO_ROOT = os.path.expanduser("~/nrs_imitation")
DEFAULT_ACT_ROOT_DIR = os.path.join(REPO_ROOT, "datasets", "ACT")
DEFAULT_LOCAL_TXT_PATH = os.path.join(REPO_ROOT, "tmp", "cmd_continue9D.txt")
DEFAULT_DEBUG_TXT_DIR = os.path.join(REPO_ROOT, "tmp", "stage1_pusher_debug")


# ============================================================
# Helpers
# ============================================================

def find_latest_stage1_dir(act_root_dir: str, output_subdir: str) -> str:
    pattern = os.path.join(act_root_dir, "*", output_subdir)
    dirs = sorted([d for d in glob.glob(pattern) if os.path.isdir(d)])
    if not dirs:
        raise FileNotFoundError(f"No stage1 episode dir found: {pattern}")
    return dirs[-1]


def detect_first_contact_idx_fz_positive(
    fz: np.ndarray,
    threshold: float = 0.0,
    consec: int = 1,
) -> Optional[int]:
    """
    Contact definition requested:
        fz > threshold

    default threshold = 0.0
    default consec = 1
    """
    fz = np.asarray(fz, dtype=np.float64).reshape(-1)
    consec = max(1, int(consec))

    count = 0
    for i, v in enumerate(fz):
        if float(v) > float(threshold):
            count += 1
            if count >= consec:
                return i - consec + 1
        else:
            count = 0
    return None


def _smoothstep01(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def build_initial_contact_slowdown_scale(
    n: int,
    contact_idx: int,
    dt: float,
    pre_sec: float,
    post_sec: float,
    scale_max: float,
    profile: str = "asymmetric_cosine",
) -> Tuple[np.ndarray, int, int]:
    """
    Build segment scale, length n-1.

    scale > 1 means the segment takes longer. After uniform resampling,
    more points are inserted into that region.

    For initial contact, we want the robot to approach very slowly before
    the first positive Fz. Therefore the scale is high near contact and
    smoothly falls to 1 away from contact.

    Returns:
      scale: (n-1,)
      s0, s1: sample-index window on original trajectory
    """
    n = int(n)
    if n < 2:
        return np.ones((0,), dtype=np.float64), 0, 0

    dt = float(dt)
    pre_n = max(0, int(round(float(pre_sec) / max(dt, 1e-9))))
    post_n = max(0, int(round(float(post_sec) / max(dt, 1e-9))))

    c = int(np.clip(contact_idx, 0, n - 1))
    s0 = max(0, c - pre_n)
    s1 = min(n - 1, c + post_n)

    scale = np.ones((n - 1,), dtype=np.float64)
    scale_max = max(1.0, float(scale_max))

    if s1 <= s0:
        return scale, s0, s1

    # Segment index k covers sample k -> k+1.
    ks = np.arange(s0, s1, dtype=np.int64)

    if profile == "flat_peak":
        # Very conservative: full slowdown in the whole window with small cosine taper.
        center = float(c)
        dist = np.abs(ks.astype(np.float64) - center)
        half = max(1.0, float(max(c - s0, s1 - c)))
        weight = 0.5 + 0.5 * np.cos(np.pi * np.clip(dist / half, 0.0, 1.0))
        weight = np.maximum(weight, 0.75)
    else:
        # asymmetric_cosine:
        #   before contact: gradually increase slowdown toward contact
        #   after contact : gradually decrease slowdown
        weight = np.zeros_like(ks, dtype=np.float64)

        before = ks <= c
        if np.any(before):
            denom_pre = max(1.0, float(c - s0))
            u_pre = (ks[before].astype(np.float64) - float(s0)) / denom_pre
            # 0 at window start, 1 at contact
            weight[before] = 0.5 - 0.5 * np.cos(np.pi * np.clip(u_pre, 0.0, 1.0))

        after = ks > c
        if np.any(after):
            denom_post = max(1.0, float(s1 - c))
            u_post = (ks[after].astype(np.float64) - float(c)) / denom_post
            # 1 at contact, 0 at post-window end
            weight[after] = 0.5 + 0.5 * np.cos(np.pi * np.clip(u_post, 0.0, 1.0))

        weight = np.clip(weight, 0.0, 1.0)

    scale[ks] = 1.0 + (scale_max - 1.0) * weight
    return scale, s0, s1


def resample_uniform_by_segment_scale(
    traj: np.ndarray,
    dt: float,
    seg_scale: np.ndarray,
) -> np.ndarray:
    """
    Time-warp trajectory with segment scale and resample at original dt.

    traj: (N,D)
    seg_scale: (N-1,)
    """
    traj = np.asarray(traj, dtype=np.float64)
    n, d = traj.shape
    if n < 2:
        return traj.copy()

    seg_scale = np.asarray(seg_scale, dtype=np.float64).reshape(-1)
    if seg_scale.shape[0] != n - 1:
        raise ValueError(f"seg_scale must be N-1={n-1}, got {seg_scale.shape}")

    dt = float(dt)
    tprime = np.zeros(n, dtype=np.float64)
    tprime[1:] = np.cumsum(dt * np.maximum(seg_scale, 1e-6))

    total_t = float(tprime[-1])
    if total_t <= 0.0:
        return traj.copy()

    m = int(round(total_t / dt)) + 1
    m = max(2, m)

    t_uniform = np.arange(m, dtype=np.float64) * dt
    t_uniform[-1] = total_t

    out = np.empty((m, d), dtype=np.float64)
    for j in range(d):
        out[:, j] = np.interp(t_uniform, tprime, traj[:, j])

    return out


def apply_initial_contact_slowdown(
    traj: np.ndarray,
    dt: float,
    threshold: float,
    consec: int,
    pre_sec: float,
    post_sec: float,
    scale_max: float,
    profile: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Apply extra slowdown around first fz > threshold.

    Returns:
      processed_traj, debug_meta
    """
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] != 9:
        raise ValueError(f"traj must be (T,9), got {traj.shape}")

    n = int(traj.shape[0])
    fz = traj[:, 8]

    contact_idx = detect_first_contact_idx_fz_positive(
        fz,
        threshold=float(threshold),
        consec=int(consec),
    )

    meta: Dict[str, object] = {
        "enabled": True,
        "contact_rule": f"first fz > {float(threshold)}",
        "contact_idx": None if contact_idx is None else int(contact_idx),
        "input_len": int(n),
        "output_len": int(n),
        "dt": float(dt),
        "pre_sec": float(pre_sec),
        "post_sec": float(post_sec),
        "scale_max": float(scale_max),
        "profile": str(profile),
        "window_start_idx": None,
        "window_end_idx": None,
    }

    if contact_idx is None:
        meta["skipped_reason"] = "no fz > threshold contact found"
        return traj.astype(np.float32), meta

    seg_scale, s0, s1 = build_initial_contact_slowdown_scale(
        n=n,
        contact_idx=contact_idx,
        dt=dt,
        pre_sec=pre_sec,
        post_sec=post_sec,
        scale_max=scale_max,
        profile=profile,
    )

    out = resample_uniform_by_segment_scale(traj, dt=dt, seg_scale=seg_scale)

    meta.update(
        {
            "window_start_idx": int(s0),
            "window_end_idx": int(s1),
            "window_start_time_sec": float(s0 * dt),
            "contact_time_sec": float(contact_idx * dt),
            "window_end_time_sec": float(s1 * dt),
            "scale_min": float(np.min(seg_scale)) if seg_scale.size else 1.0,
            "scale_max_actual": float(np.max(seg_scale)) if seg_scale.size else 1.0,
            "output_len": int(out.shape[0]),
            "length_ratio": float(out.shape[0] / max(1, n)),
        }
    )
    return out.astype(np.float32), meta


# ============================================================
# Main node
# ============================================================

class VRStage1EpisodePusher(Node):
    def __init__(self):
        super().__init__("vr_stage1_episode_pusher")

        # -------------------------
        # Input episode directory
        # -------------------------
        self.declare_parameter("act_root_dir", DEFAULT_ACT_ROOT_DIR)
        self.declare_parameter("episode_dir", "")  # empty -> latest <act_root>/*/stage1_vr_episodes
        self.declare_parameter("output_subdir", "stage1_vr_episodes")
        self.declare_parameter("traj_dataset", "traj")

        # -------------------------
        # Local/remote txt
        # -------------------------
        self.declare_parameter("local_txt_path", DEFAULT_LOCAL_TXT_PATH)
        self.declare_parameter("remote_user", "nrs_forcecon")
        self.declare_parameter("remote_ip", "192.168.0.151")
        self.declare_parameter(
            "remote_txt_path",
            "dev_ws/src/y2_ur10skku_control/Y2RobMotion/txtcmd/cmd_continue9D.txt",
        )
        self.declare_parameter("txt_fmt", "%.10f")
        self.declare_parameter("use_atomic_remote_replace", True)

        # -------------------------
        # Extra initial-contact slow-down
        # -------------------------
        self.declare_parameter("extra_contact_slowdown_enable", True)

        # Requested contact criterion:
        #   first contact = first fz > 0
        self.declare_parameter("contact_fz_threshold", 0.0)
        self.declare_parameter("contact_consec", 1)

        # Slow down window and strength.
        self.declare_parameter("contact_pre_sec", 6.0)
        self.declare_parameter("contact_post_sec", 2.0)
        self.declare_parameter("contact_scale_max", 15.0)
        self.declare_parameter("contact_slowdown_profile", "asymmetric_cosine")

        # Playback dt. If <= 0, infer from HDF5 attr dt or record_hz.
        self.declare_parameter("traj_dt", 0.0)
        self.declare_parameter("default_record_hz", 125.0)

        # Optional processed txt copy for debugging.
        self.declare_parameter("save_debug_txt_copy", True)
        self.declare_parameter("debug_txt_dir", DEFAULT_DEBUG_TXT_DIR)

        # -------------------------
        # Load params
        # -------------------------
        self.act_root_dir = os.path.expanduser(str(self.get_parameter("act_root_dir").value))
        self.output_subdir = str(self.get_parameter("output_subdir").value)
        episode_dir = str(self.get_parameter("episode_dir").value).strip()
        self.episode_dir = episode_dir if episode_dir else find_latest_stage1_dir(self.act_root_dir, self.output_subdir)

        self.traj_dataset = str(self.get_parameter("traj_dataset").value)

        self.local_txt_path = os.path.expanduser(str(self.get_parameter("local_txt_path").value))
        self.remote_user = str(self.get_parameter("remote_user").value)
        self.remote_ip = str(self.get_parameter("remote_ip").value)
        self.remote_txt_path = str(self.get_parameter("remote_txt_path").value)
        self.txt_fmt = str(self.get_parameter("txt_fmt").value)
        self.use_atomic_remote_replace = bool(self.get_parameter("use_atomic_remote_replace").value)

        self.extra_contact_slowdown_enable = bool(self.get_parameter("extra_contact_slowdown_enable").value)
        self.contact_fz_threshold = float(self.get_parameter("contact_fz_threshold").value)
        self.contact_consec = int(self.get_parameter("contact_consec").value)
        self.contact_pre_sec = float(self.get_parameter("contact_pre_sec").value)
        self.contact_post_sec = float(self.get_parameter("contact_post_sec").value)
        self.contact_scale_max = float(self.get_parameter("contact_scale_max").value)
        self.contact_slowdown_profile = str(self.get_parameter("contact_slowdown_profile").value)

        self.traj_dt = float(self.get_parameter("traj_dt").value)
        self.default_record_hz = float(self.get_parameter("default_record_hz").value)

        self.save_debug_txt_copy = bool(self.get_parameter("save_debug_txt_copy").value)
        self.debug_txt_dir = os.path.expanduser(str(self.get_parameter("debug_txt_dir").value))

        # -------------------------
        # Episode list
        # -------------------------
        self.episodes = sorted(glob.glob(os.path.join(self.episode_dir, "episode_*.hdf5")))
        if not self.episodes:
            raise RuntimeError(f"No episode_*.hdf5 found in {self.episode_dir}")

        self.cur_idx = 0
        self.stop_evt = threading.Event()
        self.kbd_thread = threading.Thread(target=self.keyboard_loop, daemon=True)

        self.get_logger().info(block("STAGE1 PUSHER READY", [
            ("episode_dir", self.episode_dir),
            ("episodes", len(self.episodes)),
            ("traj_dataset", self.traj_dataset),
            ("local_txt", self.local_txt_path),
            ("remote", f"{self.remote_user}@{self.remote_ip}:{self.remote_txt_path}"),
            ("slowdown", f"{int(self.extra_contact_slowdown_enable)} first fz>{self.contact_fz_threshold}, consec={self.contact_consec}"),
            ("window", f"pre={self.contact_pre_sec}s, post={self.contact_post_sec}s"),
            ("scale", f"max={self.contact_scale_max}, profile={self.contact_slowdown_profile}"),
        ]))
        self.print_help()
        self.get_logger().info(self.status_line())

        self.kbd_thread.start()

    # ------------------------------------------------------------
    # Episode helpers
    # ------------------------------------------------------------

    def ep_path(self, idx: int) -> str:
        idx = int(np.clip(idx, 0, len(self.episodes) - 1))
        return self.episodes[idx]

    def ep_name(self, idx: int) -> str:
        return os.path.basename(self.ep_path(idx))

    def status_line(self) -> str:
        return status("STATUS", [
            ("total", len(self.episodes)),
            ("cur_idx", self.cur_idx),
            ("cur_ep", self.ep_name(self.cur_idx)),
        ])

    def print_help(self):
        self.get_logger().info(block("KEYBOARD", [
            ("Enter", "push current episode"),
            ("d / a", "next / previous episode"),
            ("r", "re-push current"),
            ("g <idx>", "go to index"),
            ("l / s / h / q", "list / status / help / quit"),
        ], char="-"))

    def list_episodes(self, max_show: int = 80):
        self.get_logger().info(f"[LIST] total={len(self.episodes)}")
        show = self.episodes if len(self.episodes) <= max_show else self.episodes[:max_show]
        for i, p in enumerate(show):
            mark = "<--" if i == self.cur_idx else ""
            self.get_logger().info(f"  {i:4d}: {os.path.basename(p)} {mark}")
        if len(self.episodes) > max_show:
            self.get_logger().info("  ...")

    # ------------------------------------------------------------
    # Load / process / push
    # ------------------------------------------------------------

    def load_traj_and_dt(self, path: str) -> Tuple[np.ndarray, float]:
        with h5py.File(path, "r") as f:
            if self.traj_dataset not in f:
                raise KeyError(f"Dataset '{self.traj_dataset}' not found in {path}")
            traj = np.asarray(f[self.traj_dataset], dtype=np.float64)

            # dt inference priority:
            #   1) parameter traj_dt if > 0
            #   2) HDF5 attr dt
            #   3) HDF5 attr record_hz
            #   4) default_record_hz
            if self.traj_dt > 0.0:
                dt = float(self.traj_dt)
            elif "dt" in f.attrs:
                dt = float(f.attrs["dt"])
            elif "record_hz" in f.attrs:
                dt = 1.0 / max(float(f.attrs["record_hz"]), 1e-9)
            else:
                dt = 1.0 / max(float(self.default_record_hz), 1e-9)

        if traj.ndim != 2 or traj.shape[1] != 9:
            raise ValueError(f"traj must be (T,9), got {traj.shape} from {path}")

        return traj, dt

    def process_traj_for_push(self, traj: np.ndarray, dt: float) -> Tuple[np.ndarray, Dict[str, object]]:
        if not self.extra_contact_slowdown_enable:
            return traj.astype(np.float32), {
                "enabled": False,
                "input_len": int(traj.shape[0]),
                "output_len": int(traj.shape[0]),
                "dt": float(dt),
            }

        return apply_initial_contact_slowdown(
            traj=traj,
            dt=dt,
            threshold=self.contact_fz_threshold,
            consec=self.contact_consec,
            pre_sec=self.contact_pre_sec,
            post_sec=self.contact_post_sec,
            scale_max=self.contact_scale_max,
            profile=self.contact_slowdown_profile,
        )

    def save_debug_copy(self, processed: np.ndarray, idx: int, meta: Dict[str, object]):
        if not self.save_debug_txt_copy:
            return

        os.makedirs(self.debug_txt_dir, exist_ok=True)
        base = f"{idx:04d}_{self.ep_name(idx).replace('.hdf5', '')}"
        txt_path = os.path.join(self.debug_txt_dir, f"{base}_slow_contact.txt")
        meta_path = os.path.join(self.debug_txt_dir, f"{base}_slow_contact_meta.txt")

        np.savetxt(txt_path, processed, fmt=self.txt_fmt)

        with open(meta_path, "w", encoding="utf-8") as fp:
            for k, v in meta.items():
                fp.write(f"{k}: {v}\n")

        self.get_logger().info(f"[DEBUG] saved processed TXT copy: {txt_path}")
        self.get_logger().info(f"[DEBUG] saved meta: {meta_path}")

    def push_current(self):
        idx = int(self.cur_idx)
        path = self.ep_path(idx)
        t0 = time.time()

        self.get_logger().info(f"[PUSH] idx={idx} loading {path}")
        traj, dt = self.load_traj_and_dt(path)

        fz = traj[:, 8]
        raw_contact_idx = detect_first_contact_idx_fz_positive(
            fz,
            threshold=self.contact_fz_threshold,
            consec=self.contact_consec,
        )

        self.get_logger().info(status("LOAD", [
            ("shape", traj.shape),
            ("dt", f"{dt:.6f}s"),
            ("fz_min", f"{float(np.min(fz)):.4f}"),
            ("fz_max", f"{float(np.max(fz)):.4f}"),
            ("contact_idx", raw_contact_idx),
        ]))

        processed, meta = self.process_traj_for_push(traj, dt)

        self.get_logger().warn(status("SLOW-CONTACT", [
            ("enabled", meta.get("enabled")),
            ("contact_idx", meta.get("contact_idx")),
            ("window", f"[{meta.get('window_start_idx')}, {meta.get('window_end_idx')}]"),
            ("len", f"{meta.get('input_len')} -> {meta.get('output_len')}"),
            ("ratio", f"{float(meta.get('length_ratio', 1.0)):.3f}"),
            ("scale", f"{float(meta.get('scale_max_actual', 1.0)):.3f}"),
        ]))

        os.makedirs(os.path.dirname(self.local_txt_path) or ".", exist_ok=True)
        np.savetxt(self.local_txt_path, processed, fmt=self.txt_fmt)
        self.get_logger().info(f"[PUSH] saved local TXT: {self.local_txt_path} shape={processed.shape}")

        self.save_debug_copy(processed, idx, meta)

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

    # ------------------------------------------------------------
    # SSH/SCP
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------

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
