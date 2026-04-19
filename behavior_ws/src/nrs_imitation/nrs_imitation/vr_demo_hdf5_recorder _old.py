#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vr_demo_hdf5_recorder.py

- VR tracker: /calibrated_pose (Float64MultiArray: [x y z wx wy wz])   input: (m, rad)
- FT sensor : /ftsensor/measured_Cvalue (geometry_msgs/Wrench)        input: (N)

Episode rule:
  start: |fx| >= start_abs_fx
  end  : |fy| >= stop_abs_fy

This HDF5 recorder uses the EXACT SAME trajectory-generation pipeline as vr_demo_txt_recorder.py.
The ONLY conceptual difference is output storage:
  - txt_recorder: one episode -> one txt file (+ optional scp)
  - hdf5_recorder: multiple episodes -> one HDF5 file (episodes/ep_xxxx/traj)

All filtering / clamp / retime / approach slow-down / QP-guard / fz contact cleanup is identical to txt_recorder.
"""

import os
import sys
import time
import json
import math
import atexit
import threading
import select
import termios
import tty
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import h5py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Wrench


# ============================================================
# Shared utilities (same as txt_recorder)
# ============================================================
def pctl(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, q))


def norm_rows(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=1)


# ----------------------------
# Hampel filter (per-dim)
# ----------------------------
def hampel_1d(x: np.ndarray, win: int, n_sigmas: float) -> np.ndarray:
    if win <= 0:
        return x.copy()
    n = x.size
    y = x.copy()
    k = 1.4826
    for i in range(n):
        i0 = max(0, i - win)
        i1 = min(n, i + win + 1)
        w = x[i0:i1]
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        sigma = k * mad + 1e-12
        if abs(x[i] - med) > n_sigmas * sigma:
            y[i] = med
    return y


def hampel_nd(X: np.ndarray, win: int, n_sigmas: float) -> np.ndarray:
    Y = X.copy()
    for d in range(X.shape[1]):
        Y[:, d] = hampel_1d(X[:, d], win=win, n_sigmas=n_sigmas)
    return Y


# ----------------------------
# Whittaker smoother via CG (D2 penalty)
# ----------------------------
def _apply_D2(x: np.ndarray) -> np.ndarray:
    return x[:-2] - 2.0 * x[1:-1] + x[2:]


def _apply_D2t(u: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    out[:-2] += u
    out[1:-1] += -2.0 * u
    out[2:] += u
    return out


def whittaker_cg_1d(y: np.ndarray, lam: float, cg_iters: int = 200, tol: float = 1e-8) -> np.ndarray:
    n = y.size
    if n < 5 or lam <= 0.0:
        return y.copy()

    def A(x: np.ndarray) -> np.ndarray:
        d2 = _apply_D2(x)
        return x + lam * _apply_D2t(d2, n)

    x = y.copy()
    r = y - A(x)
    p = r.copy()
    rr = float(r @ r)
    if rr < tol:
        return x

    yy = float(y @ y) + 1e-12
    for _ in range(cg_iters):
        Ap = A(p)
        denom = float(p @ Ap) + 1e-12
        alpha = rr / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = float(r @ r)
        if rr_new < (tol * tol) * yy:
            break
        beta = rr_new / (rr + 1e-12)
        p = r + beta * p
        rr = rr_new
    return x


def whittaker_cg_nd(Y: np.ndarray, lam: float, cg_iters: int = 200, tol: float = 1e-8) -> np.ndarray:
    Z = np.empty_like(Y)
    for d in range(Y.shape[1]):
        Z[:, d] = whittaker_cg_1d(Y[:, d], lam=lam, cg_iters=cg_iters, tol=tol)
    return Z


def ema_nd(Y: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0 or alpha >= 1.0:
        return Y.copy()
    Z = Y.copy()
    for i in range(1, Y.shape[0]):
        Z[i] = alpha * Y[i] + (1.0 - alpha) * Z[i - 1]
    return Z


# ----------------------------
# Jerk-penalty smoother via CG (D3 penalty)
# ----------------------------
def _apply_D3(x: np.ndarray) -> np.ndarray:
    return x[:-3] - 3.0 * x[1:-2] + 3.0 * x[2:-1] - x[3:]


def _apply_D3t(u: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    out[:-3] += u
    out[1:-2] += -3.0 * u
    out[2:-1] += 3.0 * u
    out[3:] += -1.0 * u
    return out


def whittaker_jerk_cg_1d(y: np.ndarray, lam: float, cg_iters: int = 200, tol: float = 1e-8) -> np.ndarray:
    n = y.size
    if n < 6 or lam <= 0.0:
        return y.copy()

    def A(x: np.ndarray) -> np.ndarray:
        d3 = _apply_D3(x)
        return x + lam * _apply_D3t(d3, n)

    x = y.copy()
    r = y - A(x)
    p = r.copy()
    rr = float(r @ r)
    if rr < tol:
        return x

    yy = float(y @ y) + 1e-12
    for _ in range(cg_iters):
        Ap = A(p)
        denom = float(p @ Ap) + 1e-12
        alpha = rr / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = float(r @ r)
        if rr_new < (tol * tol) * yy:
            break
        beta = rr_new / (rr + 1e-12)
        p = r + beta * p
        rr = rr_new
    return x


def whittaker_jerk_cg_nd(Y: np.ndarray, lam: float, cg_iters: int = 200, tol: float = 1e-8) -> np.ndarray:
    Z = np.empty_like(Y)
    for d in range(Y.shape[1]):
        Z[:, d] = whittaker_jerk_cg_1d(Y[:, d], lam=lam, cg_iters=cg_iters, tol=tol)
    return Z


# ----------------------------
# Resampling helpers (same as txt_recorder)
# ----------------------------
def upsample_linear(X: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return X.copy()
    N, D = X.shape
    outN = (N - 1) * factor + 1
    out = np.empty((outN, D), dtype=np.float64)

    frac = (np.arange(factor, dtype=np.float64) / float(factor)).reshape(-1, 1)
    for i in range(N - 1):
        base = i * factor
        delta = (X[i + 1] - X[i]).reshape(1, -1)
        out[base:base + factor, :] = X[i].reshape(1, -1) + frac * delta
    out[-1, :] = X[-1, :]
    return out


def resample_uniform_by_timewarp(P: np.ndarray, F: np.ndarray, dt: float, seg_scale: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N = P.shape[0]
    assert seg_scale.shape[0] == N - 1

    tprime = np.zeros(N, dtype=np.float64)
    tprime[1:] = np.cumsum(dt * seg_scale)

    T = float(tprime[-1])
    if T <= 0.0:
        return P.copy(), F.copy()

    M = int(np.round(T / dt)) + 1
    t_u = np.arange(M, dtype=np.float64) * dt
    t_u[-1] = T

    Pn = np.empty((M, P.shape[1]), dtype=np.float64)
    Fn = np.empty((M, F.shape[1]), dtype=np.float64)
    for d in range(P.shape[1]):
        Pn[:, d] = np.interp(t_u, tprime, P[:, d])
    for d in range(F.shape[1]):
        Fn[:, d] = np.interp(t_u, tprime, F[:, d])
    return Pn, Fn


# ----------------------------
# Contact logic (same as txt_recorder)
# ----------------------------
def detect_contact_on_idx(fz: np.ndarray, thr: float, consec_on: int) -> Optional[int]:
    cnt = 0
    for i in range(fz.size):
        if fz[i] >= thr:
            cnt += 1
            if cnt >= consec_on:
                return i - consec_on + 1
        else:
            cnt = 0
    return None


def detect_contact_off_idx(fz: np.ndarray, thr: float, consec_off: int, start_from: int) -> Optional[int]:
    if start_from < 0:
        start_from = 0
    cnt = 0
    for i in range(start_from, fz.size):
        if fz[i] < thr:
            cnt += 1
            if cnt >= consec_off:
                return i - consec_off + 1
        else:
            cnt = 0
    return None


def force_process_with_contact_cleanup(
    Fraw: np.ndarray,
    clamp_abs: float,
    ema_alpha: float,
    zero_xy: bool,
    contact_thr_N: float,
    consec_on: int,
    consec_off: int,
    fz_contact_smooth_enable: bool,
    fz_contact_lam_d2: float,
    cg_iters: int,
    cg_tol: float,
) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
    Fp = np.clip(Fraw.copy(), -clamp_abs, clamp_abs)

    if zero_xy:
        Fp[:, 0] = 0.0
        Fp[:, 1] = 0.0

    if 0.0 < ema_alpha < 1.0:
        for i in range(1, Fp.shape[0]):
            Fp[i] = ema_alpha * Fp[i] + (1.0 - ema_alpha) * Fp[i - 1]

    fz = Fp[:, 2].copy()

    on_idx = detect_contact_on_idx(fz, contact_thr_N, consec_on)
    if on_idx is None:
        Fp[:, 2] = 0.0
        return Fp, None, None

    off_idx = detect_contact_off_idx(fz, contact_thr_N, consec_off, start_from=on_idx + consec_on)

    if fz_contact_smooth_enable and fz_contact_lam_d2 > 0.0:
        i0 = on_idx
        i1 = off_idx if off_idx is not None else fz.size
        if (i1 - i0) >= 8:
            seg = fz[i0:i1]
            seg_sm = whittaker_cg_1d(seg, lam=fz_contact_lam_d2, cg_iters=cg_iters, tol=cg_tol)
            fz[i0:i1] = seg_sm

    fz[:on_idx] = 0.0
    if off_idx is not None:
        fz[off_idx:] = 0.0

    Fp[:, 2] = fz
    return Fp, on_idx, off_idx


# ----------------------------
# QP-proxy evaluation (same as txt_recorder)
# ----------------------------
@dataclass
class Limits:
    pos_vmax: float
    pos_amax: float
    ang_vmax: float
    ang_amax: float
    pos_jmax: float
    ang_jmax: float


@dataclass
class EvalStats:
    N: int
    dt: float
    T: float
    vpos_max: float
    apos_max: float
    vang_max: float
    aang_max: float
    jpos_max: float
    jang_max: float
    vpos_p95: float
    apos_p95: float
    vang_p95: float
    aang_p95: float
    jpos_p95: float
    jang_p95: float
    viol_v: float
    viol_a: float
    viol_w: float
    viol_alpha: float
    viol_jpos: float
    viol_jang: float


def eval_qp_proxy(pose6: np.ndarray, dt: float, lim: Limits, safety: float = 1.0) -> Tuple[EvalStats, Dict[str, np.ndarray]]:
    N = int(pose6.shape[0])
    T = dt * max(0, (N - 1))

    dp = pose6[1:, :3] - pose6[:-1, :3]
    dr = pose6[1:, 3:] - pose6[:-1, 3:]

    vpos = norm_rows(dp) / dt
    vang = norm_rows(dr) / dt

    v = (pose6[1:, :] - pose6[:-1, :]) / dt
    a = (v[1:, :] - v[:-1, :]) / dt
    apos = norm_rows(a[:, :3])
    aang = norm_rows(a[:, 3:])

    j = (a[1:, :] - a[:-1, :]) / dt
    jpos = norm_rows(j[:, :3])
    jang = norm_rows(j[:, 3:])

    vpos_lim = lim.pos_vmax * safety
    apos_lim = lim.pos_amax * safety
    vang_lim = lim.ang_vmax * safety
    aang_lim = lim.ang_amax * safety
    jpos_lim = lim.pos_jmax * safety
    jang_lim = lim.ang_jmax * safety

    st = EvalStats(
        N=N, dt=dt, T=T,
        vpos_max=float(vpos.max()) if vpos.size else 0.0,
        apos_max=float(apos.max()) if apos.size else 0.0,
        vang_max=float(vang.max()) if vang.size else 0.0,
        aang_max=float(aang.max()) if aang.size else 0.0,
        jpos_max=float(jpos.max()) if jpos.size else 0.0,
        jang_max=float(jang.max()) if jang.size else 0.0,
        vpos_p95=pctl(vpos, 95),
        apos_p95=pctl(apos, 95),
        vang_p95=pctl(vang, 95),
        aang_p95=pctl(aang, 95),
        jpos_p95=pctl(jpos, 95),
        jang_p95=pctl(jang, 95),
        viol_v=float(np.mean(vpos > vpos_lim)) if vpos.size else 0.0,
        viol_a=float(np.mean(apos > apos_lim)) if apos.size else 0.0,
        viol_w=float(np.mean(vang > vang_lim)) if vang.size else 0.0,
        viol_alpha=float(np.mean(aang > aang_lim)) if aang.size else 0.0,
        viol_jpos=float(np.mean(jpos > jpos_lim)) if jpos.size else 0.0,
        viol_jang=float(np.mean(jang > jang_lim)) if jang.size else 0.0,
    )

    debug = {
        "vpos": vpos, "vang": vang,
        "apos": apos, "aang": aang,
        "jpos": jpos, "jang": jang,
    }
    return st, debug


def constraints_ok(st: EvalStats) -> bool:
    return (st.viol_v == 0.0 and st.viol_a == 0.0 and st.viol_w == 0.0 and st.viol_alpha == 0.0 and
            st.viol_jpos == 0.0 and st.viol_jang == 0.0)


# ============================================================
# Keyboard watcher (press 'q' to quit; no Enter)
# ============================================================
class _KeyboardQuitter:
    def __init__(self, quit_key: str = 'q'):
        self.quit_key = (quit_key or 'q').lower()
        self._stop_evt = threading.Event()
        self._hit_quit = threading.Event()
        self._thread = None
        self._enabled = False
        self._fd = None
        self._old_term = None

    def start(self):
        if not sys.stdin.isatty():
            self._enabled = False
            return False
        self._enabled = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self._restore_term()

    def hit(self) -> bool:
        return self._hit_quit.is_set()

    def _restore_term(self):
        try:
            if self._enabled and self._fd is not None and self._old_term is not None:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term)
        except Exception:
            pass
        self._fd = None
        self._old_term = None

    def _loop(self):
        try:
            self._fd = sys.stdin.fileno()
            self._old_term = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            while not self._stop_evt.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if not ch:
                    continue
                if ch.lower() == self.quit_key:
                    self._hit_quit.set()
                    break
        except Exception:
            pass
        finally:
            self._restore_term()


# ============================================================
# Main Node
# ============================================================
class VRDemoHDF5Recorder(Node):
    def __init__(self):
        super().__init__("vr_demo_hdf5_recorder")

        # -------------------------
        # HDF5 / run control (KEEP CURRENT PATH PARAMS)
        # -------------------------
        self.declare_parameter("save_dir", "/home/eunseop/dev_ws/src/y2_ur10skku_control/Y2RobMotion/datasets")
        self.declare_parameter("hdf5_name", "vr_demo_stage1.hdf5")
        self.declare_parameter("overwrite", True)
        self.declare_parameter("flush_each_episode", True)

        self.declare_parameter("num_episodes", 50)
        self.declare_parameter("min_raw_samples", 10)
        self.declare_parameter("quit_key", "q")

        # -------------------------
        # topics
        # -------------------------
        self.declare_parameter("pose_topic", "/calibrated_pose")
        self.declare_parameter("force_topic", "/ftsensor/measured_Cvalue")

        # -------------------------
        # recorder timing
        # -------------------------
        self.declare_parameter("record_hz", 125.0)
        self.declare_parameter("require_fresh_sec", 0.2)

        # -------------------------
        # episode rule
        # -------------------------
        self.declare_parameter("start_abs_fx", 10.0)
        self.declare_parameter("stop_abs_fy", 10.0)

        # -------------------------
        # force shaping (IDENTICAL to txt)
        # -------------------------
        self.declare_parameter("zero_xy_forces", True)
        self.declare_parameter("force_clamp_abs", 200.0)
        self.declare_parameter("force_ema_alpha", 0.2)

        self.declare_parameter("contact_thr_N", 5.0)
        self.declare_parameter("consec_on", 10)
        self.declare_parameter("consec_off", 10)

        self.declare_parameter("fz_contact_smooth_enable", True)
        self.declare_parameter("fz_contact_lam_d2", 4000.0)

        # -------------------------
        # pose smoothing (IDENTICAL to txt)
        # -------------------------
        self.declare_parameter("hampel_enable", True)
        self.declare_parameter("hampel_win", 16)
        self.declare_parameter("hampel_sig", 2.0)

        self.declare_parameter("lam_pos_d2", 250000.0)
        self.declare_parameter("lam_ang_d2", 6000.0)
        self.declare_parameter("pose_ema_enable", True)
        self.declare_parameter("pose_ema_alpha", 0.10)

        # retime fixed x2
        self.retime_k = 2

        # approach slow-down (IDENTICAL to txt)
        self.declare_parameter("approach_slowdown_enable", True)
        self.declare_parameter("approach_pre_sec", 5.0)
        self.declare_parameter("approach_post_sec", 0.3)
        self.declare_parameter("approach_scale_max", 30.0)
        self.declare_parameter("approach_profile", "cosine")
        self.declare_parameter("approach_use_fz_ramp", True)
        self.declare_parameter("approach_fz_full", 20.0)

        # post jerk penalty (D3)
        self.declare_parameter("post_enable", True)
        self.declare_parameter("lam_pos_d3", 2.0e7)
        self.declare_parameter("lam_ang_d3", 6.0e5)

        # QP-guard loop
        self.declare_parameter("qp_guard_enable", True)
        self.declare_parameter("qp_guard_safety", 0.75)
        self.declare_parameter("qp_guard_max_iter", 8)
        self.declare_parameter("qp_guard_growth", 2.2)
        self.declare_parameter("max_dev_pos_mm", 8.0)
        self.declare_parameter("max_dev_ang_rad", 0.06)

        # CG
        self.declare_parameter("cg_iters", 400)
        self.declare_parameter("cg_tol", 1e-8)

        # QP-proxy limits
        self.declare_parameter("pos_vmax", 30.0)
        self.declare_parameter("pos_amax", 120.0)
        self.declare_parameter("ang_vmax", 0.6)
        self.declare_parameter("ang_amax", 3.0)
        self.declare_parameter("pos_jmax", 5000.0)
        self.declare_parameter("ang_jmax", 80.0)

        # -------------------------
        # load params
        # -------------------------
        self.save_dir = str(self.get_parameter("save_dir").value)
        self.hdf5_name = str(self.get_parameter("hdf5_name").value)
        self.overwrite = bool(self.get_parameter("overwrite").value)
        self.flush_each_episode = bool(self.get_parameter("flush_each_episode").value)

        self.num_episodes = int(self.get_parameter("num_episodes").value)
        self.min_raw_samples = int(self.get_parameter("min_raw_samples").value)
        self.quit_key = str(self.get_parameter("quit_key").value)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.force_topic = str(self.get_parameter("force_topic").value)

        self.record_hz = float(self.get_parameter("record_hz").value)
        self.dt = 1.0 / max(1e-9, self.record_hz)
        self.require_fresh_sec = float(self.get_parameter("require_fresh_sec").value)

        self.start_abs_fx = float(self.get_parameter("start_abs_fx").value)
        self.stop_abs_fy = float(self.get_parameter("stop_abs_fy").value)

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

        self.approach_slowdown_enable = bool(self.get_parameter("approach_slowdown_enable").value)
        self.approach_pre_sec = float(self.get_parameter("approach_pre_sec").value)
        self.approach_post_sec = float(self.get_parameter("approach_post_sec").value)
        self.approach_scale_max = float(self.get_parameter("approach_scale_max").value)
        self.approach_profile = str(self.get_parameter("approach_profile").value)
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

        self.lim = Limits(
            pos_vmax=float(self.get_parameter("pos_vmax").value),
            pos_amax=float(self.get_parameter("pos_amax").value),
            ang_vmax=float(self.get_parameter("ang_vmax").value),
            ang_amax=float(self.get_parameter("ang_amax").value),
            pos_jmax=float(self.get_parameter("pos_jmax").value),
            ang_jmax=float(self.get_parameter("ang_jmax").value),
        )

        # -------------------------
        # file open (KEEP CURRENT)
        # -------------------------
        os.makedirs(self.save_dir, exist_ok=True)
        self.hdf5_path = os.path.join(self.save_dir, self.hdf5_name)
        if self.overwrite and os.path.exists(self.hdf5_path):
            os.remove(self.hdf5_path)

        self.h5_lock = threading.Lock()
        self.h5 = h5py.File(self.hdf5_path, "a")
        self.grp_eps = self.h5.require_group("episodes")

        self.episode_count = self._detect_existing_episode_count()
        self._write_root_meta_once()

        # -------------------------
        # runtime state
        # -------------------------
        self.state_lock = threading.Lock()

        self.latest_pose6_mm_rad: Optional[np.ndarray] = None
        self.latest_force3_N: Optional[np.ndarray] = None
        self.latest_pose_t: float = 0.0
        self.latest_force_t: float = 0.0

        self.episode_active = False
        self.finishing_ = False

        self.buf_pose: List[np.ndarray] = []
        self.buf_force: List[np.ndarray] = []

        self.stop_requested = False
        self.stop_reason = ""

        # -------------------------
        # ROS IO
        # -------------------------
        self.sub_pose = self.create_subscription(Float64MultiArray, self.pose_topic, self.cb_pose, 50)
        self.sub_force = self.create_subscription(Wrench, self.force_topic, self.cb_force, 10)
        self.timer = self.create_timer(self.dt, self.cb_timer)
        self.timer_stop = self.create_timer(0.05, self._check_stop)

        # keyboard
        self.kb = _KeyboardQuitter(quit_key=self.quit_key)
        enabled = self.kb.start()
        atexit.register(self.kb.stop)

        # logs
        self.get_logger().info("============================================================")
        self.get_logger().info("VRDemoHDF5Recorder initialized (txt-identical pipeline)")
        self.get_logger().info(f"  HDF5: {self.hdf5_path}")
        self.get_logger().info(f"  Topics: pose={self.pose_topic}, force={self.force_topic}")
        self.get_logger().info(f"  record_hz={self.record_hz}, require_fresh_sec={self.require_fresh_sec}")
        self.get_logger().info(f"  Episode rule: start=|fx|>={self.start_abs_fx}, end=|fy|>={self.stop_abs_fy}")
        self.get_logger().info(f"  Target episodes: {self.num_episodes} (existing: {self.episode_count})")
        self.get_logger().info(
            f"  Force: clamp_abs={self.force_clamp_abs}, EMA={self.force_ema_alpha}, zero_xy={self.zero_xy_forces}, "
            f"thr={self.contact_thr_N}, consec_on={self.consec_on}, consec_off={self.consec_off}, "
            f"fz_contact_smooth={self.fz_contact_smooth_enable}(lam={self.fz_contact_lam_d2})"
        )
        self.get_logger().info(
            f"  Pose pre: Hampel={self.hampel_enable}(win={self.hampel_win}, sig={self.hampel_sig}), "
            f"D2(lam_pos={self.lam_pos_d2}, lam_ang={self.lam_ang_d2}), PoseEMA={self.pose_ema_enable}(alpha={self.pose_ema_alpha})"
        )
        self.get_logger().info(
            f"  Retime: fixed x2, Approach slow-down: enable={self.approach_slowdown_enable}, pre={self.approach_pre_sec}, post={self.approach_post_sec}, scale_max={self.approach_scale_max}"
        )
        self.get_logger().info(
            f"  Post: D3(lam_pos={self.lam_pos_d3}, lam_ang={self.lam_ang_d3}), QP-guard enable={self.qp_guard_enable} safety={self.qp_guard_safety}"
        )
        if enabled:
            self.get_logger().info(f"  Press '{self.quit_key}' to stop (no Enter). Ctrl+C also works.")
        else:
            self.get_logger().warn("  stdin is not a TTY -> 'q' quit disabled. Use Ctrl+C instead.")
        self.get_logger().info("============================================================")

    # ============================================================
    # HDF5 helpers
    # ============================================================
    def _detect_existing_episode_count(self) -> int:
        max_idx = -1
        for k in self.grp_eps.keys():
            if k.startswith("ep_"):
                try:
                    idx = int(k.split("_")[1])
                    max_idx = max(max_idx, idx)
                except Exception:
                    pass
        return max_idx + 1

    def _write_root_meta_once(self):
        if "created_unix" not in self.h5.attrs:
            self.h5.attrs["created_unix"] = float(time.time())
        self.h5.attrs["columns"] = np.string_("x_mm,y_mm,z_mm,wx,wy,wz,fx,fy,fz")
        self.h5.attrs["note_pose"] = np.string_("pose xyz input meters -> stored millimeters; omega stored radians")
        self.h5.attrs["record_hz"] = float(self.record_hz)
        self.h5.attrs["dt"] = float(self.dt)
        self.h5.attrs["episode_rule"] = np.string_(f"start=|fx|>={self.start_abs_fx}, end=|fy|>={self.stop_abs_fy}")
        self.h5.flush()

    # ============================================================
    # stop control
    # ============================================================
    def request_stop(self, reason: str = "user_request"):
        self.stop_requested = True
        self.stop_reason = str(reason)
        self.get_logger().warn(f"[STOP REQUEST] reason={self.stop_reason}")

    def _check_stop(self):
        if self.stop_requested and (not self.finishing_) and (not self.episode_active):
            self.finalize_and_shutdown()

    def finalize_and_shutdown(self):
        self.get_logger().warn("Finalizing HDF5 and shutting down...")
        try:
            with self.h5_lock:
                try:
                    self.h5.flush()
                except Exception:
                    pass
                try:
                    self.h5.close()
                except Exception:
                    pass
        finally:
            try:
                self.kb.stop()
            except Exception:
                pass
            try:
                self.destroy_node()
            except Exception:
                pass
            try:
                rclpy.shutdown()
            except Exception:
                pass

    # ============================================================
    # ROS callbacks (collect raw)
    # ============================================================
    def cb_pose(self, msg: Float64MultiArray):
        if len(msg.data) < 6:
            return
        x, y, z, wx, wy, wz = msg.data[:6]
        pose = np.array([1000.0 * x, 1000.0 * y, 1000.0 * z, wx, wy, wz], dtype=np.float64)
        with self.state_lock:
            self.latest_pose6_mm_rad = pose
            self.latest_pose_t = time.time()

    def cb_force(self, msg: Wrench):
        fx = float(msg.force.x)
        fy = float(msg.force.y)
        fz = float(msg.force.z)
        F = np.array([fx, fy, fz], dtype=np.float64)

        with self.state_lock:
            self.latest_force3_N = F
            self.latest_force_t = time.time()

        if self.stop_requested:
            return

        if self.kb.hit() and (not self.stop_requested):
            self.request_stop(reason=f"keyboard_{self.quit_key}")

        if self.finishing_:
            return

        # start
        if (not self.episode_active) and (abs(fx) >= self.start_abs_fx):
            with self.state_lock:
                self.episode_active = True
                self.buf_pose.clear()
                self.buf_force.clear()
            self.get_logger().info(f"=== EPISODE STARTED (idx={self.episode_count:04d}) ===")
            return

        # end
        if self.episode_active and (abs(fy) >= self.stop_abs_fy):
            self.get_logger().info(f"=== EPISODE ENDED (idx={self.episode_count:04d}) by |fy| threshold ===")
            self._start_finish_thread(reason="fy_threshold")
            return

    def cb_timer(self):
        if self.stop_requested and self.episode_active and (not self.finishing_):
            self.get_logger().warn("Stop requested while recording -> closing current episode.")
            self._start_finish_thread(reason=self.stop_reason or "stop_requested")
            return

        if (not self.episode_active) or self.finishing_ or self.stop_requested:
            return

        now = time.time()
        with self.state_lock:
            if self.latest_pose6_mm_rad is None or (now - self.latest_pose_t) > self.require_fresh_sec:
                return
            if self.latest_force3_N is None or (now - self.latest_force_t) > self.require_fresh_sec:
                return

            self.buf_pose.append(self.latest_pose6_mm_rad.copy())
            self.buf_force.append(self.latest_force3_N.copy())

    # ============================================================
    # Pipeline blocks (IDENTICAL to txt_recorder)
    # ============================================================
    def _pose_pre_smooth(self, P: np.ndarray) -> np.ndarray:
        P0 = P.copy()
        if self.hampel_enable:
            P0 = hampel_nd(P0, win=self.hampel_win, n_sigmas=self.hampel_sig)

        P1 = P0.copy()
        P1[:, :3] = whittaker_cg_nd(P1[:, :3], lam=self.lam_pos_d2, cg_iters=self.cg_iters, tol=self.cg_tol)
        P1[:, 3:] = whittaker_cg_nd(P1[:, 3:], lam=self.lam_ang_d2, cg_iters=self.cg_iters, tol=self.cg_tol)

        if self.pose_ema_enable:
            P1 = ema_nd(P1, alpha=self.pose_ema_alpha)
        return P1

    def _retime_x2(self, P: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Pr = upsample_linear(P, self.retime_k)
        Fr = upsample_linear(F, self.retime_k)
        return Pr, Fr

    def _apply_contact_approach_slowdown(self, Pr: np.ndarray, Fr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.approach_slowdown_enable:
            return Pr, Fr

        fz = Fr[:, 2]
        cidx = detect_contact_on_idx(fz, self.contact_thr_N, self.consec_on)
        if cidx is None:
            self.get_logger().warn("[APPROACH] contact not found -> skip approach slow-down")
            return Pr, Fr

        preN = int(round(self.approach_pre_sec * self.record_hz))
        postN = int(round(self.approach_post_sec * self.record_hz))

        N = Pr.shape[0]
        seg_scale = np.ones(N - 1, dtype=np.float64)

        s0 = max(0, cidx - preN)
        s1 = min(N - 1, cidx + postN)
        if s1 <= s0 + 2:
            self.get_logger().warn("[APPROACH] window too small -> skip")
            return Pr, Fr

        idx = np.arange(s0, s1, dtype=np.float64)
        u = (idx - float(s0)) / max(1.0, float(s1 - s0))

        bump = 0.5 - 0.5 * np.cos(2.0 * np.pi * u)
        bump = np.clip(bump, 0.0, 1.0)

        scale_target = 1.0 + (self.approach_scale_max - 1.0) * bump

        if self.approach_use_fz_ramp:
            fz_win = fz[s0:s1]
            ramp = np.clip(fz_win / max(1e-6, self.approach_fz_full), 0.0, 1.0)
            scale_target = 1.0 + (scale_target - 1.0) * ramp

        seg_scale[s0:s1] = np.maximum(seg_scale[s0:s1], scale_target)

        Pn, Fn = resample_uniform_by_timewarp(Pr, Fr, self.dt, seg_scale)
        return Pn, Fn

    def _pose_post_smooth_d3(self, P: np.ndarray, lam_pos_d3: float, lam_ang_d3: float) -> np.ndarray:
        if not self.post_enable:
            return P
        P2 = P.copy()
        P2[:, :3] = whittaker_jerk_cg_nd(P2[:, :3], lam=lam_pos_d3, cg_iters=self.cg_iters, tol=self.cg_tol)
        P2[:, 3:] = whittaker_jerk_cg_nd(P2[:, 3:], lam=lam_ang_d3, cg_iters=self.cg_iters, tol=self.cg_tol)
        return P2

    def _qp_guard(self, Pref: np.ndarray) -> np.ndarray:
        if not self.qp_guard_enable:
            return self._pose_post_smooth_d3(Pref, self.lam_pos_d3, self.lam_ang_d3)

        lam_p = self.lam_pos_d3
        lam_a = self.lam_ang_d3
        best = None
        best_score = 1e18

        for _ in range(max(1, self.qp_guard_max_iter)):
            Pk = self._pose_post_smooth_d3(Pref, lam_p, lam_a)

            dpos = norm_rows(Pk[:, :3] - Pref[:, :3])
            dang = norm_rows(Pk[:, 3:] - Pref[:, 3:])
            if float(dpos.max()) > self.max_dev_pos_mm or float(dang.max()) > self.max_dev_ang_rad:
                break

            st, _ = eval_qp_proxy(Pk, self.dt, self.lim, safety=self.qp_guard_safety)
            score = max(
                st.jpos_p95 / (self.lim.pos_jmax * self.qp_guard_safety + 1e-9),
                st.jang_p95 / (self.lim.ang_jmax * self.qp_guard_safety + 1e-9),
                st.apos_p95 / (self.lim.pos_amax * self.qp_guard_safety + 1e-9),
                st.aang_p95 / (self.lim.ang_amax * self.qp_guard_safety + 1e-9),
            )
            if score < best_score:
                best_score = score
                best = Pk

            if constraints_ok(st):
                return Pk

            lam_p *= self.qp_guard_growth
            lam_a *= self.qp_guard_growth

        return best if best is not None else self._pose_post_smooth_d3(Pref, self.lam_pos_d3, self.lam_ang_d3)

    # ============================================================
    # Finish episode (threaded)
    # ============================================================
    def _start_finish_thread(self, reason: str):
        if self.finishing_:
            return
        self.finishing_ = True

        with self.state_lock:
            self.episode_active = False
            P_list = self.buf_pose.copy()
            F_list = self.buf_force.copy()
            self.buf_pose.clear()
            self.buf_force.clear()

        th = threading.Thread(target=self._finish_episode_worker, args=(P_list, F_list, reason), daemon=True)
        th.start()

    def _finish_episode_worker(self, P_list: List[np.ndarray], F_list: List[np.ndarray], reason: str):
        try:
            if len(P_list) < max(1, self.min_raw_samples):
                self.get_logger().warn(f"Episode dropped (too short): raw_len={len(P_list)} < {self.min_raw_samples}, reason={reason}")
                return

            rawP = np.asarray(P_list, dtype=np.float64)   # (N,6) [mm, rad]
            rawF = np.asarray(F_list, dtype=np.float64)   # (N,3) [N]
            rawN = int(rawP.shape[0])

            # 1) force process (IDENTICAL)
            Fp, on_idx, off_idx = force_process_with_contact_cleanup(
                rawF,
                clamp_abs=self.force_clamp_abs,
                ema_alpha=self.force_ema_alpha,
                zero_xy=self.zero_xy_forces,
                contact_thr_N=self.contact_thr_N,
                consec_on=self.consec_on,
                consec_off=self.consec_off,
                fz_contact_smooth_enable=self.fz_contact_smooth_enable,
                fz_contact_lam_d2=self.fz_contact_lam_d2,
                cg_iters=self.cg_iters,
                cg_tol=self.cg_tol,
            )

            # 2) pose pre smooth (IDENTICAL)
            Ps = self._pose_pre_smooth(rawP)

            # 3) retime fixed x2 (IDENTICAL)
            Pr, Fr = self._retime_x2(Ps, Fp)

            # 4) approach slow-down (IDENTICAL)
            Pr_slow, Fr_slow = self._apply_contact_approach_slowdown(Pr, Fr)

            # 5) final pose smoothing + QP-guard (IDENTICAL)
            Pf = self._qp_guard(Pr_slow)

            out = np.hstack([Pf, Fr_slow]).astype(np.float32)  # (M,9)

            # save
            ep_idx = self.episode_count
            used = {
                "contact_thr_N": float(self.contact_thr_N),
                "consec_on": int(self.consec_on),
                "consec_off": int(self.consec_off),
                "fz_contact_lam_d2": float(self.fz_contact_lam_d2),
                "fz_on_idx": None if on_idx is None else int(on_idx),
                "fz_off_idx": None if off_idx is None else int(off_idx),
                "retime_k": int(self.retime_k),
            }
            self._save_episode_to_hdf5(ep_idx, out, reason=reason, raw_len=rawN, used_meta=used)
            self.episode_count += 1

            self.get_logger().info(
                f"=== EPISODE SAVED (idx={ep_idx:04d}) raw_len={rawN} -> out_len={out.shape[0]} reason={reason} ==="
            )

            if self.episode_count >= self.num_episodes:
                self.request_stop(reason="reached_num_episodes")

        except Exception as e:
            self.get_logger().error(f"Episode processing failed: {e}")
        finally:
            self.finishing_ = False

    # ============================================================
    # HDF5 save (KEEP CURRENT GROUP LAYOUT)
    # ============================================================
    def _save_episode_to_hdf5(
        self,
        ep_idx: int,
        out: np.ndarray,
        reason: str,
        raw_len: int,
        used_meta: Dict[str, object],
    ):
        ep_name = f"ep_{ep_idx:04d}"
        with self.h5_lock:
            if ep_name in self.grp_eps:
                del self.grp_eps[ep_name]
            g = self.grp_eps.create_group(ep_name)

            g.attrs["saved_unix"] = float(time.time())
            g.attrs["reason"] = np.string_(str(reason))
            g.attrs["raw_len"] = int(raw_len)
            g.attrs["out_len"] = int(out.shape[0])
            g.attrs["dtype"] = np.string_(str(out.dtype))
            g.attrs["record_hz"] = float(self.record_hz)
            g.attrs["dt"] = float(self.dt)

            # store key params for reproducibility
            g.attrs["force_clamp_abs"] = float(self.force_clamp_abs)
            g.attrs["force_ema_alpha"] = float(self.force_ema_alpha)
            g.attrs["zero_xy_forces"] = int(bool(self.zero_xy_forces))

            g.attrs["pose_hampel_enable"] = int(bool(self.hampel_enable))
            g.attrs["pose_hampel_win"] = int(self.hampel_win)
            g.attrs["pose_hampel_sig"] = float(self.hampel_sig)
            g.attrs["lam_pos_d2"] = float(self.lam_pos_d2)
            g.attrs["lam_ang_d2"] = float(self.lam_ang_d2)
            g.attrs["pose_ema_enable"] = int(bool(self.pose_ema_enable))
            g.attrs["pose_ema_alpha"] = float(self.pose_ema_alpha)

            g.attrs["post_enable"] = int(bool(self.post_enable))
            g.attrs["lam_pos_d3"] = float(self.lam_pos_d3)
            g.attrs["lam_ang_d3"] = float(self.lam_ang_d3)

            g.attrs["qp_guard_enable"] = int(bool(self.qp_guard_enable))
            g.attrs["qp_guard_safety"] = float(self.qp_guard_safety)
            g.attrs["qp_guard_max_iter"] = int(self.qp_guard_max_iter)
            g.attrs["qp_guard_growth"] = float(self.qp_guard_growth)
            g.attrs["max_dev_pos_mm"] = float(self.max_dev_pos_mm)
            g.attrs["max_dev_ang_rad"] = float(self.max_dev_ang_rad)

            g.attrs["used_meta_json"] = np.string_(json.dumps(used_meta))

            g.create_dataset(
                "traj",
                data=out,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            if self.flush_each_episode:
                self.h5.flush()


def main(args=None):
    rclpy.init(args=args)
    node = VRDemoHDF5Recorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        try:
            node.request_stop(reason="KeyboardInterrupt")
        except Exception:
            pass
        time.sleep(0.1)
        try:
            if rclpy.ok():
                node.finalize_and_shutdown()
        except Exception:
            pass
    finally:
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
