#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vr_demo_hdf5_recorder.py

Patched version for IL HDF5 dataset generation.

Inputs
------
- VR tracker: /calibrated_pose
    std_msgs/Float64MultiArray: [x, y, z, wx, wy, wz]
    input xyz unit: meter
    stored xyz unit: millimeter

- FT sensor: /ftsensor/measured_Cvalue
    geometry_msgs/Wrench
    stored force unit: Newton

Episode rule is unchanged.
- start: |Fx| >= start_abs_fx
- end  : |Fy| >= stop_abs_fy

Major changes
-------------
1) Pose and force are collected at their own callback rates.
   - Pose samples and force samples are buffered independently in memory.
   - The saved IL trajectory episodes/ep_xxxx/traj is merged on a uniform pose-time axis.
   - Force is processed on the original force-time axis and interpolated to the pose-time axis.
   - HDF5 storage layout is kept identical to the original HDF5 recorder: each episode stores only traj.

2) Save path is generated when the first episode is saved, not at node startup.
   Default:
     /home/eunseop/nrs_act/datasets/ACT/YYYYMMDD_HHMM/episodes_ft/
     vr_demo_merged_YYYYMMDD_HHMM.hdf5

3) Force filtering follows the uploaded vr_demo_txt_recorder.py behavior.
   - Fx, Fy -> 0 if zero_xy_forces=True
   - Fz -> raw Fz + EMA only
   - first/last force_edge_zero_sec seconds -> all force values zero
   - no Fz clamp
   - no contact pre-zero/post-zero cleanup
   - approach slow-down contact detection uses abs(EMA Fz)
"""

import os
import sys
import time
import json
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
# HDF5 string helper: works with NumPy 1.x and 2.x
# ============================================================
def h5str(s: object):
    return np.bytes_(str(s))


# ============================================================
# Shared utilities
# ============================================================
def pctl(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, q))


def norm_rows(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=1)


def estimate_hz_from_time(t: np.ndarray) -> float:
    if t.size < 2:
        return 0.0
    dt = np.diff(t.astype(np.float64))
    dt = dt[np.isfinite(dt) & (dt > 1e-9)]
    if dt.size == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def sanitize_time_series(t: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort by time and remove duplicate/non-increasing samples."""
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    X = np.asarray(X, dtype=np.float64)
    if t.size == 0 or X.shape[0] == 0:
        return t, X
    n = min(t.size, X.shape[0])
    t = t[:n]
    X = X[:n]

    order = np.argsort(t)
    t = t[order]
    X = X[order]

    keep = np.ones(t.size, dtype=bool)
    if t.size >= 2:
        keep[1:] = np.diff(t) > 1e-9
    return t[keep], X[keep]


def resample_to_uniform_time(t: np.ndarray, X: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Resample X(t) to a uniform time axis with spacing dt."""
    t, X = sanitize_time_series(t, X)
    if t.size == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0, X.shape[1] if X.ndim == 2 else 0), dtype=np.float64)
    if t.size == 1:
        return np.array([0.0], dtype=np.float64), X.copy()

    t0 = float(t[0])
    t1 = float(t[-1])
    T = max(0.0, t1 - t0)
    M = int(np.floor(T / dt)) + 1
    if M < 2:
        M = 2
    tu = np.arange(M, dtype=np.float64) * dt
    if tu[-1] < T:
        tu = np.append(tu, T)

    tabs = t0 + tu
    Xu = np.empty((tu.size, X.shape[1]), dtype=np.float64)
    for d in range(X.shape[1]):
        Xu[:, d] = np.interp(tabs, t, X[:, d])
    return tu, Xu


def interpolate_to_time_axis(src_t: np.ndarray, src_X: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    src_t, src_X = sanitize_time_series(src_t, src_X)
    dst_t = np.asarray(dst_t, dtype=np.float64).reshape(-1)
    if dst_t.size == 0:
        return np.zeros((0, src_X.shape[1] if src_X.ndim == 2 else 0), dtype=np.float64)
    if src_t.size == 0:
        return np.zeros((dst_t.size, src_X.shape[1] if src_X.ndim == 2 else 0), dtype=np.float64)
    if src_t.size == 1:
        return np.repeat(src_X[:1, :], dst_t.size, axis=0)

    Y = np.empty((dst_t.size, src_X.shape[1]), dtype=np.float64)
    for d in range(src_X.shape[1]):
        Y[:, d] = np.interp(dst_t, src_t, src_X[:, d], left=src_X[0, d], right=src_X[-1, d])
    return Y


# ============================================================
# Hampel filter (per-dim)
# ============================================================
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


# ============================================================
# Whittaker smoother via CG (D2 penalty)
# ============================================================
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


def ema_1d(y: np.ndarray, alpha: float) -> np.ndarray:
    if y.size == 0:
        return y.copy()
    if alpha <= 0.0 or alpha >= 1.0:
        return y.copy()
    z = y.astype(np.float64).copy()
    for i in range(1, y.size):
        z[i] = alpha * y[i] + (1.0 - alpha) * z[i - 1]
    return z


# ============================================================
# Jerk-penalty smoother via CG (D3 penalty)
# ============================================================
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


# ============================================================
# Resampling helpers
# ============================================================
def upsample_linear(X: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return X.copy()
    N, D = X.shape
    if N <= 1:
        return X.copy()
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


# ============================================================
# Contact detection for approach slow-down
# ============================================================
def detect_contact_idx(fz: np.ndarray, fz_on: float, consec_on: int) -> Optional[int]:
    cnt_on = 0
    sig = np.abs(fz)
    for i in range(sig.size):
        if sig[i] >= fz_on:
            cnt_on += 1
            if cnt_on >= consec_on:
                return i - consec_on + 1
        else:
            cnt_on = 0
    return None


# ============================================================
# Force processing: same policy as uploaded vr_demo_txt_recorder.py
# ============================================================
def process_force_keep_fz_with_ema_and_edge_zero(
    Fraw: np.ndarray,
    traw: Optional[np.ndarray],
    fz_ema_alpha: float,
    edge_zero_sec: float,
    zero_xy: bool = True,
    logger=None,
) -> np.ndarray:
    """
    - Fx, Fy -> 0 when zero_xy=True
    - Fz -> raw Fz + EMA only
    - First/last edge_zero_sec seconds -> all force = 0
    - No clamp, no Fz contact cleanup
    """
    Fp = Fraw.astype(np.float64).copy()
    N = Fp.shape[0]
    if N == 0:
        return Fp

    if zero_xy:
        Fp[:, 0] = 0.0
        Fp[:, 1] = 0.0

    Fp[:, 2] = ema_1d(Fp[:, 2], alpha=fz_ema_alpha)

    if edge_zero_sec > 0.0:
        if traw is not None and np.asarray(traw).size == N:
            t = np.asarray(traw, dtype=np.float64).reshape(-1)
            T = float(t[-1] - t[0]) if N >= 2 else 0.0
            rel = t - float(t[0])
            mask = (rel <= edge_zero_sec) | (rel >= max(0.0, T - edge_zero_sec))
            Fp[mask, :] = 0.0
            edge_info = f"edge_zero_sec={edge_zero_sec}, edge_zero_by_time=True, zeroed={int(np.sum(mask))}/{N}"
        else:
            edge_n = int(round(edge_zero_sec * estimate_hz_from_time(np.arange(N, dtype=np.float64))))
            edge_n = max(0, min(edge_n, N))
            Fp[:edge_n, :] = 0.0
            Fp[max(0, N - edge_n):, :] = 0.0
            edge_info = f"edge_zero_sec={edge_zero_sec}, edge_zero_by_time=False, edge_zero_samples={edge_n}"
    else:
        edge_info = "edge_zero_sec=0.0"

    if logger is not None:
        logger.info(
            f"[FORCE] zero_xy={zero_xy}, fz_ema_alpha={fz_ema_alpha}, {edge_info}, N={N}"
        )
        logger.info(
            f"[FORCE] raw_fz_abs_max={float(np.max(np.abs(Fraw[:, 2]))):.3f}, "
            f"proc_fz_abs_max={float(np.max(np.abs(Fp[:, 2]))):.3f}"
        )

    return Fp


# ============================================================
# QP-proxy evaluation
# ============================================================
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

    vpos = norm_rows(dp) / dt if dp.size else np.zeros((0,), dtype=np.float64)
    vang = norm_rows(dr) / dt if dr.size else np.zeros((0,), dtype=np.float64)

    v = (pose6[1:, :] - pose6[:-1, :]) / dt if N >= 2 else np.zeros((0, 6), dtype=np.float64)
    a = (v[1:, :] - v[:-1, :]) / dt if v.shape[0] >= 2 else np.zeros((0, 6), dtype=np.float64)
    apos = norm_rows(a[:, :3]) if a.size else np.zeros((0,), dtype=np.float64)
    aang = norm_rows(a[:, 3:]) if a.size else np.zeros((0,), dtype=np.float64)

    j = (a[1:, :] - a[:-1, :]) / dt if a.shape[0] >= 2 else np.zeros((0, 6), dtype=np.float64)
    jpos = norm_rows(j[:, :3]) if j.size else np.zeros((0,), dtype=np.float64)
    jang = norm_rows(j[:, 3:]) if j.size else np.zeros((0,), dtype=np.float64)

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


def print_eval(logger, title: str, st: EvalStats, lim: Limits, safety: float):
    logger.info(f"[QP-EVAL] ===== {title} =====")
    logger.info(
        f"\n  N={st.N}  dt={st.dt:.6f}s  T={st.T:.3f}s"
        f"\n  pos |v|: max={st.vpos_max:.3f} (lim {lim.pos_vmax*safety:.3f}), p95={st.vpos_p95:.3f}  [mm/s]"
        f"\n  pos |a|: max={st.apos_max:.3f} (lim {lim.pos_amax*safety:.3f}), p95={st.apos_p95:.3f}  [mm/s^2]"
        f"\n  rotvec |r_dot|: max={st.vang_max:.3f} (lim {lim.ang_vmax*safety:.3f}), p95={st.vang_p95:.3f}  [rad/s]"
        f"\n  rotvec |r_ddot|: max={st.aang_max:.3f} (lim {lim.ang_amax*safety:.3f}), p95={st.aang_p95:.3f}  [rad/s^2]"
        f"\n  jerk: pos max={st.jpos_max:.3f} (lim {lim.pos_jmax*safety:.3f}), p95={st.jpos_p95:.3f}  [mm/s^3]"
        f"\n        ang max={st.jang_max:.3f} (lim {lim.ang_jmax*safety:.3f}), p95={st.jang_p95:.3f}  [rad/s^3]"
        f"\n  violation_rate: vpos={100*st.viol_v:.3f}%, apos={100*st.viol_a:.3f}%, "
        f"rdot={100*st.viol_w:.3f}%, rddot={100*st.viol_alpha:.3f}%, "
        f"jpos={100*st.viol_jpos:.3f}%, jang={100*st.viol_jang:.3f}%"
    )


def constraints_ok(st: EvalStats) -> bool:
    return (
        st.viol_v == 0.0 and st.viol_a == 0.0 and st.viol_w == 0.0 and
        st.viol_alpha == 0.0 and st.viol_jpos == 0.0 and st.viol_jang == 0.0
    )


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
        # Save path / run control
        # -------------------------
        self.declare_parameter("save_root", "/home/eunseop/nrs_act/datasets/ACT")
        self.declare_parameter("subdir_name", "episodes_ft")
        self.declare_parameter("file_ext", ".hdf5")  # requested spelling
        self.declare_parameter("overwrite", True)
        self.declare_parameter("flush_each_episode", True)

        self.declare_parameter("num_episodes", 50)
        self.declare_parameter("min_pose_samples", 10)
        self.declare_parameter("min_force_samples", 10)
        self.declare_parameter("quit_key", "q")

        # -------------------------
        # Topics
        # -------------------------
        self.declare_parameter("pose_topic", "/calibrated_pose")
        self.declare_parameter("force_topic", "/ftsensor/measured_Cvalue")

        # -------------------------
        # Timing
        # -------------------------
        self.declare_parameter("pose_record_hz", 125.0)
        self.declare_parameter("force_record_hz", 500.0)
        self.declare_parameter("require_fresh_sec", 0.2)

        # -------------------------
        # Episode rule: unchanged
        # -------------------------
        self.declare_parameter("start_abs_fx", 10.0)
        self.declare_parameter("stop_abs_fy", 10.0)

        # -------------------------
        # Force processing: same policy as uploaded txt recorder
        # -------------------------
        self.declare_parameter("zero_xy_forces", True)
        self.declare_parameter("fz_ema_alpha", 0.2)
        self.declare_parameter("force_edge_zero_sec", 3.0)

        # Contact detection only for optional approach slow-down
        self.declare_parameter("fz_gate_N", 10.0)
        self.declare_parameter("consec_on", 10)

        # -------------------------
        # Pose smoothing / retime / approach slow-down
        # -------------------------
        self.declare_parameter("hampel_enable", True)
        self.declare_parameter("hampel_win", 16)
        self.declare_parameter("hampel_sig", 2.0)

        self.declare_parameter("lam_pos_d2", 250000.0)
        self.declare_parameter("lam_ang_d2", 6000.0)
        self.declare_parameter("pose_ema_enable", True)
        self.declare_parameter("pose_ema_alpha", 0.10)

        self.retime_k = 2

        self.declare_parameter("approach_slowdown_enable", False)
        self.declare_parameter("approach_pre_sec", 5.0)
        self.declare_parameter("approach_post_sec", 0.3)
        self.declare_parameter("approach_scale_max", 30.0)
        self.declare_parameter("approach_profile", "cosine")
        self.declare_parameter("approach_use_fz_ramp", True)
        self.declare_parameter("approach_fz_full", 20.0)

        # Post jerk penalty (D3)
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
        # Load params
        # -------------------------
        self.save_root = str(self.get_parameter("save_root").value)
        self.subdir_name = str(self.get_parameter("subdir_name").value)
        self.file_ext = str(self.get_parameter("file_ext").value)
        if not self.file_ext.startswith("."):
            self.file_ext = "." + self.file_ext
        self.overwrite = bool(self.get_parameter("overwrite").value)
        self.flush_each_episode = bool(self.get_parameter("flush_each_episode").value)

        self.num_episodes = int(self.get_parameter("num_episodes").value)
        self.min_pose_samples = int(self.get_parameter("min_pose_samples").value)
        self.min_force_samples = int(self.get_parameter("min_force_samples").value)
        self.quit_key = str(self.get_parameter("quit_key").value)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.force_topic = str(self.get_parameter("force_topic").value)

        self.pose_record_hz = float(self.get_parameter("pose_record_hz").value)
        self.force_record_hz = float(self.get_parameter("force_record_hz").value)
        self.dt = 1.0 / max(1e-9, self.pose_record_hz)
        self.require_fresh_sec = float(self.get_parameter("require_fresh_sec").value)

        self.start_abs_fx = float(self.get_parameter("start_abs_fx").value)
        self.stop_abs_fy = float(self.get_parameter("stop_abs_fy").value)

        self.zero_xy_forces = bool(self.get_parameter("zero_xy_forces").value)
        self.fz_ema_alpha = float(self.get_parameter("fz_ema_alpha").value)
        self.force_edge_zero_sec = float(self.get_parameter("force_edge_zero_sec").value)

        self.fz_gate_N = float(self.get_parameter("fz_gate_N").value)
        self.consec_on = int(self.get_parameter("consec_on").value)

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
        # Lazy HDF5 open: path is created at first episode save time
        # -------------------------
        self.h5_lock = threading.Lock()
        self.h5: Optional[h5py.File] = None
        self.grp_eps = None
        self.hdf5_path: Optional[str] = None
        self.save_dir: Optional[str] = None
        self.run_stamp: Optional[str] = None
        self.episode_count = 0

        # -------------------------
        # Runtime state
        # -------------------------
        self.state_lock = threading.Lock()

        self.latest_pose6_mm_rad: Optional[np.ndarray] = None
        self.latest_force3_N: Optional[np.ndarray] = None
        self.latest_pose_t_abs: float = 0.0
        self.latest_force_t_abs: float = 0.0

        self.episode_active = False
        self.finishing_ = False
        self.episode_start_t_abs: Optional[float] = None

        self.buf_pose_t: List[float] = []
        self.buf_pose: List[np.ndarray] = []
        self.buf_force_t: List[float] = []
        self.buf_force: List[np.ndarray] = []

        self.stop_requested = False
        self.stop_reason = ""

        # -------------------------
        # ROS IO
        # -------------------------
        self.sub_pose = self.create_subscription(Float64MultiArray, self.pose_topic, self.cb_pose, 200)
        self.sub_force = self.create_subscription(Wrench, self.force_topic, self.cb_force, 2000)
        self.timer_stop = self.create_timer(0.05, self._check_stop)

        # Keyboard
        self.kb = _KeyboardQuitter(quit_key=self.quit_key)
        enabled = self.kb.start()
        atexit.register(self.kb.stop)

        # Logs
        self.get_logger().info("============================================================")
        self.get_logger().info("VRDemoHDF5Recorder initialized (multi-rate pose/force recorder)")
        self.get_logger().info(f"  Save root: {self.save_root}/YYYYMMDD_HHMM/{self.subdir_name}/")
        self.get_logger().info(f"  HDF5 name: vr_demo_merged_YYYYMMDD_HHMM{self.file_ext}")
        self.get_logger().info(f"  HDF5 path will be decided at first episode save time, not node startup time.")
        self.get_logger().info(f"  Topics: pose={self.pose_topic}, force={self.force_topic}")
        self.get_logger().info(f"  Expected rates: pose={self.pose_record_hz:.3f} Hz, force={self.force_record_hz:.3f} Hz")
        self.get_logger().info(f"  Merge axis: uniform pose axis dt={self.dt:.6f}s")
        self.get_logger().info("  HDF5 layout: episodes/ep_xxxx/traj only")
        self.get_logger().info(f"  Episode rule: start=|fx|>={self.start_abs_fx}, end=|fy|>={self.stop_abs_fy}")
        self.get_logger().info(
            f"  Force policy: zero_xy={self.zero_xy_forces}, fz_ema_alpha={self.fz_ema_alpha}, "
            f"edge_zero_sec={self.force_edge_zero_sec}, no clamp/contact-cleanup"
        )
        self.get_logger().info(
            f"  Contact for approach only: fz_gate_N={self.fz_gate_N}, consec_on={self.consec_on}, "
            f"approach_enable={self.approach_slowdown_enable}"
        )
        self.get_logger().info(
            f"  Pose pre: Hampel={self.hampel_enable}(win={self.hampel_win}, sig={self.hampel_sig}), "
            f"D2(lam_pos={self.lam_pos_d2}, lam_ang={self.lam_ang_d2}), PoseEMA={self.pose_ema_enable}(alpha={self.pose_ema_alpha})"
        )
        self.get_logger().info(
            f"  Post: D3(lam_pos={self.lam_pos_d3}, lam_ang={self.lam_ang_d3}), "
            f"QP-guard={self.qp_guard_enable}, safety={self.qp_guard_safety}"
        )
        if enabled:
            self.get_logger().info(f"  Press '{self.quit_key}' to stop (no Enter). Ctrl+C also works.")
        else:
            self.get_logger().warn("  stdin is not a TTY -> 'q' quit disabled. Use Ctrl+C instead.")
        self.get_logger().info("============================================================")

    # ============================================================
    # HDF5 helpers
    # ============================================================
    def _make_stamp_from_save_time(self, save_time: Optional[float] = None) -> str:
        if save_time is None:
            save_time = time.time()
        return time.strftime("%Y%m%d_%H%M", time.localtime(save_time))

    def _ensure_hdf5_open(self, save_time: Optional[float] = None):
        with self.h5_lock:
            if self.h5 is not None:
                return

            self.run_stamp = self._make_stamp_from_save_time(save_time)
            self.save_dir = os.path.join(self.save_root, self.run_stamp, self.subdir_name)
            hdf5_name = f"vr_demo_merged_{self.run_stamp}{self.file_ext}"
            self.hdf5_path = os.path.join(self.save_dir, hdf5_name)

            os.makedirs(self.save_dir, exist_ok=True)
            if self.overwrite and os.path.exists(self.hdf5_path):
                os.remove(self.hdf5_path)

            self.h5 = h5py.File(self.hdf5_path, "a")
            self.grp_eps = self.h5.require_group("episodes")
            self.episode_count = self._detect_existing_episode_count_unlocked()
            self._write_root_meta_unlocked()

            self.get_logger().info(f"[HDF5] Opened: {self.hdf5_path}")
            self.get_logger().info(f"[HDF5] Existing episode_count={self.episode_count}")

    def _detect_existing_episode_count_unlocked(self) -> int:
        if self.grp_eps is None:
            return 0
        max_idx = -1
        for k in self.grp_eps.keys():
            if k.startswith("ep_"):
                try:
                    idx = int(k.split("_")[1])
                    max_idx = max(max_idx, idx)
                except Exception:
                    pass
        return max_idx + 1

    def _write_root_meta_unlocked(self):
        assert self.h5 is not None
        if "created_unix" not in self.h5.attrs:
            self.h5.attrs["created_unix"] = float(time.time())
        self.h5.attrs["save_time_basis"] = h5str("first episode save time")
        self.h5.attrs["columns"] = h5str("x_mm,y_mm,z_mm,wx,wy,wz,fx,fy,fz")
        self.h5.attrs["note_pose"] = h5str("pose xyz input meters -> stored millimeters; orientation vector stored as received")
        self.h5.attrs["note_force"] = h5str("traj force: fx/fy zeroed, fz EMA, edge-zeroed, then interpolated to pose axis")
        self.h5.attrs["pose_topic"] = h5str(self.pose_topic)
        self.h5.attrs["force_topic"] = h5str(self.force_topic)
        self.h5.attrs["expected_pose_hz"] = float(self.pose_record_hz)
        self.h5.attrs["expected_force_hz"] = float(self.force_record_hz)
        self.h5.attrs["merged_traj_dt"] = float(self.dt)
        self.h5.attrs["episode_rule"] = h5str(f"start=|fx|>={self.start_abs_fx}, end=|fy|>={self.stop_abs_fy}")
        self.h5.attrs["hdf5_layout"] = h5str("episodes/ep_xxxx/traj")
        self.h5.flush()

    # ============================================================
    # Stop control
    # ============================================================
    def request_stop(self, reason: str = "user_request"):
        self.stop_requested = True
        self.stop_reason = str(reason)
        self.get_logger().warn(f"[STOP REQUEST] reason={self.stop_reason}")

    def _check_stop(self):
        if self.kb.hit() and (not self.stop_requested):
            self.request_stop(reason=f"keyboard_{self.quit_key}")

        if self.stop_requested and self.episode_active and (not self.finishing_):
            self.get_logger().warn("Stop requested while recording -> closing current episode.")
            self._start_finish_thread(reason=self.stop_reason or "stop_requested")
            return

        if self.stop_requested and (not self.finishing_) and (not self.episode_active):
            self.finalize_and_shutdown()

    def finalize_and_shutdown(self):
        self.get_logger().warn("Finalizing HDF5 and shutting down...")
        try:
            with self.h5_lock:
                if self.h5 is not None:
                    try:
                        self.h5.flush()
                    except Exception:
                        pass
                    try:
                        self.h5.close()
                    except Exception:
                        pass
                    self.h5 = None
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
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

    # ============================================================
    # ROS callbacks: record each stream at its own callback rate
    # ============================================================
    def cb_pose(self, msg: Float64MultiArray):
        if len(msg.data) < 6:
            return

        now = time.time()
        x, y, z, wx, wy, wz = msg.data[:6]
        pose = np.array([1000.0 * x, 1000.0 * y, 1000.0 * z, wx, wy, wz], dtype=np.float64)

        with self.state_lock:
            self.latest_pose6_mm_rad = pose
            self.latest_pose_t_abs = now

            if self.episode_active and (not self.finishing_) and (self.episode_start_t_abs is not None):
                self.buf_pose_t.append(now - self.episode_start_t_abs)
                self.buf_pose.append(pose.copy())

    def cb_force(self, msg: Wrench):
        now = time.time()
        fx = float(msg.force.x)
        fy = float(msg.force.y)
        fz = float(msg.force.z)
        F = np.array([fx, fy, fz], dtype=np.float64)

        start_trigger = False
        stop_trigger = False
        ep_idx_for_log = self.episode_count

        with self.state_lock:
            self.latest_force3_N = F
            self.latest_force_t_abs = now

            if self.stop_requested or self.finishing_:
                return

            if (not self.episode_active) and (abs(fx) >= self.start_abs_fx):
                self.episode_active = True
                self.episode_start_t_abs = now
                self.buf_pose_t.clear()
                self.buf_pose.clear()
                self.buf_force_t.clear()
                self.buf_force.clear()

                # Append the start force sample at t=0.
                self.buf_force_t.append(0.0)
                self.buf_force.append(F.copy())

                # If a fresh pose is already available, append it as the first pose sample.
                if self.latest_pose6_mm_rad is not None and (now - self.latest_pose_t_abs) <= self.require_fresh_sec:
                    self.buf_pose_t.append(0.0)
                    self.buf_pose.append(self.latest_pose6_mm_rad.copy())

                start_trigger = True

            elif self.episode_active and (self.episode_start_t_abs is not None):
                self.buf_force_t.append(now - self.episode_start_t_abs)
                self.buf_force.append(F.copy())
                if abs(fy) >= self.stop_abs_fy:
                    stop_trigger = True

        if start_trigger:
            self.get_logger().info(f"=== EPISODE STARTED (idx={ep_idx_for_log:04d}, |fx| >= start_abs_fx) ===")
            return

        if stop_trigger:
            self.get_logger().info(f"=== EPISODE ENDED (idx={ep_idx_for_log:04d}, |fy| >= stop_abs_fy) ===")
            self._start_finish_thread(reason="fy_threshold")

    # ============================================================
    # Pipeline blocks
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
        cidx = detect_contact_idx(fz, self.fz_gate_N, self.consec_on)
        if cidx is None:
            self.get_logger().warn("[APPROACH] contact not found -> skip approach slow-down")
            return Pr, Fr

        preN = int(round(self.approach_pre_sec * self.pose_record_hz))
        postN = int(round(self.approach_post_sec * self.pose_record_hz))

        N = Pr.shape[0]
        seg_scale = np.ones(N - 1, dtype=np.float64)

        s0 = max(0, cidx - preN)
        s1 = min(N - 1, cidx + postN)
        if s1 <= s0 + 2:
            self.get_logger().warn("[APPROACH] window too small -> skip")
            return Pr, Fr

        idx = np.arange(s0, s1, dtype=np.float64)
        u = (idx - float(s0)) / max(1.0, float(s1 - s0))

        # The txt recorder currently uses cosine profile.
        bump = 0.5 - 0.5 * np.cos(2.0 * np.pi * u)
        bump = np.clip(bump, 0.0, 1.0)
        scale_target = 1.0 + (self.approach_scale_max - 1.0) * bump

        if self.approach_use_fz_ramp:
            fz_win = np.abs(fz[s0:s1])
            ramp = np.clip(fz_win / max(1e-6, self.approach_fz_full), 0.0, 1.0)
            scale_target = 1.0 + (scale_target - 1.0) * ramp

        seg_scale[s0:s1] = np.maximum(seg_scale[s0:s1], scale_target)

        Pn, Fn = resample_uniform_by_timewarp(Pr, Fr, self.dt, seg_scale)
        self.get_logger().info(
            f"[APPROACH] contact idx={cidx} (t={cidx*self.dt:.3f}s), "
            f"slow window [{s0},{s1}] -> rows {Pr.shape[0]} -> {Pn.shape[0]}"
        )
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

        for it in range(max(1, self.qp_guard_max_iter)):
            Pk = self._pose_post_smooth_d3(Pref, lam_p, lam_a)

            dpos = norm_rows(Pk[:, :3] - Pref[:, :3])
            dang = norm_rows(Pk[:, 3:] - Pref[:, 3:])
            if float(dpos.max()) > self.max_dev_pos_mm or float(dang.max()) > self.max_dev_ang_rad:
                self.get_logger().warn(
                    f"[QP-GUARD] stop by deviation: max_dpos={float(dpos.max()):.3f}mm (allow {self.max_dev_pos_mm}), "
                    f"max_dang={float(dang.max()):.4f}rad (allow {self.max_dev_ang_rad})"
                )
                break

            st, _ = eval_qp_proxy(Pk, self.dt, self.lim, safety=self.qp_guard_safety)
            print_eval(
                self.get_logger(),
                f"QP-GUARD iter={it} (lam_p={lam_p:.3e}, lam_a={lam_a:.3e}, safety={self.qp_guard_safety})",
                st,
                self.lim,
                self.qp_guard_safety,
            )

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
                self.get_logger().info("[QP-GUARD] constraints satisfied.")
                return Pk

            lam_p *= self.qp_guard_growth
            lam_a *= self.qp_guard_growth

        self.get_logger().warn("[QP-GUARD] could not fully satisfy constraints. Returning best smoothed.")
        return best if best is not None else self._pose_post_smooth_d3(Pref, self.lam_pos_d3, self.lam_ang_d3)

    def _make_merged_uniform_inputs(
        self,
        raw_pose_t: np.ndarray,
        rawP: np.ndarray,
        raw_force_t: np.ndarray,
        rawF: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          pose_axis_t: uniform pose time axis [s]
          P_uniform: raw pose resampled to pose axis
          F_uniform_processed: processed force resampled to pose axis
          F_processed_raw_axis: processed force at raw force axis
          force_t_sanitized: sanitized force time axis
        """
        pose_t, rawP = sanitize_time_series(raw_pose_t, rawP)
        force_t, rawF = sanitize_time_series(raw_force_t, rawF)

        # Use the overlapping interval between pose and force streams.
        t0 = max(float(pose_t[0]), float(force_t[0]))
        t1 = min(float(pose_t[-1]), float(force_t[-1]))
        if t1 <= t0:
            raise RuntimeError(f"No overlapping time range between pose and force. pose=[{pose_t[0]:.3f},{pose_t[-1]:.3f}], force=[{force_t[0]:.3f},{force_t[-1]:.3f}]")

        pose_mask = (pose_t >= t0) & (pose_t <= t1)
        force_mask = (force_t >= t0) & (force_t <= t1)
        pose_t_crop = pose_t[pose_mask] - t0
        rawP_crop = rawP[pose_mask]
        force_t_crop = force_t[force_mask] - t0
        rawF_crop = rawF[force_mask]

        if pose_t_crop.size < self.min_pose_samples:
            raise RuntimeError(f"Too few pose samples after overlap crop: {pose_t_crop.size}")
        if force_t_crop.size < self.min_force_samples:
            raise RuntimeError(f"Too few force samples after overlap crop: {force_t_crop.size}")

        pose_axis_t, P_uniform = resample_to_uniform_time(pose_t_crop, rawP_crop, self.dt)

        F_processed_raw_axis = process_force_keep_fz_with_ema_and_edge_zero(
            rawF_crop,
            traw=force_t_crop,
            fz_ema_alpha=self.fz_ema_alpha,
            edge_zero_sec=self.force_edge_zero_sec,
            zero_xy=self.zero_xy_forces,
            logger=self.get_logger(),
        )

        F_uniform_processed = interpolate_to_time_axis(force_t_crop, F_processed_raw_axis, pose_axis_t)
        return pose_axis_t, P_uniform, F_uniform_processed, F_processed_raw_axis, force_t_crop

    # ============================================================
    # Finish episode (threaded)
    # ============================================================
    def _start_finish_thread(self, reason: str):
        if self.finishing_:
            return
        self.finishing_ = True

        with self.state_lock:
            self.episode_active = False
            pose_t_list = self.buf_pose_t.copy()
            P_list = self.buf_pose.copy()
            force_t_list = self.buf_force_t.copy()
            F_list = self.buf_force.copy()
            self.buf_pose_t.clear()
            self.buf_pose.clear()
            self.buf_force_t.clear()
            self.buf_force.clear()
            self.episode_start_t_abs = None

        th = threading.Thread(
            target=self._finish_episode_worker,
            args=(pose_t_list, P_list, force_t_list, F_list, reason),
            daemon=True,
        )
        th.start()

    def _finish_episode_worker(
        self,
        pose_t_list: List[float],
        P_list: List[np.ndarray],
        force_t_list: List[float],
        F_list: List[np.ndarray],
        reason: str,
    ):
        try:
            if len(P_list) < max(1, self.min_pose_samples):
                self.get_logger().warn(
                    f"Episode dropped: pose samples too short: {len(P_list)} < {self.min_pose_samples}, reason={reason}"
                )
                return
            if len(F_list) < max(1, self.min_force_samples):
                self.get_logger().warn(
                    f"Episode dropped: force samples too short: {len(F_list)} < {self.min_force_samples}, reason={reason}"
                )
                return

            raw_pose_t = np.asarray(pose_t_list, dtype=np.float64)
            raw_force_t = np.asarray(force_t_list, dtype=np.float64)
            rawP = np.asarray(P_list, dtype=np.float64)   # (Np,6) [mm, rad]
            rawF = np.asarray(F_list, dtype=np.float64)   # (Nf,3) [N]

            pose_hz_est = estimate_hz_from_time(raw_pose_t)
            force_hz_est = estimate_hz_from_time(raw_force_t)
            self.get_logger().info(
                f"[RAW] pose_len={rawP.shape[0]}, force_len={rawF.shape[0]}, "
                f"pose_hz_est={pose_hz_est:.2f}, force_hz_est={force_hz_est:.2f}"
            )

            pose_axis_t, P_uniform, F_uniform_processed, F_processed_raw_axis, force_t_crop = self._make_merged_uniform_inputs(
                raw_pose_t, rawP, raw_force_t, rawF
            )
            self.get_logger().info(
                f"[MERGE] pose_axis_len={P_uniform.shape[0]}, force_raw_processed_len={F_processed_raw_axis.shape[0]}, "
                f"merged_dt={self.dt:.6f}s"
            )

            st0, _ = eval_qp_proxy(P_uniform, self.dt, self.lim, safety=1.0)
            print_eval(self.get_logger(), "RAW POSE UNIFORM (before)", st0, self.lim, 1.0)

            # 1) Pose pre smooth
            Ps = self._pose_pre_smooth(P_uniform)

            # 2) Retime fixed x2
            Pr, Fr = self._retime_x2(Ps, F_uniform_processed)
            self.get_logger().info(f"[RETIME] x2 applied: rows {Ps.shape[0]} -> {Pr.shape[0]}")

            # 3) Optional approach slow-down. Default is False, same as uploaded txt recorder.
            Pr_slow, Fr_slow = self._apply_contact_approach_slowdown(Pr, Fr)

            # 4) Final pose smoothing + QP-guard
            Pf = self._qp_guard(Pr_slow)

            st2, _ = eval_qp_proxy(Pf, self.dt, self.lim, safety=self.qp_guard_safety)
            print_eval(self.get_logger(), "FINAL pose (retime x2 + optional approach + D3)", st2, self.lim, self.qp_guard_safety)

            out = np.hstack([Pf, Fr_slow]).astype(np.float32)  # (M,9)
            traj_time = np.arange(out.shape[0], dtype=np.float64) * self.dt

            used = {
                "merge_mode": "pose_uniform_axis_force_interpolated",
                "pose_record_hz": float(self.pose_record_hz),
                "force_record_hz": float(self.force_record_hz),
                "pose_hz_est": float(pose_hz_est),
                "force_hz_est": float(force_hz_est),
                "fz_ema_alpha": float(self.fz_ema_alpha),
                "force_edge_zero_sec": float(self.force_edge_zero_sec),
                "zero_xy_forces": bool(self.zero_xy_forces),
                "fz_gate_N": float(self.fz_gate_N),
                "consec_on": int(self.consec_on),
                "retime_k": int(self.retime_k),
                "approach_slowdown_enable": bool(self.approach_slowdown_enable),
            }

            save_time = time.time()
            self._ensure_hdf5_open(save_time=save_time)
            ep_idx = self.episode_count
            self._save_episode_to_hdf5(
                ep_idx=ep_idx,
                out=out,
                traj_time=traj_time,
                reason=reason,
                raw_pose_t=raw_pose_t,
                rawP=rawP,
                raw_force_t=raw_force_t,
                rawF=rawF,
                pose_axis_t=pose_axis_t,
                P_uniform=P_uniform,
                F_uniform_processed=F_uniform_processed,
                force_t_processed=force_t_crop,
                F_processed_raw_axis=F_processed_raw_axis,
                used_meta=used,
            )
            self.episode_count += 1

            self.get_logger().info(
                f"=== EPISODE SAVED (idx={ep_idx:04d}) "
                f"raw_pose_len={rawP.shape[0]}, raw_force_len={rawF.shape[0]} -> out_len={out.shape[0]} "
                f"reason={reason} ==="
            )
            self.get_logger().info(f"[HDF5] {self.hdf5_path}")

            if self.episode_count >= self.num_episodes:
                self.request_stop(reason="reached_num_episodes")

        except Exception as e:
            self.get_logger().error(f"Episode processing failed: {e}")
        finally:
            self.finishing_ = False

    # ============================================================
    # HDF5 save
    # ============================================================
    def _save_episode_to_hdf5(
        self,
        ep_idx: int,
        out: np.ndarray,
        traj_time: np.ndarray,
        reason: str,
        raw_pose_t: np.ndarray,
        rawP: np.ndarray,
        raw_force_t: np.ndarray,
        rawF: np.ndarray,
        pose_axis_t: np.ndarray,
        P_uniform: np.ndarray,
        F_uniform_processed: np.ndarray,
        force_t_processed: np.ndarray,
        F_processed_raw_axis: np.ndarray,
        used_meta: Dict[str, object],
    ):
        ep_name = f"ep_{ep_idx:04d}"
        with self.h5_lock:
            assert self.h5 is not None
            assert self.grp_eps is not None

            if ep_name in self.grp_eps:
                del self.grp_eps[ep_name]
            g = self.grp_eps.create_group(ep_name)

            g.attrs["saved_unix"] = float(time.time())
            g.attrs["saved_local_stamp"] = h5str(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
            g.attrs["reason"] = h5str(str(reason))
            g.attrs["dtype"] = h5str(str(out.dtype))
            g.attrs["columns"] = h5str("x_mm,y_mm,z_mm,wx,wy,wz,fx,fy,fz")

            g.attrs["raw_pose_len"] = int(rawP.shape[0])
            g.attrs["raw_force_len"] = int(rawF.shape[0])
            g.attrs["pose_uniform_len"] = int(P_uniform.shape[0])
            g.attrs["out_len"] = int(out.shape[0])
            g.attrs["merged_traj_dt"] = float(self.dt)
            g.attrs["pose_hz_est"] = float(estimate_hz_from_time(raw_pose_t))
            g.attrs["force_hz_est"] = float(estimate_hz_from_time(raw_force_t))

            # Store key params for reproducibility.
            g.attrs["zero_xy_forces"] = int(bool(self.zero_xy_forces))
            g.attrs["fz_ema_alpha"] = float(self.fz_ema_alpha)
            g.attrs["force_edge_zero_sec"] = float(self.force_edge_zero_sec)
            g.attrs["fz_gate_N"] = float(self.fz_gate_N)
            g.attrs["consec_on"] = int(self.consec_on)

            g.attrs["pose_hampel_enable"] = int(bool(self.hampel_enable))
            g.attrs["pose_hampel_win"] = int(self.hampel_win)
            g.attrs["pose_hampel_sig"] = float(self.hampel_sig)
            g.attrs["lam_pos_d2"] = float(self.lam_pos_d2)
            g.attrs["lam_ang_d2"] = float(self.lam_ang_d2)
            g.attrs["pose_ema_enable"] = int(bool(self.pose_ema_enable))
            g.attrs["pose_ema_alpha"] = float(self.pose_ema_alpha)

            g.attrs["retime_k"] = int(self.retime_k)
            g.attrs["approach_slowdown_enable"] = int(bool(self.approach_slowdown_enable))
            g.attrs["approach_pre_sec"] = float(self.approach_pre_sec)
            g.attrs["approach_post_sec"] = float(self.approach_post_sec)
            g.attrs["approach_scale_max"] = float(self.approach_scale_max)
            g.attrs["approach_use_fz_ramp"] = int(bool(self.approach_use_fz_ramp))
            g.attrs["approach_fz_full"] = float(self.approach_fz_full)

            g.attrs["post_enable"] = int(bool(self.post_enable))
            g.attrs["lam_pos_d3"] = float(self.lam_pos_d3)
            g.attrs["lam_ang_d3"] = float(self.lam_ang_d3)

            g.attrs["qp_guard_enable"] = int(bool(self.qp_guard_enable))
            g.attrs["qp_guard_safety"] = float(self.qp_guard_safety)
            g.attrs["qp_guard_max_iter"] = int(self.qp_guard_max_iter)
            g.attrs["qp_guard_growth"] = float(self.qp_guard_growth)
            g.attrs["max_dev_pos_mm"] = float(self.max_dev_pos_mm)
            g.attrs["max_dev_ang_rad"] = float(self.max_dev_ang_rad)

            g.attrs["used_meta_json"] = h5str(json.dumps(used_meta))

            # Main IL trajectory only.
            # Keep the same storage layout as the original HDF5 recorder:
            #   episodes/ep_xxxx/traj
            # Raw multi-rate streams and intermediate arrays are used internally
            # for merging, but are not written to the HDF5 file.
            g.create_dataset("traj", data=out, compression="gzip", compression_opts=4, shuffle=True)

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