#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
filter_cmd_continue9D.py

[동작]
- input: cmd_continue9D.txt (x y z [mm], wx wy wz [rad], fx fy fz [N]) 각 행 9개 실수
- 실행하면:
  (1) raw를 non_filtered_cmd_continue9D.txt 로 백업
  (2) VrDemoTxtRecorder.finish_episode() 후처리 파이프라인 동일 적용
      - force_process
      - pose_smooth (Hampel + WhittakerAuto + optional EMA)
      - retime_uniform (조건부 upsample_linear)
      - precontact_gating
      - edge_force_window
  (3) 결과를 cmd_continue9D.txt 에 덮어쓰기 저장
  (4) viz_YYYYMMDD_HHMMSS 폴더 생성 후, png 3장만 저장:
      1) plot_1_lin_kinematics.png : x y z / vx vy vz / ax ay az / jx jy jz
      2) plot_2_ang_kinematics.png : wx wy wz / wdot / wddot / wdddot
      3) plot_3_forces.png         : fx fy fz

[시각화(time-align)]
- 저장값(Pr/Fr)은 그대로 두고,
- plot 단계에서만 before/after를 "같은 time grid"로 보간해서 겹쳐 그림.
- 보간 축은 항상 seconds 기준으로 통일:
    raw: 0..T_raw  -> 0..T_common 로 시간 스케일만 늘려서 보간
    after: 0..T_after -> 0..T_common 로 보간
- 이렇게 하면 before/after 시간축이 완전히 동일하게 맞음.
"""

import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Dict

import numpy as np

from nrs_imitation.pretty_print import block, status

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Utility
# ----------------------------
def pctl(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, q))


def norm_rows(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=1)


def finite_diff_pad(y: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    y: (N,) signal
    returns v,a,j each shaped (N,) with NaN padding so plots align on same index/time.
      v[0]=nan, v[1:]=diff(y)/dt
      a[:2]=nan, a[2:]=diff(v)/dt
      j[:3]=nan, j[3:]=diff(a)/dt
    """
    y = y.astype(np.float64).reshape(-1)
    N = y.size
    v = np.full(N, np.nan, dtype=np.float64)
    a = np.full(N, np.nan, dtype=np.float64)
    j = np.full(N, np.nan, dtype=np.float64)
    if N >= 2:
        v[1:] = (y[1:] - y[:-1]) / dt
    if N >= 3:
        a[2:] = (v[2:] - v[1:-1]) / dt
    if N >= 4:
        j[3:] = (a[3:] - a[2:-1]) / dt
    return v, a, j


def make_common_time_and_align(raw_y: np.ndarray, after_y: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Plot용 time-align 전용.
    - raw_y: (N0,)
    - after_y: (N1,)
    - dt: 원래 sampling dt (초)

    리타임 후 after는 샘플 수가 늘어나면서 총 duration이 길어짐.
    여기서는 before/after를 동일 time축(0..T_common)에 겹쳐 그리기 위해,
    둘 다 0..T_common로 '시간 스케일만' 맞춘 뒤 공통 grid(t_common)로 보간한다.

    반환:
      t_common, raw_aligned, after_aligned, dt_plot
    """
    raw_y = raw_y.reshape(-1).astype(np.float64)
    after_y = after_y.reshape(-1).astype(np.float64)
    N0 = raw_y.size
    N1 = after_y.size
    if N0 < 2 or N1 < 2:
        # 너무 짧으면 그냥 그대로
        T_common = dt * max(0, max(N0, N1) - 1)
        t_common = np.arange(max(N0, N1), dtype=np.float64) * dt
        # 길이 맞추기: 가능한 경우만
        raw_pad = np.full_like(t_common, np.nan)
        aft_pad = np.full_like(t_common, np.nan)
        raw_pad[:N0] = raw_y
        aft_pad[:N1] = after_y
        dt_plot = dt if t_common.size < 2 else float(t_common[1] - t_common[0])
        return t_common, raw_pad, aft_pad, dt_plot

    # durations
    T0 = dt * (N0 - 1)
    T1 = dt * (N1 - 1)
    T_common = max(T0, T1)

    # common grid length: 더 촘촘한 쪽(보통 after)이 정보 손실이 적음
    N_common = max(N0, N1)
    if N_common < 2:
        N_common = 2

    t_common = np.linspace(0.0, T_common, N_common, dtype=np.float64)

    # 각 신호의 "스케일된 시간축" (0..T_common)
    t0_scaled = np.linspace(0.0, T_common, N0, dtype=np.float64)
    t1_scaled = np.linspace(0.0, T_common, N1, dtype=np.float64)

    raw_aligned = np.interp(t_common, t0_scaled, raw_y)
    aft_aligned = np.interp(t_common, t1_scaled, after_y)

    dt_plot = float(t_common[1] - t_common[0])
    return t_common, raw_aligned, aft_aligned, dt_plot


# ----------------------------
# Hampel filter (per-dim)
# ----------------------------
def hampel_1d(x: np.ndarray, win: int, n_sigmas: float) -> np.ndarray:
    if win <= 0:
        return x.copy()
    n = x.size
    y = x.copy()
    k = 1.4826  # MAD -> std approx
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
# Whittaker smoother via CG
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
# QP-proxy eval
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
    vpos_mean: float
    apos_mean: float
    vang_mean: float
    aang_mean: float
    viol_v: float
    viol_a: float
    viol_w: float
    viol_alpha: float


def eval_qp_proxy(pose6: np.ndarray, dt: float, lim: Limits, safety: float = 1.0) -> Tuple[EvalStats, Dict[str, np.ndarray]]:
    N = int(pose6.shape[0])
    T = dt * max(0, (N - 1))

    dp = pose6[1:, :3] - pose6[:-1, :3]
    dw = pose6[1:, 3:] - pose6[:-1, 3:]
    vpos = norm_rows(dp) / dt
    vang = norm_rows(dw) / dt

    v = (pose6[1:, :] - pose6[:-1, :]) / dt
    a = (v[1:, :] - v[:-1, :]) / dt
    apos = norm_rows(a[:, :3])
    aang = norm_rows(a[:, 3:])

    j = (a[1:, :] - a[:-1, :]) / dt
    jpos = norm_rows(j[:, :3])
    jang = norm_rows(j[:, 3:])

    vpos_max = float(vpos.max()) if vpos.size else 0.0
    vang_max = float(vang.max()) if vang.size else 0.0
    apos_max = float(apos.max()) if apos.size else 0.0
    aang_max = float(aang.max()) if aang.size else 0.0
    jpos_max = float(jpos.max()) if jpos.size else 0.0
    jang_max = float(jang.max()) if jang.size else 0.0

    vpos_lim = lim.pos_vmax * safety
    apos_lim = lim.pos_amax * safety
    vang_lim = lim.ang_vmax * safety
    aang_lim = lim.ang_amax * safety

    viol_v = float(np.mean(vpos > vpos_lim)) if vpos.size else 0.0
    viol_w = float(np.mean(vang > vang_lim)) if vang.size else 0.0
    viol_a = float(np.mean(apos > apos_lim)) if apos.size else 0.0
    viol_alpha = float(np.mean(aang > aang_lim)) if aang.size else 0.0

    st = EvalStats(
        N=N, dt=dt, T=T,
        vpos_max=vpos_max, apos_max=apos_max, vang_max=vang_max, aang_max=aang_max,
        jpos_max=jpos_max, jang_max=jang_max,
        vpos_p95=pctl(vpos, 95), apos_p95=pctl(apos, 95), vang_p95=pctl(vang, 95), aang_p95=pctl(aang, 95),
        jpos_p95=pctl(jpos, 95), jang_p95=pctl(jang, 95),
        vpos_mean=float(vpos.mean()) if vpos.size else 0.0,
        apos_mean=float(apos.mean()) if apos.size else 0.0,
        vang_mean=float(vang.mean()) if vang.size else 0.0,
        aang_mean=float(aang.mean()) if aang.size else 0.0,
        viol_v=viol_v, viol_a=viol_a, viol_w=viol_w, viol_alpha=viol_alpha
    )
    debug = {"vpos": vpos, "vang": vang, "apos": apos, "aang": aang, "jpos": jpos, "jang": jang}
    return st, debug


def print_eval(title: str, st: EvalStats, lim: Limits, safety: float):
    print(block(f"QP-EVAL {title}", [
        ("samples", f"N={st.N}, dt={st.dt:.6f}s, T={st.T:.3f}s"),
        ("pos |v|", f"max={st.vpos_max:.3f}, lim={lim.pos_vmax:.3f}, p95={st.vpos_p95:.3f}, mean={st.vpos_mean:.3f} mm/s"),
        ("pos |a|", f"max={st.apos_max:.3f}, lim={lim.pos_amax:.3f}, p95={st.apos_p95:.3f}, mean={st.apos_mean:.3f} mm/s^2"),
        ("ang |w|", f"max={st.vang_max:.3f}, lim={lim.ang_vmax:.3f}, p95={st.vang_p95:.3f}, mean={st.vang_mean:.3f} rad/s"),
        ("ang |alpha|", f"max={st.aang_max:.3f}, lim={lim.ang_amax:.3f}, p95={st.aang_p95:.3f}, mean={st.aang_mean:.3f} rad/s^2"),
        ("jerk(ref)", f"pos={st.jpos_max:.3f} mm/s^3, ang={st.jang_max:.3f} rad/s^3"),
        ("violations", f"safety={safety:.3f}, vpos={100*st.viol_v:.3f}%, apos={100*st.viol_a:.3f}%, vang={100*st.viol_w:.3f}%, aang={100*st.viol_alpha:.3f}%"),
    ], char="-"))


# ----------------------------
# Upsample + contact + edge window
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


def detect_contact_idx(fz: np.ndarray, fz_on: float, fz_off: float, consec_on: int, consec_off: int) -> Optional[int]:
    on = False
    cnt_on = 0
    first_on_idx = None
    for i in range(fz.size):
        if not on:
            if fz[i] >= fz_on:
                cnt_on += 1
                if cnt_on >= consec_on:
                    on = True
                    first_on_idx = i - consec_on + 1
                    break
            else:
                cnt_on = 0
        else:
            break
    return first_on_idx


def apply_edge_force_window(F: np.ndarray, hz: float, edge_force_zero_sec: float, edge_force_fade_sec: float) -> np.ndarray:
    out = F.copy()
    n = out.shape[0]
    zN = int(round(edge_force_zero_sec * hz))
    fN = int(round(edge_force_fade_sec * hz))
    zN = max(0, min(n, zN))
    fN = max(0, min(n, fN))

    if zN > 0:
        out[:zN, :] = 0.0
    if fN > 0 and (zN + fN) < n:
        w = np.linspace(0.0, 1.0, fN, dtype=np.float64).reshape(-1, 1)
        out[zN:zN + fN, :] = w * out[zN:zN + fN, :]

    if zN > 0:
        out[n - zN:, :] = 0.0
    if fN > 0 and (n - zN - fN) > 0:
        w = np.linspace(1.0, 0.0, fN, dtype=np.float64).reshape(-1, 1)
        out[n - zN - fN:n - zN, :] = w * out[n - zN - fN:n - zN, :]

    return out


# ----------------------------
# Pipeline blocks
# ----------------------------
def force_process(F: np.ndarray, zero_xy_forces: bool, force_clamp_abs: float, force_ema_alpha: float) -> np.ndarray:
    Fp = F.copy()
    Fp = np.clip(Fp, -force_clamp_abs, force_clamp_abs)
    if zero_xy_forces:
        Fp[:, 0] = 0.0
        Fp[:, 1] = 0.0
    if 0.0 < force_ema_alpha < 1.0:
        Fp = ema_nd(Fp, alpha=force_ema_alpha)
    return Fp


def pose_smooth(
    P: np.ndarray,
    dt: float,
    lim: Limits,
    safety: float,
    hampel_enable: bool,
    hampel_win: int,
    hampel_sig: float,
    whittaker_auto: bool,
    lam_pos_init: float,
    lam_ang_init: float,
    lam_growth: float,
    lam_iters: int,
    cg_iters: int,
    cg_tol: float,
    pose_ema_enable: bool,
    pose_ema_alpha: float
) -> Tuple[np.ndarray, Dict[str, float]]:
    P0 = P.copy()

    if hampel_enable:
        P0 = hampel_nd(P0, win=hampel_win, n_sigmas=hampel_sig)

    if not whittaker_auto:
        Pp = P0.copy()
        Pp[:, :3] = whittaker_cg_nd(Pp[:, :3], lam=lam_pos_init, cg_iters=cg_iters, tol=cg_tol)
        Pp[:, 3:] = whittaker_cg_nd(Pp[:, 3:], lam=lam_ang_init, cg_iters=cg_iters, tol=cg_tol)
        if pose_ema_enable:
            Pp = ema_nd(Pp, alpha=pose_ema_alpha)
        return Pp, {"lam_pos": lam_pos_init, "lam_ang": lam_ang_init}

    lam_pos = lam_pos_init
    lam_ang = lam_ang_init

    best = None
    best_score = 1e18
    best_info = {"lam_pos": lam_pos, "lam_ang": lam_ang}

    max_pos_delta_allow = 5.0
    max_ang_delta_allow = 0.03

    for _ in range(max(1, lam_iters)):
        Pp = P0.copy()
        Pp[:, :3] = whittaker_cg_nd(Pp[:, :3], lam=lam_pos, cg_iters=cg_iters, tol=cg_tol)
        Pp[:, 3:] = whittaker_cg_nd(Pp[:, 3:], lam=lam_ang, cg_iters=cg_iters, tol=cg_tol)

        if pose_ema_enable:
            Pp = ema_nd(Pp, alpha=pose_ema_alpha)

        dpos = norm_rows(Pp[:, :3] - P[:, :3])
        dang = norm_rows(Pp[:, 3:] - P[:, 3:])
        if float(dpos.max()) > max_pos_delta_allow or float(dang.max()) > max_ang_delta_allow:
            break

        st, _ = eval_qp_proxy(Pp, dt, lim, safety=safety)
        score = max(st.apos_p95 / (lim.pos_amax + 1e-9),
                    st.aang_p95 / (lim.ang_amax + 1e-9),
                    st.jpos_p95 / (lim.pos_jmax + 1e-9),
                    st.jang_p95 / (lim.ang_jmax + 1e-9)) \
                + 0.05 * (float(dpos.mean()) / 1.0)

        if score < best_score:
            best_score = score
            best = Pp
            best_info = {"lam_pos": lam_pos, "lam_ang": lam_ang}

        lam_pos *= lam_growth
        lam_ang *= lam_growth

    if best is None:
        best = P0
    return best, best_info


def retime_uniform(
    P: np.ndarray, F: np.ndarray,
    dt: float, lim: Limits, safety: float,
    retime_enable: bool, retime_use_jerk: bool,
    retime_max_k: int, retime_passes: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    if not retime_enable:
        return P, F, 1

    Pcur = P.copy()
    Fcur = F.copy()
    k_total = 1

    for _ in range(max(1, retime_passes)):
        st, _ = eval_qp_proxy(Pcur, dt, lim, safety=safety)

        rv = max(
            st.vpos_max / (lim.pos_vmax * safety + 1e-9),
            st.vang_max / (lim.ang_vmax * safety + 1e-9),
        )
        ra = max(
            math.sqrt(st.apos_max / (lim.pos_amax * safety + 1e-9)),
            math.sqrt(st.aang_max / (lim.ang_amax * safety + 1e-9)),
        )

        rj = 1.0
        if retime_use_jerk:
            rj = max(
                (st.jpos_max / (lim.pos_jmax * safety + 1e-9)) ** (1.0 / 3.0),
                (st.jang_max / (lim.ang_jmax * safety + 1e-9)) ** (1.0 / 3.0),
            )

        r_need = max(1.0, rv, ra, rj)
        k_need = int(math.ceil(r_need))

        remaining = max(1, retime_max_k // max(1, k_total))
        k_need = min(k_need, remaining)

        if k_need <= 1:
            break

        Pcur = upsample_linear(Pcur, k_need)
        Fcur = upsample_linear(Fcur, k_need)
        k_total *= k_need

    return Pcur, Fcur, k_total


# ----------------------------
# TXT I/O
# ----------------------------
def read_txt9(path: str) -> np.ndarray:
    rows = []
    with open(path, "r") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 9:
                raise RuntimeError(f"Line {ln}: need >=9 floats, got={len(parts)} -> '{s}'")
            rows.append(list(map(float, parts[:9])))
    data = np.asarray(rows, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 9 or data.shape[0] == 0:
        raise RuntimeError(f"Invalid data shape: {data.shape}")
    return data


def write_txt9_tab6(path: str, out: np.ndarray):
    with open(path, "w") as f:
        for row in out:
            f.write("\t".join([f"{v:.6f}" for v in row.tolist()]) + "\n")


# ----------------------------
# Plot helpers (3 figures only, time-aligned)
# ----------------------------
def plot_before_after(ax, t, y0, y1, title, ylabel):
    ax.plot(t, y0, label="before")
    ax.plot(t, y1, label="after")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True)


def save_plot_1_lin_kinematics(viz_dir: str, dt: float, rawP: np.ndarray, filtP: np.ndarray):
    fig = plt.figure(figsize=(16, 12))
    names = ["x", "y", "z"]
    units = ["mm", "mm", "mm"]

    for c in range(3):
        t, y_raw, y_fil, dtp = make_common_time_and_align(rawP[:, c], filtP[:, c], dt)
        v_raw, a_raw, j_raw = finite_diff_pad(y_raw, dtp)
        v_fil, a_fil, j_fil = finite_diff_pad(y_fil, dtp)

        ax = plt.subplot(4, 3, 1 + c)
        plot_before_after(ax, t, y_raw, y_fil, f"{names[c]}", units[c])
        if c == 0:
            ax.legend()

        ax = plt.subplot(4, 3, 4 + c)
        plot_before_after(ax, t, v_raw, v_fil, f"v{names[c]}", f"{units[c]}/s")

        ax = plt.subplot(4, 3, 7 + c)
        plot_before_after(ax, t, a_raw, a_fil, f"a{names[c]}", f"{units[c]}/s^2")

        ax = plt.subplot(4, 3, 10 + c)
        plot_before_after(ax, t, j_raw, j_fil, f"j{names[c]}", f"{units[c]}/s^3")
        ax.set_xlabel("time [s]")

    fig.suptitle("Linear kinematics: pos / vel / acc / jerk (before vs after, time-aligned)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    outpath = os.path.join(viz_dir, "plot_1_lin_kinematics.png")
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def save_plot_2_ang_kinematics(viz_dir: str, dt: float, rawP: np.ndarray, filtP: np.ndarray):
    fig = plt.figure(figsize=(16, 12))
    names = ["wx", "wy", "wz"]
    units = ["rad", "rad", "rad"]

    for c in range(3):
        t, y_raw, y_fil, dtp = make_common_time_and_align(rawP[:, 3 + c], filtP[:, 3 + c], dt)
        v_raw, a_raw, j_raw = finite_diff_pad(y_raw, dtp)
        v_fil, a_fil, j_fil = finite_diff_pad(y_fil, dtp)

        ax = plt.subplot(4, 3, 1 + c)
        plot_before_after(ax, t, y_raw, y_fil, f"{names[c]}", units[c])
        if c == 0:
            ax.legend()

        ax = plt.subplot(4, 3, 4 + c)
        plot_before_after(ax, t, v_raw, v_fil, f"{names[c]}_dot", f"{units[c]}/s")

        ax = plt.subplot(4, 3, 7 + c)
        plot_before_after(ax, t, a_raw, a_fil, f"{names[c]}_dotdot", f"{units[c]}/s^2")

        ax = plt.subplot(4, 3, 10 + c)
        plot_before_after(ax, t, j_raw, j_fil, f"{names[c]}_dotdotdot", f"{units[c]}/s^3")
        ax.set_xlabel("time [s]")

    fig.suptitle("Angular kinematics: w / w_dot / w_ddot / w_dddot (before vs after, time-aligned)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    outpath = os.path.join(viz_dir, "plot_2_ang_kinematics.png")
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def save_plot_3_forces(viz_dir: str, dt: float, rawF: np.ndarray, filtF: np.ndarray):
    fig = plt.figure(figsize=(16, 4))
    names = ["fx", "fy", "fz"]

    for c in range(3):
        t, y_raw, y_fil, _dtp = make_common_time_and_align(rawF[:, c], filtF[:, c], dt)
        ax = plt.subplot(1, 3, 1 + c)
        plot_before_after(ax, t, y_raw, y_fil, f"{names[c]}", "N")
        ax.set_xlabel("time [s]")
        if c == 0:
            ax.legend()

    fig.suptitle("Forces: fx / fy / fz (before vs after, time-aligned)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    outpath = os.path.join(viz_dir, "plot_3_forces.png")
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dir", default=".", help="cmd_continue9D.txt 가 있는 디렉토리 (default: .)")
    ap.add_argument("--infile", default="cmd_continue9D.txt", help="입력 파일명 (default: cmd_continue9D.txt)")
    ap.add_argument("--backup", default="non_filtered_cmd_continue9D.txt", help="백업 파일명")

    ap.add_argument("--record_hz", type=float, default=125.0)

    # force shaping
    ap.add_argument("--zero_xy_forces", type=int, default=1)
    ap.add_argument("--force_clamp_abs", type=float, default=200.0)
    ap.add_argument("--force_ema_alpha", type=float, default=0.2)
    ap.add_argument("--edge_force_zero_sec", type=float, default=0.5)
    ap.add_argument("--edge_force_fade_sec", type=float, default=0.3)

    # precontact gating
    ap.add_argument("--precontact_gating", type=int, default=1)
    ap.add_argument("--fz_on", type=float, default=5.0)
    ap.add_argument("--fz_off", type=float, default=3.0)
    ap.add_argument("--consec_on", type=int, default=10)
    ap.add_argument("--consec_off", type=int, default=10)

    # pose smoothing
    ap.add_argument("--hampel_enable", type=int, default=1)
    ap.add_argument("--hampel_win", type=int, default=6)
    ap.add_argument("--hampel_sig", type=float, default=3.0)

    ap.add_argument("--whittaker_auto", type=int, default=1)
    ap.add_argument("--lam_pos_init", type=float, default=20000.0)
    ap.add_argument("--lam_ang_init", type=float, default=200.0)
    ap.add_argument("--lam_growth", type=float, default=3.0)
    ap.add_argument("--lam_iters", type=int, default=6)
    ap.add_argument("--cg_iters", type=int, default=200)
    ap.add_argument("--cg_tol", type=float, default=1e-8)

    ap.add_argument("--pose_ema_enable", type=int, default=0)
    ap.add_argument("--pose_ema_alpha", type=float, default=0.2)

    # QP-proxy limits + safety
    ap.add_argument("--pos_vmax", type=float, default=30.0)
    ap.add_argument("--pos_amax", type=float, default=120.0)
    ap.add_argument("--ang_vmax", type=float, default=0.6)
    ap.add_argument("--ang_amax", type=float, default=3.0)
    ap.add_argument("--pos_jmax", type=float, default=5000.0)
    ap.add_argument("--ang_jmax", type=float, default=80.0)
    ap.add_argument("--safety", type=float, default=1.05)

    # retime
    ap.add_argument("--retime_enable", type=int, default=1)
    ap.add_argument("--retime_use_jerk", type=int, default=1)
    ap.add_argument("--retime_max_k", type=int, default=20)
    ap.add_argument("--retime_passes", type=int, default=3)

    args = ap.parse_args()

    base_dir = os.path.abspath(args.dir)
    in_path = os.path.join(base_dir, args.infile)
    backup_path = os.path.join(base_dir, args.backup)

    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input not found: {in_path}")

    dt = 1.0 / max(1e-9, args.record_hz)
    lim = Limits(
        pos_vmax=args.pos_vmax,
        pos_amax=args.pos_amax,
        ang_vmax=args.ang_vmax,
        ang_amax=args.ang_amax,
        pos_jmax=args.pos_jmax,
        ang_jmax=args.ang_jmax,
    )

    # (1) read raw
    raw9 = read_txt9(in_path)
    rawN = int(raw9.shape[0])
    print(status("INFO", [("read_raw", in_path), ("N", rawN)]))

    # (2) backup raw
    write_txt9_tab6(backup_path, raw9)
    print(status("INFO", [("backup", backup_path)]))

    rawP = raw9[:, :6].copy()
    rawF = raw9[:, 6:].copy()

    # eval raw
    st_raw, _ = eval_qp_proxy(rawP, dt, lim, safety=args.safety)
    print_eval("BEFORE pose smoothing (RAW)", st_raw, lim, args.safety)

    # (3) force process
    Fp = force_process(
        rawF,
        zero_xy_forces=bool(args.zero_xy_forces),
        force_clamp_abs=args.force_clamp_abs,
        force_ema_alpha=args.force_ema_alpha
    )

    # (4) pose smoothing
    Ps, info = pose_smooth(
        rawP, dt, lim, args.safety,
        hampel_enable=bool(args.hampel_enable),
        hampel_win=args.hampel_win,
        hampel_sig=args.hampel_sig,
        whittaker_auto=bool(args.whittaker_auto),
        lam_pos_init=args.lam_pos_init,
        lam_ang_init=args.lam_ang_init,
        lam_growth=args.lam_growth,
        lam_iters=args.lam_iters,
        cg_iters=args.cg_iters,
        cg_tol=args.cg_tol,
        pose_ema_enable=bool(args.pose_ema_enable),
        pose_ema_alpha=args.pose_ema_alpha
    )
    st_sm, _ = eval_qp_proxy(Ps, dt, lim, safety=args.safety)
    print_eval("AFTER pose smoothing", st_sm, lim, args.safety)
    print(status("POSE-SMOOTH", [("used_lams", info)]))

    # (5) retime
    Pr, Fr, k_total = retime_uniform(
        Ps, Fp,
        dt=dt, lim=lim, safety=args.safety,
        retime_enable=bool(args.retime_enable),
        retime_use_jerk=bool(args.retime_use_jerk),
        retime_max_k=args.retime_max_k,
        retime_passes=args.retime_passes
    )
    st_rt, _ = eval_qp_proxy(Pr, dt, lim, safety=args.safety)
    print_eval("AFTER retiming (pose)", st_rt, lim, args.safety)
    if k_total > 1:
        print(status("QP-EVAL", [("time_scale", k_total), ("rows", f"{Ps.shape[0]} -> {Pr.shape[0]}")]))

    # (6) contact gating
    if bool(args.precontact_gating):
        cidx = detect_contact_idx(Fr[:, 2], args.fz_on, args.fz_off, args.consec_on, args.consec_off)
        if cidx is not None and cidx > 0:
            print(status("CONTACT", [("idx", f"{cidx}/{Pr.shape[0]}"), ("time", f"{cidx*dt:.3f}s"), ("zero", f"[0:{cidx})")]))
            Fr[:cidx, :] = 0.0

    # (7) edge force window
    Fr = apply_edge_force_window(
        Fr,
        hz=args.record_hz,
        edge_force_zero_sec=args.edge_force_zero_sec,
        edge_force_fade_sec=args.edge_force_fade_sec
    )

    # (8) overwrite filtered -> cmd_continue9D.txt
    filt9 = np.hstack([Pr, Fr])
    write_txt9_tab6(in_path, filt9)
    print(status("DONE", [("file", in_path), ("rows", f"{rawN} -> {filt9.shape[0]}")]))

    # (9) viz folder + 3 plots only (time-aligned)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_dir = os.path.join(base_dir, f"viz_{ts}")
    os.makedirs(viz_dir, exist_ok=True)

    save_plot_1_lin_kinematics(viz_dir, dt, rawP, Pr)
    save_plot_2_ang_kinematics(viz_dir, dt, rawP, Pr)
    save_plot_3_forces(viz_dir, dt, rawF, Fr)

    print(block("VIZ", [
        ("dir", viz_dir),
        ("plot 1", "plot_1_lin_kinematics.png"),
        ("plot 2", "plot_2_ang_kinematics.png"),
        ("plot 3", "plot_3_forces.png"),
    ], char="-"))


if __name__ == "__main__":
    main()
