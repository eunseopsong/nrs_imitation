#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Stage-1 trajectory filtering for txt and multimodal HDF5 recorders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from nrs_imitation.vr_stage1_hdf5_recorder import (
    Limits,
    constraints_ok,
    detect_contact_on_idx,
    eval_qp_proxy,
    force_process_with_contact_cleanup,
    hampel_nd,
    norm_rows,
    upsample_linear,
    whittaker_cg_nd,
    whittaker_jerk_cg_nd,
)


@dataclass
class Stage1FilterConfig:
    sample_hz: float
    filter_reference_hz: float = 125.0
    scale_filter_params_with_hz: bool = False
    force_filter_mode: str = "ema"
    zero_xy_forces: bool = False
    force_clamp_abs: float = 200.0
    force_ema_alpha: float = 0.2
    contact_thr_N: float = 5.0
    consec_on: int = 10
    consec_off: int = 10
    fz_contact_smooth_enable: bool = True
    fz_contact_lam_d2: float = 4000.0
    hampel_enable: bool = True
    hampel_win: int = 16
    hampel_sig: float = 2.0
    lam_pos_d2: float = 250000.0
    lam_ang_d2: float = 6000.0
    pose_ema_enable: bool = True
    pose_ema_alpha: float = 0.10
    retime_k: int = 2
    approach_slowdown_enable: bool = True
    approach_pre_sec: float = 5.0
    approach_post_sec: float = 0.3
    approach_scale_max: float = 30.0
    approach_use_fz_ramp: bool = True
    approach_fz_full: float = 20.0
    post_enable: bool = True
    lam_pos_d3: float = 2.0e7
    lam_ang_d3: float = 6.0e5
    qp_guard_enable: bool = True
    qp_guard_safety: float = 0.75
    qp_guard_max_iter: int = 8
    qp_guard_growth: float = 2.2
    max_dev_pos_mm: float = 8.0
    max_dev_ang_rad: float = 0.06
    cg_iters: int = 400
    cg_tol: float = 1e-8
    pos_vmax: float = 30.0
    pos_amax: float = 120.0
    ang_vmax: float = 0.6
    ang_amax: float = 3.0
    pos_jmax: float = 5000.0
    ang_jmax: float = 80.0


@dataclass
class Stage1FilterResult:
    position: np.ndarray
    force: np.ndarray
    source_index: np.ndarray
    sample_time: Optional[np.ndarray]
    meta: Dict[str, object]


def stage1_config_from_recorder(recorder, sample_hz: float) -> Stage1FilterConfig:
    cfg = Stage1FilterConfig(
        sample_hz=float(sample_hz),
        filter_reference_hz=float(getattr(recorder, "filter_reference_hz", 125.0)),
        scale_filter_params_with_hz=bool(getattr(recorder, "scale_filter_params_with_hz", False)),
        force_filter_mode=str(getattr(recorder, "force_filter_mode", "ema")),
        zero_xy_forces=bool(getattr(recorder, "zero_xy_forces", False)),
        force_clamp_abs=float(getattr(recorder, "force_clamp_abs", 200.0)),
        force_ema_alpha=float(getattr(recorder, "force_ema_alpha", getattr(recorder, "fz_ema_alpha", 0.2))),
        contact_thr_N=float(getattr(recorder, "contact_thr_N", getattr(recorder, "fz_gate_N", 5.0))),
        consec_on=int(getattr(recorder, "consec_on", 10)),
        consec_off=int(getattr(recorder, "consec_off", 10)),
        fz_contact_smooth_enable=bool(getattr(recorder, "fz_contact_smooth_enable", True)),
        fz_contact_lam_d2=float(getattr(recorder, "fz_contact_lam_d2", 4000.0)),
        hampel_enable=bool(getattr(recorder, "hampel_enable", True)),
        hampel_win=int(getattr(recorder, "hampel_win", 16)),
        hampel_sig=float(getattr(recorder, "hampel_sig", 2.0)),
        lam_pos_d2=float(getattr(recorder, "lam_pos_d2", 250000.0)),
        lam_ang_d2=float(getattr(recorder, "lam_ang_d2", 6000.0)),
        pose_ema_enable=bool(getattr(recorder, "pose_ema_enable", True)),
        pose_ema_alpha=float(getattr(recorder, "pose_ema_alpha", 0.10)),
        retime_k=int(getattr(recorder, "retime_k", 2)),
        approach_slowdown_enable=bool(getattr(recorder, "approach_slowdown_enable", True)),
        approach_pre_sec=float(getattr(recorder, "approach_pre_sec", 5.0)),
        approach_post_sec=float(getattr(recorder, "approach_post_sec", 0.3)),
        approach_scale_max=float(getattr(recorder, "approach_scale_max", 30.0)),
        approach_use_fz_ramp=bool(getattr(recorder, "approach_use_fz_ramp", True)),
        approach_fz_full=float(getattr(recorder, "approach_fz_full", 20.0)),
        post_enable=bool(getattr(recorder, "post_enable", True)),
        lam_pos_d3=float(getattr(recorder, "lam_pos_d3", 2.0e7)),
        lam_ang_d3=float(getattr(recorder, "lam_ang_d3", 6.0e5)),
        qp_guard_enable=bool(getattr(recorder, "qp_guard_enable", True)),
        qp_guard_safety=float(getattr(recorder, "qp_guard_safety", 0.75)),
        qp_guard_max_iter=int(getattr(recorder, "qp_guard_max_iter", 8)),
        qp_guard_growth=float(getattr(recorder, "qp_guard_growth", 2.2)),
        max_dev_pos_mm=float(getattr(recorder, "max_dev_pos_mm", 8.0)),
        max_dev_ang_rad=float(getattr(recorder, "max_dev_ang_rad", 0.06)),
        cg_iters=int(getattr(recorder, "cg_iters", 400)),
        cg_tol=float(getattr(recorder, "cg_tol", 1e-8)),
        pos_vmax=float(getattr(recorder, "pos_vmax", 30.0)),
        pos_amax=float(getattr(recorder, "pos_amax", 120.0)),
        ang_vmax=float(getattr(recorder, "ang_vmax", 0.6)),
        ang_amax=float(getattr(recorder, "ang_amax", 3.0)),
        pos_jmax=float(getattr(recorder, "pos_jmax", 5000.0)),
        ang_jmax=float(getattr(recorder, "ang_jmax", 80.0)),
    )
    return _with_sample_hz_scaled_params(cfg)


def _scale_whittaker_lambda(lam: float, derivative_order: int, sample_hz: float, reference_hz: float) -> float:
    if lam <= 0.0 or sample_hz <= 0.0 or reference_hz <= 0.0:
        return float(lam)
    power = 2 * int(derivative_order) - 1
    return float(lam) * (float(sample_hz) / float(reference_hz)) ** power


def _scale_ema_alpha(alpha: float, sample_hz: float, reference_hz: float) -> float:
    if alpha <= 0.0 or alpha >= 1.0 or sample_hz <= 0.0 or reference_hz <= 0.0:
        return float(alpha)
    return float(np.clip(1.0 - (1.0 - float(alpha)) ** (float(reference_hz) / float(sample_hz)), 0.0, 1.0))


def _scale_sample_count(count: int, sample_hz: float, reference_hz: float) -> int:
    if count <= 1 or sample_hz <= 0.0 or reference_hz <= 0.0:
        return int(count)
    return max(1, int(round(float(count) * float(sample_hz) / float(reference_hz))))


def _with_sample_hz_scaled_params(cfg: Stage1FilterConfig) -> Stage1FilterConfig:
    if not bool(cfg.scale_filter_params_with_hz):
        return cfg

    hz = float(cfg.sample_hz)
    ref = float(cfg.filter_reference_hz)
    cfg.fz_contact_lam_d2 = _scale_whittaker_lambda(float(cfg.fz_contact_lam_d2), 2, hz, ref)
    cfg.hampel_win = _scale_sample_count(int(cfg.hampel_win), hz, ref)
    cfg.consec_on = _scale_sample_count(int(cfg.consec_on), hz, ref)
    cfg.consec_off = _scale_sample_count(int(cfg.consec_off), hz, ref)
    cfg.lam_pos_d2 = _scale_whittaker_lambda(float(cfg.lam_pos_d2), 2, hz, ref)
    cfg.lam_ang_d2 = _scale_whittaker_lambda(float(cfg.lam_ang_d2), 2, hz, ref)
    cfg.pose_ema_alpha = _scale_ema_alpha(float(cfg.pose_ema_alpha), hz, ref)
    cfg.lam_pos_d3 = _scale_whittaker_lambda(float(cfg.lam_pos_d3), 3, hz, ref)
    cfg.lam_ang_d3 = _scale_whittaker_lambda(float(cfg.lam_ang_d3), 3, hz, ref)
    return cfg


def take_nearest_by_source_index(array: np.ndarray, source_index: np.ndarray) -> np.ndarray:
    if array.shape[0] == 0:
        return array.copy()
    idx = np.rint(source_index).astype(np.int64)
    idx = np.clip(idx, 0, array.shape[0] - 1)
    return array[idx].copy()


def interpolate_sample_times(sample_times: Optional[np.ndarray], source_index: np.ndarray) -> Optional[np.ndarray]:
    if sample_times is None:
        return None
    times = np.asarray(sample_times, dtype=np.float64).reshape(-1)
    if times.size == 0:
        return np.zeros((source_index.shape[0],), dtype=np.float64)
    if times.size == 1:
        return np.full((source_index.shape[0],), float(times[0]), dtype=np.float64)
    raw_idx = np.arange(times.size, dtype=np.float64)
    return np.interp(source_index.astype(np.float64), raw_idx, times).astype(np.float64)


def apply_stage1_filter(
    raw_position: np.ndarray,
    raw_force: np.ndarray,
    cfg: Stage1FilterConfig,
    sample_times: Optional[np.ndarray] = None,
    logger=None,
) -> Stage1FilterResult:
    rawP = np.asarray(raw_position, dtype=np.float64).reshape(-1, 6)
    rawF = np.asarray(raw_force, dtype=np.float64).reshape(-1, 3)
    if rawP.shape[0] != rawF.shape[0]:
        raise ValueError(f"raw_position and raw_force length mismatch: {rawP.shape[0]} vs {rawF.shape[0]}")

    rawN = int(rawP.shape[0])
    dt = 1.0 / max(1e-9, float(cfg.sample_hz))

    force_filter_mode = str(cfg.force_filter_mode or "ema").strip().lower()
    if force_filter_mode in ("contact", "contact_cleanup", "stage1_contact", "legacy_contact"):
        Fp, on_idx, off_idx = force_process_with_contact_cleanup(
            rawF,
            clamp_abs=float(cfg.force_clamp_abs),
            ema_alpha=float(cfg.force_ema_alpha),
            zero_xy=bool(cfg.zero_xy_forces),
            contact_thr_N=float(cfg.contact_thr_N),
            consec_on=int(cfg.consec_on),
            consec_off=int(cfg.consec_off),
            fz_contact_smooth_enable=bool(cfg.fz_contact_smooth_enable),
            fz_contact_lam_d2=float(cfg.fz_contact_lam_d2),
            cg_iters=int(cfg.cg_iters),
            cg_tol=float(cfg.cg_tol),
        )
    else:
        Fp = _force_ema_only(rawF, cfg)
        on_idx = detect_contact_on_idx(Fp[:, 2], float(cfg.contact_thr_N), int(cfg.consec_on))
        off_idx = None

    Ps = _pose_pre_smooth(rawP, cfg)

    retime_k = max(1, int(cfg.retime_k))
    Pr = upsample_linear(Ps, retime_k)
    Fr = upsample_linear(Fp, retime_k)
    source = upsample_linear(np.arange(rawN, dtype=np.float64).reshape(-1, 1), retime_k).reshape(-1)

    Pr_slow, Fr_slow, source_slow, approach_idx, approach_applied = _apply_contact_approach_slowdown(
        Pr,
        Fr,
        source,
        cfg,
        logger=logger,
    )

    Pf = _qp_guard(Pr_slow, cfg, logger=logger)
    sample_times_out = interpolate_sample_times(sample_times, source_slow)

    meta = {
        "filter_source": "stage1_vr_filtering_pipeline",
        "filter_reference_hz": float(cfg.filter_reference_hz),
        "scale_filter_params_with_hz": int(bool(cfg.scale_filter_params_with_hz)),
        "force_filter_mode": force_filter_mode,
        "zero_xy_forces": int(bool(cfg.zero_xy_forces)),
        "force_ema_alpha": float(cfg.force_ema_alpha),
        "hampel_win": int(cfg.hampel_win),
        "lam_pos_d2": float(cfg.lam_pos_d2),
        "lam_ang_d2": float(cfg.lam_ang_d2),
        "pose_ema_enable": int(bool(cfg.pose_ema_enable)),
        "pose_ema_alpha": float(cfg.pose_ema_alpha),
        "lam_pos_d3": float(cfg.lam_pos_d3),
        "lam_ang_d3": float(cfg.lam_ang_d3),
        "raw_len": rawN,
        "out_len": int(Pf.shape[0]),
        "retime_k": retime_k,
        "force_contact_on_idx": -1 if on_idx is None else int(on_idx),
        "force_contact_off_idx": -1 if off_idx is None else int(off_idx),
        "approach_contact_idx": -1 if approach_idx is None else int(approach_idx),
        "approach_applied": int(bool(approach_applied)),
        "dt": float(dt),
    }

    if logger is not None:
        logger.info(
            f"[STAGE1-FILTER] raw_len={rawN} -> out_len={Pf.shape[0]}, "
            f"retime_k={retime_k}, force_mode={force_filter_mode}, "
            f"approach={bool(approach_applied)}, "
            f"force_contact_on={meta['force_contact_on_idx']}"
        )

    return Stage1FilterResult(
        position=Pf.astype(np.float32),
        force=Fr_slow.astype(np.float32),
        source_index=source_slow.astype(np.float32),
        sample_time=sample_times_out,
        meta=meta,
    )


def _pose_pre_smooth(P: np.ndarray, cfg: Stage1FilterConfig) -> np.ndarray:
    P0 = P.copy()
    if cfg.hampel_enable:
        P0 = hampel_nd(P0, win=int(cfg.hampel_win), n_sigmas=float(cfg.hampel_sig))

    P1 = P0.copy()
    P1[:, :3] = whittaker_cg_nd(P1[:, :3], lam=float(cfg.lam_pos_d2), cg_iters=int(cfg.cg_iters), tol=float(cfg.cg_tol))
    P1[:, 3:] = whittaker_cg_nd(P1[:, 3:], lam=float(cfg.lam_ang_d2), cg_iters=int(cfg.cg_iters), tol=float(cfg.cg_tol))

    if cfg.pose_ema_enable:
        P1 = _ema_nd(P1, alpha=float(cfg.pose_ema_alpha))
    return P1


def _ema_nd(Y: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0 or alpha >= 1.0:
        return Y.copy()
    Z = Y.copy()
    for i in range(1, Y.shape[0]):
        Z[i] = alpha * Y[i] + (1.0 - alpha) * Z[i - 1]
    return Z


def _force_ema_only(F: np.ndarray, cfg: Stage1FilterConfig) -> np.ndarray:
    Fp = np.asarray(F, dtype=np.float64).reshape(-1, 3).copy()
    Fp[~np.isfinite(Fp)] = 0.0

    clamp_abs = float(cfg.force_clamp_abs)
    if clamp_abs > 0.0:
        Fp = np.clip(Fp, -clamp_abs, clamp_abs)

    if bool(cfg.zero_xy_forces):
        Fp[:, 0] = 0.0
        Fp[:, 1] = 0.0

    return _ema_nd(Fp, alpha=float(cfg.force_ema_alpha))


def _apply_contact_approach_slowdown(
    P: np.ndarray,
    F: np.ndarray,
    source_index: np.ndarray,
    cfg: Stage1FilterConfig,
    logger=None,
):
    if not cfg.approach_slowdown_enable:
        return P, F, source_index, None, False

    fz = F[:, 2]
    cidx = detect_contact_on_idx(fz, float(cfg.contact_thr_N), int(cfg.consec_on))
    if cidx is None:
        if logger is not None:
            logger.warn("[APPROACH] contact not found -> skip approach slow-down")
        return P, F, source_index, None, False

    preN = int(round(float(cfg.approach_pre_sec) * float(cfg.sample_hz)))
    postN = int(round(float(cfg.approach_post_sec) * float(cfg.sample_hz)))

    N = P.shape[0]
    seg_scale = np.ones(N - 1, dtype=np.float64)

    s0 = max(0, cidx - preN)
    s1 = min(N - 1, cidx + postN)
    if s1 <= s0 + 2:
        if logger is not None:
            logger.warn("[APPROACH] window too small -> skip approach slow-down")
        return P, F, source_index, cidx, False

    idx = np.arange(s0, s1, dtype=np.float64)
    u = (idx - float(s0)) / max(1.0, float(s1 - s0))
    bump = 0.5 - 0.5 * np.cos(2.0 * np.pi * u)
    bump = np.clip(bump, 0.0, 1.0)

    scale_target = 1.0 + (float(cfg.approach_scale_max) - 1.0) * bump
    if cfg.approach_use_fz_ramp:
        fz_win = fz[s0:s1]
        ramp = np.clip(fz_win / max(1e-6, float(cfg.approach_fz_full)), 0.0, 1.0)
        scale_target = 1.0 + (scale_target - 1.0) * ramp

    seg_scale[s0:s1] = np.maximum(seg_scale[s0:s1], scale_target)
    Pn, Fn, source_n = _resample_uniform_by_timewarp(P, F, source_index, 1.0 / max(1e-9, cfg.sample_hz), seg_scale)
    return Pn, Fn, source_n, cidx, True


def _resample_uniform_by_timewarp(
    P: np.ndarray,
    F: np.ndarray,
    source_index: np.ndarray,
    dt: float,
    seg_scale: np.ndarray,
):
    N = P.shape[0]
    if N <= 1:
        return P.copy(), F.copy(), source_index.copy()
    if seg_scale.shape[0] != N - 1:
        raise ValueError(f"seg_scale shape mismatch: {seg_scale.shape[0]} vs {N - 1}")

    tprime = np.zeros(N, dtype=np.float64)
    tprime[1:] = np.cumsum(float(dt) * seg_scale)

    T = float(tprime[-1])
    if T <= 0.0:
        return P.copy(), F.copy(), source_index.copy()

    M = int(np.round(T / float(dt))) + 1
    t_u = np.arange(M, dtype=np.float64) * float(dt)
    t_u[-1] = T

    Pn = np.empty((M, P.shape[1]), dtype=np.float64)
    Fn = np.empty((M, F.shape[1]), dtype=np.float64)
    for d in range(P.shape[1]):
        Pn[:, d] = np.interp(t_u, tprime, P[:, d])
    for d in range(F.shape[1]):
        Fn[:, d] = np.interp(t_u, tprime, F[:, d])
    source_n = np.interp(t_u, tprime, source_index)
    return Pn, Fn, source_n


def _pose_post_smooth_d3(P: np.ndarray, cfg: Stage1FilterConfig, lam_pos_d3: float, lam_ang_d3: float) -> np.ndarray:
    if not cfg.post_enable:
        return P
    P2 = P.copy()
    P2[:, :3] = whittaker_jerk_cg_nd(P2[:, :3], lam=lam_pos_d3, cg_iters=int(cfg.cg_iters), tol=float(cfg.cg_tol))
    P2[:, 3:] = whittaker_jerk_cg_nd(P2[:, 3:], lam=lam_ang_d3, cg_iters=int(cfg.cg_iters), tol=float(cfg.cg_tol))
    return P2


def _qp_guard(Pref: np.ndarray, cfg: Stage1FilterConfig, logger=None) -> np.ndarray:
    if not cfg.qp_guard_enable:
        return _pose_post_smooth_d3(Pref, cfg, float(cfg.lam_pos_d3), float(cfg.lam_ang_d3))

    lim = Limits(
        pos_vmax=float(cfg.pos_vmax),
        pos_amax=float(cfg.pos_amax),
        ang_vmax=float(cfg.ang_vmax),
        ang_amax=float(cfg.ang_amax),
        pos_jmax=float(cfg.pos_jmax),
        ang_jmax=float(cfg.ang_jmax),
    )

    lam_p = float(cfg.lam_pos_d3)
    lam_a = float(cfg.lam_ang_d3)
    best = None
    best_score = 1e18

    for _ in range(max(1, int(cfg.qp_guard_max_iter))):
        Pk = _pose_post_smooth_d3(Pref, cfg, lam_p, lam_a)

        dpos = norm_rows(Pk[:, :3] - Pref[:, :3])
        dang = norm_rows(Pk[:, 3:] - Pref[:, 3:])
        if float(dpos.max()) > float(cfg.max_dev_pos_mm) or float(dang.max()) > float(cfg.max_dev_ang_rad):
            if logger is not None:
                logger.warn(
                    f"[QP-GUARD] stop by deviation: max_dpos={float(dpos.max()):.3f}mm "
                    f"(allow {float(cfg.max_dev_pos_mm)}), max_dang={float(dang.max()):.4f}rad "
                    f"(allow {float(cfg.max_dev_ang_rad)})"
                )
            break

        st, _ = eval_qp_proxy(Pk, 1.0 / max(1e-9, float(cfg.sample_hz)), lim, safety=float(cfg.qp_guard_safety))
        score = max(
            st.jpos_p95 / (lim.pos_jmax * float(cfg.qp_guard_safety) + 1e-9),
            st.jang_p95 / (lim.ang_jmax * float(cfg.qp_guard_safety) + 1e-9),
            st.apos_p95 / (lim.pos_amax * float(cfg.qp_guard_safety) + 1e-9),
            st.aang_p95 / (lim.ang_amax * float(cfg.qp_guard_safety) + 1e-9),
        )
        if score < best_score:
            best_score = score
            best = Pk

        if constraints_ok(st):
            return Pk

        lam_p *= float(cfg.qp_guard_growth)
        lam_a *= float(cfg.qp_guard_growth)

    if logger is not None:
        logger.warn("[QP-GUARD] could not fully satisfy constraints. Returning best smoothed trajectory.")
    return best if best is not None else _pose_post_smooth_d3(Pref, cfg, float(cfg.lam_pos_d3), float(cfg.lam_ang_d3))
