#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Polishing-removal visualization for an inference run.

Same removal model as ~/nrspath_ws/src/polishing_removal (Preston: removal
rate = k * Fn * speed, accumulated on an xy grid, pad-convolved), but wired
to the LAUNCH lifecycle instead of ~/start / ~/end services:

  * recording begins as soon as this node starts (launch start)
  * on shutdown (Ctrl-C / launch end) it computes the removal map for the
    trajectory that was actually traversed and writes the 4 PNGs + a summary

Subscribes:  /ur10skku/currentP  [x, y, z, rx, ry, rz]   (mm ; rot-vec)
             /ur10skku/currentF  [fx, fy, fz, ...]        (N)
Note: currentP[3:6] here is a rotation vector, not angular velocity -- it is
only used for the direction arrows in 04_3d_path_w.png (kept for parity with
the original tool), not for the removal computation.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402
import numpy as np  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from std_msgs.msg import Float64MultiArray  # noqa: E402


# ============================================================
# removal model (ported from polishing_removal_node.py)
# ============================================================
@dataclass
class RemovalResult:
    t: np.ndarray
    state6: np.ndarray
    force3: np.ndarray
    dt: np.ndarray
    fn: np.ndarray
    speed: np.ndarray
    contact_mask: np.ndarray
    grid_removal: np.ndarray
    grid_hits: np.ndarray
    xi: np.ndarray
    yi: np.ndarray
    heatmap_value_series: np.ndarray
    meta: Dict[str, float]


def _parse_tool_axis(axis_str: str) -> np.ndarray:
    s = axis_str.strip().upper()
    table = {
        "+Z": (0, 0, 1), "Z": (0, 0, 1), "Z+": (0, 0, 1),
        "-Z": (0, 0, -1), "Z-": (0, 0, -1),
        "+X": (1, 0, 0), "X": (1, 0, 0), "X+": (1, 0, 0),
        "-X": (-1, 0, 0), "X-": (-1, 0, 0),
        "+Y": (0, 1, 0), "Y": (0, 1, 0), "Y+": (0, 1, 0),
        "-Y": (0, -1, 0), "Y-": (0, -1, 0),
    }
    if s not in table:
        raise ValueError("tool_axis must be one of +/-X, +/-Y, +/-Z")
    return np.array(table[s], dtype=float)


def _fft_convolve2d(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    h, w = a.shape
    kh, kw = k.shape
    fh = int(2 ** np.ceil(np.log2(h + kh - 1)))
    fw = int(2 ** np.ceil(np.log2(w + kw - 1)))
    y = np.fft.irfft2(np.fft.rfft2(a, s=(fh, fw)) * np.fft.rfft2(k, s=(fh, fw)), s=(fh, fw))
    sh, sw = (kh - 1) // 2, (kw - 1) // 2
    return y[sh:sh + h, sw:sw + w]


def _compute_removal(
    t: np.ndarray, state6: np.ndarray, force3: np.ndarray,
    cell_mm: float, k_preston: float, tool_axis: str, speed_mode: str,
    contact_threshold_N: float, speed_threshold_mm_s: float, pad_radius_mm: float,
    view_margin_mm: float = 40.0,
) -> RemovalResult:
    xyz = state6[:, 0:3]

    dt = np.diff(t, prepend=t[0])
    dt0 = float(np.median(np.diff(t))) if t.shape[0] >= 2 else 0.02
    dt[0] = max(dt0, 1e-6)
    dt = np.clip(dt, 1e-6, None)

    vxyz = np.zeros_like(xyz)
    if xyz.shape[0] >= 2:
        dt_seg = np.clip(np.diff(t), 1e-6, None)
        vxyz[1:] = np.diff(xyz, axis=0) / dt_seg[:, None]
        vxyz[0] = vxyz[1]

    if speed_mode.lower() == "xy":
        speed = np.linalg.norm(vxyz[:, 0:2], axis=1)
    else:
        speed = np.linalg.norm(vxyz, axis=1)

    axis_base = _parse_tool_axis(tool_axis)
    fn1 = np.sum(force3[:, :3] * axis_base[None, :], axis=1)
    fn = np.maximum(fn1 if np.mean(np.maximum(fn1, 0.0)) >= np.mean(np.maximum(-fn1, 0.0)) else -fn1, 0.0)

    contact_mask = (fn >= contact_threshold_N) & (speed >= speed_threshold_mm_s)
    removal_rate = np.zeros_like(fn)
    removal_rate[contact_mask] = k_preston * fn[contact_mask] * speed[contact_mask]
    dremoval = removal_rate * dt

    # Grid spans the traversed area plus view_margin_mm of surrounding
    # workpiece on every side, so the heatmap shows context around the
    # polished region (and the pad-footprint convolution isn't clipped).
    x, y = xyz[:, 0], xyz[:, 1]
    m = max(view_margin_mm, 0.0)
    x_min = np.floor((np.min(x) - m) / cell_mm) * cell_mm
    x_max = np.ceil((np.max(x) + m) / cell_mm) * cell_mm
    y_min = np.floor((np.min(y) - m) / cell_mm) * cell_mm
    y_max = np.ceil((np.max(y) + m) / cell_mm) * cell_mm
    W = max(int(np.round((x_max - x_min) / cell_mm)) + 1, 1)
    H = max(int(np.round((y_max - y_min) / cell_mm)) + 1, 1)
    xi = np.clip(np.round((x - x_min) / cell_mm).astype(int), 0, W - 1)
    yi = np.clip(np.round((y - y_min) / cell_mm).astype(int), 0, H - 1)

    grid_removal = np.zeros((H, W))
    grid_hits = np.zeros((H, W))
    np.add.at(grid_removal, (yi, xi), dremoval)
    np.add.at(grid_hits, (yi, xi), 1.0)

    if pad_radius_mm > 0.0:
        r_cells = int(np.ceil(pad_radius_mm / cell_mm))
        if r_cells >= 1:
            yy, xx = np.mgrid[-r_cells:r_cells + 1, -r_cells:r_cells + 1]
            kernel = ((xx ** 2 + yy ** 2) <= (pad_radius_mm / cell_mm) ** 2).astype(float)
            kernel /= np.sum(kernel)
            grid_removal = _fft_convolve2d(grid_removal, kernel)
            grid_hits = _fft_convolve2d(grid_hits, kernel)

    heatmap_value_series = grid_removal[yi, xi].copy()
    visited_vals = grid_removal[grid_hits > 0] if np.any(grid_hits > 0) else np.array([0.0])
    meta = {
        "x_min": float(x_min), "x_max": float(x_max),
        "y_min": float(y_min), "y_max": float(y_max),
        "cell_mm": float(cell_mm), "k_preston": float(k_preston),
        "contact_threshold_N": float(contact_threshold_N),
        "speed_threshold_mm_s": float(speed_threshold_mm_s),
        "pad_radius_mm": float(pad_radius_mm),
        "mean_heatmap_removal": float(np.mean(visited_vals)),
        "std_heatmap_removal": float(np.std(visited_vals)),
        "samples": int(t.size),
        "contact_samples": int(np.sum(contact_mask)),
        "duration_s": float(t[-1] - t[0]) if t.size >= 2 else 0.0,
        "max_speed_mm_s": float(np.max(speed)) if speed.size else 0.0,
        "max_fn_N": float(np.max(fn)) if fn.size else 0.0,
    }
    return RemovalResult(t, state6, force3, dt, fn, speed, contact_mask,
                         grid_removal, grid_hits, xi, yi, heatmap_value_series, meta)


def _heatmap_figure(res: RemovalResult):
    extent = [res.meta["x_min"], res.meta["x_max"], res.meta["y_min"], res.meta["y_max"]]
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(res.grid_removal, origin="lower", extent=extent, aspect="auto")
    ax.set_title("Polishing Removal Heatmap (inference trajectory)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    fig.colorbar(im, ax=ax).set_label("Cell removal [a.u.]")
    ax.text(0.02, 0.98,
            f"mean = {res.meta['mean_heatmap_removal']:.6f}\n"
            f"std  = {res.meta['std_heatmap_removal']:.6f}\n"
            f"samples = {res.meta['samples']}\n"
            f"contact = {res.meta['contact_samples']}\n"
            f"duration = {res.meta['duration_s']:.1f} s",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    fig.tight_layout()
    return fig


_DPI = 150


def _save_heatmap(res: RemovalResult, out_dir: Path) -> Path:
    """The one plot that matters -- saved first, on its own, so a later
    failure (or a second Ctrl-C mid-render) can't lose it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = _heatmap_figure(res)
    path = out_dir / "01_removal_heatmap.png"
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    return path


def _save_companion_plots(res: RemovalResult, out_dir: Path,
                          w_arrow_stride: int, w_arrow_length_mm: float) -> list:
    xyz = res.state6[:, 0:3]
    wxyz = res.state6[:, 3:6]
    force3 = res.force3
    saved = ["01_removal_heatmap.png"]

    def _plot_time_series():
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        ax.plot(res.t, res.heatmap_value_series, linewidth=1.5)
        ax.axhline(res.meta["mean_heatmap_removal"], linestyle="--", linewidth=1.2, label="heatmap mean")
        ax.set_title("Heatmap Cell Removal at Current Position vs Time")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("cell removal [a.u.]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "02_heatmap_value_vs_time.png", dpi=_DPI)
        plt.close(fig)
        return "02_heatmap_value_vs_time.png"

    def _plot_signals():
        labels = [
            (xyz[:, 0], "x [mm]"), (xyz[:, 1], "y [mm]"), (xyz[:, 2], "z [mm]"),
            (wxyz[:, 0], "rx"), (wxyz[:, 1], "ry"), (wxyz[:, 2], "rz"),
            (force3[:, 0], "fx [N]"), (force3[:, 1], "fy [N]"), (force3[:, 2], "fz [N]"),
        ]
        fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)
        for ax, (y, ylabel) in zip(axes.ravel(), labels):
            ax.plot(res.t, y, linewidth=1.3)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        for ax in axes[-1, :]:
            ax.set_xlabel("time [s]")
        fig.suptitle("Recorded Signals")
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        fig.savefig(out_dir / "03_signals_subplot.png", dpi=_DPI)
        plt.close(fig)
        return "03_signals_subplot.png"

    def _plot_3d_path():
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], linewidth=1.6, label="xyz path")
        stride = max(int(w_arrow_stride), 1)
        idx = np.arange(0, xyz.shape[0], stride)
        if idx.size:
            w_s = wxyz[idx]
            w_norm = np.linalg.norm(w_s, axis=1, keepdims=True)
            valid = w_norm[:, 0] > 1e-12
            if np.any(valid):
                dirs = np.zeros_like(w_s)
                dirs[valid] = w_s[valid] / w_norm[valid]
                ax.quiver(xyz[idx][valid, 0], xyz[idx][valid, 1], xyz[idx][valid, 2],
                          dirs[valid, 0], dirs[valid, 1], dirs[valid, 2],
                          length=float(w_arrow_length_mm), normalize=False,
                          linewidth=0.8, arrow_length_ratio=0.25)
        ax.set_title("3D Path with Orientation Direction (rx, ry, rz)")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")
        ax.legend(loc="best")
        rng = max(float(np.ptp(xyz[:, 0])), float(np.ptp(xyz[:, 1])), float(np.ptp(xyz[:, 2])), 1.0)
        mid = xyz.min(0) + (xyz.max(0) - xyz.min(0)) / 2.0
        ax.set_xlim(mid[0] - rng / 2, mid[0] + rng / 2)
        ax.set_ylim(mid[1] - rng / 2, mid[1] + rng / 2)
        ax.set_zlim(mid[2] - rng / 2, mid[2] + rng / 2)
        fig.tight_layout()
        fig.savefig(out_dir / "04_3d_path_w.png", dpi=_DPI)
        plt.close(fig)
        return "04_3d_path_w.png"

    for fn in (_plot_time_series, _plot_signals, _plot_3d_path):
        try:
            saved.append(fn())
        except Exception as e:  # noqa: BLE001
            print(f"[polishing_removal] {fn.__name__} failed: {e}")

    return saved


# ============================================================
# node
# ============================================================
class PolishingRemovalRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("polishing_removal_recorder")

        self.declare_parameter("enable", True)
        self.declare_parameter("position_topic", "/ur10skku/currentP")
        self.declare_parameter("force_topic", "/ur10skku/currentF")
        self.declare_parameter("recording_rate_hz", 50.0)
        self.declare_parameter("output_dir", os.path.expanduser(
            "~/nrs_imitation/logs/polishing_removal"))
        self.declare_parameter("run_tag", "")
        self.declare_parameter("open_on_exit", True)
        self.declare_parameter("start_delay_sec", 0.0)
        self.declare_parameter("tool_axis", "+Z")
        self.declare_parameter("speed_mode", "xy")
        self.declare_parameter("contact_threshold_N", 0.5)
        self.declare_parameter("speed_threshold_mm_s", 0.1)
        self.declare_parameter("cell_mm", 1.0)
        self.declare_parameter("pad_radius_mm", 20.0)
        self.declare_parameter("view_margin_mm", 40.0)
        self.declare_parameter("k_preston", 1.0)
        self.declare_parameter("w_arrow_stride", 20)
        self.declare_parameter("w_arrow_length_mm", 15.0)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.enable = bool(g("enable"))
        self.position_topic = str(g("position_topic"))
        self.force_topic = str(g("force_topic"))
        self.rate_hz = max(float(g("recording_rate_hz")), 1e-3)
        self.output_dir = Path(str(g("output_dir")))
        self.run_tag = str(g("run_tag"))
        self.open_on_exit = bool(g("open_on_exit"))
        self.start_delay_sec = float(g("start_delay_sec"))

        self._lock = threading.Lock()
        self.latest_state6: Optional[np.ndarray] = None
        self.latest_force3: Optional[np.ndarray] = None
        self.t_buf: list[float] = []
        self.s_buf: list[np.ndarray] = []
        self.f_buf: list[np.ndarray] = []
        self._t0: Optional[float] = None
        self._done = False

        if not self.enable:
            self.get_logger().info("polishing_removal_recorder disabled (enable:=false)")
            return

        # Warm matplotlib now (font cache / backend init) so the first
        # savefig at shutdown -- under the launch's SIGKILL grace timer --
        # isn't the cold one.
        try:
            _warm = plt.figure(figsize=(1, 1))
            _warm.savefig(os.devnull, format="png")
            plt.close(_warm)
        except Exception:  # noqa: BLE001
            pass

        self.create_subscription(Float64MultiArray, self.position_topic, self._on_pose, qos_profile_sensor_data)
        self.create_subscription(Float64MultiArray, self.force_topic, self._on_force, qos_profile_sensor_data)
        self.timer = self.create_timer(1.0 / self.rate_hz, self._on_tick)
        self.get_logger().info(
            f"polishing_removal_recorder recording {self.position_topic} + {self.force_topic} "
            f"@ {self.rate_hz:.0f}Hz; heatmap on shutdown -> {self.output_dir}")

    def _on_pose(self, msg: Float64MultiArray) -> None:
        d = np.asarray(msg.data, dtype=float)
        if d.size >= 6:
            with self._lock:
                self.latest_state6 = d[:6].copy()

    def _on_force(self, msg: Float64MultiArray) -> None:
        d = np.asarray(msg.data, dtype=float)
        if d.size >= 3:
            with self._lock:
                self.latest_force3 = d[:3].copy()

    def _on_tick(self) -> None:
        with self._lock:
            if self.latest_state6 is None or self.latest_force3 is None:
                return
            now = self.get_clock().now().nanoseconds * 1e-9
            if self._t0 is None:
                self._t0 = now
            t_rel = now - self._t0
            if t_rel < self.start_delay_sec:
                return
            self.t_buf.append(float(t_rel))
            self.s_buf.append(self.latest_state6.copy())
            self.f_buf.append(self.latest_force3.copy())

    def _finalize(self) -> None:
        if not self.enable or self._done:
            return
        self._done = True
        if getattr(self, "timer", None) is not None:
            self.timer.cancel()
        with self._lock:
            t = np.asarray(self.t_buf, dtype=float)
            s = np.asarray(self.s_buf, dtype=float)
            f = np.asarray(self.f_buf, dtype=float)
        if t.size < 5 or s.shape[0] < 5 or f.shape[0] < 5:
            self.get_logger().warn(
                f"polishing_removal_recorder: only {t.size} samples, nothing to visualize")
            return

        tag = f"{self.run_tag}_" if self.run_tag else ""
        session = self.output_dir / f"{tag}{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            res = _compute_removal(
                t, s, f,
                cell_mm=float(self.get_parameter("cell_mm").value),
                k_preston=float(self.get_parameter("k_preston").value),
                tool_axis=str(self.get_parameter("tool_axis").value),
                speed_mode=str(self.get_parameter("speed_mode").value),
                contact_threshold_N=float(self.get_parameter("contact_threshold_N").value),
                speed_threshold_mm_s=float(self.get_parameter("speed_threshold_mm_s").value),
                pad_radius_mm=float(self.get_parameter("pad_radius_mm").value),
                view_margin_mm=float(self.get_parameter("view_margin_mm").value),
            )
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"[polishing_removal] removal compute failed: {e}")
            return

        # summary + heatmap first, on their own -- a second Ctrl-C or a
        # failure in the companion plots must not lose the main output.
        session.mkdir(parents=True, exist_ok=True)
        try:
            (session / "summary.txt").write_text(
                "\n".join(f"{k}: {v}" for k, v in res.meta.items()) + "\n")
        except Exception:  # noqa: BLE001
            pass
        try:
            heatmap_path = _save_heatmap(res, session)
            self.get_logger().warn(f"[polishing_removal] heatmap saved -> {heatmap_path}")
            if self.open_on_exit and os.environ.get("DISPLAY"):
                subprocess.Popen(
                    ["xdg-open", str(heatmap_path)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"[polishing_removal] heatmap save/open failed: {e}")

        try:
            saved = _save_companion_plots(
                res, session,
                w_arrow_stride=int(self.get_parameter("w_arrow_stride").value),
                w_arrow_length_mm=float(self.get_parameter("w_arrow_length_mm").value),
            )
            self.get_logger().warn(
                f"[polishing_removal] {len(saved)} plots + summary -> {session}")
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"[polishing_removal] companion plots failed: {e}")

    def destroy_node(self):
        try:
            self._finalize()
        except Exception:  # noqa: BLE001
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolishingRemovalRecorderNode()

    def _stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _stop)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # The removal compute + 4 plots take ~1s and must finish -- a second
        # Ctrl-C or launch's SIGTERM would otherwise leave a half-written set
        # (and no heatmap for overlay_video_recorder to append). Ignore both.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
