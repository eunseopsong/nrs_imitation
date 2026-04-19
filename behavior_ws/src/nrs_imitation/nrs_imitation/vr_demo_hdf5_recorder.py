#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vr_demo_hdf5_recorder.py

New 2-step ACT dataset pipeline recorder.

Goal:
  Record VR tracker pose + F/T sensor force + single RealSense camera image
  directly into ACT merged_hdf5 format.

Input topics:
  pose  : /calibrated_pose                  Float64MultiArray [x y z wx wy wz]
          - xyz input unit is meter by default.
          - stored xyz unit is millimeter by default.
          - wx wy wz are stored as-is, assumed rotation-vector [rad].

  force : /ftsensor/measured_Cvalue         geometry_msgs/Wrench
          - uses force.x, force.y, force.z only.
          - raw Fx/Fy are used for episode start/end trigger.
          - saved force is processed:
              Fx, Fy -> 0
              Fz    -> raw Fz + EMA
              first/last force_edge_zero_sec -> all force zero

  image : /realsense/vr/color/image_raw     sensor_msgs/Image
          - stored as RGB uint8.

Episode rule:
  start : |Fx| >= start_abs_fx
  end   : |Fy| >= stop_abs_fy

Sampling:
  - final ACT merged dataset is sampled by sample_hz timer, default 30 Hz.
  - pose can be 125 Hz, force can be 500 Hz, image can be 30 Hz.
  - each sample tick stores the latest fresh pose/force/image.

Output:
  /home/eunseop/nrs_act/datasets/ACT/YYYYMMDD_HHMM/merged_hdf5/
    vr_demo_merged_YYYYMMDD_HHMM.hdf5

HDF5 layout:
  episodes/
    ep_0000/
      position        (T, 6) float32  [x_mm y_mm z_mm wx wy wz]
      ft              (T, 3) float32  [fx fy fz]
      images/
        cam0          (T, H, W, 3) uint8 RGB

This file is designed to be followed by a single-camera version of:
  demo_data_act_form.py
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
from typing import Optional, List, Tuple

import numpy as np
import h5py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Wrench
from sensor_msgs.msg import Image


# ============================================================
# Image utilities
# ============================================================
def image_to_rgb_numpy(msg: Image) -> Optional[np.ndarray]:
    """
    Convert ROS sensor_msgs/Image to RGB uint8 numpy array.

    Supported encodings:
      rgb8, bgr8, rgba8, bgra8, mono8
    """
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
        img4 = row[:, :need].reshape(h, w, 4)
        img = img4[:, :, :3]
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
        if im is not None and im.ndim == 3 and im.shape[0] > 1 and im.shape[1] > 1 and im.shape[2] == 3:
            return int(im.shape[0]), int(im.shape[1])
    return 0, 0


def stack_images_repeat_last(frames: List[Optional[np.ndarray]], logger=None) -> Optional[np.ndarray]:
    """
    Stack image frames into (T,H,W,3).

    Missing or shape-mismatched frames are replaced by the latest valid frame.
    If no valid frame exists, return None.
    """
    H, W = pick_image_shape(frames)
    if H <= 0 or W <= 0:
        return None

    T = len(frames)
    out = np.zeros((T, H, W, 3), dtype=np.uint8)
    last = np.zeros((H, W, 3), dtype=np.uint8)
    valid_count = 0
    mismatch_count = 0

    for i, im in enumerate(frames):
        if im is not None and im.ndim == 3 and im.shape == (H, W, 3):
            out[i] = im
            last = im
            valid_count += 1
        else:
            out[i] = last
            mismatch_count += 1

    if logger is not None:
        logger.info(
            f"[IMAGE] stacked images: T={T}, H={H}, W={W}, "
            f"valid={valid_count}, repeated_or_invalid={mismatch_count}"
        )

    return out


# ============================================================
# Force utilities
# ============================================================
def ema_1d(y: np.ndarray, alpha: float) -> np.ndarray:
    if y.size == 0:
        return y.astype(np.float64).copy()

    z = y.astype(np.float64).copy()
    if alpha <= 0.0 or alpha >= 1.0:
        return z

    for i in range(1, y.size):
        z[i] = alpha * y[i] + (1.0 - alpha) * z[i - 1]
    return z


def process_force_keep_fz_with_ema_and_edge_zero(
    Fraw: np.ndarray,
    fz_ema_alpha: float,
    edge_zero_sec: float,
    sample_hz: float,
    zero_xy: bool = True,
    logger=None,
) -> np.ndarray:
    """
    Filtering policy aligned with the latest vr_demo_txt_recorder.py style:
      - Fx, Fy -> 0
      - Fz -> raw Fz + EMA only
      - first edge_zero_sec and last edge_zero_sec -> all forces zero

    Fraw shape:
      (T, 3) = [Fx, Fy, Fz]
    """
    if Fraw.size == 0:
        return Fraw.astype(np.float64).copy()

    Fp = Fraw.astype(np.float64).copy()
    N = int(Fp.shape[0])

    if zero_xy:
        Fp[:, 0] = 0.0
        Fp[:, 1] = 0.0

    Fp[:, 2] = ema_1d(Fp[:, 2], alpha=float(fz_ema_alpha))

    edge_n = int(round(float(edge_zero_sec) * float(sample_hz)))
    edge_n = max(0, min(edge_n, N))

    if edge_n > 0:
        Fp[:edge_n, :] = 0.0
        Fp[max(0, N - edge_n):, :] = 0.0

    if logger is not None:
        raw_fz_abs_max = float(np.max(np.abs(Fraw[:, 2]))) if N > 0 else 0.0
        proc_fz_abs_max = float(np.max(np.abs(Fp[:, 2]))) if N > 0 else 0.0
        logger.info(
            f"[FORCE] zero_xy={zero_xy}, fz_ema_alpha={fz_ema_alpha}, "
            f"edge_zero_sec={edge_zero_sec}, edge_zero_samples={edge_n}, N={N}"
        )
        logger.info(
            f"[FORCE] raw |fz|max={raw_fz_abs_max:.3f} N, "
            f"processed |fz|max={proc_fz_abs_max:.3f} N"
        )

    return Fp


def ema_nd(X: np.ndarray, alpha: float) -> np.ndarray:
    if X.size == 0:
        return X.astype(np.float64).copy()

    Z = X.astype(np.float64).copy()
    if alpha <= 0.0 or alpha >= 1.0:
        return Z

    for i in range(1, Z.shape[0]):
        Z[i] = alpha * X[i] + (1.0 - alpha) * Z[i - 1]
    return Z


# ============================================================
# Keyboard watcher
# ============================================================
class KeyboardQuitter:
    """
    Press quit_key without Enter to request graceful stop.
    """
    def __init__(self, quit_key: str = "q"):
        self.quit_key = (quit_key or "q").lower()
        self._stop_evt = threading.Event()
        self._hit_quit = threading.Event()
        self._thread = None
        self._enabled = False
        self._fd = None
        self._old_term = None

    def start(self) -> bool:
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
# Main node
# ============================================================
class VRDemoHDF5Recorder(Node):
    def __init__(self):
        super().__init__("vr_demo_hdf5_recorder")

        # -------------------------
        # Save parameters
        # -------------------------
        self.declare_parameter("act_root_dir", "/home/eunseop/nrs_act/datasets/ACT")
        self.declare_parameter("merged_subdir", "merged_hdf5")
        self.declare_parameter("file_prefix", "vr_demo_merged")
        self.declare_parameter("overwrite", False)
        self.declare_parameter("flush_each_episode", True)

        self.declare_parameter("num_episodes", 50)
        self.declare_parameter("min_samples", 10)
        self.declare_parameter("quit_key", "q")

        # -------------------------
        # Topic parameters
        # -------------------------
        self.declare_parameter("pose_topic", "/calibrated_pose")
        self.declare_parameter("force_topic", "/ftsensor/measured_Cvalue")
        self.declare_parameter("image_topic", "/realsense/vr/color/image_raw")

        # -------------------------
        # Sampling / freshness
        # -------------------------
        self.declare_parameter("sample_hz", 30.0)
        self.declare_parameter("require_pose_fresh_sec", 0.20)
        self.declare_parameter("require_force_fresh_sec", 0.20)
        self.declare_parameter("require_image_fresh_sec", 0.50)

        # -------------------------
        # Unit convention
        # -------------------------
        self.declare_parameter("pose_xyz_scale", 1000.0)  # m -> mm

        # -------------------------
        # Episode trigger
        # -------------------------
        self.declare_parameter("start_abs_fx", 10.0)
        self.declare_parameter("stop_abs_fy", 10.0)

        # -------------------------
        # Force processing
        # -------------------------
        self.declare_parameter("zero_xy_forces", True)
        self.declare_parameter("fz_ema_alpha", 0.2)
        self.declare_parameter("force_edge_zero_sec", 3.0)

        # -------------------------
        # Optional pose smoothing
        # Default OFF because this recorder is now a synchronized
        # multimodal ACT recorder, not a robot-playback trajectory generator.
        # -------------------------
        self.declare_parameter("pose_ema_enable", False)
        self.declare_parameter("pose_ema_alpha", 0.10)

        # -------------------------
        # Image save
        # -------------------------
        self.declare_parameter("image_dataset_name", "cam0")
        self.declare_parameter("image_compression", "gzip")  # gzip, lzf, none
        self.declare_parameter("image_gzip_level", 4)

        # -------------------------
        # Load parameters
        # -------------------------
        self.act_root_dir = str(self.get_parameter("act_root_dir").value)
        self.merged_subdir = str(self.get_parameter("merged_subdir").value)
        self.file_prefix = str(self.get_parameter("file_prefix").value)
        self.overwrite = bool(self.get_parameter("overwrite").value)
        self.flush_each_episode = bool(self.get_parameter("flush_each_episode").value)

        self.num_episodes = int(self.get_parameter("num_episodes").value)
        self.min_samples = int(self.get_parameter("min_samples").value)
        self.quit_key = str(self.get_parameter("quit_key").value)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.force_topic = str(self.get_parameter("force_topic").value)
        self.image_topic = str(self.get_parameter("image_topic").value)

        self.sample_hz = float(self.get_parameter("sample_hz").value)
        self.dt = 1.0 / max(1e-9, self.sample_hz)

        self.require_pose_fresh_sec = float(self.get_parameter("require_pose_fresh_sec").value)
        self.require_force_fresh_sec = float(self.get_parameter("require_force_fresh_sec").value)
        self.require_image_fresh_sec = float(self.get_parameter("require_image_fresh_sec").value)

        self.pose_xyz_scale = float(self.get_parameter("pose_xyz_scale").value)

        self.start_abs_fx = float(self.get_parameter("start_abs_fx").value)
        self.stop_abs_fy = float(self.get_parameter("stop_abs_fy").value)

        self.zero_xy_forces = bool(self.get_parameter("zero_xy_forces").value)
        self.fz_ema_alpha = float(self.get_parameter("fz_ema_alpha").value)
        self.force_edge_zero_sec = float(self.get_parameter("force_edge_zero_sec").value)

        self.pose_ema_enable = bool(self.get_parameter("pose_ema_enable").value)
        self.pose_ema_alpha = float(self.get_parameter("pose_ema_alpha").value)

        self.image_dataset_name = str(self.get_parameter("image_dataset_name").value)
        self.image_compression = str(self.get_parameter("image_compression").value).lower()
        self.image_gzip_level = int(self.get_parameter("image_gzip_level").value)

        # -------------------------
        # HDF5 lazy-open state
        # -------------------------
        self.h5_lock = threading.Lock()
        self.h5: Optional[h5py.File] = None
        self.grp_eps: Optional[h5py.Group] = None
        self.hdf5_path: Optional[str] = None
        self.run_stamp: Optional[str] = None
        self.episode_count = 0

        # -------------------------
        # Runtime state
        # -------------------------
        self.state_lock = threading.Lock()

        self.latest_pose6: Optional[np.ndarray] = None
        self.latest_force3: Optional[np.ndarray] = None
        self.latest_image: Optional[np.ndarray] = None

        self.latest_pose_t: float = 0.0
        self.latest_force_t: float = 0.0
        self.latest_image_t: float = 0.0

        self.episode_active = False
        self.finishing = False
        self.stop_requested = False
        self.stop_reason = ""

        self.buf_position: List[np.ndarray] = []
        self.buf_ft_raw: List[np.ndarray] = []
        self.buf_images: List[Optional[np.ndarray]] = []

        # -------------------------
        # ROS IO
        # -------------------------
        self.create_subscription(Float64MultiArray, self.pose_topic, self.cb_pose, 100)
        self.create_subscription(Wrench, self.force_topic, self.cb_force, 200)
        self.create_subscription(Image, self.image_topic, self.cb_image, 10)

        self.create_timer(self.dt, self.cb_sample_timer)
        self.create_timer(1.0, self.cb_status_timer)
        self.create_timer(0.05, self.cb_stop_timer)

        # -------------------------
        # Keyboard
        # -------------------------
        self.kb = KeyboardQuitter(quit_key=self.quit_key)
        kb_enabled = self.kb.start()
        atexit.register(self.kb.stop)

        # -------------------------
        # Logs
        # -------------------------
        self.get_logger().info("============================================================")
        self.get_logger().info("VRDemoHDF5Recorder initialized (single-camera ACT merged recorder)")
        self.get_logger().info("HDF5 file is created lazily when the first episode is saved.")
        self.get_logger().info(f"  ACT root    : {self.act_root_dir}")
        self.get_logger().info(f"  merged dir  : <ACT root>/<YYYYMMDD_HHMM>/{self.merged_subdir}")
        self.get_logger().info(f"  filename    : {self.file_prefix}_YYYYMMDD_HHMM.hdf5")
        self.get_logger().info(f"  pose_topic  : {self.pose_topic}")
        self.get_logger().info(f"  force_topic : {self.force_topic}")
        self.get_logger().info(f"  image_topic : {self.image_topic}")
        self.get_logger().info(f"  image key   : images/{self.image_dataset_name}")
        self.get_logger().info(f"  sample_hz   : {self.sample_hz:.3f} Hz, dt={self.dt:.6f}s")
        self.get_logger().info(
            f"  freshness   : pose={self.require_pose_fresh_sec:.3f}s, "
            f"force={self.require_force_fresh_sec:.3f}s, image={self.require_image_fresh_sec:.3f}s"
        )
        self.get_logger().info(
            f"  trigger     : start=|Fx|>={self.start_abs_fx:.3f} N, "
            f"end=|Fy|>={self.stop_abs_fy:.3f} N"
        )
        self.get_logger().info(
            f"  force proc  : zero_xy={self.zero_xy_forces}, "
            f"fz_ema_alpha={self.fz_ema_alpha}, edge_zero_sec={self.force_edge_zero_sec}"
        )
        self.get_logger().info(
            f"  pose        : xyz_scale={self.pose_xyz_scale}, "
            f"pose_ema={self.pose_ema_enable}(alpha={self.pose_ema_alpha})"
        )
        self.get_logger().info(
            f"  image save  : compression={self.image_compression}, gzip_level={self.image_gzip_level}"
        )
        self.get_logger().info(f"  target eps  : {self.num_episodes}")
        if kb_enabled:
            self.get_logger().info(f"  Press '{self.quit_key}' to stop gracefully. Ctrl+C also works.")
        else:
            self.get_logger().warn("  stdin is not a TTY -> key quit disabled. Use Ctrl+C.")
        self.get_logger().info("============================================================")

    # ============================================================
    # HDF5 helpers
    # ============================================================
    @staticmethod
    def _stamp_now() -> str:
        return time.strftime("%Y%m%d_%H%M", time.localtime())

    def _build_hdf5_path(self, stamp: str) -> str:
        run_dir = os.path.join(self.act_root_dir, stamp, self.merged_subdir)
        os.makedirs(run_dir, exist_ok=True)
        return os.path.join(run_dir, f"{self.file_prefix}_{stamp}.hdf5")

    def _detect_existing_episode_count(self) -> int:
        if self.grp_eps is None:
            return 0

        max_idx = -1
        for k in self.grp_eps.keys():
            if not k.startswith("ep_"):
                continue
            try:
                idx = int(k.split("_")[1])
                max_idx = max(max_idx, idx)
            except Exception:
                pass
        return max_idx + 1

    def _ensure_hdf5_open(self):
        """
        Open HDF5 at first save time, not node start time.
        """
        if self.h5 is not None:
            return

        self.run_stamp = self._stamp_now()
        self.hdf5_path = self._build_hdf5_path(self.run_stamp)

        if self.overwrite and os.path.exists(self.hdf5_path):
            os.remove(self.hdf5_path)

        self.h5 = h5py.File(self.hdf5_path, "a")
        self.grp_eps = self.h5.require_group("episodes")
        self.episode_count = self._detect_existing_episode_count()

        self._write_root_meta()
        self.h5.flush()

        self.get_logger().info("============================================================")
        self.get_logger().info("[HDF5] opened")
        self.get_logger().info(f"  path          : {self.hdf5_path}")
        self.get_logger().info(f"  existing eps  : {self.episode_count}")
        self.get_logger().info("============================================================")

    def _write_root_meta(self):
        if self.h5 is None:
            return

        if "created_unix" not in self.h5.attrs:
            self.h5.attrs["created_unix"] = float(time.time())

        self.h5.attrs["format"] = np.string_("act_merged_hdf5_single_camera")
        self.h5.attrs["format_version"] = np.string_("1.0")
        self.h5.attrs["camera_names_json"] = np.string_(json.dumps([self.image_dataset_name]))
        self.h5.attrs["position_columns"] = np.string_("x_mm,y_mm,z_mm,wx,wy,wz")
        self.h5.attrs["ft_columns"] = np.string_("fx_N,fy_N,fz_N")
        self.h5.attrs["pose_note"] = np.string_("pose xyz input meters -> stored millimeters by pose_xyz_scale; wx wy wz stored as rotation-vector radians")
        self.h5.attrs["image_note"] = np.string_("RGB uint8, shape=(T,H,W,3)")
        self.h5.attrs["sample_hz"] = float(self.sample_hz)
        self.h5.attrs["dt"] = float(self.dt)
        self.h5.attrs["pose_topic"] = np.string_(self.pose_topic)
        self.h5.attrs["force_topic"] = np.string_(self.force_topic)
        self.h5.attrs["image_topic"] = np.string_(self.image_topic)
        self.h5.attrs["episode_rule"] = np.string_(f"start=|fx|>={self.start_abs_fx}, end=|fy|>={self.stop_abs_fy}")
        self.h5.attrs["force_processing"] = np.string_(
            f"zero_xy={self.zero_xy_forces}, fz_ema_alpha={self.fz_ema_alpha}, edge_zero_sec={self.force_edge_zero_sec}"
        )

    def _image_create_kwargs(self):
        if self.image_compression == "none":
            return {}
        if self.image_compression == "lzf":
            return {"compression": "lzf", "shuffle": True}
        if self.image_compression == "gzip":
            return {
                "compression": "gzip",
                "compression_opts": int(self.image_gzip_level),
                "shuffle": True,
            }

        self.get_logger().warn(
            f"[HDF5] unknown image_compression={self.image_compression}, using gzip"
        )
        return {
            "compression": "gzip",
            "compression_opts": int(self.image_gzip_level),
            "shuffle": True,
        }

    def _save_episode_to_hdf5(
        self,
        ep_idx: int,
        position: np.ndarray,
        ft: np.ndarray,
        images: np.ndarray,
        reason: str,
    ):
        with self.h5_lock:
            self._ensure_hdf5_open()

            assert self.h5 is not None
            assert self.grp_eps is not None

            ep_name = f"ep_{ep_idx:04d}"
            if ep_name in self.grp_eps:
                del self.grp_eps[ep_name]

            g = self.grp_eps.create_group(ep_name)
            g.attrs["saved_unix"] = float(time.time())
            g.attrs["reason"] = np.string_(str(reason))
            g.attrs["out_len"] = int(position.shape[0])
            g.attrs["sample_hz"] = float(self.sample_hz)
            g.attrs["dt"] = float(self.dt)
            g.attrs["pose_xyz_scale"] = float(self.pose_xyz_scale)
            g.attrs["zero_xy_forces"] = int(bool(self.zero_xy_forces))
            g.attrs["fz_ema_alpha"] = float(self.fz_ema_alpha)
            g.attrs["force_edge_zero_sec"] = float(self.force_edge_zero_sec)
            g.attrs["pose_ema_enable"] = int(bool(self.pose_ema_enable))
            g.attrs["pose_ema_alpha"] = float(self.pose_ema_alpha)
            g.attrs["image_dataset_name"] = np.string_(self.image_dataset_name)
            g.attrs["image_shape"] = np.array(images.shape[1:], dtype=np.int64)

            g.create_dataset(
                "position",
                data=position.astype(np.float32),
                dtype="float32",
            )
            g.create_dataset(
                "ft",
                data=ft.astype(np.float32),
                dtype="float32",
            )

            img_grp = g.create_group("images")
            img_grp.create_dataset(
                self.image_dataset_name,
                data=images.astype(np.uint8),
                dtype="uint8",
                chunks=(1, images.shape[1], images.shape[2], images.shape[3]),
                **self._image_create_kwargs(),
            )

            if self.flush_each_episode:
                self.h5.flush()

    # ============================================================
    # Stop / finalization
    # ============================================================
    def request_stop(self, reason: str = "user_request"):
        if self.stop_requested:
            return
        self.stop_requested = True
        self.stop_reason = str(reason)
        self.get_logger().warn(f"[STOP REQUEST] reason={self.stop_reason}")

    def cb_stop_timer(self):
        if self.kb.hit() and not self.stop_requested:
            self.request_stop(reason=f"keyboard_{self.quit_key}")

        if self.stop_requested and (not self.finishing) and (not self.episode_active):
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
    # ROS callbacks
    # ============================================================
    def cb_pose(self, msg: Float64MultiArray):
        if len(msg.data) < 6:
            return

        x, y, z, wx, wy, wz = msg.data[:6]
        pose = np.array(
            [
                self.pose_xyz_scale * float(x),
                self.pose_xyz_scale * float(y),
                self.pose_xyz_scale * float(z),
                float(wx),
                float(wy),
                float(wz),
            ],
            dtype=np.float64,
        )

        now = time.time()
        with self.state_lock:
            self.latest_pose6 = pose
            self.latest_pose_t = now

    def cb_force(self, msg: Wrench):
        fx = float(msg.force.x)
        fy = float(msg.force.y)
        fz = float(msg.force.z)
        F = np.array([fx, fy, fz], dtype=np.float64)

        now = time.time()
        with self.state_lock:
            self.latest_force3 = F
            self.latest_force_t = now

        if self.stop_requested:
            return

        if self.finishing:
            return

        # Start trigger uses raw Fx.
        if (not self.episode_active) and (abs(fx) >= self.start_abs_fx):
            self._start_episode()
            return

        # End trigger uses raw Fy.
        if self.episode_active and (abs(fy) >= self.stop_abs_fy):
            self.get_logger().info(
                f"=== EPISODE ENDED (idx={self.episode_count:04d}) by |Fy| threshold ==="
            )
            self._start_finish_thread(reason="fy_threshold")
            return

    def cb_image(self, msg: Image):
        img = image_to_rgb_numpy(msg)
        if img is None:
            self.get_logger().warn(
                f"[IMAGE] unsupported or invalid image encoding='{msg.encoding}' "
                f"shape=({msg.height},{msg.width}), step={msg.step}",
                throttle_duration_sec=2.0,
            )
            return

        now = time.time()
        with self.state_lock:
            self.latest_image = img
            self.latest_image_t = now

    def cb_sample_timer(self):
        if self.stop_requested and self.episode_active and (not self.finishing):
            self.get_logger().warn("Stop requested while recording -> closing current episode.")
            self._start_finish_thread(reason=self.stop_reason or "stop_requested")
            return

        if (not self.episode_active) or self.finishing or self.stop_requested:
            return

        now = time.time()

        with self.state_lock:
            if self.latest_pose6 is None:
                return
            if self.latest_force3 is None:
                return
            if self.latest_image is None:
                return

            pose_age = now - self.latest_pose_t
            force_age = now - self.latest_force_t
            image_age = now - self.latest_image_t

            if pose_age > self.require_pose_fresh_sec:
                self.get_logger().warn(
                    f"[SKIP] pose stale: age={pose_age:.3f}s",
                    throttle_duration_sec=2.0,
                )
                return

            if force_age > self.require_force_fresh_sec:
                self.get_logger().warn(
                    f"[SKIP] force stale: age={force_age:.3f}s",
                    throttle_duration_sec=2.0,
                )
                return

            if image_age > self.require_image_fresh_sec:
                self.get_logger().warn(
                    f"[SKIP] image stale: age={image_age:.3f}s",
                    throttle_duration_sec=2.0,
                )
                return

            self.buf_position.append(self.latest_pose6.copy())
            self.buf_ft_raw.append(self.latest_force3.copy())
            self.buf_images.append(self.latest_image.copy())

    def cb_status_timer(self):
        with self.state_lock:
            steps = len(self.buf_position)
            active = bool(self.episode_active)
            finishing = bool(self.finishing)
            has_pose = self.latest_pose6 is not None
            has_force = self.latest_force3 is not None
            has_img = self.latest_image is not None

        self.get_logger().info(
            f"[STATUS] active={active}, finishing={finishing}, "
            f"ep_idx={self.episode_count:04d}, steps={steps}, "
            f"latest(pose={has_pose}, force={has_force}, image={has_img}), "
            f"h5={'not_opened' if self.hdf5_path is None else self.hdf5_path}"
        )

    # ============================================================
    # Episode control
    # ============================================================
    def _start_episode(self):
        with self.state_lock:
            self.episode_active = True
            self.buf_position.clear()
            self.buf_ft_raw.clear()
            self.buf_images.clear()

        self.get_logger().info(
            f"=== EPISODE STARTED (idx={self.episode_count:04d}) by |Fx| threshold ==="
        )

    def _start_finish_thread(self, reason: str):
        if self.finishing:
            return

        self.finishing = True

        with self.state_lock:
            self.episode_active = False

            P_list = self.buf_position.copy()
            F_list = self.buf_ft_raw.copy()
            I_list = self.buf_images.copy()

            self.buf_position.clear()
            self.buf_ft_raw.clear()
            self.buf_images.clear()

        th = threading.Thread(
            target=self._finish_episode_worker,
            args=(P_list, F_list, I_list, reason),
            daemon=True,
        )
        th.start()

    def _finish_episode_worker(
        self,
        P_list: List[np.ndarray],
        F_list: List[np.ndarray],
        I_list: List[Optional[np.ndarray]],
        reason: str,
    ):
        try:
            Np = len(P_list)
            Nf = len(F_list)
            Ni = len(I_list)
            N = min(Np, Nf, Ni)

            if N < max(1, self.min_samples):
                self.get_logger().warn(
                    f"Episode dropped: too short. "
                    f"N={N}, min_samples={self.min_samples}, "
                    f"Np={Np}, Nf={Nf}, Ni={Ni}, reason={reason}"
                )
                return

            # All lists are appended together inside cb_sample_timer,
            # so lengths should match. Still trim defensively.
            P = np.asarray(P_list[:N], dtype=np.float64)
            Fraw = np.asarray(F_list[:N], dtype=np.float64)
            images = stack_images_repeat_last(I_list[:N], logger=self.get_logger())

            if images is None:
                self.get_logger().warn(
                    f"Episode dropped: no valid image frames. N={N}, reason={reason}"
                )
                return

            if self.pose_ema_enable:
                P_out = ema_nd(P, alpha=self.pose_ema_alpha)
            else:
                P_out = P.copy()

            F_out = process_force_keep_fz_with_ema_and_edge_zero(
                Fraw,
                fz_ema_alpha=self.fz_ema_alpha,
                edge_zero_sec=self.force_edge_zero_sec,
                sample_hz=self.sample_hz,
                zero_xy=self.zero_xy_forces,
                logger=self.get_logger(),
            )

            ep_idx = self.episode_count
            self._save_episode_to_hdf5(
                ep_idx=ep_idx,
                position=P_out,
                ft=F_out,
                images=images,
                reason=reason,
            )

            self.episode_count += 1

            self.get_logger().info(
                f"=== EPISODE SAVED (idx={ep_idx:04d}) "
                f"N={N}, position={P_out.shape}, ft={F_out.shape}, images={images.shape}, "
                f"reason={reason} ==="
            )

            if self.episode_count >= self.num_episodes:
                self.request_stop(reason="reached_num_episodes")

        except Exception as e:
            self.get_logger().error(f"Episode processing failed: {repr(e)}")
        finally:
            self.finishing = False


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