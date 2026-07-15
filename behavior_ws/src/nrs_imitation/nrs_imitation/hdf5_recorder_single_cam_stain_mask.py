#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-camera HDF5 recorder that also stores a polishing stain mask.

This wrapper leaves hdf5_recorder_single_cam.py and hdf5_recorder_base.py
unchanged. It records the same cam0 RGB stream as the baseline recorder.

Default workflow:

  ep_0000 is recorded without stain and marked as a clean reference episode.
  demo_data_imitation_form_single_cam.py then compares later stained episodes
  against ep_0000 to generate observations/images/stain_mask and blob proposals.

Optional rgb_threshold mode can still write a rough online mask:

  episodes/ep_xxxx/images/stain_mask    (T, H, W, 1) uint8, values 0 or 255

The online mask is generated from cam0 RGB with a small rule-based prior:

  stain_candidate   = dark_region
  reflection_region = very_bright_region with low saturation
  stain_mask        = stain_candidate AND NOT reflection_region
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

import numpy as np

import rclpy

from nrs_imitation.hdf5_recorder_base import HDF5Recorder, cv2
from nrs_imitation.pretty_print import block


FIXED_DEFAULTS = {
    "recording_mode": "tracker",
    "pose_topic": "/calibrated_pose",
    "force_topic": "/ftsensor/measured_Cvalue",
    "force_msg_type": "wrench",
    "image_topic": "/realsense/vr/color/image_raw",
    "enable_global_cam": False,
    "file_prefix": "hdf5_recorder_single_cam_stain_mask",
}


def _as_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"RGB image must be (H,W,3), got {arr.shape}")
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if float(np.nanmax(arr)) <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _value_and_saturation(rgb_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if cv2 is not None:
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
        return hsv[:, :, 2], hsv[:, :, 1]

    arr = rgb_u8.astype(np.float32)
    vmax = np.max(arr, axis=2)
    vmin = np.min(arr, axis=2)
    sat = np.zeros_like(vmax, dtype=np.float32)
    valid = vmax > 1e-6
    sat[valid] = (vmax[valid] - vmin[valid]) / vmax[valid] * 255.0
    return vmax.astype(np.uint8), np.clip(sat, 0, 255).astype(np.uint8)


def _morphology(mask_u8: np.ndarray, kernel_size: int) -> np.ndarray:
    k = int(kernel_size)
    if k <= 1 or cv2 is None:
        return mask_u8
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), dtype=np.uint8)
    opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def _filter_small_components(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    min_area = int(min_area)
    if min_area <= 0 or cv2 is None:
        return mask_u8

    binary = (mask_u8 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return mask_u8

    out = np.zeros_like(mask_u8, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == label] = 255
    return out


def generate_stain_mask_from_rgb(
    rgb: np.ndarray,
    stain_dark_thresh: int = 80,
    reflection_v_thresh: int = 235,
    reflection_s_thresh: int = 60,
    stain_min_area: int = 20,
    stain_morph_kernel: int = 3,
) -> np.ndarray:
    """
    Generate one stain mask from an RGB frame.

    Returns:
        (H, W) uint8 mask with values 0 or 255.
    """
    rgb_u8 = _as_uint8_rgb(rgb)
    value, saturation = _value_and_saturation(rgb_u8)

    dark_region = value < int(stain_dark_thresh)
    reflection_region = (value > int(reflection_v_thresh)) & (saturation < int(reflection_s_thresh))
    mask = (dark_region & (~reflection_region)).astype(np.uint8) * 255

    mask = _morphology(mask, stain_morph_kernel)
    mask = _filter_small_components(mask, stain_min_area)
    return mask.astype(np.uint8)


def generate_stain_mask_sequence_from_rgb(
    images_rgb: np.ndarray,
    stain_dark_thresh: int = 80,
    reflection_v_thresh: int = 235,
    reflection_s_thresh: int = 60,
    stain_min_area: int = 20,
    stain_morph_kernel: int = 3,
) -> np.ndarray:
    """
    Generate masks for a cam0 RGB sequence.

    Args:
        images_rgb: (T, H, W, 3) uint8 or float RGB.

    Returns:
        (T, H, W, 1) uint8 mask with values 0 or 255.
    """
    arr = np.asarray(images_rgb)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"images_rgb must be (T,H,W,3), got {arr.shape}")

    masks = [
        generate_stain_mask_from_rgb(
            arr[i],
            stain_dark_thresh=stain_dark_thresh,
            reflection_v_thresh=reflection_v_thresh,
            reflection_s_thresh=reflection_s_thresh,
            stain_min_area=stain_min_area,
            stain_morph_kernel=stain_morph_kernel,
        )
        for i in range(arr.shape[0])
    ]
    return np.stack(masks, axis=0)[:, :, :, None].astype(np.uint8)


def make_stain_mask_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb_u8 = _as_uint8_rgb(rgb).astype(np.float32)
    mask_2d = np.asarray(mask)
    if mask_2d.ndim == 3:
        mask_2d = mask_2d[:, :, 0]
    m = (mask_2d > 0).astype(np.float32)[:, :, None]
    red = np.zeros_like(rgb_u8)
    red[:, :, 0] = 255.0
    out = rgb_u8 * (1.0 - alpha * m) + red * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


class StainMaskHDF5Recorder(HDF5Recorder):
    def __init__(self, node_name: str, fixed_defaults: Optional[Dict[str, object]] = None):
        super().__init__(node_name=node_name, fixed_defaults=fixed_defaults)

        self.declare_parameter("use_stain_mask", True)
        self.declare_parameter("stain_mask_source", "reference_episode")
        self.declare_parameter("stain_mask_dataset_name", "stain_mask")
        self.declare_parameter("stain_reference_first_episode", True)
        self.declare_parameter("stain_reference_episode_index", 0)
        self.declare_parameter("stain_dark_thresh", 80)
        self.declare_parameter("reflection_v_thresh", 235)
        self.declare_parameter("reflection_s_thresh", 60)
        self.declare_parameter("stain_min_area", 20)
        self.declare_parameter("stain_morph_kernel", 3)
        self.declare_parameter("stain_debug_save_samples", 0)
        self.declare_parameter(
            "stain_debug_dir",
            os.path.expanduser("~/nrs_imitation/debug/stain_mask_samples"),
        )

        self.use_stain_mask = bool(self.get_parameter("use_stain_mask").value)
        self.stain_mask_source = str(self.get_parameter("stain_mask_source").value).strip().lower()
        self.stain_mask_dataset_name = str(self.get_parameter("stain_mask_dataset_name").value).strip()
        self.stain_reference_first_episode = bool(self.get_parameter("stain_reference_first_episode").value)
        self.stain_reference_episode_index = int(self.get_parameter("stain_reference_episode_index").value)
        self.stain_dark_thresh = int(self.get_parameter("stain_dark_thresh").value)
        self.reflection_v_thresh = int(self.get_parameter("reflection_v_thresh").value)
        self.reflection_s_thresh = int(self.get_parameter("reflection_s_thresh").value)
        self.stain_min_area = int(self.get_parameter("stain_min_area").value)
        self.stain_morph_kernel = int(self.get_parameter("stain_morph_kernel").value)
        self.stain_debug_save_samples = int(self.get_parameter("stain_debug_save_samples").value)
        self.stain_debug_dir = os.path.expanduser(str(self.get_parameter("stain_debug_dir").value))

        if self.stain_mask_source not in ("reference_episode", "rgb_threshold", "auto", "hdf5", "none"):
            self.get_logger().warn(
                f"[STAIN_MASK] unknown stain_mask_source={self.stain_mask_source}; using reference_episode"
            )
            self.stain_mask_source = "reference_episode"
        if self.stain_mask_source == "hdf5":
            self.get_logger().warn(
                "[STAIN_MASK] stain_mask_source=hdf5 is a loader-side concept; "
                "this recorder will use reference_episode metadata instead."
            )
            self.stain_mask_source = "reference_episode"
        if self.stain_mask_dataset_name == "":
            self.stain_mask_dataset_name = "stain_mask"

        with self.h5_lock:
            self.h5.attrs["use_stain_mask"] = int(self.use_stain_mask)
            self.h5.attrs["stain_mask_source"] = str(self.stain_mask_source)
            self.h5.attrs["stain_mask_dataset_name"] = str(self.stain_mask_dataset_name)
            self.h5.attrs["stain_reference_first_episode"] = int(self.stain_reference_first_episode)
            self.h5.attrs["stain_reference_episode_index"] = int(self.stain_reference_episode_index)
            self.h5.attrs["stain_reference_episode"] = f"ep_{int(self.stain_reference_episode_index):04d}"
            self.h5.attrs["stain_dark_thresh"] = int(self.stain_dark_thresh)
            self.h5.attrs["reflection_v_thresh"] = int(self.reflection_v_thresh)
            self.h5.attrs["reflection_s_thresh"] = int(self.reflection_s_thresh)
            self.h5.attrs["stain_min_area"] = int(self.stain_min_area)
            self.h5.attrs["stain_morph_kernel"] = int(self.stain_morph_kernel)
            self.h5.attrs["stain_mask_storage"] = "uint8_0_255"

        if cv2 is None and (self.stain_morph_kernel > 1 or self.stain_min_area > 0):
            self.get_logger().warn(
                "[STAIN_MASK] cv2 is not available; morphology and connected component "
                "filtering are disabled."
            )

        self.get_logger().info(block("STAIN MASK RECORDER READY", [
            ("enabled", int(self.use_stain_mask)),
            ("source", self.stain_mask_source),
            ("reference_first_ep", int(self.stain_reference_first_episode)),
            ("reference_ep", f"ep_{int(self.stain_reference_episode_index):04d}"),
            ("dataset", f"images/{self.stain_mask_dataset_name}"),
            ("dark_thresh", self.stain_dark_thresh),
            ("reflection_v", self.reflection_v_thresh),
            ("reflection_s", self.reflection_s_thresh),
            ("min_area", self.stain_min_area),
            ("morph_kernel", self.stain_morph_kernel),
            ("debug_samples", self.stain_debug_save_samples),
        ]))

    def _generate_masks(self, images0: np.ndarray) -> Optional[np.ndarray]:
        if not self.use_stain_mask or self.stain_mask_source == "none":
            return None
        if self.stain_mask_source == "reference_episode":
            return None
        return generate_stain_mask_sequence_from_rgb(
            images0,
            stain_dark_thresh=self.stain_dark_thresh,
            reflection_v_thresh=self.reflection_v_thresh,
            reflection_s_thresh=self.reflection_s_thresh,
            stain_min_area=self.stain_min_area,
            stain_morph_kernel=self.stain_morph_kernel,
        )

    def _save_debug_samples(self, ep_name: str, images0: np.ndarray, masks: np.ndarray):
        if self.stain_debug_save_samples <= 0:
            return
        if cv2 is None:
            self.get_logger().warn("[STAIN_MASK] cv2 is not available; debug PNG save skipped.")
            return

        n = min(int(self.stain_debug_save_samples), int(images0.shape[0]))
        out_dir = os.path.join(self.stain_debug_dir, self.timestamp, ep_name)
        os.makedirs(out_dir, exist_ok=True)

        for i in range(n):
            rgb = _as_uint8_rgb(images0[i])
            mask = masks[i, :, :, 0]
            overlay = make_stain_mask_overlay(rgb, mask)
            cv2.imwrite(os.path.join(out_dir, f"{i:04d}_rgb.png"), rgb[:, :, ::-1])
            cv2.imwrite(os.path.join(out_dir, f"{i:04d}_mask.png"), mask)
            cv2.imwrite(os.path.join(out_dir, f"{i:04d}_overlay.png"), overlay[:, :, ::-1])

        self.get_logger().info(f"[STAIN_MASK] saved debug samples -> {out_dir}")

    def _save_episode_to_hdf5(self, ep_idx, position, ft, images0, images1, sample_times, reason, **kwargs):
        is_reference_episode = (
            self.use_stain_mask
            and self.stain_reference_first_episode
            and int(ep_idx) == int(self.stain_reference_episode_index)
        )
        masks = self._generate_masks(images0)
        if is_reference_episode:
            masks = None
        if masks is not None:
            mask_float = masks.astype(np.float32) / 255.0
            self.get_logger().info(
                "[STAIN_MASK] "
                f"rgb_shape={tuple(images0.shape)} "
                f"mask_shape={tuple(masks.shape)} "
                f"min={float(mask_float.min()):.4f} "
                f"max={float(mask_float.max()):.4f} "
                f"mean={float(mask_float.mean()):.4f}"
            )

        super()._save_episode_to_hdf5(ep_idx, position, ft, images0, images1, sample_times, reason, **kwargs)

        ep_name = self._ep_name(ep_idx)
        if is_reference_episode:
            with self.h5_lock:
                g = self.grp_eps[ep_name]
                g.attrs["stain_reference_episode"] = 1
                g.attrs["exclude_from_imitation_training"] = 1
                g.attrs["stain_reference_role"] = "clean_surface_reference"
                if self.flush_each_episode:
                    self.h5.flush()
            self.get_logger().info(
                f"[STAIN_MASK] {ep_name} marked as clean reference episode; "
                "demo_data_imitation_form_single_cam.py will use it to generate masks."
            )

        if masks is None:
            return

        with self.h5_lock:
            g = self.grp_eps[ep_name]
            g_img = g["images"]
            if self.stain_mask_dataset_name in g_img:
                del g_img[self.stain_mask_dataset_name]

            ds = g_img.create_dataset(
                self.stain_mask_dataset_name,
                data=masks.astype(np.uint8),
                **self._compression_kwargs(),
            )
            ds.attrs["description"] = "Rule-based polishing stain mask generated from cam0 RGB"
            ds.attrs["shape_convention"] = "T,H,W,1"
            ds.attrs["storage"] = "uint8_0_255"
            ds.attrs["model_value_range_after_div255"] = "float32_0_1"
            ds.attrs["source_image_dataset"] = str(self.image_dataset_name)
            ds.attrs["stain_dark_thresh"] = int(self.stain_dark_thresh)
            ds.attrs["reflection_v_thresh"] = int(self.reflection_v_thresh)
            ds.attrs["reflection_s_thresh"] = int(self.reflection_s_thresh)
            ds.attrs["stain_min_area"] = int(self.stain_min_area)
            ds.attrs["stain_morph_kernel"] = int(self.stain_morph_kernel)
            g.attrs["has_stain_mask"] = 1
            g.attrs["stain_mask_dataset"] = f"images/{self.stain_mask_dataset_name}"
            if self.flush_each_episode:
                self.h5.flush()

        self._save_debug_samples(ep_name, images0, masks)


def spin_stain_mask_recorder(node_name: str, fixed_defaults: Optional[Dict[str, object]] = None, args=None):
    rclpy.init(args=args)
    node = StainMaskHDF5Recorder(node_name=node_name, fixed_defaults=fixed_defaults)
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


def main(args=None):
    spin_stain_mask_recorder(
        "hdf5_recorder_single_cam_stain_mask",
        fixed_defaults=FIXED_DEFAULTS,
        args=args,
    )


if __name__ == "__main__":
    main()
