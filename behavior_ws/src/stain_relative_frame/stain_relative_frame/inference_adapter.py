#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step [5]: the inference-side consumer of the latched `stain_origin`.

This is the ONLY thing an inference node needs to add. It subscribes to the
latched origin topic published by `stain_origin_node`, builds a
`RelativeFrameAdapter` from it -- the same `to_relative` / `to_absolute` the
dataset converter used in step [3] -- and hands back two calls:

    from stain_relative_frame.inference_adapter import StainOriginClient

    client = StainOriginClient(node, stats_path=ckpt_dir + "/dataset_stats.pkl")
    client.wait_until_ready(timeout_sec=30.0)      # once, before the episode

    qpos_rel = client.observation(qpos_abs)        # per step: policy input
    cmd_abs  = client.command(action_rel)          # per step: robot command

`use_relative_position` is read from the CHECKPOINT's dataset_stats.pkl, not
from a launch argument, so a policy trained on absolute coordinates cannot be
driven through the relative path or the other way round -- the mismatch is an
exception at startup instead of a silently wrong trajectory.

The origin is fetched ONCE and frozen. There is no method that updates it;
`observation()` and `command()` are pure functions of the frozen value.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from .relative_frame import (
    TRANSFORM_VERSION,
    TRANSFORM_VERSION_ATTR,
    USE_RELATIVE_ATTR,
    RelativeFrameAdapter,
    StainOrigin,
    to_absolute,
    to_relative,
)

DEFAULT_ORIGIN_TOPIC = "/stain_relative_frame/stain_origin"


def read_use_relative(stats_path) -> tuple:
    """(use_relative_position, transform_version) from a checkpoint's stats."""
    p = Path(stats_path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"dataset_stats.pkl not found: {p}")
    with open(p, "rb") as f:
        stats = pickle.load(f)
    use_relative = bool(stats.get(USE_RELATIVE_ATTR, False))
    version = str(stats.get(TRANSFORM_VERSION_ATTR, TRANSFORM_VERSION))
    return use_relative, version


class StainOriginClient:
    """Latched-origin subscriber + frozen `RelativeFrameAdapter`."""

    def __init__(self, node, stats_path=None, use_relative: Optional[bool] = None,
                 origin_topic: str = DEFAULT_ORIGIN_TOPIC):
        from rclpy.qos import (
            DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
        )
        from std_msgs.msg import Float64MultiArray

        if use_relative is None:
            if stats_path is None:
                raise ValueError("pass stats_path or an explicit use_relative")
            use_relative, version = read_use_relative(stats_path)
        else:
            version = TRANSFORM_VERSION
        self._use_relative = bool(use_relative)
        self._version = version
        self._adapter: Optional[RelativeFrameAdapter] = None
        self._node = node

        if not self._use_relative:
            # Absolute-coordinate policy: build the identity adapter now and
            # never touch the topic. The per-step calls stay identical, so the
            # inference node has ONE code path regardless of the flag.
            self._adapter = RelativeFrameAdapter([0.0, 0.0], use_relative=False,
                                                 transform_version=version)
            node.get_logger().info(
                "[SRF] use_relative_position=False -> identity transform "
                "(matching the checkpoint this policy was trained with)"
            )
            return

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._sub = node.create_subscription(
            Float64MultiArray, origin_topic, self._on_origin, qos
        )
        node.get_logger().info(
            f"[SRF] use_relative_position=True -> waiting for {origin_topic}"
        )

    # ------------------------------------------------------------------
    def _on_origin(self, msg) -> None:
        if self._adapter is not None:
            return  # already frozen; a republish cannot move it
        xy = np.asarray(msg.data, dtype=np.float64).reshape(-1)[:2]
        self._adapter = RelativeFrameAdapter(
            StainOrigin(xy, source="latched_topic"),
            use_relative=True, transform_version=self._version,
        )
        self._node.destroy_subscription(self._sub)  # ★ one-shot, like the publisher
        self._sub = None
        self._node.get_logger().info(
            f"[SRF] stain_origin = ({xy[0]:.3f}, {xy[1]:.3f}) mm -- FROZEN for this "
            f"episode; subscription released"
        )

    @property
    def ready(self) -> bool:
        return self._adapter is not None

    @property
    def use_relative(self) -> bool:
        return self._use_relative

    @property
    def stain_origin(self) -> np.ndarray:
        if self._adapter is None:
            raise RuntimeError("stain_origin not received yet")
        return self._adapter.stain_origin

    def wait_until_ready(self, timeout_sec: float = 30.0) -> bool:
        import rclpy
        import time as _time

        deadline = _time.monotonic() + float(timeout_sec)
        while not self.ready and _time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.1)
        if not self.ready:
            raise RuntimeError(
                f"no stain_origin within {timeout_sec}s. Is stain_origin_node running, "
                f"and did it resolve the origin at the home pose?"
            )
        return True

    # ---- the two per-step calls --------------------------------------
    def observation(self, pose_abs: np.ndarray) -> np.ndarray:
        """Absolute robot pose -> what the policy was trained on."""
        if self._adapter is None:
            raise RuntimeError("stain_origin not received yet -- call wait_until_ready()")
        return self._adapter.observation(pose_abs)

    def command(self, pose_rel: np.ndarray) -> np.ndarray:
        """Policy output -> absolute robot command. Always call before sending."""
        if self._adapter is None:
            raise RuntimeError("stain_origin not received yet -- call wait_until_ready()")
        return self._adapter.command(pose_rel)

    def stats(self) -> dict:
        return {} if self._adapter is None else self._adapter.stats()


__all__ = [
    "StainOriginClient", "read_use_relative", "DEFAULT_ORIGIN_TOPIC",
    "to_relative", "to_absolute",
]
