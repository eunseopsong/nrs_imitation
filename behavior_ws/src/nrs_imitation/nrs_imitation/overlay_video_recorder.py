#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Records the inference-time diagnostic overlays (flow-vector overlay on the
left, modality-importance dashboard on the right) into a single side-by-side
webm video -- instead of a raw desktop screen capture, which depends on
window layout/what else happens to be on screen.

Both topics are sensor_msgs/Image, published as contiguous rgb8 by
inference_core.py's _rgb_numpy_to_image_msg(). Frames are composited
side-by-side at equal (1:1) panel width -- the narrower one is scaled up to
match the wider one's width (aspect ratio preserved, not stretched), then
heights are matched via centered black padding -- on a fixed-rate timer that
always uses the latest frame received on each topic. This decouples the
output video's frame pacing from each topic's own (irregular) publish rate.
The ffmpeg process is started lazily, once at least one frame has arrived on
each topic (so the composite size is known).
"""
import os
import signal
import subprocess
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


def _pad_to_height(arr: np.ndarray, height: int) -> np.ndarray:
    h = arr.shape[0]
    if h == height:
        return arr
    top = (height - h) // 2
    bottom = height - h - top
    return np.pad(arr, ((top, bottom), (0, 0), (0, 0)), mode="constant", constant_values=0)


def _resize_to_width(arr: np.ndarray, width: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if w == width:
        return arr
    new_h = max(1, int(round(h * (width / w))))
    return cv2.resize(arr, (width, new_h), interpolation=cv2.INTER_AREA if width < w else cv2.INTER_LINEAR)


def _fit_into(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Resize arr to fit inside out_w x out_h (aspect preserved), then
    center-pad with black to exactly that size."""
    h, w = arr.shape[:2]
    scale = min(out_w / w, out_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    y0, x0 = (out_h - nh) // 2, (out_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


class OverlayVideoRecorderNode(Node):
    def __init__(self):
        super().__init__("overlay_video_recorder")

        self.declare_parameter("enable", True)
        self.declare_parameter("output_dir", os.path.expanduser("~/Videos/Screencasts"))
        self.declare_parameter("file_prefix", "polishing_inference")
        self.declare_parameter("policy_class", "unknown")
        self.declare_parameter("run_tag", "")
        self.declare_parameter("framerate", 30.0)
        self.declare_parameter("bitrate_mbps", 4.0)
        # The overlays are only recomputed once per policy replan (a few
        # seconds apart, not every control tick) -- raising the output
        # framerate alone just re-writes the same static frame, it can't
        # make an update that's genuinely infrequent look smooth. Instead
        # we crossfade from the previous composite to each new one over
        # this many seconds, so the real (slow) update cadence reads as a
        # smooth transition rather than an abrupt jump-cut.
        self.declare_parameter("transition_sec", 0.6)
        self.declare_parameter("modality_importance_topic", "/inference_single_cam/modality_importance")
        self.declare_parameter("flow_vector_overlay_topic", "/inference_single_cam/flow_vector_overlay")
        self.declare_parameter("record_modality_importance", True)
        self.declare_parameter("record_flow_vector_overlay", True)
        # Tack the polishing-removal heatmap onto the end of the video.
        # polishing_removal_recorder writes it on the same shutdown; we poll
        # for the PNG (up to grace_sec) then hold it for tail_sec.
        self.declare_parameter("append_removal_heatmap", True)
        self.declare_parameter("removal_heatmap_tail_sec", 15.0)
        self.declare_parameter("removal_heatmap_grace_sec", 8.0)
        self.declare_parameter("removal_heatmap_glob",
                               os.path.expanduser("~/nrs_imitation/logs/polishing_removal/*/01_removal_heatmap.png"))

        self.enable = bool(self.get_parameter("enable").value)
        self.proc = None
        self.out_path = None
        self._log_fh = None
        self._frames_written = 0
        self._out_w = None
        self._out_h = None
        self._start_wall = time.time()
        self.latest_flow = None
        self.latest_modality = None
        self.append_removal_heatmap = bool(self.get_parameter("append_removal_heatmap").value)
        self.removal_tail_sec = float(self.get_parameter("removal_heatmap_tail_sec").value)
        self.removal_grace_sec = float(self.get_parameter("removal_heatmap_grace_sec").value)
        self.removal_glob = str(self.get_parameter("removal_heatmap_glob").value)
        self.want_flow = bool(self.get_parameter("record_flow_vector_overlay").value)
        self.want_modality = bool(self.get_parameter("record_modality_importance").value)

        if not self.enable or not (self.want_flow or self.want_modality):
            self.get_logger().info("overlay_video_recorder disabled (enable:=false or both topics off)")
            return

        self.output_dir = os.path.expanduser(str(self.get_parameter("output_dir").value))
        os.makedirs(self.output_dir, exist_ok=True)
        self.file_prefix = str(self.get_parameter("file_prefix").value)
        self.policy_class = str(self.get_parameter("policy_class").value)
        self.run_tag = str(self.get_parameter("run_tag").value)
        self.framerate = float(self.get_parameter("framerate").value)
        self.bitrate_mbps = float(self.get_parameter("bitrate_mbps").value)
        self.transition_sec = float(self.get_parameter("transition_sec").value)
        self._prev_composite = None   # np.float32 (H,W,3), blend start point
        self._target_composite = None  # np.uint8 (H,W,3), latest real composite
        self._target_time = None       # time.monotonic() when target changed

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        if self.want_flow:
            topic = str(self.get_parameter("flow_vector_overlay_topic").value)
            self.create_subscription(Image, topic, self._cb_flow, qos)
        if self.want_modality:
            topic = str(self.get_parameter("modality_importance_topic").value)
            self.create_subscription(Image, topic, self._cb_modality, qos)

        self.timer = self.create_timer(1.0 / max(1e-3, self.framerate), self._cb_timer)

    def _cb_flow(self, msg: Image):
        arr = self._to_array(msg, "flow_vector_overlay")
        if arr is not None:
            self.latest_flow = arr
            self._update_target()

    def _cb_modality(self, msg: Image):
        arr = self._to_array(msg, "modality_importance")
        if arr is not None:
            self.latest_modality = arr
            self._update_target()

    def _update_target(self):
        if self.want_flow and self.latest_flow is None:
            return
        if self.want_modality and self.latest_modality is None:
            return
        composite = self._composite()
        if composite is None:
            return
        if self._target_composite is not None and self._target_composite.shape == composite.shape:
            self._prev_composite = self._current_blend().astype(np.float32)
        else:
            self._prev_composite = composite.astype(np.float32)
        self._target_composite = composite
        self._target_time = time.monotonic()

    def _current_blend(self) -> np.ndarray:
        """The frame as currently displayed (mid-transition or settled)."""
        if self._target_composite is None:
            return None
        if self._prev_composite is None or self._target_time is None:
            return self._target_composite
        alpha = (time.monotonic() - self._target_time) / max(1e-6, self.transition_sec)
        alpha = min(1.0, max(0.0, alpha))
        if alpha >= 1.0:
            return self._target_composite
        blended = (1.0 - alpha) * self._prev_composite + alpha * self._target_composite.astype(np.float32)
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _to_array(self, msg: Image, label: str):
        if msg.encoding != "rgb8":
            self.get_logger().warn(f"[{label}] unexpected encoding '{msg.encoding}', expected rgb8, skipping frame")
            return None
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

    def _composite(self):
        left = self.latest_flow if self.want_flow else None
        right = self.latest_modality if self.want_modality else None
        parts = [p for p in (left, right) if p is not None]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        # Equal-width (1:1) panels: scale each up to the wider panel's width
        # (preserving its own aspect ratio, not stretched), then pad heights
        # to match so hstack lines up cleanly.
        target_width = max(p.shape[1] for p in parts)
        parts = [_resize_to_width(p, target_width) for p in parts]
        height = max(p.shape[0] for p in parts)
        parts = [_pad_to_height(p, height) for p in parts]
        return np.hstack(parts)

    def _cb_timer(self):
        frame = self._current_blend()
        if frame is None:
            return
        if self.proc is None:
            self._start_ffmpeg(frame.shape[1], frame.shape[0])
        try:
            self.proc.stdin.write(frame.tobytes())
            self._frames_written += 1
        except BrokenPipeError:
            self.get_logger().warn("ffmpeg pipe closed unexpectedly")

    def _start_ffmpeg(self, width: int, height: int):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"{self.file_prefix}_{stamp}_{self.policy_class}"
        if self.run_tag:
            name += f"_{self.run_tag}"
        self.out_path = os.path.join(self.output_dir, f"{name}.webm")
        self._out_w, self._out_h = int(width), int(height)
        log_path = f"/tmp/overlay_video_recorder_{stamp}.log"
        self._log_fh = open(log_path, "wb")
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(self.framerate),
            "-i", "-",
            "-c:v", "libvpx", "-b:v", f"{self.bitrate_mbps}M",
            "-deadline", "realtime", "-cpu-used", "4",
            "-an",
            self.out_path,
        ]
        self.get_logger().info(f"recording composite {width}x{height} (flow left, modality right) -> {self.out_path}")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=self._log_fh, stderr=subprocess.STDOUT)

    def _find_fresh_heatmap(self):
        import glob
        cands = [c for c in glob.glob(self.removal_glob)
                 if os.path.getmtime(c) >= self._start_wall - 5.0]
        return max(cands, key=os.path.getmtime) if cands else None

    def _append_removal_heatmap_tail(self):
        """Poll for the polishing-removal heatmap PNG (written by
        polishing_removal_recorder on this same shutdown) and, if found,
        write it as the last removal_tail_sec of video."""
        if not self.append_removal_heatmap or self.proc is None or self._out_w is None:
            return
        deadline = time.monotonic() + self.removal_grace_sec
        png = self._find_fresh_heatmap()
        while png is None and time.monotonic() < deadline:
            time.sleep(0.3)
            png = self._find_fresh_heatmap()
        if png is None:
            self.get_logger().warn("no fresh removal heatmap found; video ends without it")
            return
        img = cv2.imread(png, cv2.IMREAD_COLOR)
        if img is None:
            self.get_logger().warn(f"could not read heatmap {png}")
            return
        frame = _fit_into(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), self._out_w, self._out_h)
        n = max(1, int(round(self.removal_tail_sec * self.framerate)))
        written = 0
        for _ in range(n):
            try:
                self.proc.stdin.write(frame.tobytes())
                written += 1
            except (BrokenPipeError, ValueError):
                break
        self._frames_written += written
        self.get_logger().info(
            f"appended removal heatmap tail: {written} frames (~{written / max(1e-6, self.framerate):.1f}s) "
            f"from {png}")

    def destroy_node(self):
        if getattr(self, "timer", None) is not None:
            self.timer.cancel()
        if self.proc is not None:
            if self._frames_written == 0:
                self.get_logger().info("no frames ever received, discarding")
                self.proc.stdin.close()
                self.proc.kill()
                self.proc.wait(timeout=5.0)
                try:
                    os.remove(self.out_path)
                except OSError:
                    pass
            else:
                self._append_removal_heatmap_tail()
                self.get_logger().info(f"stopping ({self._frames_written} frames)...")
                try:
                    self.proc.stdin.close()
                    self.proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    self.get_logger().warn("ffmpeg did not exit in time, killing it")
                    self.proc.kill()
                    self.proc.wait(timeout=5.0)
                self.get_logger().info(f"saved -> {self.out_path}")
            if self._log_fh is not None:
                try:
                    self._log_fh.close()
                except Exception:
                    pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OverlayVideoRecorderNode()

    def _stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _stop)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Finalizing (heatmap append + ffmpeg flush) takes a few seconds and
        # MUST run to completion -- a second Ctrl-C or launch's SIGTERM
        # escalation would otherwise truncate the webm. Ignore both here.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
