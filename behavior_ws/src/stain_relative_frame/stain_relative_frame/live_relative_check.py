#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live diagnostic: watch `relative = to_relative(TCP, stain_origin)` update as the
stain is drawn or the arm is jogged.

Not part of the gated pipeline -- a bring-up aid. It prints, a few times a
second:

    stain origin (base mm) | current TCP xy (base mm) | relative xy (base mm)

The relative value is computed by `relative_frame.to_relative` -- the SAME
function the dataset converter [3] and the inference adapter [5] call -- so this
is a live check of the real transform, not a re-implementation.

Two modes:

  * default (--freeze): resolve the stain origin ONCE from the first good
    detection and latch it, exactly as `stain_origin_node` does. `relative`
    then tracks the TCP 1:1 by construction; use this to confirm the pose
    stream and the transform are wired up.
  * --redetect: re-run the detector every frame. The origin should stay put
    while you jog (that also exercises the per-frame homography), but single
    frames off the home pose are noisy -- specular highlights truncate the
    strip, stray dark blobs pull the midpoint. Useful for tuning
    --plate_roi / --tool_box / --dark_thresh against the live scene.

Detection is reference-free (`stain_detect.detect_stain_dark`) with a per-frame
homography from the depth-extrinsic calibration; needs homography.json written
by `homography_depth_calibrate`.

  ros2 run stain_relative_frame live_relative_check
  ros2 run stain_relative_frame live_relative_check -- --redetect --save /tmp/srf.png
  ros2 run stain_relative_frame live_relative_check -- --plate_roi 200 68 340 140
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import load_config
from .homography import load_homography, per_frame_homography
from .relative_frame import StainOrigin, to_relative
from .ros_utils import img_msg_to_numpy
from .stain_detect import detect_stain_dark, to_gray


def _draw_overlay(frame, det, plate_roi, tool_box, path: str) -> None:
    try:
        import cv2
    except ImportError:
        return
    g = to_gray(frame).astype(np.uint8)
    vis = np.stack([g, g, g], axis=-1)
    if det is not None and det.mask is not None:
        vis[det.mask > 0] = (255, 0, 0)
    cv2.rectangle(vis, tuple(plate_roi[:2]), tuple(plate_roi[2:]), (0, 255, 0), 1)
    cv2.rectangle(vis, tuple(tool_box[:2]), tuple(tool_box[2:]), (0, 128, 255), 1)
    if det is not None and det.ok:
        cv2.circle(vis, (int(det.centroid_px[0]), int(det.centroid_px[1])), 4, (0, 0, 255), -1)
    cv2.imwrite(path, vis[:, :, ::-1])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="live stain-relative position check")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--homography", type=str, default=None)
    ap.add_argument("--image_topic", type=str, default=None)
    ap.add_argument("--pose_topic", type=str, default=None)
    ap.add_argument("--plate_roi", type=int, nargs=4, default=None,
                    metavar=("U0", "V0", "U1", "V1"))
    ap.add_argument("--tool_box", type=int, nargs=4, default=None,
                    metavar=("U0", "V0", "U1", "V1"))
    ap.add_argument("--dark_thresh", type=int, default=None)
    ap.add_argument("--min_area", type=int, default=None)
    ap.add_argument("--rate_hz", type=float, default=3.0)
    ap.add_argument("--redetect", action="store_true",
                    help="re-detect the origin every frame instead of latching it")
    ap.add_argument("--save", type=str, default="",
                    help="write a mask overlay PNG here every tick")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    image_topic = args.image_topic or cfg.image_topic
    pose_topic = args.pose_topic or cfg.pose_topic
    plate_roi = list(args.plate_roi if args.plate_roi is not None else cfg.stain_plate_roi)
    tool_box = list(args.tool_box if args.tool_box is not None else cfg.stain_tool_box)
    dark_thresh = int(args.dark_thresh if args.dark_thresh is not None else cfg.stain_dark_thresh)
    min_area = int(args.min_area if args.min_area is not None else cfg.stain_dark_min_area)

    _, h_meta = load_homography(args.homography or cfg.path("homography_file"), require_pass=True)
    hm = h_meta.get("meta", {}) or {}
    if hm.get("method") != "depth_extrinsic":
        print("[live] this tool needs homography.json from homography_depth_calibrate "
              f"(method=depth_extrinsic); got method={hm.get('method')!r}", file=sys.stderr)
        return 1

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float64MultiArray

    rclpy.init()
    node = Node("srf_live_relative_check")
    latest = {"frame": None, "pose": None}

    def on_img(msg):
        try:
            latest["frame"] = img_msg_to_numpy(msg)
        except Exception as exc:  # noqa: BLE001
            node.get_logger().warn(f"decode failed: {exc}")

    def on_pose(msg):
        v = np.asarray(msg.data, dtype=np.float64)
        if v.shape[0] >= 6:
            latest["pose"] = v[:6]

    node.create_subscription(Image, image_topic, on_img, qos_profile_sensor_data)
    node.create_subscription(Float64MultiArray, pose_topic, on_pose, qos_profile_sensor_data)

    print(f"[live] image={image_topic}  pose={pose_topic}")
    print(f"[live] plate_roi={plate_roi}  tool_box={tool_box}  dark<{dark_thresh}  "
          f"min_area={min_area}  mode={'redetect' if args.redetect else 'freeze'}")

    t_wait = time.monotonic()
    while rclpy.ok() and (latest["frame"] is None or latest["pose"] is None):
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.monotonic() - t_wait > 15:
            miss = [k for k in ("frame", "pose") if latest[k] is None]
            print(f"[live] no {', '.join(miss)} after 15s -- are the topics up?", file=sys.stderr)
            node.destroy_node()
            rclpy.shutdown()
            return 1

    def detect(frame, pose6):
        H_i = per_frame_homography(hm, pose6)
        return detect_stain_dark(frame, H_i, tuple(plate_roi), tuple(tool_box),
                                 dark_thresh=dark_thresh, min_area=min_area,
                                 keep_mask=bool(args.save))

    frozen: Optional[StainOrigin] = None
    if not args.redetect:
        for _ in range(60):
            rclpy.spin_once(node, timeout_sec=0.1)
            d = detect(latest["frame"], latest["pose"])
            if d.ok:
                frozen = StainOrigin(d.origin_mm, source="live_first_detection").freeze()
                print(f"[live] origin LATCHED at ({frozen.xy[0]:.2f}, {frozen.xy[1]:.2f}) mm "
                      f"-- relative now tracks the TCP; ctrl-C to stop\n")
                break
        if frozen is None:
            print("[live] could not detect the stain to latch an origin -- draw a "
                  "darker line or widen --plate_roi, or use --redetect", file=sys.stderr)
            node.destroy_node()
            rclpy.shutdown()
            return 1
    else:
        print("[live] re-detecting every frame; ctrl-C to stop\n")

    print(f"{'status':>18} | {'origin xy (mm)':>21} | {'TCP xy (mm)':>21} | "
          f"{'relative xy (mm)':>21} | px")
    period = 1.0 / max(0.5, args.rate_hz)
    try:
        while rclpy.ok():
            t0 = time.monotonic()
            rclpy.spin_once(node, timeout_sec=period)
            frame, pose6 = latest["frame"], latest["pose"]
            det = detect(frame, pose6)
            origin = frozen.xy if frozen is not None else (det.origin_mm if det.ok else None)
            if origin is not None:
                rel = to_relative(pose6, origin, use_relative=True)
                o, tcp = np.asarray(origin), pose6[:2]
                px = (f"({det.centroid_px[0]:.0f},{det.centroid_px[1]:.0f}) "
                      f"n={det.num_components} a={det.area_px}") if det.ok else "(no detection this frame)"
                print(f"{('OK' if det.ok else det.status):>18} | "
                      f"({o[0]:9.2f},{o[1]:9.2f}) | ({tcp[0]:9.2f},{tcp[1]:9.2f}) | "
                      f"({rel[0]:9.2f},{rel[1]:9.2f}) | {px}")
            else:
                print(f"{det.status:>18} | {'--':>21} | "
                      f"({pose6[0]:9.2f},{pose6[1]:9.2f}) | {'--':>21} | n={det.num_components}")
            if args.save:
                _draw_overlay(frame, det, plate_roi, tool_box, args.save)
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
