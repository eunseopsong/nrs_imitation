#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Move the arm through a few small XY offsets from the home pose and check that
the stain-relative position responds correctly to the motion.

Not part of the gated pipeline -- a bring-up check for the transform + the pose
stream on real hardware.

For each pose it reports two things:

  * frozen-origin relative (the pipeline's behaviour): the origin is resolved
    ONCE at the home pose and latched; `relative = adapter.observation(TCP)`
    via `RelativeFrameAdapter`. `d_relative` MUST equal the commanded `d_TCP`
    to within --track_tol_mm. This is the gate.
  * redetect origin (information only): the detector re-run at this pose. It
    should land near the home value; a large drift means the single-frame
    detector is being fooled here (specular highlight, stray blob), not that
    the transform is wrong -- the pipeline never re-detects off the home pose.

Safe by construction: Z, rx, ry, rz are held at the home values, XY offsets are
clamped to --max_offset_mm (default 25), velocity is --velocity (default 20
mm/s), and the arm is PTP'd back to the recorded home pose in a finally block.

  ros2 run stain_relative_frame ptp_relative_test
  ros2 run stain_relative_frame ptp_relative_test -- --offsets "10,0 -10,0 0,10 0,-10"
  ros2 run stain_relative_frame ptp_relative_test -- --plate_roi 200 68 340 140 --velocity 15
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import load_config
from .homography import load_homography, per_frame_homography
from .relative_frame import RelativeFrameAdapter, StainOrigin
from .report import table
from .ros_utils import img_msg_to_numpy, ptp_to
from .stain_detect import detect_stain_dark

PRESET_OFFSETS: List[Tuple[float, float]] = [(12.0, 0.0), (-12.0, 0.0), (0.0, 12.0), (0.0, -12.0)]


def _parse_offsets(spec: Optional[str]) -> List[Tuple[float, float]]:
    if not spec:
        return list(PRESET_OFFSETS)
    out: List[Tuple[float, float]] = []
    for tok in spec.replace(";", " ").split():
        dx, dy = tok.split(",")
        out.append((float(dx), float(dy)))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="PTP the arm and check stain-relative tracking")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--homography", type=str, default=None)
    ap.add_argument("--image_topic", type=str, default=None)
    ap.add_argument("--pose_topic", type=str, default=None)
    ap.add_argument("--arm_service", type=str, default=None)
    ap.add_argument("--offsets", type=str, default=None,
                    help='XY offsets from home, e.g. "12,0 -12,0 0,12 0,-12" (mm). '
                         "Default: a +/-12 mm star.")
    ap.add_argument("--max_offset_mm", type=float, default=25.0,
                    help="hard clamp on |dx|,|dy| -- a guard against a fat-fingered --offsets")
    ap.add_argument("--velocity", type=float, default=None)
    ap.add_argument("--settle_sec", type=float, default=1.5)
    ap.add_argument("--samples", type=int, default=25)
    ap.add_argument("--track_tol_mm", type=float, default=1.5,
                    help="max |d_relative - d_TCP| for the frozen-origin path to pass")
    ap.add_argument("--redetect_tol_mm", type=float, default=3.0,
                    help="redetect origin drift above this is reported as CHECK (info only)")
    ap.add_argument("--plate_roi", type=int, nargs=4, default=None, metavar=("U0", "V0", "U1", "V1"))
    ap.add_argument("--tool_box", type=int, nargs=4, default=None, metavar=("U0", "V0", "U1", "V1"))
    ap.add_argument("--dark_thresh", type=int, default=None)
    ap.add_argument("--min_area", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    image_topic = args.image_topic or cfg.image_topic
    pose_topic = args.pose_topic or cfg.pose_topic
    arm_service = args.arm_service or cfg.arm_service
    velocity = float(args.velocity if args.velocity is not None else cfg.ptp_velocity_mm_s)
    plate_roi = tuple(args.plate_roi if args.plate_roi is not None else cfg.stain_plate_roi)
    tool_box = tuple(args.tool_box if args.tool_box is not None else cfg.stain_tool_box)
    dark_thresh = int(args.dark_thresh if args.dark_thresh is not None else cfg.stain_dark_thresh)
    min_area = int(args.min_area if args.min_area is not None else cfg.stain_dark_min_area)
    out_path = Path(args.out) if args.out else cfg.path("ptp_test_report_file")

    offsets = _parse_offsets(args.offsets)
    for dx, dy in offsets:
        if abs(dx) > args.max_offset_mm or abs(dy) > args.max_offset_mm:
            print(f"[ptp] offset ({dx},{dy}) exceeds --max_offset_mm={args.max_offset_mm}",
                  file=sys.stderr)
            return 1

    _, h_meta = load_homography(args.homography or cfg.path("homography_file"), require_pass=True)
    hm = h_meta.get("meta", {}) or {}
    if hm.get("method") != "depth_extrinsic":
        print("[ptp] needs homography.json from homography_depth_calibrate "
              f"(method=depth_extrinsic); got method={hm.get('method')!r}", file=sys.stderr)
        return 1

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float64MultiArray

    rclpy.init()
    node = Node("srf_ptp_relative_test")
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

    def spin(dt: float) -> None:
        end = time.monotonic() + dt
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.02)

    def detect(frame, pose6):
        H_i = per_frame_homography(hm, pose6)
        return detect_stain_dark(frame, H_i, plate_roi, tool_box,
                                 dark_thresh=dark_thresh, min_area=min_area)

    def sample(n: int, timeout: float = 6.0):
        xs, tcps = [], []
        end = time.monotonic() + timeout
        while len(xs) < n and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
            d = detect(latest["frame"], latest["pose"])
            if d.ok:
                xs.append(d.origin_mm)
                tcps.append(latest["pose"][:2].copy())
        if not xs:
            return None
        return (np.median(xs, axis=0), np.median(tcps, axis=0),
                len(xs), np.std(np.asarray(xs), axis=0))

    spin(1.0)
    if latest["pose"] is None or latest["frame"] is None:
        print("[ptp] no pose/image stream", file=sys.stderr)
        node.destroy_node(); rclpy.shutdown()
        return 1
    home6 = latest["pose"].copy()
    print(f"[ptp] home = {np.round(home6, 2).tolist()}")
    print(f"[ptp] velocity={velocity}mm/s  offsets={offsets}  "
          f"plate_roi={list(plate_roi)}  dark<{dark_thresh}")

    s0 = sample(args.samples)
    if s0 is None:
        print("[ptp] cannot detect the stain at the home pose -- draw a darker line "
              "or adjust --plate_roi / --dark_thresh", file=sys.stderr)
        node.destroy_node(); rclpy.shutdown()
        return 1
    home_xy, _, _, _ = s0
    adapter = RelativeFrameAdapter(
        StainOrigin(home_xy, source="ptp_test_home_pose"), use_relative=True)
    rel0 = np.asarray(adapter.observation(home6[:2]))
    print(f"[ptp] origin latched at ({home_xy[0]:.2f}, {home_xy[1]:.2f}) mm  "
          f"home relative = ({rel0[0]:.2f}, {rel0[1]:.2f})\n")

    rows: List[dict] = []
    aborted = False
    try:
        for i, (dx, dy) in enumerate([(0.0, 0.0)] + offsets):
            tgt = home6.copy()
            tgt[0] += dx
            tgt[1] += dy
            resp = ptp_to(tgt, velocity_mm_s=velocity, service_name=arm_service)
            reached = False
            end = time.monotonic() + 15.0
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=0.05)
                if np.linalg.norm(latest["pose"][:2] - tgt[:2]) < 0.8:
                    reached = True
                    break
            spin(args.settle_sec)
            if not reached:
                err = float(np.linalg.norm(latest["pose"][:2] - tgt[:2]))
                print(f"[ptp] ({dx:+.0f},{dy:+.0f}) NOT REACHED (xy err {err:.1f} mm, "
                      f"resp={resp!r}) -- aborting, returning home")
                aborted = True
                break

            s = sample(args.samples)
            tcp = latest["pose"][:2].copy()
            rel = np.asarray(adapter.observation(tcp))          # frozen-origin path
            d_tcp = tcp - home6[:2]
            d_rel = rel - rel0
            track_err = float(np.linalg.norm(d_rel - d_tcp))
            row = {
                "offset_cmd": [dx, dy],
                "tcp_xy": [round(float(tcp[0]), 3), round(float(tcp[1]), 3)],
                "d_tcp": [round(float(d_tcp[0]), 3), round(float(d_tcp[1]), 3)],
                "relative_xy": [round(float(rel[0]), 3), round(float(rel[1]), 3)],
                "d_relative": [round(float(d_rel[0]), 3), round(float(d_rel[1]), 3)],
                "track_err_mm": round(track_err, 3),
                "track": "OK" if track_err <= args.track_tol_mm else "FAIL",
            }
            if s is not None:
                rx, _, k, rstd = s
                drift = float(np.linalg.norm(rx - home_xy))
                row.update({
                    "redetect_xy": [round(float(rx[0]), 3), round(float(rx[1]), 3)],
                    "redetect_drift_mm": round(drift, 3),
                    "redetect_n": int(k),
                    "redetect": "OK" if drift <= args.redetect_tol_mm else "CHECK",
                })
            else:
                row.update({"redetect_xy": None, "redetect_drift_mm": None,
                            "redetect_n": 0, "redetect": "no-detection"})
            rows.append(row)
            print(f"[ptp] ({dx:+5.0f},{dy:+5.0f})  d_TCP=({d_tcp[0]:6.2f},{d_tcp[1]:6.2f})  "
                  f"d_rel=({d_rel[0]:6.2f},{d_rel[1]:6.2f})  track_err={track_err:5.2f}mm "
                  f"[{row['track']}]   redetect_drift={row['redetect_drift_mm']}mm "
                  f"[{row['redetect']}]")
    finally:
        print("\n[ptp] returning to home ...")
        try:
            ptp_to(home6, velocity_mm_s=velocity, service_name=arm_service)
            spin(args.settle_sec)
            err = float(np.linalg.norm(latest["pose"][:2] - home6[:2]))
            print(f"[ptp] back home, xy err {err:.2f} mm")
        except Exception as exc:  # noqa: BLE001
            print(f"[ptp] WARNING: return-home PTP failed: {exc}", file=sys.stderr)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if rows:
        print()
        print(table(rows, columns=["offset_cmd", "d_tcp", "d_relative", "track_err_mm",
                                   "track", "redetect_drift_mm", "redetect"],
                    title="[ptp] stain-relative tracking"))

    n_fail = sum(1 for r in rows if r["track"] == "FAIL")
    passed = (not aborted) and rows and n_fail == 0
    payload = {
        "step": "ptp_relative_test",
        "stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "home_pose6": [round(float(x), 5) for x in home6],
        "origin_xy_mm": [round(float(home_xy[0]), 3), round(float(home_xy[1]), 3)],
        "velocity_mm_s": velocity,
        "track_tol_mm": args.track_tol_mm,
        "redetect_tol_mm": args.redetect_tol_mm,
        "detect_params": {"method": "dark", "dark_thresh": dark_thresh,
                          "min_area": min_area, "plate_roi": list(plate_roi),
                          "tool_box": list(tool_box)},
        "homography_file": str(args.homography or cfg.path("homography_file")),
        "aborted": aborted,
        "n_poses": len(rows),
        "n_track_fail": n_fail,
        "passed": bool(passed),
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[ptp] report -> {out_path}")

    if aborted:
        print("[ptp] RESULT: ABORTED (arm did not reach a commanded pose)", file=sys.stderr)
        return 1
    if n_fail:
        print(f"[ptp] RESULT: {n_fail} pose(s) FAILED the {args.track_tol_mm}mm tracking "
              "tolerance -- the frozen-origin transform is not following the TCP",
              file=sys.stderr)
        return 1
    print(f"[ptp] RESULT: PASS -- relative position tracked the TCP within "
          f"{args.track_tol_mm}mm at every pose")
    return 0


if __name__ == "__main__":
    sys.exit(main())
