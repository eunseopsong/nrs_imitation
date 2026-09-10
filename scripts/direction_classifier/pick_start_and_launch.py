#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-launch helper: grabs a few frames from the live camera topic, runs the
small direction classifier, and launches inference_clean_single_cam.launch.py
with whichever demo_start-pose wrapper checkpoint (0deg-targeted vs
90deg-targeted) matches the prediction -- so the robot's initial alignment
move actually goes toward the stain that's really in front of it, instead of
a single fixed point.

This is a standalone pre-step, not a change to inference_core.py's own
stage machine: it decides ONE thing (which ckpt_dir/demo_start wrapper to
use) before handing off to the normal launch file.
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
# torch (and anything that pulls in ssl/huggingface_hub through it) must be
# imported before rclpy: rclpy loads a system libcrypto that's incompatible
# with the conda env's OpenSSL, and whichever loads first wins for the rest
# of the process -- importing torch first keeps the working one loaded.
import torch
import torch.nn as nn
import torchvision.transforms as T
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "models"))
from dinov3_backbone import DINOv3PatchBackbone  # noqa: E402


def _img_msg_to_numpy(msg: Image) -> np.ndarray:
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        arr = arr.reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)
    raise RuntimeError(f"unsupported image encoding: {msg.encoding}")


class _FrameGrabber(Node):
    def __init__(self, image_topic: str, num_frames: int):
        super().__init__("direction_classifier_frame_grabber")
        self.frames = []
        self.num_frames = num_frames
        self.sub = self.create_subscription(Image, image_topic, self._cb, qos_profile_sensor_data)

    def _cb(self, msg: Image):
        try:
            self.frames.append(_img_msg_to_numpy(msg))
        except Exception as e:
            self.get_logger().warn(f"decode failed: {e}")


def grab_frames(image_topic: str, num_frames: int, timeout_sec: float) -> list:
    rclpy.init(args=None)
    node = _FrameGrabber(image_topic, num_frames)
    try:
        deadline = time.monotonic() + timeout_sec
        while len(node.frames) < num_frames and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return node.frames


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Load NumPy-2 pickles under a NumPy-1 interpreter -- `ros2 launch`
    itself runs under the system python3 (see its shebang), which has an
    older numpy than the conda env the checkpoints were saved from, so
    plain pickle.load() fails with "No module named 'numpy._core'". Mirrors
    inference_core.py's own _NumpyCompatUnpickler, kept in sync there."""

    def find_class(self, module, name):
        if module == "numpy._core":
            module = "numpy.core"
        elif str(module).startswith("numpy._core."):
            module = "numpy.core." + str(module)[len("numpy._core."):]
        return super().find_class(module, name)


def load_demo_start_pose_mean(ckpt_dir: str, prefer_arithmetic: bool = True) -> np.ndarray:
    """Read the (direction-agnostic) neutral pose to settle at before the
    stain is classified -- only there (not at the robot's arbitrary idle
    pose) does the wrist camera reliably see the workpiece.

    prefer_arithmetic=True (default) uses `demo_start_pose_arithmetic_mean`,
    the mean over ALL episodes -- a genuine 0deg/90deg midpoint. The plain
    `demo_start_pose_mean` is a KNN-density pick of ONE representative
    episode, which for balanced42 landed exactly on a 90deg start pose;
    classifying from there biased every prediction toward 90deg because the
    classifier only ever saw that viewpoint for the 90deg class.
    """
    with open(Path(ckpt_dir) / "dataset_stats.pkl", "rb") as f:
        stats = _NumpyCompatUnpickler(f).load()
    key = "demo_start_pose_mean"
    if prefer_arithmetic and "demo_start_pose_arithmetic_mean" in stats:
        key = "demo_start_pose_arithmetic_mean"
    return np.asarray(stats[key], dtype=np.float64)


class _SettleAndGrab(Node):
    """Subscribes to both the measured pose and the camera topic. Once the
    measured pose stays within tolerance of target_pose6 for hold_sec, starts
    collecting frames (so classification only ever sees a frame taken from a
    pose known to view the workpiece, never the robot's arbitrary post-launch
    idle pose)."""

    def __init__(self, image_topic: str, pose_topic: str, target_pose6: np.ndarray,
                 pos_tol_mm: float, rot_tol_rad: float, hold_sec: float, num_frames: int):
        super().__init__("direction_classifier_settle_and_grab")
        self.target_pose6 = target_pose6
        self.pos_tol_mm = pos_tol_mm
        self.rot_tol_rad = rot_tol_rad
        self.hold_sec = hold_sec
        self.num_frames = num_frames
        self.frames = []
        self.settled = False
        self._settled_since = None
        self._last_pose_err = (float("inf"), float("inf"))
        self.create_subscription(Float64MultiArray, pose_topic, self._on_pose, qos_profile_sensor_data)
        self.create_subscription(Image, image_topic, self._on_img, qos_profile_sensor_data)

    def _on_pose(self, msg: Float64MultiArray):
        arr = np.asarray(msg.data, dtype=np.float64)
        if arr.shape[0] < 6:
            return
        pos_err = float(np.linalg.norm(arr[:3] - self.target_pose6[:3]))
        rot_err = float(np.max(np.abs(arr[3:6] - self.target_pose6[3:6])))
        self._last_pose_err = (pos_err, rot_err)
        now = time.monotonic()
        if pos_err <= self.pos_tol_mm and rot_err <= self.rot_tol_rad:
            if self._settled_since is None:
                self._settled_since = now
            elif not self.settled and (now - self._settled_since) >= self.hold_sec:
                self.settled = True
        else:
            self._settled_since = None

    def _on_img(self, msg: Image):
        if not self.settled or len(self.frames) >= self.num_frames:
            return
        try:
            self.frames.append(_img_msg_to_numpy(msg))
        except Exception as e:
            self.get_logger().warn(f"decode failed: {e}")


def wait_settle_and_grab_frames(
    image_topic: str, pose_topic: str, target_pose6: np.ndarray,
    num_frames: int, pos_tol_mm: float = 5.0, rot_tol_rad: float = 0.05,
    hold_sec: float = 1.0, timeout_sec: float = 30.0,
) -> list:
    rclpy.init(args=None)
    node = _SettleAndGrab(image_topic, pose_topic, target_pose6, pos_tol_mm, rot_tol_rad, hold_sec, num_frames)
    try:
        deadline = time.monotonic() + timeout_sec
        last_log = 0.0
        while len(node.frames) < num_frames and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            now = time.monotonic()
            if now - last_log >= 2.0:
                pos_err, rot_err = node._last_pose_err
                print(f"[settle] settled={node.settled} pos_err={pos_err:.1f}mm rot_err={rot_err:.3f}rad frames={len(node.frames)}/{num_frames}")
                last_log = now
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return node.frames


def move_to_pose_and_grab_frames(
    target_pose6,
    image_topic: str,
    pose_topic: str,
    num_frames: int,
    pos_tol_mm: float = 6.0,
    rot_tol_rad: float = 0.06,
    hold_sec: float = 1.0,
    timeout_sec: float = 45.0,
    target_velocity_mm_s: float = 20.0,
    service_name: str = "/singleArm_cmd/single_arm_command",
) -> list:
    """Phase 1 (lightweight): PTP the arm to the neutral viewing pose through
    the SingleArmCommand service directly -- NO inference stack, NO Force
    mode, NO PTP9D stream -- then wait for it to settle and grab classifier
    frames.

    Replaces the old phase 1 (run a whole base-checkpoint inference stack
    just to move the arm), which entered TRACK + Force mode and left the
    robot force-loaded / mid-trajectory for phase 2's alignment to crash
    into. The arm is left in Position mode; phase 2's own stack does its
    PTP + Force-mode switch.
    """
    from y2_rob_motion_interfaces.srv import SingleArmCommand

    target_pose6 = np.asarray(target_pose6, dtype=np.float64)
    rclpy.init(args=None)
    try:
        ptp = rclpy.create_node("autodir_phase1_move")
        cli = ptp.create_client(SingleArmCommand, service_name)
        if not cli.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f"{service_name} unavailable for phase-1 move")

        req = SingleArmCommand.Request()
        req.command_mode = "PTP"
        # PTP_command_gen applies DegreeToRadian -- this service wants the
        # orientation triple in DEGREES, unlike every other pose field here.
        req.target_pose = [
            float(target_pose6[0]), float(target_pose6[1]), float(target_pose6[2]),
            float(np.degrees(target_pose6[3])),
            float(np.degrees(target_pose6[4])),
            float(np.degrees(target_pose6[5])),
        ]
        req.target_velocity = float(target_velocity_mm_s)
        print(f"[autodir] phase 1: PTP -> neutral {np.round(target_pose6, 3).tolist()} "
              f"@ {target_velocity_mm_s:.0f}mm/s")
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(ptp, fut, timeout_sec=max(30.0, timeout_sec))
        try:
            resp = fut.result()
        except Exception as exc:
            raise RuntimeError(f"phase-1 PTP call failed: {exc}")
        if resp is None:
            raise RuntimeError("phase-1 PTP call timed out (no service response)")
        print(f"[autodir] phase 1: PTP done: {getattr(resp, 'message', resp)}")
        ptp.destroy_node()

        node = _SettleAndGrab(
            image_topic, pose_topic, target_pose6,
            pos_tol_mm, rot_tol_rad, hold_sec, num_frames,
        )
        try:
            deadline = time.monotonic() + timeout_sec
            last_log = 0.0
            while len(node.frames) < num_frames and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.2)
                now = time.monotonic()
                if now - last_log >= 2.0:
                    pos_err, rot_err = node._last_pose_err
                    print(f"[settle] settled={node.settled} pos_err={pos_err:.1f}mm "
                          f"rot_err={rot_err:.3f}rad frames={len(node.frames)}/{num_frames}")
                    last_log = now
            return list(node.frames)
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()


def stop_arm_motion(
    service_name: str = "/singleArm_cmd/single_arm_command",
    settle_sec: float = 3.0,
) -> None:
    """Tear down the robot-side motion left running by the phase-1 stack.

    inference_core's SIGINT path (`except KeyboardInterrupt: pass`) does NOT
    stop the PTP9D stream or leave Force mode, so after the phase-1 stack is
    killed the arm keeps consuming phase-1's queued trajectory while still
    force-loaded. If phase 2 then fires its own DEMO_START PTP into that, the
    alignment lands ~40mm into the workpiece (observed twice). Send an
    explicit PTP9D_STREAM_STOP, then wait for the arm to come to rest.
    """
    from y2_rob_motion_interfaces.srv import SingleArmCommand

    rclpy.init(args=None)
    node = rclpy.create_node("autodir_stop_arm_motion")
    cli = node.create_client(SingleArmCommand, service_name)
    try:
        if not cli.wait_for_service(timeout_sec=5.0):
            print(f"[autodir] WARN: {service_name} unavailable; cannot stop phase-1 stream")
            return
        req = SingleArmCommand.Request()
        req.command_mode = "PTP9D_STREAM_STOP"
        req.target_pose = []
        req.target_velocity = 0.0
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        try:
            resp = future.result()
            print(f"[autodir] phase-1 stream stop: {getattr(resp, 'message', resp)}")
        except Exception as exc:
            print(f"[autodir] WARN: PTP9D_STREAM_STOP call raised: {exc}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print(f"[autodir] waiting {settle_sec}s for arm to come to rest before phase 2 ...")
    time.sleep(settle_sec)


def classify_direction(frames: list, classifier_ckpt: str) -> int:
    ckpt = torch.load(classifier_ckpt, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = DINOv3PatchBackbone(model_name=ckpt["dino_model_name"], pretrained=True, freeze=True).to(device)
    backbone.eval()
    head = nn.Sequential(
        nn.LayerNorm(backbone.feature_dim),
        nn.Linear(backbone.feature_dim, 128),
        nn.Mish(),
        nn.Linear(128, len(classes)),
    ).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    votes = np.zeros(len(classes), dtype=np.int64)
    with torch.no_grad():
        for frame in frames:
            img = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            global_feat, _, _ = backbone(normalize(img))
            logits = head(global_feat)
            votes[int(logits.argmax(-1))] += 1
    winner_idx = int(np.argmax(votes))
    print(f"[direction_classifier] votes={dict(zip(classes, votes.tolist()))} -> predicted {classes[winner_idx]}deg")
    return int(classes[winner_idx])


# Sentinel labels returned by the reference classifier for anything that is
# not one of the known polishing directions (0 / 90):
LABEL_NOVEL_TYPE = -1   # matched a known-but-non-direction reference (e.g. the 3-dot-row stain)
LABEL_UNMATCHED = -2    # matched NO reference within reject_dist (a stain shape never templated)


def classify_stain_by_reference(frames: list, reference_npz: str, verbose: bool = True):
    """Reference-difference stain classifier for a FIXED classification pose
    (home pose). No network, no training: compare the live frame to K stored
    reference images (one per stain type, each drawn at that same pose)
    inside the pixel region where the references differ from each other
    (= where any stain leaves a mark). Whichever reference the frame is
    closest to there wins -- unless it is close to *none* of them, in which
    case it is a stain type we have never templated (LABEL_UNMATCHED).

    Robust because the pose and background are fixed, so only the stain
    varies; brightness is matched per frame before comparison. Returns
    (label:int, info:dict); label is 0 / 90 for the known directions,
    LABEL_NOVEL_TYPE for a templated non-direction stain, or LABEL_UNMATCHED.
    Build / extend the .npz with build_home_reference.py.
    """
    ref = np.load(reference_npz, allow_pickle=True)

    # New multi-class format: `refs` (K,H,W) + `labels` (K,) + `label_names`.
    # Old 0/90-only format: ref_gray_0 / ref_gray_90 -- synthesised here so a
    # stale npz keeps working (it just cannot report the third type).
    if "refs" in ref.files:
        refs = ref["refs"].astype(np.float32)
        labels = [int(x) for x in ref["labels"]]
        names = [str(x) for x in ref["label_names"]] if "label_names" in ref.files \
            else [str(l) for l in labels]
        reject_dist = float(ref["reject_dist"]) if "reject_dist" in ref.files else float("inf")
    else:
        refs = np.stack([ref["ref_gray_0"], ref["ref_gray_90"]]).astype(np.float32)
        labels = [0, 90]
        names = ["0deg", "90deg"]
        reject_dist = float("inf")

    mask = ref["stain_mask"].astype(bool) if "stain_mask" in ref.files \
        else ref["diff_mask"].astype(bool)
    m_sum = float(mask.sum())
    if m_sum < 1:
        raise ValueError(f"{reference_npz}: empty stain mask")
    h, w = refs.shape[1:]
    ref_mean_in_mask = float(refs.mean(axis=0)[mask].mean())

    K = len(labels)
    d_acc = np.zeros(K, dtype=np.float64)
    votes = np.zeros(K, dtype=np.int64)
    best_d_acc = 0.0
    for fr in frames:
        g = np.asarray(fr).astype(np.float32).mean(axis=-1)
        if g.shape != (h, w):
            g = cv2.resize(g, (w, h))
        # match overall brightness inside the compared region
        g = g - float(g[mask].mean()) + ref_mean_in_mask
        d = np.array([float(np.sum((g[mask] - refs[k][mask]) ** 2) / m_sum) for k in range(K)])
        d_acc += d
        votes[int(np.argmin(d))] += 1
        best_d_acc += float(d.min())
    n = max(1, len(frames))
    d_mean = d_acc / n
    best_d = best_d_acc / n
    order = np.argsort(d_mean)
    winner = int(np.argmax(votes))
    label = labels[winner]
    if best_d > reject_dist:
        label = LABEL_UNMATCHED

    info = {
        "label": label,
        "label_name": ("unmatched" if label == LABEL_UNMATCHED else names[winner]),
        "votes": {labels[k]: int(votes[k]) for k in range(K)},
        "dists": {labels[k]: round(float(d_mean[k]), 1) for k in range(K)},
        "best_d": round(best_d, 2),
        "margin": round(float(d_mean[order[1]] - d_mean[order[0]]), 1),
        "reject_dist": reject_dist,
    }
    # keep the old d0/d90 keys when present, some callers/logs read them
    if 0 in info["dists"]:
        info["d0"] = info["dists"][0]
    if 90 in info["dists"]:
        info["d90"] = info["dists"][90]
    if verbose:
        print(f"[stain_classifier] reference-diff votes={info['votes']} "
              f"dists={info['dists']} best_d={info['best_d']} "
              f"(reject>{reject_dist:.0f}) -> {label} ({info['label_name']})")
    return label, info


def classify_direction_by_reference(frames: list, reference_npz: str, verbose: bool = True):
    """Back-compat alias -- see classify_stain_by_reference(). Returns the same
    (label, info); label may now be 0, 90, LABEL_NOVEL_TYPE or LABEL_UNMATCHED,
    so callers must handle the non-direction cases (older callers assumed
    0/90 only)."""
    return classify_stain_by_reference(frames, reference_npz, verbose=verbose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_topic", type=str, default="/realsense/vr/color/image_raw")
    ap.add_argument("--num_frames", type=int, default=5)
    ap.add_argument("--grab_timeout_sec", type=float, default=5.0)
    ap.add_argument(
        "--classifier_ckpt", type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "direction_classifier" / "classifier.pt"),
    )
    ap.add_argument(
        "--ckpt_dir_0deg", type=str,
        default="/home/eunseop/nrs_imitation/checkpoints/flow/polishing/single_cam/20260827_2128_phasematch90_0deg/20260827_2129",
    )
    ap.add_argument(
        "--ckpt_dir_90deg", type=str,
        default="/home/eunseop/nrs_imitation/checkpoints/flow/polishing/single_cam/20260827_2128_only90deg/20260828_0341",
    )
    ap.add_argument("--dry_run", action="store_true", help="Only print the chosen ckpt_dir/command, don't launch.")
    ap.add_argument("--extra_args", type=str, default="", help="Extra ros2 launch key:=value args, space-separated.")
    args = ap.parse_args()

    print(f"[direction_classifier] grabbing up to {args.num_frames} frames from {args.image_topic} ...")
    frames = grab_frames(args.image_topic, args.num_frames, args.grab_timeout_sec)
    if not frames:
        raise RuntimeError(f"no frames received from {args.image_topic} within {args.grab_timeout_sec}s")
    print(f"[direction_classifier] got {len(frames)} frame(s)")

    direction = classify_direction(frames, args.classifier_ckpt)
    ckpt_dir = args.ckpt_dir_0deg if direction == 0 else args.ckpt_dir_90deg
    print(f"[direction_classifier] selected ckpt_dir={ckpt_dir}")

    cmd = [
        "ros2", "launch", "nrs_imitation", "inference_clean_single_cam.launch.py",
        "policy_class:=FLOW",
        f"ckpt_dir:={ckpt_dir}",
        "inference_mode:=service_stream",
        "use_stain_mask:=false",
        "metrics_log_enable:=true",
        f"metrics_run_tag:=autodir_{direction}deg",
    ] + args.extra_args.split()

    print("[direction_classifier] launch command:")
    print("  " + " ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
