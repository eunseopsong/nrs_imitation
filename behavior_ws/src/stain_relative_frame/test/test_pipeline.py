#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end tests for steps [1]-[4] on the synthetic fixture.

  pytest behavior_ws/src/stain_relative_frame/test/test_pipeline.py -v

No ROS and no hardware: steps [0] and [1a]/[1b] need the arm and an operator,
so the correspondences come from the fixture's known homography instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture import (  # noqa: E402
    calibration_points, make_dataset, write_clean_reference,
)
from stain_relative_frame.homography import (  # noqa: E402
    apply_homography, fit_homography, load_homography, save_homography,
)
from stain_relative_frame.linear_probe import repeated_probe  # noqa: E402
from stain_relative_frame.relative_frame import (  # noqa: E402
    StainOrigin, StainOriginError, RelativeFrameAdapter, to_absolute, to_relative,
)
from stain_relative_frame.stain_detect import (  # noqa: E402
    FAIL_EMPTY_MASK, detect_stain_origin, measure_stability,
)


# ---------------------------------------------------------------- relative_frame

def test_flag_off_is_the_identity():
    pose = np.array([[100.0, 200.0, 50.0, 0.1, 0.2, 0.3]], dtype=np.float32)
    out = to_relative(pose, [10.0, 20.0], use_relative=False)
    assert np.array_equal(out, pose)
    assert out.dtype == pose.dtype


def test_only_xy_moves():
    pose = np.random.default_rng(0).normal(size=(7, 6))
    out = to_relative(pose, [12.0, -3.0])
    assert np.array_equal(out[:, 2:], pose[:, 2:])          # z + rotations untouched
    assert np.allclose(pose[:, 0] - out[:, 0], 12.0)
    assert np.allclose(pose[:, 1] - out[:, 1], -3.0)


def test_round_trip():
    pose = np.random.default_rng(1).normal(size=(5, 6))
    o = StainOrigin([4.0, -7.5]).freeze()
    assert np.allclose(to_absolute(to_relative(pose, o), o), pose)


def test_frozen_origin_cannot_be_updated_per_step():
    o = StainOrigin([1.0, 2.0]).freeze()
    with pytest.raises(StainOriginError):
        o.xy = np.array([3.0, 4.0])


def test_adapter_has_no_setter_for_the_origin():
    a = RelativeFrameAdapter([5.0, 6.0], use_relative=True)
    assert not any(n for n in dir(a) if n.startswith("set_"))
    before = a.stain_origin.copy()
    a.observation(np.zeros(6))
    a.command(np.zeros(6))
    assert np.array_equal(a.stain_origin, before)


def test_adapter_command_undoes_observation():
    a = RelativeFrameAdapter([30.0, -12.0], use_relative=True)
    q = np.array([400.0, 350.0, 210.0, 0.0, 0.1, 2.4])
    assert np.allclose(a.command(a.observation(q)), q)


# ---------------------------------------------------------------- [1] homography

def test_homography_heldout_gate_passes_on_clean_points(tmp_path):
    px, rb = calibration_points()
    res = fit_homography(px, rb, n_fit=6, tol_mm=2.0)
    assert res.passed, res.summary()
    assert res.max_heldout_error_mm < 1.0
    assert len(res.heldout_indices) == 2

    p = save_homography(tmp_path / "homography.json", res)
    H, meta = load_homography(p)
    assert np.allclose(H, res.H)
    assert meta["passed"]


def test_homography_gate_fails_on_a_tight_cluster():
    rng = np.random.default_rng(0)
    px = np.array([[200.0, 120.0]]) + rng.normal(0, 1.5, (8, 2))
    from fixture import _to_robot
    rb = _to_robot(px) + rng.normal(0, 1.0, (8, 2))
    res = fit_homography(px, rb, n_fit=6, tol_mm=2.0)
    assert not res.passed


def test_failed_homography_is_refused_downstream(tmp_path):
    px, rb = calibration_points()
    res = fit_homography(px, rb, n_fit=6, tol_mm=2.0)
    res.passed = False
    p = save_homography(tmp_path / "bad.json", res)
    with pytest.raises(RuntimeError, match="did NOT pass"):
        load_homography(p)


# ---------------------------------------------------------------- [2] detection

@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("scene")
    data = make_dataset(d / "raw", n_per_class=12, T=6, seed=5)
    ref_path = write_clean_reference(d / "clean_reference.npz", data["reference_rgb"])
    px, rb = calibration_points()
    res = fit_homography(px, rb, n_fit=6, tol_mm=2.0)
    h_path = save_homography(d / "homography.json", res)
    return {"dir": d, "raw": d / "raw", "data": data,
            "clean": ref_path, "homography": h_path, "H": res.H}


def test_detected_origin_matches_ground_truth(scene):
    import h5py
    from stain_relative_frame.clean_reference_capture import load_clean_reference

    ref, _ = load_clean_reference(scene["clean"])
    errs = []
    for stem, truth in list(scene["data"]["truth"].items())[:8]:
        with h5py.File(str(scene["raw"] / f"{stem}.hdf5"), "r") as f:
            frame = np.asarray(f["observations/images/cam0"][0])
        det = detect_stain_origin(frame, ref, scene["H"])
        assert det.ok, det.status
        errs.append(np.linalg.norm(det.origin_mm - np.asarray(truth["origin_mm"])))
    assert max(errs) < 2.5, f"worst origin error {max(errs):.2f}mm"


def test_empty_mask_is_recorded_as_a_failure(scene):
    from stain_relative_frame.clean_reference_capture import load_clean_reference

    ref, _ = load_clean_reference(scene["clean"])
    clean_rgb = np.stack([ref] * 3, axis=-1).astype(np.uint8)
    det = detect_stain_origin(clean_rgb, ref, scene["H"])
    assert det.status == FAIL_EMPTY_MASK
    assert det.origin_mm is None


def test_too_many_components_is_recorded_as_a_failure(scene):
    from fixture import clean_frame, draw_stain
    from stain_relative_frame.clean_reference_capture import load_clean_reference

    ref, _ = load_clean_reference(scene["clean"])
    rng = np.random.default_rng(0)
    frame = clean_frame(rng)
    for i in range(7):
        frame = draw_stain(frame, (50 + 50 * i, 60 + 20 * (i % 3)), 0.0, length=8, width=8)
    det = detect_stain_origin(frame, ref, scene["H"], max_components=5)
    assert det.status == "FAIL_TOO_MANY_COMPONENTS"
    assert det.num_components > 5


def test_stability_is_tight_on_a_static_stain(scene):
    import h5py
    from stain_relative_frame.clean_reference_capture import load_clean_reference

    ref, _ = load_clean_reference(scene["clean"])
    stem = next(iter(scene["data"]["truth"]))
    with h5py.File(str(scene["raw"] / f"{stem}.hdf5"), "r") as f:
        frames = np.asarray(f["observations/images/cam0"])
    rep = measure_stability(frames, ref, scene["H"], episode=stem, std_tol_mm=3.0)
    assert rep.n_ok == frames.shape[0]
    assert not rep.unstable
    assert rep.std_norm_mm < 1.0


# ---------------------------------------------------------------- [4] the gate

def _first_frame_features(dataset_dir: Path):
    from stain_relative_frame.validate_shortcut import load_first_frames
    return load_first_frames(Path(dataset_dir))


def test_gate_flags_the_absolute_shortcut(scene):
    d = _first_frame_features(scene["raw"])
    res = repeated_probe(d["position"][:, :2], d["labels"], 10, 0.2, 0)
    assert res["accuracy_mean"] > 0.9, "the fixture is supposed to be separable in absolute coords"


def test_gate_clears_relative_coordinates(scene):
    d = _first_frame_features(scene["raw"])
    truth = scene["data"]["truth"]
    rel = np.stack([
        d["position"][i, :2] - np.asarray(truth[name]["origin_mm"])
        for i, name in enumerate(d["names"])
    ])
    res = repeated_probe(rel, d["labels"], 20, 0.2, 0)
    assert res["accuracy_mean"] < 0.7, f"relative coords still leak: {res['accuracy_mean']:.3f}"


def test_label_shuffle_control_lands_on_chance(scene):
    d = _first_frame_features(scene["raw"])
    res = repeated_probe(d["position"][:, :2], d["labels"], 20, 0.2, 0, shuffle_labels=True)
    assert abs(res["accuracy_mean"] - 0.5) < 0.15


def test_gate_catches_an_injected_force_confound(tmp_path):
    make_dataset(tmp_path / "confounded", n_per_class=20, T=4, seed=7, force_confound=True)
    d = _first_frame_features(tmp_path / "confounded")
    res = repeated_probe(d["force"], d["labels"], 10, 0.2, 0)
    assert res["accuracy_mean"] > 0.9, "the gate must notice a force confound"


def test_gate_catches_an_injected_relative_position_confound(tmp_path):
    data = make_dataset(tmp_path / "confounded2", n_per_class=20, T=4, seed=8,
                        position_confound=True)
    d = _first_frame_features(tmp_path / "confounded2")
    rel = np.stack([
        d["position"][i, :2] - np.asarray(data["truth"][name]["origin_mm"])
        for i, name in enumerate(d["names"])
    ])
    res = repeated_probe(rel, d["labels"], 10, 0.2, 0)
    assert res["accuracy_mean"] > 0.9, "a type-dependent approach offset must fail (a)"


# ---------------------------------------------------------------- [3] converter

def test_flag_off_conversion_is_bit_identical(tmp_path):
    import h5py
    from stain_relative_frame.dataset_relativize import convert_episode, verify_episode
    from fixture import make_dataset

    raw = tmp_path / "raw"
    make_dataset(raw, n_per_class=2, T=4, seed=21)
    src = sorted(raw.glob("episode_*.hdf5"))[0]
    dst = tmp_path / "out.hdf5"
    convert_episode(src, dst, None, use_relative=False)
    assert verify_episode(src, dst, False, None) == []

    with h5py.File(src, "r") as a, h5py.File(dst, "r") as b:
        for k in ("observations/position", "action/position",
                  "observations/force", "observations/images/cam0"):
            assert np.array_equal(np.asarray(a[k]), np.asarray(b[k]))


def test_relative_conversion_preserves_absolute_and_untouched_columns(tmp_path):
    import h5py
    from stain_relative_frame.dataset_relativize import convert_episode, verify_episode
    from stain_relative_frame.relative_frame import (
        ABS_ACT_POSITION_KEY, ABS_OBS_POSITION_KEY, STAIN_ORIGIN_ATTR,
    )
    from fixture import make_dataset

    raw = tmp_path / "raw"
    make_dataset(raw, n_per_class=2, T=4, seed=22)
    src = sorted(raw.glob("episode_*.hdf5"))[0]
    dst = tmp_path / "out_rel.hdf5"
    origin = StainOrigin([300.0, 250.0]).freeze()
    convert_episode(src, dst, origin, use_relative=True)
    assert verify_episode(src, dst, True, origin) == []

    with h5py.File(src, "r") as a, h5py.File(dst, "r") as b:
        assert ABS_OBS_POSITION_KEY in b and ABS_ACT_POSITION_KEY in b
        assert np.array_equal(np.asarray(b[ABS_OBS_POSITION_KEY]),
                              np.asarray(a["observations/position"]))
        assert np.array_equal(np.asarray(a["observations/position"])[:, 2:],
                              np.asarray(b["observations/position"])[:, 2:])
        assert np.array_equal(np.asarray(a["observations/force"]),
                              np.asarray(b["observations/force"]))
        assert np.allclose(b.attrs[STAIN_ORIGIN_ATTR], origin.xy)
        assert int(b.attrs["rotation_aligned"]) == 0


def test_stats_shift_with_the_frame(tmp_path):
    from stain_relative_frame.dataset_relativize import compute_stats, convert_episode
    from fixture import make_dataset

    raw = tmp_path / "raw"
    make_dataset(raw, n_per_class=3, T=5, seed=23)
    srcs = sorted(raw.glob("episode_*.hdf5"))
    out = tmp_path / "rel"
    out.mkdir()
    origin = StainOrigin([300.0, 250.0]).freeze()
    for s in srcs:
        convert_episode(s, out / s.name, origin, use_relative=True)

    abs_stats = compute_stats(srcs, "minmax_m11", "minmax_m11")
    rel_stats = compute_stats(sorted(out.glob("episode_*.hdf5")), "minmax_m11", "minmax_m11")
    assert np.allclose(abs_stats["qpos_min"][:2] - rel_stats["qpos_min"][:2], origin.xy, atol=1e-3)
    assert np.allclose(abs_stats["qpos_min"][2:], rel_stats["qpos_min"][2:])   # z, rot, force


# ---------------------------------------------------------------- [5] audit

def test_audit_passes_on_this_package():
    from stain_relative_frame.audit_inference_path import PKG_DIR, audit

    result = audit([PKG_DIR])
    assert result["findings"] == [], result["findings"]
    assert all(n["verdict"] == "PASS" for n in result["notes"]), result["notes"]


def test_audit_catches_open_coded_arithmetic_and_per_step_recompute(tmp_path):
    from stain_relative_frame.audit_inference_path import PKG_DIR, audit

    bad = tmp_path / "bad.py"
    bad.write_text(
        "class B:\n"
        "    def _on_image(self, msg):\n"
        "        det = detect_stain_origin(msg, self.ref, self.H)\n"
        "        self.origin = det.origin_mm\n"
        "        return self.pose - self.origin\n"
    )
    findings = audit([PKG_DIR, tmp_path])["findings"]
    assert {f["check"] for f in findings} == {"C", "D"}


def test_audit_ignores_prose_about_the_rule(tmp_path):
    from stain_relative_frame.audit_inference_path import PKG_DIR, audit

    prose = tmp_path / "prose.py"
    prose.write_text(
        '"""We subtract stain_origin: obs - stain_origin, then + origin later."""\n'
        "# comment: origin - something\n"
        "def fine(x, adapter):\n"
        "    return adapter.observation(x)   # -- origin cannot drift\n"
    )
    assert audit([PKG_DIR, tmp_path])["findings"] == []


def test_checkpoint_flag_drives_the_inference_adapter(tmp_path):
    import pickle
    from stain_relative_frame.inference_adapter import read_use_relative
    from stain_relative_frame.relative_frame import (
        TRANSFORM_VERSION, TRANSFORM_VERSION_ATTR, USE_RELATIVE_ATTR,
    )

    p = tmp_path / "dataset_stats.pkl"
    with open(p, "wb") as f:
        pickle.dump({USE_RELATIVE_ATTR: True, TRANSFORM_VERSION_ATTR: TRANSFORM_VERSION}, f)
    assert read_use_relative(p) == (True, TRANSFORM_VERSION)

    with open(p, "wb") as f:
        pickle.dump({"qpos_min": np.zeros(9)}, f)   # a pre-existing checkpoint
    assert read_use_relative(p)[0] is False         # defaults to the old path


# ------------------------------------------------ [1a] marker-free jog calibrate

def _textured_image(h=240, w=424, seed=0):
    """Broadband texture like a machined / partly-polished metal surface."""
    rng = np.random.default_rng(seed)
    import cv2
    base = rng.normal(128, 45, (h, w)).astype(np.float32)
    fine = cv2.GaussianBlur(base, (0, 0), 0.8)
    coarse = cv2.GaussianBlur(rng.normal(0, 55, (h, w)).astype(np.float32), (0, 0), 6.0)
    return np.clip(fine + coarse, 0, 255)


def test_surface_warp_recovers_a_known_homography():
    import cv2
    from stain_relative_frame.homography_jog_calibrate import surface_warp

    ref = _textured_image(seed=1)
    h, w = ref.shape
    for truth in [(3.0, -2.0), (-7.0, 4.0), (0.0, 0.0)]:
        M = np.array([[1, 0, truth[0]], [0, 1, truth[1]]], dtype=np.float32)
        cur = cv2.warpAffine(ref, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        G, n_inl, rms, flow = surface_warp(ref, cur, feat_roi=(15, 15, w - 15, h - 15),
                                           tool_box=(0, 0, 0, 0))
        assert G is not None and n_inl >= 40
        moved = cv2.perspectiveTransform(np.array([[[200.0, 120.0]]]), G).reshape(2)
        assert np.allclose(moved - [200.0, 120.0], truth, atol=0.4), (moved, truth)
        assert rms < 0.6


def test_jog_correspondences_fit_a_homography():
    """G_i^{-1}(tip_px)  <->  base XY  must be a clean homography."""
    import cv2
    from stain_relative_frame.homography_jog_calibrate import surface_warp
    from stain_relative_frame.homography import fit_homography

    ref = _textured_image(seed=2)
    h, w = ref.shape
    tip_px = np.array([[[253.0, 120.0]]])
    # camera translation parallel to the plane => image translation; the
    # 45deg mount + ~0.6 px/mm scale sets how base mm map to scene pan.
    theta = np.deg2rad(45.0)
    A = 0.6 * np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    home_xy = np.array([445.0, 394.0])

    robot, pixel = [], []
    for dx in (-40, -20, 0, 20, 40):
        for dy in (-36, -18, 0, 18, 36):
            s = -A @ np.array([dx, dy])                       # scene pan, px
            M = np.array([[1, 0, s[0]], [0, 1, s[1]]], dtype=np.float32)
            cur = cv2.warpAffine(ref, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
            G, *_ = surface_warp(ref, cur, feat_roi=(15, 15, w - 15, h - 15),
                                 tool_box=(0, 0, 0, 0))
            px_home = cv2.perspectiveTransform(tip_px, np.linalg.inv(G)).reshape(2)
            robot.append(home_xy + np.array([dx, dy]))
            pixel.append(px_home)

    res = fit_homography(np.asarray(pixel), np.asarray(robot), n_fit=19, tol_mm=2.0)
    assert res.passed, res.summary()
    assert res.max_heldout_error_mm < 1.0


# --------------------------------------------- [1] depth extrinsic calibration

def test_solve_base_rotation_recovers_a_known_rotation():
    from scipy.spatial.transform import Rotation
    from stain_relative_frame.homography_depth_calibrate import solve_base_rotation

    rng = np.random.default_rng(3)
    R_bc_true = Rotation.from_euler("xyz", [40, 5, 135], degrees=True).as_matrix()
    dxdy = rng.uniform(-40, 40, (14, 2))
    T_cam = np.array([-(R_bc_true @ [dx, dy, 0.0]) for dx, dy in dxdy])
    T_cam += rng.normal(0, 0.05, T_cam.shape)               # 0.05 mm ICP noise

    R_bc, a, b, ortho_dev = solve_base_rotation(dxdy, T_cam)
    assert abs(a - 1) < 0.02 and abs(b - 1) < 0.02
    assert ortho_dev < 1.0
    # columns 0,1 (base x,y axes in cam frame) must match
    assert np.allclose(R_bc[:, :2], R_bc_true[:, :2], atol=0.02)


def test_tip_anchor_and_plane_reprojection_round_trip():
    from scipy.spatial.transform import Rotation
    from stain_relative_frame.homography_depth_calibrate import (
        pixels_to_plane, project, solve_tcb,
    )

    K = (301.877, 301.822, 216.737, 125.092)
    R_cb_true = Rotation.from_euler("xyz", [-135, 3, 2], degrees=True).as_matrix()
    t_cb_true = np.array([445.0, 394.0, 560.0])              # cam origin in base
    z0 = 167.0
    home_xyz = np.array([445.0, 394.0, 220.0])

    # tip_px is DERIVED so the scene is self-consistent: the tip at home_xyz
    # must actually project where we say it does.
    X_tip_cam = R_cb_true.T @ (home_xyz - t_cb_true)
    tip_px = tuple(project(X_tip_cam, K).reshape(2))
    z_tip_true = float(np.linalg.norm(X_tip_cam))

    n_base = np.array([0.0, 0.0, 1.0])
    n_cam = R_cb_true.T @ n_base
    if n_cam[2] > 0:
        n_cam = -n_cam
    p_cam = R_cb_true.T @ (np.array([445.0, 394.0, z0]) - t_cb_true)
    d_cam = float(n_cam @ p_cam)

    t_cb, z_tip = solve_tcb(R_cb_true, K, tip_px, n_cam, d_cam, home_xyz, z0)
    assert np.allclose(t_cb, t_cb_true, atol=1e-3), (t_cb, t_cb_true)
    assert abs(z_tip - z_tip_true) < 1e-3

    # a pixel that views a known base point must reproject to it
    known_base = np.array([[430.0, 380.0, z0], [470.0, 410.0, z0]])
    known_cam = (known_base - t_cb_true) @ R_cb_true            # = R_cb_true.T @ x, batched
    px = project(known_cam, K)
    got = pixels_to_plane(px, K, R_cb_true, t_cb, z0)
    assert np.allclose(got[:, :2], known_base[:, :2], atol=0.05)


def test_plane_homography_matches_pixels_to_plane():
    from scipy.spatial.transform import Rotation
    from stain_relative_frame.homography import apply_homography, plane_homography
    from stain_relative_frame.homography_depth_calibrate import pixels_to_plane

    K = (301.877, 301.822, 216.737, 125.092)
    R_cb = Rotation.from_euler("xyz", [-134, 4, 1], degrees=True).as_matrix()
    t_cb = np.array([540.0, 525.0, 388.0])
    z0 = 167.0
    px = np.array([[200.0, 120.0], [150.0, 90.0], [300.0, 160.0], [253.0, 118.0]])
    H = plane_homography(K, R_cb, t_cb, z0)
    assert np.allclose(apply_homography(H, px), pixels_to_plane(px, K, R_cb, t_cb, z0)[:, :2],
                       atol=1e-6)


def test_extrinsic_at_pose_is_identity_at_home_and_tracks_translation():
    from scipy.spatial.transform import Rotation
    from stain_relative_frame.homography import extrinsic_at_pose

    R_home = Rotation.from_rotvec([0.0, 0.0, 2.356]).as_matrix()
    t_home = np.array([540.0, 525.0, 388.0])
    home_pose6 = np.array([445.0, 394.0, 220.0, 0.0, 0.0, 2.356])

    R0, t0 = extrinsic_at_pose(R_home, t_home, home_pose6, home_pose6)
    assert np.allclose(R0, R_home) and np.allclose(t0, t_home)

    # pure TCP translation -> camera translates by the same base vector
    moved = home_pose6.copy(); moved[:3] += [30.0, -20.0, 5.0]
    R1, t1 = extrinsic_at_pose(R_home, t_home, home_pose6, moved)
    assert np.allclose(R1, R_home, atol=1e-9)
    assert np.allclose(t1 - t_home, [30.0, -20.0, 5.0], atol=1e-6)


def test_dark_blob_finds_a_black_strip_without_a_reference():
    import cv2
    from stain_relative_frame.stain_detect import detect_stain_dark

    img = np.full((240, 424, 3), 180, np.uint8)                 # bright plate
    cv2.rectangle(img, (190, 70), (210, 150), (12, 12, 12), -1)  # vertical black strip
    img[:] = np.clip(img.astype(int) + np.random.default_rng(0).integers(-8, 8, img.shape), 0, 255)
    H = np.eye(3)                                                # pixel == mm for the test
    det = detect_stain_dark(img, H, plate_roi=(120, 20, 340, 200), tool_box=(0, 0, 0, 0))
    assert det.ok
    assert abs(det.centroid_px[0] - 200) < 3 and abs(det.centroid_px[1] - 110) < 4


def test_icp_translation_recovers_a_shift_with_structure():
    """A tilted plate plus a few raised pads (bolt heads / fixture) -- the
    pads' edges give in-plane observability, like the real scene. Rendered
    analytically per scene translation, so D0/D1 carry no resampling error."""
    from stain_relative_frame.homography_depth_calibrate import icp_translation

    K = (301.877, 301.822, 216.737, 125.092)
    fx, fy, cx, cy = K
    H, W = 240, 424
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    rx = (uu - cx) / fx
    ry = (vv - cy) / fy

    # world surface (t=0 frame):  Z = plate(X,Y) + ripple(X,Y)
    a, b, c = -0.07, -0.98, 300.0                   # Z = a X + b Y + c  (tilted plate)

    def gworld(X, Y):
        return (a * X + b * Y + c
                + 4.0 * np.cos(X / 22.0) + 3.0 * np.sin(Y / 19.0)
                + 2.5 * np.cos((X + Y) / 15.0))

    def render(t):
        s = np.full_like(rx, c)                     # fixed-point solve for depth
        for _ in range(40):
            s = gworld(s * rx - t[0], s * ry - t[1]) + t[2]
        s[s <= 0] = 0.0
        return s.astype(np.float32)

    D0 = render(np.zeros(3))
    t_true = np.array([6.0, -3.0, 2.0])
    D1 = render(t_true)

    mask = (D0 > 0) & (D1 > 0)
    mask[:10] = mask[-10:] = mask[:, :10] = mask[:, -10:] = False
    t, rms, n = icp_translation(D0, D1, K, mask, iters=30)
    assert np.allclose(t, t_true, atol=0.4), (t, t_true)
