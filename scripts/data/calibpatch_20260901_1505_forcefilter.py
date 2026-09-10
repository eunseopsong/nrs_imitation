#!/usr/bin/env python3
"""Force-gate the 20260901_1505_calibpatch dataset: zero fx/fy/fz in every
non-contact frame, where contact is decided purely by end-effector height
(obs z near the 173 mm surface plane).

Runs in place on datasets/polishing/single_cam/20260901_1505_calibpatch/
imitation_form/episode_*.hdf5 AFTER calibpatch_20260901_1505.py (needs the
z already corrected to the 173 plane).

Gate: contact  <=>  observations/position[:,2] < Z_THRESHOLD (185.0 mm).
  contact-phase z ranges ~170-184; the lift between the two dots peaks at
  ~194-215 and the retract at 215+, so 185 sits in a clean ~10 mm gap
  (verified: zeroes 0.5% of |F|>5 frames, keeps 0% of the lift/retract).
Both observations/force and action/force are gated (identical arrays).
Force inside contact is left untouched.
"""
import json
from pathlib import Path
import h5py, numpy as np

DST = Path("datasets/polishing/single_cam/20260901_1505_calibpatch/imitation_form")
Z_THRESHOLD = 185.0

def main():
    paths = sorted(DST.glob("episode_*.hdf5"), key=lambda p: int(p.stem.split("_")[1]))
    rep = []
    for p in paths:
        with h5py.File(p, "a") as f:
            if f.attrs.get("force_noncontact_zeroed", 0):
                print(f"{p.name}: already gated, skipping"); continue
            z = np.asarray(f["observations/position"][:, 2], dtype=np.float64)
            contact = z < Z_THRESHOLD
            for key in ("observations/force", "action/force"):
                fo = np.asarray(f[key], dtype=np.float64)
                fo[~contact] = 0.0
                f[key][...] = fo.astype(f[key].dtype)
            f.attrs["force_noncontact_zeroed"] = 1
            f.attrs["force_contact_gate"] = f"observations z < {Z_THRESHOLD} mm"
            f.attrs["force_gate_z_threshold"] = float(Z_THRESHOLD)
            T = len(z); nz = int((~contact).sum())
        rep.append({"episode": p.name, "T": T, "noncontact_frames_zeroed": nz,
                    "contact_frac": round(1 - nz / T, 3)})
        print(f"{p.name}: zeroed {nz}/{T} frames ({100*(1-nz/T):.0f}% contact)")

    mpath = DST.parent / "calib_patch_manifest.json"
    m = json.loads(mpath.read_text())
    m["force_gate"] = {"z_threshold_mm": Z_THRESHOLD,
                       "rule": "contact <=> observations z < z_threshold; fx/fy/fz set to 0 elsewhere",
                       "applies_to": ["observations/force", "action/force"],
                       "per_episode": rep}
    mpath.write_text(json.dumps(m, indent=2))
    print(f"\nmanifest updated -> {mpath}")

if __name__ == "__main__":
    main()
