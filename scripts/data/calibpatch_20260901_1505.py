#!/usr/bin/env python3
"""Build 20260901_1505_calibpatch: drop bad episodes + apply the constant
kinematic-calibration offset that aligns the ep15-49 regime onto the ep1-11
reference, and pull both regimes' contact z to the true surface (z=173.0).

Removed (8): ep 0, 2, 12, 13, 14, 40, 43, 46
  0,2      -> merged/malformed contact (1 phase, >65% in contact, noisy z)
  12,13,14 -> gross 3-axis frame offset (contact at x+25 y-15 z+26 mm)
  40,43    -> phantom mid-air contact segment + ends |F|~8-9N (truncated)
  46       -> starts |F|~6N (fx bias ~-4N), no clean initial non-contact

Patch (added to observations/position AND action/position; force & images untouched):
  regime A  ep 1,3-11        : dz = -1.4061                         (173.0 target)
  regime C  ep 15-49 (kept)  : [dx,dy,dz,dwx,dwy,dwz] =
             [-8.3849, +3.6371, -6.3002, +0.0148, +0.0117, +0.0460]
  (regC contact-phase mean -> regA contact-phase mean for x,y,wx,wy,wz;
   z -> 173.0 absolute for both regimes)
"""
import json, shutil, sys
from pathlib import Path
import h5py, numpy as np

SRC = Path("datasets/polishing/single_cam/20260901_1505/imitation_form")
DST = Path("datasets/polishing/single_cam/20260901_1505_calibpatch/imitation_form")

REMOVE = {0, 2, 12, 13, 14, 40, 43, 46}
REG_A = {1, 3, 4, 5, 6, 7, 8, 9, 10, 11}
DELTA_A = np.array([0.0, 0.0, -1.4061, 0.0, 0.0, 0.0])
DELTA_C = np.array([-8.3849, 3.6371, -6.3002, 0.0148, 0.0117, 0.0460])

def main():
    DST.mkdir(parents=True, exist_ok=True)
    kept = sorted(e for e in range(50) if e not in REMOVE)
    manifest = {"source": str(SRC), "removed": sorted(REMOVE),
                "delta_regA": DELTA_A.tolist(), "delta_regC": DELTA_C.tolist(),
                "regA_episodes": sorted(REG_A), "episodes": []}
    for new_i, old_i in enumerate(kept):
        src = SRC / f"episode_{old_i}.hdf5"
        dst = DST / f"episode_{new_i}.hdf5"
        regime = "A" if old_i in REG_A else "C"
        delta = DELTA_A if regime == "A" else DELTA_C
        shutil.copy2(src, dst)
        with h5py.File(dst, "a") as f:
            for key in ("observations/position", "action/position"):
                p = np.asarray(f[key], dtype=np.float64) + delta
                f[key][...] = p.astype(f[key].dtype)
            f.attrs["calib_patch"] = "20260901_1505_calibpatch_v1"
            f.attrs["calib_patch_regime"] = regime
            f.attrs["calib_patch_src_episode"] = int(old_i)
            f.attrs["calib_patch_delta_pos6"] = delta
        with h5py.File(dst, "r") as f:
            of = np.asarray(f["observations/force"], float)
            pos = np.asarray(f["observations/position"], float)
            fm = np.linalg.norm(of, axis=1)
            c = fm > max(4.0, np.median(fm[:20]) + 2.0)
            zc = float(np.median(pos[c, 2])) if c.any() else float("nan")
        manifest["episodes"].append(
            {"new": new_i, "old": old_i, "regime": regime, "contact_z_after": round(zc, 2)})
        print(f"episode_{new_i:2d}  <- ep{old_i:2d}  ({regime})  contact_z->{zc:.2f}")
    (DST.parent / "calib_patch_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(kept)} episodes -> {DST}")
    print(f"manifest -> {DST.parent/'calib_patch_manifest.json'}")

if __name__ == "__main__":
    main()
