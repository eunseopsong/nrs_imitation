#!/usr/bin/env python3
"""Build 20260901_2141_calibpatch from 20260901_2141_clean (3-dot-row removal,
= the daytime 2-dot row + 1 extra dot along the same diagonal).

Steps:
  1. drop episode_31 (src ep_0035): contact-z std 14.5mm, 4 contact segments,
     shallow lifts, trailing scrape -- messier than the rest.
  2. per-regime constant offset added to observations/position AND
     action/position (force + images untouched):
       MAIN (27 eps: 0-7,13-30,32) dxy=(+1.292,-6.593)  dz=+1.993
       LOW  (5 eps: 8,9,10,11,12)  dxy=(+5.538,-5.730)  dz=+9.574
     xy: each regime's first-two-dot contact centroid -> the
         20260901_1505_calibpatch 2-dot centroid (so the 3-dot row sits on
         the same diagonal the 2-dot row extends along; 3rd dot then lands
         at ~2.05x the 2-dot spacing, ~2mm off the line).
     z : each regime's contact-phase z median -> 173.0.
     orientation (wx/wy/wz): untouched.
  2b. extra xy shift for new episode_0..7 (= clean 0-7, merged ep_0000-0008):
     the stain was drawn in the wrong place for those first takes. Shift
     their dot1/dot2 contact centroid onto the pooled centroid of the OTHER
     episodes (new 8..31): dxy = (-12.952, +21.301). Later drifting episodes
     (new ~17-24) are left as-is per user.
  3. force gate: contact <=> observations z < 185.0 mm; fx/fy/fz set to 0 in
     every non-contact frame. Applied to observations/force and action/force.

Renumbered episode_0..episode_31 (32 kept).
"""
import json, shutil
from pathlib import Path
import h5py, numpy as np

SRC = Path("datasets/polishing/single_cam/20260901_2141_clean/imitation_form")
DST = Path("datasets/polishing/single_cam/20260901_2141_calibpatch/imitation_form")

DROP = {31}
LOW = {8, 9, 10, 11, 12}
DELTA_MAIN = np.array([ 1.292, -6.593,  1.993, 0.0, 0.0, 0.0])
DELTA_LOW  = np.array([ 5.538, -5.730,  9.574, 0.0, 0.0, 0.0])
# new episode_0..7 (clean 0-7): stain mis-drawn on the first takes -> extra
# xy shift onto the pooled dot1/dot2 centroid of the other episodes (8..31).
EP07_CLEAN = {0, 1, 2, 3, 4, 5, 6, 7}
DELTA_EP07_XY = np.array([-12.952, 21.301, 0.0, 0.0, 0.0, 0.0])
Z_GATE = 185.0

def main():
    DST.mkdir(parents=True, exist_ok=True)
    kept = sorted(e for e in range(33) if e not in DROP)
    manifest = {"source": str(SRC), "dropped_clean_idx": sorted(DROP),
                "delta_main": DELTA_MAIN.tolist(), "delta_low": DELTA_LOW.tolist(),
                "low_clean_idx": sorted(LOW), "force_gate_z_threshold": Z_GATE,
                "ep0-7_extra_xy": DELTA_EP07_XY[:2].tolist(),
                "ep0-7_clean_idx": sorted(EP07_CLEAN),
                "episodes": []}
    for new_i, clean_i in enumerate(kept):
        regime = "LOW" if clean_i in LOW else "MAIN"
        delta = (DELTA_LOW if clean_i in LOW else DELTA_MAIN).copy()
        if clean_i in EP07_CLEAN:
            delta = delta + DELTA_EP07_XY
            regime = "MAIN_ep0-7_xyfix"
        src = SRC / f"episode_{clean_i}.hdf5"
        dst = DST / f"episode_{new_i}.hdf5"
        shutil.copy2(src, dst)
        with h5py.File(dst, "a") as f:
            for key in ("observations/position", "action/position"):
                p = np.asarray(f[key], dtype=np.float64) + delta
                f[key][...] = p.astype(f[key].dtype)
            z = np.asarray(f["observations/position"][:, 2], dtype=np.float64)
            contact = z < Z_GATE
            for key in ("observations/force", "action/force"):
                fo = np.asarray(f[key], dtype=np.float64)
                fo[~contact] = 0.0
                f[key][...] = fo.astype(f[key].dtype)
            f.attrs["calib_patch"] = "20260901_2141_calibpatch_v1"
            f.attrs["calib_patch_regime"] = regime
            f.attrs["calib_patch_src_clean_episode"] = int(clean_i)
            f.attrs["calib_patch_src_merged_episode"] = int(f.attrs.get("clean_src_episode", -1))
            f.attrs["calib_patch_delta_pos6"] = delta
            f.attrs["force_noncontact_zeroed"] = 1
            f.attrs["force_contact_gate"] = f"observations z < {Z_GATE} mm"
            f.attrs["force_gate_z_threshold"] = float(Z_GATE)
            zc = float(np.median(z[contact])) if contact.any() else float("nan")
            nz = int((~contact).sum()); T = len(z)
        manifest["episodes"].append({"new": new_i, "clean": clean_i, "regime": regime,
                                     "contact_z_after": round(zc, 2),
                                     "noncontact_frames_zeroed": nz, "contact_frac": round(1-nz/T, 3)})
        print(f"episode_{new_i:2d} <- clean episode_{clean_i:2d} ({regime})  contact_z->{zc:.2f}  contact {100*(1-nz/T):.0f}%")
    (DST.parent / "calib_patch_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(kept)} episodes -> {DST}")

if __name__ == "__main__":
    main()
