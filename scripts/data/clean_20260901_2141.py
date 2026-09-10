#!/usr/bin/env python3
"""Build 20260901_2141_clean: drop the episodes whose position feed was cut
mid-recording (multi-second time gaps in sample_time_unix -> teleporting /
filter-smeared trajectory in the last 5-20% of the take).

Removed (7): imitation_form episode_3, 9, 15, 20, 37, 38, 39
  (merged ep_0003/0009/0015/0020/0037/0038/0039). Lost 4.5-18s each; see
  scratchpad/gaps_2141.png and scan_dataset_calibration.py-style analysis.
Kept: the other 33 episodes (only isolated 1-2 frame drops, filter-clean).
No data transform -- just selection + renumber episode_0..episode_32.
"""
import json, shutil
from pathlib import Path
import h5py

SRC = Path("datasets/polishing/single_cam/20260901_2141/imitation_form")
DST = Path("datasets/polishing/single_cam/20260901_2141_clean/imitation_form")
REMOVE = {3, 9, 15, 20, 37, 38, 39}

def main():
    DST.mkdir(parents=True, exist_ok=True)
    kept = sorted(e for e in range(40) if e not in REMOVE)
    manifest = {"source": str(SRC), "removed_imitation_form_idx": sorted(REMOVE),
                "reason": "position feed cut mid-recording (multi-second gaps, teleport)",
                "episodes": []}
    for new_i, old_i in enumerate(kept):
        src = SRC / f"episode_{old_i}.hdf5"
        dst = DST / f"episode_{new_i}.hdf5"
        shutil.copy2(src, dst)
        with h5py.File(dst, "a") as f:
            f.attrs["clean_src_episode"] = int(old_i)
            f.attrs["clean_dataset"] = "20260901_2141_clean_v1"
        manifest["episodes"].append({"new": new_i, "old": old_i})
        print(f"episode_{new_i:2d}  <- episode_{old_i}")
    (DST.parent / "clean_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(kept)} episodes -> {DST}")
    print(f"manifest -> {DST.parent/'clean_manifest.json'}")

if __name__ == "__main__":
    main()
