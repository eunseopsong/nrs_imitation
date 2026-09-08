#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step [4] GATE: does anything except the image still give the defect type away?

Four checks, all on the FIRST frame of each episode, split 8:2 by EPISODE and
repeated over many splits:

  (a) relative position -> defect type.  Must be near chance.
  (b) force             -> defect type.  Must be near chance. Removing the
      absolute position can push the shortcut onto force, so this is not
      optional.
  (c) label-shuffle control: (a) rerun with the labels permuted. Must land on
      chance -- if it does not, the split leaks and (a)'s result means nothing.
  (d) scatter of the relative start positions per type. The distributions have
      to overlap; the plot is saved for inspection.

For reference the same probe is run on the ABSOLUTE position ("baseline"),
which is expected to be HIGH -- that is the shortcut being removed, and a
baseline that is already at chance means the conversion is not what is being
measured.

Any of (a), (b) materially above chance -> BLOCKED. That is a data-collection
problem: the defect types were not drawn at overlapping locations / contact
conditions. No code change fixes it.

  ros2 run stain_relative_frame validate_shortcut -- \
      --dataset_dir datasets/polishing/single_cam/<run>/imitation_form_rel
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from .config import load_config
from .linear_probe import probe_with_null, repeated_probe
from .relative_frame import ABS_OBS_POSITION_KEY, USE_RELATIVE_ATTR
from .report import table, verdict_line

LABEL_ATTR = "stain_direction_deg"


def load_first_frames(dataset_dir: Path, label_attr: str = LABEL_ATTR) -> Dict:
    """One row per episode: first-frame position, force, and the label."""
    files = sorted(dataset_dir.glob("episode_*.hdf5")) or sorted(dataset_dir.glob("episode_*.h5"))
    if not files:
        raise FileNotFoundError(f"no episode_*.hdf5 under {dataset_dir}")

    pos, force, absolute, labels, names, missing = [], [], [], [], [], []
    use_relative = None
    for p in files:
        with h5py.File(str(p), "r") as f:
            if label_attr not in f.attrs:
                missing.append(p.stem)
                continue
            labels.append(int(f.attrs[label_attr]))
            names.append(p.stem)
            pos.append(np.asarray(f["observations/position"][0], dtype=np.float64))
            force.append(np.asarray(f["observations/force"][0], dtype=np.float64))
            if ABS_OBS_POSITION_KEY in f:
                absolute.append(np.asarray(f[ABS_OBS_POSITION_KEY][0], dtype=np.float64))
            if use_relative is None and USE_RELATIVE_ATTR in f.attrs:
                use_relative = bool(int(f.attrs[USE_RELATIVE_ATTR]))
    if not labels:
        raise ValueError(
            f"no episode under {dataset_dir} carries the '{label_attr}' attribute -- "
            f"the gate needs defect-type labels"
        )
    return {
        "position": np.stack(pos),
        "force": np.stack(force),
        "absolute": np.stack(absolute) if absolute else None,
        "labels": np.asarray(labels, dtype=np.int64),
        "names": names,
        "missing_label": missing,
        "use_relative": use_relative,
    }


def _verdict(res: dict, margin: float, alpha: float = 0.05) -> Tuple[bool, float]:
    """Is this result "near chance"?

    Two conditions must BOTH hold for a FAIL, because either one alone
    misreads this project's episode counts:

      * accuracy above chance + margin -- the specified rule; on its own it
        fires on noise, because ~17 test episodes make chance-level accuracy
        swing by 6 points per episode.
      * permutation p < alpha -- the result is also outside the null this
        dataset produces when the labels carry nothing. A result inside its
        own null IS near chance, whatever the point estimate reads.

    So a large-but-not-significant accuracy passes, and a modest-but-highly-
    significant one fails. Both numbers are reported either way.
    """
    chance = max(res["chance_uniform"], res["chance_majority"])
    above = bool(res["accuracy_mean"] > chance + margin)
    p = res.get("p_value")
    significant = True if p is None or not np.isfinite(p) else bool(p < alpha)
    return (not (above and significant)), chance


def _row(name: str, res: dict, margin: float, expect: str, alpha: float = 0.05) -> dict:
    near, chance = _verdict(res, margin, alpha)
    verdict = ("PASS" if near else "FAIL") if expect == "near_chance" else "info"
    p = res.get("p_value")
    return {
        "check": name,
        "acc_mean": f"{res['accuracy_mean']:.3f}",
        "acc_std": f"{res['accuracy_std']:.3f}",
        "chance": f"{chance:.3f}",
        "null_mean": f"{res.get('null_mean', float('nan')):.3f}",
        "null_p95": f"{res.get('null_p95', float('nan')):.3f}",
        "p_value": "n/a" if p is None else f"{p:.4f}",
        "splits": res["n_splits"],
        "verdict": verdict,
    }


def scatter_plot(data: Dict, out_path: Path, title: str) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[validate] (d) scatter skipped, matplotlib unavailable: {exc}",
              file=sys.stderr)
        return None

    labels = data["labels"]
    uniq = np.unique(labels)
    has_abs = data["absolute"] is not None
    fig, axes = plt.subplots(1, 2 if has_abs else 1, figsize=(11 if has_abs else 6, 5),
                             squeeze=False)

    def draw(ax, xy, name):
        for c in uniq:
            m = labels == c
            ax.scatter(xy[m, 0], xy[m, 1], s=42, alpha=0.75, label=f"{int(c)} deg (n={m.sum()})")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=8)

    draw(axes[0][0], data["position"][:, :2],
         "relative start position\n(distributions must OVERLAP)")
    if has_abs:
        draw(axes[0][1], data["absolute"][:, :2],
             "absolute start position\n(the shortcut being removed)")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def overlap_metrics(xy: np.ndarray, labels: np.ndarray) -> List[dict]:
    """Per-class centroid + spread, and the class separation in spread units.

    Two clusters whose centroids sit several spreads apart do not overlap, no
    matter what the probe accuracy says on a small sample.
    """
    rows = []
    uniq = np.unique(labels)
    cents, spreads = {}, {}
    for c in uniq:
        m = labels == c
        cents[c] = xy[m].mean(axis=0)
        spreads[c] = xy[m].std(axis=0)
    for c in uniq:
        rows.append({
            "type": int(c),
            "n": int((labels == c).sum()),
            "cx_mm": f"{cents[c][0]:.2f}", "cy_mm": f"{cents[c][1]:.2f}",
            "sx_mm": f"{spreads[c][0]:.2f}", "sy_mm": f"{spreads[c][1]:.2f}",
        })
    for i, a in enumerate(uniq):
        for b in uniq[i + 1:]:
            d = np.linalg.norm(cents[a] - cents[b])
            pooled = float(np.sqrt(((spreads[a] ** 2).sum() + (spreads[b] ** 2).sum()) / 2.0))
            rows.append({
                "type": f"{int(a)} vs {int(b)}",
                "n": "-",
                "cx_mm": f"d={d:.2f}", "cy_mm": "-",
                "sx_mm": f"pooled={pooled:.2f}",
                "sy_mm": f"sep={d / max(pooled, 1e-6):.2f} sd",
            })
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="[4] pre-training shortcut-leakage gate")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--label_attr", type=str, default=LABEL_ATTR)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--test_fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--margin", type=float, default=None,
                    help="accuracy above chance+margin is the first FAIL condition")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="permutation p below this is the second FAIL condition")
    ap.add_argument("--n_permutations", type=int, default=200,
                    help="label shuffles used to build the empirical null")
    ap.add_argument("--position_dims", type=int, default=2,
                    help="2 = xy only (what the transform touches); 6 = full pose")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--plot", type=str, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    repeats = int(args.repeats or cfg.val_repeats)
    test_fraction = float(args.test_fraction or cfg.val_test_fraction)
    seed = int(args.seed if args.seed is not None else cfg.val_seed)
    margin = float(args.margin if args.margin is not None else cfg.val_chance_margin)
    out_path = Path(args.out) if args.out else cfg.path("validation_report_file")
    plot_path = Path(args.plot) if args.plot else out_path.with_name("validation_scatter.png")

    d = Path(args.dataset_dir).expanduser()
    data = load_first_frames(d, args.label_attr)
    labels = data["labels"]
    n_classes = int(np.unique(labels).size)
    pos = data["position"][:, : int(args.position_dims)]

    print(f"[validate] dataset      : {d}")
    print(f"[validate] episodes     : {labels.shape[0]}"
          + (f"  ({len(data['missing_label'])} without a label, skipped)"
             if data["missing_label"] else ""))
    print(f"[validate] defect types : {n_classes} -> "
          + ", ".join(f"{int(c)}deg x{int((labels == c).sum())}" for c in np.unique(labels)))
    if n_classes < 2 or min(int((labels == c).sum()) for c in np.unique(labels)) < 5:
        print(f"[validate] CANNOT RUN THE GATE: need >=2 defect types with >=5 episodes "
              f"each, got {n_classes} type(s) "
              + ", ".join(f"{int(c)}x{int((labels == c).sum())}" for c in np.unique(labels))
              + ".\n           Upstream [2] stain detection probably dropped most "
              "episodes (check stain_origin_stability.json 'unstable_episodes'). "
              "A gate that cannot classify is a FAIL, not a pass.", file=sys.stderr)
        return 1
    print(f"[validate] use_relative_position attr in the data: {data['use_relative']}")
    if data["use_relative"] is False:
        print("[validate] WARNING: this dataset was written with use_relative_position=False, "
              "so check (a) is measuring ABSOLUTE position.")
    if data["use_relative"] is None:
        print("[validate] WARNING: no use_relative_position attribute -- this looks like a "
              "raw dataset that never went through step [3].")
    print(f"[validate] protocol     : {repeats} stratified episode-level splits, "
          f"test_fraction={test_fraction}, chance margin={margin}\n")

    n_perm = int(args.n_permutations)
    alpha = float(args.alpha)
    probe = lambda X: probe_with_null(X, labels, repeats, test_fraction, seed, n_perm=n_perm)

    res_a = probe(pos)
    res_b = probe(data["force"])
    res_c = repeated_probe(pos, labels, int(cfg.val_shuffle_repeats), test_fraction,
                           seed + 1000, shuffle_labels=True)
    res_c.update({"null_mean": res_a.get("null_mean"), "null_p95": res_a.get("null_p95")})
    res_ab = probe(np.concatenate([pos, data["force"]], axis=1))

    rows = [
        _row("(a) relative position -> type", res_a, margin, "near_chance", alpha),
        _row("(b) force -> type", res_b, margin, "near_chance", alpha),
        _row("(c) shuffled-label control", res_c, margin, "near_chance", alpha),
        _row("    position+force (joint)", res_ab, margin, "near_chance", alpha),
    ]
    baseline = None
    if data["absolute"] is not None:
        baseline = probe(data["absolute"][:, : int(args.position_dims)])
        rows.append(_row("    absolute position (baseline)", baseline, margin, "baseline", alpha))

    print(table(rows, title="[4] shortcut-leakage gate"))

    print()
    print(table(overlap_metrics(data["position"][:, :2], labels),
                title="[4d] relative start position per type"))

    saved_plot = scatter_plot(data, plot_path, f"[4d] start positions -- {d.name}")
    if saved_plot:
        print(f"\n[validate] (d) scatter -> {saved_plot}")

    pass_a, chance_a = _verdict(res_a, margin, alpha)
    pass_b, chance_b = _verdict(res_b, margin, alpha)
    pass_c, chance_c = _verdict(res_c, margin, alpha)
    passed = pass_a and pass_b and pass_c

    print()
    print(verdict_line("(a) relative position near chance", pass_a,
                       f"{res_a['accuracy_mean']:.3f} vs chance {chance_a:.3f}, "
                       f"p={res_a.get('p_value', float('nan')):.4f}"))
    print(verdict_line("(b) force near chance", pass_b,
                       f"{res_b['accuracy_mean']:.3f} vs chance {chance_b:.3f}, "
                       f"p={res_b.get('p_value', float('nan')):.4f}"))
    print(verdict_line("(c) shuffled-label control at chance", pass_c,
                       f"{res_c['accuracy_mean']:.3f} vs chance {chance_c:.3f}"))
    if baseline is not None:
        print(f"[info] absolute-position baseline: {baseline['accuracy_mean']:.3f} "
              f"(high is expected -- it is the shortcut being removed)")

    payload = {
        "step": "4_validation_gate",
        "stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_dir": str(d),
        "n_episodes": int(labels.shape[0]),
        "n_classes": n_classes,
        "class_counts": {int(c): int((labels == c).sum()) for c in np.unique(labels)},
        "use_relative_position_attr": data["use_relative"],
        "repeats": repeats,
        "test_fraction": test_fraction,
        "chance_margin": margin,
        "alpha": alpha,
        "n_permutations": n_perm,
        "a_relative_position": res_a,
        "b_force": res_b,
        "c_label_shuffle": res_c,
        "joint_position_force": res_ab,
        "baseline_absolute_position": baseline,
        "scatter_plot": None if saved_plot is None else str(saved_plot),
        "passed": bool(passed),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[validate] report -> {out_path}")

    if not passed:
        print("\n[validate] GATE FAILED -> BLOCKED. Do not start training.", file=sys.stderr)
        if not pass_c:
            print("[validate]   (c) failed: the shuffled-label control is above chance, so "
                  "the split itself leaks. Fix that before reading (a) or (b) at all.",
                  file=sys.stderr)
        if not pass_a:
            print("[validate]   (a) failed: the relative start position still predicts the "
                  "defect type. The types were drawn at systematically different offsets "
                  "from their own stain, so subtracting the stain origin does not make them "
                  "overlap. This is a data-collection problem -- vary the approach offset "
                  "per episode independently of the defect type and re-record.",
                  file=sys.stderr)
        if not pass_b:
            print("[validate]   (b) failed: the first-frame force predicts the defect type. "
                  "The contact conditions differ per type (approach speed, dwell, pressure). "
                  "Re-record with the contact protocol held constant across types.",
                  file=sys.stderr)
        return 1
    print("\n[validate] GATE PASSED -> READY_TO_TRAIN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
