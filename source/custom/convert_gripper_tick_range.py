#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Copy imitation_form episodes while remapping gripper position ticks.

This is intended for gripper datasets recorded with one open/close tick range
and reused on hardware whose open/close ticks differ.

Only gripper position fields are remapped:
  - action/gripper_present_position
  - observations/gripper/present_position

Gripper current remains in mA and is copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np


GRIPPER_POSITION_KEYS = (
    "action/gripper_present_position",
    "observations/gripper/present_position",
)


def _episode_index(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.split("_")[-1])
    except Exception:
        return 10**12


def _episode_files(input_dir: Path) -> List[Path]:
    files = sorted(input_dir.glob("episode_*.hdf5"), key=_episode_index)
    if not files:
        files = sorted(input_dir.glob("episode_*.h5"), key=_episode_index)
    if not files:
        raise FileNotFoundError(f"No episode_*.hdf5/.h5 files found in {input_dir}")
    return files


def _copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _dataset_create_kwargs(ds: h5py.Dataset) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    if ds.chunks is not None:
        kwargs["chunks"] = ds.chunks
    if ds.compression is not None:
        kwargs["compression"] = ds.compression
        if ds.compression_opts is not None:
            kwargs["compression_opts"] = ds.compression_opts
    if ds.shuffle:
        kwargs["shuffle"] = True
    if ds.fletcher32:
        kwargs["fletcher32"] = True
    return kwargs


def remap_ticks(
    values: np.ndarray,
    src_open_tick: float,
    src_close_tick: float,
    dst_open_tick: float,
    dst_close_tick: float,
    clip: bool = True,
) -> np.ndarray:
    src_span = float(src_close_tick) - float(src_open_tick)
    if abs(src_span) < 1e-9:
        raise ValueError("src_open_tick and src_close_tick must differ")

    alpha = (np.asarray(values, dtype=np.float32) - float(src_open_tick)) / src_span
    out = float(dst_open_tick) + alpha * (float(dst_close_tick) - float(dst_open_tick))
    if clip:
        lo = min(float(dst_open_tick), float(dst_close_tick))
        hi = max(float(dst_open_tick), float(dst_close_tick))
        out = np.clip(out, lo, hi)
    return np.rint(out).astype(np.int32)


def _replace_dataset_with_remapped_ticks(
    src_file: h5py.File,
    dst_file: h5py.File,
    key: str,
    src_open_tick: float,
    src_close_tick: float,
    dst_open_tick: float,
    dst_close_tick: float,
    clip: bool,
) -> Tuple[int, int, int, int]:
    src_ds = src_file[key]
    src_values = np.asarray(src_ds)
    out_values = remap_ticks(
        src_values,
        src_open_tick=src_open_tick,
        src_close_tick=src_close_tick,
        dst_open_tick=dst_open_tick,
        dst_close_tick=dst_close_tick,
        clip=clip,
    )

    parent_name, leaf = key.rsplit("/", 1)
    parent = dst_file[parent_name]
    del parent[leaf]
    dst_ds = parent.create_dataset(
        leaf,
        data=out_values.astype(np.int32),
        dtype=np.int32,
        **_dataset_create_kwargs(src_ds),
    )
    _copy_attrs(src_ds, dst_ds)

    return (
        int(np.min(src_values)),
        int(np.max(src_values)),
        int(np.min(out_values)),
        int(np.max(out_values)),
    )


def convert_episode(
    src_path: Path,
    dst_path: Path,
    src_open_tick: float,
    src_close_tick: float,
    dst_open_tick: float,
    dst_close_tick: float,
    clip: bool,
) -> Dict[str, object]:
    if dst_path.exists():
        dst_path.unlink()

    stats: Dict[str, object] = {"file": dst_path.name, "converted_keys": {}}
    with h5py.File(str(src_path), "r") as src, h5py.File(str(dst_path), "w") as dst:
        for name in src:
            src.copy(name, dst)
        _copy_attrs(src, dst)

        dst.attrs["gripper_tick_range_converted"] = 1
        dst.attrs["gripper_src_open_tick"] = float(src_open_tick)
        dst.attrs["gripper_src_close_tick"] = float(src_close_tick)
        dst.attrs["gripper_dst_open_tick"] = float(dst_open_tick)
        dst.attrs["gripper_dst_close_tick"] = float(dst_close_tick)
        dst.attrs["gripper_tick_conversion_clip"] = int(bool(clip))
        dst.attrs["gripper_tick_conversion_note"] = (
            "linear open/close remap; current_mA copied unchanged"
        )

        for key in GRIPPER_POSITION_KEYS:
            if key not in src:
                continue
            src_min, src_max, dst_min, dst_max = _replace_dataset_with_remapped_ticks(
                src_file=src,
                dst_file=dst,
                key=key,
                src_open_tick=src_open_tick,
                src_close_tick=src_close_tick,
                dst_open_tick=dst_open_tick,
                dst_close_tick=dst_close_tick,
                clip=clip,
            )
            stats["converted_keys"][key] = {
                "src_min": src_min,
                "src_max": src_max,
                "dst_min": dst_min,
                "dst_max": dst_max,
            }

    return stats


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    src_open_tick: float,
    src_close_tick: float,
    dst_open_tick: float,
    dst_close_tick: float,
    clip: bool,
    overwrite: bool,
) -> Dict[str, object]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")

    files = _episode_files(input_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output_dir already exists: {output_dir} (use --overwrite)")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_episode = []
    for src_path in files:
        dst_path = output_dir / src_path.name
        stats = convert_episode(
            src_path=src_path,
            dst_path=dst_path,
            src_open_tick=src_open_tick,
            src_close_tick=src_close_tick,
            dst_open_tick=dst_open_tick,
            dst_close_tick=dst_close_tick,
            clip=clip,
        )
        per_episode.append(stats)
        key_stats = stats.get("converted_keys", {})
        brief = []
        for key, item in key_stats.items():
            brief.append(
                f"{key}: {item['src_min']}..{item['src_max']} -> {item['dst_min']}..{item['dst_max']}"
            )
        print(f"[OK] {src_path.name} | " + "; ".join(brief))

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_episodes": len(per_episode),
        "src_open_tick": float(src_open_tick),
        "src_close_tick": float(src_close_tick),
        "dst_open_tick": float(dst_open_tick),
        "dst_close_tick": float(dst_close_tick),
        "clip": bool(clip),
        "position_keys": list(GRIPPER_POSITION_KEYS),
        "episodes": per_episode,
    }
    summary_path = output_dir / "gripper_tick_range_conversion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    print(f"[INFO] wrote summary: {summary_path}")
    print(f"[DONE] converted {len(per_episode)} episodes -> {output_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy imitation_form HDF5 episodes while remapping gripper position ticks."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--src_open_tick", type=float, required=True)
    parser.add_argument("--src_close_tick", type=float, required=True)
    parser.add_argument("--dst_open_tick", type=float, required=True)
    parser.add_argument("--dst_close_tick", type=float, required=True)
    parser.add_argument("--no_clip", action="store_true", help="Do not clip mapped ticks to destination range.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    convert_directory(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        src_open_tick=args.src_open_tick,
        src_close_tick=args.src_close_tick,
        dst_open_tick=args.dst_open_tick,
        dst_close_tick=args.dst_close_tick,
        clip=not bool(args.no_clip),
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
