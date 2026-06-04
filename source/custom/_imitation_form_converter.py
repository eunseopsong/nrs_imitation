#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared converter for hdf5_recorder_* merged HDF5 files.

Output episode layout is intentionally compact:

  episode_0.hdf5
  ├── action/
  │   ├── position
  │   └── force
  └── observations/
      ├── position
      ├── force
      └── images/
          ├── cam0
          └── cam1  # dual-camera only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = PROJECT_ROOT / "datasets"


def _hdf5_files_under(path: Path, recursive: bool = False) -> List[Path]:
    patterns = ["**/*.hdf5", "**/*.h5"] if recursive else ["*.hdf5", "*.h5"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(path.glob(pattern))
    return sorted({p.resolve() for p in files if p.is_file()})


def _newest_file(files: Iterable[Path]) -> Path:
    candidates = list(files)
    if not candidates:
        raise FileNotFoundError("No HDF5 files found.")
    candidates.sort(key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
    return candidates[0]


def _has_completed_episodes(path: Path) -> bool:
    try:
        with h5py.File(str(path), "r") as f:
            if "episodes" not in f:
                return False
            return any(not str(name).endswith("__writing") for name in f["episodes"].keys())
    except Exception:
        return False


def _newest_usable_file(files: Iterable[Path]) -> Path:
    candidates = list(files)
    usable = [p for p in candidates if _has_completed_episodes(p)]
    if usable:
        return _newest_file(usable)
    return _newest_file(candidates)


def resolve_input_h5(input_h5: str, default_root: Path) -> Path:
    raw = str(input_h5 or "").strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.is_file():
            return p
        if p.is_dir():
            merged_dir = p / "merged_hdf5"
            if merged_dir.is_dir():
                files = _hdf5_files_under(merged_dir, recursive=False)
                if files:
                    return _newest_usable_file(files)
            files = _hdf5_files_under(p, recursive=False)
            if files:
                return _newest_usable_file(files)
            files = _hdf5_files_under(p, recursive=True)
            if files:
                return _newest_usable_file(files)
            raise FileNotFoundError(f"No .hdf5/.h5 file found under input directory: {p}")
        raise FileNotFoundError(f"input_h5 does not exist: {p}")

    root = default_root.expanduser().resolve()
    files = []
    for pattern in ("*/merged_hdf5/*.hdf5", "*/merged_hdf5/*.h5"):
        files.extend(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No merged HDF5 found under {root}/*/merged_hdf5/")
    usable = [p for p in files if _has_completed_episodes(p)]
    candidates = usable if usable else files
    candidates.sort(key=lambda p: (p.parent.parent.name, p.stat().st_mtime, str(p)), reverse=True)
    return candidates[0].resolve()


def infer_output_dir(input_h5: Path, output_dir: str, output_name: str = "imitation_form") -> Path:
    raw = str(output_dir or "").strip()
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p
    if input_h5.parent.name == "merged_hdf5":
        return input_h5.parent.parent / output_name
    return input_h5.parent / output_name


def _read_optional_array(g: h5py.Group, paths: Sequence[str], dtype=None) -> Tuple[np.ndarray | None, str]:
    for path in paths:
        try:
            if path in g:
                arr = np.asarray(g[path])
                if dtype is not None:
                    arr = arr.astype(dtype)
                return arr, path
        except Exception:
            pass
    return None, ""


def _ensure_2d_min_dim(arr: np.ndarray, min_dim: int, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1 and arr.size >= min_dim:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < min_dim:
        raise ValueError(f"{name} must be (T,{min_dim}+) but got {arr.shape}")
    return arr[:, :min_dim]


def _ensure_image4(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must be (T,H,W,3) but got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _trim_to_min_len(items: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], int]:
    lengths = [int(v.shape[0]) for v in items.values()]
    if not lengths:
        raise ValueError("No arrays to trim.")
    T = min(lengths)
    return {k: v[:T] for k, v in items.items()}, T


def load_episode(g_ep: h5py.Group, camera_names: Sequence[str]) -> Dict[str, np.ndarray]:
    position, _ = _read_optional_array(
        g_ep,
        ["position", "observations/position"],
        dtype=np.float32,
    )
    if position is None:
        raise KeyError(f"Missing position in {g_ep.name}")

    force, _ = _read_optional_array(
        g_ep,
        ["ft", "force", "observations/force"],
        dtype=np.float32,
    )
    if force is None:
        raise KeyError(f"Missing force/ft in {g_ep.name}")

    data: Dict[str, np.ndarray] = {
        "position": _ensure_2d_min_dim(position, 6, "position").astype(np.float32),
        "force": _ensure_2d_min_dim(force, 3, "force").astype(np.float32),
    }

    for cam in camera_names:
        aliases = [
            f"images/{cam}",
            f"observations/images/{cam}",
        ]
        if cam == "cam0":
            aliases.extend(["image", "images/rgb", "observations/image"])
        if cam == "cam1":
            aliases.extend(["images/global", "observations/images/global"])

        image, _ = _read_optional_array(g_ep, aliases, dtype=None)
        if image is None:
            raise KeyError(f"Missing {cam} image in {g_ep.name}")
        data[cam] = _ensure_image4(image, cam)

    data, _ = _trim_to_min_len(data)
    return data


def truncate_episode(data: Dict[str, np.ndarray], max_len: int) -> Tuple[Dict[str, np.ndarray], bool, int]:
    T = int(data["position"].shape[0])
    if int(max_len) <= 0 or T <= int(max_len):
        return data, False, T
    return {k: v[: int(max_len)] for k, v in data.items()}, True, int(max_len)


def _compression_kwargs(mode: str, gzip_level: int) -> Dict:
    mode = str(mode).lower()
    if mode == "gzip":
        return {"compression": "gzip", "compression_opts": int(gzip_level), "shuffle": True}
    if mode == "lzf":
        return {"compression": "lzf", "shuffle": True}
    return {}


def write_episode(
    out_path: Path,
    data: Dict[str, np.ndarray],
    camera_names: Sequence[str],
    source_h5: Path,
    source_episode: str,
    compression: str,
    gzip_level: int,
    orig_len: int,
    truncated: bool,
) -> None:
    if out_path.exists():
        out_path.unlink()

    kwargs = _compression_kwargs(compression, gzip_level)
    with h5py.File(str(out_path), "w") as f:
        f.attrs["schema_version"] = "imitation_form_compact_v1"
        f.attrs["source_h5"] = str(source_h5)
        f.attrs["source_episode"] = str(source_episode)
        f.attrs["camera_names_json"] = json.dumps(list(camera_names))
        f.attrs["orig_len"] = int(orig_len)
        f.attrs["truncated"] = int(bool(truncated))

        g_action = f.create_group("action")
        g_action.create_dataset("position", data=data["position"].astype(np.float32), **kwargs)
        g_action.create_dataset("force", data=data["force"].astype(np.float32), **kwargs)

        g_obs = f.create_group("observations")
        g_obs.create_dataset("position", data=data["position"].astype(np.float32), **kwargs)
        g_obs.create_dataset("force", data=data["force"].astype(np.float32), **kwargs)

        g_images = g_obs.create_group("images")
        for cam in camera_names:
            g_images.create_dataset(cam, data=data[cam].astype(np.uint8), **kwargs)


def convert_merged_h5(
    input_h5: Path,
    output_dir: Path,
    camera_names: Sequence[str],
    min_len: int,
    max_len: int,
    compression: str,
    gzip_level: int,
    overwrite: bool,
    write_summary: bool,
) -> List[Path]:
    camera_names = [str(c).strip() for c in camera_names if str(c).strip()]
    if not camera_names:
        raise ValueError("camera_names must not be empty.")

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif list(output_dir.glob("episode_*.hdf5")):
            raise RuntimeError(f"Output dir already contains episode files: {output_dir}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    failed: List[Tuple[str, str]] = []

    with h5py.File(str(input_h5), "r") as f:
        if "episodes" not in f:
            raise KeyError(f"{input_h5} does not contain /episodes group")

        ep_names = sorted(name for name in f["episodes"].keys() if not str(name).endswith("__writing"))
        if not ep_names:
            raise RuntimeError(f"No episodes found under {input_h5}/episodes")

        print(f"[INFO] input_h5       = {input_h5}")
        print(f"[INFO] output_dir     = {output_dir}")
        print(f"[INFO] camera_names   = {camera_names}")
        print(f"[INFO] episodes found = {len(ep_names)}")

        out_idx = 0
        for ep_name in ep_names:
            try:
                data = load_episode(f["episodes"][ep_name], camera_names)
                orig_len = int(data["position"].shape[0])
                if orig_len < int(min_len):
                    raise RuntimeError(f"too short: T={orig_len} < min_len={min_len}")

                data, truncated, T_out = truncate_episode(data, max_len=max_len)
                out_path = output_dir / f"episode_{out_idx}.hdf5"
                write_episode(
                    out_path=out_path,
                    data=data,
                    camera_names=camera_names,
                    source_h5=input_h5,
                    source_episode=ep_name,
                    compression=compression,
                    gzip_level=gzip_level,
                    orig_len=orig_len,
                    truncated=truncated,
                )

                image_shapes = ", ".join(f"{cam}={data[cam].shape}" for cam in camera_names)
                print(
                    f"[OK] {ep_name} -> episode_{out_idx}.hdf5 | "
                    f"T={T_out}, position={data['position'].shape}, force={data['force'].shape}, {image_shapes}"
                )
                written.append(out_path)
                out_idx += 1

            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                failed.append((ep_name, msg))
                print(f"[FAIL] {ep_name}: {msg}")

    if write_summary:
        summary_path = output_dir / "conversion_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "input_h5": str(input_h5),
                    "output_dir": str(output_dir),
                    "camera_names": list(camera_names),
                    "num_written": len(written),
                    "written": [str(p) for p in written],
                    "num_failed": len(failed),
                    "failed": [{"episode": ep, "error": err} for ep, err in failed],
                    "schema_version": "imitation_form_compact_v1",
                },
                fp,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[INFO] wrote summary: {summary_path}")

    print(f"[DONE] converted episodes: {len(written)} / {len(written) + len(failed)}")
    if not written:
        raise RuntimeError("No episodes were converted successfully.")
    return written


def build_parser(description: str, default_root: Path, camera_names: Sequence[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input_h5",
        "--input",
        type=str,
        default="",
        help="Merged HDF5 file, merged_hdf5 directory, or run directory. If omitted, latest is selected.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(default_root),
        help="Dataset root used for auto-latest search.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory. If omitted, use <run_dir>/imitation_form.",
    )
    parser.add_argument("--min_len", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=0, help="0 means no truncation.")
    parser.add_argument("--compression", type=str, default="gzip", choices=["gzip", "lzf", "none"])
    parser.add_argument("--gzip_level", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="Replace output_dir if it contains old episodes.")
    parser.add_argument("--write_summary", action="store_true", help="Write conversion_summary.json into output_dir.")
    parser.set_defaults(camera_names=list(camera_names))
    return parser


def run_cli(description: str, default_root: Path, camera_names: Sequence[str]) -> None:
    parser = build_parser(description, default_root, camera_names)
    args = parser.parse_args()

    input_h5 = resolve_input_h5(args.input_h5, Path(args.dataset_root))
    output_dir = infer_output_dir(input_h5, args.output_dir)
    convert_merged_h5(
        input_h5=input_h5,
        output_dir=output_dir,
        camera_names=args.camera_names,
        min_len=int(args.min_len),
        max_len=int(args.max_len),
        compression=str(args.compression).lower(),
        gzip_level=int(args.gzip_level),
        overwrite=bool(args.overwrite),
        write_summary=bool(args.write_summary),
    )
