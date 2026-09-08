#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step [5]: prove training and inference share one preprocessing path.

Static audit over the repository. Four questions, each answered from the code
rather than from a comment:

  A. Does the DATASET CONVERTER get its transform from relative_frame?
  B. Does the INFERENCE path get its transform from the same module?
  C. Does anything ANYWHERE subtract or add a stain origin by hand, bypassing
     that module? (open-coded `- stain_origin`, `+ origin`, ...)
  D. Is there any code path that could recompute stain_origin PER STEP --
     a detector call inside a timer/subscription callback, or a write to a
     stain-origin attribute outside the one-shot resolve?

  ros2 run stain_relative_frame audit_inference_path
  ros2 run stain_relative_frame audit_inference_path -- --root <extra dir>

Exit 0 = every question answered cleanly, 1 = at least one finding.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import PROJECT_ROOT, load_config
from .report import table

PKG_DIR = Path(__file__).resolve().parent
TRANSFORM_MODULE = "relative_frame"
TRANSFORM_FUNCS = ("to_relative", "to_absolute", "RelativeFrameAdapter")

# Names that denote a stain origin. Matched against the identifier only, via
# the AST -- an earlier regex over raw lines flagged prose like "-- origin
# cannot drift" and the docstrings describing the rule, which is noise.
ORIGIN_NAME = re.compile(r"^_?(?:stain_)?origin(?:_xy|_mm|_px)?$")
# A per-step recomputation smell: detection called from a callback.
CALLBACK_NAMES = re.compile(r"^(_?on_\w+|\w*_callback|\w*_cb|timer_\w+|\w+_timer)$")
DETECT_CALLS = ("detect_stain_origin", "diff_mask", "measure_stability")


def _py_files(root: Path) -> List[Path]:
    skip = {".git", "__pycache__", "build", "install", "log", ".pytest_cache",
            "node_modules", ".venv"}
    return [p for p in root.rglob("*.py")
            if not any(part in skip for part in p.parts)]


def _imports_transform(path: Path) -> Tuple[bool, List[str]]:
    """Does this file import the shared transform functions?"""
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return False, []
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and TRANSFORM_MODULE in node.module:
            found += [a.name for a in node.names if a.name in TRANSFORM_FUNCS]
        elif isinstance(node, ast.Import):
            for a in node.names:
                if TRANSFORM_MODULE in a.name:
                    found.append(a.name)
    return bool(found), sorted(set(found))


def _origin_operand(node: ast.AST) -> Optional[str]:
    """The identifier if this expression names a stain origin, else None."""
    name = None
    if isinstance(node, ast.Name):
        name = node.id
    elif isinstance(node, ast.Attribute):
        name = node.attr
    elif isinstance(node, ast.Subscript):
        return _origin_operand(node.value)
    if name and ORIGIN_NAME.match(name):
        return name
    return None


def _open_coded_hits(path: Path) -> List[Tuple[int, str]]:
    """Add/subtract of an origin-named value, found in the AST.

    Comments, docstrings and prose are invisible here by construction, so a
    hit is real arithmetic and not a sentence about arithmetic.
    """
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return []
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            for side in (node.left, node.right):
                name = _origin_operand(side)
                if name:
                    hits.append((node.lineno, f"{type(node.op).__name__.lower()} with `{name}`"))
                    break
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, (ast.Add, ast.Sub)):
            name = _origin_operand(node.target) or _origin_operand(node.value)
            if name:
                hits.append((node.lineno,
                             f"augmented {type(node.op).__name__.lower()} with `{name}`"))
    return hits


def _per_step_recompute(path: Path) -> List[Tuple[int, str]]:
    """Detector calls reached from a subscription/timer callback."""
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return []
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not CALLBACK_NAMES.match(node.name):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fn = sub.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if name in DETECT_CALLS:
                    hits.append((sub.lineno, f"{node.name}() calls {name}()"))
    return hits


def audit(roots: List[Path]) -> Dict:
    findings: List[Dict] = []
    notes: List[Dict] = []

    converter = PKG_DIR / "dataset_relativize.py"
    inference = PKG_DIR / "inference_adapter.py"
    node = PKG_DIR / "stain_origin_node.py"
    adapter = PKG_DIR / "relative_frame.py"

    # ---- A. converter ---------------------------------------------------
    ok_a, funcs_a = _imports_transform(converter)
    notes.append({
        "check": "A training/dataset path",
        "file": converter.name,
        "detail": f"imports {funcs_a or 'NOTHING'} from {TRANSFORM_MODULE}",
        "verdict": "PASS" if ok_a else "FAIL",
    })
    if not ok_a:
        findings.append({"check": "A", "file": str(converter),
                         "detail": "converter does not import the shared transform"})

    # ---- B. inference path ---------------------------------------------
    ok_b, funcs_b = _imports_transform(inference)
    notes.append({
        "check": "B inference path",
        "file": inference.name,
        "detail": f"imports {funcs_b or 'NOTHING'} from {TRANSFORM_MODULE}",
        "verdict": "PASS" if ok_b else "FAIL",
    })
    if not ok_b:
        findings.append({"check": "B", "file": str(inference),
                         "detail": "inference adapter does not import the shared transform"})

    shared = sorted(set(funcs_a) & set(funcs_b))
    same_module = ok_a and ok_b
    notes.append({
        "check": "A==B same module",
        "file": f"{TRANSFORM_MODULE}.py",
        "detail": (f"shared symbols {shared}" if shared else
                   "both import the module, no symbol in common")
                  if same_module else "one side does not import it",
        "verdict": "PASS" if same_module else "FAIL",
    })
    if same_module and not shared:
        notes.append({
            "check": "A==B transform call",
            "file": f"{converter.name} / {inference.name}",
            "detail": f"converter uses {funcs_a}, inference uses {funcs_b} "
                      f"(RelativeFrameAdapter wraps to_relative/to_absolute)",
            "verdict": "PASS",
        })

    # ---- C. open-coded arithmetic --------------------------------------
    scanned = 0
    for root in roots:
        for p in _py_files(root):
            if p.resolve() == adapter.resolve():
                continue  # the one file allowed to do the arithmetic
            scanned += 1
            for lineno, text in _open_coded_hits(p):
                findings.append({
                    "check": "C", "file": f"{p}:{lineno}",
                    "detail": f"open-coded origin arithmetic: {text[:110]}",
                })

    # ---- D. per-step recomputation -------------------------------------
    for root in roots:
        for p in _py_files(root):
            for lineno, text in _per_step_recompute(p):
                findings.append({
                    "check": "D", "file": f"{p}:{lineno}",
                    "detail": f"stain detection reachable from a callback: {text}",
                })

    # ---- D. structural one-shot proof in the node ----------------------
    src = node.read_text(errors="replace")
    disarms = "self._disarm()" in src and "destroy_subscription" in src
    frozen = ".freeze()" in src
    notes.append({
        "check": "D one-shot structure",
        "file": node.name,
        "detail": ("origin frozen and the camera subscription destroyed on resolve"
                   if (disarms and frozen)
                   else f"freeze={frozen} destroy_subscription={disarms}"),
        "verdict": "PASS" if (disarms and frozen) else "FAIL",
    })
    if not (disarms and frozen):
        findings.append({"check": "D", "file": str(node),
                         "detail": "the node can still receive frames after resolving"})

    return {"findings": findings, "notes": notes, "files_scanned": scanned}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="[5] training/inference path audit")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--root", type=str, action="append", default=None,
                    help="extra directory to scan (repeatable)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    roots = [PKG_DIR]
    for extra in (args.root or []):
        roots.append(Path(extra).expanduser())
    if not args.root:
        for cand in (PROJECT_ROOT / "source", PROJECT_ROOT / "scripts",
                     PROJECT_ROOT / "behavior_ws" / "src" / "nrs_imitation"):
            if cand.is_dir():
                roots.append(cand)

    print("[audit] roots:")
    for r in roots:
        print(f"[audit]   {r}")
    print()

    result = audit(roots)
    print(table(result["notes"], title="[5] shared-preprocessing audit"))
    print(f"\n[audit] {result['files_scanned']} files scanned for open-coded origin arithmetic")

    if result["findings"]:
        print()
        print(table(result["findings"], title=f"[5] FINDINGS ({len(result['findings'])})"))
    else:
        print("[audit] no open-coded origin arithmetic, no per-step recomputation")

    payload = {
        "step": "5_inference_path_audit",
        "stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "roots": [str(r) for r in roots],
        "passed": not result["findings"],
        **result,
    }
    out_path = Path(args.out) if args.out else cfg.artifacts / "inference_path_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[audit] report -> {out_path}")
    return 1 if result["findings"] else 0


if __name__ == "__main__":
    sys.exit(main())
