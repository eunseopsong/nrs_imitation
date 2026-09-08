#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plain-text table rendering for the step reports."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence


def table(rows: Sequence[Dict], columns: Optional[Sequence[str]] = None,
          title: str = "", max_rows: int = 0) -> str:
    rows = list(rows)
    if not rows:
        return f"{title}\n  (no rows)" if title else "  (no rows)"
    cols = list(columns) if columns else list(rows[0].keys())

    def cell(r, c):
        v = r.get(c, "")
        return "-" if v is None else str(v)

    shown = rows[:max_rows] if max_rows and len(rows) > max_rows else rows
    width = {c: max(len(c), *(len(cell(r, c)) for r in shown)) for c in cols}
    line = "  ".join(c.ljust(width[c]) for c in cols)
    sep = "  ".join("-" * width[c] for c in cols)
    body = [ "  ".join(cell(r, c).ljust(width[c]) for c in cols) for r in shown ]
    out = [line, sep] + body
    if max_rows and len(rows) > max_rows:
        out.append(f"... {len(rows) - max_rows} more rows")
    return (f"{title}\n" if title else "") + "\n".join(out)


def verdict_line(name: str, passed: bool, detail: str = "") -> str:
    tag = "PASS" if passed else "FAIL"
    return f"[{tag}] {name}" + (f" -- {detail}" if detail else "")
