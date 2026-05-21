#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import textwrap
from typing import Iterable, Sequence, Tuple


TERMINAL_WIDTH = int(os.environ.get("NRS_PRINT_WIDTH", "88"))
VALUE_WIDTH = max(36, TERMINAL_WIDTH - 18)


def rule(title: str = "", char: str = "=") -> str:
    title = str(title).strip()
    if not title:
        return char * TERMINAL_WIDTH
    label = f" {title} "
    side = max(2, (TERMINAL_WIDTH - len(label)) // 2)
    line = (char * side) + label
    return line + (char * max(0, TERMINAL_WIDTH - len(line)))


def kv_lines(items: Sequence[Tuple[str, object]], indent: int = 2) -> str:
    out = []
    pad = " " * indent
    for key, value in items:
        prefix = f"{pad}{key:<14}: "
        wrapped = textwrap.wrap(
            str(value),
            width=VALUE_WIDTH,
            subsequent_indent=" " * len(prefix),
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        out.append(prefix + wrapped[0])
        out.extend(wrapped[1:])
    return "\n".join(out)


def block(title: str, items: Sequence[Tuple[str, object]], char: str = "=") -> str:
    return "\n".join([rule(title, char), kv_lines(items), rule(char=char)])


def status(tag: str, items: Sequence[Tuple[str, object]]) -> str:
    head = f"[{tag}]"
    parts = [f"{k}={v}" for k, v in items]
    lines = textwrap.wrap(
        " ".join(parts),
        width=max(20, TERMINAL_WIDTH - len(head) - 1),
        subsequent_indent=" " * (len(head) + 1),
        break_long_words=False,
        break_on_hyphens=False,
    )
    return head if not lines else f"{head} {lines[0]}" + ("\n" + "\n".join(lines[1:]) if len(lines) > 1 else "")


def bullet_lines(title: str, lines: Iterable[object]) -> str:
    body = "\n".join(f"  - {line}" for line in lines)
    return f"{rule(title)}\n{body}\n{rule()}"
