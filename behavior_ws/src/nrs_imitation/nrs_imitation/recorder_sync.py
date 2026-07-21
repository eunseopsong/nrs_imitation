#!/usr/bin/env python3
"""
Small timestamp-buffer utilities shared by recorder nodes.

ROS messages used by this project (Float64MultiArray, Wrench, Int32 and
Float32) do not carry a Header.  Consequently all recorder streams use one
clock domain: the UNIX receive time measured at the subscriber callback.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SyncedValue:
    """One value aligned to a requested master timestamp."""

    value: np.ndarray
    source_time: float
    error_sec: float
    interpolated: bool


class TimedValueBuffer:
    """Bounded, monotonically ordered buffer with nearest/linear lookup."""

    def __init__(self, history_sec: float = 1.0, maxlen: int = 4096):
        """Initialize a time-ordered buffer with bounded history."""
        self.history_sec = max(0.05, float(history_sec))
        self._items: Deque[Tuple[float, np.ndarray]] = deque(maxlen=max(2, int(maxlen)))

    def clear(self) -> None:
        """Discard all buffered samples."""
        self._items.clear()

    def add(self, timestamp: float, value) -> None:
        """Append one value and its receive timestamp."""
        t = float(timestamp)
        v = np.asarray(value).copy()
        if self._items and t < self._items[-1][0]:
            # time.time() should be monotonic enough on one host, but keep the
            # lookup well-defined if the system clock is adjusted backwards.
            self._items.clear()
        self._items.append((t, v))
        cutoff = t - self.history_sec
        while len(self._items) > 2 and self._items[1][0] < cutoff:
            self._items.popleft()

    def sample(self, target_time: float, mode: str = "nearest") -> Optional[SyncedValue]:
        """Return the nearest value or a linear interpolation at target time."""
        if not self._items:
            return None

        target = float(target_time)
        items = self._items
        right = 0
        while right < len(items) and items[right][0] < target:
            right += 1

        if right == 0:
            t, v = items[0]
            return SyncedValue(v.copy(), float(t), abs(float(t) - target), False)
        if right >= len(items):
            t, v = items[-1]
            return SyncedValue(v.copy(), float(t), abs(target - float(t)), False)

        t0, v0 = items[right - 1]
        t1, v1 = items[right]
        if str(mode).lower() == "linear" and t1 > t0:
            ratio = (target - t0) / (t1 - t0)
            value = np.asarray(v0, dtype=np.float64) + ratio * (
                np.asarray(v1, dtype=np.float64) - np.asarray(v0, dtype=np.float64)
            )
            # Conservative diagnostic: the farther of the two samples needed
            # for interpolation, rather than reporting an artificial zero.
            error = max(target - t0, t1 - target)
            return SyncedValue(value.astype(np.asarray(v0).dtype), target, float(error), True)

        if (target - t0) <= (t1 - target):
            t, v = t0, v0
        else:
            t, v = t1, v1
        return SyncedValue(v.copy(), float(t), abs(float(t) - target), False)


def sync_error_summary(sync_rows: np.ndarray, error_columns) -> dict:
    """Return compact millisecond percentiles for recorder logging."""
    rows = np.asarray(sync_rows, dtype=np.float64)
    out = {}
    if rows.ndim != 2 or rows.shape[0] == 0:
        return out
    for name, column in error_columns:
        values = rows[:, int(column)] * 1000.0
        values = values[np.isfinite(values)]
        if values.size:
            out[str(name)] = (
                float(np.percentile(values, 50)),
                float(np.percentile(values, 95)),
                float(np.max(values)),
            )
    return out
