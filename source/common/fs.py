#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ACT_ROOT = PROJECT_ROOT / "datasets" / "ACT"
DATASETS_POLISHING_ROOT = PROJECT_ROOT / "datasets" / "polishing"
DATASETS_GRIPPER_ROOT = PROJECT_ROOT / "datasets" / "gripper"
CHECKPOINTS_ROOT = PROJECT_ROOT / "checkpoints"
ANALYSIS_LOGS_ROOT = PROJECT_ROOT / "analysis_logs"


def find_latest_timestamped_subdir(root_dir: str) -> Optional[str]:
    if not os.path.isdir(root_dir):
        return None

    candidates = []
    for name in os.listdir(root_dir):
        sub = os.path.join(root_dir, name)
        if not os.path.isdir(sub):
            continue

        ok = False
        for fmt in ("%Y%m%d_%H%M", "%m%d_%H%M"):
            try:
                datetime.strptime(name, fmt)
                ok = True
                break
            except ValueError:
                pass

        if ok:
            candidates.append((name, sub))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]
