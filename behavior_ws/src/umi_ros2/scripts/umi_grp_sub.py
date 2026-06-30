#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from umi_ros2.umi_grp_sub import main


if __name__ == "__main__":
    main()
