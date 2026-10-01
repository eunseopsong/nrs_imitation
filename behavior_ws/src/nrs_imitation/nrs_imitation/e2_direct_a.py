"""Config-free A runtime settings from a checkpoint and ROS parameters.

No checkpoint copies, experiment JSON reads, or hardware I/O. Control constants
match the archived B/C timed executor. Protective limits remain explicit.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .e2_direct_abc import (ROOT, CHECKPOINTS, PARAMETERS, object_hash,
                            build_runtime as build_common_runtime, from_node,
                            launch_arguments, launch_values, node_parameters)

SCHEMA = 'E2_direct_A_v1'
DEFAULT_CHECKPOINT = CHECKPOINTS['A']


def is_direct_a(config):
    return config.get('schema') in (SCHEMA, 'E2_direct_ABC_v1') and config.get('condition') == 'A'


def build_runtime(parameters, require_protection=True):
    """A-only compatibility entry; matched experiments use e2_abc_common.launch.py."""
    p = dict(PARAMETERS, **parameters)
    return build_common_runtime(p, require_protection=require_protection,
                                legacy_a=not p['e2_direct_abc'])


def main():
    """Actual A weights + held-out frame, without creating a ROS context."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config = build_runtime(dict(checkpoint=args.checkpoint), require_protection=False)
    args.output.mkdir(parents=True, exist_ok=False)
    from .e2_offline import registered_A_sample
    action, _, model = registered_A_sample(config, args.output)
    np.save(args.output/'policy_motion6.npy', action[:, :6])
    report = dict(status='offline_pass', hardware_io=False, model=model,
                  runtime_sha256=object_hash(config), output=str(args.output))
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
