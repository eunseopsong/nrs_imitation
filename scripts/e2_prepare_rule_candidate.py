#!/usr/bin/env python3
"""Build a reviewable, OFFLINE-ONLY R recipe from an explicit teacher source.

This does not select a successful T episode, certify a path, or enable hardware.
Source force gate is a geometry proxy, not processing/contact ground truth.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'behavior_ws/src/nrs_imitation'))
from nrs_imitation.e2_providers import TimedActions, file_hash, rule_actions


def prepare(config_path, template_path, output):
    cfg = json.loads(config_path.read_text())
    source = TimedActions.load(template_path)
    # Only an explicitly supplied source; no first/longest-episode selection.
    if source.metadata.get('task_type') != 'line':
        raise ValueError('A line teacher template is required')
    indices = np.flatnonzero(source.action[:, 8] > cfg['executor']['contact_on_N'])
    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1)
    groups = [g for g in groups if len(g) >= 2]
    if not groups:
        raise ValueError('No contiguous source segment above existing contact threshold')
    segment = max(groups, key=lambda g: source.time[g[-1]]-source.time[g[0]])
    start, end = int(segment[0]), int(segment[-1])
    work = source.action[segment]
    height = float(np.median(work[:, 2]))
    rotation = Rotation.from_rotvec(work[:, 3:6]).mean().as_rotvec()
    ends = source.action[[start, end], :3].astype(float)
    ends[:, 2] = height
    first_work = np.r_[ends[0], rotation]
    # Five recorded poses at equally spaced source times over the approach,
    # including both boundaries; then an explicit connector to the work pose.
    source_indices = np.unique(np.searchsorted(source.time, np.linspace(0., source.time[start], 5)))
    waypoints = source.action[source_indices, :6].astype(float)
    if not np.allclose(waypoints[0], cfg['common']['demo_start_pose6'], atol=1e-4, rtol=0):
        raise ValueError('Source does not start at common demo_start; do not invent a connector')
    waypoints = np.vstack([waypoints, first_work])
    force = cfg['recipe']['force_reference_N']
    speed = float(cfg['executor']['linear_speed_mm_s'])
    ramp = 1.875*abs(force)/cfg['executor']['force_rate_N_s']
    candidate = copy.deepcopy(cfg)
    candidate['task'].update(endpoints_relative_mm=ends.tolist(), orientation_rotvec_rad=rotation.tolist())
    candidate['recipe'].update(simulation_only=True, feed_speed_mm_s=speed,
        pass_count=2, force_ramp_s=ramp, task_time_budget_s=1e6,
        approach=dict(waypoints_pose6=waypoints.tolist(), peak_speed_mm_s=speed,
            peak_angular_speed_rad_s=cfg['executor']['angular_speed_rad_s'],
            force_N=0., interpolation='quintic rest-to-rest XYZ; SO3 shortest-arc orientation',
            source_indices=source_indices.tolist(), source_template=str(template_path.resolve()),
            source_template_sha256=file_hash(template_path)),
        validation_status='offline_candidate_pending_operator_review',
        main_experiment_ready=False)
    plan = rule_actions(candidate['task'], candidate['recipe'])
    # Reviewable proposed common ceiling, rounded up to 10 s. It is not an
    # empirically validated time budget and is never silently activated.
    budget = float(math.ceil((plan.time[-1]+cfg['executor']['control_period_s'])/10.)*10.)
    candidate['recipe']['task_time_budget_s'] = budget
    candidate['common']['task_time_budget_s'] = budget
    candidate['common']['approach_retract_protocol']['approach'] = (
        'Common existing demo-start alignment; R explicit source-backed waypoint approach, '
        'T original teacher approach, C policy approach; common executor/gates/rate limits')
    provenance = dict(source_template=str(template_path.resolve()),
        source_template_sha256=file_hash(template_path),
        source_episode=source.metadata['source_episode'],
        source_episode_sha256=source.metadata['episode_sha256'],
        source_success_confirmed=source.metadata.get('selected_success', False),
        source_selection='episode_29 already previewed; proposed R geometry only, not selected as successful T',
        geometry_selection='longest contiguous source Fz > existing 3 N gate threshold; proxy only',
        source_work_indices=[start, end], source_work_time_s=source.time[[start, end]].tolist(),
        endpoints='source segment boundary XY; constant median source Z, no pixel-to-mm conversion',
        orientation='SO3 mean over the same source segment',
        approach='five actual source poses at equal source-time intervals plus first working pose',
        feed='existing common executor 10 mm/s limit, interpreted as norm peak for R',
        passes='one round trip (two strokes), initial recipe proposal',
        force_ramp='quintic ramp peak derivative equals existing 30 N/s limit',
        budget='proposed common ceiling: round total R duration up to next 10 s',
        force='E1-derived +18 N signed controller Fz target; not calibrated surface-normal force',
        status='OFFLINE CANDIDATE; path and recipe not validated on hardware')
    candidate['recipe']['geometry_recipe_source'] = provenance
    output.mkdir(parents=True, exist_ok=True)
    plan = rule_actions(candidate['task'], candidate['recipe'])
    plan.save(output/'rule_candidate.npz')
    (output/'config.R_validation_candidate.json').write_text(json.dumps(candidate, indent=2, ensure_ascii=False)+'\n')
    result = dict(provenance=provenance, endpoints_relative_mm=ends.tolist(),
        orientation_rotvec_rad=rotation.tolist(), line_length_mm=float(np.linalg.norm(ends[1]-ends[0])),
        peak_speed_mm_s=speed, force_reference_N=force, pass_count=2, force_ramp_s=ramp,
        approach_duration_s=plan.metadata['approach_duration_s'], total_duration_s=float(plan.time[-1]),
        proposed_common_budget_s=budget, hardware_enabled=False, physical_robot_used=False)
    (output/'rule_candidate_report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(source.action[:, 0], source.action[:, 1], color='0.7', label='Teacher source')
    axes[0].plot(plan.action[:, 0], plan.action[:, 1], label='Proposed R incl. approach')
    axes[0].scatter(ends[:, 0], ends[:, 1], color='red', label='Work endpoints')
    axes[0].set(xlabel='Relative X (mm)', ylabel='Relative Y (mm)', title='XY path')
    axes[0].axis('equal'); axes[0].legend(fontsize=8)
    axes[1].plot(plan.time, plan.action[:, 2]); axes[1].set(xlabel='Elapsed (s)', ylabel='Base Z (mm)', title='Requested height')
    axes[2].plot(plan.time, plan.action[:, 8]); axes[2].set(xlabel='Elapsed (s)', ylabel='Requested Fz (N)', title='Requested force before gate')
    for ax in axes:
        ax.grid(alpha=.3)
    fig.suptitle('R offline candidate: source-derived proposal; hardware validation NOT performed')
    fig.tight_layout(); fig.savefig(output/'rule_candidate_preview.png', dpi=150); plt.close(fig)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.config, args.template, args.output)
