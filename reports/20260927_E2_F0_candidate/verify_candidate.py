"""Independent numerical checks and CPU-only command-channel checks; no ROS init."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path('/home/eunseop/nrs_imitation')
OUT = Path(__file__).resolve().parent
EXP = ROOT / 'experiments/e2_force_ablation_20260926'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    candidate = json.loads((OUT / 'candidate.json').read_text())
    phases = json.loads((OUT / 'pose_phase_proposals.json').read_text())
    split = json.loads((EXP / 'teacher_preview/split.json').read_text())
    cfg = json.loads((EXP / 'config.json').read_text())
    config_sha = sha(EXP / 'config.json')
    assert config_sha == candidate['runtime_config_sha256']
    names = [e['episode'] for e in phases['episodes']]
    assert len(names) == len(set(names)) == 38
    assert set(names) == set(split['train'])
    assert not set(names).intersection(split['validation'])
    assert candidate['normal_force_N'] is None and candidate['command_fz_N'] is None
    assert candidate['validated_operating_range_N'] is None
    assert not cfg['calibration']['normal_force_verified']
    assert cfg['f0']['status'] != 'frozen' and cfg['f0']['command_fz_N'] is None

    # Independent formulation: clip every original sample's held-time segment
    # against the interval, rather than constructing interval edges/searchsorted.
    means, second_moments, low_count, nonpositive_count = [], [], 0, 0
    for e in phases['episodes']:
        with h5py.File(e['source_h5']) as f:
            g = f['episodes/' + e['source_group']]
            t = np.asarray(g['sample_time_unix'], dtype=np.float64)
            fz = np.asarray(g['ft'], dtype=np.float64)[:, 2]
        t -= t[0]
        a, b = e['start_s'], e['end_s']
        weights = np.maximum(0., np.minimum(t[1:], b) - np.maximum(t[:-1], a))
        np.testing.assert_allclose(weights.sum(), b - a, atol=1e-12, rtol=0)
        assert np.isfinite(t).all() and np.isfinite(fz).all() and (np.diff(t) > 0).all()
        assert np.max(np.diff(t)[weights > 0]) <= .2
        means.append(float(np.dot(weights, fz[:-1]) / (b - a)))
        second_moments.append(float(np.dot(weights, fz[:-1] ** 2) / (b - a)))
        low_count += int(np.sum((weights > 0) & (fz[:-1] < 3.)))
        nonpositive_count += int(np.sum((weights > 0) & (fz[:-1] <= 0.)))
    independent = float(np.mean(means))
    expected = candidate['teacher_Fz_constant_candidate_N']
    np.testing.assert_allclose(independent, expected, atol=1e-10, rtol=0)

    def objective(c):
        return float(np.mean(second_moments) - 2 * c * independent + c * c)

    for delta in (-1., 1.):
        np.testing.assert_allclose(objective(independent + delta) - objective(independent),
                                   1., atol=1e-10, rtol=0)
    assert objective(23.) < objective(22.) and objective(23.) < objective(24.)

    # Bind the actual repository methods to an in-memory fake. This is a
    # numerical channel identity check, not robot/teacher physical calibration.
    sys.path[:0] = [str(ROOT / 'source'), str(ROOT / 'behavior_ws/src/nrs_imitation'),
                   str(ROOT / 'behavior_ws/src/nrs_imitation/test')]
    import torch
    from nrs_imitation import inference_core as core
    from test_execution_metrics import FakeNode
    channel_checks = {}
    for condition in 'BC':
        node = FakeNode(OUT / 'unused_metrics', force_on=(condition == 'C'))
        assert node._metrics is None
        node.fz_hard_limit = cfg['executor']['fz_hard_limit_N']
        assert node.fz_hard_limit == 0.
        native = np.tile(np.r_[np.zeros(6), 2., -4., 23.].astype(np.float32), (3, 1))
        native[:, 8] = [-3., 23., 31.]
        stats = node.stats
        aa, ab = np.asarray(stats.act_a), np.asarray(stats.act_b)
        if stats.act_mode in ('minmax_01', 'minmax_m11'):
            normalized = (native - aa) / np.maximum(ab - aa, 1e-6)
            if stats.act_mode == 'minmax_m11':
                normalized = 2 * normalized - 1
        else:
            normalized = (native - aa) / np.maximum(ab, 1e-6)
        denormalized = core._denorm_action_seq(torch.as_tensor(normalized, dtype=torch.float32), stats).numpy()
        processed = node._postprocess_provider_action(denormalized.copy())
        np.testing.assert_allclose(processed[:, 8], native[:, 8], atol=1e-5, rtol=0)
        np.testing.assert_array_equal(processed[:, 6:8], np.zeros((3, 2)))
        assert not node.sent_messages
        channel_checks[condition] = dict(input_Fz_N=native[:, 8].tolist(),
            output_Fz_before_executor_N=processed[:, 8].tolist(), Fx_Fy_zero=True,
            actual_denormalization_and_postprocessor=True, robot_messages_sent=0)
    assert sha(EXP / 'config.json') == config_sha
    result = dict(checked_at=datetime.now().astimezone().isoformat(), status='pass',
        independent_clipped_sample_mean_N=independent, training_episodes=38,
        held_out_episodes_used=False, objective='Equal-episode time-normalized squared error',
        objective_at_optimum_N2=objective(independent),
        objective_at_23N_N2=objective(23.),
        objective_delta_for_plus_or_minus_1N_N2=1.,
        retained_samples_below_3N=low_count, retained_samples_at_or_below_zero_N=nonpositive_count,
        actual_software_channel_checks=channel_checks,
        physical_calibration_verified=False, validated_operating_range=None,
        runtime_F0_still_unset=True, runtime_config_sha256=config_sha,
        candidate_sha256=sha(OUT / 'candidate.json'), hardware_executed=False)
    (OUT / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
