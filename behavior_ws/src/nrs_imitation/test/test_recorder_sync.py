import numpy as np

from nrs_imitation.recorder_sync import TimedValueBuffer, sync_error_summary


def test_linear_interpolation_uses_target_time():
    buf = TimedValueBuffer(history_sec=1.0)
    buf.add(10.0, np.array([0.0, 10.0], dtype=np.float32))
    buf.add(10.2, np.array([2.0, 20.0], dtype=np.float32))

    result = buf.sample(10.1, mode="linear")

    assert result is not None
    assert result.interpolated
    assert np.allclose(result.value, [1.0, 15.0])
    assert np.isclose(result.source_time, 10.1)
    assert np.isclose(result.error_sec, 0.1)


def test_nearest_selects_closest_value():
    buf = TimedValueBuffer(history_sec=1.0)
    buf.add(5.0, np.asarray(10, dtype=np.int32))
    buf.add(5.1, np.asarray(20, dtype=np.int32))

    result = buf.sample(5.08, mode="nearest")

    assert result is not None
    assert int(result.value) == 20
    assert np.isclose(result.error_sec, 0.02)


def test_sync_error_summary_is_in_milliseconds():
    rows = np.array([[0.0, 0.001], [0.0, 0.003]], dtype=np.float64)
    summary = sync_error_summary(rows, (("sensor", 1),))

    assert np.isclose(summary["sensor"][0], 2.0)
    assert np.isclose(summary["sensor"][2], 3.0)
