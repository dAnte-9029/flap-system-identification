"""Numerical checks for the reference-assisted diagnostic, not causal validity."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_control_reference_iv import flight_weights, iv_gain


def test_instrument_recovers_gain_with_endogenous_input():
    # Instrument and disturbance are exactly orthogonal in this constructed case.
    z = np.tile([-1., -1., 1., 1.], 20)
    disturbance = np.tile([-1., 1., -1., 1.], 20)
    u = z + disturbance
    y = -3 * u + 2 * disturbance
    beta, denominator, signal = iv_gain(u, y, z[:, None], np.repeat(['a', 'b'], 40))
    np.testing.assert_allclose(beta, -3., atol=1e-12)
    assert denominator > 0 and signal > 0
    assert abs(np.dot(u, y) / np.dot(u, u) - beta) > .5


def test_replicating_one_flight_preserves_equal_flight_estimate():
    rng = np.random.default_rng(123)
    z = rng.normal(size=(40, 3))
    u = z[:, 0] + rng.normal(size=40)
    y = 2 * u + rng.normal(size=40)
    ids = np.repeat(['a', 'b'], 20)
    original = iv_gain(u, y, z, ids)
    idx = np.r_[np.arange(40), np.arange(20), np.arange(20)]
    np.testing.assert_allclose(iv_gain(u[idx], y[idx], z[idx], ids[idx]), original)
    w = flight_weights(ids[idx])
    np.testing.assert_allclose([w[ids[idx] == g].sum() for g in ['a', 'b']], [.5, .5])


def test_absent_reference_signal_rejects_gain():
    beta, denominator, signal = iv_gain(np.arange(10.), np.arange(10.), np.zeros((10, 3)), np.repeat('a', 10))
    assert np.isnan(beta)
    assert denominator == 0 and signal == 0


def test_validation_features_need_no_future_states_and_ignore_command_tail():
    from types import SimpleNamespace
    from evaluate_control_reference_iv import features
    rng = np.random.default_rng(12)
    batch = SimpleNamespace(
        history_mask=np.ones((2, 26), dtype=bool),
        history_state_features=rng.normal(size=(2, 26, 12)),
        history_controls=rng.normal(size=(2, 26, 4)),
        trajectory=SimpleNamespace(controls=rng.normal(size=(2, 50, 4)), dt_s=np.full((2, 50), .02)),
    )
    before = features(batch)  # No truth/reference fields even exist.
    batch.trajectory.controls[:, 5:] = 1e6
    batch.trajectory.dt_s[:, 5:] = 999
    after = features(batch)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
