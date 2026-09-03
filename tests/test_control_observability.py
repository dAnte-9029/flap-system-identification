from __future__ import annotations

import numpy as np

from system_identification.analysis.control_observability import (
    control_summary_features,
    distribution_shift_summary,
    fit_standardized_ridge,
    paired_log_bootstrap,
    quaternion_relative_rotation_vector,
)


def test_standardized_ridge_uses_train_statistics_and_recovers_mapping() -> None:
    rng = np.random.default_rng(3)
    train_x = rng.normal(size=(200, 3))
    weights = np.array([[1.0, -2.0], [0.5, 0.25], [-0.2, 0.1]])
    train_y = train_x @ weights + np.array([2.0, -1.0])
    validation_x = rng.normal(loc=2.0, size=(30, 3))

    prediction, fit = fit_standardized_ridge(
        train_x, train_y, validation_x, alpha=1.0e-9
    )

    np.testing.assert_allclose(prediction, validation_x @ weights + [2.0, -1.0], atol=1e-7)
    np.testing.assert_allclose(fit["feature_mean"], np.mean(train_x, axis=0))


def test_control_summary_features_are_horizon_bounded() -> None:
    controls = np.array([[[0.0], [1.0], [3.0], [7.0]]])
    dt_s = np.full((1, 4), 0.1)

    summary, names = control_summary_features(
        controls, dt_s, steps=3, channel_names=("motor",)
    )

    assert names == (
        "motor_first",
        "motor_last",
        "motor_mean",
        "motor_std",
        "motor_total_variation",
        "motor_lpf_tau_0p05",
        "motor_lpf_tau_0p15",
        "motor_lpf_tau_0p40",
    )
    np.testing.assert_allclose(summary[0, :5], [0.0, 3.0, 4.0 / 3.0, np.std([0, 1, 3]), 3.0])


def test_relative_rotation_vector_is_sign_safe() -> None:
    half = np.deg2rad(45.0)
    q0 = np.array([[1.0, 0.0, 0.0, 0.0]])
    q1 = np.array([[np.cos(half), 0.0, 0.0, np.sin(half)]])

    first = quaternion_relative_rotation_vector(q0, q1)
    second = quaternion_relative_rotation_vector(-q0, q1)

    np.testing.assert_allclose(first, [[0.0, 0.0, np.pi / 2.0]], atol=1e-10)
    np.testing.assert_allclose(first, second, atol=1e-10)


def test_paired_log_bootstrap_resamples_logs_not_rows() -> None:
    reference = {"a": 2.0, "b": 4.0, "c": 6.0}
    candidate = {"a": 1.0, "b": 3.0, "c": 5.0}

    result = paired_log_bootstrap(reference, candidate, seed=5, draws=500)

    assert result["log_count"] == 3
    assert result["logs_improved"] == 3
    assert result["mean_gain"] == 1.0
    assert result["ci_low"] == 1.0
    assert result["ci_high"] == 1.0


def test_distribution_shift_summary_reports_out_of_train_support() -> None:
    train = np.arange(100.0)
    validation = np.array([-10.0, 50.0, 200.0])

    result = distribution_shift_summary(train, validation)

    assert result["validation_outside_train_p01_p99_fraction"] == 2.0 / 3.0
    assert result["standardized_mean_shift"] > 0.0
    assert result["normalized_wasserstein_distance"] > 0.0
