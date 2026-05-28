from __future__ import annotations

import numpy as np
import pandas as pd

from system_identification.signal_preprocessing import (
    apply_groupwise_time_shift,
    finite_difference_quality_metrics,
    groupwise_cubic_spline_derivative,
    groupwise_lowpass_filter,
    groupwise_savgol_derivative,
)


def _two_log_frame() -> pd.DataFrame:
    time_s = np.tile(np.arange(60, dtype=float) * 0.01, 2)
    log_id = np.repeat(["log_a", "log_b"], 60)
    value = time_s**2
    return pd.DataFrame({"log_id": log_id, "time_s": time_s, "value": value})


def test_groupwise_savgol_derivative_recovers_quadratic_without_crossing_logs():
    frame = _two_log_frame()

    derivative = groupwise_savgol_derivative(
        frame,
        "value",
        window_s=0.11,
        polyorder=2,
        group_columns=["log_id"],
    )

    expected = 2.0 * frame["time_s"].to_numpy()
    np.testing.assert_allclose(derivative.to_numpy()[5:55], expected[5:55], atol=1e-8)
    np.testing.assert_allclose(derivative.to_numpy()[65:115], expected[65:115], atol=1e-8)


def test_groupwise_cubic_spline_derivative_recovers_sinusoid():
    time_s = np.arange(100, dtype=float) * 0.01
    frame = pd.DataFrame({"log_id": "log_a", "time_s": time_s, "value": np.sin(2.0 * np.pi * time_s)})

    derivative = groupwise_cubic_spline_derivative(frame, "value", group_columns=["log_id"])

    expected = 2.0 * np.pi * np.cos(2.0 * np.pi * time_s)
    np.testing.assert_allclose(derivative.to_numpy()[5:-5], expected[5:-5], atol=2e-3)


def test_groupwise_lowpass_filter_reduces_high_frequency_noise():
    time_s = np.arange(500, dtype=float) * 0.01
    clean = np.sin(2.0 * np.pi * 1.0 * time_s)
    noisy = clean + 0.4 * np.sin(2.0 * np.pi * 30.0 * time_s)
    frame = pd.DataFrame({"log_id": "log_a", "time_s": time_s, "value": noisy})

    filtered = groupwise_lowpass_filter(frame, "value", cutoff_hz=5.0, order=2, group_columns=["log_id"])

    raw_metrics = finite_difference_quality_metrics(clean, noisy, sample_rate_hz=100.0, highpass_cutoff_hz=8.0)
    filtered_metrics = finite_difference_quality_metrics(clean, filtered, sample_rate_hz=100.0, highpass_cutoff_hz=8.0)
    assert filtered_metrics["rmse"] < raw_metrics["rmse"]


def test_groupwise_time_shift_does_not_wrap_across_logs():
    frame = _two_log_frame()

    shifted = apply_groupwise_time_shift(frame, "value", lag_s=0.02, group_columns=["log_id"])

    assert np.isnan(shifted.iloc[0])
    assert np.isnan(shifted.iloc[60])
    assert np.isfinite(shifted.iloc[5])
    assert np.isfinite(shifted.iloc[65])
    assert shifted.iloc[60] != shifted.iloc[59]
