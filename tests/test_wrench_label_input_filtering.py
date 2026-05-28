from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.build_time_aligned_smoothed_label_split import apply_input_filters
from system_identification.signal_preprocessing import finite_difference_quality_metrics


def test_apply_input_filters_preserves_raw_and_adds_filtered_columns():
    time_s = np.arange(500, dtype=float) * 0.01
    low = np.sin(2.0 * np.pi * 1.0 * time_s)
    noisy = low + 0.5 * np.sin(2.0 * np.pi * 30.0 * time_s)
    frame = pd.DataFrame(
        {
            "log_id": "log_a",
            "time_s": time_s,
            "flap_frequency_hz": noisy,
            "servo_left_elevon": noisy,
        }
    )

    filtered = apply_input_filters(
        frame,
        {
            "flap_frequency_hz": {"method": "butterworth", "order": 2, "cutoff_hz": 5.0},
            "servo_left_elevon": {"method": "first_order", "time_constant_s": 0.04},
        },
    )

    assert "flap_frequency_hz_filt" in filtered.columns
    assert "servo_left_elevon_filt" in filtered.columns
    np.testing.assert_allclose(filtered["flap_frequency_hz"].to_numpy(), noisy)
    raw_metrics = finite_difference_quality_metrics(low, noisy, sample_rate_hz=100.0)
    filt_metrics = finite_difference_quality_metrics(low, filtered["flap_frequency_hz_filt"], sample_rate_hz=100.0)
    assert filt_metrics["rmse"] < raw_metrics["rmse"]


def test_apply_input_filters_uses_aligned_column_when_available():
    frame = pd.DataFrame(
        {
            "log_id": "log_a",
            "time_s": np.arange(20, dtype=float) * 0.01,
            "servo_rudder": np.zeros(20),
            "servo_rudder_aligned": np.ones(20),
        }
    )

    filtered = apply_input_filters(
        frame,
        {"servo_rudder": {"method": "first_order", "time_constant_s": 0.04}},
    )

    assert filtered["servo_rudder_filt"].iloc[-1] > 0.9
