from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.diagnose_fyb_learnability import (
    compute_band_r2_table,
    compute_filter_sweep_table,
    compute_phase_binned_table,
    compute_spike_capture_table,
    fft_filter,
)


def test_fft_filter_keeps_requested_frequency_band():
    sample_rate_hz = 100.0
    time_s = np.arange(1000, dtype=float) / sample_rate_hz
    low_signal = np.sin(2.0 * np.pi * 2.0 * time_s)
    high_signal = 0.5 * np.sin(2.0 * np.pi * 12.0 * time_s)
    mixed = low_signal + high_signal

    filtered = fft_filter(mixed, sample_rate_hz=sample_rate_hz, low_hz=1.0, high_hz=3.0)

    assert np.corrcoef(filtered, low_signal)[0, 1] > 0.98
    assert np.std(filtered - low_signal) < 0.08


def test_compute_band_r2_table_reports_frequency_specific_fit():
    sample_rate_hz = 100.0
    time_s = np.arange(1000, dtype=float) / sample_rate_hz
    low_signal = np.sin(2.0 * np.pi * 2.0 * time_s)
    high_signal = 0.5 * np.sin(2.0 * np.pi * 12.0 * time_s)
    frame = pd.DataFrame(
        {
            "log_id": "log_a",
            "segment_id": 0,
            "time_s": time_s,
            "true_fy_b": low_signal + high_signal,
            "pred_fy_b": low_signal,
        }
    )

    table = compute_band_r2_table(
        frame,
        target="fy_b",
        bands=[("low", 1.0, 3.0), ("high", 10.0, 14.0)],
    )

    low_r2 = float(table.loc[table["band"] == "low", "r2"].iloc[0])
    high_r2 = float(table.loc[table["band"] == "high", "r2"].iloc[0])
    assert low_r2 > 0.95
    assert high_r2 < 0.1


def test_compute_spike_capture_table_detects_underpredicted_spikes():
    true = np.zeros(100)
    pred = np.zeros(100)
    true[-5:] = 10.0
    frame = pd.DataFrame(
        {
            "log_id": "log_a",
            "segment_id": 0,
            "time_s": np.arange(100, dtype=float) * 0.01,
            "true_fy_b": true,
            "pred_fy_b": pred,
        }
    )

    table = compute_spike_capture_table(frame, target="fy_b", quantiles=(0.95,))

    row = table.iloc[0]
    assert row["sample_count"] == 5
    assert row["pred_abs_mean"] < row["true_abs_mean"]


def test_compute_filter_sweep_median_preserves_all_group_rows():
    time_s = np.arange(20, dtype=float) * 0.01
    frame = pd.DataFrame(
        {
            "log_id": ["a"] * 10 + ["b"] * 10,
            "segment_id": [0] * 10 + [0] * 10,
            "time_s": np.concatenate([time_s[:10], time_s[:10]]),
            "true_fy_b": np.sin(time_s),
            "pred_fy_b": np.sin(time_s),
        }
    )

    table = compute_filter_sweep_table(frame, median_windows_s=(0.05,))
    median_row = table.loc[table["filter"] == "rolling_median_s"].iloc[0]

    assert median_row["sample_count"] == len(frame)


def test_compute_phase_binned_table_recovers_phase_locked_high_frequency():
    sample_rate_hz = 100.0
    time_s = np.arange(1000, dtype=float) / sample_rate_hz
    phase = np.mod(2.0 * np.pi * 5.0 * time_s, 2.0 * np.pi)
    phase_locked = np.sin(phase)
    frame = pd.DataFrame(
        {
            "log_id": "a",
            "segment_id": 0,
            "time_s": time_s,
            "phase_corrected_rad": phase,
            "true_fy_b": phase_locked,
            "pred_fy_b": phase_locked,
        }
    )

    table = compute_phase_binned_table(
        frame,
        target="fy_b",
        highpass_hz=3.0,
        phase_bins=24,
    )

    assert table["true_hpf_mean"].max() - table["true_hpf_mean"].min() > 1.5
    assert table["sample_count"].min() > 0
