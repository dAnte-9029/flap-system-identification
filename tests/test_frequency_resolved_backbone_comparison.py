from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.frequency_resolved_backbone_comparison import (
    compute_frequency_component_metrics,
    filtered_arrays_for_frequency_component,
)


def test_frequency_component_metrics_separate_low_and_high_residual():
    sample_rate_hz = 100.0
    time_s = np.arange(1000, dtype=float) / sample_rate_hz
    low = np.sin(2.0 * np.pi * 0.5 * time_s)
    mid = 0.5 * np.sin(2.0 * np.pi * 2.0 * time_s)
    flap = 0.25 * np.sin(2.0 * np.pi * 5.0 * time_s)
    high = 0.4 * np.sin(2.0 * np.pi * 18.0 * time_s)
    true = low + mid + flap + high
    pred = low + mid + flap
    frame = pd.DataFrame(
        {
            "log_id": "a",
            "segment_id": 0,
            "time_s": time_s,
            "cycle_flap_frequency_hz": 5.0,
            "true_fy_b": true,
            "pred_fy_b": pred,
        }
    )

    table = compute_frequency_component_metrics(frame, targets=("fy_b",))

    low_r2 = float(table.loc[(table["target"] == "fy_b") & (table["component"] == "low_0_1hz"), "r2"].iloc[0])
    residual_r2 = float(
        table.loc[(table["target"] == "fy_b") & (table["component"] == "high_frequency_residual"), "r2"].iloc[0]
    )

    assert low_r2 > 0.95
    assert residual_r2 < 0.1


def test_high_frequency_residual_subtracts_structured_bands():
    sample_rate_hz = 100.0
    time_s = np.arange(1000, dtype=float) / sample_rate_hz
    low = np.sin(2.0 * np.pi * 0.5 * time_s)
    mid = 0.5 * np.sin(2.0 * np.pi * 2.0 * time_s)
    flap = 0.25 * np.sin(2.0 * np.pi * 5.0 * time_s)
    high = 0.4 * np.sin(2.0 * np.pi * 18.0 * time_s)
    frame = pd.DataFrame(
        {
            "log_id": "a",
            "segment_id": 0,
            "time_s": time_s,
            "cycle_flap_frequency_hz": 5.0,
            "true_fy_b": low + mid + flap + high,
            "pred_fy_b": low + mid + flap,
        }
    )

    true_resid, pred_resid = filtered_arrays_for_frequency_component(frame, target="fy_b", component="high_frequency_residual")

    assert np.corrcoef(true_resid, high)[0, 1] > 0.90
    assert np.std(pred_resid) < np.std(high)
