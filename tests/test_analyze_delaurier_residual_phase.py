from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.analyze_delaurier_residual_phase import phase_bin_table, phase_summary_table


def test_phase_bin_table_wraps_phase_and_reports_residual_medians() -> None:
    frame = pd.DataFrame(
        {
            "phase_corrected_rad": [-math.pi / 2.0, math.pi / 4.0, 3.0 * math.pi / 4.0, 9.0 * math.pi / 4.0],
            "label_fx_b": [10.0, 8.0, 4.0, 12.0],
            "prior_fx_b": [7.0, 4.0, 5.0, 9.0],
            "pred_fx_b": [2.0, 3.0, -2.0, 4.0],
        }
    )

    table = phase_bin_table(frame, targets=("fx_b",), phase_bins=4)

    assert list(table["phase_bin"]) == [0, 1, 2, 3]
    assert table["sample_count"].tolist() == [2, 1, 0, 1]

    bin0 = table.loc[table["phase_bin"] == 0].iloc[0]
    assert bin0["phase_center_rad"] == np.pi / 4.0
    assert bin0["true_residual_median"] == 3.5
    assert bin0["pred_residual_median"] == 3.5
    assert bin0["remaining_residual_median"] == 0.0

    bin3 = table.loc[table["phase_bin"] == 3].iloc[0]
    assert bin3["true_residual_median"] == 3.0
    assert bin3["pred_residual_median"] == 2.0
    assert bin3["remaining_residual_median"] == 1.0


def test_phase_summary_table_quantifies_phase_structure_and_residual_capture() -> None:
    phase = np.tile(np.array([0.25 * math.pi, 0.75 * math.pi, 1.25 * math.pi, 1.75 * math.pi]), 3)
    true_residual = np.tile(np.array([2.0, -2.0, 1.0, -1.0]), 3)
    pred_residual = true_residual * 0.5
    frame = pd.DataFrame(
        {
            "phase_corrected_rad": phase,
            "label_fx_b": true_residual,
            "prior_fx_b": np.zeros_like(true_residual),
            "pred_fx_b": pred_residual,
        }
    )

    bins = phase_bin_table(frame, targets=("fx_b",), phase_bins=4)
    summary = phase_summary_table(frame, bins, targets=("fx_b",))
    row = summary.iloc[0]

    assert row["target"] == "fx_b"
    assert row["sample_count"] == len(frame)
    assert row["true_residual_rmse"] == np.sqrt(2.5)
    assert row["remaining_residual_rmse"] == np.sqrt(0.625)
    assert row["rmse_reduction_fraction"] == 0.5
    assert row["true_residual_bias"] == 0.0
    assert row["remaining_residual_bias"] == 0.0
    assert row["true_phase_peak_to_peak"] == 4.0
    assert row["pred_phase_peak_to_peak"] == 2.0
    assert row["remaining_phase_peak_to_peak"] == 2.0
    assert row["phase_peak_to_peak_reduction_fraction"] == 0.5
    assert row["phase_r2_true_residual"] == 1.0
    assert row["phase_r2_remaining_residual"] == 1.0
    assert row["pred_phase_pattern_r2"] == 0.75
