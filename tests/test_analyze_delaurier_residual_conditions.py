from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_delaurier_residual_conditions import condition_bin_table, condition_summary_table


def test_condition_bin_table_reports_true_and_remaining_residual_rmse() -> None:
    frame = pd.DataFrame(
        {
            "airspeed": [1.0, 2.0, 3.0, 4.0, np.nan],
            "label_fx_b": [4.0, 6.0, 12.0, 16.0, 100.0],
            "prior_fx_b": [1.0, 2.0, 4.0, 8.0, 0.0],
            "pred_fx_b": [1.0, 2.0, 6.0, 6.0, 0.0],
        }
    )

    table = condition_bin_table(
        frame,
        condition_columns=("airspeed",),
        targets=("fx_b",),
        quantile_bins=2,
        min_samples=1,
    )

    assert table["condition"].tolist() == ["airspeed", "airspeed"]
    assert table["target"].tolist() == ["fx_b", "fx_b"]
    assert table["sample_count"].tolist() == [2, 2]

    low = table.iloc[0]
    assert low["value_min"] == 1.0
    assert low["value_max"] == 2.0
    assert low["value_median"] == 1.5
    assert low["true_residual_rmse"] == np.sqrt((3.0**2 + 4.0**2) / 2.0)
    assert low["remaining_residual_rmse"] == np.sqrt((2.0**2 + 2.0**2) / 2.0)
    assert low["rmse_reduction_fraction"] == 1.0 - low["remaining_residual_rmse"] / low["true_residual_rmse"]

    high = table.iloc[1]
    assert high["value_min"] == 3.0
    assert high["value_max"] == 4.0
    assert high["true_residual_rmse"] == 8.0
    assert high["remaining_residual_rmse"] == 2.0
    assert high["rmse_reduction_fraction"] == 0.75


def test_condition_summary_table_reports_variation_and_worst_bin() -> None:
    bins = pd.DataFrame(
        {
            "condition": ["airspeed", "airspeed", "alpha", "alpha"],
            "target": ["fx_b", "fx_b", "fx_b", "fx_b"],
            "condition_bin": [0, 1, 0, 1],
            "bin_label": ["low", "high", "low", "high"],
            "sample_count": [10, 10, 10, 10],
            "value_median": [1.0, 2.0, 0.1, 0.2],
            "true_residual_rmse": [2.0, 6.0, 5.0, 10.0],
            "remaining_residual_rmse": [1.0, 2.0, 2.0, 4.0],
            "rmse_reduction_fraction": [0.5, 2.0 / 3.0, 0.6, 0.6],
        }
    )

    summary = condition_summary_table(bins)

    airspeed = summary.loc[(summary["condition"] == "airspeed") & (summary["target"] == "fx_b")].iloc[0]
    assert airspeed["bin_count"] == 2
    assert airspeed["true_rmse_min"] == 2.0
    assert airspeed["true_rmse_max"] == 6.0
    assert airspeed["true_rmse_max_to_min"] == 3.0
    assert airspeed["worst_bin_label"] == "high"
    assert airspeed["worst_bin_value_median"] == 2.0

    alpha = summary.loc[(summary["condition"] == "alpha") & (summary["target"] == "fx_b")].iloc[0]
    assert alpha["true_rmse_max_to_min"] == 2.0
