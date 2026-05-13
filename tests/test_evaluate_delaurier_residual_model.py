import numpy as np
import pandas as pd

from scripts.evaluate_delaurier_residual_model import TARGET_COLUMNS, combined_metrics_from_aligned


def test_combined_metrics_add_prior_and_predicted_residual() -> None:
    aligned = pd.DataFrame(
        {
            "true_fx_b": [10.0, 12.0],
            "label_fx_b": [18.0, 22.0],
            "true_fy_b": [0.0, 0.0],
            "label_fy_b": [0.0, 0.0],
            "true_fz_b": [0.0, 0.0],
            "label_fz_b": [0.0, 0.0],
            "true_mx_b": [0.0, 0.0],
            "label_mx_b": [0.0, 0.0],
            "true_my_b": [0.0, 0.0],
            "label_my_b": [0.0, 0.0],
            "true_mz_b": [0.0, 0.0],
            "label_mz_b": [0.0, 0.0],
            "prior_fx_b": [8.0, 9.0],
            "prior_fy_b": [0.0, 0.0],
            "prior_fz_b": [0.0, 0.0],
            "prior_mx_b": [0.0, 0.0],
            "prior_my_b": [0.0, 0.0],
            "prior_mz_b": [0.0, 0.0],
            "pred_fx_b": [1.0, 2.0],
            "pred_fy_b": [0.0, 0.0],
            "pred_fz_b": [0.0, 0.0],
            "pred_mx_b": [0.0, 0.0],
            "pred_my_b": [0.0, 0.0],
            "pred_mz_b": [0.0, 0.0],
        }
    )

    metrics = combined_metrics_from_aligned(aligned)

    fx = metrics.loc[metrics["target"] == "fx_b"].iloc[0]
    assert fx["prior_rmse"] == np.sqrt((10.0**2 + 13.0**2) / 2.0)
    assert fx["combined_rmse"] == np.sqrt((9.0**2 + 11.0**2) / 2.0)
    assert set(metrics["target"]) == set(TARGET_COLUMNS)
