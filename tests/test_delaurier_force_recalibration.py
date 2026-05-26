from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_delaurier_force_recalibration import (
    FORCE_COLUMNS,
    ForceAffineModel,
    apply_force_model,
    fit_per_channel_affine,
    fit_shared_gain_channel_bias,
    force_metrics_table,
)


def test_per_channel_affine_recovers_channel_gains_and_biases() -> None:
    prior = pd.DataFrame(
        {
            "fx_b": [0.0, 1.0, 2.0, 3.0],
            "fy_b": [1.0, 2.0, 3.0, 4.0],
            "fz_b": [-2.0, -1.0, 0.0, 1.0],
        }
    )
    true = pd.DataFrame(
        {
            "fx_b": 2.0 * prior["fx_b"] + 1.0,
            "fy_b": -1.0 * prior["fy_b"] + 0.5,
            "fz_b": 0.25 * prior["fz_b"] - 3.0,
        }
    )

    model = fit_per_channel_affine(prior, true)
    corrected = apply_force_model(prior, model)

    np.testing.assert_allclose(model.gain_matrix, np.diag([2.0, -1.0, 0.25]), atol=1e-12)
    np.testing.assert_allclose(model.bias, [1.0, 0.5, -3.0], atol=1e-12)
    np.testing.assert_allclose(corrected[list(FORCE_COLUMNS)], true[list(FORCE_COLUMNS)], atol=1e-12)


def test_shared_gain_channel_bias_uses_channel_weights() -> None:
    prior = pd.DataFrame(
        {
            "fx_b": [0.0, 1.0, 2.0, 3.0],
            "fy_b": [0.0, 1.0, 2.0, 3.0],
            "fz_b": [0.0, 1.0, 2.0, 3.0],
        }
    )
    true = pd.DataFrame(
        {
            "fx_b": 2.0 * prior["fx_b"] + 10.0,
            "fy_b": 5.0 * prior["fy_b"] - 2.0,
            "fz_b": 2.0 * prior["fz_b"] + 4.0,
        }
    )

    unweighted = fit_shared_gain_channel_bias(prior, true, channel_weights={"fx_b": 1.0, "fy_b": 1.0, "fz_b": 1.0})
    weighted = fit_shared_gain_channel_bias(prior, true, channel_weights={"fx_b": 10.0, "fy_b": 0.01, "fz_b": 10.0})

    assert abs(unweighted.gain_matrix[0, 0] - 3.0) < 1e-12
    assert abs(weighted.gain_matrix[0, 0] - 2.0) < abs(unweighted.gain_matrix[0, 0] - 2.0)
    assert np.allclose(np.diag(weighted.gain_matrix), weighted.gain_matrix[0, 0])
    assert np.allclose(weighted.gain_matrix - np.diag(np.diag(weighted.gain_matrix)), 0.0)


def test_force_metrics_table_reports_rmse_r2_and_weighted_score() -> None:
    true = pd.DataFrame(
        {
            "fx_b": [1.0, 3.0, 5.0],
            "fy_b": [2.0, 2.0, 2.0],
            "fz_b": [-1.0, -2.0, -3.0],
        }
    )
    pred = pd.DataFrame(
        {
            "fx_b": [1.0, 2.0, 6.0],
            "fy_b": [1.0, 2.0, 3.0],
            "fz_b": [-2.0, -2.0, -2.0],
        }
    )
    rows = force_metrics_table(
        true,
        pred,
        split="unit",
        variant="A_test",
        channel_weights={"fx_b": 2.0, "fy_b": 0.5, "fz_b": 2.0},
    )

    by_target = {row["target"]: row for row in rows}
    assert by_target["fx_b"]["rmse"] == np.sqrt(2.0 / 3.0)
    assert by_target["fx_b"]["r2"] == 0.75
    assert np.isnan(by_target["fy_b"]["r2"])

    summary = by_target["force_mean"]
    assert summary["rmse"] == np.mean([by_target[target]["rmse"] for target in FORCE_COLUMNS])
    assert summary["weighted_rmse"] == np.average(
        [by_target[target]["rmse"] for target in FORCE_COLUMNS],
        weights=[2.0, 0.5, 2.0],
    )


def test_force_affine_model_rejects_bad_shapes() -> None:
    prior = pd.DataFrame({"fx_b": [1.0], "fy_b": [2.0], "fz_b": [3.0]})
    model = ForceAffineModel(name="bad", gain_matrix=np.ones((2, 2)), bias=np.zeros(3), description="bad")

    with np.testing.assert_raises(ValueError):
        apply_force_model(prior, model)
