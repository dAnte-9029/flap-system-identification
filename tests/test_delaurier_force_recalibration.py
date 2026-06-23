from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_delaurier_force_recalibration import (
    FORCE_COLUMNS,
    ForceAffineModel,
    apply_force_model,
    fit_per_channel_affine,
    fit_shared_gain_channel_bias,
    force_metrics_table,
    run_force_recalibration,
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


def _write_keyed_recalibration_split(split_root: Path, prior_root: Path, split: str) -> None:
    rows = []
    priors = []
    for index in range(6):
        prior_fx = float(index)
        prior_fy = float(index + 1)
        prior_fz = float(index - 2)
        rows.append(
            {
                "dataset_id": "dataset_a",
                "log_id": f"{split}_log",
                "segment_id": 0,
                "time_s": float(index) * 0.01,
                "phase_corrected_rad": float(index) * 0.2,
                "fx_b": 2.0 * prior_fx + 1.0,
                "fy_b": -1.0 * prior_fy + 0.5,
                "fz_b": 0.25 * prior_fz - 3.0,
            }
        )
        priors.append(
            {
                "dataset_id": "dataset_a",
                "log_id": f"{split}_log",
                "segment_id": 0,
                "time_s": float(index) * 0.01,
                "fx_b": prior_fx,
                "fy_b": prior_fy,
                "fz_b": prior_fz,
            }
        )
    pd.DataFrame(rows).to_parquet(split_root / f"{split}_samples.parquet", index=False)
    pd.DataFrame(priors).iloc[::-1].reset_index(drop=True).to_parquet(
        prior_root / f"{split}_predictions.parquet",
        index=False,
    )


def test_run_force_recalibration_aligns_shuffled_keyed_prior_rows(tmp_path: Path) -> None:
    split_root = tmp_path / "split"
    prior_root = tmp_path / "prior"
    output_root = tmp_path / "out"
    split_root.mkdir()
    prior_root.mkdir()
    for split in ("train", "val", "test"):
        _write_keyed_recalibration_split(split_root, prior_root, split)

    run_force_recalibration(
        split_root=split_root,
        prior_root=prior_root,
        output_root=output_root,
        channel_weights={"fx_b": 1.0, "fy_b": 1.0, "fz_b": 1.0},
        phase_bins=3,
        skip_frequency=True,
    )

    parameters = pd.read_csv(output_root / "parameters.csv")
    a1_gains = parameters.loc[
        (parameters["variant"] == "A1_per_channel_affine")
        & (parameters["term"] == "gain")
        & (parameters["output_channel"] == parameters["input_channel"])
    ].sort_values("output_channel")
    np.testing.assert_allclose(a1_gains["value"].to_numpy(), [2.0, -1.0, 0.25], atol=1e-12)
    manifest = pd.read_json(output_root / "manifest.json", typ="series")
    assert manifest["alignment"]["train"]["alignment_mode"] == "key_merge"
