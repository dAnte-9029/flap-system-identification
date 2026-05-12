from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_model_sanity_checks import (
    build_split_protocol_table,
    compute_mean_baseline_metrics,
    compute_permuted_target_metrics,
    scan_bundle_inputs,
)


TARGET_COLUMNS = ["fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b"]


def _target_frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, columns=TARGET_COLUMNS)


def test_build_split_protocol_table_flags_shared_log_ids(tmp_path):
    pd.DataFrame({"log_id": ["log_a", "log_b"], "fx_b": [1.0, 2.0]}).to_parquet(
        tmp_path / "train_samples.parquet"
    )
    pd.DataFrame({"log_id": ["log_b", "log_c"], "fx_b": [3.0, 4.0]}).to_parquet(
        tmp_path / "val_samples.parquet"
    )
    pd.DataFrame({"log_id": ["log_d"], "fx_b": [5.0]}).to_parquet(tmp_path / "test_samples.parquet")

    table = build_split_protocol_table(tmp_path)

    overlap = table[(table["check"] == "log_overlap") & (table["split_a"] == "train") & (table["split_b"] == "val")]
    assert len(overlap) == 1
    assert bool(overlap.iloc[0]["passed"]) is False
    assert overlap.iloc[0]["matched_columns"] == "log_b"


def test_scan_bundle_inputs_flags_acceleration_and_target_leakage():
    bundle = {
        "feature_columns": [
            "phase_corrected_sin",
            "vehicle_local_position.ax",
            "fx_b",
        ],
        "sequence_feature_columns": [
            "servo_left_elevon",
            "vehicle_angular_velocity.xyz_derivative[0]",
            "fy_b@t-1",
        ],
        "current_feature_columns": ["airspeed_validated.true_airspeed_m_s"],
        "target_columns": TARGET_COLUMNS,
    }

    table = scan_bundle_inputs(bundle)

    failed = set(table.loc[~table["passed"], "check"])
    assert "no_acceleration_inputs" in failed
    assert "no_target_columns_in_inputs" in failed


def test_baseline_and_permuted_target_metrics_report_sanity_values():
    rng = np.random.default_rng(12)
    train_targets = _target_frame(
        rng.normal(size=(24, len(TARGET_COLUMNS)))
    )
    test_targets = _target_frame(
        rng.normal(size=(24, len(TARGET_COLUMNS)))
    )

    mean_metrics = compute_mean_baseline_metrics(train_targets, test_targets, target_columns=TARGET_COLUMNS)
    assert mean_metrics["baseline"] == "train_target_mean"
    assert "overall_r2" in mean_metrics
    assert "fx_b_r2" in mean_metrics

    perfect_predictions = test_targets.to_numpy(dtype=float)
    permuted = compute_permuted_target_metrics(
        test_targets.to_numpy(dtype=float),
        perfect_predictions,
        target_columns=TARGET_COLUMNS,
        seed=7,
    )
    assert permuted["baseline"] == "model_predictions_vs_permuted_test_targets"
    assert permuted["overall_r2"] < 0.5
