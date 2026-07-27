from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from system_identification.artifacts.static_correction_bundle import load_static_bundle, save_static_bundle
from system_identification.evaluation.static_correction_metrics import (
    circular_peak_phase_error,
    half_stroke_integral_error,
    per_log_total_metrics,
    per_log_waveform_metrics,
)
from system_identification.evaluation.static_correction_validation import _prior_value_comparisons
from system_identification.models.correction.bundles import StaticCorrectionBundle, compute_bundle_hash
from system_identification.models.correction.specifications import StaticCorrectionSpec


def test_per_log_macro_is_not_sample_pooled() -> None:
    frame = pd.DataFrame(
        {
            "log_id": ["long"] * 100 + ["short"],
            "label_n": np.zeros(101),
            "prediction_n": [1.0] * 100 + [3.0],
        }
    )
    per_log = per_log_total_metrics(frame)
    assert per_log["rmse"].mean() == pytest.approx(2.0)
    assert np.sqrt(np.mean(frame["prediction_n"] ** 2)) != pytest.approx(2.0)


def test_total_metrics_rmse_mae_and_bias() -> None:
    frame = pd.DataFrame({"log_id": ["a", "a"], "label_n": [1.0, 3.0], "prediction_n": [2.0, 1.0]})
    row = per_log_total_metrics(frame).iloc[0]
    assert row["rmse"] == pytest.approx(np.sqrt(2.5))
    assert row["mae"] == pytest.approx(1.5)
    assert row["bias"] == pytest.approx(-0.5)


def test_waveform_metric_averages_cycle_rmse_before_log() -> None:
    frame = pd.DataFrame(
        {
            "log_id": ["a"] * 5,
            "cycle_id": ["short", "long", "long", "long", "long"],
            "label_fx_waveform_n": np.zeros(5),
        }
    )
    prediction = np.array([3.0, 1.0, 1.0, 1.0, 1.0])
    metric = per_log_waveform_metrics(frame, prediction, "fx").iloc[0]
    assert metric["waveform_rmse"] == pytest.approx(2.0)


def test_half_stroke_integral_synthetic() -> None:
    phase = [0.0, 0.5, 1.0]
    assert half_stroke_integral_error(phase, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]) == pytest.approx(2.0)


def test_circular_peak_phase_wrap() -> None:
    phase = [0.01, np.pi, 2 * np.pi - 0.01]
    label = [0.0, 0.0, 2.0]
    prediction = [3.0, 0.0, 0.0]
    assert circular_peak_phase_error(phase, label, prediction) == pytest.approx(0.02)


def test_metric_missing_columns_fails() -> None:
    with pytest.raises(ValueError, match="missing"):
        per_log_total_metrics(pd.DataFrame({"log_id": ["a"]}))


def test_selected_train_only_bundle_status_round_trip(tmp_path) -> None:
    spec = StaticCorrectionSpec(model_type="raw_prior", force_component="fx", fit_intercept=False)
    provisional = StaticCorrectionBundle(
        bundle_schema_version="static_correction_bundle_v1",
        model_id="selected_synthetic",
        created_at="2026-07-27T00:00:00+00:00",
        status="selected_static_train_only",
        spec=spec,
        mean_solution=None,
        waveform_solution=None,
        component_scale=None,
        normalization={},
        training_provenance={
            "included_partitions": ["train"],
            "correction_ready_artifact_id": "synthetic",
            "correction_ready_manifest_hash": "0" * 64,
            "dataset_id": "dataset",
            "dataset_hash": "1" * 64,
            "prior_id": "prior",
            "prior_hash": "2" * 64,
            "ratio_contract": "ratio8_v1",
            "phase_contract": "phase",
            "test_labels_loaded": False,
            "dynamic_audit_pending": True,
        },
        fit_summary={
            "train_cycle_count": 2,
            "train_waveform_row_count": 8,
            "coefficient_count": 0,
            "finite_checks": True,
            "selection_performed": True,
        },
        bundle_hash="",
    )
    bundle = StaticCorrectionBundle(
        **{**provisional.__dict__, "bundle_hash": compute_bundle_hash(provisional.hash_payload())}
    )
    loaded = load_static_bundle(save_static_bundle(bundle, tmp_path / "bundle"))
    assert loaded.status == "selected_static_train_only"
    assert loaded.training_provenance["included_partitions"] == ["train"]
    assert loaded.training_provenance["test_labels_loaded"] is False


def test_validation_residual_identity() -> None:
    label = np.array([1.0, -2.0, 3.0])
    prediction = np.array([0.5, -1.0, 2.5])
    residual = label - prediction
    np.testing.assert_allclose(residual, [0.5, -1.0, 0.5])


def test_matched_prior_comparison_can_report_stable_value() -> None:
    common_spec = {
        "force_component": "fx",
        "harmonic_order": 1,
        "mean_condition_set": "none",
        "waveform_condition_set": "none",
        "ridge_lambda_mean": 1e-4,
        "ridge_lambda_waveform": 1e-4,
        "mean_weighting": "equal_log",
        "waveform_weighting": "equal_log",
        "fit_intercept": True,
    }
    metrics = [
        {
            "candidate_id": "no",
            "component": "fx",
            "model_type": "no_prior_mean_wb",
            "macro_total_rmse": 2.0,
            "worst_log_total_rmse": 2.2,
            "spec": {
                **common_spec,
                "model_type": "no_prior_mean_wb",
                "mean_prior_retention": 0.0,
                "waveform_prior_retention": 0.0,
            },
        },
        {
            "candidate_id": "prior",
            "component": "fx",
            "model_type": "fixed_prior_mean_wb",
            "macro_total_rmse": 1.5,
            "worst_log_total_rmse": 2.0,
            "spec": {
                **common_spec,
                "model_type": "fixed_prior_mean_wb",
                "mean_prior_retention": 1.0,
                "waveform_prior_retention": 1.0,
            },
        },
    ]
    per_log = pd.DataFrame(
        {
            "component": ["fx"] * 6,
            "candidate_id": ["no"] * 3 + ["prior"] * 3,
            "log_id": ["a", "b", "c"] * 2,
            "flight_date": ["d1", "d1", "d2"] * 2,
            "rmse": [2.0, 2.2, 1.8, 1.5, 1.7, 1.4],
        }
    )
    result = _prior_value_comparisons(metrics, per_log)
    assert result.iloc[0]["verdict"] == "Stable incremental predictive value demonstrated"


def test_matched_prior_comparison_can_reject_unstable_value() -> None:
    common = {
        "component": "fz",
        "macro_total_rmse": 2.0,
        "worst_log_total_rmse": 2.0,
    }
    base_spec = StaticCorrectionSpec(
        model_type="no_prior_mean_wb",
        force_component="fz",
        harmonic_order=1,
        mean_prior_retention=0.0,
        waveform_prior_retention=0.0,
    )
    prior_spec = StaticCorrectionSpec(
        model_type="shaped_prior_mean_wb",
        force_component="fz",
        harmonic_order=1,
        mean_prior_retention=0.25,
        waveform_prior_retention=0.25,
    )
    metrics = [
        {**common, "candidate_id": "no", "model_type": base_spec.model_type, "spec": base_spec.to_dict()},
        {
            **common,
            "candidate_id": "prior",
            "model_type": prior_spec.model_type,
            "spec": prior_spec.to_dict(),
            "macro_total_rmse": 2.1,
            "worst_log_total_rmse": 2.3,
        },
    ]
    per_log = pd.DataFrame(
        {
            "component": ["fz"] * 4,
            "candidate_id": ["no", "no", "prior", "prior"],
            "log_id": ["a", "b", "a", "b"],
            "flight_date": ["d1", "d2", "d1", "d2"],
            "rmse": [2.0, 2.0, 1.9, 2.3],
        }
    )
    result = _prior_value_comparisons(metrics, per_log)
    assert result.iloc[0]["verdict"] == "No stable incremental predictive value demonstrated"
