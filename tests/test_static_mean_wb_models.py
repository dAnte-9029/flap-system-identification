from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from system_identification.models.correction.prediction import predict_cycle_mean, predict_total, predict_waveform
from system_identification.models.correction.specifications import StaticCorrectionSpec
from system_identification.training.correction.fitting import fit_candidate


def _provenance() -> dict[str, object]:
    return {
        "correction_ready_artifact_id": "synthetic",
        "correction_ready_manifest_hash": "0" * 64,
        "dataset_id": "canonical_v0.4_training_ready_split_measured_massprops_ratio8_phasefix_v3",
        "dataset_hash": "1" * 64,
        "prior_id": "delaurier_attitude_aware_3b5d4ec_ratio8_phasefix_trainval_v4",
        "prior_hash": "2" * 64,
        "ratio_contract": "ratio8_v1",
        "phase_contract": "hall_indexed_mechanical_phase_ratio8_v1",
        "included_partitions": ["train"],
    }


def _spec(model_type: str = "shaped_prior_mean_wb", component: str = "fx", **kwargs: object) -> StaticCorrectionSpec:
    retentions: dict[str, object]
    if model_type == "fixed_prior_mean_wb":
        retentions = {"mean_prior_retention": 1.0, "waveform_prior_retention": 1.0}
    elif model_type == "no_prior_mean_wb":
        retentions = {"mean_prior_retention": 0.0, "waveform_prior_retention": 0.0}
    elif model_type == "shaped_prior_mean_wb":
        retentions = {"mean_prior_retention": 0.5, "waveform_prior_retention": 0.75}
    else:
        retentions = {"mean_prior_retention": None, "waveform_prior_retention": None}
    values: dict[str, object] = {
        "model_type": model_type,
        "force_component": component,
        "harmonic_order": 1 if model_type.endswith("mean_wb") else None,
        "condition_set": "none",
        **retentions,
        "ridge_lambda_mean": 0.0,
        "ridge_lambda_waveform": 0.0,
        "mean_weighting": "equal_cycle",
        "waveform_weighting": "equal_sample",
        "fit_intercept": model_type not in {"raw_prior", "physical_component_scale"},
    }
    values.update(kwargs)
    return StaticCorrectionSpec(**values)


def _frames(component: str = "fx", cycles: int = 5, samples: int = 16) -> tuple[pd.DataFrame, pd.DataFrame]:
    cycle_rows: list[dict[str, object]] = []
    wave_rows: list[dict[str, object]] = []
    for c in range(cycles):
        cycle_id = f"c{c}"
        prior_mean = 0.5 * c - 0.25
        label_mean = 0.5 * prior_mean + 1.25
        cycle_rows.append(
            {
                "cycle_id": cycle_id,
                "partition": "train",
                "log_id": f"l{c % 2}",
                "flight_date": "2026-04-12",
                "alpha_mean_std": float(c),
                "flapping_frequency_mean_std": float(-c),
                f"prior_{component}_mean_n": prior_mean,
                f"label_{component}_mean_n": label_mean,
                "weight_equal_cycle": 1.0,
                "weight_equal_log": 0.5,
                "weight_equal_date": 0.2,
            }
        )
        phase = np.arange(samples) * 2 * np.pi / samples
        prior_wave = np.sin(phase)
        label_wave = 0.75 * prior_wave + 0.4 * np.cos(phase)
        for i, value in enumerate(phase):
            row: dict[str, object] = {
                "cycle_id": cycle_id,
                "partition": "train",
                "log_id": f"l{c % 2}",
                "flight_date": "2026-04-12",
                "timestamp_us": c * 1_000_000 + i,
                "alpha_mean_std": float(c),
                "flapping_frequency_mean_std": float(-c),
                f"prior_{component}_mean_n": prior_mean,
                f"label_{component}_mean_n": label_mean,
                f"prior_{component}_waveform_n": prior_wave[i],
                f"label_{component}_waveform_n": label_wave[i],
                f"prior_{component}_n": prior_mean + prior_wave[i],
                f"label_{component}_n": label_mean + label_wave[i],
                "weight_equal_cycle_sample": 1.0 / samples,
                "weight_equal_log_sample": 1.0 / samples,
                "weight_equal_date_sample": 1.0 / samples,
            }
            for k in range(1, 5):
                row[f"sin_{k}_phase_centered"] = np.sin(k * value)
                row[f"cos_{k}_phase_centered"] = np.cos(k * value)
            wave_rows.append(row)
    return pd.DataFrame(cycle_rows), pd.DataFrame(wave_rows)


def test_raw_prior_is_exact_identity_and_preserves_row_order() -> None:
    cycle, waveform = _frames()
    bundle = fit_candidate(_spec("raw_prior"), cycle, waveform, {}, _provenance())
    shuffled = waveform.sample(frac=1.0, random_state=12).reset_index(drop=True)
    predicted = predict_total(bundle, cycle, shuffled)
    np.testing.assert_array_equal(predicted["timestamp_us"], shuffled["timestamp_us"])
    np.testing.assert_allclose(predicted["prediction_n"], shuffled["prior_fx_n"], atol=1e-14)
    np.testing.assert_allclose(predicted["prediction_n"], predicted["predicted_mean_n"] + predicted["predicted_waveform_n"])


def test_gain_bias_recovers_exact_total_mapping() -> None:
    cycle, waveform = _frames()
    waveform["label_fx_n"] = 1.2 * waveform["prior_fx_n"] - 0.4
    bundle = fit_candidate(_spec("gain_bias"), cycle, waveform, {}, _provenance())
    prediction = predict_total(bundle, cycle, waveform)
    np.testing.assert_allclose(prediction["prediction_n"], waveform["label_fx_n"], atol=1e-10)
    np.testing.assert_allclose(bundle.waveform_solution.coefficients, [1.2, -0.4], atol=1e-10)


def test_component_scale_recovers_bounded_normal_force_scale() -> None:
    cycle, waveform = _frames(component="fz")
    normal = np.linspace(-4.0, 4.0, len(waveform))
    other = np.linspace(1.0, 2.0, len(waveform))
    waveform["prior_fz_normal_component_n"] = normal
    waveform["prior_fz_other_component_n"] = other
    waveform["prior_fz_n"] = normal + other
    waveform["label_fz_n"] = 0.7 * normal + other
    spec = _spec(
        "physical_component_scale",
        component="fz",
        physical_component="normal_force",
        coefficient_constraints={"scale_min": 0.0, "scale_max": 2.0, "strategy": "clip_after_fit"},
    )
    bundle = fit_candidate(spec, cycle, waveform, {}, _provenance())
    assert bundle.component_scale == pytest.approx(0.7, abs=1e-10)
    np.testing.assert_allclose(predict_total(bundle, cycle, waveform)["prediction_n"], waveform["label_fz_n"], atol=1e-10)


def test_component_sum_mismatch_fails_closed() -> None:
    cycle, waveform = _frames(component="fz")
    waveform["prior_fz_normal_component_n"] = 1.0
    waveform["prior_fz_other_component_n"] = 1.0
    spec = _spec("physical_component_scale", component="fz", physical_component="normal_force")
    with pytest.raises(ValueError, match="component sum"):
        fit_candidate(spec, cycle, waveform, {}, _provenance())


def test_fixed_and_no_prior_degenerations_match_shaped_endpoints() -> None:
    cycle, waveform = _frames()
    fixed = fit_candidate(_spec("fixed_prior_mean_wb"), cycle, waveform, {}, _provenance())
    shaped_one = fit_candidate(
        _spec("shaped_prior_mean_wb", mean_prior_retention=1.0, waveform_prior_retention=1.0),
        cycle,
        waveform,
        {},
        _provenance(),
    )
    no_prior = fit_candidate(_spec("no_prior_mean_wb"), cycle, waveform, {}, _provenance())
    shaped_zero = fit_candidate(
        _spec("shaped_prior_mean_wb", mean_prior_retention=0.0, waveform_prior_retention=0.0),
        cycle,
        waveform,
        {},
        _provenance(),
    )
    np.testing.assert_allclose(predict_total(fixed, cycle, waveform)["prediction_n"], predict_total(shaped_one, cycle, waveform)["prediction_n"])
    np.testing.assert_allclose(predict_total(no_prior, cycle, waveform)["prediction_n"], predict_total(shaped_zero, cycle, waveform)["prediction_n"])


def test_no_prior_prediction_does_not_require_prior_columns() -> None:
    cycle, waveform = _frames()
    bundle = fit_candidate(_spec("no_prior_mean_wb"), cycle, waveform, {}, _provenance())
    cycle_without_prior = cycle.drop(columns="prior_fx_mean_n")
    waveform_without_prior = waveform.drop(columns=["prior_fx_mean_n", "prior_fx_waveform_n", "prior_fx_n"])
    result = predict_total(bundle, cycle_without_prior, waveform_without_prior)
    assert np.isfinite(result["prediction_n"]).all()


def test_waveform_is_zero_mean_and_total_equals_branch_sum() -> None:
    cycle, waveform = _frames()
    bundle = fit_candidate(_spec(), cycle, waveform, {}, _provenance())
    mean = predict_cycle_mean(bundle, cycle)
    wave = predict_waveform(bundle, waveform)
    total = predict_total(bundle, cycle, waveform)
    assert np.max(np.abs(wave.groupby("cycle_id")["predicted_waveform_n"].mean())) < 1e-12
    np.testing.assert_allclose(total["prediction_n"], total["predicted_mean_n"] + total["predicted_waveform_n"])
    assert len(mean) == len(cycle)


def test_missing_or_duplicate_cycle_ids_fail_closed() -> None:
    cycle, waveform = _frames()
    bundle = fit_candidate(_spec(), cycle, waveform, {}, _provenance())
    with pytest.raises(ValueError, match="duplicate cycle_id"):
        predict_total(bundle, pd.concat([cycle, cycle.iloc[[0]]], ignore_index=True), waveform)
    with pytest.raises(ValueError, match="missing cycle mean"):
        predict_total(bundle, cycle.iloc[1:].copy(), waveform)


def test_fx_and_fz_models_have_independent_coefficients() -> None:
    fx_cycle, fx_wave = _frames("fx")
    fz_cycle, fz_wave = _frames("fz")
    fz_cycle["label_fz_mean_n"] += 10.0
    fx = fit_candidate(_spec(component="fx"), fx_cycle, fx_wave, {}, _provenance())
    fz = fit_candidate(_spec(component="fz"), fz_cycle, fz_wave, {}, _provenance())
    assert fx.spec.force_component == "fx"
    assert fz.spec.force_component == "fz"
    assert not np.array_equal(fx.mean_solution.coefficients, fz.mean_solution.coefficients)
