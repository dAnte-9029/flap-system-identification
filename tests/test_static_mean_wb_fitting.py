from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from system_identification.models.correction.specifications import StaticCorrectionSpec
from system_identification.training.correction.fitting import fit_candidate, fit_weighted_ridge


def _spec(**overrides: object) -> StaticCorrectionSpec:
    values: dict[str, object] = {
        "model_type": "shaped_prior_mean_wb",
        "force_component": "fx",
        "harmonic_order": 1,
        "condition_set": "none",
        "mean_prior_retention": 0.5,
        "waveform_prior_retention": 0.75,
        "ridge_lambda_mean": 0.0,
        "ridge_lambda_waveform": 0.0,
        "mean_weighting": "equal_cycle",
        "waveform_weighting": "equal_sample",
        "fit_intercept": True,
    }
    values.update(overrides)
    return StaticCorrectionSpec(**values)


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


def _synthetic_frames(cycle_count: int = 12, samples_per_cycle: int = 16) -> tuple[pd.DataFrame, pd.DataFrame]:
    cycle_rows: list[dict[str, object]] = []
    waveform_rows: list[dict[str, object]] = []
    for cycle_index in range(cycle_count):
        cycle_id = f"c{cycle_index}"
        alpha = -1.0 + 2.0 * cycle_index / max(cycle_count - 1, 1)
        frequency = np.cos(cycle_index)
        prior_mean = 0.2 * cycle_index
        cycle_rows.append(
            {
                "cycle_id": cycle_id,
                "partition": "train",
                "log_id": f"log_{cycle_index % 3}",
                "flight_date": f"2026-04-{12 + cycle_index % 2:02d}",
                "alpha_mean_std": alpha,
                "flapping_frequency_mean_std": frequency,
                "prior_fx_mean_n": prior_mean,
                "label_fx_mean_n": 0.5 * prior_mean + 2.0 + 3.0 * alpha,
                "weight_equal_cycle": 1.0,
                "weight_equal_log": 1.0 / 4.0,
                "weight_equal_date": 1.0 / 6.0,
            }
        )
        phase = np.arange(samples_per_cycle) * 2.0 * np.pi / samples_per_cycle
        prior_wave = 0.3 * np.sin(phase) - 0.1 * np.cos(phase)
        label_wave = 0.75 * prior_wave + 2.0 * np.sin(phase) - 0.5 * np.cos(phase)
        for sample_index, value in enumerate(phase):
            row: dict[str, object] = {
                "cycle_id": cycle_id,
                "partition": "train",
                "log_id": f"log_{cycle_index % 3}",
                "flight_date": f"2026-04-{12 + cycle_index % 2:02d}",
                "timestamp_us": cycle_index * 1_000_000 + sample_index,
                "alpha_mean_std": alpha,
                "flapping_frequency_mean_std": frequency,
                "prior_fx_mean_n": prior_mean,
                "label_fx_mean_n": cycle_rows[-1]["label_fx_mean_n"],
                "prior_fx_waveform_n": prior_wave[sample_index],
                "label_fx_waveform_n": label_wave[sample_index],
                "prior_fx_n": prior_mean + prior_wave[sample_index],
                "label_fx_n": cycle_rows[-1]["label_fx_mean_n"] + label_wave[sample_index],
                "weight_equal_cycle_sample": 1.0 / samples_per_cycle,
                "weight_equal_log_sample": 1.0 / (4.0 * samples_per_cycle),
                "weight_equal_date_sample": 1.0 / (6.0 * samples_per_cycle),
            }
            for harmonic in range(1, 5):
                row[f"sin_{harmonic}_phase_centered"] = np.sin(harmonic * value)
                row[f"cos_{harmonic}_phase_centered"] = np.cos(harmonic * value)
            waveform_rows.append(row)
    return pd.DataFrame(cycle_rows), pd.DataFrame(waveform_rows)


def test_mean_only_and_single_harmonic_coefficients_are_recovered() -> None:
    cycle, waveform = _synthetic_frames()
    bundle = fit_candidate(_spec(condition_set="alpha"), cycle, waveform, {}, _provenance())
    assert bundle.mean_solution is not None
    assert bundle.waveform_solution is not None
    np.testing.assert_allclose(bundle.mean_solution.coefficients, [2.0, 3.0], atol=1e-10)
    np.testing.assert_allclose(bundle.waveform_solution.coefficients[:2], [2.0, -0.5], atol=1e-10)
    np.testing.assert_allclose(bundle.waveform_solution.coefficients[2:], 0.0, atol=1e-10)


def test_conditioned_second_harmonic_coefficients_are_recovered() -> None:
    cycle, waveform = _synthetic_frames(samples_per_cycle=24)
    waveform["prior_fx_waveform_n"] = 0.0
    waveform["label_fx_waveform_n"] = (
        (1.0 + 0.4 * waveform["alpha_mean_std"]) * waveform["sin_2_phase_centered"]
        + (-0.2 + 0.3 * waveform["flapping_frequency_mean_std"]) * waveform["cos_2_phase_centered"]
    )
    spec = _spec(
        harmonic_order=2,
        condition_set="alpha_frequency",
        mean_prior_retention=0.0,
        waveform_prior_retention=0.0,
    )
    bundle = fit_candidate(spec, cycle, waveform, {}, _provenance())
    names = bundle.waveform_solution.feature_names
    coefficients = dict(zip(names, bundle.waveform_solution.coefficients))
    assert coefficients["sin_2_phase_centered"] == pytest.approx(1.0, abs=1e-10)
    assert coefficients["alpha_x_sin_2"] == pytest.approx(0.4, abs=1e-10)
    assert coefficients["cos_2_phase_centered"] == pytest.approx(-0.2, abs=1e-10)
    assert coefficients["frequency_x_cos_2"] == pytest.approx(0.3, abs=1e-10)


def test_weighted_ridge_intercept_penalty_and_diagnostics() -> None:
    x = np.column_stack([np.arange(6, dtype=float), np.ones(6)])
    y = 2.0 * x[:, 0] + 3.0
    solution = fit_weighted_ridge(
        x,
        y,
        np.array([1, 1, 1, 10, 10, 10], dtype=float),
        ridge_lambda=1.0,
        feature_names=("slope", "intercept"),
        penalize_mask=np.array([True, False]),
    )
    assert solution.coefficients[1] > 2.9
    assert np.isfinite(solution.coefficients).all()
    assert solution.diagnostics.matrix_rank == 2
    assert np.isfinite(solution.diagnostics.condition_number)


def test_fitting_is_deterministic_and_reports_rank_deficiency() -> None:
    x = np.column_stack([np.arange(5, dtype=float)] * 2)
    y = np.arange(5, dtype=float)
    first = fit_weighted_ridge(x, y, np.ones(5), 0.0, ("a", "b"), np.ones(2, dtype=bool))
    second = fit_weighted_ridge(x, y, np.ones(5), 0.0, ("a", "b"), np.ones(2, dtype=bool))
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    assert first.diagnostics.matrix_rank == 1
    assert first.diagnostics.rank_deficient is True


@pytest.mark.parametrize("partition", ["validation", "test"])
def test_non_train_fitting_is_rejected(partition: str) -> None:
    cycle, waveform = _synthetic_frames()
    cycle["partition"] = partition
    waveform["partition"] = partition
    with pytest.raises(ValueError, match="train-only"):
        fit_candidate(_spec(), cycle, waveform, {}, _provenance())


def test_zero_rows_nan_and_bad_weights_fail_closed() -> None:
    with pytest.raises(ValueError, match="zero rows"):
        fit_weighted_ridge(np.empty((0, 1)), np.empty(0), np.empty(0), 0.0, ("x",), np.ones(1, bool))
    with pytest.raises(ValueError, match="non-finite"):
        fit_weighted_ridge(np.array([[np.nan]]), np.array([1.0]), np.ones(1), 0.0, ("x",), np.ones(1, bool))
    with pytest.raises(ValueError, match="weights"):
        fit_weighted_ridge(np.array([[1.0]]), np.array([1.0]), np.zeros(1), 0.0, ("x",), np.ones(1, bool))
