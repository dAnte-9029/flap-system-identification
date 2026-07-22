from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from system_identification.models.correction.features import (
    build_mean_design,
    build_waveform_design,
)
from system_identification.models.correction.specifications import StaticCorrectionSpec


def _spec(**overrides: object) -> StaticCorrectionSpec:
    values: dict[str, object] = {
        "model_type": "shaped_prior_mean_wb",
        "force_component": "fx",
        "harmonic_order": 2,
        "condition_set": "alpha_frequency",
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


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    cycle = pd.DataFrame(
        {
            "cycle_id": ["c0", "c1"],
            "alpha_mean_std": [-1.0, 1.0],
            "flapping_frequency_mean_std": [0.25, -0.25],
        }
    )
    rows: list[dict[str, object]] = []
    for cycle_id, alpha, frequency in (("c0", -1.0, 0.25), ("c1", 1.0, -0.25)):
        phase = np.arange(8) * 2.0 * np.pi / 8.0
        for index, value in enumerate(phase):
            row: dict[str, object] = {
                "cycle_id": cycle_id,
                "timestamp_us": index,
                "alpha_mean_std": alpha,
                "flapping_frequency_mean_std": frequency,
            }
            for harmonic in range(1, 5):
                row[f"sin_{harmonic}_phase_centered"] = np.sin(harmonic * value)
                row[f"cos_{harmonic}_phase_centered"] = np.cos(harmonic * value)
            rows.append(row)
    return cycle, pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("model_type", "kwargs"),
    [
        ("raw_prior", {"harmonic_order": None, "condition_set": "none", "mean_prior_retention": None, "waveform_prior_retention": None, "fit_intercept": False}),
        ("gain_bias", {"harmonic_order": None, "condition_set": "none", "mean_prior_retention": None, "waveform_prior_retention": None}),
        (
            "physical_component_scale",
            {
                "harmonic_order": None,
                "condition_set": "none",
                "force_component": "fz",
                "mean_prior_retention": None,
                "waveform_prior_retention": None,
                "physical_component": "normal_force",
                "fit_intercept": False,
            },
        ),
        ("fixed_prior_mean_wb", {"mean_prior_retention": 1.0, "waveform_prior_retention": 1.0}),
        ("shaped_prior_mean_wb", {}),
        ("no_prior_mean_wb", {"mean_prior_retention": 0.0, "waveform_prior_retention": 0.0}),
    ],
)
def test_all_model_types_are_constructible(model_type: str, kwargs: dict[str, object]) -> None:
    _spec(model_type=model_type, **kwargs)


@pytest.mark.parametrize(
    "override",
    [
        {"model_type": "unknown"},
        {"force_component": "fy"},
        {"harmonic_order": 0},
        {"harmonic_order": 5},
        {"condition_set": "airspeed"},
        {"mean_prior_retention": -0.1},
        {"waveform_prior_retention": 1.1},
        {"mean_prior_retention": float("nan")},
        {"waveform_prior_retention": float("inf")},
        {"ridge_lambda_mean": -1.0},
        {"ridge_lambda_waveform": float("nan")},
        {"mean_weighting": "equal_sample"},
        {"waveform_weighting": "unknown"},
    ],
)
def test_invalid_specification_fields_fail(override: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _spec(**override)


def test_model_specific_invariants_fail_closed() -> None:
    with pytest.raises(ValueError, match="retention"):
        _spec(model_type="fixed_prior_mean_wb", mean_prior_retention=0.5, waveform_prior_retention=1.0)
    with pytest.raises(ValueError, match="retention"):
        _spec(model_type="no_prior_mean_wb", mean_prior_retention=0.0, waveform_prior_retention=0.25)
    with pytest.raises(ValueError, match="harmonic_order"):
        _spec(model_type="raw_prior", harmonic_order=1, condition_set="none", mean_prior_retention=None, waveform_prior_retention=None, fit_intercept=False)
    with pytest.raises(ValueError, match="retention"):
        _spec(model_type="gain_bias", harmonic_order=None, condition_set="none", mean_prior_retention=1.0, waveform_prior_retention=None)


def test_mean_and_waveform_feature_names_are_deterministic() -> None:
    cycle, waveform = _frames()
    mean_a = build_mean_design(cycle, _spec())
    mean_b = build_mean_design(cycle.loc[::-1].reset_index(drop=True), _spec())
    assert mean_a.feature_names == mean_b.feature_names == (
        "intercept",
        "alpha_mean_std",
        "flapping_frequency_mean_std",
    )

    wave_a = build_waveform_design(waveform, _spec())
    wave_b = build_waveform_design(waveform.sample(frac=1.0, random_state=7), _spec())
    assert wave_a.feature_names == wave_b.feature_names
    assert wave_a.feature_names[:6] == (
        "sin_1_phase_centered",
        "cos_1_phase_centered",
        "alpha_x_sin_1",
        "alpha_x_cos_1",
        "frequency_x_sin_1",
        "frequency_x_cos_1",
    )


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize(
    ("condition", "per_harmonic"),
    [("none", 2), ("alpha", 4), ("frequency", 4), ("alpha_frequency", 6)],
)
def test_k1_to_k4_and_condition_interactions(order: int, condition: str, per_harmonic: int) -> None:
    _, waveform = _frames()
    design = build_waveform_design(waveform, _spec(harmonic_order=order, condition_set=condition))
    assert design.values.shape == (len(waveform), order * per_harmonic)
    assert len(design.feature_names) == order * per_harmonic


def test_missing_feature_fails_and_extra_columns_do_not_change_order() -> None:
    _, waveform = _frames()
    expected = build_waveform_design(waveform, _spec()).feature_names
    with pytest.raises(ValueError, match="flapping_frequency_mean_std"):
        build_waveform_design(waveform.drop(columns="flapping_frequency_mean_std"), _spec())
    waveform["airspeed_mean_std"] = 999.0
    waveform["dynamic_pressure_mean_std"] = 999.0
    assert build_waveform_design(waveform, _spec()).feature_names == expected


def test_design_rejects_non_finite_values_without_refitting_normalization() -> None:
    cycle, _ = _frames()
    cycle.loc[0, "alpha_mean_std"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        build_mean_design(cycle, _spec())
