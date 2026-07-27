from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from system_identification.models.correction.specifications import StaticCorrectionSpec
from system_identification.training.correction.selection_specs import (
    make_mean_wb_spec,
    parse_selection_config,
)


CONFIG = Path("configs/correction/static_correction_model_selection_v1.yaml")


def test_branch_specific_condition_sets_are_independent() -> None:
    spec = make_mean_wb_spec(
        "fx",
        mean_retention=0.5,
        waveform_retention=0.5,
        mean_condition="none",
        waveform_condition="alpha_frequency",
        harmonic_order=2,
        ridge_mean=1e-4,
        ridge_waveform=1e-2,
        mean_weighting="equal_log",
        waveform_weighting="equal_date",
    )
    assert spec.mean_condition_set == "none"
    assert spec.waveform_condition_set == "alpha_frequency"
    assert spec.condition_set is None


def test_old_c2_condition_set_maps_to_both_branches() -> None:
    spec = StaticCorrectionSpec(
        model_type="fixed_prior_mean_wb",
        force_component="fx",
        harmonic_order=2,
        condition_set="alpha_frequency",
        mean_prior_retention=1.0,
        waveform_prior_retention=1.0,
    )
    assert spec.mean_condition_set == spec.waveform_condition_set == "alpha_frequency"


def test_old_c2_json_bundle_spec_loads() -> None:
    value = {
        "model_type": "no_prior_mean_wb",
        "force_component": "fz",
        "harmonic_order": 1,
        "condition_set": "frequency",
        "mean_prior_retention": 0.0,
        "waveform_prior_retention": 0.0,
    }
    loaded = StaticCorrectionSpec.from_dict(json.loads(json.dumps(value)))
    assert loaded.mean_condition_set == "frequency"
    assert loaded.waveform_condition_set == "frequency"


@pytest.mark.parametrize("field", ["mean_condition_set", "waveform_condition_set"])
def test_invalid_branch_condition_is_rejected(field: str) -> None:
    values = {
        "model_type": "no_prior_mean_wb",
        "force_component": "fx",
        "harmonic_order": 1,
        "mean_prior_retention": 0.0,
        "waveform_prior_retention": 0.0,
        field: "airspeed",
    }
    with pytest.raises(ValueError, match=field):
        StaticCorrectionSpec(**values)


def test_legacy_and_branch_condition_conflict_is_rejected() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        StaticCorrectionSpec(
            model_type="no_prior_mean_wb",
            force_component="fx",
            harmonic_order=1,
            condition_set="alpha",
            mean_condition_set="frequency",
            mean_prior_retention=0.0,
            waveform_prior_retention=0.0,
        )


def test_frozen_config_forbids_dynamic_and_q_features() -> None:
    config = parse_selection_config(yaml.safe_load(CONFIG.read_text()))
    assert {"airspeed", "dynamic_pressure", "history", "future_state"}.issubset(config.forbidden_features)


@pytest.mark.parametrize("missing", ["airspeed", "dynamic_pressure", "history", "future_state"])
def test_config_missing_forbidden_feature_fails(missing: str) -> None:
    value = yaml.safe_load(CONFIG.read_text())
    value["forbidden_features"].remove(missing)
    with pytest.raises(ValueError, match="forbidden"):
        parse_selection_config(value)


def test_non_mean_model_rejects_branch_conditions() -> None:
    with pytest.raises(ValueError, match="does not accept condition"):
        StaticCorrectionSpec(
            model_type="raw_prior",
            force_component="fx",
            fit_intercept=False,
            mean_condition_set="alpha",
        )


def test_config_partition_contract_is_train_then_validation_only() -> None:
    config = parse_selection_config(yaml.safe_load(CONFIG.read_text()))
    assert config.allowed_fit_partitions == ("train",)
    assert config.allowed_evaluation_partitions == ("validation",)


def test_config_shortlist_limits_are_frozen() -> None:
    config = parse_selection_config(yaml.safe_load(CONFIG.read_text()))
    assert config.shortlist == {
        "mean_branch_limit": 3,
        "waveform_branch_limit": 4,
        "complete_model_limit_per_component": 6,
    }


def test_shaped_prior_requires_explicit_branch_retentions() -> None:
    with pytest.raises(ValueError, match="retention must be explicit"):
        StaticCorrectionSpec(
            model_type="shaped_prior_mean_wb",
            force_component="fx",
            harmonic_order=1,
        )
