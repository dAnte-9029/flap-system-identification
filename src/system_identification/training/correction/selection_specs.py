"""Frozen configuration and model-spec helpers for C3 static selection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

from system_identification.models.correction.specifications import (
    CONDITION_SETS,
    MEAN_WEIGHTINGS,
    WAVEFORM_WEIGHTINGS,
    StaticCorrectionSpec,
)


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sequence(value: object, name: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return tuple(value)


@dataclass(frozen=True)
class StaticSelectionConfig:
    schema_version: str
    input: Mapping[str, object]
    authority: Mapping[str, object]
    force_components: tuple[str, ...]
    train_cv: Mapping[str, object]
    mean_search: Mapping[str, tuple[object, ...]]
    waveform_search: Mapping[str, tuple[object, ...]]
    weighting_refinement: Mapping[str, tuple[str, ...]]
    shortlist: Mapping[str, int]
    forbidden_features: tuple[str, ...]
    allowed_fit_partitions: tuple[str, ...]
    allowed_evaluation_partitions: tuple[str, ...]
    raw: Mapping[str, object]

    @property
    def config_hash(self) -> str:
        return canonical_hash(self.raw)


def _finite_nonnegative(values: tuple[object, ...], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        raise ValueError(f"{name} must contain finite non-negative values")
    return result


def parse_selection_config(value: Mapping[str, object]) -> StaticSelectionConfig:
    if value.get("schema_version") != "static_correction_selection_v1":
        raise ValueError("Unsupported C3 selection schema_version")
    force_components = tuple(str(item) for item in _sequence(value.get("force_components"), "force_components"))
    if force_components != ("fx", "fz"):
        raise ValueError("force_components must be exactly [fx, fz]")
    train_cv = value.get("train_cv")
    mean = value.get("mean_search")
    waveform = value.get("waveform_search")
    refinement = value.get("weighting_refinement")
    limits = value.get("shortlist")
    authority = value.get("authority")
    input_value = value.get("input")
    for name, item in (
        ("train_cv", train_cv),
        ("mean_search", mean),
        ("waveform_search", waveform),
        ("weighting_refinement", refinement),
        ("shortlist", limits),
        ("authority", authority),
        ("input", input_value),
    ):
        if not isinstance(item, Mapping):
            raise ValueError(f"{name} must be a mapping")
    if int(train_cv.get("folds", 0)) != 5 or train_cv.get("group_column") != "log_id":
        raise ValueError("C3 train_cv requires five log-grouped folds")
    if train_cv.get("primary_aggregation") != "per_log_macro":
        raise ValueError("C3 primary aggregation must be per_log_macro")

    mean_parsed = {
        "prior_retention": tuple(float(v) for v in _sequence(mean.get("prior_retention"), "mean retention")),
        "condition_sets": tuple(str(v) for v in _sequence(mean.get("condition_sets"), "mean conditions")),
        "ridge_values": _finite_nonnegative(_sequence(mean.get("ridge_values"), "mean ridge"), "mean ridge"),
        "initial_weighting": tuple(str(v) for v in _sequence(mean.get("initial_weighting"), "mean weighting")),
    }
    waveform_parsed = {
        "prior_retention": tuple(
            float(v) for v in _sequence(waveform.get("prior_retention"), "waveform retention")
        ),
        "harmonic_orders": tuple(
            int(v) for v in _sequence(waveform.get("harmonic_orders"), "waveform harmonic orders")
        ),
        "condition_sets": tuple(
            str(v) for v in _sequence(waveform.get("condition_sets"), "waveform conditions")
        ),
        "ridge_values": _finite_nonnegative(
            _sequence(waveform.get("ridge_values"), "waveform ridge"), "waveform ridge"
        ),
        "initial_weighting": tuple(
            str(v) for v in _sequence(waveform.get("initial_weighting"), "waveform weighting")
        ),
    }
    expected_retention = {0.0, 0.25, 0.5, 0.75, 1.0}
    if set(mean_parsed["prior_retention"]) != expected_retention:
        raise ValueError("mean retention grid must be the frozen five-value grid")
    if set(waveform_parsed["prior_retention"]) != expected_retention:
        raise ValueError("waveform retention grid must be the frozen five-value grid")
    if set(mean_parsed["condition_sets"]) != CONDITION_SETS:
        raise ValueError("mean condition grid must contain the four allowed condition sets")
    if set(waveform_parsed["condition_sets"]) != CONDITION_SETS:
        raise ValueError("waveform condition grid must contain the four allowed condition sets")
    if set(waveform_parsed["harmonic_orders"]) != {1, 2, 3, 4}:
        raise ValueError("waveform harmonic grid must contain K=1..4")
    if mean_parsed["initial_weighting"] != ("equal_log",):
        raise ValueError("mean first pass must use equal_log only")
    if waveform_parsed["initial_weighting"] != ("equal_log",):
        raise ValueError("waveform first pass must use equal_log only")

    refinement_parsed = {
        "mean": tuple(str(v) for v in _sequence(refinement.get("mean"), "mean refinement")),
        "waveform": tuple(str(v) for v in _sequence(refinement.get("waveform"), "waveform refinement")),
    }
    if not set(refinement_parsed["mean"]).issubset(MEAN_WEIGHTINGS):
        raise ValueError("Invalid mean weighting refinement")
    if not set(refinement_parsed["waveform"]).issubset(WAVEFORM_WEIGHTINGS - {"equal_sample"}):
        raise ValueError("Invalid waveform weighting refinement")
    forbidden = tuple(str(v) for v in _sequence(value.get("forbidden_features"), "forbidden_features"))
    required_forbidden = {"airspeed", "dynamic_pressure", "history", "future_state"}
    if not required_forbidden.issubset(forbidden):
        raise ValueError("C3 forbidden feature contract is incomplete")
    fit_partitions = tuple(
        str(v) for v in _sequence(value.get("allowed_fit_partitions"), "allowed_fit_partitions")
    )
    eval_partitions = tuple(
        str(v) for v in _sequence(value.get("allowed_evaluation_partitions"), "allowed_evaluation_partitions")
    )
    if fit_partitions != ("train",) or eval_partitions != ("validation",):
        raise ValueError("C3 partition policy must be train-only fit and validation-only evaluation")
    parsed_limits = {str(key): int(item) for key, item in limits.items()}
    if parsed_limits != {
        "mean_branch_limit": 3,
        "waveform_branch_limit": 4,
        "complete_model_limit_per_component": 6,
    }:
        raise ValueError("C3 shortlist limits differ from the frozen protocol")
    return StaticSelectionConfig(
        schema_version=str(value["schema_version"]),
        input=dict(input_value),
        authority=dict(authority),
        force_components=force_components,
        train_cv=dict(train_cv),
        mean_search=mean_parsed,
        waveform_search=waveform_parsed,
        weighting_refinement=refinement_parsed,
        shortlist=parsed_limits,
        forbidden_features=forbidden,
        allowed_fit_partitions=fit_partitions,
        allowed_evaluation_partitions=eval_partitions,
        raw=dict(value),
    )


def model_type_for_retention(mean_retention: float, waveform_retention: float) -> str:
    if mean_retention == 0.0 and waveform_retention == 0.0:
        return "no_prior_mean_wb"
    if mean_retention == 1.0 and waveform_retention == 1.0:
        return "fixed_prior_mean_wb"
    return "shaped_prior_mean_wb"


def make_mean_wb_spec(
    component: str,
    *,
    mean_retention: float,
    waveform_retention: float,
    mean_condition: str,
    waveform_condition: str,
    harmonic_order: int,
    ridge_mean: float,
    ridge_waveform: float,
    mean_weighting: str,
    waveform_weighting: str,
) -> StaticCorrectionSpec:
    return StaticCorrectionSpec(
        model_type=model_type_for_retention(mean_retention, waveform_retention),
        force_component=component,
        harmonic_order=harmonic_order,
        mean_condition_set=mean_condition,
        waveform_condition_set=waveform_condition,
        mean_prior_retention=mean_retention,
        waveform_prior_retention=waveform_retention,
        ridge_lambda_mean=ridge_mean,
        ridge_lambda_waveform=ridge_waveform,
        mean_weighting=mean_weighting,
        waveform_weighting=waveform_weighting,
    )
