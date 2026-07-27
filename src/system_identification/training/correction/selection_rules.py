"""Deterministic shortlist and one-standard-error rules for C3."""

from __future__ import annotations

from collections import Counter
import math
from typing import Mapping, Sequence

import numpy as np

from system_identification.models.correction.specifications import StaticCorrectionSpec
from system_identification.training.correction.selection_specs import canonical_hash


CONDITION_COMPLEXITY = {"none": 0, "alpha": 1, "frequency": 1, "alpha_frequency": 2}
WEIGHTING_COMPLEXITY = {"equal_cycle": 0, "equal_log": 1, "equal_date": 2, "equal_sample": 3}


def spec_complexity(spec: StaticCorrectionSpec, coefficient_count: int, condition_number: float) -> tuple[object, ...]:
    shaped_count = int(
        spec.model_type == "shaped_prior_mean_wb"
        and (spec.mean_prior_retention not in {0.0, 1.0} or spec.waveform_prior_retention not in {0.0, 1.0})
    )
    return (
        int(coefficient_count),
        int(spec.harmonic_order or 0),
        CONDITION_COMPLEXITY[str(spec.mean_condition_set)]
        + CONDITION_COMPLEXITY[str(spec.waveform_condition_set)],
        shaped_count,
        WEIGHTING_COMPLEXITY[spec.mean_weighting] + WEIGHTING_COMPLEXITY[spec.waveform_weighting],
        float(condition_number) if math.isfinite(condition_number) else float("inf"),
    )


def one_se_threshold(best_per_log_errors: Sequence[float]) -> tuple[float, float]:
    errors = np.asarray(best_per_log_errors, dtype=np.float64)
    if len(errors) == 0 or not np.isfinite(errors).all():
        raise ValueError("one-SE requires finite per-log errors")
    standard_error = float(errors.std(ddof=1) / math.sqrt(len(errors))) if len(errors) > 1 else 0.0
    return float(errors.mean() + standard_error), standard_error


def select_one_se(
    candidates: Sequence[Mapping[str, object]],
    *,
    component: str,
) -> dict[str, object]:
    if not candidates:
        raise ValueError("Cannot select from an empty candidate set")
    best = min(candidates, key=lambda item: (float(item["macro_total_rmse"]), str(item["candidate_id"])))
    threshold, standard_error = one_se_threshold(best["per_log_rmse"])
    eligible = [item for item in candidates if float(item["macro_total_rmse"]) <= threshold + 1e-12]
    secondary = (
        ("macro_waveform_rmse", "worst_log_total_rmse", "phase_bin_waveform_rmse", "peak_magnitude_error")
        if component == "fx"
        else (
            "macro_mean_rmse",
            "downstroke_integral_error_abs",
            "macro_waveform_rmse",
            "worst_log_total_rmse",
        )
    )
    selected = min(
        eligible,
        key=lambda item: (
            tuple(item["complexity"]),
            *(abs(float(item.get(name, float("inf")))) for name in secondary),
            str(item["candidate_id"]),
        ),
    )
    return {
        "selected_candidate_id": str(selected["candidate_id"]),
        "best_candidate_id": str(best["candidate_id"]),
        "best_macro_total_rmse": float(best["macro_total_rmse"]),
        "best_standard_error": standard_error,
        "one_se_threshold": threshold,
        "eligible_candidate_ids": sorted(str(item["candidate_id"]) for item in eligible),
        "selection_reason": (
            f"{component.upper()} validation primary best={float(best['macro_total_rmse']):.6f} N, "
            f"SE={standard_error:.6f} N, threshold={threshold:.6f} N; selected "
            f"{selected['candidate_id']} as the lowest-complexity eligible candidate, with frozen "
            f"{component.upper()} secondary metrics used only after the complexity ordering."
        ),
    }


def leave_one_log_out_selection(
    candidates: Sequence[Mapping[str, object]],
    *,
    component: str,
) -> dict[str, object]:
    log_ids = sorted(
        set.intersection(*(set(map(str, item["per_log_rmse_by_log"].keys())) for item in candidates))
    )
    all_result = select_one_se(candidates, component=component)
    selections: list[dict[str, object]] = []
    for omitted in log_ids:
        reduced = []
        for item in candidates:
            values = {
                str(log_id): float(value)
                for log_id, value in item["per_log_rmse_by_log"].items()
                if str(log_id) != omitted
            }
            copy = dict(item)
            copy["per_log_rmse"] = list(values.values())
            copy["macro_total_rmse"] = float(np.mean(list(values.values())))
            reduced.append(copy)
        result = select_one_se(reduced, component=component)
        selected_metric = next(
            float(item["macro_total_rmse"])
            for item in reduced
            if str(item["candidate_id"]) == str(result["selected_candidate_id"])
        )
        selections.append(
            {
                "omitted_log_id": omitted,
                "selected_candidate_id": result["selected_candidate_id"],
                "selected_macro_total_rmse": selected_metric,
                "one_se_threshold": result["one_se_threshold"],
            }
        )
    counts = Counter(str(item["selected_candidate_id"]) for item in selections)
    all_selected = str(all_result["selected_candidate_id"])
    selected_frequency = int(counts.get(all_selected, 0))
    primary_values = [float(item["selected_macro_total_rmse"]) for item in selections]
    return {
        "component": component,
        "all_log_selected_model": all_selected,
        "leave_one_log_out": selections,
        "selection_counts": dict(sorted(counts.items())),
        "all_log_selected_frequency": selected_frequency,
        "primary_metric_range": [
            min(primary_values) if primary_values else None,
            max(primary_values) if primary_values else None,
        ],
        "single_log_dominated": selected_frequency < math.ceil(len(log_ids) / 2),
        "selection_uncertainty": len(counts) > 1,
    }


def seal_shortlist(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.pop("shortlist_hash", None)
    result["shortlist_hash"] = canonical_hash(result)
    return result


def verify_sealed_shortlist(
    payload: Mapping[str, object],
    *,
    expected_config_hash: str,
    expected_artifact_hash: str,
) -> None:
    recorded = str(payload.get("shortlist_hash", ""))
    unhashed = dict(payload)
    unhashed.pop("shortlist_hash", None)
    actual = canonical_hash(unhashed)
    if recorded != actual:
        raise ValueError("Sealed shortlist hash mismatch")
    if payload.get("source_config_hash") != expected_config_hash:
        raise ValueError("Sealed shortlist source config hash mismatch")
    if payload.get("source_artifact_hash") != expected_artifact_hash:
        raise ValueError("Sealed shortlist source artifact hash mismatch")
