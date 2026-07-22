"""Stable weighted-ridge fitting for C2 static correction candidates."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Mapping

import numpy as np
import pandas as pd

from system_identification.models.correction.bundles import (
    ALLOWED_BUNDLE_STATUSES,
    StaticCorrectionBundle,
    compute_bundle_hash,
)
from system_identification.models.correction.features import build_mean_design, build_waveform_design
from system_identification.models.correction.specifications import MEAN_WB_TYPES, StaticCorrectionSpec
from system_identification.models.correction.static_models import RidgeDiagnostics, RidgeSolution


MEAN_WEIGHT_COLUMNS = {
    "equal_cycle": "weight_equal_cycle",
    "equal_log": "weight_equal_log",
    "equal_date": "weight_equal_date",
}
WAVEFORM_WEIGHT_COLUMNS = {
    "equal_sample": None,
    "equal_cycle": "weight_equal_cycle_sample",
    "equal_log": "weight_equal_log_sample",
    "equal_date": "weight_equal_date_sample",
}
REQUIRED_PROVENANCE = frozenset(
    {
        "correction_ready_artifact_id",
        "correction_ready_manifest_hash",
        "dataset_id",
        "dataset_hash",
        "prior_id",
        "prior_hash",
        "ratio_contract",
        "phase_contract",
        "included_partitions",
    }
)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required fitting columns: {missing}")


def _train_only(frame: pd.DataFrame, name: str) -> None:
    if len(frame) == 0:
        raise ValueError(f"{name} has zero rows")
    _require_columns(frame, ["partition"])
    partitions = set(frame["partition"].astype(str).unique())
    if partitions != {"train"}:
        raise ValueError(f"C2 fitting is train-only; {name} contains partitions={sorted(partitions)}")


def _weights(frame: pd.DataFrame, strategy: str, mapping: Mapping[str, str | None]) -> np.ndarray:
    column = mapping[strategy]
    if column is None:
        return np.ones(len(frame), dtype=np.float64)
    _require_columns(frame, [column])
    return frame[column].to_numpy(dtype=np.float64, copy=False)


def fit_weighted_ridge(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    ridge_lambda: float,
    feature_names: tuple[str, ...],
    penalize_mask: np.ndarray,
) -> RidgeSolution:
    """Solve weighted ridge by augmented least squares without a Gram inverse."""

    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    penalty = np.asarray(penalize_mask, dtype=bool).reshape(-1)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("weighted ridge received zero rows or a non-matrix design")
    if x.shape != (len(y), len(feature_names)) or x.shape[0] != len(w) or x.shape[1] != len(penalty):
        raise ValueError("weighted ridge shapes are inconsistent")
    if not math.isfinite(float(ridge_lambda)) or ridge_lambda < 0.0:
        raise ValueError("ridge_lambda must be finite and non-negative")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("weighted ridge contains non-finite design or target values")
    if not np.isfinite(w).all() or np.any(w < 0.0) or float(w.sum()) <= 0.0:
        raise ValueError("weights must be finite, non-negative, and have positive sum")

    sqrt_w = np.sqrt(w)
    weighted_x = x * sqrt_w[:, None]
    weighted_y = y * sqrt_w
    augmented_x = weighted_x
    augmented_y = weighted_y
    if ridge_lambda > 0.0 and np.any(penalty):
        regularizer = np.diag(np.sqrt(ridge_lambda) * penalty.astype(np.float64))
        augmented_x = np.vstack([weighted_x, regularizer])
        augmented_y = np.concatenate([weighted_y, np.zeros(x.shape[1], dtype=np.float64)])
    coefficients, _, _, singular_values = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)
    residual = x @ coefficients - y
    weighted_residual_norm = float(np.linalg.norm(sqrt_w * residual))
    matrix_rank = int(np.linalg.matrix_rank(weighted_x))
    if len(singular_values) == 0 or singular_values[-1] <= np.finfo(np.float64).eps * singular_values[0]:
        condition_number = float("inf")
    else:
        condition_number = float(singular_values[0] / singular_values[-1])
    diagnostics = RidgeDiagnostics(
        row_count=len(y),
        coefficient_count=len(coefficients),
        matrix_rank=matrix_rank,
        condition_number=condition_number,
        weighted_residual_norm=weighted_residual_norm,
        effective_weight_sum=float(w.sum()),
        rank_deficient=matrix_rank < x.shape[1],
        finite_checks=bool(np.isfinite(coefficients).all() and np.isfinite(weighted_residual_norm)),
    )
    if not diagnostics.finite_checks:
        raise ValueError("weighted ridge produced non-finite coefficients or residual diagnostics")
    return RidgeSolution(
        coefficients=coefficients,
        feature_names=tuple(feature_names),
        ridge_lambda=float(ridge_lambda),
        penalize_mask=tuple(bool(value) for value in penalty),
        diagnostics=diagnostics,
    )

def _checked_target(frame: pd.DataFrame, column: str) -> np.ndarray:
    _require_columns(frame, [column])
    target = frame[column].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(target).all():
        raise ValueError(f"Target column {column!r} contains non-finite values")
    return target


def _validate_normalization(normalization: Mapping[str, object]) -> dict[str, object]:
    result = {str(key): value for key, value in normalization.items()}
    for key, value in result.items():
        if isinstance(value, Mapping) and value.get("source_partition") not in {None, "train"}:
            raise ValueError(f"Normalization {key!r} was not fit on train")
    return result


def _validate_provenance(provenance: Mapping[str, object]) -> dict[str, object]:
    missing = sorted(REQUIRED_PROVENANCE - set(provenance))
    if missing:
        raise ValueError(f"Training provenance missing required fields: {missing}")
    result = dict(provenance)
    if result["included_partitions"] != ["train"]:
        raise ValueError("C2 bundle provenance included_partitions must be ['train']")
    return result


def _make_bundle(
    spec: StaticCorrectionSpec,
    mean_solution: RidgeSolution | None,
    waveform_solution: RidgeSolution | None,
    component_scale: float | None,
    normalization: Mapping[str, object],
    provenance: Mapping[str, object],
    fit_summary: Mapping[str, object],
    status: str,
) -> StaticCorrectionBundle:
    if status not in ALLOWED_BUNDLE_STATUSES:
        raise ValueError(f"Bundle status must be one of {sorted(ALLOWED_BUNDLE_STATUSES)}")
    model_id_source = {"spec": spec.to_dict(), "artifact": provenance["correction_ready_artifact_id"]}
    model_id = f"{spec.model_type}_{spec.force_component}_{compute_bundle_hash(model_id_source)[:12]}"
    provisional = StaticCorrectionBundle(
        bundle_schema_version="static_correction_bundle_v1",
        model_id=model_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        spec=spec,
        mean_solution=mean_solution,
        waveform_solution=waveform_solution,
        component_scale=component_scale,
        normalization=normalization,
        training_provenance=provenance,
        fit_summary=fit_summary,
        bundle_hash="",
    )
    return StaticCorrectionBundle(**{**provisional.__dict__, "bundle_hash": compute_bundle_hash(provisional.hash_payload())})


def fit_candidate(
    spec: StaticCorrectionSpec,
    cycle_frame: pd.DataFrame,
    waveform_frame: pd.DataFrame,
    normalization: Mapping[str, object],
    provenance: Mapping[str, object],
    *,
    status: str = "candidate",
) -> StaticCorrectionBundle:
    """Fit one specified candidate using train rows only; never selects candidates."""

    _train_only(cycle_frame, "cycle_frame")
    _train_only(waveform_frame, "waveform_frame")
    normalized = _validate_normalization(normalization)
    resolved_provenance = _validate_provenance(provenance)
    mean_solution: RidgeSolution | None = None
    waveform_solution: RidgeSolution | None = None
    component_scale: float | None = None
    component = spec.force_component

    if spec.model_type in MEAN_WB_TYPES:
        mean_design = build_mean_design(cycle_frame, spec)
        mean_target = _checked_target(cycle_frame, f"label_{component}_mean_n")
        if spec.mean_prior_retention != 0.0:
            mean_target = mean_target - float(spec.mean_prior_retention) * _checked_target(
                cycle_frame, f"prior_{component}_mean_n"
            )
        mean_solution = fit_weighted_ridge(
            mean_design.values,
            mean_target,
            _weights(cycle_frame, spec.mean_weighting, MEAN_WEIGHT_COLUMNS),
            spec.ridge_lambda_mean,
            mean_design.feature_names,
            np.array([name != "intercept" for name in mean_design.feature_names], dtype=bool),
        )

        waveform_design = build_waveform_design(waveform_frame, spec)
        waveform_target = _checked_target(waveform_frame, f"label_{component}_waveform_n")
        if spec.waveform_prior_retention != 0.0:
            waveform_target = waveform_target - float(spec.waveform_prior_retention) * _checked_target(
                waveform_frame, f"prior_{component}_waveform_n"
            )
        waveform_solution = fit_weighted_ridge(
            waveform_design.values,
            waveform_target,
            _weights(waveform_frame, spec.waveform_weighting, WAVEFORM_WEIGHT_COLUMNS),
            spec.ridge_lambda_waveform,
            waveform_design.feature_names,
            np.ones(len(waveform_design.feature_names), dtype=bool),
        )
    elif spec.model_type == "gain_bias":
        prior = _checked_target(waveform_frame, f"prior_{component}_n")
        target = _checked_target(waveform_frame, f"label_{component}_n")
        design = np.column_stack([prior, np.ones(len(prior), dtype=np.float64)])
        waveform_solution = fit_weighted_ridge(
            design,
            target,
            _weights(waveform_frame, spec.waveform_weighting, WAVEFORM_WEIGHT_COLUMNS),
            spec.ridge_lambda_waveform,
            ("prior_total_n", "intercept"),
            np.array([True, False]),
        )
    elif spec.model_type == "physical_component_scale":
        total = _checked_target(waveform_frame, f"prior_{component}_n")
        label = _checked_target(waveform_frame, f"label_{component}_n")
        normal = _checked_target(waveform_frame, f"prior_{component}_normal_component_n")
        other = _checked_target(waveform_frame, f"prior_{component}_other_component_n")
        if not np.allclose(total, normal + other, atol=1e-10, rtol=1e-10):
            raise ValueError("Authoritative prior component sum does not match total prior")
        waveform_solution = fit_weighted_ridge(
            normal[:, None],
            label - total,
            _weights(waveform_frame, spec.waveform_weighting, WAVEFORM_WEIGHT_COLUMNS),
            spec.ridge_lambda_waveform,
            ("normal_force_scale_minus_one",),
            np.array([True]),
        )
        unconstrained = 1.0 + float(waveform_solution.coefficients[0])
        constraints = dict(spec.coefficient_constraints or {})
        component_scale = float(
            np.clip(unconstrained, float(constraints.get("scale_min", 0.0)), float(constraints.get("scale_max", 2.0)))
        )

    coefficient_count = sum(
        len(solution.coefficients) for solution in (mean_solution, waveform_solution) if solution is not None
    )
    fit_summary = {
        "train_cycle_count": int(len(cycle_frame)),
        "train_waveform_row_count": int(len(waveform_frame)),
        "coefficient_count": int(coefficient_count),
        "finite_checks": True,
        "selection_performed": False,
    }
    return _make_bundle(
        spec,
        mean_solution,
        waveform_solution,
        component_scale,
        normalized,
        resolved_provenance,
        fit_summary,
        status,
    )
