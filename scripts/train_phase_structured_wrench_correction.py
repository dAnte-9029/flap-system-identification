#!/usr/bin/env python3
"""Train phase-structured force and wrench-consistent moment corrections."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_deployable_wrench_correction_v2 import (  # noqa: E402
    FORCE_TARGETS,
    MOMENT_TARGETS,
    FeatureTransform,
    ForceCorrectionModel,
    RidgeMultiOutputModel,
    _append_prior_force_interactions,
    _array_from_prefixed_columns,
    _build_arm_design_matrix,
    _channel_metrics,
    _feature_transform_state,
    _fit_ridge_frame,
    _fit_transform,
    _metrics_with_candidate_columns,
    _nanmean_without_warning,
    _parse_csv_floats,
    _ridge_solve,
    _ridge_solve_multi,
    build_v2_feature_frame,
    cross_arm_force,
    dataframe_to_markdown,
    force_metrics,
    moment_metrics,
    predict_force_correction,
    v2_feature_groups,
)

DEFAULT_SPLIT_ROOT = Path("dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1")
DEFAULT_PRIOR_ROOT = Path("artifacts/delaurier_physical_prior_v1")
DEFAULT_V2_REFERENCE_ROOT = Path("artifacts/20260526_deployable_wrench_correction_v2")
DEFAULT_OUTPUT_ROOT = Path("artifacts/20260527_phase_structured_wrench_correction_v1")
PROTECTED_ARTIFACT_ROOTS = {
    Path("artifacts/20260525_delaurier_greybox_force_correction_v1"),
    Path("artifacts/20260525_dynamic_arm_moment_head_v1"),
    Path("artifacts/20260526_deployable_wrench_correction_v2"),
}
VALIDATION_TIE_TOLERANCE = 1.0e-12
FORCE_FAMILY_PREFERENCE = {
    "prior": 0,
    "slow_only": 1,
    "phase_only": 2,
    "phase_structured": 3,
    "phase_structured_plus_rates_controls": 4,
}
FORCE_VARIANT_PREFERENCE = {"additive": 0, "affine": 1, "prior": 2}
MOMENT_FORM_PREFERENCE = {
    "force_arm_plus_free": 0,
    "hybrid_prior_arm_free": 1,
    "force_arm_only": 2,
    "direct_residual": 3,
    "prior_moment": 4,
}
MOMENT_FEATURE_PREFERENCE = {"phase_structured": 0, "phase_structured_plus_rates_controls": 1, "prior": 2}

SLOW_FEATURES = (
    "alpha_rad",
    "flap_frequency_hz",
    "true_airspeed_m_s",
    "dynamic_pressure_pa",
    "alpha_rad_x_flap_frequency_hz",
    "beta_proxy_rad",
    "v_air_b_y",
    "q_dyn_x_beta_proxy",
)
PHASE_FEATURES = ("phase_sin_1", "phase_cos_1", "phase_sin_2", "phase_cos_2")
PHASE_STRUCTURED_INTERACTIONS = (
    "alpha_rad_x_phase_sin_1",
    "alpha_rad_x_phase_cos_1",
    "flap_frequency_hz_x_phase_sin_1",
    "flap_frequency_hz_x_phase_cos_1",
    "true_airspeed_m_s_x_phase_sin_1",
    "true_airspeed_m_s_x_phase_cos_1",
)
RATE_FEATURES = (
    "body_rate_p",
    "body_rate_q",
    "body_rate_r",
    "q_dyn_x_body_rate_p",
    "q_dyn_x_body_rate_q",
    "q_dyn_x_body_rate_r",
)
CONTROL_FEATURES = (
    "servo_rudder",
    "servo_left_elevon",
    "servo_right_elevon",
    "elevon_sum_proxy",
    "elevon_diff_proxy",
    "q_dyn_x_servo_rudder",
    "q_dyn_x_servo_left_elevon",
    "q_dyn_x_servo_right_elevon",
)
LATERAL_INTERACTIONS = (
    "beta_proxy_x_phase_sin_1",
    "beta_proxy_x_phase_cos_1",
)
RATE_CONTROL_INTERACTIONS = (
    "body_rate_p_x_phase_sin_1",
    "body_rate_p_x_phase_cos_1",
    "body_rate_q_x_phase_sin_1",
    "body_rate_q_x_phase_cos_1",
    "body_rate_r_x_phase_sin_1",
    "body_rate_r_x_phase_cos_1",
    "servo_rudder_x_phase_sin_1",
    "servo_rudder_x_phase_cos_1",
    "elevon_diff_x_phase_sin_1",
    "elevon_diff_x_phase_cos_1",
)
FORBIDDEN_INFERENCE_COLUMNS = {
    *FORCE_TARGETS,
    *MOMENT_TARGETS,
    *(f"label_{target}" for target in (*FORCE_TARGETS, *MOMENT_TARGETS)),
}


@dataclass
class PriorForceModel:
    family: str = "prior"
    variant: str = "prior"
    selected_features: list[str] | None = None
    uses_true_force_for_inference: bool = False

    def predict_force(self, frame: pd.DataFrame) -> np.ndarray:
        return _array_from_prefixed_columns(frame, "prior", FORCE_TARGETS)


@dataclass
class PhaseStructuredMomentModel:
    form: str
    feature_family: str
    selected_features: list[str]
    alpha: float
    feature_transform: FeatureTransform
    direct_coefficients: np.ndarray | None = None
    arm_coefficients: np.ndarray | None = None
    free_coefficients: np.ndarray | None = None
    uses_true_force_for_inference: bool = False

    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        features = build_v2_feature_frame(frame)[0].loc[:, self.selected_features]
        phi = self.feature_transform.transform(features)
        force = _array_from_prefixed_columns(frame, "force_corr", FORCE_TARGETS)
        prior_moment = _array_from_prefixed_columns(frame, "prior", MOMENT_TARGETS)
        zeros = np.zeros((len(frame), 3), dtype=float)
        if self.form == "prior_moment":
            return {"moment": prior_moment, "r_hat": zeros, "arm_moment": zeros, "tau_free": zeros}
        if self.form == "direct_residual":
            if self.direct_coefficients is None:
                raise ValueError("direct coefficients are required")
            delta = phi @ self.direct_coefficients
            return {"moment": prior_moment + delta, "r_hat": zeros, "arm_moment": zeros, "tau_free": delta}
        if self.arm_coefficients is None:
            raise ValueError("arm coefficients are required")
        r_hat = phi @ self.arm_coefficients
        arm_moment = cross_arm_force(r_hat, force)
        if self.form == "force_arm_only":
            return {"moment": arm_moment, "r_hat": r_hat, "arm_moment": arm_moment, "tau_free": zeros}
        if self.form == "force_arm_plus_free":
            if self.free_coefficients is None:
                raise ValueError("free-torque coefficients are required")
            tau_free = phi @ self.free_coefficients
            return {"moment": arm_moment + tau_free, "r_hat": r_hat, "arm_moment": arm_moment, "tau_free": tau_free}
        if self.form == "hybrid_prior_arm_free":
            if self.free_coefficients is None:
                raise ValueError("free-torque coefficients are required")
            tau_free = phi @ self.free_coefficients
            return {"moment": prior_moment + arm_moment + tau_free, "r_hat": r_hat, "arm_moment": arm_moment, "tau_free": tau_free}
        raise ValueError(f"unknown moment form: {self.form}")


def _present(columns: Iterable[str], names: Iterable[str]) -> list[str]:
    available = set(columns)
    out = [name for name in names if name in available and name not in FORBIDDEN_INFERENCE_COLUMNS and not name.startswith("label_")]
    return list(dict.fromkeys(out))


def phase_structured_force_family_specs(
    columns: Iterable[str],
    groups: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    del groups
    available = list(columns)
    slow = _present(available, SLOW_FEATURES)
    phase = _present(available, PHASE_FEATURES)
    structured = _present(available, (*SLOW_FEATURES, *PHASE_FEATURES, *PHASE_STRUCTURED_INTERACTIONS))
    plus = _present(
        available,
        (
            *SLOW_FEATURES,
            *PHASE_FEATURES,
            *PHASE_STRUCTURED_INTERACTIONS,
            *RATE_FEATURES,
            *CONTROL_FEATURES,
            *LATERAL_INTERACTIONS,
            *RATE_CONTROL_INTERACTIONS,
        ),
    )
    return {
        "prior": [],
        "slow_only": slow,
        "phase_only": phase,
        "phase_structured": structured,
        "phase_structured_plus_rates_controls": plus,
    }


def build_phase_structured_feature_families(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, object]]:
    features, spec = build_v2_feature_frame(frame)
    groups = v2_feature_groups(features.columns)
    families = phase_structured_force_family_specs(features.columns, groups)
    return features, families, {**spec, "uses_true_force": False}


def _force_design_for_family(
    features: pd.DataFrame,
    prior_force: np.ndarray,
    family_columns: list[str],
    variant: str,
) -> pd.DataFrame:
    selected = features.loc[:, [column for column in family_columns if column in features.columns]].copy()
    if variant == "additive":
        return selected
    if variant == "affine":
        return _append_prior_force_interactions(selected, prior_force)
    raise ValueError(f"unknown force variant: {variant}")


def select_force_candidate(metrics: pd.DataFrame) -> pd.Series:
    val = metrics.query("split == 'val' and target == 'force_mean'").copy()
    if val.empty:
        raise ValueError("no validation force_mean rows found")
    best = float(val["rmse"].min())
    tied = val[val["rmse"] <= best + VALIDATION_TIE_TOLERANCE].copy()
    tied["_family_order"] = tied["family"].map(FORCE_FAMILY_PREFERENCE).fillna(999)
    tied["_variant_order"] = tied["variant"].map(FORCE_VARIANT_PREFERENCE).fillna(999)
    tied = tied.sort_values(["_family_order", "_variant_order", "alpha"], ascending=[True, True, True])
    return tied.iloc[0].drop(labels=[column for column in ("_family_order", "_variant_order") if column in tied.columns])


def fit_phase_structured_force_models(
    split_frames: dict[str, pd.DataFrame],
    families: dict[str, list[str]],
    alphas: tuple[float, ...],
    variants: tuple[str, ...] = ("additive", "affine"),
) -> tuple[pd.DataFrame, dict[str, object]]:
    train = split_frames["train"]
    train_features, _, _ = build_phase_structured_feature_families(train)
    train_true = _array_from_prefixed_columns(train, "label", FORCE_TARGETS)
    train_prior = _array_from_prefixed_columns(train, "prior", FORCE_TARGETS)
    rows: list[pd.DataFrame] = []
    candidates: list[dict[str, object]] = []

    prior_model = PriorForceModel(selected_features=[])
    for split, frame in split_frames.items():
        metrics = force_metrics(
            split=split,
            model_name="prior",
            true_force=_array_from_prefixed_columns(frame, "label", FORCE_TARGETS),
            predicted_force=_array_from_prefixed_columns(frame, "prior", FORCE_TARGETS),
        )
        metrics = _force_candidate_columns(metrics, "prior", "prior", 0.0, False)
        rows.append(metrics)
    prior_val = rows[1].query("target == 'force_mean'")["rmse"].iloc[0]
    candidates.append({"model": prior_model, "family": "prior", "variant": "prior", "alpha": 0.0, "val_rmse": float(prior_val)})

    for family, columns in families.items():
        if family == "prior":
            continue
        columns = [column for column in columns if column in train_features.columns]
        if not columns:
            continue
        for variant in variants:
            for alpha in alphas:
                design = _force_design_for_family(train_features, train_prior, columns, variant)
                ridge = _fit_ridge_frame(design, train_true - train_prior, float(alpha))
                model = ForceCorrectionModel(variant=variant, feature_group=family, selected_features=columns, ridge=ridge)
                val_metrics: pd.DataFrame | None = None
                for split, frame in split_frames.items():
                    pred = model.predict_force(frame)
                    metrics = force_metrics(
                        split=split,
                        model_name=f"{family}_{variant}_alpha_{alpha:g}",
                        true_force=_array_from_prefixed_columns(frame, "label", FORCE_TARGETS),
                        predicted_force=pred,
                    )
                    metrics = _force_candidate_columns(metrics, family, variant, float(alpha), False)
                    rows.append(metrics)
                    if split == "val":
                        val_metrics = metrics
                if val_metrics is None:
                    raise RuntimeError("validation split is required")
                val_rmse = float(val_metrics.query("target == 'force_mean'")["rmse"].iloc[0])
                candidates.append({"model": model, "family": family, "variant": variant, "alpha": float(alpha), "val_rmse": val_rmse})

    all_metrics = pd.concat(rows, ignore_index=True)
    selection_rows = pd.DataFrame(
        [
            {
                "split": "val",
                "target": "force_mean",
                "family": item["family"],
                "variant": item["variant"],
                "alpha": float(item["alpha"]),
                "rmse": float(item["val_rmse"]),
            }
            for item in candidates
        ]
    )
    selected_row = select_force_candidate(selection_rows)
    selected_candidate = next(
        item
        for item in candidates
        if item["family"] == selected_row["family"]
        and item["variant"] == selected_row["variant"]
        and np.isclose(float(item["alpha"]), float(selected_row["alpha"]))
    )
    selected_mask = (
        all_metrics["family"].eq(selected_candidate["family"])
        & all_metrics["variant"].eq(selected_candidate["variant"])
        & np.isclose(all_metrics["alpha"].astype(float), float(selected_candidate["alpha"]))
    )
    all_metrics.loc[selected_mask, "is_selected"] = True
    selected = {
        **{key: selected_candidate[key] for key in ("family", "variant", "alpha", "model")},
        "selection_split": "val",
        "selection_metric": "force_mean_rmse",
        "validation_tie_policy": "within 1e-12 RMSE, prefer smaller predeclared family, additive before affine, then smaller alpha",
        "uses_true_force_for_inference": False,
    }
    return all_metrics, selected


def _force_candidate_columns(metrics: pd.DataFrame, family: str, variant: str, alpha: float, selected: bool) -> pd.DataFrame:
    out = _metrics_with_candidate_columns(metrics, variant, family, alpha, variant, selected)
    out["family"] = family
    return out


def select_moment_candidate(metrics: pd.DataFrame) -> pd.Series:
    val = metrics.query("split == 'val' and target == 'moment_mean'").copy()
    if val.empty:
        raise ValueError("no validation moment_mean rows found")
    best = float(val["rmse"].min())
    tied = val[val["rmse"] <= best + VALIDATION_TIE_TOLERANCE].copy()
    tied["_form_order"] = tied["form"].map(MOMENT_FORM_PREFERENCE).fillna(999)
    tied["_feature_order"] = tied["feature_family"].map(MOMENT_FEATURE_PREFERENCE).fillna(999)
    tied = tied.sort_values(["_form_order", "_feature_order", "alpha"], ascending=[True, True, False])
    return tied.iloc[0].drop(labels=[column for column in ("_form_order", "_feature_order") if column in tied.columns])


def _moment_feature_families(families: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        "phase_structured": families.get("phase_structured", []),
        "phase_structured_plus_rates_controls": families.get("phase_structured_plus_rates_controls", []),
    }


def _build_free_torque_design_matrix(phi: np.ndarray) -> np.ndarray:
    n, p = phi.shape
    design = np.zeros((n, 3, p * 3), dtype=float)
    for idx in range(p):
        for axis in range(3):
            design[:, axis, idx * 3 + axis] = phi[:, idx]
    return design.reshape(n * 3, p * 3)


def _fit_joint_arm_free_coefficients(
    phi: np.ndarray,
    force: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    p = phi.shape[1]
    design = np.hstack([_build_arm_design_matrix(phi, force), _build_free_torque_design_matrix(phi)])
    coefficients = _ridge_solve(design, target, alpha)
    arm_coeff = coefficients[: p * 3].reshape(p, 3)
    free_coeff = coefficients[p * 3 :].reshape(p, 3)
    return arm_coeff, free_coeff


def _fit_moment_model(frame: pd.DataFrame, columns: list[str], form: str, alpha: float) -> PhaseStructuredMomentModel:
    features = build_v2_feature_frame(frame)[0].loc[:, columns]
    transform = _fit_transform(features, columns)
    phi = transform.transform(features)
    y = _array_from_prefixed_columns(frame, "label", MOMENT_TARGETS)
    prior = _array_from_prefixed_columns(frame, "prior", MOMENT_TARGETS)
    force = _array_from_prefixed_columns(frame, "force_corr", FORCE_TARGETS)
    if form == "direct_residual":
        coeff = _ridge_solve_multi(phi, y - prior, alpha)
        return PhaseStructuredMomentModel(form, "", columns, float(alpha), transform, direct_coefficients=coeff)
    if form == "force_arm_only":
        arm_coeff = _ridge_solve(_build_arm_design_matrix(phi, force), y, alpha).reshape(phi.shape[1], 3)
        return PhaseStructuredMomentModel(form, "", columns, float(alpha), transform, arm_coefficients=arm_coeff)
    if form == "force_arm_plus_free":
        arm_coeff, free_coeff = _fit_joint_arm_free_coefficients(phi, force, y, alpha)
        return PhaseStructuredMomentModel(form, "", columns, float(alpha), transform, arm_coefficients=arm_coeff, free_coefficients=free_coeff)
    if form == "hybrid_prior_arm_free":
        residual = y - prior
        arm_coeff, free_coeff = _fit_joint_arm_free_coefficients(phi, force, residual, alpha)
        return PhaseStructuredMomentModel(form, "", columns, float(alpha), transform, arm_coefficients=arm_coeff, free_coefficients=free_coeff)
    raise ValueError(f"unknown moment form: {form}")


def fit_phase_structured_moment_models(
    split_frames: dict[str, pd.DataFrame],
    selected_force_model_or_families: object | None = None,
    families: dict[str, list[str]] | None = None,
    alphas: tuple[float, ...] = (0.0, 1.0),
    forms: tuple[str, ...] = ("direct_residual", "force_arm_only", "force_arm_plus_free", "hybrid_prior_arm_free"),
) -> tuple[pd.DataFrame, dict[str, object]]:
    if families is None and isinstance(selected_force_model_or_families, dict):
        families = selected_force_model_or_families
    if families is None:
        families = phase_structured_force_family_specs(build_v2_feature_frame(split_frames["train"])[0].columns, {})
    train = split_frames["train"]
    train_features = build_v2_feature_frame(train)[0]
    rows: list[pd.DataFrame] = []
    candidates: list[dict[str, object]] = []

    for split, frame in split_frames.items():
        metrics = moment_metrics(
            split=split,
            model_name="prior_moment",
            y_true=_array_from_prefixed_columns(frame, "label", MOMENT_TARGETS),
            y_pred=_array_from_prefixed_columns(frame, "prior", MOMENT_TARGETS),
        )
        rows.append(_moment_candidate_columns(metrics, "prior_moment", "prior", 0.0, False))
    prior_val = rows[1].query("target == 'moment_mean'")["rmse"].iloc[0]
    candidates.append(
        {
            "model": PhaseStructuredMomentModel("prior_moment", "prior", [], 0.0, FeatureTransform([], [], [], [])),
            "form": "prior_moment",
            "feature_family": "prior",
            "alpha": 0.0,
            "val_rmse": float(prior_val),
        }
    )

    for feature_family, columns in _moment_feature_families(families).items():
        columns = [column for column in columns if column in train_features.columns]
        if not columns:
            continue
        for form in forms:
            for alpha in alphas:
                model = _fit_moment_model(train, columns, form, float(alpha))
                model.feature_family = feature_family
                val_metrics: pd.DataFrame | None = None
                for split, frame in split_frames.items():
                    pred = model.predict(frame)["moment"]
                    metrics = moment_metrics(
                        split=split,
                        model_name=f"{form}_{feature_family}_alpha_{alpha:g}",
                        y_true=_array_from_prefixed_columns(frame, "label", MOMENT_TARGETS),
                        y_pred=pred,
                    )
                    metrics = _moment_candidate_columns(metrics, form, feature_family, float(alpha), False)
                    rows.append(metrics)
                    if split == "val":
                        val_metrics = metrics
                if val_metrics is None:
                    raise RuntimeError("validation split is required")
                candidates.append(
                    {
                        "model": model,
                        "form": form,
                        "feature_family": feature_family,
                        "alpha": float(alpha),
                        "val_rmse": float(val_metrics.query("target == 'moment_mean'")["rmse"].iloc[0]),
                    }
                )

    all_metrics = pd.concat(rows, ignore_index=True)
    selection_rows = pd.DataFrame(
        [
            {
                "split": "val",
                "target": "moment_mean",
                "form": item["form"],
                "feature_family": item["feature_family"],
                "alpha": float(item["alpha"]),
                "rmse": float(item["val_rmse"]),
            }
            for item in candidates
        ]
    )
    selected_row = select_moment_candidate(selection_rows)
    selected_candidate = next(
        item
        for item in candidates
        if item["form"] == selected_row["form"]
        and item["feature_family"] == selected_row["feature_family"]
        and np.isclose(float(item["alpha"]), float(selected_row["alpha"]))
    )
    selected_mask = (
        all_metrics["form"].eq(selected_candidate["form"])
        & all_metrics["feature_family"].eq(selected_candidate["feature_family"])
        & np.isclose(all_metrics["alpha"].astype(float), float(selected_candidate["alpha"]))
    )
    all_metrics.loc[selected_mask, "is_selected"] = True
    selected = {
        **{key: selected_candidate[key] for key in ("form", "feature_family", "alpha", "model")},
        "selection_split": "val",
        "selection_metric": "moment_mean_rmse",
        "validation_tie_policy": "within 1e-12 RMSE, prefer force_arm_plus_free, hybrid, force_arm_only, direct; then smaller feature family; then larger alpha",
        "uses_true_force_for_inference": False,
    }
    return all_metrics, selected


def _moment_candidate_columns(metrics: pd.DataFrame, form: str, feature_family: str, alpha: float, selected: bool) -> pd.DataFrame:
    out = _metrics_with_candidate_columns(metrics, form, feature_family, alpha, form, selected)
    out["form"] = form
    out["feature_family"] = feature_family
    return out


def _prediction_path(root: Path, split: str) -> Path:
    for path in (root / f"{split}_predictions.parquet", root / "prediction_parquets" / f"{split}_predictions.parquet"):
        if path.exists():
            return path
    raise FileNotFoundError(f"could not find {split}_predictions.parquet under {root}")


def _load_split(split: str, split_root: Path, prior_root: Path) -> pd.DataFrame:
    samples = pd.read_parquet(split_root / f"{split}_samples.parquet")
    prior = pd.read_parquet(prior_root / f"{split}_predictions.parquet")
    if len(samples) != len(prior):
        raise ValueError(f"{split} row mismatch: samples={len(samples)} prior={len(prior)}")
    features, _ = build_v2_feature_frame(samples)
    frame = features.copy()
    for column in ("timestamp_us", "time_s", "log_id", "segment_id", "cycle_id", "phase_corrected_rad"):
        if column in samples.columns:
            frame[column] = samples[column].to_numpy()
    frame["split"] = split
    for target in FORCE_TARGETS:
        frame[f"label_{target}"] = samples[target].to_numpy(dtype=float)
        frame[f"prior_{target}"] = prior[target].to_numpy(dtype=float)
    for target in MOMENT_TARGETS:
        frame[f"label_{target}"] = samples[target].to_numpy(dtype=float)
        frame[f"prior_{target}"] = prior[target].to_numpy(dtype=float)
    return frame


def _assert_safe_output_root(output_root: Path, overwrite: bool) -> None:
    resolved = output_root.resolve()
    protected = {path.resolve() for path in PROTECTED_ARTIFACT_ROOTS}
    if resolved in protected:
        raise ValueError(f"refusing to write to protected baseline artifact root: {output_root}")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise ValueError(f"refusing to write to non-empty output root without --overwrite: {output_root}")


def _augment_with_force(frame: pd.DataFrame, force_model: object) -> pd.DataFrame:
    force = predict_force_correction(force_model, frame) if hasattr(force_model, "ridge") else force_model.predict_force(frame)
    out = frame.copy()
    for idx, target in enumerate(FORCE_TARGETS):
        out[f"force_corr_{target}"] = force[:, idx]
    return out


def _write_predictions(
    output_root: Path,
    split_frames: dict[str, pd.DataFrame],
    force_model: object,
    moment_model: PhaseStructuredMomentModel,
) -> dict[str, Path]:
    prediction_dir = output_root / "prediction_parquets"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    metadata_columns = ("timestamp_us", "time_s", "log_id", "segment_id", "cycle_id", "phase_corrected_rad", "split")
    for split, frame in split_frames.items():
        augmented = _augment_with_force(frame, force_model)
        moment_pred = moment_model.predict(augmented)
        out = frame.loc[:, [column for column in metadata_columns if column in frame.columns]].copy()
        for idx, target in enumerate(FORCE_TARGETS):
            out[f"label_{target}"] = frame[f"label_{target}"].to_numpy(dtype=float)
            out[f"prior_{target}"] = frame[f"prior_{target}"].to_numpy(dtype=float)
            out[f"force_corr_{target}"] = augmented[f"force_corr_{target}"].to_numpy(dtype=float)
            out[f"force_corr_residual_{target}"] = out[f"label_{target}"] - out[f"force_corr_{target}"]
        for idx, target in enumerate(MOMENT_TARGETS):
            out[f"label_{target}"] = frame[f"label_{target}"].to_numpy(dtype=float)
            out[f"prior_{target}"] = frame[f"prior_{target}"].to_numpy(dtype=float)
            out[f"moment_corr_{target}"] = moment_pred["moment"][:, idx]
            out[f"moment_corr_residual_{target}"] = out[f"label_{target}"] - out[f"moment_corr_{target}"]
            out[f"arm_moment_{target}"] = moment_pred["arm_moment"][:, idx]
            out[f"tau_free_{target}"] = moment_pred["tau_free"][:, idx]
        for idx, axis in enumerate(("x", "y", "z")):
            out[f"r_hat_{axis}"] = moment_pred["r_hat"][:, idx]
        path = prediction_dir / f"{split}_predictions.parquet"
        out.to_parquet(path, index=False)
        paths[split] = path
    return paths


def _per_log_metrics(split_frames: dict[str, pd.DataFrame], force_model: object, moment_model: PhaseStructuredMomentModel) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, frame in split_frames.items():
        log_ids = frame["log_id"].to_numpy() if "log_id" in frame.columns else np.full(len(frame), "unknown")
        augmented = _augment_with_force(frame, force_model)
        force_corr = _array_from_prefixed_columns(augmented, "force_corr", FORCE_TARGETS)
        moment_corr = moment_model.predict(augmented)["moment"]
        for log_id in pd.unique(log_ids):
            mask = log_ids == log_id
            force_mean = force_metrics(
                split=split,
                model_name="phase_structured_force",
                true_force=_array_from_prefixed_columns(frame.iloc[mask], "label", FORCE_TARGETS),
                predicted_force=force_corr[mask],
            ).query("target == 'force_mean'").iloc[0]
            moment_mean = moment_metrics(
                split=split,
                model_name="phase_structured_moment",
                y_true=_array_from_prefixed_columns(frame.iloc[mask], "label", MOMENT_TARGETS),
                y_pred=moment_corr[mask],
            ).query("target == 'moment_mean'").iloc[0]
            rows.append(
                {
                    "split": split,
                    "log_id": log_id,
                    "n": int(mask.sum()),
                    "force_rmse": float(force_mean["rmse"]),
                    "moment_rmse": float(moment_mean["rmse"]),
                }
            )
    return pd.DataFrame(rows)


def _rhat_tau_summary(test_predictions: pd.DataFrame) -> dict[str, float]:
    r_columns = ["r_hat_x", "r_hat_y", "r_hat_z"]
    tau_columns = [f"tau_free_{target}" for target in MOMENT_TARGETS]
    moment_columns = [f"moment_corr_{target}" for target in MOMENT_TARGETS]
    out: dict[str, float] = {}
    if set(r_columns).issubset(test_predictions.columns):
        r = test_predictions.loc[:, r_columns].to_numpy(dtype=float)
        mag = np.linalg.norm(r, axis=1)
        out.update({"r_hat_mag_mean": float(np.nanmean(mag)), "r_hat_mag_p95": float(np.nanpercentile(mag, 95))})
    if set(tau_columns).issubset(test_predictions.columns) and set(moment_columns).issubset(test_predictions.columns):
        tau = np.linalg.norm(test_predictions.loc[:, tau_columns].to_numpy(dtype=float), axis=1)
        moment = np.linalg.norm(test_predictions.loc[:, moment_columns].to_numpy(dtype=float), axis=1)
        fraction = tau / np.maximum(moment, 1.0e-12)
        out.update({"tau_free_fraction_mean": float(np.nanmean(fraction)), "tau_free_fraction_p95": float(np.nanpercentile(fraction, 95))})
    return out


def _array_to_json(array: np.ndarray | list[float] | None) -> list[object]:
    if array is None:
        return []
    return np.asarray(array).tolist()


def _write_inference_model_state(
    output_root: Path,
    force_model: object,
    moment_model: PhaseStructuredMomentModel,
    split_root: Path,
    prior_root: Path,
    required_features: list[str],
) -> Path:
    if isinstance(force_model, ForceCorrectionModel):
        force_state = {
            "family": force_model.feature_group,
            "variant": force_model.variant,
            "selected_features": force_model.selected_features,
            "ridge_feature_columns": force_model.ridge.feature_columns,
            "normalization": {
                "feature_fill": _array_to_json(force_model.ridge.feature_fill),
                "feature_mean": _array_to_json(force_model.ridge.feature_mean),
                "feature_scale": _array_to_json(force_model.ridge.feature_scale),
            },
            "coefficients": _array_to_json(force_model.ridge.coefficients),
            "intercept": _array_to_json(force_model.ridge.intercept),
            "alpha": float(force_model.ridge.alpha),
        }
    else:
        force_state = {"family": "prior", "variant": "prior", "selected_features": [], "normalization": {}, "coefficients": [], "alpha": 0.0}
    state = {
        "uses_true_force_for_inference": False,
        "selection_protocol": "validation metrics select models; test metrics are final reporting only",
        "split_root": str(split_root),
        "prior_root": str(prior_root),
        "required_feature_columns": required_features,
        "force_model": force_state,
        "moment_model": {
            "form": moment_model.form,
            "feature_family": moment_model.feature_family,
            "selected_features": moment_model.selected_features,
            "alpha": float(moment_model.alpha),
            "normalization": _feature_transform_state(moment_model.feature_transform),
            "direct_coefficients": _array_to_json(moment_model.direct_coefficients),
            "arm_coefficients": _array_to_json(moment_model.arm_coefficients),
            "free_coefficients": _array_to_json(moment_model.free_coefficients),
        },
    }
    path = output_root / "inference_model_state.json"
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_readme(
    output_root: Path,
    command: str,
    selected_force: dict[str, object],
    selected_moment: dict[str, object],
    force_metrics_table: pd.DataFrame,
    moment_metrics_table: pd.DataFrame,
) -> None:
    force_test = force_metrics_table.query("split == 'test' and is_selected and target in ['fx_b', 'fy_b', 'fz_b', 'force_mean']").loc[
        :, ["target", "rmse", "mae", "bias", "r2"]
    ]
    moment_test = moment_metrics_table.query("split == 'test' and is_selected and target in ['mx_b', 'my_b', 'mz_b', 'moment_mean']").loc[
        :, ["target", "rmse", "mae", "bias", "r2"]
    ]
    lines = [
        "# Phase-Structured Wrench Correction",
        "",
        "This artifact trains validation-selected phase-structured force correction and wrench-consistent moment correction. No inference feature uses true force, target wrench, future samples, or acceleration targets.",
        "Model selection uses validation RMSE only; test metrics are final reporting.",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Selected force candidate",
        "",
        f"- Family: `{selected_force['family']}`",
        f"- Variant: `{selected_force['variant']}`",
        f"- Alpha: `{float(selected_force['alpha']):g}`",
        "",
        dataframe_to_markdown(force_test.reset_index(drop=True)),
        "",
        "## Selected moment candidate",
        "",
        f"- Form: `{selected_moment['form']}`",
        f"- Feature family: `{selected_moment['feature_family']}`",
        f"- Alpha: `{float(selected_moment['alpha']):g}`",
        "",
        dataframe_to_markdown(moment_test.reset_index(drop=True)),
        "",
        "## Outputs",
        "",
        "- `force_metrics_by_split.csv`, `moment_metrics_by_split.csv`: train/val/test metrics by candidate.",
        "- `force_model_selection.csv`, `moment_model_selection.csv`: validation-only selection rows.",
        "- `prediction_parquets/`: aligned predictions and effective arm/free-torque components.",
        "- `inference_model_state.json`: coefficients and normalization state for deployable inference.",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase_structured_experiment(
    *,
    split_root: Path = DEFAULT_SPLIT_ROOT,
    prior_root: Path = DEFAULT_PRIOR_ROOT,
    v2_reference_root: Path = DEFAULT_V2_REFERENCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    alphas: tuple[float, ...] = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
    overwrite: bool = False,
    command: str = "",
) -> dict[str, object]:
    _assert_safe_output_root(output_root, overwrite)
    _prediction_path(v2_reference_root, "test")
    output_root.mkdir(parents=True, exist_ok=True)
    split_frames = {split: _load_split(split, split_root, prior_root) for split in ("train", "val", "test")}
    _, families, feature_spec = build_phase_structured_feature_families(split_frames["train"])

    force_metrics_table, selected_force = fit_phase_structured_force_models(split_frames, families, alphas, variants=("additive", "affine"))
    force_model = selected_force["model"]
    moment_frames = {split: _augment_with_force(frame, force_model) for split, frame in split_frames.items()}
    moment_metrics_table, selected_moment = fit_phase_structured_moment_models(moment_frames, families=families, alphas=alphas)
    moment_model = selected_moment["model"]

    prediction_paths = _write_predictions(output_root, split_frames, force_model, moment_model)
    per_log = _per_log_metrics(split_frames, force_model, moment_model)
    test_predictions = pd.read_parquet(prediction_paths["test"])
    summaries = _rhat_tau_summary(test_predictions)

    force_selection = force_metrics_table.query("split == 'val' and target == 'force_mean'").sort_values("rmse")
    moment_selection = moment_metrics_table.query("split == 'val' and target == 'moment_mean'").sort_values("rmse")
    force_metrics_table.to_csv(output_root / "force_metrics_by_split.csv", index=False)
    moment_metrics_table.to_csv(output_root / "moment_metrics_by_split.csv", index=False)
    force_selection.to_csv(output_root / "force_model_selection.csv", index=False)
    moment_selection.to_csv(output_root / "moment_model_selection.csv", index=False)
    per_log.to_csv(output_root / "per_log_metrics.csv", index=False)

    required_features = list(
        dict.fromkeys(
            [
                *(force_model.selected_features if isinstance(force_model, ForceCorrectionModel) else []),
                *moment_model.selected_features,
            ]
        )
    )
    inference_model_state_path = _write_inference_model_state(output_root, force_model, moment_model, split_root, prior_root, required_features)
    config = {
        "split_root": str(split_root),
        "prior_root": str(prior_root),
        "v2_reference_root": str(v2_reference_root),
        "output_root": str(output_root),
        "uses_true_force_for_inference": False,
        "selection_split": "val",
        "test_metrics_final_reporting_only": True,
        "selected": {
            "force": {
                "family": selected_force["family"],
                "variant": selected_force["variant"],
                "alpha": float(selected_force["alpha"]),
            },
            "moment": {
                "form": selected_moment["form"],
                "feature_family": selected_moment["feature_family"],
                "alpha": float(selected_moment["alpha"]),
            },
        },
        "feature_sources": feature_spec.get("feature_sources", {}),
        "required_feature_columns": required_features,
        "summary": summaries,
        "inference_model_state": str(inference_model_state_path),
        "command": command,
    }
    (output_root / "model_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    _write_readme(output_root, command, selected_force, selected_moment, force_metrics_table, moment_metrics_table)
    return {
        "output_root": output_root,
        "force_metrics": output_root / "force_metrics_by_split.csv",
        "moment_metrics": output_root / "moment_metrics_by_split.csv",
        "test_predictions": prediction_paths["test"],
        "selected_force": selected_force,
        "selected_moment": selected_moment,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--prior-root", type=Path, default=DEFAULT_PRIOR_ROOT)
    parser.add_argument("--v2-reference-root", type=Path, default=DEFAULT_V2_REFERENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--alphas", default="0,0.001,0.01,0.1,1,10,100,1000")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    command = (
        "python scripts/train_phase_structured_wrench_correction.py "
        f"--split-root {args.split_root} --prior-root {args.prior_root} "
        f"--v2-reference-root {args.v2_reference_root} --output-root {args.output_root} "
        f"--alphas {args.alphas}"
    )
    if args.overwrite:
        command += " --overwrite"
    try:
        outputs = run_phase_structured_experiment(
            split_root=args.split_root,
            prior_root=args.prior_root,
            v2_reference_root=args.v2_reference_root,
            output_root=args.output_root,
            alphas=_parse_csv_floats(args.alphas),
            overwrite=args.overwrite,
            command=command,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    printable = {key: str(value) for key, value in outputs.items() if key not in {"selected_force", "selected_moment"}}
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
