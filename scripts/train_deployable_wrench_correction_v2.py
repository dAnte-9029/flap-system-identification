#!/usr/bin/env python3
"""Train deployable v2 force and moment wrench corrections."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FORCE_TARGETS = ("fx_b", "fy_b", "fz_b")
MOMENT_TARGETS = ("mx_b", "my_b", "mz_b")
DEFAULT_SPLIT_ROOT = Path("dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1")
DEFAULT_PRIOR_ROOT = Path("artifacts/delaurier_physical_prior_v1")
DEFAULT_FORCE_V1_ROOT = Path("artifacts/20260525_delaurier_greybox_force_correction_v1")
DEFAULT_MOMENT_V1_ROOT = Path("artifacts/20260525_dynamic_arm_moment_head_v1")
DEFAULT_OUTPUT_ROOT = Path("artifacts/20260526_deployable_wrench_correction_v2")
PROTECTED_ARTIFACT_ROOTS = {
    DEFAULT_PRIOR_ROOT,
    DEFAULT_FORCE_V1_ROOT,
    DEFAULT_MOMENT_V1_ROOT,
    Path("artifacts/20260525_delaurier_force_recalibration_v1"),
    Path("artifacts/20260526_component_residual_attribution_v1"),
}
FEATURE_GROUP_PREFERENCE = {
    "base": 0,
    "base+rates": 1,
    "base+controls": 2,
    "base+lateral": 3,
    "base+rates+controls": 4,
    "base+rates+lateral": 5,
    "base+rates+controls+lateral": 6,
    "base+rates+controls+lateral+interactions": 7,
}
MODEL_FORM_PREFERENCE = {
    "hybrid": 0,
    "force_arm": 1,
    "direct_residual": 2,
}
VALIDATION_TIE_TOLERANCE = 1.0e-12

BASE_FEATURES = (
    "phase_sin_1",
    "phase_cos_1",
    "phase_sin_2",
    "phase_cos_2",
    "alpha_rad",
    "flap_frequency_hz",
    "true_airspeed_m_s",
    "dynamic_pressure_pa",
    "alpha_rad_x_phase_sin_1",
    "alpha_rad_x_phase_cos_1",
    "flap_frequency_hz_x_phase_sin_1",
    "flap_frequency_hz_x_phase_cos_1",
    "true_airspeed_m_s_x_phase_sin_1",
    "true_airspeed_m_s_x_phase_cos_1",
    "alpha_rad_x_flap_frequency_hz",
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
LATERAL_FEATURES = ("beta_proxy_rad", "v_air_b_y", "q_dyn_x_beta_proxy")
INTERACTION_FEATURES = (
    "beta_proxy_x_phase_sin_1",
    "beta_proxy_x_phase_cos_1",
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


@dataclass
class RidgeMultiOutputModel:
    feature_columns: list[str]
    coefficients: np.ndarray
    intercept: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    feature_fill: np.ndarray
    alpha: float

    def design_from_frame(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame.loc[:, self.feature_columns].to_numpy(dtype=float)
        x = np.where(np.isfinite(x), x, self.feature_fill)
        return (x - self.feature_mean) / self.feature_scale

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.design_from_frame(frame) @ self.coefficients + self.intercept


@dataclass
class ForceCorrectionModel:
    variant: str
    feature_group: str
    selected_features: list[str]
    ridge: RidgeMultiOutputModel
    uses_true_force_for_inference: bool = False

    def _design_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        features = _ensure_feature_frame(frame).loc[:, self.selected_features].copy()
        if self.variant == "additive":
            return features
        if self.variant == "affine":
            return _append_prior_force_interactions(features, _array_from_prefixed_columns(frame, "prior", FORCE_TARGETS))
        raise ValueError(f"unknown force variant: {self.variant}")

    def predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        return self.ridge.predict(self._design_frame(frame))

    def predict_force(self, frame: pd.DataFrame) -> np.ndarray:
        prior_force = _array_from_prefixed_columns(frame, "prior", FORCE_TARGETS)
        return prior_force + self.predict_raw(frame)


@dataclass
class FeatureTransform:
    columns: list[str]
    fill: list[float]
    mean: list[float]
    scale: list[float]

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        if not self.columns:
            return np.ones((len(features), 1), dtype=float)
        x = features.loc[:, self.columns].to_numpy(dtype=float)
        fill = np.asarray(self.fill, dtype=float)
        mean = np.asarray(self.mean, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        x = np.where(np.isfinite(x), x, fill)
        return np.column_stack([np.ones(len(features), dtype=float), (x - mean) / scale])


@dataclass
class MomentCorrectionModel:
    model_form: str
    feature_group: str
    selected_features: list[str]
    alpha: float
    feature_transform: FeatureTransform
    direct_coefficients: np.ndarray | None = None
    arm_coefficients: np.ndarray | None = None
    free_coefficients: np.ndarray | None = None
    uses_true_force_for_inference: bool = False

    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        features = _ensure_feature_frame(frame).loc[:, self.selected_features]
        phi = self.feature_transform.transform(features)
        force = _array_from_prefixed_columns(frame, "force_v2", FORCE_TARGETS)
        prior_moment = _array_from_prefixed_columns(frame, "prior", MOMENT_TARGETS)
        zeros = np.zeros((len(frame), 3), dtype=float)
        if self.model_form == "direct_residual":
            if self.direct_coefficients is None:
                raise ValueError("direct coefficients are required")
            delta = phi @ self.direct_coefficients
            return {"moment": prior_moment + delta, "r_hat": zeros, "arm_moment": zeros, "tau_free": delta}
        if self.arm_coefficients is None:
            raise ValueError("arm coefficients are required")
        r_hat = phi @ self.arm_coefficients
        arm_moment = cross_arm_force(r_hat, force)
        if self.model_form == "force_arm":
            if self.free_coefficients is None:
                raise ValueError("force_arm requires free torque coefficients")
            tau_free = phi @ self.free_coefficients
            return {"moment": arm_moment + tau_free, "r_hat": r_hat, "arm_moment": arm_moment, "tau_free": tau_free}
        if self.model_form == "hybrid":
            tau_free = np.zeros_like(arm_moment) if self.free_coefficients is None else phi @ self.free_coefficients
            return {"moment": prior_moment + arm_moment + tau_free, "r_hat": r_hat, "arm_moment": arm_moment, "tau_free": tau_free}
        raise ValueError(f"unknown moment form: {self.model_form}")


def _first_existing(columns: Iterable[str], frame: pd.DataFrame) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _numeric_series(frame: pd.DataFrame, column: str | None, default: float = 0.0) -> pd.Series:
    if column is None or column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _body_velocity_from_attitude(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series] | None:
    q_columns = [f"vehicle_attitude.q[{idx}]" for idx in range(4)]
    v_columns = ["vehicle_local_position.vx", "vehicle_local_position.vy", "vehicle_local_position.vz"]
    if any(column not in frame.columns for column in q_columns + v_columns):
        return None
    q0 = frame[q_columns[0]].to_numpy(dtype=float)
    q1 = frame[q_columns[1]].to_numpy(dtype=float)
    q2 = frame[q_columns[2]].to_numpy(dtype=float)
    q3 = frame[q_columns[3]].to_numpy(dtype=float)
    vn = frame[v_columns[0]].to_numpy(dtype=float) - _numeric_series(frame, "wind.windspeed_north").to_numpy()
    ve = frame[v_columns[1]].to_numpy(dtype=float) - _numeric_series(frame, "wind.windspeed_east").to_numpy()
    vd = frame[v_columns[2]].to_numpy(dtype=float)
    r00 = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
    r01 = 2.0 * (q1 * q2 - q0 * q3)
    r02 = 2.0 * (q1 * q3 + q0 * q2)
    r10 = 2.0 * (q1 * q2 + q0 * q3)
    r11 = 1.0 - 2.0 * (q1 * q1 + q3 * q3)
    r12 = 2.0 * (q2 * q3 - q0 * q1)
    r20 = 2.0 * (q1 * q3 - q0 * q2)
    r21 = 2.0 * (q2 * q3 + q0 * q1)
    r22 = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
    return (
        pd.Series(r00 * vn + r10 * ve + r20 * vd, index=frame.index, dtype=float),
        pd.Series(r01 * vn + r11 * ve + r21 * vd, index=frame.index, dtype=float),
        pd.Series(r02 * vn + r12 * ve + r22 * vd, index=frame.index, dtype=float),
    )


def _derive_body_air_velocity(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, str]:
    for x_col, y_col, z_col in (
        ("air_relative_velocity_b_x", "air_relative_velocity_b_y", "air_relative_velocity_b_z"),
        ("v_air_b_x", "v_air_b_y", "v_air_b_z"),
        ("body_air_relative_velocity_x", "body_air_relative_velocity_y", "body_air_relative_velocity_z"),
        ("air_relative_velocity_b.x", "air_relative_velocity_b.y", "air_relative_velocity_b.z"),
    ):
        if all(column in frame.columns for column in (x_col, y_col, z_col)):
            return _numeric_series(frame, x_col), _numeric_series(frame, y_col), _numeric_series(frame, z_col), f"{x_col},{y_col},{z_col}"
    body_velocity = _body_velocity_from_attitude(frame)
    if body_velocity is not None:
        return (*body_velocity, "attitude_local_velocity_wind")
    zeros = pd.Series(0.0, index=frame.index, dtype=float)
    return zeros, zeros, zeros, "zero_fallback"


def _derive_alpha_rad(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    if "alpha_rad" in frame.columns:
        return _numeric_series(frame, "alpha_rad"), "alpha_rad"
    body_velocity = _body_velocity_from_attitude(frame)
    if body_velocity is not None:
        u_b, _, w_b = body_velocity
        return pd.Series(np.arctan2(-w_b.to_numpy(dtype=float), u_b.to_numpy(dtype=float)), index=frame.index), "body_air_relative_velocity"
    column = _first_existing(("airspeed_validated.pitch_filtered", "vehicle_local_position.pitch", "pitch_rad"), frame)
    if column is not None:
        return _numeric_series(frame, column), column
    return pd.Series(0.0, index=frame.index, dtype=float), "zero_fallback"


def build_v2_feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build deployable v2 feature columns from raw samples or already-derived frames."""

    if all(column in frame.columns for column in BASE_FEATURES):
        features = frame.loc[:, [column for column in (*BASE_FEATURES, *RATE_FEATURES, *CONTROL_FEATURES, *LATERAL_FEATURES, *INTERACTION_FEATURES) if column in frame.columns]].copy()
        return features, {
            "uses_true_force": False,
            "source": "already_derived",
            "columns": features.columns.tolist(),
            "feature_sources": {column: "input_column" for column in features.columns},
        }

    spec: dict[str, object] = {"uses_true_force": False, "warnings": [], "feature_sources": {}}
    phase_column = _first_existing(("phase_corrected_rad", "wing_phase.phase_rad", "drive_phase_rad", "encoder_phase_rad", "phase_raw_rad"), frame)
    frequency_column = _first_existing(("cycle_flap_frequency_hz", "flap_frequency_hz", "encoder_rpm_est"), frame)
    airspeed_column = _first_existing(
        (
            "airspeed_validated.true_airspeed_m_s",
            "airspeed_validated.calibrated_airspeed_m_s",
            "airspeed_validated.indicated_airspeed_m_s",
            "true_airspeed_m_s",
        ),
        frame,
    )
    density_column = _first_existing(("vehicle_air_data.rho", "rho"), frame)
    phase = _numeric_series(frame, phase_column)
    flap_frequency = _numeric_series(frame, frequency_column)
    if frequency_column == "encoder_rpm_est":
        flap_frequency = flap_frequency / 60.0
    true_airspeed = _numeric_series(frame, airspeed_column)
    rho = _numeric_series(frame, density_column, 1.225)
    q_dyn = _numeric_series(frame, "dynamic_pressure_pa") if "dynamic_pressure_pa" in frame.columns else 0.5 * rho * true_airspeed * true_airspeed
    alpha_rad, alpha_source = _derive_alpha_rad(frame)
    v_air_b_x, v_air_b_y, _, beta_source = _derive_body_air_velocity(frame)
    beta_proxy = pd.Series(np.arctan2(v_air_b_y.to_numpy(dtype=float), np.maximum(np.abs(v_air_b_x.to_numpy(dtype=float)), 1.0e-6)), index=frame.index)

    features = pd.DataFrame(index=frame.index)
    features["phase_sin_1"] = np.sin(phase)
    features["phase_cos_1"] = np.cos(phase)
    features["phase_sin_2"] = np.sin(2.0 * phase)
    features["phase_cos_2"] = np.cos(2.0 * phase)
    features["alpha_rad"] = alpha_rad
    features["flap_frequency_hz"] = flap_frequency
    features["true_airspeed_m_s"] = true_airspeed
    features["dynamic_pressure_pa"] = q_dyn
    features["alpha_rad_x_phase_sin_1"] = alpha_rad * features["phase_sin_1"]
    features["alpha_rad_x_phase_cos_1"] = alpha_rad * features["phase_cos_1"]
    features["flap_frequency_hz_x_phase_sin_1"] = flap_frequency * features["phase_sin_1"]
    features["flap_frequency_hz_x_phase_cos_1"] = flap_frequency * features["phase_cos_1"]
    features["true_airspeed_m_s_x_phase_sin_1"] = true_airspeed * features["phase_sin_1"]
    features["true_airspeed_m_s_x_phase_cos_1"] = true_airspeed * features["phase_cos_1"]
    features["alpha_rad_x_flap_frequency_hz"] = alpha_rad * flap_frequency

    rate_sources = {
        "body_rate_p": ("vehicle_angular_velocity.xyz[0]", "body_rate_p", "p_rad_s", "roll_rate_rad_s"),
        "body_rate_q": ("vehicle_angular_velocity.xyz[1]", "body_rate_q", "q_rad_s", "pitch_rate_rad_s"),
        "body_rate_r": ("vehicle_angular_velocity.xyz[2]", "body_rate_r", "r_rad_s", "yaw_rate_rad_s"),
    }
    for target, candidates in rate_sources.items():
        column = _first_existing(candidates, frame)
        if column is not None:
            features[target] = _numeric_series(frame, column)
            features[f"q_dyn_x_{target}"] = q_dyn * features[target]

    for target in ("servo_rudder", "servo_left_elevon", "servo_right_elevon"):
        if target in frame.columns:
            features[target] = _numeric_series(frame, target)
            features[f"q_dyn_x_{target}"] = q_dyn * features[target]
    if {"servo_left_elevon", "servo_right_elevon"}.issubset(features.columns):
        features["elevon_sum_proxy"] = features["servo_left_elevon"] + features["servo_right_elevon"]
        features["elevon_diff_proxy"] = features["servo_left_elevon"] - features["servo_right_elevon"]
    elif {"actuator_servos.servo[0]", "actuator_servos.servo[1]"}.issubset(frame.columns):
        features["servo_left_elevon"] = _numeric_series(frame, "actuator_servos.servo[0]")
        features["servo_right_elevon"] = _numeric_series(frame, "actuator_servos.servo[1]")
        features["elevon_sum_proxy"] = features["servo_left_elevon"] + features["servo_right_elevon"]
        features["elevon_diff_proxy"] = features["servo_left_elevon"] - features["servo_right_elevon"]

    features["beta_proxy_rad"] = beta_proxy
    features["v_air_b_y"] = v_air_b_y
    features["q_dyn_x_beta_proxy"] = q_dyn * beta_proxy
    for column in ("beta_proxy", "body_rate_p", "body_rate_q", "body_rate_r", "servo_rudder", "elevon_diff"):
        if column == "beta_proxy":
            source = features["beta_proxy_rad"]
            prefix = "beta_proxy"
        elif column == "elevon_diff":
            if "elevon_diff_proxy" not in features:
                continue
            source = features["elevon_diff_proxy"]
            prefix = "elevon_diff"
        elif column in features:
            source = features[column]
            prefix = column
        else:
            continue
        features[f"{prefix}_x_phase_sin_1"] = source * features["phase_sin_1"]
        features[f"{prefix}_x_phase_cos_1"] = source * features["phase_cos_1"]

    spec.update(
        {
            "phase_column": phase_column,
            "frequency_column": frequency_column,
            "airspeed_column": airspeed_column,
            "density_column": density_column,
            "alpha_source": alpha_source,
            "body_air_velocity_source": beta_source,
            "columns": features.columns.tolist(),
            "feature_sources": {column: "derived_from_input" for column in features.columns},
        }
    )
    return features, spec


def v2_feature_groups(columns: Iterable[str]) -> dict[str, list[str]]:
    available = list(columns)
    available_set = set(available)

    def present(names: Iterable[str]) -> list[str]:
        return [name for name in names if name in available_set]

    return {
        "base": present(BASE_FEATURES),
        "rates": present(RATE_FEATURES),
        "controls": present(CONTROL_FEATURES),
        "lateral": present(LATERAL_FEATURES),
        "interactions": present(INTERACTION_FEATURES),
    }


def force_feature_group_specs(feature_groups: dict[str, list[str]]) -> dict[str, list[str]]:
    combos = {
        "base": ("base",),
        "base+rates": ("base", "rates"),
        "base+controls": ("base", "controls"),
        "base+lateral": ("base", "lateral"),
        "base+rates+controls": ("base", "rates", "controls"),
        "base+rates+lateral": ("base", "rates", "lateral"),
        "base+rates+controls+lateral": ("base", "rates", "controls", "lateral"),
        "base+rates+controls+lateral+interactions": ("base", "rates", "controls", "lateral", "interactions"),
    }
    specs: dict[str, list[str]] = {}
    for name, group_names in combos.items():
        columns: list[str] = []
        for group_name in group_names:
            columns.extend(feature_groups.get(group_name, []))
        specs[name] = list(dict.fromkeys(columns))
    return specs


def _ensure_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return build_v2_feature_frame(frame)[0]


def _array_from_prefixed_columns(frame: pd.DataFrame, prefix: str, targets: tuple[str, ...]) -> np.ndarray:
    columns = [f"{prefix}_{target}" for target in targets]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    return frame.loc[:, columns].to_numpy(dtype=float)


def fit_ridge_multi_output(x: np.ndarray, y: np.ndarray, alpha: float) -> RidgeMultiOutputModel:
    fill = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    x_filled = np.where(np.isfinite(x), x, fill)
    mean = np.mean(x_filled, axis=0)
    scale = np.std(x_filled, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    x_scaled = (x_filled - mean) / scale
    intercept = np.mean(y, axis=0)
    y_centered = y - intercept
    gram = x_scaled.T @ x_scaled
    if alpha > 0.0:
        gram = gram + float(alpha) * np.eye(gram.shape[0])
    rhs = x_scaled.T @ y_centered
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return RidgeMultiOutputModel([], coefficients, intercept, mean, scale, fill, float(alpha))


def predict_ridge_multi_output(model: RidgeMultiOutputModel, frame: pd.DataFrame) -> np.ndarray:
    return model.predict(frame)


def _fit_ridge_frame(frame: pd.DataFrame, target: np.ndarray, alpha: float) -> RidgeMultiOutputModel:
    model = fit_ridge_multi_output(frame.to_numpy(dtype=float), target, alpha)
    model.feature_columns = frame.columns.tolist()
    return model


def _append_prior_force_interactions(features: pd.DataFrame, prior_force: np.ndarray) -> pd.DataFrame:
    base_columns = features.columns.tolist()
    interaction_columns: dict[str, np.ndarray] = {}
    for idx, target in enumerate(FORCE_TARGETS):
        for column in base_columns:
            interaction_columns[f"prior_{target}_x_{column}"] = prior_force[:, idx] * features[column].to_numpy(dtype=float)
    if not interaction_columns:
        return features.copy()
    interactions = pd.DataFrame(interaction_columns, index=features.index)
    return pd.concat([features, interactions], axis=1)


def _channel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {"mae": np.nan, "rmse": np.nan, "bias": np.nan, "r2": np.nan}
    residual = y_pred[mask] - y_true[mask]
    ss_res = float(np.sum(residual * residual))
    centered = y_true[mask] - float(np.mean(y_true[mask]))
    ss_tot = float(np.sum(centered * centered))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "bias": float(np.mean(residual)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
    }


def _nanmean_without_warning(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return float("nan")
    return float(np.mean(finite))


def force_metrics(*, split: str, model_name: str, true_force: np.ndarray, predicted_force: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for idx, target in enumerate(FORCE_TARGETS):
        rows.append({"split": split, "model": model_name, "target": target, "n": int(len(true_force)), **_channel_metrics(true_force[:, idx], predicted_force[:, idx])})
    rows.append(
        {
            "split": split,
            "model": model_name,
            "target": "force_mean",
            "n": int(len(true_force)),
            "mae": float(np.nanmean([row["mae"] for row in rows])),
            "rmse": float(np.sqrt(np.nanmean(np.square([row["rmse"] for row in rows])))),
            "bias": float(np.nanmean([row["bias"] for row in rows])),
            "r2": _nanmean_without_warning(row["r2"] for row in rows),
        }
    )
    return pd.DataFrame(rows)


def moment_metrics(*, split: str, model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for idx, target in enumerate(MOMENT_TARGETS):
        rows.append({"split": split, "model": model_name, "target": target, "n": int(len(y_true)), **_channel_metrics(y_true[:, idx], y_pred[:, idx])})
    rows.append(
        {
            "split": split,
            "model": model_name,
            "target": "moment_mean",
            "n": int(len(y_true)),
            "mae": float(np.nanmean([row["mae"] for row in rows])),
            "rmse": float(np.sqrt(np.nanmean(np.square([row["rmse"] for row in rows])))),
            "bias": float(np.nanmean([row["bias"] for row in rows])),
            "r2": _nanmean_without_warning(row["r2"] for row in rows),
        }
    )
    return pd.DataFrame(rows)


def _force_design(features: pd.DataFrame, prior_force: np.ndarray, variant: str) -> pd.DataFrame:
    if variant == "additive":
        return features
    if variant == "affine":
        return _append_prior_force_interactions(features, prior_force)
    raise ValueError(f"unknown force variant: {variant}")


def fit_force_correction_models(
    split_frames: dict[str, pd.DataFrame],
    feature_group_specs: dict[str, list[str]],
    alphas: tuple[float, ...],
    variants: tuple[str, ...] = ("additive", "affine"),
) -> tuple[pd.DataFrame, dict[str, object]]:
    train = split_frames["train"]
    train_features_all = _ensure_feature_frame(train)
    train_true = _array_from_prefixed_columns(train, "label", FORCE_TARGETS)
    train_prior = _array_from_prefixed_columns(train, "prior", FORCE_TARGETS)
    rows: list[pd.DataFrame] = []
    candidates: list[dict[str, object]] = []
    for split, frame in split_frames.items():
        rows.append(_metrics_with_candidate_columns(force_metrics(split=split, model_name="prior", true_force=_array_from_prefixed_columns(frame, "label", FORCE_TARGETS), predicted_force=_array_from_prefixed_columns(frame, "prior", FORCE_TARGETS)), "prior", "prior", 0.0, "prior", False))

    for feature_group, columns in feature_group_specs.items():
        columns = [column for column in columns if column in train_features_all.columns]
        if not columns:
            continue
        train_selected = train_features_all.loc[:, columns]
        for variant in variants:
            for alpha in alphas:
                design = _force_design(train_selected, train_prior, variant)
                model = ForceCorrectionModel(
                    variant=variant,
                    feature_group=feature_group,
                    selected_features=columns,
                    ridge=_fit_ridge_frame(design, train_true - train_prior, alpha),
                )
                val_metrics_frame: pd.DataFrame | None = None
                for split, frame in split_frames.items():
                    pred = model.predict_force(frame)
                    metrics = force_metrics(split=split, model_name=f"{variant}_{feature_group}_alpha_{alpha:g}", true_force=_array_from_prefixed_columns(frame, "label", FORCE_TARGETS), predicted_force=pred)
                    metrics = _metrics_with_candidate_columns(metrics, variant, feature_group, alpha, variant, False)
                    rows.append(metrics)
                    if split == "val":
                        val_metrics_frame = metrics
                if val_metrics_frame is None:
                    raise RuntimeError("validation split is required")
                val_rmse = float(val_metrics_frame.loc[val_metrics_frame["target"].eq("force_mean"), "rmse"].iloc[0])
                candidates.append({"model": model, "variant": variant, "feature_group": feature_group, "alpha": float(alpha), "val_rmse": val_rmse})
    best_val = min(float(item["val_rmse"]) for item in candidates)
    tied = [item for item in candidates if float(item["val_rmse"]) <= best_val + VALIDATION_TIE_TOLERANCE]
    selected_candidate = min(
        tied,
        key=lambda item: (
            FEATURE_GROUP_PREFERENCE.get(str(item["feature_group"]), 999),
            str(item["variant"]),
            float(item["alpha"]),
        ),
    )
    metrics = pd.concat(rows, ignore_index=True)
    mask = (
        metrics["split"].eq("val")
        & metrics["target"].eq("force_mean")
        & metrics["variant"].eq(selected_candidate["variant"])
        & metrics["feature_group"].eq(selected_candidate["feature_group"])
        & np.isclose(metrics["alpha"].astype(float), float(selected_candidate["alpha"]))
    )
    if not mask.any():
        raise RuntimeError("selected force candidate not found in metrics")
    selected_mask = (
        metrics["variant"].eq(selected_candidate["variant"])
        & metrics["feature_group"].eq(selected_candidate["feature_group"])
        & np.isclose(metrics["alpha"].astype(float), float(selected_candidate["alpha"]))
    )
    metrics.loc[selected_mask, "is_selected"] = True
    selected = {
        **{key: selected_candidate[key] for key in ("variant", "feature_group", "alpha", "model")},
        "selection_split": "val",
        "selection_metric": "force_mean_rmse",
        "validation_tie_policy": "within 1e-12 RMSE, prefer the smallest predeclared feature group, then variant name, then alpha",
        "uses_true_force_for_inference": False,
    }
    return metrics, selected


def _metrics_with_candidate_columns(metrics: pd.DataFrame, variant: str, feature_group: str, alpha: float, model_form: str, selected: bool) -> pd.DataFrame:
    out = metrics.copy()
    out["variant"] = variant
    out["feature_group"] = feature_group
    out["alpha"] = float(alpha)
    out["model_form"] = model_form
    out["is_selected"] = bool(selected)
    return out


def predict_force_correction(model: ForceCorrectionModel, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_force(frame)


def cross_arm_force(arm: np.ndarray, force: np.ndarray) -> np.ndarray:
    arm = np.asarray(arm, dtype=float)
    force = np.asarray(force, dtype=float)
    if arm.shape != force.shape or arm.ndim != 2 or arm.shape[1] != 3:
        raise ValueError(f"arm and force must both be shaped (n, 3); got {arm.shape} and {force.shape}")
    return np.column_stack(
        [
            arm[:, 1] * force[:, 2] - arm[:, 2] * force[:, 1],
            arm[:, 2] * force[:, 0] - arm[:, 0] * force[:, 2],
            arm[:, 0] * force[:, 1] - arm[:, 1] * force[:, 0],
        ]
    )


def _fit_transform(features: pd.DataFrame, columns: list[str]) -> FeatureTransform:
    if not columns:
        return FeatureTransform([], [], [], [])
    x = features.loc[:, columns].to_numpy(dtype=float)
    fill = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    x = np.where(np.isfinite(x), x, fill)
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return FeatureTransform(columns, fill.tolist(), mean.tolist(), scale.tolist())


def _ridge_solve(design: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    y = target.reshape(-1)
    mask = np.isfinite(y) & np.isfinite(design).all(axis=1)
    x = design[mask]
    y = y[mask]
    gram = x.T @ x
    if alpha > 0.0:
        gram = gram + float(alpha) * np.eye(gram.shape[0])
    rhs = x.T @ y
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def _ridge_solve_multi(features: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    mask = np.isfinite(target).all(axis=1) & np.isfinite(features).all(axis=1)
    x = features[mask]
    y = target[mask]
    gram = x.T @ x
    if alpha > 0.0:
        gram = gram + float(alpha) * np.eye(gram.shape[0])
    rhs = x.T @ y
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def _build_arm_design_matrix(phi: np.ndarray, force: np.ndarray) -> np.ndarray:
    n, p = phi.shape
    fx, fy, fz = force[:, 0], force[:, 1], force[:, 2]
    design = np.zeros((n, 3, p * 3), dtype=float)
    for idx in range(p):
        column = phi[:, idx]
        base = idx * 3
        design[:, 0, base + 1] = column * fz
        design[:, 0, base + 2] = -column * fy
        design[:, 1, base + 0] = -column * fz
        design[:, 1, base + 2] = column * fx
        design[:, 2, base + 0] = column * fy
        design[:, 2, base + 1] = -column * fx
    return design.reshape(n * 3, p * 3)


def _fit_moment_model(frame: pd.DataFrame, columns: list[str], model_form: str, alpha: float) -> MomentCorrectionModel:
    features = _ensure_feature_frame(frame).loc[:, columns]
    transform = _fit_transform(features, columns)
    phi = transform.transform(features)
    y = _array_from_prefixed_columns(frame, "label", MOMENT_TARGETS)
    prior = _array_from_prefixed_columns(frame, "prior", MOMENT_TARGETS)
    force = _array_from_prefixed_columns(frame, "force_v2", FORCE_TARGETS)
    if model_form == "direct_residual":
        coeff = _ridge_solve_multi(phi, y - prior, alpha)
        return MomentCorrectionModel(model_form, "", columns, float(alpha), transform, direct_coefficients=coeff)
    if model_form == "force_arm":
        arm_coeff = _ridge_solve(_build_arm_design_matrix(phi, force), y, alpha).reshape(phi.shape[1], 3)
        arm_moment = cross_arm_force(phi @ arm_coeff, force)
        free_coeff = _ridge_solve_multi(phi, y - arm_moment, alpha)
        return MomentCorrectionModel(model_form, "", columns, float(alpha), transform, arm_coefficients=arm_coeff, free_coefficients=free_coeff)
    if model_form == "hybrid":
        residual = y - prior
        arm_coeff = _ridge_solve(_build_arm_design_matrix(phi, force), residual, alpha).reshape(phi.shape[1], 3)
        arm_moment = cross_arm_force(phi @ arm_coeff, force)
        free_coeff = _ridge_solve_multi(phi, residual - arm_moment, alpha)
        return MomentCorrectionModel(model_form, "", columns, float(alpha), transform, arm_coefficients=arm_coeff, free_coefficients=free_coeff)
    raise ValueError(f"unknown moment form: {model_form}")


def fit_moment_correction_models(
    split_frames: dict[str, pd.DataFrame],
    feature_group_specs: dict[str, list[str]],
    alphas: tuple[float, ...],
    model_forms: tuple[str, ...] = ("direct_residual", "force_arm", "hybrid"),
) -> tuple[pd.DataFrame, dict[str, object]]:
    train = split_frames["train"]
    train_features_all = _ensure_feature_frame(train)
    rows: list[pd.DataFrame] = []
    candidates: list[dict[str, object]] = []
    for split, frame in split_frames.items():
        rows.append(_metrics_with_candidate_columns(moment_metrics(split=split, model_name="prior", y_true=_array_from_prefixed_columns(frame, "label", MOMENT_TARGETS), y_pred=_array_from_prefixed_columns(frame, "prior", MOMENT_TARGETS)), "prior", "prior", 0.0, "prior", False))

    for feature_group, columns in feature_group_specs.items():
        columns = [column for column in columns if column in train_features_all.columns]
        if not columns:
            continue
        for model_form in model_forms:
            for alpha in alphas:
                model = _fit_moment_model(train, columns, model_form, alpha)
                model.feature_group = feature_group
                val_metrics_frame: pd.DataFrame | None = None
                for split, frame in split_frames.items():
                    pred = model.predict(frame)["moment"]
                    metrics = moment_metrics(split=split, model_name=f"{model_form}_{feature_group}_alpha_{alpha:g}", y_true=_array_from_prefixed_columns(frame, "label", MOMENT_TARGETS), y_pred=pred)
                    metrics = _metrics_with_candidate_columns(metrics, model_form, feature_group, alpha, model_form, False)
                    rows.append(metrics)
                    if split == "val":
                        val_metrics_frame = metrics
                if val_metrics_frame is None:
                    raise RuntimeError("validation split is required")
                val_rmse = float(val_metrics_frame.loc[val_metrics_frame["target"].eq("moment_mean"), "rmse"].iloc[0])
                candidates.append({"model": model, "model_form": model_form, "feature_group": feature_group, "alpha": float(alpha), "val_rmse": val_rmse})
    best_val = min(float(item["val_rmse"]) for item in candidates)
    tied = [item for item in candidates if float(item["val_rmse"]) <= best_val + VALIDATION_TIE_TOLERANCE]
    selected_candidate = min(
        tied,
        key=lambda item: (
            FEATURE_GROUP_PREFERENCE.get(str(item["feature_group"]), 999),
            MODEL_FORM_PREFERENCE.get(str(item["model_form"]), 999),
            float(item["alpha"]),
        ),
    )
    metrics = pd.concat(rows, ignore_index=True)
    selected_mask = (
        metrics["model_form"].eq(selected_candidate["model_form"])
        & metrics["feature_group"].eq(selected_candidate["feature_group"])
        & np.isclose(metrics["alpha"].astype(float), float(selected_candidate["alpha"]))
    )
    metrics.loc[selected_mask, "is_selected"] = True
    selected = {
        **{key: selected_candidate[key] for key in ("model_form", "feature_group", "alpha", "model")},
        "selection_split": "val",
        "selection_metric": "moment_mean_rmse",
        "validation_tie_policy": "within 1e-12 RMSE, prefer the smallest predeclared feature group, then hybrid/force_arm/direct_residual, then alpha",
        "uses_true_force_for_inference": False,
    }
    return metrics, selected


def predict_moment_model(model: MomentCorrectionModel, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return model.predict(frame)


def _prediction_path(root: Path, split: str) -> Path:
    for path in (root / f"{split}_predictions.parquet", root / "prediction_parquets" / f"{split}_predictions.parquet"):
        if path.exists():
            return path
    raise FileNotFoundError(f"could not find {split}_predictions.parquet under {root}")


def _load_v1_force(root: Path, split: str, n: int) -> np.ndarray:
    path = _prediction_path(root, split)
    frame = pd.read_parquet(path)
    if len(frame) != n:
        raise ValueError(f"{split} force v1 row mismatch: samples={n} force_v1={len(frame)}")
    return _array_from_prefixed_columns(frame, "corrected", FORCE_TARGETS)


def _load_cli_split(split: str, split_root: Path, prior_root: Path, force_v1_root: Path) -> pd.DataFrame:
    samples = pd.read_parquet(split_root / f"{split}_samples.parquet")
    prior = pd.read_parquet(prior_root / f"{split}_predictions.parquet")
    if len(samples) != len(prior):
        raise ValueError(f"{split} row mismatch: samples={len(samples)} prior={len(prior)}")
    features, _ = build_v2_feature_frame(samples)
    frame = features.copy()
    metadata_columns = [column for column in ("timestamp_us", "time_s", "log_id", "segment_id", "cycle_id", "phase_corrected_rad", "split") if column in samples.columns]
    for column in metadata_columns:
        frame[column] = samples[column].to_numpy()
    if "split" not in frame:
        frame["split"] = split
    for target in FORCE_TARGETS:
        frame[f"label_{target}"] = samples[target].to_numpy(dtype=float)
        frame[f"prior_{target}"] = prior[target].to_numpy(dtype=float)
        frame[f"force_v1_{target}"] = _load_v1_force(force_v1_root, split, len(samples))[:, FORCE_TARGETS.index(target)]
    for target in MOMENT_TARGETS:
        frame[f"label_{target}"] = samples[target].to_numpy(dtype=float)
        frame[f"prior_{target}"] = prior[target].to_numpy(dtype=float)
    return frame


def _assert_safe_output_root(output_root: Path, overwrite: bool) -> None:
    resolved = output_root.resolve()
    protected = {path.resolve() for path in PROTECTED_ARTIFACT_ROOTS}
    if resolved in protected and not overwrite:
        raise ValueError(f"refusing to write to protected baseline artifact root without --overwrite: {output_root}")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise ValueError(f"refusing to write to non-empty output root without --overwrite: {output_root}")


def _write_predictions(output_root: Path, split_frames: dict[str, pd.DataFrame], force_model: ForceCorrectionModel, moment_model: MomentCorrectionModel) -> dict[str, Path]:
    prediction_dir = output_root / "prediction_parquets"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    metadata_columns = ("timestamp_us", "time_s", "log_id", "segment_id", "cycle_id", "phase_corrected_rad", "split")
    for split, frame in split_frames.items():
        out = frame.loc[:, [column for column in metadata_columns if column in frame.columns]].copy()
        force_v2 = force_model.predict_force(frame)
        working = frame.copy()
        for idx, target in enumerate(FORCE_TARGETS):
            working[f"force_v2_{target}"] = force_v2[:, idx]
            out[f"label_{target}"] = frame[f"label_{target}"].to_numpy(dtype=float)
            out[f"prior_{target}"] = frame[f"prior_{target}"].to_numpy(dtype=float)
            out[f"force_v1_{target}"] = frame[f"force_v1_{target}"].to_numpy(dtype=float) if f"force_v1_{target}" in frame else np.nan
            out[f"force_v2_{target}"] = force_v2[:, idx]
            out[f"force_v2_residual_{target}"] = frame[f"label_{target}"].to_numpy(dtype=float) - force_v2[:, idx]
        moment_pred = moment_model.predict(working)["moment"]
        for idx, target in enumerate(MOMENT_TARGETS):
            out[f"label_{target}"] = frame[f"label_{target}"].to_numpy(dtype=float)
            out[f"prior_{target}"] = frame[f"prior_{target}"].to_numpy(dtype=float)
            out[f"moment_v2_{target}"] = moment_pred[:, idx]
            out[f"moment_v2_residual_{target}"] = frame[f"label_{target}"].to_numpy(dtype=float) - moment_pred[:, idx]
        path = prediction_dir / f"{split}_predictions.parquet"
        out.to_parquet(path, index=False)
        paths[split] = path
    return paths


def _per_log_metrics(split_frames: dict[str, pd.DataFrame], force_model: ForceCorrectionModel, moment_model: MomentCorrectionModel) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, frame in split_frames.items():
        log_ids = frame["log_id"].to_numpy() if "log_id" in frame.columns else np.full(len(frame), "unknown")
        force_v2 = force_model.predict_force(frame)
        working = frame.copy()
        for idx, target in enumerate(FORCE_TARGETS):
            working[f"force_v2_{target}"] = force_v2[:, idx]
        moment_v2 = moment_model.predict(working)["moment"]
        for log_id in pd.unique(log_ids):
            mask = log_ids == log_id
            force_mean = force_metrics(split=split, model_name="force_v2", true_force=_array_from_prefixed_columns(frame.iloc[mask], "label", FORCE_TARGETS), predicted_force=force_v2[mask]).query("target == 'force_mean'").iloc[0]
            moment_mean = moment_metrics(split=split, model_name="moment_v2", y_true=_array_from_prefixed_columns(frame.iloc[mask], "label", MOMENT_TARGETS), y_pred=moment_v2[mask]).query("target == 'moment_mean'").iloc[0]
            rows.append({"split": split, "log_id": log_id, "force_rmse": float(force_mean["rmse"]), "moment_rmse": float(moment_mean["rmse"]), "n": int(mask.sum())})
    return pd.DataFrame(rows)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [[str(row[column]) for column in frame.columns] for _, row in frame.iterrows()]
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |", *("| " + " | ".join(row) + " |" for row in rows)])


def _array_to_json(array: np.ndarray | list[float] | None) -> list[object]:
    if array is None:
        return []
    return np.asarray(array).tolist()


def _feature_transform_state(transform: FeatureTransform) -> dict[str, object]:
    return {
        "columns": transform.columns,
        "fill": transform.fill,
        "mean": transform.mean,
        "scale": transform.scale,
        "adds_intercept_column": True,
    }


def _write_inference_model_state(output_root: Path, force_model: ForceCorrectionModel, moment_model: MomentCorrectionModel) -> Path:
    state = {
        "uses_true_force_for_inference": False,
        "force_model": {
            "variant": force_model.variant,
            "feature_group": force_model.feature_group,
            "selected_features": force_model.selected_features,
            "ridge_feature_columns": force_model.ridge.feature_columns,
            "feature_fill": _array_to_json(force_model.ridge.feature_fill),
            "feature_mean": _array_to_json(force_model.ridge.feature_mean),
            "feature_scale": _array_to_json(force_model.ridge.feature_scale),
            "coefficients": _array_to_json(force_model.ridge.coefficients),
            "intercept": _array_to_json(force_model.ridge.intercept),
            "alpha": float(force_model.ridge.alpha),
        },
        "moment_model": {
            "model_form": moment_model.model_form,
            "feature_group": moment_model.feature_group,
            "selected_features": moment_model.selected_features,
            "alpha": float(moment_model.alpha),
            "feature_transform": _feature_transform_state(moment_model.feature_transform),
            "direct_coefficients": _array_to_json(moment_model.direct_coefficients),
            "arm_coefficients": _array_to_json(moment_model.arm_coefficients),
            "free_coefficients": _array_to_json(moment_model.free_coefficients),
        },
    }
    path = output_root / "inference_model_state.json"
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return path


def train_deployable_wrench_correction_v2(
    *,
    split_root: Path = DEFAULT_SPLIT_ROOT,
    prior_root: Path = DEFAULT_PRIOR_ROOT,
    force_v1_root: Path = DEFAULT_FORCE_V1_ROOT,
    moment_v1_root: Path = DEFAULT_MOMENT_V1_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    alphas: tuple[float, ...] = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
    overwrite: bool = False,
    command: str = "",
) -> dict[str, object]:
    _assert_safe_output_root(output_root, overwrite)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "figures").mkdir(parents=True, exist_ok=True)
    split_frames = {split: _load_cli_split(split, split_root, prior_root, force_v1_root) for split in ("train", "val", "test")}
    groups = force_feature_group_specs(v2_feature_groups(split_frames["train"].columns))
    force_metrics_table, selected_force = fit_force_correction_models(split_frames, groups, alphas, variants=("additive", "affine"))
    force_model = selected_force["model"]
    moment_frames = {}
    for split, frame in split_frames.items():
        force_v2 = predict_force_correction(force_model, frame)
        augmented = frame.copy()
        for idx, target in enumerate(FORCE_TARGETS):
            augmented[f"force_v2_{target}"] = force_v2[:, idx]
        moment_frames[split] = augmented
    moment_metrics_table, selected_moment = fit_moment_correction_models(moment_frames, groups, alphas)
    moment_model = selected_moment["model"]
    prediction_paths = _write_predictions(output_root, split_frames, force_model, moment_model)
    per_log = _per_log_metrics(split_frames, force_model, moment_model)
    selection = pd.concat(
        [
            force_metrics_table.query("split == 'val' and target == 'force_mean'").assign(stage="force_v2"),
            moment_metrics_table.query("split == 'val' and target == 'moment_mean'").assign(stage="moment_v2"),
        ],
        ignore_index=True,
    ).sort_values(["stage", "rmse"])
    feature_ablation = selection.loc[:, ["stage", "variant", "model_form", "feature_group", "alpha", "rmse", "r2", "is_selected"]]

    force_metrics_table.to_csv(output_root / "force_metrics_by_split.csv", index=False)
    moment_metrics_table.to_csv(output_root / "moment_metrics_by_split.csv", index=False)
    selection.to_csv(output_root / "model_selection.csv", index=False)
    feature_ablation.to_csv(output_root / "feature_group_ablation.csv", index=False)
    per_log.to_csv(output_root / "per_log_metrics.csv", index=False)
    inference_model_state_path = _write_inference_model_state(output_root, force_model, moment_model)
    config = {
        "split_root": str(split_root),
        "prior_root": str(prior_root),
        "force_v1_root": str(force_v1_root),
        "moment_v1_root": str(moment_v1_root),
        "output_root": str(output_root),
        "uses_true_force_for_inference": False,
        "selection_split": "val",
        "selected_force_source": "corrected_force_v2",
        "selected_force_v2": {
            "variant": selected_force["variant"],
            "feature_group": selected_force["feature_group"],
            "alpha": float(selected_force["alpha"]),
        },
        "selected_moment_v2": {
            "model_form": selected_moment["model_form"],
            "feature_group": selected_moment["feature_group"],
            "alpha": float(selected_moment["alpha"]),
        },
        "inference_model_state": str(inference_model_state_path),
        "feature_sources": build_v2_feature_frame(pd.read_parquet(split_root / "train_samples.parquet"))[1].get("feature_sources", {}),
        "inference_available_features": list(groups["base+rates+controls+lateral+interactions"]),
        "excluded_diagnostic_only_rows": "true_force rows excluded from deployable comparisons and selection",
        "validation_tie_policy": {
            "tolerance_rmse": VALIDATION_TIE_TOLERANCE,
            "force": "prefer the smallest predeclared feature group, then variant name, then alpha",
            "moment": "prefer the smallest predeclared feature group, then hybrid/force_arm/direct_residual, then alpha",
        },
        "command": command,
    }
    (output_root / "model_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(
        output_root / "coefficients_or_model.npz",
        force_coefficients=force_model.ridge.coefficients,
        force_intercept=force_model.ridge.intercept,
        force_feature_columns=np.array(force_model.ridge.feature_columns, dtype=object),
        force_feature_fill=force_model.ridge.feature_fill,
        force_feature_mean=force_model.ridge.feature_mean,
        force_feature_scale=force_model.ridge.feature_scale,
        moment_feature_columns=np.array(moment_model.feature_transform.columns, dtype=object),
        moment_feature_fill=np.array(moment_model.feature_transform.fill, dtype=float),
        moment_feature_mean=np.array(moment_model.feature_transform.mean, dtype=float),
        moment_feature_scale=np.array(moment_model.feature_transform.scale, dtype=float),
        moment_direct_coefficients=np.array([]) if moment_model.direct_coefficients is None else moment_model.direct_coefficients,
        moment_arm_coefficients=np.array([]) if moment_model.arm_coefficients is None else moment_model.arm_coefficients,
        moment_free_coefficients=np.array([]) if moment_model.free_coefficients is None else moment_model.free_coefficients,
    )
    _write_readme(output_root, command, selected_force, selected_moment, force_metrics_table, moment_metrics_table)
    return {
        "output_root": output_root,
        "force_metrics": output_root / "force_metrics_by_split.csv",
        "moment_metrics": output_root / "moment_metrics_by_split.csv",
        "test_predictions": prediction_paths["test"],
        "selected_force": selected_force,
        "selected_moment": selected_moment,
    }


def _write_readme(output_root: Path, command: str, selected_force: dict[str, object], selected_moment: dict[str, object], force_metrics_table: pd.DataFrame, moment_metrics_table: pd.DataFrame) -> None:
    force_test = force_metrics_table.query("split == 'test' and is_selected and target in ['fx_b', 'fy_b', 'fz_b', 'force_mean']").loc[:, ["target", "rmse", "mae", "bias", "r2"]]
    moment_test = moment_metrics_table.query("split == 'test' and is_selected and target in ['mx_b', 'my_b', 'mz_b', 'moment_mean']").loc[:, ["target", "rmse", "mae", "bias", "r2"]]
    lines = [
        "# Deployable Wrench Correction v2",
        "",
        "This artifact trains validation-selected deployable force and moment corrections. No deployable prediction or metric uses true_force.",
        "If validation RMSE ties within 1e-12, selection uses a predeclared deployable tie policy: smaller feature groups first, and for moment forms hybrid/force_arm before direct_residual.",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Selected force v2",
        "",
        f"- Variant: `{selected_force['variant']}`",
        f"- Feature group: `{selected_force['feature_group']}`",
        f"- Alpha: `{float(selected_force['alpha']):g}`",
        "",
        dataframe_to_markdown(force_test.reset_index(drop=True)),
        "",
        "## Selected moment v2",
        "",
        f"- Form: `{selected_moment['model_form']}`",
        f"- Feature group: `{selected_moment['feature_group']}`",
        f"- Alpha: `{float(selected_moment['alpha']):g}`",
        "",
        dataframe_to_markdown(moment_test.reset_index(drop=True)),
        "",
        "## Outputs",
        "",
        "- `force_metrics_by_split.csv` and `moment_metrics_by_split.csv`: train/val/test metrics for validation-selected candidates and baselines.",
        "- `model_selection.csv`: validation-only selection table.",
        "- `prediction_parquets/`: aligned deployable predictions.",
        "- `model_config.json`: selected configuration and true-force exclusion record.",
        "- `inference_model_state.json`: normalization state and coefficients needed for standalone deployable inference.",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--prior-root", type=Path, default=DEFAULT_PRIOR_ROOT)
    parser.add_argument("--force-v1-root", type=Path, default=DEFAULT_FORCE_V1_ROOT)
    parser.add_argument("--moment-v1-root", type=Path, default=DEFAULT_MOMENT_V1_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--alphas", default="0,0.001,0.01,0.1,1,10,100,1000")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    command = (
        "python scripts/train_deployable_wrench_correction_v2.py "
        f"--split-root {args.split_root} --prior-root {args.prior_root} "
        f"--force-v1-root {args.force_v1_root} --moment-v1-root {args.moment_v1_root} "
        f"--output-root {args.output_root} --alphas {args.alphas}"
    )
    if args.overwrite:
        command += " --overwrite"
    try:
        outputs = train_deployable_wrench_correction_v2(
            split_root=args.split_root,
            prior_root=args.prior_root,
            force_v1_root=args.force_v1_root,
            moment_v1_root=args.moment_v1_root,
            output_root=args.output_root,
            alphas=_parse_csv_floats(args.alphas),
            overwrite=args.overwrite,
            command=command,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps({key: str(value) for key, value in outputs.items() if key not in {"selected_force", "selected_moment"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
