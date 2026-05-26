#!/usr/bin/env python3
"""Attribute structured residual associations for component wrench models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FORCE_TARGETS = ("fx_b", "fy_b", "fz_b")
MOMENT_TARGETS = ("mx_b", "my_b", "mz_b")
ALL_TARGETS = FORCE_TARGETS + MOMENT_TARGETS
DEFAULT_SPLIT_ROOT = Path("dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1")
DEFAULT_FORCE_PREDICTION_ROOT = Path("artifacts/20260525_delaurier_greybox_force_correction_v1")
DEFAULT_MOMENT_PREDICTION_ROOT = Path("artifacts/20260525_dynamic_arm_moment_head_v1")
DEFAULT_PRIOR_ROOT = Path("artifacts/delaurier_physical_prior_v1")
DEFAULT_OUTPUT_ROOT = Path("artifacts/20260526_component_residual_attribution_v1")
DEFAULT_KEY_TARGETS = ("fy_b", "mx_b", "my_b", "mz_b")
DEFAULT_ALPHAS = (0.0, 0.1, 1.0, 10.0, 100.0)
DEFAULT_MI_MAX_SAMPLES = 50_000
METADATA_COLUMNS = (
    "timestamp_us",
    "time_s",
    "log_id",
    "segment_id",
    "cycle_id",
    "phase_corrected_rad",
    "split",
)
PREDICTION_PREFIXES = (
    "label_",
    "force_prior_",
    "force_corrected_",
    "moment_prior_",
    "moment_current_",
    "prior_",
    "corrected_",
    "pred_",
)

RESIDUAL_PREFIXES = (
    "force_prior_residual_",
    "force_corrected_residual_",
    "moment_prior_residual_",
    "moment_current_residual_",
)


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

    u_b = r00 * vn + r10 * ve + r20 * vd
    v_b = r01 * vn + r11 * ve + r21 * vd
    w_b = r02 * vn + r12 * ve + r22 * vd
    return (
        pd.Series(u_b, index=frame.index, dtype=float),
        pd.Series(v_b, index=frame.index, dtype=float),
        pd.Series(w_b, index=frame.index, dtype=float),
    )


def _derive_alpha_rad(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    if "alpha_rad" in frame.columns:
        return _numeric_series(frame, "alpha_rad"), "alpha_rad"
    body_velocity = _body_velocity_from_attitude(frame)
    if body_velocity is not None:
        u_b, _, w_b = body_velocity
        return pd.Series(np.arctan2(-w_b.to_numpy(dtype=float), u_b.to_numpy(dtype=float)), index=frame.index), (
            "body_air_relative_velocity"
        )
    column = _first_existing(("airspeed_validated.pitch_filtered", "vehicle_local_position.pitch", "pitch_rad"), frame)
    if column is not None:
        return _numeric_series(frame, column), column
    return pd.Series(0.0, index=frame.index, dtype=float), "zero_fallback"


def _derive_body_air_velocity(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, str]:
    direct_names = (
        ("air_relative_velocity_b_x", "air_relative_velocity_b_y", "air_relative_velocity_b_z"),
        ("v_air_b_x", "v_air_b_y", "v_air_b_z"),
        ("body_air_relative_velocity_x", "body_air_relative_velocity_y", "body_air_relative_velocity_z"),
        ("air_relative_velocity_b.x", "air_relative_velocity_b.y", "air_relative_velocity_b.z"),
    )
    for x_col, y_col, z_col in direct_names:
        if all(column in frame.columns for column in (x_col, y_col, z_col)):
            return _numeric_series(frame, x_col), _numeric_series(frame, y_col), _numeric_series(frame, z_col), (
                f"{x_col},{y_col},{z_col}"
            )
    body_velocity = _body_velocity_from_attitude(frame)
    if body_velocity is not None:
        return (*body_velocity, "attitude_local_velocity_wind")
    zeros = pd.Series(0.0, index=frame.index, dtype=float)
    return zeros, zeros, zeros, "zero_fallback"


def _copy_if_present(output: pd.DataFrame, frame: pd.DataFrame, source: str, target: str) -> None:
    if source in frame.columns:
        output[target] = _numeric_series(frame, source)


def build_candidate_variables(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build physically motivated residual association variables."""

    spec: dict[str, object] = {"warnings": []}
    phase_column = _first_existing(
        ("phase_corrected_rad", "wing_phase.phase_rad", "drive_phase_rad", "encoder_phase_rad", "phase_raw_rad"),
        frame,
    )
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

    if phase_column is None:
        spec["warnings"].append("phase column missing; phase harmonics use zero fallback")
    if frequency_column is None:
        spec["warnings"].append("flap-frequency column missing; frequency variables use zero fallback")
    if airspeed_column is None:
        spec["warnings"].append("airspeed column missing; airspeed variables use zero fallback")

    phase = _numeric_series(frame, phase_column)
    flap_frequency = _numeric_series(frame, frequency_column)
    if frequency_column == "encoder_rpm_est":
        flap_frequency = flap_frequency / 60.0
    true_airspeed = _numeric_series(frame, airspeed_column)
    rho = _numeric_series(frame, density_column, 1.225)
    dynamic_pressure = (
        _numeric_series(frame, "dynamic_pressure_pa")
        if "dynamic_pressure_pa" in frame.columns
        else 0.5 * rho * true_airspeed * true_airspeed
    )
    alpha_rad, alpha_source = _derive_alpha_rad(frame)
    v_air_b_x, v_air_b_y, v_air_b_z, beta_source = _derive_body_air_velocity(frame)
    beta_proxy = pd.Series(
        np.arctan2(v_air_b_y.to_numpy(dtype=float), np.maximum(np.abs(v_air_b_x.to_numpy(dtype=float)), 1.0e-6)),
        index=frame.index,
        dtype=float,
    )
    if beta_source == "zero_fallback":
        spec["warnings"].append("body-air velocity missing; beta proxy uses zero fallback")

    variables = pd.DataFrame(index=frame.index)
    variables["phase_sin_1"] = np.sin(phase)
    variables["phase_cos_1"] = np.cos(phase)
    variables["phase_sin_2"] = np.sin(2.0 * phase)
    variables["phase_cos_2"] = np.cos(2.0 * phase)
    variables["flap_frequency_hz"] = flap_frequency
    variables["true_airspeed_m_s"] = true_airspeed
    variables["dynamic_pressure_pa"] = dynamic_pressure
    variables["alpha_rad"] = alpha_rad
    variables["beta_proxy_rad"] = beta_proxy
    variables["v_air_b_x"] = v_air_b_x
    variables["v_air_b_y"] = v_air_b_y
    variables["v_air_b_z"] = v_air_b_z

    _copy_if_present(variables, frame, "vehicle_angular_velocity.xyz[0]", "body_rate_p")
    _copy_if_present(variables, frame, "vehicle_angular_velocity.xyz[1]", "body_rate_q")
    _copy_if_present(variables, frame, "vehicle_angular_velocity.xyz[2]", "body_rate_r")
    for target, candidates in {
        "body_rate_p": ("body_rate_p", "p_rad_s", "roll_rate_rad_s"),
        "body_rate_q": ("body_rate_q", "q_rad_s", "pitch_rate_rad_s"),
        "body_rate_r": ("body_rate_r", "r_rad_s", "yaw_rate_rad_s"),
    }.items():
        if target not in variables:
            variables[target] = _numeric_series(frame, _first_existing(candidates, frame))

    servo_columns: list[tuple[str, str]] = []
    for column in frame.columns:
        match = re.fullmatch(r"actuator_servos\.servo\[(\d+)\]", column)
        if match:
            servo_columns.append((f"servo_{int(match.group(1))}", column))
    named_servo_candidates = (
        ("servo_left_elevon", "servo_left_elevon"),
        ("servo_right_elevon", "servo_right_elevon"),
        ("servo_rudder", "servo_rudder"),
        ("motor_cmd_0", "motor_cmd_0"),
    )
    servo_columns.extend((target, source) for target, source in named_servo_candidates if source in frame.columns)
    for target, source in sorted(servo_columns):
        variables[target] = _numeric_series(frame, source)

    if {"servo_0", "servo_1"}.issubset(variables.columns):
        variables["elevon_sum_proxy"] = variables["servo_0"] + variables["servo_1"]
        variables["elevon_diff_proxy"] = variables["servo_0"] - variables["servo_1"]
    elif {"servo_left_elevon", "servo_right_elevon"}.issubset(variables.columns):
        variables["elevon_sum_proxy"] = variables["servo_left_elevon"] + variables["servo_right_elevon"]
        variables["elevon_diff_proxy"] = variables["servo_left_elevon"] - variables["servo_right_elevon"]

    variables["q_dyn_x_beta_proxy"] = dynamic_pressure * beta_proxy
    for rate in ("body_rate_p", "body_rate_q", "body_rate_r"):
        variables[f"q_dyn_x_{rate}"] = dynamic_pressure * variables[rate]
    for column in [column for column in variables.columns if column.startswith("servo_")]:
        variables[f"q_dyn_x_{column}"] = dynamic_pressure * variables[column]
    variables["alpha_rad_x_phase_sin_1"] = alpha_rad * variables["phase_sin_1"]
    variables["alpha_rad_x_phase_cos_1"] = alpha_rad * variables["phase_cos_1"]
    variables["beta_proxy_x_phase_sin_1"] = beta_proxy * variables["phase_sin_1"]
    variables["beta_proxy_x_phase_cos_1"] = beta_proxy * variables["phase_cos_1"]
    variables["flap_frequency_hz_x_phase_sin_1"] = flap_frequency * variables["phase_sin_1"]
    variables["flap_frequency_hz_x_phase_cos_1"] = flap_frequency * variables["phase_cos_1"]

    spec.update(
        {
            "phase_column": phase_column,
            "frequency_column": frequency_column,
            "airspeed_column": airspeed_column,
            "density_column": density_column,
            "alpha_source": alpha_source,
            "body_air_velocity_source": beta_source,
            "candidate_columns": variables.columns.tolist(),
        }
    )
    return variables, spec


def _validate_equal_rows(split: str, samples: pd.DataFrame, named_frames: dict[str, pd.DataFrame | None]) -> None:
    expected = len(samples)
    mismatched = {
        name: len(frame)
        for name, frame in named_frames.items()
        if frame is not None and len(frame) != expected
    }
    if mismatched:
        raise ValueError(f"{split} row-count mismatch: samples={expected}, others={mismatched}")


def _prediction_series(
    frames: tuple[pd.DataFrame, ...],
    candidates: tuple[str, ...],
    *,
    index: pd.Index,
    default: float | None = None,
) -> tuple[pd.Series, str | None]:
    for frame in frames:
        for column in candidates:
            if column in frame.columns:
                return pd.to_numeric(frame[column], errors="coerce").astype(float).reset_index(drop=True), column
    if default is None:
        raise ValueError(f"missing prediction column; tried {list(candidates)}")
    return pd.Series(default, index=index, dtype=float), None


def _residual_difference(label: pd.Series, prediction: pd.Series) -> pd.Series:
    return (label - prediction).round(12)


def _rmse(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite * finite)))


def _mae(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(np.abs(finite))) if len(finite) else float("nan")


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def _round_float(value: float) -> float:
    return round(float(value), 12) if np.isfinite(value) else float(value)


def _parse_residual_column(column: str) -> tuple[str, str]:
    for prefix in RESIDUAL_PREFIXES:
        if column.startswith(prefix):
            return prefix.removesuffix("_residual_"), column[len(prefix) :]
    return "unknown", column


def _quantile_bin_codes(values: pd.Series, quantile_bins: int) -> pd.Series:
    finite_mask = np.isfinite(values.to_numpy(dtype=float))
    finite = values.loc[finite_mask]
    if finite.nunique(dropna=True) < 2:
        return pd.Series(pd.NA, index=values.index, dtype="Int64")
    ranked = finite.rank(method="first")
    bins = pd.qcut(ranked, q=min(quantile_bins, int(finite.nunique(dropna=True))), labels=False, duplicates="drop")
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    result.loc[finite.index] = bins.astype("Int64")
    return result


def residual_variable_bin_table(
    frame: pd.DataFrame,
    *,
    residual_columns: tuple[str, ...],
    variable_columns: tuple[str, ...],
    quantile_bins: int,
    min_samples: int,
) -> pd.DataFrame:
    """Summarize residual distributions across candidate-variable quantile bins."""

    if quantile_bins <= 0:
        raise ValueError("quantile_bins must be positive")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    missing = [column for column in (*residual_columns, *variable_columns) if column not in frame.columns]
    if missing:
        raise ValueError(f"residual bin table missing columns: {missing}")

    split_groups = frame.groupby("split", observed=True) if "split" in frame.columns else [("all", frame)]
    rows: list[dict[str, float | int | str]] = []
    for split, split_frame in split_groups:
        for variable in variable_columns:
            values = pd.to_numeric(split_frame[variable], errors="coerce").astype(float)
            bin_codes = _quantile_bin_codes(values, quantile_bins)
            for bin_id in sorted(code for code in bin_codes.dropna().unique().tolist()):
                variable_mask = bin_codes == bin_id
                variable_values = values.loc[variable_mask].to_numpy(dtype=float)
                finite_variable_values = variable_values[np.isfinite(variable_values)]
                if len(finite_variable_values) < min_samples:
                    continue
                for residual_column in residual_columns:
                    residual_values = pd.to_numeric(
                        split_frame.loc[variable_mask, residual_column], errors="coerce"
                    ).to_numpy(dtype=float)
                    finite_mask = np.isfinite(residual_values)
                    if int(np.sum(finite_mask)) < min_samples:
                        continue
                    residual_kind, target = _parse_residual_column(residual_column)
                    rows.append(
                        {
                            "split": str(split),
                            "residual_kind": residual_kind,
                            "target": target,
                            "residual_column": residual_column,
                            "variable": variable,
                            "bin": int(bin_id),
                            "sample_count": int(np.sum(finite_mask)),
                            "variable_min": _round_float(np.min(finite_variable_values)),
                            "variable_max": _round_float(np.max(finite_variable_values)),
                            "variable_median": _round_float(np.median(finite_variable_values)),
                            "residual_bias": _finite_mean(residual_values),
                            "residual_mae": _mae(residual_values),
                            "residual_rmse": _rmse(residual_values),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_residual_variable_bins(bin_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize across-bin residual variation for each residual-variable pair."""

    if bin_table.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | int | str]] = []
    group_columns = ["split", "residual_kind", "target", "residual_column", "variable"]
    for keys, group in bin_table.groupby(group_columns, observed=True):
        split, residual_kind, target, residual_column, variable = keys
        ordered = group.sort_values("residual_rmse", ascending=False)
        worst = ordered.iloc[0]
        rows.append(
            {
                "split": str(split),
                "residual_kind": str(residual_kind),
                "target": str(target),
                "residual_column": str(residual_column),
                "variable": str(variable),
                "bin_count": int(group["bin"].nunique()),
                "sample_count": int(group["sample_count"].sum()),
                "residual_rmse_min": float(group["residual_rmse"].min()),
                "residual_rmse_max": float(group["residual_rmse"].max()),
                "residual_rmse_range": float(group["residual_rmse"].max() - group["residual_rmse"].min()),
                "residual_bias_min": float(group["residual_bias"].min()),
                "residual_bias_max": float(group["residual_bias"].max()),
                "worst_bin": int(worst["bin"]),
                "worst_bin_variable_median": float(worst["variable_median"]),
                "worst_bin_residual_rmse": float(worst["residual_rmse"]),
            }
        )
    return pd.DataFrame(rows)


def _corr(x: np.ndarray, y: np.ndarray, *, method: str) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        return float("nan")
    x_finite = x[mask]
    y_finite = y[mask]
    if np.nanstd(x_finite) <= 1.0e-12 or np.nanstd(y_finite) <= 1.0e-12:
        return float("nan")
    if method == "spearman":
        x_finite = pd.Series(x_finite).rank(method="average").to_numpy(dtype=float)
        y_finite = pd.Series(y_finite).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(x_finite, y_finite)[0, 1])


def _mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan")
    if int(np.sum(mask)) > DEFAULT_MI_MAX_SAMPLES:
        mask_indices = np.flatnonzero(mask)
        rng = np.random.default_rng(0)
        keep = rng.choice(mask_indices, size=DEFAULT_MI_MAX_SAMPLES, replace=False)
        sample_mask = np.zeros_like(mask, dtype=bool)
        sample_mask[keep] = True
        mask = sample_mask
    try:
        from sklearn.feature_selection import mutual_info_regression
    except Exception:
        return float("nan")
    try:
        value = mutual_info_regression(x[mask].reshape(-1, 1), y[mask], random_state=0)[0]
    except Exception:
        return float("nan")
    return float(value)


def residual_variable_ranking_table(
    frame: pd.DataFrame,
    *,
    residual_columns: tuple[str, ...],
    variable_columns: tuple[str, ...],
    group_columns: tuple[str, ...] = (),
    min_samples: int,
) -> pd.DataFrame:
    """Rank candidate variables by residual association diagnostics."""

    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    missing = [column for column in (*residual_columns, *variable_columns, *group_columns) if column not in frame.columns]
    if missing:
        raise ValueError(f"residual ranking table missing columns: {missing}")

    grouped = frame.groupby(list(group_columns), observed=True) if group_columns else [((), frame)]
    rows: list[dict[str, float | int | str]] = []
    for group_key, group in grouped:
        if group_columns and not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_values = dict(zip(group_columns, group_key if group_columns else ()))
        split = str(group["split"].iloc[0]) if "split" in group.columns and group["split"].nunique(dropna=False) == 1 else "all"
        for residual_column in residual_columns:
            residual_values = pd.to_numeric(group[residual_column], errors="coerce").to_numpy(dtype=float)
            residual_kind, target = _parse_residual_column(residual_column)
            for variable in variable_columns:
                variable_values = pd.to_numeric(group[variable], errors="coerce").to_numpy(dtype=float)
                sample_count = int(np.sum(np.isfinite(residual_values) & np.isfinite(variable_values)))
                if sample_count < min_samples:
                    continue
                pearson = _corr(variable_values, residual_values, method="pearson")
                spearman = _corr(variable_values, residual_values, method="spearman")
                mi = _mutual_information(variable_values, residual_values)
                abs_pearson = abs(pearson) if np.isfinite(pearson) else float("nan")
                abs_spearman = abs(spearman) if np.isfinite(spearman) else float("nan")
                score_parts = [value for value in (abs_pearson, abs_spearman, mi) if np.isfinite(value)]
                row: dict[str, float | int | str] = {
                    **{key: str(value) for key, value in group_values.items()},
                    "split": split,
                    "residual_kind": residual_kind,
                    "target": target,
                    "residual_column": residual_column,
                    "variable": variable,
                    "sample_count": sample_count,
                    "pearson": pearson,
                    "spearman": spearman,
                    "mutual_information": mi,
                    "abs_pearson": abs_pearson,
                    "abs_spearman": abs_spearman,
                    "combined_rank_score": float(np.mean(score_parts)) if score_parts else float("nan"),
                }
                rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["combined_rank_score", "abs_spearman", "abs_pearson"], ascending=False).reset_index(
        drop=True
    )


def default_feature_groups(variable_columns: Iterable[str]) -> dict[str, list[str]]:
    """Return transparent candidate feature groups from available variable columns."""

    columns = list(dict.fromkeys(variable_columns))

    def containing(*needles: str) -> list[str]:
        return [column for column in columns if any(needle in column for needle in needles)]

    def exact(*names: str) -> list[str]:
        wanted = set(names)
        return [column for column in columns if column in wanted]

    groups = {
        "phase": [column for column in columns if re.fullmatch(r"phase_(sin|cos)_\d+", column)],
        "longitudinal": exact(
            "alpha_rad",
            "true_airspeed_m_s",
            "dynamic_pressure_pa",
            "flap_frequency_hz",
            "v_air_b_x",
            "v_air_b_z",
        ),
        "lateral_body": exact("beta_proxy_rad", "v_air_b_y", "q_dyn_x_beta_proxy"),
        "body_rates": exact(
            "body_rate_p",
            "body_rate_q",
            "body_rate_r",
            "q_dyn_x_body_rate_p",
            "q_dyn_x_body_rate_q",
            "q_dyn_x_body_rate_r",
        ),
        "tail_controls": containing("servo_", "elevon_", "rudder", "motor_cmd"),
        "phase_interactions": containing("_x_phase_sin_1", "_x_phase_cos_1"),
        "lateral_tail": containing("q_dyn_x_beta_proxy", "beta_proxy_x_", "elevon_diff_proxy", "servo_rudder"),
        "all_candidate": columns,
    }
    return {name: list(dict.fromkeys(group_columns)) for name, group_columns in groups.items()}


def _split_xy(
    frame: pd.DataFrame,
    *,
    split: str,
    residual_column: str,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    subset = frame.loc[frame["split"].astype(str).eq(split)].copy()
    y = pd.to_numeric(subset[residual_column], errors="coerce").to_numpy(dtype=float)
    if feature_columns:
        x = subset.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    else:
        x = np.empty((len(subset), 0), dtype=float)
    return x, y, subset


def _fit_ridge_standardized(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    finite_y = np.isfinite(y)
    x = x[finite_y]
    y = y[finite_y]
    fill = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    x_filled = np.where(np.isfinite(x), x, fill)
    mean = np.mean(x_filled, axis=0)
    scale = np.std(x_filled, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    x_scaled = (x_filled - mean) / scale
    intercept = np.array([float(np.mean(y))])
    y_centered = y - intercept[0]
    gram = x_scaled.T @ x_scaled
    if alpha > 0.0:
        gram = gram + float(alpha) * np.eye(gram.shape[0])
    rhs = x_scaled.T @ y_centered
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram) @ rhs
    return {
        "coefficients": coefficients,
        "intercept": intercept,
        "fill": fill,
        "mean": mean,
        "scale": scale,
    }


def _predict_ridge(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros(x.shape[0], dtype=float)
    x_filled = np.where(np.isfinite(x), x, model["fill"])
    x_scaled = (x_filled - model["mean"]) / model["scale"]
    return x_scaled @ model["coefficients"] + model["intercept"][0]


def _zero_metrics(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    return {
        "zero_train_rmse": _rmse(y_train),
        "zero_val_rmse": _rmse(y_val),
        "zero_test_rmse": _rmse(y_test),
    }


def _ablation_metric_row(
    *,
    residual_column: str,
    feature_group: str,
    selected_alpha: float,
    feature_columns: list[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    pred_train: np.ndarray,
    pred_val: np.ndarray,
    pred_test: np.ndarray,
) -> dict[str, float | int | str]:
    residual_kind, target = _parse_residual_column(residual_column)
    train_rmse = _rmse(y_train - pred_train)
    val_rmse = _rmse(y_val - pred_val)
    test_rmse = _rmse(y_test - pred_test)
    zero = _zero_metrics(y_train, y_val, y_test)
    zero_test = zero["zero_test_rmse"]
    test_mse = test_rmse * test_rmse
    zero_mse = zero_test * zero_test
    return {
        "residual_kind": residual_kind,
        "target": target,
        "residual_column": residual_column,
        "feature_group": feature_group,
        "selected_alpha": float(selected_alpha),
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
        **zero,
        "test_rmse_reduction_fraction": float(1.0 - test_rmse / zero_test) if zero_test > 0.0 else float("nan"),
        "test_residual_r2": float(1.0 - test_mse / zero_mse) if zero_mse > 0.0 else float("nan"),
        "n_features": int(len(feature_columns)),
        "feature_columns": ",".join(feature_columns),
    }


def residual_feature_group_ablation(
    frame: pd.DataFrame,
    *,
    residual_columns: tuple[str, ...],
    feature_groups: dict[str, list[str]],
    alphas: tuple[float, ...],
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate simple feature-group residual explainability on held-out splits."""

    if "split" not in frame.columns:
        raise ValueError("frame must include a split column")
    if not alphas:
        raise ValueError("at least one alpha is required")
    missing_residuals = [column for column in residual_columns if column not in frame.columns]
    if missing_residuals:
        raise ValueError(f"missing residual columns: {missing_residuals}")

    aggregate_rows: list[dict[str, float | int | str]] = []
    per_log_rows: list[dict[str, float | int | str]] = []
    for residual_column in residual_columns:
        _, y_train, train_subset = _split_xy(frame, split=train_split, residual_column=residual_column, feature_columns=[])
        _, y_val, val_subset = _split_xy(frame, split=val_split, residual_column=residual_column, feature_columns=[])
        _, y_test, test_subset = _split_xy(frame, split=test_split, residual_column=residual_column, feature_columns=[])
        zero_pred_train = np.zeros(len(y_train), dtype=float)
        zero_pred_val = np.zeros(len(y_val), dtype=float)
        zero_pred_test = np.zeros(len(y_test), dtype=float)
        aggregate_rows.append(
            _ablation_metric_row(
                residual_column=residual_column,
                feature_group="zero_residual",
                selected_alpha=float("nan"),
                feature_columns=[],
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                pred_train=zero_pred_train,
                pred_val=zero_pred_val,
                pred_test=zero_pred_test,
            )
        )

        for group_name, raw_columns in feature_groups.items():
            feature_columns = [column for column in raw_columns if column in frame.columns]
            if not feature_columns:
                continue
            x_train, y_train, train_subset = _split_xy(
                frame, split=train_split, residual_column=residual_column, feature_columns=feature_columns
            )
            x_val, y_val, val_subset = _split_xy(
                frame, split=val_split, residual_column=residual_column, feature_columns=feature_columns
            )
            x_test, y_test, test_subset = _split_xy(
                frame, split=test_split, residual_column=residual_column, feature_columns=feature_columns
            )
            if len(train_subset) == 0 or len(val_subset) == 0 or len(test_subset) == 0:
                continue
            best_alpha = float(alphas[0])
            best_model: dict[str, np.ndarray] | None = None
            best_val_rmse = float("inf")
            for alpha in alphas:
                model = _fit_ridge_standardized(x_train, y_train, float(alpha))
                pred_val = _predict_ridge(model, x_val)
                val_rmse = _rmse(y_val - pred_val)
                if np.isfinite(val_rmse) and val_rmse < best_val_rmse:
                    best_alpha = float(alpha)
                    best_model = model
                    best_val_rmse = val_rmse
            if best_model is None:
                continue
            pred_train = _predict_ridge(best_model, x_train)
            pred_val = _predict_ridge(best_model, x_val)
            pred_test = _predict_ridge(best_model, x_test)
            aggregate_rows.append(
                _ablation_metric_row(
                    residual_column=residual_column,
                    feature_group=group_name,
                    selected_alpha=best_alpha,
                    feature_columns=feature_columns,
                    y_train=y_train,
                    y_val=y_val,
                    y_test=y_test,
                    pred_train=pred_train,
                    pred_val=pred_val,
                    pred_test=pred_test,
                )
            )

            if "log_id" in test_subset.columns:
                test_eval = test_subset.loc[:, ["log_id"]].copy()
                test_eval["_y"] = y_test
                test_eval["_pred"] = pred_test
                for log_id, log_group in test_eval.groupby("log_id", observed=True):
                    y_log = log_group["_y"].to_numpy(dtype=float)
                    pred_log = log_group["_pred"].to_numpy(dtype=float)
                    zero_log = _rmse(y_log)
                    test_log_rmse = _rmse(y_log - pred_log)
                    row = _ablation_metric_row(
                        residual_column=residual_column,
                        feature_group=group_name,
                        selected_alpha=best_alpha,
                        feature_columns=feature_columns,
                        y_train=y_train,
                        y_val=y_val,
                        y_test=y_log,
                        pred_train=pred_train,
                        pred_val=pred_val,
                        pred_test=pred_log,
                    )
                    row["log_id"] = str(log_id)
                    row["sample_count"] = int(len(log_group))
                    row["zero_test_rmse"] = zero_log
                    row["test_rmse"] = test_log_rmse
                    row["test_rmse_reduction_fraction"] = (
                        float(1.0 - test_log_rmse / zero_log) if zero_log > 0.0 else float("nan")
                    )
                    row["test_residual_r2"] = (
                        float(1.0 - (test_log_rmse * test_log_rmse) / (zero_log * zero_log))
                        if zero_log > 0.0
                        else float("nan")
                    )
                    per_log_rows.append(row)

    return pd.DataFrame(aggregate_rows), pd.DataFrame(per_log_rows)


def residual_columns_for_frame(frame: pd.DataFrame) -> tuple[str, ...]:
    columns = [
        column
        for column in frame.columns
        if any(column.startswith(prefix) for prefix in RESIDUAL_PREFIXES)
    ]
    return tuple(columns)


def candidate_columns_for_frame(frame: pd.DataFrame) -> tuple[str, ...]:
    columns: list[str] = []
    excluded = set(METADATA_COLUMNS)
    for column in frame.columns:
        if column in excluded:
            continue
        if any(column.startswith(prefix) for prefix in (*RESIDUAL_PREFIXES, *PREDICTION_PREFIXES)):
            continue
        if column in ALL_TARGETS:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return tuple(columns)


def _prediction_path(root: Path, split: str) -> Path:
    nested = root / "prediction_parquets" / f"{split}_predictions.parquet"
    if nested.exists():
        return nested
    direct = root / f"{split}_predictions.parquet"
    if direct.exists():
        return direct
    raise FileNotFoundError(f"could not find predictions for split {split!r} under {root}")


def load_residual_frames(
    *,
    split_root: Path,
    force_prediction_root: Path,
    moment_prediction_root: Path,
    prior_root: Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> tuple[pd.DataFrame, dict[str, object]]:
    frames: list[pd.DataFrame] = []
    split_specs: dict[str, object] = {}
    for split in splits:
        samples_path = split_root / f"{split}_samples.parquet"
        force_path = _prediction_path(force_prediction_root, split)
        prior_path = _prediction_path(prior_root, split)
        moment_path = _prediction_path(moment_prediction_root, split)
        residual = build_residual_frame(
            split=split,
            samples=pd.read_parquet(samples_path),
            force_predictions=pd.read_parquet(force_path),
            prior_predictions=pd.read_parquet(prior_path),
            current_moment_predictions=pd.read_parquet(moment_path),
        )
        split_specs[split] = {
            "samples_path": str(samples_path),
            "force_prediction_path": str(force_path),
            "prior_prediction_path": str(prior_path),
            "moment_prediction_path": str(moment_path),
            "candidate_variable_spec": residual.attrs.get("candidate_variable_spec", {}),
            "sample_count": int(len(residual)),
        }
        frames.append(residual)
    return pd.concat(frames, ignore_index=True), split_specs


def _parse_csv_tuple(raw: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        return default
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("comma-separated list cannot be empty")
    return values


def _parse_alpha_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("alpha list cannot be empty")
    return values


def dataframe_to_markdown(frame: pd.DataFrame, *, max_rows: int = 12) -> str:
    if frame.empty:
        return "_No rows met the reporting thresholds._"
    display = frame.head(max_rows).copy()
    columns = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in display.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _chosen_residual_kind(target: str) -> str:
    return "force_corrected" if target in FORCE_TARGETS else "moment_current"


def _plot_group_ablation(ablation: pd.DataFrame, output_path: Path, *, key_targets: tuple[str, ...]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for target in key_targets:
        kind = _chosen_residual_kind(target)
        subset = ablation.loc[
            (ablation["target"].eq(target))
            & (ablation["residual_kind"].eq(kind))
            & (~ablation["feature_group"].eq("zero_residual"))
        ].copy()
        if subset.empty:
            continue
        rows.append(subset.sort_values("test_rmse_reduction_fraction", ascending=False).iloc[0])
    if not rows:
        raise ValueError("no key-target ablation rows available to plot")
    plot_frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(plot_frame))
    ax.bar(x, plot_frame["test_rmse_reduction_fraction"].to_numpy(dtype=float), color="#0072B2")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{row.target}\n{row.feature_group}" for row in plot_frame.itertuples()], rotation=0
    )
    ax.set_ylabel("test RMSE reduction vs zero residual")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def _plot_rank_heatmap(rankings: pd.DataFrame, output_path: Path, *, key_targets: tuple[str, ...]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset_rows = []
    for target in key_targets:
        kind = _chosen_residual_kind(target)
        subset = rankings.loc[(rankings["target"].eq(target)) & (rankings["residual_kind"].eq(kind))]
        subset_rows.append(subset.sort_values("combined_rank_score", ascending=False).head(5))
    subset = pd.concat([row for row in subset_rows if not row.empty], ignore_index=True) if subset_rows else pd.DataFrame()
    if subset.empty:
        raise ValueError("no key-target ranking rows available to plot")
    variables = list(dict.fromkeys(subset["variable"].astype(str).tolist()))
    labels = [f"{_chosen_residual_kind(target)}:{target}" for target in key_targets]
    matrix = np.full((len(labels), len(variables)), np.nan)
    for row in subset.itertuples():
        label = f"{row.residual_kind}:{row.target}"
        if label in labels and row.variable in variables:
            matrix[labels.index(label), variables.index(row.variable)] = max(
                abs(float(row.pearson)) if np.isfinite(float(row.pearson)) else 0.0,
                abs(float(row.spearman)) if np.isfinite(float(row.spearman)) else 0.0,
            )
    fig, ax = plt.subplots(figsize=(max(7.2, 0.42 * len(variables)), 3.6))
    image = ax.imshow(np.nan_to_num(matrix, nan=0.0), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(variables)))
    ax.set_xticklabels(variables, rotation=55, ha="right")
    fig.colorbar(image, ax=ax, label="max |correlation|")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def _plot_bins(bin_table: pd.DataFrame, rankings: pd.DataFrame, output_path: Path, *, key_targets: tuple[str, ...]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs: list[tuple[str, str, str]] = []
    for target in key_targets:
        kind = _chosen_residual_kind(target)
        subset = rankings.loc[(rankings["target"].eq(target)) & (rankings["residual_kind"].eq(kind))]
        if subset.empty:
            continue
        variable = str(subset.sort_values("combined_rank_score", ascending=False).iloc[0]["variable"])
        pairs.append((kind, target, variable))
    if not pairs:
        raise ValueError("no key-target bin pairs available to plot")
    fig, axes = plt.subplots(len(pairs), 1, figsize=(7.2, 2.0 * len(pairs)), squeeze=False)
    for idx, (kind, target, variable) in enumerate(pairs):
        ax = axes[idx, 0]
        subset = bin_table.loc[
            (bin_table["split"].eq("test"))
            & (bin_table["residual_kind"].eq(kind))
            & (bin_table["target"].eq(target))
            & (bin_table["variable"].eq(variable))
        ].sort_values("bin")
        if subset.empty:
            ax.set_axis_off()
            continue
        ax.plot(
            subset["variable_median"].to_numpy(dtype=float),
            subset["residual_rmse"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.3,
            color="#D55E00",
        )
        ax.set_title(f"{kind}:{target} vs {variable}")
        ax.set_xlabel("bin median")
        ax.set_ylabel("residual RMSE")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def _write_plots(
    *,
    bin_table: pd.DataFrame,
    rankings: pd.DataFrame,
    ablation: pd.DataFrame,
    figures_dir: Path,
    key_targets: tuple[str, ...],
) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []
    plot_specs = (
        ("residual_group_ablation_key_targets.png", lambda path: _plot_group_ablation(ablation, path, key_targets=key_targets)),
        ("residual_rank_heatmap_key_targets.png", lambda path: _plot_rank_heatmap(rankings, path, key_targets=key_targets)),
        ("residual_bins_key_targets.png", lambda path: _plot_bins(bin_table, rankings, path, key_targets=key_targets)),
    )
    for filename, plotter in plot_specs:
        try:
            plotter(figures_dir / filename)
        except Exception as exc:
            skipped.append(f"{filename}: {exc}")
    return skipped


def _top_associations(rankings: pd.DataFrame, *, key_targets: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for target in key_targets:
        kind = _chosen_residual_kind(target)
        subset = rankings.loc[(rankings["target"].eq(target)) & (rankings["residual_kind"].eq(kind))]
        if not subset.empty:
            rows.append(subset.sort_values("combined_rank_score", ascending=False).head(5))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).loc[
        :, ["residual_kind", "target", "variable", "sample_count", "pearson", "spearman", "mutual_information", "combined_rank_score"]
    ]


def _best_feature_groups(ablation: pd.DataFrame, *, key_targets: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for target in key_targets:
        kind = _chosen_residual_kind(target)
        subset = ablation.loc[
            (ablation["target"].eq(target))
            & (ablation["residual_kind"].eq(kind))
            & (~ablation["feature_group"].eq("zero_residual"))
        ]
        if not subset.empty:
            rows.append(subset.sort_values("test_rmse_reduction_fraction", ascending=False).head(3))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).loc[
        :,
        [
            "residual_kind",
            "target",
            "feature_group",
            "selected_alpha",
            "test_rmse",
            "zero_test_rmse",
            "test_rmse_reduction_fraction",
            "test_residual_r2",
            "n_features",
        ],
    ]


def _write_readme(
    output_root: Path,
    *,
    split_root: Path,
    force_prediction_root: Path,
    moment_prediction_root: Path,
    prior_root: Path,
    rankings: pd.DataFrame,
    ablation: pd.DataFrame,
    key_targets: tuple[str, ...],
    skipped_plots: list[str],
) -> None:
    top = _top_associations(rankings, key_targets=key_targets)
    best_groups = _best_feature_groups(ablation, key_targets=key_targets)
    skipped_text = "\n".join(f"- {item}" for item in skipped_plots) if skipped_plots else "- None"
    text = f"""# Component Residual Attribution Diagnostic

This diagnostic reports observational residual attribution from held-out logs. It identifies variables and feature groups associated with residual structure and candidate mismatch sources; it does not establish strict causality.

## Residual Definitions

- `force_prior_residual = label_force - prior_force`
- `force_corrected_residual = label_force - corrected_force`
- `moment_prior_residual = label_moment - prior_moment`
- `moment_current_residual = label_moment - current_moment_prediction`

## Data Roots

- Split root: `{split_root}`
- Force correction root: `{force_prediction_root}`
- Moment prediction root: `{moment_prediction_root}`
- Physical prior root: `{prior_root}`

## Top Variable Associations

{dataframe_to_markdown(top)}

## Best Held-Out Feature Groups

{dataframe_to_markdown(best_groups)}

## Plot Status

{skipped_text}

## Limitations

These tables use observational residual attribution and held-out-log residual explainability. The associations are consistent with candidate mismatch sources and can prioritize model revisions, but they do not isolate aerodynamic mechanisms or prove causal effects without controlled interventions.
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def run_component_residual_attribution(
    *,
    split_root: Path = DEFAULT_SPLIT_ROOT,
    force_prediction_root: Path = DEFAULT_FORCE_PREDICTION_ROOT,
    moment_prediction_root: Path = DEFAULT_MOMENT_PREDICTION_ROOT,
    prior_root: Path = DEFAULT_PRIOR_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    quantile_bins: int = 10,
    min_samples: int = 500,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    key_targets: tuple[str, ...] = DEFAULT_KEY_TARGETS,
) -> dict[str, str]:
    residual_frame, split_specs = load_residual_frames(
        split_root=split_root,
        force_prediction_root=force_prediction_root,
        moment_prediction_root=moment_prediction_root,
        prior_root=prior_root,
    )
    residual_columns = residual_columns_for_frame(residual_frame)
    variable_columns = candidate_columns_for_frame(residual_frame)
    feature_groups = default_feature_groups(variable_columns)

    output_root.mkdir(parents=True, exist_ok=True)
    residual_path = output_root / "residual_frame.parquet"
    bins_path = output_root / "residual_variable_bins.csv"
    bin_summary_path = output_root / "residual_variable_bin_summary.csv"
    rankings_path = output_root / "residual_variable_rankings.csv"
    per_log_rankings_path = output_root / "per_log_residual_variable_rankings.csv"
    ablation_path = output_root / "residual_feature_group_ablation.csv"
    per_log_ablation_path = output_root / "per_log_residual_feature_group_ablation.csv"
    config_path = output_root / "residual_attribution_config.json"

    residual_frame.to_parquet(residual_path, index=False)
    bins = residual_variable_bin_table(
        residual_frame,
        residual_columns=residual_columns,
        variable_columns=variable_columns,
        quantile_bins=quantile_bins,
        min_samples=min_samples,
    )
    bin_summary = summarize_residual_variable_bins(bins)
    rankings = residual_variable_ranking_table(
        residual_frame,
        residual_columns=residual_columns,
        variable_columns=variable_columns,
        min_samples=min_samples,
    )
    per_log_rankings = residual_variable_ranking_table(
        residual_frame,
        residual_columns=residual_columns,
        variable_columns=variable_columns,
        group_columns=("log_id",),
        min_samples=min_samples,
    )
    ablation, per_log_ablation = residual_feature_group_ablation(
        residual_frame,
        residual_columns=residual_columns,
        feature_groups=feature_groups,
        alphas=alphas,
    )

    bins.to_csv(bins_path, index=False)
    bin_summary.to_csv(bin_summary_path, index=False)
    rankings.to_csv(rankings_path, index=False)
    per_log_rankings.to_csv(per_log_rankings_path, index=False)
    ablation.to_csv(ablation_path, index=False)
    per_log_ablation.to_csv(per_log_ablation_path, index=False)
    skipped_plots = _write_plots(
        bin_table=bins,
        rankings=rankings,
        ablation=ablation,
        figures_dir=output_root / "figures",
        key_targets=key_targets,
    )
    config = {
        "split_root": str(split_root),
        "force_prediction_root": str(force_prediction_root),
        "moment_prediction_root": str(moment_prediction_root),
        "prior_root": str(prior_root),
        "output_root": str(output_root),
        "quantile_bins": int(quantile_bins),
        "min_samples": int(min_samples),
        "alphas": list(alphas),
        "key_targets": list(key_targets),
        "residual_columns": list(residual_columns),
        "variable_columns": list(variable_columns),
        "feature_groups": feature_groups,
        "split_specs": split_specs,
        "mutual_information_max_samples": DEFAULT_MI_MAX_SAMPLES,
        "skipped_plots": skipped_plots,
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    _write_readme(
        output_root,
        split_root=split_root,
        force_prediction_root=force_prediction_root,
        moment_prediction_root=moment_prediction_root,
        prior_root=prior_root,
        rankings=rankings,
        ablation=ablation,
        key_targets=key_targets,
        skipped_plots=skipped_plots,
    )
    return {
        "residual_frame": str(residual_path),
        "bins": str(bins_path),
        "bin_summary": str(bin_summary_path),
        "rankings": str(rankings_path),
        "per_log_rankings": str(per_log_rankings_path),
        "ablation": str(ablation_path),
        "per_log_ablation": str(per_log_ablation_path),
        "config": str(config_path),
        "readme": str(output_root / "README.md"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--force-prediction-root", type=Path, default=DEFAULT_FORCE_PREDICTION_ROOT)
    parser.add_argument("--moment-prediction-root", type=Path, default=DEFAULT_MOMENT_PREDICTION_ROOT)
    parser.add_argument("--prior-root", type=Path, default=DEFAULT_PRIOR_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--quantile-bins", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=500)
    parser.add_argument("--alphas", type=_parse_alpha_tuple, default=DEFAULT_ALPHAS)
    parser.add_argument("--key-targets", type=lambda raw: _parse_csv_tuple(raw, DEFAULT_KEY_TARGETS), default=DEFAULT_KEY_TARGETS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = run_component_residual_attribution(
        split_root=args.split_root,
        force_prediction_root=args.force_prediction_root,
        moment_prediction_root=args.moment_prediction_root,
        prior_root=args.prior_root,
        output_root=args.output_root,
        quantile_bins=args.quantile_bins,
        min_samples=args.min_samples,
        alphas=args.alphas,
        key_targets=args.key_targets,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


def build_residual_frame(
    *,
    split: str,
    samples: pd.DataFrame,
    force_predictions: pd.DataFrame,
    prior_predictions: pd.DataFrame,
    current_moment_predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a split-aligned frame with explicit prior/current residual definitions."""

    _validate_equal_rows(
        split,
        samples,
        {
            "force_predictions": force_predictions,
            "prior_predictions": prior_predictions,
            "current_moment_predictions": current_moment_predictions,
        },
    )
    samples = samples.reset_index(drop=True)
    force_predictions = force_predictions.reset_index(drop=True)
    prior_predictions = prior_predictions.reset_index(drop=True)
    if current_moment_predictions is not None:
        current_moment_predictions = current_moment_predictions.reset_index(drop=True)

    residual = pd.DataFrame(index=samples.index)
    for column in METADATA_COLUMNS:
        if column == "split":
            residual[column] = split
        elif column in force_predictions.columns:
            residual[column] = force_predictions[column].reset_index(drop=True)
        elif current_moment_predictions is not None and column in current_moment_predictions.columns:
            residual[column] = current_moment_predictions[column].reset_index(drop=True)
        elif column in samples.columns:
            residual[column] = samples[column].reset_index(drop=True)
    if "split" not in residual:
        residual["split"] = split

    variables, spec = build_candidate_variables(samples)
    for column in variables.columns:
        residual[column] = variables[column].reset_index(drop=True)
    residual.attrs["candidate_variable_spec"] = spec

    for target in FORCE_TARGETS:
        label, _ = _prediction_series(
            (force_predictions, samples), (f"label_{target}", target), index=samples.index
        )
        prior, _ = _prediction_series(
            (force_predictions, prior_predictions), (f"prior_{target}", target), index=samples.index
        )
        corrected, _ = _prediction_series(
            (force_predictions,), (f"corrected_{target}", f"pred_{target}"), index=samples.index
        )
        residual[f"label_{target}"] = label
        residual[f"force_prior_{target}"] = prior
        residual[f"force_corrected_{target}"] = corrected
        residual[f"force_prior_residual_{target}"] = _residual_difference(label, prior)
        residual[f"force_corrected_residual_{target}"] = _residual_difference(label, corrected)

    moment_prediction_frames = (current_moment_predictions,) if current_moment_predictions is not None else ()
    for target in MOMENT_TARGETS:
        label, _ = _prediction_series(
            (*moment_prediction_frames, samples), (f"label_{target}", target), index=samples.index
        )
        prior, _ = _prediction_series(
            (prior_predictions,), (f"prior_{target}", target), index=samples.index
        )
        current, _ = _prediction_series(
            moment_prediction_frames,
            (f"pred_{target}", f"current_{target}", target),
            index=samples.index,
            default=np.nan,
        )
        residual[f"label_{target}"] = label
        residual[f"moment_prior_{target}"] = prior
        residual[f"moment_current_{target}"] = current
        residual[f"moment_prior_residual_{target}"] = _residual_difference(label, prior)
        residual[f"moment_current_residual_{target}"] = _residual_difference(label, current)
    return residual


if __name__ == "__main__":
    main()
