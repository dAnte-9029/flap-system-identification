#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_short_horizon_replay import (
    FORCE_COLUMNS,
    MOMENT_COLUMNS,
    OMEGA_COLUMNS,
    QUATERNION_COLUMNS,
    _delta_quaternion_from_body_rate,
    _quat_multiply,
    attitude_error_deg,
    load_mass_properties,
    normalize_quaternion,
    select_replay_windows,
)


ALPHA_COLUMNS = [
    "vehicle_angular_velocity.xyz_derivative[0]",
    "vehicle_angular_velocity.xyz_derivative[1]",
    "vehicle_angular_velocity.xyz_derivative[2]",
]
ALPHA_SMOOTH_COLUMNS = [
    "vehicle_angular_velocity.xyz_derivative_smooth[0]",
    "vehicle_angular_velocity.xyz_derivative_smooth[1]",
    "vehicle_angular_velocity.xyz_derivative_smooth[2]",
]
KINEMATIC_VARIANTS = [
    ("right_multiply_rate_plus", 1.0, "right"),
    ("right_multiply_rate_minus", -1.0, "right"),
    ("left_multiply_rate_plus", 1.0, "left"),
    ("left_multiply_rate_minus", -1.0, "left"),
]
OMEGA_REPLAY_VARIANTS = [
    ("logged_gyro_euler", "logged", "euler", 1),
    ("sim_gyro_euler", "sim", "euler", 1),
    ("logged_gyro_substep4", "logged", "euler", 4),
    ("sim_gyro_substep4", "sim", "euler", 4),
    ("trapezoid_alpha", "logged", "trapezoid_alpha", 1),
]
SUMMARY_QUANTILES = {
    "median": 0.50,
    "p25": 0.25,
    "p75": 0.75,
    "p90": 0.90,
    "p95": 0.95,
}


def _parse_float_list(raw: str) -> list[float]:
    return [float(part) for part in raw.split(",") if part.strip()]


def _metadata_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _load_metadata_dict(metadata_path: str | Path) -> dict[str, Any]:
    with Path(metadata_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _metadata_cg_b(metadata: dict[str, Any]) -> np.ndarray:
    mass_properties = metadata.get("mass_properties", {})
    raw = _metadata_value(mass_properties.get("cg_b_m", [0.0, 0.0, 0.0]))
    cg_b = np.asarray(raw, dtype=float)
    if cg_b.shape != (3,) or not np.isfinite(cg_b).all():
        return np.zeros(3, dtype=float)
    return cg_b


def _sort_samples(samples: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in ["log_id", "segment_id", "time_s"] if column in samples.columns]
    return samples.sort_values(sort_columns).reset_index(drop=True) if sort_columns else samples.reset_index(drop=True)


def _group_columns(frame: pd.DataFrame) -> list[str]:
    columns = ["log_id"] if "log_id" in frame.columns else []
    if "segment_id" in frame.columns:
        columns.append("segment_id")
    return columns


def _iter_groups(frame: pd.DataFrame):
    group_columns = _group_columns(frame)
    if not group_columns:
        yield (), frame.sort_values("time_s")
        return
    for key, group in frame.groupby(group_columns, sort=False, dropna=False):
        yield key, group.sort_values("time_s")


def _array(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return frame[columns].to_numpy(dtype=float, copy=True)


def _alpha_columns(samples: pd.DataFrame) -> list[str]:
    if all(column in samples.columns for column in ALPHA_COLUMNS):
        return ALPHA_COLUMNS
    if all(column in samples.columns for column in ALPHA_SMOOTH_COLUMNS):
        return ALPHA_SMOOTH_COLUMNS
    raise ValueError("samples must contain angular acceleration derivative columns")


def _window(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    return frame.iloc[int(spec["start_idx"]) : int(spec["end_idx"]) + 1]


def integrate_attitude_from_logged_rates(
    window: pd.DataFrame,
    rate_sign: float = 1.0,
    multiply_side: str = "right",
) -> np.ndarray:
    if multiply_side not in {"right", "left"}:
        raise ValueError("multiply_side must be 'right' or 'left'")
    if len(window) < 1:
        raise ValueError("window must contain at least one row")

    time_s = window["time_s"].to_numpy(dtype=float, copy=True)
    omega = _array(window, OMEGA_COLUMNS)
    quat_log = _array(window, QUATERNION_COLUMNS)
    quat = np.zeros((len(window), 4), dtype=float)
    quat[0] = normalize_quaternion(quat_log[0])

    for idx in range(len(window) - 1):
        dt_s = float(time_s[idx + 1] - time_s[idx])
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("window time_s must be strictly increasing")
        dq = _delta_quaternion_from_body_rate(float(rate_sign) * omega[idx], dt_s)
        if multiply_side == "right":
            quat[idx + 1] = normalize_quaternion(_quat_multiply(quat[idx], dq))
        else:
            quat[idx + 1] = normalize_quaternion(_quat_multiply(dq, quat[idx]))
    return quat


def recompute_moment_from_alpha(omega_b: np.ndarray, alpha_b: np.ndarray, inertia_b: np.ndarray) -> np.ndarray:
    omega = np.asarray(omega_b, dtype=float)
    alpha = np.asarray(alpha_b, dtype=float)
    inertia = np.asarray(inertia_b, dtype=float)
    angular_momentum = omega @ inertia.T
    return alpha @ inertia.T + np.cross(omega, angular_momentum)


def infer_alpha_from_moment(omega_b: np.ndarray, moment_b: np.ndarray, inertia_b: np.ndarray) -> np.ndarray:
    omega = np.asarray(omega_b, dtype=float)
    moment = np.asarray(moment_b, dtype=float)
    inertia = np.asarray(inertia_b, dtype=float)
    angular_momentum = omega @ inertia.T
    rhs = moment - np.cross(omega, angular_momentum)
    return rhs @ np.linalg.inv(inertia).T


def apply_moment_reference_transform(
    moment_b: np.ndarray,
    force_b: np.ndarray,
    r_b: np.ndarray,
    mode: str,
) -> np.ndarray:
    moment = np.asarray(moment_b, dtype=float)
    force = np.asarray(force_b, dtype=float)
    r = np.asarray(r_b, dtype=float)
    arm = np.cross(r, force)
    if mode == "none":
        return moment.copy()
    if mode in {"minus_r_cross_f", "minus_cg_cross_force"}:
        return moment - arm
    if mode in {"plus_r_cross_f", "plus_cg_cross_force"}:
        return moment + arm
    if mode == "minus_2x_cg_cross_force":
        return moment - 2.0 * arm
    if mode == "plus_2x_cg_cross_force":
        return moment + 2.0 * arm
    raise ValueError(f"unsupported reference transform mode: {mode}")


def summarize_metric_table(metrics: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=group_columns + ["n_windows"])
    numeric_columns = [
        column
        for column in metrics.select_dtypes(include=[np.number]).columns
        if column not in set(group_columns) and column not in {"start_time_s", "n_steps"}
    ]
    grouped = metrics.groupby(group_columns, sort=True, dropna=False)
    summary = grouped.size().rename("n_windows").reset_index()
    for column in numeric_columns:
        agg = grouped[column].agg(
            median="median",
            p25=lambda values: values.quantile(0.25),
            p75=lambda values: values.quantile(0.75),
            p90=lambda values: values.quantile(0.90),
            p95=lambda values: values.quantile(0.95),
            mean="mean",
        )
        agg = agg.rename(columns={name: f"{column}_{name}" for name in agg.columns})
        summary = summary.merge(agg.reset_index(), on=group_columns, how="left")
    return summary


def evaluate_attitude_kinematic_closure(
    samples: pd.DataFrame,
    horizons_s: list[float],
    stride_s: float,
    split: str,
) -> pd.DataFrame:
    frame = _sort_samples(samples)
    windows = select_replay_windows(frame, horizons_s=horizons_s, stride_s=stride_s)
    rows: list[dict[str, Any]] = []
    for spec in windows:
        window = _window(frame, spec)
        target_quat = window.iloc[-1][QUATERNION_COLUMNS].to_numpy(dtype=float)
        for variant, sign, side in KINEMATIC_VARIANTS:
            q_sim = integrate_attitude_from_logged_rates(window, rate_sign=sign, multiply_side=side)
            rows.append(
                {
                    "diagnostic": "attitude_kinematic_closure",
                    "variant": variant,
                    "split": split,
                    "log_id": spec.get("log_id", ""),
                    "segment_id": spec.get("segment_id", ""),
                    "start_time_s": spec["start_time_s"],
                    "horizon_s": spec["horizon_s"],
                    "n_steps": spec["n_steps"],
                    "attitude_error_deg": attitude_error_deg(q_sim[-1], target_quat),
                }
            )
    return pd.DataFrame(rows)


def evaluate_moment_label_closure(samples: pd.DataFrame, inertia_b: np.ndarray) -> pd.DataFrame:
    alpha_columns = _alpha_columns(samples)
    omega = _array(samples, OMEGA_COLUMNS)
    alpha = _array(samples, alpha_columns)
    moment = _array(samples, MOMENT_COLUMNS)
    recomputed = recompute_moment_from_alpha(omega, alpha, inertia_b)
    alpha_from_moment = infer_alpha_from_moment(omega, moment, inertia_b)
    moment_error = recomputed - moment
    alpha_error = alpha_from_moment - alpha
    axis_specs = [
        ("mx_b", moment_error[:, 0], alpha_error[:, 0]),
        ("my_b", moment_error[:, 1], alpha_error[:, 1]),
        ("mz_b", moment_error[:, 2], alpha_error[:, 2]),
        ("moment_norm", np.linalg.norm(moment_error, axis=1), np.linalg.norm(alpha_error, axis=1)),
    ]
    rows: list[dict[str, Any]] = []
    for axis, error, alpha_axis_error in axis_specs:
        target = moment[:, MOMENT_COLUMNS.index(axis)] if axis in MOMENT_COLUMNS else np.linalg.norm(moment, axis=1)
        predicted = recomputed[:, MOMENT_COLUMNS.index(axis)] if axis in MOMENT_COLUMNS else np.linalg.norm(recomputed, axis=1)
        finite = np.isfinite(error) & np.isfinite(alpha_axis_error)
        corr = np.nan
        if finite.sum() > 1 and np.nanstd(target[finite]) > 0.0 and np.nanstd(predicted[finite]) > 0.0:
            corr = float(np.corrcoef(target[finite], predicted[finite])[0, 1])
        rows.append(
            {
                "diagnostic": "moment_label_closure",
                "axis": axis,
                "rmse": float(np.sqrt(np.nanmean(error[finite] ** 2))) if finite.any() else np.nan,
                "mae": float(np.nanmean(np.abs(error[finite]))) if finite.any() else np.nan,
                "bias": float(np.nanmean(error[finite])) if finite.any() else np.nan,
                "correlation": corr,
                "alpha_rmse": float(np.sqrt(np.nanmean(alpha_axis_error[finite] ** 2))) if finite.any() else np.nan,
                "n": int(finite.sum()),
            }
        )
    return pd.DataFrame(rows)


def integrate_omega_from_moment(
    window: pd.DataFrame,
    inertia_b: np.ndarray,
    gyro_source: str = "logged",
    method: str = "euler",
    substeps: int = 1,
) -> np.ndarray:
    if gyro_source not in {"logged", "sim"}:
        raise ValueError("gyro_source must be 'logged' or 'sim'")
    if method not in {"euler", "trapezoid_alpha"}:
        raise ValueError("unsupported integration method")
    if substeps < 1:
        raise ValueError("substeps must be positive")

    time_s = window["time_s"].to_numpy(dtype=float, copy=True)
    omega_log = _array(window, OMEGA_COLUMNS)
    moment = _array(window, MOMENT_COLUMNS)
    inertia = np.asarray(inertia_b, dtype=float)
    omega_sim = np.zeros_like(omega_log)
    omega_sim[0] = omega_log[0]

    for idx in range(len(window) - 1):
        dt_s = float(time_s[idx + 1] - time_s[idx])
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("window time_s must be strictly increasing")
        if method == "trapezoid_alpha":
            alpha0 = infer_alpha_from_moment(omega_log[idx : idx + 1], moment[idx : idx + 1], inertia)[0]
            alpha1 = infer_alpha_from_moment(omega_log[idx + 1 : idx + 2], moment[idx + 1 : idx + 2], inertia)[0]
            omega_sim[idx + 1] = omega_sim[idx] + 0.5 * (alpha0 + alpha1) * dt_s
            continue
        state = omega_sim[idx].copy()
        for _ in range(substeps):
            gyro = omega_log[idx] if gyro_source == "logged" else state
            alpha = infer_alpha_from_moment(gyro[None, :], moment[idx : idx + 1], inertia)[0]
            state = state + alpha * (dt_s / substeps)
        omega_sim[idx + 1] = state
    return omega_sim


def evaluate_omega_replay(
    samples: pd.DataFrame,
    inertia_b: np.ndarray,
    horizons_s: list[float],
    stride_s: float,
    variants: list[tuple[str, str, str, int]] | None = None,
) -> pd.DataFrame:
    frame = _sort_samples(samples)
    windows = select_replay_windows(frame, horizons_s=horizons_s, stride_s=stride_s)
    variants = OMEGA_REPLAY_VARIANTS if variants is None else variants
    rows: list[dict[str, Any]] = []
    for spec in windows:
        window = _window(frame, spec)
        target = window.iloc[-1][OMEGA_COLUMNS].to_numpy(dtype=float)
        for variant, gyro_source, method, substeps in OMEGA_REPLAY_VARIANTS:
            omega_sim = integrate_omega_from_moment(
                window,
                inertia_b,
                gyro_source=gyro_source,
                method=method,
                substeps=substeps,
            )
            error = omega_sim[-1] - target
            rows.append(
                {
                    "diagnostic": "omega_replay",
                    "variant": variant,
                    "log_id": spec.get("log_id", ""),
                    "segment_id": spec.get("segment_id", ""),
                    "start_time_s": spec["start_time_s"],
                    "horizon_s": spec["horizon_s"],
                    "n_steps": spec["n_steps"],
                    "body_rate_error_rad_s": float(np.linalg.norm(error)),
                    "omega_x_error_rad_s": float(error[0]),
                    "omega_y_error_rad_s": float(error[1]),
                    "omega_z_error_rad_s": float(error[2]),
                }
            )
    return pd.DataFrame(rows)


def _summary_from_omega_metrics(metrics: pd.DataFrame, diagnostic: str) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    summary = summarize_metric_table(metrics, ["diagnostic", "variant", "horizon_s"])
    summary["diagnostic"] = diagnostic
    return summary


def _shift_moments_by_lag(samples: pd.DataFrame, lag_s: float) -> pd.DataFrame:
    shifted = samples.copy()
    for _, group in _iter_groups(samples):
        times = group["time_s"].to_numpy(dtype=float, copy=True)
        if len(times) < 2:
            continue
        for column in MOMENT_COLUMNS:
            values = group[column].to_numpy(dtype=float, copy=True)
            shifted.loc[group.index, column] = np.interp(times + lag_s, times, values, left=values[0], right=values[-1])
    return shifted


def evaluate_moment_lag_sweep(
    samples: pd.DataFrame,
    inertia_b: np.ndarray,
    lags_s: list[float],
    horizons_s: list[float],
    stride_s: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for lag_s in lags_s:
        shifted = _shift_moments_by_lag(samples, lag_s)
        metrics = evaluate_omega_replay(
            shifted,
            inertia_b,
            horizons_s,
            stride_s,
            variants=[("logged_gyro_euler", "logged", "euler", 1)],
        )
        metrics = metrics[metrics["variant"].eq("logged_gyro_euler")].copy()
        metrics["variant"] = "lag_sweep"
        metrics["lag_s"] = float(lag_s)
        summary = summarize_metric_table(metrics, ["diagnostic", "variant", "horizon_s", "lag_s"])
        summary["diagnostic"] = "lag_sweep"
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def smooth_grouped_signal(
    samples: pd.DataFrame,
    columns: list[str],
    method: str,
    window_s: float,
    polyorder: int = 2,
) -> pd.DataFrame:
    del polyorder
    if method not in {"rolling_median", "savgol"}:
        raise ValueError("unsupported smoothing method")
    out = samples.copy()
    for _, group in _iter_groups(samples):
        times = group["time_s"].to_numpy(dtype=float, copy=True)
        if len(times) < 3:
            continue
        dt = np.median(np.diff(times))
        window_n = max(3, int(round(window_s / dt)))
        if window_n % 2 == 0:
            window_n += 1
        for column in columns:
            series = pd.Series(group[column].to_numpy(dtype=float, copy=True), index=group.index)
            smoothed = series.rolling(window_n, center=True, min_periods=1).median()
            out.loc[group.index, column] = smoothed.to_numpy(dtype=float)
    return out


def _summary_for_variant(samples: pd.DataFrame, inertia_b: np.ndarray, horizons_s: list[float], stride_s: float, variant: str, diagnostic: str) -> pd.DataFrame:
    metrics = evaluate_omega_replay(
        samples,
        inertia_b,
        horizons_s,
        stride_s,
        variants=[("logged_gyro_euler", "logged", "euler", 1)],
    )
    metrics = metrics[metrics["variant"].eq("logged_gyro_euler")].copy()
    metrics["variant"] = variant
    summary = summarize_metric_table(metrics, ["diagnostic", "variant", "horizon_s"])
    summary["diagnostic"] = diagnostic
    return summary


def evaluate_smoothing_sensitivity(
    samples: pd.DataFrame,
    inertia_b: np.ndarray,
    horizons_s: list[float],
    stride_s: float,
) -> pd.DataFrame:
    rows = [_summary_for_variant(samples, inertia_b, horizons_s, stride_s, "raw_moment", "smoothing_sensitivity")]
    for window_s in [0.04, 0.08, 0.16, 0.32]:
        suffix = f"{window_s:.2f}".replace(".", "p")
        smoothed_moment = smooth_grouped_signal(samples, MOMENT_COLUMNS, "savgol", window_s)
        rows.append(_summary_for_variant(smoothed_moment, inertia_b, horizons_s, stride_s, f"moment_savgol_{suffix}", "smoothing_sensitivity"))
        if all(column in samples.columns for column in _alpha_columns(samples)):
            smoothed_alpha = smooth_grouped_signal(samples, _alpha_columns(samples), "savgol", window_s)
            smoothed_alpha = smoothed_alpha.copy()
            smoothed_alpha[MOMENT_COLUMNS] = recompute_moment_from_alpha(
                _array(smoothed_alpha, OMEGA_COLUMNS),
                _array(smoothed_alpha, _alpha_columns(smoothed_alpha)),
                inertia_b,
            )
            rows.append(
                _summary_for_variant(
                    smoothed_alpha,
                    inertia_b,
                    horizons_s,
                    stride_s,
                    f"alpha_savgol_{suffix}_recomputed_moment",
                    "smoothing_sensitivity",
                )
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def evaluate_inertia_scale_sensitivity(
    samples: pd.DataFrame,
    inertia_b: np.ndarray,
    scales: list[float],
    horizons_s: list[float],
    stride_s: float,
) -> pd.DataFrame:
    inertia = np.asarray(inertia_b, dtype=float)
    rows: list[pd.DataFrame] = []
    for scale in scales:
        rows.append(_summary_for_variant(samples, inertia * float(scale), horizons_s, stride_s, f"global_scale_{scale:g}", "inertia_sensitivity"))
    for axis, index in zip(["ixx", "iyy", "izz"], range(3)):
        for scale in scales:
            scaled = inertia.copy()
            scaled[index, index] *= float(scale)
            rows.append(_summary_for_variant(samples, scaled, horizons_s, stride_s, f"{axis}_scale_{scale:g}", "inertia_sensitivity"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fit_diagonal_inertia_from_logs(samples: pd.DataFrame) -> np.ndarray:
    omega = _array(samples, OMEGA_COLUMNS)
    alpha = _array(samples, _alpha_columns(samples))
    moment = _array(samples, MOMENT_COLUMNS)
    rows: list[list[float]] = []
    targets: list[float] = []
    for w, a, m in zip(omega, alpha, moment):
        wx, wy, wz = w
        ax, ay, az = a
        rows.extend(
            [
                [ax, -wy * wz, wy * wz],
                [wx * wz, ay, -wx * wz],
                [-wx * wy, wx * wy, az],
            ]
        )
        targets.extend([m[0], m[1], m[2]])
    coeffs, *_ = np.linalg.lstsq(np.asarray(rows, dtype=float), np.asarray(targets, dtype=float), rcond=None)
    return np.diag(coeffs)


def fit_symmetric_inertia_from_logs(samples: pd.DataFrame) -> np.ndarray:
    omega = _array(samples, OMEGA_COLUMNS)
    alpha = _array(samples, _alpha_columns(samples))
    moment = _array(samples, MOMENT_COLUMNS)
    basis = [
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
    ]
    design: list[list[float]] = []
    targets: list[float] = []
    for w, a, m in zip(omega, alpha, moment):
        columns = [b @ a + np.cross(w, b @ w) for b in basis]
        design.extend(np.asarray(columns).T.tolist())
        targets.extend(m.tolist())
    coeffs, *_ = np.linalg.lstsq(np.asarray(design, dtype=float), np.asarray(targets, dtype=float), rcond=None)
    return np.array(
        [
            [coeffs[0], coeffs[3], coeffs[4]],
            [coeffs[3], coeffs[1], coeffs[5]],
            [coeffs[4], coeffs[5], coeffs[2]],
        ],
        dtype=float,
    )


def evaluate_reference_point_sensitivity(
    samples: pd.DataFrame,
    inertia_b: np.ndarray,
    cg_b_m: np.ndarray,
    horizons_s: list[float],
    stride_s: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for mode in ["none", "minus_cg_cross_force", "plus_cg_cross_force", "minus_2x_cg_cross_force", "plus_2x_cg_cross_force"]:
        transformed = samples.copy()
        transformed[MOMENT_COLUMNS] = apply_moment_reference_transform(
            _array(samples, MOMENT_COLUMNS),
            _array(samples, FORCE_COLUMNS),
            cg_b_m,
            mode,
        )
        rows.append(_summary_for_variant(transformed, inertia_b, horizons_s, stride_s, mode, "reference_point_sensitivity"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _moment_norm_per_window(samples: pd.DataFrame, windows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([np.nanmax(np.linalg.norm(_array(_window(samples, spec), MOMENT_COLUMNS), axis=1)) for spec in windows], dtype=float)


def evaluate_spike_robustness(
    samples: pd.DataFrame,
    inertia_b: np.ndarray,
    horizons_s: list[float],
    stride_s: float,
) -> pd.DataFrame:
    frame = _sort_samples(samples)
    windows = select_replay_windows(frame, horizons_s=horizons_s, stride_s=stride_s)
    rows = [_summary_for_variant(frame, inertia_b, horizons_s, stride_s, "raw", "spike_robustness")]
    if windows:
        norms = _moment_norm_per_window(frame, windows)
        for percent in [0.5, 1.0]:
            threshold = np.nanpercentile(norms, 100.0 - percent)
            keep_specs = [spec for spec, norm in zip(windows, norms) if norm <= threshold]
            metrics_rows: list[dict[str, Any]] = []
            for spec in keep_specs:
                window = _window(frame, spec)
                target = window.iloc[-1][OMEGA_COLUMNS].to_numpy(dtype=float)
                omega_sim = integrate_omega_from_moment(window, inertia_b)
                err = omega_sim[-1] - target
                metrics_rows.append(
                    {
                        "diagnostic": "spike_robustness",
                        "variant": f"drop_top_{percent:.1f}_percent_moment_norm_windows".replace(".", "p"),
                        "horizon_s": spec["horizon_s"],
                        "body_rate_error_rad_s": float(np.linalg.norm(err)),
                        "removed_fraction": float(1.0 - len(keep_specs) / len(windows)),
                    }
                )
            if metrics_rows:
                rows.append(summarize_metric_table(pd.DataFrame(metrics_rows), ["diagnostic", "variant", "horizon_s"]))
    for percentile in [99.0, 99.5]:
        clipped = frame.copy()
        limits = np.nanpercentile(np.abs(_array(frame, MOMENT_COLUMNS)), percentile, axis=0)
        clipped[MOMENT_COLUMNS] = np.clip(_array(frame, MOMENT_COLUMNS), -limits, limits)
        rows.append(_summary_for_variant(clipped, inertia_b, horizons_s, stride_s, f"winsorize_moment_{percentile:.1f}".replace(".", "p"), "spike_robustness"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _gate_from_median(value: float, pass_threshold: float, conditional_threshold: float) -> str:
    if not np.isfinite(value):
        return "fail"
    if value <= pass_threshold:
        return "pass"
    if value <= conditional_threshold:
        return "conditional"
    return "fail"


def _decision(
    attitude_summary: pd.DataFrame,
    moment_closure: pd.DataFrame,
    omega_summary: pd.DataFrame,
    lag_summary: pd.DataFrame,
    smoothing_summary: pd.DataFrame,
    inertia_summary: pd.DataFrame,
    reference_summary: pd.DataFrame,
    spike_summary: pd.DataFrame,
    fitted_diagonal_inertia: np.ndarray,
    fitted_symmetric_inertia: np.ndarray,
) -> dict[str, Any]:
    nominal_att = attitude_summary[
        attitude_summary["variant"].eq("right_multiply_rate_plus") & np.isclose(attitude_summary["horizon_s"], 0.25)
    ]
    if nominal_att.empty:
        nominal_att = attitude_summary[attitude_summary["variant"].eq("right_multiply_rate_plus")]
    att_med = float(nominal_att["attitude_error_deg_median"].iloc[0]) if not nominal_att.empty else np.nan
    kin_gate = _gate_from_median(att_med, 0.5, 2.0)

    moment_norm = moment_closure[moment_closure["axis"].eq("moment_norm")]
    moment_rmse = float(moment_norm["rmse"].iloc[0]) if not moment_norm.empty else np.nan
    moment_gate = _gate_from_median(moment_rmse, 1e-9, 0.02)

    omega_nominal = omega_summary[
        omega_summary["variant"].eq("logged_gyro_euler") & np.isclose(omega_summary["horizon_s"], 0.25)
    ]
    if omega_nominal.empty:
        omega_nominal = omega_summary[omega_summary["variant"].eq("logged_gyro_euler")]
    omega_med = float(omega_nominal["body_rate_error_rad_s_median"].iloc[0]) if not omega_nominal.empty else np.nan
    omega_gate = _gate_from_median(omega_med, 0.05, 0.20)

    likely: list[str] = []
    if kin_gate == "fail":
        likely.append("quaternion/body-rate convention or timestamp alignment")
    if moment_gate == "fail":
        likely.append("moment label generation or angular-acceleration column mismatch")
    if omega_gate == "fail":
        sensitivity_tables = [
            ("time lag between moment labels and body rates", lag_summary, "body_rate_error_rad_s_median"),
            ("moment/alpha smoothing or differentiation noise", smoothing_summary, "body_rate_error_rad_s_median"),
            ("inertia metadata mismatch", inertia_summary, "body_rate_error_rad_s_median"),
            ("moment reference point or CG transform mismatch", reference_summary, "body_rate_error_rad_s_median"),
            ("rare moment spikes/outliers", spike_summary, "body_rate_error_rad_s_median"),
        ]
        baseline = omega_med
        ranked = []
        for label, table, metric in sensitivity_tables:
            if metric in table and not table.empty:
                best = float(table[metric].min())
                improvement = baseline - best if np.isfinite(baseline) and np.isfinite(best) else -np.inf
                ranked.append((improvement, label))
        likely.extend([label for _, label in sorted(ranked, reverse=True)])
    if not likely:
        likely.append("no dominant rotational diagnostic mismatch detected")

    return {
        "kinematic_attitude_gate": kin_gate,
        "moment_label_closure_gate": moment_gate,
        "forward_omega_replay_gate": omega_gate,
        "key_metrics": {
            "nominal_attitude_error_deg_median_approx_0p25s": att_med,
            "moment_norm_rmse_n_m": moment_rmse,
            "nominal_omega_error_rad_s_median_approx_0p25s": omega_med,
        },
        "likely_issue_order": likely,
        "safe_next_experiments": [
            "translational/local force replay only, clearly separated from rotational claims",
            "repeat rotational oracle diagnostics after measured CG and inertia are available",
            "inspect moment-label differentiation and timing before any six-degree-of-freedom model comparison",
        ],
        "paper_claim_boundary": (
            "Do not claim closed-loop simulator validation or validated six-degree-of-freedom replay from these diagnostics; "
            "use replay evidence only for local translational/effective-force consistency unless rotational gates pass."
        ),
        "fitted_diagonal_inertia_b_kg_m2": fitted_diagonal_inertia.tolist(),
        "fitted_symmetric_inertia_b_kg_m2": fitted_symmetric_inertia.tolist(),
        "fitted_symmetric_positive_definite": bool(np.all(np.linalg.eigvalsh(fitted_symmetric_inertia) > 0.0)),
    }


def run_rotational_diagnostics(
    split_root: str | Path,
    metadata_path: str | Path,
    output_root: str | Path,
    split: str,
    horizons_s: list[float],
    stride_s: float,
    lags_s: list[float],
    overwrite: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_root)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(f"output root exists and is not empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    samples_path = Path(split_root) / f"{split}_samples.parquet"
    samples = pd.read_parquet(samples_path)
    metadata = _load_metadata_dict(metadata_path)
    _, inertia_b, _ = load_mass_properties(metadata_path)
    cg_b_m = _metadata_cg_b(metadata)

    attitude_metrics = evaluate_attitude_kinematic_closure(samples, horizons_s, stride_s, split)
    attitude_summary = summarize_metric_table(attitude_metrics, ["diagnostic", "variant", "split", "horizon_s"])
    moment_closure = evaluate_moment_label_closure(samples, inertia_b)
    omega_metrics = evaluate_omega_replay(samples, inertia_b, horizons_s, stride_s)
    omega_summary = summarize_metric_table(omega_metrics, ["diagnostic", "variant", "horizon_s"])
    lag_summary = evaluate_moment_lag_sweep(samples, inertia_b, lags_s, horizons_s, stride_s)
    smoothing_summary = evaluate_smoothing_sensitivity(samples, inertia_b, horizons_s, stride_s)
    inertia_summary = evaluate_inertia_scale_sensitivity(samples, inertia_b, [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00, 4.00], horizons_s, stride_s)
    reference_summary = evaluate_reference_point_sensitivity(samples, inertia_b, cg_b_m, horizons_s, stride_s)
    spike_summary = evaluate_spike_robustness(samples, inertia_b, horizons_s, stride_s)
    fitted_diagonal = fit_diagonal_inertia_from_logs(samples)
    fitted_symmetric = fit_symmetric_inertia_from_logs(samples)
    fitted_rows = pd.DataFrame(
        [
            {"fit": "metadata", "matrix": json.dumps(np.asarray(inertia_b, dtype=float).tolist()), "positive_definite": bool(np.all(np.linalg.eigvalsh(inertia_b) > 0.0))},
            {"fit": "diagonal_lstsq", "matrix": json.dumps(fitted_diagonal.tolist()), "positive_definite": bool(np.all(np.diag(fitted_diagonal) > 0.0))},
            {"fit": "symmetric_lstsq", "matrix": json.dumps(fitted_symmetric.tolist()), "positive_definite": bool(np.all(np.linalg.eigvalsh(fitted_symmetric) > 0.0))},
        ]
    )

    decision = _decision(
        attitude_summary,
        moment_closure,
        omega_summary,
        lag_summary,
        smoothing_summary,
        inertia_summary,
        reference_summary,
        spike_summary,
        fitted_diagonal,
        fitted_symmetric,
    )

    attitude_metrics.to_csv(output_path / "attitude_kinematic_closure.csv", index=False)
    attitude_summary.to_csv(output_path / "attitude_kinematic_closure_summary.csv", index=False)
    moment_closure.to_csv(output_path / "moment_label_closure.csv", index=False)
    omega_summary.to_csv(output_path / "omega_replay_summary.csv", index=False)
    lag_summary.to_csv(output_path / "lag_sweep_summary.csv", index=False)
    smoothing_summary.to_csv(output_path / "smoothing_sensitivity_summary.csv", index=False)
    inertia_summary.to_csv(output_path / "inertia_sensitivity_summary.csv", index=False)
    fitted_rows.to_csv(output_path / "fitted_inertia.csv", index=False)
    reference_summary.to_csv(output_path / "reference_point_sensitivity_summary.csv", index=False)
    spike_summary.to_csv(output_path / "spike_robustness_summary.csv", index=False)
    with (output_path / "diagnostic_decision.json").open("w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_path / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "split_root": str(split_root),
                "metadata_path": str(metadata_path),
                "split": split,
                "horizons_s": horizons_s,
                "stride_s": stride_s,
                "lags_s": lags_s,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    with (output_path / "README.md").open("w", encoding="utf-8") as handle:
        handle.write("# Rotational Oracle Replay Diagnostics\n\n")
        handle.write("Diagnostic-only artifacts for oracle rotational replay gates. These outputs do not validate closed-loop simulation and do not compare prior/corrected six-degree-of-freedom models.\n")
    return decision


def main() -> None:
    argv = sys.argv[1:]
    if "--lags-s" in argv:
        lag_index = argv.index("--lags-s")
        if lag_index + 1 < len(argv) and argv[lag_index + 1].startswith("-"):
            argv = argv[:lag_index] + [f"--lags-s={argv[lag_index + 1]}"] + argv[lag_index + 2 :]

    parser = argparse.ArgumentParser(description="Diagnose rotational oracle replay mismatch.")
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--horizons", default="0.10,0.25,0.50,1.00,2.00")
    parser.add_argument("--stride-s", type=float, default=0.25)
    parser.add_argument("--lags-s", default="-0.100,-0.080,-0.060,-0.040,-0.020,0.000,0.020,0.040,0.060,0.080,0.100")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    decision = run_rotational_diagnostics(
        split_root=args.split_root,
        metadata_path=args.metadata_path,
        output_root=args.output_root,
        split=args.split,
        horizons_s=_parse_float_list(args.horizons),
        stride_s=args.stride_s,
        lags_s=_parse_float_list(args.lags_s),
        overwrite=args.overwrite,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
