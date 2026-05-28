from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.signal import butter, lfilter, savgol_filter


def existing_group_columns(frame: pd.DataFrame, group_columns: Iterable[str] | None = None) -> list[str]:
    candidates = ["log_id", "segment_id"] if group_columns is None else list(group_columns)
    return [column for column in candidates if column in frame.columns]


def iter_groups(frame: pd.DataFrame, group_columns: Iterable[str] | None = None) -> Iterable[tuple[Any, pd.DataFrame]]:
    resolved = existing_group_columns(frame, group_columns)
    if not resolved:
        yield None, frame
        return
    yield from frame.groupby(resolved, sort=False, dropna=False)


def odd_window_length(sample_period_s: float, window_s: float, sample_count: int) -> int:
    if sample_count < 3 or not np.isfinite(sample_period_s) or sample_period_s <= 0.0:
        return 0
    window = max(3, int(round(window_s / sample_period_s)))
    if window % 2 == 0:
        window += 1
    if window > sample_count:
        window = sample_count if sample_count % 2 == 1 else sample_count - 1
    return window if window >= 3 else 0


def sorted_finite_xy(
    group: pd.DataFrame,
    value_column: str,
    time_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_s = group[time_column].to_numpy(dtype=float)
    values = group[value_column].to_numpy(dtype=float)
    finite = np.isfinite(time_s) & np.isfinite(values)
    if int(finite.sum()) < 2:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)
    original_index = group.index.to_numpy()[finite]
    valid_time = time_s[finite]
    valid_values = values[finite]
    order = np.argsort(valid_time)
    return original_index[order], valid_time[order], valid_values[order]


def groupwise_savgol_derivative(
    frame: pd.DataFrame,
    value_column: str,
    *,
    time_column: str = "time_s",
    group_columns: Iterable[str] | None = None,
    window_s: float = 0.12,
    polyorder: int = 2,
) -> pd.Series:
    derivative = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in iter_groups(frame, group_columns):
        valid_index, valid_time, valid_values = sorted_finite_xy(group, value_column, time_column)
        if len(valid_values) < 3:
            continue
        dt = np.diff(valid_time)
        dt = dt[np.isfinite(dt) & (dt > 0.0)]
        if len(dt) == 0:
            continue
        sample_period_s = float(np.median(dt))
        window = odd_window_length(sample_period_s, window_s, len(valid_values))
        if window <= polyorder:
            estimated = np.gradient(valid_values, valid_time)
        else:
            estimated = savgol_filter(
                valid_values,
                window_length=window,
                polyorder=min(polyorder, window - 1),
                deriv=1,
                delta=sample_period_s,
                mode="interp",
            )
        derivative.loc[valid_index] = estimated
    return derivative


def groupwise_cubic_spline_derivative(
    frame: pd.DataFrame,
    value_column: str,
    *,
    time_column: str = "time_s",
    group_columns: Iterable[str] | None = None,
    smoothing_factor: float | None = None,
) -> pd.Series:
    derivative = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in iter_groups(frame, group_columns):
        valid_index, valid_time, valid_values = sorted_finite_xy(group, value_column, time_column)
        if len(valid_values) < 4:
            continue
        unique_time, unique_indices = np.unique(valid_time, return_index=True)
        unique_values = valid_values[unique_indices]
        unique_index = valid_index[unique_indices]
        if len(unique_values) < 4:
            continue
        spline = UnivariateSpline(unique_time, unique_values, k=3, s=0.0 if smoothing_factor is None else smoothing_factor)
        derivative.loc[unique_index] = spline.derivative(1)(unique_time)
    return derivative


def apply_groupwise_time_shift(
    frame: pd.DataFrame,
    value_column: str,
    *,
    lag_s: float,
    time_column: str = "time_s",
    group_columns: Iterable[str] | None = None,
) -> pd.Series:
    shifted = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in iter_groups(frame, group_columns):
        valid_index, valid_time, valid_values = sorted_finite_xy(group, value_column, time_column)
        if len(valid_values) < 2:
            continue
        query_time = valid_time - float(lag_s)
        inside = (query_time >= valid_time[0]) & (query_time <= valid_time[-1])
        out = np.full(len(valid_values), np.nan, dtype=float)
        out[inside] = np.interp(query_time[inside], valid_time, valid_values)
        shifted.loc[valid_index] = out
    return shifted


def groupwise_lowpass_filter(
    frame: pd.DataFrame,
    value_column: str,
    *,
    cutoff_hz: float | None = None,
    order: int = 2,
    time_column: str = "time_s",
    group_columns: Iterable[str] | None = None,
    method: str = "butterworth",
    time_constant_s: float | None = None,
) -> pd.Series:
    filtered = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in iter_groups(frame, group_columns):
        valid_index, valid_time, valid_values = sorted_finite_xy(group, value_column, time_column)
        if len(valid_values) < 3:
            continue
        dt = np.diff(valid_time)
        dt = dt[np.isfinite(dt) & (dt > 0.0)]
        if len(dt) == 0:
            continue
        sample_period_s = float(np.median(dt))
        if method == "first_order":
            tau = float(0.04 if time_constant_s is None else time_constant_s)
            alpha = sample_period_s / max(tau + sample_period_s, 1e-12)
            out = np.empty_like(valid_values, dtype=float)
            out[0] = valid_values[0]
            for idx in range(1, len(valid_values)):
                out[idx] = out[idx - 1] + alpha * (valid_values[idx] - out[idx - 1])
        elif method == "butterworth":
            if cutoff_hz is None:
                raise ValueError("cutoff_hz is required for butterworth filtering")
            nyquist_hz = 0.5 / sample_period_s
            normalized = float(cutoff_hz) / nyquist_hz
            if not (0.0 < normalized < 1.0):
                out = valid_values.astype(float, copy=True)
            else:
                b, a = butter(int(order), normalized, btype="low", analog=False)
                out = lfilter(b, a, valid_values)
        else:
            raise ValueError(f"Unknown low-pass method: {method}")
        filtered.loc[valid_index] = out
    return filtered


def nominal_sample_rate_hz(frame: pd.DataFrame, *, time_column: str = "time_s") -> float:
    values = frame[time_column].to_numpy(dtype=float)
    dt = np.diff(values[np.isfinite(values)])
    dt = dt[(dt > 0.0) & np.isfinite(dt)]
    if len(dt) == 0:
        return float("nan")
    return float(1.0 / np.median(dt))


def highpass_energy_fraction(values: np.ndarray | pd.Series, *, sample_rate_hz: float, cutoff_hz: float) -> float:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if int(finite.sum()) < 8 or not np.isfinite(sample_rate_hz):
        return float("nan")
    filled = array.copy()
    if not finite.all():
        idx = np.flatnonzero(finite)
        filled[~finite] = np.interp(np.flatnonzero(~finite), idx, array[idx])
    centered = filled - np.mean(filled)
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    total = float(np.sum(np.square(np.abs(spectrum))))
    if total <= 0.0:
        return float("nan")
    high = float(np.sum(np.square(np.abs(spectrum[freqs >= cutoff_hz]))))
    return high / total


def finite_difference_quality_metrics(
    raw: np.ndarray | pd.Series,
    variant: np.ndarray | pd.Series,
    *,
    sample_rate_hz: float | None = None,
    highpass_cutoff_hz: float = 8.0,
) -> dict[str, float]:
    raw_array = np.asarray(raw, dtype=float)
    variant_array = np.asarray(variant, dtype=float)
    finite = np.isfinite(raw_array) & np.isfinite(variant_array)
    if not np.any(finite):
        return {"sample_count": 0.0}
    raw_valid = raw_array[finite]
    variant_valid = variant_array[finite]
    corr = float("nan")
    if len(raw_valid) > 1 and np.std(raw_valid) > 1e-12 and np.std(variant_valid) > 1e-12:
        corr = float(np.corrcoef(raw_valid, variant_valid)[0, 1])
    metrics = {
        "sample_count": float(len(raw_valid)),
        "corr": corr,
        "rmse": float(np.sqrt(np.mean(np.square(variant_valid - raw_valid)))),
        "raw_std": float(np.std(raw_valid)),
        "variant_std": float(np.std(variant_valid)),
        "jump_p99_raw": float(np.quantile(np.abs(np.diff(raw_valid)), 0.99)) if len(raw_valid) > 1 else float("nan"),
        "jump_p99_variant": float(np.quantile(np.abs(np.diff(variant_valid)), 0.99))
        if len(variant_valid) > 1
        else float("nan"),
    }
    if sample_rate_hz is not None:
        metrics["highpass_energy_frac_raw"] = highpass_energy_fraction(
            raw_valid,
            sample_rate_hz=sample_rate_hz,
            cutoff_hz=highpass_cutoff_hz,
        )
        metrics["highpass_energy_frac_variant"] = highpass_energy_fraction(
            variant_valid,
            sample_rate_hz=sample_rate_hz,
            cutoff_hz=highpass_cutoff_hz,
        )
    return metrics
