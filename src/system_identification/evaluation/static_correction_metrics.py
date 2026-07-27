"""Per-log macro metrics for frozen C3 candidates."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def circular_distance(a: float, b: float) -> float:
    return float(abs(np.angle(np.exp(1j * (float(a) - float(b))))))


def circular_peak_phase_error(phase: Iterable[float], label: Iterable[float], prediction: Iterable[float]) -> float:
    phi = np.asarray(tuple(phase), dtype=np.float64)
    target = np.asarray(tuple(label), dtype=np.float64)
    estimate = np.asarray(tuple(prediction), dtype=np.float64)
    if len(phi) == 0 or not (np.isfinite(phi).all() and np.isfinite(target).all() and np.isfinite(estimate).all()):
        raise ValueError("Peak-phase metric requires finite non-empty values")
    label_phase = phi[int(np.argmax(np.abs(target)))]
    prediction_phase = phi[int(np.argmax(np.abs(estimate)))]
    return circular_distance(float(label_phase), float(prediction_phase))


def half_stroke_integral_error(
    phase: Iterable[float],
    label: Iterable[float],
    prediction: Iterable[float],
) -> float:
    phi = np.asarray(tuple(phase), dtype=np.float64)
    error = np.asarray(tuple(prediction), dtype=np.float64) - np.asarray(tuple(label), dtype=np.float64)
    order = np.argsort(phi, kind="stable")
    if len(phi) < 2:
        return 0.0
    return float(np.trapz(error[order], phi[order]))


def per_log_total_metrics(
    frame: pd.DataFrame,
    *,
    label_column: str = "label_n",
    prediction_column: str = "prediction_n",
) -> pd.DataFrame:
    required = {"log_id", label_column, prediction_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Total metric input missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for log_id, group in frame.groupby("log_id", sort=True):
        residual = group[prediction_column].to_numpy(dtype=np.float64) - group[label_column].to_numpy(
            dtype=np.float64
        )
        rows.append(
            {
                "log_id": str(log_id),
                "sample_count": int(len(group)),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(np.abs(residual))),
                "bias": float(np.mean(residual)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_per_log(per_log: pd.DataFrame, metric: str = "rmse") -> dict[str, float]:
    values = per_log[metric].to_numpy(dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Per-log aggregate requires finite non-empty values")
    return {
        f"macro_{metric}": float(values.mean()),
        f"median_log_{metric}": float(np.median(values)),
        f"worst_log_{metric}": float(values.max()),
        f"log_standard_deviation_{metric}": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        f"log_standard_error_{metric}": (
            float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
        ),
    }


def per_log_mean_metrics(cycle_frame: pd.DataFrame, prediction: np.ndarray, component: str) -> pd.DataFrame:
    label = cycle_frame[f"label_{component}_mean_n"].to_numpy(dtype=np.float64)
    residual = np.asarray(prediction, dtype=np.float64) - label
    table = cycle_frame.loc[:, ["log_id"]].copy()
    table["error"] = residual
    rows = []
    for log_id, group in table.groupby("log_id", sort=True):
        error = group["error"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "log_id": str(log_id),
                "mean_rmse": float(np.sqrt(np.mean(error**2))),
                "mean_mae": float(np.mean(np.abs(error))),
                "mean_bias": float(np.mean(error)),
            }
        )
    return pd.DataFrame(rows)


def per_log_waveform_metrics(waveform_frame: pd.DataFrame, prediction: np.ndarray, component: str) -> pd.DataFrame:
    frame = waveform_frame.loc[:, ["log_id", "cycle_id"]].copy()
    frame["squared_error"] = (
        np.asarray(prediction, dtype=np.float64)
        - waveform_frame[f"label_{component}_waveform_n"].to_numpy(dtype=np.float64)
    ) ** 2
    cycle = frame.groupby(["log_id", "cycle_id"], sort=True)["squared_error"].mean().pow(0.5).reset_index()
    return (
        cycle.groupby("log_id", sort=True)["squared_error"]
        .agg(waveform_rmse="mean", cycle_count="size")
        .reset_index()
    )


def waveform_secondary_metrics(
    waveform_frame: pd.DataFrame,
    prediction: np.ndarray,
    component: str,
    *,
    phase_bins: int = 36,
) -> dict[str, float]:
    label = waveform_frame[f"label_{component}_waveform_n"].to_numpy(dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    phase = waveform_frame["phase_rad"].to_numpy(dtype=np.float64)
    bins = np.floor(np.mod(phase, 2.0 * np.pi) / (2.0 * np.pi) * phase_bins).astype(int)
    binned = pd.DataFrame({"bin": bins, "label": label, "prediction": estimate}).groupby("bin", sort=True).mean()
    phase_bin_rmse = float(np.sqrt(np.mean((binned["prediction"] - binned["label"]) ** 2)))
    integral_errors: dict[str, list[float]] = {"upstroke": [], "downstroke": []}
    peak_magnitude_errors: list[float] = []
    peak_phase_errors: list[float] = []
    temporary = waveform_frame.loc[:, ["cycle_id", "phase_rad", "half_stroke_id"]].copy()
    temporary["label"] = label
    temporary["prediction"] = estimate
    for _, cycle in temporary.groupby("cycle_id", sort=False):
        peak_magnitude_errors.append(
            float(np.max(np.abs(cycle["prediction"])) - np.max(np.abs(cycle["label"])))
        )
        peak_phase_errors.append(
            circular_peak_phase_error(cycle["phase_rad"], cycle["label"], cycle["prediction"])
        )
        for half, half_frame in cycle.groupby("half_stroke_id", sort=False):
            if str(half) in integral_errors:
                integral_errors[str(half)].append(
                    half_stroke_integral_error(
                        half_frame["phase_rad"], half_frame["label"], half_frame["prediction"]
                    )
                )
    return {
        "phase_bin_waveform_rmse": phase_bin_rmse,
        "upstroke_integral_error": float(np.mean(integral_errors["upstroke"])),
        "downstroke_integral_error": float(np.mean(integral_errors["downstroke"])),
        "peak_magnitude_error": float(np.mean(np.abs(peak_magnitude_errors))),
        "circular_peak_phase_error": float(np.mean(peak_phase_errors)),
    }
