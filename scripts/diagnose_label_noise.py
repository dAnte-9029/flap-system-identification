#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.training import (  # noqa: E402
    DEFAULT_TARGET_COLUMNS,
    prepare_feature_target_frames,
    predict_model_bundle,
)

try:
    from scipy.signal import savgol_filter, welch
except ImportError:  # pragma: no cover - used only in lean local envs
    savgol_filter = None
    welch = None


TARGET_COLUMNS = list(DEFAULT_TARGET_COLUMNS)
DEFAULT_SPLIT_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1"
)
DEFAULT_MODEL_BUNDLE = (
    PROJECT_ROOT
    / "artifacts"
    / "baseline_torch_hq_v3_direct_airspeed_logsplit_paper_no_accel_v2_full"
    / "model_bundle.pt"
)
DEFAULT_METRICS_JSON = DEFAULT_MODEL_BUNDLE.parent / "metrics.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "label_noise_diagnostics_hq_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate label-noise diagnostics for effective wrench targets.",
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--model-bundle", type=Path, default=DEFAULT_MODEL_BUNDLE)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--phase-bins", type=int, default=72)
    parser.add_argument("--max-psd-hz", type=float, default=45.0)
    parser.add_argument("--psd-grid-hz", type=float, default=0.1)
    parser.add_argument("--lag-max-s", type=float, default=1.0)
    parser.add_argument("--lag-step-s", type=float, default=0.025)
    return parser.parse_args()


def read_split_frames(split_root: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "test"]:
        path = split_root / f"{split}_samples.parquet"
        frame = pd.read_parquet(path)
        frame = frame.copy()
        frame["_split_name"] = split
        frames[split] = frame
    return frames


def concat_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([frames["train"], frames["val"], frames["test"]], ignore_index=True)


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = finite_mask(a, b)
    if int(mask.sum()) < 3:
        return float("nan")
    x = a[mask].astype(float, copy=False)
    y = b[mask].astype(float, copy=False)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def weighted_average(values: list[float], weights: list[int]) -> float:
    finite = [(value, weight) for value, weight in zip(values, weights) if np.isfinite(value)]
    if not finite:
        return float("nan")
    total_weight = float(sum(weight for _, weight in finite))
    if total_weight <= 0.0:
        return float("nan")
    return float(sum(value * weight for value, weight in finite) / total_weight)


def robust_mad(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    median = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - median)))


def safe_r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    mask = finite_mask(y, y_hat)
    if int(mask.sum()) < 3:
        return float("nan")
    yy = y[mask].astype(float, copy=False)
    pred = y_hat[mask].astype(float, copy=False)
    total = float(np.sum((yy - yy.mean()) ** 2))
    if total <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((yy - pred) ** 2) / total)


def iter_log_groups(frame: pd.DataFrame, columns: Iterable[str]):
    required = ["time_s", "log_id", *columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns for log grouping: {missing}")
    for log_id, group in frame.loc[:, required].groupby("log_id", sort=False):
        group = group.sort_values("time_s")
        if len(group) >= 16:
            yield log_id, group


def phase_bin_statistics(
    phase_rad: np.ndarray,
    values: np.ndarray,
    *,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    phase = np.remainder(phase_rad.astype(float, copy=False), 2.0 * math.pi)
    y = values.astype(float, copy=False)
    mask = finite_mask(phase, y)
    phase = phase[mask]
    y = y[mask]

    bin_index = np.floor(phase / (2.0 * math.pi) * bins).astype(int)
    bin_index = np.clip(bin_index, 0, bins - 1)
    centers = (np.arange(bins) + 0.5) * 2.0 * math.pi / bins
    medians = np.full(bins, np.nan, dtype=float)
    mads = np.full(bins, np.nan, dtype=float)
    predicted = np.full(len(y), np.nan, dtype=float)

    for idx in range(bins):
        in_bin = bin_index == idx
        if int(in_bin.sum()) == 0:
            continue
        bin_values = y[in_bin]
        median = float(np.median(bin_values))
        medians[idx] = median
        mads[idx] = robust_mad(bin_values)
        predicted[in_bin] = median

    phase_r2 = safe_r2(y, predicted)
    target_std = float(np.nanstd(y))
    mad_to_std = float(np.nanmedian(mads) / target_std) if target_std > 0 else float("nan")
    return centers, medians, mads, phase_r2, mad_to_std


def plot_phase_bins(
    path: Path,
    phase_rad: np.ndarray,
    values_by_target: dict[str, np.ndarray],
    *,
    bins: int,
    title: str,
    ylabel_suffix: str = "",
) -> dict[str, dict[str, float]]:
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    axes_flat = axes.ravel()
    stats: dict[str, dict[str, float]] = {}

    for ax, target in zip(axes_flat, TARGET_COLUMNS):
        centers, medians, mads, phase_r2, mad_to_std = phase_bin_statistics(
            phase_rad,
            values_by_target[target],
            bins=bins,
        )
        stats[target] = {
            "phase_r2": phase_r2,
            "phase_median_mad_to_std": mad_to_std,
        }
        ax.plot(centers, medians, color="#1f77b4", linewidth=1.8)
        ax.fill_between(centers, medians - mads, medians + mads, color="#1f77b4", alpha=0.22)
        ax.set_title(f"{target}: phase R2={phase_r2:.3f}")
        ax.set_ylabel(f"{target}{ylabel_suffix}")
        ax.grid(alpha=0.25)

    for ax in axes[-1, :]:
        ax.set_xlabel("phase_corrected_rad")
        ax.set_xlim(0.0, 2.0 * math.pi)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return stats


def fallback_welch(y: np.ndarray, fs: float, max_hz: float) -> tuple[np.ndarray, np.ndarray]:
    y = y - np.nanmean(y)
    window = np.hanning(len(y))
    spectrum = np.fft.rfft(y * window)
    freqs = np.fft.rfftfreq(len(y), d=1.0 / fs)
    power = (np.abs(spectrum) ** 2) / max(float(np.sum(window * window)) * fs, 1e-12)
    mask = freqs <= max_hz
    return freqs[mask], power[mask]


def spectral_energy(freqs: np.ndarray, power: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) < 1:
        return 0.0
    widths = np.gradient(freqs)
    widths = np.where(np.isfinite(widths) & (widths > 0.0), widths, 0.0)
    return float(np.sum(power[mask] * widths[mask]))


def compute_psd_diagnostics(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    max_hz: float,
    grid_hz: float,
) -> dict[str, dict[str, float]]:
    freq_grid = np.arange(grid_hz, max_hz + grid_hz * 0.5, grid_hz)
    global_f0 = float(np.nanmedian(frame["cycle_flap_frequency_hz"].to_numpy(dtype=float)))
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    axes_flat = axes.ravel()
    stats: dict[str, dict[str, float]] = {}

    for ax, target in zip(axes_flat, TARGET_COLUMNS):
        psd_weighted = np.zeros_like(freq_grid, dtype=float)
        total_weight = 0
        harmonic_fractions: list[float] = []
        high_fractions: list[float] = []
        weights: list[int] = []

        for _, group in iter_log_groups(frame, [target, "cycle_flap_frequency_hz"]):
            time_s = group["time_s"].to_numpy(dtype=float)
            y = group[target].to_numpy(dtype=float)
            f0 = float(np.nanmedian(group["cycle_flap_frequency_hz"].to_numpy(dtype=float)))
            mask = finite_mask(time_s, y)
            time_s = time_s[mask]
            y = y[mask]
            if len(y) < 256 or not np.isfinite(f0) or f0 <= 0.0:
                continue
            dt = np.diff(time_s)
            dt = dt[np.isfinite(dt) & (dt > 0.0)]
            if len(dt) == 0:
                continue
            sample_period = float(np.median(dt))
            fs = 1.0 / sample_period
            nyquist = fs * 0.5
            if not np.isfinite(fs) or nyquist <= 2.0:
                continue

            y = y - np.nanmedian(y)
            if welch is not None:
                nperseg = min(len(y), 4096)
                freqs, power = welch(y, fs=fs, nperseg=nperseg, detrend="constant")
            else:
                freqs, power = fallback_welch(y, fs, max_hz)

            usable = np.isfinite(freqs) & np.isfinite(power) & (freqs > 0.05) & (freqs <= max_hz)
            freqs = freqs[usable]
            power = power[usable]
            if len(freqs) < 4:
                continue

            interp = np.interp(freq_grid, freqs, power, left=np.nan, right=np.nan)
            interp = np.where(np.isfinite(interp), interp, 0.0)
            psd_weighted += interp * len(y)
            total_weight += len(y)

            total_mask = (freqs >= 0.2) & (freqs <= min(max_hz, nyquist * 0.95))
            total_energy = spectral_energy(freqs, power, total_mask)
            if total_energy <= 0.0:
                continue

            harmonic_mask = np.zeros_like(freqs, dtype=bool)
            band_half_width = max(0.5, 0.12 * f0)
            for harmonic in range(1, 7):
                harmonic_mask |= np.abs(freqs - harmonic * f0) <= band_half_width
            harmonic_mask &= total_mask
            high_mask = (freqs >= 25.0) & (freqs <= min(45.0, nyquist * 0.95))

            harmonic_fractions.append(spectral_energy(freqs, power, harmonic_mask) / total_energy)
            high_fractions.append(spectral_energy(freqs, power, high_mask) / total_energy)
            weights.append(len(y))

        if total_weight > 0:
            psd_mean = psd_weighted / total_weight
            ax.semilogy(freq_grid, np.maximum(psd_mean, 1e-18), color="#2ca02c", linewidth=1.5)
        for harmonic in range(1, 7):
            harmonic_freq = harmonic * global_f0
            if harmonic_freq <= max_hz:
                ax.axvline(harmonic_freq, color="#d62728", alpha=0.22, linewidth=1.0)
        harmonic_fraction = weighted_average(harmonic_fractions, weights)
        high_fraction = weighted_average(high_fractions, weights)
        stats[target] = {
            "psd_harmonic_band_fraction_1x_to_6x": harmonic_fraction,
            "psd_high_frequency_fraction_25_to_45hz": high_fraction,
        }
        ax.set_title(
            f"{target}: harmonic={harmonic_fraction:.2f}, high={high_fraction:.2f}",
        )
        ax.set_ylabel("PSD")
        ax.grid(alpha=0.25)

    for ax in axes[-1, :]:
        ax.set_xlabel("Frequency [Hz]")
    fig.suptitle("Target PSD; red lines mark 1x to 6x median flap-frequency harmonics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return stats


def prepare_model_residuals(
    bundle_path: Path,
    test_frame: pd.DataFrame,
    *,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    features_df, targets_df = prepare_feature_target_frames(
        test_frame,
        bundle["feature_columns"],
        bundle["target_columns"],
    )
    predictions_df = predict_model_bundle(bundle, test_frame, batch_size=8192, device=device)
    residuals_df = targets_df - predictions_df
    return features_df, targets_df, residuals_df


def compute_residual_feature_correlations(
    features_df: pd.DataFrame,
    residuals_df: pd.DataFrame,
    output_csv: Path,
    output_png: Path,
) -> pd.DataFrame:
    aliases = OrderedDict(
        [
            ("phase_sin", "phase_corrected_sin"),
            ("phase_cos", "phase_corrected_cos"),
            ("tas", "airspeed_validated.true_airspeed_m_s"),
            ("qbar", "dynamic_pressure_pa"),
            ("alpha", "alpha_rad"),
            ("beta", "beta_rad"),
            ("vel_b_x", "velocity_b.x"),
            ("vel_b_y", "velocity_b.y"),
            ("vel_b_z", "velocity_b.z"),
            ("freq", "cycle_flap_frequency_hz"),
            ("elevator", "elevator_like"),
            ("rudder", "servo_rudder"),
        ],
    )
    available = OrderedDict((alias, col) for alias, col in aliases.items() if col in features_df.columns)
    corr = pd.DataFrame(index=TARGET_COLUMNS, columns=list(available.keys()), dtype=float)

    for target in TARGET_COLUMNS:
        residual = residuals_df[target].to_numpy(dtype=float)
        for alias, column in available.items():
            corr.loc[target, alias] = pearson_corr(residual, features_df[column].to_numpy(dtype=float))

    corr.to_csv(output_csv)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    values = corr.to_numpy(dtype=float)
    im = ax.imshow(values, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(corr.columns)), labels=corr.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(corr.index)), labels=corr.index)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Test residual correlation with candidate explanatory variables")
    fig.colorbar(im, ax=ax, shrink=0.88, label="Pearson r")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return corr


def smooth_for_derivative(values: np.ndarray, dt: float, window_s: float = 0.12) -> np.ndarray:
    if len(values) < 7 or not np.isfinite(dt) or dt <= 0.0:
        return values
    window = int(round(window_s / dt))
    window = max(5, window)
    if window % 2 == 0:
        window += 1
    if window >= len(values):
        window = len(values) - 1 if len(values) % 2 == 0 else len(values)
    if window < 5:
        return values
    if savgol_filter is not None:
        return savgol_filter(values, window_length=window, polyorder=2, mode="interp")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def derivative_consistency(
    frame: pd.DataFrame,
    output_csv: Path,
    output_png: Path,
) -> pd.DataFrame:
    specs = [
        ("linear", "x", "vehicle_local_position.vx", "vehicle_local_position.ax"),
        ("linear", "y", "vehicle_local_position.vy", "vehicle_local_position.ay"),
        ("linear", "z", "vehicle_local_position.vz", "vehicle_local_position.az"),
        ("angular", "x", "vehicle_angular_velocity.xyz[0]", "vehicle_angular_velocity.xyz_derivative[0]"),
        ("angular", "y", "vehicle_angular_velocity.xyz[1]", "vehicle_angular_velocity.xyz_derivative[1]"),
        ("angular", "z", "vehicle_angular_velocity.xyz[2]", "vehicle_angular_velocity.xyz_derivative[2]"),
    ]
    rows: list[dict[str, float | str | int]] = []
    scatter_samples: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    for kind, axis, state_col, derivative_col in specs:
        estimated_parts: list[np.ndarray] = []
        logged_parts: list[np.ndarray] = []
        for _, group in iter_log_groups(frame, [state_col, derivative_col]):
            time_s = group["time_s"].to_numpy(dtype=float)
            state = group[state_col].to_numpy(dtype=float)
            logged = group[derivative_col].to_numpy(dtype=float)
            mask = finite_mask(time_s, state, logged)
            time_s = time_s[mask]
            state = state[mask]
            logged = logged[mask]
            if len(time_s) < 32:
                continue
            order = np.argsort(time_s)
            time_s = time_s[order]
            state = state[order]
            logged = logged[order]
            dt_values = np.diff(time_s)
            dt_values = dt_values[np.isfinite(dt_values) & (dt_values > 0.0)]
            if len(dt_values) == 0:
                continue
            dt = float(np.median(dt_values))
            state_smooth = smooth_for_derivative(state, dt)
            estimated = np.gradient(state_smooth, time_s)
            valid = finite_mask(estimated, logged)
            estimated_parts.append(estimated[valid])
            logged_parts.append(logged[valid])

        if estimated_parts:
            estimated_all = np.concatenate(estimated_parts)
            logged_all = np.concatenate(logged_parts)
        else:
            estimated_all = np.array([], dtype=float)
            logged_all = np.array([], dtype=float)

        corr = pearson_corr(estimated_all, logged_all) if len(estimated_all) else float("nan")
        rmse = (
            float(np.sqrt(np.nanmean((estimated_all - logged_all) ** 2)))
            if len(estimated_all)
            else float("nan")
        )
        logged_std = float(np.nanstd(logged_all)) if len(logged_all) else float("nan")
        rows.append(
            {
                "kind": kind,
                "axis": axis,
                "corr": corr,
                "rmse": rmse,
                "rmse_to_logged_std": rmse / logged_std if logged_std > 0 else float("nan"),
                "sample_count": int(len(estimated_all)),
            },
        )
        if len(estimated_all):
            sample_count = min(25000, len(estimated_all))
            sample_index = np.linspace(0, len(estimated_all) - 1, sample_count).astype(int)
            scatter_samples[(kind, axis)] = (logged_all[sample_index], estimated_all[sample_index])

    summary = pd.DataFrame(rows)
    summary.to_csv(output_csv, index=False)

    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for ax, (kind, axis, _, _) in zip(axes.ravel(), specs):
        row = summary[(summary["kind"] == kind) & (summary["axis"] == axis)].iloc[0]
        if (kind, axis) in scatter_samples:
            logged, estimated = scatter_samples[(kind, axis)]
            ax.scatter(logged, estimated, s=1, alpha=0.08, color="#4c78a8")
            finite = finite_mask(logged, estimated)
            if int(finite.sum()) > 3:
                lo = float(np.nanpercentile(np.concatenate([logged[finite], estimated[finite]]), 1))
                hi = float(np.nanpercentile(np.concatenate([logged[finite], estimated[finite]]), 99))
                ax.plot([lo, hi], [lo, hi], color="#d62728", linewidth=1.0)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
        ax.set_title(
            f"{kind} {axis}: corr={row['corr']:.2f}, RMSE/std={row['rmse_to_logged_std']:.2f}",
        )
        ax.set_xlabel("logged derivative")
        ax.set_ylabel("smoothed finite-difference derivative")
        ax.grid(alpha=0.25)
    fig.suptitle("Derivative consistency check")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return summary


def lagged_correlation_for_frame(
    frame: pd.DataFrame,
    qbar: np.ndarray,
    signal_by_target: dict[str, np.ndarray],
    lags_s: np.ndarray,
) -> pd.DataFrame:
    work = frame.loc[:, ["log_id", "time_s"]].reset_index(drop=True).copy()
    work["qbar"] = qbar
    for target in TARGET_COLUMNS:
        work[target] = signal_by_target[target]

    rows: list[dict[str, float | str]] = []
    for target in TARGET_COLUMNS:
        for lag_s in lags_s:
            corrs: list[float] = []
            weights: list[int] = []
            for _, group in work.groupby("log_id", sort=False):
                group = group.sort_values("time_s")
                time_s = group["time_s"].to_numpy(dtype=float)
                qbar_values = group["qbar"].to_numpy(dtype=float)
                signal = group[target].to_numpy(dtype=float)
                mask = finite_mask(time_s, qbar_values, signal)
                time_s = time_s[mask]
                qbar_values = qbar_values[mask]
                signal = signal[mask]
                if len(time_s) < 16:
                    continue
                shifted = np.interp(time_s - lag_s, time_s, qbar_values, left=np.nan, right=np.nan)
                corr = pearson_corr(shifted, signal)
                if np.isfinite(corr):
                    corrs.append(corr)
                    weights.append(len(time_s))
            rows.append(
                {
                    "target": target,
                    "lag_s": float(lag_s),
                    "corr": weighted_average(corrs, weights),
                },
            )
    return pd.DataFrame(rows)


def compute_qbar_lag_correlations(
    all_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    all_features_df: pd.DataFrame,
    test_features_df: pd.DataFrame,
    residuals_df: pd.DataFrame,
    output_csv: Path,
    output_png: Path,
    *,
    lag_max_s: float,
    lag_step_s: float,
) -> pd.DataFrame:
    lags_s = np.arange(-lag_max_s, lag_max_s + lag_step_s * 0.5, lag_step_s)
    label_signals = {
        target: all_frame[target].to_numpy(dtype=float)
        for target in TARGET_COLUMNS
    }
    residual_signals = {
        target: residuals_df[target].to_numpy(dtype=float)
        for target in TARGET_COLUMNS
    }

    labels = lagged_correlation_for_frame(
        all_frame,
        all_features_df["dynamic_pressure_pa"].to_numpy(dtype=float),
        label_signals,
        lags_s,
    )
    labels["signal"] = "label"
    residuals = lagged_correlation_for_frame(
        test_frame,
        test_features_df["dynamic_pressure_pa"].to_numpy(dtype=float),
        residual_signals,
        lags_s,
    )
    residuals["signal"] = "residual"
    result = pd.concat([labels, residuals], ignore_index=True)
    result.to_csv(output_csv, index=False)

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, sharey=True)
    for ax, target in zip(axes.ravel(), TARGET_COLUMNS):
        for signal, color in [("label", "#1f77b4"), ("residual", "#d62728")]:
            subset = result[(result["target"] == target) & (result["signal"] == signal)]
            ax.plot(subset["lag_s"], subset["corr"], label=signal, color=color, linewidth=1.4)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(target)
        ax.grid(alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("qbar lag [s]; positive means qbar leads signal")
    for ax in axes[:, 0]:
        ax.set_ylabel("Pearson r")
    axes[0, 0].legend()
    fig.suptitle("Dynamic pressure lag correlation")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return result


def plot_target_time_window(
    frame: pd.DataFrame,
    output_png: Path,
    *,
    duration_s: float = 4.0,
) -> None:
    group_lengths = frame.groupby("log_id", sort=False).size().sort_values(ascending=False)
    if len(group_lengths) == 0:
        return
    log_id = group_lengths.index[0]
    group = frame[frame["log_id"] == log_id].sort_values("time_s")
    time_s = group["time_s"].to_numpy(dtype=float)
    if len(time_s) < 32:
        return
    start = float(np.nanpercentile(time_s, 10))
    end = start + duration_s
    mask = (time_s >= start) & (time_s <= end)
    if int(mask.sum()) < 32:
        mask = np.arange(len(group)) < min(len(group), 800)
    time_window = time_s[mask] - time_s[mask][0]
    f0 = float(np.nanmedian(group["cycle_flap_frequency_hz"].to_numpy(dtype=float)))
    dt_values = np.diff(time_s)
    dt_values = dt_values[np.isfinite(dt_values) & (dt_values > 0.0)]
    dt = float(np.median(dt_values)) if len(dt_values) else 0.01
    smooth_window_s = min(0.20, max(0.05, 1.0 / f0 if f0 > 0 else 0.1))

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for ax, target in zip(axes.ravel(), TARGET_COLUMNS):
        raw = group[target].to_numpy(dtype=float)[mask]
        smooth = smooth_for_derivative(raw, dt, window_s=smooth_window_s)
        ax.plot(time_window, raw, color="#7f7f7f", linewidth=0.6, alpha=0.55, label="raw")
        ax.plot(time_window, smooth, color="#1f77b4", linewidth=1.3, label="smoothed")
        ax.set_title(target)
        ax.set_ylabel(target)
        ax.grid(alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("time in selected log window [s]")
    axes[0, 0].legend()
    fig.suptitle(f"Raw vs smoothed target window; log_id={log_id}")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def load_test_metrics(metrics_json: Path) -> dict[str, float]:
    if not metrics_json.exists():
        return {}
    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    per_target = metrics.get("test", {}).get("per_target", {})
    return {
        target: float(per_target.get(target, {}).get("r2", float("nan")))
        for target in TARGET_COLUMNS
    }


def classify_target(row: pd.Series) -> str:
    model_r2 = row["test_r2"]
    phase_r2 = row["phase_r2"]
    high_fraction = row["psd_high_frequency_fraction_25_to_45hz"]
    derivative_corr = row["derivative_consistency_corr"]

    if model_r2 >= 0.8 and phase_r2 >= 0.65 and high_fraction <= 0.15:
        return "strong real periodic signal"
    if model_r2 >= 0.65 and phase_r2 >= 0.55:
        return "mostly real, moderate noise"
    if model_r2 < 0.5 and phase_r2 < 0.25:
        if np.isfinite(derivative_corr) and derivative_corr < 0.55:
            return "weak signal plus derivative noise"
        return "weak phase-locked signal, likely noisy"
    if model_r2 < 0.55 and high_fraction >= 0.25:
        return "mixed but noise-dominated"
    return "mixed real signal and noise"


def make_summary(
    phase_stats: dict[str, dict[str, float]],
    residual_phase_stats: dict[str, dict[str, float]],
    psd_stats: dict[str, dict[str, float]],
    residual_corr: pd.DataFrame,
    derivative_summary: pd.DataFrame,
    test_r2: dict[str, float],
    target_stds: dict[str, float],
    output_csv: Path,
) -> pd.DataFrame:
    derivative_map = {
        "fx_b": ("linear", "x"),
        "fy_b": ("linear", "y"),
        "fz_b": ("linear", "z"),
        "mx_b": ("angular", "x"),
        "my_b": ("angular", "y"),
        "mz_b": ("angular", "z"),
    }
    rows: list[dict[str, float | str]] = []
    for target in TARGET_COLUMNS:
        kind, axis = derivative_map[target]
        derivative_row = derivative_summary[
            (derivative_summary["kind"] == kind) & (derivative_summary["axis"] == axis)
        ]
        derivative_corr = (
            float(derivative_row["corr"].iloc[0])
            if len(derivative_row)
            else float("nan")
        )
        residual_corr_row = residual_corr.loc[target]
        max_corr_feature = str(residual_corr_row.abs().idxmax())
        max_corr_value = float(residual_corr_row.abs().max())
        row = {
            "target": target,
            "test_r2": test_r2.get(target, float("nan")),
            "target_std": target_stds.get(target, float("nan")),
            "phase_r2": phase_stats[target]["phase_r2"],
            "phase_median_mad_to_std": phase_stats[target]["phase_median_mad_to_std"],
            "psd_harmonic_band_fraction_1x_to_6x": psd_stats[target][
                "psd_harmonic_band_fraction_1x_to_6x"
            ],
            "psd_high_frequency_fraction_25_to_45hz": psd_stats[target][
                "psd_high_frequency_fraction_25_to_45hz"
            ],
            "residual_phase_r2": residual_phase_stats[target]["phase_r2"],
            "residual_median_mad_to_std": residual_phase_stats[target]["phase_median_mad_to_std"],
            "max_abs_residual_feature_corr": max_corr_value,
            "max_residual_corr_feature": max_corr_feature,
            "derivative_consistency_corr": derivative_corr,
        }
        row["judgement"] = classify_target(pd.Series(row))
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_csv, index=False)
    return summary


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    output = []
    output.append("| " + " | ".join(headers) + " |")
    output.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        output.append("| " + " | ".join(row) + " |")
    return "\n".join(output)


def write_readme(
    output_dir: Path,
    summary: pd.DataFrame,
    derivative_summary: pd.DataFrame,
) -> None:
    compact_rows: list[list[str]] = []
    for _, row in summary.iterrows():
        compact_rows.append(
            [
                str(row["target"]),
                f"{row['test_r2']:.3f}",
                f"{row['phase_r2']:.3f}",
                f"{row['psd_harmonic_band_fraction_1x_to_6x']:.2f}",
                f"{row['psd_high_frequency_fraction_25_to_45hz']:.2f}",
                f"{row['derivative_consistency_corr']:.2f}",
                str(row["judgement"]),
            ],
        )
    derivative_rows: list[list[str]] = []
    for _, row in derivative_summary.iterrows():
        derivative_rows.append(
            [
                str(row["kind"]),
                str(row["axis"]),
                f"{row['corr']:.3f}",
                f"{row['rmse_to_logged_std']:.3f}",
                str(int(row["sample_count"])),
            ],
        )

    readme = f"""# Label Noise Diagnostics

Dataset split:
`{DEFAULT_SPLIT_ROOT}`

Model used for residual diagnostics:
`{DEFAULT_MODEL_BUNDLE}`

## Main judgement

{markdown_table(
    [
        "target",
        "test R2",
        "phase R2",
        "harmonic frac",
        "25-45 Hz frac",
        "deriv corr",
        "judgement",
    ],
    compact_rows,
)}

Interpretation:

- High phase R2 means the label has a repeatable phase-locked flapping pattern.
- High 1x-6x harmonic fraction means target energy is concentrated near flapping harmonics.
- High 25-45 Hz fraction means more high-frequency content, which is more suspicious as label noise.
- Low residual phase R2 means the model residual is not mainly a missed phase pattern.
- Low derivative consistency corr means the acceleration/angular-acceleration source behind the label is noisy or filtered differently.

## Derivative consistency

{markdown_table(
    ["kind", "axis", "corr", "RMSE/std", "samples"],
    derivative_rows,
)}

## Files

- `target_time_window.png`: raw labels and smoothed labels in one representative log window.
- `target_psd.png`: target frequency spectra with flapping harmonics marked.
- `target_phase_bins.png`: target median +/- robust MAD by corrected phase.
- `residual_phase_bins.png`: model residual median +/- robust MAD by corrected phase.
- `residual_feature_correlations.png`: residual correlation with candidate explanatory inputs.
- `derivative_consistency.png`: logged acceleration/angular acceleration vs velocity/gyro finite differences.
- `airspeed_qbar_lag_correlation.png`: dynamic-pressure lag correlation with labels and residuals.
- `label_noise_diagnostic_summary.csv`: numeric target-level summary.
- `residual_feature_correlations.csv`: residual-feature correlation matrix.
- `derivative_consistency_summary.csv`: derivative consistency numbers.
- `qbar_lag_correlation.csv`: lag-correlation curves behind the qbar plot.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = read_split_frames(args.split_root)
    all_frame = concat_frames(frames)
    test_frame = frames["test"].reset_index(drop=True)

    bundle = torch.load(args.model_bundle, map_location="cpu", weights_only=False)
    all_features_df, _ = prepare_feature_target_frames(
        all_frame,
        bundle["feature_columns"],
        bundle["target_columns"],
    )
    test_features_df, _, residuals_df = prepare_model_residuals(
        args.model_bundle,
        test_frame,
        device=args.device,
    )

    target_values = {
        target: all_frame[target].to_numpy(dtype=float)
        for target in TARGET_COLUMNS
    }
    target_stds = {
        target: float(np.nanstd(values))
        for target, values in target_values.items()
    }
    residual_values = {
        target: residuals_df[target].to_numpy(dtype=float)
        for target in TARGET_COLUMNS
    }

    plot_target_time_window(all_frame, args.output_dir / "target_time_window.png")
    psd_stats = compute_psd_diagnostics(
        all_frame,
        args.output_dir / "target_psd.png",
        max_hz=args.max_psd_hz,
        grid_hz=args.psd_grid_hz,
    )
    phase_stats = plot_phase_bins(
        args.output_dir / "target_phase_bins.png",
        all_frame["phase_corrected_rad"].to_numpy(dtype=float),
        target_values,
        bins=args.phase_bins,
        title="Target phase-bin median +/- robust MAD",
    )
    residual_phase_stats = plot_phase_bins(
        args.output_dir / "residual_phase_bins.png",
        test_frame["phase_corrected_rad"].to_numpy(dtype=float),
        residual_values,
        bins=args.phase_bins,
        title="Test residual phase-bin median +/- robust MAD",
        ylabel_suffix=" residual",
    )
    residual_corr = compute_residual_feature_correlations(
        test_features_df,
        residuals_df,
        args.output_dir / "residual_feature_correlations.csv",
        args.output_dir / "residual_feature_correlations.png",
    )
    derivative_summary = derivative_consistency(
        all_frame,
        args.output_dir / "derivative_consistency_summary.csv",
        args.output_dir / "derivative_consistency.png",
    )
    compute_qbar_lag_correlations(
        all_frame.reset_index(drop=True),
        test_frame.reset_index(drop=True),
        all_features_df.reset_index(drop=True),
        test_features_df.reset_index(drop=True),
        residuals_df.reset_index(drop=True),
        args.output_dir / "qbar_lag_correlation.csv",
        args.output_dir / "airspeed_qbar_lag_correlation.png",
        lag_max_s=args.lag_max_s,
        lag_step_s=args.lag_step_s,
    )
    summary = make_summary(
        phase_stats,
        residual_phase_stats,
        psd_stats,
        residual_corr,
        derivative_summary,
        load_test_metrics(args.metrics_json),
        target_stds,
        args.output_dir / "label_noise_diagnostic_summary.csv",
    )
    write_readme(args.output_dir, summary, derivative_summary)

    print(f"wrote diagnostics to {args.output_dir}")
    print(summary.loc[:, ["target", "test_r2", "phase_r2", "judgement"]].to_string(index=False))


if __name__ == "__main__":
    main()
