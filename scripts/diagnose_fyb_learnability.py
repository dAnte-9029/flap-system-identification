#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_BANDS = (
    ("low_0_1hz", 0.0, 1.0),
    ("mid_1_3hz", 1.0, 3.0),
    ("flap_main", np.nan, np.nan),
    ("high_8_25hz", 8.0, 25.0),
)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    true = y_true[mask]
    pred = y_pred[mask]
    ss_res = float(np.sum(np.square(pred - true)))
    ss_tot = float(np.sum(np.square(true - true.mean())))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(y_pred[mask] - y_true[mask]))))


def corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(mask) < 2:
        return float("nan")
    true = y_true[mask]
    pred = y_pred[mask]
    if np.std(true) < 1e-12 or np.std(pred) < 1e-12:
        return float("nan")
    return float(np.corrcoef(true, pred)[0, 1])


def nominal_sample_rate_hz(time_s: np.ndarray) -> float:
    dt = np.diff(time_s.astype(float))
    valid = dt[np.isfinite(dt) & (dt > 0.0)]
    if len(valid) == 0:
        return float("nan")
    return float(1.0 / np.median(valid))


def fft_filter(
    values: np.ndarray,
    *,
    sample_rate_hz: float,
    low_hz: float | None = None,
    high_hz: float | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.full_like(values, np.nan, dtype=float)
    finite = np.isfinite(values)
    if np.sum(finite) < 4 or not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        return output

    filled = values.copy()
    if not finite.all():
        valid_idx = np.flatnonzero(finite)
        filled[~finite] = np.interp(np.flatnonzero(~finite), valid_idx, values[valid_idx])
    mean = float(np.mean(filled))
    centered = filled - mean
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    spectrum = np.fft.rfft(centered)

    low = 0.0 if low_hz is None else float(low_hz)
    high = sample_rate_hz / 2.0 if high_hz is None else float(high_hz)
    mask = (freqs >= low) & (freqs <= high)
    if low <= 0.0:
        spectrum[~mask] = 0.0
        filtered = np.fft.irfft(spectrum, n=len(centered)) + mean
    else:
        spectrum[~mask] = 0.0
        filtered = np.fft.irfft(spectrum, n=len(centered))
    output[finite] = filtered[finite]
    return output


def _grouped_segments(frame: pd.DataFrame) -> list[pd.DataFrame]:
    group_columns = [column for column in ("log_id", "segment_id") if column in frame.columns]
    if not group_columns:
        return [frame.sort_values("time_s")]
    groups = []
    for _, group in frame.groupby(group_columns, sort=False):
        groups.append(group.sort_values("time_s"))
    return groups


def _filtered_arrays_for_band(
    frame: pd.DataFrame,
    *,
    target: str,
    low_hz: float | None,
    high_hz: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    for group in _grouped_segments(frame):
        if len(group) < 8:
            continue
        sample_rate_hz = nominal_sample_rate_hz(group["time_s"].to_numpy(dtype=float))
        true_values = group[f"true_{target}"].to_numpy(dtype=float)
        pred_values = group[f"pred_{target}"].to_numpy(dtype=float)
        true_parts.append(fft_filter(true_values, sample_rate_hz=sample_rate_hz, low_hz=low_hz, high_hz=high_hz))
        pred_parts.append(fft_filter(pred_values, sample_rate_hz=sample_rate_hz, low_hz=low_hz, high_hz=high_hz))
    if not true_parts:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.concatenate(true_parts), np.concatenate(pred_parts)


def _target_values(frame: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        frame[f"true_{target}"].to_numpy(dtype=float),
        frame[f"pred_{target}"].to_numpy(dtype=float),
    )


def compute_band_r2_table(
    frame: pd.DataFrame,
    *,
    target: str = "fy_b",
    bands: Iterable[tuple[str, float, float]],
) -> pd.DataFrame:
    raw_true, _ = _target_values(frame, target)
    raw_var = float(np.nanvar(raw_true))
    rows: list[dict[str, float | str | int]] = []
    for band_name, low_hz, high_hz in bands:
        true_band, pred_band = _filtered_arrays_for_band(
            frame,
            target=target,
            low_hz=low_hz,
            high_hz=high_hz,
        )
        rows.append(
            {
                "band": band_name,
                "low_hz": float(low_hz),
                "high_hz": float(high_hz),
                "sample_count": int(np.sum(np.isfinite(true_band) & np.isfinite(pred_band))),
                "true_std": float(np.nanstd(true_band)),
                "pred_std": float(np.nanstd(pred_band)),
                "true_variance_fraction": float(np.nanvar(true_band) / raw_var) if raw_var > 0.0 else float("nan"),
                "rmse": rmse(true_band, pred_band),
                "r2": r2_score(true_band, pred_band),
                "corr": corrcoef(true_band, pred_band),
            }
        )
    return pd.DataFrame(rows)


def _rolling_median_by_segment(frame: pd.DataFrame, column: str, window_samples: int) -> np.ndarray:
    output = np.full(len(frame), np.nan, dtype=float)
    for group in _grouped_segments(frame):
        values = group[column].to_numpy(dtype=float)
        filtered = (
            pd.Series(values)
            .rolling(window=window_samples, center=True, min_periods=max(1, window_samples // 3))
            .median()
            .to_numpy(dtype=float)
        )
        output[group.index.to_numpy()] = filtered
    return output


def compute_filter_sweep_table(
    frame: pd.DataFrame,
    *,
    target: str = "fy_b",
    lowpass_cutoffs_hz: tuple[float, ...] = (1.0, 3.0, 5.0, 8.0, 12.0),
    median_windows_s: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> pd.DataFrame:
    raw_true, raw_pred = _target_values(frame, target)
    rows = [
        {
            "filter": "raw",
            "parameter": 0.0,
            "sample_count": int(np.sum(np.isfinite(raw_true) & np.isfinite(raw_pred))),
            "true_std": float(np.nanstd(raw_true)),
            "pred_std": float(np.nanstd(raw_pred)),
            "rmse_filtered_pair": rmse(raw_true, raw_pred),
            "r2_filtered_pair": r2_score(raw_true, raw_pred),
            "rmse_raw_pred_vs_filtered_true": rmse(raw_true, raw_pred),
            "r2_raw_pred_vs_filtered_true": r2_score(raw_true, raw_pred),
        }
    ]
    for cutoff in lowpass_cutoffs_hz:
        filt_true, filt_pred = _filtered_arrays_for_band(frame, target=target, low_hz=0.0, high_hz=cutoff)
        rows.append(
            {
                "filter": "fft_lowpass_hz",
                "parameter": float(cutoff),
                "sample_count": int(np.sum(np.isfinite(filt_true) & np.isfinite(filt_pred))),
                "true_std": float(np.nanstd(filt_true)),
                "pred_std": float(np.nanstd(filt_pred)),
                "rmse_filtered_pair": rmse(filt_true, filt_pred),
                "r2_filtered_pair": r2_score(filt_true, filt_pred),
                "rmse_raw_pred_vs_filtered_true": rmse(filt_true, raw_pred[: len(filt_true)]),
                "r2_raw_pred_vs_filtered_true": r2_score(filt_true, raw_pred[: len(filt_true)]),
            }
        )
    median_rate_hz = nominal_sample_rate_hz(frame.sort_values("time_s")["time_s"].to_numpy(dtype=float))
    for window_s in median_windows_s:
        window_samples = max(3, int(round(window_s * median_rate_hz)))
        if window_samples % 2 == 0:
            window_samples += 1
        med_true = _rolling_median_by_segment(frame, f"true_{target}", window_samples)
        med_pred = _rolling_median_by_segment(frame, f"pred_{target}", window_samples)
        rows.append(
            {
                "filter": "rolling_median_s",
                "parameter": float(window_s),
                "sample_count": int(np.sum(np.isfinite(med_true) & np.isfinite(med_pred))),
                "true_std": float(np.nanstd(med_true)),
                "pred_std": float(np.nanstd(med_pred)),
                "rmse_filtered_pair": rmse(med_true, med_pred),
                "r2_filtered_pair": r2_score(med_true, med_pred),
                "rmse_raw_pred_vs_filtered_true": rmse(med_true, raw_pred),
                "r2_raw_pred_vs_filtered_true": r2_score(med_true, raw_pred),
            }
        )
    return pd.DataFrame(rows)


def compute_spike_capture_table(
    frame: pd.DataFrame,
    *,
    target: str = "fy_b",
    quantiles: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> pd.DataFrame:
    true_values, pred_values = _target_values(frame, target)
    abs_true = np.abs(true_values)
    rows = []
    for quantile in quantiles:
        threshold = float(np.nanquantile(abs_true, quantile))
        mask = np.isfinite(abs_true) & (abs_true >= threshold)
        true_sel = true_values[mask]
        pred_sel = pred_values[mask]
        sign_agreement = np.mean(np.sign(true_sel) == np.sign(pred_sel)) if len(true_sel) else np.nan
        rows.append(
            {
                "quantile": float(quantile),
                "threshold_abs_true": threshold,
                "sample_count": int(len(true_sel)),
                "true_abs_mean": float(np.mean(np.abs(true_sel))) if len(true_sel) else np.nan,
                "pred_abs_mean": float(np.mean(np.abs(pred_sel))) if len(pred_sel) else np.nan,
                "amplitude_ratio": float(np.mean(np.abs(pred_sel)) / np.mean(np.abs(true_sel))) if len(true_sel) and np.mean(np.abs(true_sel)) > 0 else np.nan,
                "sign_agreement": float(sign_agreement),
                "rmse": rmse(true_sel, pred_sel),
                "r2": r2_score(true_sel, pred_sel),
            }
        )
    return pd.DataFrame(rows)


def compute_psd_table(
    frame: pd.DataFrame,
    *,
    target: str = "fy_b",
    nperseg: int = 512,
) -> pd.DataFrame:
    psd_true_parts: list[np.ndarray] = []
    psd_pred_parts: list[np.ndarray] = []
    psd_resid_parts: list[np.ndarray] = []
    freq_ref: np.ndarray | None = None
    for group in _grouped_segments(frame):
        if len(group) < max(64, nperseg // 2):
            continue
        sample_rate_hz = nominal_sample_rate_hz(group["time_s"].to_numpy(dtype=float))
        if not np.isfinite(sample_rate_hz):
            continue
        window_size = min(nperseg, len(group))
        if window_size < 64:
            continue
        step = max(1, window_size // 2)
        window = np.hanning(window_size)
        window_power = float(np.sum(window**2))
        freqs = np.fft.rfftfreq(window_size, d=1.0 / sample_rate_hz)
        for start in range(0, len(group) - window_size + 1, step):
            chunk = group.iloc[start : start + window_size]
            true_values = chunk[f"true_{target}"].to_numpy(dtype=float)
            pred_values = chunk[f"pred_{target}"].to_numpy(dtype=float)
            if not np.isfinite(true_values).all() or not np.isfinite(pred_values).all():
                continue
            true_centered = true_values - np.mean(true_values)
            pred_centered = pred_values - np.mean(pred_values)
            resid_centered = (true_values - pred_values) - np.mean(true_values - pred_values)
            true_psd = np.square(np.abs(np.fft.rfft(true_centered * window))) / (sample_rate_hz * window_power)
            pred_psd = np.square(np.abs(np.fft.rfft(pred_centered * window))) / (sample_rate_hz * window_power)
            resid_psd = np.square(np.abs(np.fft.rfft(resid_centered * window))) / (sample_rate_hz * window_power)
            if freq_ref is None:
                freq_ref = freqs
            if len(freqs) != len(freq_ref) or not np.allclose(freqs, freq_ref):
                true_psd = np.interp(freq_ref, freqs, true_psd)
                pred_psd = np.interp(freq_ref, freqs, pred_psd)
                resid_psd = np.interp(freq_ref, freqs, resid_psd)
            psd_true_parts.append(true_psd)
            psd_pred_parts.append(pred_psd)
            psd_resid_parts.append(resid_psd)
    if freq_ref is None or not psd_true_parts:
        return pd.DataFrame(columns=["freq_hz", "true_psd", "pred_psd"])
    return pd.DataFrame(
        {
            "freq_hz": freq_ref,
            "true_psd": np.mean(np.vstack(psd_true_parts), axis=0),
            "pred_psd": np.mean(np.vstack(psd_pred_parts), axis=0),
            "residual_psd": np.mean(np.vstack(psd_resid_parts), axis=0),
        }
    )


def compute_phase_binned_table(
    frame: pd.DataFrame,
    *,
    target: str = "fy_b",
    highpass_hz: float = 8.0,
    phase_bins: int = 36,
    phase_column: str = "phase_corrected_rad",
) -> pd.DataFrame:
    if phase_column not in frame.columns:
        raise ValueError(f"Missing phase column: {phase_column}")

    phase_parts: list[np.ndarray] = []
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    for group in _grouped_segments(frame):
        if len(group) < 8:
            continue
        sample_rate_hz = nominal_sample_rate_hz(group["time_s"].to_numpy(dtype=float))
        true_values = group[f"true_{target}"].to_numpy(dtype=float)
        pred_values = group[f"pred_{target}"].to_numpy(dtype=float)
        true_hpf = true_values - fft_filter(true_values, sample_rate_hz=sample_rate_hz, low_hz=0.0, high_hz=highpass_hz)
        pred_hpf = pred_values - fft_filter(pred_values, sample_rate_hz=sample_rate_hz, low_hz=0.0, high_hz=highpass_hz)
        phase = np.mod(group[phase_column].to_numpy(dtype=float), 2.0 * np.pi)
        finite = np.isfinite(phase) & np.isfinite(true_hpf) & np.isfinite(pred_hpf)
        phase_parts.append(phase[finite])
        true_parts.append(true_hpf[finite])
        pred_parts.append(pred_hpf[finite])

    if not phase_parts:
        return pd.DataFrame()

    phase_all = np.concatenate(phase_parts)
    true_all = np.concatenate(true_parts)
    pred_all = np.concatenate(pred_parts)
    edges = np.linspace(0.0, 2.0 * np.pi, phase_bins + 1)
    bin_index = np.clip(np.digitize(phase_all, edges, right=False) - 1, 0, phase_bins - 1)
    rows: list[dict[str, float | int]] = []
    for idx in range(phase_bins):
        mask = bin_index == idx
        true_bin = true_all[mask]
        pred_bin = pred_all[mask]
        rows.append(
            {
                "phase_bin": int(idx),
                "phase_center_rad": float(0.5 * (edges[idx] + edges[idx + 1])),
                "sample_count": int(len(true_bin)),
                "true_hpf_mean": float(np.mean(true_bin)) if len(true_bin) else np.nan,
                "pred_hpf_mean": float(np.mean(pred_bin)) if len(pred_bin) else np.nan,
                "true_hpf_std": float(np.std(true_bin)) if len(true_bin) else np.nan,
                "pred_hpf_std": float(np.std(pred_bin)) if len(pred_bin) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_hpf_correlation_table(
    frame: pd.DataFrame,
    *,
    target: str = "fy_b",
    highpass_hz: float = 8.0,
) -> pd.DataFrame:
    candidates = [
        "phase_corrected_sin",
        "phase_corrected_cos",
        "wing_stroke_angle_rad",
        "cycle_flap_frequency_hz",
        "servo_rudder",
        "servo_left_elevon",
        "servo_right_elevon",
        "aileron_like",
        "vehicle_angular_velocity.xyz[0]",
        "vehicle_angular_velocity.xyz[2]",
        "vehicle_local_position.ay",
        "airspeed_validated.true_airspeed_m_s",
        "wind.windspeed_north",
        "wind.windspeed_east",
    ]
    available = [column for column in candidates if column in frame.columns]
    target_parts: list[np.ndarray] = []
    feature_parts: dict[str, list[np.ndarray]] = {column: [] for column in available}
    for group in _grouped_segments(frame):
        if len(group) < 8:
            continue
        sample_rate_hz = nominal_sample_rate_hz(group["time_s"].to_numpy(dtype=float))
        true_values = group[f"true_{target}"].to_numpy(dtype=float)
        target_hpf = true_values - fft_filter(true_values, sample_rate_hz=sample_rate_hz, low_hz=0.0, high_hz=highpass_hz)
        target_parts.append(target_hpf)
        for column in available:
            values = group[column].to_numpy(dtype=float)
            feature_hpf = values - fft_filter(values, sample_rate_hz=sample_rate_hz, low_hz=0.0, high_hz=highpass_hz)
            feature_parts[column].append(feature_hpf)
    if not target_parts:
        return pd.DataFrame()
    target_all = np.concatenate(target_parts)
    rows = []
    for column in available:
        feature_all = np.concatenate(feature_parts[column])
        rows.append(
            {
                "feature": column,
                "corr_with_true_hpf": corrcoef(target_all, feature_all),
                "abs_corr_with_true_hpf": abs(corrcoef(target_all, feature_all)),
                "feature_hpf_std": float(np.nanstd(feature_all)),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_corr_with_true_hpf", ascending=False).reset_index(drop=True)


def resolve_bands(frame: pd.DataFrame) -> list[tuple[str, float, float]]:
    if "cycle_flap_frequency_hz" in frame.columns:
        flap_freq = float(np.nanmedian(frame["cycle_flap_frequency_hz"].to_numpy(dtype=float)))
    elif "flap_frequency_hz" in frame.columns:
        flap_freq = float(np.nanmedian(frame["flap_frequency_hz"].to_numpy(dtype=float)))
    else:
        flap_freq = 4.5
    bands = []
    for name, low, high in DEFAULT_BANDS:
        if name == "flap_main":
            bands.append((f"flap_main_{flap_freq:.2f}hz_pm0.75", max(0.0, flap_freq - 0.75), flap_freq + 0.75))
        else:
            bands.append((name, float(low), float(high)))
    return bands


def plot_psd(psd: pd.DataFrame, output_path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(psd["freq_hz"], psd["true_psd"], label="true fy_b", color="#1f2933")
    ax.semilogy(psd["freq_hz"], psd["pred_psd"], label="pred fy_b", color="#d95f02")
    if "residual_psd" in psd.columns:
        ax.semilogy(psd["freq_hz"], psd["residual_psd"], label="residual", color="#7570b3", alpha=0.9)
    ax.set_xlim(0.0, min(30.0, float(psd["freq_hz"].max())))
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_markdown_report(
    output_path: Path,
    *,
    split_name: str,
    aligned_path: Path,
    band_table: pd.DataFrame,
    filter_table: pd.DataFrame,
    spike_table: pd.DataFrame,
    phase_table: pd.DataFrame,
    corr_table: pd.DataFrame,
    psd_path: Path,
) -> None:
    lines = [
        f"# fy_b Learnability Diagnostics: {split_name}",
        "",
        f"- aligned predictions: `{aligned_path}`",
        f"- PSD plot: `{psd_path.name}`",
        "",
        "## Band R2",
        "",
        band_table.to_csv(index=False),
        "",
        "## Filter Sweep",
        "",
        filter_table.to_csv(index=False),
        "",
        "## Spike Capture",
        "",
        spike_table.to_csv(index=False),
        "",
        "## Phase-binned HPF",
        "",
        phase_table.to_csv(index=False),
        "",
        "## HPF Correlations",
        "",
        corr_table.to_csv(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines))


def run_diagnostics(*, aligned_path: Path, output_dir: Path, split_name: str, target: str = "fy_b") -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(aligned_path)
    bands = resolve_bands(frame)
    band_table = compute_band_r2_table(frame, target=target, bands=bands)
    filter_table = compute_filter_sweep_table(frame, target=target)
    spike_table = compute_spike_capture_table(frame, target=target)
    phase_table = compute_phase_binned_table(frame, target=target)
    corr_table = compute_hpf_correlation_table(frame, target=target)
    psd_table = compute_psd_table(frame, target=target)

    paths = {
        "band_r2": output_dir / f"{split_name}_{target}_band_r2.csv",
        "filter_sweep": output_dir / f"{split_name}_{target}_filter_sweep.csv",
        "spike_capture": output_dir / f"{split_name}_{target}_spike_capture.csv",
        "phase_binned_hpf": output_dir / f"{split_name}_{target}_phase_binned_hpf.csv",
        "hpf_correlations": output_dir / f"{split_name}_{target}_hpf_correlations.csv",
        "psd": output_dir / f"{split_name}_{target}_psd.csv",
        "psd_plot": output_dir / f"{split_name}_{target}_psd.png",
        "report": output_dir / f"{split_name}_{target}_learnability.md",
    }
    band_table.to_csv(paths["band_r2"], index=False)
    filter_table.to_csv(paths["filter_sweep"], index=False)
    spike_table.to_csv(paths["spike_capture"], index=False)
    phase_table.to_csv(paths["phase_binned_hpf"], index=False)
    corr_table.to_csv(paths["hpf_correlations"], index=False)
    psd_table.to_csv(paths["psd"], index=False)
    if not psd_table.empty:
        plot_psd(psd_table, paths["psd_plot"], title=f"{split_name} {target} PSD: true vs pred")
    _write_markdown_report(
        paths["report"],
        split_name=split_name,
        aligned_path=aligned_path,
        band_table=band_table,
        filter_table=filter_table,
        spike_table=spike_table,
        phase_table=phase_table,
        corr_table=corr_table,
        psd_path=paths["psd_plot"],
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose whether fy_b target structure is learnable by current predictions.")
    parser.add_argument("--aligned-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--target", default="fy_b")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_diagnostics(
        aligned_path=args.aligned_parquet,
        output_dir=args.output_dir,
        split_name=args.split_name,
        target=args.target,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
