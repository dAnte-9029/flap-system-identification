#!/usr/bin/env python3
"""Analyze DeLaurier residual energy by frequency band."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET_COLUMNS = ("fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b")
KEY_PLOT_TARGETS = ("fx_b", "fz_b", "my_b")
DEFAULT_HIGH_BAND = (8.0, 25.0)
COMPONENT_ORDER = (
    "low_0_1hz",
    "mid_1_3hz",
    "flap_main",
    "harmonic_2f",
    "harmonic_3f",
    "broadband_high_8_25hz_excl_structured",
)


@dataclass(frozen=True)
class Band:
    name: str
    low_hz: float
    high_hz: float
    mask: np.ndarray


def _check_columns(frame: pd.DataFrame, targets: tuple[str, ...]) -> None:
    columns = ["time_s"]
    for target in targets:
        columns.extend([f"label_{target}", f"prior_{target}", f"pred_{target}"])
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"aligned frame is missing required columns: {missing}")


def _grouped_frames(frame: pd.DataFrame) -> list[pd.DataFrame]:
    group_columns = [column for column in ("log_id", "segment_id") if column in frame.columns]
    if not group_columns:
        return [frame.sort_values("time_s").reset_index(drop=True)]
    return [group.sort_values("time_s").reset_index(drop=True) for _, group in frame.groupby(group_columns, sort=False)]


def _sample_rate_hz(time_s: np.ndarray) -> float:
    finite = time_s[np.isfinite(time_s)]
    if len(finite) < 3:
        return float("nan")
    diffs = np.diff(np.sort(finite))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if len(diffs) == 0:
        return float("nan")
    median_dt = float(np.median(diffs))
    return 1.0 / median_dt if median_dt > 0.0 else float("nan")


def _fill_finite(values: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 8:
        return None
    filled = values.copy()
    if not finite.all():
        valid_idx = np.flatnonzero(finite)
        filled[~finite] = np.interp(np.flatnonzero(~finite), valid_idx, values[valid_idx])
    return filled


def _one_sided_fft_energy(values: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    filled = _fill_finite(values)
    if filled is None or len(filled) < 8 or not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        return np.array([], dtype=float), np.array([], dtype=float)
    centered = filled - float(np.mean(filled))
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    spectrum = np.fft.rfft(centered)
    energy = np.square(np.abs(spectrum)) / float(len(centered) ** 2)
    if len(energy) > 2:
        if len(centered) % 2 == 0:
            energy[1:-1] *= 2.0
        else:
            energy[1:] *= 2.0
    return freqs, energy


def _resolve_f0_hz(group: pd.DataFrame) -> float:
    for column in ("cycle_flap_frequency_hz", "flap_frequency_hz"):
        if column in group.columns:
            values = group[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if len(finite):
                f0 = float(np.median(finite))
                if f0 > 0.0:
                    return f0
    return 4.5


def _interval_mask(freqs: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    return (freqs >= low_hz) & (freqs <= high_hz)


def _bands_for_group(
    freqs: np.ndarray,
    *,
    f0_hz: float,
    flap_half_width_hz: float,
    high_band: tuple[float, float],
) -> list[Band]:
    flap_low = max(0.0, f0_hz - flap_half_width_hz)
    flap_high = f0_hz + flap_half_width_hz
    harmonic_2f_low = max(0.0, 2.0 * f0_hz - flap_half_width_hz)
    harmonic_2f_high = 2.0 * f0_hz + flap_half_width_hz
    harmonic_3f_low = max(0.0, 3.0 * f0_hz - flap_half_width_hz)
    harmonic_3f_high = 3.0 * f0_hz + flap_half_width_hz

    # Assign each FFT bin to at most one diagnostic band. Low/mid frequency bins
    # take priority when low flap-frequency segments would otherwise overlap 1f.
    used = np.zeros_like(freqs, dtype=bool)
    low_mask = _interval_mask(freqs, 0.0, 1.0)
    used |= low_mask
    mid_mask = _interval_mask(freqs, 1.0, 3.0) & ~used
    used |= mid_mask
    flap_mask = _interval_mask(freqs, flap_low, flap_high) & ~used
    used |= flap_mask
    harmonic_2f_mask = _interval_mask(freqs, harmonic_2f_low, harmonic_2f_high) & ~used
    used |= harmonic_2f_mask
    harmonic_3f_mask = _interval_mask(freqs, harmonic_3f_low, harmonic_3f_high) & ~used
    used |= harmonic_3f_mask
    high_mask = _interval_mask(freqs, high_band[0], high_band[1]) & ~used

    return [
        Band("low_0_1hz", 0.0, 1.0, low_mask),
        Band("mid_1_3hz", 1.0, 3.0, mid_mask),
        Band("flap_main", flap_low, flap_high, flap_mask),
        Band("harmonic_2f", harmonic_2f_low, harmonic_2f_high, harmonic_2f_mask),
        Band("harmonic_3f", harmonic_3f_low, harmonic_3f_high, harmonic_3f_mask),
        Band(
            f"broadband_high_{high_band[0]:.0f}_{high_band[1]:.0f}hz_excl_structured",
            float(high_band[0]),
            float(high_band[1]),
            high_mask,
        ),
    ]


def frequency_residual_energy_table(
    frame: pd.DataFrame,
    *,
    targets: tuple[str, ...] = TARGET_COLUMNS,
    flap_half_width_hz: float = 0.75,
    high_band: tuple[float, float] = DEFAULT_HIGH_BAND,
) -> pd.DataFrame:
    """Return true and remaining residual energy by frequency band."""

    _check_columns(frame, targets)
    accum: dict[tuple[str, str], dict[str, float | int]] = {}
    target_total_energy = {target: 0.0 for target in targets}
    target_remaining_total_energy = {target: 0.0 for target in targets}

    for group in _grouped_frames(frame):
        if len(group) < 8:
            continue
        sample_rate = _sample_rate_hz(group["time_s"].to_numpy(dtype=float))
        if not np.isfinite(sample_rate) or sample_rate <= 0.0:
            continue
        f0 = _resolve_f0_hz(group)
        for target in targets:
            true_residual = group[f"label_{target}"].to_numpy(dtype=float) - group[f"prior_{target}"].to_numpy(dtype=float)
            remaining = true_residual - group[f"pred_{target}"].to_numpy(dtype=float)
            freqs, true_energy = _one_sided_fft_energy(true_residual, sample_rate)
            _, remaining_energy = _one_sided_fft_energy(remaining, sample_rate)
            if len(freqs) == 0:
                continue
            true_total = float(np.sum(true_energy))
            remaining_total = float(np.sum(remaining_energy))
            target_total_energy[target] += true_total
            target_remaining_total_energy[target] += remaining_total
            for band in _bands_for_group(
                freqs,
                f0_hz=f0,
                flap_half_width_hz=flap_half_width_hz,
                high_band=high_band,
            ):
                key = (target, band.name)
                if key not in accum:
                    accum[key] = {
                        "true_energy": 0.0,
                        "remaining_energy": 0.0,
                        "sample_count": 0,
                        "group_count": 0,
                        "low_hz_sum": 0.0,
                        "high_hz_sum": 0.0,
                    }
                entry = accum[key]
                entry["true_energy"] = float(entry["true_energy"]) + float(np.sum(true_energy[band.mask]))
                entry["remaining_energy"] = float(entry["remaining_energy"]) + float(np.sum(remaining_energy[band.mask]))
                entry["sample_count"] = int(entry["sample_count"]) + int(len(group))
                entry["group_count"] = int(entry["group_count"]) + 1
                entry["low_hz_sum"] = float(entry["low_hz_sum"]) + float(band.low_hz)
                entry["high_hz_sum"] = float(entry["high_hz_sum"]) + float(band.high_hz)

    rows: list[dict[str, float | int | str]] = []
    for (target, component), entry in sorted(accum.items()):
        true_energy_value = float(entry["true_energy"])
        remaining_energy_value = float(entry["remaining_energy"])
        total = target_total_energy[target]
        remaining_total = target_remaining_total_energy[target]
        group_count = int(entry["group_count"])
        rows.append(
            {
                "target": target,
                "component": component,
                "sample_count": int(entry["sample_count"]),
                "group_count": group_count,
                "mean_low_hz": float(entry["low_hz_sum"]) / group_count if group_count else float("nan"),
                "mean_high_hz": float(entry["high_hz_sum"]) / group_count if group_count else float("nan"),
                "true_energy": true_energy_value,
                "remaining_energy": remaining_energy_value,
                "true_energy_fraction": float(true_energy_value / total) if total > 0.0 else float("nan"),
                "remaining_energy_fraction_of_true_total": (
                    float(remaining_energy_value / total) if total > 0.0 else float("nan")
                ),
                "remaining_energy_fraction": (
                    float(remaining_energy_value / remaining_total) if remaining_total > 0.0 else float("nan")
                ),
                "remaining_energy_fraction_of_true": (
                    float(remaining_energy_value / true_energy_value) if true_energy_value > 0.0 else float("nan")
                ),
                "energy_reduction_fraction": (
                    float(1.0 - remaining_energy_value / true_energy_value) if true_energy_value > 0.0 else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def frequency_residual_summary_table(energy_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize dominant true residual bands for each target."""

    rows: list[dict[str, Any]] = []
    for target, group in energy_table.groupby("target", observed=True):
        ordered = group.sort_values("true_energy_fraction", ascending=False)
        dominant = ordered.iloc[0]
        rows.append(
            {
                "target": target,
                "dominant_component": str(dominant["component"]),
                "dominant_true_energy_fraction": float(dominant["true_energy_fraction"]),
                "dominant_remaining_energy_fraction_of_true": float(dominant["remaining_energy_fraction_of_true"]),
                "dominant_energy_reduction_fraction": float(dominant["energy_reduction_fraction"]),
                "structured_true_energy_fraction": float(
                    group.loc[
                        group["component"].isin(["low_0_1hz", "mid_1_3hz", "flap_main", "harmonic_2f", "harmonic_3f"]),
                        "true_energy_fraction",
                    ].sum()
                ),
                "broadband_high_true_energy_fraction": float(
                    group.loc[
                        group["component"].astype(str).str.startswith("broadband_high"),
                        "true_energy_fraction",
                    ].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_frequency_energy(energy_table: pd.DataFrame, output_stem: Path, *, key_targets: tuple[str, ...]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label_map = {
        "low_0_1hz": "0-1",
        "mid_1_3hz": "1-3",
        "flap_main": "1f",
        "harmonic_2f": "2f",
        "harmonic_3f": "3f",
        "broadband_high_8_25hz_excl_structured": "8-25 excl.",
    }
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
        }
    )
    fig, axes = plt.subplots(len(key_targets), 1, figsize=(7.2, 1.9 * len(key_targets)), sharex=True)
    if len(key_targets) == 1:
        axes = [axes]
    present_components = set(energy_table["component"].astype(str))
    components = [component for component in COMPONENT_ORDER if component in present_components]
    x = np.arange(len(components))
    width = 0.38
    for ax, target in zip(axes, key_targets):
        subset = energy_table.loc[energy_table["target"] == target].set_index("component").reindex(components)
        ax.bar(
            x - width / 2.0,
            subset["true_energy_fraction"].to_numpy(dtype=float),
            width=width,
            color="#0072B2",
            label="DeLaurier residual",
        )
        ax.bar(
            x + width / 2.0,
            subset["remaining_energy_fraction_of_true_total"].to_numpy(dtype=float),
            width=width,
            color="#D55E00",
            label="remaining after NN",
        )
        ax.set_ylabel(f"{target}\nenergy fraction")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([label_map.get(component, component) for component in components], rotation=30, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def run_frequency_analysis(
    aligned_parquet: Path,
    output_dir: Path,
    *,
    flap_half_width_hz: float = 0.75,
    high_band: tuple[float, float] = DEFAULT_HIGH_BAND,
) -> dict[str, str]:
    frame = pd.read_parquet(aligned_parquet)
    energy = frequency_residual_energy_table(
        frame,
        targets=TARGET_COLUMNS,
        flap_half_width_hz=flap_half_width_hz,
        high_band=high_band,
    )
    summary = frequency_residual_summary_table(energy)

    output_dir.mkdir(parents=True, exist_ok=True)
    energy_path = output_dir / "frequency_residual_energy.csv"
    summary_path = output_dir / "frequency_residual_summary.csv"
    config_path = output_dir / "frequency_residual_config.json"
    plot_stem = output_dir / "frequency_residual_energy_key_targets"

    energy.to_csv(energy_path, index=False)
    summary.to_csv(summary_path, index=False)
    _plot_frequency_energy(energy, plot_stem, key_targets=KEY_PLOT_TARGETS)
    config = {
        "aligned_parquet": str(aligned_parquet),
        "output_dir": str(output_dir),
        "targets": list(TARGET_COLUMNS),
        "flap_half_width_hz": float(flap_half_width_hz),
        "high_band": list(high_band),
        "sample_count": int(len(frame)),
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "energy": str(energy_path),
        "summary": str(summary_path),
        "config": str(config_path),
        "plot_png": str(plot_stem.with_suffix(".png")),
        "plot_pdf": str(plot_stem.with_suffix(".pdf")),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-parquet", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--flap-half-width-hz", type=float, default=0.75)
    parser.add_argument("--high-band-low-hz", type=float, default=DEFAULT_HIGH_BAND[0])
    parser.add_argument("--high-band-high-hz", type=float, default=DEFAULT_HIGH_BAND[1])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = run_frequency_analysis(
        args.aligned_parquet,
        args.output_dir,
        flap_half_width_hz=args.flap_half_width_hz,
        high_band=(args.high_band_low_hz, args.high_band_high_hz),
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
