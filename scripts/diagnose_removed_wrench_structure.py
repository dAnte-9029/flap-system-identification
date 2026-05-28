#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMNS = ["fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b"]
BANDS = ("low_0_1hz", "mid_1_3hz", "flap_main", "harmonic_2f", "broadband_high_8_25hz")


def _phase_bins(phase: np.ndarray, n_bins: int) -> np.ndarray:
    wrapped = np.mod(phase, 2.0 * np.pi)
    return np.clip(np.floor(wrapped / (2.0 * np.pi) * n_bins).astype(int), 0, n_bins - 1)


def _rmse(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(finite))))


def phase_removed_summary(
    raw_frame: pd.DataFrame,
    clean_frame: pd.DataFrame,
    *,
    candidate: str,
    split: str,
    phase_column: str = "phase_corrected_rad",
    n_bins: int = 36,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if phase_column not in raw_frame.columns:
        phase_column = "phase_raw_rad"
    phase = raw_frame[phase_column].to_numpy(dtype=float)
    bin_index = _phase_bins(phase, n_bins)
    bin_width = 2.0 * np.pi / float(n_bins)
    bin_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for channel in TARGET_COLUMNS:
        removed = raw_frame[channel].to_numpy(dtype=float) - clean_frame[channel].to_numpy(dtype=float)
        medians = np.full(n_bins, np.nan, dtype=float)
        counts = np.zeros(n_bins, dtype=int)
        for idx in range(n_bins):
            selected = removed[bin_index == idx]
            finite = selected[np.isfinite(selected)]
            counts[idx] = len(finite)
            medians[idx] = float(np.median(finite)) if len(finite) else float("nan")
            bin_rows.append(
                {
                    "candidate": candidate,
                    "split": split,
                    "channel": channel,
                    "phase_bin": idx,
                    "phase_center_rad": float((idx + 0.5) * bin_width),
                    "sample_count": int(len(finite)),
                    "removed_median": medians[idx],
                    "removed_mad": float(np.median(np.abs(finite - medians[idx]))) if len(finite) else float("nan"),
                }
            )
        removed_rmse = _rmse(removed)
        phase_rmse = _rmse(medians)
        finite_medians = medians[np.isfinite(medians)]
        summary_rows.append(
            {
                "candidate": candidate,
                "split": split,
                "channel": channel,
                "sample_count": int(np.isfinite(removed).sum()),
                "removed_rmse": removed_rmse,
                "phase_median_rmse": phase_rmse,
                "phase_coherent_fraction": float(phase_rmse / removed_rmse) if removed_rmse > 0.0 else float("nan"),
                "phase_peak_to_peak": float(np.max(finite_medians) - np.min(finite_medians))
                if len(finite_medians)
                else float("nan"),
                "min_bin_count": int(np.min(counts)) if len(counts) else 0,
            }
        )
    return pd.DataFrame(bin_rows), pd.DataFrame(summary_rows)


def _sample_rate_hz(time_s: np.ndarray) -> float:
    finite = time_s[np.isfinite(time_s)]
    if len(finite) < 3:
        return float("nan")
    dt = np.diff(np.sort(finite))
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if len(dt) == 0:
        return float("nan")
    return float(1.0 / np.median(dt))


def _fft_energy(values: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    if finite.sum() < 8 or not np.isfinite(sample_rate_hz):
        return np.array([], dtype=float), np.array([], dtype=float)
    filled = values.astype(float, copy=True)
    if not finite.all():
        idx = np.flatnonzero(finite)
        filled[~finite] = np.interp(np.flatnonzero(~finite), idx, values[idx])
    centered = filled - float(np.mean(filled))
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    energy = np.square(np.abs(np.fft.rfft(centered)))
    return freqs, energy


def _band_masks(freqs: np.ndarray, f0_hz: float) -> dict[str, np.ndarray]:
    flap_main = (freqs >= max(0.0, f0_hz - 0.75)) & (freqs <= f0_hz + 0.75)
    harmonic_2f = (freqs >= max(0.0, 2.0 * f0_hz - 0.75)) & (freqs <= 2.0 * f0_hz + 0.75)
    broadband_high = (freqs >= 8.0) & (freqs <= 25.0) & ~(flap_main | harmonic_2f)
    return {
        "low_0_1hz": (freqs >= 0.0) & (freqs < 1.0),
        "mid_1_3hz": (freqs >= 1.0) & (freqs < 3.0),
        "flap_main": flap_main,
        "harmonic_2f": harmonic_2f,
        "broadband_high_8_25hz_excl_1f_2f": broadband_high,
    }


def frequency_removed_summary(
    raw_frame: pd.DataFrame,
    clean_frame: pd.DataFrame,
    *,
    candidate: str,
    split: str,
) -> pd.DataFrame:
    group_columns = [column for column in ("log_id", "segment_id") if column in raw_frame.columns]
    grouped = raw_frame.groupby(group_columns, sort=False, dropna=False) if group_columns else [(None, raw_frame)]
    accum: dict[tuple[str, str], dict[str, float | int]] = {}
    totals = {channel: 0.0 for channel in TARGET_COLUMNS}
    for _, raw_group in grouped:
        raw_group = raw_group.sort_values("time_s")
        clean_group = clean_frame.loc[raw_group.index]
        sample_rate = _sample_rate_hz(raw_group["time_s"].to_numpy(dtype=float))
        f0 = 4.5
        for freq_column in ("cycle_flap_frequency_hz", "flap_frequency_hz"):
            if freq_column in raw_group.columns:
                finite_freq = raw_group[freq_column].to_numpy(dtype=float)
                finite_freq = finite_freq[np.isfinite(finite_freq)]
                if len(finite_freq):
                    f0 = float(np.median(finite_freq))
                    break
        for channel in TARGET_COLUMNS:
            removed = raw_group[channel].to_numpy(dtype=float) - clean_group[channel].to_numpy(dtype=float)
            freqs, energy = _fft_energy(removed, sample_rate)
            if len(freqs) == 0:
                continue
            total = float(np.sum(energy))
            totals[channel] += total
            masks = _band_masks(freqs, f0)
            for band, mask in masks.items():
                key = (channel, band)
                if key not in accum:
                    accum[key] = {"energy": 0.0, "group_count": 0, "sample_count": 0}
                accum[key]["energy"] = float(accum[key]["energy"]) + float(np.sum(energy[mask]))
                accum[key]["group_count"] = int(accum[key]["group_count"]) + 1
                accum[key]["sample_count"] = int(accum[key]["sample_count"]) + len(raw_group)
    rows: list[dict[str, float | int | str]] = []
    for (channel, band), values in sorted(accum.items()):
        total = totals[channel]
        energy = float(values["energy"])
        rows.append(
            {
                "candidate": candidate,
                "split": split,
                "channel": channel,
                "band": band,
                "energy": energy,
                "energy_fraction": float(energy / total) if total > 0.0 else float("nan"),
                "group_count": int(values["group_count"]),
                "sample_count": int(values["sample_count"]),
            }
        )
    return pd.DataFrame(rows)


def run_removed_structure_diagnostics(
    *,
    raw_split_root: str | Path,
    candidate_roots: dict[str, str | Path],
    output_dir: str | Path,
    split: str = "test",
    phase_bins: int = 36,
) -> dict[str, str]:
    raw_split_root = Path(raw_split_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_frame = pd.read_parquet(raw_split_root / f"{split}_samples.parquet").reset_index(drop=True)

    phase_bins_rows: list[pd.DataFrame] = []
    phase_summary_rows: list[pd.DataFrame] = []
    frequency_rows: list[pd.DataFrame] = []
    for candidate, root in candidate_roots.items():
        clean_frame = pd.read_parquet(Path(root) / f"{split}_samples.parquet").reset_index(drop=True)
        if len(clean_frame) != len(raw_frame):
            raise ValueError(f"candidate {candidate} length mismatch: {len(clean_frame)} vs {len(raw_frame)}")
        phase_bins_df, phase_summary_df = phase_removed_summary(
            raw_frame,
            clean_frame,
            candidate=candidate,
            split=split,
            n_bins=phase_bins,
        )
        phase_bins_rows.append(phase_bins_df)
        phase_summary_rows.append(phase_summary_df)
        frequency_rows.append(frequency_removed_summary(raw_frame, clean_frame, candidate=candidate, split=split))

    phase_bins_table = pd.concat(phase_bins_rows, ignore_index=True)
    phase_summary_table = pd.concat(phase_summary_rows, ignore_index=True)
    frequency_table = pd.concat(frequency_rows, ignore_index=True)

    phase_bins_path = output_dir / "removed_phase_bins.csv"
    phase_summary_path = output_dir / "removed_phase_summary.csv"
    frequency_path = output_dir / "removed_frequency_summary.csv"
    phase_bins_table.to_csv(phase_bins_path, index=False)
    phase_summary_table.to_csv(phase_summary_path, index=False)
    frequency_table.to_csv(frequency_path, index=False)
    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "raw_split_root": str(raw_split_root),
                "candidate_roots": {key: str(value) for key, value in candidate_roots.items()},
                "split": split,
                "phase_bins": phase_bins,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "phase_bins": str(phase_bins_path),
        "phase_summary": str(phase_summary_path),
        "frequency_summary": str(frequency_path),
        "config": str(config_path),
    }


def parse_candidate(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("candidate must be NAME=PATH")
    name, path = raw.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("candidate must be NAME=PATH")
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose phase/frequency structure of raw-clean removed wrench content.")
    parser.add_argument("--raw-split-root", required=True, type=Path)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--phase-bins", type=int, default=36)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_removed_structure_diagnostics(
        raw_split_root=args.raw_split_root,
        candidate_roots=dict(args.candidate),
        output_dir=args.output_dir,
        split=args.split,
        phase_bins=args.phase_bins,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
