#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnose_fyb_learnability import nominal_sample_rate_hz  # noqa: E402


DEFAULT_TARGETS = ("fy_b", "mx_b", "mz_b")
DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_HIGH_BAND = (8.0, 25.0)


@dataclass(frozen=True)
class BandMask:
    name: str
    low_hz: float
    high_hz: float
    mask: np.ndarray


def _group_key_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in ("log_id", "segment_id") if column in frame.columns]


def _grouped_frames(frame: pd.DataFrame) -> dict[tuple[Any, ...], pd.DataFrame]:
    group_columns = _group_key_columns(frame)
    if not group_columns:
        return {("__all__",): frame.sort_values("time_s").reset_index(drop=True)}
    grouped = {}
    for key, group in frame.groupby(group_columns, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        grouped[key] = group.sort_values("time_s").reset_index(drop=True)
    return grouped


def _resolve_f0_hz(group: pd.DataFrame) -> float:
    if "cycle_flap_frequency_hz" in group.columns:
        f0 = float(np.nanmedian(group["cycle_flap_frequency_hz"].to_numpy(dtype=float)))
    elif "flap_frequency_hz" in group.columns:
        f0 = float(np.nanmedian(group["flap_frequency_hz"].to_numpy(dtype=float)))
    else:
        f0 = float("nan")
    return f0 if np.isfinite(f0) and f0 > 0.0 else 4.5


def _one_sided_fft_energy(values: np.ndarray, sample_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if len(values) < 8 or not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        return np.array([], dtype=float), np.array([], dtype=float)

    finite = np.isfinite(values)
    if np.sum(finite) < 8:
        return np.array([], dtype=float), np.array([], dtype=float)
    filled = values.copy()
    if not finite.all():
        valid_idx = np.flatnonzero(finite)
        filled[~finite] = np.interp(np.flatnonzero(~finite), valid_idx, values[valid_idx])

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


def _interval_mask(freqs: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    return (freqs >= low_hz) & (freqs <= high_hz)


def _band_masks(
    freqs: np.ndarray,
    *,
    f0_hz: float,
    flap_band_half_width_hz: float,
    high_band: tuple[float, float],
) -> list[BandMask]:
    flap_low = max(0.0, f0_hz - flap_band_half_width_hz)
    flap_high = f0_hz + flap_band_half_width_hz
    harmonic_2f_low = max(0.0, 2.0 * f0_hz - flap_band_half_width_hz)
    harmonic_2f_high = 2.0 * f0_hz + flap_band_half_width_hz
    harmonic_3f_low = max(0.0, 3.0 * f0_hz - flap_band_half_width_hz)
    harmonic_3f_high = 3.0 * f0_hz + flap_band_half_width_hz

    flap_mask = _interval_mask(freqs, flap_low, flap_high)
    harmonic_2f_mask = _interval_mask(freqs, harmonic_2f_low, harmonic_2f_high)
    harmonic_3f_mask = _interval_mask(freqs, harmonic_3f_low, harmonic_3f_high)
    high_mask = _interval_mask(freqs, high_band[0], high_band[1]) & ~(flap_mask | harmonic_2f_mask | harmonic_3f_mask)

    return [
        BandMask("low_0_1hz", 0.0, 1.0, _interval_mask(freqs, 0.0, 1.0)),
        BandMask("mid_1_3hz", 1.0, 3.0, _interval_mask(freqs, 1.0, 3.0)),
        BandMask("flap_main", flap_low, flap_high, flap_mask),
        BandMask("harmonic_2f", harmonic_2f_low, harmonic_2f_high, harmonic_2f_mask),
        BandMask("harmonic_3f", harmonic_3f_low, harmonic_3f_high, harmonic_3f_mask),
        BandMask(
            f"broadband_high_{high_band[0]:.0f}_{high_band[1]:.0f}hz_excl_structured",
            float(high_band[0]),
            float(high_band[1]),
            high_mask,
        ),
    ]


def _aligned_group_pairs(raw_frame: pd.DataFrame, smooth_frame: pd.DataFrame) -> list[tuple[tuple[Any, ...], pd.DataFrame, pd.DataFrame]]:
    raw_groups = _grouped_frames(raw_frame)
    smooth_groups = _grouped_frames(smooth_frame)
    missing = sorted(set(raw_groups) - set(smooth_groups))
    extra = sorted(set(smooth_groups) - set(raw_groups))
    if missing or extra:
        raise ValueError(f"Raw and smoothed frames have different groups; missing={missing[:3]}, extra={extra[:3]}")

    pairs = []
    for key, raw_group in raw_groups.items():
        smooth_group = smooth_groups[key]
        if len(raw_group) != len(smooth_group):
            raise ValueError(f"Group {key} has different row counts: raw={len(raw_group)}, smooth={len(smooth_group)}")
        raw_time = raw_group["time_s"].to_numpy(dtype=float)
        smooth_time = smooth_group["time_s"].to_numpy(dtype=float)
        if not np.allclose(raw_time, smooth_time, equal_nan=True):
            raise ValueError(f"Group {key} has non-aligned time_s values")
        pairs.append((key, raw_group, smooth_group))
    return pairs


def _edge_trim_for_variant(edge_trim_s: float | Mapping[str, float], variant: str) -> float:
    if isinstance(edge_trim_s, Mapping):
        return float(edge_trim_s.get(variant, 0.0))
    return float(edge_trim_s)


def _valid_trim_mask(time_s: np.ndarray, raw_values: np.ndarray, smooth_values: np.ndarray, edge_trim_s: float) -> np.ndarray:
    mask = np.isfinite(time_s) & np.isfinite(raw_values) & np.isfinite(smooth_values)
    if edge_trim_s > 0.0 and np.sum(mask) > 0:
        valid_time = time_s[mask]
        start = float(np.min(valid_time)) + edge_trim_s
        stop = float(np.max(valid_time)) - edge_trim_s
        if stop > start:
            mask &= (time_s >= start) & (time_s <= stop)
    return mask


def compute_retained_energy_summary(
    raw_frame: pd.DataFrame,
    smooth_frames: Mapping[str, pd.DataFrame],
    *,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    flap_band_half_width_hz: float = 0.75,
    high_band: tuple[float, float] = DEFAULT_HIGH_BAND,
    edge_trim_s: float | Mapping[str, float] = 0.0,
    min_samples: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for variant, smooth_frame in smooth_frames.items():
        trim_s = _edge_trim_for_variant(edge_trim_s, variant)
        for key, raw_group, smooth_group in _aligned_group_pairs(raw_frame, smooth_frame):
            time_s = raw_group["time_s"].to_numpy(dtype=float)
            sample_rate_hz = nominal_sample_rate_hz(time_s)
            f0_hz = _resolve_f0_hz(raw_group)
            for target in targets:
                raw_values = raw_group[target].to_numpy(dtype=float)
                smooth_values = smooth_group[target].to_numpy(dtype=float)
                mask = _valid_trim_mask(time_s, raw_values, smooth_values, trim_s)
                if np.sum(mask) < min_samples:
                    continue

                freqs, raw_energy = _one_sided_fft_energy(raw_values[mask], sample_rate_hz)
                smooth_freqs, smooth_energy = _one_sided_fft_energy(smooth_values[mask], sample_rate_hz)
                if len(freqs) == 0 or len(smooth_freqs) == 0:
                    continue
                if len(freqs) != len(smooth_freqs) or not np.allclose(freqs, smooth_freqs):
                    smooth_energy = np.interp(freqs, smooth_freqs, smooth_energy)

                raw_total_energy = float(np.sum(raw_energy))
                smooth_total_energy = float(np.sum(smooth_energy))
                for band in _band_masks(
                    freqs,
                    f0_hz=f0_hz,
                    flap_band_half_width_hz=flap_band_half_width_hz,
                    high_band=high_band,
                ):
                    raw_band_energy = float(np.sum(raw_energy[band.mask]))
                    smooth_band_energy = float(np.sum(smooth_energy[band.mask]))
                    rows.append(
                        {
                            "variant": variant,
                            "group_key": "|".join(str(part) for part in key),
                            "target": target,
                            "band": band.name,
                            "low_hz": float(band.low_hz),
                            "high_hz": float(band.high_hz),
                            "f0_hz": float(f0_hz),
                            "sample_rate_hz": float(sample_rate_hz),
                            "sample_count": int(np.sum(mask)),
                            "edge_trim_s": float(trim_s),
                            "raw_band_energy": raw_band_energy,
                            "smooth_band_energy": smooth_band_energy,
                            "raw_total_energy": raw_total_energy,
                            "smooth_total_energy": smooth_total_energy,
                        }
                    )

    segment_table = pd.DataFrame(rows)
    if segment_table.empty:
        return segment_table, pd.DataFrame()
    return segment_table, _summarize_segments(segment_table)


def _weighted_sum(frame: pd.DataFrame, column: str) -> float:
    return float(np.sum(frame[column].to_numpy(dtype=float) * frame["sample_count"].to_numpy(dtype=float)))


def _summarize_segments(segment_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, target, band), group in segment_table.groupby(["variant", "target", "band"], sort=False):
        raw_band = _weighted_sum(group, "raw_band_energy")
        smooth_band = _weighted_sum(group, "smooth_band_energy")
        raw_total = _weighted_sum(group, "raw_total_energy")
        smooth_total = _weighted_sum(group, "smooth_total_energy")
        retained = smooth_band / raw_band if raw_band > 0.0 else float("nan")
        rows.append(
            {
                "variant": variant,
                "target": target,
                "band": band,
                "segment_count": int(len(group)),
                "sample_count": int(group["sample_count"].sum()),
                "low_hz_median": float(group["low_hz"].median()),
                "high_hz_median": float(group["high_hz"].median()),
                "f0_hz_median": float(group["f0_hz"].median()),
                "raw_band_energy": raw_band / float(group["sample_count"].sum()),
                "smooth_band_energy": smooth_band / float(group["sample_count"].sum()),
                "raw_total_energy": raw_total / float(group["sample_count"].sum()),
                "smooth_total_energy": smooth_total / float(group["sample_count"].sum()),
                "raw_energy_fraction": raw_band / raw_total if raw_total > 0.0 else float("nan"),
                "smooth_energy_fraction": smooth_band / smooth_total if smooth_total > 0.0 else float("nan"),
                "retained_energy_ratio": retained,
                "removed_energy_ratio": 1.0 - retained if np.isfinite(retained) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _parse_variant_specs(specs: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for spec in specs:
        if "=" in spec:
            label, path_text = spec.split("=", 1)
            if not label:
                raise argparse.ArgumentTypeError(f"Variant label must not be empty in {spec!r}")
            parsed[label] = Path(path_text)
        else:
            path = Path(spec)
            parsed[path.name] = path
    return parsed


def _manifest_edge_trim_s(split_root: Path) -> float:
    manifest_path = split_root / "dataset_manifest.json"
    if not manifest_path.exists():
        return 0.0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    window_s = manifest.get("smoothing", {}).get("window_s", 0.0)
    try:
        return 0.5 * float(window_s)
    except (TypeError, ValueError):
        return 0.0


def _plot_retained_ratios(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    targets = list(dict.fromkeys(summary["target"].tolist()))
    variants = list(dict.fromkeys(summary["variant"].tolist()))
    bands = list(dict.fromkeys(summary["band"].tolist()))
    fig, axes = plt.subplots(len(targets), 1, figsize=(11, 3.2 * len(targets)), sharex=True)
    if len(targets) == 1:
        axes = [axes]
    x = np.arange(len(bands))
    width = min(0.22, 0.8 / max(1, len(variants)))
    for ax, target in zip(axes, targets):
        for variant_idx, variant in enumerate(variants):
            values = []
            for band in bands:
                row = summary.loc[
                    (summary["target"] == target) & (summary["variant"] == variant) & (summary["band"] == band),
                    "retained_energy_ratio",
                ]
                values.append(float(row.iloc[0]) if len(row) else np.nan)
            ax.bar(x + (variant_idx - (len(variants) - 1) / 2.0) * width, values, width=width, label=variant)
        ax.axhline(1.0, color="#2f3437", linewidth=0.9, alpha=0.7)
        ax.set_ylabel(f"{target}\nretained ratio")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(bands, rotation=25, ha="right")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    output_path: Path,
    *,
    raw_split_root: Path,
    smooth_splits: Mapping[str, Path],
    summary: pd.DataFrame,
    segment_path: Path,
    summary_path: Path,
    plot_path: Path,
) -> None:
    focus_columns = [
        "split",
        "variant",
        "target",
        "band",
        "raw_energy_fraction",
        "smooth_energy_fraction",
        "retained_energy_ratio",
        "removed_energy_ratio",
        "segment_count",
    ]
    lines = [
        "# Lateral Target Retained Energy Analysis",
        "",
        f"- raw split root: `{raw_split_root}`",
        f"- smoothed split roots: `{', '.join(f'{label}={path}' for label, path in smooth_splits.items())}`",
        f"- segment CSV: `{segment_path.name}`",
        f"- summary CSV: `{summary_path.name}`",
        f"- retained-ratio plot: `{plot_path.name}`",
        "",
        "Energy is computed from de-meaned one-sided FFT components per `log_id/segment_id`, then aggregated with sample-count weighting.",
        "The broadband high-frequency row uses the configured high band and excludes the flap-main, 2f, and 3f windows.",
        "",
        "## Summary",
        "",
        "```csv",
        summary[focus_columns].to_csv(index=False),
        "```",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_retained_energy_analysis(
    *,
    raw_split_root: Path,
    smooth_splits: Mapping[str, Path],
    output_dir: Path,
    split_names: tuple[str, ...] = DEFAULT_SPLITS,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    flap_band_half_width_hz: float = 0.75,
    high_band: tuple[float, float] = DEFAULT_HIGH_BAND,
    use_manifest_edge_trim: bool = True,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_segments: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    edge_trims = {
        label: _manifest_edge_trim_s(path) if use_manifest_edge_trim else 0.0 for label, path in smooth_splits.items()
    }

    for split_name in split_names:
        raw_frame = pd.read_parquet(raw_split_root / f"{split_name}_samples.parquet")
        smooth_frames = {
            label: pd.read_parquet(split_root / f"{split_name}_samples.parquet") for label, split_root in smooth_splits.items()
        }
        segments, summary = compute_retained_energy_summary(
            raw_frame,
            smooth_frames,
            targets=targets,
            flap_band_half_width_hz=flap_band_half_width_hz,
            high_band=high_band,
            edge_trim_s=edge_trims,
        )
        if not segments.empty:
            segments.insert(0, "split", split_name)
            all_segments.append(segments)
        if not summary.empty:
            summary.insert(0, "split", split_name)
            all_summaries.append(summary)

    segment_table = pd.concat(all_segments, ignore_index=True) if all_segments else pd.DataFrame()
    summary_table = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()

    segment_path = output_dir / "lateral_retained_energy_segments.csv"
    summary_path = output_dir / "lateral_retained_energy_summary.csv"
    config_path = output_dir / "lateral_retained_energy_config.json"
    report_path = output_dir / "lateral_retained_energy_report.md"
    plot_path = output_dir / "lateral_retained_energy_ratio.png"

    segment_table.to_csv(segment_path, index=False)
    summary_table.to_csv(summary_path, index=False)
    if not summary_table.empty:
        _plot_retained_ratios(summary_table, plot_path)
    config_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "raw_split_root": str(raw_split_root),
                "smooth_splits": {label: str(path) for label, path in smooth_splits.items()},
                "split_names": list(split_names),
                "targets": list(targets),
                "flap_band_half_width_hz": float(flap_band_half_width_hz),
                "high_band": list(high_band),
                "demeaned_fft_energy": True,
                "use_manifest_edge_trim": bool(use_manifest_edge_trim),
                "edge_trim_s_by_variant": edge_trims,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_report(
        report_path,
        raw_split_root=raw_split_root,
        smooth_splits=smooth_splits,
        summary=summary_table,
        segment_path=segment_path,
        summary_path=summary_path,
        plot_path=plot_path,
    )
    return {
        "segments": segment_path,
        "summary": summary_path,
        "config": config_path,
        "report": report_path,
        "plot": plot_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze retained frequency-domain energy in smoothed lateral targets.")
    parser.add_argument("--raw-split-root", type=Path, required=True)
    parser.add_argument(
        "--smooth-splits",
        nargs="+",
        required=True,
        help="Smoothed split specs as label=path. If label is omitted, the directory name is used.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-names", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--flap-band-half-width-hz", type=float, default=0.75)
    parser.add_argument("--high-band-low-hz", type=float, default=DEFAULT_HIGH_BAND[0])
    parser.add_argument("--high-band-high-hz", type=float, default=DEFAULT_HIGH_BAND[1])
    parser.add_argument("--no-manifest-edge-trim", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_retained_energy_analysis(
        raw_split_root=args.raw_split_root,
        smooth_splits=_parse_variant_specs(args.smooth_splits),
        output_dir=args.output_dir,
        split_names=tuple(args.split_names),
        targets=tuple(args.targets),
        flap_band_half_width_hz=args.flap_band_half_width_hz,
        high_band=(float(args.high_band_low_hz), float(args.high_band_high_hz)),
        use_manifest_edge_trim=not args.no_manifest_edge_trim,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
