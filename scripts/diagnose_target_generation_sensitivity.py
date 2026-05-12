#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.metadata import load_aircraft_metadata
from system_identification.pipeline import _compute_effective_wrench_labels, compute_smoothed_kinematic_derivatives


SMOOTH_LINEAR_COLUMNS = [
    "vehicle_local_position.ax_smooth",
    "vehicle_local_position.ay_smooth",
    "vehicle_local_position.az_smooth",
]
RAW_LINEAR_COLUMNS = [
    "vehicle_local_position.ax",
    "vehicle_local_position.ay",
    "vehicle_local_position.az",
]


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = _finite_mask(a, b)
    if np.sum(mask) < 2:
        return float("nan")
    x = a[mask]
    y = b[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    mask = _finite_mask(a, b)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(a[mask] - b[mask]))))


def _nominal_sample_rate_hz(frame: pd.DataFrame) -> float:
    dt = np.diff(frame.sort_values("time_s")["time_s"].to_numpy(dtype=float))
    valid = dt[np.isfinite(dt) & (dt > 0.0)]
    if len(valid) == 0:
        return float("nan")
    return float(1.0 / np.median(valid))


def _fft_highpass_energy(values: np.ndarray, *, sample_rate_hz: float, cutoff_hz: float) -> float:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if np.sum(finite) < 8 or not np.isfinite(sample_rate_hz):
        return float("nan")
    filled = values.copy()
    if not finite.all():
        idx = np.flatnonzero(finite)
        filled[~finite] = np.interp(np.flatnonzero(~finite), idx, values[idx])
    centered = filled - np.mean(filled)
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate_hz)
    spectrum = np.fft.rfft(centered)
    total = float(np.sum(np.square(np.abs(spectrum))))
    high = float(np.sum(np.square(np.abs(spectrum[freqs >= cutoff_hz]))))
    return high / total if total > 0.0 else float("nan")


def rewrite_force_labels_for_variant(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    variant_name: str,
    window_s: float | None,
    polyorder: int = 2,
) -> pd.DataFrame:
    work = frame.copy()
    if window_s is None:
        linear_columns = RAW_LINEAR_COLUMNS
    else:
        derivatives = compute_smoothed_kinematic_derivatives(work, window_s=window_s, polyorder=polyorder)
        for column in derivatives.columns:
            work[column] = derivatives[column].to_numpy()
        linear_columns = SMOOTH_LINEAR_COLUMNS

    force_b, _, label_valid = _compute_effective_wrench_labels(
        work,
        metadata,
        linear_acceleration_columns=linear_columns,
        angular_acceleration_columns=None,
    )
    return pd.DataFrame(
        {
            "log_id": work["log_id"].astype(str).to_numpy() if "log_id" in work.columns else "__all__",
            "time_s": work["time_s"].to_numpy(dtype=float),
            f"fy_b_{variant_name}": force_b[:, 1],
            f"label_valid_{variant_name}": label_valid,
        },
        index=frame.index,
    )


def _top_overlap(reference: np.ndarray, variant: np.ndarray, quantile: float) -> float:
    mask = _finite_mask(reference, variant)
    if not np.any(mask):
        return float("nan")
    ref_abs = np.abs(reference[mask])
    var_abs = np.abs(variant[mask])
    ref_threshold = float(np.quantile(ref_abs, quantile))
    var_threshold = float(np.quantile(var_abs, quantile))
    ref_top = ref_abs >= ref_threshold
    var_top = var_abs >= var_threshold
    denom = int(np.sum(ref_top))
    if denom == 0:
        return float("nan")
    return float(np.sum(ref_top & var_top) / denom)


def compute_variant_metrics(
    reference_frame: pd.DataFrame,
    variant_frame: pd.DataFrame,
    *,
    reference_column: str,
    variant_column: str,
    split_name: str,
) -> dict[str, Any]:
    reference = reference_frame[reference_column].to_numpy(dtype=float)
    variant = variant_frame[variant_column].to_numpy(dtype=float)
    diff = variant - reference
    sample_rate_hz = _nominal_sample_rate_hz(reference_frame)
    return {
        "split": split_name,
        "variant": variant_column.removeprefix("fy_b_"),
        "sample_count": int(np.sum(_finite_mask(reference, variant))),
        "raw_std": float(np.nanstd(reference)),
        "variant_std": float(np.nanstd(variant)),
        "diff_std": float(np.nanstd(diff)),
        "corr_with_raw": _corr(reference, variant),
        "rmse_vs_raw": _rmse(reference, variant),
        "max_abs_raw": float(np.nanmax(np.abs(reference))),
        "max_abs_variant": float(np.nanmax(np.abs(variant))),
        "p99_abs_raw": float(np.nanquantile(np.abs(reference), 0.99)),
        "p99_abs_variant": float(np.nanquantile(np.abs(variant), 0.99)),
        "top1pct_overlap_fraction": _top_overlap(reference, variant, 0.99),
        "top5pct_overlap_fraction": _top_overlap(reference, variant, 0.95),
        "highpass_energy_frac_raw_8hz": _fft_highpass_energy(reference, sample_rate_hz=sample_rate_hz, cutoff_hz=8.0),
        "highpass_energy_frac_variant_8hz": _fft_highpass_energy(variant, sample_rate_hz=sample_rate_hz, cutoff_hz=8.0),
    }


def _variant_specs() -> list[tuple[str, float | None, int]]:
    return [
        ("raw_recomputed", None, 2),
        ("sg_0p06_p2", 0.06, 2),
        ("sg_0p12_p2", 0.12, 2),
        ("sg_0p20_p2", 0.20, 2),
        ("sg_0p30_p2", 0.30, 2),
    ]


def run_sensitivity(
    *,
    split_root: Path,
    metadata_path: Path,
    output_dir: Path,
    splits: tuple[str, ...] = ("val", "test"),
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_aircraft_metadata(metadata_path)
    metrics_rows: list[dict[str, Any]] = []
    paths: dict[str, Path] = {}
    for split in splits:
        frame = pd.read_parquet(split_root / f"{split}_samples.parquet")
        variant_columns = []
        variants = []
        for variant_name, window_s, polyorder in _variant_specs():
            variant = rewrite_force_labels_for_variant(
                frame,
                metadata,
                variant_name=variant_name,
                window_s=window_s,
                polyorder=polyorder,
            )
            variant_column = f"fy_b_{variant_name}"
            variant_columns.append(variant_column)
            variants.append(variant.loc[:, [variant_column]])
            metrics_rows.append(
                compute_variant_metrics(
                    frame,
                    variant,
                    reference_column="fy_b",
                    variant_column=variant_column,
                    split_name=split,
                )
            )
        combined = pd.concat([frame.loc[:, ["log_id", "time_s", "fy_b"]], *variants], axis=1)
        variant_path = output_dir / f"{split}_fy_b_target_variants.parquet"
        combined.to_parquet(variant_path, index=False)
        paths[f"{split}_variants"] = variant_path

    metrics = pd.DataFrame(metrics_rows)
    metrics_path = output_dir / "fy_b_target_generation_sensitivity.csv"
    metrics.to_csv(metrics_path, index=False)
    paths["metrics"] = metrics_path
    report_path = output_dir / "fy_b_target_generation_sensitivity.md"
    report_path.write_text(_markdown_report(metrics, split_root=split_root, metadata_path=metadata_path))
    paths["report"] = report_path
    return paths


def _markdown_report(metrics: pd.DataFrame, *, split_root: Path, metadata_path: Path) -> str:
    lines = [
        "# fy_b Target Generation Sensitivity",
        "",
        f"- split_root: `{split_root}`",
        f"- metadata: `{metadata_path}`",
        "",
        "## Metrics",
        "",
        metrics.to_csv(index=False),
        "",
    ]
    return "\n".join(lines)


def _parse_splits(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("splits must not be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose sensitivity of fy_b target to derivative smoothing settings.")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", type=_parse_splits, default=("val", "test"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_sensitivity(
        split_root=args.split_root,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        splits=args.splits,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
