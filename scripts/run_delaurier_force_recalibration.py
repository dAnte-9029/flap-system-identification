#!/usr/bin/env python3
"""Fit and evaluate force-only affine wrappers for exported DeLaurier predictions.

This script does not re-export IsaacLab DeLaurier parameters. It fits train-only
force-channel corrections on top of already exported DeLaurier force predictions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_delaurier_residual_conditions import condition_bin_table, condition_summary_table
from scripts.analyze_delaurier_residual_frequency import (
    frequency_residual_energy_table,
    frequency_residual_summary_table,
)
from scripts.analyze_delaurier_residual_phase import phase_bin_table, phase_summary_table
from scripts.build_delaurier_residual_split import align_prior_to_samples

FORCE_COLUMNS = ("fx_b", "fy_b", "fz_b")
SPLITS = ("train", "val", "test")
DEFAULT_CHANNEL_WEIGHTS = {"fx_b": 2.0, "fy_b": 0.5, "fz_b": 2.0}
DEFAULT_OUTPUT_ROOT = Path("artifacts/20260525_delaurier_force_recalibration_v1")
DEFAULT_SPLIT_ROOT = Path("dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1")
DEFAULT_PRIOR_ROOT = Path("artifacts/delaurier_physical_prior_v1")
METADATA_COLUMNS = (
    "time_s",
    "log_id",
    "segment_id",
    "phase_corrected_rad",
    "cycle_flap_frequency_hz",
    "flap_frequency_hz",
    "airspeed_validated.true_airspeed_m_s",
    "vehicle_air_data.rho",
    "airspeed_validated.pitch_filtered",
)


@dataclass(frozen=True)
class ForceAffineModel:
    """Affine force correction: corrected_force = gain_matrix @ prior_force + bias."""

    name: str
    gain_matrix: np.ndarray
    bias: np.ndarray
    description: str


def _check_force_columns(frame: pd.DataFrame, *, label: str) -> None:
    missing = [column for column in FORCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing force columns: {missing}")


def _finite_channel_pair(prior: pd.DataFrame, true: pd.DataFrame, channel: str) -> tuple[np.ndarray, np.ndarray]:
    x = prior[channel].to_numpy(dtype=float)
    y = true[channel].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        raise ValueError(f"not enough finite samples to fit {channel}")
    return x[mask], y[mask]


def fit_per_channel_affine(prior: pd.DataFrame, true: pd.DataFrame) -> ForceAffineModel:
    """Fit independent gain and bias parameters for each force channel."""

    _check_force_columns(prior, label="prior")
    _check_force_columns(true, label="true")
    gains = np.zeros((len(FORCE_COLUMNS), len(FORCE_COLUMNS)), dtype=float)
    bias = np.zeros(len(FORCE_COLUMNS), dtype=float)
    for channel_index, channel in enumerate(FORCE_COLUMNS):
        x, y = _finite_channel_pair(prior, true, channel)
        design = np.column_stack([x, np.ones_like(x)])
        gain, channel_bias = np.linalg.lstsq(design, y, rcond=None)[0]
        gains[channel_index, channel_index] = float(gain)
        bias[channel_index] = float(channel_bias)
    return ForceAffineModel(
        name="A1_per_channel_affine",
        gain_matrix=gains,
        bias=bias,
        description="Train-only independent affine correction for fx_b/fy_b/fz_b exported DeLaurier predictions.",
    )


def fit_shared_gain_channel_bias(
    prior: pd.DataFrame,
    true: pd.DataFrame,
    *,
    channel_weights: Mapping[str, float],
) -> ForceAffineModel:
    """Fit one shared gain and one bias per force channel with channel-weighted least squares."""

    _check_force_columns(prior, label="prior")
    _check_force_columns(true, label="true")
    rows: list[list[float]] = []
    targets: list[float] = []
    for channel_index, channel in enumerate(FORCE_COLUMNS):
        x, y = _finite_channel_pair(prior, true, channel)
        weight = float(channel_weights.get(channel, 1.0))
        if weight <= 0.0 or not np.isfinite(weight):
            raise ValueError(f"channel weight for {channel} must be positive and finite")
        scale = float(np.sqrt(weight))
        for x_value, y_value in zip(x, y):
            row = [scale * float(x_value), 0.0, 0.0, 0.0]
            row[1 + channel_index] = scale
            rows.append(row)
            targets.append(scale * float(y_value))
    params = np.linalg.lstsq(np.asarray(rows, dtype=float), np.asarray(targets, dtype=float), rcond=None)[0]
    gain = float(params[0])
    gain_matrix = np.eye(len(FORCE_COLUMNS), dtype=float) * gain
    bias = np.asarray(params[1:], dtype=float)
    return ForceAffineModel(
        name="A2_weighted_shared_gain_bias",
        gain_matrix=gain_matrix,
        bias=bias,
        description=(
            "Train-only shared force gain plus per-channel bias; objective weights "
            f"{dict(channel_weights)}."
        ),
    )


def identity_force_model() -> ForceAffineModel:
    """Return the unchanged exported DeLaurier prior as an affine model."""

    return ForceAffineModel(
        name="A0_current_delaurier",
        gain_matrix=np.eye(len(FORCE_COLUMNS), dtype=float),
        bias=np.zeros(len(FORCE_COLUMNS), dtype=float),
        description="Unchanged exported calibrated DeLaurier force prediction.",
    )


def apply_force_model(prior: pd.DataFrame, model: ForceAffineModel) -> pd.DataFrame:
    """Apply an affine wrapper to prior force predictions."""

    _check_force_columns(prior, label="prior")
    gain_matrix = np.asarray(model.gain_matrix, dtype=float)
    bias = np.asarray(model.bias, dtype=float)
    if gain_matrix.shape != (len(FORCE_COLUMNS), len(FORCE_COLUMNS)):
        raise ValueError(
            f"gain_matrix must have shape {(len(FORCE_COLUMNS), len(FORCE_COLUMNS))}, got {gain_matrix.shape}"
        )
    if bias.shape != (len(FORCE_COLUMNS),):
        raise ValueError(f"bias must have shape {(len(FORCE_COLUMNS),)}, got {bias.shape}")
    values = prior.loc[:, FORCE_COLUMNS].to_numpy(dtype=float)
    corrected = values @ gain_matrix.T + bias
    return pd.DataFrame(corrected, columns=FORCE_COLUMNS, index=prior.index)


def _channel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {"n": 0, "mae": np.nan, "mse": np.nan, "rmse": np.nan, "bias": np.nan, "r2": np.nan}
    residual = y_pred[mask] - y_true[mask]
    mse = float(np.mean(residual * residual))
    centered = y_true[mask] - float(np.mean(y_true[mask]))
    ss_tot = float(np.sum(centered * centered))
    ss_res = float(np.sum(residual * residual))
    return {
        "n": int(mask.sum()),
        "mae": float(np.mean(np.abs(residual))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "bias": float(np.mean(residual)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else np.nan,
    }


def force_metrics_table(
    true: pd.DataFrame,
    pred: pd.DataFrame,
    *,
    split: str,
    variant: str,
    channel_weights: Mapping[str, float],
) -> list[dict[str, float | int | str]]:
    """Return channel rows plus a force_mean summary row for one split/variant."""

    _check_force_columns(true, label="true")
    _check_force_columns(pred, label="pred")
    rows: list[dict[str, float | int | str]] = []
    weights = np.asarray([float(channel_weights.get(channel, 1.0)) for channel in FORCE_COLUMNS], dtype=float)
    channel_rmse = []
    channel_mse = []
    channel_r2 = []
    for channel in FORCE_COLUMNS:
        metrics = _channel_metrics(
            true[channel].to_numpy(dtype=float),
            pred[channel].to_numpy(dtype=float),
        )
        row = {"split": split, "variant": variant, "target": channel, **metrics}
        row["weighted_rmse"] = row["rmse"]
        row["weighted_mse"] = row["mse"]
        rows.append(row)
        channel_rmse.append(float(row["rmse"]))
        channel_mse.append(float(row["mse"]))
        channel_r2.append(float(row["r2"]))

    rows.append(
        {
            "split": split,
            "variant": variant,
            "target": "force_mean",
            "n": int(min(int(row["n"]) for row in rows)),
            "mae": float(np.nanmean([float(row["mae"]) for row in rows])),
            "mse": float(np.nanmean(channel_mse)),
            "rmse": float(np.nanmean(channel_rmse)),
            "bias": float(np.nanmean([float(row["bias"]) for row in rows])),
            "r2": float(np.nanmean(channel_r2)),
            "weighted_rmse": float(np.average(channel_rmse, weights=weights)),
            "weighted_mse": float(np.average(channel_mse, weights=weights)),
        }
    )
    return rows


def _load_split_frames(
    split_root: Path,
    prior_root: Path,
    split: str,
    *,
    allow_row_order_fallback: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    samples = pd.read_parquet(split_root / f"{split}_samples.parquet")
    raw_prior = pd.read_parquet(prior_root / f"{split}_predictions.parquet")
    prior, alignment_info = align_prior_to_samples(
        samples,
        raw_prior,
        allow_row_order_fallback=allow_row_order_fallback,
    )
    _check_force_columns(samples, label=f"{split} samples")
    _check_force_columns(prior, label=f"{split} prior")
    return samples, prior, alignment_info


def _aligned_frame(samples: pd.DataFrame, prior: pd.DataFrame, corrected: pd.DataFrame) -> pd.DataFrame:
    metadata = samples.loc[:, [column for column in METADATA_COLUMNS if column in samples.columns]].copy()
    if "vehicle_air_data.rho" in metadata.columns and "airspeed_validated.true_airspeed_m_s" in metadata.columns:
        tas = metadata["airspeed_validated.true_airspeed_m_s"].to_numpy(dtype=float)
        rho = metadata["vehicle_air_data.rho"].to_numpy(dtype=float)
        metadata["dynamic_pressure_pa"] = 0.5 * rho * tas * tas
    if "airspeed_validated.pitch_filtered" in metadata.columns:
        metadata["alpha_proxy_rad"] = metadata["airspeed_validated.pitch_filtered"].astype(float)

    aligned = metadata.reset_index(drop=True)
    for channel in FORCE_COLUMNS:
        label = samples[channel].to_numpy(dtype=float)
        prior_values = prior[channel].to_numpy(dtype=float)
        corrected_values = corrected[channel].to_numpy(dtype=float)
        aligned[f"label_{channel}"] = label
        aligned[f"true_{channel}"] = label
        aligned[f"prior_{channel}"] = prior_values
        aligned[f"pred_{channel}"] = corrected_values - prior_values
    return aligned


def _parameter_rows(models: list[ForceAffineModel]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for model in models:
        for output_index, output_channel in enumerate(FORCE_COLUMNS):
            for input_index, input_channel in enumerate(FORCE_COLUMNS):
                rows.append(
                    {
                        "variant": model.name,
                        "term": "gain",
                        "output_channel": output_channel,
                        "input_channel": input_channel,
                        "value": float(model.gain_matrix[output_index, input_index]),
                        "description": model.description,
                    }
                )
            rows.append(
                {
                    "variant": model.name,
                    "term": "bias",
                    "output_channel": output_channel,
                    "input_channel": "",
                    "value": float(model.bias[output_index]),
                    "description": model.description,
                }
            )
    return rows


def _improvement_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics.loc[metrics["variant"] == "A0_current_delaurier", ["split", "target", "rmse", "r2"]].rename(
        columns={"rmse": "baseline_rmse", "r2": "baseline_r2"}
    )
    merged = metrics.merge(baseline, on=["split", "target"], how="left")
    merged["rmse_delta_vs_A0"] = merged["rmse"] - merged["baseline_rmse"]
    merged["rmse_reduction_fraction_vs_A0"] = 1.0 - merged["rmse"] / merged["baseline_rmse"]
    merged["r2_delta_vs_A0"] = merged["r2"] - merged["baseline_r2"]
    return merged


def _condition_columns_for(frame: pd.DataFrame) -> tuple[str, ...]:
    candidates = (
        "airspeed_validated.true_airspeed_m_s",
        "dynamic_pressure_pa",
        "alpha_proxy_rad",
        "cycle_flap_frequency_hz",
    )
    return tuple(column for column in candidates if column in frame.columns)


def run_force_recalibration(
    *,
    split_root: Path,
    prior_root: Path,
    output_root: Path,
    channel_weights: Mapping[str, float],
    phase_bins: int = 36,
    condition_bins: int = 5,
    min_condition_samples: int = 500,
    skip_frequency: bool = False,
    allow_row_order_fallback: bool = False,
) -> dict[str, str]:
    """Fit train-only force wrappers and evaluate all splits."""

    output_root.mkdir(parents=True, exist_ok=True)
    train_samples, train_prior, train_alignment = _load_split_frames(
        split_root,
        prior_root,
        "train",
        allow_row_order_fallback=allow_row_order_fallback,
    )
    models = [
        identity_force_model(),
        fit_per_channel_affine(train_prior, train_samples),
        fit_shared_gain_channel_bias(train_prior, train_samples, channel_weights=channel_weights),
    ]

    metrics_rows: list[dict[str, float | int | str]] = []
    phase_tables: list[pd.DataFrame] = []
    phase_summaries: list[pd.DataFrame] = []
    condition_tables: list[pd.DataFrame] = []
    condition_summaries: list[pd.DataFrame] = []
    frequency_tables: list[pd.DataFrame] = []
    frequency_summaries: list[pd.DataFrame] = []
    row_counts: dict[str, int] = {}
    alignment: dict[str, object] = {"train": train_alignment}

    for split in SPLITS:
        if split == "train":
            samples, prior = train_samples, train_prior
        else:
            samples, prior, split_alignment = _load_split_frames(
                split_root,
                prior_root,
                split,
                allow_row_order_fallback=allow_row_order_fallback,
            )
            alignment[split] = split_alignment
        row_counts[split] = int(len(samples))
        for model in models:
            corrected = apply_force_model(prior, model)
            metrics_rows.extend(
                force_metrics_table(
                    samples,
                    corrected,
                    split=split,
                    variant=model.name,
                    channel_weights=channel_weights,
                )
            )
            aligned = _aligned_frame(samples, prior, corrected)
            aligned_path = output_root / f"{split}_{model.name}_aligned_force_predictions.parquet"
            aligned.to_parquet(aligned_path, index=False)

            phase = phase_bin_table(aligned, targets=FORCE_COLUMNS, phase_bins=phase_bins)
            phase.insert(0, "variant", model.name)
            phase.insert(0, "split", split)
            phase_tables.append(phase)
            phase_summary = phase_summary_table(aligned, phase, targets=FORCE_COLUMNS)
            phase_summary.insert(0, "variant", model.name)
            phase_summary.insert(0, "split", split)
            phase_summaries.append(phase_summary)

            condition_columns = _condition_columns_for(aligned)
            if condition_columns:
                conditions = condition_bin_table(
                    aligned,
                    condition_columns=condition_columns,
                    targets=FORCE_COLUMNS,
                    quantile_bins=condition_bins,
                    min_samples=min_condition_samples,
                )
                conditions.insert(0, "variant", model.name)
                conditions.insert(0, "split", split)
                condition_tables.append(conditions)
                condition_summary = condition_summary_table(conditions)
                condition_summary.insert(0, "variant", model.name)
                condition_summary.insert(0, "split", split)
                condition_summaries.append(condition_summary)

            if not skip_frequency:
                frequency = frequency_residual_energy_table(aligned, targets=FORCE_COLUMNS)
                frequency.insert(0, "variant", model.name)
                frequency.insert(0, "split", split)
                frequency_tables.append(frequency)
                frequency_summary = frequency_residual_summary_table(frequency)
                frequency_summary.insert(0, "variant", model.name)
                frequency_summary.insert(0, "split", split)
                frequency_summaries.append(frequency_summary)

    parameters = pd.DataFrame(_parameter_rows(models))
    metrics = pd.DataFrame(metrics_rows)
    summary = _improvement_summary(metrics)
    phase_all = pd.concat(phase_tables, ignore_index=True)
    phase_summary_all = pd.concat(phase_summaries, ignore_index=True)
    condition_all = pd.concat(condition_tables, ignore_index=True) if condition_tables else pd.DataFrame()
    condition_summary_all = pd.concat(condition_summaries, ignore_index=True) if condition_summaries else pd.DataFrame()
    frequency_all = pd.concat(frequency_tables, ignore_index=True) if frequency_tables else pd.DataFrame()
    frequency_summary_all = pd.concat(frequency_summaries, ignore_index=True) if frequency_summaries else pd.DataFrame()

    paths = {
        "parameters": output_root / "parameters.csv",
        "metrics_by_split": output_root / "metrics_by_split.csv",
        "force_metrics_summary": output_root / "force_metrics_summary.csv",
        "residual_by_phase": output_root / "residual_by_phase.csv",
        "residual_by_phase_summary": output_root / "residual_by_phase_summary.csv",
        "residual_by_condition": output_root / "residual_by_condition.csv",
        "residual_by_condition_summary": output_root / "residual_by_condition_summary.csv",
        "residual_frequency_energy": output_root / "residual_frequency_energy.csv",
        "residual_frequency_summary": output_root / "residual_frequency_summary.csv",
        "manifest": output_root / "manifest.json",
        "readme": output_root / "README.md",
    }
    parameters.to_csv(paths["parameters"], index=False)
    metrics.to_csv(paths["metrics_by_split"], index=False)
    summary.to_csv(paths["force_metrics_summary"], index=False)
    phase_all.to_csv(paths["residual_by_phase"], index=False)
    phase_summary_all.to_csv(paths["residual_by_phase_summary"], index=False)
    condition_all.to_csv(paths["residual_by_condition"], index=False)
    condition_summary_all.to_csv(paths["residual_by_condition_summary"], index=False)
    frequency_all.to_csv(paths["residual_frequency_energy"], index=False)
    frequency_summary_all.to_csv(paths["residual_frequency_summary"], index=False)

    manifest = {
        "split_root": str(split_root),
        "prior_root": str(prior_root),
        "output_root": str(output_root),
        "row_counts": row_counts,
        "force_columns": list(FORCE_COLUMNS),
        "channel_weights": dict(channel_weights),
        "variants": {model.name: model.description for model in models},
        "fit_split": "train",
        "stage": "Stage 1: Force-Only DeLaurier Recalibration",
        "method_note": (
            "A1/A2 are affine wrappers fit to exported DeLaurier force predictions. "
            "They are not IsaacLab internal DeLaurier aerodynamic parameter recalibrations."
        ),
        "skip_frequency": bool(skip_frequency),
        "alignment": alignment,
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_readme(paths["readme"], manifest, summary)
    return {key: str(path) for key, path in paths.items()}


def _write_readme(path: Path, manifest: dict[str, object], summary: pd.DataFrame) -> None:
    force_mean = summary.loc[summary["target"] == "force_mean"].copy()
    force_mean = force_mean.sort_values(["split", "variant"])
    lines = [
        "# DeLaurier Force-Only Recalibration v1",
        "",
        "This artifact evaluates train-only affine wrappers around exported DeLaurier force predictions.",
        "",
        "Important scope note: this is not a true IsaacLab DeLaurier internal parameter re-export.",
        "The fitted variants operate on exported `fx_b`, `fy_b`, and `fz_b` predictions only.",
        "",
        "## Variants",
        "",
    ]
    for name, description in dict(manifest["variants"]).items():
        lines.append(f"- `{name}`: {description}")
    lines.extend(
        [
            "",
            "## Force Mean Summary",
            "",
            "| split | variant | RMSE | weighted RMSE | R2 | RMSE reduction vs A0 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in force_mean.iterrows():
        lines.append(
            "| {split} | `{variant}` | {rmse:.6g} | {weighted_rmse:.6g} | {r2:.6g} | {reduction:.6g} |".format(
                split=row["split"],
                variant=row["variant"],
                rmse=float(row["rmse"]),
                weighted_rmse=float(row["weighted_rmse"]),
                r2=float(row["r2"]),
                reduction=float(row["rmse_reduction_fraction_vs_A0"]),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `parameters.csv`: affine coefficients and biases.",
            "- `metrics_by_split.csv`: per-channel and force-mean train/val/test metrics.",
            "- `force_metrics_summary.csv`: metrics plus deltas relative to A0.",
            "- `residual_by_phase.csv`: phase-binned residual medians.",
            "- `residual_by_condition.csv`: quantile-binned residual RMSE by available conditions.",
            "- `residual_frequency_energy.csv`: residual energy by low/mid/flap/harmonic/high frequency bands.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_channel_weights(text: str) -> dict[str, float]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    weights = dict(DEFAULT_CHANNEL_WEIGHTS)
    for part in parts:
        if "=" not in part:
            raise ValueError(f"expected channel=weight item, got {part!r}")
        channel, value = part.split("=", 1)
        channel = channel.strip()
        if channel not in FORCE_COLUMNS:
            raise ValueError(f"unknown force channel {channel!r}")
        weights[channel] = float(value)
    return weights


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--prior-root", type=Path, default=DEFAULT_PRIOR_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--channel-weights",
        default=",".join(f"{channel}={weight}" for channel, weight in DEFAULT_CHANNEL_WEIGHTS.items()),
        help="Comma-separated weights for A2 and weighted force metrics, e.g. fx_b=2,fy_b=0.5,fz_b=2.",
    )
    parser.add_argument("--phase-bins", type=int, default=36)
    parser.add_argument("--condition-bins", type=int, default=5)
    parser.add_argument("--min-condition-samples", type=int, default=500)
    parser.add_argument("--skip-frequency", action="store_true")
    parser.add_argument(
        "--allow-row-order-fallback",
        action="store_true",
        help="Allow legacy prior parquets without sample keys to be paired by row order.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = run_force_recalibration(
        split_root=args.split_root,
        prior_root=args.prior_root,
        output_root=args.output_root,
        channel_weights=_parse_channel_weights(args.channel_weights),
        phase_bins=args.phase_bins,
        condition_bins=args.condition_bins,
        min_condition_samples=args.min_condition_samples,
        skip_frequency=args.skip_frequency,
        allow_row_order_fallback=args.allow_row_order_fallback,
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
