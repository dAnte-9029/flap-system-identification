#!/usr/bin/env python3
"""Analyze phase-locked structure in DeLaurier residual predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMNS = ("fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b")


def _required_columns(targets: tuple[str, ...]) -> list[str]:
    columns = ["phase_corrected_rad"]
    for target in targets:
        columns.extend([f"label_{target}", f"prior_{target}", f"pred_{target}"])
    return columns


def _check_columns(frame: pd.DataFrame, targets: tuple[str, ...]) -> None:
    missing = [column for column in _required_columns(targets) if column not in frame.columns]
    if missing:
        raise ValueError(f"aligned frame is missing required columns: {missing}")


def _rmse(values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if not mask.any():
        return float("nan")
    finite = values[mask]
    return float(np.sqrt(np.mean(finite * finite)))


def _r2_against_zero_mean(values: np.ndarray, prediction: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(prediction)
    if not mask.any():
        return float("nan")
    y = values[mask]
    y_hat = prediction[mask]
    ss_res = float(np.sum((y - y_hat) ** 2))
    centered = y - float(np.mean(y))
    ss_tot = float(np.sum(centered * centered))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def _peak_to_peak(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite) - np.min(finite)) if len(finite) else float("nan")


def _phase_bin_indices(phase: np.ndarray, phase_bins: int) -> np.ndarray:
    wrapped = np.mod(phase, 2.0 * np.pi)
    scaled = np.floor(wrapped / (2.0 * np.pi) * float(phase_bins)).astype(int)
    return np.clip(scaled, 0, phase_bins - 1)


def phase_bin_table(frame: pd.DataFrame, *, targets: tuple[str, ...] = TARGET_COLUMNS, phase_bins: int = 36) -> pd.DataFrame:
    """Return per-target residual medians binned by corrected wingbeat phase."""

    if phase_bins <= 0:
        raise ValueError("phase_bins must be positive")
    _check_columns(frame, targets)

    phase = frame["phase_corrected_rad"].to_numpy(dtype=float)
    bin_index = _phase_bin_indices(phase, phase_bins)
    bin_width = 2.0 * np.pi / float(phase_bins)

    rows: list[dict[str, float | int | str]] = []
    for target in targets:
        true_residual = frame[f"label_{target}"].to_numpy(dtype=float) - frame[f"prior_{target}"].to_numpy(dtype=float)
        pred_residual = frame[f"pred_{target}"].to_numpy(dtype=float)
        remaining_residual = true_residual - pred_residual
        for idx in range(phase_bins):
            mask = bin_index == idx
            row = {
                "target": target,
                "phase_bin": int(idx),
                "phase_center_rad": float((idx + 0.5) * bin_width),
                "sample_count": int(mask.sum()),
            }
            for name, values in (
                ("true_residual", true_residual),
                ("pred_residual", pred_residual),
                ("remaining_residual", remaining_residual),
            ):
                selected = values[mask]
                finite = selected[np.isfinite(selected)]
                row[f"{name}_median"] = float(np.median(finite)) if len(finite) else float("nan")
                row[f"{name}_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
                row[f"{name}_mad"] = (
                    float(np.median(np.abs(finite - np.median(finite)))) if len(finite) else float("nan")
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _phase_lookup_prediction(
    frame: pd.DataFrame,
    phase_table: pd.DataFrame,
    *,
    target: str,
    column: str,
    phase_bins: int,
) -> np.ndarray:
    values = (
        phase_table.loc[phase_table["target"] == target]
        .sort_values("phase_bin")[column]
        .to_numpy(dtype=float, copy=True)
    )
    if len(values) != phase_bins:
        raise ValueError(f"phase table for {target} has {len(values)} bins, expected {phase_bins}")
    indices = _phase_bin_indices(frame["phase_corrected_rad"].to_numpy(dtype=float), phase_bins)
    return values[indices]


def phase_summary_table(
    frame: pd.DataFrame,
    phase_table: pd.DataFrame,
    *,
    targets: tuple[str, ...] = TARGET_COLUMNS,
) -> pd.DataFrame:
    """Summarize phase-locked residual structure and model residual capture."""

    _check_columns(frame, targets)
    phase_bins = int(phase_table["phase_bin"].max()) + 1
    rows: list[dict[str, float | int | str]] = []
    for target in targets:
        true_residual = frame[f"label_{target}"].to_numpy(dtype=float) - frame[f"prior_{target}"].to_numpy(dtype=float)
        pred_residual = frame[f"pred_{target}"].to_numpy(dtype=float)
        remaining_residual = true_residual - pred_residual
        true_phase_median = _phase_lookup_prediction(
            frame, phase_table, target=target, column="true_residual_median", phase_bins=phase_bins
        )
        pred_phase_median = _phase_lookup_prediction(
            frame, phase_table, target=target, column="pred_residual_median", phase_bins=phase_bins
        )
        remaining_phase_median = _phase_lookup_prediction(
            frame, phase_table, target=target, column="remaining_residual_median", phase_bins=phase_bins
        )

        true_rmse = _rmse(true_residual)
        pred_error_rmse = _rmse(true_residual - pred_residual)
        phase_rmse = _rmse(true_phase_median)
        pred_phase_error_rmse = _rmse(true_phase_median - pred_phase_median)
        true_phase_peak_to_peak = _peak_to_peak(true_phase_median)
        remaining_phase_peak_to_peak = _peak_to_peak(remaining_phase_median)
        rows.append(
            {
                "target": target,
                "sample_count": int(np.isfinite(true_residual).sum()),
                "true_residual_rmse": true_rmse,
                "pred_residual_rmse": _rmse(pred_residual),
                "remaining_residual_rmse": pred_error_rmse,
                "rmse_reduction_fraction": float(1.0 - pred_error_rmse / true_rmse) if true_rmse > 0.0 else float("nan"),
                "true_residual_bias": _finite_mean(true_residual),
                "remaining_residual_bias": _finite_mean(remaining_residual),
                "phase_median_rmse": phase_rmse,
                "phase_median_to_true_rmse": float(phase_rmse / true_rmse) if true_rmse > 0.0 else float("nan"),
                "true_phase_peak_to_peak": true_phase_peak_to_peak,
                "pred_phase_peak_to_peak": _peak_to_peak(pred_phase_median),
                "remaining_phase_peak_to_peak": remaining_phase_peak_to_peak,
                "phase_peak_to_peak_reduction_fraction": (
                    float(1.0 - remaining_phase_peak_to_peak / true_phase_peak_to_peak)
                    if true_phase_peak_to_peak > 0.0
                    else float("nan")
                ),
                "phase_r2_true_residual": _r2_against_zero_mean(true_residual, true_phase_median),
                "phase_r2_remaining_residual": _r2_against_zero_mean(remaining_residual, remaining_phase_median),
                "pred_phase_pattern_r2": _r2_against_zero_mean(true_phase_median, pred_phase_median),
                "pred_phase_pattern_rmse": pred_phase_error_rmse,
            }
        )
    return pd.DataFrame(rows)


def _plot_phase_medians(phase_table: pd.DataFrame, output_stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "true_residual_median": "#0072B2",
        "pred_residual_median": "#D55E00",
        "remaining_residual_median": "#009E73",
    }
    labels = {
        "true_residual_median": "true residual",
        "pred_residual_median": "predicted residual",
        "remaining_residual_median": "remaining residual",
    }

    targets = list(dict.fromkeys(phase_table["target"].astype(str).tolist()))
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.5), sharex=True)
    for ax, target in zip(axes.flat, targets):
        subset = phase_table.loc[phase_table["target"] == target].sort_values("phase_bin")
        x = subset["phase_center_rad"].to_numpy(dtype=float)
        for column in ("true_residual_median", "pred_residual_median", "remaining_residual_median"):
            ax.plot(x, subset[column].to_numpy(dtype=float), color=colors[column], linewidth=1.5, label=labels[column])
        ax.axhline(0.0, color="0.2", linewidth=0.7, alpha=0.5)
        ax.set_title(target)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("wingbeat phase (rad)")
    for ax in axes[:, 0]:
        ax.set_ylabel("residual wrench")
    handles, labels_out = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def run_phase_analysis(aligned_parquet: Path, output_dir: Path, *, phase_bins: int = 36) -> dict[str, str]:
    frame = pd.read_parquet(aligned_parquet)
    phase_table = phase_bin_table(frame, targets=TARGET_COLUMNS, phase_bins=phase_bins)
    summary = phase_summary_table(frame, phase_table, targets=TARGET_COLUMNS)

    output_dir.mkdir(parents=True, exist_ok=True)
    phase_table_path = output_dir / "phase_binned_residuals.csv"
    summary_path = output_dir / "phase_residual_summary.csv"
    config_path = output_dir / "phase_residual_config.json"
    plot_stem = output_dir / "phase_residual_medians"

    phase_table.to_csv(phase_table_path, index=False)
    summary.to_csv(summary_path, index=False)
    _plot_phase_medians(phase_table, plot_stem)
    config = {
        "aligned_parquet": str(aligned_parquet),
        "output_dir": str(output_dir),
        "phase_bins": int(phase_bins),
        "targets": list(TARGET_COLUMNS),
        "sample_count": int(len(frame)),
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "phase_table": str(phase_table_path),
        "summary": str(summary_path),
        "config": str(config_path),
        "plot_png": str(plot_stem.with_suffix(".png")),
        "plot_pdf": str(plot_stem.with_suffix(".pdf")),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-parquet", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phase-bins", type=int, default=36)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = run_phase_analysis(args.aligned_parquet, args.output_dir, phase_bins=args.phase_bins)
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
