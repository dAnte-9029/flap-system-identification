#!/usr/bin/env python3
"""Analyze DeLaurier residuals across flight-condition bins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMNS = ("fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b")
DEFAULT_CONDITION_COLUMNS = (
    "airspeed_validated.true_airspeed_m_s",
    "dynamic_pressure_pa",
    "alpha_rad",
    "cycle_flap_frequency_hz",
)
KEY_PLOT_TARGETS = ("fx_b", "fz_b", "my_b")
CONDITION_LABELS = {
    "airspeed_validated.true_airspeed_m_s": "airspeed",
    "dynamic_pressure_pa": "dynamic pressure",
    "alpha_rad": "angle of attack",
    "cycle_flap_frequency_hz": "flap frequency",
}


def _required_columns(condition_columns: tuple[str, ...], targets: tuple[str, ...]) -> list[str]:
    columns = list(condition_columns)
    for target in targets:
        columns.extend([f"label_{target}", f"prior_{target}", f"pred_{target}"])
    return columns


def _check_columns(frame: pd.DataFrame, condition_columns: tuple[str, ...], targets: tuple[str, ...]) -> None:
    missing = [column for column in _required_columns(condition_columns, targets) if column not in frame.columns]
    if missing:
        raise ValueError(f"aligned frame is missing required columns: {missing}")


def _rmse(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite * finite)))


def _mae(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(np.abs(finite))) if len(finite) else float("nan")


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def _quantile_bin_codes(values: pd.Series, quantile_bins: int) -> pd.Series:
    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    if finite.nunique(dropna=True) < 2:
        return pd.Series(pd.NA, index=values.index, dtype="Int64")
    ranked = finite.rank(method="first")
    binned = pd.qcut(ranked, q=min(quantile_bins, int(finite.nunique(dropna=True))), labels=False, duplicates="drop")
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    result.loc[finite.index] = binned.astype("Int64")
    return result


def condition_bin_table(
    frame: pd.DataFrame,
    *,
    condition_columns: tuple[str, ...] = DEFAULT_CONDITION_COLUMNS,
    targets: tuple[str, ...] = TARGET_COLUMNS,
    quantile_bins: int = 5,
    min_samples: int = 100,
) -> pd.DataFrame:
    """Return residual metrics for target channels binned by flight-condition quantiles."""

    if quantile_bins <= 0:
        raise ValueError("quantile_bins must be positive")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    _check_columns(frame, condition_columns, targets)

    rows: list[dict[str, float | int | str]] = []
    for condition in condition_columns:
        values = frame[condition].astype(float)
        bin_codes = _quantile_bin_codes(values, quantile_bins)
        for bin_id in sorted(code for code in bin_codes.dropna().unique().tolist()):
            mask = bin_codes == bin_id
            condition_values = values[mask].to_numpy(dtype=float)
            finite_condition = condition_values[np.isfinite(condition_values)]
            if len(finite_condition) < min_samples:
                continue
            bin_label = f"[{np.min(finite_condition):.6g}, {np.max(finite_condition):.6g}]"
            for target in targets:
                true_residual = (
                    frame.loc[mask, f"label_{target}"].to_numpy(dtype=float)
                    - frame.loc[mask, f"prior_{target}"].to_numpy(dtype=float)
                )
                pred_residual = frame.loc[mask, f"pred_{target}"].to_numpy(dtype=float)
                remaining = true_residual - pred_residual
                true_rmse = _rmse(true_residual)
                remaining_rmse = _rmse(remaining)
                rows.append(
                    {
                        "condition": condition,
                        "target": target,
                        "condition_bin": int(bin_id),
                        "bin_label": bin_label,
                        "sample_count": int(len(finite_condition)),
                        "value_min": float(np.min(finite_condition)),
                        "value_max": float(np.max(finite_condition)),
                        "value_median": float(np.median(finite_condition)),
                        "true_residual_rmse": true_rmse,
                        "remaining_residual_rmse": remaining_rmse,
                        "rmse_reduction_fraction": (
                            float(1.0 - remaining_rmse / true_rmse) if true_rmse > 0.0 else float("nan")
                        ),
                        "true_residual_mae": _mae(true_residual),
                        "remaining_residual_mae": _mae(remaining),
                        "true_residual_bias": _finite_mean(true_residual),
                        "remaining_residual_bias": _finite_mean(remaining),
                    }
                )
    return pd.DataFrame(rows)


def condition_summary_table(condition_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize residual variation across bins for each condition and target."""

    rows: list[dict[str, float | int | str]] = []
    if condition_table.empty:
        return pd.DataFrame()
    for (condition, target), group in condition_table.groupby(["condition", "target"], observed=True):
        ordered = group.sort_values("true_residual_rmse", ascending=False)
        worst = ordered.iloc[0]
        true_min = float(group["true_residual_rmse"].min())
        true_max = float(group["true_residual_rmse"].max())
        remaining_min = float(group["remaining_residual_rmse"].min())
        remaining_max = float(group["remaining_residual_rmse"].max())
        rows.append(
            {
                "condition": str(condition),
                "target": str(target),
                "bin_count": int(group["condition_bin"].nunique()),
                "sample_count": int(group["sample_count"].sum()),
                "true_rmse_min": true_min,
                "true_rmse_max": true_max,
                "true_rmse_max_to_min": float(true_max / true_min) if true_min > 0.0 else float("nan"),
                "remaining_rmse_min": remaining_min,
                "remaining_rmse_max": remaining_max,
                "remaining_rmse_max_to_min": (
                    float(remaining_max / remaining_min) if remaining_min > 0.0 else float("nan")
                ),
                "mean_rmse_reduction_fraction": float(group["rmse_reduction_fraction"].mean()),
                "worst_bin": int(worst["condition_bin"]),
                "worst_bin_label": str(worst["bin_label"]),
                "worst_bin_value_median": float(worst["value_median"]),
                "worst_bin_true_residual_rmse": float(worst["true_residual_rmse"]),
                "worst_bin_remaining_residual_rmse": float(worst["remaining_residual_rmse"]),
                "worst_bin_rmse_reduction_fraction": float(worst["rmse_reduction_fraction"]),
            }
        )
    return pd.DataFrame(rows)


def _plot_condition_rmse(condition_table: pd.DataFrame, output_stem: Path, *, key_targets: tuple[str, ...]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    conditions = list(dict.fromkeys(condition_table["condition"].astype(str).tolist()))
    fig, axes = plt.subplots(len(conditions), len(key_targets), figsize=(7.2, 1.75 * len(conditions)), squeeze=False)
    for row_idx, condition in enumerate(conditions):
        for col_idx, target in enumerate(key_targets):
            ax = axes[row_idx, col_idx]
            subset = condition_table.loc[
                (condition_table["condition"] == condition) & (condition_table["target"] == target)
            ].sort_values("condition_bin")
            if subset.empty:
                ax.set_axis_off()
                continue
            x = np.arange(len(subset))
            ax.plot(
                x,
                subset["true_residual_rmse"].to_numpy(dtype=float),
                marker="o",
                color="#0072B2",
                linewidth=1.4,
                label="DeLaurier residual",
            )
            ax.plot(
                x,
                subset["remaining_residual_rmse"].to_numpy(dtype=float),
                marker="s",
                color="#D55E00",
                linewidth=1.4,
                label="remaining after NN",
            )
            ax.set_title(f"{target}")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{value:.2g}" for value in subset["value_median"].to_numpy(dtype=float)], rotation=35)
            if col_idx == 0:
                label = CONDITION_LABELS.get(condition, condition)
                ax.set_ylabel(f"{label}\nRMSE")
            if row_idx == len(conditions) - 1:
                ax.set_xlabel("bin median")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def run_condition_analysis(
    aligned_parquet: Path,
    output_dir: Path,
    *,
    condition_columns: tuple[str, ...] = DEFAULT_CONDITION_COLUMNS,
    quantile_bins: int = 5,
    min_samples: int = 500,
) -> dict[str, str]:
    frame = pd.read_parquet(aligned_parquet)
    bins = condition_bin_table(
        frame,
        condition_columns=condition_columns,
        targets=TARGET_COLUMNS,
        quantile_bins=quantile_bins,
        min_samples=min_samples,
    )
    summary = condition_summary_table(bins)

    output_dir.mkdir(parents=True, exist_ok=True)
    bins_path = output_dir / "condition_residual_bins.csv"
    summary_path = output_dir / "condition_residual_summary.csv"
    config_path = output_dir / "condition_residual_config.json"
    plot_stem = output_dir / "condition_residual_rmse_key_targets"

    bins.to_csv(bins_path, index=False)
    summary.to_csv(summary_path, index=False)
    _plot_condition_rmse(bins, plot_stem, key_targets=KEY_PLOT_TARGETS)
    config = {
        "aligned_parquet": str(aligned_parquet),
        "output_dir": str(output_dir),
        "condition_columns": list(condition_columns),
        "targets": list(TARGET_COLUMNS),
        "quantile_bins": int(quantile_bins),
        "min_samples": int(min_samples),
        "sample_count": int(len(frame)),
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "bins": str(bins_path),
        "summary": str(summary_path),
        "config": str(config_path),
        "plot_png": str(plot_stem.with_suffix(".png")),
        "plot_pdf": str(plot_stem.with_suffix(".pdf")),
    }


def _parse_condition_columns(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return DEFAULT_CONDITION_COLUMNS
    columns = tuple(column.strip() for column in raw.split(",") if column.strip())
    if not columns:
        raise argparse.ArgumentTypeError("condition column list cannot be empty")
    return columns


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-parquet", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--condition-columns", default=None, type=_parse_condition_columns)
    parser.add_argument("--quantile-bins", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = run_condition_analysis(
        args.aligned_parquet,
        args.output_dir,
        condition_columns=args.condition_columns or DEFAULT_CONDITION_COLUMNS,
        quantile_bins=args.quantile_bins,
        min_samples=args.min_samples,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
