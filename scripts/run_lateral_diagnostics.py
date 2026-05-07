#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.training import prediction_metadata_frame_for_bundle

DEFAULT_MODEL_BUNDLE = (
    PROJECT_ROOT
    / "artifacts/20260507_transformer_focused_final/runs/"
    "transformer_focused_final_hist128_d64_l2_h4_do050/"
    "causal_transformer_paper_no_accel_v2_phase_actuator_airdata/model_bundle.pt"
)
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/20260507_lateral_diagnostics_best_transformer"

DEFAULT_TARGETS = ("fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b")
LATERAL_TARGETS = ("fy_b", "mx_b", "mz_b")
REFERENCE_TARGETS = ("fx_b", "fz_b", "my_b")

DEFAULT_BIN_SPECS = {
    "airspeed_validated.true_airspeed_m_s": [0.0, 6.0, 8.0, 10.0, 12.0, 16.0],
    "cycle_flap_frequency_hz": [0.0, 3.0, 5.0, 7.0, 10.0],
    "phase_corrected_rad": [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi],
    "servo_rudder": [-1.0, -0.05, 0.05, 1.0],
    "elevon_diff": [-1.0, -0.05, 0.05, 1.0],
}


def _rmse(true: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(true) & np.isfinite(pred)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(true[valid] - pred[valid]))))


def _r2(true: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(true) & np.isfinite(pred)
    if valid.sum() < 2:
        return float("nan")
    true_valid = true[valid]
    pred_valid = pred[valid]
    ss_res = float(np.sum(np.square(true_valid - pred_valid)))
    ss_tot = float(np.sum(np.square(true_valid - np.mean(true_valid))))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _finite_std(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return float("nan")
    return float(np.std(valid))


def _finite_quantile_abs(values: np.ndarray, q: float) -> float:
    valid = np.abs(values[np.isfinite(values)])
    if len(valid) == 0:
        return float("nan")
    return float(np.quantile(valid, q))


def _nanmean_or_nan(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or not np.isfinite(array).any():
        return float("nan")
    return float(np.nanmean(array))


def compute_target_scale_table(aligned: pd.DataFrame, *, target_columns: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for target in target_columns:
        true_column = f"true_{target}"
        pred_column = f"pred_{target}"
        if true_column not in aligned.columns or pred_column not in aligned.columns:
            continue
        true = aligned[true_column].to_numpy(dtype=float)
        pred = aligned[pred_column].to_numpy(dtype=float)
        resid = true - pred
        true_std = _finite_std(true)
        rmse = _rmse(true, pred)
        rows.append(
            {
                "target": target,
                "sample_count": int(np.isfinite(true).sum()),
                "true_mean": float(np.nanmean(true)),
                "true_std": true_std,
                "mean_abs_true": float(np.nanmean(np.abs(true))),
                "p95_abs_true": _finite_quantile_abs(true, 0.95),
                "rmse": rmse,
                "rmse_over_std": float(rmse / true_std) if np.isfinite(true_std) and true_std > 0.0 else float("nan"),
                "r2": _r2(true, pred),
                "resid_mean": float(np.nanmean(resid)),
                "resid_std": _finite_std(resid),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_over_std", ascending=False, na_position="last").reset_index(drop=True)


def _target_metrics_for_frame(frame: pd.DataFrame, targets: Iterable[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    rmse_values: list[float] = []
    r2_values: list[float] = []
    for target in targets:
        true = frame[f"true_{target}"].to_numpy(dtype=float)
        pred = frame[f"pred_{target}"].to_numpy(dtype=float)
        rmse = _rmse(true, pred)
        r2 = _r2(true, pred)
        true_std = _finite_std(true)
        metrics[f"{target}_rmse"] = rmse
        metrics[f"{target}_r2"] = r2
        metrics[f"{target}_rmse_over_std"] = (
            float(rmse / true_std) if np.isfinite(rmse) and np.isfinite(true_std) and true_std > 0.0 else float("nan")
        )
        rmse_values.append(rmse)
        r2_values.append(r2)
    metrics["lateral_rmse_mean"] = _nanmean_or_nan(rmse_values)
    metrics["lateral_r2_mean"] = _nanmean_or_nan(r2_values)
    return metrics


def compute_per_log_lateral_table(aligned: pd.DataFrame, *, lateral_targets: Iterable[str] = LATERAL_TARGETS) -> pd.DataFrame:
    if "log_id" not in aligned.columns:
        raise ValueError("aligned frame must contain log_id for per-log diagnostics")

    rows: list[dict[str, float | int | str]] = []
    for log_id, group in aligned.groupby("log_id", dropna=False):
        row: dict[str, float | int | str] = {"log_id": str(log_id), "sample_count": int(len(group))}
        row.update(_target_metrics_for_frame(group, lateral_targets))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("lateral_rmse_mean", ascending=False, na_position="last").reset_index(drop=True)


def _frame_with_derived_regime_columns(aligned: pd.DataFrame) -> pd.DataFrame:
    frame = aligned.copy()
    if "elevon_diff" not in frame.columns and {"servo_left_elevon", "servo_right_elevon"}.issubset(frame.columns):
        frame["elevon_diff"] = frame["servo_left_elevon"] - frame["servo_right_elevon"]
    return frame


def compute_regime_lateral_table(
    aligned: pd.DataFrame,
    *,
    lateral_targets: Iterable[str] = LATERAL_TARGETS,
    bin_specs: dict[str, list[float]] | None = None,
    min_samples: int = 100,
) -> pd.DataFrame:
    frame = _frame_with_derived_regime_columns(aligned)
    specs = bin_specs or DEFAULT_BIN_SPECS
    rows: list[dict[str, float | int | str]] = []

    for regime, bins in specs.items():
        if regime not in frame.columns:
            continue
        binned = pd.cut(frame[regime], bins=bins, include_lowest=True, right=False)
        for interval, group in frame.groupby(binned, observed=True):
            if len(group) < min_samples:
                continue
            row: dict[str, float | int | str] = {
                "regime": regime,
                "bin": str(interval),
                "sample_count": int(len(group)),
                "value_min": float(group[regime].min()),
                "value_max": float(group[regime].max()),
            }
            row.update(_target_metrics_for_frame(group, lateral_targets))
            rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "regime",
                "bin",
                "sample_count",
                "value_min",
                "value_max",
                "lateral_rmse_mean",
                "lateral_r2_mean",
            ]
        )
    return pd.DataFrame(rows).sort_values("lateral_rmse_mean", ascending=False, na_position="last").reset_index(drop=True)


def _top_rows_markdown(table: pd.DataFrame, columns: list[str], *, n: int = 8) -> str:
    if table.empty:
        return "_No rows._"
    visible_columns = [column for column in columns if column in table.columns]
    rows = table.loc[:, visible_columns].head(n)
    rendered = ["| " + " | ".join(visible_columns) + " |", "| " + " | ".join(["---"] * len(visible_columns)) + " |"]
    for _, row in rows.iterrows():
        values: list[str] = []
        for column in visible_columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}" if np.isfinite(value) else "nan")
            else:
                values.append(str(value))
        rendered.append("| " + " | ".join(values) + " |")
    return "\n".join(rendered)


def write_batch1_summary(
    *,
    output_path: Path,
    target_scale: pd.DataFrame,
    per_log: pd.DataFrame,
    per_regime: pd.DataFrame,
) -> None:
    lateral_scale = target_scale[target_scale["target"].isin(LATERAL_TARGETS)]
    reference_scale = target_scale[target_scale["target"].isin(REFERENCE_TARGETS)]
    lateral_ratio = float(lateral_scale["rmse_over_std"].mean()) if not lateral_scale.empty else float("nan")
    reference_ratio = float(reference_scale["rmse_over_std"].mean()) if not reference_scale.empty else float("nan")

    lines = [
        "# Lateral Diagnostics Batch 1 Summary",
        "",
        "## Scale Check",
        "",
        f"- Mean lateral RMSE/std: {lateral_ratio:.4f}",
        f"- Mean reference RMSE/std: {reference_ratio:.4f}",
        f"- Lateral worse by RMSE/std: {bool(lateral_ratio > reference_ratio)}",
        "",
        _top_rows_markdown(
            target_scale,
            ["target", "true_std", "rmse", "rmse_over_std", "r2", "mean_abs_true", "p95_abs_true"],
            n=6,
        ),
        "",
        "## Worst Logs",
        "",
        _top_rows_markdown(
            per_log,
            ["log_id", "sample_count", "lateral_rmse_mean", "lateral_r2_mean", "fy_b_rmse", "mx_b_rmse", "mz_b_rmse"],
            n=12,
        ),
        "",
        "## Worst Regime Bins",
        "",
        _top_rows_markdown(
            per_regime,
            ["regime", "bin", "sample_count", "lateral_rmse_mean", "lateral_r2_mean", "fy_b_rmse", "mx_b_rmse", "mz_b_rmse"],
            n=16,
        ),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_batch1(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = torch.load(args.model_bundle, map_location="cpu", weights_only=False)
    frame = pd.read_parquet(Path(args.split_root) / f"{args.split}_samples.parquet")
    target_columns = tuple(bundle.get("target_columns", DEFAULT_TARGETS))

    aligned = prediction_metadata_frame_for_bundle(
        bundle,
        frame,
        split_name=args.split,
        batch_size=args.batch_size,
        device=args.device,
    )
    target_scale = compute_target_scale_table(aligned, target_columns=target_columns)
    lateral_targets = tuple(target for target in LATERAL_TARGETS if target in target_columns)
    per_log = compute_per_log_lateral_table(aligned, lateral_targets=lateral_targets)
    per_regime = compute_regime_lateral_table(aligned, lateral_targets=lateral_targets, min_samples=args.min_bin_samples)

    paths = {
        "aligned_predictions": output_dir / "aligned_predictions.parquet",
        "target_scale": output_dir / "target_scale.csv",
        "per_log": output_dir / "per_log_lateral_metrics.csv",
        "per_regime": output_dir / "per_regime_lateral_metrics.csv",
        "summary": output_dir / "batch1_summary.md",
        "config": output_dir / "diagnostics_config.json",
    }
    aligned.to_parquet(paths["aligned_predictions"], index=False)
    target_scale.to_csv(paths["target_scale"], index=False)
    per_log.to_csv(paths["per_log"], index=False)
    per_regime.to_csv(paths["per_regime"], index=False)
    write_batch1_summary(output_path=paths["summary"], target_scale=target_scale, per_log=per_log, per_regime=per_regime)

    config = {
        "model_bundle": str(Path(args.model_bundle)),
        "split_root": str(Path(args.split_root)),
        "split": args.split,
        "batch": "first",
        "batch_size": args.batch_size,
        "device": args.device,
        "min_bin_samples": args.min_bin_samples,
        "target_columns": list(target_columns),
        "lateral_targets": list(lateral_targets),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    paths["config"].write_text(json.dumps(config, indent=2), encoding="utf-8")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-bundle", type=Path, default=DEFAULT_MODEL_BUNDLE)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch", default="first", choices=("first", "second"))
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default=None)
    parser.add_argument("--min-bin-samples", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch != "first":
        raise NotImplementedError("Batch 2 is intentionally not implemented in the Batch 1 checkpoint.")
    paths = run_batch1(args)
    print("Batch 1 diagnostics written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
