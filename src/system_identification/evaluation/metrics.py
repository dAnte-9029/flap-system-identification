"""Pure evaluation metric and aggregation helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

def _metrics_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_columns: list[str],
    split_name: str,
) -> dict[str, Any]:
    residual = y_pred - y_true
    overall_mae = float(np.mean(np.abs(residual)))
    overall_rmse = float(np.sqrt(np.mean(np.square(residual))))

    per_target: dict[str, dict[str, float]] = {}
    r2_values: list[float] = []
    for idx, target_name in enumerate(target_columns):
        target_true = y_true[:, idx]
        target_pred = y_pred[:, idx]
        target_residual = target_pred - target_true
        ss_res = float(np.sum(np.square(target_residual)))
        ss_tot = float(np.sum(np.square(target_true - target_true.mean())))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
        r2_values.append(r2)
        per_target[target_name] = {
            "mae": float(np.mean(np.abs(target_residual))),
            "rmse": float(np.sqrt(np.mean(np.square(target_residual)))),
            "r2": float(r2),
        }

    return {
        "split": split_name,
        "sample_count": int(len(y_true)),
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "overall_r2": float(np.mean(r2_values)),
        "per_target": per_target,
    }


def _validate_bin_edges(column: str, edges: list[float]) -> list[float]:
    resolved = [float(edge) for edge in edges]
    if len(resolved) < 2:
        raise ValueError(f"Bin spec for {column} must contain at least two edges")
    if any(not math.isfinite(edge) for edge in resolved):
        raise ValueError(f"Bin spec for {column} must contain finite edges")
    if any(right <= left for left, right in zip(resolved, resolved[1:])):
        raise ValueError(f"Bin edges for {column} must be strictly increasing")
    return resolved


def _combine_disjoint_target_metrics(split_name: str, group_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_target: dict[str, dict[str, float]] = {}
    sample_counts: list[int] = []
    for metrics in group_metrics.values():
        sample_counts.append(int(metrics["sample_count"]))
        for target_name, target_metrics in metrics["per_target"].items():
            per_target[target_name] = {
                "mae": float(target_metrics["mae"]),
                "rmse": float(target_metrics["rmse"]),
                "r2": float(target_metrics["r2"]),
            }

    if not per_target:
        raise ValueError("Cannot combine split-axis metrics without per-target metrics")

    mae_values = np.array([metrics["mae"] for metrics in per_target.values()], dtype=float)
    rmse_values = np.array([metrics["rmse"] for metrics in per_target.values()], dtype=float)
    r2_values = np.array([metrics["r2"] for metrics in per_target.values()], dtype=float)

    return {
        "split": split_name,
        "sample_count": int(min(sample_counts)) if sample_counts else 0,
        "overall_mae": float(np.mean(mae_values)),
        "overall_rmse": float(np.sqrt(np.mean(np.square(rmse_values)))),
        "overall_r2": float(np.mean(r2_values)),
        "per_target": per_target,
    }
