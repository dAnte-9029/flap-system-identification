"""Pure evaluation report-row and table preparation helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

def _history_frame(history: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(history)


def _flatten_split_metrics(split_name: str, metrics: dict[str, Any]) -> dict[str, float | int]:
    flat: dict[str, float | int] = {
        f"{split_name}_sample_count": int(metrics["sample_count"]),
        f"{split_name}_overall_mae": float(metrics["overall_mae"]),
        f"{split_name}_overall_rmse": float(metrics["overall_rmse"]),
        f"{split_name}_overall_r2": float(metrics["overall_r2"]),
    }
    for target_name, target_metrics in metrics["per_target"].items():
        for metric_name, value in target_metrics.items():
            flat[f"{split_name}_{target_name}_{metric_name}"] = float(value)
    return flat


def _metrics_table_row(
    metrics: dict[str, Any],
    *,
    split_name: str,
    diagnostic_type: str,
    group_column: str,
    group_value: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "split": split_name,
        "diagnostic_type": diagnostic_type,
        "group_column": group_column,
        "group_value": group_value,
    }
    row.update(_flatten_split_metrics(split_name, metrics))
    return row


def _target_groups_label(target_groups: dict[str, list[str]]) -> str:
    return ";".join(f"{group_name}:{'|'.join(targets)}" for group_name, targets in target_groups.items())
