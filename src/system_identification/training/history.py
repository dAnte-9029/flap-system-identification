"""Validation-history row construction with the existing schemas."""

from __future__ import annotations

from typing import Any


def build_sequence_validation_history_row(
    *,
    epoch: int,
    learning_rate: float,
    train_loss: float,
    train_supervised_loss: float,
    train_prior_loss: float,
    val_loss: float,
    val_metrics: dict[str, Any],
) -> dict[str, float]:
    row: dict[str, float] = {
        "epoch": float(epoch),
        "learning_rate": float(learning_rate),
        "train_loss": float(train_loss),
        "train_total_loss": float(train_loss),
        "train_supervised_loss": float(train_supervised_loss),
        "train_prior_loss": float(train_prior_loss),
        "val_loss": float(val_loss),
        "val_overall_mae": float(val_metrics["overall_mae"]),
        "val_overall_rmse": float(val_metrics["overall_rmse"]),
        "val_overall_r2": float(val_metrics["overall_r2"]),
    }
    for target_name, metrics in val_metrics["per_target"].items():
        row[f"val_{target_name}_mae"] = float(metrics["mae"])
        row[f"val_{target_name}_rmse"] = float(metrics["rmse"])
        row[f"val_{target_name}_r2"] = float(metrics["r2"])
    return row


def build_rollout_validation_history_row(
    *,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_metrics: dict[str, Any],
    latent_rms: float,
    delta_latent_rms: float,
    latent_derivative_rms: float,
) -> dict[str, float]:
    row: dict[str, float] = {
        "epoch": float(epoch),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_overall_mae": float(val_metrics["overall_mae"]),
        "val_overall_rmse": float(val_metrics["overall_rmse"]),
        "val_overall_r2": float(val_metrics["overall_r2"]),
        "latent_rms": float(latent_rms),
        "delta_latent_rms": float(delta_latent_rms),
        "latent_derivative_rms": float(latent_derivative_rms),
    }
    for target_name, metrics in val_metrics["per_target"].items():
        row[f"val_{target_name}_mae"] = float(metrics["mae"])
        row[f"val_{target_name}_rmse"] = float(metrics["rmse"])
        row[f"val_{target_name}_r2"] = float(metrics["r2"])
    return row


def build_validation_history_row(
    *,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_metrics: dict[str, Any],
) -> dict[str, float]:
    row: dict[str, float] = {
        "epoch": float(epoch),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_overall_mae": float(val_metrics["overall_mae"]),
        "val_overall_rmse": float(val_metrics["overall_rmse"]),
        "val_overall_r2": float(val_metrics["overall_r2"]),
    }
    for target_name, metrics in val_metrics["per_target"].items():
        row[f"val_{target_name}_mae"] = float(metrics["mae"])
        row[f"val_{target_name}_rmse"] = float(metrics["rmse"])
        row[f"val_{target_name}_r2"] = float(metrics["r2"])
    return row
