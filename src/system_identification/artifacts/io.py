"""Existing training curve and prediction diagnostic artifact writers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def sha256_file(path: str | Path) -> str:
    """Hash one artifact without interpreting or modifying it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> Path:
    """Write deterministic JSON while refusing to overwrite an existing file."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def write_table(path: str | Path, frame: pd.DataFrame) -> Path:
    """Write CSV or Parquet by suffix while refusing overwrite."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".parquet":
        frame.to_parquet(destination, index=False)
    elif destination.suffix.lower() == ".csv":
        frame.to_csv(destination, index=False)
    else:
        raise ValueError(f"Unsupported table suffix: {destination.suffix}")
    return destination


def _save_training_curves(history: pd.DataFrame, output_path: str | Path) -> None:
    from system_identification.training.data_preparation import DEFAULT_TARGET_COLUMNS

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = history["epoch"].to_numpy()

    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Scaled MSE")
    axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["val_overall_rmse"], label="val_overall_rmse")
    axes[1].plot(epochs, history["val_overall_mae"], label="val_overall_mae")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Wrench Error")
    axes[1].set_title("Validation Error")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    for target_name in DEFAULT_TARGET_COLUMNS:
        r2_column = f"val_{target_name}_r2"
        if r2_column in history.columns:
            axes[2].plot(epochs, history[r2_column], label=target_name)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("R^2")
    axes[2].set_title("Validation Per-Target R^2")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_pred_vs_true_plot(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    batch_size: int,
    device: str | None = None,
) -> None:
    from system_identification.evaluation.diagnostics import _targets_for_bundle, predict_model_bundle

    targets_df = _targets_for_bundle(bundle, frame)
    predictions_df = predict_model_bundle(bundle, frame, batch_size=batch_size, device=device)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()
    for idx, target_name in enumerate(bundle["target_columns"]):
        ax = axes_flat[idx]
        y_true = targets_df[target_name].to_numpy()
        y_pred = predictions_df[target_name].to_numpy()
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        ax.scatter(y_true, y_pred, s=5, alpha=0.15)
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
        ax.set_title(target_name)
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_residual_hist_plot(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    batch_size: int,
    device: str | None = None,
) -> None:
    from system_identification.evaluation.diagnostics import _targets_for_bundle, predict_model_bundle

    targets_df = _targets_for_bundle(bundle, frame)
    predictions_df = predict_model_bundle(bundle, frame, batch_size=batch_size, device=device)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()
    for idx, target_name in enumerate(bundle["target_columns"]):
        ax = axes_flat[idx]
        residual = predictions_df[target_name].to_numpy() - targets_df[target_name].to_numpy()
        ax.hist(residual, bins=50, alpha=0.8, color="steelblue", edgecolor="black")
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_title(target_name)
        ax.set_xlabel("Residual")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
