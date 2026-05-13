#!/usr/bin/env python3
"""Evaluate a residual model as prior + predicted residual against true effective wrench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

TARGET_COLUMNS = ["fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b"]


def _channel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {"mae": np.nan, "rmse": np.nan, "bias": np.nan, "r2": np.nan}
    residual = y_pred[mask] - y_true[mask]
    ss_res = float(np.sum(residual * residual))
    centered = y_true[mask] - float(np.mean(y_true[mask]))
    ss_tot = float(np.sum(centered * centered))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "bias": float(np.mean(residual)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else np.nan,
    }


def combined_metrics_from_aligned(aligned: pd.DataFrame) -> pd.DataFrame:
    """Compute prior-only, residual-only, and combined prior+residual metrics."""

    rows = []
    for target in TARGET_COLUMNS:
        true_column = f"label_{target}" if f"label_{target}" in aligned.columns else f"true_{target}"
        true = aligned[true_column].to_numpy(dtype=float)
        prior = aligned[f"prior_{target}"].to_numpy(dtype=float)
        pred_residual = aligned[f"pred_{target}"].to_numpy(dtype=float)
        combined = prior + pred_residual
        prior_metrics = _channel_metrics(true, prior)
        combined_metrics = _channel_metrics(true, combined)
        residual_true = true - prior
        residual_metrics = _channel_metrics(residual_true, pred_residual)
        row = {"target": target, "n": int(np.isfinite(true).sum())}
        for prefix, metrics in (
            ("prior", prior_metrics),
            ("residual", residual_metrics),
            ("combined", combined_metrics),
        ):
            for name, value in metrics.items():
                row[f"{prefix}_{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _load_bundle(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _align_prior_columns(aligned: pd.DataFrame) -> pd.DataFrame:
    missing = [f"prior_{target}" for target in TARGET_COLUMNS if f"prior_{target}" not in aligned.columns]
    if missing:
        raise ValueError(f"Residual split is missing prior columns in aligned metadata: {missing}")
    return aligned


def evaluate_residual_model(
    *,
    residual_split_root: Path,
    model_bundle: Path,
    output_dir: Path,
    split: str = "test",
    batch_size: int = 8192,
    device: str = "auto",
) -> dict[str, str]:
    from system_identification.training import _resolve_device, prediction_metadata_frame_for_bundle

    bundle = _load_bundle(model_bundle)
    resolved_device = _resolve_device(device)
    frame = pd.read_parquet(residual_split_root / f"{split}_samples.parquet")
    aligned = prediction_metadata_frame_for_bundle(
        bundle,
        frame,
        split_name=split,
        batch_size=batch_size,
        device=str(resolved_device),
    )
    aligned = _align_prior_columns(aligned)
    metrics = combined_metrics_from_aligned(aligned)

    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = output_dir / f"{split}_aligned_residual_predictions.parquet"
    metrics_path = output_dir / f"{split}_combined_metrics.csv"
    summary_path = output_dir / "summary.json"
    aligned.to_parquet(aligned_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    overall = {
        "prior_overall_rmse": float(np.sqrt(np.nanmean(metrics["prior_rmse"].to_numpy(dtype=float) ** 2))),
        "residual_overall_rmse": float(np.sqrt(np.nanmean(metrics["residual_rmse"].to_numpy(dtype=float) ** 2))),
        "combined_overall_rmse": float(np.sqrt(np.nanmean(metrics["combined_rmse"].to_numpy(dtype=float) ** 2))),
        "prior_overall_mae": float(np.nanmean(metrics["prior_mae"].to_numpy(dtype=float))),
        "combined_overall_mae": float(np.nanmean(metrics["combined_mae"].to_numpy(dtype=float))),
    }
    summary = {
        "residual_split_root": str(residual_split_root),
        "model_bundle": str(model_bundle),
        "split": split,
        "device": str(device),
        "resolved_device": str(resolved_device),
        "aligned_rows": int(len(aligned)),
        **overall,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "aligned_path": str(aligned_path),
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-split-root", required=True, type=Path)
    parser.add_argument("--model-bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = evaluate_residual_model(
        residual_split_root=args.residual_split_root,
        model_bundle=args.model_bundle,
        output_dir=args.output_dir,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
