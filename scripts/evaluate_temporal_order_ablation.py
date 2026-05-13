#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.training import (  # noqa: E402
    _as_numpy_array,
    _build_sequence_model_from_bundle,
    _inverse_transform_targets,
    _load_split_frame,
    _metrics_from_arrays,
    _predict_sequence_scaled_batches,
    _resolve_device,
    _sequence_arrays_for_bundle,
    _transform_features,
    _transform_sequence_features,
    apply_sequence_order_ablation,
)


def _evaluate_sequence_order_mode(
    *,
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    split_name: str,
    order_mode: str,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    sequence_features, current_features, targets_df = _sequence_arrays_for_bundle(bundle, frame)
    sequence_features = apply_sequence_order_ablation(sequence_features, order_mode, seed=seed)
    sequence_scaled = _transform_sequence_features(
        sequence_features,
        _as_numpy_array(bundle["sequence_feature_medians"]),
        _as_numpy_array(bundle["sequence_feature_means"]),
        _as_numpy_array(bundle["sequence_feature_stds"]),
    )
    if current_features.shape[1] > 0:
        current_scaled = _transform_features(
            current_features,
            _as_numpy_array(bundle["current_feature_medians"]),
            _as_numpy_array(bundle["current_feature_means"]),
            _as_numpy_array(bundle["current_feature_stds"]),
        )
    else:
        current_scaled = current_features.astype(np.float32, copy=False)

    model = _build_sequence_model_from_bundle(bundle, device)
    predictions_scaled = _predict_sequence_scaled_batches(
        model,
        sequence_scaled,
        current_scaled,
        batch_size=batch_size,
        device=device,
        use_amp=bool(bundle.get("use_amp", False)),
    )
    predictions = _inverse_transform_targets(
        predictions_scaled,
        _as_numpy_array(bundle["target_means"]),
        _as_numpy_array(bundle["target_stds"]),
    )
    return _metrics_from_arrays(
        targets_df.to_numpy(dtype=np.float64, copy=False),
        predictions.astype(np.float64, copy=False),
        target_columns=list(bundle["target_columns"]),
        split_name=split_name,
    )


def _flatten_metrics(split_name: str, order_mode: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "split": split_name,
        "order_mode": order_mode,
        "sample_count": int(metrics["sample_count"]),
        "overall_mae": float(metrics["overall_mae"]),
        "overall_rmse": float(metrics["overall_rmse"]),
        "overall_r2": float(metrics["overall_r2"]),
    }
    for target_name, target_metrics in metrics["per_target"].items():
        row[f"{target_name}_mae"] = float(target_metrics["mae"])
        row[f"{target_name}_rmse"] = float(target_metrics["rmse"])
        row[f"{target_name}_r2"] = float(target_metrics["r2"])
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sequence order perturbations for a trained sequence model")
    parser.add_argument("--model-bundle", required=True, help="Path to a trained sequence model_bundle.pt")
    parser.add_argument("--split-root", required=True, help="Dataset split root")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--split-names", nargs="+", default=["val"], help="Split names to evaluate")
    parser.add_argument("--order-modes", nargs="+", default=["normal", "reverse", "shuffle"], help="Order modes")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(args.model_bundle, map_location="cpu", weights_only=False)
    device = _resolve_device(args.device)

    if "sequence_feature_columns" not in bundle:
        raise ValueError("Temporal order ablation requires a sequence model bundle")

    rows: list[dict[str, Any]] = []
    for split_name in args.split_names:
        frame = _load_split_frame(args.split_root, split_name, None, args.seed)
        for order_mode in args.order_modes:
            metrics = _evaluate_sequence_order_mode(
                bundle=bundle,
                frame=frame,
                split_name=split_name,
                order_mode=order_mode,
                batch_size=args.batch_size,
                device=device,
                seed=args.seed,
            )
            rows.append(_flatten_metrics(split_name, order_mode, metrics))

    summary = pd.DataFrame(rows)
    summary_csv = output_dir / "temporal_order_eval_ablation_summary.csv"
    summary_json = output_dir / "temporal_order_eval_ablation_summary.json"
    config_json = output_dir / "temporal_order_eval_ablation_config.json"
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    config_json.write_text(
        json.dumps(
            {
                "model_bundle": str(args.model_bundle),
                "split_root": str(args.split_root),
                "split_names": list(args.split_names),
                "order_modes": list(args.order_modes),
                "batch_size": int(args.batch_size),
                "device": str(args.device),
                "seed": int(args.seed),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
