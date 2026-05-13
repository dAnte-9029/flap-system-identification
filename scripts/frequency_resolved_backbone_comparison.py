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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from system_identification.training import prediction_metadata_frame_for_bundle  # noqa: E402

from scripts.diagnose_fyb_learnability import (  # noqa: E402
    corrcoef,
    fft_filter,
    nominal_sample_rate_hz,
    r2_score,
    rmse,
)


DEFAULT_TARGETS = ("fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b")
STRUCTURED_BANDS = (
    ("low_0_1hz", 0.0, 1.0),
    ("mid_1_3hz", 1.0, 3.0),
    ("flap_main", np.nan, np.nan),
)


def _band_edges_for_group(group: pd.DataFrame, band_name: str, low_hz: float, high_hz: float) -> tuple[float, float]:
    if band_name != "flap_main":
        return float(low_hz), float(high_hz)
    if "cycle_flap_frequency_hz" in group.columns:
        f0 = float(np.nanmedian(group["cycle_flap_frequency_hz"].to_numpy(dtype=float)))
    elif "flap_frequency_hz" in group.columns:
        f0 = float(np.nanmedian(group["flap_frequency_hz"].to_numpy(dtype=float)))
    else:
        f0 = float("nan")
    if not np.isfinite(f0) or f0 <= 0.0:
        f0 = 4.36
    return max(0.0, f0 - 0.75), f0 + 0.75


def _grouped_segments(frame: pd.DataFrame) -> list[pd.DataFrame]:
    group_columns = [column for column in ("log_id", "segment_id") if column in frame.columns]
    if not group_columns:
        return [frame.sort_values("time_s")]
    return [group.sort_values("time_s") for _, group in frame.groupby(group_columns, sort=False)]


def _filtered_pair_for_group(
    group: pd.DataFrame,
    *,
    target: str,
    low_hz: float,
    high_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    sample_rate_hz = nominal_sample_rate_hz(group["time_s"].to_numpy(dtype=float))
    true_values = group[f"true_{target}"].to_numpy(dtype=float)
    pred_values = group[f"pred_{target}"].to_numpy(dtype=float)
    return (
        fft_filter(true_values, sample_rate_hz=sample_rate_hz, low_hz=low_hz, high_hz=high_hz),
        fft_filter(pred_values, sample_rate_hz=sample_rate_hz, low_hz=low_hz, high_hz=high_hz),
    )


def filtered_arrays_for_frequency_component(
    frame: pd.DataFrame,
    *,
    target: str,
    component: str,
) -> tuple[np.ndarray, np.ndarray]:
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    for group in _grouped_segments(frame):
        if len(group) < 8:
            continue
        true_raw = group[f"true_{target}"].to_numpy(dtype=float)
        pred_raw = group[f"pred_{target}"].to_numpy(dtype=float)
        if component == "high_frequency_residual":
            true_component = true_raw.copy()
            pred_component = pred_raw.copy()
            for band_name, low_hz, high_hz in STRUCTURED_BANDS:
                low, high = _band_edges_for_group(group, band_name, low_hz, high_hz)
                true_band, pred_band = _filtered_pair_for_group(group, target=target, low_hz=low, high_hz=high)
                true_component = true_component - true_band
                pred_component = pred_component - pred_band
        else:
            matched = [band for band in STRUCTURED_BANDS if band[0] == component]
            if not matched:
                raise ValueError(f"Unknown frequency component: {component}")
            band_name, low_hz, high_hz = matched[0]
            low, high = _band_edges_for_group(group, band_name, low_hz, high_hz)
            true_component, pred_component = _filtered_pair_for_group(group, target=target, low_hz=low, high_hz=high)
        true_parts.append(true_component)
        pred_parts.append(pred_component)
    if not true_parts:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.concatenate(true_parts), np.concatenate(pred_parts)


def compute_frequency_component_metrics(
    frame: pd.DataFrame,
    *,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    components: tuple[str, ...] = ("low_0_1hz", "mid_1_3hz", "flap_main", "high_frequency_residual"),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for component in components:
        target_r2_values: list[float] = []
        target_rmse_values: list[float] = []
        for target in targets:
            true_component, pred_component = filtered_arrays_for_frequency_component(
                frame,
                target=target,
                component=component,
            )
            true_raw = frame[f"true_{target}"].to_numpy(dtype=float)
            true_var = float(np.nanvar(true_raw))
            row = {
                "component": component,
                "target": target,
                "sample_count": int(np.sum(np.isfinite(true_component) & np.isfinite(pred_component))),
                "true_std": float(np.nanstd(true_component)),
                "pred_std": float(np.nanstd(pred_component)),
                "true_variance_fraction": float(np.nanvar(true_component) / true_var) if true_var > 0.0 else float("nan"),
                "rmse": rmse(true_component, pred_component),
                "r2": r2_score(true_component, pred_component),
                "corr": corrcoef(true_component, pred_component),
            }
            rows.append(row)
            if np.isfinite(row["r2"]):
                target_r2_values.append(float(row["r2"]))
            if np.isfinite(row["rmse"]):
                target_rmse_values.append(float(row["rmse"]))
        rows.append(
            {
                "component": component,
                "target": "__mean__",
                "sample_count": int(sum(row["sample_count"] for row in rows if row["component"] == component and row["target"] != "__mean__")),
                "true_std": float("nan"),
                "pred_std": float("nan"),
                "true_variance_fraction": float("nan"),
                "rmse": float(np.mean(target_rmse_values)) if target_rmse_values else float("nan"),
                "r2": float(np.mean(target_r2_values)) if target_r2_values else float("nan"),
                "corr": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _parse_model_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise argparse.ArgumentTypeError(f"Model spec must be label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        if not label:
            raise argparse.ArgumentTypeError(f"Model label must not be empty in {spec!r}")
        parsed.append((label, Path(path)))
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frequency-resolved backbone comparison")
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="Model specs as label=path/to/model_bundle.pt")
    parser.add_argument("--split-names", nargs="+", default=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_specs = _parse_model_specs(args.models)

    rows: list[pd.DataFrame] = []
    for model_label, bundle_path in model_specs:
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        for split_name in args.split_names:
            frame_path = Path(args.split_root) / f"{split_name}_samples.parquet"
            split_frame = pd.read_parquet(frame_path)
            aligned = prediction_metadata_frame_for_bundle(
                bundle,
                split_frame,
                split_name=split_name,
                batch_size=args.batch_size,
                device=args.device,
            )
            aligned_path = output_dir / f"{model_label}_{split_name}_aligned_predictions.parquet"
            aligned.to_parquet(aligned_path, index=False)
            metrics = compute_frequency_component_metrics(aligned)
            metrics.insert(0, "split", split_name)
            metrics.insert(0, "model", model_label)
            metrics.insert(0, "model_bundle", str(bundle_path))
            rows.append(metrics)

    summary = pd.concat(rows, ignore_index=True)
    summary_csv = output_dir / "frequency_resolved_backbone_summary.csv"
    mean_csv = output_dir / "frequency_resolved_backbone_mean_summary.csv"
    config_path = output_dir / "frequency_resolved_backbone_config.json"
    summary.to_csv(summary_csv, index=False)
    mean_summary = summary.loc[summary["target"] == "__mean__"].copy()
    mean_summary.to_csv(mean_csv, index=False)
    config_path.write_text(
        json.dumps(
            {
                "split_root": args.split_root,
                "models": {label: str(path) for label, path in model_specs},
                "split_names": list(args.split_names),
                "batch_size": int(args.batch_size),
                "device": str(args.device),
                "components": ["low_0_1hz", "mid_1_3hz", "flap_main", "high_frequency_residual"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(mean_summary[["model", "split", "component", "rmse", "r2"]].to_string(index=False))
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
