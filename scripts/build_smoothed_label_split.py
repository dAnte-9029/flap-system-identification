#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.metadata import load_aircraft_metadata  # noqa: E402
from system_identification.pipeline import (  # noqa: E402
    _compute_effective_wrench_labels,
    compute_smoothed_kinematic_derivatives,
)


TARGET_COLUMNS = ["fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b"]
LINEAR_ACCELERATION_COLUMNS = [
    "vehicle_local_position.ax_smooth",
    "vehicle_local_position.ay_smooth",
    "vehicle_local_position.az_smooth",
]
ANGULAR_ACCELERATION_COLUMNS = [
    "vehicle_angular_velocity.xyz_derivative_smooth[0]",
    "vehicle_angular_velocity.xyz_derivative_smooth[1]",
    "vehicle_angular_velocity.xyz_derivative_smooth[2]",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite a split with smoothed-derivative effective-wrench labels")
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window-s", type=float, default=0.12)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--force-label-source", choices=["raw", "smooth"], default="smooth")
    parser.add_argument("--moment-label-source", choices=["raw", "smooth"], default="smooth")
    return parser.parse_args()


def _rewrite_split_frame(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    window_s: float,
    polyorder: int,
    force_label_source: str,
    moment_label_source: str,
) -> pd.DataFrame:
    rewritten = frame.copy()
    derivatives = compute_smoothed_kinematic_derivatives(
        rewritten,
        window_s=window_s,
        polyorder=polyorder,
    )
    for column in derivatives.columns:
        rewritten[column] = derivatives[column].to_numpy()

    force_b, moment_b, label_valid = _compute_effective_wrench_labels(
        rewritten,
        metadata,
        linear_acceleration_columns=LINEAR_ACCELERATION_COLUMNS if force_label_source == "smooth" else None,
        angular_acceleration_columns=ANGULAR_ACCELERATION_COLUMNS if moment_label_source == "smooth" else None,
    )
    rewritten["fx_b"] = force_b[:, 0]
    rewritten["fy_b"] = force_b[:, 1]
    rewritten["fz_b"] = force_b[:, 2]
    rewritten["mx_b"] = moment_b[:, 0]
    rewritten["my_b"] = moment_b[:, 1]
    rewritten["mz_b"] = moment_b[:, 2]
    rewritten["label_valid"] = label_valid
    return rewritten


def build_smoothed_label_split(
    *,
    split_root: str | Path,
    metadata_path: str | Path,
    output_root: str | Path,
    window_s: float = 0.12,
    polyorder: int = 2,
    force_label_source: str = "smooth",
    moment_label_source: str = "smooth",
) -> dict[str, str]:
    if force_label_source not in {"raw", "smooth"}:
        raise ValueError(f"Unknown force_label_source: {force_label_source}")
    if moment_label_source not in {"raw", "smooth"}:
        raise ValueError(f"Unknown moment_label_source: {moment_label_source}")

    split_root = Path(split_root)
    metadata_path = Path(metadata_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = load_aircraft_metadata(metadata_path)

    split_sample_counts: dict[str, int] = {}
    label_valid_ratios: dict[str, float] = {}
    for split_name in ["train", "val", "test"]:
        input_path = split_root / f"{split_name}_samples.parquet"
        output_path = output_root / f"{split_name}_samples.parquet"
        frame = pd.read_parquet(input_path)
        rewritten = _rewrite_split_frame(
            frame,
            metadata,
            window_s=window_s,
            polyorder=polyorder,
            force_label_source=force_label_source,
            moment_label_source=moment_label_source,
        )
        rewritten.to_parquet(output_path, index=False)
        split_sample_counts[split_name] = int(len(rewritten))
        label_valid_ratios[split_name] = float(rewritten["label_valid"].mean())

    for filename in ["train_logs.csv", "val_logs.csv", "test_logs.csv", "all_logs.csv"]:
        source = split_root / filename
        if source.exists():
            shutil.copy2(source, output_root / filename)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": output_root.name,
        "output_root": str(output_root),
        "source_split_root": str(split_root),
        "metadata_path": str(metadata_path),
        "label_policy": "effective wrench recomputed from smoothed velocity and angular-velocity derivatives",
        "smoothing": {
            "window_s": float(window_s),
            "polyorder": int(polyorder),
            "force_label_source": force_label_source,
            "moment_label_source": moment_label_source,
            "linear_acceleration_columns": LINEAR_ACCELERATION_COLUMNS,
            "angular_acceleration_columns": ANGULAR_ACCELERATION_COLUMNS,
        },
        "split_sample_counts": split_sample_counts,
        "label_valid_ratios": label_valid_ratios,
    }
    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
    }


def main() -> None:
    args = parse_args()
    outputs = build_smoothed_label_split(
        split_root=args.split_root,
        metadata_path=args.metadata,
        output_root=args.output,
        window_s=args.window_s,
        polyorder=args.polyorder,
        force_label_source=args.force_label_source,
        moment_label_source=args.moment_label_source,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
