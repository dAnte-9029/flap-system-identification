#!/usr/bin/env python3
"""Build a residual-target split from effective-wrench labels and DeLaurier prior predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TARGET_COLUMNS = ["fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b"]
SPLITS = ("train", "val", "test")


def _check_target_columns(frame: pd.DataFrame, *, label: str) -> None:
    missing = [column for column in TARGET_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing target columns: {missing}")


def build_residual_frame(samples: pd.DataFrame, prior: pd.DataFrame, *, prior_name: str) -> pd.DataFrame:
    """Return a copy of ``samples`` whose target columns are residual targets."""

    if len(samples) != len(prior):
        raise ValueError(f"samples/prior row mismatch: {len(samples)} != {len(prior)}")
    _check_target_columns(samples, label="samples")
    _check_target_columns(prior, label="prior")

    residual = samples.copy()
    for column in TARGET_COLUMNS:
        true_values = samples[column].astype(float).to_numpy()
        prior_values = prior[column].astype(float).to_numpy()
        residual[f"label_{column}"] = true_values
        residual[f"true_{column}"] = true_values
        residual[f"prior_{column}"] = prior_values
        residual[column] = true_values - prior_values
    residual.attrs["residual_prior_name"] = str(prior_name)
    return residual


def build_residual_split(split_root: Path, prior_root: Path, output_root: Path, *, prior_name: str) -> dict[str, object]:
    """Build train/val/test residual parquet files."""

    output_root.mkdir(parents=True, exist_ok=True)
    row_counts: dict[str, int] = {}
    for split in SPLITS:
        samples_path = split_root / f"{split}_samples.parquet"
        prior_path = prior_root / f"{split}_predictions.parquet"
        samples = pd.read_parquet(samples_path)
        prior = pd.read_parquet(prior_path)
        residual = build_residual_frame(samples, prior, prior_name=prior_name)
        residual.to_parquet(output_root / f"{split}_samples.parquet", index=False)
        row_counts[split] = int(len(residual))

    manifest = {
        "source_split_root": str(split_root),
        "prior_root": str(prior_root),
        "output_root": str(output_root),
        "prior_name": str(prior_name),
        "target_columns": TARGET_COLUMNS,
        "target_semantics": "residual = true_effective_wrench - prior_effective_wrench",
        "row_counts": row_counts,
    }
    (output_root / "residual_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--prior-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--prior-name", default="delaurier_physical")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_residual_split(args.split_root, args.prior_root, args.output_root, prior_name=args.prior_name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
