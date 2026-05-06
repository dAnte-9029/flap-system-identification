#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.training import DEFAULT_REGIME_BIN_SPECS, run_diagnostic_evaluation


def _parse_split_names(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("split names must contain at least one value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-log and per-regime diagnostics for a trained model bundle")
    parser.add_argument("--model-bundle", required=True, help="Path to model_bundle.pt")
    parser.add_argument("--split-root", required=True, help="Dataset split root containing train/val/test parquet files")
    parser.add_argument("--output-dir", required=True, help="Output directory for diagnostic CSV files")
    parser.add_argument("--splits", type=_parse_split_names, default=("test",), help="Comma-separated splits to evaluate")
    parser.add_argument("--min-samples", type=int, default=16, help="Minimum samples required for a log/bin metric row")
    parser.add_argument("--batch-size", type=int, default=8192, help="Evaluation batch size")
    parser.add_argument("--device", default="auto", help="Evaluation device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--list-default-bins",
        action="store_true",
        help="Print the built-in binned-evaluation specs before running",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_default_bins:
        for column, edges in DEFAULT_REGIME_BIN_SPECS.items():
            print(f"{column}: {edges}")

    outputs = run_diagnostic_evaluation(
        model_bundle_path=args.model_bundle,
        split_root=args.split_root,
        output_dir=args.output_dir,
        split_names=args.splits,
        min_samples=args.min_samples,
        batch_size=args.batch_size,
        device=args.device,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
