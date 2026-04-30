#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.dataset_split import materialize_cycle_block_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a training-ready cycle-block split from canonical parquet datasets")
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Input dataset_manifest.json path. Repeat for multiple packaged cohorts.",
    )
    parser.add_argument("--output", required=True, help="Output directory for the merged split dataset")
    parser.add_argument("--block-size-cycles", type=int, default=60, help="Number of consecutive cycles per shuffled block")
    parser.add_argument("--purge-cycles", type=int, default=8, help="Cycles to drop from train around val/test blocks")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for block shuffling")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = materialize_cycle_block_split(
        manifest_paths=args.manifest,
        output_root=args.output,
        block_size_cycles=args.block_size_cycles,
        purge_cycles=args.purge_cycles,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
