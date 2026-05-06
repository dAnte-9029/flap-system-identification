#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.dataset_split import materialize_log_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a training-ready whole-log split from canonical parquet datasets")
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Input dataset_manifest.json path. Repeat for multiple packaged cohorts.",
    )
    parser.add_argument("--output", required=True, help="Output directory for the merged split dataset")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for log shuffling")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument(
        "--altitude-window-min-m",
        type=float,
        default=None,
        help="If set, keep each log from the first to last sample with -vehicle_local_position.z above this altitude",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = materialize_log_split(
        manifest_paths=args.manifest,
        output_root=args.output,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        altitude_window_min_m=args.altitude_window_min_m,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
