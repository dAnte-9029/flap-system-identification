#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.pipeline import run_ulog_to_canonical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PX4 ULog to canonical parquet dataset")
    parser.add_argument("--ulg", required=True, help="Input PX4 ULog path")
    parser.add_argument("--metadata", required=True, help="Aircraft metadata YAML path")
    parser.add_argument("--output", required=True, help="Output dataset root directory")
    parser.add_argument("--rate-hz", type=float, default=100.0, help="Canonical sample rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_ulog_to_canonical(
        ulg_path=args.ulg,
        metadata_path=args.metadata,
        output_root=args.output,
        rate_hz=args.rate_hz,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
