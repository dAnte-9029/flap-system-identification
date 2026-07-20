#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.data.hall_phase import build_hall_ratio8_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild one Hall-indexed ratio-8 phase and corrected frequency.")
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--accepted-logs-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--partitions", nargs="+", default=["train", "validation"])
    parser.add_argument("--logged-ratio", type=float, default=7.5)
    parser.add_argument("--true-ratio", type=float, default=8.0)
    parser.add_argument("--counts-per-encoder-revolution", type=float, default=4096.0)
    parser.add_argument("--maximum-cycle-count-relative-error", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        outputs = build_hall_ratio8_dataset(
            source_dataset_root=args.source_dataset_root,
            accepted_logs_csv=args.accepted_logs_csv,
            output_root=args.output_root,
            partitions=args.partitions,
            logged_ratio=args.logged_ratio,
            true_ratio=args.true_ratio,
            counts_per_encoder_revolution=args.counts_per_encoder_revolution,
            maximum_cycle_count_relative_error=args.maximum_cycle_count_relative_error,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
