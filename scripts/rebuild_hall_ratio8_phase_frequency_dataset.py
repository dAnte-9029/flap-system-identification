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
    parser.add_argument(
        "--aircraft-metadata",
        type=Path,
        default=Path("metadata/aircraft/flapper_01/aircraft_metadata.yaml"),
    )
    parser.add_argument("--partitions", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--maximum-cycle-count-relative-error", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        outputs = build_hall_ratio8_dataset(
            source_dataset_root=args.source_dataset_root,
            accepted_logs_csv=args.accepted_logs_csv,
            output_root=args.output_root,
            aircraft_metadata=args.aircraft_metadata,
            partitions=args.partitions,
            maximum_cycle_count_relative_error=args.maximum_cycle_count_relative_error,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
