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

from system_identification.data.trajectory_dataset import build_trajectory_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen Step 1 trajectory dataset")
    parser.add_argument(
        "--audit-summary",
        type=Path,
        default=PROJECT_ROOT / "docs/audits/results/2026-09-02_august_ulg_audit_summary.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "dataset/trajectory_v1_august_f5_c4",
    )
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=("train", "validation", "sealed_test"),
        default=("train", "validation"),
        help="Sealed test is excluded by default and must be requested explicitly.",
    )
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--maximum-gap-s", type=float, default=0.05)
    parser.add_argument("--horizon-s", type=float, default=2.0)
    parser.add_argument("--stride-s", type=float, default=0.2)
    parser.add_argument("--summary-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_trajectory_dataset(
        audit_summary_path=args.audit_summary,
        output_root=args.output_root,
        partitions=args.partitions,
        expected_rate_hz=args.rate_hz,
        maximum_gap_s=args.maximum_gap_s,
        horizon_s=args.horizon_s,
        stride_s=args.stride_s,
        repository_root=PROJECT_ROOT,
    )
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "dataset_version": manifest["dataset_version"],
            "generated_at": manifest["generated_at"],
            "builder_git_head": manifest["builder_git_head"],
            "source": manifest["source"],
            "split_contract": {
                "split_dates": manifest["split_contract"]["split_dates"],
                "materialized_partitions": manifest["split_contract"]["materialized_partitions"],
                "sealed_test_opened": manifest["split_contract"]["sealed_test_opened"],
            },
            "sampling": manifest["sampling"],
            "partitions": manifest["partitions"],
        }
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"output_root: {args.output_root.resolve()}")
    print(json.dumps(manifest["partitions"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
