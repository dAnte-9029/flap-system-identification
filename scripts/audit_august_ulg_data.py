#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.data.ulg_audit import build_audit_summary, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit one calendar month of PX4 ULog data for trajectory modeling")
    parser.add_argument("--source-root", type=Path, default=Path("/home/zn/QgcLogs"))
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--month", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "docs/audits/results/2026-09-02_august_ulg_audit_summary.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=PROJECT_ROOT / "docs/audits/2026-09-02_august_ulg_data_audit.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    summary = build_audit_summary(
        args.source_root,
        year=args.year,
        month=args.month,
        workers=args.workers,
        audit_repository=PROJECT_ROOT,
    )
    json_path = write_json(summary, args.output_json)
    report_path = write_markdown(summary, args.output_report)
    print(f"summary_json: {json_path}")
    print(f"report_markdown: {report_path}")
    print(f"counts: {summary['counts']}")
    print(f"durations_s: {summary['durations_s']}")


if __name__ == "__main__":
    main()
