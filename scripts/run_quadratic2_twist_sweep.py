#!/usr/bin/env python3
"""Run the train/validation-only Quadratic-2 dynamic-twist experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from system_identification.analysis.quadratic2_twist_sweep import (  # noqa: E402
    dry_run_summary,
    resolve_experiment,
    run_baseline,
    run_coarse,
    run_oat,
    run_refine,
    run_report,
)


DEFAULT_CONFIG = Path("configs/delaurier/quadratic2_twist_sweep_v1.yaml")
STAGES = ("baseline", "oat", "coarse", "refine", "report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    resolved = resolve_experiment(args.config, project_root=PROJECT_ROOT)
    summary = dry_run_summary(resolved)
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    workers = (
        int(args.workers)
        if args.workers is not None
        else int(resolved.config["runtime"]["default_workers"])
    )
    command_path = resolved.output_root / "run_commands.txt"
    resolved.output_root.mkdir(parents=True, exist_ok=True)
    with command_path.open("a", encoding="utf-8") as stream:
        stream.write(" ".join(shlex.quote(value) for value in sys.argv) + "\n")
    if args.stage == "baseline":
        result = run_baseline(resolved)
        print(json.dumps({
            "baseline_gate_passed": result["baseline_gate_passed"],
            "baseline_full_cycle_fx_peak_deg": result["baseline_full_cycle_fx_peak_deg"],
        }, indent=2, sort_keys=True))
    elif args.stage == "oat":
        result = run_oat(resolved, workers=workers, resume=args.resume)
        print(f"oat_metric_rows: {len(result)}")
    elif args.stage == "coarse":
        result = run_coarse(resolved, workers=workers, resume=args.resume)
        print(f"coarse_metric_rows: {len(result)}")
    elif args.stage == "refine":
        result = run_refine(resolved, workers=workers, resume=args.resume)
        print(f"refined_metric_rows: {len(result)}")
    else:
        result = run_report(resolved)
        print(json.dumps({
            "conclusion_code": result["conclusion_code"],
            "conclusion_text": result["conclusion_text"],
        }, indent=2, sort_keys=True))
    print(f"output_root: {resolved.output_root}")
    print("test_partition_loaded: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
