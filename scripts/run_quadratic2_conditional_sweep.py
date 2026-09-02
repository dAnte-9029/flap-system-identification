#!/usr/bin/env python3
"""Run the directed train/validation-only Quadratic-2 conditional sweep."""

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

from system_identification.analysis.quadratic2_conditional_sweep import (  # noqa: E402
    conditional_dry_run_summary,
    run_conditional,
    run_conditional_report,
    seed_existing_results,
)
from system_identification.analysis.quadratic2_twist_sweep import (  # noqa: E402
    resolve_experiment,
    run_baseline,
)


DEFAULT_CONFIG = Path("configs/delaurier/quadratic2_conditional_sweep_v2.yaml")
STAGES = ("baseline", "seed", "conditional", "report")


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
    if args.dry_run:
        print(json.dumps(conditional_dry_run_summary(resolved), indent=2, sort_keys=True))
        return 0
    workers = (
        int(args.workers)
        if args.workers is not None
        else int(resolved.config["runtime"]["default_workers"])
    )
    resolved.output_root.mkdir(parents=True, exist_ok=True)
    with (resolved.output_root / "run_commands.txt").open("a", encoding="utf-8") as stream:
        stream.write(" ".join(shlex.quote(value) for value in sys.argv) + "\n")
    if args.stage == "baseline":
        result = run_baseline(resolved)
    elif args.stage == "seed":
        result = seed_existing_results(resolved)
    elif args.stage == "conditional":
        result = {"metric_rows": len(run_conditional(
            resolved, workers=workers, resume=args.resume
        ))}
    else:
        manifest = run_conditional_report(resolved)
        result = {
            "boundary_diagnostic_decision": manifest["boundary_diagnostic_decision"],
            "unique_candidates": manifest["grid"]["unique_candidates"],
        }
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print(f"output_root: {resolved.output_root}")
    print("test_partition_loaded: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
