#!/usr/bin/env python3
"""Thin CLI for the C0/C1 longitudinal correction-ready artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

from system_identification.artifacts.correction_ready import (
    build_longitudinal_correction_ready_artifact,
)
from system_identification.artifacts.prior_registry import resolve_delaurier_prior
from system_identification.data.correction_ready import (
    CorrectionReadyConfig,
    normalize_correction_partitions,
)


DEFAULT_CONFIG = Path("configs/correction/longitudinal_force_correction_v1.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build C1 longitudinal correction-ready tables without training a model."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument(
        "--prior-registry", default="configs/physics/delaurier_prior_registry.yaml"
    )
    parser.add_argument("--prior-id")
    parser.add_argument("--partitions", nargs="+", default=["train", "validation"])
    parser.add_argument("--output-root", default="artifacts/correction_ready")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--phase-bins", type=int)
    parser.add_argument("--minimum-cycle-samples", type=int)
    parser.add_argument("--minimum-phase-coverage", type=float)
    parser.add_argument("--maximum-phase-gap", type=float)
    parser.add_argument("--harmonic-max-order", type=int)
    parser.add_argument("--condition-aggregation")
    parser.add_argument("--output-format", choices=("parquet", "csv", "both"), default="parquet")
    parser.add_argument("--random-seed", type=int)
    return parser


def _config(args: argparse.Namespace) -> CorrectionReadyConfig:
    path = Path(args.config)
    mapping = yaml.safe_load(path.read_text(encoding="utf-8"))
    if mapping.get("schema_version") != "longitudinal_force_correction_contract_v1":
        raise ValueError(f"Unsupported correction config schema in {path}")
    cycle = mapping["cycle"]
    decomposition = mapping["decomposition"]
    conditions = mapping["conditions"]
    normalization = mapping["normalization"]
    alignment = mapping["alignment"]
    config = CorrectionReadyConfig(
        phase_bins=args.phase_bins or int(cycle["phase_bins"]),
        minimum_cycle_samples=args.minimum_cycle_samples or int(cycle["minimum_cycle_samples"]),
        minimum_phase_coverage_rad=(
            args.minimum_phase_coverage
            if args.minimum_phase_coverage is not None
            else float(cycle["minimum_phase_coverage_rad"])
        ),
        maximum_phase_gap_rad=(
            args.maximum_phase_gap
            if args.maximum_phase_gap is not None
            else float(cycle["maximum_phase_gap_rad"])
        ),
        harmonic_max_order=args.harmonic_max_order or int(decomposition["harmonic_max_order"]),
        condition_aggregation=args.condition_aggregation or str(conditions["aggregation"]),
        reconstruction_tolerance=float(decomposition["reconstruction_tolerance"]),
        zero_mean_tolerance=float(decomposition["zero_mean_tolerance"]),
        normalization_std_epsilon=float(normalization["standard_deviation_epsilon"]),
        minimum_accepted_cycle_fraction=float(cycle["minimum_accepted_cycle_fraction"]),
        maximum_missing_alignment_fraction=float(alignment["maximum_missing_fraction"]),
        random_seed=args.random_seed or int(mapping["random_seed"]),
    )
    config.validate()
    return config


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        partitions = normalize_correction_partitions(args.partitions)
        config = _config(args)
        prior = resolve_delaurier_prior(
            prior_id=args.prior_id,
            registry_path=args.prior_registry,
            requested_partitions=partitions,
        )
        command = [sys.executable, str(Path(__file__)), *(argv if argv is not None else sys.argv[1:])]
        result = build_longitudinal_correction_ready_artifact(
            dataset_root=args.dataset_root,
            split_manifest=args.split_manifest,
            prior=prior,
            partitions=partitions,
            output_root=args.output_root,
            config=config,
            output_format=args.output_format,
            command=command,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if result.exit_code == 2:
        print(
            f"STRICT QUALITY FAILURE: {result.quality_checks['strict_failures']}",
            file=sys.stderr,
        )
        print(f"Artifact: {result.output_dir}", file=sys.stderr)
        return 2
    print(f"Artifact: {result.output_dir}")
    print(
        "Cycles: "
        f"accepted={result.dataset_summary['accepted_cycle_count']} "
        f"rejected={result.dataset_summary['rejected_cycle_count']}"
    )
    print("Quality checks: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
