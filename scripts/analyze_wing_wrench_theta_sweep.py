#!/usr/bin/env python3
"""Run real-log wing-only DeLaurier six-axis dynamic-twist sensitivity."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.analysis.wing_wrench_theta_sweep import (
    load_canonical_samples,
    run_wing_wrench_theta_sweep,
)
from system_identification.baselines import AIRFLOW_MODES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a wing-only DeLaurier baseline with total reconstructed effective-wrench labels."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--samples-parquet", type=Path)
    source.add_argument("--split-root", type=Path)
    parser.add_argument("--partition", default="test")
    parser.add_argument("--aircraft-metadata", type=Path, required=True)
    parser.add_argument(
        "--geometry-csv",
        type=Path,
        default=PROJECT_ROOT / "metadata/aircraft/flapper_01/wing_geometry_isaaclab_3b5d4ec.csv",
    )
    parser.add_argument(
        "--legacy-comparison-aligned",
        type=Path,
        help="Optional legacy aligned_predictions.parquet for direct airflow-mode comparison figures.",
    )
    parser.add_argument("--theta-tip-deg", type=float, nargs="+", default=[0.0, 5.0, 10.0, 15.0])
    parser.add_argument("--window-manifest", type=Path)
    parser.add_argument("--auto-window-count", type=int, default=5)
    parser.add_argument("--phase-bins", type=int, default=72)
    parser.add_argument(
        "--airflow-mode",
        choices=sorted(AIRFLOW_MODES),
        default="attitude_ground_wind_3d",
        help="Explicit real-log airflow reconstruction contract.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    samples, samples_path = load_canonical_samples(
        samples_parquet=args.samples_parquet,
        split_root=args.split_root,
        partition=args.partition,
    )
    command = " ".join(shlex.quote(value) for value in sys.argv)
    run_wing_wrench_theta_sweep(
        samples=samples,
        samples_path=samples_path,
        aircraft_metadata_path=args.aircraft_metadata,
        geometry_path=args.geometry_csv,
        theta_tip_deg=args.theta_tip_deg,
        output_dir=args.output_dir,
        window_manifest_path=args.window_manifest,
        phase_bins=args.phase_bins,
        auto_window_count=args.auto_window_count,
        command=command,
        airflow_mode=args.airflow_mode,
        legacy_comparison_aligned_path=args.legacy_comparison_aligned,
    )
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
