#!/usr/bin/env python3
"""Thin CLI for the active July-14 attitude-aware DeLaurier prior."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from system_identification.artifacts.prior_registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    load_prior_registry,
)
from system_identification.physics.priors import (  # noqa: E402
    AUTHORITATIVE_PRIOR_ID,
    materialize_authoritative_delaurier_prior,
)


DEFAULT_DATASET = Path("dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1")
DEFAULT_METADATA = Path("metadata/aircraft/flapper_01/aircraft_metadata.yaml")
DEFAULT_GEOMETRY = Path("metadata/aircraft/flapper_01/wing_geometry_isaaclab_3b5d4ec.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--prior-registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--prior-id", default=AUTHORITATIVE_PRIOR_ID)
    parser.add_argument("--aircraft-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--wing-geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--partitions", nargs="+", default=("train", "validation"))
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--physics-repository-root", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    registry_path, registry = load_prior_registry(args.prior_registry)
    entries = registry["priors"]
    entry = entries.get(args.prior_id)
    if entry is None and args.output_root is None:
        raise ValueError(
            f"Unknown prior ID {args.prior_id!r} in {registry_path}; "
            "an explicit --output-root is required before registry promotion"
        )
    if entry is not None and entry.get("lifecycle_status") != "active":
        raise ValueError(f"Materializer only writes active priors, got {entry.get('lifecycle_status')!r}")
    output_root = args.output_root or Path(str(entry["artifact_root"]))
    manifest = materialize_authoritative_delaurier_prior(
        dataset_root=args.dataset_root,
        output_root=output_root,
        metadata_path=args.aircraft_metadata,
        geometry_path=args.wing_geometry,
        partitions=args.partitions,
        chunk_size=args.chunk_size,
        project_root=PROJECT_ROOT,
        artifact_id=args.prior_id,
        physics_repository_root=args.physics_repository_root,
    )
    print(f"artifact_id: {manifest['artifact_id']}")
    print(f"output_root: {Path(output_root).resolve()}")
    print(f"partitions: {manifest['partitions']}")
    print(f"row_counts: {manifest['row_counts']}")
    print(f"physics_commit: {manifest['physics_source']['commit']}")
    print(f"airflow_contract: {manifest['contracts']['airflow_contract']}")
    print("test_partition_loaded: false")
    print("execution_backend: numpy_cpu_vectorized (validated frozen implementation; CUDA rewrite not used)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
