#!/usr/bin/env python3
"""Thin CLI for the fixed train-only C2 static-family smoke run."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from system_identification.artifacts.static_correction_data import load_static_correction_training_data
from system_identification.models.correction.specifications import parse_model_family_config
from system_identification.training.correction.smoke import run_static_correction_smoke


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction-ready-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--partition", default="train")
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.partition != "train":
        raise ValueError("C2 fitting is train-only; validation and test requests are forbidden")
    config_path = Path(args.config)
    config_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config_value, dict):
        raise ValueError(f"Expected YAML mapping in {config_path}")
    config = parse_model_family_config(config_value)
    project_root = Path(__file__).resolve().parents[1]
    data = load_static_correction_training_data(
        args.correction_ready_root,
        authority=config.authority,
        partition=args.partition,
        project_root=project_root,
    )
    summary = run_static_correction_smoke(data, config, args.output_root, project_root=project_root)
    print(
        "C2 train-only smoke complete: "
        f"candidates={summary['candidate_count']} output={Path(args.output_root).resolve()} "
        "selection_performed=false validation_labels_loaded=false test_labels_loaded=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
