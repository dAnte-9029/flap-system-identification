#!/usr/bin/env python3
"""Evaluate only sealed C3 finalists on validation and freeze selected train bundles."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from system_identification.artifacts.static_correction_selection_data import (
    load_static_correction_validation_data,
)
from system_identification.evaluation.static_correction_validation import run_validation_selection
from system_identification.training.correction.selection_rules import verify_sealed_shortlist
from system_identification.training.correction.selection_specs import parse_selection_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction-ready-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--shortlist", required=True)
    parser.add_argument("--partition", default="validation")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--selected-bundle-root", required=True)
    return parser


def _git(project_root: Path) -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--short"], cwd=project_root, text=True).strip())
    return commit, dirty


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.partition != "validation":
        raise ValueError("C3 Stage B only accepts partition='validation'; test is forbidden")
    project_root = Path(__file__).resolve().parents[1]
    config_value = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    shortlist_value = json.loads(Path(args.shortlist).read_text(encoding="utf-8"))
    if not isinstance(config_value, dict) or not isinstance(shortlist_value, dict):
        raise ValueError("C3 config and sealed shortlist must be mappings")
    config = parse_selection_config(config_value)
    verify_sealed_shortlist(
        shortlist_value,
        expected_config_hash=config.config_hash,
        expected_artifact_hash=str(config.authority["correction_ready_manifest_sha256"]),
    )
    expected_id = str(config.input["correction_ready_artifact_id"])
    if Path(args.correction_ready_root).name != expected_id:
        raise ValueError("Correction-ready artifact ID differs from frozen C3 config")
    commit, dirty = _git(project_root)
    data = load_static_correction_validation_data(
        args.correction_ready_root,
        authority=config.authority,
        project_root=project_root,
        partition="validation",
    )
    command = " ".join(shlex.quote(item) for item in [sys.executable, __file__, *(argv or sys.argv[1:])])
    summary = run_validation_selection(
        data,
        config,
        shortlist_value,
        args.output_root,
        args.selected_bundle_root,
        git_commit=commit,
        git_dirty=dirty,
        run_command=command,
    )
    print(
        "C3 Stage B complete: "
        f"selected={summary['selected']} quality={summary['quality_status']} "
        "test_labels_loaded=false dynamic_model_trained=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
