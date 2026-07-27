#!/usr/bin/env python3
"""Build and seal the C3 train-only grouped-CV finalist shortlist."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from system_identification.artifacts.static_correction_data import load_static_correction_training_data
from system_identification.training.correction.selection_specs import parse_selection_config
from system_identification.training.correction.static_selection import run_train_cv_selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction-ready-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--partition", default="train")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=6)
    return parser


def _git(project_root: Path) -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--short"], cwd=project_root, text=True).strip())
    return commit, dirty


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.partition != "train":
        raise ValueError("C3 Stage A only accepts partition='train'; validation and test are forbidden")
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("C3 selection config must be a mapping")
    config = parse_selection_config(value)
    expected_id = str(config.input["correction_ready_artifact_id"])
    if Path(args.correction_ready_root).name != expected_id:
        raise ValueError("Correction-ready artifact ID differs from frozen C3 config")
    commit, dirty = _git(project_root)
    data = load_static_correction_training_data(
        args.correction_ready_root,
        authority=config.authority,
        partition="train",
        project_root=project_root,
    )
    command = " ".join(shlex.quote(item) for item in [sys.executable, __file__, *(argv or sys.argv[1:])])
    summary = run_train_cv_selection(
        data,
        config,
        args.output_root,
        git_commit=commit,
        git_dirty=dirty,
        config_path=config_path,
        run_command=command,
        workers=args.workers,
    )
    print(
        "C3 Stage A complete: "
        f"mean={summary['mean_candidate_count']} waveform={summary['waveform_candidate_count']} "
        f"complete={summary['complete_candidate_count']} shortlist={summary['shortlist_hash']} "
        "validation_labels_loaded=false test_labels_loaded=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
