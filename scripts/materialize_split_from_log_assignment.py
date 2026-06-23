#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.dataset_split import _apply_altitude_window_trim, _valid_row_mask


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _load_manifest_records(source_manifest: str | Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = _resolve_path(source_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    accepted_path = _resolve_path(manifest["accepted_logs_json"])
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    records = {str(item["log_id"]): str(item["samples_path"]) for item in accepted}
    return manifest, records


def materialize_split_from_assignment(
    *,
    source_manifest: str | Path,
    assignment_csv: str | Path,
    output_root: str | Path,
    altitude_window_min_m: float | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    manifest, log_to_samples = _load_manifest_records(source_manifest)
    assignment_path = _resolve_path(assignment_csv)
    output = _resolve_path(output_root)

    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output root already exists and is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    assignments = pd.read_csv(assignment_path)
    required = {"log_id", "split"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"{assignment_path} missing required columns: {sorted(missing)}")

    split_frames: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    output_rows: list[dict[str, Any]] = []

    for row in assignments.itertuples(index=False):
        log_id = str(row.log_id)
        split = str(row.split)
        if split not in split_frames:
            continue
        if log_id not in log_to_samples:
            raise KeyError(f"log_id {log_id!r} from assignment CSV not found in source manifest")

        samples_path = log_to_samples[log_id]
        samples = pd.read_parquet(samples_path)
        valid_samples = samples.loc[_valid_row_mask(samples)].copy()
        valid_samples = _apply_altitude_window_trim(valid_samples, altitude_window_min_m)
        valid_samples["dataset_id"] = output.name
        valid_samples["log_id"] = log_id
        valid_samples["source_samples_path"] = samples_path
        valid_samples["split"] = split
        if not valid_samples.empty:
            split_frames[split].append(valid_samples)

        output_row = row._asdict()
        output_row["dataset_id"] = output.name
        output_row["manifest_path"] = str(_resolve_path(source_manifest))
        output_row["source_samples_path"] = samples_path
        output_row["valid_sample_count"] = int(len(valid_samples))
        output_rows.append(output_row)

    split_tables: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        split_tables[split] = pd.concat(split_frames[split], ignore_index=True) if split_frames[split] else pd.DataFrame()
        split_tables[split].to_parquet(output / f"{split}_samples.parquet", index=False)

    all_logs = pd.DataFrame(output_rows).sort_values(["split", "dataset_id", "log_id"])
    all_logs.to_csv(output / "all_logs.csv", index=False)
    for split in ("train", "val", "test"):
        all_logs.loc[all_logs["split"] == split].to_csv(output / f"{split}_logs.csv", index=False)

    output_manifest = {
        "altitude_window_min_m": altitude_window_min_m,
        "assignment_csv": str(assignment_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": output.name,
        "input_log_count": int(len(all_logs)),
        "output_root": str(output),
        "source_dataset_id": manifest.get("dataset_id"),
        "source_manifest_path": str(_resolve_path(source_manifest)),
        "split_log_counts": {
            split: int((all_logs["split"] == split).sum())
            for split in ("train", "val", "test")
        },
        "split_policy": "whole_log_reused_assignment",
        "split_sample_counts": {
            split: int(len(split_tables[split]))
            for split in ("train", "val", "test")
        },
        "all_logs_csv": str(output / "all_logs.csv"),
        "train_logs_csv": str(output / "train_logs.csv"),
        "val_logs_csv": str(output / "val_logs.csv"),
        "test_logs_csv": str(output / "test_logs.csv"),
        "train_samples_parquet": str(output / "train_samples.parquet"),
        "val_samples_parquet": str(output / "val_samples.parquet"),
        "test_samples_parquet": str(output / "test_samples.parquet"),
    }
    manifest_path = output / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(output_manifest, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "dataset_manifest_path": manifest_path,
        "all_logs_path": output / "all_logs.csv",
        "train_samples_path": output / "train_samples.parquet",
        "val_samples_path": output / "val_samples.parquet",
        "test_samples_path": output / "test_samples.parquet",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a whole-log split using an existing log assignment CSV")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--assignment-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--altitude-window-min-m", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = materialize_split_from_assignment(
        source_manifest=args.source_manifest,
        assignment_csv=args.assignment_csv,
        output_root=args.output,
        altitude_window_min_m=args.altitude_window_min_m,
        overwrite=args.overwrite,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
