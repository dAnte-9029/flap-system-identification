#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _filter_frame_by_log_id(frame: pd.DataFrame, exclude_log_ids: set[str]) -> tuple[pd.DataFrame, int]:
    if "log_id" not in frame.columns:
        raise ValueError("split frame must contain log_id")
    mask = ~frame["log_id"].astype(str).isin(exclude_log_ids)
    return frame.loc[mask].reset_index(drop=True), int((~mask).sum())


def _filter_csv_if_exists(input_path: Path, output_path: Path, exclude_log_ids: set[str]) -> int:
    if not input_path.exists():
        return 0
    frame = pd.read_csv(input_path)
    if "log_id" not in frame.columns:
        frame.to_csv(output_path, index=False)
        return 0
    filtered, removed = _filter_frame_by_log_id(frame, exclude_log_ids)
    filtered.to_csv(output_path, index=False)
    return removed


def filter_split_logs(
    *,
    input_root: str | Path,
    output_root: str | Path,
    exclude_log_ids: list[str],
    reason: str,
    overwrite: bool = False,
) -> dict[str, str]:
    input_path = Path(input_root)
    output_path = Path(output_root)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(f"output_root already exists and is not empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    exclude_set = {str(value) for value in exclude_log_ids}
    if not exclude_set:
        raise ValueError("exclude_log_ids must not be empty")

    removed_sample_counts: dict[str, int] = {}
    kept_sample_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        split_input = input_path / f"{split}_samples.parquet"
        if not split_input.exists():
            raise FileNotFoundError(split_input)
        frame = pd.read_parquet(split_input)
        filtered, removed = _filter_frame_by_log_id(frame, exclude_set)
        filtered.to_parquet(output_path / f"{split}_samples.parquet", index=False)
        removed_sample_counts[split] = removed
        kept_sample_counts[split] = int(len(filtered))
        _filter_csv_if_exists(input_path / f"{split}_logs.csv", output_path / f"{split}_logs.csv", exclude_set)

    _filter_csv_if_exists(input_path / "all_logs.csv", output_path / "all_logs.csv", exclude_set)

    manifest = _read_manifest(input_path / "dataset_manifest.json")
    manifest.update(
        {
            "filtered_from_split_root": str(input_path),
            "excluded_log_ids": sorted(exclude_set),
            "excluded_log_reason": reason,
            "removed_sample_counts_by_split": removed_sample_counts,
            "kept_sample_counts_by_split": kept_sample_counts,
            "split_policy": manifest.get("split_policy", "whole_log"),
        }
    )
    manifest_path = output_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "dataset_manifest_path": str(manifest_path),
        "output_root": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter one or more log ids out of an existing train/val/test split")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--exclude-log-ids", nargs="+", required=True)
    parser.add_argument("--reason", default="excluded suspect log")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = filter_split_logs(
        input_root=args.input_root,
        output_root=args.output_root,
        exclude_log_ids=args.exclude_log_ids,
        reason=args.reason,
        overwrite=args.overwrite,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
