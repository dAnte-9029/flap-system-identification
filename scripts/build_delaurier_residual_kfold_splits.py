#!/usr/bin/env python3
"""Build log-level k-fold residual splits from effective-wrench labels and DeLaurier priors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_delaurier_residual_split import SPLITS, TARGET_COLUMNS, build_residual_frame


def _date_from_log_id(log_id: str) -> str:
    match = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", str(log_id))
    return match.group(1) if match else "unknown"


def _log_key_frame(logs: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset_id", "log_id"}
    missing = sorted(required - set(logs.columns))
    if missing:
        raise ValueError(f"log table is missing required columns: {missing}")
    output = logs.copy()
    output["dataset_id"] = output["dataset_id"].astype(str)
    output["log_id"] = output["log_id"].astype(str)
    if "valid_sample_count" not in output.columns:
        output["valid_sample_count"] = 1
    output["valid_sample_count"] = output["valid_sample_count"].fillna(0).astype(int)
    if "date" not in output.columns:
        output["date"] = output["log_id"].map(_date_from_log_id)
    output["date"] = output["date"].fillna("unknown").astype(str)
    return output


def assign_balanced_log_folds(logs: pd.DataFrame, *, n_folds: int, seed: int) -> pd.DataFrame:
    """Assign whole logs to approximately date-stratified, sample-balanced folds."""

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    output = _log_key_frame(logs)
    unique_logs = output.drop_duplicates(["dataset_id", "log_id"]).reset_index(drop=True)
    if len(unique_logs) < n_folds:
        raise ValueError(f"need at least n_folds logs: {len(unique_logs)} < {n_folds}")

    fold_loads = np.zeros(n_folds, dtype=np.int64)
    unique_logs["fold"] = -1
    rng = np.random.default_rng(seed)
    date_order = (
        unique_logs.groupby("date", sort=True)["valid_sample_count"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    for date in date_order:
        date_mask = unique_logs["date"] == date
        date_logs = unique_logs.loc[date_mask].copy()
        date_logs["_tie_break"] = rng.random(len(date_logs))
        date_logs = date_logs.sort_values(
            ["valid_sample_count", "_tie_break"],
            ascending=[False, True],
            kind="mergesort",
        )
        date_used_folds: set[int] = set()
        for row in date_logs.itertuples():
            if len(date_used_folds) < n_folds:
                candidates = [fold for fold in range(n_folds) if fold not in date_used_folds]
            else:
                candidates = list(range(n_folds))
            fold = min(candidates, key=lambda candidate: (fold_loads[candidate], candidate))
            unique_logs.loc[row.Index, "fold"] = fold
            fold_loads[fold] += int(row.valid_sample_count)
            date_used_folds.add(fold)
    unique_logs["fold"] = unique_logs["fold"].astype(int)
    return unique_logs.sort_values(["fold", "dataset_id", "log_id"]).reset_index(drop=True)


def split_date_coverage(fold_logs: pd.DataFrame) -> pd.DataFrame:
    """Return per-split date coverage counts for fold logs."""

    logs = _log_key_frame(fold_logs)
    if "split" not in logs.columns:
        raise ValueError("fold log table is missing required column: split")
    coverage = (
        logs.groupby(["split", "date"], observed=True)
        .agg(log_count=("log_id", "count"), sample_count=("valid_sample_count", "sum"))
        .reset_index()
        .sort_values(["split", "date"])
        .reset_index(drop=True)
    )
    return coverage


def _read_residual_source(
    split_root: Path,
    prior_root: Path,
    *,
    prior_name: str,
    allow_row_order_fallback: bool,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in SPLITS:
        samples_path = split_root / f"{split}_samples.parquet"
        prior_path = prior_root / f"{split}_predictions.parquet"
        samples = pd.read_parquet(samples_path)
        prior = pd.read_parquet(prior_path)
        residual = build_residual_frame(
            samples,
            prior,
            prior_name=prior_name,
            allow_row_order_fallback=allow_row_order_fallback,
        )
        residual["source_split"] = split
        frames.append(residual)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    for column in ("dataset_id", "log_id"):
        if column not in combined.columns:
            raise ValueError(f"sample tables are missing required column: {column}")
        combined[column] = combined[column].astype(str)
    return combined


def _write_fold_logs(logs: pd.DataFrame, fold_root: Path, *, test_fold: int, val_fold: int) -> dict[str, int]:
    fold_logs = logs.copy()
    fold_logs["split"] = "train"
    fold_logs.loc[fold_logs["fold"] == val_fold, "split"] = "val"
    fold_logs.loc[fold_logs["fold"] == test_fold, "split"] = "test"
    counts = {
        split: int((fold_logs["split"] == split).sum())
        for split in SPLITS
    }
    fold_logs.sort_values(["split", "dataset_id", "log_id"]).to_csv(fold_root / "all_logs.csv", index=False)
    for split in SPLITS:
        fold_logs.loc[fold_logs["split"] == split].to_csv(fold_root / f"{split}_logs.csv", index=False)
    split_date_coverage(fold_logs).to_csv(fold_root / "date_coverage.csv", index=False)
    return counts


def build_residual_kfold_splits(
    *,
    split_root: Path,
    prior_root: Path,
    output_root: Path,
    n_folds: int,
    seed: int,
    prior_name: str,
    allow_row_order_fallback: bool = False,
) -> dict[str, object]:
    """Materialize residual train/val/test parquet files for log-level k-fold evaluation."""

    all_logs_path = split_root / "all_logs.csv"
    if not all_logs_path.exists():
        raise FileNotFoundError(f"missing log table: {all_logs_path}")
    assigned_logs = assign_balanced_log_folds(pd.read_csv(all_logs_path), n_folds=n_folds, seed=seed)
    residual_source = _read_residual_source(
        split_root,
        prior_root,
        prior_name=prior_name,
        allow_row_order_fallback=allow_row_order_fallback,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    assigned_logs.to_csv(output_root / "fold_assignments.csv", index=False)

    fold_manifests: list[dict[str, object]] = []
    for fold_index in range(n_folds):
        val_fold = int((fold_index + 1) % n_folds)
        fold_root = output_root / f"fold_{fold_index:02d}"
        fold_root.mkdir(parents=True, exist_ok=True)
        log_counts = _write_fold_logs(assigned_logs, fold_root, test_fold=fold_index, val_fold=val_fold)
        fold_logs = pd.read_csv(fold_root / "all_logs.csv")

        sample_counts: dict[str, int] = {}
        for split in SPLITS:
            split_logs = fold_logs.loc[fold_logs["split"] == split, ["dataset_id", "log_id"]].astype(str)
            key_index = pd.MultiIndex.from_frame(split_logs)
            sample_keys = pd.MultiIndex.from_frame(residual_source[["dataset_id", "log_id"]].astype(str))
            frame = residual_source.loc[sample_keys.isin(key_index)].copy()
            frame["split"] = split
            frame.to_parquet(fold_root / f"{split}_samples.parquet", index=False)
            sample_counts[split] = int(len(frame))

        fold_manifest = {
            "fold_index": int(fold_index),
            "test_fold": int(fold_index),
            "val_fold": val_fold,
            "split_root": str(split_root),
            "prior_root": str(prior_root),
            "prior_name": str(prior_name),
            "log_counts": log_counts,
            "sample_counts": sample_counts,
            "date_coverage_csv": str(fold_root / "date_coverage.csv"),
            "date_coverage": split_date_coverage(fold_logs).to_dict(orient="records"),
            "target_semantics": "residual = true_effective_wrench - prior_effective_wrench",
            "target_columns": TARGET_COLUMNS,
        }
        (fold_root / "residual_manifest.json").write_text(
            json.dumps(fold_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        fold_manifests.append(fold_manifest)

    manifest = {
        "split_root": str(split_root),
        "prior_root": str(prior_root),
        "output_root": str(output_root),
        "n_folds": int(n_folds),
        "seed": int(seed),
        "prior_name": str(prior_name),
        "allow_row_order_fallback": bool(allow_row_order_fallback),
        "fold_policy": "whole-log approximately date-stratified sample-balanced folds",
        "folds": fold_manifests,
    }
    (output_root / "kfold_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--prior-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prior-name", default="delaurier_physical_calibrated_v1")
    parser.add_argument(
        "--allow-row-order-fallback",
        action="store_true",
        help="Allow legacy prior parquets without sample keys to be paired by row order.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_residual_kfold_splits(
        split_root=args.split_root,
        prior_root=args.prior_root,
        output_root=args.output_root,
        n_folds=args.n_folds,
        seed=args.seed,
        prior_name=args.prior_name,
        allow_row_order_fallback=args.allow_row_order_fallback,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
