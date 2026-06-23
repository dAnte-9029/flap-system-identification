#!/usr/bin/env python3
"""Reorder legacy prior prediction parquets onto a target split by sample keys."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SPLITS = ("train", "val", "test")
IDENTITY_KEY_COLUMNS = ("log_id", "segment_id")
TIME_KEY_COLUMN = "__time_key_100hz"


def _add_time_key(frame: pd.DataFrame) -> pd.DataFrame:
    if "time_s" not in frame.columns:
        raise ValueError("frame is missing required column time_s")
    keyed = frame.copy()
    keyed[TIME_KEY_COLUMN] = (keyed["time_s"].astype(float) * 100.0).round().astype("int64")
    return keyed


def _key_columns(source_samples: pd.DataFrame, target_samples: pd.DataFrame) -> list[str]:
    if "log_id" not in source_samples.columns or "log_id" not in target_samples.columns:
        raise ValueError("source and target samples must both contain log_id")
    keys = [
        column
        for column in IDENTITY_KEY_COLUMNS
        if column in source_samples.columns and column in target_samples.columns
    ]
    if "log_id" not in keys:
        raise ValueError("source and target samples must both contain log_id")
    keys.append(TIME_KEY_COLUMN)
    return keys


def materialize_split(
    *,
    source_split_root: Path,
    target_split_root: Path,
    prior_root: Path,
    output_root: Path,
    split: str,
) -> dict[str, object]:
    source_samples = pd.read_parquet(source_split_root / f"{split}_samples.parquet")
    target_samples = pd.read_parquet(target_split_root / f"{split}_samples.parquet")
    prior = pd.read_parquet(prior_root / f"{split}_predictions.parquet")
    if len(source_samples) != len(prior):
        raise ValueError(f"{split}: source samples/prior row mismatch {len(source_samples)} != {len(prior)}")

    keyed_source = _add_time_key(source_samples)
    keyed_target = _add_time_key(target_samples)
    keys = _key_columns(source_samples, target_samples)
    source_keys = keyed_source.loc[:, keys].copy()
    target_keys = keyed_target.loc[:, keys].copy()
    if source_keys.duplicated().any():
        duplicate = source_keys.loc[source_keys.duplicated(keep=False)].head(3).to_dict(orient="records")
        raise ValueError(f"{split}: source keys are not unique: {duplicate}")
    if target_keys.duplicated().any():
        duplicate = target_keys.loc[target_keys.duplicated(keep=False)].head(3).to_dict(orient="records")
        raise ValueError(f"{split}: target keys are not unique: {duplicate}")

    value_columns = [column for column in prior.columns if column not in keys and column != "time_s"]
    keyed_prior = pd.concat([source_keys.reset_index(drop=True), prior.loc[:, value_columns].reset_index(drop=True)], axis=1)
    keyed_target["__target_row"] = range(len(keyed_target))
    merged = keyed_target.loc[:, keys + ["__target_row"]].merge(
        keyed_prior,
        on=keys,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if merged[value_columns].isna().any().any():
        missing_count = int(merged[value_columns].isna().any(axis=1).sum())
        raise ValueError(f"{split}: missing prior rows for {missing_count} target samples")
    merged = merged.sort_values("__target_row", kind="mergesort").reset_index(drop=True)

    output_key_columns = [column for column in ("dataset_id", *IDENTITY_KEY_COLUMNS, "time_s") if column in target_samples.columns]
    output = target_samples.loc[:, output_key_columns].copy()
    for column in value_columns:
        output[column] = merged[column].to_numpy()
    output.to_parquet(output_root / f"{split}_predictions.parquet", index=False)
    return {
        "split": split,
        "row_count": int(len(output)),
        "alignment_key_columns": keys,
        "value_columns": value_columns,
    }


def materialize_keyed_prior_for_split(
    *,
    source_split_root: Path,
    target_split_root: Path,
    prior_root: Path,
    output_root: Path,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    split_summaries = [
        materialize_split(
            source_split_root=source_split_root,
            target_split_root=target_split_root,
            prior_root=prior_root,
            output_root=output_root,
            split=split,
        )
        for split in SPLITS
    ]
    manifest = {
        "source_split_root": str(source_split_root),
        "target_split_root": str(target_split_root),
        "prior_root": str(prior_root),
        "output_root": str(output_root),
        "method": "legacy prior predictions keyed from source split and reordered to target split",
        "splits": split_summaries,
    }
    (output_root / "keyed_prior_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-split-root", required=True, type=Path)
    parser.add_argument("--target-split-root", required=True, type=Path)
    parser.add_argument("--prior-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = materialize_keyed_prior_for_split(
        source_split_root=args.source_split_root,
        target_split_root=args.target_split_root,
        prior_root=args.prior_root,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
