#!/usr/bin/env python3
"""Build a residual-target split from effective-wrench labels and DeLaurier prior predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TARGET_COLUMNS = ["fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b"]
SPLITS = ("train", "val", "test")
IDENTITY_KEY_COLUMNS = ("log_id", "segment_id")
TIME_KEY_COLUMN = "__time_key_100hz"


def _check_target_columns(frame: pd.DataFrame, *, label: str) -> None:
    missing = [column for column in TARGET_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing target columns: {missing}")


def _alignment_key_columns(samples: pd.DataFrame, prior: pd.DataFrame) -> list[str]:
    if "log_id" not in samples.columns or "log_id" not in prior.columns:
        return []
    if "time_s" not in samples.columns or "time_s" not in prior.columns:
        return []
    keys = [column for column in IDENTITY_KEY_COLUMNS if column in samples.columns and column in prior.columns]
    if "log_id" not in keys:
        return []
    keys.append(TIME_KEY_COLUMN)
    return keys


def _with_alignment_time_key(frame: pd.DataFrame) -> pd.DataFrame:
    keyed = frame.copy()
    keyed[TIME_KEY_COLUMN] = (keyed["time_s"].astype(float) * 100.0).round().astype("int64")
    return keyed


def align_prior_to_samples(
    samples: pd.DataFrame,
    prior: pd.DataFrame,
    *,
    allow_row_order_fallback: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return ``prior`` rows ordered to match ``samples`` using stable sample keys.

    If prior predictions do not carry sample identity columns, row-order fallback is
    allowed only when requested explicitly. This avoids silently pairing prior rows
    from one split order with labels from another split order.
    """

    if len(samples) != len(prior):
        raise ValueError(f"samples/prior row mismatch: {len(samples)} != {len(prior)}")

    key_columns = _alignment_key_columns(samples, prior)
    if not key_columns:
        if not allow_row_order_fallback:
            raise ValueError(
                "prior predictions do not include stable alignment keys. "
                "Regenerate/key the prior predictions or pass allow_row_order_fallback=True "
                "only for known same-order legacy artifacts."
            )
        return prior.reset_index(drop=True).copy(), {
            "alignment_mode": "row_order_fallback",
            "alignment_key_columns": [],
            "allow_row_order_fallback": True,
        }

    keyed_samples = _with_alignment_time_key(samples)
    keyed_prior = _with_alignment_time_key(prior)
    sample_keys = keyed_samples.loc[:, key_columns].copy()
    prior_keys = keyed_prior.loc[:, key_columns].copy()
    if sample_keys.duplicated().any():
        duplicate = sample_keys.loc[sample_keys.duplicated(keep=False)].head(3).to_dict(orient="records")
        raise ValueError(f"sample alignment keys are not unique; examples: {duplicate}")
    if prior_keys.duplicated().any():
        duplicate = prior_keys.loc[prior_keys.duplicated(keep=False)].head(3).to_dict(orient="records")
        raise ValueError(f"prior alignment keys are not unique; examples: {duplicate}")

    keyed_samples = sample_keys.copy()
    keyed_samples["__sample_row"] = range(len(keyed_samples))
    prior_value_columns = [column for column in prior.columns if column not in key_columns and column != "time_s"]
    keyed_prior = pd.concat(
        [
            keyed_prior.loc[:, key_columns].reset_index(drop=True),
            prior.loc[:, prior_value_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    merged = keyed_samples.merge(keyed_prior, on=key_columns, how="left", validate="one_to_one", sort=False)
    if merged[prior_value_columns].isna().any().any():
        missing_count = int(merged[prior_value_columns].isna().any(axis=1).sum())
        raise ValueError(f"missing keyed prior rows for {missing_count} samples")
    merged = merged.sort_values("__sample_row", kind="mergesort").reset_index(drop=True)
    aligned = merged.loc[:, prior_value_columns].copy()
    return aligned, {
        "alignment_mode": "key_merge",
        "alignment_key_columns": key_columns,
        "allow_row_order_fallback": bool(allow_row_order_fallback),
    }


def build_residual_frame(
    samples: pd.DataFrame,
    prior: pd.DataFrame,
    *,
    prior_name: str,
    allow_row_order_fallback: bool = True,
) -> pd.DataFrame:
    """Return a copy of ``samples`` whose target columns are residual targets."""

    prior, alignment_info = align_prior_to_samples(
        samples,
        prior,
        allow_row_order_fallback=allow_row_order_fallback,
    )
    _check_target_columns(samples, label="samples")
    _check_target_columns(prior, label="prior")

    residual = samples.copy()
    for column in TARGET_COLUMNS:
        true_values = samples[column].astype(float).to_numpy()
        prior_values = prior[column].astype(float).to_numpy()
        residual[f"label_{column}"] = true_values
        residual[f"true_{column}"] = true_values
        residual[f"prior_{column}"] = prior_values
        residual[column] = true_values - prior_values
    residual.attrs["residual_prior_name"] = str(prior_name)
    residual.attrs["alignment_info"] = alignment_info
    return residual


def build_residual_split(
    split_root: Path,
    prior_root: Path,
    output_root: Path,
    *,
    prior_name: str,
    allow_row_order_fallback: bool = False,
) -> dict[str, object]:
    """Build train/val/test residual parquet files."""

    output_root.mkdir(parents=True, exist_ok=True)
    row_counts: dict[str, int] = {}
    alignment_modes: dict[str, object] = {}
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
        residual.to_parquet(output_root / f"{split}_samples.parquet", index=False)
        row_counts[split] = int(len(residual))
        alignment_modes[split] = dict(residual.attrs.get("alignment_info", {}))

    manifest = {
        "source_split_root": str(split_root),
        "prior_root": str(prior_root),
        "output_root": str(output_root),
        "prior_name": str(prior_name),
        "target_columns": TARGET_COLUMNS,
        "target_semantics": "residual = true_effective_wrench - prior_effective_wrench",
        "row_counts": row_counts,
        "alignment": alignment_modes,
    }
    (output_root / "residual_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--prior-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--prior-name", default="delaurier_physical")
    parser.add_argument(
        "--allow-row-order-fallback",
        action="store_true",
        help="Allow legacy prior parquets without sample keys to be paired by row order.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_residual_split(
        args.split_root,
        args.prior_root,
        args.output_root,
        prior_name=args.prior_name,
        allow_row_order_fallback=args.allow_row_order_fallback,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
