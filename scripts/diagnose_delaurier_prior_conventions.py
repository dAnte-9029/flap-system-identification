#!/usr/bin/env python3
"""Diagnose force-axis/sign convention candidates for DeLaurier prior predictions."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_delaurier_residual_split import align_prior_to_samples

FORCE_COLUMNS = ("fx_b", "fy_b", "fz_b")
SPLITS = ("train", "val", "test")
Transform = dict[str, tuple[str, float]]


def _transform_candidates() -> dict[str, Transform]:
    candidates: dict[str, Transform] = {}
    for perm in itertools.permutations(FORCE_COLUMNS):
        perm_name = "".join(channel.split("_", 1)[0].replace("f", "") for channel in perm)
        for signs in itertools.product((-1.0, 1.0), repeat=len(FORCE_COLUMNS)):
            sign_name = "".join("p" if sign > 0 else "m" for sign in signs)
            mapping = {
                output_channel: (input_channel, float(sign))
                for output_channel, input_channel, sign in zip(FORCE_COLUMNS, perm, signs)
            }
            candidates[f"perm_{perm_name}_{sign_name}"] = mapping
    aliases = {
        "identity": {"fx_b": ("fx_b", 1.0), "fy_b": ("fy_b", 1.0), "fz_b": ("fz_b", 1.0)},
        "neg_all": {"fx_b": ("fx_b", -1.0), "fy_b": ("fy_b", -1.0), "fz_b": ("fz_b", -1.0)},
        "yz_flip_flu_to_frd": {"fx_b": ("fx_b", 1.0), "fy_b": ("fy_b", -1.0), "fz_b": ("fz_b", -1.0)},
    }
    return {**aliases, **candidates}


def _load_aligned(split_root: Path, prior_root: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = pd.read_parquet(split_root / f"{split}_samples.parquet")
    prior_raw = pd.read_parquet(prior_root / f"{split}_predictions.parquet")
    prior, _ = align_prior_to_samples(samples, prior_raw, allow_row_order_fallback=False)
    return samples, prior


def _transform(prior: pd.DataFrame, mapping: Transform) -> pd.DataFrame:
    transformed = pd.DataFrame(index=prior.index)
    for output_channel in FORCE_COLUMNS:
        input_channel, sign = mapping[output_channel]
        transformed[output_channel] = float(sign) * prior[input_channel].astype(float)
    return transformed


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    residual = y_pred[mask] - y_true[mask]
    return float(np.sqrt(np.mean(residual * residual)))


def _corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(y_true[mask], y_pred[mask])[0, 1])


def _fit_affine(train_samples: pd.DataFrame, train_prior: pd.DataFrame) -> dict[str, tuple[float, float]]:
    params: dict[str, tuple[float, float]] = {}
    for channel in FORCE_COLUMNS:
        x = train_prior[channel].to_numpy(dtype=float)
        y = train_samples[channel].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        design = np.column_stack([x[mask], np.ones(mask.sum())])
        gain, bias = np.linalg.lstsq(design, y[mask], rcond=None)[0]
        params[channel] = (float(gain), float(bias))
    return params


def _phase_median_corr(samples: pd.DataFrame, prior: pd.DataFrame, channel: str, phase_bins: int) -> float:
    if "phase_corrected_rad" not in samples.columns:
        return float("nan")
    phase = np.mod(samples["phase_corrected_rad"].to_numpy(dtype=float), 2.0 * np.pi)
    bin_index = np.floor(phase / (2.0 * np.pi) * float(phase_bins)).astype(int)
    bin_index = np.clip(bin_index, 0, phase_bins - 1)
    label_medians = []
    prior_medians = []
    for index in range(phase_bins):
        mask = bin_index == index
        if mask.sum() == 0:
            continue
        label_medians.append(float(np.median(samples[channel].to_numpy(dtype=float)[mask])))
        prior_medians.append(float(np.median(prior[channel].to_numpy(dtype=float)[mask])))
    if len(label_medians) < 2:
        return float("nan")
    return float(np.corrcoef(np.asarray(label_medians), np.asarray(prior_medians))[0, 1])


def run_diagnostics(
    *,
    split_root: Path,
    prior_root: Path,
    output_root: Path,
    phase_bins: int,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    loaded = {split: _load_aligned(split_root, prior_root, split) for split in SPLITS}
    transforms = _transform_candidates()
    rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []

    for transform_name, mapping in transforms.items():
        train_samples, train_prior_raw = loaded["train"]
        train_prior = _transform(train_prior_raw, mapping)
        affine = _fit_affine(train_samples, train_prior)
        for channel, (gain, bias) in affine.items():
            input_channel, sign = mapping[channel]
            parameter_rows.append(
                {
                    "transform": transform_name,
                    "channel": channel,
                    "input_channel": input_channel,
                    "sign": sign,
                    "affine_gain": gain,
                    "affine_bias": bias,
                }
            )
        for split, (samples, prior_raw) in loaded.items():
            prior = _transform(prior_raw, mapping)
            for channel in FORCE_COLUMNS:
                input_channel, sign = mapping[channel]
                y = samples[channel].to_numpy(dtype=float)
                p = prior[channel].to_numpy(dtype=float)
                gain, bias = affine[channel]
                aligned = gain * p + bias
                rows.append(
                    {
                        "split": split,
                        "transform": transform_name,
                        "channel": channel,
                        "input_channel": input_channel,
                        "sign": sign,
                        "raw_corr": _corr(y, p),
                        "raw_rmse": _rmse(y, p),
                        "affine_gain": gain,
                        "affine_bias": bias,
                        "affine_rmse": _rmse(y, aligned),
                        "label_std": float(np.nanstd(y, ddof=1)),
                        "phase_median_corr": _phase_median_corr(samples, prior, channel, phase_bins),
                    }
                )

    summary = pd.DataFrame(rows)
    params = pd.DataFrame(parameter_rows)
    summary.to_csv(output_root / "summary.csv", index=False)
    params.to_csv(output_root / "affine_parameters.csv", index=False)
    best = (
        summary.loc[summary["split"] == "test"]
        .sort_values(["channel", "affine_rmse", "raw_rmse"])
        .groupby("channel", as_index=False)
        .head(3)
    )
    best.to_csv(output_root / "test_best_by_channel.csv", index=False)
    manifest = {
        "split_root": str(split_root),
        "prior_root": str(prior_root),
        "output_root": str(output_root),
        "phase_bins": int(phase_bins),
        "transform_count": len(transforms),
        "files": {
            "summary": str(output_root / "summary.csv"),
            "affine_parameters": str(output_root / "affine_parameters.csv"),
            "test_best_by_channel": str(output_root / "test_best_by_channel.csv"),
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--prior-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--phase-bins", type=int, default=36)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_diagnostics(
        split_root=args.split_root,
        prior_root=args.prior_root,
        output_root=args.output_root,
        phase_bins=args.phase_bins,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
