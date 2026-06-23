#!/usr/bin/env python3
"""Train a focused fx/fz correction on top of a raw force prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_delaurier_residual_split import align_prior_to_samples
from scripts.train_deployable_wrench_correction_v2 import (
    BASE_FEATURES,
    CONTROL_FEATURES,
    INTERACTION_FEATURES,
    LATERAL_FEATURES,
    PHASE_METADATA_COLUMNS,
    RATE_FEATURES,
    _fit_ridge_frame,
    build_v2_feature_frame,
)

TARGETS = ("fx_b", "fz_b")
SPLITS = ("train", "val", "test")


def _array(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return frame.loc[:, columns].to_numpy(dtype=float)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    mask = np.isfinite(y) & np.isfinite(p)
    if int(mask.sum()) == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "r2": np.nan, "corr": np.nan}
    err = p[mask] - y[mask]
    centered = y[mask] - float(np.mean(y[mask]))
    ss_tot = float(np.sum(centered * centered))
    ss_res = float(np.sum(err * err))
    corr = np.nan
    if int(mask.sum()) > 2 and np.std(y[mask]) > 1.0e-12 and np.std(p[mask]) > 1.0e-12:
        corr = float(np.corrcoef(y[mask], p[mask])[0, 1])
    return {
        "n": int(mask.sum()),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else np.nan,
        "corr": corr,
    }


def _metrics_rows(frame: pd.DataFrame, pred: np.ndarray, *, split: str, model: str) -> list[dict[str, object]]:
    rows = []
    for idx, target in enumerate(TARGETS):
        rows.append(
            {
                "split": split,
                "model": model,
                "target": target,
                **_metrics(frame[f"label_{target}"].to_numpy(dtype=float), pred[:, idx]),
            }
        )
    rows.append(
        {
            "split": split,
            "model": model,
            "target": "fx_fz_mean",
            "n": int(min(row["n"] for row in rows)),
            "rmse": float(np.nanmean([row["rmse"] for row in rows])),
            "mae": float(np.nanmean([row["mae"] for row in rows])),
            "bias": float(np.nanmean([row["bias"] for row in rows])),
            "r2": float(np.nanmean([row["r2"] for row in rows])),
            "corr": float(np.nanmean([row["corr"] for row in rows])),
        }
    )
    return rows


def _feature_groups(columns: list[str]) -> dict[str, list[str]]:
    available = set(columns)

    def present(names: tuple[str, ...]) -> list[str]:
        return [name for name in names if name in available]

    return {
        "base": present(BASE_FEATURES),
        "base+rates": present(BASE_FEATURES + RATE_FEATURES),
        "base+controls": present(BASE_FEATURES + CONTROL_FEATURES),
        "base+rates+controls": present(BASE_FEATURES + RATE_FEATURES + CONTROL_FEATURES),
        "base+rates+controls+lateral": present(BASE_FEATURES + RATE_FEATURES + CONTROL_FEATURES + LATERAL_FEATURES),
        "base+rates+controls+lateral+interactions": present(
            BASE_FEATURES + RATE_FEATURES + CONTROL_FEATURES + LATERAL_FEATURES + INTERACTION_FEATURES
        ),
    }


def _load_split(split_root: Path, prior_root: Path, split: str) -> pd.DataFrame:
    samples = pd.read_parquet(split_root / f"{split}_samples.parquet").reset_index(drop=True)
    prior_raw = pd.read_parquet(prior_root / f"{split}_predictions.parquet")
    prior, _ = align_prior_to_samples(samples, prior_raw, allow_row_order_fallback=False)
    frame = samples.copy()
    for target in TARGETS:
        frame[f"label_{target}"] = samples[target].to_numpy(dtype=float)
        frame[f"prior_{target}"] = prior[target].to_numpy(dtype=float)
    return frame


def _design(features: pd.DataFrame, prior: np.ndarray, variant: str) -> pd.DataFrame:
    if variant == "additive":
        return features
    if variant == "affine":
        out = features.copy()
        for idx, target in enumerate(TARGETS):
            for column in features.columns:
                out[f"prior_{target}_x_{column}"] = prior[:, idx] * features[column].to_numpy(dtype=float)
        return out
    raise ValueError(f"unknown variant: {variant}")


def run(*, split_root: Path, prior_root: Path, output_root: Path, alphas: tuple[float, ...]) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    pred_root = output_root / "prediction_parquets"
    pred_root.mkdir(exist_ok=True)
    frames = {split: _load_split(split_root, prior_root, split) for split in SPLITS}
    feature_frames = {split: build_v2_feature_frame(frame)[0] for split, frame in frames.items()}
    groups = _feature_groups(feature_frames["train"].columns.tolist())

    y_train = _array(frames["train"], tuple(f"label_{target}" for target in TARGETS))
    prior_train = _array(frames["train"], tuple(f"prior_{target}" for target in TARGETS))
    rows: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []

    for split, frame in frames.items():
        prior = _array(frame, tuple(f"prior_{target}" for target in TARGETS))
        rows.extend(_metrics_rows(frame, prior, split=split, model="raw_prior"))

    for group_name, columns in groups.items():
        if not columns:
            continue
        train_features = feature_frames["train"].loc[:, columns]
        for variant in ("additive", "affine"):
            for alpha in alphas:
                design = _design(train_features, prior_train, variant)
                model = _fit_ridge_frame(design, y_train - prior_train, float(alpha))
                val_pred = None
                for split, frame in frames.items():
                    prior = _array(frame, tuple(f"prior_{target}" for target in TARGETS))
                    x = _design(feature_frames[split].loc[:, columns], prior, variant)
                    pred = prior + model.predict(x)
                    rows.extend(
                        _metrics_rows(
                            frame,
                            pred,
                            split=split,
                            model=f"{variant}_{group_name}_alpha_{float(alpha):g}",
                        )
                    )
                    if split == "val":
                        val_pred = pred
                if val_pred is None:
                    raise RuntimeError("validation prediction was not computed")
                val_rows = _metrics_rows(frames["val"], val_pred, split="val", model="candidate")
                val_rmse = [row for row in val_rows if row["target"] == "fx_fz_mean"][0]["rmse"]
                candidates.append(
                    {
                        "variant": variant,
                        "feature_group": group_name,
                        "alpha": float(alpha),
                        "columns": columns,
                        "model": model,
                        "val_rmse": float(val_rmse),
                    }
                )

    selected = min(candidates, key=lambda item: (item["val_rmse"], len(item["columns"]), item["variant"], item["alpha"]))
    selected_name = f"{selected['variant']}_{selected['feature_group']}_alpha_{selected['alpha']:g}"
    metrics = pd.DataFrame(rows)
    metrics["is_selected"] = metrics["model"].eq(selected_name)
    metrics.to_csv(output_root / "fx_fz_correction_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "variant": item["variant"],
                "feature_group": item["feature_group"],
                "alpha": item["alpha"],
                "n_features": len(item["columns"]),
                "val_rmse": item["val_rmse"],
                "is_selected": item is selected,
            }
            for item in candidates
        ]
    ).to_csv(output_root / "fx_fz_model_selection.csv", index=False)

    for split, frame in frames.items():
        prior = _array(frame, tuple(f"prior_{target}" for target in TARGETS))
        x = _design(feature_frames[split].loc[:, selected["columns"]], prior, selected["variant"])
        pred = prior + selected["model"].predict(x)
        out_cols = [
            column
            for column in ("timestamp_us", "time_s", "log_id", "segment_id", "cycle_id", *PHASE_METADATA_COLUMNS, "split")
            if column in frame.columns
        ]
        out = frame.loc[:, out_cols].copy()
        for idx, target in enumerate(TARGETS):
            out[f"label_{target}"] = frame[f"label_{target}"].to_numpy(dtype=float)
            out[f"prior_{target}"] = frame[f"prior_{target}"].to_numpy(dtype=float)
            out[f"force_v2_{target}"] = pred[:, idx]
            out[f"force_v2_residual_{target}"] = out[f"label_{target}"] - out[f"force_v2_{target}"]
        out.to_parquet(pred_root / f"{split}_predictions.parquet", index=False)

    manifest = {
        "split_root": str(split_root),
        "prior_root": str(prior_root),
        "output_root": str(output_root),
        "targets": list(TARGETS),
        "selected": {
            "variant": selected["variant"],
            "feature_group": selected["feature_group"],
            "alpha": selected["alpha"],
            "n_features": len(selected["columns"]),
            "val_rmse": selected["val_rmse"],
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _parse_alphas(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alphas", default="0,0.001,0.01,0.1,1,10,100")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                split_root=args.split_root,
                prior_root=args.prior_root,
                output_root=args.output_root,
                alphas=_parse_alphas(args.alphas),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
