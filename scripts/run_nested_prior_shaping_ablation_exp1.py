#!/usr/bin/env python3
"""Experiment 1: nested no-shaping ablation for the fx/fz gain-bias correction.

This script compares a fixed nominal DeLaurier prior against the current shaped
DeLaurier prior under the same six-fold nested whole-log splits used by the
paper. For each prior and each outer fold, the gain-bias ridge parameter is
selected on the inner validation logs, then the selected model is refit on the
outer development logs and evaluated on the held-out outer logs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_deployable_wrench_correction_v2 import PHASE_METADATA_COLUMNS, build_v2_feature_frame
from scripts.train_fx_fz_correction import TARGETS, _array, _metrics_rows
from scripts.train_fx_fz_structured_correction import (
    PHASE_FREQ_Q_COLUMNS,
    _fit_direct_model,
    _gain_bias_design,
    _predict_direct,
    _with_intercept,
)


DEFAULT_SPLITS_ROOT = Path(
    "/home/zn/paper/AeroConf_effective_aero/research_notes/20260612_nested_outer_cv/splits"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/20260703_prior_shaping_ablation_exp1"
PRIOR_SWEEP_ROOT = PROJECT_ROOT / (
    "artifacts/20260604_delaurier_other_parameter_sweep_fixed_twist10_ratio8_sg0p03_v1/priors"
)
DEFAULT_PRIORS = {
    "nominal": PRIOR_SWEEP_ROOT
    / "attached__twist_eta_max_deg_10p0__alpha0_deg_0p0__eta_s_0p65__cd_f_0p0__enable_separation_sep_off__alpha_stall_max_deg_12p0__cd_cf_1p95__xi_0p0",
    "shaped": PRIOR_SWEEP_ROOT
    / "separation__twist_eta_max_deg_10p0__alpha0_deg_4p0__eta_s_0p65__cd_f_0p0__enable_separation_sep_on__alpha_stall_max_deg_18p0__cd_cf_1p2__xi_1p0",
}
DEFAULT_ALPHA_GRID = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
SPLITS = ("train", "val", "test")
PRIOR_TARGET_COLUMNS = ("fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b")
ALIGN_TIME_KEY = "__time_key_100hz"
ALIGN_KEY_COLUMNS = ("log_id", "segment_id", ALIGN_TIME_KEY)


def _parse_alphas(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def _parse_folds(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepare_output(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(f"output exists; pass --force to overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _alignment_key_frame(frame: pd.DataFrame, *, row_column: str | None = None) -> pd.DataFrame:
    missing = [column for column in ("log_id", "segment_id", "time_s") if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing alignment columns: {missing}")
    keyed = pd.DataFrame(
        {
            "log_id": frame["log_id"].astype(str).to_numpy(),
            "segment_id": pd.to_numeric(frame["segment_id"], errors="raise").astype("int64").to_numpy(),
            ALIGN_TIME_KEY: np.round(
                pd.to_numeric(frame["time_s"], errors="raise").astype(float).to_numpy() * 100.0
            ).astype("int64"),
        }
    )
    if row_column is not None:
        keyed[row_column] = np.arange(len(frame), dtype=np.int64)
    return keyed


def _read_prior_all(prior_root: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for split in SPLITS:
        path = prior_root / f"{split}_predictions.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path).reset_index(drop=True)
        frame["prior_source_split"] = split
        parts.append(frame)
    out = pd.concat(parts, ignore_index=True)
    missing = [column for column in ("fx_b", "fz_b") if column not in out.columns]
    if missing:
        raise ValueError(f"prior predictions are missing columns: {missing}")
    return out


def _align_prior_to_subset(samples: pd.DataFrame, prior_all: pd.DataFrame) -> pd.DataFrame:
    sample_keys = _alignment_key_frame(samples, row_column="__sample_row")
    prior_keys = _alignment_key_frame(prior_all)
    prior_value_columns = [column for column in PRIOR_TARGET_COLUMNS if column in prior_all.columns]
    if not {"fx_b", "fz_b"}.issubset(prior_value_columns):
        raise ValueError("prior predictions are missing fx_b/fz_b")

    if sample_keys.loc[:, list(ALIGN_KEY_COLUMNS)].duplicated().any():
        duplicate = sample_keys.loc[sample_keys.loc[:, list(ALIGN_KEY_COLUMNS)].duplicated(keep=False)].head(3)
        raise ValueError(f"sample alignment keys are not unique; examples={duplicate.to_dict(orient='records')}")
    if prior_keys.loc[:, list(ALIGN_KEY_COLUMNS)].duplicated().any():
        duplicate = prior_keys.loc[prior_keys.loc[:, list(ALIGN_KEY_COLUMNS)].duplicated(keep=False)].head(3)
        raise ValueError(f"prior alignment keys are not unique; examples={duplicate.to_dict(orient='records')}")

    prior_payload = pd.concat(
        [
            prior_keys.loc[:, list(ALIGN_KEY_COLUMNS)].reset_index(drop=True),
            prior_all.loc[:, prior_value_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    merged = sample_keys.merge(prior_payload, on=list(ALIGN_KEY_COLUMNS), how="left", validate="one_to_one", sort=False)
    if merged[prior_value_columns].isna().any().any():
        missing_count = int(merged[prior_value_columns].isna().any(axis=1).sum())
        raise ValueError(f"missing keyed prior rows for {missing_count} samples")
    merged = merged.sort_values("__sample_row", kind="mergesort").reset_index(drop=True)
    return merged.loc[:, prior_value_columns].copy()


def _load_frames(fold_root: Path, prior_all: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        samples = pd.read_parquet(fold_root / f"{split}_samples.parquet").reset_index(drop=True)
        prior = _align_prior_to_subset(samples, prior_all)
        frame = samples.copy()
        for target in TARGETS:
            if target not in samples.columns:
                raise ValueError(f"{fold_root}/{split} is missing label column {target}")
            frame[f"label_{target}"] = samples[target].to_numpy(dtype=float)
            frame[f"prior_{target}"] = prior[target].to_numpy(dtype=float)
        frames[split] = frame
    return frames


def _build_features(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    outputs = {split: build_v2_feature_frame(frame) for split, frame in frames.items()}
    features = {split: output[0] for split, output in outputs.items()}
    specs = {split: output[1] for split, output in outputs.items()}
    missing = sorted(set(PHASE_FREQ_Q_COLUMNS) - set(features["train"].columns))
    if missing:
        raise ValueError(f"missing phase/frequency/q feature columns: {missing}")
    return features, {"by_split": specs}


def _fit_gain_bias(train_frame: pd.DataFrame, train_features: pd.DataFrame, *, alpha: float) -> list[object]:
    y_train = _array(train_frame, tuple(f"label_{target}" for target in TARGETS))
    prior_train = _array(train_frame, tuple(f"prior_{target}" for target in TARGETS))
    phi_train = _with_intercept(train_features, list(PHASE_FREQ_Q_COLUMNS))
    train_designs = [_gain_bias_design(phi_train, prior_train, idx) for idx in range(len(TARGETS))]
    return _fit_direct_model(train_designs, y_train, float(alpha))


def _predict_gain_bias(models: list[object], frame: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
    prior = _array(frame, tuple(f"prior_{target}" for target in TARGETS))
    phi = _with_intercept(features, list(PHASE_FREQ_Q_COLUMNS))
    designs = [_gain_bias_design(phi, prior, idx) for idx in range(len(TARGETS))]
    return _predict_direct(models, designs)


def _fx_fz_mean_rmse(rows: list[dict[str, object]], *, split: str, model: str) -> float:
    for row in rows:
        if row["split"] == split and row["model"] == model and row["target"] == "fx_fz_mean":
            return float(row["rmse"])
    raise ValueError(f"missing {split} fx_fz_mean RMSE for {model}")


def _prediction_frame(frame: pd.DataFrame, pred: np.ndarray, *, model_name: str) -> pd.DataFrame:
    out_cols = [
        column
        for column in (
            "timestamp_us",
            "time_s",
            "log_id",
            "segment_id",
            "cycle_id",
            *PHASE_METADATA_COLUMNS,
            "source_split",
            "nested_split",
            "split",
        )
        if column in frame.columns
    ]
    out = frame.loc[:, out_cols].copy()
    out["model"] = model_name
    for idx, target in enumerate(TARGETS):
        out[f"label_{target}"] = frame[f"label_{target}"].to_numpy(dtype=float)
        out[f"prior_{target}"] = frame[f"prior_{target}"].to_numpy(dtype=float)
        out[f"force_v2_{target}"] = pred[:, idx]
        out[f"force_v2_residual_{target}"] = out[f"label_{target}"] - out[f"force_v2_{target}"]
    return out


def _select_alpha_and_evaluate(
    *,
    prior_name: str,
    prior_root: Path,
    fold_root: Path,
    fold_output_root: Path,
    prior_all: pd.DataFrame,
    alphas: tuple[float, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    fold_output_root.mkdir(parents=True, exist_ok=True)
    outer_fold = int(fold_root.name.split("_")[-1])
    frames = _load_frames(fold_root, prior_all)
    features, feature_spec = _build_features(frames)

    selection_rows: list[dict[str, object]] = []
    selection_metric_rows: list[dict[str, object]] = []
    prior_predictions = {split: _array(frame, tuple(f"prior_{target}" for target in TARGETS)) for split, frame in frames.items()}
    prior_model_name = f"{prior_name}_prior_only"
    gain_bias_model_name = f"{prior_name}_gain_bias"

    for split in ("train", "val"):
        rows = _metrics_rows(frames[split], prior_predictions[split], split=split, model=prior_model_name)
        selection_metric_rows.extend(
            {
                **row,
                "outer_fold": outer_fold,
                "prior_name": prior_name,
                "prior_root": str(prior_root),
                "selection_scope": "inner_train_val_only",
                "alpha": np.nan,
            }
            for row in rows
        )

    for alpha in alphas:
        candidate_name = f"{gain_bias_model_name}_alpha_{float(alpha):g}"
        models = _fit_gain_bias(frames["train"], features["train"], alpha=float(alpha))
        rows: list[dict[str, object]] = []
        for split in ("train", "val"):
            pred = _predict_gain_bias(models, frames[split], features[split])
            rows.extend(_metrics_rows(frames[split], pred, split=split, model=candidate_name))
        val_rmse = _fx_fz_mean_rmse(rows, split="val", model=candidate_name)
        selection_rows.append(
            {
                "outer_fold": outer_fold,
                "prior_name": prior_name,
                "prior_root": str(prior_root),
                "model": gain_bias_model_name,
                "candidate_model": candidate_name,
                "alpha": float(alpha),
                "val_rmse": val_rmse,
                "columns": ",".join(PHASE_FREQ_Q_COLUMNS),
                "n_features": int(2 * (len(PHASE_FREQ_Q_COLUMNS) + 1) * len(TARGETS)),
            }
        )
        selection_metric_rows.extend(
            {
                **row,
                "outer_fold": outer_fold,
                "prior_name": prior_name,
                "prior_root": str(prior_root),
                "selection_scope": "inner_train_val_only",
                "alpha": float(alpha),
            }
            for row in rows
        )

    selection = pd.DataFrame(selection_rows).sort_values(["val_rmse", "alpha"], kind="mergesort").reset_index(drop=True)
    selected = selection.iloc[0].replace({np.nan: None}).to_dict()
    selected_alpha = float(selected["alpha"])
    selection["is_selected"] = selection["candidate_model"].eq(str(selected["candidate_model"]))
    selection.to_csv(fold_output_root / "gain_bias_alpha_selection.csv", index=False)
    pd.DataFrame(selection_metric_rows).to_csv(fold_output_root / "selection_metrics.csv", index=False)

    dev_frame = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    dev_features = pd.concat([features["train"], features["val"]], ignore_index=True)
    final_models = _fit_gain_bias(dev_frame, dev_features, alpha=selected_alpha)
    pred_test = _predict_gain_bias(final_models, frames["test"], features["test"])

    fold_meta = {
        "outer_fold": outer_fold,
        "prior_name": prior_name,
        "prior_root": str(prior_root),
        "selection_scope": "inner_train_val_only",
        "final_fit_scope": "outer_development_train_plus_val",
        "alpha": selected_alpha,
        "train_rows": int(len(frames["train"])),
        "val_rows": int(len(frames["val"])),
        "test_rows": int(len(frames["test"])),
        "train_logs": int(frames["train"]["log_id"].nunique()),
        "val_logs": int(frames["val"]["log_id"].nunique()),
        "test_logs": int(frames["test"]["log_id"].nunique()),
    }

    test_rows: list[dict[str, object]] = []
    for row in _metrics_rows(frames["test"], prior_predictions["test"], split="test", model=prior_model_name):
        test_rows.append({**fold_meta, **row})
    for row in _metrics_rows(frames["test"], pred_test, split="test", model=gain_bias_model_name):
        test_rows.append({**fold_meta, **row})

    per_log_rows: list[dict[str, object]] = []
    for log_id, log_frame in frames["test"].groupby("log_id", sort=True):
        idx = log_frame.index.to_numpy(dtype=int)
        log_meta = {**fold_meta, "log_id": str(log_id), "log_rows": int(len(log_frame))}
        for row in _metrics_rows(log_frame, prior_predictions["test"][idx, :], split="test", model=prior_model_name):
            per_log_rows.append({**log_meta, **row})
        for row in _metrics_rows(log_frame, pred_test[idx, :], split="test", model=gain_bias_model_name):
            per_log_rows.append({**log_meta, **row})

    pd.DataFrame(test_rows).to_csv(fold_output_root / "outer_test_metrics.csv", index=False)
    pd.DataFrame(per_log_rows).to_csv(fold_output_root / "outer_test_per_log_metrics.csv", index=False)
    _prediction_frame(frames["test"], pred_test, model_name=gain_bias_model_name).to_parquet(
        fold_output_root / "test_predictions.parquet",
        index=False,
    )
    manifest = {
        **fold_meta,
        "selected": selected,
        "feature_columns": list(PHASE_FREQ_Q_COLUMNS),
        "feature_spec": feature_spec,
        "outputs": {
            "selection": str(fold_output_root / "gain_bias_alpha_selection.csv"),
            "selection_metrics": str(fold_output_root / "selection_metrics.csv"),
            "outer_test_metrics": str(fold_output_root / "outer_test_metrics.csv"),
            "outer_test_per_log_metrics": str(fold_output_root / "outer_test_per_log_metrics.csv"),
            "test_predictions": str(fold_output_root / "test_predictions.parquet"),
        },
    }
    _write_json(fold_output_root / "manifest.json", manifest)
    return test_rows, per_log_rows, selection_metric_rows, manifest


def _summary_by_fold(metrics: pd.DataFrame) -> pd.DataFrame:
    value_columns = ("rmse", "mae", "bias", "r2", "corr")
    grouped = metrics.groupby(["model", "target"], dropna=False)
    summary = grouped.agg(folds=("outer_fold", "nunique")).reset_index()
    for value in value_columns:
        stats = grouped[value].agg(["mean", "std", "min", "max"]).reset_index()
        stats = stats.rename(
            columns={
                "mean": f"{value}_mean",
                "std": f"{value}_std",
                "min": f"{value}_min",
                "max": f"{value}_max",
            }
        )
        summary = summary.merge(stats, on=["model", "target"], how="left")
    order = {
        "nominal_prior_only": 0,
        "nominal_gain_bias": 1,
        "shaped_prior_only": 2,
        "shaped_gain_bias": 3,
    }
    summary["model_order"] = summary["model"].map(order).fillna(99)
    return summary.sort_values(["model_order", "target"]).drop(columns=["model_order"]).reset_index(drop=True)


def _display_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in ("nominal_prior_only", "nominal_gain_bias", "shaped_prior_only", "shaped_gain_bias"):
        model_summary = summary.loc[summary["model"].eq(model)]
        if model_summary.empty:
            continue
        row: dict[str, object] = {"model": model}
        for target, prefix in (("fx_b", "fx"), ("fz_b", "fz"), ("fx_fz_mean", "mean")):
            item = model_summary.loc[model_summary["target"].eq(target)]
            if item.empty:
                continue
            item = item.iloc[0]
            row[f"{prefix}_rmse_mean"] = float(item["rmse_mean"])
            row[f"{prefix}_rmse_std"] = float(item["rmse_std"])
            row[f"{prefix}_r2_mean"] = float(item["r2_mean"])
            row[f"{prefix}_r2_std"] = float(item["r2_std"])
        rows.append(row)
    return pd.DataFrame(rows)


def run(
    *,
    splits_root: Path,
    output_root: Path,
    priors: dict[str, Path],
    folds: list[int],
    alphas: tuple[float, ...],
    force: bool,
) -> dict[str, object]:
    _prepare_output(output_root, force=force)
    all_test_rows: list[dict[str, object]] = []
    all_per_log_rows: list[dict[str, object]] = []
    all_selection_metric_rows: list[dict[str, object]] = []
    fold_manifests: dict[str, object] = {}

    prior_candidates = pd.DataFrame(
        [{"prior_name": name, "prior_root": str(root), "exists": root.exists()} for name, root in priors.items()]
    )
    prior_candidates.to_csv(output_root / "prior_candidates.csv", index=False)
    missing_priors = prior_candidates.loc[~prior_candidates["exists"], "prior_root"].tolist()
    if missing_priors:
        raise FileNotFoundError(f"missing prior roots: {missing_priors}")

    for prior_name, prior_root in priors.items():
        prior_all = _read_prior_all(prior_root)
        for fold in folds:
            fold_root = splits_root / f"fold_{fold}"
            if not fold_root.exists():
                raise FileNotFoundError(fold_root)
            fold_output_root = output_root / prior_name / f"fold_{fold}"
            test_rows, per_log_rows, selection_metric_rows, manifest = _select_alpha_and_evaluate(
                prior_name=prior_name,
                prior_root=prior_root,
                fold_root=fold_root,
                fold_output_root=fold_output_root,
                prior_all=prior_all,
                alphas=alphas,
            )
            all_test_rows.extend(test_rows)
            all_per_log_rows.extend(per_log_rows)
            all_selection_metric_rows.extend(selection_metric_rows)
            fold_manifests[f"{prior_name}_fold_{fold}"] = manifest

    test_metrics = pd.DataFrame(all_test_rows)
    per_log_metrics = pd.DataFrame(all_per_log_rows)
    selection_metrics = pd.DataFrame(all_selection_metric_rows)
    summary = _summary_by_fold(test_metrics)
    table = _display_table(summary)
    selected_alpha = (
        test_metrics.loc[test_metrics["model"].str.endswith("_gain_bias"), ["prior_name", "outer_fold", "alpha"]]
        .drop_duplicates()
        .sort_values(["prior_name", "outer_fold"])
        .reset_index(drop=True)
    )

    test_metrics.to_csv(output_root / "experiment1_outer_test_metrics.csv", index=False)
    per_log_metrics.to_csv(output_root / "experiment1_outer_test_per_log_metrics.csv", index=False)
    selection_metrics.to_csv(output_root / "experiment1_selection_metrics.csv", index=False)
    selected_alpha.to_csv(output_root / "experiment1_selected_alpha_by_fold.csv", index=False)
    summary.to_csv(output_root / "experiment1_per_target_summary.csv", index=False)
    table.to_csv(output_root / "experiment1_table_fx_fz_rmse.csv", index=False)

    manifest = {
        "protocol": "six_fold_nested_whole_log_fixed_prior_ablation",
        "experiment": "Experiment 1: no prior-shaping ablation",
        "splits_root": str(splits_root),
        "output_root": str(output_root),
        "priors": {name: str(root) for name, root in priors.items()},
        "folds": folds,
        "alpha_grid": list(alphas),
        "feature_columns": list(PHASE_FREQ_Q_COLUMNS),
        "selection_policy": "for each fixed prior and outer fold, select gain-bias ridge alpha on inner validation logs",
        "final_fit_scope": "train_plus_val_logs_only_within_each_outer_fold",
        "outputs": {
            "prior_candidates": str(output_root / "prior_candidates.csv"),
            "outer_test_metrics": str(output_root / "experiment1_outer_test_metrics.csv"),
            "outer_test_per_log_metrics": str(output_root / "experiment1_outer_test_per_log_metrics.csv"),
            "selection_metrics": str(output_root / "experiment1_selection_metrics.csv"),
            "selected_alpha_by_fold": str(output_root / "experiment1_selected_alpha_by_fold.csv"),
            "per_target_summary": str(output_root / "experiment1_per_target_summary.csv"),
            "table": str(output_root / "experiment1_table_fx_fz_rmse.csv"),
        },
        "fold_manifests": {
            key: value["outputs"] for key, value in fold_manifests.items()
        },
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--folds", default="0,1,2,3,4,5")
    parser.add_argument("--alphas", default=",".join(str(alpha) for alpha in DEFAULT_ALPHA_GRID))
    parser.add_argument("--nominal-prior-root", type=Path, default=DEFAULT_PRIORS["nominal"])
    parser.add_argument("--shaped-prior-root", type=Path, default=DEFAULT_PRIORS["shaped"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = run(
        splits_root=args.splits_root,
        output_root=args.output_root,
        priors={"nominal": args.nominal_prior_root, "shaped": args.shaped_prior_root},
        folds=_parse_folds(args.folds),
        alphas=_parse_alphas(args.alphas),
        force=args.force,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
