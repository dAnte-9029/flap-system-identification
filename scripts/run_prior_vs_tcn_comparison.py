#!/usr/bin/env python3
"""Compare prior gain-bias corrections with TCN residual baselines."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
for path in (PROJECT_SRC, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.run_nested_prior_shaping_ablation_exp1 import (  # noqa: E402
    DEFAULT_ALPHA_GRID,
    DEFAULT_SPLITS_ROOT,
    PHASE_FREQ_Q_COLUMNS,
    _align_prior_to_subset,
    _build_features,
    _fit_gain_bias,
    _load_frames,
    _predict_gain_bias,
    _read_prior_all,
    _with_intercept,
)
from scripts.train_fx_fz_correction import TARGETS  # noqa: E402
from scripts.train_fx_fz_structured_correction import _gain_bias_design  # noqa: E402
from system_identification.training import (  # noqa: E402
    _history_frame,
    _save_training_curves,
    _to_serializable_bundle,
    _with_derived_columns,
    fit_torch_sequence_regressor,
    prediction_metadata_frame_for_bundle,
    resolve_feature_set_columns,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/20260707_prior_vs_tcn_comparison"
DEFAULT_CALIBRATION_TABLE = (
    PROJECT_ROOT
    / "artifacts/20260706_prior_nonlinear_calibration_smooth_lsq_inmem_full"
    / "experiment3_calibrated_parameters_by_fold.csv"
)
DEFAULT_PRIOR_NAME = "nonlinear_regularized"
SPLITS = ("train", "val", "test")
KEY_COLUMNS = ("log_id", "segment_id", "time_s")
CAPACITY_PRESETS: dict[str, dict[str, object]] = {
    "tiny": {"sequence_history_size": 16, "tcn_channels": 16, "tcn_num_blocks": 1, "hidden_sizes": (32,)},
    "small": {"sequence_history_size": 32, "tcn_channels": 32, "tcn_num_blocks": 2, "hidden_sizes": (64,)},
    "base": {"sequence_history_size": 64, "tcn_channels": 128, "tcn_num_blocks": 4, "hidden_sizes": (128, 128)},
}
OOD_PRESETS: dict[str, dict[str, object]] = {
    "airspeed_ge8": {
        "variable": "airspeed",
        "threshold": 8.0,
        "train_relation": "lt",
        "test_relation": "ge",
        "description": "train/val V < 8 m/s, test V >= 8 m/s",
    },
    "alpha_abs_ge20": {
        "variable": "alpha_abs_deg",
        "threshold": 20.0,
        "train_relation": "lt",
        "test_relation": "ge",
        "description": "train/val |alpha_eff| < 20 deg, test |alpha_eff| >= 20 deg",
    },
}


def _parse_folds(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_alphas(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def _parse_string_tuple(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepare_output(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(f"output exists; pass --force to overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def select_training_fraction(frame: pd.DataFrame, *, fraction: float, seed: int) -> pd.DataFrame:
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if float(fraction) >= 1.0:
        return frame.copy().reset_index(drop=True)
    missing = [column for column in ("log_id", "segment_id", "time_s") if column not in frame.columns]
    if missing:
        raise ValueError(f"frame missing columns required for group subsampling: {missing}")

    keyed = frame.copy()
    keyed["log_id"] = keyed["log_id"].astype(str)
    keyed["segment_id"] = pd.to_numeric(keyed["segment_id"], errors="raise").astype("int64")
    groups = keyed.loc[:, ["log_id", "segment_id"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    target_rows = max(1, int(round(len(keyed) * float(fraction))))
    selected_keys: list[tuple[str, int]] = []
    selected_rows = 0
    for group_index in rng.permutation(len(groups)):
        group = groups.iloc[int(group_index)]
        log_id = str(group["log_id"])
        segment_id = int(group["segment_id"])
        selected_keys.append((log_id, segment_id))
        selected_rows += int(((keyed["log_id"] == log_id) & (keyed["segment_id"] == segment_id)).sum())
        if selected_rows >= target_rows:
            break

    selected = pd.DataFrame(selected_keys, columns=["log_id", "segment_id"])
    out = keyed.merge(selected, on=["log_id", "segment_id"], how="inner", sort=False)
    return out.sort_values(["log_id", "segment_id", "time_s"], kind="mergesort").reset_index(drop=True)


def expand_experiment_configs(
    *,
    experiment: str,
    train_fractions: tuple[float, ...],
    capacity_presets: tuple[str, ...],
    ood_presets: tuple[str, ...] = ("none",),
    prior_loss_weights: tuple[float, ...] = (0.0,),
) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    resolved_ood_presets = ood_presets if experiment == "ood" else ("none",)
    resolved_prior_loss_weights = prior_loss_weights if experiment == "prior_anchor" else (0.0,)
    for fraction in train_fractions:
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError("train fractions must be in (0, 1]")
        for preset in capacity_presets:
            if preset not in CAPACITY_PRESETS:
                raise ValueError(f"unknown capacity preset: {preset}")
            for ood_preset in resolved_ood_presets:
                if ood_preset != "none" and ood_preset not in OOD_PRESETS:
                    raise ValueError(f"unknown OOD preset: {ood_preset}")
                for prior_loss_weight in resolved_prior_loss_weights:
                    if not math.isfinite(float(prior_loss_weight)) or float(prior_loss_weight) < 0.0:
                        raise ValueError("prior loss weights must be finite and nonnegative")
                    fraction_tag = f"{float(fraction):g}".replace(".", "p")
                    ood_tag = "" if ood_preset == "none" else f"__{ood_preset}"
                    prior_loss_tag = (
                        f"__lambda_{float(prior_loss_weight):g}".replace(".", "p")
                        if experiment == "prior_anchor"
                        else ""
                    )
                    configs.append(
                        {
                            "experiment": experiment,
                            "config_id": f"{experiment}__frac_{fraction_tag}__{preset}{ood_tag}{prior_loss_tag}",
                            "train_fraction": float(fraction),
                            "capacity_preset": preset,
                            "ood_preset": ood_preset,
                            "ood_description": OOD_PRESETS.get(ood_preset, {}).get("description", "none"),
                            "prior_loss_weight": float(prior_loss_weight),
                            **CAPACITY_PRESETS[preset],
                        }
                    )
    return configs


def _ood_variable_values(frame: pd.DataFrame, *, variable: str) -> np.ndarray:
    if variable == "airspeed":
        column = "airspeed_validated.true_airspeed_m_s"
        if column not in frame.columns:
            raise ValueError(f"OOD airspeed split requires column {column}")
        return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if variable == "alpha_abs_deg":
        derived = _with_derived_columns(frame)
        alpha = _alpha_eff_deg(derived)
        return np.abs(alpha)
    raise ValueError(f"unknown OOD variable: {variable}")


def _relation_mask(values: np.ndarray, *, relation: str, threshold: float) -> np.ndarray:
    if relation == "lt":
        return values < float(threshold)
    if relation == "ge":
        return values >= float(threshold)
    raise ValueError(f"unknown OOD relation: {relation}")


def apply_ood_split_filter(frames: dict[str, pd.DataFrame], *, ood_preset: str) -> dict[str, pd.DataFrame]:
    if ood_preset == "none":
        return {split: frame.copy().reset_index(drop=True) for split, frame in frames.items()}
    if ood_preset not in OOD_PRESETS:
        raise ValueError(f"unknown OOD preset: {ood_preset}")
    spec = OOD_PRESETS[ood_preset]
    out: dict[str, pd.DataFrame] = {}
    for split, frame in frames.items():
        values = _ood_variable_values(frame, variable=str(spec["variable"]))
        relation = str(spec["test_relation"] if split == "test" else spec["train_relation"])
        mask = _relation_mask(values, relation=relation, threshold=float(spec["threshold"]))
        mask &= np.isfinite(values)
        filtered = frame.loc[mask].copy().reset_index(drop=True)
        if filtered.empty:
            raise ValueError(f"OOD preset {ood_preset} produced empty {split} split")
        out[split] = filtered
    return out


def _ridge_fit(design: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(design, dtype=float)
    target = np.asarray(y, dtype=float)
    penalty = float(alpha) * np.eye(x.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ target)


def _with_constant(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.column_stack([np.ones(len(arr), dtype=float), arr])


def fit_global_gain_bias(prior: np.ndarray, truth: np.ndarray, *, alpha: float) -> np.ndarray:
    """Fit per-axis y_j = g_j prior_j + b_j."""
    coefs = []
    for idx in range(prior.shape[1]):
        design = _with_constant(prior[:, idx])
        coefs.append(_ridge_fit(design, truth[:, idx], alpha=float(alpha)))
    return np.vstack(coefs)


def predict_global_gain_bias(model: np.ndarray, prior: np.ndarray) -> np.ndarray:
    pred = np.zeros((len(prior), prior.shape[1]), dtype=float)
    for idx in range(prior.shape[1]):
        pred[:, idx] = _with_constant(prior[:, idx]) @ model[idx]
    return pred


def fit_matrix_gain_bias(prior: np.ndarray, truth: np.ndarray, *, alpha: float) -> np.ndarray:
    """Fit y = A prior + b with cross-axis coupling."""
    return _ridge_fit(_with_constant(prior), truth, alpha=float(alpha))


def predict_matrix_gain_bias(model: np.ndarray, prior: np.ndarray) -> np.ndarray:
    return _with_constant(prior) @ model


def _select_linear_alpha(
    *,
    train_prior: np.ndarray,
    train_truth: np.ndarray,
    val_prior: np.ndarray,
    val_truth: np.ndarray,
    alphas: tuple[float, ...],
    fit_fn,
    predict_fn,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for alpha in alphas:
        model = fit_fn(train_prior, train_truth, alpha=float(alpha))
        pred = predict_fn(model, val_prior)
        row = overall_metric_row(model="candidate", fold=-1, frame=pd.DataFrame(index=np.arange(len(val_truth))), truth=val_truth, pred=pred)
        rows.append({"alpha": float(alpha), "selection_rmse_force_norm": row["rmse_force_norm"]})
    table = pd.DataFrame(rows).sort_values(["selection_rmse_force_norm", "alpha"], kind="mergesort").reset_index(drop=True)
    return float(table.iloc[0]["alpha"]), table


def _finite_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y, dtype=float)
    estimate = np.asarray(pred, dtype=float)
    mask = np.isfinite(truth) & np.isfinite(estimate)
    if int(mask.sum()) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "bias": np.nan}
    err = estimate[mask] - truth[mask]
    ss_res = float(np.sum(err * err))
    centered = truth[mask] - float(np.mean(truth[mask]))
    ss_tot = float(np.sum(centered * centered))
    return {
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else np.nan,
        "bias": float(np.mean(err)),
    }


def overall_metric_row(
    *,
    model: str,
    fold: int,
    frame: pd.DataFrame,
    truth: np.ndarray,
    pred: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(truth, dtype=float)
    p = np.asarray(pred, dtype=float)
    err = p - y
    row: dict[str, Any] = {
        "outer_fold": int(fold),
        "model": model,
        "n_samples": int(len(y)),
        "n_logs": int(frame["log_id"].nunique()) if "log_id" in frame.columns else np.nan,
        "rmse_force_norm": float(np.sqrt(np.mean(np.sum(err * err, axis=1)))) if len(y) else np.nan,
    }
    for idx, target in enumerate(TARGETS):
        metrics = _finite_metrics(y[:, idx], p[:, idx])
        suffix = target
        row[f"rmse_{suffix}"] = metrics["rmse"]
        row[f"mae_{suffix}"] = metrics["mae"]
        row[f"r2_{suffix}"] = metrics["r2"]
        row[f"bias_{suffix}"] = metrics["bias"]
    return row


def _error_metric_subset(truth: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    if len(truth) == 0:
        return {
            "n_samples": 0,
            "rmse_fx_b": np.nan,
            "rmse_fz_b": np.nan,
            "rmse_force_norm": np.nan,
            "mae_fx_b": np.nan,
            "mae_fz_b": np.nan,
            "mean_residual_fx_b": np.nan,
            "mean_residual_fz_b": np.nan,
        }
    err = pred - truth
    return {
        "n_samples": int(len(truth)),
        "rmse_fx_b": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "rmse_fz_b": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "rmse_force_norm": float(np.sqrt(np.mean(np.sum(err * err, axis=1)))),
        "mae_fx_b": float(np.mean(np.abs(err[:, 0]))),
        "mae_fz_b": float(np.mean(np.abs(err[:, 1]))),
        "mean_residual_fx_b": float(np.mean(truth[:, 0] - pred[:, 0])),
        "mean_residual_fz_b": float(np.mean(truth[:, 1] - pred[:, 1])),
    }


def _safe_numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _alpha_eff_deg(frame: pd.DataFrame) -> np.ndarray:
    if "alpha_rad" in frame.columns:
        return np.rad2deg(_safe_numeric(frame, "alpha_rad"))
    if {"relative_air_velocity_b.x", "relative_air_velocity_b.z"}.issubset(frame.columns):
        vx = _safe_numeric(frame, "relative_air_velocity_b.x")
        vz = _safe_numeric(frame, "relative_air_velocity_b.z")
        return np.rad2deg(np.arctan2(-vz, vx))
    if "pitch_rad" in frame.columns:
        return np.rad2deg(_safe_numeric(frame, "pitch_rad"))
    return np.full(len(frame), np.nan, dtype=float)


def _phase_bin_index(frame: pd.DataFrame, phase_bins: int) -> np.ndarray:
    phase = np.mod(_safe_numeric(frame, "phase_corrected_rad", default=0.0), 2.0 * np.pi)
    return np.floor(phase / (2.0 * np.pi) * int(phase_bins)).astype(int).clip(0, int(phase_bins) - 1)


def _stroke_labels(frame: pd.DataFrame) -> np.ndarray:
    if "wing_stroke_direction" in frame.columns:
        raw = frame["wing_stroke_direction"]
        if not pd.api.types.is_numeric_dtype(raw):
            labels = raw.astype(str).str.lower().to_numpy(dtype=object)
            if np.isin(labels, ["upstroke", "downstroke"]).all():
                return labels
        direction = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
        if np.isfinite(direction).any():
            return np.where(direction >= 0.0, "downstroke", "upstroke")
    phase = np.mod(_safe_numeric(frame, "phase_corrected_rad", default=0.0), 2.0 * np.pi)
    return np.where(np.cos(phase) >= 0.0, "downstroke", "upstroke")


def condition_bin_table(
    *,
    frame: pd.DataFrame,
    truth: np.ndarray,
    pred: np.ndarray,
    model: str,
    fold: int,
    phase_bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_rows(family: str, labels: np.ndarray, indices: np.ndarray | None = None) -> None:
        unique_labels = list(dict.fromkeys(labels.tolist()))
        for label in unique_labels:
            mask = labels == label
            row: dict[str, Any] = {
                "outer_fold": int(fold),
                "model": model,
                "bin_family": family,
                "bin_label": str(label),
                "bin_index": int(indices[mask][0]) if indices is not None and np.any(mask) else np.nan,
            }
            row.update(_error_metric_subset(truth[mask], pred[mask]))
            rows.append(row)

    airspeed = _safe_numeric(frame, "airspeed_validated.true_airspeed_m_s")
    air_edges = np.array([0.0, 4.0, 6.0, 8.0, np.inf])
    air_labels = np.array(["[0,4)", "[4,6)", "[6,8)", "[8,inf)"], dtype=object)
    air_idx = np.digitize(airspeed, air_edges[1:-1], right=False)
    add_rows("airspeed", air_labels[air_idx], air_idx)

    alpha = _alpha_eff_deg(frame)
    alpha_edges = np.array([-np.inf, -10.0, 0.0, 10.0, 20.0, np.inf])
    alpha_labels = np.array(["[-inf,-10)", "[-10,0)", "[0,10)", "[10,20)", "[20,inf)"], dtype=object)
    alpha_idx = np.digitize(alpha, alpha_edges[1:-1], right=False)
    valid_alpha = np.isfinite(alpha)
    if np.any(valid_alpha):
        add_rows("alpha_eff", alpha_labels[alpha_idx][valid_alpha], alpha_idx[valid_alpha])

    phase_idx = _phase_bin_index(frame, phase_bins=phase_bins)
    phase_labels = np.array([f"{idx:02d}" for idx in phase_idx], dtype=object)
    add_rows("phase", phase_labels, phase_idx)

    stroke = _stroke_labels(frame)
    add_rows("stroke", stroke)
    return pd.DataFrame(rows)


def _key_frame(frame: pd.DataFrame, *, row_column: str | None = None) -> pd.DataFrame:
    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"frame missing key columns: {missing}")
    out = pd.DataFrame(
        {
            "log_id": frame["log_id"].astype(str).to_numpy(),
            "segment_id": pd.to_numeric(frame["segment_id"], errors="raise").astype("int64").to_numpy(),
            "time_key": np.round(pd.to_numeric(frame["time_s"], errors="raise").astype(float).to_numpy() * 100.0).astype("int64"),
        }
    )
    if row_column is not None:
        out[row_column] = np.arange(len(frame), dtype=np.int64)
    return out


def _align_array_to_reference(source_frame: pd.DataFrame, values: np.ndarray, reference_frame: pd.DataFrame) -> np.ndarray:
    source_keys = _key_frame(source_frame)
    reference_keys = _key_frame(reference_frame, row_column="__ref_row")
    payload = pd.concat(
        [
            source_keys.reset_index(drop=True),
            pd.DataFrame(values, columns=[f"v{idx}" for idx in range(values.shape[1])]),
        ],
        axis=1,
    )
    merged = reference_keys.merge(payload, on=["log_id", "segment_id", "time_key"], how="left", validate="one_to_one", sort=False)
    value_columns = [f"v{idx}" for idx in range(values.shape[1])]
    if merged[value_columns].isna().any().any():
        raise ValueError("failed to align predictions to common evaluation rows")
    merged = merged.sort_values("__ref_row", kind="mergesort")
    return merged.loc[:, value_columns].to_numpy(dtype=float)


def _subset_frame_to_reference(source_frame: pd.DataFrame, reference_frame: pd.DataFrame) -> pd.DataFrame:
    source = _key_frame(source_frame, row_column="__source_row")
    reference = _key_frame(reference_frame, row_column="__ref_row")
    merged = reference.merge(source, on=["log_id", "segment_id", "time_key"], how="left", validate="one_to_one", sort=False)
    if merged["__source_row"].isna().any():
        raise ValueError("failed to align frame to common evaluation rows")
    rows = merged.sort_values("__ref_row", kind="mergesort")["__source_row"].to_numpy(dtype=int)
    return source_frame.iloc[rows].reset_index(drop=True)


def _make_prediction_frame(frame: pd.DataFrame, truth: np.ndarray, prior: np.ndarray, pred: np.ndarray, *, model: str, fold: int) -> pd.DataFrame:
    derived = _with_derived_columns(frame)
    keep = [
        column
        for column in (
            "log_id",
            "segment_id",
            "time_s",
            "phase_corrected_rad",
            "wing_stroke_direction",
            "airspeed_validated.true_airspeed_m_s",
            "alpha_rad",
            "pitch_rad",
            "velocity_b.x",
            "velocity_b.z",
            "relative_air_velocity_b.x",
            "relative_air_velocity_b.z",
        )
        if column in derived.columns
    ]
    out = derived.loc[:, keep].reset_index(drop=True).copy()
    out["outer_fold"] = int(fold)
    out["model"] = model
    for idx, target in enumerate(TARGETS):
        out[f"label_{target}"] = truth[:, idx]
        out[f"prior_{target}"] = prior[:, idx]
        out[f"pred_{target}"] = pred[:, idx]
        out[f"residual_{target}"] = truth[:, idx] - pred[:, idx]
        out[f"prior_residual_{target}"] = truth[:, idx] - prior[:, idx]
    return out


def _save_bundle(output_dir: Path, bundle: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_to_serializable_bundle(bundle), output_dir / "model_bundle.pt")
    _history_frame(bundle["history"]).to_csv(output_dir / "history.csv", index=False)
    _save_training_curves(_history_frame(bundle["history"]), output_dir / "training_curves.png")
    _write_json(
        output_dir / "training_config.json",
        {
            "model_type": bundle["model_type"],
            "target_columns": bundle["target_columns"],
            "feature_columns": bundle["feature_columns"],
            "sequence_feature_columns": bundle["sequence_feature_columns"],
            "current_feature_columns": bundle["current_feature_columns"],
            "sequence_history_size": bundle["sequence_history_size"],
            "best_epoch": bundle["best_epoch"],
            "best_val_loss": bundle["best_val_loss"],
            "device_type": bundle["device_type"],
            "prior_target_columns": bundle.get("prior_target_columns", []),
            "prior_loss_weight": bundle.get("prior_loss_weight", 0.0),
        },
    )


def _train_tcn(
    *,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    target_columns: list[str],
    feature_columns: list[str],
    output_dir: Path,
    random_seed: int,
    args: argparse.Namespace,
    prior_target_columns: list[str] | None = None,
    prior_loss_weight: float = 0.0,
) -> dict[str, Any]:
    bundle = fit_torch_sequence_regressor(
        train_frame=train_frame,
        val_frame=val_frame,
        feature_columns=feature_columns,
        target_columns=target_columns,
        prior_target_columns=prior_target_columns,
        prior_loss_weight=prior_loss_weight,
        model_type="causal_tcn",
        hidden_sizes=tuple(args.hidden_sizes),
        dropout=float(args.dropout),
        batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        early_stopping_patience=int(args.early_stopping_patience),
        device=str(args.device),
        random_seed=int(random_seed),
        num_workers=int(args.num_workers),
        use_amp=not bool(args.disable_amp),
        loss_type=str(args.loss_type),
        huber_delta=float(args.huber_delta),
        sequence_history_size=int(args.sequence_history_size),
        sequence_feature_mode=str(args.sequence_feature_mode),
        current_feature_mode=str(args.current_feature_mode),
        tcn_channels=int(args.tcn_channels),
        tcn_num_blocks=int(args.tcn_num_blocks),
        tcn_kernel_size=int(args.tcn_kernel_size),
        lr_scheduler=args.lr_scheduler,
        lr_warmup_ratio=float(args.lr_warmup_ratio),
        gradient_clip_norm=args.gradient_clip_norm,
        ema_decay=float(args.ema_decay),
    )
    _save_bundle(output_dir, bundle)
    return bundle


def _tcn_residual_prediction(bundle: dict[str, Any], frame: pd.DataFrame, *, device: str, batch_size: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    meta = prediction_metadata_frame_for_bundle(bundle, frame, split_name="test", batch_size=batch_size, device=device)
    truth = np.column_stack([meta[f"label_{target}"].to_numpy(dtype=float) for target in TARGETS])
    prior = np.column_stack([meta[f"prior_{target}"].to_numpy(dtype=float) for target in TARGETS])
    residual = np.column_stack([meta[f"pred_residual_{target}"].to_numpy(dtype=float) for target in TARGETS])
    return meta, truth, prior, prior + residual


def _tcn_pure_prediction(bundle: dict[str, Any], frame: pd.DataFrame, *, device: str, batch_size: int) -> tuple[pd.DataFrame, np.ndarray]:
    meta = prediction_metadata_frame_for_bundle(bundle, frame, split_name="test", batch_size=batch_size, device=device)
    pred = np.column_stack([meta[f"pred_{target}"].to_numpy(dtype=float) for target in TARGETS])
    return meta, pred


def _select_conditioned_gain_bias(
    frames: dict[str, pd.DataFrame],
    features: dict[str, pd.DataFrame],
    alphas: tuple[float, ...],
) -> tuple[float, pd.DataFrame]:
    rows = []
    y_val = np.column_stack([frames["val"][f"label_{target}"].to_numpy(dtype=float) for target in TARGETS])
    for alpha in alphas:
        models = _fit_gain_bias(frames["train"], features["train"], alpha=float(alpha))
        pred = _predict_gain_bias(models, frames["val"], features["val"])
        row = overall_metric_row(model="conditioned_candidate", fold=-1, frame=frames["val"], truth=y_val, pred=pred)
        rows.append({"alpha": float(alpha), "selection_rmse_force_norm": row["rmse_force_norm"]})
    table = pd.DataFrame(rows).sort_values(["selection_rmse_force_norm", "alpha"], kind="mergesort").reset_index(drop=True)
    return float(table.iloc[0]["alpha"]), table


def _residual_spectrum_rows(frame: pd.DataFrame, truth: np.ndarray, pred: np.ndarray, *, model: str, fold: int, sample_rate_hz: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    work = frame.loc[:, [column for column in ("log_id", "segment_id") if column in frame.columns]].copy()
    for idx, target in enumerate(TARGETS):
        work["residual"] = truth[:, idx] - pred[:, idx]
        for (log_id, segment_id), group in work.groupby(["log_id", "segment_id"], sort=True):
            values = group["residual"].to_numpy(dtype=float)
            if len(values) < 32:
                continue
            values = values - float(np.mean(values))
            window = np.hanning(len(values))
            denom = float(sample_rate_hz * np.sum(window * window))
            freq = np.fft.rfftfreq(len(values), d=1.0 / sample_rate_hz)
            psd = np.square(np.abs(np.fft.rfft(values * window))) / max(denom, 1.0e-12)
            for f, p in zip(freq, psd):
                rows.append(
                    {
                        "outer_fold": int(fold),
                        "model": model,
                        "target": target,
                        "log_id": str(log_id),
                        "segment_id": int(segment_id),
                        "frequency_hz": float(f),
                        "residual_psd": float(p),
                    }
                )
    return rows


def _save_residual_plots(predictions: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        residual_col = f"residual_{target}"
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for model, group in predictions.groupby("model", sort=True):
            sample = group.iloc[:: max(1, len(group) // 5000)]
            axes[0, 0].plot(np.arange(len(sample)), sample[residual_col], linewidth=0.8, alpha=0.75, label=model)
            if "phase_corrected_rad" in sample.columns:
                axes[0, 1].scatter(sample["phase_corrected_rad"], sample[residual_col], s=2, alpha=0.25, label=model)
            if "airspeed_validated.true_airspeed_m_s" in sample.columns:
                axes[0, 2].scatter(sample["airspeed_validated.true_airspeed_m_s"], sample[residual_col], s=2, alpha=0.25, label=model)
            alpha = _alpha_eff_deg(sample)
            if np.isfinite(alpha).any():
                axes[1, 0].scatter(alpha, sample[residual_col], s=2, alpha=0.25, label=model)
            axes[1, 1].hist(group[residual_col].to_numpy(dtype=float), bins=80, alpha=0.35, density=True, label=model)
        axes[0, 0].set_title("Residual vs sample")
        axes[0, 1].set_title("Residual vs phase")
        axes[0, 2].set_title("Residual vs airspeed")
        axes[1, 0].set_title("Residual vs alpha")
        axes[1, 1].set_title("Residual histogram")
        axes[1, 2].axis("off")
        for ax in axes.ravel()[:5]:
            ax.grid(True, alpha=0.25)
        axes[1, 2].legend(*axes[0, 0].get_legend_handles_labels(), loc="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(output_dir / f"residual_diagnostics_{target}.png", dpi=180)
        plt.close(fig)


def _fold_prior_roots(calibration_table: Path, prior_name: str) -> dict[int, Path]:
    table = pd.read_csv(calibration_table)
    subset = table.loc[table["prior_name"].eq(prior_name)].copy()
    if subset.empty:
        raise ValueError(f"prior_name {prior_name!r} not found in {calibration_table}")
    return {int(row.outer_fold): Path(str(row.exact_prior_root)) for row in subset.itertuples(index=False)}


def _evaluate_fold(
    *,
    fold: int,
    fold_root: Path,
    prior_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    config: dict[str, object],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fold_output = output_root / f"fold_{fold}"
    fold_output.mkdir(parents=True, exist_ok=True)
    prior_all = _read_prior_all(prior_root)
    frames = _load_frames(fold_root, prior_all)
    frames = apply_ood_split_filter(frames, ood_preset=str(config.get("ood_preset", "none")))
    include_linear_baselines = bool(args.include_linear_baselines or config["experiment"] == "baseline")
    features: dict[str, pd.DataFrame] = {}
    feature_spec: dict[str, object] = {}
    if include_linear_baselines:
        features, feature_spec = _build_features(frames)

    for split in SPLITS:
        for target in TARGETS:
            frames[split][f"residual_{target}"] = frames[split][f"label_{target}"] - frames[split][f"prior_{target}"]

    tcn_args = copy.copy(args)
    tcn_args.sequence_history_size = int(config["sequence_history_size"])
    tcn_args.tcn_channels = int(config["tcn_channels"])
    tcn_args.tcn_num_blocks = int(config["tcn_num_blocks"])
    tcn_args.hidden_sizes = tuple(int(value) for value in config["hidden_sizes"])
    train_fraction = float(config["train_fraction"])
    tcn_train_frame = select_training_fraction(
        frames["train"],
        fraction=train_fraction,
        seed=int(args.random_seed) + 10000 * int(fold) + int(round(train_fraction * 1000.0)),
    )

    base_features = resolve_feature_set_columns(args.feature_set)
    residual_features = list(dict.fromkeys([*base_features, "prior_fx_b", "prior_fz_b"]))
    pure_features = list(base_features)

    residual_bundle = _train_tcn(
        train_frame=tcn_train_frame,
        val_frame=frames["val"],
        target_columns=[f"residual_{target}" for target in TARGETS],
        feature_columns=residual_features,
        output_dir=fold_output / "tcn_residual",
        random_seed=int(args.random_seed) + fold,
        args=tcn_args,
    )
    pure_bundle = _train_tcn(
        train_frame=tcn_train_frame,
        val_frame=frames["val"],
        target_columns=list(TARGETS),
        feature_columns=pure_features,
        output_dir=fold_output / "pure_tcn",
        random_seed=int(args.random_seed) + 1000 + fold,
        args=tcn_args,
    )
    prior_anchor_bundle = None
    if str(config["experiment"]) == "prior_anchor":
        prior_anchor_bundle = _train_tcn(
            train_frame=tcn_train_frame,
            val_frame=frames["val"],
            target_columns=list(TARGETS),
            prior_target_columns=[f"prior_{target}" for target in TARGETS],
            prior_loss_weight=float(config["prior_loss_weight"]),
            feature_columns=pure_features,
            output_dir=fold_output / "prior_anchor_tcn",
            random_seed=int(args.random_seed) + 1000 + fold,
            args=tcn_args,
        )

    common_meta, tcn_truth, tcn_prior, tcn_residual_pred = _tcn_residual_prediction(
        residual_bundle,
        frames["test"],
        device=str(args.device),
        batch_size=int(args.batch_size),
    )
    pure_meta, pure_pred_full = _tcn_pure_prediction(
        pure_bundle,
        frames["test"],
        device=str(args.device),
        batch_size=int(args.batch_size),
    )
    pure_pred = _align_array_to_reference(pure_meta, pure_pred_full, common_meta)
    prior_anchor_pred = None
    if prior_anchor_bundle is not None:
        prior_anchor_meta, prior_anchor_pred_full = _tcn_pure_prediction(
            prior_anchor_bundle,
            frames["test"],
            device=str(args.device),
            batch_size=int(args.batch_size),
        )
        prior_anchor_pred = _align_array_to_reference(prior_anchor_meta, prior_anchor_pred_full, common_meta)
    common_frame = _subset_frame_to_reference(frames["test"], common_meta)
    truth = tcn_truth
    prior = tcn_prior

    test_prior_full = np.column_stack([frames["test"][f"prior_{target}"].to_numpy(dtype=float) for target in TARGETS])
    test_prior = _align_array_to_reference(frames["test"], test_prior_full, common_meta)

    predictions = {
        "TCN residual": tcn_residual_pred,
        "Pure TCN": pure_pred,
        **({"Prior-anchor TCN": prior_anchor_pred} if prior_anchor_pred is not None else {}),
    }
    selected_alphas: dict[str, float] = {}
    if include_linear_baselines:
        train_prior = np.column_stack([frames["train"][f"prior_{target}"].to_numpy(dtype=float) for target in TARGETS])
        val_prior = np.column_stack([frames["val"][f"prior_{target}"].to_numpy(dtype=float) for target in TARGETS])
        dev_prior = np.vstack([train_prior, val_prior])
        train_truth = np.column_stack([frames["train"][f"label_{target}"].to_numpy(dtype=float) for target in TARGETS])
        val_truth = np.column_stack([frames["val"][f"label_{target}"].to_numpy(dtype=float) for target in TARGETS])
        dev_truth = np.vstack([train_truth, val_truth])

        global_alpha, global_selection = _select_linear_alpha(
            train_prior=train_prior,
            train_truth=train_truth,
            val_prior=val_prior,
            val_truth=val_truth,
            alphas=args.alphas,
            fit_fn=fit_global_gain_bias,
            predict_fn=predict_global_gain_bias,
        )
        global_model = fit_global_gain_bias(dev_prior, dev_truth, alpha=global_alpha)
        global_pred = predict_global_gain_bias(global_model, test_prior)
        selected_alphas["global_gain_bias"] = float(global_alpha)

        matrix_alpha, matrix_selection = _select_linear_alpha(
            train_prior=train_prior,
            train_truth=train_truth,
            val_prior=val_prior,
            val_truth=val_truth,
            alphas=args.alphas,
            fit_fn=fit_matrix_gain_bias,
            predict_fn=predict_matrix_gain_bias,
        )
        matrix_model = fit_matrix_gain_bias(dev_prior, dev_truth, alpha=matrix_alpha)
        matrix_pred = predict_matrix_gain_bias(matrix_model, test_prior)
        selected_alphas["matrix_gain_bias"] = float(matrix_alpha)

        conditioned_alpha, conditioned_selection = _select_conditioned_gain_bias(frames, features, args.alphas)
        dev_frame = pd.concat([frames["train"], frames["val"]], ignore_index=True)
        dev_features = pd.concat([features["train"], features["val"]], ignore_index=True)
        conditioned_models = _fit_gain_bias(dev_frame, dev_features, alpha=conditioned_alpha)
        conditioned_full = _predict_gain_bias(conditioned_models, frames["test"], features["test"])
        conditioned_pred = _align_array_to_reference(frames["test"], conditioned_full, common_meta)
        selected_alphas["conditioned_gain_bias"] = float(conditioned_alpha)

        predictions = {
            "Raw prior": test_prior,
            "Global gain-bias": global_pred,
            "Matrix gain-bias": matrix_pred,
            "Conditioned gain-bias": conditioned_pred,
            **predictions,
        }
        global_selection.assign(model="Global gain-bias").to_csv(fold_output / "global_gain_bias_alpha_selection.csv", index=False)
        matrix_selection.assign(model="Matrix gain-bias").to_csv(fold_output / "matrix_gain_bias_alpha_selection.csv", index=False)
        conditioned_selection.assign(model="Conditioned gain-bias").to_csv(
            fold_output / "conditioned_gain_bias_alpha_selection.csv",
            index=False,
        )

    metric_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    prediction_frames = []
    config_meta = {
        "experiment": str(config["experiment"]),
        "config_id": str(config["config_id"]),
        "train_fraction": float(config["train_fraction"]),
        "capacity_preset": str(config["capacity_preset"]),
        "ood_preset": str(config.get("ood_preset", "none")),
        "prior_loss_weight": float(config.get("prior_loss_weight", 0.0)),
        "sequence_history_size": int(config["sequence_history_size"]),
        "tcn_channels": int(config["tcn_channels"]),
        "tcn_num_blocks": int(config["tcn_num_blocks"]),
    }
    for model_name, pred in predictions.items():
        metric_rows.append(
            {
                **config_meta,
                **overall_metric_row(model=model_name, fold=fold, frame=common_frame, truth=truth, pred=pred),
            }
        )
        bin_rows.extend(
            {
                **config_meta,
                **row,
            }
            for row in condition_bin_table(
                frame=common_frame,
                truth=truth,
                pred=pred,
                model=model_name,
                fold=fold,
                phase_bins=int(args.phase_bins),
            ).to_dict(orient="records")
        )
        spectrum_rows.extend(
            {
                **config_meta,
                **row,
            }
            for row in _residual_spectrum_rows(
                common_frame,
                truth,
                pred,
                model=model_name,
                fold=fold,
                sample_rate_hz=float(args.sample_rate_hz),
            )
        )
        prediction_frames.append(_make_prediction_frame(common_frame, truth, prior, pred, model=model_name, fold=fold))

    fold_predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_predictions.to_parquet(fold_output / "test_predictions_common_rows.parquet", index=False)
    pd.DataFrame(metric_rows).to_csv(fold_output / "overall_metrics.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(fold_output / "condition_binned_metrics.csv", index=False)
    pd.DataFrame(spectrum_rows).to_csv(fold_output / "residual_spectrum.csv", index=False)
    _save_residual_plots(fold_predictions, fold_output / "plots")

    manifest = {
        "outer_fold": int(fold),
        **config_meta,
        "prior_root": str(prior_root),
        "fold_root": str(fold_root),
        "common_test_rows": int(len(common_frame)),
        "raw_test_rows": int(len(frames["test"])),
        "tcn_train_rows": int(len(tcn_train_frame)),
        "full_train_rows": int(len(frames["train"])),
        "val_rows_after_filter": int(len(frames["val"])),
        "test_rows_after_filter": int(len(frames["test"])),
        "history_size": int(tcn_args.sequence_history_size),
        "feature_set": args.feature_set,
        "residual_tcn_feature_columns": residual_features,
        "pure_tcn_feature_columns": pure_features,
        "prior_anchor_target_columns": [f"prior_{target}" for target in TARGETS] if prior_anchor_bundle is not None else [],
        "prior_anchor_loss_weight": float(config.get("prior_loss_weight", 0.0)) if prior_anchor_bundle is not None else None,
        "conditioned_gain_bias_columns": list(PHASE_FREQ_Q_COLUMNS) if include_linear_baselines else [],
        "selected_alphas": selected_alphas,
        "feature_spec": feature_spec,
        "outputs": {
            "overall_metrics": str(fold_output / "overall_metrics.csv"),
            "condition_binned_metrics": str(fold_output / "condition_binned_metrics.csv"),
            "residual_spectrum": str(fold_output / "residual_spectrum.csv"),
            "test_predictions": str(fold_output / "test_predictions_common_rows.parquet"),
        },
    }
    _write_json(fold_output / "manifest.json", manifest)
    return metric_rows, bin_rows, spectrum_rows, manifest


def _summary_table(overall: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for column in overall.columns if column.startswith(("rmse_", "mae_", "r2_", "bias_"))]
    group_columns = [
        column
        for column in ("experiment", "config_id", "train_fraction", "capacity_preset", "ood_preset", "prior_loss_weight", "sequence_history_size", "tcn_channels", "tcn_num_blocks", "model")
        if column in overall.columns
    ]
    grouped = overall.groupby(group_columns, sort=False, dropna=False)
    rows = []
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row: dict[str, Any] = dict(zip(group_columns, key, strict=True))
        row["folds"] = int(group["outer_fold"].nunique())
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    order = {
        "Raw prior": 0,
        "Global gain-bias": 1,
        "Matrix gain-bias": 2,
        "Conditioned gain-bias": 3,
        "TCN residual": 4,
        "Pure TCN": 5,
        "Prior-anchor TCN": 6,
    }
    out = pd.DataFrame(rows)
    out["order"] = out["model"].map(order).fillna(99)
    sort_columns = [column for column in ("experiment", "train_fraction", "capacity_preset", "ood_preset", "prior_loss_weight", "order") if column in out.columns]
    return out.sort_values(sort_columns).drop(columns="order").reset_index(drop=True)


def _write_tcn_compact_tables(summary: pd.DataFrame, output_root: Path, experiment: str) -> None:
    keep_models = ("TCN residual", "Pure TCN", "Prior-anchor TCN")
    compact_columns = [
        column
        for column in (
            "experiment",
            "config_id",
            "train_fraction",
            "capacity_preset",
            "ood_preset",
            "prior_loss_weight",
            "sequence_history_size",
            "tcn_channels",
            "tcn_num_blocks",
            "model",
            "rmse_fx_b_mean",
            "rmse_fz_b_mean",
            "rmse_force_norm_mean",
            "mae_fx_b_mean",
            "mae_fz_b_mean",
            "r2_fx_b_mean",
            "r2_fz_b_mean",
        )
        if column in summary.columns
    ]
    compact = summary.loc[summary["model"].isin(keep_models), compact_columns].copy()
    if experiment == "ood":
        comparison_name = "ood_tcn_comparison.csv"
        delta_name = "ood_tcn_delta.csv"
    elif experiment == "prior_anchor":
        comparison_name = "prior_anchor_tcn_comparison.csv"
        delta_name = "prior_anchor_tcn_delta.csv"
    elif "sample_efficiency" in experiment:
        comparison_name = "sample_efficiency_tcn_comparison.csv"
        delta_name = "sample_efficiency_tcn_delta.csv"
    else:
        comparison_name = "capacity_tcn_comparison.csv"
        delta_name = "capacity_tcn_delta.csv"
    compact.to_csv(output_root / comparison_name, index=False)

    index_columns = [
        column
        for column in ("experiment", "config_id", "train_fraction", "capacity_preset", "ood_preset", "prior_loss_weight", "sequence_history_size", "tcn_channels", "tcn_num_blocks")
        if column in compact.columns
    ]
    pivot = compact.pivot_table(
        index=index_columns,
        columns="model",
        values=["rmse_fx_b_mean", "rmse_fz_b_mean", "rmse_force_norm_mean", "mae_fx_b_mean", "mae_fz_b_mean", "r2_fx_b_mean", "r2_fz_b_mean"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}__{model}".replace(" ", "_").lower() for metric, model in pivot.columns]
    delta = pivot.reset_index()
    if {"rmse_force_norm_mean__pure_tcn", "rmse_force_norm_mean__tcn_residual"}.issubset(delta.columns):
        delta["delta_rmse_force_norm_pure_minus_prior_tcn"] = (
            delta["rmse_force_norm_mean__pure_tcn"] - delta["rmse_force_norm_mean__tcn_residual"]
        )
    if {"rmse_fx_b_mean__pure_tcn", "rmse_fx_b_mean__tcn_residual"}.issubset(delta.columns):
        delta["delta_rmse_fx_b_pure_minus_prior_tcn"] = delta["rmse_fx_b_mean__pure_tcn"] - delta["rmse_fx_b_mean__tcn_residual"]
    if {"rmse_fz_b_mean__pure_tcn", "rmse_fz_b_mean__tcn_residual"}.issubset(delta.columns):
        delta["delta_rmse_fz_b_pure_minus_prior_tcn"] = delta["rmse_fz_b_mean__pure_tcn"] - delta["rmse_fz_b_mean__tcn_residual"]
    if {"rmse_force_norm_mean__pure_tcn", "rmse_force_norm_mean__prior-anchor_tcn"}.issubset(delta.columns):
        delta["delta_rmse_force_norm_pure_minus_anchor_tcn"] = (
            delta["rmse_force_norm_mean__pure_tcn"] - delta["rmse_force_norm_mean__prior-anchor_tcn"]
        )
    delta.to_csv(output_root / delta_name, index=False)


def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    _prepare_output(output_root, force=bool(args.force))
    prior_roots = _fold_prior_roots(Path(args.calibration_table), str(args.prior_name))
    folds = list(args.folds)
    configs = expand_experiment_configs(
        experiment=str(args.experiment),
        train_fractions=tuple(args.train_fractions),
        capacity_presets=tuple(args.capacity_presets),
        ood_presets=tuple(args.ood_presets),
        prior_loss_weights=tuple(args.prior_loss_weights),
    )

    all_metric_rows: list[dict[str, Any]] = []
    all_bin_rows: list[dict[str, Any]] = []
    all_spectrum_rows: list[dict[str, Any]] = []
    fold_manifests: dict[str, Any] = {}
    for config in configs:
        config_output_root = output_root / str(config["config_id"])
        for fold in folds:
            if fold not in prior_roots:
                raise ValueError(f"missing prior root for fold {fold}")
            metric_rows, bin_rows, spectrum_rows, manifest = _evaluate_fold(
                fold=fold,
                fold_root=Path(args.splits_root) / f"fold_{fold}",
                prior_root=prior_roots[fold],
                output_root=config_output_root,
                args=args,
                config=config,
            )
            all_metric_rows.extend(metric_rows)
            all_bin_rows.extend(bin_rows)
            all_spectrum_rows.extend(spectrum_rows)
            fold_manifests[f"{config['config_id']}_fold_{fold}"] = manifest

    overall = pd.DataFrame(all_metric_rows)
    binned = pd.DataFrame(all_bin_rows)
    spectrum = pd.DataFrame(all_spectrum_rows)
    summary = _summary_table(overall)
    overall.to_csv(output_root / "overall_metrics_by_fold.csv", index=False)
    summary.to_csv(output_root / "overall_metrics_summary.csv", index=False)
    binned.to_csv(output_root / "condition_binned_metrics_by_fold.csv", index=False)
    if not binned.empty:
        bin_group_columns = [
            column
            for column in ("experiment", "config_id", "train_fraction", "capacity_preset", "ood_preset", "prior_loss_weight", "sequence_history_size", "tcn_channels", "tcn_num_blocks", "model", "bin_family", "bin_label")
            if column in binned.columns
        ]
        binned.groupby(bin_group_columns, dropna=False).mean(numeric_only=True).reset_index().to_csv(
            output_root / "condition_binned_metrics_summary.csv",
            index=False,
        )
    spectrum.to_csv(output_root / "residual_spectrum_by_fold.csv", index=False)
    if not spectrum.empty:
        spectrum_group_columns = [
            column
            for column in ("experiment", "config_id", "train_fraction", "capacity_preset", "ood_preset", "prior_loss_weight", "sequence_history_size", "tcn_channels", "tcn_num_blocks", "model", "target", "frequency_hz")
            if column in spectrum.columns
        ]
        spectrum.groupby(spectrum_group_columns, dropna=False)["residual_psd"].mean().reset_index().to_csv(
            output_root / "residual_spectrum_summary.csv",
            index=False,
        )
    if str(args.experiment) in {"sample_efficiency", "capacity", "sample_efficiency_capacity", "ood", "prior_anchor"}:
        _write_tcn_compact_tables(summary, output_root, str(args.experiment))
    _write_json(
        output_root / "manifest.json",
        {
            "experiment": str(args.experiment),
            "splits_root": str(args.splits_root),
            "calibration_table": str(args.calibration_table),
            "prior_name": str(args.prior_name),
            "output_root": str(output_root),
            "folds": folds,
            "configs": configs,
            "model_set": sorted(overall["model"].unique().tolist()) if not overall.empty else [],
            "common_row_policy": "All reported test metrics use rows with complete causal TCN history.",
            "outputs": {
                "overall_metrics_by_fold": str(output_root / "overall_metrics_by_fold.csv"),
                "overall_metrics_summary": str(output_root / "overall_metrics_summary.csv"),
                "condition_binned_metrics_by_fold": str(output_root / "condition_binned_metrics_by_fold.csv"),
                "condition_binned_metrics_summary": str(output_root / "condition_binned_metrics_summary.csv"),
                "residual_spectrum_by_fold": str(output_root / "residual_spectrum_by_fold.csv"),
                "residual_spectrum_summary": str(output_root / "residual_spectrum_summary.csv"),
            },
            "include_linear_baselines": bool(args.include_linear_baselines),
            "fold_manifests": fold_manifests,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--calibration-table", type=Path, default=DEFAULT_CALIBRATION_TABLE)
    parser.add_argument("--prior-name", default=DEFAULT_PRIOR_NAME)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--folds", type=_parse_folds, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--alphas", type=_parse_alphas, default=DEFAULT_ALPHA_GRID)
    parser.add_argument(
        "--experiment",
        default="baseline",
        choices=["baseline", "sample_efficiency", "capacity", "sample_efficiency_capacity", "ood", "prior_anchor"],
    )
    parser.add_argument("--train-fractions", type=_parse_float_tuple, default=(1.0,))
    parser.add_argument("--capacity-presets", type=_parse_string_tuple, default=("base",))
    parser.add_argument("--ood-presets", type=_parse_string_tuple, default=("airspeed_ge8", "alpha_abs_ge20"))
    parser.add_argument("--prior-loss-weights", type=_parse_float_tuple, default=(0.0, 0.001, 0.01, 0.1))
    parser.add_argument(
        "--include-linear-baselines",
        action="store_true",
        help="Include raw prior and gain-bias baselines for non-baseline experiments.",
    )
    parser.add_argument("--phase-bins", type=int, default=12)
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--feature-set", default="paper_no_accel_v2_phase_harmonic")
    parser.add_argument("--sequence-feature-mode", default="phase_harmonic_actuator_airdata")
    parser.add_argument("--current-feature-mode", default="remaining_current")
    parser.add_argument("--sequence-history-size", type=int, default=64)
    parser.add_argument("--hidden-sizes", type=lambda text: tuple(int(part) for part in text.split(",") if part), default=(128, 128))
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--loss-type", default="huber", choices=["mse", "huber"])
    parser.add_argument("--huber-delta", type=float, default=1.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--tcn-channels", type=int, default=128)
    parser.add_argument("--tcn-num-blocks", type=int, default=4)
    parser.add_argument("--tcn-kernel-size", type=int, default=3)
    parser.add_argument("--lr-scheduler", default=None, choices=[None, "warmup_cosine"])
    parser.add_argument("--lr-warmup-ratio", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.phase_bins < 2:
        raise ValueError("--phase-bins must be at least 2")
    if args.sequence_history_size < 1:
        raise ValueError("--sequence-history-size must be positive")
    args.alphas = tuple(args.alphas)
    args.folds = list(args.folds)
    args.hidden_sizes = tuple(args.hidden_sizes)
    args.train_fractions = tuple(args.train_fractions)
    args.capacity_presets = tuple(args.capacity_presets)
    args.ood_presets = tuple(args.ood_presets)
    args.prior_loss_weights = tuple(args.prior_loss_weights)
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
