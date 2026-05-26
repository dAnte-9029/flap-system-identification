#!/usr/bin/env python3
"""Train dynamic equivalent-arm moment heads from force predictions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FORCE_TARGETS = ("fx_b", "fy_b", "fz_b")
MOMENT_TARGETS = ("mx_b", "my_b", "mz_b")
DEFAULT_SPLIT_ROOT = Path("dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1")
DEFAULT_FORCE_PREDICTION_ROOT = Path("artifacts/20260525_delaurier_greybox_force_correction_v1")
DEFAULT_STAGE1_ROOT = Path("artifacts/20260525_delaurier_force_recalibration_v1")
DEFAULT_OUTPUT_ROOT = Path("artifacts/20260525_dynamic_arm_moment_head_v1")

FEATURE_CANDIDATES = (
    "phase_sin",
    "phase_cos",
    "phase_sin_1",
    "phase_cos_1",
    "phase_sin_2",
    "phase_cos_2",
    "alpha_rad",
    "flap_frequency_hz",
    "true_airspeed_m_s",
    "dynamic_pressure_pa",
    "alpha_rad_x_phase_sin_1",
    "alpha_rad_x_phase_cos_1",
    "flap_frequency_hz_x_phase_sin_1",
    "flap_frequency_hz_x_phase_cos_1",
    "true_airspeed_m_s_x_phase_sin_1",
    "true_airspeed_m_s_x_phase_cos_1",
    "alpha_rad_x_flap_frequency_hz",
)


@dataclass(frozen=True)
class FeatureTransform:
    raw_columns: list[str]
    model_columns: list[str]
    fill: list[float]
    mean: list[float]
    scale: list[float]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.raw_columns:
            return np.ones((len(frame), 1), dtype=float)
        x = frame.loc[:, self.raw_columns].to_numpy(dtype=float)
        fill = np.asarray(self.fill, dtype=float)
        mean = np.asarray(self.mean, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        x = np.where(np.isfinite(x), x, fill)
        x = (x - mean) / scale
        return np.column_stack([np.ones(len(frame), dtype=float), x])


@dataclass
class LinearMomentHead:
    model_name: str
    force_source: str
    alpha: float
    feature_transform: FeatureTransform
    arm_coefficients: np.ndarray | None = None
    free_coefficients: np.ndarray | None = None
    direct_coefficients: np.ndarray | None = None

    def predict(self, *, force: np.ndarray, features: pd.DataFrame | None = None) -> dict[str, np.ndarray]:
        phi = (
            np.ones((len(force), 1), dtype=float)
            if features is None
            else self.feature_transform.transform(features)
        )
        if self.direct_coefficients is not None:
            moment = phi @ self.direct_coefficients
            zeros = np.zeros_like(moment)
            return {"moment": moment, "r_hat": zeros, "tau_free": moment, "arm_moment": zeros}

        if self.arm_coefficients is None:
            raise ValueError("arm coefficients are required for force-arm models")
        r_hat = phi @ self.arm_coefficients
        arm_moment = cross_arm_force(r_hat, force)
        tau_free = np.zeros_like(arm_moment)
        if self.free_coefficients is not None:
            tau_free = phi @ self.free_coefficients
        return {"moment": arm_moment + tau_free, "r_hat": r_hat, "tau_free": tau_free, "arm_moment": arm_moment}


def cross_arm_force(arm: np.ndarray, force: np.ndarray) -> np.ndarray:
    """Return row-wise ``arm x force`` moments."""

    arm = np.asarray(arm, dtype=float)
    force = np.asarray(force, dtype=float)
    if arm.shape != force.shape or arm.shape[1] != 3:
        raise ValueError(f"arm and force must both be shaped (n, 3); got {arm.shape} and {force.shape}")
    return np.column_stack(
        [
            arm[:, 1] * force[:, 2] - arm[:, 2] * force[:, 1],
            arm[:, 2] * force[:, 0] - arm[:, 0] * force[:, 2],
            arm[:, 0] * force[:, 1] - arm[:, 1] * force[:, 0],
        ]
    )


def build_arm_design_matrix(features: np.ndarray, force: np.ndarray) -> np.ndarray:
    """Build the linear design matrix mapping flattened arm coefficients to ``r(x) x F``."""

    phi = np.asarray(features, dtype=float)
    force = np.asarray(force, dtype=float)
    if phi.ndim != 2 or force.ndim != 2 or force.shape[1] != 3 or len(phi) != len(force):
        raise ValueError(f"features must be (n, p) and force must be (n, 3); got {phi.shape} and {force.shape}")
    n, p = phi.shape
    fx, fy, fz = force[:, 0], force[:, 1], force[:, 2]
    design = np.zeros((n, 3, p * 3), dtype=float)
    for idx in range(p):
        column = phi[:, idx]
        base = idx * 3
        design[:, 0, base + 1] = column * fz
        design[:, 0, base + 2] = -column * fy
        design[:, 1, base + 0] = -column * fz
        design[:, 1, base + 2] = column * fx
        design[:, 2, base + 0] = column * fy
        design[:, 2, base + 1] = -column * fx
    return design.reshape(n * 3, p * 3)


def _build_direct_design_matrix(features: np.ndarray) -> np.ndarray:
    n, p = features.shape
    design = np.zeros((n, 3, p * 3), dtype=float)
    for idx in range(p):
        column = features[:, idx]
        base = idx * 3
        design[:, 0, base + 0] = column
        design[:, 1, base + 1] = column
        design[:, 2, base + 2] = column
    return design.reshape(n * 3, p * 3)


def _ridge_solve(design: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    y = np.asarray(target, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(design).all(axis=1)
    x = design[mask]
    y = y[mask]
    gram = x.T @ x
    if alpha > 0.0:
        gram = gram + float(alpha) * np.eye(gram.shape[0])
    rhs = x.T @ y
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def _ridge_solve_multi(features: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    mask = np.isfinite(target).all(axis=1) & np.isfinite(features).all(axis=1)
    x = features[mask]
    y = target[mask]
    gram = x.T @ x
    if alpha > 0.0:
        gram = gram + float(alpha) * np.eye(gram.shape[0])
    rhs = x.T @ y
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def _fit_feature_transform(frame: pd.DataFrame, columns: list[str]) -> FeatureTransform:
    if not columns:
        return FeatureTransform(raw_columns=[], model_columns=["bias"], fill=[], mean=[], scale=[])
    x = frame.loc[:, columns].to_numpy(dtype=float)
    finite_x = np.where(np.isfinite(x), x, np.nan)
    fill = np.nanmedian(finite_x, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    x = np.where(np.isfinite(x), x, fill)
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return FeatureTransform(
        raw_columns=columns,
        model_columns=["bias", *columns],
        fill=fill.tolist(),
        mean=mean.tolist(),
        scale=scale.tolist(),
    )


def fit_fixed_arm(*, force: np.ndarray, moment: np.ndarray, alpha: float, force_source: str = "unit") -> LinearMomentHead:
    transform = _fit_feature_transform(pd.DataFrame(index=range(len(force))), [])
    phi = np.ones((len(force), 1), dtype=float)
    coefficients = _ridge_solve(build_arm_design_matrix(phi, force), moment, alpha).reshape(1, 3)
    return LinearMomentHead(
        model_name="fixed_arm",
        force_source=force_source,
        alpha=float(alpha),
        feature_transform=transform,
        arm_coefficients=coefficients,
    )


def fit_dynamic_arm_linear(
    *,
    features: pd.DataFrame,
    feature_columns: list[str],
    force: np.ndarray,
    moment: np.ndarray,
    alpha: float,
    force_source: str,
) -> LinearMomentHead:
    """Fit ``r_hat = Phi(x) B`` so that ``M = r_hat x F``."""

    transform = _fit_feature_transform(features, feature_columns)
    phi = transform.transform(features)
    coefficients = _ridge_solve(build_arm_design_matrix(phi, force), moment, alpha).reshape(phi.shape[1], 3)
    return LinearMomentHead(
        model_name="dynamic_arm_linear",
        force_source=force_source,
        alpha=float(alpha),
        feature_transform=transform,
        arm_coefficients=coefficients,
    )


def fit_dynamic_arm_plus_free_linear(
    *,
    features: pd.DataFrame,
    feature_columns: list[str],
    force: np.ndarray,
    moment: np.ndarray,
    alpha: float,
    force_source: str,
) -> LinearMomentHead:
    """Fit dynamic arm first, then an unconstrained linear residual moment."""

    arm_model = fit_dynamic_arm_linear(
        features=features,
        feature_columns=feature_columns,
        force=force,
        moment=moment,
        alpha=alpha,
        force_source=force_source,
    )
    arm_prediction = arm_model.predict(force=force, features=features)["arm_moment"]
    residual = moment - arm_prediction
    phi = arm_model.feature_transform.transform(features)
    free_coefficients = _ridge_solve_multi(phi, residual, alpha)
    return LinearMomentHead(
        model_name="dynamic_arm_plus_free_linear",
        force_source=force_source,
        alpha=float(alpha),
        feature_transform=arm_model.feature_transform,
        arm_coefficients=arm_model.arm_coefficients,
        free_coefficients=free_coefficients,
    )


def _fit_direct_moment(
    *,
    transform: FeatureTransform,
    features: pd.DataFrame,
    moment: np.ndarray,
    alpha: float,
) -> LinearMomentHead:
    phi = transform.transform(features)
    coefficients = _ridge_solve_multi(phi, moment, alpha)
    return LinearMomentHead(
        model_name="direct_moment_linear",
        force_source="features_only",
        alpha=float(alpha),
        feature_transform=transform,
        direct_coefficients=coefficients,
    )


def _channel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {"mae": np.nan, "rmse": np.nan, "bias": np.nan, "r2": np.nan}
    residual = y_pred[mask] - y_true[mask]
    ss_res = float(np.sum(residual * residual))
    centered = y_true[mask] - float(np.mean(y_true[mask]))
    ss_tot = float(np.sum(centered * centered))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "bias": float(np.mean(residual)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
    }


def moment_metrics(
    *,
    split: str,
    model_name: str,
    force_source: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for idx, target in enumerate(MOMENT_TARGETS):
        metrics = _channel_metrics(y_true[:, idx], y_pred[:, idx])
        rows.append(
            {
                "split": split,
                "model": model_name,
                "force_source": force_source,
                "target": target,
                "n": int(np.isfinite(y_true[:, idx]).sum()),
                **metrics,
            }
        )
    rows.append(
        {
            "split": split,
            "model": model_name,
            "force_source": force_source,
            "target": "moment_mean",
            "n": int(len(y_true)),
            "mae": float(np.nanmean([row["mae"] for row in rows])),
            "rmse": float(np.sqrt(np.nanmean(np.square([row["rmse"] for row in rows])))),
            "bias": float(np.nanmean([row["bias"] for row in rows])),
            "r2": float(np.nanmean([row["r2"] for row in rows])),
        }
    )
    return pd.DataFrame(rows)


def _r_hat_distribution(*, split: str, model: LinearMomentHead, predictions: dict[str, np.ndarray]) -> dict[str, float | str | int]:
    r_hat = predictions["r_hat"]
    norm = np.linalg.norm(r_hat, axis=1)
    return {
        "split": split,
        "model": model.model_name,
        "force_source": model.force_source,
        "n": int(len(r_hat)),
        "rx_mean": float(np.mean(r_hat[:, 0])),
        "ry_mean": float(np.mean(r_hat[:, 1])),
        "rz_mean": float(np.mean(r_hat[:, 2])),
        "norm_mean": float(np.mean(norm)),
        "norm_median": float(np.median(norm)),
        "norm_p90": float(np.quantile(norm, 0.90)),
        "norm_p99": float(np.quantile(norm, 0.99)),
        "norm_max": float(np.max(norm)),
    }


def compute_tau_free_energy(
    *,
    moment: np.ndarray,
    arm_moment: np.ndarray,
    tau_free: np.ndarray,
) -> dict[str, float]:
    """Compute split-level energy fractions for the unconstrained free moment."""

    tau_energy = float(np.sum(tau_free * tau_free))
    arm_energy = float(np.sum(arm_moment * arm_moment))
    pred_energy = float(np.sum(moment * moment))
    return {
        "arm_energy": arm_energy,
        "tau_free_energy": tau_energy,
        "predicted_moment_energy": pred_energy,
        "tau_free_fraction_of_predicted": float(tau_energy / pred_energy) if pred_energy > 0.0 else np.nan,
        "tau_free_fraction_of_arm_plus_tau": float(tau_energy / (arm_energy + tau_energy))
        if (arm_energy + tau_energy) > 0.0
        else np.nan,
    }


def _tau_free_energy(*, split: str, model: LinearMomentHead, predictions: dict[str, np.ndarray]) -> dict[str, float | str | int]:
    energy = compute_tau_free_energy(
        moment=predictions["moment"],
        arm_moment=predictions["arm_moment"],
        tau_free=predictions["tau_free"],
    )
    return {
        "split": split,
        "model": model.model_name,
        "force_source": model.force_source,
        "n": int(len(predictions["moment"])),
        **energy,
    }


def _per_log_metrics(
    *,
    split_frame: pd.DataFrame,
    split: str,
    model: LinearMomentHead,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    log_ids = split_frame["log_id"].to_numpy() if "log_id" in split_frame.columns else np.full(len(split_frame), "unknown")
    rows: list[dict[str, float | int | str]] = []
    for log_id in pd.unique(log_ids):
        mask = log_ids == log_id
        metrics = moment_metrics(
            split=split,
            model_name=model.model_name,
            force_source=model.force_source,
            y_true=y_true[mask],
            y_pred=y_pred[mask],
        )
        mean = metrics.loc[metrics["target"] == "moment_mean"].iloc[0].to_dict()
        mean["log_id"] = log_id
        rows.append(mean)
    return pd.DataFrame(rows)


def _default_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in FEATURE_CANDIDATES
        if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _force_from_columns(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    return frame.loc[:, [f"{prefix}_{target}" for target in FORCE_TARGETS]].to_numpy(dtype=float)


def _has_force_columns(frame: pd.DataFrame, prefix: str) -> bool:
    return all(f"{prefix}_{target}" in frame.columns for target in FORCE_TARGETS)


def _prediction_path(force_prediction_root: Path, split: str) -> Path:
    direct = force_prediction_root / f"{split}_predictions.parquet"
    nested = force_prediction_root / "prediction_parquets" / f"{split}_predictions.parquet"
    if direct.exists():
        return direct
    if nested.exists():
        return nested
    raise FileNotFoundError(
        f"could not find {split}_predictions.parquet under {force_prediction_root} "
        "or its prediction_parquets/ subdirectory"
    )


def _load_stage1_force(stage1_root: Path, split: str, variant: str) -> np.ndarray | None:
    path = stage1_root / f"{split}_{variant}_aligned_force_predictions.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    return _force_from_columns(frame, "pred")


def _load_split(
    *,
    split: str,
    split_root: Path,
    force_prediction_root: Path,
    stage1_root: Path,
) -> dict[str, object]:
    prediction_path = _prediction_path(force_prediction_root, split)
    predictions = pd.read_parquet(prediction_path)
    samples = pd.read_parquet(split_root / f"{split}_samples.parquet")
    if len(samples) != len(predictions):
        raise ValueError(f"{split} row count mismatch: samples={len(samples)} predictions={len(predictions)}")

    if _has_force_columns(predictions, "label"):
        true_force = _force_from_columns(predictions, "label")
    else:
        true_force = samples.loc[:, FORCE_TARGETS].to_numpy(dtype=float)
    force_sources: dict[str, np.ndarray] = {
        "true_force": true_force,
        "prior_force": _force_from_columns(predictions, "prior"),
        "corrected_force": _force_from_columns(predictions, "corrected"),
    }

    return {
        "frame": predictions,
        "moment": samples.loc[:, MOMENT_TARGETS].to_numpy(dtype=float),
        "force_sources": force_sources,
    }


def _select_row(metrics: pd.DataFrame, split: str, model: str, force_source: str, target: str = "moment_mean") -> pd.Series:
    rows = metrics[
        (metrics["split"] == split)
        & (metrics["model"] == model)
        & (metrics["force_source"] == force_source)
        & (metrics["target"] == target)
    ]
    if rows.empty:
        raise KeyError((split, model, force_source, target))
    return rows.iloc[0]


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [[str(row[column]) for column in frame.columns] for _, row in frame.iterrows()]

    def render(values: list[str]) -> str:
        return "| " + " | ".join(values) + " |"

    separator = "| " + " | ".join("---" for _ in columns) + " |"
    return "\n".join([render(columns), separator, *(render(row) for row in rows)])


def _write_prediction_parquets(
    *,
    output_root: Path,
    best_model: LinearMomentHead,
    loaded: dict[str, dict[str, object]],
) -> dict[str, Path]:
    prediction_dir = output_root / "prediction_parquets"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, data in loaded.items():
        frame = data["frame"]
        moment = data["moment"]
        force = data["force_sources"][best_model.force_source] if best_model.force_source != "features_only" else data["force_sources"]["corrected_force"]
        predictions = best_model.predict(force=force, features=frame)
        metadata_columns = [
            column
            for column in ("timestamp_us", "time_s", "log_id", "segment_id", "cycle_id", "phase_corrected_rad", "split")
            if column in frame.columns
        ]
        out = frame.loc[:, metadata_columns].copy()
        for idx, target in enumerate(FORCE_TARGETS):
            out[f"force_{target}"] = force[:, idx]
        for idx, target in enumerate(MOMENT_TARGETS):
            out[f"label_{target}"] = moment[:, idx]
            out[f"pred_{target}"] = predictions["moment"][:, idx]
            out[f"residual_{target}"] = moment[:, idx] - predictions["moment"][:, idx]
            out[f"arm_moment_{target}"] = predictions["arm_moment"][:, idx]
            out[f"tau_free_{target}"] = predictions["tau_free"][:, idx]
        for idx, column in enumerate(("rx_b_m", "ry_b_m", "rz_b_m")):
            out[f"r_hat_{column}"] = predictions["r_hat"][:, idx]
        path = prediction_dir / f"{split}_predictions.parquet"
        out.to_parquet(path, index=False)
        paths[split] = path
    return paths


def _write_model_artifacts(
    *,
    output_root: Path,
    best_model: LinearMomentHead,
    feature_columns: list[str],
    force_sources: Iterable[str],
    alphas: tuple[float, ...],
    config: dict[str, object],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    model_config = {
        **config,
        "best_model": {
            "model_name": best_model.model_name,
            "force_source": best_model.force_source,
            "alpha": best_model.alpha,
        },
        "feature_columns": feature_columns,
        "force_sources": list(force_sources),
        "alphas": list(alphas),
        "feature_transform": asdict(best_model.feature_transform),
    }
    (output_root / "model_config.json").write_text(json.dumps(model_config, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(
        output_root / "coefficients_or_model.npz",
        model_name=np.array(best_model.model_name),
        force_source=np.array(best_model.force_source),
        alpha=np.array(best_model.alpha),
        feature_columns=np.array(best_model.feature_transform.model_columns, dtype=object),
        arm_coefficients=np.array([]) if best_model.arm_coefficients is None else best_model.arm_coefficients,
        free_coefficients=np.array([]) if best_model.free_coefficients is None else best_model.free_coefficients,
        direct_coefficients=np.array([]) if best_model.direct_coefficients is None else best_model.direct_coefficients,
    )


def _write_readme(
    *,
    output_root: Path,
    command: str,
    best_model: LinearMomentHead,
    metrics: pd.DataFrame,
    r_distribution: pd.DataFrame,
    tau_energy: pd.DataFrame,
    split_root: Path,
    force_prediction_root: Path,
    stage1_root: Path,
) -> None:
    test_summary = metrics[(metrics["split"] == "test") & (metrics["target"] == "moment_mean")].copy()
    test_summary = test_summary.sort_values(["rmse", "model", "force_source"]).loc[
        :, ["model", "force_source", "rmse", "r2", "mae", "bias"]
    ]
    best_r = r_distribution[
        (r_distribution["split"] == "test")
        & (r_distribution["model"] == best_model.model_name)
        & (r_distribution["force_source"] == best_model.force_source)
    ]
    best_tau = tau_energy[
        (tau_energy["split"] == "test")
        & (tau_energy["model"] == best_model.model_name)
        & (tau_energy["force_source"] == best_model.force_source)
    ]
    lines = [
        "# Dynamic Equivalent-Arm Moment Head v1",
        "",
        "This artifact fits train-only ridge heads for moment models of the form `M = r_hat(x) x F` and `M = r_hat(x) x F + tau_free_hat(x)`.",
        "The learned equivalent arm is a structured predictive parameterization, not a validated center-of-pressure measurement.",
        "",
        "## Inputs",
        "",
        f"- Split root: `{split_root}`",
        f"- Stage 2 force predictions: `{force_prediction_root}`",
        f"- Stage 1 force recalibration: `{stage1_root}`",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Selected model",
        "",
        f"- Model: `{best_model.model_name}`",
        f"- Force source: `{best_model.force_source}`",
        f"- Ridge alpha: `{best_model.alpha:g}`",
        "",
        "## Test moment mean ranking",
        "",
        dataframe_to_markdown(test_summary.reset_index(drop=True)),
        "",
        "## Selected test arm distribution",
        "",
        dataframe_to_markdown(best_r.reset_index(drop=True)),
        "",
        "## Selected test tau_free energy",
        "",
        dataframe_to_markdown(best_tau.reset_index(drop=True)),
        "",
        "## Outputs",
        "",
        "- `model_config.json`: configuration, selected model metadata, and feature transform.",
        "- `coefficients_or_model.npz`: selected model coefficients.",
        "- `model_selection.csv`: validation-selection rows for every fitted candidate.",
        "- `metrics_by_split.csv`: train/val/test moment metrics for selected candidates.",
        "- `moment_metrics_summary.csv`: compact test ranking.",
        "- `r_hat_distribution.csv`: split-level equivalent-arm component and norm statistics.",
        "- `tau_free_energy.csv`: split-level arm and free-moment energy fractions.",
        "- `per_log_moment_metrics.csv`: test per-log moment mean metrics.",
        "- `prediction_parquets/`: train/val/test predictions for the best validation-selected model.",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_dynamic_arm_moment_head(
    *,
    split_root: Path = DEFAULT_SPLIT_ROOT,
    force_prediction_root: Path = DEFAULT_FORCE_PREDICTION_ROOT,
    stage1_root: Path = DEFAULT_STAGE1_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
    command: str = "",
) -> dict[str, Path | str]:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "prediction_curves").mkdir(parents=True, exist_ok=True)
    loaded = {
        split: _load_split(
            split=split,
            split_root=split_root,
            force_prediction_root=force_prediction_root,
            stage1_root=stage1_root,
        )
        for split in ("train", "val", "test")
    }
    train_frame = loaded["train"]["frame"]
    feature_columns = _default_feature_columns(train_frame)
    transform = _fit_feature_transform(train_frame, feature_columns)
    train_phi = transform.transform(train_frame)
    val_phi = transform.transform(loaded["val"]["frame"])
    train_moment = loaded["train"]["moment"]
    val_moment = loaded["val"]["moment"]
    force_source_names = tuple(loaded["train"]["force_sources"].keys())

    best_by_combo: list[LinearMomentHead] = []
    selection_rows: list[dict[str, float | str]] = []

    for force_source in force_source_names:
        train_force = loaded["train"]["force_sources"][force_source]
        val_force = loaded["val"]["force_sources"][force_source]
        for model_name in ("fixed_arm", "dynamic_arm_linear", "dynamic_arm_plus_free_linear"):
            val_design: np.ndarray | None = None
            train_design: np.ndarray | None = None
            if model_name == "fixed_arm":
                train_design = build_arm_design_matrix(np.ones((len(train_force), 1), dtype=float), train_force)
                val_design = build_arm_design_matrix(np.ones((len(val_force), 1), dtype=float), val_force)
            elif model_name in {"dynamic_arm_linear", "dynamic_arm_plus_free_linear"}:
                train_design = build_arm_design_matrix(train_phi, train_force)
                val_design = build_arm_design_matrix(val_phi, val_force)

            best_model: LinearMomentHead | None = None
            best_val_rmse = float("inf")
            for alpha in alphas:
                coefficients = _ridge_solve(train_design, train_moment, alpha)
                if model_name == "fixed_arm":
                    transform_for_model = _fit_feature_transform(pd.DataFrame(index=range(len(train_force))), [])
                    model = LinearMomentHead(
                        model_name=model_name,
                        force_source=force_source,
                        alpha=float(alpha),
                        feature_transform=transform_for_model,
                        arm_coefficients=coefficients.reshape(1, 3),
                    )
                    val_prediction = (val_design @ coefficients).reshape(-1, 3)
                elif model_name == "dynamic_arm_linear":
                    model = LinearMomentHead(
                        model_name=model_name,
                        force_source=force_source,
                        alpha=float(alpha),
                        feature_transform=transform,
                        arm_coefficients=coefficients.reshape(train_phi.shape[1], 3),
                    )
                    val_prediction = (val_design @ coefficients).reshape(-1, 3)
                else:
                    arm_coefficients = coefficients.reshape(train_phi.shape[1], 3)
                    train_arm_prediction = (train_design @ coefficients).reshape(-1, 3)
                    residual = train_moment - train_arm_prediction
                    free_coefficients = _ridge_solve_multi(train_phi, residual, alpha)
                    model = LinearMomentHead(
                        model_name=model_name,
                        force_source=force_source,
                        alpha=float(alpha),
                        feature_transform=transform,
                        arm_coefficients=arm_coefficients,
                        free_coefficients=free_coefficients,
                    )
                    val_prediction = (val_design @ coefficients).reshape(-1, 3) + val_phi @ free_coefficients
                val_metric = moment_metrics(
                    split="val",
                    model_name=model_name,
                    force_source=force_source,
                    y_true=val_moment,
                    y_pred=val_prediction,
                )
                val_rmse = float(val_metric.loc[val_metric["target"] == "moment_mean", "rmse"].iloc[0])
                selection_rows.append(
                    {
                        "model": model_name,
                        "force_source": force_source,
                        "alpha": float(alpha),
                        "val_moment_mean_rmse": val_rmse,
                        "val_moment_mean_r2": float(val_metric.loc[val_metric["target"] == "moment_mean", "r2"].iloc[0]),
                    }
                )
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_model = model
            if best_model is None:
                raise RuntimeError(f"no selected model for {model_name}/{force_source}")
            best_by_combo.append(best_model)

    best_direct: LinearMomentHead | None = None
    best_direct_val_rmse = float("inf")
    for alpha in alphas:
        coefficients = _ridge_solve_multi(train_phi, train_moment, alpha)
        model = LinearMomentHead(
            model_name="direct_moment_linear",
            force_source="features_only",
            alpha=float(alpha),
            feature_transform=transform,
            direct_coefficients=coefficients,
        )
        val_prediction = val_phi @ coefficients
        val_metric = moment_metrics(
            split="val",
            model_name="direct_moment_linear",
            force_source="features_only",
            y_true=val_moment,
            y_pred=val_prediction,
        )
        val_rmse = float(val_metric.loc[val_metric["target"] == "moment_mean", "rmse"].iloc[0])
        selection_rows.append(
            {
                "model": "direct_moment_linear",
                "force_source": "features_only",
                "alpha": float(alpha),
                "val_moment_mean_rmse": val_rmse,
                "val_moment_mean_r2": float(val_metric.loc[val_metric["target"] == "moment_mean", "r2"].iloc[0]),
            }
        )
        if val_rmse < best_direct_val_rmse:
            best_direct_val_rmse = val_rmse
            best_direct = model
    if best_direct is None:
        raise RuntimeError("no selected direct moment model")
    best_by_combo.append(best_direct)

    metrics_frames: list[pd.DataFrame] = []
    r_rows: list[dict[str, float | str | int]] = []
    tau_rows: list[dict[str, float | str | int]] = []
    per_log_frames: list[pd.DataFrame] = []
    for model in best_by_combo:
        for split, data in loaded.items():
            force_source = model.force_source
            force = data["force_sources"][force_source] if force_source != "features_only" else data["force_sources"]["corrected_force"]
            predictions = model.predict(force=force, features=data["frame"])
            metrics_frames.append(
                moment_metrics(
                    split=split,
                    model_name=model.model_name,
                    force_source=model.force_source,
                    y_true=data["moment"],
                    y_pred=predictions["moment"],
                )
            )
            r_rows.append(_r_hat_distribution(split=split, model=model, predictions=predictions))
            tau_rows.append(_tau_free_energy(split=split, model=model, predictions=predictions))
            if split == "test":
                per_log_frames.append(
                    _per_log_metrics(
                        split_frame=data["frame"],
                        split=split,
                        model=model,
                        y_true=data["moment"],
                        y_pred=predictions["moment"],
                    )
                )

    metrics = pd.concat(metrics_frames, ignore_index=True)
    selection = pd.DataFrame(selection_rows).sort_values(["val_moment_mean_rmse", "model", "force_source"])
    r_distribution = pd.DataFrame(r_rows)
    tau_energy = pd.DataFrame(tau_rows)
    per_log = pd.concat(per_log_frames, ignore_index=True)
    test_summary = metrics[metrics["split"].eq("test") & metrics["target"].eq("moment_mean")].copy()
    test_summary = test_summary.sort_values(["rmse", "model", "force_source"])

    metrics.to_csv(output_root / "metrics_by_split.csv", index=False)
    selection.to_csv(output_root / "model_selection.csv", index=False)
    test_summary.to_csv(output_root / "moment_metrics_summary.csv", index=False)
    r_distribution.to_csv(output_root / "r_hat_distribution.csv", index=False)
    tau_energy.to_csv(output_root / "tau_free_energy.csv", index=False)
    per_log.to_csv(output_root / "per_log_moment_metrics.csv", index=False)

    best_row = selection.iloc[0]
    best_model = next(
        model
        for model in best_by_combo
        if model.model_name == best_row["model"]
        and model.force_source == best_row["force_source"]
        and abs(model.alpha - float(best_row["alpha"])) < 1.0e-12
    )
    prediction_paths = _write_prediction_parquets(output_root=output_root, best_model=best_model, loaded=loaded)
    _write_model_artifacts(
        output_root=output_root,
        best_model=best_model,
        feature_columns=feature_columns,
        force_sources=force_source_names,
        alphas=alphas,
        config={
            "split_root": str(split_root),
            "force_prediction_root": str(force_prediction_root),
            "stage1_root": str(stage1_root),
            "output_root": str(output_root),
            "selection_metric": "validation moment_mean RMSE",
        },
    )
    _write_readme(
        output_root=output_root,
        command=command,
        best_model=best_model,
        metrics=metrics,
        r_distribution=r_distribution,
        tau_energy=tau_energy,
        split_root=split_root,
        force_prediction_root=force_prediction_root,
        stage1_root=stage1_root,
    )
    return {
        "output_root": output_root,
        "best_model": best_model.model_name,
        "best_force_source": best_model.force_source,
        "metrics": output_root / "metrics_by_split.csv",
        "test_predictions": prediction_paths["test"],
    }


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--force-prediction-root", type=Path, default=DEFAULT_FORCE_PREDICTION_ROOT)
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100,1000")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    command = (
        "python scripts/train_dynamic_arm_moment_head.py "
        f"--split-root {args.split_root} --force-prediction-root {args.force_prediction_root} "
        f"--stage1-root {args.stage1_root} --output-root {args.output_root} --alphas {args.alphas}"
    )
    outputs = train_dynamic_arm_moment_head(
        split_root=args.split_root,
        force_prediction_root=args.force_prediction_root,
        stage1_root=args.stage1_root,
        output_root=args.output_root,
        alphas=_parse_csv_floats(args.alphas),
        command=command,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
