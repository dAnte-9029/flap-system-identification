#!/usr/bin/env python3
"""Train low-dimensional grey-box corrections for DeLaurier force predictions."""

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
DEFAULT_SPLIT_ROOT = Path("dataset/canonical_v0.2_training_ready_split_hq_v4_direct_airspeed_logsplit_paper_alt5_v1")
DEFAULT_PRIOR_ROOT = Path("artifacts/delaurier_physical_prior_v1")
DEFAULT_OUTPUT_ROOT = Path("artifacts/20260525_delaurier_greybox_force_correction_v1")


@dataclass(frozen=True)
class FeatureSpec:
    phase_column: str
    frequency_column: str
    airspeed_column: str
    density_column: str | None
    aoa_source: str
    columns: list[str]


@dataclass
class CorrectionModel:
    name: str
    targets: tuple[str, ...]
    feature_columns: list[str]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    alpha: float
    feature_fill: np.ndarray | None = None

    def design_from_frame(self, features: pd.DataFrame) -> np.ndarray:
        x = features.loc[:, self.feature_columns].to_numpy(dtype=float)
        fill = self.feature_fill if self.feature_fill is not None else self.feature_mean
        x = np.where(np.isfinite(x), x, fill)
        return (x - self.feature_mean) / self.feature_scale

    def predict_raw(self, features: pd.DataFrame) -> np.ndarray:
        return self.design_from_frame(features) @ self.coefficients + self.intercept

    def apply_correction(self, prior_force: np.ndarray, raw_prediction: np.ndarray, *, variant: str | None = None) -> np.ndarray:
        correction = self.name if variant is None else variant
        if correction == "additive":
            return prior_force + raw_prediction
        if correction == "multiplicative":
            return prior_force * (1.0 + raw_prediction)
        if correction == "affine":
            return prior_force + raw_prediction
        raise ValueError(f"unknown correction variant: {correction}")

    def predict_force(self, features: pd.DataFrame, prior_force: np.ndarray) -> np.ndarray:
        return self.apply_correction(prior_force, self.predict_raw(features))


def _first_existing(columns: Iterable[str], frame: pd.DataFrame, *, kind: str) -> str:
    for column in columns:
        if column in frame.columns:
            return column
    raise ValueError(f"could not find a {kind} column; tried {list(columns)}")


def _series_or_default(frame: pd.DataFrame, column: str | None, default: float) -> pd.Series:
    if column and column in frame.columns:
        return frame[column].astype(float)
    return pd.Series(default, index=frame.index, dtype=float)


def _body_velocity_from_attitude(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series] | None:
    q_columns = [f"vehicle_attitude.q[{idx}]" for idx in range(4)]
    v_columns = ["vehicle_local_position.vx", "vehicle_local_position.vy", "vehicle_local_position.vz"]
    if any(column not in frame.columns for column in q_columns + v_columns):
        return None

    q0 = frame[q_columns[0]].to_numpy(dtype=float)
    q1 = frame[q_columns[1]].to_numpy(dtype=float)
    q2 = frame[q_columns[2]].to_numpy(dtype=float)
    q3 = frame[q_columns[3]].to_numpy(dtype=float)
    vn = frame[v_columns[0]].to_numpy(dtype=float) - _series_or_default(frame, "wind.windspeed_north", 0.0).to_numpy()
    ve = frame[v_columns[1]].to_numpy(dtype=float) - _series_or_default(frame, "wind.windspeed_east", 0.0).to_numpy()
    vd = frame[v_columns[2]].to_numpy(dtype=float)

    r00 = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
    r01 = 2.0 * (q1 * q2 - q0 * q3)
    r02 = 2.0 * (q1 * q3 + q0 * q2)
    r10 = 2.0 * (q1 * q2 + q0 * q3)
    r11 = 1.0 - 2.0 * (q1 * q1 + q3 * q3)
    r12 = 2.0 * (q2 * q3 - q0 * q1)
    r20 = 2.0 * (q1 * q3 - q0 * q2)
    r21 = 2.0 * (q2 * q3 + q0 * q1)
    r22 = 1.0 - 2.0 * (q1 * q1 + q2 * q2)

    # PX4 attitude quaternions rotate body to local NED; transpose maps local NED to body.
    u_b = r00 * vn + r10 * ve + r20 * vd
    v_b = r01 * vn + r11 * ve + r21 * vd
    w_b = r02 * vn + r12 * ve + r22 * vd
    return (
        pd.Series(u_b, index=frame.index, dtype=float),
        pd.Series(v_b, index=frame.index, dtype=float),
        pd.Series(w_b, index=frame.index, dtype=float),
    )


def _derive_alpha_rad(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    if "alpha_rad" in frame.columns:
        return frame["alpha_rad"].astype(float), "alpha_rad"
    body_velocity = _body_velocity_from_attitude(frame)
    if body_velocity is not None:
        u_b, _, w_b = body_velocity
        return pd.Series(np.arctan2(-w_b.to_numpy(dtype=float), u_b.to_numpy(dtype=float)), index=frame.index), (
            "body_air_relative_velocity"
        )
    for column in ("airspeed_validated.pitch_filtered", "vehicle_local_position.pitch", "pitch_rad"):
        if column in frame.columns:
            return frame[column].astype(float), column
    return pd.Series(0.0, index=frame.index, dtype=float), "zero_fallback"


def build_feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, FeatureSpec]:
    """Build low-dimensional phase/frequency/AoA/airspeed correction features."""

    phase_column = _first_existing(
        ("phase_corrected_rad", "wing_phase.phase_rad", "drive_phase_rad", "encoder_phase_rad", "phase_raw_rad"),
        frame,
        kind="phase",
    )
    frequency_column = _first_existing(
        ("cycle_flap_frequency_hz", "flap_frequency_hz", "encoder_rpm_est"), frame, kind="flap-frequency"
    )
    airspeed_column = _first_existing(
        (
            "airspeed_validated.true_airspeed_m_s",
            "airspeed_validated.calibrated_airspeed_m_s",
            "airspeed_validated.indicated_airspeed_m_s",
        ),
        frame,
        kind="airspeed",
    )
    density_column = "vehicle_air_data.rho" if "vehicle_air_data.rho" in frame.columns else None

    phase = frame[phase_column].astype(float)
    flap_frequency = frame[frequency_column].astype(float)
    if frequency_column == "encoder_rpm_est":
        flap_frequency = flap_frequency / 60.0
    true_airspeed = frame[airspeed_column].astype(float)
    rho = _series_or_default(frame, density_column, 1.225)
    dynamic_pressure = (
        frame["dynamic_pressure_pa"].astype(float)
        if "dynamic_pressure_pa" in frame.columns
        else 0.5 * rho * true_airspeed * true_airspeed
    )
    alpha_rad, aoa_source = _derive_alpha_rad(frame)

    features = pd.DataFrame(index=frame.index)
    features["phase_sin_1"] = np.sin(phase)
    features["phase_cos_1"] = np.cos(phase)
    features["phase_sin_2"] = np.sin(2.0 * phase)
    features["phase_cos_2"] = np.cos(2.0 * phase)
    features["alpha_rad"] = alpha_rad
    features["flap_frequency_hz"] = flap_frequency
    features["true_airspeed_m_s"] = true_airspeed
    features["dynamic_pressure_pa"] = dynamic_pressure
    features["alpha_rad_x_phase_sin_1"] = alpha_rad * features["phase_sin_1"]
    features["alpha_rad_x_phase_cos_1"] = alpha_rad * features["phase_cos_1"]
    features["flap_frequency_hz_x_phase_sin_1"] = flap_frequency * features["phase_sin_1"]
    features["flap_frequency_hz_x_phase_cos_1"] = flap_frequency * features["phase_cos_1"]
    features["true_airspeed_m_s_x_phase_sin_1"] = true_airspeed * features["phase_sin_1"]
    features["true_airspeed_m_s_x_phase_cos_1"] = true_airspeed * features["phase_cos_1"]
    features["alpha_rad_x_flap_frequency_hz"] = alpha_rad * flap_frequency

    spec = FeatureSpec(
        phase_column=phase_column,
        frequency_column=frequency_column,
        airspeed_column=airspeed_column,
        density_column=density_column,
        aoa_source=aoa_source,
        columns=features.columns.tolist(),
    )
    return features, spec


def _append_affine_force_features(features: pd.DataFrame, prior_force: np.ndarray) -> pd.DataFrame:
    affine = features.copy()
    base_columns = features.columns.tolist()
    for target_idx, target in enumerate(FORCE_TARGETS):
        for column in base_columns:
            affine[f"prior_{target}_x_{column}"] = prior_force[:, target_idx] * features[column].to_numpy(dtype=float)
    return affine


def design_features_for_variant(features: pd.DataFrame, prior_force: np.ndarray, variant: str) -> pd.DataFrame:
    if variant in {"additive", "multiplicative"}:
        return features
    if variant == "affine":
        return _append_affine_force_features(features, prior_force)
    raise ValueError(f"unknown model variant: {variant}")


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fill = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    x_filled = np.where(np.isfinite(x), x, fill)
    mean = np.mean(x_filled, axis=0)
    scale = np.std(x_filled, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    x_scaled = (x_filled - mean) / scale

    y_mean = np.mean(y, axis=0)
    y_centered = y - y_mean
    gram = x_scaled.T @ x_scaled
    if alpha > 0.0:
        gram = gram + float(alpha) * np.eye(gram.shape[0])
    coefficients = np.linalg.solve(gram, x_scaled.T @ y_centered)
    return coefficients, y_mean, mean, scale, fill


def _fit_variant(
    *,
    name: str,
    features: pd.DataFrame,
    prior_force: np.ndarray,
    true_force: np.ndarray,
    alpha: float,
) -> CorrectionModel:
    design = design_features_for_variant(features, prior_force, name)
    if name == "multiplicative":
        denominator = np.where(np.abs(prior_force) >= 1.0e-6, prior_force, np.sign(prior_force) * 1.0e-6 + 1.0e-6)
        target = (true_force - prior_force) / denominator
    else:
        target = true_force - prior_force
    coefficients, intercept, mean, scale, fill = _ridge_fit(design.to_numpy(dtype=float), target, alpha)
    return CorrectionModel(
        name=name,
        targets=FORCE_TARGETS,
        feature_columns=design.columns.tolist(),
        feature_mean=mean,
        feature_scale=scale,
        coefficients=coefficients,
        intercept=intercept,
        alpha=float(alpha),
        feature_fill=fill,
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


def force_metrics(
    *,
    split: str,
    model_name: str,
    true_force: np.ndarray,
    predicted_force: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for idx, target in enumerate(FORCE_TARGETS):
        metrics = _channel_metrics(true_force[:, idx], predicted_force[:, idx])
        rows.append(
            {
                "split": split,
                "model": model_name,
                "target": target,
                "n": int(np.isfinite(true_force[:, idx]).sum()),
                **metrics,
            }
        )
    rmse_values = [row["rmse"] for row in rows]
    r2_values = [row["r2"] for row in rows]
    rows.append(
        {
            "split": split,
            "model": model_name,
            "target": "force_mean",
            "n": int(len(true_force)),
            "mae": float(np.nanmean([row["mae"] for row in rows])),
            "rmse": float(np.sqrt(np.nanmean(np.square(rmse_values)))),
            "bias": float(np.nanmean([row["bias"] for row in rows])),
            "r2": float(np.nanmean(r2_values)),
        }
    )
    return pd.DataFrame(rows)


def _load_split(split_root: Path, prior_root: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame, FeatureSpec, np.ndarray, np.ndarray]:
    samples = pd.read_parquet(split_root / f"{split}_samples.parquet")
    prior = pd.read_parquet(prior_root / f"{split}_predictions.parquet")
    if len(samples) != len(prior):
        raise ValueError(f"{split} row count mismatch: samples={len(samples)} prior={len(prior)}")
    missing_targets = [target for target in FORCE_TARGETS if target not in samples.columns or target not in prior.columns]
    if missing_targets:
        raise ValueError(f"{split} is missing force target/prior columns: {missing_targets}")
    features, spec = build_feature_frame(samples)
    return samples, features, spec, samples.loc[:, FORCE_TARGETS].to_numpy(dtype=float), prior.loc[:, FORCE_TARGETS].to_numpy(dtype=float)


def _prediction_frame(
    samples: pd.DataFrame,
    features: pd.DataFrame,
    true_force: np.ndarray,
    prior_force: np.ndarray,
    corrected_force: np.ndarray,
    raw_correction: np.ndarray,
) -> pd.DataFrame:
    metadata_columns = [
        column
        for column in (
            "timestamp_us",
            "time_s",
            "log_id",
            "segment_id",
            "cycle_id",
            "phase_corrected_rad",
            "cycle_flap_frequency_hz",
            "flap_frequency_hz",
            "split",
        )
        if column in samples.columns
    ]
    out = samples.loc[:, metadata_columns].copy()
    for column in features.columns:
        out[column] = features[column].to_numpy(dtype=float)
    for idx, target in enumerate(FORCE_TARGETS):
        out[f"label_{target}"] = true_force[:, idx]
        out[f"prior_{target}"] = prior_force[:, idx]
        out[f"pred_{target}"] = raw_correction[:, idx]
        out[f"corrected_{target}"] = corrected_force[:, idx]
        out[f"remaining_{target}"] = true_force[:, idx] - corrected_force[:, idx]
    return out


def _write_model(output_root: Path, model: CorrectionModel, feature_spec: FeatureSpec, config: dict) -> None:
    model_config = {
        **config,
        "best_model": {
            "name": model.name,
            "alpha": model.alpha,
            "targets": list(model.targets),
            "feature_count": len(model.feature_columns),
        },
        "feature_spec": asdict(feature_spec),
    }
    (output_root / "model_config.json").write_text(json.dumps(model_config, indent=2, sort_keys=True), encoding="utf-8")
    feature_columns = {
        "base_feature_columns": feature_spec.columns,
        "model_feature_columns": model.feature_columns,
    }
    (output_root / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2, sort_keys=True), encoding="utf-8")
    np.savez(
        output_root / "coefficients_or_model.npz",
        coefficients=model.coefficients,
        intercept=model.intercept,
        feature_mean=model.feature_mean,
        feature_scale=model.feature_scale,
        feature_fill=model.feature_fill,
        feature_columns=np.array(model.feature_columns, dtype=object),
        targets=np.array(model.targets, dtype=object),
        name=np.array(model.name),
        alpha=np.array(model.alpha),
    )


def _write_diagnostics(output_root: Path, aligned_test: pd.DataFrame) -> None:
    from scripts.analyze_delaurier_residual_conditions import condition_bin_table, condition_summary_table
    from scripts.analyze_delaurier_residual_frequency import frequency_residual_energy_table, frequency_residual_summary_table
    from scripts.analyze_delaurier_residual_phase import phase_bin_table, phase_summary_table

    phase_table = phase_bin_table(aligned_test, targets=FORCE_TARGETS, phase_bins=36)
    phase_summary = phase_summary_table(aligned_test, phase_table, targets=FORCE_TARGETS)
    phase_summary.to_csv(output_root / "phase_residual_comparison.csv", index=False)
    phase_table.to_csv(output_root / "phase_residual_bins.csv", index=False)

    condition_columns = tuple(
        column
        for column in ("true_airspeed_m_s", "dynamic_pressure_pa", "alpha_rad", "flap_frequency_hz")
        if column in aligned_test.columns
    )
    condition_bins = condition_bin_table(
        aligned_test,
        condition_columns=condition_columns,
        targets=FORCE_TARGETS,
        quantile_bins=5,
        min_samples=500,
    )
    condition_summary = condition_summary_table(condition_bins)
    condition_summary.to_csv(output_root / "condition_bin_comparison.csv", index=False)
    condition_bins.to_csv(output_root / "condition_bins.csv", index=False)

    frequency_energy = frequency_residual_energy_table(aligned_test, targets=FORCE_TARGETS)
    frequency_summary = frequency_residual_summary_table(frequency_energy)
    frequency_summary.to_csv(output_root / "frequency_residual_comparison.csv", index=False)
    frequency_energy.to_csv(output_root / "frequency_residual_energy.csv", index=False)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""

    columns = [str(column) for column in frame.columns]
    rows = [[str(row[column]) for column in frame.columns] for _, row in frame.iterrows()]

    def render(values: list[str]) -> str:
        return "| " + " | ".join(values) + " |"

    separator = "| " + " | ".join("---" for _ in columns) + " |"
    return "\n".join([render(columns), separator, *(render(row) for row in rows)])


def _write_readme(
    output_root: Path,
    *,
    command: str,
    best_model: CorrectionModel,
    metrics: pd.DataFrame,
    split_root: Path,
    prior_root: Path,
) -> None:
    test_rows = metrics[
        (metrics["split"] == "test") & (metrics["model"] == best_model.name) & (metrics["target"].isin(list(FORCE_TARGETS) + ["force_mean"]))
    ]
    prior_rows = metrics[
        (metrics["split"] == "test") & (metrics["model"] == "prior") & (metrics["target"].isin(list(FORCE_TARGETS) + ["force_mean"]))
    ]
    lines = [
        "# DeLaurier Grey-Box Force Correction v1",
        "",
        "This artifact trains residual-guided effective force corrections on top of the exported DeLaurier prior.",
        "The learned coefficients are wrapper-level empirical corrections; they are not evidence that DeLaurier internal aerodynamic constants were physically reidentified.",
        "",
        "## Inputs",
        "",
        f"- Split root: `{split_root}`",
        f"- Prior root: `{prior_root}`",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Selected model",
        "",
        f"- Variant: `{best_model.name}`",
        f"- Ridge alpha: `{best_model.alpha:g}`",
        f"- Model feature count: `{len(best_model.feature_columns)}`",
        "",
        "## Test metrics",
        "",
        "### Current DeLaurier prior",
        "",
        dataframe_to_markdown(prior_rows.reset_index(drop=True)),
        "",
        "### Selected grey-box correction",
        "",
        dataframe_to_markdown(test_rows.reset_index(drop=True)),
        "",
        "## Outputs",
        "",
        "- `model_config.json`: training configuration and selected model metadata.",
        "- `feature_columns.json`: base and expanded model feature columns.",
        "- `coefficients_or_model.npz`: ridge coefficients, intercepts, scaling, fill values, and target names.",
        "- `metrics_by_split.csv`: train/val/test metrics for the prior and fitted candidates.",
        "- `prediction_parquets/`: aligned labels, prior force, predicted correction, corrected force, and residuals.",
        "- `phase_residual_comparison.csv`, `condition_bin_comparison.csv`, `frequency_residual_comparison.csv`: held-out residual diagnostics for the selected test predictions.",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_greybox_force_correction(
    *,
    split_root: Path = DEFAULT_SPLIT_ROOT,
    prior_root: Path = DEFAULT_PRIOR_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    variants: tuple[str, ...] = ("additive", "multiplicative", "affine"),
    alphas: tuple[float, ...] = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
    command: str = "",
) -> dict[str, Path | str]:
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_root / "prediction_parquets"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    loaded = {split: _load_split(split_root, prior_root, split) for split in ("train", "val", "test")}
    train_samples, train_base_features, train_spec, train_true, train_prior = loaded["train"]
    _, val_base_features, _, val_true, val_prior = loaded["val"]

    metrics_frames: list[pd.DataFrame] = []
    candidate_rows: list[dict[str, float | str]] = []
    for split, (_, _, _, true_force, prior_force) in loaded.items():
        metrics_frames.append(force_metrics(split=split, model_name="prior", true_force=true_force, predicted_force=prior_force))

    best_model: CorrectionModel | None = None
    best_val_rmse = float("inf")
    for variant in variants:
        for alpha in alphas:
            model = _fit_variant(
                name=variant,
                features=train_base_features,
                prior_force=train_prior,
                true_force=train_true,
                alpha=alpha,
            )
            for split, (_, base_features, _, true_force, prior_force) in loaded.items():
                design = design_features_for_variant(base_features, prior_force, variant)
                predicted = model.predict_force(design, prior_force)
                split_metrics = force_metrics(
                    split=split,
                    model_name=f"{variant}_alpha_{alpha:g}",
                    true_force=true_force,
                    predicted_force=predicted,
                )
                metrics_frames.append(split_metrics)
                if split == "val":
                    val_rmse = float(split_metrics.loc[split_metrics["target"] == "force_mean", "rmse"].iloc[0])
                    candidate_rows.append({"variant": variant, "alpha": float(alpha), "val_force_mean_rmse": val_rmse})
                    if val_rmse < best_val_rmse:
                        best_val_rmse = val_rmse
                        best_model = model

    if best_model is None:
        raise RuntimeError("no candidate model was fitted")

    all_metrics = [pd.concat(metrics_frames, ignore_index=True)]
    best_prediction_paths: dict[str, Path] = {}
    for split, (samples, base_features, _, true_force, prior_force) in loaded.items():
        design = design_features_for_variant(base_features, prior_force, best_model.name)
        raw = best_model.predict_raw(design)
        corrected = best_model.apply_correction(prior_force, raw)
        all_metrics.append(force_metrics(split=split, model_name=best_model.name, true_force=true_force, predicted_force=corrected))
        aligned = _prediction_frame(samples, base_features, true_force, prior_force, corrected, raw)
        path = prediction_dir / f"{split}_predictions.parquet"
        aligned.to_parquet(path, index=False)
        best_prediction_paths[split] = path

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(output_root / "metrics_by_split.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(output_root / "model_selection.csv", index=False)
    _write_model(
        output_root,
        best_model,
        train_spec,
        {
            "split_root": str(split_root),
            "prior_root": str(prior_root),
            "output_root": str(output_root),
            "variants": list(variants),
            "alphas": list(alphas),
            "selection_metric": "val force_mean RMSE",
            "best_val_force_mean_rmse": best_val_rmse,
        },
    )
    _write_diagnostics(output_root, pd.read_parquet(best_prediction_paths["test"]))
    _write_readme(
        output_root,
        command=command,
        best_model=best_model,
        metrics=metrics,
        split_root=split_root,
        prior_root=prior_root,
    )
    return {
        "output_root": output_root,
        "best_model": best_model.name,
        "metrics": output_root / "metrics_by_split.csv",
        "test_predictions": best_prediction_paths["test"],
    }


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--prior-root", type=Path, default=DEFAULT_PRIOR_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--variants", default="additive,multiplicative,affine")
    parser.add_argument("--alphas", default="0,0.001,0.01,0.1,1,10,100")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    command = (
        "python scripts/train_delaurier_greybox_force_correction.py "
        f"--split-root {args.split_root} --prior-root {args.prior_root} --output-root {args.output_root} "
        f"--variants {args.variants} --alphas {args.alphas}"
    )
    outputs = train_greybox_force_correction(
        split_root=args.split_root,
        prior_root=args.prior_root,
        output_root=args.output_root,
        variants=_parse_csv_strings(args.variants),
        alphas=_parse_csv_floats(args.alphas),
        command=command,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
