#!/usr/bin/env python3
"""Additional diagnostics for Experiment 1 prior-shaping ablation."""

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

from scripts.analyze_delaurier_residual_frequency import frequency_residual_energy_table
from scripts.analyze_delaurier_residual_phase import phase_bin_table, phase_summary_table
from scripts.run_nested_prior_shaping_ablation_exp1 import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PRIORS,
    DEFAULT_SPLITS_ROOT,
    PHASE_FREQ_Q_COLUMNS,
    SPLITS,
    TARGETS,
    _array,
    _build_features,
    _fit_gain_bias,
    _load_frames,
    _parse_folds,
    _predict_gain_bias,
    _read_prior_all,
    _with_intercept,
)
from scripts.train_fx_fz_structured_correction import _gain_bias_design


PRIOR_PARAMS = {
    "nominal": {
        "stage": "nominal_attached",
        "twist_eta_max_deg": 10.0,
        "alpha0_deg": 0.0,
        "eta_s": 0.65,
        "cd_f": 0.0,
        "enable_separation": False,
        "alpha_stall_max_deg": 12.0,
        "cd_cf": 1.95,
        "xi": 0.0,
    },
    "shaped": {
        "stage": "selected_separation",
        "twist_eta_max_deg": 10.0,
        "alpha0_deg": 4.0,
        "eta_s": 0.65,
        "cd_f": 0.0,
        "enable_separation": True,
        "alpha_stall_max_deg": 18.0,
        "cd_cf": 1.2,
        "xi": 1.0,
    },
}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _effective_linear_terms(model: object) -> tuple[np.ndarray, float]:
    coefficients = np.asarray(model.coefficients, dtype=float)[:, 0]
    scale = np.asarray(model.feature_scale, dtype=float)
    mean = np.asarray(model.feature_mean, dtype=float)
    scale = np.where(np.abs(scale) > 1.0e-12, scale, 1.0)
    beta = coefficients / scale
    intercept = float(np.asarray(model.intercept, dtype=float)[0] - np.sum(mean * beta))
    return beta, intercept


def _decompose_gain_bias(
    *,
    models: list[object],
    frame: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    prior = _array(frame, tuple(f"prior_{target}" for target in TARGETS))
    phi = _with_intercept(features, list(PHASE_FREQ_Q_COLUMNS))
    rows: list[dict[str, object]] = []
    max_abs_error = 0.0

    for target_index, target in enumerate(TARGETS):
        model = models[target_index]
        design = _gain_bias_design(phi, prior, target_index)
        beta, intercept = _effective_linear_terms(model)
        beta_by_column = dict(zip(model.feature_columns, beta, strict=True))
        gain_coefficients = {
            column.removeprefix("gain_"): value
            for column, value in beta_by_column.items()
            if column.startswith("gain_")
        }
        bias_coefficients = {
            column.removeprefix("bias_"): value
            for column, value in beta_by_column.items()
            if column.startswith("bias_")
        }
        gain_factor = np.zeros(len(frame), dtype=float)
        for column, value in gain_coefficients.items():
            gain_factor += float(value) * phi[column].to_numpy(dtype=float)
        bias_component = np.full(len(frame), intercept, dtype=float)
        for column, value in bias_coefficients.items():
            bias_component += float(value) * phi[column].to_numpy(dtype=float)

        predicted_from_parts = gain_factor * prior[:, target_index] + bias_component
        predicted_direct = model.predict(design)[:, 0]
        max_abs_error = max(max_abs_error, float(np.nanmax(np.abs(predicted_from_parts - predicted_direct))))
        correction = predicted_direct - prior[:, target_index]
        gain_coeff_values = np.asarray(list(gain_coefficients.values()), dtype=float)
        bias_coeff_values = np.asarray(list(bias_coefficients.values()) + [intercept], dtype=float)
        rows.append(
            {
                "target": target,
                "gain_factor_mean": float(np.nanmean(gain_factor)),
                "gain_factor_std": float(np.nanstd(gain_factor)),
                "gain_factor_q05": float(np.nanquantile(gain_factor, 0.05)),
                "gain_factor_q50": float(np.nanquantile(gain_factor, 0.50)),
                "gain_factor_q95": float(np.nanquantile(gain_factor, 0.95)),
                "gain_factor_mean_abs_error_from_one": float(np.nanmean(np.abs(gain_factor - 1.0))),
                "gain_component_rmse": float(np.sqrt(np.nanmean((gain_factor * prior[:, target_index]) ** 2))),
                "bias_component_rmse": float(np.sqrt(np.nanmean(bias_component * bias_component))),
                "bias_component_mean": float(np.nanmean(bias_component)),
                "bias_component_std": float(np.nanstd(bias_component)),
                "total_correction_rmse": float(np.sqrt(np.nanmean(correction * correction))),
                "gain_coefficient_l2": float(np.sqrt(np.sum(gain_coeff_values * gain_coeff_values))),
                "gain_coefficient_mean_abs": float(np.mean(np.abs(gain_coeff_values))),
                "bias_coefficient_l2_including_intercept": float(np.sqrt(np.sum(bias_coeff_values * bias_coeff_values))),
                "bias_coefficient_mean_abs_including_intercept": float(np.mean(np.abs(bias_coeff_values))),
            }
        )
    return pd.DataFrame(rows), max_abs_error


def _diagnostic_frame(frame: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    out_columns = [
        column
        for column in (
            "log_id",
            "segment_id",
            "time_s",
            "cycle_flap_frequency_hz",
            "flap_frequency_hz",
            "phase_ratio8_clipped_rad",
            "phase_corrected_rad",
        )
        if column in frame.columns
    ]
    out = frame.loc[:, out_columns].copy()
    phase_source = "phase_ratio8_clipped_rad" if "phase_ratio8_clipped_rad" in frame.columns else "phase_corrected_rad"
    out["phase_corrected_rad"] = frame[phase_source].to_numpy(dtype=float)
    out["phase_source_column"] = phase_source
    for target_index, target in enumerate(TARGETS):
        prior = frame[f"prior_{target}"].to_numpy(dtype=float)
        out[f"label_{target}"] = frame[f"label_{target}"].to_numpy(dtype=float)
        out[f"prior_{target}"] = prior
        out[f"pred_{target}"] = pred[:, target_index] - prior
        out[f"force_v2_{target}"] = pred[:, target_index]
    return out


def _aggregate_mean_std(frame: pd.DataFrame, group_columns: list[str], value_columns: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_columns, dropna=False)
    out = grouped.size().rename("fold_rows").reset_index()
    for column in value_columns:
        stats = grouped[column].agg(["mean", "std", "min", "max"]).reset_index()
        stats = stats.rename(
            columns={
                "mean": f"{column}_mean",
                "std": f"{column}_std",
                "min": f"{column}_min",
                "max": f"{column}_max",
            }
        )
        out = out.merge(stats, on=group_columns, how="left")
    return out


def _combined_fold_report(
    *,
    experiment_root: Path,
    phase_all: pd.DataFrame,
    flap_main: pd.DataFrame,
    decomposition_all: pd.DataFrame,
    parameters: pd.DataFrame,
) -> pd.DataFrame:
    metrics = pd.read_csv(experiment_root / "experiment1_outer_test_metrics.csv")
    metric_parts = []
    for prior_name in ("nominal", "shaped"):
        prior_model = f"{prior_name}_prior_only"
        final_model = f"{prior_name}_gain_bias"
        prior = metrics.loc[
            metrics["model"].eq(prior_model) & metrics["target"].isin(TARGETS),
            ["prior_name", "outer_fold", "target", "rmse", "r2"],
        ].rename(columns={"rmse": "prior_only_rmse", "r2": "prior_only_r2"})
        final = metrics.loc[
            metrics["model"].eq(final_model) & metrics["target"].isin(TARGETS),
            ["prior_name", "outer_fold", "target", "rmse", "r2"],
        ].rename(columns={"rmse": "gain_bias_final_rmse", "r2": "gain_bias_final_r2"})
        metric_parts.append(prior.merge(final, on=["prior_name", "outer_fold", "target"], how="inner"))
    report = pd.concat(metric_parts, ignore_index=True)
    report = report.merge(
        phase_all.loc[
            :,
            [
                "prior_name",
                "outer_fold",
                "target",
                "true_phase_peak_to_peak",
                "remaining_phase_peak_to_peak",
                "phase_peak_to_peak_reduction_fraction",
            ],
        ].rename(
            columns={
                "true_phase_peak_to_peak": "prior_phase_peak_to_peak",
                "remaining_phase_peak_to_peak": "gain_bias_remaining_phase_peak_to_peak",
            }
        ),
        on=["prior_name", "outer_fold", "target"],
        how="left",
    )
    report = report.merge(
        flap_main.loc[
            :,
            [
                "prior_name",
                "outer_fold",
                "target",
                "true_energy",
                "remaining_energy",
                "true_energy_fraction",
                "remaining_energy_fraction_of_true",
                "energy_reduction_fraction",
            ],
        ].rename(
            columns={
                "true_energy": "prior_flap_fundamental_energy",
                "remaining_energy": "gain_bias_remaining_flap_fundamental_energy",
                "true_energy_fraction": "prior_flap_fundamental_energy_fraction",
            }
        ),
        on=["prior_name", "outer_fold", "target"],
        how="left",
    )
    report = report.merge(
        decomposition_all.loc[
            :,
            [
                "prior_name",
                "outer_fold",
                "target",
                "gain_factor_mean",
                "gain_factor_std",
                "gain_factor_mean_abs_error_from_one",
                "gain_coefficient_l2",
                "bias_component_rmse",
                "bias_coefficient_l2_including_intercept",
                "total_correction_rmse",
            ],
        ],
        on=["prior_name", "outer_fold", "target"],
        how="left",
    )
    report = report.merge(
        parameters.drop(columns=["prior_root"]),
        on=["prior_name", "outer_fold"],
        how="left",
    )
    return report.sort_values(["prior_name", "outer_fold", "target"]).reset_index(drop=True)


def run(
    *,
    splits_root: Path,
    experiment_root: Path,
    output_root: Path,
    folds: list[int],
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    selected_alpha_path = experiment_root / "experiment1_selected_alpha_by_fold.csv"
    if not selected_alpha_path.exists():
        raise FileNotFoundError(selected_alpha_path)
    selected_alpha = pd.read_csv(selected_alpha_path)

    phase_rows: list[pd.DataFrame] = []
    phase_bin_rows: list[pd.DataFrame] = []
    frequency_rows: list[pd.DataFrame] = []
    decomposition_rows: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    reconstruction_error_rows: list[dict[str, object]] = []

    for prior_name, prior_root in DEFAULT_PRIORS.items():
        prior_all = _read_prior_all(prior_root)
        for fold in folds:
            alpha_match = selected_alpha.loc[
                selected_alpha["prior_name"].astype(str).eq(prior_name)
                & selected_alpha["outer_fold"].astype(int).eq(int(fold)),
                "alpha",
            ]
            if len(alpha_match) != 1:
                raise ValueError(f"missing selected alpha for {prior_name} fold {fold}")
            alpha = float(alpha_match.iloc[0])
            fold_root = splits_root / f"fold_{fold}"
            frames = _load_frames(fold_root, prior_all)
            features, _ = _build_features(frames)
            dev_frame = pd.concat([frames["train"], frames["val"]], ignore_index=True)
            dev_features = pd.concat([features["train"], features["val"]], ignore_index=True)
            models = _fit_gain_bias(dev_frame, dev_features, alpha=alpha)
            pred_test = _predict_gain_bias(models, frames["test"], features["test"])

            diag = _diagnostic_frame(frames["test"], pred_test)
            phase_bins = phase_bin_table(diag, targets=TARGETS, phase_bins=36)
            phase_summary = phase_summary_table(diag, phase_bins, targets=TARGETS)
            frequency = frequency_residual_energy_table(diag, targets=TARGETS)
            decomposition, max_abs_reconstruction_error = _decompose_gain_bias(
                models=models,
                frame=frames["test"],
                features=features["test"],
            )

            for table in (phase_bins, phase_summary, frequency, decomposition):
                table.insert(0, "alpha", alpha)
                table.insert(0, "outer_fold", int(fold))
                table.insert(0, "prior_name", prior_name)

            phase_bin_rows.append(phase_bins)
            phase_rows.append(phase_summary)
            frequency_rows.append(frequency)
            decomposition_rows.append(decomposition)
            reconstruction_error_rows.append(
                {
                    "prior_name": prior_name,
                    "outer_fold": int(fold),
                    "alpha": alpha,
                    "max_abs_prediction_decomposition_error": max_abs_reconstruction_error,
                }
            )
            parameter_rows.append(
                {
                    "prior_name": prior_name,
                    "outer_fold": int(fold),
                    "alpha": alpha,
                    "prior_root": str(prior_root),
                    **PRIOR_PARAMS[prior_name],
                }
            )

    phase_bins_all = pd.concat(phase_bin_rows, ignore_index=True)
    phase_all = pd.concat(phase_rows, ignore_index=True)
    frequency_all = pd.concat(frequency_rows, ignore_index=True)
    decomposition_all = pd.concat(decomposition_rows, ignore_index=True)
    parameters = pd.DataFrame(parameter_rows)
    reconstruction_errors = pd.DataFrame(reconstruction_error_rows)
    flap_main = frequency_all.loc[frequency_all["component"].eq("flap_main")].copy().reset_index(drop=True)

    phase_summary = _aggregate_mean_std(
        phase_all,
        ["prior_name", "target"],
        [
            "true_residual_rmse",
            "remaining_residual_rmse",
            "true_phase_peak_to_peak",
            "remaining_phase_peak_to_peak",
            "phase_peak_to_peak_reduction_fraction",
        ],
    )
    flap_summary = _aggregate_mean_std(
        flap_main,
        ["prior_name", "target"],
        [
            "true_energy",
            "remaining_energy",
            "true_energy_fraction",
            "remaining_energy_fraction_of_true",
            "energy_reduction_fraction",
        ],
    )
    decomposition_summary = _aggregate_mean_std(
        decomposition_all,
        ["prior_name", "target"],
        [
            "gain_factor_mean",
            "gain_factor_std",
            "gain_factor_mean_abs_error_from_one",
            "bias_component_rmse",
            "total_correction_rmse",
            "gain_coefficient_l2",
            "bias_coefficient_l2_including_intercept",
        ],
    )
    fold_report = _combined_fold_report(
        experiment_root=experiment_root,
        phase_all=phase_all,
        flap_main=flap_main,
        decomposition_all=decomposition_all,
        parameters=parameters,
    )

    phase_bins_all.to_csv(output_root / "experiment1_phase_bins_by_fold.csv", index=False)
    phase_all.to_csv(output_root / "experiment1_phase_summary_by_fold.csv", index=False)
    frequency_all.to_csv(output_root / "experiment1_frequency_energy_by_fold.csv", index=False)
    flap_main.to_csv(output_root / "experiment1_flap_fundamental_energy_by_fold.csv", index=False)
    decomposition_all.to_csv(output_root / "experiment1_gain_bias_decomposition_by_fold.csv", index=False)
    parameters.to_csv(output_root / "experiment1_prior_parameters_by_fold.csv", index=False)
    reconstruction_errors.to_csv(output_root / "experiment1_decomposition_reconstruction_check.csv", index=False)
    phase_summary.to_csv(output_root / "experiment1_phase_summary_mean_std.csv", index=False)
    flap_summary.to_csv(output_root / "experiment1_flap_fundamental_energy_mean_std.csv", index=False)
    decomposition_summary.to_csv(output_root / "experiment1_gain_bias_decomposition_mean_std.csv", index=False)
    fold_report.to_csv(output_root / "experiment1_fold_report.csv", index=False)

    manifest = {
        "experiment_root": str(experiment_root),
        "output_root": str(output_root),
        "folds": folds,
        "targets": list(TARGETS),
        "diagnostics": {
            "phase_residual_peak_to_peak": str(output_root / "experiment1_phase_summary_by_fold.csv"),
            "flap_fundamental_energy": str(output_root / "experiment1_flap_fundamental_energy_by_fold.csv"),
            "gain_bias_decomposition": str(output_root / "experiment1_gain_bias_decomposition_by_fold.csv"),
            "prior_parameters": str(output_root / "experiment1_prior_parameters_by_fold.csv"),
            "combined_fold_report": str(output_root / "experiment1_fold_report.csv"),
        },
        "summary_outputs": {
            "phase": str(output_root / "experiment1_phase_summary_mean_std.csv"),
            "flap_fundamental": str(output_root / "experiment1_flap_fundamental_energy_mean_std.csv"),
            "gain_bias_decomposition": str(output_root / "experiment1_gain_bias_decomposition_mean_std.csv"),
        },
        "decomposition_reconstruction_max_abs_error": float(
            reconstruction_errors["max_abs_prediction_decomposition_error"].max()
        ),
        "phase_column_policy": "phase_ratio8_clipped_rad if present, otherwise phase_corrected_rad",
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "diagnostics")
    parser.add_argument("--folds", default="0,1,2,3,4,5")
    args = parser.parse_args()
    manifest = run(
        splits_root=args.splits_root,
        experiment_root=args.experiment_root,
        output_root=args.output_root,
        folds=_parse_folds(args.folds),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
