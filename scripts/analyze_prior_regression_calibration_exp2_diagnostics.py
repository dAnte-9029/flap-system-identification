#!/usr/bin/env python3
"""Diagnostics for Experiment 2 bounded prior-regression calibration."""

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
from scripts.analyze_nested_prior_shaping_ablation_exp1_diagnostics import (
    _aggregate_mean_std,
    _decompose_gain_bias,
    _diagnostic_frame,
)
from scripts.run_nested_prior_regression_calibration_exp2 import DEFAULT_OUTPUT_ROOT
from scripts.run_nested_prior_shaping_ablation_exp1 import (
    DEFAULT_SPLITS_ROOT,
    TARGETS,
    _build_features,
    _fit_gain_bias,
    _load_frames,
    _parse_folds,
    _predict_gain_bias,
    _read_prior_all,
    _write_json,
)


def _combined_fold_report(
    *,
    experiment_root: Path,
    phase_all: pd.DataFrame,
    flap_main: pd.DataFrame,
    decomposition_all: pd.DataFrame,
    parameters: pd.DataFrame,
) -> pd.DataFrame:
    metrics = pd.read_csv(experiment_root / "experiment2_outer_test_metrics.csv")
    prior = metrics.loc[
        metrics["model"].eq("calibrated_prior_only") & metrics["target"].isin(TARGETS),
        ["prior_name", "outer_fold", "target", "rmse", "r2"],
    ].rename(columns={"rmse": "prior_only_rmse", "r2": "prior_only_r2"})
    final = metrics.loc[
        metrics["model"].eq("calibrated_gain_bias") & metrics["target"].isin(TARGETS),
        ["prior_name", "outer_fold", "target", "rmse", "r2"],
    ].rename(columns={"rmse": "gain_bias_final_rmse", "r2": "gain_bias_final_r2"})
    report = prior.merge(final, on=["prior_name", "outer_fold", "target"], how="inner")
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
    parameter_columns = [
        column
        for column in parameters.columns
        if column
        not in {
            "prior_name",
            "exact_prior_root",
        }
    ]
    report = report.merge(
        parameters.loc[:, ["prior_name", *parameter_columns]],
        on=["prior_name", "outer_fold"],
        how="left",
    )
    return report.sort_values(["outer_fold", "target"]).reset_index(drop=True)


def run(
    *,
    splits_root: Path,
    experiment_root: Path,
    output_root: Path,
    folds: list[int],
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    selected_alpha_path = experiment_root / "experiment2_selected_alpha_by_fold.csv"
    calibrated_path = experiment_root / "experiment2_calibrated_parameters_by_fold.csv"
    if not selected_alpha_path.exists():
        raise FileNotFoundError(selected_alpha_path)
    if not calibrated_path.exists():
        raise FileNotFoundError(calibrated_path)

    selected_alpha = pd.read_csv(selected_alpha_path)
    calibrated = pd.read_csv(calibrated_path)

    phase_rows: list[pd.DataFrame] = []
    phase_bin_rows: list[pd.DataFrame] = []
    frequency_rows: list[pd.DataFrame] = []
    decomposition_rows: list[pd.DataFrame] = []
    reconstruction_error_rows: list[dict[str, object]] = []

    for fold in folds:
        fold = int(fold)
        alpha_match = selected_alpha.loc[
            selected_alpha["prior_name"].astype(str).eq("calibrated")
            & selected_alpha["outer_fold"].astype(int).eq(fold),
            "alpha",
        ]
        parameter_match = calibrated.loc[calibrated["outer_fold"].astype(int).eq(fold)]
        if len(alpha_match) != 1:
            raise ValueError(f"missing selected alpha for calibrated fold {fold}")
        if len(parameter_match) != 1:
            raise ValueError(f"missing calibrated parameters for fold {fold}")
        alpha = float(alpha_match.iloc[0])
        prior_root = Path(str(parameter_match.iloc[0]["exact_prior_root"]))
        prior_all = _read_prior_all(prior_root)
        frames = _load_frames(splits_root / f"fold_{fold}", prior_all)
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
            table.insert(0, "outer_fold", fold)
            table.insert(0, "prior_name", "calibrated")

        phase_bin_rows.append(phase_bins)
        phase_rows.append(phase_summary)
        frequency_rows.append(frequency)
        decomposition_rows.append(decomposition)
        reconstruction_error_rows.append(
            {
                "prior_name": "calibrated",
                "outer_fold": fold,
                "alpha": alpha,
                "max_abs_prediction_decomposition_error": max_abs_reconstruction_error,
            }
        )

    phase_bins_all = pd.concat(phase_bin_rows, ignore_index=True)
    phase_all = pd.concat(phase_rows, ignore_index=True)
    frequency_all = pd.concat(frequency_rows, ignore_index=True)
    decomposition_all = pd.concat(decomposition_rows, ignore_index=True)
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
        parameters=calibrated,
    )

    phase_bins_all.to_csv(output_root / "experiment2_phase_bins_by_fold.csv", index=False)
    phase_all.to_csv(output_root / "experiment2_phase_summary_by_fold.csv", index=False)
    frequency_all.to_csv(output_root / "experiment2_frequency_energy_by_fold.csv", index=False)
    flap_main.to_csv(output_root / "experiment2_flap_fundamental_energy_by_fold.csv", index=False)
    decomposition_all.to_csv(output_root / "experiment2_gain_bias_decomposition_by_fold.csv", index=False)
    reconstruction_errors.to_csv(output_root / "experiment2_decomposition_reconstruction_check.csv", index=False)
    phase_summary.to_csv(output_root / "experiment2_phase_summary_mean_std.csv", index=False)
    flap_summary.to_csv(output_root / "experiment2_flap_fundamental_energy_mean_std.csv", index=False)
    decomposition_summary.to_csv(output_root / "experiment2_gain_bias_decomposition_mean_std.csv", index=False)
    fold_report.to_csv(output_root / "experiment2_fold_report.csv", index=False)

    exp1_report = PROJECT_ROOT / "artifacts/20260703_prior_shaping_ablation_exp1/diagnostics/experiment1_fold_report.csv"
    if exp1_report.exists():
        comparison = pd.concat([pd.read_csv(exp1_report), fold_report], ignore_index=True, sort=False)
        comparison.to_csv(output_root / "experiment2_comparison_with_exp1_fold_report.csv", index=False)

    manifest = {
        "experiment_root": str(experiment_root),
        "output_root": str(output_root),
        "folds": folds,
        "targets": list(TARGETS),
        "diagnostics": {
            "phase_residual_peak_to_peak": str(output_root / "experiment2_phase_summary_by_fold.csv"),
            "flap_fundamental_energy": str(output_root / "experiment2_flap_fundamental_energy_by_fold.csv"),
            "gain_bias_decomposition": str(output_root / "experiment2_gain_bias_decomposition_by_fold.csv"),
            "combined_fold_report": str(output_root / "experiment2_fold_report.csv"),
        },
        "summary_outputs": {
            "phase": str(output_root / "experiment2_phase_summary_mean_std.csv"),
            "flap_fundamental": str(output_root / "experiment2_flap_fundamental_energy_mean_std.csv"),
            "gain_bias_decomposition": str(output_root / "experiment2_gain_bias_decomposition_mean_std.csv"),
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
