#!/usr/bin/env python3
"""Experiment 4: bounded attached-only DeLaurier prior calibration."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_nested_prior_regression_calibration_exp2 import (
    DEFAULT_EXPORT_SPLIT_ROOT,
    DEFAULT_EXPORTER,
    DEFAULT_METADATA,
    DEFAULT_PYTHON_EXE,
    _target_scales,
)
from scripts.run_nested_prior_shaping_ablation_exp1 import (
    DEFAULT_ALPHA_GRID,
    DEFAULT_SPLITS_ROOT,
    SPLITS,
    TARGETS,
    _align_prior_to_subset,
    _array,
    _fx_fz_mean_rmse,
    _metrics_rows,
    _parse_alphas,
    _parse_folds,
    _prepare_output,
    _read_prior_all,
    _select_alpha_and_evaluate,
    _summary_by_fold,
    _write_json,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/20260706_attached_prior_calibration_exp4"
DEFAULT_PRIOR_BANK_ROOT = PROJECT_ROOT / "artifacts/20260706_attached_prior_calibration_prior_bank"
DEFAULT_REGULARIZED_LAMBDA = 10.0

PARAMETER_NAMES = ("alpha0_deg", "eta_s", "cd_f")
CENTER_THETA = np.asarray([4.0, 0.65, 0.0], dtype=float)
PENALTY_SCALES = np.asarray([6.0, 0.35, 0.05], dtype=float)
LOWER_THETA = np.asarray([0.0, 0.0, 0.0], dtype=float)
UPPER_THETA = np.asarray([16.0, 1.0, 0.2], dtype=float)

FIXED_PRIOR_PARAMS: dict[str, float | bool] = {
    "twist_eta_max_deg": 10.0,
    "alpha0_deg": 4.0,
    "eta_s": 0.65,
    "cd_f": 0.0,
    "enable_separation": False,
    "theta_w_deg": 0.0,
    "alpha_stall_min_deg": -18.0,
    "alpha_stall_max_deg": 18.0,
    "cd_cf": 1.2,
    "xi": 0.0,
}


def _slug_value(value: float | bool) -> str:
    if isinstance(value, bool):
        return "sep_on" if value else "sep_off"
    text = f"{float(value):.8g}".replace("-", "m").replace(".", "p")
    return text


def _params_from_theta(theta: np.ndarray) -> dict[str, float | bool]:
    params = dict(FIXED_PRIOR_PARAMS)
    for name, value in zip(PARAMETER_NAMES, np.asarray(theta, dtype=float), strict=True):
        params[name] = float(value)
    return params


def _prior_name(stage: str, params: dict[str, float | bool]) -> str:
    keys = (
        "twist_eta_max_deg",
        "alpha0_deg",
        "eta_s",
        "cd_f",
        "theta_w_deg",
        "enable_separation",
    )
    parts = [stage]
    parts.extend(f"{key}_{_slug_value(params[key])}" for key in keys)
    return "__".join(parts)


def _load_exporter_module(exporter: Path):
    spec = importlib.util.spec_from_file_location("delaurier_prior_exporter", exporter)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load exporter module: {exporter}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AttachedPriorEvaluator:
    def __init__(
        self,
        *,
        sample_frame: pd.DataFrame,
        metadata: Path,
        exporter: Path,
        chunk_size: int,
        device: str,
    ) -> None:
        self.sample_frame = sample_frame
        self.metadata = metadata
        self.exporter = exporter
        self.chunk_size = int(chunk_size)
        self.device = device
        self.n_evaluations = 0
        self._exporter_module = _load_exporter_module(exporter)
        metadata_dict = self._exporter_module._load_metadata(metadata)
        self.amplitude_rad = self._exporter_module._metadata_value(
            metadata_dict,
            ("flapping_drive", "wing_stroke_amplitude_rad"),
            np.deg2rad(30.0),
        )
        self.phase_column = self._exporter_module.choose_first_column(
            sample_frame,
            self._exporter_module.PHASE_PRIORITY,
        )
        self.frequency_column = self._exporter_module.choose_first_column(
            sample_frame,
            self._exporter_module.FREQUENCY_PRIORITY,
        )
        self.wing_geom_csv = (
            self._exporter_module.PROJECT_ROOT / "outputs_DeLaurier" / "right_wing_te_fit_poly5_gap50.csv"
        )

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        params = _params_from_theta(np.asarray(theta, dtype=float))
        prior = self._exporter_module.compute_delaurier_force_prior(
            self.sample_frame,
            phase_column=self.phase_column,
            frequency_column=self.frequency_column,
            amplitude_rad=self.amplitude_rad,
            wing_geom_csv=self.wing_geom_csv,
            num_strips=80,
            chunk_size=self.chunk_size,
            device=self.device,
            alpha0_deg=float(params["alpha0_deg"]),
            eta_s=float(params["eta_s"]),
            cd_cf=float(params["cd_cf"]),
            alpha_stall_min_deg=float(params["alpha_stall_min_deg"]),
            alpha_stall_max_deg=float(params["alpha_stall_max_deg"]),
            xi=float(params["xi"]),
            c_mac=0.0,
            cd_f=float(params["cd_f"]),
            theta_w_deg=float(params["theta_w_deg"]),
            twist_eta_max_deg=float(params["twist_eta_max_deg"]),
            twist_eta_limit_deg=float(params["twist_eta_max_deg"]),
            enable_separation=False,
            stall_smoothing_width_deg=0.0,
            include_diagnostics=False,
        )
        self.n_evaluations += 1
        return prior.loc[:, list(TARGETS)].to_numpy(dtype=float)


def _sample_frames(fold_root: Path) -> dict[str, pd.DataFrame]:
    return {
        split: pd.read_parquet(fold_root / f"{split}_samples.parquet").reset_index(drop=True)
        for split in SPLITS
    }


def _label_array(frame: pd.DataFrame) -> np.ndarray:
    return _array(frame, tuple(TARGETS))


def _dev_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([frames["train"], frames["val"]], ignore_index=True)


def _fit_bounded_attached_theta(
    *,
    labels: np.ndarray,
    sample_frame: pd.DataFrame,
    lambda_value: float,
    metadata: Path,
    exporter: Path,
    chunk_size: int,
    device: str,
    initial_theta: np.ndarray,
    max_function_evaluations: int,
    diff_step: float,
) -> tuple[dict[str, object], AttachedPriorEvaluator]:
    labels = np.asarray(labels, dtype=float)
    scales = _target_scales(labels)
    evaluator = AttachedPriorEvaluator(
        sample_frame=sample_frame,
        metadata=metadata,
        exporter=exporter,
        chunk_size=chunk_size,
        device=device,
    )

    def residual_vector(theta: np.ndarray) -> np.ndarray:
        prediction = evaluator(theta)
        scaled = ((labels - prediction) / scales[None, :]).reshape(-1)
        if lambda_value <= 0:
            return scaled
        regularizer = np.sqrt(float(lambda_value)) * ((np.asarray(theta, dtype=float) - CENTER_THETA) / PENALTY_SCALES)
        return np.concatenate([scaled, regularizer])

    initial = np.clip(np.asarray(initial_theta, dtype=float), LOWER_THETA, UPPER_THETA)
    fit = least_squares(
        residual_vector,
        initial,
        bounds=(LOWER_THETA, UPPER_THETA),
        max_nfev=int(max_function_evaluations),
        diff_step=float(diff_step),
        xtol=1.0e-5,
        ftol=1.0e-8,
        gtol=1.0e-8,
        method="trf",
    )
    theta = np.clip(np.asarray(fit.x, dtype=float), LOWER_THETA, UPPER_THETA)
    residual = residual_vector(theta)
    normalized_delta = (theta - CENTER_THETA) / PENALTY_SCALES
    if lambda_value > 0:
        data_residual = residual[: labels.size]
        reg_loss = float(lambda_value) * float(np.sum(normalized_delta * normalized_delta))
    else:
        data_residual = residual
        reg_loss = 0.0
    data_loss = float(np.sum(data_residual * data_residual))
    result = {
        "theta": theta,
        "delta": theta - CENTER_THETA,
        "normalized_delta": normalized_delta,
        "success": bool(fit.success),
        "hit_bounds": [
            f"{name}:lower" for name, value, lower in zip(PARAMETER_NAMES, theta, LOWER_THETA, strict=True)
            if np.isclose(value, lower, rtol=0.0, atol=1.0e-7)
        ] + [
            f"{name}:upper" for name, value, upper in zip(PARAMETER_NAMES, theta, UPPER_THETA, strict=True)
            if np.isclose(value, upper, rtol=0.0, atol=1.0e-7)
        ],
        "message": str(fit.message),
        "n_function_evaluations": int(getattr(fit, "nfev", 0)),
        "n_prior_evaluations": int(evaluator.n_evaluations),
        "lambda": float(lambda_value),
        "data_loss": data_loss,
        "regularization_loss": reg_loss,
        "total_loss": data_loss + reg_loss,
    }
    return result, evaluator


def _export_prior_command(
    *,
    python_exe: Path,
    exporter: Path,
    split_root: Path,
    metadata: Path,
    output_root: Path,
    params: dict[str, float | bool],
    chunk_size: int,
    device: str,
) -> list[str]:
    return [
        str(python_exe),
        str(exporter),
        "--split-root",
        str(split_root),
        "--metadata",
        str(metadata),
        "--output-root",
        str(output_root),
        "--overwrite",
        "--chunk-size",
        str(int(chunk_size)),
        "--device",
        str(device),
        "--theta-w-deg",
        str(float(params["theta_w_deg"])),
        "--twist-eta-max-deg",
        str(float(params["twist_eta_max_deg"])),
        "--twist-eta-limit-deg",
        str(float(params["twist_eta_max_deg"])),
        "--alpha0-deg",
        str(float(params["alpha0_deg"])),
        "--eta-s",
        str(float(params["eta_s"])),
        "--cd-f",
        str(float(params["cd_f"])),
        "--alpha-stall-min-deg",
        str(float(params["alpha_stall_min_deg"])),
        "--alpha-stall-max-deg",
        str(float(params["alpha_stall_max_deg"])),
        "--cd-cf",
        str(float(params["cd_cf"])),
        "--xi",
        str(float(params["xi"])),
    ]


def _export_params_prior(
    *,
    prior_root: Path,
    params: dict[str, float | bool],
    export_split_root: Path,
    metadata: Path,
    exporter: Path,
    python_exe: Path,
    chunk_size: int,
    device: str,
    reuse_existing: bool,
) -> None:
    prior_root = prior_root.resolve()
    if reuse_existing and all((prior_root / f"{split}_predictions.parquet").exists() for split in SPLITS):
        return
    cmd = _export_prior_command(
        python_exe=python_exe,
        exporter=exporter,
        split_root=export_split_root,
        metadata=metadata,
        output_root=prior_root,
        params=params,
        chunk_size=chunk_size,
        device=device,
    )
    completed = subprocess.run(
        cmd,
        check=False,
        cwd=exporter.parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            cmd,
            output=completed.stdout,
            stderr=completed.stderr,
        )


def _prior_rmse_for_frame(frame: pd.DataFrame, prior_all: pd.DataFrame, model_name: str) -> float:
    prior = _align_prior_to_subset(frame, prior_all)
    eval_frame = frame.copy()
    for target in TARGETS:
        eval_frame[f"label_{target}"] = eval_frame[target].to_numpy(dtype=float)
    rows = _metrics_rows(eval_frame, prior.loc[:, list(TARGETS)].to_numpy(dtype=float), split="val", model=model_name)
    return _fx_fz_mean_rmse(rows, split="val", model=model_name)


def _display_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in summary["model"].drop_duplicates().tolist():
        model_summary = summary.loc[summary["model"].eq(model)]
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
    export_split_root: Path,
    metadata: Path,
    output_root: Path,
    prior_bank_root: Path,
    exporter: Path,
    python_exe: Path,
    folds: list[int],
    regularized_lambda: float,
    alphas: tuple[float, ...],
    chunk_size: int,
    device: str,
    max_function_evaluations: int,
    diff_step: float,
    force: bool,
    overwrite_priors: bool,
) -> dict[str, object]:
    output_root = output_root.resolve()
    prior_bank_root = prior_bank_root.resolve()
    splits_root = splits_root.resolve()
    export_split_root = export_split_root.resolve()
    metadata = metadata.resolve()
    exporter = exporter.resolve()
    python_exe = python_exe.resolve()
    _prepare_output(output_root, force=force)
    prior_bank_root.mkdir(parents=True, exist_ok=True)
    reuse_existing = not overwrite_priors

    calibration_methods = {
        "attached_bounded_unregularized": 0.0,
        "attached_bounded_regularized": float(regularized_lambda),
    }
    all_test_rows: list[dict[str, object]] = []
    all_per_log_rows: list[dict[str, object]] = []
    all_selection_metric_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    fold_manifests: dict[str, object] = {}

    for fold in folds:
        outer_fold = int(fold)
        fold_root = splits_root / f"fold_{outer_fold}"
        if not fold_root.exists():
            raise FileNotFoundError(fold_root)
        sample_frames = _sample_frames(fold_root)

        for prior_name, lambda_value in calibration_methods.items():
            method_output_root = output_root / prior_name / f"fold_{outer_fold}"
            method_output_root.mkdir(parents=True, exist_ok=True)
            method_prior_bank = prior_bank_root / prior_name / f"fold_{outer_fold}"

            train_result, train_evaluator = _fit_bounded_attached_theta(
                labels=_label_array(sample_frames["train"]),
                sample_frame=sample_frames["train"],
                lambda_value=lambda_value,
                metadata=metadata,
                exporter=exporter,
                chunk_size=chunk_size,
                device=device,
                initial_theta=CENTER_THETA,
                max_function_evaluations=max_function_evaluations,
                diff_step=diff_step,
            )
            train_params = _params_from_theta(train_result["theta"])
            train_prior_root = (
                method_prior_bank
                / "train_selected"
                / f"fold_{outer_fold}__lambda_{_slug_value(float(lambda_value))}__"
                f"{_prior_name('train_selected', train_params)}"
            )
            _export_params_prior(
                prior_root=train_prior_root,
                params=train_params,
                export_split_root=export_split_root,
                metadata=metadata,
                exporter=exporter,
                python_exe=python_exe,
                chunk_size=chunk_size,
                device=device,
                reuse_existing=reuse_existing,
            )
            train_prior_all = _read_prior_all(train_prior_root)
            val_rmse = _prior_rmse_for_frame(
                sample_frames["val"],
                train_prior_all,
                model_name=f"{prior_name}_train_selected_prior",
            )
            validation_rows.append(
                {
                    "prior_name": prior_name,
                    "outer_fold": outer_fold,
                    "lambda": float(lambda_value),
                    "fit_scope": "train",
                    "val_rmse": float(val_rmse),
                    "optimizer_success": train_result["success"],
                    "optimizer_message": train_result["message"],
                    "n_function_evaluations": train_result["n_function_evaluations"],
                    "n_prior_evaluations": train_evaluator.n_evaluations,
                    "optimizer": "least_squares_bounded",
                    "hit_bounds": ",".join(train_result["hit_bounds"]),
                    **{
                        f"theta_{name}": float(value)
                        for name, value in zip(PARAMETER_NAMES, train_result["theta"], strict=True)
                    },
                }
            )

            dev = _dev_frame(sample_frames)
            final_result, final_evaluator = _fit_bounded_attached_theta(
                labels=_label_array(dev),
                sample_frame=dev,
                lambda_value=lambda_value,
                metadata=metadata,
                exporter=exporter,
                chunk_size=chunk_size,
                device=device,
                initial_theta=train_result["theta"],
                max_function_evaluations=max_function_evaluations,
                diff_step=diff_step,
            )
            calibrated_params = _params_from_theta(final_result["theta"])
            exact_prior_root = (
                prior_bank_root
                / "exact_calibrated"
                / prior_name
                / f"fold_{outer_fold}__lambda_{_slug_value(float(lambda_value))}__"
                f"{_prior_name('calibrated', calibrated_params)}"
            )
            _export_params_prior(
                prior_root=exact_prior_root,
                params=calibrated_params,
                export_split_root=export_split_root,
                metadata=metadata,
                exporter=exporter,
                python_exe=python_exe,
                chunk_size=chunk_size,
                device=device,
                reuse_existing=reuse_existing,
            )
            exact_prior_all = _read_prior_all(exact_prior_root)
            test_rows, per_log_rows, selection_metric_rows, gain_bias_manifest = _select_alpha_and_evaluate(
                prior_name=prior_name,
                prior_root=exact_prior_root,
                fold_root=fold_root,
                fold_output_root=method_output_root,
                prior_all=exact_prior_all,
                alphas=alphas,
            )
            calibration_row = {
                "outer_fold": outer_fold,
                "prior_name": prior_name,
                "exact_prior_root": str(exact_prior_root),
                "lambda": float(lambda_value),
                "inner_val_exact_prior_rmse": float(val_rmse),
                "train_optimizer_success": train_result["success"],
                "train_optimizer_message": train_result["message"],
                "train_n_function_evaluations": train_result["n_function_evaluations"],
                "train_n_prior_evaluations": train_result["n_prior_evaluations"],
                "final_optimizer_success": final_result["success"],
                "final_optimizer_message": final_result["message"],
                "final_n_function_evaluations": final_result["n_function_evaluations"],
                "final_n_prior_evaluations": final_result["n_prior_evaluations"],
                "final_total_loss": final_result["total_loss"],
                "final_data_loss": final_result["data_loss"],
                "final_regularization_loss": final_result["regularization_loss"],
                "hit_bounds": ",".join(final_result["hit_bounds"]),
                **{
                    f"theta_{name}": float(value)
                    for name, value in zip(PARAMETER_NAMES, final_result["theta"], strict=True)
                },
                **{
                    f"delta_{name}": float(value)
                    for name, value in zip(PARAMETER_NAMES, final_result["delta"], strict=True)
                },
                **calibrated_params,
            }
            calibration_rows.append(calibration_row)
            all_test_rows.extend(test_rows)
            all_per_log_rows.extend(per_log_rows)
            all_selection_metric_rows.extend(selection_metric_rows)

            manifest_path = method_output_root / "manifest.json"
            fold_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            fold_manifest["prior_calibration"] = calibration_row
            fold_manifest["train_selection_prior_calibration"] = validation_rows[-1]
            fold_manifest["prior_calibration_outputs"] = {
                "validation_selection": str(output_root / "experiment4_validation_selection.csv"),
            }
            _write_json(manifest_path, fold_manifest)
            fold_manifests[f"{prior_name}_fold_{outer_fold}"] = fold_manifest

    test_metrics = pd.DataFrame(all_test_rows)
    per_log_metrics = pd.DataFrame(all_per_log_rows)
    selection_metrics = pd.DataFrame(all_selection_metric_rows)
    calibration = pd.DataFrame(calibration_rows).sort_values(["prior_name", "outer_fold"]).reset_index(drop=True)
    validation = pd.DataFrame(validation_rows).sort_values(["prior_name", "outer_fold"]).reset_index(drop=True)
    summary = _summary_by_fold(test_metrics)
    table = _display_table(summary)
    selected_alpha = (
        test_metrics.loc[test_metrics["model"].str.endswith("_gain_bias"), ["prior_name", "outer_fold", "alpha"]]
        .drop_duplicates()
        .sort_values(["prior_name", "outer_fold"])
        .reset_index(drop=True)
    )

    test_metrics.to_csv(output_root / "experiment4_outer_test_metrics.csv", index=False)
    per_log_metrics.to_csv(output_root / "experiment4_outer_test_per_log_metrics.csv", index=False)
    selection_metrics.to_csv(output_root / "experiment4_gain_bias_selection_metrics.csv", index=False)
    selected_alpha.to_csv(output_root / "experiment4_selected_alpha_by_fold.csv", index=False)
    calibration.to_csv(output_root / "experiment4_calibrated_parameters_by_fold.csv", index=False)
    validation.to_csv(output_root / "experiment4_validation_selection.csv", index=False)
    summary.to_csv(output_root / "experiment4_per_target_summary.csv", index=False)
    table.to_csv(output_root / "experiment4_table_fx_fz_rmse.csv", index=False)

    manifest = {
        "protocol": "six_fold_nested_whole_log_bounded_attached_only_prior_calibration",
        "experiment": "Experiment 4: attached-only three-parameter bounded calibration plus deployable gain-bias correction",
        "splits_root": str(splits_root),
        "export_split_root": str(export_split_root),
        "metadata": str(metadata),
        "output_root": str(output_root),
        "prior_bank_root": str(prior_bank_root),
        "exporter": str(exporter),
        "python_exe": str(python_exe),
        "folds": folds,
        "calibration_methods": calibration_methods,
        "alpha_grid": list(alphas),
        "max_function_evaluations": int(max_function_evaluations),
        "diff_step": float(diff_step),
        "parameter_names": list(PARAMETER_NAMES),
        "center_theta": CENTER_THETA.tolist(),
        "penalty_scales": PENALTY_SCALES.tolist(),
        "lower_theta": LOWER_THETA.tolist(),
        "upper_theta": UPPER_THETA.tolist(),
        "fixed_prior_params": FIXED_PRIOR_PARAMS,
        "selection_policy": (
            "for each outer fold and regularization condition, fit bounded attached-only theta on inner train logs, "
            "record inner validation prior RMSE, refit theta on train+val, export exact calibrated prior, "
            "then select gain-bias ridge alpha on inner validation logs and evaluate on held-out outer logs"
        ),
        "outputs": {
            "calibrated_parameters": str(output_root / "experiment4_calibrated_parameters_by_fold.csv"),
            "validation_selection": str(output_root / "experiment4_validation_selection.csv"),
            "outer_test_metrics": str(output_root / "experiment4_outer_test_metrics.csv"),
            "outer_test_per_log_metrics": str(output_root / "experiment4_outer_test_per_log_metrics.csv"),
            "gain_bias_selection_metrics": str(output_root / "experiment4_gain_bias_selection_metrics.csv"),
            "selected_alpha_by_fold": str(output_root / "experiment4_selected_alpha_by_fold.csv"),
            "per_target_summary": str(output_root / "experiment4_per_target_summary.csv"),
            "table": str(output_root / "experiment4_table_fx_fz_rmse.csv"),
        },
        "fold_manifests": {key: value.get("outputs", {}) for key, value in fold_manifests.items()},
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--export-split-root", type=Path, default=DEFAULT_EXPORT_SPLIT_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prior-bank-root", type=Path, default=DEFAULT_PRIOR_BANK_ROOT)
    parser.add_argument("--exporter", type=Path, default=DEFAULT_EXPORTER)
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON_EXE)
    parser.add_argument("--folds", default="0,1,2,3,4,5")
    parser.add_argument("--regularized-lambda", type=float, default=DEFAULT_REGULARIZED_LAMBDA)
    parser.add_argument("--alphas", default=",".join(str(alpha) for alpha in DEFAULT_ALPHA_GRID))
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-function-evaluations", type=int, default=80)
    parser.add_argument("--diff-step", type=float, default=1.0e-2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite-priors", action="store_true")
    args = parser.parse_args()
    manifest = run(
        splits_root=args.splits_root,
        export_split_root=args.export_split_root,
        metadata=args.metadata,
        output_root=args.output_root,
        prior_bank_root=args.prior_bank_root,
        exporter=args.exporter,
        python_exe=args.python_exe,
        folds=_parse_folds(args.folds),
        regularized_lambda=float(args.regularized_lambda),
        alphas=_parse_alphas(args.alphas),
        chunk_size=int(args.chunk_size),
        device=args.device,
        max_function_evaluations=int(args.max_function_evaluations),
        diff_step=float(args.diff_step),
        force=bool(args.force),
        overwrite_priors=bool(args.overwrite_priors),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
