#!/usr/bin/env python3
"""Experiment 2: bounded regression calibration for the DeLaurier fx/fz prior."""

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

from scripts.prior_regression_calibration import ParameterSpec, solve_bounded_linearized_delta
from scripts.run_nested_prior_shaping_ablation_exp1 import (
    DEFAULT_ALPHA_GRID,
    DEFAULT_SPLITS_ROOT,
    PHASE_FREQ_Q_COLUMNS,
    SPLITS,
    TARGETS,
    _align_prior_to_subset,
    _array,
    _fx_fz_mean_rmse,
    _load_frames,
    _metrics_rows,
    _parse_alphas,
    _parse_folds,
    _prepare_output,
    _read_prior_all,
    _select_alpha_and_evaluate,
    _summary_by_fold,
    _write_json,
)
from scripts.sweep_delaurier_original_parameters import _export_prior


DEFAULT_EXPORT_SPLIT_ROOT = (
    PROJECT_ROOT / "dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1"
)
DEFAULT_METADATA = PROJECT_ROOT / "metadata/aircraft/flapper_01/aircraft_metadata.yaml"
DEFAULT_EXPORTER = Path("/home/zn/IsaacLab/scripts/flapping_px4/export_delaurier_prior_predictions.py")
DEFAULT_PYTHON_EXE = Path("/home/zn/anaconda3/envs/env_isaaclab/bin/python")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/20260703_prior_regression_calibration_exp2"
DEFAULT_PRIOR_BANK_ROOT = PROJECT_ROOT / "artifacts/20260703_prior_regression_calibration_prior_bank"
DEFAULT_LAMBDA_GRID = (0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0)

FIXED_PRIOR_PARAMS: dict[str, float | bool] = {
    "twist_eta_max_deg": 10.0,
    "alpha0_deg": 4.0,
    "eta_s": 0.65,
    "cd_f": 0.0,
    "enable_separation": True,
    "alpha_stall_max_deg": 18.0,
    "cd_cf": 1.2,
    "xi": 1.0,
}

PARAMETER_SPECS = (
    ParameterSpec("alpha0_deg", center=4.0, lower=0.0, upper=8.0, step=1.0, penalty_scale=4.0),
    ParameterSpec("alpha_stall_max_deg", center=18.0, lower=12.0, upper=24.0, step=2.0, penalty_scale=6.0),
    ParameterSpec("cd_cf", center=1.2, lower=0.6, upper=2.0, step=0.2, penalty_scale=0.7),
    ParameterSpec("xi", center=1.0, lower=0.0, upper=1.5, step=0.25, penalty_scale=0.5),
)


def _parse_lambdas(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def _slug_value(value: float | bool) -> str:
    if isinstance(value, bool):
        return "sep_on" if value else "sep_off"
    text = f"{float(value):.8g}".replace("-", "m").replace(".", "p")
    return text


def _prior_name(stage: str, params: dict[str, float | bool]) -> str:
    keys = (
        "twist_eta_max_deg",
        "alpha0_deg",
        "eta_s",
        "cd_f",
        "enable_separation",
        "alpha_stall_max_deg",
        "cd_cf",
        "xi",
    )
    parts = [stage]
    parts.extend(f"{key}_{_slug_value(params[key])}" for key in keys)
    return "__".join(parts)


def _params_from_theta(theta: np.ndarray, specs: tuple[ParameterSpec, ...]) -> dict[str, float | bool]:
    params = dict(FIXED_PRIOR_PARAMS)
    for spec, value in zip(specs, theta, strict=True):
        params[spec.name] = float(value)
    return params


def _center_params(specs: tuple[ParameterSpec, ...]) -> dict[str, float | bool]:
    return _params_from_theta(np.asarray([spec.center for spec in specs], dtype=float), specs)


def _basis_specs(specs: tuple[ParameterSpec, ...]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = [{"basis_name": "center", "parameter": None, "side": "center", "value": None}]
    for spec in specs:
        minus = max(spec.lower, spec.center - spec.step)
        plus = min(spec.upper, spec.center + spec.step)
        if np.isclose(minus, plus):
            raise ValueError(f"finite-difference step for {spec.name} collapses at bounds")
        records.append({"basis_name": f"{spec.name}_minus", "parameter": spec.name, "side": "minus", "value": minus})
        records.append({"basis_name": f"{spec.name}_plus", "parameter": spec.name, "side": "plus", "value": plus})
    return records


def _basis_params(record: dict[str, object], specs: tuple[ParameterSpec, ...]) -> dict[str, float | bool]:
    theta = np.asarray([spec.center for spec in specs], dtype=float)
    if record["parameter"] is not None:
        index = [spec.name for spec in specs].index(str(record["parameter"]))
        theta[index] = float(record["value"])
    return _params_from_theta(theta, specs)


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
    overwrite_priors: bool,
) -> None:
    _export_prior(
        python_exe=python_exe,
        exporter=exporter,
        split_root=export_split_root,
        metadata=metadata,
        output_root=prior_root,
        params=params,
        chunk_size=chunk_size,
        device=device,
        reuse_existing=not overwrite_priors,
    )


def _export_basis_priors(
    *,
    prior_bank_root: Path,
    export_split_root: Path,
    metadata: Path,
    exporter: Path,
    python_exe: Path,
    specs: tuple[ParameterSpec, ...],
    chunk_size: int,
    device: str,
    overwrite_priors: bool,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for basis in _basis_specs(specs):
        params = _basis_params(basis, specs)
        prior_root = prior_bank_root / "basis" / _prior_name(str(basis["basis_name"]), params)
        _export_params_prior(
            prior_root=prior_root,
            params=params,
            export_split_root=export_split_root,
            metadata=metadata,
            exporter=exporter,
            python_exe=python_exe,
            chunk_size=chunk_size,
            device=device,
            overwrite_priors=overwrite_priors,
        )
        records.append(
            {
                **basis,
                "prior_root": str(prior_root),
                **params,
            }
        )
    return pd.DataFrame(records)


def _target_arrays(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    labels = {
        split: _array(frame, tuple(f"label_{target}" for target in TARGETS))
        for split, frame in frames.items()
    }
    prior = {
        split: _array(frame, tuple(f"prior_{target}" for target in TARGETS))
        for split, frame in frames.items()
    }
    return labels, prior


def _jacobian_by_split(
    *,
    fold_frames: dict[str, pd.DataFrame],
    basis_records: pd.DataFrame,
    basis_prior_all: dict[str, pd.DataFrame],
    specs: tuple[ParameterSpec, ...],
) -> dict[str, np.ndarray]:
    jacobians: dict[str, np.ndarray] = {}
    for split, frame in fold_frames.items():
        jacobian = np.zeros((len(frame), len(TARGETS), len(specs)), dtype=float)
        for param_index, spec in enumerate(specs):
            minus_record = basis_records.loc[
                basis_records["basis_name"].eq(f"{spec.name}_minus")
            ].iloc[0]
            plus_record = basis_records.loc[
                basis_records["basis_name"].eq(f"{spec.name}_plus")
            ].iloc[0]
            minus = _align_prior_to_subset(frame, basis_prior_all[str(minus_record["basis_name"])])
            plus = _align_prior_to_subset(frame, basis_prior_all[str(plus_record["basis_name"])])
            denominator = float(plus_record["value"]) - float(minus_record["value"])
            jacobian[:, :, param_index] = (
                plus.loc[:, list(TARGETS)].to_numpy(dtype=float)
                - minus.loc[:, list(TARGETS)].to_numpy(dtype=float)
            ) / denominator
        jacobians[split] = jacobian
    return jacobians


def _linearized_prior(base_prior: np.ndarray, jacobian: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return base_prior + np.einsum("ntp,p->nt", jacobian, delta)


def _target_scales(labels: np.ndarray) -> np.ndarray:
    scales = np.nanstd(labels, axis=0)
    return np.where(np.isfinite(scales) & (scales > 1.0e-6), scales, 1.0)


def _select_lambda(
    *,
    frames: dict[str, pd.DataFrame],
    labels: dict[str, np.ndarray],
    base_prior: dict[str, np.ndarray],
    jacobians: dict[str, np.ndarray],
    specs: tuple[ParameterSpec, ...],
    lambdas: tuple[float, ...],
    outer_fold: int,
    fold_output_root: Path,
) -> tuple[dict[str, object], pd.DataFrame, object]:
    selection_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    scales = _target_scales(labels["train"])

    for lambda_value in lambdas:
        result = solve_bounded_linearized_delta(
            residual=labels["train"] - base_prior["train"],
            jacobian=jacobians["train"],
            parameter_specs=specs,
            lambda_value=float(lambda_value),
            target_scales=scales,
        )
        model_name = f"linearized_calibrated_prior_lambda_{float(lambda_value):g}"
        rows: list[dict[str, object]] = []
        for split in ("train", "val"):
            pred = _linearized_prior(base_prior[split], jacobians[split], result.delta)
            rows.extend(_metrics_rows(frames[split], pred, split=split, model=model_name))
        val_rmse = _fx_fz_mean_rmse(rows, split="val", model=model_name)
        selection_rows.append(
            {
                "outer_fold": int(outer_fold),
                "lambda": float(lambda_value),
                "val_rmse": val_rmse,
                "cost": result.cost,
                "optimality": result.optimality,
                "hit_bounds": ",".join(result.hit_bounds),
                **{f"theta_{spec.name}": float(value) for spec, value in zip(specs, result.theta, strict=True)},
                **{f"delta_{spec.name}": float(value) for spec, value in zip(specs, result.delta, strict=True)},
            }
        )
        metric_rows.extend(
            {
                **row,
                "outer_fold": int(outer_fold),
                "lambda": float(lambda_value),
                "target_scale_fx_b": float(scales[0]),
                "target_scale_fz_b": float(scales[1]),
            }
            for row in rows
        )

    selection = pd.DataFrame(selection_rows).sort_values(["val_rmse", "lambda"], kind="mergesort").reset_index(drop=True)
    selected = selection.iloc[0].to_dict()
    selection["is_selected"] = selection["lambda"].eq(float(selected["lambda"]))

    dev_labels = np.concatenate([labels["train"], labels["val"]], axis=0)
    dev_prior = np.concatenate([base_prior["train"], base_prior["val"]], axis=0)
    dev_jacobian = np.concatenate([jacobians["train"], jacobians["val"]], axis=0)
    final_result = solve_bounded_linearized_delta(
        residual=dev_labels - dev_prior,
        jacobian=dev_jacobian,
        parameter_specs=specs,
        lambda_value=float(selected["lambda"]),
        target_scales=_target_scales(dev_labels),
    )

    fold_output_root.mkdir(parents=True, exist_ok=True)
    selection.to_csv(fold_output_root / "prior_calibration_lambda_selection.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(fold_output_root / "prior_calibration_selection_metrics.csv", index=False)
    return selected, selection, final_result


def _display_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in ("calibrated_prior_only", "calibrated_gain_bias"):
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
    export_split_root: Path,
    metadata: Path,
    output_root: Path,
    prior_bank_root: Path,
    exporter: Path,
    python_exe: Path,
    folds: list[int],
    lambdas: tuple[float, ...],
    alphas: tuple[float, ...],
    chunk_size: int,
    device: str,
    force: bool,
    overwrite_priors: bool,
) -> dict[str, object]:
    _prepare_output(output_root, force=force)
    prior_bank_root.mkdir(parents=True, exist_ok=True)

    basis_records = _export_basis_priors(
        prior_bank_root=prior_bank_root,
        export_split_root=export_split_root,
        metadata=metadata,
        exporter=exporter,
        python_exe=python_exe,
        specs=PARAMETER_SPECS,
        chunk_size=chunk_size,
        device=device,
        overwrite_priors=overwrite_priors,
    )
    basis_records.to_csv(output_root / "prior_basis_records.csv", index=False)

    basis_prior_all = {
        str(row["basis_name"]): _read_prior_all(Path(str(row["prior_root"])))
        for _, row in basis_records.iterrows()
    }
    center_prior_all = basis_prior_all["center"]

    all_test_rows: list[dict[str, object]] = []
    all_per_log_rows: list[dict[str, object]] = []
    all_selection_metric_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    fold_manifests: dict[str, object] = {}

    for fold in folds:
        outer_fold = int(fold)
        fold_root = splits_root / f"fold_{outer_fold}"
        if not fold_root.exists():
            raise FileNotFoundError(fold_root)
        fold_output_root = output_root / "calibrated" / f"fold_{outer_fold}"

        frames = _load_frames(fold_root, center_prior_all)
        labels, base_prior = _target_arrays(frames)
        jacobians = _jacobian_by_split(
            fold_frames=frames,
            basis_records=basis_records,
            basis_prior_all=basis_prior_all,
            specs=PARAMETER_SPECS,
        )
        selected_lambda, _selection, final_calibration = _select_lambda(
            frames=frames,
            labels=labels,
            base_prior=base_prior,
            jacobians=jacobians,
            specs=PARAMETER_SPECS,
            lambdas=lambdas,
            outer_fold=outer_fold,
            fold_output_root=fold_output_root,
        )

        calibrated_params = _params_from_theta(final_calibration.theta, PARAMETER_SPECS)
        exact_prior_root = (
            prior_bank_root
            / "exact_calibrated"
            / f"fold_{outer_fold}__lambda_{_slug_value(float(selected_lambda['lambda']))}__"
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
            overwrite_priors=overwrite_priors,
        )
        exact_prior_all = _read_prior_all(exact_prior_root)
        test_rows, per_log_rows, selection_metric_rows, gain_bias_manifest = _select_alpha_and_evaluate(
            prior_name="calibrated",
            prior_root=exact_prior_root,
            fold_root=fold_root,
            fold_output_root=fold_output_root,
            prior_all=exact_prior_all,
            alphas=alphas,
        )

        calibration_row = {
            "outer_fold": outer_fold,
            "prior_name": "calibrated",
            "exact_prior_root": str(exact_prior_root),
            "lambda": float(selected_lambda["lambda"]),
            "inner_val_linearized_prior_rmse": float(selected_lambda["val_rmse"]),
            "linearized_fit_cost": final_calibration.cost,
            "linearized_fit_optimality": final_calibration.optimality,
            "hit_bounds": ",".join(final_calibration.hit_bounds),
            **{
                f"theta_{spec.name}": float(value)
                for spec, value in zip(PARAMETER_SPECS, final_calibration.theta, strict=True)
            },
            **{
                f"delta_{spec.name}": float(value)
                for spec, value in zip(PARAMETER_SPECS, final_calibration.delta, strict=True)
            },
            **calibrated_params,
        }
        calibration_rows.append(calibration_row)
        all_test_rows.extend(test_rows)
        all_per_log_rows.extend(per_log_rows)
        all_selection_metric_rows.extend(selection_metric_rows)

        manifest_path = fold_output_root / "manifest.json"
        fold_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fold_manifest["prior_calibration"] = calibration_row
        fold_manifest["prior_calibration_outputs"] = {
            "lambda_selection": str(fold_output_root / "prior_calibration_lambda_selection.csv"),
            "selection_metrics": str(fold_output_root / "prior_calibration_selection_metrics.csv"),
        }
        _write_json(manifest_path, fold_manifest)
        fold_manifests[f"fold_{outer_fold}"] = fold_manifest

    test_metrics = pd.DataFrame(all_test_rows)
    per_log_metrics = pd.DataFrame(all_per_log_rows)
    selection_metrics = pd.DataFrame(all_selection_metric_rows)
    calibration = pd.DataFrame(calibration_rows).sort_values("outer_fold").reset_index(drop=True)
    summary = _summary_by_fold(test_metrics)
    table = _display_table(summary)
    selected_alpha = (
        test_metrics.loc[test_metrics["model"].str.endswith("_gain_bias"), ["prior_name", "outer_fold", "alpha"]]
        .drop_duplicates()
        .sort_values(["prior_name", "outer_fold"])
        .reset_index(drop=True)
    )

    test_metrics.to_csv(output_root / "experiment2_outer_test_metrics.csv", index=False)
    per_log_metrics.to_csv(output_root / "experiment2_outer_test_per_log_metrics.csv", index=False)
    selection_metrics.to_csv(output_root / "experiment2_gain_bias_selection_metrics.csv", index=False)
    selected_alpha.to_csv(output_root / "experiment2_selected_alpha_by_fold.csv", index=False)
    calibration.to_csv(output_root / "experiment2_calibrated_parameters_by_fold.csv", index=False)
    summary.to_csv(output_root / "experiment2_per_target_summary.csv", index=False)
    table.to_csv(output_root / "experiment2_table_fx_fz_rmse.csv", index=False)

    exp1_table_path = PROJECT_ROOT / "artifacts/20260703_prior_shaping_ablation_exp1/experiment1_table_fx_fz_rmse.csv"
    if exp1_table_path.exists():
        exp1_table = pd.read_csv(exp1_table_path)
        comparison = pd.concat([exp1_table, table], ignore_index=True, sort=False)
        comparison.to_csv(output_root / "experiment2_comparison_with_exp1_table_fx_fz_rmse.csv", index=False)

    manifest = {
        "protocol": "six_fold_nested_whole_log_bounded_prior_regression_calibration",
        "experiment": "Experiment 2: bounded regression calibration plus deployable gain-bias correction",
        "splits_root": str(splits_root),
        "export_split_root": str(export_split_root),
        "metadata": str(metadata),
        "output_root": str(output_root),
        "prior_bank_root": str(prior_bank_root),
        "exporter": str(exporter),
        "python_exe": str(python_exe),
        "folds": folds,
        "lambda_grid": list(lambdas),
        "alpha_grid": list(alphas),
        "parameter_specs": [spec.__dict__ for spec in PARAMETER_SPECS],
        "fixed_prior_params": FIXED_PRIOR_PARAMS,
        "feature_columns": list(PHASE_FREQ_Q_COLUMNS),
        "selection_policy": (
            "for each outer fold, select the prior-calibration ridge lambda on inner validation logs, "
            "refit theta on train+val, export exact calibrated prior, then select gain-bias ridge alpha "
            "on inner validation logs and evaluate on held-out outer logs"
        ),
        "outputs": {
            "basis_records": str(output_root / "prior_basis_records.csv"),
            "calibrated_parameters": str(output_root / "experiment2_calibrated_parameters_by_fold.csv"),
            "outer_test_metrics": str(output_root / "experiment2_outer_test_metrics.csv"),
            "outer_test_per_log_metrics": str(output_root / "experiment2_outer_test_per_log_metrics.csv"),
            "gain_bias_selection_metrics": str(output_root / "experiment2_gain_bias_selection_metrics.csv"),
            "selected_alpha_by_fold": str(output_root / "experiment2_selected_alpha_by_fold.csv"),
            "per_target_summary": str(output_root / "experiment2_per_target_summary.csv"),
            "table": str(output_root / "experiment2_table_fx_fz_rmse.csv"),
        },
        "fold_manifests": {
            key: value.get("outputs", {}) for key, value in fold_manifests.items()
        },
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
    parser.add_argument("--lambdas", default=",".join(str(value) for value in DEFAULT_LAMBDA_GRID))
    parser.add_argument("--alphas", default=",".join(str(alpha) for alpha in DEFAULT_ALPHA_GRID))
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--device", default="cpu")
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
        lambdas=_parse_lambdas(args.lambdas),
        alphas=_parse_alphas(args.alphas),
        chunk_size=args.chunk_size,
        device=args.device,
        force=args.force,
        overwrite_priors=args.overwrite_priors,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
