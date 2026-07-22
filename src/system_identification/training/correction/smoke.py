"""Fixed-config, train-only smoke orchestration for the implemented C2 family."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Mapping

import numpy as np

from system_identification.artifacts.io import sha256_file, write_json
from system_identification.artifacts.static_correction_bundle import save_static_bundle
from system_identification.artifacts.static_correction_data import StaticCorrectionTrainingData
from system_identification.models.correction.prediction import predict_total
from system_identification.models.correction.specifications import StaticCorrectionSpec, StaticModelFamilyConfig
from system_identification.training.correction.fitting import fit_candidate


def _git_state(project_root: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=project_root, text=True).strip()

    status = run("status", "--porcelain=v1")
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_branch": run("branch", "--show-current"),
        "git_dirty_paths": status.splitlines(),
    }


def representative_smoke_specs(config: StaticModelFamilyConfig) -> tuple[StaticCorrectionSpec, ...]:
    """Return the fixed interface-smoke set; this is not a grid or selector."""

    defaults = config.smoke_defaults
    common = {
        "harmonic_order": int(defaults["harmonic_order"]),
        "condition_set": str(defaults["condition_set"]),
        "ridge_lambda_mean": float(defaults["ridge_lambda_mean"]),
        "ridge_lambda_waveform": float(defaults["ridge_lambda_waveform"]),
        "mean_weighting": str(defaults["mean_weighting"]),
        "waveform_weighting": str(defaults["waveform_weighting"]),
        "fit_intercept": True,
    }
    result: list[StaticCorrectionSpec] = []
    for component in config.force_components:
        result.append(
            StaticCorrectionSpec(
                model_type="raw_prior",
                force_component=component,
                harmonic_order=None,
                condition_set="none",
                mean_prior_retention=None,
                waveform_prior_retention=None,
                ridge_lambda_mean=0.0,
                ridge_lambda_waveform=0.0,
                mean_weighting="equal_cycle",
                waveform_weighting="equal_sample",
                fit_intercept=False,
            )
        )
        result.append(
            StaticCorrectionSpec(
                model_type="gain_bias",
                force_component=component,
                harmonic_order=None,
                condition_set="none",
                mean_prior_retention=None,
                waveform_prior_retention=None,
                ridge_lambda_mean=0.0,
                ridge_lambda_waveform=float(defaults["ridge_lambda_waveform"]),
                mean_weighting="equal_cycle",
                waveform_weighting=str(defaults["waveform_weighting"]),
                fit_intercept=True,
            )
        )
        result.append(
            StaticCorrectionSpec(
                model_type="fixed_prior_mean_wb",
                force_component=component,
                mean_prior_retention=1.0,
                waveform_prior_retention=1.0,
                **common,
            )
        )
        result.append(
            StaticCorrectionSpec(
                model_type="shaped_prior_mean_wb",
                force_component=component,
                mean_prior_retention=float(defaults["mean_prior_retention"]),
                waveform_prior_retention=float(defaults["waveform_prior_retention"]),
                **common,
            )
        )
        result.append(
            StaticCorrectionSpec(
                model_type="no_prior_mean_wb",
                force_component=component,
                mean_prior_retention=0.0,
                waveform_prior_retention=0.0,
                **common,
            )
        )
    return tuple(result)


def _rmse(residual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(residual, dtype=np.float64)))))


def _smoke_metrics(bundle, data: StaticCorrectionTrainingData) -> dict[str, object]:
    component = bundle.spec.force_component
    prediction = predict_total(bundle, data.cycle_frame, data.waveform_frame)
    truth = data.waveform_frame[f"label_{component}_n"].to_numpy(dtype=np.float64, copy=False)
    true_waveform = data.waveform_frame[f"label_{component}_waveform_n"].to_numpy(dtype=np.float64, copy=False)
    predicted_cycle = prediction.groupby("cycle_id", sort=False)["predicted_mean_n"].first()
    true_cycle = data.cycle_frame.set_index("cycle_id")[f"label_{component}_mean_n"].loc[predicted_cycle.index]
    solutions = [solution for solution in (bundle.mean_solution, bundle.waveform_solution) if solution is not None]
    coefficients = np.concatenate([solution.coefficients for solution in solutions]) if solutions else np.empty(0)
    conditions = [solution.diagnostics.condition_number for solution in solutions]
    ranks = [solution.diagnostics.matrix_rank for solution in solutions]
    weighted_norms = [solution.diagnostics.weighted_residual_norm for solution in solutions]
    effective_weight_sums = [solution.diagnostics.effective_weight_sum for solution in solutions]
    return {
        "metric_scope": "train_only_interface_smoke_not_for_selection",
        "train_rmse_n": _rmse(prediction["prediction_n"].to_numpy() - truth),
        "train_cycle_mean_rmse_n": _rmse(predicted_cycle.to_numpy() - true_cycle.to_numpy()),
        "train_waveform_rmse_n": _rmse(prediction["predicted_waveform_n"].to_numpy() - true_waveform),
        "coefficient_count": int(len(coefficients)),
        "maximum_absolute_coefficient": float(np.max(np.abs(coefficients))) if len(coefficients) else 0.0,
        "matrix_ranks": ranks,
        "condition_numbers": conditions,
        "weighted_residual_norms": weighted_norms,
        "effective_weight_sums": effective_weight_sums,
        "prediction_finite": bool(np.isfinite(prediction["prediction_n"]).all()),
        "waveform_cycle_mean_max_abs_n": float(
            prediction.groupby("cycle_id", sort=False)["predicted_waveform_n"].mean().abs().max()
        ),
    }


def run_static_correction_smoke(
    data: StaticCorrectionTrainingData,
    config: StaticModelFamilyConfig,
    output_root: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, object]:
    """Fit and serialize only the fixed representative smoke set."""

    root = Path(output_root)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite smoke output root: {root}")
    root.mkdir(parents=True, exist_ok=False)
    provenance = {**dict(data.provenance), **_git_state(Path(project_root))}
    rows: list[dict[str, object]] = []
    for spec in representative_smoke_specs(config):
        bundle = fit_candidate(
            spec,
            data.cycle_frame,
            data.waveform_frame,
            data.normalization,
            provenance,
            status="smoke_test",
        )
        bundle_dir = root / bundle.model_id
        save_static_bundle(bundle, bundle_dir)
        rows.append(
            {
                "model_id": bundle.model_id,
                "model_type": spec.model_type,
                "force_component": spec.force_component,
                "bundle_path": str(bundle_dir.resolve()),
                "bundle_hash": bundle.bundle_hash,
                **_smoke_metrics(bundle, data),
            }
        )

    availability = dict(data.component_availability)
    if availability.get("physical_component_scale_fz") == "available":
        component_spec = StaticCorrectionSpec(
            model_type="physical_component_scale",
            force_component="fz",
            harmonic_order=None,
            condition_set="none",
            mean_prior_retention=None,
            waveform_prior_retention=None,
            ridge_lambda_mean=0.0,
            ridge_lambda_waveform=float(config.smoke_defaults["ridge_lambda_waveform"]),
            mean_weighting="equal_cycle",
            waveform_weighting=str(config.smoke_defaults["waveform_weighting"]),
            fit_intercept=False,
            physical_component="normal_force",
            coefficient_constraints={"scale_min": 0.0, "scale_max": 2.0, "strategy": "clip_after_fit"},
        )
        bundle = fit_candidate(
            component_spec,
            data.cycle_frame,
            data.waveform_frame,
            data.normalization,
            provenance,
            status="smoke_test",
        )
        bundle_dir = root / bundle.model_id
        save_static_bundle(bundle, bundle_dir)
        rows.append(
            {
                "model_id": bundle.model_id,
                "model_type": component_spec.model_type,
                "force_component": "fz",
                "bundle_path": str(bundle_dir.resolve()),
                "bundle_hash": bundle.bundle_hash,
                **_smoke_metrics(bundle, data),
            }
        )
    write_json(root / "physical_component_availability.json", availability)

    after_hashes = {path: sha256_file(path) for path in data.input_hashes}
    if after_hashes != dict(data.input_hashes):
        raise RuntimeError("Correction-ready input artifact changed during smoke fitting")
    model_types = {row["model_type"] for row in rows}
    if len(model_types) < 5:
        raise RuntimeError("Representative smoke did not exercise the five core static candidate types")
    summary = {
        "schema_version": "static_correction_smoke_v1",
        "scope": "train_only_interface_smoke_not_model_selection",
        "included_partitions": ["train"],
        "validation_labels_loaded": False,
        "test_labels_loaded": False,
        "candidate_count": len(rows),
        "candidate_metrics": rows,
        "physical_component_availability": availability,
        "input_artifacts_unmodified": True,
        "selection_performed": False,
        "dynamic_model_trained": False,
        "provenance": provenance,
    }
    write_json(root / "smoke_summary.json", summary)
    return summary
