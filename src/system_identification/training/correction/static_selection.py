"""C3 train-CV search and sealed-shortlist construction."""

from __future__ import annotations

from dataclasses import replace
import json
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from system_identification.artifacts.io import sha256_file, write_json
from system_identification.artifacts.static_correction_data import StaticCorrectionTrainingData
from system_identification.evaluation.static_correction_metrics import (
    aggregate_per_log,
    per_log_mean_metrics,
    per_log_total_metrics,
    per_log_waveform_metrics,
    waveform_secondary_metrics,
)
from system_identification.models.correction.bundles import StaticCorrectionBundle
from system_identification.models.correction.features import build_mean_design, build_waveform_design
from system_identification.models.correction.prediction import predict_total
from system_identification.models.correction.specifications import StaticCorrectionSpec
from system_identification.models.correction.static_models import RidgeSolution
from system_identification.training.correction.fitting import (
    fit_candidate,
    fit_mean_branch,
    fit_waveform_branch,
)
from system_identification.training.correction.grouped_cv import (
    GroupedFoldManifest,
    build_date_aware_grouped_folds,
)
from system_identification.training.correction.selection_rules import (
    one_se_threshold,
    seal_shortlist,
    spec_complexity,
)
from system_identification.training.correction.selection_specs import (
    StaticSelectionConfig,
    canonical_hash,
    make_mean_wb_spec,
)

_WORKER_FOLDS: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] | None = None
_WORKER_DATA: StaticCorrectionTrainingData | None = None


def _mean_worker(spec: StaticCorrectionSpec) -> dict[str, object]:
    if _WORKER_FOLDS is None:
        raise RuntimeError("C3 mean worker was not initialized")
    return _mean_cv_record(spec, _WORKER_FOLDS)


def _waveform_worker(spec: StaticCorrectionSpec) -> dict[str, object]:
    if _WORKER_FOLDS is None:
        raise RuntimeError("C3 waveform worker was not initialized")
    return _waveform_cv_record(spec, _WORKER_FOLDS)


def _complete_worker(spec: StaticCorrectionSpec) -> dict[str, object]:
    if _WORKER_FOLDS is None or _WORKER_DATA is None:
        raise RuntimeError("C3 complete worker was not initialized")
    return _complete_cv_record(spec, _WORKER_FOLDS, _WORKER_DATA)


def _parallel_map(function, values: list[StaticCorrectionSpec], workers: int):
    if workers <= 1:
        return [function(value) for value in values]
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        return pool.map(function, values)


def _candidate_id(prefix: str, value: Mapping[str, object]) -> str:
    return f"{prefix}_{canonical_hash(value)[:12]}"


def _fold_frames(
    cycle: pd.DataFrame,
    waveform: pd.DataFrame,
    fold_manifest: GroupedFoldManifest,
) -> Iterable[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    for fold in fold_manifest.folds:
        held_logs = set(str(item) for item in fold["log_ids"])
        cycle_held = cycle[cycle["log_id"].astype(str).isin(held_logs)].copy()
        cycle_fit = cycle[~cycle["log_id"].astype(str).isin(held_logs)].copy()
        waveform_held = waveform[waveform["log_id"].astype(str).isin(held_logs)].copy()
        waveform_fit = waveform[~waveform["log_id"].astype(str).isin(held_logs)].copy()
        yield cycle_fit, cycle_held, waveform_fit, waveform_held


def _predict_mean_solution(spec: StaticCorrectionSpec, solution: RidgeSolution, frame: pd.DataFrame) -> np.ndarray:
    design = build_mean_design(frame, spec)
    prediction = design.values @ solution.coefficients
    if spec.mean_prior_retention:
        prediction = prediction + float(spec.mean_prior_retention) * frame[
            f"prior_{spec.force_component}_mean_n"
        ].to_numpy(dtype=np.float64)
    return prediction


def _predict_waveform_solution(
    spec: StaticCorrectionSpec,
    solution: RidgeSolution,
    frame: pd.DataFrame,
) -> np.ndarray:
    design = build_waveform_design(frame, spec)
    prediction = design.values @ solution.coefficients
    if spec.waveform_prior_retention:
        prediction = prediction + float(spec.waveform_prior_retention) * frame[
            f"prior_{spec.force_component}_waveform_n"
        ].to_numpy(dtype=np.float64)
    return prediction


def _branch_spec(
    component: str,
    *,
    mean_retention: float = 0.0,
    waveform_retention: float = 0.0,
    mean_condition: str = "none",
    waveform_condition: str = "none",
    harmonic_order: int = 1,
    ridge_mean: float = 0.0,
    ridge_waveform: float = 0.0,
    mean_weighting: str = "equal_log",
    waveform_weighting: str = "equal_log",
) -> StaticCorrectionSpec:
    return make_mean_wb_spec(
        component,
        mean_retention=mean_retention,
        waveform_retention=waveform_retention,
        mean_condition=mean_condition,
        waveform_condition=waveform_condition,
        harmonic_order=harmonic_order,
        ridge_mean=ridge_mean,
        ridge_waveform=ridge_waveform,
        mean_weighting=mean_weighting,
        waveform_weighting=waveform_weighting,
    )


def _mean_cv_record(
    spec: StaticCorrectionSpec,
    folds: Iterable[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> dict[str, object]:
    per_logs = []
    diagnostics = []
    coefficient_norms = []
    for cycle_fit, cycle_held, _, _ in folds:
        solution = fit_mean_branch(spec, cycle_fit)
        prediction = _predict_mean_solution(spec, solution, cycle_held)
        per_logs.append(per_log_mean_metrics(cycle_held, prediction, spec.force_component))
        diagnostics.append(solution.diagnostics)
        coefficient_norms.append(float(np.linalg.norm(solution.coefficients)))
    per_log = pd.concat(per_logs, ignore_index=True).sort_values("log_id")
    values = per_log["mean_rmse"].to_numpy(dtype=np.float64)
    coefficient_count = max(item.coefficient_count for item in diagnostics)
    condition_number = max(item.condition_number for item in diagnostics)
    payload = {
        "component": spec.force_component,
        "branch": "mean",
        "mean_prior_retention": float(spec.mean_prior_retention),
        "mean_condition_set": spec.mean_condition_set,
        "ridge_lambda_mean": spec.ridge_lambda_mean,
        "mean_weighting": spec.mean_weighting,
    }
    return {
        "candidate_id": _candidate_id("mean", payload),
        **payload,
        "macro_mean_rmse": float(values.mean()),
        "mean_mae": float(per_log["mean_mae"].mean()),
        "mean_bias": float(per_log["mean_bias"].mean()),
        "worst_log_mean_rmse": float(values.max()),
        "median_log_mean_rmse": float(np.median(values)),
        "fold_standard_deviation": float(values.std(ddof=1)),
        "log_standard_error": float(values.std(ddof=1) / np.sqrt(len(values))),
        "coefficient_count": coefficient_count,
        "coefficient_norm_mean": float(np.mean(coefficient_norms)),
        "maximum_condition_number": condition_number,
        "rank_deficient": any(item.rank_deficient for item in diagnostics),
        "per_log_errors_json": json.dumps(
            dict(zip(per_log["log_id"].astype(str), per_log["mean_rmse"].astype(float))),
            sort_keys=True,
        ),
        "spec_json": json.dumps(spec.to_dict(), sort_keys=True),
    }


def _waveform_cv_record(
    spec: StaticCorrectionSpec,
    folds: Iterable[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    *,
    include_secondary: bool = False,
) -> dict[str, object]:
    per_logs = []
    diagnostics = []
    coefficient_norms = []
    secondary = []
    for _, _, waveform_fit, waveform_held in folds:
        solution = fit_waveform_branch(spec, waveform_fit)
        prediction = _predict_waveform_solution(spec, solution, waveform_held)
        per_logs.append(per_log_waveform_metrics(waveform_held, prediction, spec.force_component))
        if include_secondary:
            secondary.append(waveform_secondary_metrics(waveform_held, prediction, spec.force_component))
        diagnostics.append(solution.diagnostics)
        coefficient_norms.append(float(np.linalg.norm(solution.coefficients)))
    per_log = pd.concat(per_logs, ignore_index=True).sort_values("log_id")
    values = per_log["waveform_rmse"].to_numpy(dtype=np.float64)
    payload = {
        "component": spec.force_component,
        "branch": "waveform",
        "waveform_prior_retention": float(spec.waveform_prior_retention),
        "harmonic_order": int(spec.harmonic_order),
        "waveform_condition_set": spec.waveform_condition_set,
        "ridge_lambda_waveform": spec.ridge_lambda_waveform,
        "waveform_weighting": spec.waveform_weighting,
    }
    return {
        "candidate_id": _candidate_id("waveform", payload),
        **payload,
        "macro_waveform_rmse": float(values.mean()),
        "worst_log_waveform_rmse": float(values.max()),
        "median_log_waveform_rmse": float(np.median(values)),
        "fold_standard_deviation": float(values.std(ddof=1)),
        "log_standard_error": float(values.std(ddof=1) / np.sqrt(len(values))),
        "phase_bin_waveform_rmse": (
            float(np.mean([item["phase_bin_waveform_rmse"] for item in secondary])) if secondary else None
        ),
        "upstroke_integral_error": (
            float(np.mean([item["upstroke_integral_error"] for item in secondary])) if secondary else None
        ),
        "downstroke_integral_error": (
            float(np.mean([item["downstroke_integral_error"] for item in secondary])) if secondary else None
        ),
        "peak_magnitude_error": (
            float(np.mean([item["peak_magnitude_error"] for item in secondary])) if secondary else None
        ),
        "circular_peak_phase_error": (
            float(np.mean([item["circular_peak_phase_error"] for item in secondary])) if secondary else None
        ),
        "coefficient_count": max(item.coefficient_count for item in diagnostics),
        "coefficient_norm_mean": float(np.mean(coefficient_norms)),
        "maximum_condition_number": max(item.condition_number for item in diagnostics),
        "rank_deficient": any(item.rank_deficient for item in diagnostics),
        "per_log_errors_json": json.dumps(
            dict(zip(per_log["log_id"].astype(str), per_log["waveform_rmse"].astype(float))),
            sort_keys=True,
        ),
        "spec_json": json.dumps(spec.to_dict(), sort_keys=True),
    }


def _simple_branch_key(row: Mapping[str, object], branch: str) -> tuple[object, ...]:
    if branch == "mean":
        return (
            int(row["coefficient_count"]),
            {"none": 0, "alpha": 1, "frequency": 1, "alpha_frequency": 2}[str(row["mean_condition_set"])],
            int(float(row["mean_prior_retention"]) not in {0.0, 1.0}),
            float(row["maximum_condition_number"]),
            str(row["candidate_id"]),
        )
    return (
        int(row["coefficient_count"]),
        int(row["harmonic_order"]),
        {"none": 0, "alpha": 1, "frequency": 1, "alpha_frequency": 2}[
            str(row["waveform_condition_set"])
        ],
        int(float(row["waveform_prior_retention"]) not in {0.0, 1.0}),
        float(row["maximum_condition_number"]),
        str(row["candidate_id"]),
    )


def _branch_shortlist(frame: pd.DataFrame, branch: str, limit: int) -> pd.DataFrame:
    metric = "macro_mean_rmse" if branch == "mean" else "macro_waveform_rmse"
    retention = "mean_prior_retention" if branch == "mean" else "waveform_prior_retention"
    best = frame.sort_values([metric, "candidate_id"], kind="stable").iloc[0]
    errors = list(json.loads(best["per_log_errors_json"]).values())
    threshold, _ = one_se_threshold(errors)
    eligible = frame[frame[metric] <= threshold + 1e-12]
    simple = min((row for _, row in eligible.iterrows()), key=lambda row: _simple_branch_key(row, branch))
    reasons: dict[str, set[str]] = {}

    def add(row: pd.Series, reason: str) -> None:
        reasons.setdefault(str(row["candidate_id"]), set()).add(reason)

    add(best, "train_cv_primary_best")
    add(simple, "train_cv_one_se_simplest")
    no_prior = frame[frame[retention] == 0.0]
    if len(no_prior):
        add(no_prior.sort_values([metric, "candidate_id"], kind="stable").iloc[0], "best_no_prior")
    prior = frame[frame[retention] > 0.0]
    if len(prior):
        add(prior.sort_values([metric, "candidate_id"], kind="stable").iloc[0], "best_prior_retaining")
    if branch == "waveform":
        near = frame[frame[metric] <= threshold + 1e-12]
        for _, row in near.sort_values([metric, "harmonic_order"], kind="stable").iterrows():
            if int(row["harmonic_order"]) not in {
                int(frame.loc[frame["candidate_id"] == candidate_id, "harmonic_order"].iloc[0])
                for candidate_id in reasons
            }:
                add(row, "near_one_se_distinct_harmonic_order")
                break
    ordered_ids = sorted(
        reasons,
        key=lambda candidate_id: (
            float(frame.loc[frame["candidate_id"] == candidate_id, metric].iloc[0]),
            candidate_id,
        ),
    )
    mandatory = [str(best["candidate_id"]), str(simple["candidate_id"])]
    ordered_ids = list(dict.fromkeys(mandatory + ordered_ids))[:limit]
    result = frame[frame["candidate_id"].isin(ordered_ids)].copy()
    result["shortlist_reason"] = result["candidate_id"].map(
        {candidate_id: ";".join(sorted(reasons[candidate_id])) for candidate_id in ordered_ids}
    )
    return result.sort_values(metric, kind="stable").reset_index(drop=True)


def _refine_mean(
    initial: pd.DataFrame,
    config: StaticSelectionConfig,
    fold_frames: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    seed = pd.concat(
        [
            _branch_shortlist(
                initial[initial["component"] == component],
                "mean",
                config.shortlist["mean_branch_limit"],
            )
            for component in config.force_components
        ],
        ignore_index=True,
    )
    for _, row in seed.iterrows():
        base = StaticCorrectionSpec.from_dict(json.loads(row["spec_json"]))
        for weighting in config.weighting_refinement["mean"]:
            rows.append(_mean_cv_record(replace(base, mean_weighting=weighting), fold_frames))
    return pd.DataFrame(rows)


def _refine_waveform(
    initial: pd.DataFrame,
    config: StaticSelectionConfig,
    fold_frames: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    seed = pd.concat(
        [
            _branch_shortlist(
                initial[initial["component"] == component],
                "waveform",
                config.shortlist["waveform_branch_limit"],
            )
            for component in config.force_components
        ],
        ignore_index=True,
    )
    for _, row in seed.iterrows():
        base = StaticCorrectionSpec.from_dict(json.loads(row["spec_json"]))
        for weighting in config.weighting_refinement["waveform"]:
            rows.append(
                _waveform_cv_record(
                    replace(base, waveform_weighting=weighting),
                    fold_frames,
                    include_secondary=True,
                )
            )
    return pd.DataFrame(rows)


def _prediction_metric_table(
    waveform_frame: pd.DataFrame,
    prediction: pd.DataFrame,
    component: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    table = waveform_frame.loc[
        :,
        [
            "cycle_id",
            "log_id",
            "flight_date",
            "timestamp_us",
            "phase_rad",
            "half_stroke_id",
            f"label_{component}_n",
            f"label_{component}_waveform_n",
        ],
    ].reset_index(drop=True)
    table = table.rename(
        columns={
            f"label_{component}_n": "label_n",
            f"label_{component}_waveform_n": "label_waveform_n",
        }
    )
    table["prediction_n"] = prediction["prediction_n"].to_numpy(dtype=np.float64)
    table["predicted_waveform_n"] = prediction["predicted_waveform_n"].to_numpy(dtype=np.float64)
    per_log = per_log_total_metrics(table)
    total = aggregate_per_log(per_log)
    waveform = per_log_waveform_metrics(
        waveform_frame, table["predicted_waveform_n"].to_numpy(dtype=np.float64), component
    )
    secondary = waveform_secondary_metrics(
        waveform_frame, table["predicted_waveform_n"].to_numpy(dtype=np.float64), component
    )
    metrics = {
        "macro_total_rmse": total["macro_rmse"],
        "median_log_total_rmse": total["median_log_rmse"],
        "worst_log_total_rmse": total["worst_log_rmse"],
        "macro_waveform_rmse": float(waveform["waveform_rmse"].mean()),
        **secondary,
    }
    return per_log, metrics


def _complete_cv_record(
    spec: StaticCorrectionSpec,
    folds: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    data: StaticCorrectionTrainingData,
) -> dict[str, object]:
    per_logs = []
    fold_mean = []
    fold_waveform = []
    diagnostics = []
    corrections = []
    priors = []
    for cycle_fit, cycle_held, waveform_fit, waveform_held in folds:
        bundle = fit_candidate(
            spec,
            cycle_fit,
            waveform_fit,
            data.normalization,
            data.provenance,
        )
        prediction = predict_total(bundle, cycle_held, waveform_held)
        per_log, metrics = _prediction_metric_table(waveform_held, prediction, spec.force_component)
        per_logs.append(per_log)
        mean_map = prediction.groupby("cycle_id", sort=False)["predicted_mean_n"].first()
        cycle_prediction = cycle_held["cycle_id"].map(mean_map).to_numpy(dtype=np.float64)
        fold_mean.append(per_log_mean_metrics(cycle_held, cycle_prediction, spec.force_component))
        fold_waveform.append(metrics["macro_waveform_rmse"])
        corrections.append(
            prediction["prediction_n"].to_numpy(dtype=np.float64)
            - waveform_held[f"prior_{spec.force_component}_n"].to_numpy(dtype=np.float64)
        )
        priors.append(waveform_held[f"prior_{spec.force_component}_n"].to_numpy(dtype=np.float64))
        for solution in (bundle.mean_solution, bundle.waveform_solution):
            if solution is not None:
                diagnostics.append(solution.diagnostics)
    per_log = pd.concat(per_logs, ignore_index=True).sort_values("log_id")
    mean_per_log = pd.concat(fold_mean, ignore_index=True)
    correction = np.concatenate(corrections)
    prior = np.concatenate(priors)
    payload = spec.to_dict()
    candidate_id = _candidate_id("complete", payload)
    full_bundle = fit_candidate(
        spec,
        data.cycle_frame,
        data.waveform_frame,
        data.normalization,
        data.provenance,
    )
    coefficient_count = sum(
        len(solution.coefficients)
        for solution in (full_bundle.mean_solution, full_bundle.waveform_solution)
        if solution is not None
    )
    maximum_condition = max((item.condition_number for item in diagnostics), default=0.0)
    return {
        "candidate_id": candidate_id,
        "component": spec.force_component,
        "model_type": spec.model_type,
        "macro_total_rmse": float(per_log["rmse"].mean()),
        "median_log_total_rmse": float(per_log["rmse"].median()),
        "worst_log_total_rmse": float(per_log["rmse"].max()),
        "macro_mean_rmse": float(mean_per_log["mean_rmse"].mean()),
        "macro_waveform_rmse": float(np.mean(fold_waveform)),
        "coefficient_count": coefficient_count,
        "correction_rms": float(np.sqrt(np.mean(correction**2))),
        "correction_peak": float(np.max(np.abs(correction))),
        "correction_prior_rms_ratio": float(
            np.sqrt(np.mean(correction**2)) / max(np.sqrt(np.mean(prior**2)), np.finfo(float).eps)
        ),
        "maximum_condition_number": maximum_condition,
        "rank_deficient": any(item.rank_deficient for item in diagnostics),
        "per_log_errors_json": json.dumps(
            dict(zip(per_log["log_id"].astype(str), per_log["rmse"].astype(float))), sort_keys=True
        ),
        "spec_json": json.dumps(spec.to_dict(), sort_keys=True),
        "complexity_json": json.dumps(
            spec_complexity(spec, coefficient_count, maximum_condition), allow_nan=False
        ),
    }


def _baseline_specs(component: str) -> tuple[StaticCorrectionSpec, StaticCorrectionSpec]:
    return (
        StaticCorrectionSpec(
            model_type="raw_prior",
            force_component=component,
            fit_intercept=False,
        ),
        StaticCorrectionSpec(
            model_type="gain_bias",
            force_component=component,
            fit_intercept=True,
            waveform_weighting="equal_log",
        ),
    )


def _complete_shortlist(frame: pd.DataFrame, component: str, limit: int) -> tuple[pd.DataFrame, str]:
    selectable = frame[frame["model_type"] != "raw_prior"].copy()
    best = selectable.sort_values(["macro_total_rmse", "candidate_id"], kind="stable").iloc[0]
    threshold, _ = one_se_threshold(list(json.loads(best["per_log_errors_json"]).values()))
    eligible = selectable[selectable["macro_total_rmse"] <= threshold + 1e-12]
    simple = min(
        (row for _, row in eligible.iterrows()),
        key=lambda row: (tuple(json.loads(row["complexity_json"])), str(row["candidate_id"])),
    )
    reasons: dict[str, set[str]] = {}

    def add(row: pd.Series, reason: str) -> None:
        reasons.setdefault(str(row["candidate_id"]), set()).add(reason)

    add(best, "train_cv_total_best")
    add(simple, "train_cv_one_se_simplest")
    matched_priority: list[str] = []
    matched_group_id = ""
    no_prior_rows = selectable[selectable["model_type"] == "no_prior_mean_wb"].sort_values(
        ["macro_total_rmse", "candidate_id"], kind="stable"
    )
    for _, no_prior in no_prior_rows.iterrows():
        no_spec = StaticCorrectionSpec.from_dict(json.loads(no_prior["spec_json"]))
        capacity = (
            no_spec.harmonic_order,
            no_spec.mean_condition_set,
            no_spec.waveform_condition_set,
            no_spec.ridge_lambda_mean,
            no_spec.ridge_lambda_waveform,
            no_spec.mean_weighting,
            no_spec.waveform_weighting,
        )
        matches = {"no_prior_mean_wb": no_prior}
        for family in ("fixed_prior_mean_wb", "shaped_prior_mean_wb"):
            for _, row in selectable[selectable["model_type"] == family].iterrows():
                spec = StaticCorrectionSpec.from_dict(json.loads(row["spec_json"]))
                if (
                    spec.harmonic_order,
                    spec.mean_condition_set,
                    spec.waveform_condition_set,
                    spec.ridge_lambda_mean,
                    spec.ridge_lambda_waveform,
                    spec.mean_weighting,
                    spec.waveform_weighting,
                ) == capacity:
                    matches[family] = row
                    break
        if len(matches) == 3:
            matched_group_id = canonical_hash({"component": component, "capacity": capacity})[:12]
            for family in ("no_prior_mean_wb", "fixed_prior_mean_wb", "shaped_prior_mean_wb"):
                row = matches[family]
                add(row, f"matched_capacity_{matched_group_id}")
                matched_priority.append(str(row["candidate_id"]))
            break
    for family in ("no_prior_mean_wb", "fixed_prior_mean_wb", "shaped_prior_mean_wb", "gain_bias"):
        subset = selectable[selectable["model_type"] == family]
        if len(subset):
            add(
                subset.sort_values(["macro_total_rmse", "candidate_id"], kind="stable").iloc[0],
                f"best_{family}",
            )
    gain_ids = [
        candidate_id
        for candidate_id, candidate_reasons in reasons.items()
        if "best_gain_bias" in candidate_reasons
    ]
    ordered = list(
        dict.fromkeys(
            [str(best["candidate_id"]), str(simple["candidate_id"])]
            + matched_priority
            + gain_ids
            + sorted(reasons)
        )
    )
    ordered = ordered[:limit]
    result = selectable[selectable["candidate_id"].isin(ordered)].copy()
    result["shortlist_reason"] = result["candidate_id"].map(
        {candidate_id: ";".join(sorted(reasons[candidate_id])) for candidate_id in ordered}
    )
    raw = frame[frame["model_type"] == "raw_prior"].iloc[[0]].copy()
    raw["shortlist_reason"] = "fixed_nonselectable_baseline"
    result = pd.concat([result, raw], ignore_index=True).sort_values("macro_total_rmse", kind="stable")
    return result.reset_index(drop=True), matched_group_id


def run_train_cv_selection(
    data: StaticCorrectionTrainingData,
    config: StaticSelectionConfig,
    output_root: str | Path,
    *,
    git_commit: str,
    git_dirty: bool,
    config_path: str | Path,
    run_command: str,
    workers: int = 1,
) -> dict[str, object]:
    """Run Stage A without loading validation and seal the finalist specs."""

    if git_dirty:
        raise ValueError("Formal C3 Stage A requires a clean implementation commit")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite C3 Stage A output: {output}")
    output.mkdir(parents=True)
    config_file = Path(config_path)
    config_before = sha256_file(config_file)
    artifact_manifest_path = Path(str(data.provenance["correction_ready_artifact_path"])) / "manifest.json"
    artifact_before = sha256_file(artifact_manifest_path)
    folds = build_date_aware_grouped_folds(
        data.cycle_frame,
        fold_count=int(config.train_cv["folds"]),
        random_seed=int(config.train_cv["random_seed"]),
    )
    fold_frames = list(_fold_frames(data.cycle_frame, data.waveform_frame, folds))

    if workers < 1:
        raise ValueError("workers must be positive")
    mean_specs = []
    waveform_specs = []
    for component in config.force_components:
        for retention in config.mean_search["prior_retention"]:
            for condition in config.mean_search["condition_sets"]:
                for ridge in config.mean_search["ridge_values"]:
                    spec = _branch_spec(
                        component,
                        mean_retention=float(retention),
                        mean_condition=str(condition),
                        ridge_mean=float(ridge),
                    )
                    mean_specs.append(spec)
        for retention in config.waveform_search["prior_retention"]:
            for harmonic in config.waveform_search["harmonic_orders"]:
                for condition in config.waveform_search["condition_sets"]:
                    for ridge in config.waveform_search["ridge_values"]:
                        spec = _branch_spec(
                            component,
                            waveform_retention=float(retention),
                            waveform_condition=str(condition),
                            harmonic_order=int(harmonic),
                            ridge_waveform=float(ridge),
                        )
                        waveform_specs.append(spec)
    global _WORKER_FOLDS, _WORKER_DATA
    _WORKER_FOLDS = fold_frames
    _WORKER_DATA = data
    mean_rows = _parallel_map(_mean_worker, mean_specs, workers)
    waveform_rows = _parallel_map(_waveform_worker, waveform_specs, workers)
    mean_first = pd.DataFrame(mean_rows)
    waveform_first = pd.DataFrame(waveform_rows)
    mean_refined = _refine_mean(mean_first, config, fold_frames)
    waveform_refined = _refine_waveform(waveform_first, config, fold_frames)
    mean_results = pd.concat([mean_first.assign(search_pass="initial"), mean_refined.assign(search_pass="weighting_refinement")])
    waveform_results = pd.concat(
        [waveform_first.assign(search_pass="initial"), waveform_refined.assign(search_pass="weighting_refinement")]
    )
    mean_shortlist = pd.concat(
        [
            _branch_shortlist(
                mean_refined[mean_refined["component"] == component],
                "mean",
                config.shortlist["mean_branch_limit"],
            )
            for component in config.force_components
        ],
        ignore_index=True,
    )
    waveform_shortlist = pd.concat(
        [
            _branch_shortlist(
                waveform_refined[waveform_refined["component"] == component],
                "waveform",
                config.shortlist["waveform_branch_limit"],
            )
            for component in config.force_components
        ],
        ignore_index=True,
    )

    complete_specs: dict[str, StaticCorrectionSpec] = {}
    for component in config.force_components:
        mean_component = mean_shortlist[mean_shortlist["component"] == component]
        waveform_component = waveform_shortlist[waveform_shortlist["component"] == component]
        for _, mean_row in mean_component.iterrows():
            for _, wave_row in waveform_component.iterrows():
                spec = make_mean_wb_spec(
                    component,
                    mean_retention=float(mean_row["mean_prior_retention"]),
                    waveform_retention=float(wave_row["waveform_prior_retention"]),
                    mean_condition=str(mean_row["mean_condition_set"]),
                    waveform_condition=str(wave_row["waveform_condition_set"]),
                    harmonic_order=int(wave_row["harmonic_order"]),
                    ridge_mean=float(mean_row["ridge_lambda_mean"]),
                    ridge_waveform=float(wave_row["ridge_lambda_waveform"]),
                    mean_weighting=str(mean_row["mean_weighting"]),
                    waveform_weighting=str(wave_row["waveform_weighting"]),
                )
                complete_specs[canonical_hash(spec.to_dict())] = spec
        for baseline in _baseline_specs(component):
            complete_specs[canonical_hash(baseline.to_dict())] = baseline

        component_no_prior = [
            spec for spec in complete_specs.values()
            if spec.force_component == component and spec.model_type == "no_prior_mean_wb"
        ]
        if component_no_prior:
            anchor = component_no_prior[0]
            for mean_retention, waveform_retention in ((1.0, 1.0), (0.5, 0.5)):
                matched = make_mean_wb_spec(
                    component,
                    mean_retention=mean_retention,
                    waveform_retention=waveform_retention,
                    mean_condition=str(anchor.mean_condition_set),
                    waveform_condition=str(anchor.waveform_condition_set),
                    harmonic_order=int(anchor.harmonic_order),
                    ridge_mean=anchor.ridge_lambda_mean,
                    ridge_waveform=anchor.ridge_lambda_waveform,
                    mean_weighting=anchor.mean_weighting,
                    waveform_weighting=anchor.waveform_weighting,
                )
                complete_specs[canonical_hash(matched.to_dict())] = matched

    complete_results = pd.DataFrame(
        _parallel_map(_complete_worker, list(complete_specs.values()), workers)
    )
    finalists: dict[str, list[dict[str, object]]] = {}
    matched_groups: dict[str, str] = {}
    for component in config.force_components:
        shortlist, matched_group_id = _complete_shortlist(
            complete_results[complete_results["component"] == component],
            component,
            config.shortlist["complete_model_limit_per_component"],
        )
        finalists[component] = [
            {
                "candidate_id": str(row["candidate_id"]),
                "model_spec": json.loads(row["spec_json"]),
                "train_cv_metrics": {
                    "macro_total_rmse": float(row["macro_total_rmse"]),
                    "macro_mean_rmse": float(row["macro_mean_rmse"]),
                    "macro_waveform_rmse": float(row["macro_waveform_rmse"]),
                    "worst_log_total_rmse": float(row["worst_log_total_rmse"]),
                    "per_log_rmse": json.loads(row["per_log_errors_json"]),
                },
                "complexity": json.loads(row["complexity_json"]),
                "shortlist_reason": str(row["shortlist_reason"]),
                "selectable": str(row["model_type"]) != "raw_prior",
            }
            for _, row in shortlist.iterrows()
        ]
        matched_groups[component] = matched_group_id

    frozen_config_path = output / "frozen_selection_config.yaml"
    frozen_config_path.write_text(yaml.safe_dump(dict(config.raw), sort_keys=False), encoding="utf-8")
    write_json(output / "train_cv_folds.json", folds.to_dict())
    mean_results.to_parquet(output / "mean_branch_cv_results.parquet", index=False)
    waveform_results.to_parquet(output / "waveform_branch_cv_results.parquet", index=False)
    complete_results.to_parquet(output / "complete_model_cv_results.parquet", index=False)
    mean_shortlist.to_csv(output / "mean_branch_shortlist.csv", index=False)
    waveform_shortlist.to_csv(output / "waveform_branch_shortlist.csv", index=False)
    shortlist_payload = seal_shortlist(
        {
            "schema_version": "static_correction_train_cv_shortlist_v1",
            "stage": "train_only_grouped_cv",
            "generation_rule": "branch_grid_then_weighting_refinement_then_complete_one_se_family_shortlist",
            "source_config_hash": config.config_hash,
            "source_artifact_hash": str(data.provenance["correction_ready_manifest_hash"]),
            "fold_manifest_hash": folds.assignment_hash,
            "git_commit": git_commit,
            "git_dirty": False,
            "normalization_source": "full_train_partition_from_C1",
            "normalization_validation_participated": False,
            "validation_labels_loaded": False,
            "test_labels_loaded": False,
            "physical_component_scale": data.component_availability["physical_component_scale_fz"],
            "matched_capacity_group_ids": matched_groups,
            "finalists": finalists,
        }
    )
    write_json(output / "train_cv_shortlist.json", shortlist_payload)
    (output / "run_command.txt").write_text(run_command.rstrip() + "\n", encoding="utf-8")
    selection_manifest = {
        "schema_version": "static_correction_selection_manifest_v1",
        "run_stage": "stage_a_train_only",
        "git_commit": git_commit,
        "git_dirty": False,
        "config_hash": config.config_hash,
        "frozen_config_hash": sha256_file(frozen_config_path),
        "correction_ready_artifact_id": data.provenance["correction_ready_artifact_id"],
        "correction_ready_manifest_hash": data.provenance["correction_ready_manifest_hash"],
        "fold_assignment_hash": folds.assignment_hash,
        "shortlist_hash": shortlist_payload["shortlist_hash"],
        "train_logs": sorted(data.cycle_frame["log_id"].astype(str).unique().tolist()),
        "train_cycle_count": int(len(data.cycle_frame)),
        "train_waveform_row_count": int(len(data.waveform_frame)),
        "compute_workers": int(workers),
        "validation_labels_loaded": False,
        "test_labels_loaded": False,
    }
    write_json(output / "stage_a_manifest.json", selection_manifest)
    if sha256_file(config_file) != config_before or sha256_file(artifact_manifest_path) != artifact_before:
        raise ValueError("Stage A input config or correction-ready artifact changed during execution")
    return {
        "output_root": str(output.resolve()),
        "mean_candidate_count": int(len(mean_results)),
        "waveform_candidate_count": int(len(waveform_results)),
        "complete_candidate_count": int(len(complete_results)),
        "shortlist_hash": shortlist_payload["shortlist_hash"],
        "finalist_counts": {key: len(value) for key, value in finalists.items()},
        "validation_labels_loaded": False,
        "test_labels_loaded": False,
    }
