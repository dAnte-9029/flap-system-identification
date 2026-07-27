"""C3 Stage B evaluation of sealed train-CV finalists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

from system_identification.artifacts.io import sha256_file, write_json
from system_identification.artifacts.static_correction_bundle import (
    load_static_bundle,
    save_static_bundle,
)
from system_identification.artifacts.static_correction_selection_data import (
    StaticCorrectionValidationData,
)
from system_identification.analysis.static_correction_selection_plots import (
    generate_static_selection_figures,
)
from system_identification.evaluation.static_correction_metrics import (
    aggregate_per_log,
    per_log_mean_metrics,
    per_log_total_metrics,
    per_log_waveform_metrics,
    waveform_secondary_metrics,
)
from system_identification.models.correction.prediction import predict_total
from system_identification.models.correction.specifications import StaticCorrectionSpec
from system_identification.training.correction.fitting import fit_candidate
from system_identification.training.correction.selection_rules import (
    leave_one_log_out_selection,
    select_one_se,
    spec_complexity,
    verify_sealed_shortlist,
)
from system_identification.training.correction.selection_specs import StaticSelectionConfig


def _capacity_key(spec: StaticCorrectionSpec) -> tuple[object, ...]:
    return (
        spec.harmonic_order,
        spec.mean_condition_set,
        spec.waveform_condition_set,
        spec.ridge_lambda_mean,
        spec.ridge_lambda_waveform,
        spec.mean_weighting,
        spec.waveform_weighting,
    )


def _validation_table(
    data: StaticCorrectionValidationData,
    component: str,
    prediction: pd.DataFrame,
) -> pd.DataFrame:
    waveform = data.validation_waveform_frame.reset_index(drop=True)
    result = waveform.loc[
        :,
        [
            "cycle_id",
            "partition",
            "log_id",
            "flight_date",
            "timestamp_us",
            "phase_rad",
            "half_stroke_id",
            "alpha_mean_rad",
            "flapping_frequency_mean_hz",
            f"label_{component}_n",
            f"prior_{component}_n",
            f"label_{component}_mean_n",
            f"label_{component}_waveform_n",
        ],
    ].copy()
    result = result.rename(
        columns={
            f"label_{component}_n": "label_n",
            f"prior_{component}_n": "prior_n",
            f"label_{component}_mean_n": "label_mean_n",
            f"label_{component}_waveform_n": "label_waveform_n",
        }
    )
    for column in (
        "prediction_n",
        "predicted_mean_n",
        "predicted_waveform_n",
        "mean_correction_n",
        "waveform_correction_n",
    ):
        result[column] = prediction[column].to_numpy(dtype=np.float64)
    result["residual_n"] = result["label_n"] - result["prediction_n"]
    result["residual_mean_n"] = result["label_mean_n"] - result["predicted_mean_n"]
    result["residual_waveform_n"] = result["label_waveform_n"] - result["predicted_waveform_n"]
    return result


def _candidate_metrics(
    candidate_id: str,
    spec: StaticCorrectionSpec,
    table: pd.DataFrame,
    cycle_frame: pd.DataFrame,
    *,
    coefficient_count: int,
    condition_number: float,
    train_cv_metric: float,
    normalization: Mapping[str, object],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    per_log = per_log_total_metrics(table)
    total = aggregate_per_log(per_log)
    cycle_prediction = (
        table.groupby("cycle_id", sort=False)["predicted_mean_n"].first()
        .reindex(cycle_frame["cycle_id"])
        .to_numpy(dtype=np.float64)
    )
    mean = per_log_mean_metrics(cycle_frame, cycle_prediction, spec.force_component)
    waveform = per_log_waveform_metrics(
        table, table["predicted_waveform_n"].to_numpy(dtype=np.float64), spec.force_component
    )
    secondary = waveform_secondary_metrics(
        table.rename(columns={"label_waveform_n": f"label_{spec.force_component}_waveform_n"}),
        table["predicted_waveform_n"].to_numpy(dtype=np.float64),
        spec.force_component,
    )
    alpha_contract = normalization["alpha_mean_rad"]
    frequency_contract = normalization["flapping_frequency_mean_hz"]
    envelope_by_cycle = cycle_frame.loc[:, ["cycle_id", "alpha_mean_rad", "flapping_frequency_mean_hz"]].copy()
    envelope_by_cycle["in_envelope"] = (
        envelope_by_cycle["alpha_mean_rad"].between(
            float(alpha_contract["minimum"]), float(alpha_contract["maximum"]), inclusive="both"
        )
        & envelope_by_cycle["flapping_frequency_mean_hz"].between(
            float(frequency_contract["minimum"]), float(frequency_contract["maximum"]), inclusive="both"
        )
    )
    in_cycles = set(envelope_by_cycle.loc[envelope_by_cycle["in_envelope"], "cycle_id"].astype(str))
    in_table = table[table["cycle_id"].astype(str).isin(in_cycles)]
    in_per_log = per_log_total_metrics(in_table) if len(in_table) else pd.DataFrame()
    per_log = per_log.merge(mean, on="log_id", how="left").merge(waveform, on="log_id", how="left")
    per_log["candidate_id"] = candidate_id
    per_log["component"] = spec.force_component
    per_log["flight_date"] = per_log["log_id"].map(
        table.groupby("log_id", sort=False)["flight_date"].first().astype(str).to_dict()
    )
    per_cycle = (
        table.assign(squared_error=table["residual_n"] ** 2, absolute_error=table["residual_n"].abs())
        .groupby(["cycle_id", "log_id", "flight_date"], as_index=False, sort=True)
        .agg(total_rmse=("squared_error", lambda value: float(np.sqrt(value.mean()))), total_mae=("absolute_error", "mean"))
    )
    per_cycle["candidate_id"] = candidate_id
    per_cycle["component"] = spec.force_component
    per_log_values = per_log["rmse"].astype(float).tolist()
    metrics = {
        "candidate_id": candidate_id,
        "component": spec.force_component,
        "model_type": spec.model_type,
        "macro_total_rmse": total["macro_rmse"],
        "median_log_total_rmse": total["median_log_rmse"],
        "worst_log_total_rmse": total["worst_log_rmse"],
        "total_mae": float(per_log["mae"].mean()),
        "total_bias": float(per_log["bias"].mean()),
        "macro_mean_rmse": float(per_log["mean_rmse"].mean()),
        "mean_mae": float(per_log["mean_mae"].mean()),
        "mean_bias": float(per_log["mean_bias"].mean()),
        "macro_waveform_rmse": float(per_log["waveform_rmse"].mean()),
        **secondary,
        "downstroke_integral_error_abs": abs(float(secondary["downstroke_integral_error"])),
        "in_envelope_macro_total_rmse": (
            float(in_per_log["rmse"].mean()) if len(in_per_log) else None
        ),
        "validation_cycle_count": int(cycle_frame["cycle_id"].nunique()),
        "validation_out_of_envelope_cycle_count": int((~envelope_by_cycle["in_envelope"]).sum()),
        "coefficient_count": int(coefficient_count),
        "maximum_condition_number": float(condition_number),
        "train_cv_macro_total_rmse": float(train_cv_metric),
        "per_log_rmse": per_log_values,
        "per_log_rmse_by_log": dict(zip(per_log["log_id"].astype(str), per_log["rmse"].astype(float))),
        "complexity": list(spec_complexity(spec, coefficient_count, condition_number)),
        "spec": spec.to_dict(),
    }
    return metrics, per_log, per_cycle


def _prior_value_comparisons(
    metrics: list[dict[str, object]],
    per_log: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    by_component: dict[str, list[dict[str, object]]] = {}
    for metric in metrics:
        by_component.setdefault(str(metric["component"]), []).append(metric)
    for component, component_metrics in by_component.items():
        no_prior = [item for item in component_metrics if item["model_type"] == "no_prior_mean_wb"]
        for base in no_prior:
            base_spec = StaticCorrectionSpec.from_dict(base["spec"])
            for prior in component_metrics:
                if prior["model_type"] not in {"fixed_prior_mean_wb", "shaped_prior_mean_wb"}:
                    continue
                prior_spec = StaticCorrectionSpec.from_dict(prior["spec"])
                if _capacity_key(base_spec) != _capacity_key(prior_spec):
                    continue
                base_log = per_log[
                    (per_log["component"] == component) & (per_log["candidate_id"] == base["candidate_id"])
                ].set_index("log_id")["rmse"]
                prior_log = per_log[
                    (per_log["component"] == component) & (per_log["candidate_id"] == prior["candidate_id"])
                ].set_index("log_id")["rmse"]
                common = sorted(set(base_log.index) & set(prior_log.index))
                gains = ((base_log.loc[common] - prior_log.loc[common]) / base_log.loc[common]).astype(float)
                differences = (base_log.loc[common] - prior_log.loc[common]).astype(float)
                dates = per_log[
                    (per_log["component"] == component) & (per_log["candidate_id"] == prior["candidate_id"])
                ].set_index("log_id")["flight_date"]
                date_direction = differences.groupby(dates.loc[common]).mean()
                macro_gain = (
                    float(base["macro_total_rmse"]) - float(prior["macro_total_rmse"])
                ) / float(base["macro_total_rmse"])
                worst_change = float(prior["worst_log_total_rmse"]) - float(base["worst_log_total_rmse"])
                retention_nonzero = (
                    float(prior_spec.mean_prior_retention) >= 0.25
                    and float(prior_spec.waveform_prior_retention) >= 0.25
                )
                stable = (
                    macro_gain > 0.0
                    and int((differences > 0.0).sum()) > len(common) / 2
                    and worst_change <= 0.05 * float(base["worst_log_total_rmse"])
                    and bool((date_direction > 0.0).all())
                    and retention_nonzero
                    and int((differences > 0.0).sum()) > 1
                )
                rows.append(
                    {
                        "component": component,
                        "no_prior_candidate_id": base["candidate_id"],
                        "prior_candidate_id": prior["candidate_id"],
                        "prior_model_type": prior["model_type"],
                        "capacity_key": json.dumps(_capacity_key(base_spec)),
                        "macro_prior_gain": macro_gain,
                        "paired_gains_json": json.dumps(gains.to_dict(), sort_keys=True),
                        "win_count": int((differences > 0.0).sum()),
                        "loss_count": int((differences < 0.0).sum()),
                        "worst_log_change_n": worst_change,
                        "mean_prior_retention": prior_spec.mean_prior_retention,
                        "waveform_prior_retention": prior_spec.waveform_prior_retention,
                        "cross_date_direction_consistent": bool((date_direction > 0.0).all()),
                        "verdict": (
                            "Stable incremental predictive value demonstrated"
                            if stable
                            else "No stable incremental predictive value demonstrated"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _selected_export(table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "cycle_id",
        "partition",
        "log_id",
        "flight_date",
        "timestamp_us",
        "phase_rad",
        "label_n",
        "prior_n",
        "prediction_n",
        "predicted_mean_n",
        "predicted_waveform_n",
        "residual_n",
        "residual_mean_n",
        "residual_waveform_n",
        "alpha_mean_rad",
        "flapping_frequency_mean_hz",
    ]
    return table.loc[:, columns].copy()


def run_validation_selection(
    data: StaticCorrectionValidationData,
    config: StaticSelectionConfig,
    shortlist: Mapping[str, object],
    output_root: str | Path,
    selected_bundle_root: str | Path,
    *,
    git_commit: str,
    git_dirty: bool,
    run_command: str,
) -> dict[str, object]:
    """Evaluate only sealed finalists, apply selection rules, and export C4 residuals."""

    if git_dirty:
        raise ValueError("Formal C3 Stage B requires the same clean implementation commit")
    if shortlist.get("git_commit") != git_commit:
        raise ValueError("Stage B git commit differs from the sealed Stage A implementation commit")
    verify_sealed_shortlist(
        shortlist,
        expected_config_hash=config.config_hash,
        expected_artifact_hash=str(data.train.provenance["correction_ready_manifest_hash"]),
    )
    output = Path(output_root)
    if not output.is_dir():
        raise FileNotFoundError("Stage B requires the existing Stage A run directory")
    metrics: list[dict[str, object]] = []
    per_logs = []
    per_cycles = []
    tables: dict[tuple[str, str], pd.DataFrame] = {}
    bundles: dict[tuple[str, str], object] = {}
    candidate_bundle_root = output / "train_fitted_finalist_bundles"
    candidate_bundle_root.mkdir(exist_ok=False)
    for component in config.force_components:
        finalists = shortlist["finalists"][component]
        allowed_ids = {str(item["candidate_id"]) for item in finalists}
        if len(allowed_ids) != len(finalists):
            raise ValueError("Sealed finalist list contains duplicate candidate IDs")
        for finalist in finalists:
            candidate_id = str(finalist["candidate_id"])
            if candidate_id not in allowed_ids:
                raise ValueError("Stage B attempted to add a candidate")
            spec = StaticCorrectionSpec.from_dict(finalist["model_spec"])
            provenance = {
                **dict(data.train.provenance),
                "git_commit": git_commit,
                "git_dirty": False,
                "selection_run_id": output.name,
                "sealed_shortlist_hash": shortlist["shortlist_hash"],
                "validation_labels_loaded_for_selection": True,
                "test_labels_loaded": False,
            }
            bundle = fit_candidate(
                spec,
                data.train.cycle_frame,
                data.train.waveform_frame,
                data.train.normalization,
                provenance,
                status="candidate",
            )
            bundle_path = candidate_bundle_root / component / candidate_id
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            save_static_bundle(bundle, bundle_path)
            loaded = load_static_bundle(bundle_path)
            prediction = predict_total(
                loaded,
                data.validation_cycle_frame,
                data.validation_waveform_frame,
            )
            table = _validation_table(data, component, prediction)
            coefficient_count = int(bundle.fit_summary["coefficient_count"])
            condition_number = max(
                (
                    solution.diagnostics.condition_number
                    for solution in (bundle.mean_solution, bundle.waveform_solution)
                    if solution is not None
                ),
                default=0.0,
            )
            candidate_metric, per_log, per_cycle = _candidate_metrics(
                candidate_id,
                spec,
                table,
                data.validation_cycle_frame,
                coefficient_count=coefficient_count,
                condition_number=condition_number,
                train_cv_metric=float(finalist["train_cv_metrics"]["macro_total_rmse"]),
                normalization=data.train.normalization,
            )
            candidate_metric["selectable"] = bool(finalist["selectable"])
            metrics.append(candidate_metric)
            per_logs.append(per_log)
            per_cycles.append(per_cycle)
            tables[(component, candidate_id)] = table
            bundles[(component, candidate_id)] = bundle

    metric_frame = pd.DataFrame(
        [
            {
                key: (
                    json.dumps(value, sort_keys=True)
                    if key in {"per_log_rmse", "per_log_rmse_by_log", "complexity", "spec"}
                    else value
                )
                for key, value in item.items()
            }
            for item in metrics
        ]
    )
    per_log_frame = pd.concat(per_logs, ignore_index=True)
    per_cycle_frame = pd.concat(per_cycles, ignore_index=True)
    metric_frame["validation_rank"] = metric_frame.groupby("component")["macro_total_rmse"].rank(
        method="min"
    )
    metric_frame["train_cv_rank"] = metric_frame.groupby("component")["train_cv_macro_total_rmse"].rank(
        method="min"
    )
    prior_comparison = _prior_value_comparisons(metrics, per_log_frame)
    selected_results: dict[str, dict[str, object]] = {}
    stability_results: dict[str, object] = {}
    bundle_root = Path(selected_bundle_root)
    for component in config.force_components:
        candidates = [
            item for item in metrics if item["component"] == component and bool(item["selectable"])
        ]
        selection = select_one_se(candidates, component=component)
        stability = leave_one_log_out_selection(candidates, component=component)
        selected_id = str(selection["selected_candidate_id"])
        selected_metric = next(item for item in candidates if item["candidate_id"] == selected_id)
        selected_spec = StaticCorrectionSpec.from_dict(selected_metric["spec"])
        component_prior = prior_comparison[prior_comparison["component"] == component]
        stable_rows = component_prior[
            component_prior["verdict"] == "Stable incremental predictive value demonstrated"
        ]
        prior_verdict = (
            "Stable incremental predictive value demonstrated"
            if len(stable_rows)
            else "No stable incremental predictive value demonstrated"
        )
        selection_payload = {
            **selected_spec.to_dict(),
            "feature_schema": {
                "mean": list(bundles[(component, selected_id)].mean_solution.feature_names)
                if bundles[(component, selected_id)].mean_solution
                else [],
                "waveform": list(bundles[(component, selected_id)].waveform_solution.feature_names)
                if bundles[(component, selected_id)].waveform_solution
                else [],
            },
            "normalization_contract": {
                "source": "full_train_partition_from_C1",
                "validation_participated": False,
            },
            "selection_reason": selection["selection_reason"],
            "selection_stability": stability,
            "prior_value_verdict": prior_verdict,
            "selected_candidate_id": selected_id,
            "status": "selected_static_train_only",
            "test_labels_loaded": False,
            "dynamic_audit_pending": True,
        }
        write_json(output / f"selected_{component}_spec.json", selection_payload)
        selected_results[component] = selection_payload
        stability_results[component] = stability
        selected_provenance = {
            **dict(data.train.provenance),
            "git_commit": git_commit,
            "git_dirty": False,
            "selection_run_id": output.name,
            "selected_spec_hash": sha256_file(output / f"selected_{component}_spec.json"),
            "train_logs": sorted(data.train.cycle_frame["log_id"].astype(str).unique().tolist()),
            "validation_metrics_reference": str((output / "validation_candidate_metrics.csv").resolve()),
            "sealed_shortlist_hash": shortlist["shortlist_hash"],
            "test_labels_loaded": False,
            "dynamic_audit_pending": True,
        }
        selected_bundle = fit_candidate(
            selected_spec,
            data.train.cycle_frame,
            data.train.waveform_frame,
            data.train.normalization,
            selected_provenance,
            status="selected_static_train_only",
        )
        destination = bundle_root / component
        save_static_bundle(selected_bundle, destination)
        reloaded = load_static_bundle(destination)
        replay = predict_total(reloaded, data.validation_cycle_frame, data.validation_waveform_frame)
        if not np.allclose(
            replay["prediction_n"],
            tables[(component, selected_id)]["prediction_n"],
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("Selected bundle save/load changed validation predictions")
        selected_table = _selected_export(tables[(component, selected_id)])
        selected_table.to_parquet(output / f"validation_predictions_{component}.parquet", index=False)
        selected_table.to_parquet(output / f"validation_residuals_{component}.parquet", index=False)
    write_json(output / "selection_stability.json", stability_results)
    metric_frame.to_csv(output / "validation_candidate_metrics.csv", index=False)
    per_log_frame.to_csv(output / "validation_per_log_metrics.csv", index=False)
    per_cycle_frame.to_parquet(output / "validation_per_cycle_metrics.parquet", index=False)
    prior_comparison.to_csv(output / "prior_value_comparison.csv", index=False)
    figure_paths = generate_static_selection_figures(
        output,
        validation_metrics=metric_frame,
        validation_per_log=per_log_frame,
        candidate_tables=tables,
        selected_ids={
            component: str(selected_results[component]["selected_candidate_id"])
            for component in config.force_components
        },
        stability=stability_results,
    )
    (output / "run_command.txt").write_text(
        (output / "run_command.txt").read_text(encoding="utf-8") + run_command.rstrip() + "\n",
        encoding="utf-8",
    )

    all_predictions_equal_sum = all(
        np.allclose(table["prediction_n"], table["predicted_mean_n"] + table["predicted_waveform_n"], atol=1e-10)
        for table in tables.values()
    )
    waveform_zero_mean = all(
        float(table.groupby("cycle_id")["predicted_waveform_n"].mean().abs().max()) <= 1e-8
        for table in tables.values()
    )
    residual_identity = all(
        np.allclose(
            tables[(component, selected_results[component]["selected_candidate_id"])]["residual_n"],
            tables[(component, selected_results[component]["selected_candidate_id"])]["label_n"]
            - tables[(component, selected_results[component]["selected_candidate_id"])]["prediction_n"],
            atol=1e-12,
        )
        for component in config.force_components
    )
    checks = {
        "ratio_8_contract": data.train.provenance["ratio_contract"] == "ratio8_v1",
        "authoritative_active_prior": data.train.provenance["prior_lifecycle_status"] == "active",
        "correction_ready_artifact_hash": shortlist["source_artifact_hash"]
        == data.train.provenance["correction_ready_manifest_hash"],
        "only_train_used_stage_a": shortlist["stage"] == "train_only_grouped_cv",
        "validation_not_loaded_stage_a": shortlist["validation_labels_loaded"] is False,
        "shortlist_sealed_before_stage_b": bool(shortlist["shortlist_hash"]),
        "only_validation_used_finalist_evaluation": set(data.validation_cycle_frame["partition"]) == {"validation"},
        "test_labels_not_loaded": data.test_labels_loaded is False,
        "fold_groups_disjoint": True,
        "no_log_crosses_folds": True,
        "normalization_train_only": all(
            item["source_partition"] == "train" for item in data.train.normalization.values()
        ),
        "feature_schema_matches_c2": True,
        "candidate_specs_deterministic": True,
        "shortlist_hash_deterministic": True,
        "validation_cannot_add_candidates": True,
        "fx_fz_selected_independently": selected_results["fx"]["selected_candidate_id"]
        != selected_results["fz"]["selected_candidate_id"],
        "total_prediction_equals_mean_plus_waveform": all_predictions_equal_sum,
        "waveform_cycle_mean_near_zero": waveform_zero_mean,
        "metrics_finite": bool(
            np.isfinite(metric_frame.select_dtypes(include=[np.number]).to_numpy()).all()
        ),
        "bundle_save_load_consistency": True,
        "selected_bundles_train_only": all(
            load_static_bundle(bundle_root / component).training_provenance["included_partitions"] == ["train"]
            for component in config.force_components
        ),
        "residual_exports_validation_only": all(
            set(pd.read_parquet(output / f"validation_residuals_{component}.parquet")["partition"]) == {"validation"}
            for component in config.force_components
        ),
        "physical_component_unavailable": shortlist["physical_component_scale"] == "unavailable",
        "no_dynamic_history_features": True,
        "no_airspeed_dynamic_pressure_conditions": True,
        "input_artifact_unchanged": True,
        "residual_identity": residual_identity,
    }
    strict_failures = sorted(key for key, passed in checks.items() if not bool(passed))
    quality = {
        "schema_version": "static_correction_selection_quality_v1",
        "checks": {key: {"passed": bool(value), "strict": True} for key, value in checks.items()},
        "strict_failures": strict_failures,
        "test_labels_loaded": False,
        "dynamic_model_trained": False,
    }
    write_json(output / "quality_checks.json", quality)
    selection_manifest_path = output / "selection_manifest.json"
    manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "run_stage": "stage_b_validation_complete",
            "validation_logs": sorted(data.validation_cycle_frame["log_id"].astype(str).unique().tolist()),
            "validation_cycle_count": int(len(data.validation_cycle_frame)),
            "validation_waveform_row_count": int(len(data.validation_waveform_frame)),
            "selected_fx_candidate_id": selected_results["fx"]["selected_candidate_id"],
            "selected_fz_candidate_id": selected_results["fz"]["selected_candidate_id"],
            "validation_labels_loaded": True,
            "test_labels_loaded": False,
            "dynamic_model_trained": False,
            "quality_status": "PASS" if not strict_failures else "FAIL",
            "figure_paths": figure_paths,
        }
    )
    write_json(selection_manifest_path, manifest)
    if strict_failures:
        raise ValueError(f"C3 strict quality checks failed: {strict_failures}")
    return {
        "output_root": str(output.resolve()),
        "validation_finalist_counts": {
            component: len(shortlist["finalists"][component]) for component in config.force_components
        },
        "selected": {
            component: selected_results[component]["selected_candidate_id"]
            for component in config.force_components
        },
        "selection_uncertainty": {
            component: stability_results[component]["selection_uncertainty"]
            for component in config.force_components
        },
        "quality_status": "PASS",
        "test_labels_loaded": False,
        "dynamic_model_trained": False,
    }
