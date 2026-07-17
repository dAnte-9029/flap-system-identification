"""Package orchestration for the EDA0 force-discrepancy attribution audit."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from system_identification.analysis.component_attribution import (
    evaluate_component_diagnostics,
    physical_sensitivity_audit,
    summarize_component_attribution,
)
from system_identification.analysis.force_discrepancy_attribution import (
    AlignmentResult,
    AuditConfig,
    SCHEMA_VERSION,
    audit_summary_metrics,
    condition_dependence_table,
    decision_summary,
    decompose_cycle_residuals,
    derive_condition_columns,
    diagnostic_history_probes,
    diagnostic_static_probes,
    half_stroke_attribution,
    harmonic_cycle_summary,
    keyed_align_label_and_prior,
    label_robustness,
    matched_capacity_prior_probe,
    normalize_partitions,
    phase_alignment_cycles,
    phase_binned_waveform,
    select_complete_cycles,
    summarize_phase_alignment,
    trim_impact_estimate,
    waveform_repeatability,
)


@dataclass
class AuditRunResult:
    """All authoritative tables and structured summaries from one audit."""

    tables: dict[str, pd.DataFrame]
    alignment_report: dict[str, object]
    phase_alignment_summary: dict[str, object]
    decision: dict[str, object]
    summary: dict[str, object]
    condition_sources: dict[str, str]
    availability: dict[str, str]
    strict_failures: list[str]


def _partition_path(root: Path, partition: str, *, prior: bool) -> Path:
    filename = f"{partition}_predictions.parquet" if prior else f"{partition}_samples.parquet"
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_label_variants(
    roots: Sequence[str | Path],
    partitions: Sequence[str],
) -> Mapping[str, pd.DataFrame]:
    variants: dict[str, pd.DataFrame] = {}
    for root_value in roots:
        root = Path(root_value)
        pieces: list[pd.DataFrame] = []
        for partition in partitions:
            path = _partition_path(root, partition, prior=False)
            frame = pd.read_parquet(
                path,
                columns=["log_id", "timestamp_us", "fx_b", "fz_b", "split"],
            ).rename(columns={"split": "partition"})
            frame["partition"] = partition
            pieces.append(frame)
        variants[root.name] = pd.concat(pieces, ignore_index=True)
    return variants


def run_force_discrepancy_audit(
    *,
    dataset_root: str | Path,
    prior_root: str | Path,
    partitions: Sequence[str],
    config: AuditConfig,
    label_variant_roots: Sequence[str | Path] = (),
    wt1_sensitivity_artifact: str | Path | None = None,
    component_metadata_path: str | Path | None = None,
    component_geometry_path: str | Path | None = None,
    component_theta_tip_deg: float = 10.0,
    component_airflow_mode: str = "legacy_scalar_true_airspeed",
    component_primary_prior_parameters: Mapping[str, object] | None = None,
    sensitivity_phase_step_rad: float = 0.03490658503988659,
    sensitivity_incidence_step_rad: float = 0.017453292519943295,
    sensitivity_twist_step_deg: float = 1.0,
    run_component_diagnostics: bool = True,
    run_physical_sensitivity: bool = True,
) -> AuditRunResult:
    """Run EDA0 on train/validation only and return tables without writing files."""

    config.validate()
    resolved_partitions = normalize_partitions(partitions)
    dataset = Path(dataset_root)
    prior = Path(prior_root)
    aligned_pieces: list[pd.DataFrame] = []
    mismatch_pieces: list[pd.DataFrame] = []
    alignment_by_partition: dict[str, object] = {}
    condition_sources: dict[str, str] = {}
    for partition in resolved_partitions:
        sample_path = _partition_path(dataset, partition, prior=False)
        prior_path = _partition_path(prior, partition, prior=True)
        samples = pd.read_parquet(sample_path)
        predictions = pd.read_parquet(prior_path)
        alignment: AlignmentResult = keyed_align_label_and_prior(
            samples,
            predictions,
            partition=partition,
            maximum_missing_fraction=config.maximum_missing_alignment_fraction,
        )
        conditioned, sources = derive_condition_columns(alignment.aligned)
        condition_sources.update(sources)
        aligned_pieces.append(conditioned)
        if not alignment.mismatches.empty:
            mismatches = alignment.mismatches.copy()
            mismatches["partition"] = partition
            mismatch_pieces.append(mismatches)
        alignment_by_partition[partition] = alignment.report
    aligned = pd.concat(aligned_pieces, ignore_index=True)
    mismatches = pd.concat(mismatch_pieces, ignore_index=True) if mismatch_pieces else pd.DataFrame(
        columns=["log_id", "timestamp_us", "mismatch_type", "partition"]
    )
    cycle_selection = select_complete_cycles(aligned, config)
    accepted = cycle_selection.accepted_rows
    strict_failures: list[str] = []
    if accepted.empty:
        raise ValueError("No complete cycles passed EDA0 quality thresholds")

    phase_cycles = phase_alignment_cycles(accepted, config)
    phase_by_log, phase_summary = summarize_phase_alignment(phase_cycles)
    cycle_means, decomposed = decompose_cycle_residuals(accepted)
    waveform = phase_binned_waveform(decomposed, config)
    half_strokes, half_by_log = half_stroke_attribution(decomposed, config)
    harmonics, harmonic_by_log = harmonic_cycle_summary(decomposed, config)
    repeatability, log_correlations, date_summary = waveform_repeatability(decomposed, waveform)
    condition_table = condition_dependence_table(
        cycle_means,
        half_strokes,
        harmonics,
        phase_cycles,
        bins=config.condition_bins,
    )
    static_probes = diagnostic_static_probes(decomposed, config)
    history_probes = diagnostic_history_probes(decomposed, config)
    prior_probe = matched_capacity_prior_probe(decomposed, config)
    variants = _read_label_variants(label_variant_roots, resolved_partitions)
    label_variant_table, label_uncertainty = label_robustness(aligned, variants, config)
    trim = trim_impact_estimate(cycle_means, wt1_sensitivity_artifact)

    availability: dict[str, str] = {
        "tail_subtracted_residual": "not_available_no_authoritative_keyed_tail_artifact",
        "label_robustness": "available" if variants else "not_available_no_label_variants",
        "trim_impact": str(trim.iloc[0]["status"]),
    }
    component_phase = pd.DataFrame()
    component_half = pd.DataFrame()
    component_span = pd.DataFrame()
    component_similarity = pd.DataFrame()
    component_contract = pd.DataFrame()
    physical_sensitivity = pd.DataFrame()
    if run_component_diagnostics and component_metadata_path and component_geometry_path:
        component_aligned, component_checks = evaluate_component_diagnostics(
            decomposed,
            metadata_path=component_metadata_path,
            geometry_path=component_geometry_path,
            theta_tip_deg=component_theta_tip_deg,
            airflow_mode=component_airflow_mode,
            primary_prior_parameters=component_primary_prior_parameters,
        )
        component_phase, component_half, component_span, component_similarity = summarize_component_attribution(
            component_aligned,
            config,
        )
        component_contract = pd.DataFrame([component_checks])
        tolerance = 1.0e-9
        for key, value in component_checks.items():
            if ("sum_max_abs_error" in key or "left_right_max_abs_error" in key or "spanwise_max_abs_error" in key) and float(value) > tolerance:
                strict_failures.append(f"{key}={value} exceeds {tolerance}")
        availability["component_attribution"] = "available_opt_in_frozen_diagnostic"
        if run_physical_sensitivity:
            physical_sensitivity = physical_sensitivity_audit(
                decomposed,
                metadata_path=component_metadata_path,
                geometry_path=component_geometry_path,
                theta_tip_deg=component_theta_tip_deg,
                config=config,
                phase_step_rad=sensitivity_phase_step_rad,
                incidence_step_rad=sensitivity_incidence_step_rad,
                twist_step_deg=sensitivity_twist_step_deg,
                primary_prior_parameters=component_primary_prior_parameters,
                airflow_mode=component_airflow_mode,
            )
            availability["physical_sensitivity"] = "available_bounded_local_probe"
        else:
            availability["physical_sensitivity"] = "not_available_disabled"
    else:
        availability["component_attribution"] = "not_available_missing_metadata_or_geometry"
        availability["physical_sensitivity"] = "not_available_missing_component_diagnostics"

    decision = decision_summary(
        phase_summary=phase_summary,
        cycle_means=cycle_means,
        harmonic_by_log=harmonic_by_log,
        static_probes=static_probes,
        history_probes=history_probes,
        label_uncertainty_phase=label_uncertainty,
        prior_probe=prior_probe,
        config=config,
        physical_sensitivity=physical_sensitivity,
    )
    summary = audit_summary_metrics(cycle_means, phase_summary, decision)
    waveform_metrics_columns = [
        column
        for column in cycle_means.columns
        if column in {"partition", "log_id", "segment_id", "cycle_id", "sample_count"}
        or column.startswith("waveform_")
        or column.startswith("mean_energy_fraction_")
    ]
    tables = {
        "alignment_mismatches": mismatches,
        "cycle_quality": cycle_selection.quality,
        "cycle_rejection_reasons": cycle_selection.rejections,
        "phase_alignment_cycles": phase_cycles,
        "phase_alignment_by_log": phase_by_log,
        "cycle_mean_residuals": cycle_means,
        "wingbeat_waveform_metrics": cycle_means.loc[:, waveform_metrics_columns],
        "phase_waveform": waveform,
        "half_stroke_residuals": half_strokes,
        "half_stroke_summary_by_log": half_by_log,
        "harmonic_cycle_summary": harmonics,
        "harmonic_by_log": harmonic_by_log,
        "condition_dependence": condition_table,
        "condition_probe_metrics": static_probes,
        "waveform_repeatability": repeatability,
        "log_waveform_correlations": log_correlations,
        "date_level_summary": date_summary,
        "label_robustness": label_variant_table,
        "label_uncertainty_phase": label_uncertainty,
        "static_history_probe_metrics": history_probes,
        "matched_capacity_prior_probe": prior_probe,
        "trim_impact_estimate": trim,
        "component_phase_contributions": component_phase,
        "component_half_stroke_summary": component_half,
        "component_spanwise_summary": component_span,
        "component_residual_similarity": component_similarity,
        "component_contract_checks": component_contract,
        "physical_sensitivity_similarity": physical_sensitivity,
    }
    alignment_report = {
        "schema_version": SCHEMA_VERSION,
        "partitions": alignment_by_partition,
        "total_sample_rows": int(len(aligned)),
        "total_aligned_rows": int(len(aligned) - len(mismatches.loc[mismatches.get("mismatch_type") == "missing_prior"])),
        "test_rows_loaded": 0,
        "status": "ok" if not strict_failures else "strict_failure",
    }
    return AuditRunResult(
        tables=tables,
        alignment_report=alignment_report,
        phase_alignment_summary=phase_summary,
        decision=decision,
        summary=summary,
        condition_sources=condition_sources,
        availability=availability,
        strict_failures=strict_failures,
    )
