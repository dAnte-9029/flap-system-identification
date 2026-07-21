#!/usr/bin/env python3
"""Thin CLI for EDA0 DeLaurier longitudinal-force attribution."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from system_identification.analysis.force_discrepancy_attribution import (  # noqa: E402
    AuditConfig,
    SCHEMA_VERSION,
    sha256_file,
)
from system_identification.analysis.force_discrepancy_audit import run_force_discrepancy_audit  # noqa: E402
from system_identification.analysis.force_discrepancy_report import build_chinese_report  # noqa: E402
from system_identification.artifacts.prior_registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    resolve_delaurier_prior,
)
from system_identification.plotting.force_discrepancy_attribution import (  # noqa: E402
    PlotContext,
    write_audit_figures,
)


DEFAULT_DATASET = Path("dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_sg0p03_v1")
DEFAULT_LABEL_VARIANT = Path("dataset/canonical_v0.2_training_ready_split_measured_massprops_ratio8_v1")
DEFAULT_METADATA = Path("metadata/aircraft/flapper_01/aircraft_metadata.yaml")
TABLE_FILENAMES = {
    "alignment_mismatches": "alignment_mismatches.csv",
    "cycle_quality": "cycle_quality.csv",
    "cycle_rejection_reasons": "cycle_rejection_reasons.csv",
    "phase_alignment_cycles": "phase_alignment_cycles.csv",
    "phase_alignment_by_log": "phase_alignment_by_log.csv",
    "cycle_mean_residuals": "cycle_mean_residuals.csv",
    "wingbeat_waveform_metrics": "wingbeat_waveform_metrics.csv",
    "phase_waveform": "phase_binned_waveform.csv",
    "half_stroke_residuals": "half_stroke_residuals.csv",
    "half_stroke_summary_by_log": "half_stroke_summary_by_log.csv",
    "harmonic_cycle_summary": "harmonic_cycle_summary.csv",
    "harmonic_by_log": "harmonic_by_log.csv",
    "condition_dependence": "condition_dependence.csv",
    "condition_probe_metrics": "condition_probe_metrics.csv",
    "waveform_repeatability": "waveform_repeatability.csv",
    "log_waveform_correlations": "log_waveform_correlations.csv",
    "date_level_summary": "date_level_summary.csv",
    "label_robustness": "label_robustness.csv",
    "label_uncertainty_phase": "label_uncertainty_phase.csv",
    "static_history_probe_metrics": "static_history_probe_metrics.csv",
    "matched_capacity_prior_probe": "matched_capacity_prior_probe.csv",
    "trim_impact_estimate": "trim_impact_estimate.csv",
    "component_phase_contributions": "component_phase_contributions.csv",
    "component_half_stroke_summary": "component_half_stroke_summary.csv",
    "component_spanwise_summary": "component_spanwise_summary.csv",
    "component_residual_similarity": "component_residual_similarity.csv",
    "component_contract_checks": "component_contract_checks.csv",
    "physical_sensitivity_similarity": "physical_sensitivity_similarity.csv",
}


def _git(args: Sequence[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for package in ("numpy", "pandas", "scipy", "matplotlib", "pyarrow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def _parse_csv_numbers(raw: str, *, integer: bool = False) -> tuple[float, ...] | tuple[int, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    return tuple(int(value) for value in values) if integer else tuple(float(value) for value in values)


def _validate_ratio_contract(
    dataset_manifest: dict[str, object], prior_manifest: dict[str, object]
) -> dict[str, object]:
    keys = (
        "wing_transmission_ratio",
        "ratio_contract_version",
        "ratio_source",
        "phase_contract_version",
        "frequency_contract_version",
    )
    missing_dataset = [key for key in keys if not dataset_manifest.get(key)]
    missing_prior = [key for key in keys if not prior_manifest.get(key)]
    if missing_dataset or missing_prior:
        raise ValueError(
            "ratio contract missing from EDA0 inputs: "
            f"dataset={missing_dataset}, prior={missing_prior}"
        )
    for key in ("wing_transmission_ratio", "ratio_contract_version", "phase_contract_version", "frequency_contract_version"):
        if str(dataset_manifest[key]) != str(prior_manifest[key]):
            raise ValueError(f"dataset/prior ratio contract mismatch for {key}")
    return {key: dataset_manifest[key] for key in keys}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--prior-root", type=Path)
    parser.add_argument("--prior-id")
    parser.add_argument("--prior-registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--allow-legacy-prior",
        action="store_true",
        help="Allow an explicitly requested legacy prior for historical reproduction only",
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/force_discrepancy_attribution"))
    parser.add_argument("--partitions", nargs="+", default=("train", "validation"))
    parser.add_argument("--label-variant-root", type=Path, action="append", default=None)
    parser.add_argument("--no-label-variants", action="store_true")
    parser.add_argument("--wt1-sensitivity-artifact", type=Path)
    parser.add_argument("--component-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--component-geometry", type=Path)
    parser.add_argument("--component-theta-tip-deg", type=float)
    parser.add_argument("--skip-component-diagnostics", action="store_true")
    parser.add_argument("--skip-physical-sensitivity", action="store_true")
    parser.add_argument("--strict-require-label-variants", action="store_true")
    parser.add_argument("--strict-require-component-diagnostics", action="store_true")
    parser.add_argument("--phase-bins", type=int, default=72)
    parser.add_argument("--minimum-cycle-samples", type=int, default=12)
    parser.add_argument("--minimum-phase-coverage-rad", type=float, default=5.5)
    parser.add_argument("--maximum-cycle-missing-fraction", type=float, default=0.0)
    parser.add_argument("--shift-search-range-deg", type=float, default=180.0)
    parser.add_argument("--condition-bins", type=int, default=5)
    parser.add_argument("--harmonic-max-order", type=int, default=4)
    parser.add_argument("--ridge-alphas", default="0,0.1,1,10,100")
    parser.add_argument("--history-lengths", default="1,2,4")
    parser.add_argument("--phase-sensitivity-step-deg", type=float, default=2.0)
    parser.add_argument("--incidence-sensitivity-step-deg", type=float, default=1.0)
    parser.add_argument("--twist-sensitivity-step-deg", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=20260717)
    parser.add_argument("--phase-offset-threshold-deg", type=float, default=10.0)
    parser.add_argument("--phase-component-agreement-threshold-deg", type=float, default=10.0)
    parser.add_argument("--fixed-delay-improvement-fraction", type=float, default=0.10)
    parser.add_argument("--mean-energy-threshold", type=float, default=0.15)
    parser.add_argument("--phase-energy-threshold", type=float, default=0.30)
    parser.add_argument("--condition-probe-gain-threshold", type=float, default=0.05)
    parser.add_argument("--history-probe-gain-threshold", type=float, default=0.05)
    parser.add_argument("--prior-gain-threshold", type=float, default=0.02)
    parser.add_argument("--label-uncertainty-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--physical-similarity-threshold", type=float, default=0.70)
    parser.add_argument("--run-id", help="Override timestamp_gitsha run directory name (mainly for tests)")
    parser.add_argument("--report-path", type=Path)
    return parser


def _resolved_component_contract(args: argparse.Namespace, prior_manifest: dict[str, object]) -> tuple[Path | None, float]:
    geometry = args.component_geometry
    if geometry is None and prior_manifest.get("wing_geom_csv"):
        geometry = Path(str(prior_manifest["wing_geom_csv"]))
    theta = args.component_theta_tip_deg
    parameters = prior_manifest.get("delaurier_parameters", {})
    if theta is None and prior_manifest.get("theta_tip_deg") is not None:
        theta = float(prior_manifest["theta_tip_deg"])
    if theta is None and isinstance(parameters, dict):
        theta = float(parameters.get("twist_eta_max_deg", 10.0))
    return geometry, float(theta if theta is not None else 10.0)


def _main(args: argparse.Namespace) -> int:
    dataset_root = args.dataset_root.resolve()
    prior_resolution = resolve_delaurier_prior(
        prior_root=args.prior_root,
        prior_id=args.prior_id,
        registry_path=args.prior_registry,
        allow_legacy=args.allow_legacy_prior,
        requested_partitions=args.partitions,
    )
    prior_root = prior_resolution.artifact_root
    split_manifest = (args.split_manifest or dataset_root / "dataset_manifest.json").resolve()
    for path, label in ((dataset_root, "dataset root"), (prior_root, "prior root"), (split_manifest, "split manifest")):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    dataset_manifest_path = dataset_root / "dataset_manifest.json"
    dataset_manifest = _read_json(dataset_manifest_path)
    prior_manifest_path = prior_root / "manifest.json"
    prior_manifest = _read_json(prior_manifest_path)
    ratio_contract = _validate_ratio_contract(dataset_manifest, prior_manifest)
    component_airflow_mode = str(
        prior_manifest.get("airflow_mode")
        or prior_resolution.airflow_contract
    )
    geometry_path, theta_tip_deg = _resolved_component_contract(args, prior_manifest)
    if geometry_path is not None:
        geometry_path = geometry_path.resolve()
    label_variants = [] if args.no_label_variants else args.label_variant_root
    if label_variants is None:
        label_variants = [DEFAULT_LABEL_VARIANT] if DEFAULT_LABEL_VARIANT.exists() else []

    git_commit = _git(["rev-parse", "HEAD"])
    git_short = git_commit[:7]
    branch = _git(["branch", "--show-current"])
    dirty_before = _git(["status", "--porcelain=v1"])
    timestamp = datetime.now(timezone.utc)
    run_id = args.run_id or f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}_{git_short}"
    run_dir = (args.output_root / run_id).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    command = "python " + " ".join(shlex.quote(value) for value in sys.argv)
    (run_dir / "run_command.txt").write_text(command + "\n", encoding="utf-8")

    config = AuditConfig(
        phase_bins=args.phase_bins,
        minimum_cycle_samples=args.minimum_cycle_samples,
        minimum_phase_coverage_rad=args.minimum_phase_coverage_rad,
        maximum_cycle_missing_fraction=args.maximum_cycle_missing_fraction,
        shift_search_range_rad=math.radians(args.shift_search_range_deg),
        harmonic_max_order=args.harmonic_max_order,
        condition_bins=args.condition_bins,
        ridge_alphas=tuple(_parse_csv_numbers(args.ridge_alphas)),
        history_lengths=tuple(_parse_csv_numbers(args.history_lengths, integer=True)),
        random_seed=args.random_seed,
        phase_offset_threshold_rad=math.radians(args.phase_offset_threshold_deg),
        phase_component_agreement_threshold_rad=math.radians(args.phase_component_agreement_threshold_deg),
        fixed_delay_improvement_fraction=args.fixed_delay_improvement_fraction,
        mean_energy_threshold=args.mean_energy_threshold,
        phase_energy_threshold=args.phase_energy_threshold,
        condition_probe_gain_threshold=args.condition_probe_gain_threshold,
        history_probe_gain_threshold=args.history_probe_gain_threshold,
        prior_gain_threshold=args.prior_gain_threshold,
        label_uncertainty_ratio_threshold=args.label_uncertainty_ratio_threshold,
        physical_similarity_threshold=args.physical_similarity_threshold,
    )

    input_hashes_before = {
        "dataset_manifest": sha256_file(dataset_manifest_path),
        "split_manifest": sha256_file(split_manifest),
        "prior_manifest": sha256_file(prior_manifest_path) if prior_manifest_path.is_file() else "missing",
    }
    result = run_force_discrepancy_audit(
        dataset_root=dataset_root,
        prior_root=prior_root,
        partitions=args.partitions,
        config=config,
        label_variant_roots=[Path(value).resolve() for value in label_variants],
        wt1_sensitivity_artifact=args.wt1_sensitivity_artifact,
        component_metadata_path=args.component_metadata.resolve() if args.component_metadata else None,
        component_geometry_path=geometry_path,
        component_theta_tip_deg=theta_tip_deg,
        component_airflow_mode=component_airflow_mode,
        component_primary_prior_parameters=(
            prior_manifest.get("delaurier_parameters")
            if isinstance(prior_manifest.get("delaurier_parameters"), dict)
            else None
        ),
        sensitivity_phase_step_rad=math.radians(args.phase_sensitivity_step_deg),
        sensitivity_incidence_step_rad=math.radians(args.incidence_sensitivity_step_deg),
        sensitivity_twist_step_deg=args.twist_sensitivity_step_deg,
        run_component_diagnostics=not args.skip_component_diagnostics,
        run_physical_sensitivity=not args.skip_physical_sensitivity,
    )
    if args.strict_require_label_variants and result.availability["label_robustness"] != "available":
        result.strict_failures.append("label variants required by strict mode but unavailable")
    if args.strict_require_component_diagnostics and not result.availability["component_attribution"].startswith("available"):
        result.strict_failures.append("component diagnostics required by strict mode but unavailable")

    written_tables: dict[str, str] = {}
    for name, table in result.tables.items():
        filename = TABLE_FILENAMES[name]
        if table.empty and name not in {"alignment_mismatches", "cycle_rejection_reasons"}:
            continue
        path = run_dir / filename
        table.to_csv(path, index=False)
        written_tables[name] = str(path)
    (run_dir / "alignment_report.json").write_text(
        json.dumps(result.alignment_report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "phase_alignment_summary.json").write_text(
        json.dumps(result.phase_alignment_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "decision_summary.json").write_text(
        json.dumps(result.decision, indent=2, sort_keys=True), encoding="utf-8"
    )

    quality = result.tables["cycle_quality"]
    accepted_quality = quality.loc[quality["accepted"]]
    used_logs = sorted(accepted_quality["log_id"].astype(str).unique())
    all_logs = sorted(quality["log_id"].astype(str).unique())
    used_partitions = sorted(accepted_quality["partition"].astype(str).unique())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp": timestamp.isoformat(),
        "branch": branch,
        "git_commit": git_commit,
        "git_short_sha": git_short,
        "dirty_status_before_run": dirty_before.splitlines(),
        "python_package_versions": _package_versions(),
        "dataset_path": str(dataset_root),
        "dataset_manifest": str(dataset_manifest_path.resolve()),
        "dataset_manifest_hash": input_hashes_before["dataset_manifest"],
        "split_manifest": str(split_manifest),
        "split_manifest_hash": input_hashes_before["split_manifest"],
        "split_identity": dataset_manifest.get("dataset_id", dataset_root.name),
        **ratio_contract,
        "phase_column": dataset_manifest.get("phase_column", "mechanical_phase_rad"),
        "used_partitions": used_partitions,
        "test_partition_loaded": False,
        "test_rows_loaded": 0,
        "used_log_ids": used_logs,
        "excluded_log_ids": sorted(set(all_logs) - set(used_logs)),
        "prior_artifact": str(prior_root),
        "prior_id": prior_resolution.prior_id,
        "prior_lifecycle_status": prior_resolution.lifecycle_status,
        "prior_registry": str(prior_resolution.registry_path) if prior_resolution.registry_path else None,
        "prior_is_legacy": prior_resolution.is_legacy,
        "prior_manifest_hash": input_hashes_before["prior_manifest"],
        "prior_source_commit": prior_resolution.physics_source_commit,
        "prior_contracts": {
            "frame_contract": prior_resolution.frame_contract,
            "airflow_contract": prior_resolution.airflow_contract,
            "phase_contract": prior_resolution.phase_contract,
        },
        "label_artifact": str(dataset_root),
        "label_variant_roots": [str(Path(value).resolve()) for value in label_variants],
        "alignment_keys": ["log_id", "timestamp_us"],
        "cycle_quality_thresholds": {
            "minimum_cycle_samples": config.minimum_cycle_samples,
            "minimum_phase_coverage_rad": config.minimum_phase_coverage_rad,
            "maximum_missing_fraction": config.maximum_cycle_missing_fraction,
            "frequency_range_hz": config.frequency_range_hz,
        },
        "phase_grid": {"bins": config.phase_bins, "endpoint": False, "interval_rad": [0.0, 2.0 * math.pi]},
        "condition_variables": result.condition_sources,
        "sensitivity_steps": {
            "phase_offset_deg": args.phase_sensitivity_step_deg,
            "mean_incidence_deg": args.incidence_sensitivity_step_deg,
            "dynamic_twist_tip_deg": args.twist_sensitivity_step_deg,
            "scale_probe": 0.05,
        },
        "diagnostic_probe_settings": {
            "harmonic_max_order": config.harmonic_max_order,
            "ridge_alphas": config.ridge_alphas,
            "history_lengths_samples": config.history_lengths,
            "fit_partition": "train",
            "evaluation_partition": "validation",
            "sample_weighting": "equal-log then equal-cycle then equal-sample",
            "deployable_model_written": False,
        },
        "decision_thresholds": {
            key: value for key, value in asdict(config).items() if "threshold" in key
        },
        "random_seed": config.random_seed,
        "output_schema_version": SCHEMA_VERSION,
        "availability": result.availability,
        "component_diagnostic_contract": {
            "metadata": str(args.component_metadata.resolve()) if args.component_metadata else None,
            "geometry": str(geometry_path) if geometry_path else None,
            "theta_tip_deg": theta_tip_deg,
            "airflow_mode": component_airflow_mode,
            "primary_prior_may_use_different_exporter": False,
            "primary_scalar_parameters_applied": prior_manifest.get("delaurier_parameters", {}),
            "separation_component_policy": "attached-flow closure matching active primary prior",
        },
        "implementation_inventory": {
            "reused": [
                "conventions.phase wrap/direction",
                "physics.delaurier airflow/dynamic-twist/strip-wrench",
                "physics.baselines.wing_only aggregate component diagnostics",
                "canonical dataset phase/cycle/sample identity",
            ],
            "migrated": [
                "phase/condition diagnostic patterns into authoritative analysis package",
                "component residual attribution away from script-only orchestration",
            ],
            "new": [
                "active/legacy prior registry with fail-closed resolution",
                "attitude-aware train/validation keyed prior materialization",
                "strict keyed audit alignment",
                "phase-delay hypotheses",
                "mean/WB decomposition",
                "authoritative half-stroke, harmonic, repeatability, probes, decision matrix",
            ],
            "legacy_duplicates_not_removed": [
                "scripts/analyze_delaurier_residual_phase.py",
                "scripts/analyze_delaurier_residual_frequency.py",
                "scripts/analyze_delaurier_residual_conditions.py",
                "scripts/analyze_component_residual_attribution.py",
                "legacy label diagnostic scripts",
            ],
        },
        "written_tables": written_tables,
        "strict_failures": result.strict_failures,
    }
    figures = write_audit_figures(
        output_dir=run_dir / "figures",
        context=PlotContext(run_id=run_id, git_short_sha=git_short, partitions=tuple(used_partitions)),
        tables=result.tables,
        decision=result.decision,
    )
    manifest["figures"] = [str(path) for path in figures]
    input_hashes_after = {
        "dataset_manifest": sha256_file(dataset_manifest_path),
        "split_manifest": sha256_file(split_manifest),
        "prior_manifest": sha256_file(prior_manifest_path) if prior_manifest_path.is_file() else "missing",
    }
    manifest["input_hashes_unchanged"] = input_hashes_before == input_hashes_after
    if not manifest["input_hashes_unchanged"]:
        result.strict_failures.append("input artifact hash changed during audit")

    report_path = args.report_path or PROJECT_ROOT / "docs" / "analysis" / f"{timestamp.date().isoformat()}_force_discrepancy_attribution_audit.md"
    manifest["chinese_report_path"] = str(report_path.resolve())
    build_chinese_report(
        output_directory=run_dir,
        report_path=report_path,
        manifest=manifest,
        summary=result.summary,
        decision=result.decision,
        tables=result.tables,
        run_command=command,
    )
    manifest["strict_failures"] = result.strict_failures
    manifest["strict_failure_count"] = len(result.strict_failures)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"branch: {branch}")
    print(f"git commit: {git_commit}")
    print(f"output directory: {run_dir}")
    print(f"used partitions: {used_partitions}")
    print(f"used logs: {used_logs}")
    print(f"accepted/rejected cycles: {int(quality['accepted'].sum())}/{int((~quality['accepted']).sum())}")
    print(f"Fx main discrepancy type: mean={result.decision['mean_correction_fx']}, phase={result.decision['phase_correction_fx']}")
    print(f"Fz main discrepancy type: mean={result.decision['mean_correction_fz']}, phase={result.decision['phase_correction_fz']}")
    print(f"phase offset verdict: {result.decision['fix_phase_convention_first']}")
    print(f"fixed delay verdict: {result.decision['fix_fixed_delay_first']}")
    print(f"mean correction verdict: Fx={result.decision['mean_correction_fx']}, Fz={result.decision['mean_correction_fz']}")
    print(f"phase correction verdict: Fx={result.decision['phase_correction_fx']}, Fz={result.decision['phase_correction_fz']}")
    print(f"condition features recommendation: {result.decision['condition_features_recommended']}")
    print(f"dynamic model verdict: {result.decision['dynamic_model_needed']}; TCN={result.decision['tcn_needed']}")
    print(f"label uncertainty verdict: {result.decision['label_uncertainty_blocks_correction']}")
    print(f"prior incremental value verdict: {result.decision['prior_has_incremental_value']}")
    print(f"Chinese report path: {report_path.resolve()}")
    print(f"strict failure count: {len(result.strict_failures)}")
    return 2 if result.strict_failures else 0


def main() -> int:
    try:
        return _main(build_parser().parse_args())
    except Exception as exc:  # CLI boundary: configuration/input/execution failure -> 1
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
