"""Immutable artifact writer for the C1 correction-ready dataset."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import subprocess
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from system_identification.artifacts.io import sha256_file, write_json, write_table
from system_identification.artifacts.prior_registry import PriorResolution
from system_identification.data.correction_ready import (
    CONDITION_COLUMNS,
    FORCE_COMPONENTS,
    SCHEMA_VERSION,
    CorrectionReadyConfig,
    align_correction_partition,
    apply_condition_normalization,
    build_correction_tables,
    fit_condition_normalization,
    normalize_correction_partitions,
    segment_complete_cycles,
    table_semantic_hash,
    validate_correction_contract,
)


PARTITION_FILES = {"train": "train_samples.parquet", "validation": "val_samples.parquet"}
PRIOR_FILES = {"train": "train_predictions.parquet", "validation": "val_predictions.parquet"}


@dataclass(frozen=True)
class CorrectionReadyArtifactResult:
    """Paths and summaries produced by one immutable artifact build."""

    output_dir: Path
    manifest: Mapping[str, object]
    quality_checks: Mapping[str, object]
    dataset_summary: Mapping[str, object]
    exit_code: int


def _read_mapping(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(value: object, *, manifest_path: Path, project_root: Path) -> Path:
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate.resolve()
    project_candidate = (project_root / candidate).resolve()
    if project_candidate.exists():
        return project_candidate
    return (manifest_path.parent / candidate).resolve()


def _dataset_manifest_chain(
    manifest_path: Path,
    *,
    project_root: Path,
) -> list[tuple[Path, dict[str, object]]]:
    chain: list[tuple[Path, dict[str, object]]] = []
    pending = [manifest_path.resolve()]
    seen: set[Path] = set()
    while pending:
        current = pending.pop(0)
        if current in seen or not current.is_file():
            continue
        seen.add(current)
        manifest = _read_mapping(current)
        chain.append((current, manifest))
        for key in ("source_manifest_path", "dataset_manifest"):
            if manifest.get(key):
                pending.append(
                    _resolve_path(manifest[key], manifest_path=current, project_root=project_root)
                )
        if manifest.get("source_split_root"):
            root = _resolve_path(
                manifest["source_split_root"], manifest_path=current, project_root=project_root
            )
            pending.append(root / "dataset_manifest.json")
    return chain


def _metadata_path(
    chain: Sequence[tuple[Path, Mapping[str, object]]],
    *,
    project_root: Path,
) -> Path:
    for manifest_path, manifest in chain:
        if manifest.get("metadata_path"):
            path = _resolve_path(
                manifest["metadata_path"], manifest_path=manifest_path, project_root=project_root
            )
            if path.is_file():
                return path
    raise ValueError("Dataset provenance chain has no readable aircraft metadata_path")


def _git_state(project_root: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=project_root, text=True).strip()

    status = run("status", "--porcelain=v1")
    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "dirty_paths": status.splitlines(),
    }


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("numpy", "pandas", "pyarrow", "PyYAML", "scipy"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not_installed"
    return result


def _log_ids_from_chain(
    chain: Sequence[tuple[Path, Mapping[str, object]]],
    *,
    project_root: Path,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    keys = {
        "train": "train_logs_csv",
        "validation": "val_logs_csv",
        "test": "test_logs_csv",
    }
    for partition, key in keys.items():
        for manifest_path, manifest in chain:
            if not manifest.get(key):
                continue
            path = _resolve_path(manifest[key], manifest_path=manifest_path, project_root=project_root)
            if not path.is_file():
                continue
            table = pd.read_csv(path)
            if "log_id" in table.columns:
                result[partition] = sorted(table["log_id"].astype(str).unique().tolist())
                break
    return result


def _condition_coverage(cycle_table: pd.DataFrame) -> dict[str, object]:
    coverage: dict[str, object] = {}
    train = cycle_table.loc[cycle_table["partition"] == "train"]
    for partition, group in cycle_table.groupby("partition", sort=True):
        coverage[partition] = {
            column: {
                "minimum": float(group[column].min()),
                "maximum": float(group[column].max()),
                "mean": float(group[column].mean()),
                "std": float(group[column].std(ddof=0)),
            }
            for column in CONDITION_COLUMNS
        }
    validation = cycle_table.loc[cycle_table["partition"] == "validation"]
    coverage["validation_outside_training_range"] = {
        column: int(
            ((validation[column] < train[column].min()) | (validation[column] > train[column].max())).sum()
        )
        for column in CONDITION_COLUMNS
    }
    return coverage


def _max_group_mean(frame: pd.DataFrame, columns: Sequence[str]) -> float:
    if frame.empty or not columns:
        return 0.0
    return float(frame.groupby("cycle_id")[list(columns)].mean().abs().to_numpy().max())


def _quality_checks(
    *,
    prior: PriorResolution,
    prior_hash_matches: bool,
    alignment_report: Mapping[str, object],
    cycle_quality: pd.DataFrame,
    cycle_table: pd.DataFrame,
    waveform: pd.DataFrame,
    normalization: Mapping[str, Mapping[str, object]],
    deterministic: bool,
    inputs_unchanged: bool,
    production_prior_unchanged: bool,
    config: CorrectionReadyConfig,
) -> dict[str, object]:
    checks: dict[str, dict[str, object]] = {}

    def record(name: str, passed: bool, detail: object) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    partitions = alignment_report["partitions"]
    duplicate_count = sum(int(item.get("duplicate_keys", 0)) for item in partitions.values())
    missing_count = sum(int(item.get("missing_prior_rows", 0)) for item in partitions.values())
    coverage_complete = all(int(item.get("missing_prior_rows", 0)) == 0 for item in partitions.values())
    record("authoritative_active_prior_resolved", prior.lifecycle_status == "active", prior.lifecycle_status)
    record("legacy_prior_not_used", not prior.is_legacy, prior.prior_id)
    record("prior_artifact_hash_matches", prior_hash_matches, prior.manifest.get("prediction_sha256", {}))
    record("train_validation_prior_coverage_complete", coverage_complete, {"missing_rows": missing_count})
    record("test_label_not_loaded", alignment_report.get("test_labels_loaded") is False, False)
    record("alignment_keys_unique", duplicate_count == 0, duplicate_count)
    record(
        "alignment_does_not_use_row_order",
        all(not bool(item.get("row_order_used", True)) for item in partitions.values()),
        {name: item.get("row_order_used") for name, item in partitions.items()},
    )
    record(
        "cycles_do_not_cross_logs",
        int(waveform.groupby("cycle_id")["log_id"].nunique().max()) == 1,
        int(waveform.groupby("cycle_id")["log_id"].nunique().max()),
    )
    record(
        "cycles_do_not_cross_partitions",
        int(waveform.groupby("cycle_id")["partition"].nunique().max()) == 1,
        int(waveform.groupby("cycle_id")["partition"].nunique().max()),
    )
    accepted_fraction = float(cycle_quality["accepted"].mean()) if len(cycle_quality) else 0.0
    record(
        "cycle_phase_quality_passes",
        bool(
            len(cycle_table)
            and accepted_fraction >= config.minimum_accepted_cycle_fraction
            and (cycle_table["phase_coverage_rad"] >= config.minimum_phase_coverage_rad).all()
            and (cycle_table["maximum_phase_gap_rad"] <= config.maximum_phase_gap_rad).all()
        ),
        {"accepted_fraction": accepted_fraction, "minimum": config.minimum_accepted_cycle_fraction},
    )
    endpoint_rows = waveform.loc[np.isclose(waveform["phase_rad"], 0.0, atol=1.0e-10)]
    duplicate_endpoints = int(endpoint_rows.duplicated(["cycle_id"], keep="first").sum())
    record("phase_endpoint_not_repeated", duplicate_endpoints == 0, duplicate_endpoints)
    decomposition_errors: dict[str, float] = {}
    zero_means: dict[str, float] = {}
    for component in FORCE_COMPONENTS:
        decomposition_errors[f"label_{component}"] = float(
            np.max(
                np.abs(
                    waveform[f"label_{component}_n"]
                    - waveform[f"label_{component}_mean_n"]
                    - waveform[f"label_{component}_waveform_n"]
                )
            )
        )
        decomposition_errors[f"prior_{component}"] = float(
            np.max(
                np.abs(
                    waveform[f"prior_{component}_n"]
                    - waveform[f"prior_{component}_mean_n"]
                    - waveform[f"prior_{component}_waveform_n"]
                )
            )
        )
        decomposition_errors[f"residual_{component}"] = float(
            np.max(
                np.abs(
                    waveform[f"residual_{component}_n"]
                    - waveform[f"residual_{component}_mean_n"]
                    - waveform[f"residual_{component}_waveform_n"]
                )
            )
        )
        for kind in ("label", "prior", "residual"):
            zero_means[f"{kind}_{component}"] = _max_group_mean(
                waveform, [f"{kind}_{component}_waveform_n"]
            )
    record(
        "label_decomposition_conserved",
        max(value for key, value in decomposition_errors.items() if key.startswith("label"))
        <= config.reconstruction_tolerance,
        decomposition_errors,
    )
    record(
        "prior_decomposition_conserved",
        max(value for key, value in decomposition_errors.items() if key.startswith("prior"))
        <= config.reconstruction_tolerance,
        decomposition_errors,
    )
    record(
        "residual_mean_waveform_conserved",
        max(value for key, value in decomposition_errors.items() if key.startswith("residual"))
        <= config.reconstruction_tolerance,
        decomposition_errors,
    )
    for kind in ("label", "prior", "residual"):
        maximum = max(value for key, value in zero_means.items() if key.startswith(kind))
        record(
            f"{kind}_waveform_zero_mean_per_cycle",
            maximum <= config.zero_mean_tolerance,
            maximum,
        )
    centered = [column for column in waveform if column.endswith("_phase_centered")]
    centered_maximum = _max_group_mean(waveform, centered)
    record(
        "centered_harmonic_basis_zero_mean_per_cycle",
        centered_maximum <= config.zero_mean_tolerance,
        centered_maximum,
    )
    record(
        "normalization_uses_train_only",
        all(item.get("source_partition") == "train" for item in normalization.values()),
        {column: item.get("source_partition") for column, item in normalization.items()},
    )
    train_cycles = int((cycle_table["partition"] == "train").sum())
    record(
        "validation_not_used_for_normalization",
        all(int(item.get("training_cycle_count", -1)) == train_cycles for item in normalization.values()),
        {"training_cycle_count": train_cycles},
    )
    weight_columns = [column for column in waveform if column.startswith("weight_")]
    weights = waveform[weight_columns].to_numpy(dtype=float)
    record("weights_finite_and_nonnegative", bool(np.isfinite(weights).all() and (weights >= 0).all()), weight_columns)
    record("artifact_rebuild_deterministic", deterministic, deterministic)
    record("input_artifacts_unmodified", inputs_unchanged, inputs_unchanged)
    record("production_delaurier_predictions_unmodified", production_prior_unchanged, production_prior_unchanged)
    failures = [name for name, item in checks.items() if not item["passed"]]
    return {
        "schema_version": f"{SCHEMA_VERSION}.quality_checks",
        "checks": checks,
        "strict_failure_count": len(failures),
        "strict_failures": failures,
        "status": "passed" if not failures else "failed",
    }


def _schema(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [{"name": column, "dtype": str(frame[column].dtype)} for column in frame.columns]


def build_longitudinal_correction_ready_artifact(
    *,
    dataset_root: str | Path,
    split_manifest: str | Path,
    prior: PriorResolution,
    partitions: Sequence[str],
    output_root: str | Path,
    config: CorrectionReadyConfig,
    output_format: str = "parquet",
    command: Sequence[str] = (),
    project_root: str | Path | None = None,
) -> CorrectionReadyArtifactResult:
    """Build and serialize one immutable C1 artifact without loading test labels."""

    started_at = time.perf_counter()
    config.validate()
    requested = normalize_correction_partitions(partitions)
    if set(requested) != {"train", "validation"}:
        raise ValueError("A correction-ready training artifact requires both train and validation")
    if output_format not in {"parquet", "csv", "both"}:
        raise ValueError("output_format must be parquet, csv, or both")
    root = Path(project_root).resolve() if project_root is not None else _project_root()
    dataset = Path(dataset_root).resolve()
    split_path = Path(split_manifest).resolve()
    output_parent = Path(output_root).resolve()
    dataset_manifest_path = dataset / "dataset_manifest.json"
    dataset_manifest = _read_mapping(dataset_manifest_path)
    split_mapping = _read_mapping(split_path)
    if str(split_mapping.get("dataset_id", "")) != str(dataset_manifest.get("dataset_id", "")):
        raise ValueError("split identity does not match dataset identity")
    chain = _dataset_manifest_chain(dataset_manifest_path, project_root=root)
    metadata_path = _metadata_path(chain, project_root=root)
    metadata = _read_mapping(metadata_path)
    contract = validate_correction_contract(prior, dataset_manifest, metadata)
    if set(prior.required_partitions) < {"train", "validation"}:
        raise ValueError("Authoritative prior does not cover train and validation")
    expected_dataset_hash = prior.manifest.get("dataset_manifest_sha256")
    if expected_dataset_hash and str(expected_dataset_hash) != sha256_file(dataset_manifest_path):
        raise ValueError("dataset manifest hash differs from authoritative prior provenance")

    input_paths: list[Path] = [dataset_manifest_path, split_path, prior.artifact_root / "manifest.json", metadata_path]
    for partition in requested:
        input_paths.extend(
            [dataset / PARTITION_FILES[partition], prior.artifact_root / PRIOR_FILES[partition]]
        )
    unique_input_paths = list(dict.fromkeys(path.resolve() for path in input_paths))
    input_hashes_before = {str(path): sha256_file(path) for path in unique_input_paths}

    aligned_pieces: list[pd.DataFrame] = []
    mismatch_pieces: list[pd.DataFrame] = []
    alignment_partitions: dict[str, object] = {}
    prior_hash_matches = True
    prediction_hashes = prior.manifest.get("prediction_sha256", {})
    prediction_hash_mapping = prediction_hashes if isinstance(prediction_hashes, Mapping) else {}
    for partition in requested:
        sample_path = dataset / PARTITION_FILES[partition]
        prior_path = prior.artifact_root / PRIOR_FILES[partition]
        expected_source_hash = prior.manifest.get("source_partition_sha256", {})
        if isinstance(expected_source_hash, Mapping):
            expected = expected_source_hash.get(partition)
            if expected and str(expected) != sha256_file(sample_path):
                raise ValueError(f"{partition} label artifact hash differs from authoritative prior provenance")
        expected_prior_hash = prediction_hash_mapping.get(partition)
        if expected_prior_hash and str(expected_prior_hash) != sha256_file(prior_path):
            prior_hash_matches = False
            raise ValueError(f"{partition} authoritative prior prediction hash mismatch")
        samples = pd.read_parquet(sample_path)
        predictions = pd.read_parquet(prior_path)
        aligned = align_correction_partition(
            samples,
            predictions,
            partition=partition,
            maximum_missing_fraction=config.maximum_missing_alignment_fraction,
        )
        aligned_pieces.append(aligned.aligned)
        if not aligned.mismatches.empty:
            mismatches = aligned.mismatches.copy()
            mismatches["partition"] = partition
            mismatch_pieces.append(mismatches)
        alignment_partitions[partition] = {
            **aligned.report,
            "duplicate_keys": 0,
            "missing_label_rows": int(aligned.report.get("orphan_prior_rows", 0)),
        }
    aligned_frame = pd.concat(aligned_pieces, ignore_index=True)
    mismatches = (
        pd.concat(mismatch_pieces, ignore_index=True)
        if mismatch_pieces
        else pd.DataFrame(columns=["log_id", "timestamp_us", "mismatch_type", "partition"])
    )
    cycles = segment_complete_cycles(aligned_frame, config)
    if cycles.accepted_rows.empty:
        raise ValueError("No complete cycles passed C1 quality thresholds")
    cycle_table, waveform = build_correction_tables(cycles.accepted_rows, config)
    normalization = fit_condition_normalization(cycle_table)
    cycle_table, waveform = apply_condition_normalization(cycle_table, waveform, normalization)

    second_cycle, second_waveform = build_correction_tables(
        cycles.accepted_rows.sample(frac=1.0, random_state=config.random_seed), config
    )
    second_cycle, second_waveform = apply_condition_normalization(
        second_cycle, second_waveform, normalization
    )
    deterministic = (
        table_semantic_hash(cycle_table) == table_semantic_hash(second_cycle)
        and table_semantic_hash(waveform) == table_semantic_hash(second_waveform)
    )
    input_hashes_after = {str(path): sha256_file(path) for path in unique_input_paths}
    inputs_unchanged = input_hashes_before == input_hashes_after
    production_prior_unchanged = all(
        input_hashes_before[str(prior.artifact_root / PRIOR_FILES[partition])]
        == input_hashes_after[str(prior.artifact_root / PRIOR_FILES[partition])]
        for partition in requested
    )
    alignment_report: dict[str, object] = {
        "schema_version": f"{SCHEMA_VERSION}.alignment",
        "key_columns": ["log_id", "timestamp_us"],
        "partitions": alignment_partitions,
        "matched_rows": int(sum(item["aligned_rows"] for item in alignment_partitions.values())),
        "missing_label_rows": int(sum(item["missing_label_rows"] for item in alignment_partitions.values())),
        "missing_prior_rows": int(sum(item["missing_prior_rows"] for item in alignment_partitions.values())),
        "duplicate_keys": 0,
        "test_labels_loaded": False,
        "row_order_used": False,
        "status": "ok",
    }
    quality_checks = _quality_checks(
        prior=prior,
        prior_hash_matches=prior_hash_matches,
        alignment_report=alignment_report,
        cycle_quality=cycles.quality,
        cycle_table=cycle_table,
        waveform=waveform,
        normalization=normalization,
        deterministic=deterministic,
        inputs_unchanged=inputs_unchanged,
        production_prior_unchanged=production_prior_unchanged,
        config=config,
    )
    condition_coverage = _condition_coverage(cycle_table)
    readiness_limitations: list[str] = []
    if contract["physics_dirty"] == "not_recorded_in_prior_manifest":
        readiness_limitations.append("physics_source_dirty_status_not_recorded_upstream")
    outside_training = condition_coverage["validation_outside_training_range"]
    if any(int(value) > 0 for value in outside_training.values()):
        readiness_limitations.append("some_validation_cycles_outside_training_condition_range")
    quality_checks["readiness_limitations"] = readiness_limitations
    if quality_checks["strict_failure_count"]:
        quality_checks["c2_readiness"] = "NOT READY"
    elif readiness_limitations:
        quality_checks["c2_readiness"] = "READY WITH LIMITATIONS"
    else:
        quality_checks["c2_readiness"] = "READY"
    rejection_counts = (
        cycles.rejections["rejection_reason"].value_counts().sort_index().to_dict()
        if not cycles.rejections.empty
        else {}
    )
    dataset_summary: dict[str, object] = {
        "dataset_id": dataset_manifest["dataset_id"],
        "partitions": list(requested),
        "test_labels_loaded": False,
        "aligned_sample_count": int(len(aligned_frame)),
        "waveform_sample_count": int(len(waveform)),
        "total_cycle_count": int(len(cycles.quality)),
        "accepted_cycle_count": int(cycles.quality["accepted"].sum()),
        "rejected_cycle_count": int((~cycles.quality["accepted"]).sum()),
        "rejection_reason_counts": {str(key): int(value) for key, value in rejection_counts.items()},
        "per_partition_cycle_count": {
            str(key): int(value) for key, value in cycle_table.groupby("partition")["cycle_id"].nunique().items()
        },
        "per_log_cycle_count": {
            str(key): int(value) for key, value in cycle_table.groupby("log_id")["cycle_id"].nunique().items()
        },
        "per_date_cycle_count": {
            str(key): int(value) for key, value in cycle_table.groupby("flight_date")["cycle_id"].nunique().items()
        },
        "condition_coverage": condition_coverage,
        "label_uncertainty": "not_available_no_reliable_keyed_sample_uncertainty_artifact",
    }
    git = _git_state(root)
    created_at = datetime.now(timezone.utc)
    artifact_id = f"longitudinal_mean_wb_{created_at.strftime('%Y%m%dT%H%M%SZ')}_{str(git['commit'])[:7]}"
    output_dir = output_parent / artifact_id
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite correction-ready artifact: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    log_ids = _log_ids_from_chain(chain, project_root=root)
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "created_at": created_at.isoformat(),
        "branch": git["branch"],
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "git_dirty_paths": git["dirty_paths"],
        "command": shlex.join(command) if command else "not_recorded",
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "dataset_root": str(dataset),
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_manifest_hash": sha256_file(dataset_manifest_path),
        "label_artifact": {partition: str(dataset / PARTITION_FILES[partition]) for partition in requested},
        "label_schema": ["fx_b", "fz_b"],
        "mass_property_version": {
            "metadata_schema": metadata.get("schema_version", "unknown"),
            "metadata_sha256": sha256_file(metadata_path),
            "mass_status": metadata.get("mass_properties", {}).get("mass_kg", {}).get("status", "unknown"),
        },
        "preprocessing_version": {
            "label_policy": dataset_manifest.get("label_policy", "unknown"),
            "derivative": dataset_manifest.get("derivative", "unknown"),
            "input_filtering": dataset_manifest.get("input_filtering", "unknown"),
        },
        "split_manifest": str(split_path),
        "split_hash": sha256_file(split_path),
        "included_partitions": list(requested),
        "excluded_partitions": ["test"],
        "included_log_ids": sorted(set(log_ids["train"] + log_ids["validation"])),
        "excluded_log_ids": log_ids["test"],
        "test_labels_loaded": False,
        "prior_registry": str(prior.registry_path),
        "prior_artifact_hash": sha256_file(prior.artifact_root / "manifest.json"),
        **contract,
        "prior_artifact_hashes": {
            partition: sha256_file(prior.artifact_root / PRIOR_FILES[partition]) for partition in requested
        },
        "separation_enabled": bool(prior.manifest.get("enable_separation", False)),
        "prior_generator_git": prior.manifest.get("generator_git", "not_recorded"),
        "key_columns": ["log_id", "timestamp_us"],
        "matched_rows": alignment_report["matched_rows"],
        "missing_label_rows": alignment_report["missing_label_rows"],
        "missing_prior_rows": alignment_report["missing_prior_rows"],
        "duplicate_keys": alignment_report["duplicate_keys"],
        "minimum_cycle_samples": config.minimum_cycle_samples,
        "minimum_phase_coverage": config.minimum_phase_coverage_rad,
        "maximum_phase_gap": config.maximum_phase_gap_rad,
        "phase_endpoint_policy": "remove_duplicate_wrapped_zero_endpoints_after_quality_accounting",
        "accepted_cycle_count": dataset_summary["accepted_cycle_count"],
        "rejected_cycle_count": dataset_summary["rejected_cycle_count"],
        "force_components": ["Fx", "Fz"],
        "mean_definition": "arithmetic_sample_mean_per_complete_cycle",
        "waveform_definition": "sample_value_minus_its_complete_cycle_mean",
        "reconstruction_tolerance": config.reconstruction_tolerance,
        "zero_mean_tolerance": config.zero_mean_tolerance,
        "harmonic_max_order": config.harmonic_max_order,
        "centered_basis": True,
        "condition_columns": list(CONDITION_COLUMNS),
        "condition_aggregation": config.condition_aggregation,
        "normalization_source_partition": "train",
        "target_scope": "provisional_effective_longitudinal_force",
        "tail_subtracted": False,
        "body_subtracted": False,
        "moment_in_scope": False,
        "input_hashes_before": input_hashes_before,
        "input_hashes_after": input_hashes_after,
        "semantic_hashes": {
            "cycle_table": table_semantic_hash(cycle_table),
            "waveform_table": table_semantic_hash(waveform),
        },
        "output_format": output_format,
        "run_duration_seconds": float(time.perf_counter() - started_at),
        "c2_readiness": quality_checks["c2_readiness"],
    }
    schema = {
        "schema_version": SCHEMA_VERSION,
        "cycle_table": _schema(cycle_table),
        "waveform_table": _schema(waveform),
        "identity_keys": {"cycle": ["cycle_id"], "waveform": ["cycle_id", "timestamp_us"]},
    }
    weight_contract = {
        "weight_equal_cycle": "1 per cycle row",
        "weight_equal_log": "1 / number_of_accepted_cycles_in_log per cycle row",
        "weight_equal_date": "1 / number_of_accepted_cycles_on_date per cycle row",
        "weight_equal_cycle_sample": "1 / number_of_samples_in_cycle",
        "weight_equal_log_sample": "1 / (accepted_cycles_in_log * samples_in_cycle)",
        "weight_equal_date_sample": "1 / (accepted_cycles_on_date * samples_in_cycle)",
        "selection_status": "weights_generated_only_no_strategy_selected",
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "schema.json", schema)
    write_json(output_dir / "alignment_report.json", alignment_report)
    write_json(output_dir / "normalization.json", normalization)
    write_json(output_dir / "weight_contract.json", weight_contract)
    write_json(output_dir / "dataset_summary.json", dataset_summary)
    write_json(output_dir / "quality_checks.json", quality_checks)
    write_table(output_dir / "alignment_mismatches.csv", mismatches)
    write_table(output_dir / "cycle_quality.csv", cycles.quality)
    write_table(output_dir / "cycle_rejection_reasons.csv", cycles.rejections)
    if output_format in {"parquet", "both"}:
        write_table(output_dir / "cycle_table.parquet", cycle_table)
        write_table(output_dir / "waveform_table.parquet", waveform)
    if output_format in {"csv", "both"}:
        write_table(output_dir / "cycle_table.csv", cycle_table)
        write_table(output_dir / "waveform_table.csv", waveform)
    write_table(output_dir / "cycle_table_preview.csv", cycle_table.head(100))
    write_table(output_dir / "waveform_table_preview.csv", waveform.head(200))
    (output_dir / "run_command.txt").write_text(
        (shlex.join(command) if command else "not_recorded") + "\n", encoding="utf-8"
    )
    exit_code = 0 if quality_checks["strict_failure_count"] == 0 else 2
    return CorrectionReadyArtifactResult(
        output_dir=output_dir,
        manifest=manifest,
        quality_checks=quality_checks,
        dataset_summary=dataset_summary,
        exit_code=exit_code,
    )
