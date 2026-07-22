"""Fail-closed train-only reader for authoritative C1 correction-ready artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import pandas as pd
import pyarrow.dataset as ds
import yaml

from system_identification.artifacts.io import sha256_file


@dataclass(frozen=True)
class StaticCorrectionTrainingData:
    cycle_frame: pd.DataFrame
    waveform_frame: pd.DataFrame
    normalization: Mapping[str, object]
    provenance: Mapping[str, object]
    component_availability: Mapping[str, object]
    input_hashes: Mapping[str, str]


def _read_mapping(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def _expected(authority: Mapping[str, object], key: str, actual: object) -> None:
    expected = authority.get(key)
    if expected is None:
        raise ValueError(f"Static family authority is missing {key!r}")
    if str(expected) != str(actual):
        raise ValueError(f"Authoritative provenance mismatch for {key}: expected={expected!r}, actual={actual!r}")


def _read_train_table(path: Path) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    if "partition" not in dataset.schema.names:
        raise ValueError(f"Correction-ready table has no partition column: {path}")
    table = dataset.to_table(filter=ds.field("partition") == "train")
    frame = table.to_pandas()
    if len(frame) == 0 or set(frame["partition"].astype(str).unique()) != {"train"}:
        raise ValueError(f"Train-only scan returned invalid partitions for {path}")
    return frame


def _registry_provenance(
    authority: Mapping[str, object], project_root: Path | None
) -> dict[str, object]:
    dataset_registry_value = authority.get("dataset_registry_path")
    prior_registry_value = authority.get("prior_registry_path")
    if dataset_registry_value is None and prior_registry_value is None:
        return {}
    if project_root is None or dataset_registry_value is None or prior_registry_value is None:
        raise ValueError("Registry-backed authority requires project_root and both registry paths")
    dataset_registry_path = (project_root / str(dataset_registry_value)).resolve()
    prior_registry_path = (project_root / str(prior_registry_value)).resolve()
    for path in (dataset_registry_path, prior_registry_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    dataset_registry = yaml.safe_load(dataset_registry_path.read_text(encoding="utf-8"))
    prior_registry = yaml.safe_load(prior_registry_path.read_text(encoding="utf-8"))
    if not isinstance(dataset_registry, Mapping) or not isinstance(prior_registry, Mapping):
        raise ValueError("Authority registries must contain mappings")
    dataset_id = str(authority["dataset_id"])
    prior_id = str(authority["prior_id"])
    if dataset_registry.get("default_dataset_id") != dataset_id:
        raise ValueError("Configured dataset is not the canonical registry default")
    if prior_registry.get("default_prior_id") != prior_id:
        raise ValueError("Configured prior is not the DeLaurier registry default")
    dataset_entry = dataset_registry.get("datasets", {}).get(dataset_id, {})
    prior_entry = prior_registry.get("priors", {}).get(prior_id, {})
    if dataset_entry.get("lifecycle_status") != "active":
        raise ValueError("Canonical dataset registry entry is not active")
    if prior_entry.get("lifecycle_status") != "active":
        raise ValueError("DeLaurier prior registry entry is not active")
    if dataset_entry.get("manifest_sha256") != authority["dataset_manifest_sha256"]:
        raise ValueError("Canonical dataset registry hash does not match C2 authority")
    if prior_entry.get("artifact_manifest_sha256") != authority["prior_manifest_sha256"]:
        raise ValueError("DeLaurier prior registry hash does not match C2 authority")
    if prior_entry.get("dataset_id") != dataset_id:
        raise ValueError("DeLaurier prior registry dataset identity mismatch")
    for registry_key, authority_key in (
        ("ratio_contract_version", "ratio_contract_version"),
        ("phase_contract_version", "phase_contract_version"),
        ("frequency_contract_version", "frequency_contract_version"),
    ):
        if prior_entry.get(registry_key) != authority[authority_key]:
            raise ValueError(f"DeLaurier prior registry {registry_key} mismatch")
    return {
        "dataset_registry_path": str(dataset_registry_path),
        "dataset_registry_hash": sha256_file(dataset_registry_path),
        "dataset_lifecycle_status": dataset_entry["lifecycle_status"],
        "dataset_repository_relative_path": dataset_entry["repository_relative_root"],
        "prior_registry_path": str(prior_registry_path),
        "prior_registry_hash": sha256_file(prior_registry_path),
        "prior_lifecycle_status": prior_entry["lifecycle_status"],
        "prior_physics_source_commit": prior_entry["physics_source_commit"],
    }


def _hash_by_suffix(manifest: Mapping[str, object], suffix: str) -> str:
    hashes = manifest.get("input_hashes_before", {})
    if not isinstance(hashes, Mapping):
        return "not_recorded"
    matches = [str(value) for path, value in hashes.items() if str(path).endswith(suffix)]
    return matches[0] if len(matches) == 1 else "not_recorded"


def load_static_correction_training_data(
    root: str | Path,
    *,
    authority: Mapping[str, object],
    partition: str = "train",
    project_root: str | Path | None = None,
) -> StaticCorrectionTrainingData:
    """Load only logical train rows; validation and test labels are never returned."""

    if partition != "train":
        raise ValueError("C2 fitting is train-only; validation and test requests are forbidden")
    artifact_root = Path(root).resolve()
    registry = _registry_provenance(
        authority,
        None if project_root is None else Path(project_root).resolve(),
    )
    required = [
        artifact_root / "manifest.json",
        artifact_root / "normalization.json",
        artifact_root / "quality_checks.json",
        artifact_root / "cycle_table.parquet",
        artifact_root / "waveform_table.parquet",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Correction-ready artifact is incomplete: {missing}")
    manifest = _read_mapping(required[0])
    normalization = _read_mapping(required[1])
    quality = _read_mapping(required[2])
    manifest_hash = sha256_file(required[0])
    _expected(authority, "correction_ready_manifest_sha256", manifest_hash)
    _expected(authority, "dataset_id", manifest.get("dataset_id"))
    _expected(authority, "dataset_manifest_sha256", manifest.get("dataset_manifest_hash"))
    _expected(authority, "prior_id", manifest.get("resolved_prior_id"))
    _expected(authority, "prior_manifest_sha256", manifest.get("prior_artifact_hash"))
    _expected(authority, "ratio_contract_version", manifest.get("ratio_contract_version"))
    _expected(authority, "phase_contract_version", manifest.get("phase_contract_version"))
    _expected(authority, "frequency_contract_version", manifest.get("frequency_contract_version"))
    if float(manifest.get("wing_transmission_ratio", 0.0)) != 8.0:
        raise ValueError("C2 requires the authoritative wing transmission ratio 8.0")
    if manifest.get("prior_lifecycle_status") != "active":
        raise ValueError("C2 requires an active authoritative prior")
    if bool(manifest.get("git_dirty", True)):
        raise ValueError("Dirty correction-ready artifact provenance is forbidden")
    if bool(manifest.get("test_labels_loaded", True)):
        raise ValueError("Correction-ready artifact reports test labels loaded")
    if "test" in set(str(item) for item in manifest.get("included_partitions", [])):
        raise ValueError("Correction-ready artifact includes test labels")
    if "test" not in set(str(item) for item in manifest.get("excluded_partitions", [])):
        raise ValueError("Correction-ready artifact does not explicitly exclude test")
    checks = quality.get("checks", {})
    if not isinstance(checks, Mapping) or not bool(checks.get("test_label_not_loaded", {}).get("passed", False)):
        raise ValueError("Correction-ready quality checks do not prove test-label isolation")
    if quality.get("strict_failures"):
        raise ValueError(f"Correction-ready artifact has strict failures: {quality['strict_failures']}")
    for name, record in normalization.items():
        if not isinstance(record, Mapping) or record.get("source_partition") != "train":
            raise ValueError(f"Normalization field {name!r} is not frozen from train")

    cycle = _read_train_table(required[3])
    waveform = _read_train_table(required[4])
    if cycle["cycle_id"].duplicated().any():
        raise ValueError("Correction-ready cycle table contains duplicate train cycle_id values")
    if waveform[["cycle_id", "timestamp_us"]].duplicated().any():
        raise ValueError("Correction-ready waveform table contains duplicate stable keys")
    missing_cycles = set(waveform["cycle_id"]) - set(cycle["cycle_id"])
    if missing_cycles:
        raise ValueError("Correction-ready waveform table has missing cycle means")

    component_columns = {
        "prior_fz_normal_component_n",
        "prior_fz_other_component_n",
    }
    component_available = component_columns.issubset(waveform.columns)
    component_availability = {
        "physical_component_scale_fz": "available" if component_available else "unavailable",
        "reason": (
            "authoritative component columns present in correction-ready waveform table"
            if component_available
            else "authoritative C1 and active prior artifacts contain total force only; no reliable row-level component artifact"
        ),
        "required_columns": sorted(component_columns),
        "fabricated": False,
    }
    input_hashes = {str(path): sha256_file(path) for path in required}
    provenance = {
        "correction_ready_artifact_id": manifest["artifact_id"],
        "correction_ready_artifact_path": str(artifact_root),
        "correction_ready_manifest_hash": manifest_hash,
        "dataset_id": manifest["dataset_id"],
        "dataset_hash": manifest["dataset_manifest_hash"],
        "prior_id": manifest["resolved_prior_id"],
        "prior_hash": manifest["prior_artifact_hash"],
        "ratio_contract": manifest["ratio_contract_version"],
        "phase_contract": manifest["phase_contract_version"],
        "frequency_contract": manifest["frequency_contract_version"],
        "ratio_source": manifest.get("ratio_source", "not_recorded"),
        "wing_transmission_ratio": manifest["wing_transmission_ratio"],
        "dataset_root": manifest.get("dataset_root", "not_recorded"),
        "dataset_sample_artifact_hashes": {
            "train": _hash_by_suffix(manifest, "/train_samples.parquet")
        },
        "prior_artifact_path": manifest.get("prior_artifact_path", "not_recorded"),
        "prior_prediction_artifact_hashes": {
            "train": _hash_by_suffix(manifest, "/train_predictions.parquet")
        },
        "prior_lifecycle_status": manifest["prior_lifecycle_status"],
        "physics_source_commit": manifest.get("physics_commit", "not_recorded"),
        "frame_contract": manifest.get("frame_contract", "not_recorded"),
        "airflow_contract": manifest.get("airflow_contract", "not_recorded"),
        "separation_contract": manifest.get("separation_contract", "not_recorded"),
        "dynamic_twist_contract": manifest.get("dynamic_twist_contract", "not_recorded"),
        "prior_partition_coverage": list(manifest.get("prior_partition_coverage", [])),
        "included_partitions": ["train"],
        "validation_labels_loaded": False,
        "test_labels_loaded": False,
        "source_included_partitions": list(manifest.get("included_partitions", [])),
        **registry,
    }
    return StaticCorrectionTrainingData(
        cycle_frame=cycle,
        waveform_frame=waveform,
        normalization=normalization,
        provenance=provenance,
        component_availability=component_availability,
        input_hashes=input_hashes,
    )
