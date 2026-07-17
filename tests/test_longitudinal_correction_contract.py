from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from system_identification.artifacts.prior_registry import resolve_delaurier_prior
from system_identification.data.correction_ready import (
    normalize_correction_partitions,
    validate_correction_contract,
)


def _write_prior(
    tmp_path: Path,
    *,
    lifecycle: str = "active",
    partitions: tuple[str, ...] = ("train", "validation"),
    frame_contract: str = "body_frd_force_at_imu_origin_moment_about_cg",
    artifact_id: str = "active-prior",
) -> tuple[Path, Path]:
    artifact = tmp_path / artifact_id
    artifact.mkdir()
    manifest = {
        "artifact_id": artifact_id,
        "lifecycle_status": lifecycle,
        "partitions": list(partitions),
        "test_partition_loaded": False,
        "physics_source": {
            "repository": "https://example.invalid/physics",
            "commit": "abc123",
        },
        "contracts": {
            "frame_contract": frame_contract,
            "phase_contract": "canonical_mechanical_phase_to_delaurier_v1",
            "airflow_contract": "attitude_ground_wind_3d",
            "dynamic_twist_contract": "disabled_zero_tip_amplitude",
            "separation_contract": "disabled_attached_flow",
        },
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "schema_version": "delaurier_prior_registry_v1",
                "default_prior_id": artifact_id,
                "priors": {
                    artifact_id: {
                        "lifecycle_status": lifecycle,
                        "artifact_root": str(artifact),
                        "physics_source_commit": "abc123",
                        "frame_contract": frame_contract,
                        "phase_contract": "canonical_mechanical_phase_to_delaurier_v1",
                        "airflow_contract": "attitude_ground_wind_3d",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return registry, artifact


def _dataset_contract() -> tuple[dict[str, object], dict[str, object]]:
    manifest = {
        "dataset_id": "synthetic-ratio8",
        "label_policy": "effective wrench recomputed from pre-smoothed kinematic derivatives",
        "derivative": {"method": "savgol", "window_s": 0.03, "polyorder": 3},
        "split_sample_counts": {"train": 20, "val": 20, "test": 10},
    }
    metadata = {
        "schema_version": "aircraft_metadata_v0.1",
        "frames": {"body_frame": "FRD", "local_frame": "NED"},
        "mass_properties": {"mass_kg": {"status": "measured-v1"}},
        "flapping_drive": {
            "encoder_to_drive_ratio": {"value": 8.0},
            "drive_phase_zero_definition": "sine_argument_zero_crossing",
        },
        "label_definition": {"force_definition": "effective_non_gravity_external_force"},
    }
    return manifest, metadata


def test_default_registry_resolves_active_authoritative_prior() -> None:
    resolution = resolve_delaurier_prior(requested_partitions=("train", "validation"))
    assert resolution.prior_id == "delaurier_attitude_aware_3b5d4ec_trainval_v1"
    assert resolution.lifecycle_status == "active"
    assert resolution.physics_source_commit == "3b5d4ec1d28f1384cf042402992ad7ea59995f49"


def test_legacy_prior_is_not_silently_used(tmp_path: Path) -> None:
    registry, _ = _write_prior(tmp_path, lifecycle="legacy", artifact_id="legacy-prior")
    with pytest.raises(ValueError, match="Refusing legacy"):
        resolve_delaurier_prior(registry_path=registry)


def test_missing_active_prior_fails_without_fallback(tmp_path: Path) -> None:
    registry, artifact = _write_prior(tmp_path)
    (artifact / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        resolve_delaurier_prior(registry_path=registry)


def test_prior_contract_mismatch_fails(tmp_path: Path) -> None:
    registry, _ = _write_prior(tmp_path, frame_contract="world_ned")
    resolution = resolve_delaurier_prior(registry_path=registry)
    dataset_manifest, metadata = _dataset_contract()
    with pytest.raises(ValueError, match="frame contract mismatch"):
        validate_correction_contract(resolution, dataset_manifest, metadata)


def test_test_window_prior_is_rejected(tmp_path: Path) -> None:
    registry, artifact = _write_prior(tmp_path)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["test_partition_loaded"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="test_partition_loaded"):
        resolve_delaurier_prior(registry_path=registry)


def test_manifest_contract_records_real_resolved_prior(tmp_path: Path) -> None:
    registry, artifact = _write_prior(tmp_path)
    resolution = resolve_delaurier_prior(registry_path=registry)
    dataset_manifest, metadata = _dataset_contract()
    contract = validate_correction_contract(resolution, dataset_manifest, metadata)
    assert contract["resolved_prior_id"] == "active-prior"
    assert contract["prior_artifact_path"] == str(artifact.resolve())
    assert contract["frame_contract"] == "body_frd_force_at_imu_origin_moment_about_cg"


def test_partition_contract_refuses_test() -> None:
    with pytest.raises(ValueError, match="test"):
        normalize_correction_partitions(("train", "test"))
    assert normalize_correction_partitions(("train", "validation")) == ("train", "validation")
