from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from system_identification.artifacts.prior_registry import resolve_delaurier_prior
from system_identification.physics.priors import materialize_authoritative_delaurier_prior


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(
    root: Path,
    *,
    artifact_id: str,
    status: str,
    commit: str,
    partitions: tuple[str, ...] = ("train", "validation"),
) -> None:
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_id": artifact_id,
                "lifecycle_status": status,
                "partitions": list(partitions),
                "test_partition_loaded": False,
                "physics_source": {"commit": commit},
                "contracts": {
                    "frame_contract": "body_frd",
                    "airflow_contract": "attitude_ground_wind_3d",
                    "phase_contract": "phase_v1",
                },
            }
        ),
        encoding="utf-8",
    )


def _write_registry(path: Path, active: Path, legacy: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "delaurier_prior_registry_v1",
                "default_prior_id": "active_v1",
                "priors": {
                    "active_v1": {
                        "lifecycle_status": "active",
                        "artifact_root": str(active),
                        "physics_source_commit": "active_commit",
                        "required_partitions": ["train", "validation"],
                    },
                    "legacy_v0": {
                        "lifecycle_status": "legacy",
                        "artifact_root": str(legacy),
                        "physics_source_commit": "legacy_commit",
                        "superseded_by": "active_v1",
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_registry_defaults_to_active_and_refuses_legacy(tmp_path: Path) -> None:
    active = tmp_path / "active"
    legacy = tmp_path / "legacy"
    _write_manifest(active, artifact_id="active_v1", status="active", commit="active_commit")
    _write_manifest(legacy, artifact_id="legacy_v0", status="legacy", commit="legacy_commit")
    registry = tmp_path / "registry.yaml"
    _write_registry(registry, active, legacy)

    resolved = resolve_delaurier_prior(registry_path=registry)
    assert resolved.prior_id == "active_v1"
    assert resolved.lifecycle_status == "active"
    assert resolved.artifact_root == active.resolve()

    with pytest.raises(ValueError, match="Refusing legacy"):
        resolve_delaurier_prior(prior_id="legacy_v0", registry_path=registry)
    historical = resolve_delaurier_prior(
        prior_id="legacy_v0",
        registry_path=registry,
        allow_legacy=True,
    )
    assert historical.is_legacy


def test_missing_authoritative_prior_never_falls_back_to_legacy(tmp_path: Path) -> None:
    missing = tmp_path / "missing_active"
    legacy = tmp_path / "legacy"
    _write_manifest(legacy, artifact_id="legacy_v0", status="legacy", commit="legacy_commit")
    registry = tmp_path / "registry.yaml"
    _write_registry(registry, missing, legacy)
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        resolve_delaurier_prior(registry_path=registry)


def _canonical_rows(log_id: str, start_timestamp: int, count: int = 16) -> pd.DataFrame:
    phase = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return pd.DataFrame(
        {
            "dataset_id": "synthetic",
            "log_id": log_id,
            "segment_id": "s0",
            "time_s": np.arange(count, dtype=float) * 0.01,
            "timestamp_us": start_timestamp + np.arange(count, dtype=np.int64) * 10_000,
            "mechanical_phase_rad": phase,
            "flap_frequency_hz": np.full(count, 4.0),
            "vehicle_air_data.rho": np.full(count, 1.2),
            "vehicle_attitude.q[0]": np.ones(count),
            "vehicle_attitude.q[1]": np.zeros(count),
            "vehicle_attitude.q[2]": np.zeros(count),
            "vehicle_attitude.q[3]": np.zeros(count),
            "vehicle_local_position.vx": np.full(count, 8.0),
            "vehicle_local_position.vy": np.zeros(count),
            "vehicle_local_position.vz": np.zeros(count),
            "wind.windspeed_north": np.zeros(count),
            "wind.windspeed_east": np.zeros(count),
        }
    )


def test_materializer_writes_keyed_train_validation_without_test(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    train = _canonical_rows("train_log", 1_700_000_000_000_000)
    validation = _canonical_rows("validation_log", 1_700_000_100_000_000)
    test = _canonical_rows("test_log", 1_700_000_200_000_000)
    train.to_parquet(dataset / "train_samples.parquet", index=False)
    validation.to_parquet(dataset / "val_samples.parquet", index=False)
    test.to_parquet(dataset / "test_samples.parquet", index=False)
    (dataset / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "synthetic",
                "wing_transmission_ratio": 8.0,
                "ratio_contract_version": "ratio8_v1",
                "ratio_source": "confirmed_physical_hardware",
                "phase_contract_version": "hall_indexed_mechanical_phase_ratio8_v1",
                "frequency_contract_version": "flap_frequency_ratio8_v1",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "prior"

    manifest = materialize_authoritative_delaurier_prior(
        dataset_root=dataset,
        output_root=output,
        metadata_path=PROJECT_ROOT / "metadata/aircraft/flapper_01/aircraft_metadata.yaml",
        geometry_path=PROJECT_ROOT / "metadata/aircraft/flapper_01/wing_geometry_isaaclab_3b5d4ec.csv",
        partitions=("train", "validation"),
        chunk_size=8,
        project_root=PROJECT_ROOT,
    )

    assert manifest["lifecycle_status"] == "active"
    assert manifest["test_partition_loaded"] is False
    assert manifest["test_rows_loaded"] == 0
    assert manifest["wing_transmission_ratio"] == 8.0
    assert manifest["ratio_contract_version"] == "ratio8_v1"
    assert manifest["contracts"]["airflow_contract"] == "attitude_ground_wind_3d"
    assert manifest["contracts"]["dynamic_twist_contract"] == "disabled_zero_tip_amplitude"
    assert not (output / "test_predictions.parquet").exists()
    for partition, source in (("train", train), ("val", validation)):
        predictions = pd.read_parquet(output / f"{partition}_predictions.parquet")
        assert len(predictions) == len(source)
        assert predictions[["log_id", "timestamp_us"]].equals(source[["log_id", "timestamp_us"]])
        assert np.isfinite(predictions[["fx_b", "fz_b"]].to_numpy()).all()


def test_materializer_rejects_test_partition(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Only train/validation"):
        materialize_authoritative_delaurier_prior(
            dataset_root=tmp_path,
            output_root=tmp_path / "prior",
            metadata_path=tmp_path / "metadata.yaml",
            geometry_path=tmp_path / "geometry.csv",
            partitions=("test",),
            project_root=PROJECT_ROOT,
        )


def test_materializer_rejects_timestamp_unit_mismatch(tmp_path: Path) -> None:
    from system_identification.physics.priors.export import _validate_source_rows

    frame = _canonical_rows("bad_units", 1_700_000_000_000_000)
    frame["timestamp_us"] = frame["timestamp_us"] // 1000
    with pytest.raises(ValueError, match="timestamp_us increments disagree"):
        _validate_source_rows(frame, partition="train")
