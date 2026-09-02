from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.build_multibody_wrench_dataset import build_multibody_wrench_dataset


DATASET_ID = "canonical_v0.4_training_ready_split_measured_massprops_ratio8_phasefix_v3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(partition: str, frequency_hz: float) -> pd.DataFrame:
    time_s = np.arange(31, dtype=float) * 0.01
    phase = np.mod(2.0 * np.pi * frequency_hz * time_s, 2.0 * np.pi)
    return pd.DataFrame(
        {
            "timestamp_us": np.rint(time_s * 1e6).astype(np.int64),
            "time_s": time_s,
            "log_id": [f"log_{partition}"] * len(time_s),
            "segment_id": 0,
            "split": partition,
            "dataset_id": DATASET_ID,
            "mechanical_phase_rad": phase,
            "flap_frequency_hz": frequency_hz,
            "phase_valid": True,
            "label_reconstruction_valid": True,
            "vehicle_local_position.ax_smooth": 0.0,
            "vehicle_local_position.ay_smooth": 0.0,
            "vehicle_local_position.az_smooth": 9.81,
            "vehicle_angular_velocity.xyz[0]": 0.0,
            "vehicle_angular_velocity.xyz[1]": 0.0,
            "vehicle_angular_velocity.xyz[2]": 0.0,
            "vehicle_angular_velocity.xyz_derivative_smooth[0]": 0.0,
            "vehicle_angular_velocity.xyz_derivative_smooth[1]": 0.0,
            "vehicle_angular_velocity.xyz_derivative_smooth[2]": 0.0,
            "vehicle_attitude.q[0]": 1.0,
            "vehicle_attitude.q[1]": 0.0,
            "vehicle_attitude.q[2]": 0.0,
            "vehicle_attitude.q[3]": 0.0,
            "fx_b": 1.0,
            "fy_b": 2.0,
            "fz_b": 3.0,
            "mx_b": 4.0,
            "my_b": 5.0,
            "mz_b": 6.0,
            "label_valid": True,
            "label_variant": "rigid_v01",
        }
    )


def _source_registry(tmp_path: Path) -> Path:
    source = tmp_path / DATASET_ID
    source.mkdir()
    train_path = source / "train_samples.parquet"
    val_path = source / "val_samples.parquet"
    _frame("train", 4.0).to_parquet(train_path, index=False)
    _frame("validation", 3.0).to_parquet(val_path, index=False)
    (source / "test_samples.parquet").write_text("must not be read", encoding="utf-8")
    manifest_path = source / "dataset_manifest.json"
    manifest_path.write_text(json.dumps({"dataset_id": DATASET_ID}), encoding="utf-8")
    registry = {
        "default_dataset_id": DATASET_ID,
        "datasets": {
            DATASET_ID: {
                "lifecycle_status": "active",
                "repository_relative_root": str(source),
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "artifact_sha256": {
                    "train_samples.parquet": _sha256(train_path),
                    "val_samples.parquet": _sha256(val_path),
                },
                "phase_contract": {"column": "mechanical_phase_rad"},
                "frequency_contract": {"column": "flap_frequency_hz"},
            }
        },
    }
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    return registry_path


def test_builder_relabels_only_train_validation_and_preserves_keys(tmp_path: Path):
    registry_path = _source_registry(tmp_path)
    output = tmp_path / "candidate"
    result = build_multibody_wrench_dataset(
        registry_path=registry_path,
        source_dataset_id=DATASET_ID,
        config_path=Path("configs/data/multibody_wrench_label_v1.yaml"),
        output_root=output,
    )

    assert result["manifest_path"] == output / "dataset_manifest.json"
    assert not (output / "test_samples.parquet").exists()
    source = tmp_path / DATASET_ID
    for filename in ("train_samples.parquet", "val_samples.parquet"):
        before = pd.read_parquet(source / filename)
        after = pd.read_parquet(output / filename)
        pd.testing.assert_frame_equal(after[["log_id", "timestamp_us"]], before[["log_id", "timestamp_us"]])
        np.testing.assert_allclose(after["fx_b_rigid_v01"], 1.0)
        np.testing.assert_allclose(after["fx_b"], after["fx_b_multibody"])
        assert after["dataset_id"].eq("candidate").all()
        assert after["label_variant"].eq("measured_three_body_fixed_neutral_com_v1").all()

    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["included_partitions"] == ["train", "validation"]
    assert manifest["excluded_partitions"] == ["test"]
    assert manifest["test_labels_loaded"] is False


def test_builder_refuses_to_overwrite_existing_output(tmp_path: Path):
    registry_path = _source_registry(tmp_path)
    output = tmp_path / "candidate"
    output.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_multibody_wrench_dataset(
            registry_path=registry_path,
            source_dataset_id=DATASET_ID,
            config_path=Path("configs/data/multibody_wrench_label_v1.yaml"),
            output_root=output,
        )
