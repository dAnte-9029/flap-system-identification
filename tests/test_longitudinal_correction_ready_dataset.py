from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from system_identification.artifacts.io import sha256_file
from system_identification.data.correction_ready import (
    CONDITION_COLUMNS,
    CorrectionReadyConfig,
    STANDARDIZED_CONDITION_COLUMNS,
    align_correction_partition,
    apply_condition_normalization,
    build_correction_tables,
    fit_condition_normalization,
    segment_complete_cycles,
    table_semantic_hash,
)


def _samples(*, partition: str = "train", log_id: str = "log_0_2026-4-14-10-00-00") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    timestamp = 1_000_000
    for source_cycle in range(3):
        phase = np.linspace(0.0, 2.0 * np.pi, 17)
        for sample_index, phi in enumerate(phase):
            rows.append(
                {
                    "dataset_id": "synthetic-ratio8",
                    "partition": partition,
                    "split": "val" if partition == "validation" else partition,
                    "log_id": log_id,
                    "segment_id": 0,
                    "cycle_id": source_cycle,
                    "timestamp_us": timestamp,
                    "time_s": timestamp * 1.0e-6,
                    "phase_corrected_rad": phi,
                    "cycle_flap_frequency_hz": 5.0 + 0.1 * source_cycle,
                    "airspeed_validated.true_airspeed_m_s": 8.0 + source_cycle,
                    "vehicle_air_data.rho": 1.18,
                    "condition_alpha_rad": 0.1 + 0.01 * source_cycle,
                    "fx_b": 2.0 + source_cycle + np.sin(phi),
                    "fz_b": -3.0 + 0.5 * source_cycle + 2.0 * np.cos(phi),
                }
            )
            timestamp += 10_000
        timestamp += 50_000
    return pd.DataFrame(rows)


def _prior(samples: pd.DataFrame) -> pd.DataFrame:
    phase = samples["phase_corrected_rad"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "log_id": samples["log_id"].to_numpy(),
            "timestamp_us": samples["timestamp_us"].to_numpy(),
            "fx_b": 0.5 + 0.25 * np.sin(phase),
            "fz_b": -1.0 + 0.5 * np.cos(phase),
        }
    )


def _prepared(partition: str = "train") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = _samples(partition=partition)
    aligned = align_correction_partition(samples, _prior(samples), partition=partition)
    selection = segment_complete_cycles(
        aligned.aligned,
        CorrectionReadyConfig(minimum_cycle_samples=12, maximum_phase_gap_rad=0.8),
    )
    return aligned.aligned, selection.accepted_rows, selection.quality


def test_keyed_alignment_is_row_order_independent() -> None:
    samples = _samples()
    prior = _prior(samples)
    first = align_correction_partition(samples, prior, partition="train").aligned
    shuffled = align_correction_partition(
        samples.sample(frac=1.0, random_state=3),
        prior.sample(frac=1.0, random_state=7),
        partition="train",
    ).aligned
    keys = ["log_id", "timestamp_us"]
    columns = [*keys, "label_fx_b", "prior_fx_b", "residual_fx_b"]
    pd.testing.assert_frame_equal(
        first[columns].sort_values(keys).reset_index(drop=True),
        shuffled[columns].sort_values(keys).reset_index(drop=True),
    )


def test_alignment_duplicate_missing_partition_and_timestamp_failures() -> None:
    samples = _samples()
    prior = _prior(samples)
    with pytest.raises(ValueError, match="Duplicate alignment keys"):
        align_correction_partition(samples, pd.concat([prior, prior.iloc[[0]]]), partition="train")
    missing = align_correction_partition(
        samples,
        prior.iloc[1:],
        partition="train",
        maximum_missing_fraction=0.1,
    )
    assert missing.report["missing_prior_rows"] == 1
    assert set(missing.mismatches["mismatch_type"]) == {"missing_prior"}
    wrong_partition = samples.assign(split="validation")
    with pytest.raises(ValueError, match="partition identity conflict"):
        align_correction_partition(wrong_partition, prior, partition="train")
    seconds = prior.assign(timestamp_us=prior["timestamp_us"] // 1_000_000)
    with pytest.raises(ValueError, match="timestamp unit mismatch"):
        align_correction_partition(samples, seconds, partition="train")


def test_cycle_identity_quality_and_shuffle_are_deterministic() -> None:
    aligned, accepted, quality = _prepared()
    assert accepted.groupby("correction_cycle_id")["log_id"].nunique().max() == 1
    assert accepted.groupby("correction_cycle_id")["partition"].nunique().max() == 1
    assert quality["accepted"].all()
    shuffled = segment_complete_cycles(
        aligned.sample(frac=1.0, random_state=17),
        CorrectionReadyConfig(minimum_cycle_samples=12, maximum_phase_gap_rad=0.8),
    )
    assert set(accepted["correction_cycle_id"]) == set(shuffled.accepted_rows["correction_cycle_id"])


def test_incomplete_cycle_is_reported_and_repeated_endpoint_removed() -> None:
    samples = _samples()
    prior = _prior(samples)
    incomplete_mask = ~((samples["cycle_id"] == 2) & (samples["phase_corrected_rad"] > np.pi))
    aligned = align_correction_partition(
        samples.loc[incomplete_mask],
        prior.loc[incomplete_mask],
        partition="train",
    )
    selected = segment_complete_cycles(
        aligned.aligned,
        CorrectionReadyConfig(minimum_cycle_samples=12, maximum_phase_gap_rad=0.8),
    )
    assert (~selected.quality["accepted"]).any()
    assert selected.rejections["rejection_reason"].str.contains("incomplete_phase_coverage").any()
    accepted_counts = selected.accepted_rows.groupby("correction_cycle_id").size()
    assert (accepted_counts == 16).all()


def test_mean_waveform_harmonic_basis_and_weights_contract() -> None:
    _, accepted, _ = _prepared()
    cycle_table, waveform = build_correction_tables(accepted, CorrectionReadyConfig())
    for component in ("fx", "fz"):
        reconstructed = waveform[f"label_{component}_mean_n"] + waveform[f"label_{component}_waveform_n"]
        np.testing.assert_allclose(reconstructed, waveform[f"label_{component}_n"], atol=1e-12)
        residual_reconstructed = (
            waveform[f"residual_{component}_mean_n"]
            + waveform[f"residual_{component}_waveform_n"]
        )
        np.testing.assert_allclose(residual_reconstructed, waveform[f"residual_{component}_n"], atol=1e-12)
        grouped = waveform.groupby("cycle_id")
        assert grouped[f"label_{component}_waveform_n"].mean().abs().max() < 1e-12
        assert grouped[f"prior_{component}_waveform_n"].mean().abs().max() < 1e-12
        assert grouped[f"residual_{component}_waveform_n"].mean().abs().max() < 1e-12
    for order in range(1, 5):
        for trig in ("sin", "cos"):
            assert f"{trig}_{order}_phase" in waveform
            centered = f"{trig}_{order}_phase_centered"
            assert waveform.groupby("cycle_id")[centered].mean().abs().max() < 1e-12
    assert np.allclose(waveform.groupby("cycle_id")["weight_equal_cycle_sample"].sum(), 1.0)
    assert np.allclose(waveform.groupby("log_id")["weight_equal_log_sample"].sum(), 1.0)
    assert np.allclose(waveform.groupby("flight_date")["weight_equal_date_sample"].sum(), 1.0)
    assert np.isfinite(cycle_table.select_dtypes(include=[np.number])).all().all()


def test_partition_aware_weights_isolate_train_from_validation() -> None:
    _, train_rows, _ = _prepared("train")
    _, validation_rows, _ = _prepared("validation")
    _, train_waveform = build_correction_tables(train_rows, CorrectionReadyConfig())
    combined_cycles, combined_waveform = build_correction_tables(
        pd.concat([train_rows, validation_rows], ignore_index=True), CorrectionReadyConfig()
    )
    train_only = combined_waveform.loc[combined_waveform["partition"] == "train"]
    np.testing.assert_allclose(
        train_waveform["weight_equal_log_sample"], train_only["weight_equal_log_sample"]
    )
    assert np.allclose(
        combined_waveform.groupby(["partition", "log_id"])["weight_equal_log_sample"].sum(),
        1.0,
    )
    assert np.allclose(
        combined_waveform.groupby(["partition", "flight_date"])["weight_equal_date_sample"].sum(),
        1.0,
    )
    assert np.allclose(
        combined_cycles.groupby(["partition", "log_id"])["weight_equal_log"].sum(), 1.0
    )


def test_negative_airspeed_is_preserved_and_marked_invalid() -> None:
    samples = _samples()
    samples["airspeed_validated.true_airspeed_m_s"] = -2.5
    aligned = align_correction_partition(samples, _prior(samples), partition="train")
    selection = segment_complete_cycles(aligned.aligned, CorrectionReadyConfig())
    cycles, waveform = build_correction_tables(selection.accepted_rows, CorrectionReadyConfig())
    assert (cycles["airspeed_mean_mps"] == -2.5).all()
    assert (waveform["airspeed_mean_mps"] == -2.5).all()
    assert (cycles["airspeed_min_mps"] == -2.5).all()
    assert (cycles["airspeed_negative_fraction"] == 1.0).all()
    assert not cycles["airspeed_condition_valid"].any()
    assert not cycles["dynamic_pressure_condition_valid"].any()


def test_phase_wrap_and_nonuniform_sampling_keep_centered_basis_zero() -> None:
    _, accepted, _ = _prepared()
    accepted = accepted.loc[accepted.groupby("correction_cycle_id").cumcount() % 3 != 0].copy()
    accepted["phase_corrected_rad"] += 4.0 * np.pi
    _, waveform = build_correction_tables(accepted, CorrectionReadyConfig(minimum_cycle_samples=4))
    centered = [column for column in waveform if column.endswith("_phase_centered")]
    assert waveform.groupby("cycle_id")[centered].mean().abs().to_numpy().max() < 1e-12


def test_normalization_uses_training_only_and_handles_zero_variance() -> None:
    _, train_rows, _ = _prepared("train")
    _, val_rows, _ = _prepared("validation")
    train_cycles, train_waveform = build_correction_tables(train_rows, CorrectionReadyConfig())
    val_cycles, val_waveform = build_correction_tables(val_rows, CorrectionReadyConfig())
    val_cycles.loc[:, "airspeed_mean_mps"] = 10_000.0
    stats = fit_condition_normalization(train_cycles)
    transformed_cycle, transformed_waveform = apply_condition_normalization(
        val_cycles,
        val_waveform,
        stats,
    )
    assert stats["airspeed_mean_mps"]["mean"] != 10_000.0
    assert stats["rho_mean_kgpm3"]["zero_variance"] is True
    assert stats["rho_mean_kgpm3"]["std"] == 1.0
    standardized = [STANDARDIZED_CONDITION_COLUMNS[column] for column in CONDITION_COLUMNS]
    assert np.isfinite(transformed_cycle[standardized]).all().all()
    assert len(transformed_waveform) == len(val_waveform)
    assert json.loads(json.dumps(stats)) == stats


def test_semantic_hash_and_parquet_round_trip_are_deterministic(tmp_path: Path) -> None:
    _, accepted, _ = _prepared()
    cycle_table, waveform = build_correction_tables(accepted, CorrectionReadyConfig())
    assert table_semantic_hash(cycle_table) == table_semantic_hash(
        cycle_table.sample(frac=1.0, random_state=9)
    )
    path = tmp_path / "waveform.parquet"
    waveform.to_parquet(path, index=False)
    restored = pd.read_parquet(path)
    pd.testing.assert_frame_equal(waveform, restored)


def test_cli_rejects_test_partition() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_longitudinal_correction_ready_dataset.py",
            "--dataset-root",
            "missing",
            "--split-manifest",
            "missing.json",
            "--partitions",
            "test",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "test" in result.stderr.lower()


def test_cli_rejects_prior_root_and_prior_id_together() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_longitudinal_correction_ready_dataset.py",
            "--dataset-root",
            "missing",
            "--split-manifest",
            "missing.json",
            "--prior-root",
            "missing-prior",
            "--prior-id",
            "also-missing",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "at most one" in result.stderr.lower()


def _artifact_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset = tmp_path / "dataset"
    prior_root = tmp_path / "prior"
    dataset.mkdir()
    prior_root.mkdir()
    metadata = tmp_path / "aircraft_metadata.yaml"
    metadata.write_text(
        yaml.safe_dump(
            {
                "schema_version": "aircraft_metadata_v0.1",
                "frames": {"body_frame": "FRD", "local_frame": "NED"},
                "mass_properties": {"mass_kg": {"status": "synthetic_measured_v1"}},
                "label_definition": {
                    "force_definition": "effective_non_gravity_external_force"
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "dataset_id": "synthetic-ratio8",
        "wing_transmission_ratio": 8.0,
        "ratio_contract_version": "ratio8_v1",
        "ratio_source": "confirmed_physical_hardware",
        "phase_contract_version": "hall_indexed_mechanical_phase_ratio8_v1",
        "frequency_contract_version": "flap_frequency_ratio8_v1",
        "preprocessing_version": {
            "label_policy": "synthetic reconstructed effective force",
            "derivative": {"method": "analytic"},
            "input_filtering": {"enabled": False},
        },
        "metadata_path": str(metadata),
        "label_policy": "synthetic reconstructed effective force",
        "derivative": {"method": "analytic"},
        "input_filtering": {"enabled": False},
        "split_sample_counts": {},
    }
    manifest_path = dataset / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    source_hashes: dict[str, str] = {}
    prediction_hashes: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    for partition, sample_name, prior_name, log_id in (
        ("train", "train_samples.parquet", "train_predictions.parquet", "log_0_2026-4-14-10-00-00"),
        ("validation", "val_samples.parquet", "val_predictions.parquet", "log_1_2026-4-15-10-00-00"),
    ):
        samples = _samples(partition=partition, log_id=log_id)
        keep = ~((samples["cycle_id"] == 2) & (samples["phase_corrected_rad"] > np.pi))
        samples = samples.loc[keep].reset_index(drop=True)
        predictions = _prior(samples)
        sample_path = dataset / sample_name
        prediction_path = prior_root / prior_name
        samples.to_parquet(sample_path, index=False)
        predictions.to_parquet(prediction_path, index=False)
        source_hashes[partition] = sha256_file(sample_path)
        prediction_hashes[partition] = sha256_file(prediction_path)
        row_counts[partition] = len(samples)
    prior_manifest = {
        "schema_version": "delaurier_keyed_prior_v2",
        "artifact_id": "synthetic-active-prior",
        "lifecycle_status": "active",
        "partitions": ["train", "validation"],
        "row_counts": row_counts,
        "test_partition_loaded": False,
        "wing_transmission_ratio": 8.0,
        "ratio_contract_version": "ratio8_v1",
        "ratio_source": "confirmed_physical_hardware",
        "phase_contract_version": "hall_indexed_mechanical_phase_ratio8_v1",
        "frequency_contract_version": "flap_frequency_ratio8_v1",
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "source_partition_sha256": source_hashes,
        "prediction_sha256": prediction_hashes,
        "physics_source": {
            "repository": "https://example.invalid/physics",
            "commit": "abc123",
            "dirty": False,
        },
        "contracts": {
            "frame_contract": "body_frd_force_at_imu_origin_moment_about_cg",
            "phase_contract": "canonical_mechanical_phase_to_delaurier_v1",
            "airflow_contract": "attitude_ground_wind_3d",
            "dynamic_twist_contract": "disabled_zero_tip_amplitude",
            "separation_contract": "disabled_attached_flow",
        },
    }
    (prior_root / "manifest.json").write_text(
        json.dumps(prior_manifest, sort_keys=True), encoding="utf-8"
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "schema_version": "delaurier_prior_registry_v1",
                "default_prior_id": "synthetic-active-prior",
                "priors": {
                    "synthetic-active-prior": {
                        "lifecycle_status": "active",
                        "artifact_root": str(prior_root),
                        "physics_source_commit": "abc123",
                        "frame_contract": "body_frd_force_at_imu_origin_moment_about_cg",
                        "phase_contract": "canonical_mechanical_phase_to_delaurier_v1",
                        "airflow_contract": "attitude_ground_wind_3d",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return dataset, manifest_path, registry


def test_cli_headless_artifact_schema_and_strict_exit(tmp_path: Path) -> None:
    dataset, split_manifest, registry = _artifact_fixture(tmp_path)
    output = tmp_path / "output"
    command = [
        sys.executable,
        "scripts/build_longitudinal_correction_ready_dataset.py",
        "--dataset-root",
        str(dataset),
        "--split-manifest",
        str(split_manifest),
        "--prior-registry",
        str(registry),
        "--partitions",
        "train",
        "validation",
        "--output-root",
        str(output),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    artifact = next(output.iterdir())
    required = {
        "manifest.json",
        "schema.json",
        "alignment_report.json",
        "alignment_mismatches.csv",
        "cycle_quality.csv",
        "cycle_rejection_reasons.csv",
        "cycle_table.parquet",
        "waveform_table.parquet",
        "normalization.json",
        "weight_contract.json",
        "dataset_summary.json",
        "quality_checks.json",
        "run_command.txt",
    }
    assert required.issubset({path.name for path in artifact.iterdir()})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((artifact / "schema.json").read_text(encoding="utf-8"))
    assert manifest["resolved_prior_id"] == "synthetic-active-prior"
    assert manifest["test_labels_loaded"] is False
    assert manifest["target_scope"] == "provisional_effective_longitudinal_force"
    assert manifest["preprocessing_version"]["derivative"] == {"method": "analytic"}
    assert {"cycle_table", "waveform_table", "identity_keys"}.issubset(schema)

    strict_config = yaml.safe_load(
        Path("configs/correction/longitudinal_force_correction_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    strict_config["cycle"]["minimum_accepted_cycle_fraction"] = 0.9
    strict_path = tmp_path / "strict.yaml"
    strict_path.write_text(yaml.safe_dump(strict_config), encoding="utf-8")
    strict_result = subprocess.run(
        [*command, "--output-root", str(tmp_path / "strict-output"), "--config", str(strict_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert strict_result.returncode == 2
    assert "STRICT QUALITY FAILURE" in strict_result.stderr
