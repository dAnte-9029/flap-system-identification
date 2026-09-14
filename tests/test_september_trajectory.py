from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from system_identification.data.september_trajectory import (
    _validate_config, build_september_dataset, cohort_samples, hall_diagnostics,
    observed_phase, publication_hold,
    file_hash, verify_registered_dataset,
)
from system_identification.data.trajectory_dataset import build_window_index


def test_publication_alignment_does_not_backdate_measurements():
    data = {"timestamp": np.array([110, 210]), "timestamp_sample": np.array([90, 190]),
            "value": np.array([1., 2.])}
    values, valid, times = publication_hold([100, 110, 200], data, "value", 1)
    np.testing.assert_array_equal(valid, [False, True, True])
    assert np.isnan(values[0])
    np.testing.assert_array_equal(values[1:], [1, 1])
    np.testing.assert_array_equal(times, [-1, 110, 110])


def test_later_hall_cannot_validate_an_earlier_phase_packet():
    wing = {"timestamp": np.array([100_000, 200_000, 300_000]),
            "phase_rad": np.array([.1, .2, .3]), "phase_valid": np.array([1, 1, 0])}
    hall = {"timestamp": np.array([150_000]), "pulse_count": np.array([1])}
    result = observed_phase([90_000, 180_000, 220_000, 310_000, 500_000], wing, hall)
    assert result.valid_logged_phase.tolist() == [False, True, True, False, False]
    assert result.hall_reference_valid.tolist() == [False, False, True, False, False]
    assert np.isnan(result.logged_flap_phase_rad.iloc[-1])
    assert result.loc[result.hall_reference_valid, "hall_source_timestamp_us"].le(
        result.loc[result.hall_reference_valid, "phase_source_timestamp_us"]).all()


def test_missing_hall_preserves_logged_phase_without_absolute_claim():
    wing = {"timestamp": np.array([100]), "phase_rad": np.array([1.]), "phase_valid": np.array([1])}
    result = observed_phase([100], wing, None)
    assert result.valid_logged_phase.iloc[0]
    assert not result.hall_reference_valid.iloc[0]


def test_cohort_windows_never_bridge_phase_gaps_or_logs():
    frame = pd.DataFrame({"split": ["train"] * 12, "log_id": ["a"] * 8 + ["b"] * 4,
        "timestamp_us": [0,20000,40000,60000,80000,100000,120000,140000,0,20000,40000,60000],
        "valid_core": [True]*12, "valid_logged_phase": [True,True,True,False,True,True,True,True]+[True]*4,
        "hall_reference_valid": [False]*12, "sample_in_log": list(range(8))+list(range(4))})
    chosen = cohort_samples(frame, "logged_phase")
    windows = build_window_index(chosen, horizon_steps=2, stride_steps=1, dt_s=.02)
    assert len(windows) == 5
    for row in windows.itertuples():
        assert not (row.log_id == "a" and row.start_sample_in_log <= 3 <= row.end_sample_in_log)
    assert chosen.loc[~chosen.valid_core, "segment_id"].eq(-1).all()


def test_malformed_timestamp_stream_fails_instead_of_sorting_across_reset():
    with pytest.raises(ValueError, match="non-increasing"):
        publication_hold([100], {"timestamp": [100,90], "x": [1,2]}, "x", 1)


def test_hall_ratio_mismatch_is_not_silently_repaired():
    times = np.arange(0, 500_001, 10_000)
    encoder = {"timestamp": times, "total_count": times * (32768 / 250_000)}
    hall = {"timestamp": np.array([0,250_000,500_000]), "pulse_count": np.array([1,2,3])}
    assert hall_diagnostics(encoder, hall, 8)["status"] == "median_ratio_consistent"
    with pytest.raises(ValueError, match="ratio mismatch"):
        hall_diagnostics(encoder, hall, 7.5)


def test_frozen_day_split_rejects_reassignment():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root/"configs/data/trajectory_september_v2.yaml").read_text())
    assert _validate_config(config)["split_dates"]["sealed_test"] == ["2026-09-08"]
    config["partitions"]["validation"], config["partitions"]["sealed_test"] = (
        config["partitions"]["sealed_test"],config["partitions"]["validation"])
    with pytest.raises(ValueError, match="frozen contract"):
        _validate_config(config)


def test_nonempty_build_output_is_preserved(tmp_path):
    root = Path(__file__).resolve().parents[1]
    marker = tmp_path/"user_file"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        build_september_dataset(root/"configs/data/trajectory_september_v2.yaml",root,tmp_path)
    assert marker.read_text() == "preserve"


@pytest.mark.parametrize("corruption", ["artifact", "manifest", "sealed_test"])
def test_registered_dataset_rejects_corruption_or_test_materialization(tmp_path, corruption):
    artifact = tmp_path / "samples_train.parquet"
    artifact.write_bytes(b"fixture artifact")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"dataset_id": "fixture",
        "split_contract": {"sealed_test_opened": False},
        "artifact_sha256": {artifact.name: file_hash(artifact)}}))
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(yaml.safe_dump({"default_dataset_id": "fixture", "datasets": {
        "fixture": {"manifest_path": manifest_path.name, "manifest_sha256": file_hash(manifest_path)}}}))
    assert verify_registered_dataset(registry_path, tmp_path)["dataset_id"] == "fixture"
    if corruption == "artifact":
        artifact.write_bytes(b"changed")
    elif corruption == "manifest":
        manifest_path.write_text("{}")
    else:
        (tmp_path / "samples_sealed_test.parquet").write_bytes(b"forbidden")
    with pytest.raises(ValueError):
        verify_registered_dataset(registry_path, tmp_path)
