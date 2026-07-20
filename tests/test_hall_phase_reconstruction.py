from __future__ import annotations

import numpy as np
import pandas as pd

from system_identification.data.hall_phase import (
    TWO_PI,
    _fallback_reconstruction,
    _hall_reconstruction,
    _rewrite_sample_frame,
    _resample_corrected_frequency,
)


def test_hall_reconstruction_uses_interpolated_zero_and_ratio8() -> None:
    encoder_t = np.arange(0, 800_001, 10_000, dtype=np.int64)
    counts_per_us = 32_768.0 / 250_000.0
    encoder = pd.DataFrame({"timestamp": encoder_t, "total_count": encoder_t * counts_per_us})
    hall = pd.DataFrame(
        {
            "timestamp": [5_000, 5_000, 255_000, 255_000, 505_000, 505_000, 755_000],
            "pulse_count": [1, 1, 2, 2, 3, 3, 4],
        }
    )
    target = np.array([10_000, 130_000, 250_000, 260_000, 380_000, 500_000], dtype=np.int64)
    frequency = np.full(len(target), 4.0)

    rebuilt, quality = _hall_reconstruction(
        encoder=encoder,
        hall_event=hall,
        target_timestamp_us=target,
        frequency_hz=frequency,
        counts_per_encoder_revolution=4096.0,
        true_ratio=8.0,
        maximum_cycle_count_relative_error=0.01,
    )

    expected = np.array([0.02, 0.5, 0.98, 0.02, 0.5, 0.98]) * TWO_PI
    np.testing.assert_allclose(rebuilt["mechanical_phase_rad"], expected, atol=1.0e-12)
    assert rebuilt["cycle_valid"].all()
    assert quality["hall_event_count"] == 4
    assert quality["valid_cycle_count"] == 3


def test_frequency_is_scaled_by_logged_over_true_ratio() -> None:
    source = pd.DataFrame({"timestamp": [0, 10_000, 20_000], "frequency_hz": [4.8, 4.0, 3.2]})
    corrected = _resample_corrected_frequency(
        source,
        np.array([0, 5_000, 20_000], dtype=np.int64),
        logged_ratio=7.5,
        true_ratio=8.0,
    )
    np.testing.assert_allclose(corrected, np.array([4.8, 4.4, 3.2]) * 7.5 / 8.0)


def test_rewritten_table_exports_one_phase_coordinate() -> None:
    source = pd.DataFrame(
        {
            "timestamp_us": [0, 10_000],
            "log_id": ["log", "log"],
            "encoder_phase_rad": [1.0, 2.0],
            "drive_phase_rad": [1.0, 2.0],
            "wing_phase.phase_rad": [1.0, 2.0],
            "phase_raw_rad": [1.0, 2.0],
            "mechanical_phase_rad": [1.0, 2.0],
            "phase_corrected_rad": [1.0, 2.0],
            "flap_frequency_hz": [4.8, 4.8],
            "wing_phase.flap_frequency_hz": [4.8, 4.8],
        }
    )
    rebuilt = pd.DataFrame(
        {
            "timestamp_us": [0, 10_000],
            "mechanical_phase_rad": [0.0, 0.2],
            "flap_frequency_hz": [4.5, 4.5],
            "phase_valid": [True, True],
            "cycle_id": [0, 0],
            "cycle_valid": [True, True],
            "cycle_duration_s": [0.25, 0.25],
            "flap_active": [True, True],
        }
    )
    output = _rewrite_sample_frame(source, rebuilt, dataset_id="rebuilt")
    phase_columns = [column for column in output if column.endswith("phase_rad") or column.endswith("phase_unwrapped_rad")]
    assert phase_columns == ["mechanical_phase_rad"]
    np.testing.assert_allclose(output["flap_frequency_hz"], 4.5)


def test_hall_reconstruction_rejects_bad_count_cycle() -> None:
    encoder = pd.DataFrame(
        {
            "timestamp": [0, 250_000, 500_000],
            "total_count": [0.0, 32_768.0, 70_000.0],
        }
    )
    hall = pd.DataFrame({"timestamp": [0, 250_000, 500_000], "pulse_count": [1, 2, 3]})
    target = np.array([100_000, 300_000], dtype=np.int64)
    rebuilt, quality = _hall_reconstruction(
        encoder=encoder,
        hall_event=hall,
        target_timestamp_us=target,
        frequency_hz=np.array([4.0, 4.0]),
        counts_per_encoder_revolution=4096.0,
        true_ratio=8.0,
        maximum_cycle_count_relative_error=0.01,
    )
    assert rebuilt["cycle_valid"].tolist() == [True, False]
    assert quality["invalid_cycle_count"] == 1


def test_fallback_recovers_hall_zero_count_without_reset_latency() -> None:
    source_t = np.arange(0, 1_010_000, 10_000, dtype=np.int64)
    counts_per_us = 32_768.0 / 250_000.0
    encoder_count = source_t * counts_per_us
    hall_t = np.arange(0, 1_000_001, 250_000, dtype=np.int64)
    reset_publish_t = hall_t + 10_000
    published_zero_index = np.searchsorted(reset_publish_t, source_t, side="right") - 1
    hall_zero_counts = hall_t * counts_per_us
    published_zero_count = np.where(
        published_zero_index >= 0,
        hall_zero_counts[np.clip(published_zero_index, 0, len(hall_zero_counts) - 1)],
        -32_768.0,
    )
    logged_unwrapped = (encoder_count - published_zero_count) * TWO_PI / (4096.0 * 7.5)
    wing_phase = pd.DataFrame(
        {
            "timestamp": source_t,
            "encoder_total_count": encoder_count,
            "phase_unwrapped_rad": logged_unwrapped,
            "phase_valid": np.ones(len(source_t), dtype=bool),
        }
    )
    encoder = pd.DataFrame({"timestamp": source_t, "total_count": encoder_count})
    target = np.arange(0, 1_000_000, 5_000, dtype=np.int64)
    true_phase = np.mod(TWO_PI * 4.0 * target * 1.0e-6, TWO_PI)
    rebuilt, quality = _fallback_reconstruction(
        encoder=encoder,
        wing_phase=wing_phase,
        target_timestamp_us=target,
        frequency_hz=np.full(len(target), 4.0),
        logged_ratio=7.5,
        true_ratio=8.0,
        counts_per_encoder_revolution=4096.0,
        maximum_cycle_count_relative_error=0.01,
    )
    valid = rebuilt["phase_valid"].to_numpy(dtype=bool)
    phase_error = np.angle(
        np.exp(1j * (rebuilt.loc[valid, "mechanical_phase_rad"].to_numpy() - true_phase[valid]))
    )
    np.testing.assert_allclose(phase_error, 0.0, atol=1.0e-12)
    assert quality["phase_method"] == "logged_hall_zero_count_encoder_inversion_fallback"
    assert quality["median_reset_publication_delay_ms"] == 10.0


def test_dataset_builder_rejects_test_partition_before_loading(tmp_path) -> None:
    from system_identification.data.hall_phase import build_hall_ratio8_dataset

    try:
        build_hall_ratio8_dataset(
            source_dataset_root=tmp_path / "missing",
            accepted_logs_csv=tmp_path / "missing.csv",
            output_root=tmp_path / "out",
            partitions=["test"],
        )
    except ValueError as exc:
        assert "test is forbidden" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("test partition was not rejected")
