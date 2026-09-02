from __future__ import annotations

from datetime import datetime

import numpy as np

from system_identification.data.ulg_audit import (
    _event_times,
    classify_maneuvers,
    durations_by_label,
    duration_from_mask,
    parse_log_datetime,
    summarize_timestamps,
)


class _Dataset:
    def __init__(self, data: dict[str, np.ndarray]) -> None:
        self.data = data


def test_parse_log_datetime_supports_both_august_filename_styles() -> None:
    assert parse_log_datetime("log_17_2026-8-20-06-29-34.ulg") == datetime(2026, 8, 20, 6, 29, 34)
    assert parse_log_datetime("2026-08-26_054812_log12.ulg") == datetime(2026, 8, 26, 5, 48, 12)
    assert parse_log_datetime("log_6_2026-4-15-10-48-46.ulg") == datetime(2026, 4, 15, 10, 48, 46)


def test_duration_from_mask_does_not_bridge_invalid_samples_or_large_gaps() -> None:
    timestamps_us = np.array([0, 20_000, 40_000, 60_000, 200_000, 220_000], dtype=np.int64)
    valid = np.array([True, True, False, True, True, True])

    assert duration_from_mask(timestamps_us, valid, max_gap_s=0.05) == 0.04


def test_summarize_timestamps_reports_sampling_gaps_and_duplicates() -> None:
    timestamps_us = np.array([0, 20_000, 20_000, 40_000, 120_000], dtype=np.int64)

    summary = summarize_timestamps(timestamps_us)

    assert summary["sample_count"] == 5
    assert summary["duplicate_count"] == 1
    assert summary["backward_count"] == 0
    assert summary["large_gap_count"] == 1
    assert summary["median_rate_hz"] == 50.0


def test_event_times_falls_back_when_timestamp_sample_is_all_zero() -> None:
    dataset = _Dataset(
        {
            "timestamp": np.array([1_000_000, 2_000_000], dtype=np.int64),
            "timestamp_sample": np.array([0, 0], dtype=np.int64),
        }
    )

    np.testing.assert_array_equal(_event_times(dataset), dataset.data["timestamp"])


def test_classify_maneuvers_uses_exclusive_kinematic_categories() -> None:
    vertical_velocity_ned_m_s = np.array([0.0, -1.0, 1.0, 0.0, 0.6])
    roll_rad = np.deg2rad([0.0, 0.0, 0.0, 25.0, 12.0])
    yaw_rate_rad_s = np.deg2rad([0.0, 0.0, 0.0, 0.0, 12.0])
    ground_speed_m_s = np.full(5, 8.0)

    labels = classify_maneuvers(
        vertical_velocity_ned_m_s,
        roll_rad,
        yaw_rate_rad_s,
        ground_speed_m_s,
    )

    assert labels.tolist() == ["stable_level", "climb", "descent", "turn", "transition"]


def test_durations_by_label_assigns_each_valid_interval_once() -> None:
    timestamps_us = np.array([0, 20_000, 40_000, 60_000], dtype=np.int64)
    valid = np.array([True, True, True, True])
    labels = np.array(["stable", "turn", "turn", "climb"], dtype=object)

    durations = durations_by_label(timestamps_us, valid, labels, max_gap_s=0.05)

    assert durations == {"climb": 0.0, "stable": 0.02, "turn": 0.04}
