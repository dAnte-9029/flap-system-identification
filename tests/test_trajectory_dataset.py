from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from system_identification.data.trajectory_dataset import (
    ABSOLUTE_PHASE_COLUMNS,
    CONTROL_COLUMNS,
    FUTURE_FORBIDDEN_INPUT_COLUMNS,
    assign_contiguous_segments,
    build_window_index,
    relative_phase_from_total_count,
    validate_split_assignments,
)


def test_relative_phase_is_log_local_and_does_not_claim_absolute_phase() -> None:
    counts_per_cycle = 4096.0 * 8.0
    total_count = np.array([12_345.0, 12_345.0 + counts_per_cycle / 4.0, 12_345.0 + counts_per_cycle])

    unwrapped, wrapped = relative_phase_from_total_count(total_count, transmission_ratio=8.0)

    np.testing.assert_allclose(unwrapped, [0.0, np.pi / 2.0, 2.0 * np.pi])
    np.testing.assert_allclose(wrapped, [0.0, np.pi / 2.0, 0.0], atol=1.0e-12)
    assert set(ABSOLUTE_PHASE_COLUMNS).isdisjoint(CONTROL_COLUMNS)


def test_contiguous_segments_break_on_invalid_samples_and_time_gaps() -> None:
    timestamps_us = np.array([0, 20_000, 40_000, 60_000, 120_000, 140_000, 160_000])
    valid = np.array([True, True, False, True, True, True, False])

    segments = assign_contiguous_segments(
        timestamps_us,
        valid,
        expected_dt_us=20_000,
        maximum_gap_us=30_000,
    )

    np.testing.assert_array_equal(segments, [0, 0, -1, 1, 2, 2, -1])


def test_window_index_never_crosses_segment_and_uses_one_fewer_control_step() -> None:
    samples = pd.DataFrame(
        {
            "split": ["train"] * 10,
            "log_id": ["a"] * 6 + ["b"] * 4,
            "segment_id": [0] * 6 + [0] * 4,
            "sample_in_segment": list(range(6)) + list(range(4)),
            "timestamp_us": np.arange(10, dtype=np.int64) * 20_000,
        }
    )

    windows = build_window_index(samples, horizon_steps=3, stride_steps=2, dt_s=0.02)

    assert list(windows["log_id"]) == ["a", "a", "b"]
    assert list(windows["start_sample_in_segment"]) == [0, 2, 0]
    assert windows["state_sample_count"].eq(4).all()
    assert windows["control_step_count"].eq(3).all()
    assert windows["horizon_s"].eq(0.06).all()


def test_contract_excludes_future_realized_signals_from_control_sequence() -> None:
    forbidden = {
        "position_ned_m_x",
        "velocity_ned_m_s_x",
        "attitude_q_w",
        "angular_velocity_body_rad_s_x",
        "relative_flap_phase_rad",
        "flap_frequency_hz",
        "true_airspeed_m_s",
        "wind_ned_m_s_n",
    }
    assert forbidden <= set(FUTURE_FORBIDDEN_INPUT_COLUMNS)
    assert set(CONTROL_COLUMNS) == {
        "control_flap_motor_normalized",
        "control_left_elevon_normalized",
        "control_right_elevon_normalized",
        "control_rudder_normalized",
    }
    assert forbidden.isdisjoint(CONTROL_COLUMNS)


def test_split_assignments_require_disjoint_days_and_logs() -> None:
    good = {
        "train": ["batch/log_1_2026-8-19-10-00-00.ulg"],
        "validation": ["batch/log_1_2026-8-20-10-00-00.ulg"],
        "sealed_test": ["batch/2026-08-26_100000_log01.ulg"],
    }
    validated = validate_split_assignments(good)
    assert validated["split_dates"] == {
        "train": ["2026-08-19"],
        "validation": ["2026-08-20"],
        "sealed_test": ["2026-08-26"],
    }

    overlapping = {**good, "validation": good["train"]}
    with pytest.raises(ValueError, match="overlap"):
        validate_split_assignments(overlapping)

    same_day = {**good, "validation": ["batch/log_2_2026-8-19-11-00-00.ulg"]}
    with pytest.raises(ValueError, match="date"):
        validate_split_assignments(same_day)
