from __future__ import annotations

import importlib

import numpy as np
import pandas as pd


MODULE_CASES = (
    (
        "system_identification.phase",
        "system_identification.conventions.phase",
        (
            "annotate_phase_cycles",
            "compute_drive_phase_rad",
            "compute_wing_stroke_angle_rad",
            "compute_wing_stroke_direction",
            "encoder_phase_from_counts",
            "wrap_to_2pi",
        ),
    ),
    (
        "system_identification.resample",
        "system_identification.data.resampling",
        (
            "bin_mean_resample",
            "build_uniform_grid_us",
            "ceil_to_step_us",
            "floor_to_step_us",
            "linear_resample",
            "zoh_resample",
        ),
    ),
    (
        "system_identification.signal_preprocessing",
        "system_identification.data.preprocessing",
        (
            "apply_groupwise_time_shift",
            "existing_group_columns",
            "finite_difference_quality_metrics",
            "groupwise_cubic_spline_derivative",
            "groupwise_lowpass_filter",
            "groupwise_savgol_derivative",
            "highpass_energy_fraction",
            "iter_groups",
            "nominal_sample_rate_hz",
            "odd_window_length",
            "sorted_finite_xy",
        ),
    ),
    (
        "system_identification.dataset_split",
        "system_identification.data.splits",
        (
            "VALID_ROW_COLUMNS",
            "assign_cycle_block_splits",
            "assign_log_splits",
            "build_train_purge_intervals",
            "extract_cycle_blocks",
            "materialize_cycle_block_split",
            "materialize_log_split",
        ),
    ),
)


def test_legacy_and_canonical_modules_export_the_same_public_objects():
    for legacy_name, canonical_name, symbols in MODULE_CASES:
        legacy = importlib.import_module(legacy_name)
        canonical = importlib.import_module(canonical_name)
        for symbol in symbols:
            assert hasattr(legacy, symbol)
            assert hasattr(canonical, symbol)
            assert getattr(legacy, symbol) is getattr(canonical, symbol)


def test_split_wrapper_preserves_repository_private_helper_imports():
    legacy = importlib.import_module("system_identification.dataset_split")
    canonical = importlib.import_module("system_identification.data.splits")

    assert legacy._valid_row_mask is canonical._valid_row_mask
    assert legacy._apply_altitude_window_trim is canonical._apply_altitude_window_trim


def test_phase_legacy_and_canonical_calls_match():
    legacy = importlib.import_module("system_identification.phase")
    canonical = importlib.import_module("system_identification.conventions.phase")
    values = np.array([-0.25, 0.0, 2.0 * np.pi + 0.25])

    np.testing.assert_array_equal(legacy.wrap_to_2pi(values), canonical.wrap_to_2pi(values))


def test_resampling_legacy_and_canonical_calls_match():
    legacy = importlib.import_module("system_identification.resample")
    canonical = importlib.import_module("system_identification.data.resampling")
    source_t = np.array([0, 10_000, 20_000], dtype=np.int64)
    source_v = np.array([0.0, 1.0, 2.0])
    target_t = np.array([5_000, 15_000], dtype=np.int64)

    np.testing.assert_array_equal(
        legacy.linear_resample(source_t, source_v, target_t),
        canonical.linear_resample(source_t, source_v, target_t),
    )


def test_preprocessing_legacy_and_canonical_calls_match():
    legacy = importlib.import_module("system_identification.signal_preprocessing")
    canonical = importlib.import_module("system_identification.data.preprocessing")

    assert legacy.odd_window_length(0.01, 0.12, 100) == canonical.odd_window_length(0.01, 0.12, 100)


def test_split_legacy_and_canonical_calls_match():
    legacy = importlib.import_module("system_identification.dataset_split")
    canonical = importlib.import_module("system_identification.data.splits")
    logs = pd.DataFrame({"log_id": ["a", "b", "c", "d"]})
    kwargs = {"seed": 7, "train_ratio": 0.5, "val_ratio": 0.25, "test_ratio": 0.25}

    pd.testing.assert_frame_equal(legacy.assign_log_splits(logs, **kwargs), canonical.assign_log_splits(logs, **kwargs))
