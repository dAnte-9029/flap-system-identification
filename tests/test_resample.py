import numpy as np

from system_identification.resample import (
    build_uniform_grid_us,
    linear_resample,
    zoh_resample,
)


def test_build_uniform_grid_us_includes_start_and_end_points():
    grid = build_uniform_grid_us(start_us=0, end_us=20_000, dt_us=10_000)

    assert grid.tolist() == [0, 10_000, 20_000]


def test_linear_resample_matches_simple_line():
    source_t = np.array([0, 10_000, 20_000], dtype=np.int64)
    source_v = np.array([0.0, 1.0, 2.0])
    target_t = np.array([0, 5_000, 10_000, 15_000, 20_000], dtype=np.int64)

    out = linear_resample(source_t, source_v, target_t)

    assert np.allclose(out, np.array([0.0, 0.5, 1.0, 1.5, 2.0]))


def test_zoh_resample_marks_values_invalid_when_too_old():
    source_t = np.array([0, 20_000], dtype=np.int64)
    source_v = np.array([1.0, 2.0])
    target_t = np.array([0, 5_000, 15_000, 25_000], dtype=np.int64)

    values, age_s, valid = zoh_resample(
        source_t,
        source_v,
        target_t,
        freshness_s=0.01,
    )

    assert np.allclose(values, np.array([1.0, 1.0, 1.0, 2.0]))
    assert np.allclose(age_s, np.array([0.0, 0.005, 0.015, 0.005]))
    assert valid.tolist() == [True, True, False, True]
