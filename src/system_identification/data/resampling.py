from __future__ import annotations

import numpy as np


def build_uniform_grid_us(start_us: int, end_us: int, dt_us: int) -> np.ndarray:
    if end_us < start_us:
        raise ValueError("end_us must be >= start_us")
    return np.arange(start_us, end_us + dt_us, dt_us, dtype=np.int64)


def ceil_to_step_us(value_us: int, step_us: int) -> int:
    return int(((int(value_us) + step_us - 1) // step_us) * step_us)


def floor_to_step_us(value_us: int, step_us: int) -> int:
    return int((int(value_us) // step_us) * step_us)


def linear_resample(source_t_us: np.ndarray, source_v: np.ndarray, target_t_us: np.ndarray) -> np.ndarray:
    source_t = np.asarray(source_t_us, dtype=np.int64)
    source_values = np.asarray(source_v, dtype=float)
    target_t = np.asarray(target_t_us, dtype=np.int64)

    if source_t.size == 0:
        return np.full(target_t.shape, np.nan, dtype=float)

    result = np.interp(target_t.astype(float), source_t.astype(float), source_values)
    outside = (target_t < source_t[0]) | (target_t > source_t[-1])
    result[outside] = np.nan
    return result


def zoh_resample(
    source_t_us: np.ndarray,
    source_v: np.ndarray,
    target_t_us: np.ndarray,
    freshness_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_t = np.asarray(source_t_us, dtype=np.int64)
    source_values = np.asarray(source_v, dtype=float)
    target_t = np.asarray(target_t_us, dtype=np.int64)

    values = np.full(target_t.shape, np.nan, dtype=float)
    age_s = np.full(target_t.shape, np.inf, dtype=float)
    valid = np.zeros(target_t.shape, dtype=bool)

    if source_t.size == 0:
        return values, age_s, valid

    idx = np.searchsorted(source_t, target_t, side="right") - 1
    ok = idx >= 0

    values[ok] = source_values[idx[ok]]
    age_s[ok] = (target_t[ok] - source_t[idx[ok]]) * 1e-6
    valid = ok & (age_s <= freshness_s)
    return values, age_s, valid


def bin_mean_resample(
    source_t_us: np.ndarray,
    source_v: np.ndarray,
    grid_t_us: np.ndarray,
    dt_us: int,
) -> np.ndarray:
    source_t = np.asarray(source_t_us, dtype=np.int64)
    source_values = np.asarray(source_v, dtype=float)
    grid = np.asarray(grid_t_us, dtype=np.int64)
    out = np.full(grid.shape, np.nan, dtype=float)

    if source_t.size == 0:
        return out

    for i, start in enumerate(grid):
        end = start + dt_us
        mask = (source_t >= start) & (source_t < end)
        if np.any(mask):
            values = source_values[mask]
            finite = values[np.isfinite(values)]
            if finite.size:
                out[i] = float(np.mean(finite))

    return out
