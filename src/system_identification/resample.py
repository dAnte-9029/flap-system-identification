"""Compatibility wrapper for :mod:`system_identification.data.resampling`.

New code should import resampling utilities from the canonical module.
"""

from system_identification.data.resampling import (
    bin_mean_resample,
    build_uniform_grid_us,
    ceil_to_step_us,
    floor_to_step_us,
    linear_resample,
    zoh_resample,
)

__all__ = [
    "bin_mean_resample",
    "build_uniform_grid_us",
    "ceil_to_step_us",
    "floor_to_step_us",
    "linear_resample",
    "zoh_resample",
]
