"""Compatibility wrapper for :mod:`system_identification.data.preprocessing`.

New code should import preprocessing utilities from the canonical module.
"""

from system_identification.data.preprocessing import (
    apply_groupwise_time_shift,
    existing_group_columns,
    finite_difference_quality_metrics,
    groupwise_cubic_spline_derivative,
    groupwise_lowpass_filter,
    groupwise_savgol_derivative,
    highpass_energy_fraction,
    iter_groups,
    nominal_sample_rate_hz,
    odd_window_length,
    sorted_finite_xy,
)

__all__ = [
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
]
