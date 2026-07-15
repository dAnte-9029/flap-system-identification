"""Compatibility wrapper for :mod:`system_identification.conventions.phase`.

New code should import phase conventions from the canonical module.
"""

from system_identification.conventions.phase import (
    annotate_phase_cycles,
    compute_drive_phase_rad,
    compute_wing_stroke_angle_rad,
    compute_wing_stroke_direction,
    encoder_phase_from_counts,
    wrap_to_2pi,
)

__all__ = [
    "annotate_phase_cycles",
    "compute_drive_phase_rad",
    "compute_wing_stroke_angle_rad",
    "compute_wing_stroke_direction",
    "encoder_phase_from_counts",
    "wrap_to_2pi",
]
