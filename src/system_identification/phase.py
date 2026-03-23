from __future__ import annotations

import numpy as np


def wrap_to_2pi(values: np.ndarray | float) -> np.ndarray:
    values_arr = np.asarray(values, dtype=float)
    wrapped = np.mod(values_arr, 2.0 * np.pi)
    wrapped[np.isclose(wrapped, 2.0 * np.pi)] = 0.0
    return wrapped


def compute_drive_phase_rad(
    encoder_phase_unwrapped_rad: np.ndarray,
    encoder_to_drive_ratio: float,
    encoder_to_drive_sign: float,
    drive_phase_zero_offset_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    ratio = float(encoder_to_drive_ratio)
    sign = float(encoder_to_drive_sign)
    offset = float(drive_phase_zero_offset_rad)

    drive_unwrapped = sign * np.asarray(encoder_phase_unwrapped_rad, dtype=float) / ratio + offset
    drive_wrapped = wrap_to_2pi(drive_unwrapped)
    return drive_unwrapped, drive_wrapped


def compute_wing_stroke_angle_rad(
    drive_phase_rad: np.ndarray,
    wing_stroke_amplitude_rad: float,
    wing_stroke_phase_offset_rad: float,
) -> np.ndarray:
    drive = np.asarray(drive_phase_rad, dtype=float)
    amplitude = float(wing_stroke_amplitude_rad)
    offset = float(wing_stroke_phase_offset_rad)
    return amplitude * np.sin(drive + offset)


def compute_wing_stroke_direction(
    drive_phase_rad: np.ndarray,
    wing_stroke_phase_offset_rad: float,
    zero_tol: float = 1e-9,
) -> np.ndarray:
    drive = np.asarray(drive_phase_rad, dtype=float)
    derivative = np.cos(drive + float(wing_stroke_phase_offset_rad))
    labels = np.empty(drive.shape, dtype=object)
    labels[derivative > zero_tol] = "upstroke"
    labels[derivative < -zero_tol] = "downstroke"
    labels[np.abs(derivative) <= zero_tol] = "zero_crossing"
    return labels


def encoder_phase_from_counts(
    total_count: np.ndarray,
    position_raw: np.ndarray,
    encoder_counts_per_rev: float,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.asarray(total_count, dtype=float)
    position = np.asarray(position_raw, dtype=float)
    counts_per_rev = float(encoder_counts_per_rev)

    continuous_count = (total - total[0]) + position[0]
    encoder_phase_unwrapped = 2.0 * np.pi * continuous_count / counts_per_rev
    encoder_phase_wrapped = wrap_to_2pi(encoder_phase_unwrapped)
    return encoder_phase_unwrapped, encoder_phase_wrapped
