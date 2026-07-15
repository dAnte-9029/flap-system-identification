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


def annotate_phase_cycles(
    time_s: np.ndarray,
    phase_rad: np.ndarray,
    flap_frequency_hz: np.ndarray,
    phase_valid: np.ndarray | None = None,
    active_frequency_threshold_hz: float = 0.1,
    reset_threshold_rad: float = np.pi,
    min_cycle_samples: int = 4,
    min_cycle_duration_s: float = 0.05,
    min_cycle_span_rad: float = 5.0,
) -> dict[str, np.ndarray]:
    time_arr = np.asarray(time_s, dtype=float)
    phase_arr = np.asarray(phase_rad, dtype=float)
    flap_freq_arr = np.asarray(flap_frequency_hz, dtype=float)
    n = len(phase_arr)

    if phase_valid is None:
        phase_valid_arr = np.ones(n, dtype=bool)
    else:
        phase_valid_arr = np.asarray(phase_valid, dtype=bool)

    flap_active = np.isfinite(flap_freq_arr) & (flap_freq_arr > float(active_frequency_threshold_hz))
    usable = np.isfinite(time_arr) & np.isfinite(phase_arr) & phase_valid_arr
    active_mask = usable & flap_active

    cycle_id = np.full(n, -1, dtype=int)
    cycle_valid = np.zeros(n, dtype=bool)
    phase_corrected = np.full(n, np.nan, dtype=float)
    phase_corrected_unwrapped = np.full(n, np.nan, dtype=float)
    cycle_duration_s = np.full(n, np.nan, dtype=float)
    cycle_flap_frequency_hz = np.full(n, np.nan, dtype=float)

    next_cycle_id = 0
    idx = 0

    while idx < n:
        if not active_mask[idx]:
            idx += 1
            continue

        run_start = idx
        while idx + 1 < n and active_mask[idx + 1]:
            idx += 1
        run_end = idx

        run_phase = phase_arr[run_start : run_end + 1]
        reset_idx = np.flatnonzero(np.diff(run_phase) < -abs(float(reset_threshold_rad)))
        cycle_starts = np.r_[0, reset_idx + 1]
        cycle_ends = np.r_[reset_idx, len(run_phase) - 1]

        for run_cycle_idx, (cycle_start, cycle_end) in enumerate(zip(cycle_starts, cycle_ends)):
            sample_idx = np.arange(run_start + cycle_start, run_start + cycle_end + 1, dtype=int)
            segment_phase = phase_arr[sample_idx]
            segment_time = time_arr[sample_idx]
            segment_flap_freq = flap_freq_arr[sample_idx]

            cycle_id[sample_idx] = next_cycle_id

            phase_start = float(segment_phase[0])
            phase_max = float(np.nanmax(segment_phase))
            phase_span = phase_max - phase_start
            duration_s = float(segment_time[-1] - segment_time[0]) if len(sample_idx) > 1 else 0.0
            mean_flap_freq_hz = float(np.nanmean(segment_flap_freq))

            cycle_duration_s[sample_idx] = duration_s
            cycle_flap_frequency_hz[sample_idx] = mean_flap_freq_hz

            if phase_span > 0.0:
                corrected = np.clip(2.0 * np.pi * (segment_phase - phase_start) / phase_span, 0.0, 2.0 * np.pi)
                phase_corrected[sample_idx] = corrected
                phase_corrected_unwrapped[sample_idx] = next_cycle_id * 2.0 * np.pi + corrected

            monotonic_violation = bool(np.any(np.diff(segment_phase) < -0.25))
            is_edge_cycle = run_cycle_idx == 0 or run_cycle_idx == len(cycle_starts) - 1
            is_valid_cycle = (
                len(sample_idx) >= int(min_cycle_samples)
                and duration_s >= float(min_cycle_duration_s)
                and phase_span >= float(min_cycle_span_rad)
                and not monotonic_violation
                and not is_edge_cycle
            )
            cycle_valid[sample_idx] = is_valid_cycle
            next_cycle_id += 1

        idx += 1

    return {
        "flap_active": flap_active,
        "cycle_id": cycle_id,
        "cycle_valid": cycle_valid,
        "phase_corrected_rad": phase_corrected,
        "phase_corrected_unwrapped_rad": phase_corrected_unwrapped,
        "cycle_duration_s": cycle_duration_s,
        "cycle_flap_frequency_hz": cycle_flap_frequency_hz,
    }
