"""Prescribed DeLaurier pitch kinematics for offline evaluation.

Synchronized from dAnte-9029/IsaacLab commit
3b5d4ec1d28f1384cf042402992ad7ea59995f49. ``theta_tip`` is the amplitude at
the theoretical geometric tip ``y=R``, not at the last strip center.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeLaurierTwistKinematics:
    """Strip pitch fields with shapes ``(B,N)`` in rad, rad/s, and rad/s^2."""

    theta: np.ndarray
    theta_dot: np.ndarray
    theta_ddot: np.ndarray
    delta_theta: np.ndarray
    delta_theta_dot: np.ndarray
    delta_theta_ddot: np.ndarray
    span_fraction: np.ndarray
    phase: np.ndarray
    phase_rate: np.ndarray
    phase_acceleration: np.ndarray


def map_canonical_phase_to_delaurier(phase_rad: np.ndarray) -> np.ndarray:
    """Map canonical ``q=A sin(phi_C)`` to frozen ``q=Gamma cos(phi_D)``.

    ``phi_D = wrap(phi_C - pi/2)`` preserves the physical pose and direction;
    its first and second derivative signs are unchanged.
    """

    return np.mod(np.asarray(phase_rad, dtype=float) - 0.5 * np.pi, 2.0 * np.pi)


def _strip_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array[None, :]
    if array.ndim == 2:
        return array
    raise ValueError(f"{name} must have shape (N,) or (B,N)")


def _batch_column(values: np.ndarray | float, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim == 2 and array.shape[1] == 1:
        return array
    raise ValueError(f"{name} must be scalar or have shape (B,) or (B,1)")


def compute_delaurier_dynamic_twist(
    *,
    strip_span_m: np.ndarray,
    strip_width_m: np.ndarray,
    mean_pitch_rad: np.ndarray | float,
    tip_twist_amplitude_rad: np.ndarray | float,
    phase_rad: np.ndarray,
    phase_rate_rad_s: np.ndarray,
    phase_acceleration_rad_s2: np.ndarray,
    enabled: bool,
    semi_span_m: np.ndarray | float | None = None,
) -> DeLaurierTwistKinematics:
    """Compute linear-span prescribed dynamic twist and analytic derivatives."""

    span = _strip_matrix(strip_span_m, name="strip_span_m")
    width = _strip_matrix(strip_width_m, name="strip_width_m")
    if span.shape[-1] != width.shape[-1]:
        raise ValueError("strip_span_m and strip_width_m must have the same strip count")
    if np.any(span < 0.0) or np.any(width <= 0.0):
        raise ValueError("strip spans must be non-negative and widths positive")

    pitch = np.asarray(mean_pitch_rad, dtype=float)
    if pitch.ndim == 0:
        pitch = pitch.reshape(1, 1)
    elif pitch.ndim == 1:
        pitch = pitch[:, None]
    elif pitch.ndim != 2 or pitch.shape[1] not in {1, span.shape[1]}:
        raise ValueError("mean_pitch_rad must be scalar or have shape (B,), (B,1), or (B,N)")

    tip = _batch_column(tip_twist_amplitude_rad, name="tip_twist_amplitude_rad")
    phase = _batch_column(phase_rad, name="phase_rad")
    rate = _batch_column(phase_rate_rad_s, name="phase_rate_rad_s")
    acceleration = _batch_column(phase_acceleration_rad_s2, name="phase_acceleration_rad_s2")
    sizes = {a.shape[0] for a in (span, width, pitch, tip, phase, rate, acceleration) if a.shape[0] != 1}
    if len(sizes) > 1:
        raise ValueError(f"Incompatible batch dimensions: {sorted(sizes)}")
    batch = sizes.pop() if sizes else 1

    def expand(array: np.ndarray) -> np.ndarray:
        return np.broadcast_to(array, (batch, *array.shape[1:]))

    span, width, pitch, tip, phase, rate, acceleration = map(
        expand,
        (span, width, pitch, tip, phase, rate, acceleration),
    )
    pitch = np.broadcast_to(pitch, (batch, span.shape[1]))
    if semi_span_m is None:
        semi_span = np.max(span + 0.5 * width, axis=1, keepdims=True)
    else:
        semi_span = expand(_batch_column(semi_span_m, name="semi_span_m"))
    if np.any(semi_span <= 0.0):
        raise ValueError("semi_span_m must be positive")
    fraction = span / semi_span
    if np.any(fraction > 1.0 + 1.0e-6):
        raise ValueError("A strip center lies beyond the geometric semi-span")

    if enabled:
        delta = -tip * fraction * np.sin(phase)
        delta_dot = -tip * fraction * np.cos(phase) * rate
        delta_ddot = tip * fraction * (
            np.sin(phase) * np.square(rate) - np.cos(phase) * acceleration
        )
    else:
        delta = np.zeros_like(fraction)
        delta_dot = np.zeros_like(fraction)
        delta_ddot = np.zeros_like(fraction)
    return DeLaurierTwistKinematics(
        theta=pitch + delta,
        theta_dot=delta_dot,
        theta_ddot=delta_ddot,
        delta_theta=delta,
        delta_theta_dot=delta_dot,
        delta_theta_ddot=delta_ddot,
        span_fraction=fraction,
        phase=phase,
        phase_rate=rate,
        phase_acceleration=acceleration,
    )
