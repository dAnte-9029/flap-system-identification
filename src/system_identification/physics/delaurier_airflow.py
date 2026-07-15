"""Offline DeLaurier airflow frame helpers.

Synchronized from the pure convention helpers in dAnte-9029/IsaacLab commit
3b5d4ec1d28f1384cf042402992ad7ea59995f49. This module has no Isaac runtime
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

BodyFrameConvention = Literal["FLU", "FRD"]


@dataclass(frozen=True)
class ReconstructedBodyAirflow:
    """Air-relative velocity reconstructed from NED navigation data.

    ``velocity_body_frd_m_s`` is aircraft velocity relative to air, expressed
    in body FRD. ``rotation_body_to_ned`` follows the PX4 wxyz quaternion
    convention. Vertical wind is an explicit caller input rather than a hidden
    assumption.
    """

    velocity_body_frd_m_s: np.ndarray
    speed_m_s: np.ndarray
    alpha_rad: np.ndarray
    beta_rad: np.ndarray
    rotation_body_to_ned: np.ndarray
    quaternion_valid: np.ndarray


def quaternion_wxyz_to_rotation_body_to_ned(
    quaternion_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PX4 body-FRD to NED rotation matrices and a validity mask."""

    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError("quaternion_wxyz must have shape (B,4)")
    norms = np.linalg.norm(quaternion, axis=1)
    valid = np.isfinite(quaternion).all(axis=1) & (norms > 1.0e-12)
    rotation = np.full((len(quaternion), 3, 3), np.nan, dtype=float)
    if not np.any(valid):
        return rotation, valid
    normalized = np.zeros_like(quaternion)
    normalized[valid] = quaternion[valid] / norms[valid, None]
    w, x, y, z = normalized.T
    rotation[valid, 0, 0] = 1.0 - 2.0 * (y[valid] ** 2 + z[valid] ** 2)
    rotation[valid, 0, 1] = 2.0 * (x[valid] * y[valid] - z[valid] * w[valid])
    rotation[valid, 0, 2] = 2.0 * (x[valid] * z[valid] + y[valid] * w[valid])
    rotation[valid, 1, 0] = 2.0 * (x[valid] * y[valid] + z[valid] * w[valid])
    rotation[valid, 1, 1] = 1.0 - 2.0 * (x[valid] ** 2 + z[valid] ** 2)
    rotation[valid, 1, 2] = 2.0 * (y[valid] * z[valid] - x[valid] * w[valid])
    rotation[valid, 2, 0] = 2.0 * (x[valid] * z[valid] - y[valid] * w[valid])
    rotation[valid, 2, 1] = 2.0 * (y[valid] * z[valid] + x[valid] * w[valid])
    rotation[valid, 2, 2] = 1.0 - 2.0 * (x[valid] ** 2 + y[valid] ** 2)
    return rotation, valid


def reconstruct_body_airflow_from_ned(
    *,
    ground_velocity_ned_m_s: np.ndarray,
    wind_velocity_ned_m_s: np.ndarray,
    quaternion_body_to_ned_wxyz: np.ndarray,
) -> ReconstructedBodyAirflow:
    """Rotate ``ground velocity - wind`` from NED into body FRD.

    All vector inputs have shape ``(B,3)``. The returned vector describes the
    aircraft velocity relative to the air, so positive body ``u`` is forward.
    """

    ground = np.asarray(ground_velocity_ned_m_s, dtype=float)
    wind = np.asarray(wind_velocity_ned_m_s, dtype=float)
    if ground.ndim != 2 or ground.shape[1] != 3:
        raise ValueError("ground_velocity_ned_m_s must have shape (B,3)")
    if wind.shape != ground.shape:
        raise ValueError("wind_velocity_ned_m_s must match ground velocity shape")
    rotation, quaternion_valid = quaternion_wxyz_to_rotation_body_to_ned(
        quaternion_body_to_ned_wxyz
    )
    if len(rotation) != len(ground):
        raise ValueError("quaternion and velocity batch sizes must match")
    air_velocity_ned = ground - wind
    velocity_body = np.einsum("nji,nj->ni", rotation, air_velocity_ned)
    finite = (
        quaternion_valid
        & np.isfinite(ground).all(axis=1)
        & np.isfinite(wind).all(axis=1)
    )
    velocity_body[~finite] = np.nan
    speed = np.linalg.norm(velocity_body, axis=1)
    alpha = np.arctan2(velocity_body[:, 2], velocity_body[:, 0])
    beta = np.arctan2(
        velocity_body[:, 1],
        np.sqrt(np.square(velocity_body[:, 0]) + np.square(velocity_body[:, 2])),
    )
    alpha[~finite] = np.nan
    beta[~finite] = np.nan
    speed[~finite] = np.nan
    return ReconstructedBodyAirflow(
        velocity_body_frd_m_s=velocity_body,
        speed_m_s=speed,
        alpha_rad=alpha,
        beta_rad=beta,
        rotation_body_to_ned=rotation,
        quaternion_valid=finite,
    )


def body_air_velocity_to_delaurier_section_velocity(
    air_velocity_body: np.ndarray,
    *,
    body_frame: BodyFrameConvention,
) -> np.ndarray:
    """Return a polar air-relative velocity in internal section FRD axes.

    Input and output have shape ``(..., 3)`` and units m/s. FRD is ``+x``
    forward, ``+y`` right, ``+z`` down. FLU is converted exactly once by
    ``diag(1,-1,-1)``.
    """

    values = np.asarray(air_velocity_body, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("air_velocity_body must have shape (...,3)")
    if body_frame == "FRD":
        return values.copy()
    if body_frame == "FLU":
        result = values.copy()
        result[..., 1:] *= -1.0
        return result
    raise ValueError(f"Unsupported body frame {body_frame!r}; expected 'FLU' or 'FRD'")


def compute_delaurier_axis_incidence(
    *,
    air_velocity_body: np.ndarray,
    body_frame: BodyFrameConvention,
    minimum_forward_speed_mps: float = 1.0e-3,
) -> np.ndarray:
    """Return ``theta_a=atan2(w_D,u_D)`` in radians."""

    minimum = float(minimum_forward_speed_mps)
    if minimum <= 0.0:
        raise ValueError("minimum_forward_speed_mps must be positive")
    section_velocity = body_air_velocity_to_delaurier_section_velocity(
        air_velocity_body,
        body_frame=body_frame,
    )
    forward = section_velocity[..., 0]
    downward = section_velocity[..., 2]
    return np.arctan2(downward, forward)
