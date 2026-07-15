"""Whole-aircraft effective-wrench label reconstruction.

The reconstructed force and moment preserve the existing body-frame, sign,
unit, validity-mask, and aircraft-CG reference conventions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from system_identification.metadata import metadata_has_complete_labels, nested_value

def _as_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def _frame_columns_or_none(samples: pd.DataFrame, columns: list[str]) -> np.ndarray | None:
    if any(column not in samples.columns for column in columns):
        return None
    return samples[columns].to_numpy(dtype=float, copy=True)


def _rotation_body_to_world_from_quaternions(quaternions_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quaternions = np.asarray(quaternions_wxyz, dtype=float)
    rotation = np.full((len(quaternions), 3, 3), np.nan, dtype=float)

    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        return rotation, np.zeros(len(quaternions), dtype=bool)

    norms = np.linalg.norm(quaternions, axis=1)
    valid = np.isfinite(quaternions).all(axis=1) & (norms > 0.0)

    if not np.any(valid):
        return rotation, valid

    q = np.zeros_like(quaternions)
    q[valid] = quaternions[valid] / norms[valid, None]
    w, x, y, z = q.T

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
def _compute_effective_wrench_labels(
    samples: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    linear_acceleration_columns: list[str] | None = None,
    angular_acceleration_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_count = len(samples)
    nan_vectors = np.full((sample_count, 3), np.nan, dtype=float)
    label_valid = np.zeros(sample_count, dtype=bool)

    if not metadata_has_complete_labels(metadata):
        return nan_vectors.copy(), nan_vectors.copy(), label_valid

    mass_kg = nested_value(metadata, "mass_properties", "mass_kg")
    inertia_raw = nested_value(metadata, "mass_properties", "inertia_b_kg_m2")

    try:
        mass_kg = float(mass_kg)
        inertia_b = np.asarray(inertia_raw, dtype=float)
    except (TypeError, ValueError):
        return nan_vectors.copy(), nan_vectors.copy(), label_valid

    if not np.isfinite(mass_kg) or inertia_b.shape != (3, 3) or not np.isfinite(inertia_b).all():
        return nan_vectors.copy(), nan_vectors.copy(), label_valid

    resolved_linear_acceleration_columns = linear_acceleration_columns or [
        "vehicle_local_position.ax",
        "vehicle_local_position.ay",
        "vehicle_local_position.az",
    ]
    resolved_angular_acceleration_columns = angular_acceleration_columns or [
        "vehicle_angular_velocity.xyz_derivative[0]",
        "vehicle_angular_velocity.xyz_derivative[1]",
        "vehicle_angular_velocity.xyz_derivative[2]",
    ]

    acc_n = _frame_columns_or_none(samples, resolved_linear_acceleration_columns)
    quat_nb = _frame_columns_or_none(
        samples,
        [
            "vehicle_attitude.q[0]",
            "vehicle_attitude.q[1]",
            "vehicle_attitude.q[2]",
            "vehicle_attitude.q[3]",
        ],
    )
    omega_b = _frame_columns_or_none(
        samples,
        [
            "vehicle_angular_velocity.xyz[0]",
            "vehicle_angular_velocity.xyz[1]",
            "vehicle_angular_velocity.xyz[2]",
        ],
    )
    alpha_b = _frame_columns_or_none(samples, resolved_angular_acceleration_columns)

    if acc_n is None or quat_nb is None or omega_b is None or alpha_b is None:
        return nan_vectors.copy(), nan_vectors.copy(), label_valid

    rot_nb, quat_valid = _rotation_body_to_world_from_quaternions(quat_nb)
    gravity_n = np.array(
        [0.0, 0.0, _as_float(nested_value(metadata, "label_definition", "gravity_m_s2"), default=9.81)],
        dtype=float,
    )
    specific_acc_n = acc_n - gravity_n
    force_b = mass_kg * np.einsum("nji,nj->ni", rot_nb, specific_acc_n)

    angular_momentum_b = np.einsum("ij,nj->ni", inertia_b, omega_b)
    inertia_alpha_b = np.einsum("ij,nj->ni", inertia_b, alpha_b)
    moment_b = inertia_alpha_b + np.cross(omega_b, angular_momentum_b)

    finite_mask = (
        np.isfinite(acc_n).all(axis=1)
        & np.isfinite(quat_nb).all(axis=1)
        & np.isfinite(omega_b).all(axis=1)
        & np.isfinite(alpha_b).all(axis=1)
        & quat_valid
    )

    for column in [
        "vehicle_local_position.xy_valid",
        "vehicle_local_position.z_valid",
        "vehicle_local_position.v_xy_valid",
        "vehicle_local_position.v_z_valid",
    ]:
        if column in samples.columns:
            finite_mask &= samples[column].fillna(False).astype(bool).to_numpy()

    force_b[~finite_mask] = np.nan
    moment_b[~finite_mask] = np.nan
    return force_b, moment_b, finite_mask


compute_effective_wrench_labels = _compute_effective_wrench_labels

__all__ = ["compute_effective_wrench_labels"]
