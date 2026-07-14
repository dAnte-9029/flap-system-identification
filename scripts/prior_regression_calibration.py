"""Linearized bounded calibration for DeLaurier prior parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    center: float
    lower: float
    upper: float
    step: float
    penalty_scale: float


@dataclass(frozen=True)
class CalibrationResult:
    delta: np.ndarray
    theta: np.ndarray
    normalized_delta: np.ndarray
    hit_bounds: list[str]
    cost: float
    optimality: float
    active_mask: np.ndarray
    lambda_value: float


def solve_bounded_linearized_delta(
    *,
    residual: np.ndarray,
    jacobian: np.ndarray,
    parameter_specs: Sequence[ParameterSpec],
    lambda_value: float,
    target_scales: np.ndarray,
) -> CalibrationResult:
    """Solve a bounded ridge fit for a local prior-parameter update.

    The fitted variable is the normalized update
    ``u_p = (theta_p - center_p) / penalty_scale_p``. This keeps the ridge
    penalty comparable across parameters with different physical units.
    """

    residual = np.asarray(residual, dtype=float)
    jacobian = np.asarray(jacobian, dtype=float)
    target_scales = np.asarray(target_scales, dtype=float)

    if residual.ndim != 2:
        raise ValueError("residual must have shape (n_samples, n_targets)")
    if jacobian.ndim != 3:
        raise ValueError("jacobian must have shape (n_samples, n_targets, n_parameters)")
    if jacobian.shape[:2] != residual.shape:
        raise ValueError("jacobian and residual sample/target dimensions do not match")
    if jacobian.shape[2] != len(parameter_specs):
        raise ValueError("jacobian parameter dimension does not match parameter_specs")
    if target_scales.shape != (residual.shape[1],):
        raise ValueError("target_scales must have shape (n_targets,)")
    if np.any(~np.isfinite(target_scales)) or np.any(target_scales <= 0):
        raise ValueError("target_scales must be finite and positive")
    if lambda_value < 0:
        raise ValueError("lambda_value must be nonnegative")

    centers = np.asarray([spec.center for spec in parameter_specs], dtype=float)
    lowers = np.asarray([spec.lower for spec in parameter_specs], dtype=float)
    uppers = np.asarray([spec.upper for spec in parameter_specs], dtype=float)
    penalty_scales = np.asarray([spec.penalty_scale for spec in parameter_specs], dtype=float)
    if np.any(penalty_scales <= 0):
        raise ValueError("parameter penalty_scale values must be positive")
    if np.any(lowers > centers) or np.any(centers > uppers):
        raise ValueError("each parameter center must lie within its bounds")

    weights = 1.0 / target_scales
    weighted_residual = (residual * weights[None, :]).reshape(-1)
    weighted_jacobian = jacobian * weights[None, :, None]
    design = weighted_jacobian.reshape(-1, jacobian.shape[2]) * penalty_scales[None, :]

    if lambda_value > 0:
        regularizer = np.sqrt(lambda_value) * np.eye(jacobian.shape[2], dtype=float)
        design = np.vstack([design, regularizer])
        weighted_residual = np.concatenate(
            [weighted_residual, np.zeros(jacobian.shape[2], dtype=float)]
        )

    lower_u = (lowers - centers) / penalty_scales
    upper_u = (uppers - centers) / penalty_scales
    fit = lsq_linear(design, weighted_residual, bounds=(lower_u, upper_u), method="trf")

    normalized_delta = fit.x
    delta = normalized_delta * penalty_scales
    theta = centers + delta

    tol = 1.0e-8
    hit_bounds: list[str] = []
    for spec, value in zip(parameter_specs, theta):
        if np.isclose(value, spec.lower, rtol=0.0, atol=tol):
            hit_bounds.append(f"{spec.name}:lower")
        if np.isclose(value, spec.upper, rtol=0.0, atol=tol):
            hit_bounds.append(f"{spec.name}:upper")

    return CalibrationResult(
        delta=delta,
        theta=theta,
        normalized_delta=normalized_delta,
        hit_bounds=hit_bounds,
        cost=float(fit.cost),
        optimality=float(fit.optimality),
        active_mask=np.asarray(fit.active_mask, dtype=int),
        lambda_value=float(lambda_value),
    )
