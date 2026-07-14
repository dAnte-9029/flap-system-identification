"""Exact bounded nonlinear calibration for DeLaurier prior parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import least_squares, minimize

from scripts.prior_regression_calibration import ParameterSpec


PriorEvaluator = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class NonlinearObjectiveResult:
    total_loss: float
    data_loss: float
    regularization_loss: float


@dataclass(frozen=True)
class NonlinearCalibrationResult:
    theta: np.ndarray
    delta: np.ndarray
    normalized_delta: np.ndarray
    hit_bounds: list[str]
    total_loss: float
    data_loss: float
    regularization_loss: float
    success: bool
    message: str
    n_function_evaluations: int
    lambda_value: float
    optimizer: str


def _centers(parameter_specs: Sequence[ParameterSpec]) -> np.ndarray:
    return np.asarray([spec.center for spec in parameter_specs], dtype=float)


def _lowers(parameter_specs: Sequence[ParameterSpec]) -> np.ndarray:
    return np.asarray([spec.lower for spec in parameter_specs], dtype=float)


def _uppers(parameter_specs: Sequence[ParameterSpec]) -> np.ndarray:
    return np.asarray([spec.upper for spec in parameter_specs], dtype=float)


def _penalty_scales(parameter_specs: Sequence[ParameterSpec]) -> np.ndarray:
    scales = np.asarray([spec.penalty_scale for spec in parameter_specs], dtype=float)
    if np.any(scales <= 0):
        raise ValueError("parameter penalty_scale values must be positive")
    return scales


def _validate_problem(
    *,
    labels: np.ndarray,
    parameter_specs: Sequence[ParameterSpec],
    lambda_value: float,
    target_scales: np.ndarray,
) -> None:
    if labels.ndim != 2:
        raise ValueError("labels must have shape (n_samples, n_targets)")
    if not parameter_specs:
        raise ValueError("parameter_specs must not be empty")
    centers = _centers(parameter_specs)
    lowers = _lowers(parameter_specs)
    uppers = _uppers(parameter_specs)
    if np.any(lowers > centers) or np.any(centers > uppers):
        raise ValueError("each parameter center must lie within its bounds")
    if lambda_value < 0:
        raise ValueError("lambda_value must be nonnegative")
    if target_scales.shape != (labels.shape[1],):
        raise ValueError("target_scales must have shape (n_targets,)")
    if np.any(~np.isfinite(target_scales)) or np.any(target_scales <= 0):
        raise ValueError("target_scales must be finite and positive")


def _hit_bounds(theta: np.ndarray, parameter_specs: Sequence[ParameterSpec]) -> list[str]:
    hits: list[str] = []
    tol = 1.0e-6
    for spec, value in zip(parameter_specs, theta, strict=True):
        if np.isclose(value, spec.lower, rtol=0.0, atol=tol):
            hits.append(f"{spec.name}:lower")
        if np.isclose(value, spec.upper, rtol=0.0, atol=tol):
            hits.append(f"{spec.name}:upper")
    return hits


def nonlinear_objective(
    *,
    theta: np.ndarray,
    labels: np.ndarray,
    evaluate_prior: PriorEvaluator,
    parameter_specs: Sequence[ParameterSpec],
    lambda_value: float,
    target_scales: np.ndarray,
) -> NonlinearObjectiveResult:
    """Evaluate scaled data loss plus normalized parameter regularization."""

    theta = np.asarray(theta, dtype=float)
    labels = np.asarray(labels, dtype=float)
    target_scales = np.asarray(target_scales, dtype=float)
    _validate_problem(
        labels=labels,
        parameter_specs=parameter_specs,
        lambda_value=float(lambda_value),
        target_scales=target_scales,
    )
    if theta.shape != (len(parameter_specs),):
        raise ValueError("theta must have shape (n_parameters,)")

    prediction = np.asarray(evaluate_prior(theta), dtype=float)
    if prediction.shape != labels.shape:
        raise ValueError("evaluate_prior returned an array with the wrong shape")

    residual = (labels - prediction) / target_scales[None, :]
    data_loss = float(np.sum(residual * residual))
    normalized_delta = (theta - _centers(parameter_specs)) / _penalty_scales(parameter_specs)
    regularization_loss = float(lambda_value) * float(np.sum(normalized_delta * normalized_delta))
    return NonlinearObjectiveResult(
        total_loss=data_loss + regularization_loss,
        data_loss=data_loss,
        regularization_loss=regularization_loss,
    )


def make_theta_cache_key(
    theta: np.ndarray,
    parameter_specs: Sequence[ParameterSpec],
    *,
    precision: int = 6,
) -> str:
    """Create a stable file-system-safe key for a physical parameter vector."""

    values: list[str] = []
    for spec, value in zip(parameter_specs, np.asarray(theta, dtype=float), strict=True):
        rounded = round(float(value), int(precision))
        text = f"{rounded:.{precision}f}".rstrip("0").rstrip(".")
        if text == "-0":
            text = "0"
        text = text.replace("-", "m").replace(".", "p")
        values.append(f"{spec.name}_{text}")
    return "__".join(values)


def solve_bounded_nonlinear_calibration(
    *,
    labels: np.ndarray,
    evaluate_prior: PriorEvaluator,
    parameter_specs: Sequence[ParameterSpec],
    lambda_value: float,
    target_scales: np.ndarray,
    initial_theta: np.ndarray | None = None,
    maxiter: int = 200,
    max_function_evaluations: int | None = None,
    optimizer: str = "least_squares",
    diff_step: float = 1.0e-2,
) -> NonlinearCalibrationResult:
    """Solve exact bounded nonlinear prior calibration."""

    labels = np.asarray(labels, dtype=float)
    target_scales = np.asarray(target_scales, dtype=float)
    _validate_problem(
        labels=labels,
        parameter_specs=parameter_specs,
        lambda_value=float(lambda_value),
        target_scales=target_scales,
    )
    centers = _centers(parameter_specs)
    lowers = _lowers(parameter_specs)
    uppers = _uppers(parameter_specs)
    penalty_scales = _penalty_scales(parameter_specs)

    if initial_theta is None:
        initial = centers.copy()
    else:
        initial = np.asarray(initial_theta, dtype=float)
        if initial.shape != centers.shape:
            raise ValueError("initial_theta must have shape (n_parameters,)")
        initial = np.clip(initial, lowers, uppers)

    optimizer = str(optimizer)
    if optimizer == "pattern_search":
        max_evals = 200 if max_function_evaluations is None else int(max_function_evaluations)
        theta = initial.copy()
        steps = np.asarray([spec.penalty_scale for spec in parameter_specs], dtype=float) * 0.5
        min_steps = np.maximum(np.asarray([spec.step for spec in parameter_specs], dtype=float) * 0.05, 1.0e-4)
        current = nonlinear_objective(
            theta=theta,
            labels=labels,
            evaluate_prior=evaluate_prior,
            parameter_specs=parameter_specs,
            lambda_value=float(lambda_value),
            target_scales=target_scales,
        )
        nfev = 1
        sweeps = 0
        while nfev < max_evals and sweeps < int(maxiter) and np.any(steps > min_steps):
            improved_in_sweep = False
            for param_index in range(len(parameter_specs)):
                if nfev >= max_evals:
                    break
                best_theta = theta
                best = current
                for direction in (1.0, -1.0):
                    if nfev >= max_evals:
                        break
                    candidate = theta.copy()
                    candidate[param_index] = np.clip(
                        candidate[param_index] + direction * steps[param_index],
                        lowers[param_index],
                        uppers[param_index],
                    )
                    if np.allclose(candidate, theta, rtol=0.0, atol=1.0e-12):
                        continue
                    candidate_result = nonlinear_objective(
                        theta=candidate,
                        labels=labels,
                        evaluate_prior=evaluate_prior,
                        parameter_specs=parameter_specs,
                        lambda_value=float(lambda_value),
                        target_scales=target_scales,
                    )
                    nfev += 1
                    if candidate_result.total_loss < best.total_loss:
                        best_theta = candidate
                        best = candidate_result
                if best.total_loss < current.total_loss:
                    theta = best_theta
                    current = best
                    improved_in_sweep = True
            if not improved_in_sweep:
                steps = steps * 0.5
            sweeps += 1
        success = bool(np.all(steps <= min_steps) or nfev >= 1)
        message = f"pattern_search finished after {sweeps} sweeps and {nfev} evaluations"
    elif optimizer == "least_squares":
        def residual_vector(theta: np.ndarray) -> np.ndarray:
            theta = np.asarray(theta, dtype=float)
            prediction = np.asarray(evaluate_prior(theta), dtype=float)
            if prediction.shape != labels.shape:
                raise ValueError("evaluate_prior returned an array with the wrong shape")
            scaled = ((labels - prediction) / target_scales[None, :]).reshape(-1)
            if lambda_value <= 0:
                return scaled
            normalized_delta = (theta - centers) / penalty_scales
            regularizer = np.sqrt(float(lambda_value)) * normalized_delta
            return np.concatenate([scaled, regularizer])

        fit = least_squares(
            residual_vector,
            initial,
            bounds=(lowers, uppers),
            max_nfev=max_function_evaluations,
            diff_step=float(diff_step),
            xtol=1.0e-5,
            ftol=1.0e-8,
            gtol=1.0e-8,
            method="trf",
        )
        theta = np.clip(np.asarray(fit.x, dtype=float), lowers, uppers)
        success = bool(fit.success)
        message = str(fit.message)
        nfev = int(getattr(fit, "nfev", 0))
    elif optimizer == "powell":
        def objective(theta: np.ndarray) -> float:
            return nonlinear_objective(
                theta=np.asarray(theta, dtype=float),
                labels=labels,
                evaluate_prior=evaluate_prior,
                parameter_specs=parameter_specs,
                lambda_value=float(lambda_value),
                target_scales=target_scales,
            ).total_loss

        fit = minimize(
            objective,
            initial,
            method="Powell",
            bounds=list(zip(lowers, uppers, strict=True)),
            options={
                "maxiter": int(maxiter),
                "maxfev": None if max_function_evaluations is None else int(max_function_evaluations),
                "xtol": 1.0e-5,
                "ftol": 1.0e-8,
            },
        )
        theta = np.clip(np.asarray(fit.x, dtype=float), lowers, uppers)
        success = bool(fit.success)
        message = str(fit.message)
        nfev = int(getattr(fit, "nfev", 0))
    else:
        raise ValueError(f"unsupported optimizer: {optimizer}")

    final = nonlinear_objective(
        theta=theta,
        labels=labels,
        evaluate_prior=evaluate_prior,
        parameter_specs=parameter_specs,
        lambda_value=float(lambda_value),
        target_scales=target_scales,
    )
    delta = theta - centers
    return NonlinearCalibrationResult(
        theta=theta,
        delta=delta,
        normalized_delta=delta / penalty_scales,
        hit_bounds=_hit_bounds(theta, parameter_specs),
        total_loss=final.total_loss,
        data_loss=final.data_loss,
        regularization_loss=final.regularization_loss,
        success=success,
        message=message,
        n_function_evaluations=nfev,
        lambda_value=float(lambda_value),
        optimizer=optimizer,
    )
