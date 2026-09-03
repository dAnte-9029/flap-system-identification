"""Inference-only trajectory baseline models for the Step 1 contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch import nn

from system_identification.physics.delaurier.airflow import quaternion_wxyz_to_rotation_body_to_ned


GRAVITY_NED_M_S2 = np.array([0.0, 0.0, 9.80665], dtype=float)


@dataclass
class InitialTrajectoryState:
    position_n: np.ndarray
    velocity_n: np.ndarray
    quaternion_nb: np.ndarray
    angular_velocity_b: np.ndarray
    relative_phase_rad: np.ndarray
    flap_frequency_hz: np.ndarray


@dataclass
class TrajectoryPrediction:
    position_n: np.ndarray
    velocity_n: np.ndarray
    quaternion_nb: np.ndarray
    angular_velocity_b: np.ndarray
    relative_phase_rad: np.ndarray
    flap_frequency_hz: np.ndarray


def normalize_quaternion_batch(quaternion: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternion, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("quaternion must have shape (batch, 4)")
    norm = np.linalg.norm(values, axis=1)
    if not np.isfinite(values).all() or np.any(norm <= 1.0e-12):
        raise ValueError("quaternion must be finite with nonzero norm")
    return values / norm[:, None]


def attitude_error_deg(first: np.ndarray, second: np.ndarray) -> float | np.ndarray:
    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)
    scalar = first_values.ndim == 1
    if scalar:
        first_values = first_values[None, :]
        second_values = second_values[None, :]
    first_normalized = normalize_quaternion_batch(first_values)
    second_normalized = normalize_quaternion_batch(second_values)
    dot = np.clip(np.abs(np.sum(first_normalized * second_normalized, axis=1)), 0.0, 1.0)
    result = np.degrees(2.0 * np.arccos(dot))
    return float(result[0]) if scalar else result


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = left.T
    w1, x1, y1, z1 = right.T
    return np.column_stack(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ]
    )


def _delta_quaternion(body_rotation_vector: np.ndarray) -> np.ndarray:
    rotation = np.asarray(body_rotation_vector, dtype=float)
    angle = np.linalg.norm(rotation, axis=1)
    delta = np.zeros((len(rotation), 4), dtype=float)
    delta[:, 0] = 1.0
    nonzero = angle > 1.0e-12
    half_angle = 0.5 * angle[nonzero]
    delta[nonzero, 0] = np.cos(half_angle)
    delta[nonzero, 1:] = rotation[nonzero] * (np.sin(half_angle) / angle[nonzero])[:, None]
    return delta


def _allocate_prediction(initial: InitialTrajectoryState, step_count: int) -> TrajectoryPrediction:
    batch_size = len(initial.position_n)
    return TrajectoryPrediction(
        position_n=np.empty((batch_size, step_count + 1, 3), dtype=float),
        velocity_n=np.empty((batch_size, step_count + 1, 3), dtype=float),
        quaternion_nb=np.empty((batch_size, step_count + 1, 4), dtype=float),
        angular_velocity_b=np.empty((batch_size, step_count + 1, 3), dtype=float),
        relative_phase_rad=np.empty((batch_size, step_count + 1), dtype=float),
        flap_frequency_hz=np.empty((batch_size, step_count + 1), dtype=float),
    )


def _set_initial(prediction: TrajectoryPrediction, initial: InitialTrajectoryState) -> None:
    prediction.position_n[:, 0] = initial.position_n
    prediction.velocity_n[:, 0] = initial.velocity_n
    prediction.quaternion_nb[:, 0] = normalize_quaternion_batch(initial.quaternion_nb)
    prediction.angular_velocity_b[:, 0] = initial.angular_velocity_b
    prediction.relative_phase_rad[:, 0] = initial.relative_phase_rad
    prediction.flap_frequency_hz[:, 0] = initial.flap_frequency_hz


def _validate_rollout_inputs(initial: InitialTrajectoryState, controls: np.ndarray, dt_s: np.ndarray) -> None:
    if controls.ndim != 3 or controls.shape[2] != 4:
        raise ValueError("controls must have shape (batch, steps, 4)")
    if dt_s.shape != controls.shape[:2]:
        raise ValueError("dt_s must match the first two control dimensions")
    if len(initial.position_n) != len(controls):
        raise ValueError("initial state batch size must match controls")
    if not np.isfinite(controls).all() or not np.isfinite(dt_s).all() or np.any(dt_s <= 0.0):
        raise ValueError("controls and dt_s must be finite and dt_s positive")


class TrajectoryPredictor(Protocol):
    def rollout(
        self, initial: InitialTrajectoryState, controls: np.ndarray, dt_s: np.ndarray
    ) -> TrajectoryPrediction: ...


class PersistencePredictor:
    """Hold every initial state component fixed for the entire horizon."""

    def rollout(
        self, initial: InitialTrajectoryState, controls: np.ndarray, dt_s: np.ndarray
    ) -> TrajectoryPrediction:
        _validate_rollout_inputs(initial, controls, dt_s)
        prediction = _allocate_prediction(initial, controls.shape[1])
        _set_initial(prediction, initial)
        prediction.position_n[:] = prediction.position_n[:, :1]
        prediction.velocity_n[:] = prediction.velocity_n[:, :1]
        prediction.quaternion_nb[:] = prediction.quaternion_nb[:, :1]
        prediction.angular_velocity_b[:] = prediction.angular_velocity_b[:, :1]
        prediction.relative_phase_rad[:] = prediction.relative_phase_rad[:, :1]
        prediction.flap_frequency_hz[:] = prediction.flap_frequency_hz[:, :1]
        return prediction


class ConstantTwistPredictor:
    """Integrate constant NED velocity, body rate, and flap frequency."""

    def rollout(
        self, initial: InitialTrajectoryState, controls: np.ndarray, dt_s: np.ndarray
    ) -> TrajectoryPrediction:
        _validate_rollout_inputs(initial, controls, dt_s)
        prediction = _allocate_prediction(initial, controls.shape[1])
        _set_initial(prediction, initial)
        for step in range(controls.shape[1]):
            dt = dt_s[:, step]
            prediction.position_n[:, step + 1] = (
                prediction.position_n[:, step] + prediction.velocity_n[:, step] * dt[:, None]
            )
            prediction.velocity_n[:, step + 1] = prediction.velocity_n[:, step]
            prediction.angular_velocity_b[:, step + 1] = prediction.angular_velocity_b[:, step]
            delta = _delta_quaternion(prediction.angular_velocity_b[:, step] * dt[:, None])
            prediction.quaternion_nb[:, step + 1] = normalize_quaternion_batch(
                _quaternion_multiply(prediction.quaternion_nb[:, step], delta)
            )
            frequency = prediction.flap_frequency_hz[:, step]
            prediction.flap_frequency_hz[:, step + 1] = frequency
            prediction.relative_phase_rad[:, step + 1] = np.mod(
                prediction.relative_phase_rad[:, step] + 2.0 * np.pi * frequency * dt,
                2.0 * np.pi,
            )
        return prediction


def dynamics_features(
    *,
    velocity_n: np.ndarray,
    quaternion_nb: np.ndarray,
    angular_velocity_b: np.ndarray,
    relative_phase_rad: np.ndarray,
    flap_frequency_hz: np.ndarray,
    controls: np.ndarray | None,
) -> np.ndarray:
    rotation, valid = quaternion_wxyz_to_rotation_body_to_ned(quaternion_nb)
    if not np.all(valid):
        raise ValueError("dynamics features require finite valid quaternions")
    velocity_b = np.einsum("nji,nj->ni", rotation, velocity_n)
    gravity_b = np.einsum(
        "nji,nj->ni", rotation, np.broadcast_to(GRAVITY_NED_M_S2, velocity_n.shape)
    )
    columns = [
        velocity_b,
        angular_velocity_b,
        gravity_b,
        np.sin(relative_phase_rad)[:, None],
        np.cos(relative_phase_rad)[:, None],
        flap_frequency_hz[:, None],
    ]
    if controls is not None:
        if controls.shape != (len(velocity_n), 4):
            raise ValueError("controls must have shape (batch, 4)")
        columns.append(controls)
    features = np.column_stack(columns)
    if not np.isfinite(features).all():
        raise ValueError("dynamics features must be finite")
    return features


class DerivativePredictor(Protocol):
    use_controls: bool

    def predict_derivatives(
        self, state: InitialTrajectoryState, controls: np.ndarray
    ) -> np.ndarray: ...


class RidgeDynamicsPredictor:
    def __init__(
        self,
        *,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        coefficients: np.ndarray,
        intercept: np.ndarray,
        use_controls: bool,
    ) -> None:
        self.feature_mean = np.asarray(feature_mean, dtype=float)
        self.feature_std = np.asarray(feature_std, dtype=float)
        self.target_mean = np.asarray(target_mean, dtype=float)
        self.target_std = np.asarray(target_std, dtype=float)
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.intercept = np.asarray(intercept, dtype=float)
        self.use_controls = bool(use_controls)

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        scaled = (np.asarray(features, dtype=float) - self.feature_mean) / self.feature_std
        prediction_scaled = scaled @ self.coefficients + self.intercept
        return prediction_scaled * self.target_std + self.target_mean

    def predict_derivatives(self, state: InitialTrajectoryState, controls: np.ndarray) -> np.ndarray:
        features = dynamics_features(
            velocity_n=state.velocity_n,
            quaternion_nb=state.quaternion_nb,
            angular_velocity_b=state.angular_velocity_b,
            relative_phase_rad=state.relative_phase_rad,
            flap_frequency_hz=state.flap_frequency_hz,
            controls=controls if self.use_controls else None,
        )
        return self.predict_features(features)


class MLPDynamicsPredictor:
    def __init__(
        self,
        *,
        module: nn.Module,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        use_controls: bool = True,
    ) -> None:
        self.module = module.cpu().eval()
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)
        self.target_mean = np.asarray(target_mean, dtype=np.float32)
        self.target_std = np.asarray(target_std, dtype=np.float32)
        self.use_controls = bool(use_controls)

    def predict_derivatives(self, state: InitialTrajectoryState, controls: np.ndarray) -> np.ndarray:
        features = dynamics_features(
            velocity_n=state.velocity_n,
            quaternion_nb=state.quaternion_nb,
            angular_velocity_b=state.angular_velocity_b,
            relative_phase_rad=state.relative_phase_rad,
            flap_frequency_hz=state.flap_frequency_hz,
            controls=controls if self.use_controls else None,
        )
        scaled = ((features - self.feature_mean) / self.feature_std).astype(np.float32)
        with torch.no_grad():
            prediction_scaled = self.module(torch.from_numpy(scaled)).numpy()
        return prediction_scaled * self.target_std + self.target_mean


class IntegratedDynamicsPredictor:
    """Roll out fitted body acceleration, angular acceleration, and frequency rate."""

    def __init__(self, derivative_predictor: DerivativePredictor) -> None:
        self.derivative_predictor = derivative_predictor

    def rollout(
        self, initial: InitialTrajectoryState, controls: np.ndarray, dt_s: np.ndarray
    ) -> TrajectoryPrediction:
        _validate_rollout_inputs(initial, controls, dt_s)
        prediction = _allocate_prediction(initial, controls.shape[1])
        _set_initial(prediction, initial)
        for step in range(controls.shape[1]):
            dt = dt_s[:, step]
            state = InitialTrajectoryState(
                position_n=prediction.position_n[:, step],
                velocity_n=prediction.velocity_n[:, step],
                quaternion_nb=prediction.quaternion_nb[:, step],
                angular_velocity_b=prediction.angular_velocity_b[:, step],
                relative_phase_rad=prediction.relative_phase_rad[:, step],
                flap_frequency_hz=prediction.flap_frequency_hz[:, step],
            )
            derivative = self.derivative_predictor.predict_derivatives(state, controls[:, step])
            if derivative.shape != (len(controls), 7) or not np.isfinite(derivative).all():
                raise ValueError("dynamics derivative prediction must be finite with shape (batch, 7)")
            acceleration_b = derivative[:, :3]
            angular_acceleration_b = derivative[:, 3:6]
            frequency_rate = derivative[:, 6]
            rotation, valid = quaternion_wxyz_to_rotation_body_to_ned(state.quaternion_nb)
            if not np.all(valid):
                raise ValueError("rollout produced an invalid quaternion")
            acceleration_n = np.einsum("nij,nj->ni", rotation, acceleration_b)
            prediction.position_n[:, step + 1] = (
                state.position_n
                + state.velocity_n * dt[:, None]
                + 0.5 * acceleration_n * np.square(dt)[:, None]
            )
            prediction.velocity_n[:, step + 1] = state.velocity_n + acceleration_n * dt[:, None]
            next_omega = state.angular_velocity_b + angular_acceleration_b * dt[:, None]
            midpoint_omega = 0.5 * (state.angular_velocity_b + next_omega)
            delta = _delta_quaternion(midpoint_omega * dt[:, None])
            prediction.quaternion_nb[:, step + 1] = normalize_quaternion_batch(
                _quaternion_multiply(state.quaternion_nb, delta)
            )
            prediction.angular_velocity_b[:, step + 1] = next_omega
            next_frequency = np.clip(state.flap_frequency_hz + frequency_rate * dt, 0.5, 20.0)
            prediction.flap_frequency_hz[:, step + 1] = next_frequency
            prediction.relative_phase_rad[:, step + 1] = np.mod(
                state.relative_phase_rad
                + 2.0 * np.pi * 0.5 * (state.flap_frequency_hz + next_frequency) * dt,
                2.0 * np.pi,
            )
        return prediction
