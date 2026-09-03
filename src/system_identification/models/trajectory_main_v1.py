"""Causal history-aware trajectory dynamics for Main V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch import nn

from system_identification.models.trajectory import TrajectoryPrediction, dynamics_features


STATE_FEATURE_DIM = 12
CONTROL_DIM = 4
DERIVATIVE_DIM = 7


def offset_invariant_dynamics_features(
    *,
    velocity_n: np.ndarray,
    quaternion_nb: np.ndarray,
    angular_velocity_b: np.ndarray,
    relative_phase_rad: np.ndarray,
    phase_anchor_rad: np.ndarray,
    flap_frequency_hz: np.ndarray,
) -> np.ndarray:
    """Build Step 2 state features without assuming a shared phase zero."""
    return dynamics_features(
        velocity_n=velocity_n,
        quaternion_nb=quaternion_nb,
        angular_velocity_b=angular_velocity_b,
        relative_phase_rad=np.asarray(relative_phase_rad) - np.asarray(phase_anchor_rad),
        flap_frequency_hz=flap_frequency_hz,
        controls=None,
    )


@dataclass
class TorchTrajectoryPrediction:
    position_n: torch.Tensor
    velocity_n: torch.Tensor
    quaternion_nb: torch.Tensor
    angular_velocity_b: torch.Tensor
    relative_phase_rad: torch.Tensor
    flap_frequency_hz: torch.Tensor

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(
            (
                self.position_n,
                self.velocity_n,
                self.quaternion_nb,
                self.angular_velocity_b,
                self.relative_phase_rad,
                self.flap_frequency_hz,
            )
        )

    def to_numpy(self) -> TrajectoryPrediction:
        values = [value.detach().cpu().numpy().astype(float) for value in self]
        return TrajectoryPrediction(*values)


def _normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    w0, x0, y0, z0 = left.unbind(dim=-1)
    w1, x1, y1, z1 = right.unbind(dim=-1)
    return torch.stack(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ),
        dim=-1,
    )


def _delta_quaternion(rotation_vector: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    scale = torch.sin(half_angle) / angle.clamp_min(1.0e-8)
    small_scale = 0.5 - torch.square(angle) / 48.0
    scale = torch.where(angle < 1.0e-4, small_scale, scale)
    return torch.cat((torch.cos(half_angle), rotation_vector * scale), dim=-1)


def _rotation_body_to_ned(quaternion: torch.Tensor) -> torch.Tensor:
    q = _normalize_quaternion(quaternion)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def _state_features(
    velocity_n: torch.Tensor,
    quaternion_nb: torch.Tensor,
    angular_velocity_b: torch.Tensor,
    relative_phase_rad: torch.Tensor,
    phase_anchor_rad: torch.Tensor,
    flap_frequency_hz: torch.Tensor,
) -> torch.Tensor:
    rotation = _rotation_body_to_ned(quaternion_nb)
    velocity_b = torch.einsum("bji,bj->bi", rotation, velocity_n)
    gravity_n = torch.zeros_like(velocity_n)
    gravity_n[:, 2] = 9.80665
    gravity_b = torch.einsum("bji,bj->bi", rotation, gravity_n)
    phase_delta = relative_phase_rad - phase_anchor_rad
    return torch.cat(
        (
            velocity_b,
            angular_velocity_b,
            gravity_b,
            torch.sin(phase_delta)[:, None],
            torch.cos(phase_delta)[:, None],
            flap_frequency_hz[:, None],
        ),
        dim=1,
    )


class CausalHistoryTrajectoryModel(nn.Module):
    """Encode causal history and autoregressively predict physical derivatives."""

    def __init__(
        self,
        *,
        hidden_size: int,
        use_controls: bool,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        control_mean: np.ndarray,
        control_std: np.ndarray,
        derivative_mean: np.ndarray,
        derivative_std: np.ndarray,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = int(hidden_size)
        self.use_controls = bool(use_controls)
        input_dim = STATE_FEATURE_DIM + (CONTROL_DIM if self.use_controls else 0)
        # Sharing the recurrent transition keeps the local-vs-multistep ablation matched:
        # history encoding trains the same state update used during open-loop rollout.
        self.recurrent_cell = nn.GRUCell(input_dim, self.hidden_size)
        self.derivative_head = nn.Sequential(
            nn.Linear(self.hidden_size + input_dim, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, DERIVATIVE_DIM),
        )
        nn.init.zeros_(self.derivative_head[-1].weight)
        nn.init.zeros_(self.derivative_head[-1].bias)
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.as_tensor(feature_std, dtype=torch.float32))
        self.register_buffer("control_mean", torch.as_tensor(control_mean, dtype=torch.float32))
        self.register_buffer("control_std", torch.as_tensor(control_std, dtype=torch.float32))
        self.register_buffer("derivative_mean", torch.as_tensor(derivative_mean, dtype=torch.float32))
        self.register_buffer("derivative_std", torch.as_tensor(derivative_std, dtype=torch.float32))
        if self.feature_mean.shape != (STATE_FEATURE_DIM,) or self.feature_std.shape != (
            STATE_FEATURE_DIM,
        ):
            raise ValueError("state feature statistics must have shape (12,)")
        if self.control_mean.shape != (CONTROL_DIM,) or self.control_std.shape != (CONTROL_DIM,):
            raise ValueError("control statistics must have shape (4,)")
        if self.derivative_mean.shape != (DERIVATIVE_DIM,) or self.derivative_std.shape != (
            DERIVATIVE_DIM,
        ):
            raise ValueError("derivative statistics must have shape (7,)")

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.feature_mean) / self.feature_std

    def _normalize_controls(self, controls: torch.Tensor) -> torch.Tensor:
        return (controls - self.control_mean) / self.control_std

    def _model_input(self, features: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        normalized = self._normalize_features(features)
        if self.use_controls:
            normalized = torch.cat((normalized, self._normalize_controls(controls)), dim=-1)
        return normalized

    def _encode_history(
        self,
        state_features: torch.Tensor,
        controls: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if state_features.ndim != 3 or state_features.shape[2] != STATE_FEATURE_DIM:
            raise ValueError("history_state_features must have shape [batch, history, 12]")
        if controls.shape != (*state_features.shape[:2], CONTROL_DIM):
            raise ValueError("history_controls must have shape [batch, history, 4]")
        if mask.shape != state_features.shape[:2]:
            raise ValueError("history_mask must have shape [batch, history]")
        hidden = state_features.new_zeros((len(state_features), self.hidden_size))
        for step in range(state_features.shape[1]):
            inputs = self._model_input(state_features[:, step], controls[:, step])
            candidate = self.recurrent_cell(inputs, hidden)
            hidden = torch.where(mask[:, step, None], candidate, hidden)
        return hidden

    def forward(
        self,
        *,
        history_state_features: torch.Tensor,
        history_controls: torch.Tensor,
        history_mask: torch.Tensor,
        position_n: torch.Tensor,
        velocity_n: torch.Tensor,
        quaternion_nb: torch.Tensor,
        angular_velocity_b: torch.Tensor,
        relative_phase_rad: torch.Tensor,
        flap_frequency_hz: torch.Tensor,
        future_controls: torch.Tensor,
        dt_s: torch.Tensor,
    ) -> TorchTrajectoryPrediction:
        if future_controls.ndim != 3 or future_controls.shape[2] != CONTROL_DIM:
            raise ValueError("future_controls must have shape [batch, steps, 4]")
        if dt_s.shape != future_controls.shape[:2]:
            raise ValueError("dt_s must have shape [batch, steps]")
        hidden = self._encode_history(history_state_features, history_controls, history_mask)
        phase_anchor = relative_phase_rad.clone()
        position_values = [position_n]
        velocity_values = [velocity_n]
        quaternion_values = [_normalize_quaternion(quaternion_nb)]
        rate_values = [angular_velocity_b]
        phase_values = [relative_phase_rad]
        frequency_values = [flap_frequency_hz]

        for step in range(future_controls.shape[1]):
            position = position_values[-1]
            velocity = velocity_values[-1]
            quaternion = quaternion_values[-1]
            rate = rate_values[-1]
            phase = phase_values[-1]
            frequency = frequency_values[-1]
            controls = future_controls[:, step]
            dt = dt_s[:, step]
            features = _state_features(
                velocity, quaternion, rate, phase, phase_anchor, frequency
            )
            model_input = self._model_input(features, controls)
            derivative_scaled = torch.clamp(
                self.derivative_head(torch.cat((hidden, model_input), dim=1)), -6.0, 6.0
            )
            derivative = self.derivative_mean + self.derivative_std * derivative_scaled
            acceleration_b = derivative[:, :3]
            angular_acceleration_b = derivative[:, 3:6]
            frequency_rate = derivative[:, 6]
            rotation = _rotation_body_to_ned(quaternion)
            acceleration_n = torch.einsum("bij,bj->bi", rotation, acceleration_b)
            next_position = (
                position
                + velocity * dt[:, None]
                + 0.5 * acceleration_n * torch.square(dt)[:, None]
            )
            next_velocity = velocity + acceleration_n * dt[:, None]
            next_rate = rate + angular_acceleration_b * dt[:, None]
            midpoint_rate = 0.5 * (rate + next_rate)
            next_quaternion = _normalize_quaternion(
                _quaternion_multiply(quaternion, _delta_quaternion(midpoint_rate * dt[:, None]))
            )
            next_frequency = torch.clamp(frequency + frequency_rate * dt, 0.5, 20.0)
            next_phase = torch.remainder(
                phase + 2.0 * torch.pi * 0.5 * (frequency + next_frequency) * dt,
                2.0 * torch.pi,
            )
            next_features = _state_features(
                next_velocity,
                next_quaternion,
                next_rate,
                next_phase,
                phase_anchor,
                next_frequency,
            )
            hidden = self.recurrent_cell(self._model_input(next_features, controls), hidden)
            position_values.append(next_position)
            velocity_values.append(next_velocity)
            quaternion_values.append(next_quaternion)
            rate_values.append(next_rate)
            phase_values.append(next_phase)
            frequency_values.append(next_frequency)

        return TorchTrajectoryPrediction(
            position_n=torch.stack(position_values, dim=1),
            velocity_n=torch.stack(velocity_values, dim=1),
            quaternion_nb=torch.stack(quaternion_values, dim=1),
            angular_velocity_b=torch.stack(rate_values, dim=1),
            relative_phase_rad=torch.stack(phase_values, dim=1),
            flap_frequency_hz=torch.stack(frequency_values, dim=1),
        )
