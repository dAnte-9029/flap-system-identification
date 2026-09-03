"""Actuator-aware residual dynamics on a frozen history-only trajectory model."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from system_identification.models.trajectory_main_v1 import (
    CONTROL_DIM,
    DERIVATIVE_DIM,
    STATE_FEATURE_DIM,
    CausalHistoryTrajectoryModel,
    TorchTrajectoryPrediction,
    _delta_quaternion,
    _normalize_quaternion,
    _quaternion_multiply,
    _rotation_body_to_ned,
    _state_features,
)


TAIL_DIM = 3


def transform_tail_controls(controls: torch.Tensor) -> torch.Tensor:
    """Map [motor, left, right, rudder] to [symmetric, differential, rudder]."""
    if controls.shape[-1] != CONTROL_DIM:
        raise ValueError("controls must end in four Step 1 channels")
    left = controls[..., 1]
    right = controls[..., 2]
    return torch.stack((0.5 * (left + right), 0.5 * (left - right), controls[..., 3]), dim=-1)


def causal_first_order_filter(
    commands: torch.Tensor,
    mask: torch.Tensor,
    *,
    dt_s: float,
    tau_s: float,
) -> torch.Tensor:
    """Return the final state of a masked causal first-order actuator filter."""
    if commands.ndim not in (2, 3) or mask.shape != commands.shape[:2]:
        raise ValueError("commands must be [batch, history, ...] with a matching mask")
    if dt_s <= 0.0 or tau_s <= 0.0:
        raise ValueError("dt_s and tau_s must be positive")
    state = torch.zeros_like(commands[:, 0])
    seen = torch.zeros(commands.shape[0], dtype=torch.bool, device=commands.device)
    alpha = 1.0 - np.exp(-float(dt_s) / float(tau_s))
    for step in range(commands.shape[1]):
        valid = mask[:, step]
        first = valid & ~seen
        update = valid & seen
        selector = (slice(None),) + (None,) * (commands.ndim - 2)
        state = torch.where(first[selector], commands[:, step], state)
        candidate = state + alpha * (commands[:, step] - state)
        state = torch.where(update[selector], candidate, state)
        seen |= valid
    if not torch.all(seen):
        raise ValueError("every history row must contain at least one valid sample")
    return state


class ActuatorAwareTrajectoryModel(nn.Module):
    """Add bounded drive/tail residuals while preserving a frozen no-control backbone."""

    def __init__(
        self,
        *,
        base_model: CausalHistoryTrajectoryModel,
        use_drive: bool,
        use_tail: bool,
        gated_tail: bool,
        tail_mean: np.ndarray,
        tail_std: np.ndarray,
        drive_tau_s: float = 0.10,
        tail_tau_s: float = 0.04,
        history_dt_s: float = 0.02,
        initial_tail_gate: float = 0.05,
    ) -> None:
        super().__init__()
        if base_model.use_controls:
            raise ValueError("V2 requires a no-control history backbone")
        if min(drive_tau_s, tail_tau_s, history_dt_s) <= 0.0:
            raise ValueError("actuator time constants and history dt must be positive")
        if not 0.0 < initial_tail_gate < 1.0:
            raise ValueError("initial_tail_gate must be between zero and one")
        self.base_model = copy.deepcopy(base_model)
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.use_drive = bool(use_drive)
        self.use_tail = bool(use_tail)
        self.gated_tail = bool(gated_tail)
        self.drive_tau_s = float(drive_tau_s)
        self.tail_tau_s = float(tail_tau_s)
        self.history_dt_s = float(history_dt_s)
        tail_mean_array = np.asarray(tail_mean, dtype=float)
        tail_std_array = np.asarray(tail_std, dtype=float)
        if tail_mean_array.shape != (TAIL_DIM,) or tail_std_array.shape != (TAIL_DIM,):
            raise ValueError("tail statistics must have shape (3,)")
        self.register_buffer("tail_mean", torch.as_tensor(tail_mean_array, dtype=torch.float32))
        self.register_buffer("tail_std", torch.as_tensor(tail_std_array, dtype=torch.float32))

        self.drive_head = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )
        self.tail_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(STATE_FEATURE_DIM, 16),
                    nn.Tanh(),
                    nn.Linear(16, DERIVATIVE_DIM),
                )
                for _ in range(TAIL_DIM)
            ]
        )
        nn.init.normal_(self.drive_head[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.drive_head[-1].bias)
        for head in self.tail_heads:
            nn.init.normal_(head[-1].weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(head[-1].bias)
        if not self.use_drive:
            for parameter in self.drive_head.parameters():
                parameter.requires_grad_(False)
        if not self.use_tail:
            for parameter in self.tail_heads.parameters():
                parameter.requires_grad_(False)
        gate_logit = float(np.log(initial_tail_gate / (1.0 - initial_tail_gate)))
        if self.gated_tail:
            self.tail_gate_logits = nn.Parameter(torch.full((TAIL_DIM,), gate_logit))
        else:
            self.register_buffer("tail_gate_logits", torch.full((TAIL_DIM,), np.inf))
        self.register_buffer(
            "tail_output_mask",
            torch.tensor(
                [
                    [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
            ),
        )
        self._last_control_residuals: list[torch.Tensor] = []

    def tail_gate_values(self) -> torch.Tensor:
        if not self.use_tail:
            return self.tail_gate_logits.new_zeros(TAIL_DIM)
        if self.gated_tail:
            return torch.sigmoid(self.tail_gate_logits)
        return self.tail_gate_logits.new_ones(TAIL_DIM)

    def control_regularization_loss(
        self, *, residual_l2: float, tail_gate_l1: float
    ) -> torch.Tensor:
        if residual_l2 < 0.0 or tail_gate_l1 < 0.0:
            raise ValueError("control regularization weights must be non-negative")
        reference = next(self.parameters())
        penalty = reference.new_zeros(())
        if self._last_control_residuals:
            penalty = penalty + residual_l2 * torch.mean(
                torch.square(torch.stack(self._last_control_residuals, dim=1))
            )
        if self.use_tail and self.gated_tail:
            penalty = penalty + tail_gate_l1 * torch.sum(self.tail_gate_values())
        return penalty

    def _normalized_motor(self, controls: torch.Tensor) -> torch.Tensor:
        return (controls[..., 0] - self.base_model.control_mean[0]) / self.base_model.control_std[0]

    def _normalized_tail(self, controls: torch.Tensor) -> torch.Tensor:
        return (transform_tail_controls(controls) - self.tail_mean) / self.tail_std

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
        if not self.use_drive and not self.use_tail:
            return self.base_model(
                history_state_features=history_state_features,
                history_controls=history_controls,
                history_mask=history_mask,
                position_n=position_n,
                velocity_n=velocity_n,
                quaternion_nb=quaternion_nb,
                angular_velocity_b=angular_velocity_b,
                relative_phase_rad=relative_phase_rad,
                flap_frequency_hz=flap_frequency_hz,
                future_controls=future_controls,
                dt_s=dt_s,
            )
        if future_controls.ndim != 3 or future_controls.shape[2] != CONTROL_DIM:
            raise ValueError("future_controls must have shape [batch, steps, 4]")
        if dt_s.shape != future_controls.shape[:2]:
            raise ValueError("dt_s must have shape [batch, steps]")
        hidden = self.base_model._encode_history(
            history_state_features, history_controls, history_mask
        )
        drive_state = causal_first_order_filter(
            self._normalized_motor(history_controls),
            history_mask,
            dt_s=self.history_dt_s,
            tau_s=self.drive_tau_s,
        )
        tail_state = causal_first_order_filter(
            self._normalized_tail(history_controls),
            history_mask,
            dt_s=self.history_dt_s,
            tau_s=self.tail_tau_s,
        )
        phase_anchor = relative_phase_rad.clone()
        position_values = [position_n]
        velocity_values = [velocity_n]
        quaternion_values = [_normalize_quaternion(quaternion_nb)]
        rate_values = [angular_velocity_b]
        phase_values = [relative_phase_rad]
        frequency_values = [flap_frequency_hz]
        self._last_control_residuals = []

        for step in range(future_controls.shape[1]):
            position = position_values[-1]
            velocity = velocity_values[-1]
            quaternion = quaternion_values[-1]
            rate = rate_values[-1]
            phase = phase_values[-1]
            frequency = frequency_values[-1]
            controls = future_controls[:, step]
            dt = dt_s[:, step]
            features = _state_features(velocity, quaternion, rate, phase, phase_anchor, frequency)
            normalized_features = self.base_model._normalize_features(features)
            model_input = self.base_model._model_input(features, controls)
            derivative_scaled = self.base_model.derivative_head(
                torch.cat((hidden, model_input), dim=1)
            )
            control_residual = torch.zeros_like(derivative_scaled)
            if self.use_drive:
                drive_inputs = torch.stack((drive_state, normalized_features[:, -1]), dim=1)
                drive_residual = torch.clamp(self.drive_head(drive_inputs).squeeze(1), -2.0, 2.0)
                control_residual[:, 6] = drive_residual
            if self.use_tail:
                gates = self.tail_gate_values()
                for channel, head in enumerate(self.tail_heads):
                    effectiveness = head(normalized_features)
                    channel_residual = (
                        effectiveness
                        * tail_state[:, channel : channel + 1]
                        * self.tail_output_mask[channel]
                        * gates[channel]
                    )
                    control_residual = control_residual + channel_residual
            control_residual = torch.clamp(control_residual, -2.0, 2.0)
            self._last_control_residuals.append(control_residual)
            derivative_scaled = torch.clamp(derivative_scaled + control_residual, -6.0, 6.0)
            derivative = self.base_model.derivative_mean + self.base_model.derivative_std * derivative_scaled
            acceleration_b = derivative[:, :3]
            angular_acceleration_b = derivative[:, 3:6]
            frequency_rate = derivative[:, 6]
            rotation = _rotation_body_to_ned(quaternion)
            acceleration_n = torch.einsum("bij,bj->bi", rotation, acceleration_b)
            next_position = position + velocity * dt[:, None] + 0.5 * acceleration_n * torch.square(dt)[:, None]
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
                next_velocity, next_quaternion, next_rate, next_phase, phase_anchor, next_frequency
            )
            hidden = self.base_model.recurrent_cell(
                self.base_model._model_input(next_features, controls), hidden
            )
            drive_alpha = 1.0 - torch.exp(-dt / self.drive_tau_s)
            drive_state = drive_state + drive_alpha * (self._normalized_motor(controls) - drive_state)
            tail_alpha = 1.0 - torch.exp(-dt / self.tail_tau_s)
            normalized_tail = self._normalized_tail(controls)
            tail_state = tail_state + tail_alpha[:, None] * (normalized_tail - tail_state)
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
