"""Inference adapter for the unchanged Main V2 transition and checkpoint schema.

All dynamic memory lives in SimulatorState. Weights, normalization and time
constants are fixed configuration; diagnostics never feed back into evolution.
The legacy forward remains untouched and is an independent parity reference.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

import torch

from system_identification.models.trajectory_main_v1 import (
    _delta_quaternion, _normalize_quaternion, _quaternion_multiply,
    _rotation_body_to_ned, _state_features,
)
from system_identification.models.trajectory_main_v2 import (
    ActuatorAwareTrajectoryModel, causal_first_order_filter,
)


@dataclass(frozen=True)
class SimulatorState:
    """Batched minimal persistent state, tied to a fixed model configuration.

    tail_state holds symmetric, differential and rudder normalized command
    proxies, not measured surface angles. drive_state is also dimensionless.
    phase_anchor is retained across save/restore and must never be re-anchored.
    """
    position_n: torch.Tensor
    velocity_n: torch.Tensor
    quaternion_nb: torch.Tensor
    angular_velocity_b: torch.Tensor
    relative_phase_rad: torch.Tensor
    flap_frequency_hz: torch.Tensor
    gru_hidden: torch.Tensor
    drive_state: torch.Tensor
    tail_state: torch.Tensor
    phase_anchor: torch.Tensor

    def snapshot(self) -> dict[str, torch.Tensor]:
        """Detached CPU tensor mapping, safe for torch.save / weights_only load."""
        return {f.name: getattr(self, f.name).detach().cpu().clone() for f in fields(self)}

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, torch.Tensor], *, device="cpu") -> SimulatorState:
        if set(snapshot) != {f.name for f in fields(cls)}:
            raise ValueError("snapshot must contain exactly the SimulatorState fields")
        return cls(**{k: v.to(device=device).clone() for k, v in snapshot.items()})


@dataclass(frozen=True)
class StepDiagnostics:
    acceleration_b: torch.Tensor
    angular_acceleration_b: torch.Tensor
    frequency_clipped: torch.Tensor
    derivative_clipped: torch.Tensor
    drive_clipped: torch.Tensor
    residual_clipped: torch.Tensor


class MainV2Simulator:
    """Functional reset/step API; no mutable hidden state inside the adapter.

    Use model.eval() and torch.inference_mode() for deployment. step returns
    (next_state, diagnostics); the observable physical state is in next_state.
    The adapter does not alter weights or the legacy training residual cache.
    """

    def __init__(self, model: ActuatorAwareTrajectoryModel):
        self.model = model

    def reset(self, *, state: SimulatorState | None = None, **warm_inputs) -> SimulatorState:
        if state is not None:
            if warm_inputs:
                raise ValueError("explicit state and history initialization are mutually exclusive")
            # No normalization, history encoding, or phase reset on restore.
            restored = SimulatorState(**{f.name: getattr(state, f.name).clone() for f in fields(state)})
            self._validate_state(restored)
            return restored
        return self._warm_reset(**warm_inputs)

    def _validate_state(self, state: SimulatorState) -> None:
        batch = state.position_n.shape[0]
        sizes = dict(position_n=3, velocity_n=3, quaternion_nb=4,
                     angular_velocity_b=3, gru_hidden=self.model.base_model.hidden_size,
                     tail_state=3)
        reference = self.model.base_model.feature_mean
        for f in fields(state):
            value = getattr(state, f.name)
            shape = (batch, sizes[f.name]) if f.name in sizes else (batch,)
            if value.shape != shape or value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"invalid shape/device/dtype for {f.name}")
            if not torch.isfinite(value).all():
                raise ValueError(f"nonfinite initial state: {f.name}")
        if torch.any((torch.linalg.vector_norm(state.quaternion_nb, dim=-1) - 1).abs() > 1e-4):
            raise ValueError("explicit quaternion must already be unit length")

    def _warm_reset(self, *, history_state_features, history_controls, history_mask,
                    position_n, velocity_n, quaternion_nb, angular_velocity_b,
                    relative_phase_rad, flap_frequency_hz) -> SimulatorState:
        hidden = self.model.base_model._encode_history(history_state_features, history_controls, history_mask)
        drive = causal_first_order_filter(self.model._normalized_motor(history_controls), history_mask,
                                         dt_s=self.model.history_dt_s, tau_s=self.model.drive_tau_s)
        tail = causal_first_order_filter(self.model._normalized_tail(history_controls), history_mask,
                                        dt_s=self.model.history_dt_s, tau_s=self.model.tail_tau_s)
        state = SimulatorState(position_n, velocity_n, _normalize_quaternion(quaternion_nb),
                               angular_velocity_b, relative_phase_rad, flap_frequency_hz,
                               hidden, drive, tail, relative_phase_rad.clone())
        self._validate_state(state)
        return state

    def cold_reset(self, *, strategy: str, command: torch.Tensor, **physical) -> SimulatorState:
        """Diagnostic alternatives, never silently substituted for warm history.

        single: encode t0 once (existing short-history boundary behavior).
        zero: no GRU observations, zero hidden, steady-command actuator proxies.
        repeated26: encode the same t0 26 times; tests fictitious steady history.
        All strategies initialize actuator proxies at the current command.
        """
        if strategy not in {"single", "zero", "repeated26"}:
            raise ValueError("unknown cold initialization strategy")
        n = 26 if strategy == "repeated26" else 1
        feature = _state_features(physical["velocity_n"], _normalize_quaternion(physical["quaternion_nb"]),
                                  physical["angular_velocity_b"], physical["relative_phase_rad"],
                                  physical["relative_phase_rad"], physical["flap_frequency_hz"])
        state = self.reset(**physical, history_state_features=feature[:, None].repeat(1,n,1),
                           history_controls=command[:, None].repeat(1,n,1),
                           history_mask=torch.ones((len(command),n),device=command.device,dtype=torch.bool))
        if strategy == "zero":
            from dataclasses import replace
            state = replace(state, gru_hidden=torch.zeros_like(state.gru_hidden))
        return state

    def step(self, state: SimulatorState, command: torch.Tensor, dt: torch.Tensor):
        """Unique next state given this state, command, dt and fixed model.

        dt is [batch] in seconds. The legacy update ordering is preserved:
        derivatives use current actuator proxies, commands update them last.
        """
        if command.shape != (len(state.position_n), 4) or dt.shape != (len(command),):
            raise ValueError("command must be [batch,4] and dt [batch]")
        if not torch.all(torch.isfinite(dt) & (dt > 0)) or not torch.isfinite(command).all():
            raise ValueError("commands and positive dt must be finite")
        position, velocity = state.position_n, state.velocity_n
        quaternion, rate = state.quaternion_nb, state.angular_velocity_b
        phase, frequency = state.relative_phase_rad, state.flap_frequency_hz
        hidden, drive_state, tail_state = state.gru_hidden, state.drive_state, state.tail_state
        phase_anchor, controls = state.phase_anchor, command
        drive_clipped = torch.zeros_like(frequency, dtype=torch.int64)
        features = _state_features(velocity, quaternion, rate, phase, phase_anchor, frequency)
        normalized_features = self.model.base_model._normalize_features(features)
        model_input = self.model.rollout_model_input(features, controls)
        derivative_scaled = self.model.base_model.derivative_head(
            torch.cat((hidden, model_input), dim=1)
        )
        control_residual = torch.zeros_like(derivative_scaled)
        if self.model.use_drive:
            drive_inputs = torch.stack((drive_state, normalized_features[:, -1]), dim=1)
            drive_raw = self.model.drive_head(drive_inputs).squeeze(1)
            drive_clipped = (drive_raw.abs() > 2.0).to(torch.int64)
            drive_residual = torch.clamp(drive_raw, -2.0, 2.0)
            control_residual[:, 6] = drive_residual
        if self.model.use_tail:
            gates = self.model.tail_gate_values()
            for channel, head in enumerate(self.model.tail_heads):
                effectiveness = self.model.tail_effectiveness(channel, normalized_features)
                channel_residual = (
                    effectiveness
                    * tail_state[:, channel : channel + 1]
                    * self.model.tail_output_mask[channel]
                    * gates[channel]
                )
                control_residual = control_residual + channel_residual
        residual_clipped = (control_residual.abs() > 2.0).sum(dim=1)
        control_residual = torch.clamp(control_residual, -2.0, 2.0)
        derivative_clipped = ((derivative_scaled + control_residual).abs() > 6.0).sum(dim=1)
        derivative_scaled = torch.clamp(derivative_scaled + control_residual, -6.0, 6.0)
        derivative = self.model.base_model.derivative_mean + self.model.base_model.derivative_std * derivative_scaled
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
        frequency_raw = frequency + frequency_rate * dt
        frequency_clipped = ((frequency_raw < 0.5) | (frequency_raw > 20.0)).to(torch.int64)
        next_frequency = torch.clamp(frequency_raw, 0.5, 20.0)
        next_phase = torch.remainder(
            phase + 2.0 * torch.pi * 0.5 * (frequency + next_frequency) * dt,
            2.0 * torch.pi,
        )
        next_features = _state_features(
            next_velocity, next_quaternion, next_rate, next_phase, phase_anchor, next_frequency
        )
        hidden = self.model.base_model.recurrent_cell(
            self.model.rollout_model_input(next_features, controls), hidden
        )
        drive_alpha = 1.0 - torch.exp(-dt / self.model.drive_tau_s)
        drive_state = drive_state + drive_alpha * (self.model._normalized_motor(controls) - drive_state)
        tail_alpha = 1.0 - torch.exp(-dt / self.model.tail_tau_s)
        normalized_tail = self.model._normalized_tail(controls)
        tail_state = tail_state + tail_alpha[:, None] * (normalized_tail - tail_state)

        next_state = SimulatorState(next_position, next_velocity, next_quaternion, next_rate,
                                    next_phase, next_frequency, hidden, drive_state, tail_state,
                                    phase_anchor)
        diagnostics = StepDiagnostics(acceleration_b, angular_acceleration_b,
                                      frequency_clipped, derivative_clipped, drive_clipped, residual_clipped)
        return next_state, diagnostics
