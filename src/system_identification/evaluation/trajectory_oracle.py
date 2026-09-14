"""Offline oracle interventions; never exposed through the production predictor API."""
from __future__ import annotations
import torch
from system_identification.models.trajectory_main_v1 import (
    CONTROL_DIM, TorchTrajectoryPrediction, _state_features, _rotation_body_to_ned,
    _normalize_quaternion, _quaternion_multiply, _delta_quaternion,
)
MODES = ('free_run', 'oracle_attitude_rotation', 'oracle_attitude_feedback', 'oracle_rate')


@torch.no_grad()
def oracle_rollout(model, *, mode='free_run', oracle_quaternion=None, oracle_rate=None, **inputs):
    """Clone of frozen Main V1 rollout with explicit single-channel interventions.

    Attitude rotation changes only the rotation used for acceleration. Indirect
    velocity-to-network feedback remains active. Attitude feedback also replaces
    q in features and next-state recurrent updates; q metrics are forced, unscored.
    Rate intervention uses true omega[k], omega[k+1] for trapezoid integration;
    predicted angular acceleration is discarded, q is integrated and never forced.
    """
    if mode not in MODES:
        raise ValueError('Unknown oracle mode')
    requires_q = mode in ('oracle_attitude_rotation', 'oracle_attitude_feedback')
    requires_rate = mode == 'oracle_rate'
    if requires_q != (oracle_quaternion is not None) or requires_rate != (oracle_rate is not None):
        raise ValueError('Supply only the explicitly selected oracle channel')
    history_state_features = inputs['history_state_features']
    history_controls = inputs['history_controls']
    history_mask = inputs['history_mask']
    position_n = inputs['position_n']
    velocity_n = inputs['velocity_n']
    quaternion_nb = inputs['quaternion_nb']
    angular_velocity_b = inputs['angular_velocity_b']
    relative_phase_rad = inputs['relative_phase_rad']
    flap_frequency_hz = inputs['flap_frequency_hz']
    future_controls = inputs['future_controls']
    dt_s = inputs['dt_s']
    for oracle, width, initial in [(oracle_quaternion, 4, quaternion_nb), (oracle_rate, 3, angular_velocity_b)]:
        if oracle is None:
            continue
        if oracle.shape != (len(position_n), future_controls.shape[1]+1, width) or not torch.isfinite(oracle).all():
            raise ValueError('Oracle shape or finite-value contract failed')
        if not torch.allclose(oracle[:,0], initial, atol=1e-6, rtol=1e-6):
            raise ValueError('Oracle initial state differs from common t0')
        if width == 4 and (torch.linalg.vector_norm(oracle,dim=-1)<1e-8).any():
            raise ValueError('Invalid oracle quaternion')
    if future_controls.ndim != 3 or future_controls.shape[2] != CONTROL_DIM:
        raise ValueError("future_controls must have shape [batch, steps, 4]")
    if dt_s.shape != future_controls.shape[:2]:
        raise ValueError("dt_s must have shape [batch, steps]")
    hidden = model._encode_history(history_state_features, history_controls, history_mask)
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
        if mode == "oracle_attitude_feedback":
            quaternion = _normalize_quaternion(oracle_quaternion[:, step])
        if mode == "oracle_rate":
            rate = oracle_rate[:, step]
        features = _state_features(
            velocity, quaternion, rate, phase, phase_anchor, frequency
        )
        model_input = model._model_input(features, controls)
        derivative_scaled = torch.clamp(
            model.derivative_head(torch.cat((hidden, model_input), dim=1)), -6.0, 6.0
        )
        derivative = model.derivative_mean + model.derivative_std * derivative_scaled
        acceleration_b = derivative[:, :3]
        angular_acceleration_b = derivative[:, 3:6]
        frequency_rate = derivative[:, 6]
        rotation_q = oracle_quaternion[:, step] if mode == "oracle_attitude_rotation" else quaternion
        rotation = _rotation_body_to_ned(rotation_q)
        acceleration_n = torch.einsum("bij,bj->bi", rotation, acceleration_b)
        next_position = (
            position
            + velocity * dt[:, None]
            + 0.5 * acceleration_n * torch.square(dt)[:, None]
        )
        next_velocity = velocity + acceleration_n * dt[:, None]
        next_rate = rate + angular_acceleration_b * dt[:, None]
        if mode == "oracle_rate":
            next_rate = oracle_rate[:, step + 1]
        midpoint_rate = 0.5 * (rate + next_rate)
        next_quaternion = _normalize_quaternion(
            _quaternion_multiply(quaternion, _delta_quaternion(midpoint_rate * dt[:, None]))
        )
        if mode == "oracle_attitude_feedback":
            next_quaternion = _normalize_quaternion(oracle_quaternion[:, step + 1])
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
        hidden = model.recurrent_cell(model._model_input(next_features, controls), hidden)
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
