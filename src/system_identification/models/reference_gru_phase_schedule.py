"""Differentiable reference GRU with an explicit history-predicted phase schedule."""
import torch
from .reference_gru_unit_gravity import ReferenceGRUUnitGravity, unit_state_features
from .trajectory_main_v1 import (_normalize_quaternion, _rotation_body_to_ned,
    _delta_quaternion, _quaternion_multiply)


class ReferenceGRUPhaseSchedule(ReferenceGRUUnitGravity):
    """Same weights and held-frequency context; optional start-of-step phase input.

    Operational callers generate the schedule from observed history only. Training
    must use out-of-flight phase predictions, never the sample's future phase label.
    With no schedule this is exactly the immutable unit-gravity baseline.
    """
    def forward(self, history_state, history_reference, history_actuator,
                future_reference, initial_quaternion, initial_velocity_n,
                initial_rates_b, current_mode, dt_s, phase_start=None):
        if phase_start is None:
            return super().forward(history_state, history_reference, history_actuator,
                future_reference, initial_quaternion, initial_velocity_n,
                initial_rates_b, current_mode, dt_s)
        if phase_start.shape != dt_s.shape or future_reference.shape[-1] != 5:
            raise ValueError('phase schedule must match rollout; five references required')
        if not torch.isfinite(phase_start).all() or not (phase_start[:, 0] == 0).all():
            raise ValueError('finite relative phase schedule must start at zero')
        history = torch.cat([(history_state-self.state_mean)/self.state_scale,
            (history_reference-self.reference_mean)/self.reference_scale,
            (history_actuator-self.actuator_mean)/self.actuator_scale,
            current_mode[:, None, :].expand(-1, history_state.shape[1], -1)], dim=-1)
        _, h = self.encoder(history)
        h = h[0]
        quat = _normalize_quaternion(initial_quaternion)
        vel, rate = initial_velocity_n, initial_rates_b
        freq = history_state[:, -1, 11]
        anchor = torch.zeros_like(freq)
        air = history_state[:, -1, 12:14]
        result = {k: [] for k in ['velocity_n', 'quaternion', 'rates_b']}
        for k in range(future_reference.shape[1]):
            features = torch.cat([unit_state_features(vel, quat, rate, phase_start[:, k], anchor, freq), air], -1)
            ref = (future_reference[:, k]-self.reference_mean)/self.reference_scale
            inp = torch.cat([(features-self.state_mean)/self.state_scale, ref, current_mode], -1)
            h = self.cell(inp, h)
            derivative = self.head(torch.cat([h, inp], -1)).clamp(-6., 6.)*self.derivative_scale
            dt = dt_s[:, k, None]
            vel = vel + torch.einsum('bij,bj->bi', _rotation_body_to_ned(quat), derivative[:, :3])*dt
            newrate = rate + derivative[:, 3:]*dt
            quat = _normalize_quaternion(_quaternion_multiply(quat, _delta_quaternion(.5*(rate+newrate)*dt)))
            rate = newrate
            for key, value in [('velocity_n', vel), ('quaternion', quat), ('rates_b', rate)]:
                result[key].append(value)
        return {k: torch.stack(v, 1) for k, v in result.items()}
