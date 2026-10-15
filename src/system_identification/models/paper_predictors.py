"""Rao-inspired velocity/attitude predictors with autonomous joint rollout.

Unlike the published component-wise oracle rollout, no future measured state
is accepted. Angular velocity belongs to the velocity module, not attitude.
All variants use a body-frame SO(3) increment for the attitude output.
"""
import torch
from torch import nn

from .actuator_command_gru import ActuatorCommandGRU
from .reference_gru_unit_gravity import unit_state_features
from .trajectory_main_v1 import (
    _normalize_quaternion, _rotation_body_to_ned,
    _delta_quaternion, _quaternion_multiply,
)


def _head(hidden, width, outputs):
    head = nn.Sequential(nn.Linear(hidden + 21, width), nn.Tanh(),
                         nn.Linear(width, outputs))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class PaperPredictor(ActuatorCommandGRU):
    """Parameter-budget matched full, multihead, and decoupled GRU variants.

    full: one encoder/cell/head, 9 outputs (dv, domega, attitude tangent).
    multihead: shared encoder/cell, independent 6- and 3-output MLPs.
    decoupled: independent encoder/cell/head per module. Predicted complementary
      states are exchanged with stop-gradient, including the body-to-NED map;
      own-state temporal gradients remain intact. There is no teacher forcing.

    Zero outputs reproduce constant velocity/rate and constant-rate attitude.
    The attitude head predicts a bounded correction (rad/s) to the *observed
    initial* angular rate, not to the velocity module's predicted rate. Thus
    attitude accuracy alone cannot certify angular-rate consistency.
    """
    ARMS = ('full', 'multihead', 'decoupled')

    def __init__(self, stats, arm):
        if arm not in self.ARMS:
            raise ValueError(f'unknown predictor arm: {arm}')
        hidden = 42 if arm == 'decoupled' else 64
        super().__init__(stats, hidden=hidden)
        self.arm = arm
        if arm == 'full':
            self.head = _head(hidden, hidden, 9)
        elif arm == 'multihead':
            self.head = _head(hidden, 32, 6)
            self.attitude_head = _head(hidden, 32, 3)
        else:
            # Keep the inherited 6-output velocity head.
            self.attitude_encoder = nn.GRU(25, hidden, batch_first=True)
            self.attitude_cell = nn.GRUCell(21, hidden)
            self.attitude_head = _head(hidden, hidden, 3)

    def forward(self, history_state, history_command, history_actuator,
                future_command, initial_quaternion, initial_velocity_n,
                initial_rates_b, current_mode, dt_s):
        history = torch.cat([
            (history_state - self.state_mean) / self.state_scale,
            (self.padded(history_command) - self.reference_mean) / self.reference_scale,
            (history_actuator - self.actuator_mean) / self.actuator_scale,
            current_mode[:, None, :].expand(-1, history_state.shape[1], -1),
        ], dim=-1)
        _, hidden = self.encoder(history)
        hidden = hidden[0]
        separate = self.arm == 'decoupled'
        if separate:
            _, attitude_hidden = self.attitude_encoder(history)
            attitude_hidden = attitude_hidden[0]
        quat = _normalize_quaternion(initial_quaternion)
        vel, rate = initial_velocity_n, initial_rates_b
        freq, air = history_state[:, -1, 11], history_state[:, -1, 12:14]
        phase, anchor = torch.zeros_like(freq), torch.zeros_like(freq)
        commands = (self.padded(future_command) - self.reference_mean) / self.reference_scale
        velocities, quaternions, rates = [], [], []

        def inputs(v, q, w, ref):
            state = torch.cat([unit_state_features(v, q, w, phase, anchor, freq), air], -1)
            return torch.cat([(state - self.state_mean) / self.state_scale, ref, current_mode], -1)

        for k in range(future_command.shape[1]):
            velocity_quat = quat.detach() if separate else quat
            inp = inputs(vel, velocity_quat, rate, commands[:, k])
            hidden = self.cell(inp, hidden)
            encoded = torch.cat([hidden, inp], -1)
            if self.arm == 'full':
                output = self.head(encoded)
                derivative, tangent = output[:, :6], output[:, 6:]
            else:
                derivative = self.head(encoded)
                if separate:
                    att_inp = inputs(vel.detach(), quat, rate.detach(), commands[:, k])
                    attitude_hidden = self.attitude_cell(att_inp, attitude_hidden)
                    tangent = self.attitude_head(torch.cat([attitude_hidden, att_inp], -1))
                else:
                    tangent = self.attitude_head(encoded)
            derivative = derivative.clamp(-6., 6.) * self.derivative_scale
            dt = dt_s[:, k, None]
            vel = vel + torch.einsum('bij,bj->bi', _rotation_body_to_ned(velocity_quat),
                                    derivative[:, :3]) * dt
            rate = rate + derivative[:, 3:] * dt
            rotation_vector = (initial_rates_b + tangent.clamp(-6., 6.)) * dt
            quat = _normalize_quaternion(_quaternion_multiply(quat, _delta_quaternion(rotation_vector)))
            phase = phase + 2 * torch.pi * freq * dt[:, 0]
            velocities.append(vel)
            quaternions.append(quat)
            rates.append(rate)
        return dict(velocity_n=torch.stack(velocities, 1),
                    quaternion=torch.stack(quaternions, 1), rates_b=torch.stack(rates, 1))
