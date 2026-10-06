"""Persistent, inference-only adapter for the frozen H26 actuator GRU.

No data access or fitting. Each call consumes one post-allocation command.
Frequency and airspeed remain initial-history proxies, exactly as in forward.
"""
from dataclasses import dataclass, fields
import copy
import hashlib
import torch

from .actuator_command_gru import ActuatorCommandGRU
from .reference_gru_unit_gravity import unit_state_features
from .trajectory_main_v1 import (
    _normalize_quaternion, _rotation_body_to_ned, _delta_quaternion,
    _quaternion_multiply,
)


@dataclass(frozen=True)
class ActuatorRolloutState:
    model_id: str
    steps: int
    hidden: torch.Tensor
    quaternion: torch.Tensor
    velocity_n: torch.Tensor
    rates_b: torch.Tensor
    frequency: torch.Tensor
    phase: torch.Tensor
    anchor: torch.Tensor
    air: torch.Tensor
    mode: torch.Tensor
    # Audit metadata only: native decoder has no previous-command input slot.
    previous_command: torch.Tensor


class ActuatorCommandStepper:
    """Independent sessions share frozen weights, never hidden-state storage.

    Supports the original 20ms, maximum 50-step inference contract. Snapshot
    tensors are CPU clones suitable for torch.save and weights_only=True load.
    Restore requires identical model weights, normalization and this version.
    """
    VERSION = 1
    MAX_STEPS = 50

    def __init__(self, model):
        if type(model) is not ActuatorCommandGRU:
            raise TypeError('Expected the frozen ActuatorCommandGRU architecture')
        # Preserve requires_grad flags: changing them selects a different CPU
        # GRU encoder kernel even under no_grad, altering float32 rounding.
        # Inference methods use no_grad and this private copy has no optimizer.
        self._model = copy.deepcopy(model).eval()
        self.device = next(self._model.parameters()).device
        self.dtype = next(self._model.parameters()).dtype
        if self.dtype != torch.float32:
            raise ValueError('This adapter contract requires float32')
        digest = hashlib.sha256(b'ActuatorCommandStepper-v1-H26-K50-dt20ms')
        for name, parameter in self._model.named_parameters():
            digest.update(f'{name}:{parameter.requires_grad}'.encode())
        for name, value in self._model.state_dict().items():
            digest.update(name.encode())
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.model_id = digest.hexdigest()

    def _tensor(self, value, shape):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
            raise ValueError(f'Expected tensor shape {shape}')
        if value.dtype != self.dtype or value.device != self.device:
            raise ValueError('Tensor device/dtype does not match adapter')
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite inference input/state')

    def _validate(self, state):
        if not isinstance(state, ActuatorRolloutState) or state.model_id != self.model_id:
            raise ValueError('Model/state contract mismatch')
        if type(state.steps) is not int or not 0 <= state.steps <= self.MAX_STEPS:
            raise ValueError('Invalid step count')
        b = state.quaternion.shape[0]
        if b < 1:
            raise ValueError('Empty session')
        for name, width in [('hidden', self._model.cell.hidden_size), ('quaternion', 4),
                            ('velocity_n', 3), ('rates_b', 3), ('air', 2),
                            ('mode', 2), ('previous_command', 4)]:
            self._tensor(getattr(state, name), (b, width))
        for name in ['frequency', 'phase', 'anchor']:
            self._tensor(getattr(state, name), (b,))
        if not torch.allclose(state.quaternion.norm(dim=-1), torch.ones(b, device=self.device), rtol=0, atol=1e-5):
            raise ValueError('State quaternion must be normalized')

    @torch.no_grad()
    def initialize(self, history_state, history_command, history_actuator,
                   initial_quaternion, initial_velocity_n, initial_rates_b,
                   current_mode):
        b = history_state.shape[0]
        for value, shape in [(history_state, (b, 26, 14)), (history_command, (b, 26, 4)),
                             (history_actuator, (b, 26, 4)), (initial_quaternion, (b, 4)),
                             (initial_velocity_n, (b, 3)), (initial_rates_b, (b, 3)),
                             (current_mode, (b, 2))]:
            self._tensor(value, shape)
        if (initial_quaternion.norm(dim=-1) < 1e-8).any():
            raise ValueError('Invalid initial quaternion')
        m = self._model
        history = torch.cat([(history_state-m.state_mean)/m.state_scale,
            (m.padded(history_command)-m.reference_mean)/m.reference_scale,
            (history_actuator-m.actuator_mean)/m.actuator_scale,
            current_mode[:, None, :].expand(-1, history_state.shape[1], -1)], dim=-1)
        _, h = m.encoder(history)
        frequency = history_state[:, -1, 11].clone()
        state = ActuatorRolloutState(self.model_id, 0, h[0].clone(),
            _normalize_quaternion(initial_quaternion), initial_velocity_n.clone(),
            initial_rates_b.clone(), frequency, torch.zeros_like(frequency),
            torch.zeros_like(frequency), history_state[:, -1, 12:14].clone(),
            current_mode.clone(), history_command[:, -1].clone())
        self._validate(state)
        return state

    @torch.no_grad()
    def step(self, state, command, dt_s):
        self._validate(state)
        if state.steps == self.MAX_STEPS:
            raise ValueError('Frozen 50-step horizon exhausted')
        b = state.quaternion.shape[0]
        self._tensor(command, (b, 4))
        self._tensor(dt_s, (b,))
        if not torch.allclose(dt_s, torch.full_like(dt_s, .02), rtol=0, atol=1e-8):
            raise ValueError('Frozen model requires a 20ms step')
        m = self._model
        features = torch.cat([unit_state_features(state.velocity_n, state.quaternion,
            state.rates_b, state.phase, state.anchor, state.frequency), state.air], dim=-1)
        ref = (m.padded(command)-m.reference_mean)/m.reference_scale
        inp = torch.cat([(features-m.state_mean)/m.state_scale, ref, state.mode], dim=-1)
        h = m.cell(inp, state.hidden)
        derivative = m.head(torch.cat([h, inp], dim=-1)).clamp(-6., 6.)*m.derivative_scale
        dt = dt_s[:, None]
        rotation = _rotation_body_to_ned(state.quaternion)
        vel = state.velocity_n + torch.einsum('bij,bj->bi', rotation, derivative[:, :3])*dt
        rate = state.rates_b + derivative[:, 3:]*dt
        quat = _normalize_quaternion(_quaternion_multiply(state.quaternion,
            _delta_quaternion(.5*(state.rates_b+rate)*dt)))
        phase = state.phase + 2*torch.pi*state.frequency*dt[:, 0]
        next_state = ActuatorRolloutState(self.model_id, state.steps+1, h, quat, vel, rate,
            state.frequency.clone(), phase, state.anchor.clone(), state.air.clone(),
            state.mode.clone(), command.clone())
        self._validate(next_state)
        return next_state, dict(velocity_n=vel.clone(), quaternion=quat.clone(), rates_b=rate.clone())

    def snapshot(self, state):
        self._validate(state)
        result = dict(version=self.VERSION)
        for field in fields(state):
            value = getattr(state, field.name)
            result[field.name] = value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        return result

    def restore(self, snapshot):
        if set(snapshot) != {'version', *(f.name for f in fields(ActuatorRolloutState))} or snapshot['version'] != self.VERSION:
            raise ValueError('Incomplete or incompatible snapshot')
        values = {key: value.to(self.device).clone() if isinstance(value, torch.Tensor) else value
                  for key, value in snapshot.items() if key != 'version'}
        state = ActuatorRolloutState(**values)
        self._validate(state)
        return state
