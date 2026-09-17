"""Frozen-weight, predicted-history representation probe (no truth input)."""
from dataclasses import dataclass, replace
import torch
from .main_v2_simulator import SimulatorState
from .trajectory_main_v1 import _state_features


def state_features(state):
    return _state_features(state.velocity_n, state.quaternion_nb, state.angular_velocity_b,
                           state.relative_phase_rad, state.phase_anchor, state.flap_frequency_hz)


@dataclass(frozen=True)
class RollingHistoryState:
    physical: SimulatorState
    features: torch.Tensor
    controls: torch.Tensor
    mask: torch.Tensor

    def snapshot(self):
        return dict(physical=self.physical.snapshot(), features=self.features.detach().cpu().clone(),
                    controls=self.controls.detach().cpu().clone(), mask=self.mask.detach().cpu().clone())

    @classmethod
    def from_snapshot(cls, value, device='cpu'):
        return cls(SimulatorState.from_snapshot(value['physical'], device=device),
                   **{k:value[k].to(device).clone() for k in ('features','controls','mask')})


class RollingHistorySimulator:
    """Keep original phase anchor/proxies/integration; reencode only predicted history.

    t0 retains the common warm26 hidden. After each transition, keep K samples,
    encode from zero for the next transition. The appended control is u_t,
    matching the original recurrent update's (x_(t+1), u_t) alignment. Current
    Main V2 base ignores controls, but the alignment is explicit nonetheless.
    """
    def __init__(self, simulator, history_steps):
        if history_steps not in (13, 26):
            raise ValueError('audit history length must be 13 or 26')
        self.simulator, self.history_steps = simulator, history_steps

    def reset(self, initial, features, controls, mask):
        k=self.history_steps
        return RollingHistoryState(self.simulator.reset(state=initial), features[:,-k:].clone(),
                                   controls[:,-k:].clone(), mask[:,-k:].clone())

    def step(self, state, command, dt):
        next_state, diagnostic = self.simulator.step(state.physical, command, dt)
        features=torch.cat((state.features[:,1:], state_features(next_state)[:,None]),1)
        controls=torch.cat((state.controls[:,1:],command[:,None]),1)
        mask=torch.cat((state.mask[:,1:],torch.ones_like(state.mask[:,:1])),1)
        hidden=self.simulator.model.base_model._encode_history(features, controls, mask)
        return RollingHistoryState(replace(next_state,gru_hidden=hidden),features,controls,mask),diagnostic
