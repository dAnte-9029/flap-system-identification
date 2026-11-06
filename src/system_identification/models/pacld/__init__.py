"""Phase-Anchored Coupled Latent Dynamics, independent of PA-HBD model behavior."""
from torch import nn
from .types import State, Observation, ControlTape, Window, initial_state
from .dynamics import Config, phase_carrier
from .flow import LieGroupFlow
from .belief import CausalBeliefEncoder


class PACLD(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or Config()
        self.flow = LieGroupFlow(self.config)
        self.belief = CausalBeliefEncoder(self.config)

    def encode(self, window):
        # A single FULL history posterior is shared by both future prediction modes.
        return self.belief(self.flow, initial_state(window, self.config), window.start,
                           window.history, window.controls)

    def forward(self, window, times=None, mode='FULL'):
        belief = self.encode(window)
        if times is None:
            times = sorted(set(e.measurement for e in window.targets))
        return belief, self.flow.rollout(belief, times, window.controls, mode=mode)

    def predict_pair(self, window, times=None):
        """Infer once; BASE/FULL share the posterior and all parameters."""
        belief = self.encode(window)
        if times is None:
            times = sorted(set(e.measurement for e in window.targets))
        times = list(times)
        return belief, {mode: self.flow.rollout(belief, times, window.controls, mode=mode)
                        for mode in ('BASE', 'FULL')}
