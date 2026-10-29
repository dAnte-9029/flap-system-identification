"""Phase-Anchored Hierarchical Belief Dynamics (PA-HBD)."""
from torch import nn
from dataclasses import replace
from .types import State, Observation, ControlTape, Window
from .dynamics import Config
from .flow import LieGroupFlow
from .belief import CausalBeliefEncoder


class _PreparedControls(ControlTape):
    """Forward-local, read-only tape; never retained across calls."""
    def canonical(self):
        return self


class PAHBD(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config=config or Config()
        self.flow=LieGroupFlow(self.config)
        self.belief=CausalBeliefEncoder(self.config)

    def encode(self,window):
        return self.belief(self.flow,window.initial,window.start,window.history,window.controls)

    def forward(self,window,times=None):
        # Keep differentiable controls on the original sensitivity path.
        if not window.controls.values.requires_grad:
            tape=window.controls.canonical()
            window=replace(window,controls=_PreparedControls(tape.times,tape.values))
        belief=self.encode(window)
        if times is None:times=sorted(set(e.measurement for e in window.targets))
        return belief,self.flow.rollout(belief,times,window.controls)
