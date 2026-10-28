"""Phase-Anchored Hierarchical Belief Dynamics (PA-HBD)."""
from torch import nn
from .types import State, Observation, ControlTape, Window
from .dynamics import Config
from .flow import LieGroupFlow
from .belief import CausalBeliefEncoder


class PAHBD(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config=config or Config()
        self.flow=LieGroupFlow(self.config)
        self.belief=CausalBeliefEncoder(self.config)

    def encode(self,window):
        return self.belief(self.flow,window.initial,window.start,window.history,window.controls)

    def forward(self,window,times=None):
        belief=self.encode(window)
        if times is None:times=sorted(set(e.measurement for e in window.targets))
        return belief,self.flow.rollout(belief,times,window.controls)
