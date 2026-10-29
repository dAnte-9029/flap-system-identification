"""Explicit runtime contracts. Times are seconds relative to each window decision."""
from dataclasses import dataclass, replace
import torch


@dataclass
class State:
    x: torch.Tensor  # (...,13): p_N, v_N, q_NB(wxyz), omega_B
    h: torch.Tensor  # (...,latent)
    f: torch.Tensor  # (...,1), Hz
    phi: torch.Tensor  # (...,1), unwrapped rad
    actuator: torch.Tensor  # (...,3), filtered normalized left/right elevon, rudder
    variance: torch.Tensor  # (...,14): p,v,body rotation,omega,f,phi diagonal proxy
    phase_quality: torch.Tensor  # (...,1) fixed causal reference confidence [0,1]

    phase_covariance: torch.Tensor | None = None  # joint unwrapped (f, phi), never modulo covariance

    def with_(self, **kwargs):
        return replace(self, **kwargs)


@dataclass
class Observation:
    measurement: float
    availability: float
    value: torch.Tensor  # (...,15): x(13),f,phi; unused quaternion must be identity
    mask: torch.Tensor  # (...,14): tangent p,v,R,omega,f,phi
    variance: torch.Tensor  # (...,14), source noise floor, not learned per window
    quality: torch.Tensor  # (...,14), [0,1]
    source: str
    frame: str = 'NED/FRD/Hamilton-wxyz'


@dataclass
class ControlTape:
    times: list[float]
    values: torch.Tensor  # (events,4) or (events,batch,4); motor,left,right,rudder

    def canonical(self):
        if not self.times or len(self.times) != len(self.values):
            raise ValueError('empty/mismatched control tape')
        if any(b <= a for a,b in zip(self.times, self.times[1:])):
            raise ValueError('control times must strictly increase')
        if not torch.isfinite(self.values).all():
            raise ValueError('nonfinite controls')
        indices = [0]
        for i in range(1, len(self.times)):
            if not torch.equal(self.values[i].detach(), self.values[indices[-1]].detach()):
                indices.append(i)
        # Do not merge distinct differentiable control variables during sensitivity evaluation.
        if self.values.requires_grad:
            indices = list(range(len(self.times)))
        return ControlTape([self.times[i] for i in indices], self.values[indices])


@dataclass
class Window:
    initial: State
    start: float
    history: list[Observation]
    controls: ControlTape
    targets: list[Observation]
    metadata: dict
