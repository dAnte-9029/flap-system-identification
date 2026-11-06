"""PACLD runtime state; physical coordinates follow the shared PA-HBD contracts."""
from dataclasses import dataclass, replace
import torch
from ..pahbd.types import Observation, ControlTape, Window


@dataclass
class State:
    x: torch.Tensor  # (...,13), p_N[m], v_N[m/s], q_NB[wxyz], omega_B[rad/s]
    z_s: torch.Tensor  # (...,12), dimensionless
    c: torch.Tensor  # (...,6), dimensionless, three 2D envelopes
    f: torch.Tensor  # (...,1), Hz
    phi: torch.Tensor  # (...,1), unwrapped Hall mechanical radians
    actuator: torch.Tensor  # (...,3), normalized tail states
    variance: torch.Tensor  # (...,14), tangent physical-state/frequency/phase proxy
    phase_quality: torch.Tensor  # (...,1), causal reference confidence
    phase_covariance: torch.Tensor | None = None
    time: float = 0.  # decision-relative seconds
    last_measurement: float | None = None
    last_availability: float | None = None

    def with_(self, **kwargs):
        return replace(self, **kwargs)


def initial_state(window, config):
    """Thin adapter: reuse causal physical bootstrap, NEVER import PA-HBD h or targets."""
    s = window.initial
    return State(s.x, s.x.new_zeros((*s.x.shape[:-1], config.slow_dim)),
                 s.x.new_zeros((*s.x.shape[:-1], config.fast_dim)),
                 s.f, s.phi, s.actuator, s.variance, s.phase_quality,
                 s.phase_covariance, time=window.start)
