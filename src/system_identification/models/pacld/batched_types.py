"""Independent tensor runtime representation for a batch of native windows."""
from dataclasses import dataclass, replace
import torch

@dataclass
class BatchedState:
    # Leading dimension is window (or concatenated FULL/BASE window). Native
    # clocks live in the CPU-compiled plan; no Python scalar time is read on GPU.
    x: torch.Tensor
    z_s: torch.Tensor
    c: torch.Tensor
    f: torch.Tensor
    phi: torch.Tensor
    actuator: torch.Tensor
    variance: torch.Tensor
    phase_quality: torch.Tensor
    phase_covariance: torch.Tensor

    def with_(self, **kwargs):
        return replace(self, **kwargs)
