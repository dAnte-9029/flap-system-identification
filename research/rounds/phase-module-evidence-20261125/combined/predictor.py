"""Load and apply the first-order phase correction and the second/third-order transport."""
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
sys.path.insert(0, str(ROUND / "coherence"))
from module import MomentPredictor, prior, causal_conditions
sys.path.insert(0, str(ROUND / "waveform/coherence"))
from inference import HarmonicTransport


class WingbeatPredictor(nn.Module):
    def __init__(self, first_order, high_order):
        super().__init__()
        self.first_order = first_order
        self.high_order = high_order

    @classmethod
    def from_seed(cls, seed, device="cpu"):
        root = ROUND / "coherence"
        state = torch.load(root / f"conditional_seed{seed}.pt", map_location="cpu", weights_only=True)
        low = MomentPredictor(state["mean"], state["std"])
        low.load_state_dict(state)
        with np.load(root / f"heads_seed{seed}.npz") as f, torch.no_grad():
            low.weights.copy_(torch.from_numpy(f["direction"]))
        return cls(low, HarmonicTransport.from_files(seed)).to(device).eval()

    def contributions(self, history, pooled, initial_command, time, harmonic_coefficients):
        low = self.first_order.forecast(history, pooled, initial_command, time, arm="direction")
        theta = self.first_order.clocks(history, pooled, initial_command, time)[..., -1]
        conditions = causal_conditions(history, pooled, initial_command, time)
        high = self.high_order(harmonic_coefficients, time, theta, conditions)
        return low, high

    def forward(self, history, pooled, initial_command, time, harmonic_coefficients):
        low, high = self.contributions(history, pooled, initial_command, time, harmonic_coefficients)
        return low + high
