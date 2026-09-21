"""Memoryless paper baseline using the existing Main V1 integration contract."""
from __future__ import annotations

import torch
from torch import nn

from system_identification.models.neural import MLPRegressor
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel


class _NoMemory(nn.Module):
    def forward(self, inputs, hidden):
        return torch.zeros_like(hidden)


class _CurrentInputHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLPRegressor(16, 7, (64, 64), dropout=0.0)
        # Match the recurrent dynamics' initial normalized derivative of zero.
        nn.init.zeros_(self.mlp.network[-1].weight)
        nn.init.zeros_(self.mlp.network[-1].bias)

    def forward(self, hidden_and_input):
        return self.mlp(hidden_and_input[:, -16:])


class MemorylessTrajectoryModel(CausalHistoryTrajectoryModel):
    """Current physical features + current command; no history or hidden state.

    Inherit the exact derivative scaling, clipping, phase handling, and physical
    integration of the existing GRU. The zero-valued placeholder is never read
    by the MLP and has no parameters. No actuator proxy is used.
    """
    def __init__(self, **stats):
        super().__init__(hidden_size=64, use_controls=True, **stats)
        self.recurrent_cell = _NoMemory()
        self.derivative_head = _CurrentInputHead()

    def _encode_history(self, state_features, controls, mask):
        return state_features.new_zeros((len(state_features), self.hidden_size))
