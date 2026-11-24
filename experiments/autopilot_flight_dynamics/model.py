"""Previously retained centered command-reference baseline extended to all eight endpoints."""
import torch
from torch import nn


class FlightPredictor(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("tau", torch.tensor([0.2, 1.0]))
        self.encoder = nn.Sequential(nn.Linear(len(mean), 64), nn.SiLU(), nn.Linear(64, 48), nn.SiLU())
        self.nominal = nn.Linear(48, 6 * 3)
        self.coupling = nn.Linear(48, 6 * 16)

    def forward(self, hist, pooled, u0, time):
        context = self.encoder((hist - self.mean) / self.std)
        nominal = self.nominal(context).reshape(-1, 2, 3, 3)
        coupling = self.coupling(context).reshape(-1, 2, 3, 16)
        basis = torch.stack((time, -torch.expm1(-time / 0.2), -torch.expm1(-time)), dim=-1)
        reference = torch.cat((u0, torch.tanh(2 * u0)), dim=-1)
        mass = -torch.expm1(-time.unsqueeze(-1) / self.tau)
        held = (mass.unsqueeze(-1) * reference[:, None, None, None, :]).flatten(-2)
        return (torch.einsum("nhcb,ncdb->nhcd", basis, nominal)
                + torch.einsum("nhcb,ncdb->nhcd", pooled - held, coupling))
