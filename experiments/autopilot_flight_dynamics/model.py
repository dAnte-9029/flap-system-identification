"""Predict trajectories by integrating history-conditioned, interval-local latent rates."""
import torch
from torch import nn


class FlightPredictor(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("tau", mean.new_tensor([0.2, 1.0]))
        latent = 32
        self.encoder = nn.Sequential(nn.Linear(len(mean), 64), nn.SiLU(), nn.Linear(64, latent))
        # 20 command features, native interval duration, midpoint and two decaying clocks.
        inputs = latent + 20 + 4
        self.field = nn.Sequential(nn.Linear(inputs, 48), nn.Tanh(), nn.Linear(48, latent))
        self.field_linear = nn.Linear(inputs, latent)
        self.readout = nn.Sequential(nn.Linear(latent, 16), nn.Tanh(), nn.Linear(16, 6, bias=False))
        self.readout_linear = nn.Linear(latent, 6, bias=False)

    def interval_features(self, pooled, u0, time):
        previous = torch.cat((torch.zeros_like(time[:, :1]), time[:, :-1]), dim=1)
        dt = (time - previous).unsqueeze(-1)
        midpoint = ((time + previous) * 0.5).unsqueeze(-1)
        prefix = pooled.unflatten(-1, (2, 8))
        prior = torch.cat((torch.zeros_like(prefix[:, :1]), prefix[:, :-1]), dim=1)
        decay = torch.exp(-dt / self.tau)
        mass = -torch.expm1(-dt / self.tau)
        average = (prefix - decay.unsqueeze(-1) * prior) / mass.unsqueeze(-1)
        held = torch.cat((u0, torch.tanh(2 * u0)), dim=-1)
        centered = (average - held[:, None, None, None]).flatten(-2)
        command = torch.cat((u0[:, None, None].expand(-1, time.shape[1], 2, -1), centered), dim=-1)
        return dt, torch.cat((command, dt, midpoint, torch.exp(-midpoint / self.tau)), dim=-1)

    def rate(self, state, forcing):
        features = torch.cat((state, forcing), dim=-1)
        return self.field_linear(features) + self.field(features)

    def observe(self, state):
        return self.readout_linear(state) + self.readout(state)

    def forward(self, hist, pooled, u0, time):
        initial = self.encoder((hist - self.mean) / self.std)[:, None, None]
        dt, forcing = self.interval_features(pooled, u0, time)
        # All local rates query the same initial state. Prefix addition enforces causality.
        query = initial.expand(-1, time.shape[1], 2, -1)
        states = initial + torch.cumsum(dt * self.rate(query, forcing), dim=1)
        change = self.observe(states) - self.observe(initial)
        return torch.stack((change[:, :, 0, :3], change[:, :, 1, 3:]), dim=2)
