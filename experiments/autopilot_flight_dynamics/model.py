"""Controlled slow dynamics and an origin-conditioned, decaying motion carrier."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class FlightPredictor(nn.Module):
    carrier_basis = "slow"

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
        self.memory_rate = nn.Linear(16, latent, bias=False)
        nn.init.zeros_(self.memory_rate.weight)
        # Six 30-channel snapshots, followed by native-history harmonic fits and masks.
        # All models, including the matched time-basis control, receive the same history.
        self.carrier_coefficients = nn.Linear(latent + len(mean) - 180, 12)
        nn.init.zeros_(self.carrier_coefficients.weight)
        nn.init.zeros_(self.carrier_coefficients.bias)
        # One learned coherence rate per physical axis, shared by the two quadratures.
        # Start at a one-second e-folding time. Positive decay makes the envelope
        # uncertain far from the last measured wingbeat.
        self.decay_logits = nn.Parameter(mean.new_full((2, 3), math.log(math.expm1(1.))))

    def interval_features(self, pooled, u0, time):
        previous = torch.cat((torch.zeros_like(time[:, :1]), time[:, :-1]), dim=1)
        dt = (time - previous).unsqueeze(-1)
        midpoint = ((time + previous) * 0.5).unsqueeze(-1)
        prefix = pooled.unflatten(-1, (2, 8))
        prior = torch.cat((torch.zeros_like(prefix[:, :1]), prefix[:, :-1]), dim=1)
        decay = torch.exp(-dt / self.tau)
        mass = -torch.expm1(-dt / self.tau)
        # Zero-duration queries include the exact origin in component diagnostics.
        safe_mass = torch.where(dt > 0, mass, torch.ones_like(mass))
        average = (prefix - decay.unsqueeze(-1) * prior) / safe_mass.unsqueeze(-1)
        held = torch.cat((u0, torch.tanh(2 * u0)), dim=-1)
        centered = (average - held[:, None, None, None]).flatten(-2)
        command = torch.cat((u0[:, None, None].expand(-1, time.shape[1], 2, -1), centered), dim=-1)
        return dt, torch.cat((command, dt, midpoint, torch.exp(-midpoint / self.tau)), dim=-1)

    def command_moments(self, centered_command, time):
        # For each native clock, integrate piecewise-constant interval means on [0,T].
        # time: (batch, query, clock). Normalized edges past each endpoint both equal 1,
        # so those intervals get exactly zero weight at that query.
        edges = time.transpose(1, 2)
        denom = torch.where(edges > 0, edges, torch.ones_like(edges))
        upper = (edges.unsqueeze(-2) / denom.unsqueeze(-1)).clamp(max=1.)
        upper = torch.where(edges.unsqueeze(-1) > 0, upper, 0.)
        # Primitives of P_k(2s-1) on [0,1], including orthonormal scale sqrt(2k+1).
        product = upper * (upper - 1.)
        cumulative = torch.stack((upper,
                                  3.**.5 * product,
                                  5.**.5 * product * (2. * upper - 1.),
                                  7.**.5 * product * (5. * product + 1.)), dim=-2)
        bounds = torch.cat((torch.zeros_like(cumulative[..., :1]), cumulative), dim=-1)
        weights = bounds[..., 1:] - bounds[..., :-1]
        signal = centered_command.transpose(1, 2)
        return torch.einsum("bcqkj,bcjd->bqckd", weights, signal)

    def memory_features(self, moments):
        c0, c1 = moments[..., :2, :].unbind(dim=-2)
        features = (c0, c1, torch.tanh(2 * c0), torch.tanh(2 * c1))
        return torch.cat(features, dim=-1)

    def rate(self, state, forcing):
        features = torch.cat((state, forcing), dim=-1)
        return self.field_linear(features) + self.field(features)

    def observe(self, state):
        return self.readout_linear(state) + self.readout(state)

    def slow_motion(self, initial, pooled, u0, time):
        dt, forcing = self.interval_features(pooled, u0, time)
        # The four raw channels of the tau=1 s exponential interval summary.
        moments = self.command_moments(forcing[..., 12:16], time)
        memory_rate = self.memory_rate(self.memory_features(moments))
        # Each interval queries the latent state reached at its own native start time.
        state = initial[:, 0].expand(-1, 2, -1)
        trajectory = []
        for duration, command, extra_rate in zip(dt.unbind(1), forcing.unbind(1), memory_rate.unbind(1)):
            state = state + duration * (self.rate(state, command) + extra_rate)
            trajectory.append(state)
        states = torch.stack(trajectory, dim=1)
        change = self.observe(states) - self.observe(initial)
        slow_delta = torch.stack((change[:, :, 0, :3], change[:, :, 1, 3:]), dim=2)
        return torch.where(time.unsqueeze(-1) > 0, slow_delta, 0.)

    def motion_carrier(self, initial, standardized, hist, time):
        # hist[..., 150:180] is the decision-time snapshot. Its phase is relative
        # to this same origin; the network estimates quadratures in that window's
        # phase convention from the causal history. Only the origin frequency is
        # extrapolated. No measured future phase or motion enters this branch.
        context = torch.cat((initial, standardized[:, 180:]), dim=-1)
        coefficients = self.carrier_coefficients(context).reshape(-1, 2, 3, 2)
        phase_valid = hist[:, 171] * hist[:, 172]
        coefficients = coefficients * phase_valid[:, None, None, None]
        frequency = hist[:, 165]
        if self.carrier_basis == "slow":
            # Across a two-second forecast this traverses half a cycle. Sine and
            # cosine stay independent and use both sets of learned coefficients.
            frequency = torch.full_like(frequency, 0.25)
        angle = (2 * math.pi * frequency[:, None, None]) * time
        quadratures = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
        wave = (coefficients[:, None] * quadratures.unsqueeze(-2)).sum(-1)
        envelope = torch.exp(-F.softplus(self.decay_logits) * time.unsqueeze(-1))
        return envelope * wave, coefficients[:, None, :, :, 0]

    def components(self, hist, pooled, u0, time):
        """Return slow delta, absolute carrier, and origin carrier in native v/rate frames."""
        standardized = (hist - self.mean) / self.std
        initial = self.encoder(standardized)
        slow_delta = self.slow_motion(initial[:, None, None], pooled, u0, time)
        wave, origin_wave = self.motion_carrier(initial, standardized, hist, time)
        return slow_delta, wave, origin_wave

    def forward(self, hist, pooled, u0, time):
        slow_delta, wave, origin_wave = self.components(hist, pooled, u0, time)
        return slow_delta + (wave - origin_wave)
