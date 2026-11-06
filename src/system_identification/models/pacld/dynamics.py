"""Coupled latent vector field, with a single nonlinear physical decoder."""
from dataclasses import dataclass, fields
import math
import torch
from torch import nn
from ..pahbd import geometry as so3
from ..pahbd.dynamics import PhasePropagator


@dataclass
class Config:
    slow_dim: int = 12
    fast_dim: int = 6
    width: int = 128
    max_step: float = .005
    actuator_tau: float = .06
    slow_rate_limit: float = 10.
    envelope_bound: float = 2.
    lambda_min: float = .5
    lambda_max: float = 20.
    coupling_bound: float = 1.
    initial_slow_bound: float = 2.
    replay_seconds: float = .65
    variance_min: float = 1e-5
    variance_max: float = 25.
    frequency_noise_psd: float = .04
    phase_noise_psd: float = .01
    frequency_coefficients: tuple = (3.6354511983566393, 23.393570068972757, -5.126725395181777)

    def __post_init__(self):
        if (self.slow_dim, self.fast_dim) != (12, 6):
            raise ValueError('PACLD v1 requires 12 slow and three 2D fast modes')
        if not isinstance(self.width, int) or self.width < 1:
            raise ValueError('positive integer width required')
        for field in fields(self):
            v = getattr(self, field.name)
            if isinstance(v, (int, float)) and not math.isfinite(v):
                raise ValueError('finite configuration required')
        if min(self.max_step, self.actuator_tau, self.slow_rate_limit,
               self.envelope_bound, self.lambda_min, self.coupling_bound,
               self.initial_slow_bound, self.replay_seconds, self.variance_min) <= 0:
            raise ValueError('positive numerical bounds required')
        if self.lambda_max <= self.lambda_min or self.variance_max <= self.variance_min:
            raise ValueError('invalid bound interval')
        if min(self.frequency_noise_psd, self.phase_noise_psd) < 0:
            raise ValueError('negative process PSD')
        if len(self.frequency_coefficients) != 3 or not all(math.isfinite(v) for v in self.frequency_coefficients):
            raise ValueError('three finite frozen frequency coefficients required')


def mlp(inputs, outputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.SiLU(),
                         nn.Linear(width, width), nn.SiLU(), nn.Linear(width, outputs))


def phase_carrier(phi, c):
    """Q(phi)c: block rotations by phi, 2phi, 3phi; no phase-zero parameter."""
    angle = phi * torch.arange(1, 4, dtype=phi.dtype, device=phi.device)
    pairs = c.reshape(*c.shape[:-1], 3, 2)
    a, b = pairs.unbind(-1)
    return torch.stack((angle.cos()*a-angle.sin()*b,
                        angle.sin()*a+angle.cos()*b), -1).flatten(-2)


def physical_features(x):
    """Dimensionless [v_B/10 m/s, omega_B/3 rad/s, gravity direction_B]."""
    inverse = so3.conjugate(x[..., 6:10])
    down = torch.zeros_like(x[..., :3])
    down[..., 2] = 1.
    return torch.cat((so3.rotate(inverse, x[..., 3:6])/10., x[..., 10:13]/3.,
                      so3.rotate(inverse, down)), -1)


class CoupledDynamics(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # z_s, chi(x), actuator, motor, f/5Hz; no sin/cos or phi.
        context_dim = cfg.slow_dim + 9 + 3 + 1 + 1
        self.slow = mlp(context_dim + cfg.fast_dim, cfg.slow_dim, cfg.width)
        self.equilibrium = mlp(context_dim, cfg.fast_dim, cfg.width)
        self.relaxation = mlp(context_dim, cfg.fast_dim, cfg.width)
        self.coupling = mlp(context_dim, cfg.fast_dim * 3, cfg.width)
        self.decoder = mlp(context_dim + 2*cfg.fast_dim, 6, cfg.width)
        # Nominal level-flight initialization, not a force/mass calibration.
        with torch.no_grad():
            self.decoder[-1].bias[2] = -9.80665

    def context(self, s, u):
        return torch.cat((s.z_s, physical_features(s.x), s.actuator,
                          u[..., :1], s.f/5.), -1)

    def latent_rates(self, s, u):
        context = self.context(s, u)
        eq = self.cfg.envelope_bound * torch.tanh(self.equilibrium(context))
        rate = self.cfg.lambda_min + (self.cfg.lambda_max-self.cfg.lambda_min)*torch.sigmoid(self.relaxation(context))
        coupling = self.cfg.coupling_bound*torch.tanh(self.coupling(context)).reshape(*context.shape[:-1], 6, 3)
        da = (u[..., 1:]-s.actuator)/self.cfg.actuator_tau
        dc = rate*(eq-s.c) + (coupling @ da.unsqueeze(-1)).squeeze(-1)
        dz = self.cfg.slow_rate_limit*torch.tanh(self.slow(torch.cat((context, s.c), -1)))
        return dz, dc

    def decode(self, s, u, mode='FULL'):
        if mode not in ('BASE', 'FULL'):
            raise ValueError('mode must be BASE or FULL')
        zf = phase_carrier(s.phi, s.c)
        # With no accepted Hall reference there is no meaningful explicit carrier.
        zf = zf*s.phase_quality if mode == 'FULL' else torch.zeros_like(zf)
        return self.decoder(torch.cat((self.context(s, u), s.c, zf), -1))

    def forward(self, s, u, mode='FULL'):
        d = self.decode(s, u, mode)
        gravity = torch.zeros_like(d[..., :3])
        gravity[..., 2] = 9.80665
        dz, dc = self.latent_rates(s, u)
        return gravity+so3.rotate(s.x[..., 6:10], d[..., :3]), d[..., 3:], dz, dc
