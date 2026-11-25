"""Conditional first circular moment of residual Hall phase, paired with a frozen dynamics readout."""
import itertools
from pathlib import Path
import importlib.util

import torch

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parents[1] / 'periodic-motion-20261125'
spec = importlib.util.spec_from_file_location('periodic_predecessor', PRIOR / 'readout/model.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)

ARMS = ('global', 'direction', 'magnitude', 'conditional', 'sham')
PAIRS = tuple(itertools.combinations_with_replacement(range(4), 2))


def build_design(t, dynamic):
    # Fixed physical scales avoid using any validation data to normalize.
    scale = t.new_tensor([0.5, 0.12, 0.2, 0.1])
    offsets = t.new_tensor([0., 0.15, 0., 0.])
    z = ((dynamic - offsets) / scale).clamp(-3, 3)
    features = [torch.ones_like(t), *z.unbind(-1)]
    features.extend(z[..., i] * z[..., j] for i, j in PAIRS)
    times = torch.stack((t, t.square(), t.pow(3)), -1)
    return (torch.stack(features, -1).unsqueeze(-1) * times.unsqueeze(-2)).flatten(-2)


def unit_projection(z):
    return z / torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp_min(1.)


def complex_unit(z):
    norm = torch.linalg.vector_norm(z, dim=-1, keepdim=True)
    return torch.where(norm > 1e-10, z / norm.clamp_min(1e-10), z.new_tensor([1., 0.]))


def circular_factors(design, parameters):
    origin = design.new_tensor([1., 0.])
    m_global = unit_projection(origin + design[..., :3] @ parameters['global'])
    m_cond = unit_projection(origin + design @ parameters['conditional'])
    m_sham = unit_projection(origin + design @ parameters['sham'])
    r_global = torch.linalg.vector_norm(m_global, dim=-1, keepdim=True)
    r_cond = torch.linalg.vector_norm(m_cond, dim=-1, keepdim=True)
    return dict(global_=m_global, direction=r_global * complex_unit(m_cond),
                magnitude=r_cond * complex_unit(m_global), conditional=m_cond, sham=m_sham)


def causal_conditions(hist, pooled, u0, t):
    seq = hist[:, :180].reshape(-1, 6, 30)
    measured = seq[..., 22]
    recent = (seq[..., 15] * measured).sum(-1) / measured.sum(-1).clamp_min(1.)
    f = seq[:, -1, 15, None, None]
    motor = u0[:, 0, None, None]
    mass = -torch.expm1(-t)
    motor_average = pooled[..., 8] / torch.where(t > 0, mass, 1.)
    return torch.stack(torch.broadcast_tensors(f - 3.8, motor - .5,
                        f - recent[:, None, None], motor_average - motor), -1)


class MomentPredictor(prior.FlightPredictor):
    def __init__(self, mean, std):
        super().__init__(mean, std)
        self.register_buffer('time_moment', torch.zeros(3, 2))
        self.register_buffer('conditional_moment', torch.zeros(45, 2))
        self.register_buffer('sham_moment', torch.zeros(45, 2))

    def coefficients(self):
        return {'global': self.time_moment, 'conditional': self.conditional_moment, 'sham': self.sham_moment}

    def bases(self, hist, pooled, u0, time):
        theta = self.clocks(hist, pooled, u0, time)[..., 2]
        slow = -torch.expm1(-time / .35)
        cosine, sine = theta.cos(), theta.sin()
        design = build_design(time, causal_conditions(hist, pooled, u0, time))
        moments = circular_factors(design, self.coefficients())
        moments['global'] = moments.pop('global_')
        out = {}
        for arm, moment in moments.items():
            real, imag = moment.unbind(-1)
            c = real * cosine - imag * sine
            s = real * sine + imag * cosine
            out[arm] = torch.stack((slow, c - 1, s), -1)
        return out

    def components(self, hist, pooled, u0, time, arm='conditional'):
        coef = torch.einsum('bh,ckha->bcka', self.context(hist), self.weights)
        basis = self.bases(hist, pooled, u0, time)[arm]
        base = self.core(hist[:, :self.mean.numel()], pooled, u0, time)
        slow = base + basis[..., 0, None] * coef[:, None, :, 0, :]
        origin = coef[:, None, :, 1, :]
        wave = ((basis[..., 1, None] + 1) * origin
                + basis[..., 2, None] * coef[:, None, :, 2, :])
        return slow, wave, origin

    def forecast(self, hist, pooled, u0, time, arm='conditional'):
        ctx = self.context(hist)
        coef = torch.einsum('bh,ckha->bcka', ctx, self.weights)
        basis = self.bases(hist, pooled, u0, time)[arm]
        base = self.core(hist[:, :self.mean.numel()], pooled, u0, time)
        return base + torch.einsum('bqck,bcka->bqca', basis, coef)

    def forward(self, hist, pooled, u0, time):
        return self.forecast(hist, pooled, u0, time)
