"""Add a learned, order-specific phase-conditioned motion increment to any compatible predictor."""
import itertools
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PAIRS = tuple(itertools.combinations_with_replacement(range(4), 2))


def causal_conditions(hist, pooled, u0, time):
    """Use only the origin and available command prefix, in the H2 calibration units."""
    h = hist[:, :180].reshape(-1, 6, 30)
    valid = h[..., 22]
    recent = (h[..., 15] * valid).sum(-1) / valid.sum(-1).clamp_min(1.)
    freq = h[:, -1, 15, None, None]
    motor = u0[:, 0, None, None]
    mass = -torch.expm1(-time)
    avg = pooled[..., 8] / torch.where(time > 0, mass, 1.)
    return torch.stack(torch.broadcast_tensors(freq - 3.8, motor - .5,
                        freq - recent[:, None, None], avg - motor), -1)


def moment_basis(time, conditions):
    # Identical feature order and scalings to the fitted shared Hall calibration.
    shifted = conditions - time.new_tensor([0., .15, 0., 0.])
    z = (shifted / time.new_tensor([.5, .12, .2, .1])).clamp(-3, 3)
    cols = [torch.ones_like(time), *z.unbind(-1)]
    cols.extend(z[..., i] * z[..., j] for i, j in PAIRS)
    powers = torch.stack((time, time.square(), time.pow(3)), -1)
    return (torch.stack(cols, -1).unsqueeze(-1) * powers.unsqueeze(-2)).flatten(-2)


class HarmonicTransport(nn.Module):
    """Pure additive head. Time is in seconds, propagated phase in radians relative to decision time."""

    def __init__(self, moment_coefficients, readout_coefficients):
        super().__init__()
        if moment_coefficients.shape != (45, 4) or readout_coefficients.shape != (2, 3, 12):
            raise ValueError("Expected moment parameters [45,4] and readout parameters [2,3,12].")
        self.register_buffer("moment_coefficients", moment_coefficients.float())
        self.register_buffer("readout_coefficients", readout_coefficients.float())
        self.register_buffer("harmonic_orders", torch.tensor([2., 3.]))
        self.register_buffer("decay_times", torch.tensor([.25, 1., 4.]))

    @classmethod
    def from_files(cls, seed, directory=None, device="cpu"):
        root = HERE if directory is None else Path(directory)
        with np.load(root / "clock.npz") as clock, np.load(root / f"weights_{seed}.npz") as head:
            model = cls(torch.from_numpy(clock["conditional"]), torch.from_numpy(head["conditional"]))
        return model.to(device).eval()

    def forward(self, coefficients, time, phase, conditions):
        """[B,2,3,2,2], [B,Q,2], [B,Q,2], [B,Q,2,4] -> [B,Q,2,3].

        The streams are velocity and angular rate. The last dimensions of coefficients
        are harmonic order (2,3) and the two quadratures (cosine,sine). Set coefficients
        to zero for ineligible histories. Each query uses only its own phase and conditions.
        """
        basis = moment_basis(time, conditions)
        moments = (basis @ self.moment_coefficients).reshape(*time.shape, 2, 2)
        moments = moments + moments.new_tensor([1., 0.])
        moments = moments / torch.linalg.vector_norm(moments, dim=-1, keepdim=True).clamp_min(1.)
        angle = (phase[..., None] * self.harmonic_orders).unsqueeze(-2)
        a, b = coefficients.unbind(-1)
        c = a[:, None] * angle.cos() + b[:, None] * angle.sin()
        q = -a[:, None] * angle.sin() + b[:, None] * angle.cos()
        real, imag = moments.unbind(-1)
        real, imag = real.unsqueeze(-2), imag.unsqueeze(-2)
        corrected = torch.stack((real * c + imag * q, real * q - imag * c), -1)
        envelope = -torch.expm1(-time / .02)[..., None] * torch.exp(-time[..., None] / self.decay_times)
        features = (corrected[..., None] * envelope[:, :, :, None, None, None, :]).flatten(-3)
        return torch.einsum("bqcxm,cxm->bqcx", features, self.readout_coefficients)
