"""Two directional scale couplings on top of a native-query flight predictor."""
import numpy as np
import torch
from torch import nn


def encode_history(history, main_rate):
    """Fit trend plus three harmonic residuals in decision-relative phase.

    Input [window,26,30] follows history_slow_fast_v1.prepare.history.
    Return [window,stream,axis,8] with trend intercept, half-second slope,
    then the cosine/sine quadratures of orders 1,2,3, and stream eligibility.
    """
    time = np.broadcast_to(np.linspace(-1., 0., history.shape[1]), history.shape[:2])
    phase = np.arctan2(history[..., 13], history[..., 14])
    columns = [np.ones_like(time), time]
    for order in (1, 2, 3):
        columns.extend([np.cos(order * phase), np.sin(order * phase)])
    basis = np.stack(columns, -1)
    coeff = np.zeros((len(time), 2, 3, 8), np.float32)
    valid = np.zeros((len(time), 2), bool)
    for stream in (0, 1):
        mask = (history[..., 21] > 0) & (history[..., 16+stream] > 0)
        if stream == 0:
            mask &= (history[..., 18] > 0) & main_rate[:, None]
        design = basis * mask[..., None]
        gram = design.transpose(0, 2, 1) @ design
        eig = np.linalg.eigvalsh(gram)
        ok = (mask.sum(1) >= 16) & mask[:, -1] & (eig[:, 0] > .0001*eig[:, -1])
        rhs = design.transpose(0, 2, 1) @ history[..., 3*stream:3*stream+3]
        coeff[ok, stream] = np.linalg.solve(gram[ok], rhs[ok]).transpose(0, 2, 1)
        valid[:, stream] = ok
    return coeff, valid


def features(coeff, valid, frequency, motor, time, mean_frequency, arm):
    """Design [window, horizon, stream, axis, coordinate]; each row depends on its own time."""
    level, slope = coeff[...,0], coeff[...,1]
    # The history mask is identical in the two source-routing controls.
    clip_l, clip_s = torch.tanh(level), torch.tanh(slope)
    if arm.endswith('self'):
        message = torch.stack((clip_l, clip_s, clip_l*slope, clip_s*level,
                               clip_l.square(), clip_s.square()), -1)
    elif arm.endswith('deleted'):
        message = coeff[..., 2:2]
    else:
        message = coeff[..., 2:]
    message = message * valid[:,:,None,None]
    one = torch.ones_like(level)
    common = torch.stack((one, level, slope, one*frequency[:,None,None],
                          one*motor[:,None,None]), -1)
    drive = torch.cat((common, message), -1)
    timescales = time.new_tensor([.15, .6, 2.])
    step = -torch.expm1(-time[...,None] / timescales)
    smooth = (drive[:,None,:,:,:,None] * step[:,:,:,None,None,:]).flatten(-2)

    hz = mean_frequency.expand_as(frequency) if arm.startswith('fixed') else frequency
    phase = 2*torch.pi*time[...,None]*hz[:,None,None,None]*time.new_tensor([1.,2.,3.])
    a,b = coeff[...,2:].unflatten(-1,(3,2)).unbind(-1)
    co,si = phase.cos().unsqueeze(-2), phase.sin().unsqueeze(-2)
    rot = torch.stack((a[:,None]*co + b[:,None]*si, -a[:,None]*si + b[:,None]*co), -1)
    envelope = (-torch.expm1(-time[...,None]/.02))*torch.exp(-time[...,None]/timescales)
    fast = (rot[...,None] * envelope[:,:,:,None,None,None,:]).flatten(-3)
    return torch.cat((smooth, fast), -1)


class ScaleExchange(nn.Module):
    """Frozen native ODE forecast plus the learned per-axis endpoint correction."""
    def __init__(self, weights, mean_frequency, arm='full'):
        super().__init__()
        self.arm = arm
        self.register_buffer('weights', weights)
        self.register_buffer('mean_frequency', mean_frequency)

    def forward(self, base, coeff, valid, frequency, motor, time):
        x = features(coeff, valid, frequency, motor, time, self.mean_frequency, self.arm)
        return base + (x*self.weights).sum(-1)
