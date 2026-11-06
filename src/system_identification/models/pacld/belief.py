"""Shared causal history features and separate Slow/Fast inference heads.

Measurement-time replay follows PA-HBD's bounded decision-time filter. This is
an approximate learned filter, not a Bayesian smoother or independent truth.
"""
import bisect
import math
import torch
from torch import nn
from ..pahbd import geometry as so3
from .dynamics import physical_features


class CausalBeliefEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # residual/mask/quality/log variance, age/dt, z_s/c, chi, a, motor, f.
        inputs = 14*4+2+cfg.slow_dim+cfg.fast_dim+9+3+1+1
        self.shared = nn.Sequential(nn.Linear(inputs, cfg.width), nn.SiLU(),
                                    nn.Linear(cfg.width, cfg.width), nn.SiLU())
        self.slow_head = nn.Linear(cfg.width, cfg.slow_dim)
        self.fast_head = nn.Linear(cfg.width, cfg.fast_dim)
        self.correction_head = nn.Linear(cfg.width, 14)
        self.gate_head = nn.Linear(cfg.width, 2)
        self.register_buffer('residual_scale', torch.tensor([10.]*6+[1.]*3+[3.]*3+[5., math.pi]))

    def update(self, s, event, dt, control):
        if event.frame != 'NED/FRD/Hamilton-wxyz':
            raise ValueError('unsupported observation frame')
        if event.measurement > event.availability or dt < 0:
            raise ValueError('invalid observation time order')
        if abs(s.time-event.measurement) > 1e-9:
            raise ValueError('correction must occur at measurement time')
        if any(not torch.isfinite(v).all() for v in (event.value, event.mask, event.quality, event.variance)):
            raise ValueError('observations must be finite and masked values sanitized')
        mask = event.mask.to(s.x)*event.quality.clamp(0, 1)
        # Batch-safe first reliable Hall reference, never a zero-quality anchor.
        anchor = (mask[..., 13:14] > 0) & (s.phase_quality <= 0)
        if event.source != 'wing_phase':
            anchor = torch.zeros_like(anchor)
        noise = event.variance.clamp_min(self.cfg.variance_min)
        cov = torch.diag_embed(s.variance[..., 12:14]) if s.phase_covariance is None else s.phase_covariance
        anchor_cov = torch.diag_embed(torch.cat((s.variance[..., 12:13], noise[..., 13:14]), -1))
        cov = torch.where(anchor.unsqueeze(-1), anchor_cov, cov)
        variance = torch.cat((s.variance[..., :12], cov.diagonal(dim1=-2, dim2=-1)), -1)
        s = s.with_(phi=torch.where(anchor, event.value[..., 14:15], s.phi),
                    phase_quality=torch.where(anchor, mask[..., 13:14], s.phase_quality),
                    variance=variance, phase_covariance=cov)
        residual = torch.cat((so3.state_residual(s.x, event.value[..., :13]),
                              event.value[..., 13:14]-s.f,
                              torch.atan2(torch.sin(event.value[..., 14:15]-s.phi),
                                          torch.cos(event.value[..., 14:15]-s.phi))), -1)*mask
        ones = torch.ones_like(s.f)
        features = torch.cat((residual/self.residual_scale, event.mask, event.quality,
                              torch.log(s.variance.clamp_min(self.cfg.variance_min)),
                              ones*(event.availability-event.measurement), ones*dt,
                              s.z_s/self.cfg.initial_slow_bound, s.c/self.cfg.envelope_bound,
                              physical_features(s.x), s.actuator, control[..., :1], s.f/5.), -1)
        shared = self.shared(features)
        gain = torch.sigmoid(self.correction_head(shared))*s.variance/(s.variance+noise)*mask
        gain = torch.cat((gain[..., :13], torch.where(anchor, torch.zeros_like(gain[..., 13:14]), gain[..., 13:14])), -1)
        correction = gain*residual
        gates = torch.sigmoid(self.gate_head(shared))*mask.amax(-1, keepdim=True)
        zs = (1-gates[..., :1])*s.z_s + gates[..., :1]*self.cfg.initial_slow_bound*torch.tanh(self.slow_head(shared))
        c = (1-gates[..., 1:])*s.c + gates[..., 1:]*self.cfg.envelope_bound*torch.tanh(self.fast_head(shared))
        variance = ((1-gain)**2*s.variance+gain**2*noise).clamp(self.cfg.variance_min, self.cfg.variance_max)
        K = torch.diag_embed(gain[..., 12:14])
        I = torch.eye(2, device=K.device, dtype=K.dtype)
        cov = (I-K)@cov@(I-K).transpose(-1, -2)+K@torch.diag_embed(noise[..., 12:14])@K.transpose(-1, -2)
        variance = torch.cat((variance[..., :12], cov.diagonal(dim1=-2, dim2=-1)), -1)
        return s.with_(x=so3.retract(s.x, correction[..., :12]), z_s=zs, c=c,
                       f=s.f+correction[..., 12:13], phi=s.phi+correction[..., 13:14],
                       variance=variance, phase_covariance=cov,
                       last_measurement=event.measurement, last_availability=event.availability)

    def forward(self, flow, initial, start, events, controls, decision=0.):
        if not all(math.isfinite(t) for t in (start, decision)) or start > decision:
            raise ValueError('invalid history interval')
        if decision-start > self.cfg.replay_seconds+1e-6:
            raise ValueError('history exceeds replay buffer capacity')
        # Filter by publication time BEFORE reading any observation values.
        published = [e for e in events if start <= e.availability <= decision]
        if any(not math.isfinite(e.measurement) or e.measurement > e.availability for e in published):
            raise ValueError('invalid measurement time')
        available = sorted((e for e in published if e.measurement >= start and
                            e.availability-e.measurement <= self.cfg.replay_seconds),
                           key=lambda e: (e.measurement, e.availability, e.source))
        self.last_replay = dict(accepted=len(available), rejected=len(published)-len(available))
        tape = controls.canonical()
        s, t, last_update = initial, start, start
        for e in available:
            s = flow._propagate(s, t, e.measurement, tape, 'FULL')
            i = bisect.bisect_right(tape.times, e.measurement)-1
            s = self.update(s, e, e.measurement-last_update, tape.values[i])
            t = last_update = e.measurement
        return flow._propagate(s, t, decision, tape, 'FULL')
