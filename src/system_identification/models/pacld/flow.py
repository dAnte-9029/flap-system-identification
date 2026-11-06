"""Common Lie-midpoint integrator for BASE/FULL and causal history replay."""
import bisect
import math
import torch
from torch import nn
from ..pahbd import geometry as so3
from ..pahbd.dynamics import PhasePropagator
from .dynamics import CoupledDynamics


class LieGroupFlow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.phase = PhasePropagator(cfg.frequency_coefficients)
        self.dynamics = CoupledDynamics(cfg)
        self.process_logit = nn.Parameter(torch.full((12,), -6.))

    def step(self, s, u, dt, mode='FULL'):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('positive finite step required')
        x = s.x
        acc, alpha, dz, dc = self.dynamics(s, u, mode)
        fm, pm = self.phase(s.f, s.phi, u[..., :1], dt/2)
        am = u[..., 1:] + (s.actuator-u[..., 1:])*math.exp(-dt/(2*self.cfg.actuator_tau))
        midx = torch.cat((x[..., :3]+x[..., 3:6]*dt/2, x[..., 3:6]+acc*dt/2,
                          so3.unit(so3.multiply(x[..., 6:10], so3.exp(x[..., 10:13]*dt/2))),
                          x[..., 10:13]+alpha*dt/2), -1)
        mid = s.with_(x=midx, z_s=s.z_s+dz*dt/2, c=s.c+dc*dt/2,
                      f=fm, phi=pm, actuator=am)
        acc, alpha, dz, dc = self.dynamics(mid, u, mode)
        newx = torch.cat((x[..., :3]+midx[..., 3:6]*dt, x[..., 3:6]+acc*dt,
                          so3.unit(so3.multiply(x[..., 6:10], so3.exp(midx[..., 10:13]*dt))),
                          x[..., 10:13]+alpha*dt), -1)
        f, phi = self.phase(s.f, s.phi, u[..., :1], dt)
        actuator = u[..., 1:] + (s.actuator-u[..., 1:])*math.exp(-dt/self.cfg.actuator_tau)
        cov = torch.diag_embed(s.variance[..., 12:14]) if s.phase_covariance is None else s.phase_covariance
        noise = x.new_tensor([self.cfg.frequency_noise_psd, self.cfg.phase_noise_psd]).diag()
        cov = self.phase.covariance(cov, dt, noise)
        variance = (s.variance[..., :12]+torch.sigmoid(self.process_logit)*dt).clamp(self.cfg.variance_min, self.cfg.variance_max)
        return s.with_(x=newx, z_s=s.z_s+dz*dt, c=s.c+dc*dt, f=f, phi=phi,
                       actuator=actuator, variance=torch.cat((variance, cov.diagonal(dim1=-2, dim2=-1)), -1),
                       phase_covariance=cov, time=s.time+dt)

    def _propagate(self, s, start, end, tape, mode):
        if mode not in ('BASE', 'FULL'):
            raise ValueError('mode must be BASE or FULL')
        if not all(math.isfinite(t) for t in (start, end, *tape.times)):
            raise ValueError('finite times required')
        if end < start:
            raise ValueError('backwards propagation forbidden')
        if abs(s.time-start) > 1e-9:
            raise ValueError('state time and propagation start disagree')
        i = bisect.bisect_right(tape.times, start)-1
        if i < 0:
            raise ValueError('no causal control at integration start')
        t = start
        while t < end:
            boundary = min(end, tape.times[i+1] if i+1 < len(tape.times) else end)
            span = boundary-t
            n = max(1, math.ceil(span/self.cfg.max_step))
            for _ in range(n):
                s = self.step(s, tape.values[i], span/n, mode)
            t = boundary
            if i+1 < len(tape.times) and t >= tape.times[i+1]:
                i += 1
        return s.with_(time=end)

    def propagate(self, s, start, end, controls, mode='FULL'):
        return self._propagate(s, start, end, controls.canonical(), mode)

    def rollout(self, s, times, controls, start=0., mode='FULL'):
        # Canonicalize once; advance one trajectory through native query timestamps.
        tape = controls.canonical()
        out = []
        t = start
        for end in times:
            s = self._propagate(s, t, end, tape, mode)
            out.append(s)
            t = end
        return out
