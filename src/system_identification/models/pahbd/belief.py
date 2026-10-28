"""Availability-ordered, mask/age/quality-aware observation correction on SO(3).

Delayed readings use a zero-order observation model with age-inflated noise.
This is a causal filter, not an exact delayed-state Bayesian smoother.
"""
import torch
from torch import nn
from . import geometry as so3


class CausalBeliefEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        # residual, mask, quality, log variance, age, event dt, current h
        self.update_net=nn.Sequential(nn.Linear(14*4+2+cfg.latent,cfg.width),nn.SiLU(),
                                      nn.Linear(cfg.width,14+cfg.latent+1))
        self.register_buffer('residual_scale',torch.tensor([10.]*3+[10.]*3+[1.]*3+[3.]*3+[5.,3.14159265]))
        self.register_buffer('age_noise_rate',torch.tensor([10.]*3+[5.]*3+[3.]*3+[10.]*3+[10.,40.]))

    def update(self,s,event,dt):
        if event.measurement > event.availability+1e-12:
            raise ValueError('measurement occurs after availability')
        if not torch.isfinite(event.value).all():raise ValueError('masked values must be sanitized')
        mask=event.mask.to(s.x)*event.quality.clamp(0,1)
        age=event.availability-event.measurement
        residual=torch.cat((so3.state_residual(s.x,event.value[...,:13]),
                            event.value[...,13:14]-s.f,
                            torch.atan2(torch.sin(event.value[...,14:15]-s.phi),
                                        torch.cos(event.value[...,14:15]-s.phi))),-1)
        residual=residual*mask
        extras=torch.ones_like(s.f)
        features=torch.cat((residual/self.residual_scale,event.mask,event.quality,
                            torch.log(s.variance.clamp_min(self.cfg.variance_min)),
                            extras*age,extras*dt,s.h/self.cfg.latent_bound),-1)
        raw=self.update_net(features)
        noise=event.variance.clamp_min(self.cfg.variance_min)+(self.age_noise_rate*age)**2
        gain=torch.sigmoid(raw[...,:14])*s.variance/(s.variance+noise)*mask
        correction=gain*residual
        x=so3.retract(s.x,correction[...,:12])
        gate=torch.sigmoid(raw[...,-1:])*mask.amax(-1,keepdim=True)
        h=(1-gate)*s.h+gate*self.cfg.latent_bound*torch.tanh(raw[...,14:-1])
        var=((1-gain)**2*s.variance+gain**2*noise).clamp(self.cfg.variance_min,self.cfg.variance_max)
        return s.with_(x=x,h=h,f=s.f+correction[...,12:13],phi=s.phi+correction[...,13:14],variance=var)

    def forward(self,flow,initial,start,events,controls,decision=0.):
        if start>decision:raise ValueError('history starts after decision')
        s=initial;t=start;last_update=start
        # Filtering occurs before sorting/using values: post-decision events cannot influence belief.
        available=sorted((e for e in events if start<=e.availability<=decision),key=lambda e:e.availability)
        for e in available:
            s=flow.propagate(s,t,e.availability,controls)
            s=self.update(s,e,e.availability-last_update)
            t=e.availability;last_update=t
        return flow.propagate(s,t,decision,controls)
