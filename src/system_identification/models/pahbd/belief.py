"""Availability-ordered, mask/age/quality-aware observation correction on SO(3).

Decision-time bounded replay uses only already published observations, corrected
at measurement time. This is an approximate learned filter, not a Bayesian smoother.
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
        noise=event.variance.clamp_min(self.cfg.variance_min)
        gain=torch.sigmoid(raw[...,:14])*s.variance/(s.variance+noise)*mask
        correction=gain*residual
        x=so3.retract(s.x,correction[...,:12])
        gate=torch.sigmoid(raw[...,-1:])*mask.amax(-1,keepdim=True)
        h=(1-gate)*s.h+gate*self.cfg.latent_bound*torch.tanh(raw[...,14:-1])
        var=((1-gain)**2*s.variance+gain**2*noise).clamp(self.cfg.variance_min,self.cfg.variance_max)
        cov=torch.diag_embed(s.variance[...,12:14]) if s.phase_covariance is None else s.phase_covariance
        # Joseph update with a masked diagonal learned gain preserves cross covariance.
        K=torch.diag_embed(gain[...,12:14]);I=torch.eye(2,device=K.device,dtype=K.dtype)
        cov=(I-K)@cov@(I-K).transpose(-1,-2)+K@torch.diag_embed(noise[...,12:14])@K.transpose(-1,-2)
        var=torch.cat((var[...,:12],cov.diagonal(dim1=-2,dim2=-1)),-1)
        return s.with_(x=x,h=h,f=s.f+correction[...,12:13],phi=s.phi+correction[...,13:14],variance=var,phase_covariance=cov)

    def forward(self,flow,initial,start,events,controls,decision=0.):
        if start>decision:raise ValueError('history starts after decision')
        if decision-start > self.cfg.replay_seconds+1e-6:
            raise ValueError('history exceeds replay buffer capacity')
        # Batch the final replay at the decision. Sorting by measurement time is
        # equivalent to inserting each arrived reading into a timestamped buffer
        # and replaying its suffix, including all later accepted corrections.
        # Filtering BEFORE reading values prevents unpublished data leakage.
        published=[e for e in events if start<=e.availability<=decision]
        if any(e.measurement>e.availability+1e-12 for e in published):
            raise ValueError('measurement occurs after availability')
        available=sorted((e for e in published if e.measurement>=start and
                          e.availability-e.measurement<=self.cfg.replay_seconds),
                         key=lambda e:(e.measurement,e.availability,e.source))
        self.last_replay=dict(accepted=len(available),rejected=len(published)-len(available),
                              max_delay=max((e.availability-e.measurement for e in available),default=0.))
        s=initial;t=start;last_update=start
        for e in available:
            s=flow.propagate(s,t,e.measurement,controls)
            s=self.update(s,e,e.measurement-last_update)
            t=e.measurement;last_update=t
        return flow.propagate(s,t,decision,controls)
