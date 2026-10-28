"""One differentiable Lie-midpoint flow for both history prediction and future rollout."""
import bisect
import math
import torch
from torch import nn
from . import geometry as so3
from .dynamics import PhasePropagator, SlowBackbone, ControlMixer


class LieGroupFlow(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        self.phase=PhasePropagator(cfg.frequency_coefficients)
        self.slow=SlowBackbone(cfg);self.mixer=ControlMixer(cfg)
        self.process_logit=nn.Parameter(torch.full((14,),-6.))

    def derivative(self,s):
        c,dh=self.slow(s);d,_=self.mixer(c,s.actuator,s.phi,s.phase_quality)
        g=torch.zeros_like(d[...,:3]);g[...,2]=9.80665
        return g+so3.rotate(s.x[...,6:10],d[...,:3]),d[...,3:],dh

    def step(self,s,u,dt):
        if dt <= 0:raise ValueError('positive step required')
        cfg=self.cfg;x=s.x
        a,alpha,dh=self.derivative(s)
        fmid,pmid=self.phase(s.f,s.phi,u[...,:1],dt/2)
        amid=u[...,1:]+(s.actuator-u[...,1:])*math.exp(-dt/(2*cfg.actuator_tau))
        midx=torch.cat((x[...,:3]+x[...,3:6]*dt/2,x[...,3:6]+a*dt/2,
                        so3.unit(so3.multiply(x[...,6:10],so3.exp(x[...,10:13]*dt/2))),
                        x[...,10:13]+alpha*dt/2),-1)
        mid=s.with_(x=midx,h=s.h+dh*dt/2,f=fmid,phi=pmid,actuator=amid)
        am,alpham,dhm=self.derivative(mid)
        newx=torch.cat((x[...,:3]+midx[...,3:6]*dt,x[...,3:6]+am*dt,
                        so3.unit(so3.multiply(x[...,6:10],so3.exp(midx[...,10:13]*dt))),
                        x[...,10:13]+alpham*dt),-1)
        f,phi=self.phase(s.f,s.phi,u[...,:1],dt)
        actuator=u[...,1:]+(s.actuator-u[...,1:])*math.exp(-dt/cfg.actuator_tau)
        # Bounded diagonal proxy; full f/phi covariance available in phase.covariance.
        variance=(s.variance+torch.sigmoid(self.process_logit)*dt).clamp(cfg.variance_min,cfg.variance_max)
        return s.with_(x=newx,h=s.h+dhm*dt,f=f,phi=phi,actuator=actuator,variance=variance)

    def propagate(self,s,start,end,controls):
        if end < start-1e-12:raise ValueError('backwards propagation forbidden')
        tape=controls.canonical()
        i=bisect.bisect_right(tape.times,start+1e-12)-1
        if i<0:raise ValueError('no causal control at integration start')
        t=start
        while t < end-1e-12:
            boundary=min(end,tape.times[i+1] if i+1<len(tape.times) else end)
            span=boundary-t
            if span <= 1e-12:i+=1;continue
            u=tape.values[i]
            fend,_=self.phase(s.f,s.phi,u[...,:1],span)
            fmax=max(float(s.f.detach().abs().max()),float(fend.detach().abs().max()),1.)
            H=max(self.cfg.forcing_harmonics,self.cfg.control_harmonics)
            step=min(self.cfg.max_step,self.cfg.phase_step/(2*math.pi*H*fmax),
                     self.cfg.latent_bound/self.cfg.latent_rate)
            n=max(1,math.ceil(span/step));dt=span/n
            for _ in range(n):s=self.step(s,u,dt)
            t=boundary
            if i+1<len(tape.times) and t>=tape.times[i+1]-1e-12:i+=1
        return s

    def rollout(self,s,times,controls,start=0.):
        out=[];t=start
        for end in times:
            if end < t:raise ValueError('rollout queries must be sorted')
            s=self.propagate(s,t,end,controls);out.append(s);t=end
        return out
