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
        c,dh=self.slow(s)
        cov=self.phase_covariance(s)
        d,_=self.mixer(c,s.actuator,s.phi,s.phase_quality,cov[...,1,1:2])
        g=torch.zeros_like(d[...,:3]);g[...,2]=9.80665
        return g+so3.rotate(s.x[...,6:10],d[...,:3]),d[...,3:],dh

    @staticmethod
    def phase_covariance(s):
        return torch.diag_embed(s.variance[...,12:14]) if s.phase_covariance is None else s.phase_covariance

    def propagate_covariance(self,s,dt):
        key=(dt,s.x.device,s.x.dtype,self.cfg.frequency_noise_psd,self.cfg.phase_noise_psd,id(self.phase.coefficients),self.phase.coefficients._version)
        if not hasattr(self,'_noise_cache'):self._noise_cache={}
        if key not in self._noise_cache:
            q=s.x.new_tensor([self.cfg.frequency_noise_psd,self.cfg.phase_noise_psd]).diag()
            F,_=self.phase.transition(dt,s.x)
            D=self.phase.covariance(s.x.new_zeros(2,2),dt,q,validated=True)
            if len(self._noise_cache)>=4096:self._noise_cache.clear()
            self._noise_cache[key]=(F,D)
        F,D=self._noise_cache[key]
        result=F@self.phase_covariance(s)@F.T+D
        return (result+result.transpose(-1,-2))/2

    def step(self,s,u,dt):
        if dt <= 0:raise ValueError('positive step required')
        cfg=self.cfg;x=s.x
        a,alpha,dh=self.derivative(s)
        fmid,pmid=self.phase(s.f,s.phi,u[...,:1],dt/2)
        amid=u[...,1:]+(s.actuator-u[...,1:])*math.exp(-dt/(2*cfg.actuator_tau))
        midx=torch.cat((x[...,:3]+x[...,3:6]*dt/2,x[...,3:6]+a*dt/2,
                        so3.unit(so3.multiply(x[...,6:10],so3.exp(x[...,10:13]*dt/2))),
                        x[...,10:13]+alpha*dt/2),-1)
        mid=s.with_(x=midx,h=s.h+dh*dt/2,f=fmid,phi=pmid,actuator=amid,phase_covariance=self.propagate_covariance(s,dt/2))
        am,alpham,dhm=self.derivative(mid)
        newx=torch.cat((x[...,:3]+midx[...,3:6]*dt,x[...,3:6]+am*dt,
                        so3.unit(so3.multiply(x[...,6:10],so3.exp(midx[...,10:13]*dt))),
                        x[...,10:13]+alpham*dt),-1)
        f,phi=self.phase(s.f,s.phi,u[...,:1],dt)
        actuator=u[...,1:]+(s.actuator-u[...,1:])*math.exp(-dt/cfg.actuator_tau)
        # Rigid-body diagonal proxy plus separately propagated full f/phi covariance.
        variance=(s.variance+torch.sigmoid(self.process_logit)*dt).clamp(cfg.variance_min,cfg.variance_max)
        covariance=self.propagate_covariance(s,dt)
        variance=torch.cat((variance[...,:12],covariance.diagonal(dim1=-2,dim2=-1)),-1)
        return s.with_(x=newx,h=s.h+dhm*dt,f=f,phi=phi,actuator=actuator,variance=variance,phase_covariance=covariance)

    def propagate(self,s,start,end,controls):
        if end < start-1e-12:raise ValueError('backwards propagation forbidden')
        tape=controls.canonical()
        for i,span in tape.intervals(start,end):
            u=tape.values[i]
            # One host transfer after a learned belief correction, then analytic
            # scalar scheduling through uninterrupted autonomous propagation.
            if getattr(self,'_frequency_tensor',None) is not s.f:
                self._frequency_host=s.f.detach().cpu().numpy().copy()
            b0,bu,bf=self.phase.host_coefficients()
            motor=tape._cpu_values[i,...,0:1].numpy()
            f0=self._frequency_host
            steady=-(b0+bu*motor)/bf if abs(bf)>=1e-9 else None
            fend=steady+(f0-steady)*math.exp(bf*span) if steady is not None else f0+(b0+bu*motor)*span
            fmax=max(float(abs(f0).max()),float(abs(fend).max()),1.)
            H=max(self.cfg.forcing_harmonics,self.cfg.control_harmonics)
            step=min(self.cfg.max_step,self.cfg.phase_step/(2*math.pi*H*fmax),
                     self.cfg.latent_bound/self.cfg.latent_rate)
            n=max(1,math.ceil(span/step));dt=span/n
            for _ in range(n):
                s=self.step(s,u,dt)
                self._frequency_host=steady+(self._frequency_host-steady)*math.exp(bf*dt) if steady is not None else self._frequency_host+(b0+bu*motor)*dt
            self._frequency_tensor=s.f
        return s

    def rollout(self,s,times,controls,start=0.):
        out=[];t=start
        for end in times:
            if end < t:raise ValueError('rollout queries must be sorted')
            s=self.propagate(s,t,end,controls);out.append(s);t=end
        return out
