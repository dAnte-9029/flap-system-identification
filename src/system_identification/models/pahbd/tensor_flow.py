"""Opt-in execution experiment. Reference flow.py and checkpoint layout stay unchanged.

Tensor inputs retain all state and gradients. Host event scheduling is copied
verbatim from the reference, and blocks never cross its control/query boundaries.
"""
import math
from collections import Counter
import torch
from torch import nn
from . import geometry as so3


def unpack(s,flow):
    return (s.x,s.h,s.f,s.phi,s.actuator,s.variance,s.phase_quality,flow.phase_covariance(s))


def repack(s,t):
    return s.with_(x=t[0],h=t[1],f=t[2],phi=t[3],actuator=t[4],variance=t[5],phase_quality=t[6],phase_covariance=t[7])


class TensorStep(nn.Module):
    def __init__(self,flow):
        super().__init__();self.cfg=flow.cfg
        self.slow=flow.slow;self.mixer=flow.mixer;self.process_logit=flow.process_logit
        self.linear_frequency=abs(flow.phase.host_coefficients()[2])<1e-9

    def derivative(self,x,h,f,phi,actuator,quality,cov):
        cfg=self.cfg;q=x[...,6:10];g=torch.zeros_like(x[...,:3]);g[...,2]=9.80665
        features=torch.cat((so3.rotate(so3.conjugate(q),x[...,3:6]),x[...,10:13],
                            so3.rotate(so3.conjugate(q),g),f),-1)
        c=self.slow.encoder(torch.cat(((features-self.slow.input_mean)/self.slow.input_scale,h/cfg.latent_bound),-1))
        target=cfg.latent_bound*torch.tanh(self.slow.memory(torch.cat((c,actuator),-1)))
        rate=cfg.latent_rate*torch.sigmoid(self.slow.retention)
        dh=rate*(target-h)/(2*cfg.latent_bound)
        lead=c.shape[:-1];raw=self.mixer.mean(c);mean=raw[...,:6];B=raw[...,6:].reshape(*lead,6,3)
        F=self.mixer.forcing(c).reshape(*lead,cfg.forcing_harmonics,2,6)
        left=self.mixer.control(c).reshape(*lead,cfg.control_harmonics,2,6*cfg.rank)
        if torch.compiler.is_compiling():
            F=self.mixer.rotate_coefficients(F,self.mixer.gauge_offset)
            left=self.mixer.rotate_coefficients(left,self.mixer.gauge_offset)
        else:
            F=self.mixer._rotate(F);left=self.mixer._rotate(left)
        left=left.reshape(*lead,cfg.control_harmonics,2,6,cfg.rank)
        G=torch.einsum('...hkor,hrj->...hkoj',left,self.mixer.right)
        da=actuator-self.mixer.reference_control
        avg=mean+torch.einsum('...ij,...j->...i',B,da)
        def basis(H):
            k=torch.arange(1,H+1,device=phi.device,dtype=phi.dtype)
            result=torch.stack((torch.cos(phi*k),torch.sin(phi*k)),-1)
            if cfg.uncertainty_mode=='gaussian':result=result*torch.exp(-.5*k.square()*cov[...,1,1:2])[...,None]
            return result
        enabled=quality if cfg.phase_mode=='relative' else torch.zeros_like(quality)
        fb=basis(cfg.forcing_harmonics)
        forcing=torch.einsum('...hk,...hko->...o',fb,F)*enabled
        cb=fb if cfg.control_harmonics==cfg.forcing_harmonics else basis(cfg.control_harmonics)
        interaction=torch.einsum('...hk,...hkoj,...j->...o',cb,G,da)*enabled
        if cfg.mode=='B':interaction=interaction*0
        d=avg+forcing+interaction
        gravity=torch.zeros_like(d[...,:3]);gravity[...,2]=9.80665
        return gravity+so3.rotate(q,d[...,:3]),d[...,3:],dh

    def phase(self,f,phi,motor,dt,b0,bu,bf,em1,e):
        c=b0+bu*motor
        if self.linear_frequency:return f+c*dt,phi+2*math.pi*(f*dt+.5*c*dt*dt)
        steady=-c/bf
        integral=steady*dt+(f-steady)*em1/bf
        return steady+(f-steady)*e,phi+2*math.pi*integral

    @staticmethod
    def covariance(c,F,D):
        result=F@c@F.T+D
        return (result+result.transpose(-1,-2))/2

    def step(self,t,u,k,Fmid,Dmid,Fend,Dend):
        x,h,f,phi,actuator,variance,quality,cov=t
        dt,decay_mid,decay_end,b0,bu,bf,em1mid,expmid,em1end,expend=k.unbind()
        a,alpha,dh=self.derivative(x,h,f,phi,actuator,quality,cov)
        fmid,pmid=self.phase(f,phi,u[...,:1],dt/2,b0,bu,bf,em1mid,expmid)
        amid=u[...,1:]+(actuator-u[...,1:])*decay_mid
        midx=torch.cat((x[...,:3]+x[...,3:6]*dt/2,x[...,3:6]+a*dt/2,
                       so3.unit(so3.multiply(x[...,6:10],so3.exp(x[...,10:13]*dt/2))),x[...,10:13]+alpha*dt/2),-1)
        midcov=self.covariance(cov,Fmid,Dmid)
        am,alpham,dhm=self.derivative(midx,h+dh*dt/2,fmid,pmid,amid,quality,midcov)
        newx=torch.cat((x[...,:3]+midx[...,3:6]*dt,x[...,3:6]+am*dt,
                       so3.unit(so3.multiply(x[...,6:10],so3.exp(midx[...,10:13]*dt))),x[...,10:13]+alpham*dt),-1)
        nf,np=self.phase(f,phi,u[...,:1],dt,b0,bu,bf,em1end,expend)
        na=u[...,1:]+(actuator-u[...,1:])*decay_end
        var=(variance+torch.sigmoid(self.process_logit)*dt).clamp(self.cfg.variance_min,self.cfg.variance_max)
        ncov=self.covariance(cov,Fend,Dend)
        var=torch.cat((var[...,:12],ncov.diagonal(dim1=-2,dim2=-1)),-1)
        return newx,h+dhm*dt,nf,np,na,var,quality,ncov

    def block4(self,t,u,*args):
        for _ in range(4):t=self.step(t,u,*args)
        return t

    def block8(self,t,u,*args):
        for _ in range(8):t=self.step(t,u,*args)
        return t


class TensorEngine:
    def __init__(self,flow,kind='tensor',compile_mode='default'):
        self.flow=flow;self.kernel=TensorStep(flow);self.kind=kind
        self.block_size={'tensor':1,'net':1,'step':1,'block4':4,'block8':8}[kind]
        self.constants={};self.segments=Counter();self.calls=Counter();self.trace=None
        self.single=self.kernel.step
        self.block=getattr(self.kernel,kind) if kind in ('block4','block8') else self.single
        if kind=='net':self.kernel.derivative=torch.compile(self.kernel.derivative,fullgraph=True,dynamic=False,mode=compile_mode)
        elif kind!='tensor':
            self.single=torch.compile(self.kernel.step,fullgraph=True,dynamic=False,mode=compile_mode)
            self.block=self.single if kind=='step' else torch.compile(self.block,fullgraph=True,dynamic=False,mode=compile_mode)

    def prepare(self,dt,like):
        flow=self.flow
        key=(dt,like.device,like.dtype,id(flow.phase.coefficients),flow.phase.coefficients._version,
             flow.cfg.frequency_noise_psd,flow.cfg.phase_noise_psd,flow.cfg.actuator_tau)
        if key not in self.constants:
            if flow.phase.coefficients.requires_grad:raise ValueError('execution engine requires the frozen phase coefficients contract')
            b0,bu,bf=flow.phase.coefficients.to(like).unbind()
            k=torch.stack((like.new_tensor(dt),like.new_tensor(math.exp(-dt/(2*flow.cfg.actuator_tau))),
                           like.new_tensor(math.exp(-dt/flow.cfg.actuator_tau)),b0,bu,bf,
                           torch.expm1(bf*(dt/2)),torch.exp(bf*(dt/2)),torch.expm1(bf*dt),torch.exp(bf*dt)))
            q=like.new_tensor([flow.cfg.frequency_noise_psd,flow.cfg.phase_noise_psd]).diag()
            Fmid,_=flow.phase.transition(dt/2,like);Fend,_=flow.phase.transition(dt,like)
            Dmid=flow.phase.covariance(like.new_zeros(2,2),dt/2,q,validated=True)
            Dend=flow.phase.covariance(like.new_zeros(2,2),dt,q,validated=True)
            if len(self.constants)>=4096:self.constants.clear()
            self.constants[key]=(k,Fmid,Dmid,Fend,Dend)
        return self.constants[key]

    def propagate(self,s,start,end,controls):
        flow=self.flow
        if end<start-1e-12:raise ValueError('backwards propagation forbidden')
        tape=controls.canonical()
        for i,span in tape.intervals(start,end):
            u=tape.values[i]
            if getattr(flow,'_frequency_tensor',None) is not s.f:
                flow._frequency_host=s.f.detach().cpu().numpy().copy()
            b0,bu,bf=flow.phase.host_coefficients()
            motor=tape._cpu_values[i,...,0:1].numpy();f0=flow._frequency_host
            steady=-(b0+bu*motor)/bf if abs(bf)>=1e-9 else None
            fend=steady+(f0-steady)*math.exp(bf*span) if steady is not None else f0+(b0+bu*motor)*span
            fmax=max(float(abs(f0).max()),float(abs(fend).max()),1.)
            H=max(flow.cfg.forcing_harmonics,flow.cfg.control_harmonics)
            step=min(flow.cfg.max_step,flow.cfg.phase_step/(2*math.pi*H*fmax),flow.cfg.latent_bound/flow.cfg.latent_rate)
            n=max(1,math.ceil(span/step));dt=span/n
            self.segments[n]+=1
            if self.trace is not None:self.trace.extend([dt]*n)
            t=unpack(s,flow);args=self.prepare(dt,s.x)
            for _ in range(n//self.block_size):
                t=self.block(t,u,*args);self.calls[self.block_size]+=1
            for _ in range(n%self.block_size):
                t=self.single(t,u,*args);self.calls[1]+=1
            # Preserve the exact reference host frequency recurrence, including rounding.
            for _ in range(n):
                flow._frequency_host=steady+(flow._frequency_host-steady)*math.exp(bf*dt) if steady is not None else flow._frequency_host+(b0+bu*motor)*dt
            s=repack(s,t);flow._frequency_tensor=s.f
        return s


def enable_tensor_engine(model,kind='tensor',compile_mode='default'):
    before=tuple(model.state_dict())
    engine=TensorEngine(model.flow,kind,compile_mode)
    # Plain engine object: do not register aliased model parameters a second time.
    model.flow._tensor_engine=engine
    model.flow.propagate=engine.propagate
    assert tuple(model.state_dict())==before
    return engine
