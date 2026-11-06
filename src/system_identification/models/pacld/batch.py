"""Tensor batch PACLD execution. The CPU single-window reference stays untouched."""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from . import PACLD
from .batched_types import BatchedState as State
from .dynamics import physical_features
from ..pahbd import geometry as so3

FIELDS = ('x','z_s','c','f','phi','actuator','variance','phase_quality','phase_covariance')


def select(valid, new, old):
    return old.with_(**{k:torch.where(valid.unsqueeze(-1) if k=='phase_covariance' else valid,getattr(new,k),getattr(old,k)) for k in FIELDS})


def phase(coefficients, f, phi, motor, dt):
    b0,bu,bf=coefficients.unbind()
    c=b0+bu*motor
    # Frozen configuration is nonzero; zero-frequency coefficient is handled explicitly outside tracing.
    steady=-c/bf
    return steady+(f-steady)*torch.exp(bf*dt), phi+2*math.pi*(steady*dt+(f-steady)*torch.expm1(bf*dt)/bf)


def covariance(cov, dt, bf, qf, qp):
    # Exact integral of exp(A t) diag(qf,qp) exp(A.T t); no matrix_exp or device readback.
    t=dt.squeeze(-1); e=torch.exp(bf*t); h=torch.expm1(bf*t)/bf
    h2=torch.expm1(2*bf*t)/(2*bf); k=2*math.pi/bf
    zero=torch.zeros_like(t); one=torch.ones_like(t)
    F=torch.stack((e,zero,2*math.pi*h,one),-1).reshape(*t.shape,2,2)
    Q=torch.stack((qf*h2,qf*k*(h2-h),qf*k*(h2-h),qf*k*k*(h2-2*h+t)+qp*t),-1).reshape(*t.shape,2,2)
    result=F@cov@F.transpose(-1,-2)+Q
    return (result+result.transpose(-1,-2))/2


class BatchedPACLD(PACLD):
    def __init__(self, config=None, precision='fp64', compile_blocks=False, checkpoint_steps=False):
        super().__init__(config)
        if abs(self.config.frequency_coefficients[2])<1e-9: raise ValueError('batched phase requires the frozen nonzero frequency coefficient')
        self.double()
        self.precision=precision
        if precision not in ('fp64','mixed','fp32'): raise ValueError(precision)
        if precision=='mixed':
            self.flow.dynamics.float()
            for name in ('shared','slow_head','fast_head','correction_head','gate_head'): getattr(self.belief,name).float()
        elif precision=='fp32': self.float()
        self.checkpoint_steps=checkpoint_steps
        self.step_kernel=torch.compile(self.tensor_step,fullgraph=True,dynamic=False) if compile_blocks else self.tensor_step
        self.update_kernel=torch.compile(self.tensor_update,fullgraph=True,dynamic=False) if compile_blocks else self.tensor_update

    @staticmethod
    def net(module, x):
        return module(x.to(next(module.parameters()).dtype)).to(x.dtype)

    def tensor_update(self, s, data):
        event=data
        control=event['control']
        original=s
        # Reference encoder parameters; only execution and dtype routing change.
        encoder=self.belief
        return self._update(encoder,s,event,control,original)

    def _update(self, encoder, s, event, control, original):
        self_cfg=self.config
        mask = event['mask'].to(s.x)*event['quality'].clamp(0, 1)
        # Batch-safe first reliable Hall reference, never a zero-quality anchor.
        anchor = (mask[..., 13:14] > 0) & (s.phase_quality <= 0)
        anchor = anchor & event['hall']
        noise = event['variance'].clamp_min(self_cfg.variance_min)
        cov = torch.diag_embed(s.variance[..., 12:14]) if s.phase_covariance is None else s.phase_covariance
        anchor_cov = torch.diag_embed(torch.cat((s.variance[..., 12:13], noise[..., 13:14]), -1))
        cov = torch.where(anchor.unsqueeze(-1), anchor_cov, cov)
        variance = torch.cat((s.variance[..., :12], cov.diagonal(dim1=-2, dim2=-1)), -1)
        s = s.with_(phi=torch.where(anchor, event['value'][..., 14:15], s.phi),
                    phase_quality=torch.where(anchor, mask[..., 13:14], s.phase_quality),
                    variance=variance, phase_covariance=cov)
        residual = torch.cat((so3.state_residual(s.x, event['value'][..., :13]),
                              event['value'][..., 13:14]-s.f,
                              torch.atan2(torch.sin(event['value'][..., 14:15]-s.phi),
                                          torch.cos(event['value'][..., 14:15]-s.phi))), -1)*mask
        ones = torch.ones_like(s.f)
        features = torch.cat((residual/encoder.residual_scale, event['mask'], event['quality'],
                              torch.log(s.variance.clamp_min(self_cfg.variance_min)),
                              event['age'], event['elapsed'],
                              s.z_s/self_cfg.initial_slow_bound, s.c/self_cfg.envelope_bound,
                              physical_features(s.x), s.actuator, control[..., :1], s.f/5.), -1)
        shared = self.net(encoder.shared, features)
        gain = torch.sigmoid(self.net(encoder.correction_head, shared))*s.variance/(s.variance+noise)*mask
        gain = torch.cat((gain[..., :13], torch.where(anchor, torch.zeros_like(gain[..., 13:14]), gain[..., 13:14])), -1)
        correction = gain*residual
        gates = torch.sigmoid(self.net(encoder.gate_head, shared))*mask.amax(-1, keepdim=True)
        zs = (1-gates[..., :1])*s.z_s + gates[..., :1]*self_cfg.initial_slow_bound*torch.tanh(self.net(encoder.slow_head, shared))
        c = (1-gates[..., 1:])*s.c + gates[..., 1:]*self_cfg.envelope_bound*torch.tanh(self.net(encoder.fast_head, shared))
        variance = ((1-gain)**2*s.variance+gain**2*noise).clamp(self_cfg.variance_min, self_cfg.variance_max)
        K = torch.diag_embed(gain[..., 12:14])
        I = torch.eye(2, device=K.device, dtype=K.dtype)
        cov = (I-K)@cov@(I-K).transpose(-1, -2)+K@torch.diag_embed(noise[..., 12:14])@K.transpose(-1, -2)
        variance = torch.cat((variance[..., :12], cov.diagonal(dim1=-2, dim2=-1)), -1)
        result = s.with_(x=so3.retract(s.x, correction[..., :12]), z_s=zs, c=c,
                       f=s.f+correction[..., 12:13], phi=s.phi+correction[..., 13:14],
                       variance=variance, phase_covariance=cov)

        return select(event['valid'],result,original)

    def vector_field(self, s, u):
        dtype=next(self.flow.dynamics.parameters()).dtype
        low=s.with_(**{k:getattr(s,k).to(dtype) for k in FIELDS})
        return tuple(v.to(s.x.dtype) for v in self.flow.dynamics(low,u.to(dtype),'FULL'))

    def tensor_step(self, s, u, dt):
        x = s.x
        acc, alpha, dz, dc = self.vector_field(s, u)
        fm, pm = phase(self.flow.phase.coefficients, s.f, s.phi, u[..., :1], dt/2)
        am = u[..., 1:] + (s.actuator-u[..., 1:])*torch.exp(-dt/(2*self.config.actuator_tau))
        midx = torch.cat((x[..., :3]+x[..., 3:6]*dt/2, x[..., 3:6]+acc*dt/2,
                          so3.unit(so3.multiply(x[..., 6:10], so3.exp(x[..., 10:13]*dt/2))),
                          x[..., 10:13]+alpha*dt/2), -1)
        mid = s.with_(x=midx, z_s=s.z_s+dz*dt/2, c=s.c+dc*dt/2,
                      f=fm, phi=pm, actuator=am)
        acc, alpha, dz, dc = self.vector_field(mid, u)
        newx = torch.cat((x[..., :3]+midx[..., 3:6]*dt, x[..., 3:6]+acc*dt,
                          so3.unit(so3.multiply(x[..., 6:10], so3.exp(midx[..., 10:13]*dt))),
                          x[..., 10:13]+alpha*dt), -1)
        f, phi = phase(self.flow.phase.coefficients, s.f, s.phi, u[..., :1], dt)
        actuator = u[..., 1:] + (s.actuator-u[..., 1:])*torch.exp(-dt/self.config.actuator_tau)
        cov = torch.diag_embed(s.variance[..., 12:14]) if s.phase_covariance is None else s.phase_covariance
        cov = covariance(cov, dt, self.flow.phase.coefficients[2], self.config.frequency_noise_psd, self.config.phase_noise_psd)
        variance = (s.variance[..., :12]+torch.sigmoid(self.flow.process_logit)*dt).clamp(self.config.variance_min, self.config.variance_max)
        result = s.with_(x=newx, z_s=s.z_s+dz*dt, c=s.c+dc*dt, f=f, phi=phi,
                       actuator=actuator, variance=torch.cat((variance, cov.diagonal(dim1=-2, dim2=-1)), -1),
                       phase_covariance=cov)

        return select(dt>0,result,s)

    def propagate_plan(self, state, plan):
        dt,u=plan
        for i in range(dt.shape[0]):
            if self.checkpoint_steps and torch.is_grad_enabled():
                def fn(*args):
                    s=State(**dict(zip(FIELDS,args[:-2])))
                    out=self.step_kernel(s,args[-2],args[-1])
                    return tuple(getattr(out,k) for k in FIELDS)
                values=checkpoint(fn,*(getattr(state,k) for k in FIELDS),u[i],dt[i],use_reentrant=False)
                state=State(**dict(zip(FIELDS,values)))
            else: state=self.step_kernel(state,u[i],dt[i])
        return state

    def encode_batch(self, batch):
        state=batch.initial
        for plan,event in batch.history:
            state=self.propagate_plan(state,plan)
            if event is not None: state=self.update_kernel(state,event)
        return state

    def predict_batch(self,batch):
        """Return B FULL rows, plus B shared-model BASE rows only with Macro pairs."""
        with torch.profiler.record_function("pacld/history"):
            belief=self.encode_batch(batch)
        # [FULL B, BASE B] share every parameter and their initial history graph.
        has_macro=batch.pairs['batch'].numel()>0
        state=belief.with_(**{k:torch.cat((getattr(belief,k),getattr(belief,k)),0) for k in FIELDS}) if has_macro else belief
        if has_macro: state=state.with_(phase_quality=torch.cat((belief.phase_quality,torch.zeros_like(belief.phase_quality)),0))
        predictions=[]; phases=[]
        with torch.profiler.record_function("pacld/full_base_parallel" if has_macro else "pacld/full_only"):
            for dt,u in batch.future:
                state=self.propagate_plan(state,(torch.cat((dt,dt),1),torch.cat((u,u),1)) if has_macro else (dt,u))
                predictions.append(state.x); phases.append(state.phi)
        return belief,torch.stack(predictions,1),torch.stack(phases,1)

    def forward(self,batch,lambda_macro=0.):
        if not math.isfinite(lambda_macro) or lambda_macro<0: raise ValueError('nonnegative finite fixed lambda required')
        belief,predictions,phases=self.predict_batch(batch)
        B=batch.initial.x.shape[0]; t=batch.targets; p=batch.pairs
        x=predictions[t['batch'],t['query']]
        r=so3.state_residual(x,t['value'])
        score=(r.square()/t['noise']*t['weight']).sum(-1)/t['weight'].sum(-1).clamp_min(1)
        full=score.new_zeros(B).index_add(0,t['batch'],score*t['reduction'])
        macro=full*0
        if p['batch'].numel():
            a=predictions[B+p['batch'],p['a']]; b=predictions[B+p['batch'],p['b']]
            ya,yb=p['ya'],p['yb']
            pred=so3.multiply(so3.conjugate(a[...,6:10]),b[...,6:10])
            true=so3.multiply(so3.conjugate(ya[...,6:10]),yb[...,6:10])
            r=torch.cat(((b[...,:6]-a[...,:6])-(yb[...,:6]-ya[...,:6]),so3.residual(true,pred),(b[...,10:]-a[...,10:])-(yb[...,10:]-ya[...,10:])), -1)
            score=(r.square()/p['noise']*p['weight']).sum(-1)/p['weight'].sum(-1).clamp_min(1)
            macro=macro.index_add(0,p['batch'],score*p['reduction'])
        return dict(total=(full+lambda_macro*macro).mean(),full=full,macro=macro,macro_valid=p['valid_windows'],belief=belief,predictions=predictions,phases=phases)
