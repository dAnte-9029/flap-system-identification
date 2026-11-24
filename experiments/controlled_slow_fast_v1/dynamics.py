"""Differentiable control-conditioned rollout with native ZOH breakpoints and dense queries."""
import math
from pathlib import Path
import sys

import torch
from torch import nn

from backbones import BACKBONES

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'history_slow_fast_v1'))
from model import inverse, rotate, prefix_rotations, geodesic, qexp, qmul, unit


def prefix_start(increments):
    return torch.cat([torch.zeros_like(increments[:,:1]),increments.cumsum(1)[:,:-1]],1)


def clock(history, tape):
    """Exact ZOH solution of the registered train-fit df/dt = b0 + bu*m + bf*f."""
    b0,bu,bf = 3.6354511983566393,23.393570068972757,-5.126725395181777
    us = tape[...,4].flatten(1)
    end = us.cumsum(1)
    dt = us.double()*1e-6
    t = (end-us).double()*1e-6
    motor = tape[...,0].flatten(1).double()
    forcing = b0+bu*motor
    # Integrating factor gives every segment-start frequency in one differentiable scan.
    f0 = history[:,-1,15:16].double()
    f = (f0+prefix_start(forcing*torch.exp(-bf*t)*(-torch.expm1(-bf*dt))/bf))*torch.exp(bf*t)
    angle = (2*math.pi/bf)*(f-f0-prefix_start(forcing*dt))
    angle,f = angle.float(),f.float()
    gate = (history[:,-1,21:22]*history[:,-1,22:23]).unsqueeze(-1)
    c,s,c2,s2 = angle.cos(),angle.sin(),(2*angle).cos(),(2*angle).sin()
    basis = gate*torch.stack([c,s,c2,s2],-1)
    slope = (gate*(2*math.pi*f.unsqueeze(-1)))*torch.stack([-s,c,-2*s2,2*c2],-1)
    return end,dt.float().unsqueeze(-1),f.unsqueeze(-1),basis,slope


class ControlledGRU(nn.Module):
    def __init__(self, mean, scale, structured, backbone=None):
        super().__init__()
        self.structured = structured
        self.block = 5 if structured else 1
        width = 48 if structured else 53
        self.backbone = None
        if backbone is None:
            self.encoder = nn.GRU(len(mean),64,batch_first=True)
            self.init_hidden = nn.Linear(64,width)
            self.cell = nn.GRUCell(18 if structured else 22,width)
        else:
            if structured:
                raise ValueError('separate sequence backbones use the single-rate baseline')
            self.backbone = BACKBONES[backbone](len(mean),22)
            width = self.backbone.width
        self.trend = nn.Linear(width,6)
        self.control_rate = nn.Linear(4,6,bias=False)
        heads = [self.trend,self.control_rate]
        if structured:
            self.initial_amplitude = nn.Linear(64,24)
            self.amplitude = nn.Linear(width+24,24)
            self.control_amplitude = nn.Linear(4,24,bias=False)
            heads.extend([self.initial_amplitude,self.amplitude,self.control_amplitude])
        for head in heads:
            nn.init.normal_(head.weight,std=.001)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
        self.register_buffer('mean',mean)
        self.register_buffer('scale',scale)
        self.register_buffer('axis_scale',scale[:6].clamp_min(.1))
        self.register_buffer('down',torch.tensor([0.,0.,1.]))

    def rollout(self, history, initial, commands):
        """MPC interface: [batch, horizon, 4] commands -> [batch, horizon+1, 13].

        Commands are ZOH on 20 ms intervals. All candidate values remain differentiable.
        Physical state is [p_NED, v_NED, Hamilton q_NB, omega_FRD].
        """
        if commands.ndim != 3 or commands.shape[-1] != 4:
            raise ValueError('expected [batch, horizon, 4] motor/left/right/rudder')
        duration = torch.full_like(commands[..., :1], 20000.)
        tape = torch.cat([commands, duration], -1).unsqueeze(2)
        roll = self(history, initial, tape)
        horizon = commands.size(1)
        times = torch.arange(1, horizon+1, device=commands.device, dtype=commands.dtype)*.02
        times = times.expand(commands.size(0), -1)
        path = torch.cat([query(roll,times,'pv'),query(roll,times,'q'),query(roll,times,'w')], -1)
        return torch.cat([initial[:,None],path],1)

    def forward(self,history,initial,tape):
        """tape[B,20ms steps,segments,5] holds motor/left/right/rudder/duration_us.

        Constant-grid MPC candidates use one segment of 20000 us per step.
        Event-packed candidates use right-padded zero durations.
        """
        if tape.ndim!=4 or tape.shape[-1]!=5 or tape.shape[1]==0:
            raise ValueError('expected a positive-length batched four-channel control tape')
        normalized = (history-self.mean)/self.scale
        if self.backbone is None:
            _,encoded = self.encoder(normalized)
            hidden = torch.tanh(self.init_hidden(encoded[0]))
        else:
            memory = self.backbone.initialize(normalized,tape.size(1))
        end,all_dt,freq,basis,slope = clock(history,tape)
        u = tape[...,:4].flatten(1,2)
        width = self.block*tape.shape[2]
        q = initial[:,6:10]
        motion = torch.cat([initial[:,3:6],initial[:,10:13]],-1)
        slow = motion
        prev_command = history[:,-1,9:13]
        if self.structured:
            coeff = self.initial_amplitude(encoded[0])
            r0 = (coeff.reshape(-1,6,4)*basis[:,0,None]).sum(-1)*self.axis_scale
            slow = motion-torch.cat([rotate(q,r0[:,:3]),r0[:,3:]],-1)
        positions,quats,motions,trends,slopes,slow_slopes = [],[],[],[],[],[]
        p = initial[:,:3]
        for start in range(0,u.size(1),width):
            sl = slice(start,start+width)
            dt,cmd = all_dt[:,sl],u[:,sl]
            body = torch.cat([rotate(inverse(q),slow[:,:3]),slow[:,3:]],-1)/self.axis_scale
            context = [body,rotate(inverse(q),self.down),prev_command,cmd[:,0],freq[:,start]/5.]
            if not self.structured:
                context.append(basis[:,start])
            inputs = torch.cat(context,-1)
            if self.backbone is None:
                hidden = self.cell(inputs,hidden)
            else:
                hidden,memory = self.backbone.step(inputs,memory)
            # A changed command has a direct differentiable entry within this block.
            change = cmd-cmd[:,:1]
            base = self.trend(hidden)[:,None]+self.control_rate(change)
            accel = base
            if self.structured:
                target = self.amplitude(torch.cat([hidden,coeff],-1))[:,None]+self.control_amplitude(change)
                dc = 2.*(target-coeff[:,None])
                c = coeff[:,None]+prefix_start(dc*dt)
                form = (*c.shape[:2],6,4)
                harmonic = (dc.reshape(form)*basis[:,sl,None]+c.reshape(form)*slope[:,sl,None]).sum(-1)
                accel = base+harmonic
                coeff = coeff+(dc*dt).sum(1)
            def physical(rate):
                rate = rate*self.axis_scale
                return torch.cat([rotate(q[:,None],rate[...,:3]),rate[...,3:]],-1)
            acc = physical(accel)
            dv = dt*acc
            path = motion[:,None]+prefix_start(dv)
            if self.structured:
                slow_acc = physical(base)
                ds = dt*slow_acc
                trends.append(slow[:,None]+prefix_start(ds))
                slow_slopes.append(slow_acc)
                slow = slow+ds.sum(1)
            dp = dt*(path[...,:3]+.5*dv[...,:3])
            dq = qexp(dt*(path[...,3:]+.5*dv[...,3:]))
            rel = prefix_rotations(dq)
            absolute = unit(qmul(q[:,None],rel))
            positions.append(p[:,None]+prefix_start(dp))
            quats.append(torch.cat([q[:,None],absolute[:,:-1]],1))
            motions.append(path)
            slopes.append(acc)
            p,q = p+dp.sum(1),absolute[:,-1]
            motion = motion+dv.sum(1)
            if not self.structured:
                slow = motion
            prev_command = (cmd*dt).sum(1)/dt.sum(1).clamp_min(1e-8)
        state = torch.cat([torch.cat(positions,1),torch.cat(motions,1)[...,:3],torch.cat(quats,1),
                           torch.cat(motions,1)[...,3:]],-1)
        trend = (torch.cat(trends,1),torch.cat(slow_slopes,1)) if self.structured else None
        return end,state,torch.cat(slopes,1),trend


def query(roll,t,component):
    """Integrate only the causal prefix of the containing constant-command segment."""
    end,state,acc,trend = roll
    ticks = (t*1e6).round().clamp_min(0).contiguous()
    # left search places an exact boundary at the end of the preceding segment.
    ix = torch.searchsorted(end.contiguous(),ticks).clamp(max=end.shape[1]-1)
    prev = torch.cat([torch.zeros_like(end[:,:1]),end[:,:-1]],1).gather(1,ix)
    dt = (ticks-prev).unsqueeze(-1)*1e-6
    def take(x):
        return x.gather(1,ix[...,None].expand(-1,-1,x.size(-1)))
    if component in ('slow_v','slow_w'):
        sl = slice(0,3) if component=='slow_v' else slice(3,6)
        return take(trend[0][...,sl])+dt*take(trend[1][...,sl])
    if component=='pv':
        v,a = take(state[...,3:6]),take(acc[...,:3])
        return torch.cat([take(state[...,:3])+dt*(v+.5*dt*a),v+dt*a],-1)
    w,alpha = take(state[...,10:]),take(acc[...,3:])
    if component=='q':
        return unit(qmul(take(state[...,6:10]),qexp(dt*(w+.5*dt*alpha))))
    if component!='w':
        raise ValueError(component)
    return w+dt*alpha
