"""Controlled native-clock backbone with optional feedback from a slow latent state."""
from copy import deepcopy
from pathlib import Path
import sys
import torch
from torch import nn

ROOT=Path('/home/zn/flap-system-identification')
sys.path.insert(0,str(ROOT/'research/rounds/wingbeat-hyperparameter-search-20261126'))
from search import load_tuned, wave
TAU=.5

class Dynamics(nn.Module):
    def __init__(self, original, coupled):
        super().__init__()
        self.core=deepcopy(original.first_order.core).requires_grad_(True)
        self.core.readout.requires_grad_(False)
        self.core.readout_linear.requires_grad_(False)
        self.coupled=coupled
        self.anchor_init=nn.Linear(4,32,bias=False)
        nn.init.zeros_(self.anchor_init.weight)
        if coupled:
            self.seed_s=nn.Sequential(nn.Linear(36,8),nn.Tanh())
            self.s_target=nn.Sequential(nn.Linear(44,24),nn.Tanh(),nn.Linear(24,8),nn.Tanh())
            self.feedback=nn.Linear(8,32,bias=False)
            nn.init.zeros_(self.feedback.weight)
            self.s_readout=nn.Linear(8,3,bias=False)

    def context(self,hist):
        n=self.core.mean.numel()
        return (hist[:,:n]-self.core.mean)/self.core.std

    def inputs(self,pool,u,time):
        dt,f=self.core.interval_features(pool,u,time)
        moments=self.core.command_moments(f[...,12:16],time)
        memory=self.core.memory_features(moments)
        # Command summary available at each knot, for the following interval.
        filtered=pool[...,8:12]/(-torch.expm1(-time))[...,None]
        left=torch.cat((u[:,None,None].expand(-1,1,2,-1),filtered),1)
        return dt,f,memory,left

    def forward(self,hist,anchor,dt,f,memory,left,freeze_s=False,quadrature=None,trace=False):
        h0=self.core.encoder(hist)+self.anchor_init(anchor)
        h=h0[:,None].expand(-1,2,-1)
        if self.coupled:
            s0=self.seed_s(torch.cat((h0,anchor),-1))
            s=s0[:,None].expand(-1,2,-1)
        memory_rate=self.core.memory_rate(memory)
        htraj=[];s_begin=[];targets=[]
        for i in range(dt.shape[1]):
            step=dt[:,i]
            field=self.core.rate(h,f[:,i])+memory_rate[:,i]
            if self.coupled:
                equilibrium=self.s_target(torch.cat((h,s,left[:,i]),-1))
                field=field+self.feedback(s)
                if quadrature is not None or trace:
                    s_begin.append(s[:,1])
                    targets.append(equilibrium[:,1])
                if not freeze_s:
                    s=s+(-torch.expm1(-step/TAU))*(equilibrium-s)
            h=h+step*field
            htraj.append(h)
        signal=self.core.observe(torch.stack(htraj,1))-self.core.observe(h0)[:,None,None]
        motion=torch.stack((signal[:,:,0,:3],signal[:,:,1,3:]),2)
        read=None
        if self.coupled and quadrature is not None:
            s_begin.append(s[:,1])
            targets.append(self.s_target(torch.cat((h[:,1],s[:,1],left[:,-1,1]),-1)))
            starts=torch.stack(s_begin,1); eq=torch.stack(targets,1)
            mass,alpha=quadrature.unbind(-1)
            avg=(mass[:,:,None]*starts).sum(1)
            if not freeze_s:
                avg=avg+(alpha[:,:,None]*(eq-starts)).sum(1)
            read=anchor[:,:3]+self.s_readout(avg-s0)
        if trace:
            hs=torch.cat((h0[:,None],torch.stack(htraj,1)[:,:,1]),1)
            ss=None
            if self.coupled:
                # The quadrature path already appended the terminal state.
                ss=torch.stack(s_begin if quadrature is not None else [*s_begin,s[:,1]],1)
            return motion,read,hs,ss
        return motion,read

    def point(self,h,pool,u,time,anchor,qt,qp,freeze_s=False,return_s=False):
        """Single angular-rate query per row, using only already passed canonical knots."""
        hist=self.context(h)
        result=h.new_empty(len(h),3)
        slow_out=h.new_empty(len(h),3) if return_s and self.coupled else None
        before=(time[:,:,1]<qt[:,0,1,None]-1e-7).sum(1)
        for i in range(time.shape[1]+1):
            rows=torch.where(before==i)[0]
            if not len(rows):continue
            t=torch.cat((time[rows,:i,1],qt[rows,:,1]),1)[:,:,None].expand(-1,-1,2)
            p=torch.cat((pool[rows,:i,1],qp[rows,:,1]),1)[:,:,None,:].expand(-1,-1,2,-1)
            quad=None
            if slow_out is not None:
                quad=h.new_zeros(len(rows),t.shape[1]+1,2)
                quad[:,-1,0]=1
            x,ss=self(hist[rows],anchor[rows],*self.inputs(p,u[rows],t),freeze_s=freeze_s,quadrature=quad)
            if slow_out is not None:slow_out[rows]=ss
            result[rows]=x[:,-1,1]
        return result,slow_out
