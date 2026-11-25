"""A fixed recurrent flight predictor and three matched linear readouts."""
import importlib.util
from pathlib import Path
import math
import torch
from torch import nn

spec=importlib.util.spec_from_file_location("flight_recurrent_strong",Path(__file__).with_name("strong.py"))
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)


class FlightPredictor(nn.Module):
    def __init__(self,mean,std):
        super().__init__()
        self.register_buffer("mean",mean.clone())
        self.register_buffer("std",std.clone())
        self.core=core.FlightPredictor(mean,std)
        self.register_buffer("center",torch.zeros(46))
        self.register_buffer("scale",torch.ones(46))
        self.register_buffer("clock_coeff",torch.zeros(5))
        self.register_buffer("clock_choice",torch.zeros(3))
        self.weights=nn.Parameter(torch.zeros(2,3,47,3))

    def raw_context(self,hist):
        h=hist[:,:self.mean.numel()]
        return torch.cat((self.core.encoder((h-self.mean)/self.std),hist[:,180:]),-1)

    def context(self,hist):
        h=(self.raw_context(hist)-self.center)/self.scale
        return torch.cat((torch.ones_like(h[:,:1]),h),1)

    def clocks(self,hist,pooled,u0,t):
        f=hist[:,165,None,None]
        history=hist[:,:180].reshape(-1,6,30)
        valid=history[...,22]
        avg_f=(history[...,15]*valid).sum(1)/valid.sum(1).clamp_min(1)
        motor=u0[:,0,None,None]
        mass=-torch.expm1(-t)
        wm=pooled[...,8]/torch.where(t>0,mass,1.)
        ramp=t+.2*torch.expm1(-t/.2)
        b=self.clock_coeff
        correction=ramp*(b[0]+b[1]*(f-3.8)+b[2]*(motor-.5)+b[3]*(f-avg_f[:,None,None])+b[4]*(wm-motor))
        return torch.stack((t*.25,t*f,t*f+correction),-1)*2*math.pi

    def components(self,hist,pooled,u0,time):
        ctx=self.context(hist)
        coef=torch.einsum('bh,ckha->bcka',ctx,self.weights)
        slow=self.core(hist[:,:self.mean.numel()],pooled,u0,time)
        slow=slow+(-torch.expm1(-time/.35))[...,None]*coef[:,None,:,0,:]
        angle=(self.clocks(hist,pooled,u0,time)*self.clock_choice).sum(-1)
        decay=torch.exp(-time/2)
        wave=decay[...,None]*(angle.cos()[...,None]*coef[:,None,:,1,:]+angle.sin()[...,None]*coef[:,None,:,2,:])
        return slow,wave,coef[:,None,:,1,:]

    def forward(self,hist,pooled,u0,time):
        s,w,origin=self.components(hist,pooled,u0,time)
        return s+(w-origin)
