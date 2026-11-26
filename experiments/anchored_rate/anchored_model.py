"""Frozen WingbeatPredictor plus a history-anchored, command-conditioned slow rate readout."""
import sys
from pathlib import Path
import torch
from torch import nn
from labels import ROOT

sys.path.insert(0,str(ROOT/'research/rounds/wingbeat-hyperparameter-search-20261126'))
from search import wave, load_tuned, causal_conditions

class AnchoredRate(nn.Module):
    def __init__(self, seed=17):
        super().__init__()
        self.ac=load_tuned(seed,'cpu').eval().requires_grad_(False)
        n=self.ac.first_order.core.field_linear.out_features
        self.register_buffer('weights',torch.zeros(2*(n+1)+3+4,3))

    def states(self,hist,pooled,u0,time):
        """Recover A's actual propagated angular-rate stream, at the original eight queries."""
        core=self.ac.first_order.core
        h=hist[:,:core.mean.numel()]
        state=core.encoder((h-core.mean)/core.std)
        states=[state]
        dt,forcing=core.interval_features(pooled,u0,time)
        moments=core.command_moments(forcing[...,12:16],time)
        memory=core.memory_rate(core.memory_features(moments))
        for i in range(time.shape[1]):
            state=state+dt[:,i,1]*(core.rate(state,forcing[:,i,1])+memory[:,i,1])
            states.append(state)
        return torch.stack(states,1)

    def features(self,states,canonical,t,commands,anchor):
        # The final latent may contribute only *after* its own query time.
        edges=torch.cat((torch.zeros_like(canonical[:,:1]),canonical),1)
        dt=(t[:,:,None]-edges[:,None]).clamp_min(0.)
        dur=torch.cat((dt[:,:,:-1]-dt[:,:,1:],dt[:,:,-1:]),-1)
        gate=-torch.expm1(-t/.35)
        h0=torch.cat((torch.ones_like(anchor[:,:1]),states[:,0],anchor),-1)
        return torch.cat((gate[...,None]*h0[:,None],t[...,None],dur@states,commands[...,8:12]),-1)

    def periodic(self,hist,pool,u0,time,coef):
        low=self.ac.first_order
        basis=low.bases(hist,pool,u0,time)['direction']
        beta=torch.einsum('bh,kha->bka',low.context(hist),low.weights[1])
        first=(basis[:,:,1,1,None]+1)*beta[:,None,1]+basis[:,:,1,2,None]*beta[:,None,2]
        theta=low.clocks(hist,pool,u0,time)[...,-1]
        high=self.ac.high_order(coef,time,theta,causal_conditions(hist,pool,u0,time))
        return first+high[:,:,1],beta[:,1]

    def offsets(self,hist,pool,u0,time,coef,anchor,origin):
        wave_abs,wave0=self.periodic(hist,pool,u0,time,coef)
        return (anchor-origin)[:,None]+wave_abs+torch.exp(-time[:,:,1,None]/.35)*(origin-anchor-wave0)[:,None]

    def forward(self,hist,pooled,u0,time,coef,anchor,origin,eligible):
        base=self.ac(hist,pooled,u0,time,coef)
        states=self.states(hist,pooled,u0,time)
        features=self.features(states,time[:,:,1],time[:,:,1],pooled[:,:,1],anchor)
        drift=features@self.weights
        proposed=self.offsets(hist,pooled,u0,time,coef,anchor,origin)+drift
        full=torch.stack((base[:,:,0],torch.where(eligible[:,None,None],proposed,base[:,:,1])),2)
        return full,anchor[:,None]+drift
