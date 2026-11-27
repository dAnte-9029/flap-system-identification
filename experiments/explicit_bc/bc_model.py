"""Explicit B+C at arbitrary query times, with an optional additional main latent recurrence."""
import torch
from torch import nn
from search import prior
from bc_features import query,history,anchor_delta

class Predictor(nn.Module):
    def __init__(self,include_a):
        super().__init__()
        self.include_a=include_a
        # Outputs are fixed coefficients of the explicit bases during a forecast window.
        self.coefficients=nn.Sequential(nn.Linear(198,72),nn.SiLU(),nn.Linear(72,2*13*3))
        nn.init.zeros_(self.coefficients[-1].bias)
        nn.init.normal_(self.coefficients[-1].weight,std=.005)
        if include_a:
            self.a=prior.core.FlightPredictor(torch.zeros(180),torch.ones(180))
            nn.init.zeros_(self.a.readout_linear.weight)
            nn.init.zeros_(self.a.readout[-1].weight)

    def encode_coefficients(self,x):
        return self.coefficients(x).reshape(-1,2,13,3)

    @staticmethod
    def explicit(coeff,features,anchor):
        # Broadcast over query times; also works for individual native observations.
        smooth=(coeff[...,:11,:]*features[...,:11,None]).sum(-2)+anchor
        first=(coeff[...,11:,:]*features[...,11:13,None]).sum(-2)
        # The causal bridge makes the complete increment zero at the forecast origin.
        bridge=(1-features[...,1,None])*(anchor+coeff[...,11,:])
        return smooth+first+features[...,13:]-bridge,smooth

    def state_path(self,x,dt,forcing,memory):
        initial=self.a.encoder(x[:,:180])
        state=initial[:,None].expand(-1,2,-1)
        states=[state]
        for interval,f,m in zip(dt.unbind(1),forcing.unbind(1),self.a.memory_rate(memory).unbind(1)):
            state=state+interval*(self.a.rate(state,f)+m)
            states.append(state)
        return torch.stack(states,1)

    def endpoints(self,coeff,features,anchor,states):
        full,slow=self.explicit(coeff[:,None],features,anchor[:,None])
        if states is not None:
            delta=self.a.observe(states[:,1:])-self.a.observe(states[:,:1])
            motion=torch.stack((delta[:,:,0,:3],delta[:,:,1,3:]),2)
            full=full+motion
            slow=slow+motion
        return full,slow

    def native(self,coeff,features,anchor,states,owner,slot,path):
        full,slow=self.explicit(coeff[owner,1],features,anchor[owner,1])
        if states is not None:
            dt,f,m=path.split((1,24,16),-1)
            state=states[owner,slot,1]
            nxt=state+dt*(self.a.rate(state,f)+self.a.memory_rate(m))
            delta=self.a.observe(nxt)[:,3:]-self.a.observe(states[:,0,1])[owner,3:]
            full=full+delta
            slow=slow+delta
        return full,slow

    def forecast(self,raw,pool,u,t,coef,anchor,origin,stats,clock):
        x=history(raw,anchor,stats); ac=anchor_delta(anchor,origin)
        features=query(clock,raw,pool,u,t,coef)
        states=None
        if self.include_a:
            dt,f=self.a.interval_features(pool,u,t)
            m=self.a.memory_features(self.a.command_moments(f[...,12:16],t))
            states=self.state_path(x,dt,f,m)
        return self.endpoints(self.encode_coefficients(x),features,ac,states)
