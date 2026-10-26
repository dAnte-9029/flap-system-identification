"""Six-dimensional discrete updates; explicit control deviation, original integrator."""
import torch
from torch import nn
from system_identification.models.velocity_rate_gru import VelocityRateGRU, integrate
from system_identification.models.control_events import canonicalize_control_events

def coordinates(u):
    return torch.stack((u[...,0], (u[...,1]+u[...,2])/2,
                        (u[...,2]-u[...,1])/2, u[...,3]),-1)

class StructuredGRU(VelocityRateGRU):
    def __init__(self,normalization,phase_modulation=False):
        super().__init__(normalization,control=True)
        # Remove the future-event path into the recurrent base branch entirely.
        del self.event_gru
        self.step_gru=nn.GRUCell(18,64)
        self.phase_modulation=phase_modulation
        self.effect_head=nn.Linear(64,48*(3 if phase_modulation else 1))
        nn.init.normal_(self.effect_head.weight,std=.001)
        nn.init.zeros_(self.effect_head.bias)
        # Transform raw deviations first; common/differential share a fixed RMS scale.
        s=self.scale[7:11]
        lateral=.5*torch.sqrt(s[1].square()+s[2].square())
        self.register_buffer('coordinate_scale',torch.stack((s[0],lateral,lateral,s[3])))

    def control_moments(self,events,lengths,dt,reference):
        events,lengths=canonicalize_control_events(events,lengths,dt)
        duration=events[...,4]
        end=duration.cumsum(-1);start=end-duration
        delta=coordinates(events[...,:4]-reference[:,None,None,:])/self.coordinate_scale
        # Exact integral of two time basis functions over each constant segment.
        m0=duration
        m1=duration*((start+end)/dt.clamp_min(1e-8)[...,None]-1)
        return torch.cat(((delta*m0[...,None]).sum(-2),
                          (delta*m1[...,None]).sum(-2)),-1)/.02

    def branch(self,hidden,state,phase,mask,dt,moment):
        # Neither the base state update nor B reads the CURRENT event/moment.
        hidden=self.step_gru(self.features(state,phase,mask,dt),hidden)
        base=self.head(hidden)
        matrix=self.effect_head(hidden).reshape(len(hidden),-1,6,8)
        effect=matrix[:,0]
        if self.phase_modulation:
            effect=effect+mask[:,:1,None]*(phase[:,:1,None]*matrix[:,1]+phase[:,1:2,None]*matrix[:,2])
        control=torch.einsum('bij,bj->bi',effect,moment)
        return hidden,base,control

    def forward(self,inputs):
        h,hm=inputs['history'],inputs['history_mask']
        masks=torch.stack((hm[...,17]&hm[...,18],hm[...,19]),-1).float()
        feat=self.features(h[...,:13],h[...,17:20],masks,inputs['history_dt'])
        command=(h[...,13:17]-self.mean[7:11])/self.scale[7:11]
        sequence,_=self.history_gru(torch.cat((feat,command),-1))
        batch=torch.arange(len(h),device=h.device);ix=inputs['history_length']-1
        hidden=sequence[batch,ix];last=h[batch,ix];mask=masks[batch,ix]
        phase0=torch.atan2(last[:,17],last[:,18]);freq=last[:,19]*mask[:,1]
        future_mask=torch.stack((mask.prod(-1),mask[:,1]),-1)
        dt=inputs['dt'];moment=self.control_moments(inputs['events'],inputs['event_length'],dt,last[:,13:17])
        state=inputs['initial'];elapsed=torch.zeros_like(freq);out=[]
        for k in range(dt.shape[1]):
            angle=phase0+2*torch.pi*freq*elapsed
            phase=torch.stack((torch.sin(angle),torch.cos(angle),freq),-1)
            hidden,base,control=self.branch(hidden,state,phase,future_mask,dt[:,k],moment[:,k])
            vw=torch.cat((state[:,3:6],state[:,10:13]),-1)+(base+control)*self.output_scale
            state=integrate(state,vw[:,:3],vw[:,3:],dt[:,k]);out.append(state);elapsed=elapsed+dt[:,k]
        return torch.stack(out,1)

def make_model(name,norm):
    if name=='M0':return VelocityRateGRU(norm,control=True)
    if name not in ('M1','M2'):raise ValueError(name)
    return StructuredGRU(norm,phase_modulation=name=='M2')
