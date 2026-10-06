"""Unit-gravity counterpart of the immutable legacy ReferenceGRU implementation."""
import torch
from torch import nn
from .trajectory_main_v1 import (_normalize_quaternion,_rotation_body_to_ned,
    _delta_quaternion,_quaternion_multiply,_state_features)


def unit_state_features(*args):
    features=_state_features(*args)
    # The reference dataset stores a unit gravity direction, not acceleration.
    return torch.cat([features[:,:6],features[:,6:9]/9.80665,features[:,9:]],dim=-1)


class ReferenceGRUUnitGravity(nn.Module):
    """Encode observed history, then integrate predicted physical derivatives.

    Current flap frequency and airspeed are held as latent-context proxies.
    This is an explicit modeling limitation, not future measurement access.
    """
    def __init__(self,stats,hidden=64):
        super().__init__()
        for name in ['state_mean','state_scale','reference_mean','reference_scale','actuator_mean','actuator_scale','rate_scale','velocity_scale']:
            self.register_buffer(name,torch.as_tensor(stats[name],dtype=torch.float32))
        self.encoder=nn.GRU(25,hidden,batch_first=True)
        self.cell=nn.GRUCell(21,hidden)
        self.head=nn.Sequential(nn.Linear(hidden+21,hidden),nn.Tanh(),nn.Linear(hidden,6))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        # Fixed derivative output units: m/s2 and rad/s2, shared across all arms.
        self.register_buffer('derivative_scale',torch.tensor([10.,10.,10.,20.,20.,20.]))

    def forward(self,history_state,history_reference,history_actuator,future_reference,
                initial_quaternion,initial_velocity_n,initial_rates_b,current_mode,dt_s):
        if future_reference.shape[-1]!=5:raise ValueError('five reference channels required')
        history=torch.cat([(history_state-self.state_mean)/self.state_scale,
            (history_reference-self.reference_mean)/self.reference_scale,
            (history_actuator-self.actuator_mean)/self.actuator_scale,
            current_mode[:,None,:].expand(-1,history_state.shape[1],-1)],dim=-1)
        _,h=self.encoder(history);h=h[0]
        quat=_normalize_quaternion(initial_quaternion);vel=initial_velocity_n;rate=initial_rates_b
        freq=history_state[:,-1,11];phase=torch.zeros_like(freq);anchor=torch.zeros_like(freq)
        air=history_state[:,-1,12:14]
        velocities=[];quaternions=[];rates=[]
        for k in range(future_reference.shape[1]):
            features=torch.cat([unit_state_features(vel,quat,rate,phase,anchor,freq),air],dim=-1)
            ref=(future_reference[:,k]-self.reference_mean)/self.reference_scale
            inp=torch.cat([(features-self.state_mean)/self.state_scale,ref,current_mode],dim=-1)
            h=self.cell(inp,h)
            derivative=self.head(torch.cat([h,inp],dim=-1)).clamp(-6.,6.)*self.derivative_scale
            dt=dt_s[:,k,None]
            rotation=_rotation_body_to_ned(quat)
            vel=vel+torch.einsum('bij,bj->bi',rotation,derivative[:,:3])*dt
            newrate=rate+derivative[:,3:]*dt
            quat=_normalize_quaternion(_quaternion_multiply(quat,_delta_quaternion(.5*(rate+newrate)*dt)))
            rate=newrate;phase=phase+2*torch.pi*freq*dt[:,0]
            velocities.append(vel);quaternions.append(quat);rates.append(rate)
        return dict(velocity_n=torch.stack(velocities,1),quaternion=torch.stack(quaternions,1),rates_b=torch.stack(rates,1))
