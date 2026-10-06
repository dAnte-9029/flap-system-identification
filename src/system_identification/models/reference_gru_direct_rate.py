"""Single-factor direct rate-residual counterpart of the unit-gravity GRU.

Velocity still integrates body acceleration. Rate is decoded relative to the
observed origin instead of accumulating angular acceleration. Both then use
the same midpoint quaternion update. No future measured states are consumed.
"""
import torch
from .reference_gru_unit_gravity import ReferenceGRUUnitGravity,unit_state_features
from .trajectory_main_v1 import (_normalize_quaternion,_rotation_body_to_ned,
    _delta_quaternion,_quaternion_multiply)


class ReferenceGRUDirectRate(ReferenceGRUUnitGravity):
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
        air=history_state[:,-1,12:14];velocities=[];quaternions=[];rates=[]
        for k in range(future_reference.shape[1]):
            features=torch.cat([unit_state_features(vel,quat,rate,phase,anchor,freq),air],dim=-1)
            ref=(future_reference[:,k]-self.reference_mean)/self.reference_scale
            inp=torch.cat([(features-self.state_mean)/self.state_scale,ref,current_mode],dim=-1)
            h=self.cell(inp,h);output=self.head(torch.cat([h,inp],dim=-1)).clamp(-6.,6.)
            dt=dt_s[:,k,None];rotation=_rotation_body_to_ned(quat)
            vel=vel+torch.einsum('bij,bj->bi',rotation,output[:,:3]*self.derivative_scale[:3])*dt
            newrate=initial_rates_b+output[:,3:]*self.rate_scale
            quat=_normalize_quaternion(_quaternion_multiply(quat,_delta_quaternion(.5*(rate+newrate)*dt)))
            rate=newrate;phase=phase+2*torch.pi*freq*dt[:,0]
            velocities.append(vel);quaternions.append(quat);rates.append(rate)
        return dict(velocity_n=torch.stack(velocities,1),quaternion=torch.stack(quaternions,1),rates_b=torch.stack(rates,1))
