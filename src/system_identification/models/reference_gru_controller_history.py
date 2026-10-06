"""Encoder-only controller observations; no data access or normalization fitting."""
import torch
from torch import nn
from .reference_gru_unit_gravity import ReferenceGRUUnitGravity,unit_state_features
from .trajectory_main_v1 import (_normalize_quaternion,_rotation_body_to_ned,
    _delta_quaternion,_quaternion_multiply)


class ReferenceGRUControllerHistory(ReferenceGRUUnitGravity):
    def __init__(self,stats,controller_mean,controller_scale,use_controller=True,hidden=64):
        # Initialize every original parameter first, in exactly the v23 order.
        super().__init__(stats,hidden)
        original=self.encoder
        self.encoder=nn.GRU(33,hidden,batch_first=True)
        with torch.no_grad():
            self.encoder.weight_ih_l0.zero_()
            self.encoder.weight_ih_l0[:,:25].copy_(original.weight_ih_l0)
            self.encoder.weight_hh_l0.copy_(original.weight_hh_l0)
            self.encoder.bias_ih_l0.copy_(original.bias_ih_l0)
            self.encoder.bias_hh_l0.copy_(original.bias_hh_l0)
        self.register_buffer('controller_mean',torch.as_tensor(controller_mean,dtype=torch.float32))
        self.register_buffer('controller_scale',torch.as_tensor(controller_scale,dtype=torch.float32))
        if self.controller_mean.shape!=(8,) or self.controller_scale.shape!=(8,):
            raise ValueError('eight controller channels required')
        if not torch.isfinite(self.controller_mean).all() or not torch.isfinite(self.controller_scale).all() or not (self.controller_scale>0).all():
            raise ValueError('invalid controller normalization')
        self.use_controller=bool(use_controller)

    def forward(self,history_state,history_reference,history_actuator,future_reference,
                initial_quaternion,initial_velocity_n,initial_rates_b,current_mode,dt_s,history_controller):
        if future_reference.shape[-1]!=5:raise ValueError('five reference channels required')
        if history_controller.shape!=(*history_state.shape[:2],8) or not torch.isfinite(history_controller).all():
            raise ValueError('finite controller observations must match past history')
        extra=(history_controller-self.controller_mean)/self.controller_scale
        if not self.use_controller:extra=torch.zeros_like(extra)
        history=torch.cat([(history_state-self.state_mean)/self.state_scale,
            (history_reference-self.reference_mean)/self.reference_scale,
            (history_actuator-self.actuator_mean)/self.actuator_scale,
            current_mode[:,None,:].expand(-1,history_state.shape[1],-1),extra],dim=-1)
        _,h=self.encoder(history);h=h[0]
        quat=_normalize_quaternion(initial_quaternion);vel=initial_velocity_n;rate=initial_rates_b
        freq=history_state[:,-1,11];phase=torch.zeros_like(freq);anchor=torch.zeros_like(freq)
        air=history_state[:,-1,12:14];velocities=[];quaternions=[];rates=[]
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
