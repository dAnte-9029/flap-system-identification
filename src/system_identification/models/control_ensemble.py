"""Frozen GRU derivative ensemble with one physical state and explicit memory.

Member spread is a diagnostic, not calibrated uncertainty or a safety bound.
All members retain their trained feature scaling and derivative clipping.
"""
from dataclasses import dataclass

import torch
from torch import nn

from .trajectory_main_v1 import (
    TorchTrajectoryPrediction, _state_features, _normalize_quaternion,
    _rotation_body_to_ned, _delta_quaternion, _quaternion_multiply,
)


PHYSICAL = ('position_n','velocity_n','quaternion_nb','angular_velocity_b',
            'relative_phase_rad','flap_frequency_hz')


@dataclass(frozen=True)
class EnsembleState:
    position_n: torch.Tensor
    velocity_n: torch.Tensor
    quaternion_nb: torch.Tensor
    angular_velocity_b: torch.Tensor
    relative_phase_rad: torch.Tensor
    flap_frequency_hz: torch.Tensor
    phase_anchor: torch.Tensor
    hidden: tuple[torch.Tensor, ...]


class ControlDerivativeEnsemble(nn.Module):
    """Equal derivative average at shared state, followed by the original integrator."""

    def __init__(self,members):
        super().__init__()
        if not members:
            raise ValueError('at least one frozen model required')
        self.members=nn.ModuleList(members)
        for member in self.members:
            member.requires_grad_(False)

    def reset(self, *, history_state_features,history_controls,history_mask,**physical):
        if set(physical)!=set(PHYSICAL):
            raise ValueError('explicit physical initial state required')
        hidden=tuple(m._encode_history(history_state_features,history_controls,history_mask) for m in self.members)
        values={k:physical[k].clone() for k in PHYSICAL}
        if not all(torch.isfinite(v).all() for v in values.values()):
            raise ValueError('nonfinite initial state')
        values['quaternion_nb']=_normalize_quaternion(values['quaternion_nb'])
        return EnsembleState(**values,phase_anchor=physical['relative_phase_rad'].clone(),hidden=hidden)

    def step(self,state,command,dt):
        if command.shape!=(len(state.position_n),4) or dt.shape!=(len(command),):
            raise ValueError('command [batch,4] and dt [batch] required')
        if not torch.isfinite(command).all() or not torch.all(torch.isfinite(dt)&(dt>0)&(dt<=.05)):
            raise ValueError('finite command and dt in (0,0.05] required')
        if len(state.hidden)!=len(self.members):
            raise ValueError('hidden state member count mismatch')
        features=_state_features(state.velocity_n,state.quaternion_nb,state.angular_velocity_b,
            state.relative_phase_rad,state.phase_anchor,state.flap_frequency_hz)
        derivatives=[];clipped=[]
        for m,h in zip(self.members,state.hidden):
            raw=m.derivative_head(torch.cat((h,m._model_input(features,command)),dim=1))
            derivatives.append(m.derivative_mean+m.derivative_std*torch.clamp(raw,-6.,6.))
            clipped.append((raw.abs()>6.).any(1))
        stack=torch.stack(derivatives)
        derivative=stack.mean(0)
        acc=torch.einsum('bij,bj->bi',_rotation_body_to_ned(state.quaternion_nb),derivative[:,:3])
        pos=state.position_n+state.velocity_n*dt[:,None]+.5*acc*torch.square(dt)[:,None]
        vel=state.velocity_n+acc*dt[:,None]
        rate=state.angular_velocity_b+derivative[:,3:6]*dt[:,None]
        midpoint=.5*(state.angular_velocity_b+rate)
        quat=_normalize_quaternion(_quaternion_multiply(state.quaternion_nb,_delta_quaternion(midpoint*dt[:,None])))
        freq=torch.clamp(state.flap_frequency_hz+derivative[:,6]*dt,.5,20.)
        phase=torch.remainder(state.relative_phase_rad+2.*torch.pi*.5*(state.flap_frequency_hz+freq)*dt,2.*torch.pi)
        nxt=_state_features(vel,quat,rate,phase,state.phase_anchor,freq)
        hidden=tuple(m.recurrent_cell(m._model_input(nxt,command),h) for m,h in zip(self.members,state.hidden))
        result=EnsembleState(pos,vel,quat,rate,phase,freq,state.phase_anchor,hidden)
        return result,dict(member_derivatives=stack,derivative_std=stack.std(0,unbiased=False),
            member_clipped=torch.stack(clipped),frequency_clipped=((state.flap_frequency_hz+derivative[:,6]*dt)<.5)|((state.flap_frequency_hz+derivative[:,6]*dt)>20.))

    def forward(self, *, future_controls,dt_s,**initial):
        state=self.reset(**initial)
        values=[[getattr(state,k)] for k in PHYSICAL]
        for step in range(future_controls.shape[1]):
            state,_=self.step(state,future_controls[:,step],dt_s[:,step])
            for name,array in zip(PHYSICAL,values):array.append(getattr(state,name))
        return TorchTrajectoryPrediction(*(torch.stack(x,dim=1) for x in values))
