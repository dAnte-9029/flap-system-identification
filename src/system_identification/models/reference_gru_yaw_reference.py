"""Prefix-causal yaw-mean correction with corrected-state GRU feedback."""
import torch
from .reference_gru_unit_gravity import ReferenceGRUUnitGravity,unit_state_features
from .trajectory_main_v1 import (_normalize_quaternion,_rotation_body_to_ned,
    _delta_quaternion,_quaternion_multiply)


from .reference_gru_yaw_mean import historical_full_features


def reference_prefix_features(history_state, history_reference, history_actuator, current_mode, future_reference):
    """104 features at each step; no planned references after that step."""
    history = historical_full_features(history_state, history_reference, history_actuator, current_mode)
    future = future_reference.to(torch.float64)
    start = history_reference[:, -1:, :].to(torch.float64)
    count = torch.arange(1, future.shape[1] + 1, dtype=future.dtype, device=future.device)[None, :, None]
    return torch.cat([history[:, None, :].expand(-1, future.shape[1], -1),
                      future - start, future.cumsum(1) / count - start], -1)


class ReferenceGRUYawReference(ReferenceGRUUnitGravity):
    """Research-only causal mean forecast; empirical limits are not flight safety limits.

    Future cumulative means use history plus the current planned reference prefix. The additive yaw-rate bias
    uses only the simulated prefix, and is fed back with the corrected physical state.
    """
    def __init__(self,stats,mean_parameters,enabled=True):
        super().__init__(stats)
        for key in ['mean','scale','coefficient','target_mean']:
            self.register_buffer('yaw_'+key,torch.as_tensor(mean_parameters[key],dtype=torch.float64))
        if self.yaw_mean.shape != (50,104) or self.yaw_scale.shape != (50,104) or self.yaw_coefficient.shape != (50,104) or self.yaw_target_mean.shape != (50,):
            raise ValueError('expected 50 prefix predictors with 104 features')
        if not all(torch.isfinite(getattr(self,'yaw_'+k)).all() for k in ['mean','scale','coefficient','target_mean']) or not (self.yaw_scale>0).all():
            raise ValueError('finite parameters and positive scales required')
        self.enabled=enabled

    def forward(self,history_state,history_reference,history_actuator,future_reference,
                initial_quaternion,initial_velocity_n,initial_rates_b,current_mode,dt_s,diagnostics=False):
        if future_reference.shape[1]>50 or not torch.all(abs(dt_s-.02)<1e-7):
            raise ValueError('trained schedule supports at most50 steps at20ms')
        if not self.enabled:
            return super().forward(history_state,history_reference,history_actuator,future_reference,
                initial_quaternion,initial_velocity_n,initial_rates_b,current_mode,dt_s)
        if future_reference.shape[-1]!=5:raise ValueError('five reference channels required')
        features=reference_prefix_features(history_state,history_reference,history_actuator,current_mode,future_reference)
        steps=future_reference.shape[1]
        means=((((features-self.yaw_mean[:steps])/self.yaw_scale[:steps])*self.yaw_coefficient[:steps]).sum(-1)+self.yaw_target_mean[:steps]).to(history_state.dtype)
        history=torch.cat([(history_state-self.state_mean)/self.state_scale,
            (history_reference-self.reference_mean)/self.reference_scale,
            (history_actuator-self.actuator_mean)/self.actuator_scale,
            current_mode[:,None,:].expand(-1,history_state.shape[1],-1)],-1)
        _,h=self.encoder(history);h=h[0]
        quat=_normalize_quaternion(initial_quaternion);vel=initial_velocity_n;rate=initial_rates_b
        freq=history_state[:,-1,11];phase=torch.zeros_like(freq);anchor=torch.zeros_like(freq)
        air=history_state[:,-1,12:14]
        bias=torch.zeros_like(freq);total=torch.zeros_like(freq);elapsed=torch.zeros_like(freq)
        result={k:[] for k in ['velocity_n','quaternion','rates_b']};biases=[]
        limit=self.rate_scale[2]
        for k in range(future_reference.shape[1]):
            state=torch.cat([unit_state_features(vel,quat,rate,phase,anchor,freq),air],-1)
            ref=(future_reference[:,k]-self.reference_mean)/self.reference_scale
            inp=torch.cat([(state-self.state_mean)/self.state_scale,ref,current_mode],-1)
            h=self.cell(inp,h)
            derivative=self.head(torch.cat([h,inp],-1)).clamp(-6.,6.)*self.derivative_scale
            dt=dt_s[:,k,None]
            vel=vel+torch.einsum('bij,bj->bi',_rotation_body_to_ned(quat),derivative[:,:3])*dt
            proposal=rate+derivative[:,3:]*dt
            # Remove the previous offset before updating the virtual uncorrected prefix.
            raw_yaw=proposal[:,2]-bias;total=total+raw_yaw;elapsed=elapsed+dt[:,0]
            u=(elapsed/.1).clamp(0.,1.);envelope=u*u*(3-2*u)
            desired=envelope*(means[:,k]-total/(k+1)).clamp(-limit,limit)
            slew=2*limit*dt[:,0]
            bias=bias+(desired-bias).clamp(-slew,slew)
            newrate=torch.cat([proposal[:,:2],(raw_yaw+bias)[:,None]],-1)
            quat=_normalize_quaternion(_quaternion_multiply(quat,_delta_quaternion(.5*(rate+newrate)*dt)))
            rate=newrate;phase=phase+2*torch.pi*freq*dt[:,0]
            for key,value in [('velocity_n',vel),('quaternion',quat),('rates_b',rate)]:result[key].append(value)
            biases.append(bias)
        output={k:torch.stack(v,1) for k,v in result.items()}
        if diagnostics:output['yaw_correction']=torch.stack(biases,1)
        return output
