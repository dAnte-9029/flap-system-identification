"""Offline instrumented GRU replay. Oracle labels never become deployable inputs."""
import numpy as np
import torch
from system_identification.models.reference_gru_unit_gravity import unit_state_features
from system_identification.models.trajectory_main_v1 import (
    _normalize_quaternion, _rotation_body_to_ned, _quaternion_multiply, _delta_quaternion)

VARIANTS = ['free', 'true_state_free_hidden', 'true_state_recursive_hidden']


@torch.no_grad()
def trace(model, x, truth):
    """All branches share history/control and held phase/frequency/airspeed.

    free is the exact model rollout. true_state_free_hidden forks one step from
    each free pre-update hidden state, with previous true physical state; its
    hidden update is discarded. true_state_recursive_hidden keeps its own hidden
    trajectory, updating it from previous true physical states at every step.
    Neither oracle reads the current target before predicting that step.
    """
    hs = x['history_state']
    if 'history_command' in x:
        hr = model.padded(x['history_command']); future = model.padded(x['future_command'])
    else:
        hr = x['history_reference']; future = x['future_reference']
    history = torch.cat([(hs-model.state_mean)/model.state_scale,
        (hr-model.reference_mean)/model.reference_scale,
        (x['history_actuator']-model.actuator_mean)/model.actuator_scale,
        x['current_mode'][:,None,:].expand(-1,hs.shape[1],-1)],-1)
    _, encoded = model.encoder(history); h = encoded[0]; ht = h
    quat = _normalize_quaternion(x['initial_quaternion'])
    vel = x['initial_velocity_n']; rate = x['initial_rates_b']
    freq = hs[:,-1,11]; phase = torch.zeros_like(freq); anchor = torch.zeros_like(freq)
    air = hs[:,-1,12:14]
    records = {name:{} for name in VARIANTS}

    def advance(q,v,r,hidden,control,k):
        features = torch.cat([unit_state_features(v,q,r,phase,anchor,freq),air],-1)
        inp = torch.cat([(features-model.state_mean)/model.state_scale,
                         (control-model.reference_mean)/model.reference_scale,x['current_mode']],-1)
        after = model.cell(inp,hidden)
        raw = model.head(torch.cat([after,inp],-1))
        derivative = raw.clamp(-6.,6.)*model.derivative_scale
        dt = x['dt_s'][:,k,None]
        newv = v+torch.einsum('bij,bj->bi',_rotation_body_to_ned(q),derivative[:,:3])*dt
        newr = r+derivative[:,3:]*dt
        newq = _normalize_quaternion(_quaternion_multiply(q,_delta_quaternion(.5*(r+newr)*dt)))
        return dict(velocity_n=newv,quaternion=newq,rates_b=newr,derivative=derivative,
                    raw_head=raw,hidden=after,recurrent_input=inp)

    for k in range(future.shape[1]):
        if k == 0:
            qt,vt,rt = quat,vel,rate
        else:
            qt,vt,rt = truth['quaternion'][:,k-1],truth['velocity_n'][:,k-1],truth['rates_b'][:,k-1]
        free = advance(quat,vel,rate,h,future[:,k],k)
        fork = advance(qt,vt,rt,h,future[:,k],k)
        teacher = advance(qt,vt,rt,ht,future[:,k],k)
        for name,values in zip(VARIANTS,[free,fork,teacher]):
            for key,value in values.items(): records[name].setdefault(key,[]).append(value)
        quat,vel,rate,h = free['quaternion'],free['velocity_n'],free['rates_b'],free['hidden']
        ht = teacher['hidden'];phase = phase+2*torch.pi*freq*x['dt_s'][:,k]
    out = {'encoded_hidden':encoded[0].numpy()}
    for name,values in records.items():
        for key,value in values.items(): out[name+'_'+key] = torch.stack(value,1).numpy()
    return out


def accumulated_rates(initial, derivative, dt):
    """Offline sum of predicted acceleration on each branch's state path.

    Unlike reset one-step rates, this exposes cumulative signed increment error.
    It is not a consistent closed-loop trajectory for either oracle branch.
    """
    return np.asarray(initial,dtype=float)[:,None,:]+np.cumsum(
        np.asarray(derivative,dtype=float)[:,:,3:]*np.asarray(dt,dtype=float)[:,:,None],axis=1)
