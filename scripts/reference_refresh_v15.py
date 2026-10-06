"""Observed-history refresh diagnostics over the same original target samples."""
import numpy as np
import torch
from system_identification.data.reference_windows import extract_window


def refresh_inputs(arrays,origin,period):
    if period not in [5,10,50]:raise ValueError('frozen periods are5,10,50 samples')
    blocks=[]
    for offset in range(0,50,period):
        inp,_=extract_window(arrays,origin+offset,horizon=period)
        blocks.append(inp)
    return {k:np.stack([b[k] for b in blocks]) for k in blocks[0]}


def stitch(values,windows,period):
    if period not in [5,10,50] or values.shape[:2]!=(windows*(50//period),period):raise ValueError('invalid refresh dimensions')
    return values.reshape(windows,50,*values.shape[2:])


def persistence(inputs):
    steps=inputs['future_reference'].shape[1]
    return {key:np.repeat(inputs[source][:,None,:],steps,axis=1) for key,source in
        [('rates_b','initial_rates_b'),('velocity_n','initial_velocity_n'),('quaternion','initial_quaternion')]}


def squared_errors(pred,truth):
    rate=(pred['rates_b'].astype(float)-truth['rates_b'])**2
    vel=np.mean((pred['velocity_n'].astype(float)-truth['velocity_n'])**2,axis=-1)
    dot=np.clip(abs(np.sum(pred['quaternion'].astype(float)*truth['quaternion'],axis=-1)),0,1)
    return np.concatenate([rate,vel[:,:,None],(2*np.arccos(dot))[:,:,None]**2],axis=-1)


def paired_local_age(errors,period):
    """Same global target positions are paired at each local prediction age."""
    if period not in [5,10]:raise ValueError('local age requires5 or10')
    return errors.reshape(len(errors),50//period,period,5).mean(axis=1)
