"""Original-origin single-step objective; no teacher forcing or new windows."""
import numpy as np
import train_control_windows_v2 as base

OUT = base.ROOT/'docs/analysis/results/control_windows_v17'
V8 = OUT.parent/'control_windows_v8'
METRICS = ['p','q','r','velocity','attitude']


def first_step_inputs(inputs):
    return {k: v[:, :1] if k in ['future_reference','dt_s'] else v for k,v in inputs.items()}


def first_step_loss(pred, target, model):
    return base.loss_fn({k:v[:,:1] for k,v in pred.items()},
                        {k:v[:,:1] for k,v in target.items()},model)


def errors(pred, truth):
    rate=(pred['rates_b'].astype(float)-truth['rates_b'])**2
    vel=np.mean((pred['velocity_n'].astype(float)-truth['velocity_n'])**2,axis=-1)
    # Normalize both quaternions to avoid a float32 norm floor at20ms.
    a=pred['quaternion'].astype(float);b=truth['quaternion'].astype(float)
    a/=np.linalg.norm(a,axis=-1,keepdims=True);b/=np.linalg.norm(b,axis=-1,keepdims=True)
    # Chord-based sign-invariant distance is stable near zero angle.
    sign=np.where(np.sum(a*b,axis=-1,keepdims=True)<0,-1.,1.)
    chord=np.linalg.norm(a-sign*b,axis=-1)
    angle=4*np.arcsin(np.clip(chord/2,0,1))
    return np.concatenate([rate,vel[:,:,None],angle[:,:,None]**2],axis=-1)
