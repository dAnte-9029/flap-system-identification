"""Causal tensor extraction for reference-driven closed-loop identification.

No future measured actuator/state features are returned as predictor inputs.
Normalization fitting and candidate selection belong to the preparation script.
"""
import numpy as np


def extract_window(arrays, origin, history=26, horizon=50):
    if origin < history or origin+horizon >= len(arrays['timestamp_us']):
        raise ValueError('window crosses array boundary')
    region=slice(origin-history,origin+horizon+1)
    if not np.asarray(arrays['valid'][region]).all():
        raise ValueError('window contains invalid samples')
    if np.asarray(arrays['reset'][origin-history+1:origin+horizon+1]).any():
        raise ValueError('window crosses estimator reset')
    times=np.asarray(arrays['timestamp_us'][region])
    if not np.all(np.diff(times)==20000):
        raise ValueError('window crosses a timestamp gap')
    modes=np.asarray(arrays['nav_state'][region])
    if not np.all(modes==modes[0]):
        raise ValueError('window crosses mode boundary')
    h=slice(origin-history+1,origin+1)
    past=slice(origin-history,origin)
    future=slice(origin,origin+horizon)
    target=slice(origin+1,origin+horizon+1)
    state=np.array(arrays['state_features'][h],copy=True)
    # Express logged phase relative to the origin; do not infer physical zero.
    phi=np.asarray(arrays['phase_rad'][h])-arrays['phase_rad'][origin]
    state[:,9]=np.sin(phi);state[:,10]=np.cos(phi)
    inputs=dict(history_state=state,history_reference=np.array(arrays['references'][h],copy=True),
        history_actuator=np.array(arrays['actuator_commands'][past],copy=True),
        future_reference=np.array(arrays['references'][future],copy=True),
        initial_quaternion=np.array(arrays['quaternion'][origin],copy=True),
        initial_velocity_n=np.array(arrays['velocity_n'][origin],copy=True),
        initial_rates_b=np.array(arrays['rates_b'][origin],copy=True),
        dt_s=np.full(horizon,.02,dtype=np.float32),
        current_mode=np.array([modes[0]==3,modes[0]==15],dtype=np.float32))
    labels={name:np.array(arrays[name][target],copy=True) for name in ['quaternion','velocity_n','rates_b']}
    for value in inputs.values():
        if not np.isfinite(value).all():raise ValueError('nonfinite predictor input')
    for value in labels.values():
        if not np.isfinite(value).all():raise ValueError('nonfinite target')
    return inputs,labels
