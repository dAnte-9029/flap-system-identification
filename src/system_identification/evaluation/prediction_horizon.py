"""Short-horizon metrics and explicitly non-deployable teacher refresh reference."""
from dataclasses import fields,replace
import numpy as np
import torch
from system_identification.models.trajectory_main_v1 import _state_features,_delta_quaternion,_quaternion_multiply,_normalize_quaternion

STEPS=(1,2,5,10,25)
PHYSICAL=('position_n','velocity_n','quaternion_nb','angular_velocity_b','relative_phase_rad','flap_frequency_hz')


def teacher_refresh_rollout(simulator,initial,physical_history,history_commands,commands,dt):
    """ORACLE diagnostic: each endpoint is ONE step from truth at its predecessor.

    physical_history index 25 is t0; index 25+i is t_i. History length is 26.
    Proxies evolve only along commands, as in Step 7 A_native. The history and
    current phase feature are reanchored at the current teacher time. No claim
    that stitched endpoints are a k-step forecast from t0.
    """
    state=initial;states=[state]
    for k in range(commands.shape[1]):
        if k:
            now={key:physical_history[key][:,25+k] for key in PHYSICAL}
            anchor=now['relative_phase_rad'];sl=slice(k,k+26)
            def flat(key,width=None):
                x=physical_history[key][:,sl]
                return x.reshape(-1,width) if width else x.reshape(-1)
            feature=_state_features(flat('velocity_n',3),flat('quaternion_nb',4),flat('angular_velocity_b',3),
                flat('relative_phase_rad'),anchor[:,None].expand(-1,26).reshape(-1),flat('flap_frequency_hz')).reshape(len(commands),26,12)
            hidden=simulator.model.base_model._encode_history(feature,history_commands[:,sl],torch.ones(feature.shape[:2],device=feature.device,dtype=torch.bool))
            state=replace(state,**now,gru_hidden=hidden,phase_anchor=anchor)
        state,_=simulator.step(state,commands[:,k],dt[:,k]);states.append(state)
    return {f.name:torch.stack([getattr(s,f.name) for s in states],1).detach().cpu().numpy() for f in fields(initial)}


def kinematic_hold(initial,dt):
    """No learned dynamics: hold v/omega/f, integrate p/q/phase with native dt."""
    state=initial;states=[state]
    for k in range(dt.shape[1]):
        step=dt[:,k]
        state=replace(state,position_n=state.position_n+state.velocity_n*step[:,None],
            quaternion_nb=_normalize_quaternion(_quaternion_multiply(state.quaternion_nb,_delta_quaternion(state.angular_velocity_b*step[:,None]))),
            relative_phase_rad=torch.remainder(state.relative_phase_rad+2*torch.pi*state.flap_frequency_hz*step,2*torch.pi))
        states.append(state)
    return {key:torch.stack([getattr(s,key) for s in states],1).detach().cpu().numpy() for key in PHYSICAL}


def variation_ratio(predicted,truth,k):
    numerator=np.linalg.norm(predicted[:,:k+1].std(axis=1),axis=-1)
    denominator=np.linalg.norm(truth[:,:k+1].std(axis=1),axis=-1)
    valid=denominator>1e-6
    ratio=np.full(len(denominator),np.nan);ratio[valid]=numerator[valid]/denominator[valid]
    return ratio,denominator


def error_statistics(values,log_ids):
    values=np.asarray(values);log_ids=np.asarray(log_ids);finite=np.isfinite(values)
    if not finite.all():raise ValueError('nonfinite endpoint errors: cannot silently drop failed predictions')
    return dict(n=len(values),pooled_rmse=float(np.sqrt(np.mean(values**2))),
        equal_flight_rmse=float(np.mean([np.sqrt(np.mean(values[log_ids==log]**2)) for log in np.unique(log_ids)])),
        mean=float(values.mean()),median=float(np.median(values)),p90=float(np.quantile(values,.9)),p95=float(np.quantile(values,.95)),max=float(values.max()))
