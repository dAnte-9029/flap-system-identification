"""Exact forecast-error decomposition and strictly history-only diagnostic features."""
import numpy as np
from diagnose_reference_predictability_v6 import fit_ridge


def decompose(prediction, truth):
    """Time axis is1; all arrays are [window,time,axis], errors in float64."""
    p=np.asarray(prediction,dtype=float);t=np.asarray(truth,dtype=float)
    if p.shape!=t.shape or p.ndim!=3:raise ValueError('matching [window,time,axis] required')
    pc=p-p.mean(1,keepdims=True);tc=t-t.mean(1,keepdims=True)
    sp=np.sqrt(np.mean(pc**2,1));st=np.sqrt(np.mean(tc**2,1));cov=np.mean(pc*tc,1)
    bias=(p-t).mean(1);amplitude=(sp-st)**2
    shape=np.maximum(2*(sp*st-cov),0.)
    return dict(total_mse=np.mean((p-t)**2,1),signed_bias=bias,bias_mse=bias**2,
        centered_mse=np.mean((pc-tc)**2,1),amplitude_mse=amplitude,shape_mse=shape,
        predicted_std=sp,truth_std=st)


def summaries(values):
    t=(np.arange(values.shape[1])-values.shape[1]+1)*.02;t-=t.mean()
    slope=np.einsum('ntd,t->nd',values,t)/(t@t)
    return np.concatenate([values[:,-1],values.mean(1),values.std(1),slope],axis=1)


def history_features(x,kind):
    """No future references, labels, IDs, flight means or prediction errors as features."""
    state=np.asarray(x['history_state'])
    parts=[summaries(state),np.asarray(x['current_mode'])]
    if kind=='full':
        parts.extend(summaries(np.asarray(x[key])) for key in ['history_reference','history_actuator'])
    elif kind!='state':raise ValueError(kind)
    return np.concatenate(parts,axis=1).astype(float)


def fit_predict(features,targets,query_features,logs,kind):
    if kind=='constant':
        _,c,n=np.unique(logs,return_inverse=True,return_counts=True);weights=1/n[c];weights/=weights.sum()
        value=np.sum(targets*weights[:,None],axis=0)
        return np.tile(value,(len(query_features),1)),dict(target_mean=value)
    return fit_ridge(features,targets,query_features,logs,alpha=100.)


def score_mean_readout(base_prediction,truth,estimated_bias):
    """Algebraic constant rate-readout correction only; NOT a corrected dynamics rollout."""
    d=decompose(base_prediction,truth)
    residual=d['signed_bias']-estimated_bias
    return dict(mean_error_squared=residual**2,
        waveform_error_squared=residual**2+d['centered_mse'])
