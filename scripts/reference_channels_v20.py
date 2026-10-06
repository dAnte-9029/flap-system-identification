"""Reference-mask experiments and input-only empirical support diagnostics."""
import numpy as np
import pandas as pd
from reference_condition_diagnostics_v12 import covariates

CHANNELS=['roll','pitch','yaw_rate','thrust','manual_yaw']
MASKS={'factual':31,'held':0,**{f'only_{c}':1<<j for j,c in enumerate(CHANNELS)},
       **{f'hold_{c}':31^(1<<j) for j,c in enumerate(CHANNELS)}}


def masked_reference(reference,mask):
    if mask<0 or mask>31:raise ValueError('five reference channels')
    out=reference[:,:1].expand_as(reference).clone()
    for j in range(5):
        if mask&(1<<j):out[:,:,j]=reference[:,:,j]
    return out


def features(x):
    r=np.asarray(x['future_reference'],float);h=np.asarray(x['history_state'],float)
    data={}
    for j,c in enumerate(CHANNELS):
        data[c+'_initial']=r[:,0,j];data[c+'_range']=np.ptp(r[:,:,j],axis=1)
        data[c+'_step_rms']=np.sqrt(np.mean(np.diff(r[:,:,j],axis=1)**2,axis=1));data[c+'_net']=r[:,-1,j]-r[:,0,j]
    out=pd.DataFrame(data);hist=covariates(x,1).add_prefix('history_')
    hist['history_q_mean']=h[:,:,4].mean(1);hist['history_p_mean']=h[:,:,3].mean(1)
    hist['history_p_std']=h[:,:,3].std(1);hist['history_r_std']=h[:,:,5].std(1)
    return pd.concat([out,hist],axis=1)


def weights(logs):
    _,code,n=np.unique(logs,return_inverse=True,return_counts=True)
    w=1/n[code];return w/w.sum()


def quantiles(v,logs,ps):
    order=np.argsort(v,kind='stable');cdf=np.cumsum(weights(logs)[order]);cdf[-1]=1
    return np.asarray(v)[order[np.searchsorted(cdf,ps)]]


def support(fit,query,logs):
    """Calibrate distance by nearest OTHER-flight fit window, no outcome labels."""
    fit=np.asarray(fit,float);query=np.asarray(query,float);w=weights(logs)
    mean=np.sum(fit*w[:,None],axis=0);scale=np.sqrt(np.sum((fit-mean)**2*w[:,None],axis=0)).clip(1e-4)
    z=(fit-mean)/scale;v=(query-mean)/scale
    distance=np.sqrt(np.mean((z[:,None,:]-z[None,:,:])**2,axis=-1))
    distance[np.asarray(logs)[:,None]==np.asarray(logs)[None,:]]=np.inf
    calibration=distance.min(1);threshold=float(quantiles(calibration,logs,[.95])[0])
    d=np.sqrt(np.mean((v[:,None,:]-z[None,:,:])**2,axis=-1));nearest=d.argmin(1)
    return dict(mean=mean,scale=scale,threshold=threshold,calibration=calibration,distance=d[np.arange(len(query)),nearest],nearest_fit_position=nearest)
