"""Fixed-support offline alignment diagnostics; none is a deployable correction."""
import numpy as np
LAGS=np.arange(-5,6)
CORES=[np.arange(5,20),np.arange(30,45)]


def best_index(cost):
    order=np.array(sorted(range(11),key=lambda j:(abs(j-5),j)))
    return order[np.argmin(cost[:,order],axis=1)]


def curves(prediction,truth,ids):
    p=np.asarray(prediction,dtype=float);t=np.asarray(truth,dtype=float)
    if p.shape!=t.shape or p.ndim!=2 or min(ids)<5 or max(ids)+5>=p.shape[1]:raise ValueError('invalid common support')
    target=t[:,ids];target=target-target.mean(1,keepdims=True);unit=[];gain=[];corr=[];coeff=[]
    for lag in LAGS:
        z=p[:,ids+lag];z=z-z.mean(1,keepdims=True)
        cov=np.sum(z*target,1);var=np.sum(z*z,1)
        a=np.clip(cov/np.maximum(var,1e-12),0,3)
        unit.append(np.mean((z-target)**2,1));gain.append(np.mean((a[:,None]*z-target)**2,1));coeff.append(a)
        corr.append(cov/np.sqrt(np.maximum(var*np.sum(target*target,1),1e-24)))
    return dict(unit=np.stack(unit,1),gain=np.stack(gain,1),coefficient=np.stack(coeff,1),correlation=np.stack(corr,1))


def effective_indices(timestamp,sample,query):
    t=np.asarray(timestamp,dtype=np.int64);s=np.asarray(sample,dtype=np.int64);q=np.asarray(query,dtype=np.int64)
    e=np.maximum(t,s)
    if np.any(np.diff(t)<0) or np.any(np.diff(e)<0):raise ValueError('nonmonotonic publication/effective time')
    ix=np.searchsorted(e,q,side='right')-1
    if (ix<0).any():raise ValueError('query precedes first eligible message')
    assert (t[ix]<=q).all() and (s[ix]<=q).all()
    return ix
