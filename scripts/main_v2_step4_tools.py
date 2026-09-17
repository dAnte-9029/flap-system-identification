"""Pure offline diagnostic math; filters must never feed simulator inference."""
import numpy as np
from scipy.signal import butter, sosfiltfilt


def harmonic_design(phi, order=3):
    phi=np.asarray(phi)
    return np.column_stack([np.ones(len(phi))]+[f(k*phi) for k in range(1,order+1) for f in (np.sin,np.cos)])


def offline_derivatives(t, y, cutoffs=(4.,6.,8.,12.)):
    """Native-interval derivatives; zero-phase filters are target diagnosis only."""
    t=np.asarray(t);y=np.asarray(y);dt=np.diff(t)
    if np.any(dt<=0):raise ValueError('nonmonotonic time')
    result={'D0_raw':np.diff(y,axis=0)/dt[:,None]}
    grid=np.arange(t[0],t[-1]+1e-8,.02)
    z=np.column_stack([np.interp(grid,t,a) for a in y.T])
    for cutoff in cutoffs:
        filtered=sosfiltfilt(butter(4,cutoff,fs=50,output='sos'),z,axis=0)
        native=np.column_stack([np.interp(t,grid,a) for a in filtered.T])
        result[f'D1_lp{cutoff:g}']=np.diff(native,axis=0)/dt[:,None]
    for lag in (2,5,10):
        left=np.clip(np.arange(len(dt))-(lag-1)//2,0,len(t)-lag-1);right=left+lag
        result[f'D2_center{lag}']=(y[right]-y[left])/(t[right]-t[left])[:,None]
    return result


def integrate_intervals(derivative, dt, steps):
    c=np.vstack([np.zeros((1,derivative.shape[1])),np.cumsum(derivative*np.asarray(dt)[:,None],axis=0)])
    return c[steps:]-c[:-steps]


def vector_r2(pred,truth):
    return 1-float(np.sum((np.asarray(pred)-truth)**2)/max(np.sum((truth-np.mean(truth,axis=0))**2),1e-20))


def nearest_indices(query, reference, forbidden, k=20, device='cuda:1'):
    import torch
    ref=torch.as_tensor(reference,dtype=torch.float32,device=device)
    out=[];dist=[]
    for start in range(0,len(query),128):
        q=torch.as_tensor(query[start:start+128],dtype=torch.float32,device=device)
        d=torch.cdist(q,ref)/np.sqrt(ref.shape[1])
        d[torch.as_tensor(forbidden[start:start+128],device=device)]=torch.inf
        v,i=torch.topk(d,k,largest=False)
        if not torch.isfinite(v).all():raise ValueError('insufficient admissible neighbors')
        out.append(i.cpu().numpy());dist.append(v.cpu().numpy())
    return np.concatenate(out),np.concatenate(dist)
