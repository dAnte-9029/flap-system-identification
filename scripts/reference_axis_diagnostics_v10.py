"""Fixed-support amplitude/timing oracles for diagnosis, not predictive corrections."""
import numpy as np


def align_diagnostic(prediction,truth,max_lag=5):
    """Compare every lag on the same truth interior; intercept is removed per crop."""
    p=np.asarray(prediction,dtype=float);t=np.asarray(truth,dtype=float)
    if p.shape!=t.shape or p.ndim!=2 or p.shape[1]<=2*max_lag+2:raise ValueError('insufficient common support')
    ids=np.arange(max_lag,p.shape[1]-max_lag)
    target=t[:,ids];target=target-target.mean(1,keepdims=True)
    zero=p[:,ids];zero=zero-zero.mean(1,keepdims=True)
    baseline=np.mean((zero-target)**2,1)
    rows=[]
    for lag in range(-max_lag,max_lag+1):
        candidate=p[:,ids+lag];candidate=candidate-candidate.mean(1,keepdims=True)
        gain=np.sum(candidate*target,1)/np.maximum(np.sum(candidate**2,1),1e-12);gain=np.clip(gain,0.,3.)
        mse=np.mean((candidate*gain[:,None]-target)**2,1)
        corr=np.sum(candidate*target,1)/np.sqrt(np.maximum(np.sum(candidate**2,1)*np.sum(target**2,1),1e-24))
        rows.append((gain,mse,corr))
    gain=np.stack([r[0] for r in rows],1);mse=np.stack([r[1] for r in rows],1);corr=np.stack([r[2] for r in rows],1)
    # Stable tie preference: smallest |lag|, then negative lag.
    order=sorted(range(2*max_lag+1),key=lambda j:(abs(j-max_lag),j));best=np.array(order)[np.argmin(mse[:,order],1)];index=np.arange(len(p))
    return dict(core_centered_mse=baseline,zero_lag_gain_mse=mse[:,max_lag],best_lag_gain_mse=mse[index,best],
        zero_lag_gain=gain[:,max_lag],zero_lag_correlation=corr[:,max_lag],best_lag_gain=gain[index,best],
        best_lag_s=(best-max_lag)*.02,best_correlation=corr[index,best],
        predicted_std=np.std(p,1),truth_std=np.std(t,1))
