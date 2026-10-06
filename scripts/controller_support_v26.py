"""Outcome-free, fit-flight-only support diagnostics for historical observations."""
import numpy as np

BLOCKS={'integral':[0,1,2],'compression':[3,4,5],'age':[6,7],'all':list(range(8))}


def weights(logs):
    _,codes,counts=np.unique(logs,return_inverse=True,return_counts=True)
    w=1./counts[codes]
    return w/w.sum()


def quantile(values,logs,probability):
    v=np.asarray(values,dtype=float);order=np.argsort(v,kind='stable')
    if not np.isfinite(v).all():raise ValueError('nonfinite observations')
    c=np.cumsum(weights(logs)[order]);c[-1]=1.
    return v[order[np.searchsorted(c,probability)]]


def summaries(history,channels):
    h=np.asarray(history,dtype=float)[:,:,channels]
    return np.concatenate([h.mean(1),h.std(1),h[:,-1]],axis=1)


def support(fit,fit_logs,query):
    """Calibration excludes the same flight; query cannot affect scaling/threshold."""
    fit=np.asarray(fit,dtype=float);query=np.asarray(query,dtype=float)
    if len(np.unique(fit_logs))<2:raise ValueError('multiple fit flights required')
    w=weights(fit_logs);mean=np.sum(w[:,None]*fit,axis=0)
    scale=np.sqrt(np.sum(w[:,None]*(fit-mean)**2,axis=0));scale=np.maximum(scale,1e-6)
    a=(fit-mean)/scale;b=(query-mean)/scale
    distance=np.sqrt(np.mean((a[:,None]-a[None,:])**2,axis=2))
    distance[np.asarray(fit_logs)[:,None]==np.asarray(fit_logs)[None,:]]=np.inf
    calibration=distance.min(1)
    query_distance=np.sqrt(np.mean((b[:,None]-a[None,:])**2,axis=2))
    neighbor=query_distance.argmin(1);qd=query_distance[np.arange(len(query)),neighbor]
    threshold=float(quantile(calibration,fit_logs,.95))
    return dict(mean=mean,scale=scale,fit_distance=calibration,query_distance=qd,
                neighbor=neighbor,threshold=threshold)


def eta_squared(values,logs):
    w=weights(logs);m=np.sum(w*values);total=np.sum(w*(values-m)**2)
    between=np.mean([(values[np.asarray(logs)==log].mean()-m)**2 for log in np.unique(logs)])
    return between/total if total>1e-14 else np.nan


def correlation(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if len(a)<8 or np.std(a)<1e-12 or np.std(b)<1e-12:return np.nan
    return np.corrcoef(a,b)[0,1]


def q_errors(pred,true):
    residual=pred-true
    total=np.mean(residual**2,axis=1);bias=residual.mean(1)**2
    centered=np.mean((residual-residual.mean(1,keepdims=True))**2,axis=1)
    return dict(mse=total,bias_mse=bias,centered_mse=centered,
                early_mse=np.mean(residual[:,:25]**2,axis=1),late_mse=np.mean(residual[:,25:]**2,axis=1))
