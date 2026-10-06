"""Fixed flight-balanced per-prefix regression; fitting stays outside inference."""
import numpy as np
from reference_error_structure_v9 import history_features
from diagnose_reference_predictability_v6 import fit_ridge


def prefix_features(x):
    past=history_features(x,'full')
    future=x['future_reference'].numpy().astype(float)
    start=x['history_reference'][:,-1:,:].numpy().astype(float)
    return np.concatenate([np.broadcast_to(past[:,None,:],(len(past),future.shape[1],94)),
        future-start,np.cumsum(future,axis=1)/np.arange(1,future.shape[1]+1)[None,:,None]-start],axis=-1)


def fit_prefix(features,targets,fit,query,logs):
    if set(logs[fit]) & set(logs[query]):
        raise ValueError('fit/query flights must be disjoint')
    means=[];scales=[];coefficients=[];intercepts=[];predictions=[]
    for k in range(50):
        pred,p=fit_ridge(features[fit,k],targets[fit,k:k+1],features[query,k],logs[fit],alpha=100.)
        predictions.append(pred[:,0]);means.append(p['mean']);scales.append(p['scale'])
        coefficients.append(p['coefficient'][:,0]);intercepts.append(p['target_mean'][0])
    return np.stack(predictions,1),dict(mean=np.array(means),scale=np.array(scales),coefficient=np.array(coefficients),target_mean=np.array(intercepts))
