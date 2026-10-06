"""Train-only bounded frequency balancing and isolated q loss change."""
import numpy as np
import torch
import train_control_windows_v2 as base
from reference_condition_diagnostics_v12 import weighted_terciles,bins


def frequency_weights(frequency,logs):
    frequency=np.asarray(frequency,dtype=float);logs=np.asarray(logs)
    cuts=weighted_terciles(frequency,logs);labels=bins(frequency,cuts)
    weights=np.ones(len(logs));records=[]
    for log in np.unique(logs):
        ids=np.flatnonzero(logs==log);names,code,count=np.unique(labels[ids],return_inverse=True,return_counts=True)
        raw=len(ids)/(len(names)*count[code])
        # Project scaled inverse-count weights to [.5,2], retaining mean one per flight.
        lo,hi=0.,4./raw.min()
        for _ in range(80):
            mid=(lo+hi)/2
            if np.clip(mid*raw,.5,2.).mean()<1:lo=mid
            else:hi=mid
        weights[ids]=np.clip((lo+hi)/2*raw,.5,2.)
        assert abs(weights[ids].mean()-1)<1e-12
        for name,n in zip(names,count):
            w=weights[ids][labels[ids]==name]
            records.append(dict(log_id=str(log),stratum=str(name),windows=int(n),weight=float(w[0]),original_mass=float(n/len(ids)),weighted_mass=float(w.sum()/len(ids))))
    return weights,cuts,labels,records


def weighted_q_loss(pred,target,model,weight):
    original=base.loss_fn(pred,target,model)
    q=((pred['rates_b'][:,:,1]-target['rates_b'][:,:,1])/model.rate_scale[1]).square().mean(1)
    return original+((weight-1)*q).mean()/3
