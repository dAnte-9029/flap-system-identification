"""Fixed parent-preserving local sampling and joint rate-increment supervision."""
import numpy as np
import torch
import train_control_windows_v2 as base

LOCAL_WEIGHT=1.0


def balanced_offsets(draws,fit_indices,fold,seed):
    """Independent per-parent shuffled cycles; every complete cycle covers0..49.

    Never consumes model initialization RNG or changes the parent draw sequence.
    No query inputs, labels, model errors or outcome-selected thresholds enter.
    """
    draws=np.asarray(draws)
    if draws.ndim!=2 or not np.issubdtype(draws.dtype,np.integer):raise ValueError('integer parent draw matrix required')
    if not set(draws.ravel())<=set(fit_indices):raise ValueError('query parent in training draws')
    rng={int(i):np.random.default_rng(np.random.SeedSequence([34,int(fold),int(seed),int(i)])) for i in fit_indices}
    counts={int(i):0 for i in fit_indices};permutations={}
    offsets=np.empty(draws.shape,dtype=np.int64)
    for pos,parent in np.ndenumerate(draws):
        parent=int(parent);j=counts[parent]%50
        if j==0:permutations[parent]=rng[parent].permutation(50)
        offsets[pos]=permutations[parent][j];counts[parent]+=1
    return offsets


def select_local(x,y,parents,offsets,fit_indices):
    parents=np.asarray(parents);offsets=np.asarray(offsets)
    if parents.ndim!=1 or offsets.shape!=parents.shape:raise ValueError('one local anchor per parent required')
    if not np.issubdtype(parents.dtype,np.integer) or not np.issubdtype(offsets.dtype,np.integer):raise ValueError('integer indices required')
    if not set(parents)<=set(fit_indices):raise ValueError('query parent selected for local loss')
    if not ((offsets>=0)&(offsets<50)).all():raise ValueError('offset outside parent support')
    keep=set(x)-{'history_reference','future_reference'}
    local_x={k:x[k][parents,offsets] for k in keep}
    local_y={k:v[parents,offsets][:,None] for k,v in y.items()}
    return local_x,local_y


def local_increment_loss(pred,target,rate_scale):
    """Common observed initial rate cancels in predicted-minus-true increments.

    Use rate units/rate_scale, no division by dt and no instantaneous-acceleration
    target. Equivalent to normalized local endpoint MSE, averaged over all3axes.
    """
    return ((pred['rates_b']-target['rates_b'])/rate_scale).square().mean()


def objective(model,full_x,full_y,local_x=None,local_y=None,weight=LOCAL_WEIGHT):
    if weight<0 or not np.isfinite(weight):raise ValueError('invalid loss weight')
    full=base.loss_fn(model(**full_x),full_y,model)
    # Exact legacy computation graph for the zero-weight reproducibility control.
    if weight==0:return full,full,None
    if local_x is None or local_y is None:raise ValueError('local batch required')
    local=local_increment_loss(model(**local_x),local_y,model.rate_scale)
    return full+weight*local,full,local
