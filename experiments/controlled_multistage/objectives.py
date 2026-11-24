import torch
from workspace import BASELINES, conditional_loss


def training_loss(name,model,b,units,drift_scale):
    base,score,errors,valid=conditional_loss(model,b,units)
    if name in BASELINES:
        return base,score,errors,valid
    late=torch.stack((errors[:,2:,0],errors[:,2:,2]),-1)
    mask=torch.stack((valid[:,2:,0],valid[:,2:,2]),-1)
    drift=(late/drift_scale*mask).sum()/mask.sum().clamp_min(1)
    loss=base+.5*drift
    if name in ('s3b','s3c'):
        # Equal relative changes in each endpoint have comparable gradients.
        sq=errors[:,1:]
        ok=valid[:,1:]
        mse=(sq*ok).sum(0)/ok.sum(0).clamp_min(1)
        terms=torch.log1p(mse/(1e-4*units.square()))
        weights=[.5,1.,2.,3.][:terms.shape[0]]
        weight=.16 if name=='s3b' else .05
        loss=loss+weight*sum(w*terms[i].mean() for i,w in enumerate(weights))/sum(weights)
    return loss,score,errors,valid
