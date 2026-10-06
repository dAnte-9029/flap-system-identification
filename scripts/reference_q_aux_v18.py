"""Keep original 50-step objective and add a fixed first-step q term."""
import train_control_windows_v2 as base
from reference_single_step_v17 import errors, METRICS

OUT=base.ROOT/'docs/analysis/results/control_windows_v18'
V8=OUT.parent/'control_windows_v8'
AUX_WEIGHT=1/3


def loss_terms(pred,target,model,weight=AUX_WEIGHT):
    full=base.loss_fn(pred,target,model)
    q=((pred['rates_b'][:,0,1]-target['rates_b'][:,0,1])/model.rate_scale[1]).square().mean()
    return full,weight*q


def auxiliary_loss(pred,target,model,weight=AUX_WEIGHT):
    full,aux=loss_terms(pred,target,model,weight)
    return full+aux
