"""Frozen original four-component trajectory loss, with optional offline weights.

Caller supplies unchanged fold-fit loss_scales and component_weights. No model,
 optimizer, checkpoint, normalization fitting or data loading occurs here.
"""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from train_velocity_rate_learning import window_loss as original_window_loss
from system_identification.models.velocity_rate_gru import errors

def _clean_padding(pred,target,mask):
 # Finite padding preserves bitwise legacy behavior. Nonfinite padding must not
 # contaminate masked losses/gradients through 0*NaN.
 bad=(~mask)&(~torch.isfinite(pred).all(-1)|~torch.isfinite(target).all(-1))
 if bad.any():
  identity=torch.zeros_like(pred);identity[...,6]=1
  pred=torch.where(bad[...,None],identity,pred)
  target=torch.where(bad[...,None],identity,target)
 return pred,target

def _validate_weight(w,shape,mask=None):
 if tuple(w.shape)!=tuple(shape):raise ValueError('weight shape mismatch')
 w=w.detach()
 if mask is not None:w=torch.where(mask,w,torch.ones_like(w))
 if not torch.isfinite(w).all() or (w<1).any() or (w>2).any():raise ValueError('weights must be finite in [1,2]')
 return w

def window_loss(pred,target,mask,scales,component_weights,*,scheme='W0',step_weight=None):
 """Returns original per-window losses or W2 within-window normalized losses."""
 if scheme not in ('W0','W1','W2'):raise ValueError('unknown scheme')
 if pred.shape!=target.shape or pred.shape[-1]!=13 or mask.shape!=pred.shape[:-1]:raise ValueError('state/mask shape mismatch')
 mask=mask.bool();pred,target=_clean_padding(pred,target,mask)
 if scheme!='W2':return original_window_loss(pred,target,mask,scales,component_weights)
 if step_weight is None:raise ValueError('W2 requires precomputed step_weight')
 w=_validate_weight(step_weight,mask.shape,mask)
 if torch.all(w==1):return original_window_loss(pred,target,mask,scales,component_weights)
 loss=(errors(pred,target)/scales).square()*component_weights
 wm=w*mask
 return (loss.sum(-1)*wm).sum(-1)/wm.sum(-1).clamp_min(1)

def weighted_loss(pred,target,mask,scales,component_weights,*,scheme='W0',window_weight=None,step_weight=None):
 """Scalar batch loss; W1 is self-normalized inside this batch, as requested."""
 v=window_loss(pred,target,mask,scales,component_weights,scheme=scheme,step_weight=step_weight)
 if scheme!='W1':return v.mean()
 if window_weight is None:raise ValueError('W1 requires precomputed window_weight')
 w=_validate_weight(window_weight,v.shape)
 if torch.all(w==1):return v.mean()
 return (v*w).sum()/w.sum()
