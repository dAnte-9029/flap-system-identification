"""Causal controller-history extraction and algebraic scaling diagnostics."""
import numpy as np
from reference_timing_v14 import effective_indices

FEATURES=['integral_p_post','integral_q_post','integral_r_post',
          'compression_p','compression_q','compression_r','integral_publication_age_s','compression_publication_age_s']


def history_block(status,compression,query):
    q=np.asarray(query,dtype=np.int64)
    si=effective_indices(status['timestamp'],status['timestamp'],q)
    gi=effective_indices(compression['timestamp'],compression['timestamp'],q)
    values=np.column_stack([status[f'{ax}speed_integ'][si] for ax in ['roll','pitch','yaw']]+
                           [compression[f'compression_gains[{j}]'][gi] for j in range(3)]+
                           [(q-status['timestamp'][si])*1e-6,(q-compression['timestamp'][gi])*1e-6])
    if not np.isfinite(values).all():raise ValueError('invalid historical controller observation')
    return values.astype(np.float32),si,gi


def infer_scaling(p_terms,ff_terms,rates,p_gain,ff_gain,epsilon=1e-3):
    """Requires EXACT shared gyro sample time and a non-VTOL rate controller.

    Diagnostics only. Cannot use asynchronously held rate references here.
    """
    p_gain=np.asarray(p_gain,float);ff_gain=np.asarray(ff_gain,float)
    if (p_gain<=0).any() or (ff_gain<=0).any():raise ValueError('positive gains required')
    used_reference=np.asarray(p_terms,float)/p_gain+rates
    ff=np.asarray(ff_terms,float);valid=abs(ff)>=epsilon
    ratio=np.divide(used_reference*ff_gain,ff,out=np.full_like(ff,np.nan,dtype=float),where=valid)
    valid &= np.isfinite(ratio)&(ratio>0)
    ratio=np.where(valid,ratio,np.nan)
    strength=np.where(valid,abs(ff),-1.)
    chosen=np.argmax(strength,axis=1)
    selected=ratio[np.arange(len(ratio)),chosen]
    return selected,ratio,used_reference


def fit_history_stats(block):
    values=np.asarray(block,float)
    if values.ndim!=2 or values.shape[1]!=8 or not np.isfinite(values).all():raise ValueError('invalid fit rows')
    return values.mean(0),values.std(0).clip(1e-4)
