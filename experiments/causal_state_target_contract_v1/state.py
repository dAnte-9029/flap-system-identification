"""Causal two-observation rate estimate, independent of offline targets."""
import numpy as np

def causal_rate(measurement_us,publication_us,rate,state_time_us,decision_us,c0):
    """Last two valid published observations. All three axes share admission.

    Bracketed t0: linear interpolation. Outside bracket: distance to nearest
    endpoint <= min(15000 us, actual observation interval). Otherwise C0.
    No B target or future publication is an input. Times are integer us.
    """
    tm=np.asarray(measurement_us,np.int64);tp=np.asarray(publication_us,np.int64);y=np.asarray(rate,float)
    t=np.atleast_1d(state_time_us).astype(np.int64);dec=np.atleast_1d(decision_us).astype(np.int64);base=np.asarray(c0,float)
    assert y.shape==(len(tm),3) and base.shape==(len(t),3) and dec.shape==t.shape
    assert np.all(np.diff(tp)>0),'publication clock must be strictly increasing'
    valid=(tm>0)&(tm<=tp)&np.isfinite(y).all(1)
    ids=np.flatnonzero(valid);out=base.copy();reason=np.full(len(t),'insufficient_valid_observations',dtype='U40');source=np.full((len(t),2),-1,np.int64);gap=np.zeros(len(t),np.int64);distance=np.zeros(len(t),np.int64)
    if len(ids)<2:return out,dict(reason=reason,source_indices=source,gap_us=gap,extrapolation_us=distance)
    last=np.searchsorted(tp[ids],dec,side='right')-1;exists=last>=1;safe=np.maximum(last,1);a=ids[safe-1];b=ids[safe];gap=tm[b]-tm[a]
    finite=np.isfinite(base).all(1);ordered=gap>0;near=gap<=20000
    distance=np.maximum(np.maximum(tm[a]-t,t-tm[b]),0)
    reason[exists]='invalid_or_nonmonotonic';reason[exists&ordered&finite]='gap_over_20ms'
    good=exists&ordered&near&finite;reason[good]='extrapolation_limit'
    good&=(distance<=15000)&(distance<=gap)
    source[exists]=np.c_[a[exists],b[exists]]
    ratio=np.divide(t-tm[a],gap,out=np.zeros(len(t),float),where=gap>0)
    estimate=y[a]+ratio[:,None]*(y[b]-y[a]);out[good]=estimate[good]
    reason[good]='interpolation';reason[good&(t>tm[b])]='forward_extrapolation';reason[good&(t<tm[a])]='backward_extrapolation'
    assert np.all(tp[b[good]]<=dec[good]) and np.all(tp[a[good]]<=dec[good])
    return out,dict(reason=reason,source_indices=source,gap_us=gap,extrapolation_us=distance)

def apply_rates(history,initial,history_rates,lengths):
    """Change only valid historical p/q/r and matching initial; preserve padding."""
    h=history.copy();s=initial.copy();valid=np.arange(h.shape[1])[None,:]<np.asarray(lengths)[:,None]
    h[...,10:13][valid]=history_rates[valid];s[:,10:13]=h[np.arange(len(h)),np.asarray(lengths)-1,10:13]
    return h,s
