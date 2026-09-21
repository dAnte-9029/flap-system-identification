"""Control-only grouping and paired flight statistics; no models or fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd

CHANNELS = ('drive', 'common', 'differential', 'rudder')
BIN_NAMES = ('Low', 'Medium', 'High', 'Top20')


def control_coordinates(controls):
    u = np.asarray(controls, dtype=float)
    if u.shape[-1] != 4 or not np.isfinite(u).all():
        raise ValueError('finite four-channel controls required')
    return np.stack((u[..., 0], (u[..., 1]+u[..., 2])/2,
                     (u[..., 1]-u[..., 2])/2, u[..., 3]), axis=-1)


def activity(controls, reference, scales):
    """Controls contain only t0...t(K-1), the tape actually applied to K steps."""
    u = control_coordinates(controls)
    reference, scales = np.asarray(reference), np.asarray(scales)
    if u.ndim != 3 or u.shape[1] < 1 or reference.shape != (4,) or scales.shape != (4,):
        raise ValueError('invalid activity array dimensions')
    if not np.isfinite(scales).all() or np.any(scales <= 0) or not np.isfinite(reference).all():
        raise ValueError('invalid frozen training scales/reference')
    magnitude = np.sqrt(np.mean((u-reference)**2, axis=1))
    delta = np.max(np.abs(u-u[:, :1]), axis=1)
    tv = np.sum(np.abs(np.diff(u, axis=1)), axis=1)
    normalized = delta/scales
    return dict(magnitude=magnitude, delta=delta, tv=tv, normalized_delta=normalized,
                total=np.sqrt(np.mean(normalized**2, axis=-1)))


def control_bins(scores):
    """Global, per-horizon control quantiles; retain ties, never rank on errors.

    Low <= q33, Medium (q33,q67], High > q67. Top20 >= q80 and
    strictly positive: a constant command is never called strong excitation.
    Ties may make realized fractions differ from nominal 33/34/33 or 20%.
    """
    scores=np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError('finite nonnegative scores required')
    a,b,c=np.quantile(scores,[.33,.67,.8],method='linear')
    labels=np.where(scores<=a,'Low',np.where(scores<=b,'Medium','High'))
    top=(scores>=c)&(scores>0)
    return labels,top,dict(q33=float(a),q67=float(b),q80=float(c),
        low=int(np.sum(labels=='Low')),medium=int(np.sum(labels=='Medium')),
        high=int(np.sum(labels=='High')),top20=int(top.sum()),zero_score=int(np.sum(scores==0)))


def attitude_rotation_vector_error(predicted, truth):
    """Principal Log(q_truth^-1 * q_pred) in truth-body axes, degrees.

    Component diagnostics are roll/pitch/yaw associated, not Euler differences
    or three independent rotations. Their Euclidean norm is geodesic error.
    """
    p=np.asarray(predicted,dtype=float);t=np.asarray(truth,dtype=float)
    p=p/np.linalg.norm(p,axis=-1,keepdims=True)
    t=t/np.linalg.norm(t,axis=-1,keepdims=True)
    w=np.sum(p*t,axis=-1)
    v=t[...,:1]*p[...,1:]-p[...,:1]*t[...,1:]-np.cross(t[...,1:],p[...,1:])
    v=np.where((w<0)[...,None],-v,v);w=np.abs(w)
    norm=np.linalg.norm(v,axis=-1)
    angle=2*np.arctan2(norm,w)
    scale=np.divide(angle,norm,out=np.full_like(angle,2.),where=norm>1e-12)
    return np.rad2deg(v*scale[...,None])


def bootstrap_weights(cohorts, *, draws=10000, seed=17023):
    """Equal-flight means; resample flights within cohort with fixed counts."""
    cohorts=np.asarray(cohorts);n=len(cohorts)
    if n<1:
        raise ValueError('bootstrap requires flights')
    rng=np.random.default_rng(seed)
    weights=np.zeros((draws,n))
    for c in sorted(set(cohorts)):
        indices=np.flatnonzero(cohorts==c);size=len(indices)
        weights[:,indices]=rng.multinomial(size,np.full(size,1/size),size=draws)/n
    return weights


def paired_statistics(b2, b3, metadata, metric_names, *, draws=10000):
    """Both descriptive origin-paired errors and flight-paired RMS comparisons.

    CI resampling unit is an entire flight, not an origin. Quantile thresholds
    remain fixed. No early selection, fitted model or p-value is involved.
    """
    b2=np.asarray(b2);b3=np.asarray(b3)
    if b2.shape!=b3.shape or b2.shape!=(len(metadata),len(metric_names)):
        raise ValueError('paired arrays must have identical keyed shapes')
    if not np.isfinite(b2).all() or not np.isfinite(b3).all():
        raise ValueError('nonfinite paired errors')
    delta=b3-b2
    flights=[];mean_delta=[];rms_delta=[];win_fraction=[]
    for log in sorted(metadata.log_id.unique()):
        indices=np.flatnonzero(metadata.log_id.to_numpy()==log)
        cohorts=metadata.iloc[indices].cohort.unique()
        if len(cohorts)!=1:
            raise ValueError('one flight assigned to multiple cohorts')
        x,y=b2[indices],b3[indices];d=delta[indices]
        rm2=np.sqrt(np.mean(x*x,axis=0));rm3=np.sqrt(np.mean(y*y,axis=0))
        mean_delta.append(d.mean(axis=0));rms_delta.append(rm3-rm2)
        win_fraction.append((d<0).mean(axis=0))
        for j,m in enumerate(metric_names):
            flights.append(dict(log_id=log,cohort=cohorts[0],n_origins=len(indices),metric=m,
                B2_error=float(rm2[j]),B3_error=float(rm3[j]),delta=float(rm3[j]-rm2[j]),
                mean_paired_difference=float(d[:,j].mean()),
                median_paired_difference=float(np.median(d[:,j])),
                B3_origin_win_fraction=float(np.mean(d[:,j]<0))))
    flight_frame=pd.DataFrame(flights)
    flight_cohorts=flight_frame.drop_duplicates('log_id').cohort.to_numpy()
    weights=bootstrap_weights(flight_cohorts,draws=draws)
    mean_delta=np.stack(mean_delta);rms_delta=np.stack(rms_delta)
    ci_mean=np.quantile(weights@mean_delta,[.025,.975],axis=0)
    ci_rms=np.quantile(weights@rms_delta,[.025,.975],axis=0)
    records=[]
    for j,m in enumerate(metric_names):
        records.append(dict(metric=m,n_origins=len(metadata),n_flights=len(mean_delta),
            origin_pooled_mean_difference=float(delta[:,j].mean()),
            origin_pooled_median_difference=float(np.median(delta[:,j])),
            origin_pooled_B3_win_fraction=float(np.mean(delta[:,j]<0)),
            flight_equal_mean_paired_difference=float(mean_delta[:,j].mean()),
            mean_paired_ci95_low=float(ci_mean[0,j]),mean_paired_ci95_high=float(ci_mean[1,j]),
            flight_equal_mean_RMSE_difference=float(rms_delta[:,j].mean()),
            RMSE_difference_ci95_low=float(ci_rms[0,j]),RMSE_difference_ci95_high=float(ci_rms[1,j]),
            flight_equal_B3_origin_win_fraction=float(np.mean(win_fraction,axis=0)[j]),
            B3_flight_wins=int(np.sum(rms_delta[:,j]<0)),
            B2_flight_wins=int(np.sum(rms_delta[:,j]>0)),
            flight_ties=int(np.sum(rms_delta[:,j]==0))))
    return pd.DataFrame(records),flight_frame
