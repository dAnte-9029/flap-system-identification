"""Validation-only window selection and diagnostics for a frozen simulator.

Observed-envelope gates measure data support, not certified physical safety.
No threshold is selected by optimizing validation performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from system_identification.evaluation.trajectory import (
    POSITION_COLUMNS, VELOCITY_COLUMNS, QUATERNION_COLUMNS, BODY_RATE_COLUMNS,
)
from system_identification.models.trajectory import attitude_error_deg

HORIZONS = {0.2:10, 0.5:25, 1.0:50, 2.0:100, 3.0:150, 5.0:250}
PHYSICAL_FIELDS = ('position_n','velocity_n','quaternion_nb','angular_velocity_b',
                   'relative_phase_rad','flap_frequency_hz')


def select_windows(samples, *, stride_steps=50):
    """Full 26-sample history and 251 future states in one contiguous segment."""
    rows, coverage = [], []
    for (log_id,segment_id), g in samples.loc[samples.valid_core].groupby(['log_id','segment_id'],sort=True):
        g=g.sort_values('sample_in_segment')
        stamps=g.timestamp_us.to_numpy()
        indices=g.sample_in_segment.to_numpy()
        if not np.array_equal(indices,np.arange(len(g))):
            raise ValueError(f'noncontiguous segment {log_id}:{segment_id}')
        dt=np.diff(stamps)*1e-6
        if np.any((dt<=0)|(dt>0.05)):
            raise ValueError(f'invalid time gap {log_id}:{segment_id}')
        count=0
        for start in range(25,len(g)-250,stride_steps):
            rows.append(dict(window_id=f'validation:{log_id}:{segment_id}:{start}:5s',
                             log_id=log_id,segment_id=segment_id,start_sample_in_segment=start,
                             state_sample_count=251,start_timestamp_us=int(stamps[start]),
                             start_time_s=float(stamps[start]*1e-6)))
            count+=1
        coverage.append(dict(log_id=log_id,segment_id=segment_id,valid_samples=len(g),
                             selected_starts=count,excluded_reason='' if count else 'shorter_than_26_history_plus_250_future'))
    if not rows:
        raise ValueError('no eligible five-second windows')
    return pd.DataFrame(rows),pd.DataFrame(coverage)


def magnitude(x):
    return np.linalg.norm(x,axis=-1)


def envelope_values(samples):
    g=samples.loc[samples.valid_core]
    values=dict(speed=magnitude(g[list(VELOCITY_COLUMNS)].to_numpy()),
                body_rate=magnitude(g[list(BODY_RATE_COLUMNS)].to_numpy()),
                frequency=g.flap_frequency_hz.to_numpy())
    a,alpha=[],[]
    for _,s in g.groupby(['log_id','segment_id']):
        s=s.sort_values('sample_in_segment')
        dt=np.diff(s.timestamp_us.to_numpy())*1e-6
        valid=(dt>0)&(dt<=0.05)&(np.diff(s.sample_in_segment.to_numpy())==1)
        a.extend(magnitude(np.diff(s[list(VELOCITY_COLUMNS)].to_numpy(),axis=0)[valid]/dt[valid,None]))
        alpha.extend(magnitude(np.diff(s[list(BODY_RATE_COLUMNS)].to_numpy(),axis=0)[valid]/dt[valid,None]))
    values.update(acceleration=np.asarray(a),angular_acceleration=np.asarray(alpha))
    return values


def describe(values):
    v=np.asarray(values).reshape(-1)
    v=v[np.isfinite(v)]
    if not len(v):
        return dict(count=0,min=None,max=None,p1=None,p99=None,mean=None,median=None)
    return dict(count=len(v),min=float(v.min()),max=float(v.max()),
                p1=float(np.quantile(v,.01)),p99=float(np.quantile(v,.99)),
                mean=float(v.mean()),median=float(np.median(v)))


def endpoint_errors(pred,truth,k):
    dp=pred['position_n'][:,k]-truth.position_n[:,k]
    dv=pred['velocity_n'][:,k]-truth.velocity_n[:,k]
    dw=pred['angular_velocity_b'][:,k]-truth.angular_velocity_b[:,k]
    phase=pred['relative_phase_rad'][:,k]-truth.relative_phase_rad[:,k]
    result=dict(position_horizontal_m=magnitude(dp[:,:2]), position_vertical_m=np.abs(dp[:,2]),
                position_m=magnitude(dp), velocity_m_s=magnitude(dv),
                attitude_deg=attitude_error_deg(pred['quaternion_nb'][:,k],truth.quaternion_nb[:,k]),
                body_rate_rad_s=magnitude(dw),
                frequency_hz=np.abs(pred['flap_frequency_hz'][:,k]-truth.flap_frequency_hz[:,k]),
                phase_rad=np.abs(np.arctan2(np.sin(phase),np.cos(phase))))
    for j,axis in enumerate('xyz'):
        result[f'velocity_{axis}_m_s']=np.abs(dv[:,j])
    for j,axis in enumerate('pqr'):
        result[f'body_rate_{axis}_rad_s']=np.abs(dw[:,j])
    return result


def aggregate(frame,keys,metric_names):
    rows=[]
    for key,g in frame.groupby(keys,sort=True):
        key=key if isinstance(key,tuple) else (key,)
        row=dict(zip(keys,key))
        row.update(n_rollouts=len(g),n_valid=int((~g.failed).sum()),n_failed=int(g.failed.sum()),
                   failure_pct=float(g.failed.mean()*100),numerical_failure_pct=float(g.numerical_failed.mean()*100))
        for name in metric_names:
            values=g[name].to_numpy()
            finite=values[np.isfinite(values)]
            row[f'{name}_finite_count']=len(finite)
            for label,q in [('median',.5),('p90',.9),('p95',.95)]:
                row[f'{name}_{label}']=float(np.quantile(finite,q)) if len(finite) else np.nan
            row[f'{name}_mean']=float(finite.mean()) if len(finite) else np.nan
            row[f'{name}_max']=float(finite.max()) if len(finite) else np.nan
            row[f'{name}_rmse']=float(np.sqrt(np.mean(finite**2))) if len(finite) else np.nan
            log_rmse=g.groupby('log_id')[name].apply(lambda v: np.sqrt(np.mean(v[np.isfinite(v)]**2)))
            row[f'{name}_equal_log_rmse']=float(log_rmse.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def stability(pred,diag,dt,train_envelope):
    """Per-step gate arrays; continue finite failed paths for failure analysis."""
    signals=dict(speed=magnitude(pred['velocity_n'][:,1:]),
                 body_rate=magnitude(pred['angular_velocity_b'][:,1:]),
                 frequency=pred['flap_frequency_hz'][:,1:],
                 acceleration=magnitude(diag['acceleration_b']),
                 angular_acceleration=magnitude(diag['angular_acceleration_b']))
    flags={}
    for name,x in signals.items():
        e=train_envelope[name]
        low=e['min']
        flags['outside_'+name]=(x<low)|(x>e['max'])
        flags['outside_p1_p99_'+name]=(x<e['p1'])|(x>e['p99'])
    flags['nonfinite']=np.zeros_like(dt,dtype=bool)
    for x in pred.values():
        v=np.isfinite(x[:,1:])
        flags['nonfinite'] |= ~v if v.ndim==2 else ~v.all(axis=-1)
    for x in diag.values():
        v=np.isfinite(x)
        flags['nonfinite'] |= ~v if v.ndim==2 else ~v.all(axis=-1)
    flags['quaternion_norm']=np.abs(magnitude(pred['quaternion_nb'][:,1:])-1)>1e-4
    flags['numerical']=flags['nonfinite']|flags['quaternion_norm']
    flags['envelope']=np.logical_or.reduce([flags['outside_'+n] for n in signals])
    clipped=sum(diag[n]>0 for n in ('frequency_clipped','derivative_clipped','drive_clipped','residual_clipped'))>0
    flags['clipping']=clipped
    flags['failed']=flags['numerical']|flags['envelope']|flags['clipping']
    return signals,flags
