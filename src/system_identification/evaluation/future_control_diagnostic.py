"""Frozen future-input replay diagnostics; no fitting or model selection."""
from dataclasses import replace
import numpy as np
import pandas as pd
from system_identification.models.trajectory import attitude_error_deg

METRICS=('position_rmse_m','velocity_rmse_m_s','attitude_error_deg','body_rate_rmse_rad_s')
GROUPS=('low','middle','high')


def hold_controls(batch):
    """Replace only future commands; preserve histories, truth and dt by identity."""
    held=np.repeat(batch.trajectory.controls[:,:1,:],batch.trajectory.controls.shape[1],axis=1)
    return replace(batch,trajectory=replace(batch.trajectory,controls=held))


def excitation(controls,scale,k):
    u=np.asarray(controls);scale=np.asarray(scale)
    if scale.shape!=(4,) or np.any(scale<=0) or not np.isfinite(scale).all():
        raise ValueError('four finite positive frozen control_std values required')
    if u.ndim!=3 or u.shape[2]!=4 or not 1<=k<=u.shape[1] or not np.isfinite(u).all():
        raise ValueError('invalid control tape/horizon')
    return np.sqrt(np.mean(((u[:,:k]-u[:,:1])/scale)**2,axis=(1,2)))


def train_thresholds(values):
    values=np.asarray(values)
    if values.ndim!=1 or not np.isfinite(values).all():raise ValueError('invalid train excitation')
    low,high=np.quantile(values,[.25,.75],method='linear')
    return float(low),float(high),bool(low<high)


def assign_groups(values,low,high):
    if not low<high:return np.full(len(values),'unavailable',dtype='<U11')
    # Ties deterministic: low <= q25, middle (q25,q75], high > q75.
    return np.where(values<=low,'low',np.where(values<=high,'middle','high'))


def squared_errors(prediction,truth):
    arrays=[]
    for field in ('position_n','velocity_n','quaternion_nb','angular_velocity_b'):
        p=getattr(prediction,field)[:,1:];t=getattr(truth,field)[:,1:]
        if p.shape!=t.shape or not np.isfinite(p).all():raise ValueError('nonfinite/incomplete prediction')
        if field=='quaternion_nb':
            angle=attitude_error_deg(p.reshape(-1,4),t.reshape(-1,4)).reshape(p.shape[:2])
            arrays.append(angle**2)
        else:arrays.append(np.sum((p-t)**2,axis=-1))
    return np.stack(arrays,axis=-1)


def interval_mse(error_squared,dt,k):
    if dt.shape!=error_squared.shape[:2] or np.any(dt<=0) or not np.isfinite(dt).all():
        raise ValueError('invalid native dt')
    if not 1<=k<=dt.shape[1]:raise ValueError('invalid interval length')
    # Right-endpoint rectangular weighting, t0 excluded.
    return np.sum(error_squared[:,:k]*dt[:,:k,None],axis=1)/np.sum(dt[:,:k],axis=1)[:,None]


def flight_metrics(origin_mse,log_ids,cohorts,mask,**meta):
    """RMSE within flight with equal origin weights, NOT duration-pooled origins."""
    rows=[]
    for log in sorted(set(log_ids[mask])):
        selected=mask&(log_ids==log)
        c=set(cohorts[selected])
        if len(c)!=1:raise ValueError('flight cohort mismatch')
        values=np.sqrt(np.mean(origin_mse[selected],axis=0))
        for metric,value in zip(METRICS,values):
            rows.append(dict(**meta,flight_id=log,cohort=next(iter(c)),n_origins=int(selected.sum()),metric=metric,value=float(value)))
    return rows


KEYS=['kind','condition','seed','group','group_rule','step','horizon_s','metric']


def aggregate(per_flight):
    if per_flight.duplicated(KEYS+['flight_id']).any():raise ValueError('duplicate flight metric')
    if not np.isfinite(per_flight.value).all():raise ValueError('nonfinite flight error')
    expanded=pd.concat([per_flight,per_flight.assign(cohort='ALL')],ignore_index=True)
    keys=KEYS+['cohort']
    per_seed=expanded.groupby(keys,sort=True).agg(value=('value','mean'),
        n_flights=('flight_id','nunique'),n_origins=('n_origins','sum')).reset_index()
    groupkeys=[k for k in keys if k!='seed']
    for _,g in per_seed.groupby(groupkeys):
        if set(g.seed)!={17,23,42}:raise ValueError('missing/extra seed')
        if g.n_flights.nunique()!=1 or g.n_origins.nunique()!=1:raise ValueError('seed coverage mismatch')
    multi=per_seed.groupby(groupkeys,sort=True).agg(mean=('value','mean'),std=('value','std'),
        min=('value','min'),max=('value','max'),n_seeds=('seed','nunique'),
        n_flights=('n_flights','first'),n_origins=('n_origins','first')).reset_index()
    return per_seed,multi


def gains(actual,hold):
    delta=np.asarray(hold)-np.asarray(actual)
    denominator=np.asarray(hold,dtype=float)
    relative=np.full_like(denominator,np.nan)
    np.divide(100*delta,denominator,out=relative,where=denominator!=0)
    return delta,relative


def paired_comparisons(per_seed,per_flight):
    """Positive gain favors Actual; strict one-to-one seed/flight pairs."""
    keys=[k for k in per_seed.columns if k not in ('condition','value')]
    actual=per_seed.query('condition=="Actual"').rename(columns={'value':'actual'}).drop(columns='condition')
    hold=per_seed.query('condition=="Hold"').rename(columns={'value':'hold'}).drop(columns='condition')
    paired=actual.merge(hold,on=keys,validate='one_to_one',how='outer',indicator=True)
    if not (paired._merge=='both').all():raise ValueError('Actual/Hold coverage mismatch')
    paired=paired.drop(columns='_merge')
    paired['absolute_gain'],paired['relative_gain_pct']=gains(paired.actual,paired.hold)
    paired['level']='seed'
    fkeys=['kind','group','group_rule','step','horizon_s','metric','flight_id','cohort','condition','n_origins']
    means=per_flight.groupby(fkeys).value.agg(['mean','count']).reset_index()
    if not (means['count']==3).all():raise ValueError('flight missing seed')
    fp=means.pivot(index=[k for k in fkeys if k!='condition'],columns='condition',values='mean').reset_index()
    fp=fp.rename(columns={'Actual':'actual','Hold':'hold'})
    if fp[['actual','hold']].isna().any().any():raise ValueError('unpaired flight')
    fp['absolute_gain'],fp['relative_gain_pct']=gains(fp.actual,fp.hold)
    fp['level']='flight_mean_over_seeds'
    macrokeys=['kind','group','group_rule','step','horizon_s','metric','cohort']
    macro=paired.groupby(macrokeys).agg(actual=('actual','mean'),hold=('hold','mean'),
        seeds_improved=('absolute_gain',lambda x:int((x>0).sum())),seeds_worse=('absolute_gain',lambda x:int((x<0).sum())),
        n_seeds=('seed','nunique'),n_flights=('n_flights','first'),n_origins=('n_origins','first')).reset_index()
    both=pd.concat([fp,fp.assign(cohort='ALL')],ignore_index=True)
    directions=both.groupby(macrokeys).absolute_gain.agg(
        flights_improved=lambda x:int((x>0).sum()),flights_worse=lambda x:int((x<0).sum()),flights_tied=lambda x:int((x==0).sum())).reset_index()
    macro=macro.merge(directions,on=macrokeys,validate='one_to_one')
    macro['absolute_gain'],macro['relative_gain_pct']=gains(macro.actual,macro.hold)
    macro['level']='macro'
    return pd.concat([paired,fp,macro],ignore_index=True)
