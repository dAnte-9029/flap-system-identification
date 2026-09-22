"""Explicit partition adapter; unchanged numerical errors and flight-first statistics."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from system_identification.evaluation.trajectory import evaluate_trajectory_predictions,ERROR_COLUMNS
from system_identification.evaluation.paper_baselines import HORIZONS
from system_identification.evaluation.future_control_diagnostic import METRICS,squared_errors,flight_metrics,interval_mse,gains


def endpoint_adapter(pred,batch,*,model,seed,date,allowlist):
    b=batch.trajectory
    if not set(b.log_ids)<=set(allowlist):raise ValueError('flight outside explicit partition allowlist')
    if len(set(b.window_ids))!=len(b.window_ids):raise ValueError('duplicate origin')
    if not np.isfinite(b.dt_s).all() or np.any((b.dt_s<=0)|(b.dt_s>.05)):raise ValueError('invalid native dt')
    for key,value in vars(pred).items():
        expected=getattr(b.truth,key)
        if value.shape!=expected.shape or not np.isfinite(value).all():raise ValueError('invalid prediction: '+key)
        np.testing.assert_allclose(value[:,0],expected[:,0],rtol=1e-6,atol=2e-5)
    if np.max(abs(np.linalg.norm(pred.quaternion_nb,axis=-1)-1))>1e-5:raise ValueError('nonunit quaternion')
    frame=evaluate_trajectory_predictions(pred,b.truth,model_name=model,split=date,window_ids=b.window_ids,
        log_ids=b.log_ids,segment_ids=b.segment_ids,horizon_steps=HORIZONS,dt_s=b.dt_s)
    frame['seed']=seed;frame['date']=date
    errors=squared_errors(pred,b.truth)
    for h,k in HORIZONS.items():
        f=frame[frame.horizon_s==h]
        np.testing.assert_allclose(f.observed_horizon_s,b.dt_s[:,:k].sum(1),rtol=0,atol=1e-12)
        np.testing.assert_allclose(f[list(ERROR_COLUMNS)],np.sqrt(errors[:,k-1]),rtol=1e-10,atol=1e-9)
    return frame,errors


KEYS=['date','model','history_steps','condition','kind','group','step','horizon_s','metric']


def aggregate(per):
    if per.duplicated(KEYS+['seed','flight_id']).any():raise ValueError('duplicate flight/seed')
    if not np.isfinite(per.value).all():raise ValueError('invalid per-flight error')
    seed=per.groupby(KEYS+['seed'],dropna=False,sort=True).agg(value=('value','mean'),
        n_flights=('flight_id','nunique'),n_origins=('n_origins','sum')).reset_index()
    rows=[]
    for keys,g in seed.groupby(KEYS,dropna=False,sort=True):
        row=dict(zip(KEYS,keys));deterministic=row['model']=='B0'
        complete=(len(g)==1 and g.seed.isna().all()) if deterministic else (len(g)==3 and set(g.seed)=={17,23,42})
        if g.n_flights.nunique()!=1 or g.n_origins.nunique()!=1:raise ValueError('seed coverage mismatch')
        row.update(mean=g.value.mean() if complete else np.nan,std=g.value.std(ddof=1) if complete and not deterministic else np.nan,
            min=g.value.min() if complete else np.nan,max=g.value.max() if complete else np.nan,
            n_seeds=0 if deterministic else len(g),n_flights=int(g.n_flights.iloc[0]),n_origins=int(g.n_origins.iloc[0]),
            status='valid' if complete else 'incomplete_seeds')
        rows.append(row)
    return seed,pd.DataFrame(rows)


COMPARISONS=[('GRU_vs_MLP','MLP','Actual','H26','Actual'),('H1_to_H26','H1','Actual','H26','Actual'),
             ('H13_to_H26','H13','Actual','H26','Actual'),('Actual_vs_Hold','H26','Hold','H26','Actual')]


def paired(seed,per):
    srows=[];frows=[]
    for name,rm,rc,cm,cc in COMPARISONS:
        for level,source in [('seed',seed),('flight',per)]:
            d=source[source.kind=='endpoint']
            if name!='Actual_vs_Hold':d=d[d.group=='ALL']
            keys=['date','group','step','horizon_s','metric']+(['seed'] if level=='seed' else ['flight_id'])
            a=d[(d.model==rm)&(d.condition==rc)];b=d[(d.model==cm)&(d.condition==cc)]
            if level=='flight':
                def means(x):
                    g=x.groupby(keys).agg(value=('value','mean'),count=('seed','nunique'),n_origins=('n_origins','first')).reset_index()
                    return g[g['count']==3].drop(columns='count')
                a=means(a);b=means(b)
            cols=keys+['value','n_origins']
            a=a[cols].rename(columns={'value':'reference_error'});b=b[cols].rename(columns={'value':'comparison_error'})
            pair=a.merge(b,on=keys+['n_origins'],how='outer',validate='one_to_one')
            pair['comparison']=name;pair['absolute_gain'],pair['relative_gain_pct']=gains(pair.comparison_error,pair.reference_error)
            pair['status']=np.where(pair[['reference_error','comparison_error']].isna().any(axis=1),'incomplete_pair','valid')
            (srows if level=='seed' else frows).append(pair)
    ps=pd.concat(srows,ignore_index=True);pf=pd.concat(frows,ignore_index=True);rows=[]
    keys=['comparison','date','group','step','horizon_s','metric']
    for kk,g in ps.groupby(keys):
        row=dict(zip(keys,kk));ok=len(g)==3 and set(g.seed)=={17,23,42} and (g.status=='valid').all()
        f=pf.copy()
        for k,v in row.items():f=f[f[k]==v]
        a=g.reference_error.mean() if ok else np.nan;b=g.comparison_error.mean() if ok else np.nan
        gain,percent=gains(b,a)
        row.update(reference_error=a,comparison_error=b,absolute_gain=float(gain),relative_gain_pct=float(percent),
            seeds_improved=int((g.absolute_gain>0).sum()),seeds_worse=int((g.absolute_gain<0).sum()),
            flights_improved=int((f.absolute_gain>0).sum()),flights_worse=int((f.absolute_gain<0).sum()),
            flights_tied=int((f.absolute_gain==0).sum()),n_flights=len(f),n_origins=int(g.n_origins.iloc[0]),status='valid' if ok else 'incomplete')
        rows.append(row)
    return ps,pf,pd.DataFrame(rows)
