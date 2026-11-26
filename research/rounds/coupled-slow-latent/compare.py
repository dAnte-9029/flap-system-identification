"""Describe paired flight-level gains; positive differences are improvements."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/zn/flap-system-identification')
STUDY=ROOT/'research/experiments/study-27657a60d6e6'

def scores(path):
    ep=pd.read_csv(path/'endpoints.csv').set_index('flight')
    native=pd.read_csv(path/'native_by_flight.csv').set_index('flight')
    groups=pd.read_csv(path/'trends_by_flight.csv').set_index(['group','flight'])
    shift=pd.read_csv(path/'shift_by_flight.csv').set_index(['horizon','flight'])
    result={k:ep[[k]] for k in ep}
    result.update({k:native[[k]] for k in native})
    for period in (.5,1.,2.):
        for shift_name in ('minus125','minus20'):
            result[f'{period}_{shift_name}']=shift.loc[period][[shift_name]]
    for key in ('full_all','full_high','future_first'):
        values=groups.loc[key]
        result[key]=values[['error']]
        result[key+'_align']=values[['dot','pred2','reference']]
    return result

def bootstrap(a,b,align,subset=None):
    a,b=a.align(b,join='inner',axis=0)
    if subset is not None:
        a=a.loc[a.index.intersection(subset)];b=b.loc[a.index]
    assert len(a)>0
    idx=np.random.default_rng(17).integers(len(a),size=(10000,len(a)))
    def reduce(v):
        v=v.to_numpy()
        def statistic(mean):
            if align:return mean[...,0]/np.sqrt(mean[...,1]*mean[...,2])
            return np.sqrt(mean[...,0])
        return statistic(v.mean(0)),statistic(v[idx].mean(1)),statistic(v)
    ref,bsr,fr=reduce(a);test,bst,ft=reduce(b)
    sign=1 if align else -1
    delta=sign*(test-ref)
    interval=np.quantile(sign*(bst-bsr),[.025,.975])
    return dict(ref=ref,proposed=test,gain=delta,improved=int((sign*(ft-fr)>0).sum()),flights=len(a),ci95=interval.tolist())

def main():
    state=json.loads((STUDY/'state.json').read_text())
    eid=next(iter(state['experiments']))
    anchor=STUDY/'baseline/attempt-1/seed-17/output'
    candidate=STUDY/eid/'attempt-1/seed-17/output'
    runs={tag:scores(p) for tag,p in [('original',anchor/'original'),('matched',anchor/'normal'),('coupled',candidate/'normal'),('held',candidate/'held_s0')]}
    with np.load(ROOT/'outputs/autopilot_flight_dynamics_20261124/validation.npz') as z:ids=z['flight']
    meta=pd.read_parquet(ROOT/'outputs/history_slow_fast_20261119_cache/validation_windows.parquet',columns=['flight_id','sampling_group']).drop_duplicates().sort_values('flight_id')
    check=set(meta.loc[meta.groupby('sampling_group').cumcount()%3==0,'flight_id'])
    assert len(check)==13 and check<=set(ids)
    comparisons={}
    for ref,mod in [('matched','coupled'),('original','coupled'),('held','coupled'),('original','matched')]:
        k=mod+'_vs_'+ref
        comparisons[k]={m:bootstrap(runs[ref][m],runs[mod][m],m.endswith('_align')) for m in runs[ref]}
        comparisons[k]['confirmation13']={m:bootstrap(runs[ref][m],runs[mod][m],False,check) for m in ('w05','w1','w2','v05','v1','v2')}
    (HERE/'paired.json').write_text(json.dumps(comparisons,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
