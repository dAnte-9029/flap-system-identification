"""Fit an origin-relative clock with only retrospective targets and origin/command-prefix regressors."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DATA=ROOT/'dataset/velocity_rate_multistep_v1'
MEM=ROOT/'outputs/history_slow_fast_20261119_cache'
OLD=ROOT/'outputs/autopilot_flight_dynamics_20261124'


def features(hist,pooled,u0,time):
    t=time.astype('float64')
    h=hist[:,:180].reshape(-1,6,30)
    f0=h[:,-1,15,None,None]
    valid=h[...,22]
    fav=(h[...,15]*valid).sum(1)/np.maximum(valid.sum(1),1)
    now=u0[:,0,None,None]
    mavg=pooled[...,8]/(-np.expm1(-t))
    ones=np.ones_like(t)
    ramp=t+.2*np.expm1(-t/.2)
    x=np.stack([ones,ones*(f0-3.8),ones*(now-.5),ones*(f0-fav[:,None,None]),mavg-now],-1)
    return ramp[...,None]*x, t*f0


def rows(part,meta):
    encoder=pd.read_parquet(DATA/f'native_encoder_count_{part}.parquet',
        columns=['flight_id','timestamp_us','total_count','in_reviewed_flight']).groupby('flight_id',sort=False)
    windows=pd.read_parquet(MEM/f'{part}_windows.parquet').set_index('window_id')
    with np.load(OLD/f'{part}.npz') as z:
        ids=z['flight'];w=windows.loc[z['window']]; assert np.array_equal(w.flight_id,ids)
        raw={k:z[k] for k in ('hist','u0','pooled','time')}
    origin=w.decision_us.to_numpy()
    for fid,inds in pd.Series(np.arange(len(ids))).groupby(ids,sort=False):
        idx=inds.to_numpy()
        if part=='train': idx=idx[np.argsort(origin[idx])][::4]
        e=encoder.get_group(fid)
        tm=e.timestamp_us.to_numpy(); co=e.total_count.to_numpy(dtype='float64'); ok=e.in_reviewed_flight.to_numpy(dtype=bool)
        assert np.all(np.diff(tm)>0)
        ratio=meta[fid]['transmission_ratio']
        cycles=co/(4096*ratio)
        bad=np.r_[False,(np.diff(tm)>40000)|(~ok[:-1])|(~ok[1:])|(np.abs(np.diff(cycles))>.5)]
        bad=bad.cumsum()
        orig=origin[idx]; t=raw['time'][idx]
        end=orig[:,None,None]+t.astype('float64')*1e6
        left=tm.searchsorted(orig,side='right')-1; right=tm.searchsorted(end)
        mask=(left[:,None,None]>=0)&(right<len(tm))&(orig[:,None,None]>=tm[0])
        mask &= bad[right.clip(max=len(tm)-1)]==bad[left.clip(0)][:,None,None]
        mask &= ((raw['hist'][idx,171]>0)&(raw['hist'][idx,172]>0))[:,None,None]
        for reset in meta[fid]['reset_times_us']:
            mask &= ~((orig[:,None,None]<=reset)&(end>=reset))
        target=np.interp(end,tm,cycles)-np.interp(orig,tm,cycles)[:,None,None]
        x,base=features(raw['hist'][idx],raw['pooled'][idx],raw['u0'][idx],t)
        if np.any(mask):
            yield fid,x,base,target,mask,t


def main():
    meta={r['flight_id']:r for r in json.loads((DATA/'flight_provenance.json').read_text()) if r['split'] in ('train','validation')}
    normal=np.zeros((5,5)); rhs=np.zeros(5); n=0; nf=0
    for fid,x,base,target,mask,t in rows('train',meta):
        a=x[mask]; y=(target-base)[mask]
        normal+=a.T@a/len(y);rhs+=a.T@y/len(y)
        n+=len(y);nf+=1
    assert np.linalg.matrix_rank(normal)==5
    b=np.linalg.solve(normal,rhs)
    result=dict(train_flights=nf,train_clock_pairs=n,normal_condition=np.linalg.cond(normal),coefficients=b.tolist(),scores={})
    cases={}
    for fid,x,base,target,mask,t in rows('validation',meta):
        cases[fid]={}
        for limit,(lo,hi) in {'early':(0,.5),'middle':(.5,1),'late':(1,2.1),'all':(0,2.1)}.items():
            use=mask&(t>lo)&(t<=hi)
            if not use.any():continue
            cases[fid][limit]=[np.cos(2*np.pi*(y-target)[use]).mean() for y in (base,base+x@b)]
    rng=np.random.default_rng(17)
    for label in ('early','middle','late','all'):
        a=np.array([c[label] for c in cases.values() if label in c])
        d=a[:,1]-a[:,0]; draws=rng.integers(0,len(d),(10000,len(d)))
        result['scores'][label]=dict(flights=len(a),held=float(a[:,0].mean()),corrected=float(a[:,1].mean()),
                        gain=float(d.mean()),interval95=np.quantile(d[draws].mean(1),[.025,.975]).tolist())
    (HERE/'clock_fit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
