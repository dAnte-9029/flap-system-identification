"""Independent explicit interval-sum curve check and frozen-window overlap audit."""
from pathlib import Path
import json
import argparse
import hashlib
import numpy as np
import pandas as pd
from verify import component_sets
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=HERE/'results');args=ap.parse_args();out=args.results
r=pd.read_csv(out/'response_curves.csv.gz');c=pd.read_csv(out/'curve_metrics.csv');s=pd.read_csv(out/'curve_summary.csv')
p=pd.read_csv(out/'pairs.csv');w=pd.read_csv(out/'windows.csv');m=pd.read_csv(out/'cohorts.csv')
curves=c.set_index(['pair_uid','model','seed']);largest=0.
for key,d in r[(r.variable=='p')&r.supported].groupby(['pair_uid','model','seed']):
    d=d.sort_values('time_ms').drop_duplicates('time_ms');t=d.time_ms.to_numpy()/1000;dur=t[-1];dt=np.diff(t)
    integrate=lambda y:float(sum(float(h)*(float(a)+float(b))/2 for h,a,b in zip(dt,y[:-1],y[1:]))/dur)
    real=d.real.to_numpy();pred=d.pred.to_numpy();e=pred-real
    vals={'curve_rmse':np.sqrt(integrate(e**2)),'curve_bias':integrate(e),'real_rms':np.sqrt(integrate(real**2)),'pred_rms':np.sqrt(integrate(pred**2)),
        'real_peak_ms':t[np.argmax(abs(real))]*1000,'pred_peak_ms':t[np.argmax(abs(pred))]*1000}
    for k,v in vals.items():
        err=abs(curves.loc[key,k]-v);largest=max(largest,err);assert err<1e-10,(key,k,err)
maxagg=0.
for row in s.itertuples():
    ids=m[(m.split==row.split)&(m.cohort==row.cohort)].pair_uid;pp=p[p.pair_uid.isin(ids)]
    d=c[c.pair_uid.isin(ids)]
    d=d[d.model==row.model]
    if row.seed!='pooled':d=d[d.seed==int(row.seed)]
    vals=d.assign(e2=d.curve_rmse**2,real2=d.real_rms**2,pred2=d.pred_rms**2).groupby('pair_uid')[['e2','real2','pred2','curve_bias']].mean()
    if row.weighting=='pair':a=vals.mean()
    elif row.weighting=='flight':a=pd.DataFrame([vals.loc[pp[(pp.flight_a==f)|(pp.flight_b==f)].pair_uid].mean() for f in set(pp.flight_a)|set(pp.flight_b)]).mean()
    else:a=pd.DataFrame([vals.loc[list(g)].mean() for g in component_sets(pp)]).mean()
    for key,value in {'curve_rmse':np.sqrt(a.e2),'real_rms':np.sqrt(a.real2),'pred_rms':np.sqrt(a.pred2),'curve_bias':a.curve_bias}.items():
        err=abs(getattr(row,key)-value);maxagg=max(maxagg,err);assert err<1e-10
# Memberships within the main cohort retain nonoverlapping full support intervals.
for split in ['train','validation']:
    pp=p[p.pair_uid.isin(m[(m.split==split)&(m.cohort=='main')].pair_uid)]
    ww=w[w.window_id.isin(set(pp.window_a)|set(pp.window_b))]
    for fid,g in ww.groupby('flight_id'):
        g=g.sort_values('support_start_us')
        assert np.all(g.support_start_us.to_numpy()[1:]>g.support_end_us.to_numpy()[:-1]),fid
    dis=p[p.pair_uid.isin(m[(m.split==split)&(m.cohort=='flight_disjoint')].pair_uid)]
    assert len(set(dis.flight_a)|set(dis.flight_b))==2*len(dis)
assert not (set(w[w.split=='train'].flight_id)&set(w[w.split=='validation'].flight_id))
result=dict(passed=True,curve_rows=len(c),curve_summary_rows=len(s),curve_max_abs_difference=largest,curve_aggregate_max_abs_difference=maxagg,
    main_support_intervals_nonoverlapping=True,flight_disjoint_flags_verified=True,train_validation_flights_disjoint=True,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'independent_curve_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
