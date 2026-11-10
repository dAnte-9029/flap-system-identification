"""Nested train-date validation and fixed-budget dynamic model comparisons."""
import json,pickle,time,os
from pathlib import Path
import numpy as np
import pandas as pd
from px4_clid_models import *
OUT=Path('experiments/px4_closed_loop_dynamic_id_v2')
# Small predeclared paired candidates; no exhaustive order-delay product.
CANDIDATES=[(3,5,0,1e-4),(3,5,1,1e-4),(5,10,1,1e-4),(5,10,2,1e-4),(3,5,1,1e-2),(5,10,1,1e-2)]

def score(model,seq):
    rows=[]
    for s in seq:
        origins,pred=rollout(model,s)
        if len(origins)==0:continue
        for h in [1,2,5,10]:
            err=pred[:,h-1]-s['y'][origins+h]
            for axis in range(3):rows.append(dict(horizon_ms=h*20,axis='pqr'[axis],n=len(origins),sse=float((err[:,axis]**2).sum()),error_sum=float(err[:,axis].sum())))
    return rows

def score_value(rows):
    f=pd.DataFrame(rows);f=f[f.horizon_ms==20];return float(f.sse.sum()/f.n.sum())

def save_model(m):return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in m.items()}

def main():
    seq=pickle.load((OUT/'local/sequences.pkl').open('rb'));dates=sorted({s['date'] for s in seq});metrics=[];responses=[];selection=[];models={}
    for fold,date in enumerate(dates):
        shard=os.environ.get('CLID_SHARD')
        if shard is not None and fold%3!=int(shard):continue
        path=OUT/'local'/f'fold_{date}.pkl'
        if path.exists():
            result=pickle.load(path.open('rb'));print('reuse',date,flush=True)
        else:
            train=[s for s in seq if s['date']!=date];test=[s for s in seq if s['date']==date];td=[d for d in dates if d!=date]
            inner=[set(td[i::3]) for i in range(3)]
            # Date sufficient statistics avoid repeatedly traversing all ARX rows.
            cache={}
            for cfg in CANDIDATES:
                per={dd:stats(blocks_for([s for s in train if s['date']==dd],cfg),[1.]) for dd in td};scores=[]
                for vi,vdates in enumerate(inner):
                    st=tuple(sum(per[dd][i] for dd in td if dd not in vdates) for i in range(4));theta,_=solve_stats(st,cfg[3]);rho=radius(theta,cfg[0])
                    mm=dict(theta=theta,c=np.ones(1),cfg=cfg,rho=rho)
                    sc=score_value(score(mm,[s for s in train if s['date'] in vdates])) if rho<.9999 else float('inf')
                    scores.append(sc)
                cache[cfg]=np.mean(scores)
            cfg=min(cache,key=cache.get);assert np.isfinite(cache[cfg])
            print('fold',date,'selected',cfg,flush=True)
            blocks=blocks_for(train,cfg);m0=fit(blocks,cfg);base=fit(blocks_for(train,cfg,True),cfg)
            # Same deterministic structure; finite innovation orders 1 and 2.
            ncs={}
            for nc in (1,2):
                vals=[]
                for vdates in inner:
                    mm=fit(blocks_for([s for s in train if s['date'] not in vdates],cfg),cfg,nc)
                    vals.append(score_value(score(mm,[s for s in train if s['date'] in vdates])))
                ncs[nc]=float(np.mean(vals));print('noise',date,nc,ncs[nc],flush=True)
            chosen=min(ncs,key=ncs.get)
            fitted={'AR':base,'M0':m0,'M1_nc1':fit(blocks,cfg,1),'M1_nc2':fit(blocks,cfg,2)}
            fitted['M1']=fitted[f'M1_nc{chosen}']
            for name in ['M0','M1']:
                for sign in [-1,1]:fitted[name+('_negative' if sign<0 else '_positive')]=sign_refit(fitted[name],blocks,sign)
            rr=[];pp=[]
            for name,mm in fitted.items():
                rr.extend([dict(date=date,model=name,**r) for r in score(mm,test)])
                curve=pulse(mm)
                pp.extend([dict(date=date,model=name,horizon_ms=(k+1)*20,gain_p=float(v[0]),positive_p=float(.036*v[0]),negative_p=float(-.036*v[0]),difference_p=float(.072*v[0])) for k,v in enumerate(curve)])
            result=dict(models=fitted,metrics=rr,responses=pp,selection=dict(date=date,cfg=list(cfg),arx_candidates={str(k):float(v) for k,v in cache.items()},innovation_scores=ncs,chosen_nc=chosen))
            with path.open('wb') as f:pickle.dump(result,f)
        if 'M0_positive100' not in result['models']:
            train=[s for s in seq if s['date']!=date];test=[s for s in seq if s['date']==date]
            blocks=blocks_for(train,result['models']['M0']['cfg'])
            for name in ['M0','M1']:
                for sign in [-1,1]:
                    key=name+('_negative100' if sign<0 else '_positive100')
                    mm=sign_refit(result['models'][name],blocks,sign,horizon_index=4);result['models'][key]=mm
                    result['metrics'].extend([dict(date=date,model=key,**rr) for rr in score(mm,test)])
                    result['responses'].extend([dict(date=date,model=key,horizon_ms=(k+1)*20,gain_p=float(v[0]),positive_p=float(.036*v[0]),negative_p=float(-.036*v[0]),difference_p=float(.072*v[0])) for k,v in enumerate(pulse(mm))])
            with path.open('wb') as fh:pickle.dump(result,fh)
        if shard is not None:continue
        metrics+=result['metrics'];responses+=result['responses'];selection.append(result['selection']);models[date]={k:save_model(v) for k,v in result['models'].items()}
        f=pd.DataFrame(metrics).groupby(['date','model','horizon_ms','axis'],as_index=False)[['n','sse','error_sum']].sum();f['rmse']=np.sqrt(f.sse/f.n);f['bias']=f.error_sum/f.n;f.to_csv(OUT/'date_metrics.csv',index=False)
        pd.DataFrame(responses).to_csv(OUT/'responses.csv',index=False)
        (OUT/'selection.json').write_text(json.dumps(selection,indent=2));(OUT/'models.json').write_text(json.dumps(models,indent=2));print('completed',date,flush=True)
if __name__=='__main__':main()
