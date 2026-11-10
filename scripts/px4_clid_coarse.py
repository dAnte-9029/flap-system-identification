"""Necessary 100ms fallback for two dates with 10Hz command logging.
Fixed structure, no order search. Held date excluded from all estimation.
"""
import pickle,json
from pathlib import Path
import pandas as pd
import numpy as np
from px4_clid_models import *
from px4_clid_run import save_model
OUT=Path('experiments/px4_closed_loop_dynamic_id_v2/coarse100')
def main():
 seq=pickle.load((OUT/'local/sequences.pkl').open('rb'));dates=sorted({s['date'] for s in seq});assert len(dates)==17
 cfg=(3,5,1,1e-4);metrics=[];responses=[];models={}
 for date in ['2026-10-10','2026-10-12']:
  train=[s for s in seq if s['date']!=date];test=[s for s in seq if s['date']==date];blocks=blocks_for(train,cfg)
  fitted={'AR':fit(blocks_for(train,cfg,True),cfg),'M0':fit(blocks,cfg),'M1_nc1':fit(blocks,cfg,1),'M1_nc2':fit(blocks,cfg,2)}
  for name,m in fitted.items():
   m['pulse_steps']=1
   for s in test:
    origins,pred=rollout(m,s,stride=2)
    for h in [1,2]:
     err=pred[:,h-1]-s['y'][origins+h]
     for a,ch in enumerate('pqr'):metrics.append(dict(date=date,model=name,horizon_ms=h*100,axis=ch,n=len(origins),sse=float((err[:,a]**2).sum()),error_sum=float(err[:,a].sum())))
   for k,v in enumerate(pulse(m)[:2]):responses.append(dict(date=date,model=name,horizon_ms=(k+1)*100,gain_p=float(v[0]),positive_p=.036*float(v[0]),negative_p=-.036*float(v[0])))
  models[date]={name:save_model(m) for name,m in fitted.items()};print('coarse complete',date,flush=True)
 f=pd.DataFrame(metrics).groupby(['date','model','horizon_ms','axis'],as_index=False)[['n','sse','error_sum']].sum();f['rmse']=np.sqrt(f.sse/f.n);f['bias']=f.error_sum/f.n;f.to_csv(OUT/'date_metrics.csv',index=False)
 pd.DataFrame(responses).to_csv(OUT/'responses.csv',index=False);(OUT/'models.json').write_text(json.dumps(models,indent=2))
if __name__=='__main__':main()
