import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from evaluate_merge import METRICS

def main():
 d=pd.read_parquet(OUT/'endpoint_errors.parquet');rows=[];flightrows=[]
 for scope,subset in [('all',d),('affected',d[d.affected])]:
  for (h,variant),g in subset.groupby(['horizon_ms','variant']):
   for seed in [17,29,43,'all']:
    x=g if seed=='all' else g[g.seed==seed]
    for metric in METRICS:
     per=x.assign(e2=x[metric]**2).groupby('flight_id').agg(mse=('e2','mean'),mean=(metric,'mean'))
     rows.append(dict(scope=scope,horizon_ms=h,variant=variant,seed=seed,metric=metric,windows=x.window_id.nunique(),flights=x.flight_id.nunique(),
       flight_rmse=np.sqrt(per.mse.mean()),flight_mean_error=per['mean'].mean(),window_rmse=np.sqrt(np.mean(x[metric]**2)),window_mean_error=x[metric].mean()))
     for fid,r in per.iterrows():flightrows.append(dict(scope=scope,horizon_ms=h,variant=variant,seed=seed,metric=metric,flight_id=fid,rmse=np.sqrt(r.mse),mean=r['mean']))
 s=pd.DataFrame(rows);s.to_csv(OUT/'summary.csv',index=False);f=pd.DataFrame(flightrows);f.to_csv(OUT/'per_flight.csv',index=False)
 keys=['scope','horizon_ms','seed','metric'];a=s[s.variant=='native'].set_index(keys);b=s[s.variant=='merged'].set_index(keys)
 p=a[['windows','flights','flight_rmse']].rename(columns={'flight_rmse':'native_rmse'});p['merged_rmse']=b.flight_rmse;p['delta']=p.merged_rmse-p.native_rmse;p['change_pct']=100*p.delta/p.native_rmse;p.reset_index().to_csv(OUT/'paired_comparisons.csv',index=False)
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for ax,h in zip(axs,[500,1000]):
  g=f[(f.scope=='all')&(f.seed=='all')&(f.metric=='velocity_m_s')&(f.horizon_ms==h)].pivot(index='flight_id',columns='variant',values='rmse')
  delta=(g.merged-g.native).sort_values();ax.bar(np.arange(len(delta)),delta,color=np.where(delta>0,'tab:red','tab:blue'));ax.axhline(0,color='k',lw=.5)
  ax.set(title=f'{h} ms velocity RMSE change',xlabel='Flight rank',ylabel='Merged - native (m/s)');ax.grid(axis='y',alpha=.2)
 fig.tight_layout();fig.savefig(OUT/'merge_effect_per_flight.png',dpi=160);plt.close(fig)
 print('Paired full-validation summaries saved')
if __name__=='__main__':main()
