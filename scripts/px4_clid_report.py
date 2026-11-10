"""Render aggregate results; never opens raw, validation or test data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('experiments/px4_closed_loop_dynamic_id_v2')
def main():
 f=pd.read_csv(OUT/'date_metrics.csv');r=pd.read_csv(OUT/'responses.csv');c=pd.read_csv(OUT/'innovation_correlations.csv');u=json.loads((OUT/'response_uncertainty.json').read_text())
 agg=f.groupby(['model','horizon_ms','axis'],as_index=False)[['n','sse','error_sum']].sum();agg['rmse']=np.sqrt(agg.sse/agg.n);agg['bias']=agg.error_sum/agg.n;agg.to_csv(OUT/'aggregate_metrics.csv',index=False)
 fig,axes=plt.subplots(1,3,figsize=(12,3.4))
 for ax,ch in zip(axes,'pqr'):
  for name in ['AR','M0','M1_nc1','M1_nc2']:
   g=agg[(agg.model==name)&(agg.axis==ch)];ax.plot(g.horizon_ms,g.rmse,'o-',label=name)
  ax.set(xlabel='Horizon (ms)',ylabel=f'{ch} RMSE (rad/s)');ax.grid(alpha=.2)
 axes[0].legend();fig.tight_layout();fig.savefig(OUT/'prediction_errors.png',dpi=170);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,3.6))
 for name in ['M0','M1']:
  g=c[(c.model==name)&(c.kind=='innovation_acf')&(c.channel=='p')].groupby('lag_ms').correlation.agg(['mean','min','max']);axes[0].plot(g.index,g['mean'],'o-',label=name);axes[0].fill_between(g.index,g['min'],g['max'],alpha=.15)
  g=c[(c.model==name)&(c.kind=='partial_diff_given_previous_state_other_control')].groupby('lag_ms').correlation.agg(['mean','min','max']);axes[1].plot(g.index,g['mean'],'o-',label=name);axes[1].fill_between(g.index,g['min'],g['max'],alpha=.15)
 axes[0].set(xlabel='Lag (ms)',ylabel='p innovation ACF');axes[1].set(xlabel='Control lag (ms); negative = future',ylabel='Partial correlation with differential')
 for ax in axes:ax.legend();ax.grid(alpha=.2);ax.axhline(0,color='gray',lw=.7)
 fig.tight_layout();fig.savefig(OUT/'innovation_diagnostics.png',dpi=170);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,3.6))
 for ax,name in zip(axes,['M0','M1_nc1','M1_nc2']):
  for date,g in r[r.model==name].groupby('date'):ax.plot(g.horizon_ms,g.positive_p,lw=.8,alpha=.65)
  ax.set(title=name,xlabel='Horizon (ms)',ylabel='p change for +0.036 pulse (rad/s)');ax.axhline(0,color='black',lw=.7)
 fig.tight_layout();fig.savefig(OUT/'pulse_responses.png',dpi=170);plt.close(fig)
 profile=[];fig,axes=plt.subplots(1,2,figsize=(11,4))
 for ax,name in zip(axes,['M0','M1']):
  for target in [100,200]:
   suffix='100' if target==100 else ''
   for h in [20,target]:
    g=f[(f.axis=='p')&(f.horizon_ms==h)].pivot(index='date',columns='model',values='rmse');delta=100*(g[name+'_positive'+suffix]/g[name+'_negative'+suffix]-1)
    ax.plot(range(len(delta)),delta,'o-',label=f'constraint {target}, eval {h} ms')
    for date,v in delta.items():profile.append(dict(date=date,model=name,target_horizon_ms=target,horizon_ms=h,positive_vs_negative_rmse_pct=float(v)))
  ax.axhline(0,color='black',lw=.7);ax.set(title=name,xlabel='Held date index',ylabel='RMSE positive / negative - 1 (%)');ax.legend()
 fig.tight_layout();fig.savefig(OUT/'sign_profile.png',dpi=170);plt.close(fig);pd.DataFrame(profile).to_csv(OUT/'sign_profile.csv',index=False)
 fig,ax=plt.subplots(figsize=(7,4))
 for name in ['M0','M1']:
  z=u['alltrain_'+name];x=np.arange(1,11)*20;y=np.array(z['gain'])*.036;se=np.array(z['cluster_se'])*.036;ax.plot(x,y,'o-',label=name);ax.fill_between(x,y-1.96*se,y+1.96*se,alpha=.2)
 ax.set(xlabel='Horizon (ms)',ylabel='p change (+0.036); local date-cluster 95% interval');ax.legend();ax.axhline(0,color='black',lw=.7);fig.tight_layout();fig.savefig(OUT/'response_uncertainty.png',dpi=170);plt.close(fig)
 print(agg[agg.axis=='p'].to_string(index=False));print(r[r.horizon_ms.isin([100,200])].groupby(['model','horizon_ms']).gain_p.agg(['mean','min','max']).to_string())
if __name__=='__main__':main()
