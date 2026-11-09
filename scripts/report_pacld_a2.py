"""Flight-bootstrap summaries and figures for the executed train-only diagnostic."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('experiments/phase_anchored_coupled_latent_v1/control_identifiability');rng=np.random.default_rng(17)
e=pd.read_csv(O/'flight_errors.csv');g=pd.read_csv(O/'local_g.csv');ex=pd.read_csv(O/'excitation.csv');df=pd.read_pickle(O/'features.pkl');rows=[];b=[]
for (h,axis),a in e.groupby(['horizon','axis']):
 ids=rng.integers(0,len(a),(2000,len(a)));v=a.m0.to_numpy();z=a.m1.to_numpy();delta=np.sqrt(v[ids].mean(1))-np.sqrt(z[ids].mean(1));b.append(dict(horizon=h,axis=axis,m0_rmse=np.sqrt(v.mean()),m1_rmse=np.sqrt(z.mean()),gain=np.sqrt(v.mean())-np.sqrt(z.mean()),ci_low=np.quantile(delta,.025),ci_high=np.quantile(delta,.975)))
 for c in ['motor','differential','symmetric','rudder']:
  removed=a[c].to_numpy();d=np.sqrt(removed[ids].mean(1))-np.sqrt(z[ids].mean(1));gg=g[(g.horizon==h)&(g.axis==axis)&(g.channel==c)];gg=gg.groupby(['flight','width']).G.mean().unstack();med=gg.median();rows.append(dict(channel=c,horizon=h,axis=axis,flights=len(a),conditional_gain=np.sqrt(removed.mean())-np.sqrt(z.mean()),ci_low=np.quantile(d,.025),ci_high=np.quantile(d,.975),G_width_half=med.get(.5),G_width_one=med.get(1.),G_width_two=med.get(2.),G_q025=gg[1.].quantile(.025),G_q975=gg[1.].quantile(.975),positive_flight_fraction=(gg[1.]>0).mean()))
pd.DataFrame(b).to_csv(O/'prediction_summary.csv',index=False);pd.DataFrame(rows).to_csv(O/'channel_summary.csv',index=False)
fig,ax=plt.subplots(1,2,figsize=(10,4))
for k,h in enumerate([.2,.5]):
 a=pd.DataFrame(b).query('horizon==@h');x=np.arange(3);ax[k].bar(x-.18,a.m0_rmse,.36,label='M0');ax[k].bar(x+.18,a.m1_rmse,.36,label='M1');ax[k].set_xticks(x,a.axis);ax[k].set_title(f'{h}s flight-balanced RMSE');ax[k].set_ylabel('rad/s');ax[k].legend()
fig.tight_layout();fig.savefig(O/'prediction.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(2,4,figsize=(14,6))
for i,h in enumerate([.2,.5]):
 for j,c in enumerate(['motor','differential','symmetric','rudder']):
  for axis in 'pqr':
   a=g[(g.horizon==h)&(g.channel==c)&(g.axis==axis)].groupby('width').G.median();ax[i,j].plot(a.index,a.values,'o-',label=axis)
  ax[i,j].axhline(0,color='gray');ax[i,j].set_title(f'{c}, {h}s');ax[i,j].set_xlabel('soft width multiplier');ax[i,j].legend()
fig.tight_layout();fig.savefig(O/'local_g.png',dpi=160);plt.close(fig)
# Coverage/excitation by flight, date and mode; descriptive, not independent windows.
r=np.load(O/'predictions.npz');records=[]
for key in ['flight','date','mode']:
 for val,ix in df.groupby(key).groups.items():
  for c,name in enumerate(['motor','differential','symmetric','rudder']):records.append(dict(group=key,value=str(val),channel=name,n=len(ix),flights=df.loc[ix].flight.nunique(),residual_rms=np.sqrt(np.mean(r['residual'][ix,4+c]**2))))
pd.DataFrame(records).to_csv(O/'excitation_coverage.csv',index=False)
summary={'prediction':b,'channels':rows,'saturation_raw_motor_left_right_rudder':np.stack(df.saturation).mean(0).tolist(),'target_offset_us_quantiles':np.quantile(np.stack(df.offset),[0,.5,1],axis=0).tolist(),'spaced_decisions':len(df),'flights':df.flight.nunique(),'dates':df.date.nunique()};(O/'summary.json').write_text(json.dumps(summary,indent=2));print(pd.DataFrame(b).to_string(index=False));print(pd.DataFrame(rows).to_string(index=False))
err=pd.read_csv(O/'flight_errors.csv');err.groupby(['date','horizon','axis'])[['m0','m1','plain0','plain1']].mean().pow(.5).to_csv(O/'date_errors.csv')
fig,ax=plt.subplots(1,2,figsize=(11,4));a=pd.read_csv(O/'excitation.csv').query('window_ms==100');x=np.arange(4);ax[0].bar(x-.18,a.raw_rms,.36,label='raw');ax[0].bar(x+.18,a.residual_rms,.36,label='crossfit residual');ax[0].set_xticks(x,a.channel,rotation=15);ax[0].set_ylabel('100ms command RMS');ax[0].legend();cov=pd.read_csv(O/'coverage.csv');cov.groupby('date').spaced.sum().plot.bar(ax=ax[1]);ax[1].set_ylabel('1s-spaced decisions');fig.tight_layout();fig.savefig(O/'excitation_coverage.png',dpi=160);plt.close(fig)
for width in range(2):np.savetxt(O/f'raw_control_corr_{width}.csv',np.corrcoef(r['U'][:,width*4:width*4+4].T),delimiter=',')
