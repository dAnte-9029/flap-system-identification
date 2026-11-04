"""Two historical angular-rate direction definitions on fixed PA-HBD validation predictions."""
import json,math
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).parent;repo=Path('/home/zn/flap-system-identification/.worktrees/pahbd-compile-v4')
old=json.loads((repo/'docs/analysis/results/model_flight_response_diagnostic_v1/protocol.json').read_text());floors={a:old['direction_floor'][a] for a in 'pqr'}
endpoints=[.02,.2,.5,1.,2.];intervals=[('early',0.,.1),('middle',.1,.3),('late',.3,.5),('extended',.5,1.),('long',1.,2.)]
protocol=dict(endpoint='sign(delta_pred)==sign(delta_true), abs(delta_true)>0.01 rad/s; abs(delta_pred)<=0.01 counts incorrect',interval='sign(dt-weighted mean delta_pred)==sign(dt-weighted mean delta_true); abs(mean delta_true)>historical per-axis frozen floor; no prediction amplitude gate (original definition)',origin='original GRU causal measured gyro at decision, verified equal to PAHBD original anchor; no future correction',endpoint_horizons_s=endpoints,endpoint_tolerance_s=.03,intervals=intervals,interval_floors_rad_s=floors,floor_source=str(repo/'docs/analysis/results/model_flight_response_diagnostic_v1/protocol.json'),floor_scope='historical TRAIN q25 thresholds reused numerically; not recalibrated on current train or validation, not a noise estimate',aggregation='eligible windows within flight then equal eligible flights, then equal seeds; amplitude ratio = mean flight RMS predicted increment / mean flight RMS true increment',selection='108 main_50hz windows of the original fixed validation cohort, 27 common flights; frozen H1000-GRU-Control and independently selected A/B best checkpoints; same paired sensor labels and native state rows',case='seed17, nominal2s, middle member after sorting flight_id; selected by identity not error')
(out/'direction_protocol.json').write_text(json.dumps(protocol,indent=2))
def endpoint_sign(t,p):return abs(t)>.01, bool(np.sign(t)==np.sign(p) and abs(p)>.01)
assert endpoint_sign(.02,0)==(True,False) and endpoint_sign(.02,.005)==(True,False) and endpoint_sign(.02,-.02)==(True,False) and endpoint_sign(.02,.03)==(True,True) and endpoint_sign(.005,.03)[0]==False
rows=[];indices={};arrays={};canonical=None
for seed in [17,29]:
 for model in ['A','B','C']:
  name=f'{model}_seed{seed}';dest=out/f'direction-{name}';idx=json.loads((dest/'index.json').read_text());indices[name]=idx
  assert (dest/'verification.json').exists()
  if canonical is None:canonical=idx
  assert idx==canonical
  for item in idx:
   z=np.load(dest/f"window_{item['i']:03d}.npz");t=z['times'];truth=z['truth'];pred=z['pred'];origin=z['origin'];true=truth-origin;estimate=pred-origin
   if model=='A':arrays[seed,item['i']]=(t.copy(),truth.copy(),origin.copy())
   else:
    for x,y in zip((t,truth,origin),arrays[seed,item['i']]):np.testing.assert_array_equal(x,y)
   common=dict(seed=seed,model=model,window_id=item['window_id'],flight_id=item['flight_id'],anchor_age_s=-item['anchor_measurement'],nominal_horizon_us=item['horizon_us'])
   for h in endpoints:
    j=int(np.argmin(abs(t-h)))
    if abs(t[j]-h)>.03:continue
    for k,axis in enumerate('pqr'):
     eligible,correct=endpoint_sign(true[j,k],estimate[j,k])
     rows.append(dict(common,kind='endpoint',period=f'{h:g}s',end_s=h,axis=axis,eligible=eligible,correct=correct,true_change=true[j,k],pred_change=estimate[j,k],true_energy=true[j,k]**2,pred_energy=estimate[j,k]**2,error_energy=(pred[j,k]-truth[j,k])**2,correlation=np.nan,native_points=1,covered_seconds=t[j],actual_time=t[j]))
   prev=np.r_[0.,t[:-1]]
   for label,start,end in intervals:
    if t[-1]<end-.03:continue
    mask=(t>start)&(t<=end);wt=np.maximum(0,np.minimum(t,end)-np.maximum(prev,start))[mask]
    if not len(wt) or wt.sum()<=0:continue
    x=true[mask];y=estimate[mask];wt=wt/wt.sum();mt=(wt[:,None]*x).sum(0);mp=(wt[:,None]*y).sum(0)
    for k,axis in enumerate('pqr'):
     dx=x[:,k]-mt[k];dy=y[:,k]-mp[k];den=math.sqrt(float((wt*dx**2).sum()*(wt*dy**2).sum()));corr=float((wt*dx*dy).sum()/den) if den>1e-12 else np.nan
     rows.append(dict(common,kind='interval',period=label,end_s=end,axis=axis,eligible=abs(mt[k])>floors[axis],correct=bool(mt[k]*mp[k]>0),true_change=mt[k],pred_change=mp[k],true_energy=float((wt*x[:,k]**2).sum()),pred_energy=float((wt*y[:,k]**2).sum()),error_energy=float((wt*(y[:,k]-x[:,k])**2).sum()),correlation=corr,native_points=int(mask.sum()),covered_seconds=float(np.maximum(0,np.minimum(t,end)-np.maximum(prev,start))[mask].sum()),actual_time=float(t[mask][-1])))
df=pd.DataFrame(rows);df.to_csv(out/'direction_windows.csv',index=False)
per=[]
for key,g in df.groupby(['seed','model','kind','period','end_s','axis','flight_id'],sort=False):
 d=dict(zip(['seed','model','kind','period','end_s','axis','flight_id'],key));ok=g[g.eligible]
 d.update(windows=len(g),eligible_windows=len(ok),correct_windows=int(ok.correct.sum()),direction=float(ok.correct.mean()) if len(ok) else np.nan,truth_rms=math.sqrt(g.true_energy.mean()),pred_rms=math.sqrt(g.pred_energy.mean()),rmse=math.sqrt(g.error_energy.mean()),correlation=float(g.correlation.mean()) if g.correlation.notna().any() else np.nan)
 per.append(d)
pf=pd.DataFrame(per);pf.to_csv(out/'direction_per_flight.csv',index=False);agg=[]
for key,g in pf.groupby(['seed','model','kind','period','end_s','axis'],sort=False):
 d=dict(zip(['seed','model','kind','period','end_s','axis'],key));d.update(direction_pct=100*g.direction.mean(),flights=len(g),eligible_flights=int(g.direction.notna().sum()),windows=int(g.windows.sum()),eligible_windows=int(g.eligible_windows.sum()),amplitude_ratio=g.pred_rms.mean()/g.truth_rms.mean(),rmse=g.rmse.mean(),correlation=g.correlation.mean(),correlation_flights=int(g.correlation.notna().sum()))
 # Endpoint correlation is across origins, flight equal, rather than waveform correlation.
 if d['kind']=='endpoint':
  z=df[(df.seed==d['seed'])&(df.model==d['model'])&(df.kind=='endpoint')&(df.period==d['period'])&(df.axis==d['axis'])];w=1/z.groupby('flight_id').flight_id.transform('size').to_numpy();w=w/w.sum();x=z.true_change.to_numpy();y=z.pred_change.to_numpy();dx=x-np.sum(w*x);dy=y-np.sum(w*y);den=math.sqrt(np.sum(w*dx**2)*np.sum(w*dy**2));d['correlation']=np.sum(w*dx*dy)/den if den>1e-12 else np.nan;d['correlation_flights']=z.flight_id.nunique()
 agg.append(d)
ps=pd.DataFrame(agg);ps.to_csv(out/'direction_per_seed.csv',index=False)
# Coverage is model-independent; eligibility is decided solely from observations.
keys=['seed','kind','period','end_s','axis','flight_id'];ac=pf[pf.model=='A'].set_index(keys);bc=pf[pf.model=='B'].set_index(keys);pd.testing.assert_frame_equal(ac[['windows','eligible_windows']],bc[['windows','eligible_windows']])
mean=ps.groupby(['model','kind','period','end_s','axis'],sort=False)[['direction_pct','amplitude_ratio','rmse','correlation','eligible_flights','eligible_windows','windows']].mean().reset_index();mean.to_csv(out/'direction_summary.csv',index=False)
fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
for row,kind in enumerate(['endpoint','interval']):
 for ax,axis in zip(axes[row],'pqr'):
  for model in ['A','B','C']:
   g=mean[(mean.kind==kind)&(mean.axis==axis)&(mean.model==model)].sort_values('end_s');ax.plot(range(len(g)),g.direction_pct,marker='o',label=model);ax.set_xticks(range(len(g)),g.period)
  ax.set(title=f'{axis}: {kind} change direction',ylabel='Agreement (%)',ylim=(0,100));ax.grid(alpha=.2);ax.legend()
fig.savefig(out/'angular_direction.png',dpi=170);plt.close(fig)
candidates=sorted([r for r in canonical if r['horizon_us']==2000000],key=lambda r:r['flight_id']);chosen=candidates[len(candidates)//2];fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True,constrained_layout=True)
for k,ax in enumerate(axes):
 for model in ['A','B','C']:
  z=np.load(out/f'direction-{model}_seed17'/f"window_{chosen['i']:03d}.npz")
  if model=='A':ax.plot(z['times'],z['truth'][:,k]-z['origin'][k],color='black',label='Observed delta')
  ax.plot(z['times'],z['pred'][:,k]-z['origin'][k],label=f'{model} predicted delta')
 ax.axhline(0,color='gray',linewidth=.5);ax.set_ylabel(f"delta {'pqr'[k]} (rad/s)");ax.legend();ax.grid(alpha=.2)
axes[0].set_title('Fixed identity example: seed 17, middle flight, 2s window');axes[-1].set_xlabel('Native elapsed time (s)');fig.savefig(out/'angular_direction_example.png',dpi=170);plt.close(fig)
(out/'direction_example.json').write_text(json.dumps(chosen,indent=2))
print(mean.to_string(index=False))
