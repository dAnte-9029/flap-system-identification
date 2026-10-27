"""Independent quadrature, full-array identities and split/ESS audit. CPU only."""
from pathlib import Path
import json,sys,time
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def main():
 cfg=json.loads((HERE/'config.json').read_text());data=Path(cfg['data_root']);a=dict(np.load(OUT/'full_train/weights.npz'));m=pd.read_csv(OUT/'full_train/window_audit.csv.gz');n=len(m)
 # Independent raw-channel lookup, not audit.tape/canonical/integral/compute_weights.
 motors=pd.read_parquet(data/'native_actuator_motors_train.parquet');servos=pd.read_parquet(data/'native_actuator_servos_train.parquet')
 clockcols=['attitude_publication_us','rate_publication_us','motor_publication_us','servo_publication_us','frequency_publication_us','encoder_publication_us']
 clock=pd.read_parquet(data/'samples_train.parquet',columns=['flight_id','segment_id','state_publication_us','state_measurement_us']+clockcols)
 segends=clock.groupby(['flight_id','segment_id']).state_publication_us.max().to_dict()
 clocks={f:g.reset_index(drop=True) for f,g in clock.groupby('flight_id')}
 for fid,rows in m.groupby('flight_id'):
  ss=clocks[fid];o=rows.origin_sample.to_numpy();t=ss.state_measurement_us.to_numpy();ix=np.searchsorted(t,t[o,None]+np.arange(-500000,1,20000),side='right')-1
  assert (ss[clockcols].to_numpy()[ix]<=rows.decision_us.to_numpy()[:,None,None]).all()
  assert (ss.state_publication_us.to_numpy()[ix]<=rows.decision_us.to_numpy()[:,None]).all()
 raw={}
 for frame,cols,channel in [(motors,['control[0]'],'motor'),(servos,['control[0]','control[1]','control[2]'],'servos')]:
  assert frame.split.eq('train').all() if 'split' in frame else True
  for fid,g in frame.groupby('flight_id'):raw.setdefault(fid,{})[channel]=(g.timestamp_us.to_numpy(),g[cols].to_numpy())
 rng=np.random.default_rng(1761);chosen=set(rng.choice(n,4096,replace=False).tolist())
 for fid,g in m.groupby('flight_id'):chosen.update([g.index[0],g.index[-1],g.w1.idxmax(),g.w2_time_ess_ratio.idxmin()])
 chosen.update(np.flatnonzero(~a['w1_supported']).tolist());chosen.update(np.flatnonzero((a['future_mask']&~a['w2_supported']).any(1)).tolist())
 maxw1=maxw2=maxint=0.
 for idx in sorted(chosen):
  row=m.iloc[idx];decision=int(row.decision_us);outputs=a['output_us'][idx];end=max(int(outputs[a['future_mask'][idx]][-1]),300000);r=raw[row.flight_id]
  times=[np.array([decision,decision+end])]
  for t,u in r.values():times.append(t[(t>decision)&(t<decision+end)])
  boundaries=np.unique(np.concatenate(times));lo=boundaries[:-1]-decision;hi=boundaries[1:]-decision;values=[];ages=[]
  for t,u in r.values():
   ii=np.searchsorted(t,boundaries[:-1],side='right')-1;values.append(u[ii]);ages.append(boundaries[:-1]-t[ii])
  u=np.concatenate(values,axis=1).astype(np.float64);age=np.max(ages,axis=0);valid=np.isfinite(u).all(1)&(u[:,0]>=0)&(u[:,0]<=1)&(abs(u[:,1:])<=1).all(1)
  cutoff=min(end,int(segends[(row.flight_id,row.segment_id)])-decision)
  for left,right,ag in zip(lo,hi,age):
   expiry=left+160000-ag
   if expiry<right:
    cutoff=min(cutoff,int(max(left,expiry)));break
  cutoff=max(0,cutoff);assert cutoff==int(a['support_until_us'][idx])
  invalid=np.flatnonzero(~valid)
  if len(invalid):cutoff=min(cutoff,int(lo[invalid[0]]))
  d=(u[:,2]-u[:,1])/2;difference=d-d[0]
  if cutoff>=300000:
   durations=np.maximum(0,np.minimum(hi,300000)-lo);mean=float(np.dot(durations,difference)/300000)
   maxint=max(maxint,abs(mean-a['delta_d'][idx]));assert abs(mean-a['delta_d'][idx])<1e-12,(idx,mean,a['delta_d'][idx],cutoff)
   w=1+max(0,min(1,(abs(mean)-.02)/.08)) if row.auto_stable else 1
  else:w=1
  maxw1=max(maxw1,abs(w-a['w1'][idx]));assert abs(w-a['w1'][idx])<1e-12
  for k,t in enumerate(outputs):
   w=1.
   if row.auto_stable and a['future_mask'][idx,k] and 0<=t<=cutoff:
    duration=np.maximum(0,np.minimum(hi,t)-np.maximum(lo,max(0,t-300000)))
    exposure=float(duration[abs(difference)>=.05].sum());w=1+min(1,exposure/100000)
   maxw2=max(maxw2,abs(w-a['w2'][idx,k]));assert abs(w-a['w2'][idx,k])<1e-12
 # Independently recompute all weighted shares and ESS from serialized arrays.
 checked={}
 for name in ['full_train','fold0','fold1','fold2']:
  z=dict(np.load(OUT/name/'weights.npz'));s=json.loads((OUT/name/'summary.json').read_text());mask=z['future_mask'];w=z['w1'];ww=z['w2'];N=len(w)
  assert np.isfinite(ww).all() and np.isfinite(w).all() and ((w>=1)&(w<=2)).all() and ((ww>=1)&(ww<=2)).all()
  assert (ww[~mask]==1).all();assert np.array_equal(mask.sum(1),m.future_count.to_numpy()[z['registered_index']])
  for times,valid in zip(z['output_us'],mask):assert (np.diff(times[valid])>0).all()
  direct_ess=1/np.sum((w/w.sum())**2);normalized=np.where(mask,ww,0);normalized/=normalized.sum(1,keepdims=True);ratios=1/(normalized**2).sum(1)/mask.sum(1)
  assert abs(direct_ess-s['w1_ess'])<1e-8 and abs(ratios.min()-s['w2_time_ess_ratio_min'])<1e-12
  flights=z['flight_id'];maxratio=max(w[flights==f].sum()/w.sum()/(np.sum(flights==f)/N) for f in set(flights));assert abs(maxratio-s['max_flight_share_ratio_W1'])<1e-12
  for key in ['w0','w1','w2','future_mask','output_us','w1_supported','w2_supported','window_id','flight_id']:
   assert np.array_equal(z[key],a[key][z['registered_index']]),key
  if name!='full_train':
   iso=json.loads((OUT/name/'isolation.json').read_text());assert set(flights)==set(iso['train_flights']);assert not set(flights)&set(iso['held_flights'])
   assert not {f.split(':')[0] for f in flights}&{f.split(':')[0] for f in iso['held_flights']+iso['validation_flights']}
  checked[name]=dict(windows=N,ess_absolute_difference=abs(direct_ess-s['w1_ess']),time_ess_min=float(ratios.min()),flight_share_ratio=float(maxratio),identities_and_isolation_pass=True)
 # Future labels are not a parameter to either weight generator. No model/optimizer constructed.
 dump(OUT/'independent_verification.json',dict(native_integral_windows=len(chosen),native_integral_flights=int(m.iloc[list(chosen)].flight_id.nunique()),max_mean_integral_difference=maxint,max_W1_difference=maxw1,max_W2_difference=maxw2,full_array_checks=checked,training_runs=0,test_opened=False,method='independent raw-channel searchsorted and clipped interval overlap; independent normalized-weight ESS',all_window_history_source_publications_available=True,status='passed'))
 print(json.dumps(checked,indent=2));print('independent quadrature windows',len(chosen),'passed')
if __name__=='__main__':main()
