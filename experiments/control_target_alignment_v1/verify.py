"""Independent SciPy interpolation, explicit piecewise integration and group reductions."""
from common import *
from scipy.interpolate import interp1d

def linear(t,y,q,gap=.04):
 t=np.asarray(t);q=np.atleast_1d(q);assert np.all(np.diff(t)>0)
 out=np.full((len(q),y.shape[1]),np.nan);f=interp1d(t,y,axis=0,bounds_error=False,fill_value=np.nan)
 for i,v in enumerate(q):
  if v<t[0]-1e-10 or v>t[-1]+1e-10:continue
  v=np.clip(v,t[0],t[-1]);j=np.searchsorted(t,v)
  if j<len(t) and abs(t[j]-v)<1e-10:out[i]=y[j]
  elif j>0 and j<len(t) and t[j]-t[j-1]<=gap+1e-10:out[i]=f(v)
 return out

def mean(t,y,c,width=.27,gap=.04):
 lo=c-width/2;hi=c+width/2
 if lo<t[0]-1e-10 or hi>t[-1]+1e-10:return np.full(y.shape[1],np.nan)
 knots=np.unique(np.r_[lo,t[(t>lo)&(t<hi)],hi]);vals=linear(t,y,knots,gap)
 if np.any(np.diff(knots)>gap+1e-10) or not np.isfinite(vals).all():return np.full(y.shape[1],np.nan)
 # Sum analytic integral y_left*dt + slope*dt^2/2, independently of main trapz.
 step=np.diff(knots);slope=np.diff(vals,axis=0)/step[:,None]
 return np.sum(vals[:-1]*step[:,None]+slope*step[:,None]**2/2,axis=0)/width

def groupweights(pp,kind):
 ids=pp.pair_uid.tolist()
 if kind=='pair':return {u:1/len(ids) for u in ids}
 if kind=='flight':
  flights=sorted(set(pp.flight_a)|set(pp.flight_b));r=dict.fromkeys(ids,0.)
  for f in flights:
   u=pp[(pp.flight_a==f)|(pp.flight_b==f)].pair_uid.tolist()
   for k in u:r[k]+=1/(len(flights)*len(u))
  return r
 adj={f:set() for f in set(pp.flight_a)|set(pp.flight_b)}
 for a,b in zip(pp.flight_a,pp.flight_b):adj[a].add(b);adj[b].add(a)
 left=set(adj);groups=[]
 while left:
  frontier={min(left)};visited=set()
  while frontier:
   x=frontier.pop();visited.add(x);frontier|=adj[x]-visited
  left-=visited;groups.append(pp[pp.flight_a.isin(visited)].pair_uid.tolist())
 return {u:1/(len(groups)*len(g)) for g in groups for u in g}

def fixed():
 z,n,w,p,coh=read_assets();case_ends=pd.read_csv(SWAP/'results/cases.csv').groupby('pair_uid').support_end_s.min().to_dict();d=pd.read_csv(OUT/'oracle_points.csv.gz');maxerr=0.;count=0;cache={};idx=p.set_index('pair_uid')
 def calc(i,time,method,oracle):
  key=i,time,method,oracle
  if key in cache:return cache[key]
  steps=int(z['future_count'][i]);t=z['native_time_us'][i,:steps+1]*1e-6;ini=z['initial'][i,10:13].astype(float);a=np.vstack([z['truth_initial'][i,10:13],z['truth'][i,:steps,10:13]]).astype(float);nt=n[f'rate_t_{i}'];ny=n[f'rate_y_{i}'];anchor=linear(nt,ny,[0],.02)[0]
  if oracle in ['O0','O1']:
   start=ini if oracle=='O0' else a[0].copy();y=a.copy();y[0]=start;clock=t;gap=.04
  else:start=ini if oracle=='O2' else anchor;y=ny;clock=nt;gap=.02
  out=(mean(clock,y,time,gap=gap) if method=='D' else linear(clock,y,[time],gap)[0])-start
  if method=='D' and (time<.135-1e-10 or time+.135>min(.5,t[-1])+1e-10):out[:]=np.nan
  if time>min(.5,t[-1])+1e-10:out[:]=np.nan
  cache[key]=out;return out
 for r in d.itertuples():
  pp=idx.loc[r.pair_uid];v=calc(int(pp.a),r.time_ms/1000,r.method,r.model)-calc(int(pp.b),r.time_ms/1000,r.method,r.model)
  truth=(calc(int(pp.a),r.time_ms/1000,r.method,'O1')-calc(int(pp.b),r.time_ms/1000,r.method,'O1')) if r.method=='A' else (calc(int(pp.a),r.time_ms/1000,r.method,'O3')-calc(int(pp.b),r.time_ms/1000,r.method,'O3'))
  j=VARS.index(r.variable)
  if r.supported:
   assert np.isfinite(v[j]) and np.isfinite(truth[j]);maxerr=max(maxerr,abs(v[j]-r.pred),abs(truth[j]-r.real));count+=1
  else:
   end=min(.5,case_ends[r.pair_uid]);outside=(r.time_ms/1000>end+1e-10) or (r.method=='D' and (r.time_ms/1000<.135-1e-10 or r.time_ms/1000+.135>end+1e-10))
   assert outside or not np.isfinite(v[j]) or not np.isfinite(truth[j]), (r.pair_uid,r.model,r.method,r.time_label)
 assert maxerr<1e-10,maxerr
 dec=pd.read_csv(OUT/'pair_decomposition.csv.gz');sd=pd.read_csv(OUT/'window_decomposition.csv.gz');de=max(float(abs(frame.increment-frame.future+frame.origin-frame.bridge).max()) if 'increment' in frame else float(abs(frame.E_increment-frame.E_future+frame.E_origin-frame.bridge).max()) for frame in [dec,sd]);assert de<1e-14
 summary=pd.read_csv(OUT/'oracle_summary.csv');me=0.;sn=0
 for row in summary[summary.time_label=='300'].itertuples():
  ids=coh[(coh.split==row.split)&(coh.cohort==row.cohort)].pair_uid;dd=d[(d.time_label=='300')&(d.model==row.model)&(d.method==row.method)&(d.variable==row.variable)&d.pair_uid.isin(ids)&d.supported];pp=p[p.pair_uid.isin(dd.pair_uid)];wm=groupweights(pp,row.weighting);wt=np.array([wm[u] for u in dd.pair_uid]);wt/=wt.sum();r=dd.real.to_numpy();v=dd.pred.to_numpy();active=abs(r)>.1
  rms=lambda x:np.sqrt(np.sum(wt*x*x));metrics=dict(rmse=rms(v-r),scale_ratio=rms(v)/rms(r),signed_gain=np.sum(wt*v*r)/np.sum(wt*r*r))
  if active.any():metrics['direction_accuracy']=np.sum(wt[active]*((v[active]*r[active]>0)&(abs(v[active])>1e-8)))/np.sum(wt[active])
  assert row.active_pairs==active.sum()
  for field,val in metrics.items():me=max(me,abs(val-getattr(row,field)))
  sn+=1
 assert me<1e-11,me
 save(OUT/'independent_fixed_verification.json',dict(passed=True,point_rows=count,point_max_error=maxerr,decomposition_max_error=de,summary_rows=sn,summary_max_error=me,method='SciPy linear interpolation, analytic segment integrals, direct per-flight and BFS-component averaging',test_read=False))
 print('Independent fixed checks passed',maxerr,me,flush=True)

def full():
 audit=pd.read_csv(OUT/'full_window_audit.csv.gz');checks=[];mx=0.
 for split in ['train','validation']:
  meta=pd.read_parquet(TRAIN/f'{split}_windows.parquet');s=pd.read_parquet(DATA/f'samples_{split}.parquet',columns=['flight_id','state_measurement_us']+RATE+TARGET);native=pd.read_parquet(DATA/f'native_vehicle_angular_velocity_{split}.parquet',columns=['flight_id','timestamp_sample_us','xyz[0]','xyz[1]','xyz[2]']);sg=dict(tuple(s.groupby('flight_id')));ng=dict(tuple(native.groupby('flight_id')))
  for fid,mm in meta.groupby('flight_id'):
   ss=sg[fid].reset_index(drop=True);nn=ng[fid]
   # Three deterministic registered identities per flight, spanning the flight.
   for original in mm.index[np.unique(np.linspace(0,len(mm)-1,3,dtype=int))]:
    row=meta.loc[original];saved=audit[(audit.split==split)&(audit.registered_index==original)].iloc[0];o=int(row.origin_sample);k=int(row.count_1000ms);t0=int(ss.state_measurement_us.iloc[o]);nt=(nn.timestamp_sample_us.to_numpy()-t0)*1e-6;ny=nn[['xyz[0]','xyz[1]','xyz[2]']].to_numpy(float)
    bounded=(nn.timestamp_sample_us>=row.support_start_us)&(nn.timestamp_sample_us<=row.support_end_us);nt=nt[bounded];ny=ny[bounded];tt=(ss.state_measurement_us.to_numpy()[o:o+k+1]-t0)*1e-6;aa=ss.iloc[o:o+k+1][TARGET].to_numpy(float).astype(np.float32).astype(float);ini=ss.iloc[o][RATE].to_numpy(float).astype(np.float32).astype(float);bb=linear(nt,ny,tt,.02);eo=ini-bb[0];ef=aa[1:]-bb[1:];good=np.isfinite(ef).all(1)
    assert int(good.sum())==saved.supported_steps
    b300=linear(nt,ny,[.3],.02)[0];a300=linear(tt,aa,[.3])[0];bs=mean(nt,ny,.3,gap=.02);aa0=mean(tt,aa,.3)
    for j,v in enumerate(VARS):
     values={f'{v}_origin_error':eo[j],f'{v}_future_mse':np.mean(ef[good,j]**2),f'{v}_increment_mse':np.mean((ef[good,j]-eo[j])**2),f'{v}_B300_future':a300[j]-b300[j],f'{v}_D300_future':aa0[j]-bs[j]}
     for field,value in values.items():mx=max(mx,abs(value-saved[field]))
    checks.append(dict(split=split,registered_index=int(original),flight_id=fid,nodes=k,supported=int(good.sum())))
 assert mx<1e-7,mx
 # Full-table identity, no independent-contribution percentage assumption.
 identity=0.
 for v in VARS:identity=max(identity,float(abs(audit[f'{v}_increment_mse']-audit[f'{v}_future_mse']-audit[f'{v}_origin_error']**2+2*audit[f'{v}_cross_moment']).max()))
 assert identity<1e-10
 save(OUT/'independent_full_verification.json',dict(passed=True,windows=len(checks),flights=len({r['flight_id'] for r in checks}),max_error=mx,all_window_mse_identity_max_error=identity,full_table_rows=len(audit),checks=checks,training_runs=0))
 print('Independent full checks passed',len(checks),mx,flush=True)
if __name__=='__main__':fixed();full()
