"""Independent bracket interpolation, algebra, aggregation, and raw-channel ZOH audit.
Does not import the analysis metrics, splice_tape, event_pack, or model forward.
"""
from common import HERE,OUT,OLD,DATA,sha,save,np,pd,json

def lerp(t,y,q):
 if q<t[0]-1e-10 or q>t[-1]+1e-10:return np.full(y.shape[1],np.nan)
 k=int(np.searchsorted(t,q,side='right')-1)
 if k>=len(t)-1:return y[-1].astype(float)
 a=(q-t[k])/(t[k+1]-t[k]);return (1-a)*y[k].astype(float)+a*y[k+1].astype(float)

def components(pairs):
 graph={f:set() for f in set(pairs.flight_a)|set(pairs.flight_b)}
 for r in pairs.itertuples():graph[r.flight_a].add(r.flight_b);graph[r.flight_b].add(r.flight_a)
 seen={};label=0
 for f in sorted(graph):
  if f in seen:continue
  stack=[f]
  while stack:
   node=stack.pop()
   if node in seen:continue
   seen[node]=label;stack.extend(graph[node]-set(seen))
  label+=1
 return {r.pair_uid:seen[r.flight_a] for r in pairs.itertuples()}

def main():
 x=dict(np.load(OUT/'inputs.npz'));pred=dict(np.load(OUT/'predictions.npz'));old=dict(np.load(OLD/'results/native_rollouts.npz'));cases=pd.read_csv(OUT/'cases.csv');windows=pd.read_csv(OUT/'windows.csv');pairs=pd.read_csv(OUT/'pairs.csv');d=pd.read_csv(OUT/'point_decomposition.csv.gz');mem=pd.read_csv(OUT/'cohorts.csv');summary=pd.read_csv(OUT/'summary.csv');assert len(cases)==452 and len(pairs)==113
 lookup={u:i*4 for i,u in enumerate(pairs.pair_uid)};maxerr=0;count=0
 for row in d.itertuples():
  ids=range(lookup[row.pair_uid],lookup[row.pair_uid]+4);j={'vx':3,'vy':4,'vz':5,'p':10,'q':11,'r':12}[row.variable];yy=[]
  for ci in ids:
   n=int(x['valid_steps'][ci]);t=x['native_time_us'][ci,:n+1]*1e-6;assert np.all(np.diff(t)>0);y=np.vstack([x['initial'][ci],pred[f'{row.model}_{row.seed}'][ci,:n]]);yy.append((lerp(t,y,row.time_ms*.001)-y[0])[j])
  aa,ab,ba,bb=yy
  if not row.supported:
   assert any(np.isnan(yy)) and np.isnan(row.C);continue
  values=[aa,ab,ba,bb,aa-ab,ba-bb,aa-ba,ab-bb,(aa-ab+ba-bb)/2,(aa-ba+ab-bb)/2,aa-bb,(aa-ab)-(ba-bb)];saved=[getattr(row,k) for k in ['YAA','YAB','YBA','YBB','CA','CB','HA','HB','C','H','F','interaction']];maxerr=max(maxerr,float(np.max(abs(np.array(values)-saved))));count+=1
 assert maxerr<1e-12,maxerr
 closure=float(abs(d.F-d.C-d.H).max());assert closure<1e-12
 for key,p in pred.items():
  if key.startswith('GRU-NoControl'):np.testing.assert_array_equal(p[::4],p[1::4]);np.testing.assert_array_equal(p[2::4],p[3::4])
 for k in ['history','history_mask','history_dt','history_length','initial','dt']:
  np.testing.assert_array_equal(x[k][::4],x[k][1::4]);np.testing.assert_array_equal(x[k][2::4],x[k][3::4])
 # Independent global flight/component equal weighting and direct reductions.
 errors=[]
 for row in summary.itertuples():
  ids=mem[(mem.split==row.split)&(mem.cohort==row.cohort)].pair_uid;g=d[d.pair_uid.isin(ids)&(d.model==row.model)&(d.variable==row.variable)&(d.time_label.astype(str)==str(row.time_label))&d.supported]
  if str(row.seed)!='pooled':g=g[g.seed==int(row.seed)]
  pp=pairs[pairs.pair_uid.isin(g.pair_uid)];graph=components(pp);counts={k:list(graph.values()).count(k) for k in set(graph.values())};flightcounts={f:int(((pp.flight_a==f)|(pp.flight_b==f)).sum()) for f in set(pp.flight_a)|set(pp.flight_b)};pw={}
  for p in pp.itertuples():pw[p.pair_uid]=1 if row.weighting=='pair' else 1/counts[graph[p.pair_uid]] if row.weighting=='component' else 1/flightcounts[p.flight_a]+1/flightcounts[p.flight_b]
  w=np.array([pw[u] for u in g.pair_uid]);w=w/w.sum()
  for k in ['CA','CB','C','H','F','interaction']:errors.append(abs(float(np.sqrt(sum(w*g[k].to_numpy()**2)))-getattr(row,k+'_rms')))
  active=(abs(g.CA.to_numpy())>.1)&(abs(g.CB.to_numpy())>.1);expected=sum(w[active&(g.CA.to_numpy()*g.CB.to_numpy()>0)])/sum(w[active]) if active.any() else np.nan
  if np.isfinite(expected):errors.append(abs(expected-row.same_direction))
  ok=abs(g.g.to_numpy())>=.02;den=sum(w[ok]*g.g.to_numpy()[ok]**2)
  if den>1e-15:errors.append(abs(sum(w[ok]*g.C.to_numpy()[ok]*g.g.to_numpy()[ok])/den-row.C_signed_gain))
 assert max(errors)<1e-11,max(errors)
 # Direct native motor/servo ZOH, independently checking every union interval.
 raw={};source=json.loads((OUT/'source_receipt.json').read_text())
 for split in ['train','validation']:
  for kind in ['motors','servos']:
   file=f'native_actuator_{kind}_{split}.parquet';assert sha(DATA/file)==source['data_sha256'][file];frame=pd.read_parquet(DATA/file)
   for fid,g in frame.groupby('flight_id'):
    cols=['control[0]'] if kind=='motors' else ['control[0]','control[1]','control[2]'];g=g.sort_values('timestamp_us').drop_duplicates('timestamp_us',keep='last');raw[(fid,kind)]=(g.timestamp_us.to_numpy(),g[cols].to_numpy())
 checks=0;maxcmd=0.;mindonor=10**12;maxdonor=0;encoded_tapes={}
 for r in cases.itertuples():
  wr=windows.iloc[r.receiver];wd=windows.iloc[r.donor];offset=0;stit=[];stiu=[]
  for k in range(int(x['valid_steps'][r.case_id])):
   n=int(x['event_length'][r.case_id,k]);ev=x['events'][r.case_id,k,:n];du=np.rint(ev[:,4].astype(float)*1e6).astype(int);assert du.sum()==round(float(x['dt'][r.case_id,k])*1e6)
   for e,span in zip(ev,du):stit.append(offset);stiu.append(e[:4]);offset+=int(span)
  stit=np.array(stit);stiu=np.array(stiu,dtype=float);encoded_tapes[r.case_id]=(stit,stiu,offset);assert offset==int(x['native_time_us'][r.case_id,r.valid_steps]);assert wr.initial_measurement_us+offset<=wr.decision_us+500000
  times=[0,offset,int(wr.decision_us-wr.initial_measurement_us)];times.extend(stit)
  for fid,shift in [(wr.flight_id,0),(wd.flight_id,int(wr.decision_us-wd.decision_us))]:
   for kind in ['motors','servos']:
    t,_=raw[fid,kind];relative=t+shift-int(wr.initial_measurement_us);times.extend(relative[(relative>0)&(relative<offset)])
  knots=np.unique(times);q=(knots[:-1]+knots[1:])/2;absolute=q+wr.initial_measurement_us;own=absolute<wr.decision_us;expected=np.zeros((len(q),4))
  for mask,fid,shift in [(own,wr.flight_id,0),(~own,wd.flight_id,wr.decision_us-wd.decision_us)]:
   query=absolute[mask]-shift
   if not len(query):continue
   for kind,sl in [('motors',slice(0,1)),('servos',slice(1,4))]:
    t,u=raw[fid,kind];assert query.min()>=t[0] and query.max()<=t[-1];expected[mask,sl]=u[np.searchsorted(t,query,side='right')-1]
  relative=absolute[~own]-(wr.decision_us-wd.decision_us)-wd.decision_us;mindonor=min(mindonor,float(relative.min()));maxdonor=max(maxdonor,float(relative.max()));assert relative.min()>=0 and relative.max()<=500000
  actual=stiu[np.searchsorted(stit,q,side='right')-1];maxcmd=max(maxcmd,float(abs(actual-expected).max()));checks+=len(q)
 assert maxcmd<1e-7,maxcmd
 # Independent single-tape clipped-duration integrals (no union difference helper).
 ctrl=pd.read_csv(OUT/'control_differences.csv');gainerr=0.
 for row in ctrl[ctrl.supported&(ctrl.time_ms>0)].itertuples():
  start=lookup[row.pair_uid];means=[];end=int(round(row.time_ms*1000))
  for ci in range(start,start+4):
   t,u,last=encoded_tapes[ci];dur=np.maximum(0,np.minimum(np.r_[t[1:],last],end)-np.minimum(t,end));means.append(float(np.sum(dur*(u[:,2]-u[:,1])/2)/end))
  gainerr=max(gainerr,abs(means[0]-means[1]-row.gA),abs(means[2]-means[3]-row.gB))
 assert gainerr<1e-12,gainerr
 train=set(windows[windows.split=='train'].flight_id);val=set(windows[windows.split=='validation'].flight_id);assert not train&val;assert set(cases.split)=={'train','validation'}
 for r in cases.itertuples():assert windows.iloc[r.receiver].split==windows.iloc[r.donor].split==r.split
 support=pd.read_csv(OUT/'time_support.csv');assert support[support.pair_uid.isin(mem.query("split=='train' and cohort=='main'").pair_uid)].main300.all()
 save(OUT/'independent_verification.json',dict(passed=True,point_rows_verified=count,max_interpolation_algebra_error=maxerr,closure_max=closure,summary_rows_verified=len(summary),max_summary_error=max(errors),raw_ZOH_union_intervals_checked=checks,max_command_error=maxcmd,max_differential_mean_error=gainerr,donor_relative_time_min_us=mindonor,donor_relative_time_max_us=maxdonor,NoControl_exact=True,non_event_inputs_identical=True,no_extrapolation=True,all_main300_supported=True,train_validation_disjoint=True,test_read=False))
 print(json.dumps(json.loads((OUT/'independent_verification.json').read_text()),indent=2))
if __name__=='__main__':main()
