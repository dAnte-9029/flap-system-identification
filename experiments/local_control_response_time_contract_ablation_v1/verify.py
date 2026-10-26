"""Independent phase formula, interpolation/integration and main summaries."""
from common import *

def linear(t,y,q):
 if q<t[0]-1e-10 or q>t[-1]+1e-10:return np.full(y.shape[1],np.nan)
 k=min(max(int(np.searchsorted(t,q,side='right')-1),0),len(t)-2);a=(q-t[k])/(t[k+1]-t[k]);return (1-a)*y[k].astype(float)+a*y[k+1].astype(float)
def average(t,y,center,width):
 lo,hi=center-width/2,center+width/2
 if lo<t[0]-1e-10 or hi>t[-1]+1e-10:return np.full(y.shape[1],np.nan)
 total=np.zeros(y.shape[1])
 for i in range(len(t)-1):
  a=max(lo,t[i]);b=min(hi,t[i+1])
  if b<=a:continue
  slope=(y[i+1].astype(float)-y[i].astype(float))/(t[i+1]-t[i]);total+=y[i]*(b-a)+slope*((b-t[i])**2-(a-t[i])**2)/2
 return total/width

def main():
 old=dict(np.load(SWAP/'results/inputs.npz'));new=dict(np.load(OUT/'V1_inputs.npz'));z=dict(np.load(OLD/'results/native_rollouts.npz'));cases=pd.read_csv(SWAP/'results/cases.csv');pairs=pd.read_csv(SWAP/'results/pairs.csv');native=dict(np.load(TIMING/'results/native_diagnostic_inputs.npz'));hrows=pd.read_csv(OUT/'input_changes.csv.gz');w=pd.read_csv(SWAP/'results/windows.csv');wi=dict(zip(w.window_id,w.rollout_index));errors=[]
 for r in hrows.itertuples():
  delta=2*np.pi*r.frequency*(r.state_us-r.encoder_source_us)*1e-6;assert r.encoder_available_upper_us<=r.decision_us and r.frequency_available_upper_us<=r.decision_us and r.rate_available_us<=r.decision_us
  # Rotate the original unit-circle feature directly, independently of atan2 conversion.
  expected=np.array([r.old_sin*np.cos(delta)+r.old_cos*np.sin(delta),r.old_cos*np.cos(delta)-r.old_sin*np.sin(delta)]);errors.append(float(abs(expected-[r.new_sin,r.new_cos]).max()))
 assert max(errors)<3e-7,max(errors)
 for k in old:
  if k=='history':np.testing.assert_array_equal(old[k][...,[i for i in range(20) if i not in [17,18]]],new[k][...,[i for i in range(20) if i not in [17,18]]])
  else:np.testing.assert_array_equal(old[k],new[k])
 for k in ['history','history_mask','history_dt','history_length','initial','dt']:
  np.testing.assert_array_equal(new[k][::4],new[k][1::4]);np.testing.assert_array_equal(new[k][2::4],new[k][3::4])
 pred={'V0':dict(np.load(SWAP/'results/predictions.npz')),'V1_source_assumption':dict(np.load(OUT/'V1_predictions.npz'))}
 for version,models in pred.items():
  for key,arr in models.items():
   if key.startswith('GRU-NoControl'):np.testing.assert_array_equal(arr[::4],arr[1::4]);np.testing.assert_array_equal(arr[2::4],arr[3::4])
 facts=pd.read_csv(OUT/'factual_points.csv.gz');internal=pd.read_csv(OUT/'internal_points.csv.gz');index={r.pair_uid:r for r in pairs.itertuples()};rowindex={r.pair_uid:i*4 for i,r in enumerate(pairs.itertuples())};err=0.;count=0
 for r in facts[(facts.time_label=='300')&facts.supported].itertuples():
  pair=index[r.pair_uid];ids=[rowindex[r.pair_uid],rowindex[r.pair_uid]+3];j=CHANNELS[VARS.index(r.variable)];values=[];truth=[]
  for ci,window in zip(ids,[pair.a,pair.b]):
   n=int(old['valid_steps'][ci]);t=old['native_time_us'][ci,:n+1]*1e-6;y=np.vstack([old['initial'][ci],pred[r.variant][f'{r.model}_{r.seed}'][ci,:n]]).astype(float);y-=y[0].copy();values.append((average(t,y,.3,.27) if r.method=='D' else linear(t,y,.3))[j])
   if r.method=='A':
    nn=int(z['future_count'][window]);tt=z['native_time_us'][window,:nn+1]*1e-6;yy=np.vstack([z['truth_initial'][window],z['truth'][window,:nn]]).astype(float);yy-=yy[0].copy();truth.append(linear(tt,yy,.3)[j])
   else:
    tt=native[f'rate_t_{window}'];yy=native[f'rate_y_{window}'].astype(float);yy=yy-linear(tt,yy,0);truth.append((average(tt,yy,.3,.27) if r.method=='D' else linear(tt,yy,.3))[j-10])
  err=max(err,abs(values[0]-values[1]-r.pred),abs(truth[0]-truth[1]-r.real));count+=1
 assert err<1e-11,err
 algebra=float(abs(internal.F-internal.C-internal.H).max());assert algebra<1e-12
 # Full internal primary points directly from four saved trajectories.
 ierr=0.
 for r in internal[(internal.time_label=='300')&internal.supported].itertuples():
  ys=[];j=CHANNELS[VARS.index(r.variable)]
  for ci in range(rowindex[r.pair_uid],rowindex[r.pair_uid]+4):
   n=int(old['valid_steps'][ci]);t=old['native_time_us'][ci,:n+1]*1e-6;y=np.vstack([old['initial'][ci],pred[r.variant][f'{r.model}_{r.seed}'][ci,:n]]).astype(float);ys.append((linear(t,y,.3)-y[0])[j])
  aa,ab,ba,bb=ys;expected=[aa-ab,ba-bb,(aa-ab+ba-bb)/2,(aa-ba+ab-bb)/2];ierr=max(ierr,float(abs(np.array(expected)-[r.CA,r.CB,r.C,r.H]).max()))
 assert ierr<1e-12
 # Independent equal-flight and graph-component weights for all 300 ms summary rows.
 membership=pd.read_csv(SWAP/'results/cohorts.csv');s=pd.read_csv(OUT/'factual_summary.csv');s=s[s.time_label=='300'];maxstat=0;grouped=dict(tuple(facts[(facts.time_label=='300')&facts.supported].groupby(['variant','model','method','variable'])))
 for r in s.itertuples():
  ids=membership[(membership.split==r.split)&(membership.cohort==r.cohort)].pair_uid;source=grouped[(r.variant,r.model,r.method,r.variable)];g=source[source.pair_uid.isin(ids)]
  if str(r.seed)!='pooled':g=g[g.seed==int(r.seed)]
  pp=pairs[pairs.pair_uid.isin(g.pair_uid)];adj={f:set() for f in set(pp.flight_a)|set(pp.flight_b)}
  for p in pp.itertuples():adj[p.flight_a].add(p.flight_b);adj[p.flight_b].add(p.flight_a)
  component={};ncomp=0
  for fid in adj:
   if fid in component:continue
   stack=[fid]
   while stack:
    node=stack.pop()
    if node in component:continue
    component[node]=ncomp;stack.extend(adj[node]-set(component))
   ncomp+=1
  labels={p.pair_uid:component[p.flight_a] for p in pp.itertuples()};degree={f:int(((pp.flight_a==f)|(pp.flight_b==f)).sum()) for f in adj};wm={p.pair_uid:(1 if r.weighting=='pair' else 1/degree[p.flight_a]+1/degree[p.flight_b] if r.weighting=='flight' else 1/list(labels.values()).count(labels[p.pair_uid])) for p in pp.itertuples()};weights=np.array([wm[u] for u in g.pair_uid],float);weights/=weights.sum();a=g.real.to_numpy();b=g.pred.to_numpy();rr=sum(weights*a*a);ppower=sum(weights*b*b);gain=sum(weights*a*b)/rr if rr else np.nan;ratio=np.sqrt(ppower/rr) if rr else np.nan;active=abs(a)>.1;direction=sum(weights[active]*(a[active]*b[active]>0))/sum(weights[active]) if any(active) else np.nan
  for expected,actual in [(gain,r.signed_gain),(ratio,r.scale_ratio),(direction,r.direction_accuracy)]:
   if np.isfinite(expected):maxstat=max(maxstat,abs(expected-actual))
 assert maxstat<1e-11
 # Independent internal C/H RMS reductions, including component and flight weights.
 ins=pd.read_csv(OUT/'internal_summary.csv');ins=ins[ins.time_label=='300'];igroups=dict(tuple(internal[(internal.time_label=='300')&internal.supported].groupby(['variant','model','variable'])));internal_stat=0.
 for r in ins.itertuples():
  ids=membership[(membership.split==r.split)&(membership.cohort==r.cohort)].pair_uid;source=igroups[(r.variant,r.model,r.variable)];g=source[source.pair_uid.isin(ids)]
  if str(r.seed)!='pooled':g=g[g.seed==int(r.seed)]
  pp=pairs[pairs.pair_uid.isin(g.pair_uid)];adj={f:set() for f in set(pp.flight_a)|set(pp.flight_b)}
  for p in pp.itertuples():adj[p.flight_a].add(p.flight_b);adj[p.flight_b].add(p.flight_a)
  labels={};label=0
  for fid in adj:
   if fid in labels:continue
   stack=[fid]
   while stack:
    node=stack.pop()
    if node in labels:continue
    labels[node]=label;stack.extend(adj[node]-set(labels))
   label+=1
  counts={c:sum(labels[p.flight_a]==c for p in pp.itertuples()) for c in set(labels.values())};degree={f:sum(p.flight_a==f or p.flight_b==f for p in pp.itertuples()) for f in adj};pw={p.pair_uid:(1. if r.weighting=='pair' else 1/degree[p.flight_a]+1/degree[p.flight_b] if r.weighting=='flight' else 1/counts[labels[p.flight_a]]) for p in pp.itertuples()};ww=np.array([pw[u] for u in g.pair_uid]);ww/=ww.sum()
  for key in ['CA','CB','C','H']:internal_stat=max(internal_stat,abs(float(np.sqrt(sum(ww*g[key].to_numpy()**2)))-getattr(r,key+'_rms')))
 assert internal_stat<1e-12
 receipt=json.loads((OUT/'input_receipt.json').read_text());assert not set(receipt['train_flights'])&set(receipt['validation_flights']);assert not receipt['test_read']
 save(OUT/'independent_verification.json',dict(passed=True,phase_rotation_max_error=max(errors),factual_primary_rows=count,max_factual_error=err,max_internal_error=ierr,closure_max=algebra,summary_primary_rows=len(s),summary_max_error=maxstat,internal_summary_rows=len(ins),internal_summary_max_error=internal_stat,NoControl_invariant=True,only_phase_sin_cos_changed=True,no_future_inputs=True,train_validation_disjoint=True,test_read=False))
 print(json.dumps(json.loads((OUT/'independent_verification.json').read_text()),indent=2))
if __name__=='__main__':main()
