from common import *
from state import causal_rate,apply_rates

def run():
 torch.set_num_threads(4);receipt=json.loads((PREV/'results/source_receipt.json').read_text());sources={};rows=[];states=[];cases=[];histchecks=0
 def checked(p):
  candidates=[h for old,h in receipt['files'].items() if old==str(p) or (str(p).startswith(str(ROOT)) and old.endswith(str(p.relative_to(ROOT))))]
  assert len(set(candidates))==1,p
  actual=sha(p);assert actual==candidates[0],p;sources[str(p)]=actual
 checked(DATA/'dataset_manifest.json');checked(ROOT/'src/system_identification/models/velocity_rate_gru.py')
 fixed=pd.read_csv(PREV/'results/windows.csv');pairs=pd.read_csv(PREV/'results/pairs.csv');wanted=pairs[pairs.pair_uid.isin(['train:124','train:336'])]
 caseids=set(wanted.window_a)|set(wanted.window_b)
 for split in ['train','validation']:
  for p in [DATA/f'samples_{split}.parquet',DATA/f'native_vehicle_angular_velocity_{split}.parquet',CACHE/f'{split}.npz',TRAIN/f'{split}_windows.parquet']:checked(p)
  meta=pd.read_parquet(TRAIN/f'{split}_windows.parquet');s=pd.read_parquet(DATA/f'samples_{split}.parquet',columns=['flight_id','state_measurement_us','state_publication_us','rate_measurement_us','rate_publication_us']+STATE);native=pd.read_parquet(DATA/f'native_vehicle_angular_velocity_{split}.parquet');sg=dict(tuple(s.groupby('flight_id',sort=False)));ng=dict(tuple(native.groupby('flight_id',sort=False)))
  cache=np.load(CACHE/f'{split}.npz');initial=cache['initial'];target=cache['target'];dt=cache['dt']
  for flight_no,(fid,mm) in enumerate(meta.groupby('flight_id',sort=False)):
   ss=sg[fid].reset_index(drop=True);nn=ng[fid];tm=nn.timestamp_sample_us.to_numpy(np.int64);pub=nn.timestamp_us.to_numpy(np.int64);y=nn[['xyz[0]','xyz[1]','xyz[2]']].to_numpy(float);t=ss.state_measurement_us.to_numpy(np.int64);dec=ss.state_publication_us.to_numpy(np.int64);c0=ss[RATE].to_numpy(float).astype(np.float32).astype(float)
   c1,info=causal_rate(tm,pub,y,t,dec,c0);c1=c1.astype(np.float32).astype(float);accepted=np.isin(info['reason'],ACCEPT)
   ref=int(t[0]);b,bok,bl,br=sample((tm-ref)*1e-6,y,(t-ref)*1e-6)
   o=mm.origin_sample.to_numpy(int);start=mm.start_sample.to_numpy(int);count=mm.count_1000ms.to_numpy(int);ids=mm.index.to_numpy();k=int(count.max());fi=np.minimum(o[:,None]+np.arange(1,k+1),(o+count)[:,None]);mask=np.arange(k)[None,:]<count[:,None]
   np.testing.assert_array_equal(initial[ids,10:13],c0[o]);assert (t[o]==mm.initial_measurement_us.to_numpy()).all() and (dec[o]==mm.decision_us.to_numpy()).all()
   hlen=mm.history_count.to_numpy(int);hi=np.minimum(start[:,None]+np.arange(int(hlen.max())),o[:,None]);hm=np.arange(hi.shape[1])[None,:]<hlen[:,None]
   # Build real history states with C1 at each row's own deadline; initial identical to final history row.
   h=ss[STATE].to_numpy(float).astype(np.float32)[hi];hc,ic=apply_rates(h,initial[ids],c1[hi],hlen)
   np.testing.assert_array_equal(hc[np.arange(len(o)),hlen-1,10:13],ic[:,10:13]);np.testing.assert_array_equal(ic[:,:10],initial[ids,:10]);np.testing.assert_array_equal(hc[...,:10],h[...,:10]);histchecks+=len(o)
   limitlo=mm.support_start_us.to_numpy();limithi=mm.support_end_us.to_numpy();bvalid=bok[fi]&(tm[bl[fi]]>=limitlo[:,None])&(tm[br[fi]]<=limithi[:,None])&mask
   prefix=np.cumprod(bvalid|~mask,axis=1).astype(bool)&mask;complete=(bvalid.sum(1)==count);bn=b[fi].copy()
   # Stop at the first unsupported target; missing labels are never filled or scored.
   tar=target[ids,:k].astype(float);ini=initial[ids].astype(float);ini1=ini.copy();ini1[:,10:13]=ic[:,10:13];d=dt[ids,:k].astype(float)
   q0,p0=kinematics(ini,tar[:,:,3:6],tar[:,:,10:13],d,tar,prefix);q1,p1=kinematics(ini1,tar[:,:,3:6],bn,d,tar,prefix)
   np.testing.assert_array_equal(p0,p1)
   used=np.unique(hi[hm]);src=info['source_indices'];a=np.maximum(src[:,0],0);bb=np.maximum(src[:,1],0);change=np.linalg.norm(y[bb]-y[a],axis=1);latest_age=t-tm[bb]
   for j in used:
    r=dict(split=split,flight_id=fid,sample_index=int(j),state_us=int(t[j]),decision_us=int(dec[j]),reason=info['reason'][j],accepted=bool(accepted[j]),source_left=int(src[j,0]),source_right=int(src[j,1]),left_measurement_us=int(tm[a[j]]) if src[j,0]>=0 else -1,right_measurement_us=int(tm[bb[j]]) if src[j,1]>=0 else -1,left_publication_us=int(pub[a[j]]) if src[j,0]>=0 else -1,right_publication_us=int(pub[bb[j]]) if src[j,1]>=0 else -1,gap_us=int(info['gap_us'][j]),extrapolation_us=int(info['extrapolation_us'][j]),B_supported=bool(bok[j]))
    for ax,v in enumerate(VARS):r.update({f'C0_{v}':c0[j,ax],f'C1_{v}':c1[j,ax],f'B_{v}':b[j,ax]})
    states.append(r)
   for local,ix in enumerate(ids):
    j=o[local];valid=prefix[local];n=int(valid.sum());last=n-1
    r=dict(split=split,registered_index=int(ix),window_id=mm.window_id.iloc[local],flight_id=fid,log_sha=fid.split(':')[0],origin_sample=int(j),reason=info['reason'][j],accepted=bool(accepted[j]),latest_age_us=int(latest_age[j]),extrapolation_us=int(info['extrapolation_us'][j]),observation_change_norm=change[j],history_rows=int(hlen[local]),history_fallback_rows=int((~accepted[hi[local,hm[local]]]).sum()),future_steps=int(count[local]),B_supported_steps=int(bvalid[local].sum()),common_prefix_steps=n,complete_B=bool(complete[local]),common_duration_s=float(d[local,:n].sum()),T0_attitude_mse=float(np.mean(q0[local,valid]**2)) if n else np.nan,T1_attitude_mse=float(np.mean(q1[local,valid]**2)) if n else np.nan,T0_endpoint_rad=float(q0[local,last]) if n else np.nan,T1_endpoint_rad=float(q1[local,last]) if n else np.nan,position_mse=float(np.mean(p0[local,valid]**2)) if n else np.nan)
    for ax,v in enumerate(VARS):r.update({f'C0_{v}_error':c0[j,ax]-b[j,ax],f'C1_{v}_error':c1[j,ax]-b[j,ax],f'C0_{v}':c0[j,ax],f'C1_{v}':c1[j,ax],f'B_{v}':b[j,ax]})
    # Initial-to-first-label jump is descriptive, contains actual dynamics, not itself an error target.
    r['T0_first_jump_norm']=float(np.linalg.norm(tar[local,0,10:13]-c0[j]));r['T1_first_jump_norm']=float(np.linalg.norm(bn[local,0]-c1[j])) if bvalid[local,0] else np.nan
    rows.append(r)
   # Fixed local cases retain exact origin identity; no matching or full Oracle analysis.
   for wr in fixed[(fixed.flight_id==fid)&fixed.window_id.isin(caseids)].itertuples():
    j=wr.origin_sample;found=np.flatnonzero(o==j);assert len(found)==1;local=found[0]
    for uid in wanted[(wanted.window_a==wr.window_id)|(wanted.window_b==wr.window_id)].pair_uid:
     for ax,v in enumerate(VARS):cases.append(dict(pair_uid=uid,window_id=wr.window_id,flight_id=fid,axis=v,C0=c0[j,ax],C1=c1[j,ax],B=b[j,ax],C0_error=c0[j,ax]-b[j,ax],C1_error=c1[j,ax]-b[j,ax],reason=info['reason'][j],T0_attitude_path_rms=np.sqrt(np.mean(q0[local,prefix[local]]**2)),T1_attitude_path_rms=np.sqrt(np.mean(q1[local,prefix[local]]**2))))
   if flight_no%20==0:print(split,flight_no+1,'flights processed',flush=True)
  del cache,initial,target,dt,s,native,sg,ng
 data=pd.DataFrame(rows);data.to_csv(OUT/'windows.csv.gz',index=False);pd.DataFrame(states).astype({f'{c}_{v}':'float32' for c in ['C0','C1'] for v in VARS}).to_parquet(OUT/'history_states.parquet',index=False,compression='zstd');pd.DataFrame(cases).to_csv(OUT/'cases.csv',index=False)
 save(OUT/'receipt.json',dict(base_sha='3ee850af770631f6be3d5a9326987aa443307394',historical_source_receipt_sha256=sha(PREV/'results/source_receipt.json'),files=sources,training_runs=0,model_inference=0,test_read=False,history_initial_identity_checked=histchecks,history_uses_each_row_deadline=True,CPU=True))
 print('Complete',len(data),'windows',len(states),'unique historical rows',flush=True)

if __name__=='__main__':run()
