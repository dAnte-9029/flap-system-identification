"""Vectorized by-flight audit of every registered H1000 train/validation window."""
from common import *

def sample(t,y,x,gap=.02):
 """Vectorized, exact timestamp bracketing; return values, support and indices."""
 t=np.asarray(t,float);x=np.asarray(x,float);assert np.all(np.diff(t)>0)
 j=np.searchsorted(t,x,side='left');right=np.clip(j,0,len(t)-1);left=np.maximum(right-1,0)
 exact=np.abs(t[right]-x)<1e-10;left=np.where(exact,right,left)
 good=(x>=t[0]-1e-10)&(x<=t[-1]+1e-10)&((t[right]-t[left])<=gap+1e-10)
 denom=t[right]-t[left];f=np.divide(x-t[left],denom,out=np.zeros_like(x),where=denom>0)
 v=y[left]*(1-f[...,None])+y[right]*f[...,None];v=np.where(good[...,None],v,np.nan)
 return v,good,left,right

def average(t,y,lo,hi,gap=.02):
 """Antiderivative with explicit rejection of any intersected oversized gap."""
 t=np.asarray(t,float);lo=np.asarray(lo,float);hi=np.asarray(hi,float)
 cumulative=np.vstack([np.zeros(y.shape[1]),np.cumsum(np.diff(t)[:,None]*(y[:-1]+y[1:])/2,axis=0)])
 def primitive(q):
  v,ok,l,r=sample(t,y,q,gap);delta=q-t[l];return cumulative[l]+delta[...,None]*(y[l]+v)/2,ok,l,r
 a,ok1,l1,r1=primitive(lo);b,ok2,l2,r2=primitive(hi)
 bad=np.r_[0,np.cumsum(np.diff(t)>gap+1e-10)];ok=ok1&ok2&(bad[r2]==bad[l1])&(hi>lo)
 return np.where(ok[...,None],(b-a)/(hi-lo)[...,None],np.nan),ok

def audit():
 rows=[];flightrows=[];checks=[];nodecounts={}
 for split,expected,nflights in [('train',107920,94),('validation',36332,27)]:
  meta=pd.read_parquet(TRAIN/f'{split}_windows.parquet');assert len(meta)==expected and meta.flight_id.nunique()==nflights and set(meta.split)=={split}
  source=pd.read_parquet(DATA/f'samples_{split}.parquet',columns=['flight_id','split','segment_id','valid_core','state_measurement_us','state_publication_us','rate_measurement_us','rate_publication_us','target_rate_measurement_us','target_rate_publication_us']+RATE+TARGET)
  native=pd.read_parquet(DATA/f'native_vehicle_angular_velocity_{split}.parquet',columns=['flight_id','timestamp_sample_us','timestamp_us','xyz[0]','xyz[1]','xyz[2]'])
  assert set(source.split)=={split};sg=dict(tuple(source.groupby('flight_id',sort=False)));ng=dict(tuple(native.groupby('flight_id',sort=False)))
  cache=np.load(CACHE/f'{split}.npz');targets=cache['target'];initials=cache['initial'];cdt=cache['dt'];mask=cache['future_mask'];total=0;validtotal=0
  for fi,(fid,mm) in enumerate(meta.groupby('flight_id',sort=False)):
   s=sg[fid].reset_index(drop=True);nat=ng[fid];ref=float(s.state_measurement_us.iloc[0]);t=(s.state_measurement_us.to_numpy()-ref)*1e-6;nt=(nat.timestamp_sample_us.to_numpy()-ref)*1e-6;ny=nat[['xyz[0]','xyz[1]','xyz[2]']].to_numpy(float)
   idx=mm.index.to_numpy();o=mm.origin_sample.to_numpy(int);counts=mm.count_1000ms.to_numpy(int);k=int(counts.max());future=o[:,None]+np.arange(1,k+1);future=np.minimum(future,(o+counts)[:,None]);fm=np.arange(k)[None,:]<counts[:,None]
   assert (s.segment_id.to_numpy()[o]==s.segment_id.to_numpy()[o+counts]).all() and s.valid_core.to_numpy()[future].all()
   assert (s.state_measurement_us.to_numpy()[o]==mm.initial_measurement_us).all() and (s.state_publication_us.to_numpy()[o]==mm.decision_us).all()
   av=s[TARGET].to_numpy(float).astype(np.float32).astype(float);ini=s[RATE].to_numpy(float).astype(np.float32).astype(float)[o]
   np.testing.assert_array_equal(targets[idx,:k,10:13][fm],av[future][fm]);np.testing.assert_array_equal(initials[idx,10:13],ini)
   np.testing.assert_array_equal(mask[idx,:k],fm);dt=(s.state_measurement_us.to_numpy()[future]-s.state_measurement_us.to_numpy()[np.column_stack([o,future[:,:-1]])])*1e-6
   np.testing.assert_array_equal(cdt[idx,:k][fm],dt.astype(np.float32)[fm])
   b,bgood,left,right=sample(nt,ny,t);j0=o;bf=b[future];b0=b[o];ef=av[future]-bf;eo=ini-b0;inc=ef-eo[:,None,:]
   # Window's registered support is an additional provenance boundary.
   boundlo=(mm.support_start_us.to_numpy()-ref)*1e-6;boundhi=(mm.support_end_us.to_numpy()-ref)*1e-6
   originok=bgood[o]&(nt[left[o]]>=boundlo-1e-10)&(nt[right[o]]<=boundhi+1e-10)
   good=fm&bgood[future]&(nt[left[future]]>=boundlo[:,None]-1e-10)&(nt[right[future]]<=boundhi[:,None]+1e-10)&originok[:,None]
   total+=int(fm.sum());validtotal+=int(good.sum());countgood=good.sum(1)
   at=t[o]+.3;aa,ag,_,_=sample(t,av,at,.04);bb,bg,bl,br=sample(nt,ny,at)
   aslow,asok=average(t,av,at-.135,at+.135,.04);bslow,bsok=average(nt,ny,at-.135,at+.135,.02)
   rawok=ag&bg&originok&(at<=t[o+counts]+1e-10)&(nt[bl]>=boundlo)&(nt[br]<=boundhi)
   _,_,slo,_=sample(nt,ny,at-.135);_,_,_,shi=sample(nt,ny,at+.135)
   slowok=asok&bsok&originok&(nt[slo]>=boundlo)&(nt[shi]<=boundhi)&(at+.135<=t[o+counts]+1e-10)&(at-.135>=t[o]-1e-10)
   # Output centers with full diagnostic-filter support, never fill edges.
   _,filterok=average(nt,ny,t[future]-.135,t[future]+.135,.02)
   _,_,flo,_=sample(nt,ny,t[future]-.135);_,_,_,fhi=sample(nt,ny,t[future]+.135)
   filterok&=(nt[flo]>=boundlo[:,None])&(nt[fhi]<=boundhi[:,None])
   filterok&=fm&(t[future]-.135>=t[o,None]-1e-10)&(t[future]+.135<=t[o+counts,None]+1e-10)&originok[:,None]
   mean=lambda v:np.divide(np.where(good[...,None],v,0).sum(1),countgood[:,None],out=np.full((len(o),3),np.nan),where=countgood[:,None]>0)
   msf=mean(ef**2);msi=mean(inc**2);cross=mean(ef*eo[:,None,:]);bias=mean(ef)
   # Exact source IDs: nearest-neighbor target is checked against raw native measurements.
   ti=np.searchsorted(nat.timestamp_sample_us.to_numpy(),s.target_rate_measurement_us.to_numpy()[np.unique(np.r_[o,future[fm]])]);si=np.unique(np.r_[o,future[fm]])
   np.testing.assert_array_equal(nat.timestamp_sample_us.to_numpy()[ti],s.target_rate_measurement_us.to_numpy()[si]);np.testing.assert_array_equal(ny[ti],s[TARGET].to_numpy(float)[si])
   assert (s.rate_publication_us.to_numpy()[o]<=s.state_publication_us.to_numpy()[o]).all()
   for local,registered in enumerate(idx):
    r=dict(split=split,registered_index=int(registered),window_id=mm.window_id.iloc[local],flight_id=fid,log_sha=fid.split(':')[0],origin_sample=int(o[local]),state_time_us=int(s.state_measurement_us.iloc[o[local]]),decision_us=int(mm.decision_us.iloc[local]),future_steps=int(counts[local]),supported_steps=int(countgood[local]),unsupported_steps=int(counts[local]-countgood[local]),origin_supported=bool(originok[local]),B300_supported=bool(rawok[local]),D300_supported=bool(slowok[local]),D_supported_steps=int(filterok[local].sum()),initial_source_age_us=int(s.state_measurement_us.iloc[o[local]]-s.rate_measurement_us.iloc[o[local]]),truth_initial_offset_us=int(s.target_rate_measurement_us.iloc[o[local]]-s.state_measurement_us.iloc[o[local]]),truth_initial_published_after_decision=bool(s.target_rate_publication_us.iloc[o[local]]>mm.decision_us.iloc[local]),target_offset_abs_mean_us=float(abs(s.target_rate_measurement_us.to_numpy()[future[local,fm[local]]]-s.state_measurement_us.to_numpy()[future[local,fm[local]]]).mean()),future_target_after_origin_fraction=float((s.target_rate_publication_us.to_numpy()[future[local,fm[local]]]>mm.decision_us.iloc[local]).mean()),reason='supported' if countgood[local]==counts[local] else ('origin_missing_bracket_or_registered_support' if not originok[local] else 'future_missing_bracket_or_gap_or_registered_support'))
    for j,v in enumerate(VARS):
     r.update({f'{v}_origin_error':eo[local,j],f'{v}_future_mse':msf[local,j],f'{v}_future_bias':bias[local,j],f'{v}_increment_mse':msi[local,j],f'{v}_cross_moment':cross[local,j],f'{v}_B300_future':aa[local,j]-bb[local,j] if rawok[local] else np.nan,f'{v}_D300_future':aslow[local,j]-bslow[local,j] if slowok[local] else np.nan})
    rows.append(r)
   checks.append(dict(split=split,flight_id=fid,windows=len(o),target_cache_exact=True,initial_cache_exact=True,dt_cache_exact=True,source_native_values_exact=True))
   if fi%20==0:print(split,fi+1,'/',nflights,'windows',len(rows),flush=True)
  nodecounts[split]=dict(windows=len(meta),flights=nflights,valid_future_nodes=total,supported_future_nodes=validtotal)
  del targets,initials,cdt,mask,cache,source,native,sg,ng
 d=pd.DataFrame(rows);d.to_csv(OUT/'full_window_audit.csv.gz',index=False)
 summary=[]
 for split,dd in d.groupby('split'):
  for axis in VARS:
   for weighting in ['window','flight']:
    weights=np.ones(len(dd)) if weighting=='window' else 1/dd.groupby('flight_id').flight_id.transform('size').to_numpy();weights/=weights.sum()
    avg=lambda v:np.nansum(weights*np.asarray(v))/np.sum(weights[np.isfinite(v)])
    origin=dd[f'{axis}_origin_error'].to_numpy();future=dd[f'{axis}_future_mse'].to_numpy();incr=dd[f'{axis}_increment_mse'].to_numpy();cross=dd[f'{axis}_cross_moment'].to_numpy()
    summary.append(dict(split=split,axis=axis,weighting=weighting,windows=len(dd),flights=dd.flight_id.nunique(),origin_rms=np.sqrt(avg(origin**2)),future_rms=np.sqrt(avg(future)),increment_rms=np.sqrt(avg(incr)),cross_moment=avg(cross),future_bias=avg(dd[f'{axis}_future_bias']),B300_future_rms=np.sqrt(avg(dd[f'{axis}_B300_future']**2)),D300_future_rms=np.sqrt(avg(dd[f'{axis}_D300_future']**2)),B300_increment_rms=np.sqrt(avg((dd[f'{axis}_B300_future']-origin)**2)),D300_increment_rms=np.sqrt(avg((dd[f'{axis}_D300_future']-origin)**2))))
 d.groupby(['split','flight_id']).mean(numeric_only=True).to_csv(OUT/'full_per_flight.csv');pd.DataFrame(summary).to_csv(OUT/'full_summary.csv',index=False)
 for split,dd in d.groupby('split'):
  nodecounts[split].update(origin_supported=int(dd.origin_supported.sum()),B300_supported=int(dd.B300_supported.sum()),D300_supported=int(dd.D300_supported.sum()),D_supported_nodes=int(dd.D_supported_steps.sum()),fully_supported_windows=int((dd.unsupported_steps==0).sum()),missing_reasons=dd.reason.value_counts().to_dict(),initial_age_median_us=float(dd.initial_source_age_us.median()),initial_age_min_us=int(dd.initial_source_age_us.min()),initial_age_max_us=int(dd.initial_source_age_us.max()),truth_initial_after_decision_fraction=float(dd.truth_initial_published_after_decision.mean()))
 save(OUT/'full_coverage.json',dict(passed=True,counts=nodecounts,training_runs=0,test_read=False,all_windows_processed=True,checks=checks))
 print(pd.DataFrame(summary).to_string(index=False),flush=True)
if __name__=='__main__':audit()
