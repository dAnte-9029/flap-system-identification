"""Fixed physical-time and centered cycle-scale diagnostics; no model is imported."""
from common import *

def metric(d,w,floor=.1):
 ok=np.isfinite(d.real)&np.isfinite(d.pred);d=d[ok];w=np.asarray(w)[ok]
 if not len(d):return dict(pairs=0)
 w=w/w.sum();r=d.real.to_numpy();p=d.pred.to_numpy();e=p-r;avg=lambda v:float(np.dot(w,v));active=abs(r)>floor
 cond=lambda v,mask:float(np.dot(w[mask],v[mask])/w[mask].sum()) if np.any(mask) else np.nan
 result=dict(pairs=d.pair_uid.nunique(),seed_rows=len(d),active_pairs=d.loc[active,'pair_uid'].nunique(),near_zero_fraction=avg(~active),
   direction_accuracy=cond(((np.sign(p)==np.sign(r))&(abs(p)>1e-8)).astype(float),active),
   real_rms=np.sqrt(avg(r*r)),pred_rms=np.sqrt(avg(p*p)),scale_ratio=np.sqrt(avg(p*p)/avg(r*r)) if avg(r*r)>1e-18 else np.nan,
   signed_gain=avg(p*r)/avg(r*r) if avg(r*r)>1e-18 else np.nan,bias=avg(e),mae=avg(abs(e)),rmse=np.sqrt(avg(e*e)),
   positive_pairs=d.loc[r>floor,'pair_uid'].nunique(),negative_pairs=d.loc[r<-floor,'pair_uid'].nunique(),
   positive_bias=cond(e,r>floor),negative_bias=cond(e,r<-floor),
   positive_amplitude_bias=cond(abs(p)-abs(r),r>floor),negative_amplitude_bias=cond(abs(p)-abs(r),r<-floor))
 for f in [.05,.2]:
  a=abs(r)>f;tag=str(f).replace('.','p')
  result[f'active_pairs_{tag}']=d.loc[a,'pair_uid'].nunique();result[f'near_zero_{tag}']=avg(~a)
  result[f'direction_{tag}']=cond(((np.sign(p)==np.sign(r))&(abs(p)>1e-8)).astype(float),a)
 return result

def main():
 OUT=RESULTS;cfg=json.loads((HERE/'config.json').read_text());z=np.load(OLD/'results/native_rollouts.npz');native=np.load(OUT/'native_diagnostic_inputs.npz')
 p=pd.read_csv(OLD/'results/pairs.csv');w=pd.read_csv(OLD/'results/windows.csv');mem=pd.read_csv(OLD/'results/cohorts.csv')
 assert set(w.split)=={'train','validation'}
 widths=[cfg['main_width_s'],cfg['sensitivity_width_s']];tapes={};support=[];periodicity=[]
 for x in w.itertuples():
  i=x.rollout_index;n=int(z['future_count'][i]);t=z['native_time_us'][i,:n+1]*1e-6;ya=np.vstack([z['truth_initial'][i,10:13],z['truth'][i,:n,10:13]])
  nt=native[f'rate_t_{i}'];ny=native[f'rate_y_{i}'];anchor=native[f'native_anchor_{i}'];end=min(.5,t[-1])
  # Clip with real bracketing samples: no padded points and no extrapolation.
  bt=np.r_[0,nt[(nt>0)&(nt<end)],end];by=interp(nt,ny,bt,cfg['max_rate_gap_s'])
  tapes[i]={'A':(t,ya-ya[0]),'B':(bt,by-anchor),'B_sparse':(native[f'sparse_t_{i}'],native[f'sparse_y_{i}']-interp(native[f'sparse_t_{i}'],native[f'sparse_y_{i}'],[0])[0])}
  for model,seed in MODELS:
   py=np.vstack([z['initial'][i,10:13],z[f'{model}_{seed}'][i,:n,10:13]]).astype(float)
   tapes[i][f'{model}_{seed}']=(t,py-py[0])
  support.append(dict(window_id=x.window_id,split=x.split,physical_origin_supported=bool(np.isfinite(anchor).all()),physical_future_supported=bool(np.isfinite(by).all()),native_samples=len(bt),future_duration_s=end,
   main_valid_center_start_s=widths[0]/2,main_valid_center_end_s=end-widths[0]/2,history_frequency=x.history_frequency,future_cycles=end*x.history_frequency))
  # Descriptive repeatability at one fixed history-derived period over ~1s history+future.
  last=int(z['history_length'][i])-1;f0=float(z['history'][i,last,19]);period=1/f0;ww=widths[0]
  ht=np.cumsum(z['history_dt'][i,:last+1].astype(float));ht-=ht[-1]
  for model,seed in [('Recorded',0)]+MODELS:
   if model=='Recorded':tt,yy=nt,ny
   else:tt=np.r_[ht,t[1:]];yy=np.vstack([z['history'][i,:last+1,10:13],z[f'{model}_{seed}'][i,:n,10:13]])
   lo=max(-.5,tt[0])+ww/2;hi=min(end,tt[-1])-ww/2-period
   grid=np.arange(lo,hi+1e-9,.01)
   fast1=interp(tt,yy,grid)-boxmean(tt,yy,grid,ww)
   fast2=interp(tt,yy,grid+period)-boxmean(tt,yy,grid+period,ww)
   a,b=fast1[:,0],fast2[:,0];valid=np.isfinite(a)&np.isfinite(b)
   corr=np.corrcoef(a[valid],b[valid])[0,1] if valid.sum()>2 and np.std(a[valid])>1e-8 and np.std(b[valid])>1e-8 else np.nan
   periodicity.append(dict(window_id=x.window_id,flight_id=x.flight_id,split=x.split,model=model,seed=seed,period_s=period,overlap_samples=int(valid.sum()),one_period_fast_correlation=corr,fast_rms=float(np.sqrt(np.nanmean(np.r_[a*a,b*b])))))
 pd.DataFrame(support).to_csv(OUT/'time_support.csv',index=False);pd.DataFrame(periodicity).to_csv(OUT/'one_period_repeatability.csv',index=False)
 rows=[];peaks=[];decomp=[]
 for pair in p.itertuples():
  a,b=int(pair.a),int(pair.b);end=min(.5,tapes[a]['A'][0][-1],tapes[b]['A'][0][-1]);delta=lambda typ,ts:interp(*tapes[a][typ],ts)-interp(*tapes[b][typ],ts)
  for width in [0]+widths:
   domain='full' if width==0 else 'common_filter';lo=width/2;hi=end-width/2
   ts=np.unique(np.r_[lo,np.arange(0,.500001,cfg['grid_s']),hi]);ts=ts[(ts>=lo-1e-10)&(ts<=hi+1e-10)]
   truths={method:delta(method,ts) for method in ['A','B','B_sparse']}
   if width:
    for dest,src in [('C','A'),('D','B')]:
     truths[dest]=boxmean(*tapes[a][src],ts,width)-boxmean(*tapes[b][src],ts,width)
     truths[dest+'_cycle']=truths[dest]-truths[dest][0]
    truths['fast_A']=truths['A']-truths['C'];truths['fast_B']=truths['B']-truths['D']
   for model,seed in MODELS:
    key=f'{model}_{seed}';raw=delta(key,ts)
    slow=boxmean(*tapes[a][key],ts,width)-boxmean(*tapes[b][key],ts,width) if width else None
    for method,real in truths.items():
     if method in ['C','D']:pred=slow
     elif method.endswith('_cycle'):pred=slow-slow[0]
     elif method.startswith('fast_'):pred=raw-slow
     else:pred=raw
     for j,var in enumerate(VARS):
      finite=np.isfinite(real[:,j])&np.isfinite(pred[:,j]);rd=real[:,j];pd_=pred[:,j];er=pd_-rd
      for k,tv in enumerate(ts):
       label='endpoint' if abs(tv-hi)<1e-10 else str(int(round(tv*1000)))
       rows.append(dict(pair_uid=pair.pair_uid,pair_id=pair.pair_id,split=pair.split,model=model,seed=seed,variable=var,method=method,width_s=width,domain=domain,time_s=tv,point=label,real=rd[k],pred=pd_[k],error=er[k],supported=bool(finite[k])))
      idx=np.flatnonzero(finite)
      if len(idx)>1:
       ri=idx[np.argmax(abs(rd[idx]))];pi=idx[np.argmax(abs(pd_[idx]))];dur=ts[idx[-1]]-ts[idx[0]]
       peaks.append(dict(pair_uid=pair.pair_uid,split=pair.split,model=model,seed=seed,variable=var,method=method,width_s=width,domain=domain,
        real_peak=rd[ri],pred_peak=pd_[pi],real_peak_time_s=ts[ri],pred_peak_time_s=ts[pi],peak_lag_ms=(ts[pi]-ts[ri])*1000,
        peak_sign_correct=bool(np.sign(rd[ri])==np.sign(pd_[pi]) and abs(rd[ri])>.1 and abs(pd_[pi])>1e-8),peak_active=bool(abs(rd[ri])>.1),
        real_peak_boundary=bool(ri in [0,len(ts)-1]),pred_peak_boundary=bool(pi in [0,len(ts)-1]),
        real_curve_rms=np.sqrt(np.trapz(rd[idx]**2,ts[idx])/dur),pred_curve_rms=np.sqrt(np.trapz(pd_[idx]**2,ts[idx])/dur),curve_rmse=np.sqrt(np.trapz(er[idx]**2,ts[idx])/dur)))
  # Exact additive decomposition at original common endpoint and 300ms.
  for point,tv in [('endpoint',end),('300',.3)]:
   old0=z['truth_initial'][a,10:13]-z['truth_initial'][b,10:13]
   new0=native[f'native_anchor_{a}']-native[f'native_anchor_{b}'];anchor=new0-old0
   ra=delta('A',[tv])[0];rb=delta('B',[tv])[0]
   for j,var in enumerate(VARS):decomp.append(dict(pair_uid=pair.pair_uid,split=pair.split,variable=var,point=point,time_s=tv,A_real=ra[j],B_real=rb[j],origin_correction=anchor[j],future_sampling_correction=(rb-ra+anchor)[j],B_minus_A=(rb-ra)[j]))
 d=pd.DataFrame(rows);d.to_csv(OUT/'responses.csv.gz',index=False);pd.DataFrame(peaks).to_csv(OUT/'peak_curve_metrics.csv.gz',index=False);pd.DataFrame(decomp).to_csv(OUT/'alignment_decomposition.csv',index=False)
 points=d[d.point.isin(['endpoint','200','300'])&~d.method.str.startswith('fast_')];points.to_csv(OUT/'points.csv.gz',index=False)
 aggregate(points,p,mem,OUT)
 print('Done responses',len(d),'points',len(points),'peak curves',len(peaks),flush=True)

def aggregate(points,p,mem,out):
 records=[];group_records=[]
 # Pre-group by estimator/time; memberships are unchanged from upstream.
 groups=points.groupby(['domain','width_s','method','point','variable','model'],sort=False)
 for key,g in groups:
  domain,width,method,point,var,model=key
  for (split,cohort),members in mem.groupby(['split','cohort'],sort=False):
   sub=g[g.pair_uid.isin(members.pair_uid)];pp=p[p.pair_uid.isin(members.pair_uid)]
   for seed in sorted(sub.seed.unique().tolist())+['pooled']:
    q=sub if seed=='pooled' else sub[sub.seed==seed]
    valid=q[np.isfinite(q.real)&np.isfinite(q.pred)];cp=pp[pp.pair_uid.isin(valid.pair_uid)]
    if cp.empty:continue
    for weighting in ['pair','flight','component']:
     wt=q.pair_uid.map(weights(cp,weighting)).to_numpy();v=metric(q,wt)
     records.append(dict(split=split,cohort=cohort,model=model,seed=seed,variable=var,method=method,width_s=width,domain=domain,point=point,weighting=weighting,requested_pairs=len(pp),missing_pairs=len(pp)-len(cp),**v))
    if cohort=='main' and var=='p' and seed=='pooled' and method in ['A','B','C','D','D_cycle'] and point=='300' and width in [0,.27]:
     for grouping in ['flight','date','controller','component']:
      if grouping=='component':sets={str(k):set(cp.loc[cp.pair_uid.map(old.labels(cp))==k,'pair_uid']) for k in set(old.labels(cp).values())}
      else:sets={f:set(cp.loc[(cp[grouping+'_a']==f)|(cp[grouping+'_b']==f),'pair_uid']) for f in set(cp[grouping+'_a'])|set(cp[grouping+'_b'])}
      for name,ids in sets.items():
       a=q[q.pair_uid.isin(ids)];group_records.append(dict(split=split,model=model,method=method,width_s=width,domain=domain,grouping=grouping,group=name,**metric(a,np.ones(len(a)))))
 pd.DataFrame(records).to_csv(out/'summary.csv',index=False);pd.DataFrame(group_records).to_csv(out/'per_group.csv',index=False)

if __name__=='__main__':main()
