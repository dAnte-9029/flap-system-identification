"""Internal factorial effects; observational truth is compared ONLY with factual F."""
from common import *

def metrics(d,w,floor=.1):
 assert d.supported.all();w=np.asarray(w)
 if not len(d):return dict(n=0,pairs=0)
 # Callers pass only supported rows.
 assert len(w)==len(d);w=w/w.sum();avg=lambda x:float(np.dot(w,np.asarray(x,float)))
 a,b,c,h,f=[d[k].to_numpy() for k in ['CA','CB','C','H','F']];active=(abs(a)>floor)&(abs(b)>floor)
 out=dict(n=len(d),pairs=d.pair_uid.nunique(),both_active_n=int(active.sum()),both_active_pairs=d.loc[active,'pair_uid'].nunique(),both_active_fraction=avg(active),both_near_fraction=avg((abs(a)<=floor)&(abs(b)<=floor)),abstention_fraction=avg(~active),same_direction=avg(active&(a*b>0))/avg(active) if avg(active)>0 else np.nan,opposite_fraction=avg(active&(a*b<0)),magnitude_ratio_B_over_A=avg(np.where(active,abs(b)/np.maximum(abs(a),1e-15),0))/avg(active) if avg(active)>0 else np.nan)
 for k in ['CA','CB','C','H','F','interaction']:
  out[k+'_rms']=np.sqrt(avg(d[k]**2));out[k+'_mean_abs']=avg(abs(d[k]));out[k+'_mean']=avg(d[k])
 out.update(C_over_H=out['C_rms']/out['H_rms'] if out['H_rms']>1e-12 else np.nan,control_scale_fraction=out['C_rms']/(out['C_rms']+out['H_rms']) if out['C_rms']+out['H_rms']>1e-12 else np.nan,CH_cross=avg(c*h),CH_opposite_fraction=avg(c*h<0),cancellation=avg(1-abs(f)/np.maximum(abs(c)+abs(h),1e-15)),closure_max=float(abs(d.F-d.C-d.H).max()))
 for response,den in [('CA','gA'),('CB','gB'),('C','g')]:
  ok=abs(d[den])>=.02;out[response+'_gain_valid_n']=int(ok.sum());out[response+'_gain_valid_pairs']=d.loc[ok,'pair_uid'].nunique()
  out[response+'_signed_gain']=avg(np.where(ok,d[response]*d[den],0))/avg(np.where(ok,d[den]**2,0)) if avg(np.where(ok,d[den]**2,0))>1e-15 else np.nan
  out[response+'_gain_mean']=avg(np.where(ok,d[response]/d[den].where(ok,1),0))/avg(ok) if avg(ok)>0 else np.nan
  out[response+'_negative_gain_fraction']=avg(ok&(d[response]*d[den]<0))/avg(ok) if avg(ok)>0 else np.nan
 out['factual_rmse']=np.sqrt(avg((d.F-d.real_F)**2));out['observed_rms']=np.sqrt(avg(d.real_F**2));out['observed_direction_F']=avg((abs(d.real_F)>.1)&(d.real_F*d.F>0))/avg(abs(d.real_F)>.1) if avg(abs(d.real_F)>.1)>0 else np.nan
 out['C_observed_sign_association']=avg((abs(d.real_F)>.1)&(abs(d.C)>.1)&(d.real_F*d.C>0))/avg((abs(d.real_F)>.1)&(abs(d.C)>.1)) if avg((abs(d.real_F)>.1)&(abs(d.C)>.1))>0 else np.nan
 return out

def make_rows():
 cfg=json.loads((HERE/'config.json').read_text());x=dict(np.load(OUT/'inputs.npz'));pred=dict(np.load(OUT/'predictions.npz'));old=dict(np.load(OLD/'results/native_rollouts.npz'));cases=pd.read_csv(OUT/'cases.csv');pairs=pd.read_csv(OUT/'pairs.csv');rows=[];windowrows=[];ctrl=[];support=[]
 tapes=[event_tape(x['events'][i],x['event_length'][i],x['dt'][i]) for i in range(len(cases))]
 for pi,pair in pairs.iterrows():
  ids=np.arange(pi*4,pi*4+4);ends=cases.iloc[ids].support_end_s.to_numpy();end=min(.5,ends.min());at=np.r_[np.arange(0,501,20)*.001,end];kinds=['grid']*26+['supported_endpoint'];valid=at<=end+1e-10
  support.append(dict(pair_uid=pair.pair_uid,split=pair.split,common_end_ms=end*1000,exact500=bool(valid[-2]),main300=bool(end>=.3)))
  controls=[]
  for t,kind,good in zip(at,kinds,valid):
   q=dict(pair_uid=pair.pair_uid,time_ms=t*1000,time_kind=kind,supported=bool(good))
   for label,ia,ib in [('A',ids[0],ids[1]),('B',ids[2],ids[3])]:
    mean,rms,duration=difference(*tapes[ia][:2],*tapes[ib][:2],int(round(t*1e6))) if good and t>0 else (np.zeros(4) if good else np.full(4,np.nan),np.zeros(4) if good else np.full(4,np.nan),0)
    for j,k in enumerate(['motor','common','differential','rudder']):q[f'{label}_{k}_mean']=mean[j];q[f'{label}_{k}_rms']=rms[j]
    q[f'{label}_differential_longest_ms']=duration*.001
   q['gA']=q['A_differential_mean'];q['gB']=q['B_differential_mean'];q['g']=(q['gA']+q['gB'])/2;controls.append(q);ctrl.append(q)
  real=[]
  for wi in [pair.a,pair.b]:
   n=int(old['future_count'][wi]);y=np.vstack([old['truth_initial'][wi],old['truth'][wi,:n]])[:,CHANNELS];real.append(interp(old['native_time_us'][wi,:n+1]*1e-6,y,at)-y[0])
  realF=real[0]-real[1]
  for key,arr in pred.items():
   model,seed=key.rsplit('_',1);ys=[]
   for ci in ids:
    n=int(x['valid_steps'][ci]);y=np.vstack([x['initial'][ci],arr[ci,:n]])[:,CHANNELS].astype(float);ys.append(interp(x['native_time_us'][ci,:n+1]*1e-6,y,at)-y[0])
    for ti,(t,kind) in enumerate(zip(at,kinds)):
     if kind=='supported_endpoint' or round(t*1000) in cfg['report_times_ms']:
      for j,var in enumerate(VARS):windowrows.append(dict(case_id=ci,pair_uid=pair.pair_uid,split=pair.split,cell=cases.cell.iloc[ci],receiver_window=cases.receiver_window.iloc[ci],donor_window=cases.donor_window.iloc[ci],model=model,seed=int(seed),variable=var,time_ms=t*1000,time_kind=kind,supported=bool(t<=ends[ci-ids[0]]+1e-10),increment=ys[-1][ti,j]))
   aa,ab,ba,bb=ys;ca=aa-ab;cb=ba-bb;ha=aa-ba;hb=ab-bb;c=(ca+cb)/2;h=(ha+hb)/2;f=aa-bb
   for ti,(t,kind,good) in enumerate(zip(at,kinds,valid)):
    for j,var in enumerate(VARS):
     row=dict(pair_uid=pair.pair_uid,split=pair.split,model=model,seed=int(seed),variable=var,time_ms=t*1000,time_kind=kind,supported=bool(good),real_F=realF[ti,j] if good else np.nan,gA=controls[ti]['gA'],gB=controls[ti]['gB'],g=controls[ti]['g'])
     for key,v in zip(['YAA','YAB','YBA','YBB','CA','CB','HA','HB','C','H','F','interaction'],[aa,ab,ba,bb,ca,cb,ha,hb,c,h,f,ca-cb]):row[key]=v[ti,j] if good else np.nan
     rows.append(row)
 d=pd.DataFrame(rows);d['time_label']=np.where(d.time_kind=='supported_endpoint','supported_endpoint',d.time_ms.round().astype(int).astype(str));d.to_csv(OUT/'decomposition_curves.csv.gz',index=False)
 points=d[(d.time_kind=='supported_endpoint')|d.time_ms.round(6).isin(cfg['report_times_ms'])];points.to_csv(OUT/'point_decomposition.csv.gz',index=False)
 pd.DataFrame(windowrows).to_csv(OUT/'window_points.csv.gz',index=False);pd.DataFrame(ctrl).to_csv(OUT/'control_differences.csv',index=False);pd.DataFrame(support).to_csv(OUT/'time_support.csv',index=False)
 return points

def aggregate(d):
 pairs=pd.read_csv(OUT/'pairs.csv');mem=pd.read_csv(OUT/'cohorts.csv');summary=[];sensitivity=[];groups=[];leave=[]
 for (split,cohort),mm in mem.groupby(['split','cohort']):
  pp=pairs[pairs.pair_uid.isin(mm.pair_uid)];dd=d[d.pair_uid.isin(mm.pair_uid)&d.supported]
  for (model,var,t),g in dd.groupby(['model','variable','time_label']):
   effective=pp[pp.pair_uid.isin(g.pair_uid)]
   for seed in ['pooled']+sorted(g.seed.unique().tolist()):
    sub=g if seed=='pooled' else g[g.seed==seed]
    for weighting in ['pair','flight','component']:
     ww=sub.pair_uid.map(stats.weights(effective,weighting));meta=dict(split=split,cohort=cohort,model=model,seed=seed,variable=var,time_label=t,weighting=weighting,frozen_pairs=len(pp),flights=len(set(effective.flight_a)|set(effective.flight_b)),components=len(set(stats.labels(effective).values())))
     summary.append(dict(**meta,**metrics(sub,ww)))
     if var=='p' and seed=='pooled':
      for floor in [.05,.1,.2]:sensitivity.append(dict(**meta,floor=floor,**metrics(sub,ww,floor)))
    if var=='p' and t=='300' and cohort in ['main','small_pair']:
     graph=stats.labels(pp)
     for field,values in [('flight',set(pp.flight_a)|set(pp.flight_b)),('date',set(pp.date_a)|set(pp.date_b)),('controller',set(pp.controller_a)|set(pp.controller_b)),('component',set(graph.values()))]:
      for value in sorted(values):
       ids=[u for u,v in graph.items() if v==value] if field=='component' else pp.loc[(pp[field+'_a']==value)|(pp[field+'_b']==value),'pair_uid'];s=sub[sub.pair_uid.isin(ids)];groups.append(dict(split=split,cohort=cohort,model=model,seed=seed,grouping=field,group=value,**metrics(s,np.ones(len(s)))))
     if split=='train' and cohort=='main' and seed=='pooled':
      for fid in sorted(set(pp.flight_a)|set(pp.flight_b)):
       ep=pp[(pp.flight_a!=fid)&(pp.flight_b!=fid)];s=sub[sub.pair_uid.isin(ep.pair_uid)];leave.append(dict(model=model,removed_flight=fid,**metrics(s,s.pair_uid.map(stats.weights(ep,'flight')))))
 pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False);pd.DataFrame(sensitivity).to_csv(OUT/'direction_sensitivity.csv',index=False);pd.DataFrame(groups).to_csv(OUT/'per_group.csv',index=False);pd.DataFrame(leave).to_csv(OUT/'leave_one_flight_out.csv',index=False)
 seedrows=[]
 for (uid,model,var,t),g in d[d.supported].groupby(['pair_uid','model','variable','time_label']):
  for key in ['CA','CB','C','H']:
   v=g[key].to_numpy();seedrows.append(dict(pair_uid=uid,model=model,variable=var,time_label=t,effect=key,n_seeds=len(v),minimum=v.min(),maximum=v.max(),mean=v.mean(),rms=np.sqrt(np.mean(v*v)),all_active=bool(np.all(abs(v)>.1)),all_same_sign=bool(np.all(v>1e-8) or np.all(v< -1e-8)),mixed_sign=bool(np.any(v>1e-8) and np.any(v< -1e-8)),near_zero_seeds=int(sum(abs(v)<=.1))))
 pd.DataFrame(seedrows).to_csv(OUT/'seed_consistency.csv',index=False)
 p=d[(d.variable=='p')&(d.time_label=='300')].copy();p['opposite_active']=(p.CA*p.CB<0)&(abs(p.CA)>.1)&(abs(p.CB)>.1);p['both_near']=(abs(p.CA)<=.1)&(abs(p.CB)<=.1);p['negative_internal_gain']=(abs(p.g)>=.02)&(p.C*p.g<0);p['history_dominates']=abs(p.H)>abs(p.C);p['factual_wrong_direction']=(abs(p.real_F)>.1)&(p.F*p.real_F<0);p.to_csv(OUT/'all_case_diagnostics.csv',index=False)
 # Covariates are frozen metadata or logged controls, not selected using outcomes.
 cp=p.merge(pairs,on=['pair_uid','split']);ctrl=pd.read_csv(OUT/'control_differences.csv');ctrl=ctrl[(ctrl.time_kind=='grid')&(ctrl.time_ms.round(6)==300)];cp=cp.merge(ctrl[['pair_uid','A_differential_rms','A_differential_longest_ms']],on='pair_uid')
 bins={'control_rms':('A_differential_rms',[0,.1,.2,.3,np.inf]),'sustained_ms':('A_differential_longest_ms',[-1,100,200,301]),'history_rate_gap':('rate_rms',[0,.25,.5,.75,np.inf]),'history_attitude_gap':('attitude_rms_deg',[0,5,10,15,np.inf])};rr=[]
 main=set(mem.query("split=='train' and cohort=='main'").pair_uid)
 for field,(col,edges) in bins.items():
  dd=cp[cp.pair_uid.isin(main)&(cp.model=='GRU-Control')].copy();dd['bin']=pd.cut(dd[col],edges,include_lowest=True)
  for label,g in dd.groupby('bin',observed=True):rr.append(dict(covariate=field,bin=str(label),**metrics(g,np.ones(len(g)))))
 pd.DataFrame(rr).to_csv(OUT/'covariate_summary.csv',index=False)
 print(pd.DataFrame(summary).query("split=='train' and cohort=='main' and variable=='p' and time_label=='300' and weighting=='flight'")[['model','seed','CA_rms','CB_rms','C_rms','H_rms','same_direction','both_active_n','both_near_fraction','C_signed_gain','control_scale_fraction']].to_string(index=False))
if __name__=='__main__':aggregate(make_rows())
