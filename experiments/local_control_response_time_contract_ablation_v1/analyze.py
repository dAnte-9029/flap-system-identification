"""Fixed A/B/D truth, original dependence weights, changed inputs only."""
from common import *

def build():
 cfg=json.loads((HERE/'config.json').read_text());cases=pd.read_csv(SWAP/'results/cases.csv');pairs=pd.read_csv(SWAP/'results/pairs.csv');x=dict(np.load(SWAP/'results/inputs.npz'));z=dict(np.load(OLD/'results/native_rollouts.npz'));native=dict(np.load(TIMING/'results/native_diagnostic_inputs.npz'));pred={'V0':dict(np.load(SWAP/'results/predictions.npz')),'V1_source_assumption':dict(np.load(OUT/'V1_predictions.npz'))};control=pd.read_csv(SWAP/'results/control_differences.csv');control=control[control.time_kind=='grid'].copy();control['time_ms']=control.time_ms.round(6);control=control.set_index(['pair_uid','time_ms']);facts=[];effects=[];windowrows=[]
 for pi,pair in pairs.iterrows():
  ids=np.arange(pi*4,pi*4+4);end=min(.5,cases.support_end_s.iloc[ids].min());slowend=min(end,native[f'rate_t_{pair.a}'][-1],native[f'rate_t_{pair.b}'][-1])-.135;at=np.r_[np.arange(0,501,20)*.001,end,slowend];kinds=['grid']*26+['supported_endpoint','filter_endpoint'];real=[]
  for wi in [pair.a,pair.b]:
   n=int(z['future_count'][wi]);t=z['native_time_us'][wi,:n+1]*1e-6;y=np.vstack([z['truth_initial'][wi],z['truth'][wi,:n]])[:,CHANNELS];a=tc.interp(t,y,at)-y[0];nt=native[f'rate_t_{wi}'];ny=native[f'rate_y_{wi}']-native[f'native_anchor_{wi}'];b=tc.interp(nt,ny,at,.02);dd=tc.boxmean(nt,ny,at,.27,.02);dd[(at<.135-1e-10)|(at+.135>end+1e-10)]=np.nan;real.append({'A':a,'B':b,'D':dd})
  for version,models in pred.items():
   for key,arr in models.items():
    model,seed=key.rsplit('_',1);ys=[];smooth=[]
    for ci in ids:
     n=int(x['valid_steps'][ci]);t=x['native_time_us'][ci,:n+1]*1e-6;y=np.vstack([x['initial'][ci],arr[ci,:n]])[:,CHANNELS].astype(float);y-=y[0].copy();ys.append(tc.interp(t,y,at));smooth.append(tc.boxmean(t,y,at,.27))
     for ti,(tt,kind) in enumerate(zip(at,kinds)):
      if kind!='grid' or round(tt*1000) in cfg['report_times_ms']:
       for j,var in enumerate(VARS):windowrows.append(dict(variant=version,case_id=ci,pair_uid=pair.pair_uid,split=pair.split,cell=cases.cell.iloc[ci],receiver_window=cases.receiver_window.iloc[ci],donor_window=cases.donor_window.iloc[ci],model=model,seed=int(seed),variable=var,time_ms=tt*1000,time_kind=kind,increment=ys[-1][ti,j]))
    aa,ab,ba,bb=ys;ca=aa-ab;cb=ba-bb;ha=aa-ba;hb=ab-bb;c=(ca+cb)/2;h=(ha+hb)/2;f=aa-bb
    for ti,(tt,kind) in enumerate(zip(at,kinds)):
     good=tt<=end+1e-10;tl=str(round(tt*1000)) if kind=='grid' else kind
     crow=control.loc[(pair.pair_uid,round(tt*1000))] if kind=='grid' else None
     if crow is None:
      # Endpoint integrals already audited by previous stage; use exact stored row for raw endpoint, derive filter endpoint from old helper.
      tapes=[sm.event_tape(x['events'][ci],x['event_length'][ci],x['dt'][ci]) for ci in ids];ga=sm.difference(*tapes[0][:2],*tapes[1][:2],round(tt*1e6))[0][2];gb=sm.difference(*tapes[2][:2],*tapes[3][:2],round(tt*1e6))[0][2]
     else:ga,gb=crow.gA,crow.gB
     for j,var in enumerate(VARS):
      base=dict(variant=version,pair_uid=pair.pair_uid,split=pair.split,model=model,seed=int(seed),variable=var,time_ms=tt*1000,time_kind=kind,time_label=tl)
      row=dict(**base,supported=bool(good),real_F=real[0]['A'][ti,j]-real[1]['A'][ti,j] if good else np.nan,gA=ga,gB=gb,g=(ga+gb)/2)
      for k,v in zip(['YAA','YAB','YBA','YBB','CA','CB','HA','HB','C','H','F','interaction'],[aa,ab,ba,bb,ca,cb,ha,hb,c,h,f,ca-cb]):row[k]=v[ti,j] if good else np.nan
      effects.append(row)
      for method in (['A','B','D'] if var in ['p','q','r'] else ['A']):
       jj=j if method=='A' else j-3;r=real[0][method][ti,jj]-real[1][method][ti,jj];p=(smooth[0][ti,j]-smooth[3][ti,j]) if method=='D' else f[ti,j];support=good and np.isfinite(r) and np.isfinite(p)
       facts.append(dict(**base,method=method,supported=bool(support),real=r if support else np.nan,pred=p if support else np.nan,error=p-r if support else np.nan))
 f=pd.DataFrame(facts);e=pd.DataFrame(effects);f.to_csv(OUT/'factual_curves.csv.gz',index=False);e.to_csv(OUT/'internal_curves.csv.gz',index=False);pd.DataFrame(windowrows).to_csv(OUT/'window_points.csv.gz',index=False)
 selected=lambda d:d[(d.time_kind!='grid')|d.time_ms.round(6).isin(cfg['report_times_ms'])].copy()
 f=selected(f);e=selected(e);f.to_csv(OUT/'factual_points.csv.gz',index=False);e.to_csv(OUT/'internal_points.csv.gz',index=False)
 # Explicit V0 primary result gates against immutable old predictions/definitions.
 old=pd.read_csv(SWAP/'results/point_decomposition.csv.gz');base=e[(e.variant=='V0')&(e.time_kind!='filter_endpoint')];join=base.merge(old,on=['pair_uid','model','seed','variable','time_label'],suffixes=('_new','_old'),validate='one_to_one');err=max(float(abs(join[k+'_new']-join[k+'_old']).max()) for k in ['YAA','YAB','YBA','YBB','CA','CB','C','H','F']);assert err<1e-12
 prior=pd.read_csv(TIMING/'results/points.csv.gz');prior=prior[(prior.point=='300')&(((prior.method.isin(['A','B']))&(prior.domain=='full'))|((prior.method=='D')&(prior.width_s==.27)))];pbase=f[(f.variant=='V0')&(f.time_label=='300')&(f.variable.isin(['p','q','r']))];join=pbase.merge(prior,on=['pair_uid','model','seed','variable','method'],suffixes=('_new','_old'),validate='one_to_one');errfact=max(float(abs(join[k+'_new']-join[k+'_old']).max()) for k in ['real','pred']);assert errfact<2e-5,errfact
 save(OUT/'baseline_statistics.json',dict(passed=True,swap_points=len(base),max_internal_error=err,factual_300_rows=len(join),max_factual_300_error=errfact,truth_identical_across_versions=True));return f,e

def aggregate(f,e):
 pairs=pd.read_csv(SWAP/'results/pairs.csv');mem=pd.read_csv(SWAP/'results/cohorts.csv');summ=[];ints=[];groups=[];leave=[]
 for (split,cohort),mm in mem.groupby(['split','cohort']):
  pp=pairs[pairs.pair_uid.isin(mm.pair_uid)];dd=f[f.pair_uid.isin(mm.pair_uid)&f.supported];ee=e[e.pair_uid.isin(mm.pair_uid)&e.supported]
  for frame,isinternal in [(dd,False),(ee,True)]:
   groupkeys=['variant','model','variable','time_label']+([] if isinternal else ['method'])
   for keys,g in frame.groupby(groupkeys,sort=False):
    meta=dict(zip(groupkeys,keys));effective=pp[pp.pair_uid.isin(g.pair_uid)];metric=sm.metrics if isinternal else tm.metric
    for seed in ['pooled']+sorted(g.seed.unique().tolist()):
     sub=g if seed=='pooled' else g[g.seed==seed]
     for weighting in (['pair','flight','component'] if meta['time_label']=='300' else ['flight']):
      wm=stats.weights(effective,weighting);r=dict(split=split,cohort=cohort,**meta,seed=seed,weighting=weighting,frozen_pairs=len(pp),flights=len(set(effective.flight_a)|set(effective.flight_b)),components=len(set(stats.labels(effective).values())),**metric(sub,sub.pair_uid.map(wm)));(ints if isinternal else summ).append(r)
    if meta['time_label']=='300' and meta['variable']=='p' and cohort in ['main','small_pair']:
     graph=stats.labels(pp)
     for field,values in [('flight',set(pp.flight_a)|set(pp.flight_b)),('date',set(pp.date_a)|set(pp.date_b)),('controller',set(pp.controller_a)|set(pp.controller_b)),('component',set(graph.values()))]:
      for value in sorted(values):
       ids=[u for u,v in graph.items() if v==value] if field=='component' else pp.loc[(pp[field+'_a']==value)|(pp[field+'_b']==value),'pair_uid'];sub=g[g.pair_uid.isin(ids)];groups.append(dict(split=split,cohort=cohort,kind='internal' if isinternal else 'factual',**meta,grouping=field,group=value,**metric(sub,np.ones(len(sub)))))
     if cohort=='main' and split=='train':
      for fid in sorted(set(pp.flight_a)|set(pp.flight_b)):
       ep=pp[(pp.flight_a!=fid)&(pp.flight_b!=fid)];sub=g[g.pair_uid.isin(ep.pair_uid)];leave.append(dict(**meta,kind='internal' if isinternal else 'factual',removed_flight=fid,**metric(sub,sub.pair_uid.map(stats.weights(ep,'flight')))))
 pd.DataFrame(summ).to_csv(OUT/'factual_summary.csv',index=False);pd.DataFrame(ints).to_csv(OUT/'internal_summary.csv',index=False);pd.DataFrame(groups).to_csv(OUT/'per_group.csv',index=False);pd.DataFrame(leave).to_csv(OUT/'leave_one_flight_out.csv',index=False)
 keys=['pair_uid','split','model','seed','variable','method','time_label'];a=f[f.variant=='V0'];b=f[f.variant!='V0'];delta=b.merge(a,on=keys,suffixes=('_new','_V0'),validate='one_to_one');np.testing.assert_allclose(delta.real_new,delta.real_V0,equal_nan=True,atol=0);delta['prediction_change']=delta.pred_new-delta.pred_V0;delta['absolute_error_change']=abs(delta.error_new)-abs(delta.error_V0);delta['amplitude_error_change']=abs(abs(delta.pred_new)-abs(delta.real_new))-abs(abs(delta.pred_V0)-abs(delta.real_V0));delta['direction_V0']=(abs(delta.real_V0)>.1)&(delta.pred_V0*delta.real_V0>0);delta['direction_new']=(abs(delta.real_new)>.1)&(delta.pred_new*delta.real_new>0);delta.to_csv(OUT/'paired_factual_changes.csv.gz',index=False)
 keys=['pair_uid','split','model','seed','variable','time_label'];delta=e[e.variant!='V0'].merge(e[e.variant=='V0'],on=keys,suffixes=('_new','_V0'),validate='one_to_one')
 for k in ['CA','CB','C','H','F']:delta[k+'_change']=delta[k+'_new']-delta[k+'_V0']
 delta.to_csv(OUT/'paired_internal_changes.csv.gz',index=False)
 seeds=[]
 for (variant,uid,model,var,method),g in f[(f.time_label=='300')&f.supported].groupby(['variant','pair_uid','model','variable','method']):seeds.append(dict(variant=variant,pair_uid=uid,model=model,variable=var,method=method,pred_min=g.pred.min(),pred_max=g.pred.max(),real=g.real.iloc[0],all_sign_same=bool((g.pred>0).all() or (g.pred<0).all()),correct_seeds=int(((g.real*g.pred>0)&(abs(g.real)>.1)).sum())))
 pd.DataFrame(seeds).to_csv(OUT/'seed_consistency.csv',index=False)
 pd.DataFrame(summ).query("variable=='p' and split=='train' and cohort=='main' and time_label=='300' and weighting=='flight'").to_csv(OUT/'main_p_300.csv',index=False)
 print(pd.read_csv(OUT/'main_p_300.csv')[['variant','model','seed','method','direction_accuracy','scale_ratio','signed_gain','rmse']].to_string(index=False))
if __name__=='__main__':aggregate(*build())
