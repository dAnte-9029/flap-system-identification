"""Matched-support comparisons and descriptive mechanisms, with no reselection."""
from common import *
from analyze import metric
OUT=RESULTS;cfg=json.loads((HERE/'config.json').read_text())
z=np.load(OLD/'results/native_rollouts.npz');native=np.load(OUT/'native_diagnostic_inputs.npz')
p=pd.read_csv(OLD/'results/pairs.csv');w=pd.read_csv(OLD/'results/windows.csv');mem=pd.read_csv(OLD/'results/cohorts.csv');origin=pd.read_csv(OUT/'origin_clock_audit.csv').set_index('rollout_index')
d=pd.read_csv(OUT/'points.csv.gz');main=set(mem.query("split=='train' and cohort=='main'").pair_uid)
# Deterministic joint endpoint from native timestamps only, avoiding differing endpoint cohorts.
rows=[];pairaudit=[]
for pair in p.itertuples():
 a,b=int(pair.a),int(pair.b);counts=[int(z['future_count'][i]) for i in [a,b]]
 ta,tb=[z['native_time_us'][i,:n+1]*1e-6 for i,n in zip([a,b],counts)]
 end=min(.5,ta[-1],tb[-1],native[f'rate_t_{a}'][-1],native[f'rate_t_{b}'][-1])
 pairaudit.append(dict(pair_uid=pair.pair_uid,split=pair.split,physical_joint_endpoint_s=end,old_endpoint_s=min(.5,ta[-1],tb[-1]),
  old_initial_pair_gap=origin.loc[a,'target_minus_initial_p']-origin.loc[b,'target_minus_initial_p'],
  target_physical_pair_gap=origin.loc[a,'target_minus_physical_p']-origin.loc[b,'target_minus_physical_p'],
  physical_initial_pair_gap=origin.loc[a,'physical_minus_initial_p']-origin.loc[b,'physical_minus_initial_p'],
  maximum_source_separation_us=max(origin.loc[a,'source_separation_us'],origin.loc[b,'source_separation_us'])))
 ya=np.vstack([z['truth_initial'][a,10:],z['truth'][a,:counts[0],10:]]);yb=np.vstack([z['truth_initial'][b,10:],z['truth'][b,:counts[1],10:]])
 ra=interp(ta,ya,[end])[0]-ya[0]-interp(tb,yb,[end])[0]+yb[0]
 rb=interp(native[f'rate_t_{a}'],native[f'rate_y_{a}'],[end])[0]-native[f'native_anchor_{a}']-interp(native[f'rate_t_{b}'],native[f'rate_y_{b}'],[end])[0]+native[f'native_anchor_{b}']
 for model,seed in MODELS:
  pa=np.vstack([z['initial'][a,10:],z[f'{model}_{seed}'][a,:counts[0],10:]]).astype(float);pb=np.vstack([z['initial'][b,10:],z[f'{model}_{seed}'][b,:counts[1],10:]]).astype(float)
  pred=interp(ta,pa,[end])[0]-pa[0]-interp(tb,pb,[end])[0]+pb[0]
  for method,real in [('A',ra),('B',rb)]:
   for j,var in enumerate(VARS):rows.append(dict(pair_uid=pair.pair_uid,split=pair.split,method=method,model=model,seed=seed,variable=var,time_s=end,real=real[j],pred=pred[j],error=pred[j]-real[j]))
e=pd.DataFrame(rows);assert np.isfinite(e[['real','pred']]).all().all();e.to_csv(OUT/'matched_endpoint_points.csv',index=False);audit=pd.DataFrame(pairaudit);audit.to_csv(OUT/'pair_clock_audit.csv',index=False)
records=[]
for (method,model,var),g in e.groupby(['method','model','variable']):
 for (split,cohort),mm in mem.groupby(['split','cohort']):
  pp=p[p.pair_uid.isin(mm.pair_uid)];sub=g[g.pair_uid.isin(mm.pair_uid)]
  for seed in sorted(sub.seed.unique().tolist())+['pooled']:
   q=sub if seed=='pooled' else sub[sub.seed==seed]
   for weight in ['pair','flight','component']:records.append(dict(split=split,cohort=cohort,method=method,model=model,seed=seed,variable=var,weighting=weight,**metric(q,q.pair_uid.map(weights(pp,weight)).to_numpy())))
pd.DataFrame(records).to_csv(OUT/'matched_endpoint_summary.csv',index=False)
# Transition counts retain all original denominator categories, separately for every seed.
trans=[];detail=[]
comparisons=[('A','B','common_300'),('B','D','common_300'),('C','D','common_300'),('A','B','matched_endpoint'),('B','D_cycle','common_300')]
for before,after,domain in comparisons:
 src=e[e.variable=='p'] if domain=='matched_endpoint' else d[(d.variable=='p')&(d.domain=='common_filter')&(d.width_s==cfg['main_width_s'])&(d.point=='300')]
 for (split,cohort),mm in mem.groupby(['split','cohort']):
  qq=src[src.pair_uid.isin(mm.pair_uid)]
  for model in ['GRU-Control','GRU-NoControl','Hold']:
   b=qq[(qq.model==model)&(qq.method==before)].set_index(['pair_uid','seed']);a=qq[(qq.model==model)&(qq.method==after)].set_index(['pair_uid','seed'])
   joint=b[['real','pred']].join(a[['real','pred']],lsuffix='_before',rsuffix='_after',how='inner').reset_index()
   for floor in cfg['direction_floors']:
    joint['active_before']=abs(joint.real_before)>floor;joint['active_after']=abs(joint.real_after)>floor
    joint['correct_before']=(np.sign(joint.real_before)==np.sign(joint.pred_before))&(abs(joint.pred_before)>1e-8)&joint.active_before
    joint['correct_after']=(np.sign(joint.real_after)==np.sign(joint.pred_after))&(abs(joint.pred_after)>1e-8)&joint.active_after
    joint['status']=np.select([~joint.active_before&~joint.active_after,~joint.active_before&joint.active_after,joint.active_before&~joint.active_after,
       joint.correct_before&joint.correct_after,~joint.correct_before&joint.correct_after,joint.correct_before&~joint.correct_after],
       ['near_zero_both','becomes_active','becomes_near_zero','correct_both','wrong_to_correct','correct_to_wrong'],default='wrong_both')
    for seed in sorted(joint.seed.unique().tolist())+['pooled']:
     sub=joint if seed=='pooled' else joint[joint.seed==seed];row=dict(split=split,cohort=cohort,model=model,seed=seed,before=before,after=after,domain=domain,floor=floor,pairs=sub.pair_uid.nunique(),seed_rows=len(sub))
     row.update({name:int(sum(sub.status==name)) for name in ['near_zero_both','becomes_active','becomes_near_zero','correct_both','wrong_to_correct','correct_to_wrong','wrong_both']});trans.append(row)
    if cohort=='main' and floor==.1:
     j=joint.copy();j['split']=split;j['model']=model;j['before']=before;j['after']=after;j['domain']=domain;detail.append(j)
pd.DataFrame(trans).to_csv(OUT/'direction_transitions.csv',index=False);pd.concat(detail).to_csv(OUT/'direction_transition_details.csv',index=False)
# Correlations are descriptive, not explanations of a causal fraction.
associations=[];peak=pd.read_csv(OUT/'peak_curve_metrics.csv.gz')
for model in ['GRU-Control','GRU-NoControl']:
 r=e[(e.pair_uid.isin(main))&(e.variable=='p')&(e.model==model)&(e.method=='A')].copy()
 r['absolute_error']=abs(r.error);r['direction_wrong']=((np.sign(r.real)!=np.sign(r.pred))&(abs(r.real)>.1)).astype(float)
 r['amplitude_error']=abs(r.pred)-abs(r.real)
 j=r.groupby('pair_uid')[['absolute_error','direction_wrong','amplitude_error']].mean().join(audit.set_index('pair_uid'))
 pk=peak[(peak.pair_uid.isin(main))&(peak.model==model)&(peak.variable=='p')&(peak.method=='A')&(peak.width_s==0)].groupby('pair_uid').peak_lag_ms.apply(lambda x:abs(x).mean())
 j=j.join(pk.rename('absolute_peak_lag_ms'))
 for x in ['old_initial_pair_gap','target_physical_pair_gap','physical_initial_pair_gap','maximum_source_separation_us']:
  for y in ['absolute_error','direction_wrong','amplitude_error','absolute_peak_lag_ms']:
   associations.append(dict(model=model,x='abs_'+x,y=y,n=len(j),spearman=j[x].abs().rank().corr(j[y].rank())))
pd.DataFrame(associations).to_csv(OUT/'clock_error_associations.csv',index=False)
# Leave-one-flight-out for slow response at a common 300ms, preserving every fixed pair otherwise.
lo=[];base=d[(d.pair_uid.isin(main))&(d.variable=='p')&(d.width_s==cfg['main_width_s'])&(d.method.isin(['D','D_cycle']))&(d.point=='300')];pp=p[p.pair_uid.isin(main)]
for f in sorted(set(pp.flight_a)|set(pp.flight_b)):
 keep=pp[(pp.flight_a!=f)&(pp.flight_b!=f)]
 for (method,model),g in base[base.pair_uid.isin(keep.pair_uid)].groupby(['method','model']):lo.append(dict(removed_flight=f,method=method,model=model,**metric(g,g.pair_uid.map(weights(keep,'flight')).to_numpy())))
pd.DataFrame(lo).to_csv(OUT/'leave_one_flight_out.csv',index=False)
# Main peak summaries; each pair/seed contributes once, no inferred time-shift correction.
ps=[]
for (sp,co),mm in mem.groupby(['split','cohort']):
 q=peak[(peak.pair_uid.isin(mm.pair_uid))&(peak.variable=='p')]
 for (model,method,width),g in q.groupby(['model','method','width_s']):
  for seed in sorted(g.seed.unique().tolist())+['pooled']:
   a=g if seed=='pooled' else g[g.seed==seed];active=a.peak_active
   ps.append(dict(split=sp,cohort=co,model=model,method=method,width_s=width,seed=seed,pairs=a.pair_uid.nunique(),active_seed_rows=int(active.sum()),peak_same_sign=a.loc[active,'peak_sign_correct'].mean(),
    peak_lag_median_ms=a.loc[active,'peak_lag_ms'].median(),peak_lag_p10_ms=a.loc[active,'peak_lag_ms'].quantile(.1),peak_lag_p90_ms=a.loc[active,'peak_lag_ms'].quantile(.9),
    peak_abs_bias=(abs(a.pred_peak)-abs(a.real_peak)).mean(),real_boundary_fraction=a.real_peak_boundary.mean(),pred_boundary_fraction=a.pred_peak_boundary.mean()))
pd.DataFrame(ps).to_csv(OUT/'peak_summary.csv',index=False)
print('Matched endpoint, transition, clock associations, peak and dependency diagnostics complete')
