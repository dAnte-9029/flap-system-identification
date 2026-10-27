"""Independent paired-summary and sign-stratum checks from serialized point values."""
from common import *
from verify import ind

def main():
 a=pd.read_csv(OUT/'component_assignments.csv');pairs=pd.read_csv(OUT/'pairs.csv');points=pd.read_csv(OUT/'factual_points.csv.gz');summ=pd.read_csv(OUT/'paired_summary.csv');mainids=set(a.pair_uid)
 x=points[(points.time_label=='300')&points.pair_uid.isin(mainids)&points.supported].copy();x['arch']=x.model.str[:2];keys=['arch','fold','seed','pair_uid','variable','method'];z=x[x.model.str.endswith('W1')].merge(x[x.model.str.endswith('W0')],on=keys,suffixes=('','_baseline'),validate='one_to_one');groups=dict(tuple(z.groupby(['model','variable','method'])))
 error=0.;checked=0
 for r in summ.itertuples():
  g=groups[(r.model,r.variable,r.method)]
  if r.scope.startswith('fold'):g=g[g.fold==int(r.scope[-1])]
  elif r.scope=='outside_fold0':g=g[g.fold!=0]
  elif r.scope.startswith('C'):g=g[g.pair_uid.isin(a[a.component_id==r.scope].pair_uid)]
  else:assert r.scope=='all_heldout'
  pp=pairs[pairs.pair_uid.isin(g.pair_uid)];w=g.pair_uid.map(ind.weights(pp,r.weighting)).to_numpy();w/=w.sum();real=g.real.to_numpy();before=g.pred_baseline.to_numpy();after=g.pred.to_numpy();active=abs(real)>.1;correct0=(real*before>0)&(abs(before)>1e-8);correct1=(real*after>0)&(abs(after)>1e-8)
  assert int((active&~correct0&correct1).sum())==r.gained_pair_seed_rows and int((active&correct0&~correct1).sum())==r.lost_pair_seed_rows
  dd=np.dot(w[active],(correct1.astype(float)-correct0)[active])/w[active].sum() if active.any() else np.nan
  mae=np.dot(w,abs(after-real)-abs(before-real));amp=np.dot(w,abs(abs(after)-abs(real))-abs(abs(before)-abs(real)))
  for v,target in [(dd,r.direction_difference),(mae,r.mean_absolute_error_change),(amp,r.mean_amplitude_error_change)]:
   if np.isfinite(v):error=max(error,abs(v-target))
   else:assert np.isnan(target)
  checked+=1
 assert error<1e-12
 signs=pd.read_csv(OUT/'sign_comparison.csv');metrics=pd.read_csv(OUT/'factual_summary.csv');metrics=metrics[(metrics.scope=='all_heldout')&(metrics.cohort=='main')&(metrics.time_label=='300')&(metrics.variable=='p')]
 for r in signs.itertuples():
  for suffix,model in [('W0',r.model[:2]+'W0'),('W1',r.model)]:
   row=metrics[(metrics.model==model)&(metrics.method==r.method)&(metrics.weighting==r.weighting)].iloc[0];assert abs(row[r.sign+'_amplitude_bias']-getattr(r,'amplitude_bias_'+suffix))<1e-12
 save(OUT/'independent_paired_verification.json',dict(passed=True,paired_summary_rows=checked,maximum_difference=error,sign_stratum_rows=len(signs),same_structure_pairs=True,test_read=False));print('Independent paired-summary checks passed',checked,error)
if __name__=='__main__':main()
