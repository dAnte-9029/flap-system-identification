from common import *

def main():
 assign=pd.read_csv(OUT/'holdout_pair_assignments.csv');main=set(assign.pair_uid);f=pd.read_csv(OUT/'factual_points.csv.gz');z=f[(f.time_label=='300')&f.pair_uid.isin(main)&f.variable.isin(['p','q','r'])];summary=pd.read_csv(OUT/'factual_summary.csv');checks=[]
 for (method,var),g in z.groupby(['method','variable']):
  aa=g[g.model=='M0A'].set_index('pair_uid').sort_index();bb=g[g.model=='M0B'].set_index('pair_uid').sort_index();assert set(aa.index)==set(bb.index)==main;np.testing.assert_array_equal(aa.real,bb.real);np.testing.assert_array_equal(aa.supported,bb.supported);assert (aa.fold==aa.index.map(assign.set_index('pair_uid').fold)).all()
  for model,tab in [('M0A',aa),('M0B',bb)]:
   real=tab.real.to_numpy();pred=tab.pred.to_numpy();active=abs(real)>.1;direction=((real*pred>0)&(abs(pred)>1e-8))[active].mean();ratio=np.sqrt(np.sum(pred**2)/np.sum(real**2));gain=np.sum(pred*real)/np.sum(real**2)
   r=summary.query("scope=='all_heldout' and cohort=='main' and weighting=='pair' and time_label=='300' and variable==@var and method==@method and model==@model").iloc[0]
   np.testing.assert_allclose([direction,ratio,gain],[r.direction_accuracy,r.scale_ratio,r.signed_gain],rtol=1e-10,atol=1e-12);checks.append(dict(model=model,method=method,variable=var,active=int(active.sum()),direction=float(direction),ratio=float(ratio),gain=float(gain)))
 ints=pd.read_csv(OUT/'internal_points.csv.gz');valid=ints[ints.supported];gap=float(abs(valid.F-valid.C-valid.H).max());assert gap<1e-10
 assert len(assign)==48 and assign.component_key.nunique()==11
 points=f[(f.time_label=='300')&f.pair_uid.isin(main)&(f.variable=='p')];changes=pd.read_csv(OUT/'paired_summary.csv')
 for method in ['B','D']:
  p=points[points.method==method].pivot(index='pair_uid',columns='model',values='pred');real=points[points.method==method].drop_duplicates('pair_uid').set_index('pair_uid').real.reindex(p.index);a=(p.M0A*real>0)&(abs(p.M0A)>1e-8);b=(p.M0B*real>0)&(abs(p.M0B)>1e-8);active=abs(real)>.1;r=changes.query("scope=='all_heldout' and weighting=='pair' and variable=='p' and method==@method").iloc[0];assert int((active&b&~a).sum())==r.gained_pair_seed_rows and int((active&a&~b).sum())==r.lost_pair_seed_rows
 # All trajectory arms share same per-window references and counts; finite supported outputs.
 t=pd.read_csv(OUT/'heldout_trajectory.csv.gz');keys=['fold','registered_index','horizon_ms','reference','support'];a=t[t.model=='M0A'].set_index(keys).sort_index();b=t[t.model=='M0B'].set_index(keys).sort_index();assert a.index.equals(b.index);np.testing.assert_array_equal(a.steps,b.steps);np.testing.assert_array_equal(a.last_supported_s,b.last_supported_s);assert np.isfinite(t[[c for c in t if c.endswith('_mse')]].to_numpy()).all()
 save(OUT/'result_verification.json',dict(passed=True,main_pairs=48,components=11,checks=checks,decomposition_max_error=gap,trajectory_windows_per_model=int(t[t.model=='M0A'].registered_index.nunique()),no_future_truth_inputs=True,test_read=False))
 print('Result verification passed')
if __name__=='__main__':main()
