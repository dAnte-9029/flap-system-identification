from common import *

def main():
 z,n,w,p,coh=read_assets();ident=[]
 for split in ['train','validation']:
  s=pd.read_parquet(DATA/f'samples_{split}.parquet',columns=['flight_id','state_measurement_us','state_publication_us','rate_measurement_us','rate_publication_us','target_rate_measurement_us','target_rate_publication_us']+RATE+TARGET);groups=dict(tuple(s.groupby('flight_id',sort=False)))
  meta=pd.read_parquet(TRAIN/f'{split}_windows.parquet');keys=['flight_id','start_sample','origin_sample','initial_measurement_us','decision_us'];assert not meta.duplicated(keys).any();lookup={tuple(r[k] for k in keys):r for _,r in meta.iterrows()}
  for row in w[w.split==split].itertuples():
   i=row.rollout_index;ss=groups[row.flight_id].reset_index(drop=True);o=ss.iloc[row.origin_sample];count=int(z['future_count'][i]);future=ss.iloc[row.origin_sample+1:row.origin_sample+1+count]
   np.testing.assert_array_equal(o[RATE].to_numpy(float).astype(np.float32),z['initial'][i,10:13]);np.testing.assert_array_equal(o[TARGET].to_numpy(float),z['truth_initial'][i,10:13]);np.testing.assert_array_equal(future[TARGET].to_numpy(float),z['truth'][i,:count,10:13]);np.testing.assert_array_equal(np.r_[0,future.state_measurement_us.to_numpy()-o.state_measurement_us],z['native_time_us'][i,:count+1])
   matched=lookup.get(tuple(getattr(row,k) for k in keys));ident.append(dict(window_id=row.window_id,split=split,flight_id=row.flight_id,log_sha=row.log_sha,registered_H1000_window_id=matched.window_id if matched is not None else None,registered_H1000_index=int(matched.name) if matched is not None else -1,exact_source_identity=True,initial_available_by_decision=bool(o.rate_publication_us<=o.state_publication_us),origin_us=int(o.state_measurement_us),decision_us=int(o.state_publication_us),initial_source_us=int(o.rate_measurement_us),initial_publication_us=int(o.rate_publication_us),truth_initial_source_us=int(o.target_rate_measurement_us),truth_initial_publication_us=int(o.target_rate_publication_us)))
 pd.DataFrame(ident).to_csv(OUT/'identity_clock_audit.csv',index=False)
 d=pd.read_csv(OUT/'pair_decomposition.csv.gz');a=pd.read_csv(OUT/'oracle_points.csv.gz');summary=[]
 for (split,cohort),members in coh.groupby(['split','cohort']):
  pp=p[p.pair_uid.isin(members.pair_uid)];dd=d[(d.time_label=='300')&d.pair_uid.isin(members.pair_uid)&d.supported]
  for (method,var),g in dd.groupby(['method','variable']):
   for weighting in ['pair','flight','component']:
    weights=g.pair_uid.map(stats.weights(pp[pp.pair_uid.isin(g.pair_uid)],weighting)).to_numpy();weights/=weights.sum();avg=lambda v:np.dot(weights,v)
    result=dict(split=split,cohort=cohort,method=method,variable=var,weighting=weighting,pairs=len(g),future_rms=np.sqrt(avg(g.future**2)),origin_rms=np.sqrt(avg(g.origin**2)),increment_rms=np.sqrt(avg(g.increment**2)),bridge_max=abs(g.bridge).max(),cross_moment=avg(g.future*g.origin),future_bias=avg(g.future),origin_bias=avg(g.origin),increment_bias=avg(g.increment),opposing_in_error_fraction=avg((g.future*g.origin)>0),enhancing_in_error_fraction=avg((g.future*g.origin)<0))
    np.testing.assert_allclose(avg(g.increment**2),avg((g.future-g.origin+g.bridge)**2),atol=1e-14);summary.append(result)
 pd.DataFrame(summary).to_csv(OUT/'decomposition_summary.csv',index=False)
 cfg=json.loads((HERE/'config.json').read_text());case=a[(a.time_label=='300')&a.pair_uid.isin(cfg['cases'])];case.to_csv(OUT/'case_oracles_300.csv',index=False)
 models=pd.read_csv(OUT/'model_error_decomposition.csv.gz');models[(models.time_label=='300')&models.pair_uid.isin(cfg['cases'])].to_csv(OUT/'case_models_300.csv',index=False)
 primary=a[(a.time_label=='300')&a.pair_uid.isin(coh.query("cohort=='main' and split=='train'").pair_uid)].copy();primary['direction_active']=abs(primary.real)>.1;primary['wrong']=primary.direction_active&((primary.real*primary.pred<=0)|(abs(primary.pred)<=1e-8));primary['near_zero_prediction']=abs(primary.pred)<=.1;primary.to_csv(OUT/'all_main_oracle_directions.csv',index=False)
 # Candidate increment supervision optimum: B(t)+initial-B(0), conflicting with A by -E_increment.
 full=pd.read_csv(OUT/'full_window_audit.csv.gz');conflict=[]
 for split,g in full.groupby('split'):
  for var in VARS:
   for method,field in [('all_supported_native_steps',f'{var}_increment_mse'),('B300',None),('D300',None)]:
    mse=g[field] if field else (g[f'{var}_{method}_future']-g[f'{var}_origin_error'])**2
    conflict.append(dict(split=split,variable=var,reference=method,windows=int(mse.notna().sum()),flights=g.loc[mse.notna(),'flight_id'].nunique(),ideal_absolute_target_candidate_loss_mean=float(mse.mean()),candidate_target_shift_rms=float(np.sqrt(mse.mean())),absolute_target_unchanged_candidate_zero=False))
 pd.DataFrame(conflict).to_csv(OUT/'candidate_increment_conflict.csv',index=False)
 save(OUT/'candidate_admission.json',dict(train_runs=0,lambda_selected=False,implemented_loss=False,decision='B: first resolve initial/target representation; do not add increment loss under the frozen initial contract',formula='target_increment_equivalent_absolute = native_B(t) + causal_initial - native_B(0)',conflict='differs from absolute target_A by -(E_future-E_origin); existing SO(3) integration uses absolute rate',full_support_source='full_coverage.json',D_centering='offline uses future half-window; admissible as targets only, no history/rollout access',scales='any later experiment must fit p/q/r scales on fold-fit flights only; no scale or lambda fitted here',not_upper_bound=True))
 print(pd.DataFrame(summary).query("split=='train' and cohort=='main' and variable=='p' and weighting=='flight'").to_string(index=False),flush=True)
if __name__=='__main__':main()
