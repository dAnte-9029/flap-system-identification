from analyze import *
def main():
 # Frozen B/D truths for ALL original main/background members, independent of trained model.
 src=HERE.parent/'local_control_response_timing_slowfast_v1/results/points.csv.gz'
 d=pd.read_csv(src);d=d[(d.variable=='p')&(d.point.astype(str)=='300')&d.supported&(((d.method=='B')&(d.width_s==0))|((d.method=='D')&(d.width_s==.27)))].copy()
 check=d.groupby(['pair_uid','method']).real.agg(['min','max']);assert np.max(check['max']-check['min'])<1e-12
 d=d.drop_duplicates(['pair_uid','method']);coh=pd.read_csv(RES/'frozen_cohorts.csv');pairs=pd.read_csv(RES/'pair_information.csv');pairs=pairs[pairs.variant=='SU_reference500'];out=[]
 for (split,cohort),g in coh.groupby(['split','cohort']):
  for method in ['B','D']:
   q=d[(d.split==split)&(d.method==method)&d.pair_uid.isin(g.pair_uid)]
   out.append(dict(split=split,cohort=cohort,method=method,pairs=len(q),active=int((abs(q.real)>.1).sum()),real_rms=np.sqrt(np.mean(q.real**2))))
 pd.DataFrame(out).to_csv(RES/'observational_background.csv',index=False)
 m=pd.read_csv(RES/'windows.csv.gz');f=pd.read_parquet(RES/'causal_features.parquet');a=dict(np.load(RES/'diagnostic_arrays.npz'));y=a['target'];out=[]
 for split in CFG['partitions']:
  for changed in [0,1]:
   mask=(m.split==split)&m.nonoverlap&(f.mission_changed_500==changed);g=m[mask];w=weights(g)
   for j,ch in enumerate(COORDS):out.append(dict(split=split,mission_changed_last500=changed,coordinate=ch,windows=len(g),flights=g.flight_id.nunique(),future300_change_rms=np.sqrt(w@(y[mask,4+j]**2))))
 pd.DataFrame(out).to_csv(RES/'mission_multi_axis.csv',index=False)
 # Source summary: actual field coverage and software freshness, not physical measurement latency.
 cov=pd.read_csv(RES/'topic_coverage.csv');cov.groupby(['split','topic']).agg(logs=('flight_id','size'),with_topic=('exists','sum'),with_records_inside_flight=('records',lambda x:(x>0).sum()),median_age_us=('age_median_us','median'),mean_fresh50=('fresh_50000_fraction','mean'),mean_fresh100=('fresh_100000_fraction','mean')).reset_index().to_csv(RES/'topic_summary.csv',index=False)
 # Literature local fulltext receipt; full copyrighted text is not committed.
 write_json(HERE/'sources/literature_text_hashes.json',{p.name:old.sha(p) for p in (HERE/'sources').glob('*.local.txt')})
 # Actual lag intervals retained to avoid reporting the registered origin grid as exact 100ms.
 rows=[]
 for s in CFG['partitions']:
  z=np.load(RES/f'gram_{s}_raw_6_command.npz')['lag_age_us']
  for k in range(z.shape[1]):rows.append(dict(split=s,lag_index=k,p10_ms=np.quantile(z[:,k],.1)/1000,median_ms=np.median(z[:,k])/1000,p90_ms=np.quantile(z[:,k],.9)/1000))
 pd.DataFrame(rows).to_csv(RES/'actual_lag_spacing.csv',index=False)
 receipt=json.loads((RES/'upstream_hashes.json').read_text());receipt[str(src.relative_to(ROOT))]=old.sha(src);write_json(RES/'upstream_hashes.json',receipt)
 # Concentration is descriptive; not an independence effective sample size.
 native=pd.read_csv(RES/'native_auto_controls.csv.gz');out=[]
 for split in CFG['partitions']:
  g=m[(m.split==split)&m.nonoverlap];ids=g.index.to_numpy();energy=pd.Series(y[ids,6]**2,index=g.flight_id).groupby(level=0).sum().sort_values(ascending=False);duration=native[native.split==split].groupby('flight_id').dt_us.sum().sort_values(ascending=False)
  out.append(dict(split=split,top5_raw_change_energy_fraction=energy.head(5).sum()/energy.sum(),top5_duration_fraction=duration.head(5).sum()/duration.sum(),dates=g.date.nunique(),firmware=g.firmware.nunique(),configurations=g.controller_id.nunique()))
 pd.DataFrame(out).to_csv(RES/'concentration.csv',index=False)
def pair_command_history():
 p=pd.read_csv(HERE.parent/'structured_control_effectiveness_group_holdout_v1/results/pairs.csv');a=dict(np.load(RES/'frozen_window_diagnostics.npz'));index=dict(zip(a['window_id'],range(len(a['window_id']))));out=[]
 for row in p.itertuples():
  i,j=index[row.window_a],index[row.window_b];x=a['control'][i]-a['control'][j]
  out.append(dict(pair_uid=row.pair_uid,split=row.split,history_d_rms_grid20ms=np.sqrt(np.mean(x[:26,2]**2)),future_d_rms_grid20ms_300=np.sqrt(np.mean(x[25:40,2]**2)),origin_d_difference=x[25,2],future_delta_difference_300=a['target'][i,6]-a['target'][j,6]))
 pd.DataFrame(out).to_csv(RES/'pair_command_history_comparison.csv',index=False)
if __name__=='__main__':
 main()
 pair_command_history()
