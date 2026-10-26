"""Read-only extraction of registered train/validation native clocks, rates and phase."""
from common import *
OUT=RESULTS
assert json.loads((OUT/'baseline_check.json').read_text())['passed']
cfg=json.loads((HERE/'config.json').read_text());z=np.load(OLD/'results/native_rollouts.npz');w=pd.read_csv(OLD/'results/windows.csv')
manifest=json.loads((DATA/'dataset_manifest.json').read_text());receipt=json.loads((OLD/'results/input_receipt.json').read_text())
assert sha(DATA/'dataset_manifest.json')==receipt['data']['manifest_sha256']
checks={};payload={};origins=[];clocks=[];phases=[];source_rows=[];frequencies=[]
ratecols=['angular_velocity_body_rad_s_'+c for c in 'xyz'];targetcols=['target_'+c for c in ratecols]
cols=['flight_id','split','state_measurement_us','state_publication_us','rate_measurement_us','rate_publication_us','target_rate_measurement_us','target_rate_publication_us','encoder_measurement_us','encoder_publication_us','encoder_source_timestamp_us','frequency_source_timestamp_us','frequency_publication_us','encoder_total_count','flap_frequency_hz','frequency_measurement_us','valid_core','segment_id']+ratecols+targetcols
for split in ['train','validation']:
 tables={}
 for stem in ['samples','native_vehicle_angular_velocity','native_encoder_count','native_flap_frequency']:
  name=f'{stem}_{split}.parquet';h=sha(DATA/name);assert h==manifest['artifact_sha256'][name];checks[name]=h
  d=pd.read_parquet(DATA/name,columns=cols if stem=='samples' else None)
  if stem=='samples':assert set(d.split)=={split}
  tables[stem]={f:g.reset_index(drop=True) for f,g in d[d.flight_id.isin(w[w.split==split].flight_id)].groupby('flight_id',sort=False)}
 for x in w[w.split==split].itertuples():
  i=x.rollout_index;n=int(z['future_count'][i]);s=tables['samples'][x.flight_id];o=s.iloc[x.origin_sample];future=s.iloc[x.origin_sample:x.end_sample+1]
  assert future.valid_core.all() and future.segment_id.nunique()==1
  np.testing.assert_allclose(o[ratecols].to_numpy(float),z['initial'][i,10:13],atol=1e-7)
  np.testing.assert_array_equal(o[targetcols].to_numpy(float),z['truth_initial'][i,10:13])
  t0=int(o.state_measurement_us);end=t0+int(z['native_time_us'][i,n])
  native=tables['native_vehicle_angular_velocity'][x.flight_id]
  native=native[(native.timestamp_sample_us>=x.support_start_us)&(native.timestamp_sample_us<=x.support_end_us)].copy()
  assert len(native)>2 and np.all(np.diff(native.timestamp_sample_us)>0)
  rt=(native.timestamp_sample_us.to_numpy(np.int64)-t0)*1e-6;ry=native[[f'xyz[{j}]' for j in range(3)]].to_numpy(float)
  at=interp(rt,ry,[0],cfg['max_rate_gap_s'])[0];status='supported' if np.isfinite(at).all() else 'no_native_bracket_or_gap'
  # Exact source identities: the apparent origin mismatch must be actual distinct native observations.
  for role,t,values in [('initial',o.rate_measurement_us,o[ratecols].to_numpy(float)),('target',o.target_rate_measurement_us,o[targetcols].to_numpy(float))]:
   v=native[native.timestamp_sample_us==t][[f'xyz[{j}]' for j in range(3)]].to_numpy(float)
   assert len(v)==1;np.testing.assert_array_equal(v[0],values)
  future_t=(future.state_measurement_us.to_numpy(np.int64)-t0)*1e-6
  sparse_t=(s.iloc[x.start_sample:x.end_sample+1].target_rate_measurement_us.to_numpy(np.int64)-t0)*1e-6
  sparse_y=s.iloc[x.start_sample:x.end_sample+1][targetcols].to_numpy(float)
  assert np.all(np.diff(sparse_t)>0)
  payload[f'rate_t_{i}']=rt;payload[f'rate_y_{i}']=ry;payload[f'sparse_t_{i}']=sparse_t;payload[f'sparse_y_{i}']=sparse_y
  payload[f'native_anchor_{i}']=at
  for j,(ts,yy) in enumerate(zip(native.timestamp_sample_us,ry)):
   source_rows.append(dict(window_id=x.window_id,rollout_index=i,split=split,flight_id=x.flight_id,measurement_us=int(ts),publication_us=int(native.timestamp_us.iloc[j]),p=yy[0],q=yy[1],r=yy[2]))
  origins.append(dict(window_id=x.window_id,rollout_index=i,split=split,flight_id=x.flight_id,status=status,
    state_measurement_us=t0,state_publication_us=int(o.state_publication_us),decision_us=int(x.decision_us),
    initial_rate_measurement_us=int(o.rate_measurement_us),initial_rate_publication_us=int(o.rate_publication_us),
    truth_rate_measurement_us=int(o.target_rate_measurement_us),truth_rate_publication_us=int(o.target_rate_publication_us),
    initial_p=float(z['initial'][i,10]),truth_initial_p=float(z['truth_initial'][i,10]),physical_initial_p=at[0],
    target_minus_initial_p=float(z['truth_initial'][i,10]-z['initial'][i,10]),target_minus_physical_p=float(z['truth_initial'][i,10]-at[0]),
    physical_minus_initial_p=float(at[0]-z['initial'][i,10]),source_separation_us=int(o.target_rate_measurement_us-o.rate_measurement_us),
    native_gap_max_us=int(np.diff(native.timestamp_sample_us).max()),initial_causal=bool(o.rate_publication_us<=o.state_publication_us),
    native_anchor_bracket_left_us=int(native.timestamp_sample_us[native.timestamp_sample_us<=t0].max()),
    native_anchor_bracket_right_us=int(native.timestamp_sample_us[native.timestamp_sample_us>=t0].min())))
  for k,row in enumerate(future.itertuples()):
   clocks.append(dict(window_id=x.window_id,split=split,step=k,model_time_s=future_t[k],position_measurement_us=int(row.state_measurement_us),position_publication_us=int(row.state_publication_us),target_rate_measurement_us=int(row.target_rate_measurement_us),target_rate_publication_us=int(row.target_rate_publication_us)))
  # Relative encoder advance within each window, never cross-flight absolute phase matching.
  enc=tables['native_encoder_count'][x.flight_id];enc=enc[(enc.timestamp_us>=x.support_start_us)&(enc.timestamp_us<=x.support_end_us)]
  et=(enc.timestamp_us.to_numpy(np.int64)-t0)*1e-6
  counts=enc.total_count.to_numpy(float);den=float(manifest['config']['phase']['counts_per_motor_revolution'])*x.ratio
  angle=(counts-counts[0])*2*np.pi/den
  e0=interp(et,angle[:,None],[0],.05)[0,0]
  obs=interp(et,angle[:,None],future_t,.05)[:,0]-e0
  encoder_staleness_phase=e0-(float(o.encoder_total_count)-counts[0])*2*np.pi/den
  last=int(z['history_length'][i])-1;freq=float(z['history'][i,last,19]);phase0=float(np.arctan2(z['history'][i,last,17],z['history'][i,last,18]))
  ff=tables['native_flap_frequency'][x.flight_id]
  available=ff[ff.availability_us<=o.state_publication_us].iloc[-1]
  assert abs(float(available.frequency_hz)-freq)<1e-6
  ft=ff[(ff.timestamp_us>=t0)&(ff.timestamp_us<=end)]
  frequencies.append(dict(window_id=x.window_id,split=split,last_input_frequency_hz=freq,future_min_hz=float(ft.frequency_hz.min()),future_max_hz=float(ft.frequency_hz.max()),future_mean_hz=float(ft.frequency_hz.mean()),future_count=len(ft),separate_encoder_measurement_timestamp_known=bool(o.encoder_measurement_us>0),separate_frequency_measurement_timestamp_known=bool(o.frequency_measurement_us>0)))
  # Model feature uses START of each step; state output is END of that step.
  for k,t in enumerate(future_t):
   phases.append(dict(window_id=x.window_id,split=split,flight_id=x.flight_id,time_s=t,last_frequency_hz=freq,
    observed_relative_phase=obs[k],predicted_relative_phase=2*np.pi*freq*t,relative_phase_error=2*np.pi*freq*t-obs[k],
    model_step_feature_relative_phase=2*np.pi*freq*future_t[max(k-1,0)],last_phase_rad=phase0,
    encoder_staleness_phase_rad=encoder_staleness_phase,total_phase_error=2*np.pi*freq*t-obs[k]-encoder_staleness_phase,encoder_source_age_at_origin_us=t0-int(o.encoder_source_timestamp_us),frequency_source_age_at_origin_us=t0-int(o.frequency_source_timestamp_us),encoder_measurement_time_known=bool(o.encoder_measurement_us>0),frequency_measurement_time_known=bool(o.frequency_measurement_us>0)))
np.savez_compressed(OUT/'native_diagnostic_inputs.npz',**payload)
pd.DataFrame(origins).to_csv(OUT/'origin_clock_audit.csv',index=False);pd.DataFrame(clocks).to_csv(OUT/'future_clock_audit.csv',index=False)
pd.DataFrame(frequencies).to_csv(OUT/'frequency_audit.csv',index=False)
pd.DataFrame(phases).to_csv(OUT/'phase_forecast_audit.csv',index=False);pd.DataFrame(source_rows).to_csv(OUT/'native_rate_extract.csv.gz',index=False)
files=['src/system_identification/data/velocity_rate_dataset.py','src/system_identification/models/velocity_rate_gru.py','experiments/local_control_response_scope_v1/config.json','experiments/local_control_response_rollout_v1/archive_manifest.json']
save(OUT/'source_receipt.json',dict(dataset_root=str(DATA),manifest_sha256=sha(DATA/'dataset_manifest.json'),artifact_sha256=checks,code_sha256={s:sha(ROOT/s) for s in files},test_read=False,partitions=['train','validation'],native_parquet_extraction_reason='Saved arrays do not contain complete native rate/encoder timestamps; native caches enable fixed physical-time interpolation without GPU or raw logs',input_keys_hashes={k:__import__('hashlib').sha256(z[k].tobytes()).hexdigest() for k in ['history','history_mask','history_dt','history_length','initial','dt','events','event_length']},outputs_sha256={n:sha(OUT/n) for n in ['native_diagnostic_inputs.npz','origin_clock_audit.csv','future_clock_audit.csv','phase_forecast_audit.csv','frequency_audit.csv','native_rate_extract.csv.gz']}))
print('Extracted',len(w),'fixed windows; origin status',pd.DataFrame(origins).status.value_counts().to_dict())
