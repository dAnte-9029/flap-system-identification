"""Input-only training reference and clock diagnostics; never changes membership."""
from common import *
replay=module(OLD/'run.py','support_replay');audit=replay.response.audit

def main():
 cfg=json.loads((HERE/'config.json').read_text());receipt={};unavailable=[];manifest=json.loads((DATA/'dataset_manifest.json').read_text())
 for f in json.loads((SCOPE/'archive_manifest.json').read_text())['files']:
  path=SCOPE/f['path']
  if not path.exists():unavailable.append(f['path']);continue
  assert sha(path)==f['sha256'];receipt[str(path.relative_to(ROOT))]=f['sha256']
 def read(name,columns=None):
  assert 'test' not in name;h=sha(DATA/name);assert h==manifest['artifact_sha256'][name];receipt[name]=h;return pd.read_parquet(DATA/name,columns=columns)
 meta=pd.read_csv(SCOPE/'results/train/candidate_index.csv.gz');ev=pd.read_csv(SCOPE/'results/train/command_events.csv.gz');ref=meta[meta.AUTO_stable].copy();ref=ref.merge(ev,on='candidate_id',validate='one_to_one')
 motors=read('native_actuator_motors_train.parquet');servos=read('native_actuator_servos_train.parquet');mg=dict(tuple(motors.groupby('flight_id')));sg=dict(tuple(servos.groupby('flight_id')));tapes={f:audit.tape(mg[f],sg[f])[:2] for f in ref.flight_id.unique()}
 rawcols=['raw_'+c+'_mean' for c in ['motor','left_elevon','right_elevon','rudder']];refmeans=np.zeros((len(ref),4));ref=ref.reset_index(drop=True)
 for fid,g in ref.groupby('flight_id'):
  t,u=tapes[fid];refmeans[g.index]=u[np.searchsorted(t,g.decision_us.to_numpy(),side='right')-1]+g[rawcols].to_numpy()
 refcoords=coords(refmeans);desc=ref[cfg['support_descriptor_columns']].to_numpy();cal=np.array(cfg['support_descriptor_calipers']);scales=np.array(cfg['support_command_scales'])
 windows=pd.read_csv(OUT/'windows.csv');cases=pd.read_csv(OUT/'cases.csv');x=dict(np.load(OUT/'inputs.npz'))
 # Build full donor 500ms means for the 225 frozen windows, including validation, from logged commands only.
 for sp in ['train','validation']:
  m=motors if sp=='train' else read('native_actuator_motors_validation.parquet');s=servos if sp=='train' else read('native_actuator_servos_validation.parquet');mm=dict(tuple(m.groupby('flight_id')));ss=dict(tuple(s.groupby('flight_id')))
  for fid in windows[windows.split==sp].flight_id.unique():
   if fid not in tapes:tapes[fid]=audit.tape(mm[fid],ss[fid])[:2]
 donor_mean=[]
 for w in windows.itertuples():
  t,u=tapes[w.flight_id];knots=np.unique(np.r_[w.decision_us,t[(t>w.decision_us)&(t<w.command_end_us)],w.command_end_us]);v=u[np.searchsorted(t,knots[:-1],side='right')-1];donor_mean.append((v*np.diff(knots)[:,None]).sum(0)/500000)
 donor_mean=np.array(donor_mean);donor_coords=coords(donor_mean)
 cols=['velocity_ned_m_s_'+a for a in 'xyz']+['angular_velocity_body_rad_s_'+a for a in 'xyz'];controls=['control_flap_motor_normalized','control_left_elevon_normalized','control_right_elevon_normalized','control_rudder_normalized'];extra=['encoder_publication_us','encoder_measurement_us','encoder_source_timestamp_us','frequency_publication_us','frequency_measurement_us','frequency_source_timestamp_us','rate_publication_us','rate_measurement_us','state_measurement_us','state_publication_us','sample_in_flight']
 fields=['flight_id','split','sampling_group','valid_history_state','valid_control','flap_frequency_hz']+cols+controls+extra
 train=read('samples_train.parquet',fields);trainmain=train[(train.sampling_group=='main_50hz')&train.valid_history_state&train.valid_control];lo=trainmain[cols+['flap_frequency_hz']].min().to_numpy();hi=trainmain[cols+['flap_frequency_hz']].max().to_numpy();ulo=trainmain[controls].min().to_numpy();uhi=trainmain[controls].max().to_numpy();rows=[]
 for r in cases.itertuples():
  w=windows.iloc[r.receiver];donor=windows.iloc[r.donor];near=(abs(desc-w[cfg['support_descriptor_columns']].to_numpy(float))<=cal).all(1)&(ref.hardware_uuid==w.hardware_uuid).to_numpy()&(abs(ref.ratio-w.ratio)<.001).to_numpy()&(~ref.flight_id.isin([w.flight_id,donor.flight_id])).to_numpy();ii=np.flatnonzero(near);u=donor_coords[r.donor];distance=np.max(abs(refcoords[ii]-u)/scales,axis=1) if len(ii) else np.array([]);cmd=x['events'][r.case_id][(np.arange(x['events'].shape[2])[None,:]<x['event_length'][r.case_id,:,None])&(x['events'][r.case_id,:,:,4]>0),:4];h=x['history'][r.case_id,:int(x['history_length'][r.case_id])];state=np.column_stack([h[:,CHANNELS],h[:,19]])
  rows.append(dict(case_id=r.case_id,pair_uid=r.pair_uid,split=r.split,cell=r.cell,receiver_window=r.receiver_window,donor_window=r.donor_window,reference_windows=len(ii),reference_flights=ref.iloc[ii].flight_id.nunique(),nearest_scaled_Linf=float(distance.min()) if len(ii) else np.nan,conditional_mean_outside_envelope=bool(np.any((u<refcoords[ii].min(0))|(u>refcoords[ii].max(0)))) if len(ii) else None,conditional_warning=bool(distance.min()>1) if len(ii) else None,conditional_unknown=not len(ii),control_outside_train_marginal=bool(np.any((cmd<ulo-1e-6)|(cmd>uhi+1e-6))),control_outside_physical_range=bool(np.any((cmd[:,0]<0)|(cmd[:,0]>1)|(abs(cmd[:,1:])>1).any(1))),history_outside_train_marginal=bool(np.any((state<lo-1e-6)|(state>hi+1e-6))),donor_motor_mean=u[0],donor_common_mean=u[1],donor_differential_mean=u[2],donor_rudder_mean=u[3]))
 pd.DataFrame(rows).to_csv(OUT/'swap_support.csv',index=False);ref[['candidate_id','window_id','flight_id']].assign(**{f'future_mean_{k}':refcoords[:,i] for i,k in enumerate(['motor','common','differential','rudder'])}).to_csv(OUT/'training_reference.csv.gz',index=False)
 # Clock extraction uses causal source columns only. No future phase/target columns loaded.
 clock=[]
 for split in ['train','validation']:
  frame=train if split=='train' else read('samples_validation.parquet',fields);frame=frame[frame.sampling_group=='main_50hz'].set_index(['flight_id','sample_in_flight'],drop=False)
  for w in windows[windows.split==split].itertuples():
   row=frame.loc[(w.flight_id,w.origin_sample)];h=x['history'][cases[cases.receiver==w.rollout_index].case_id.iloc[0],int(x['history_length'][cases[cases.receiver==w.rollout_index].case_id.iloc[0]])-1]
   rec=dict(window_id=w.window_id,split=split,flight_id=w.flight_id,initial_phase=float(np.arctan2(h[17],h[18])),frequency=float(h[19]),decision_us=w.decision_us,position_measurement_us=w.initial_measurement_us)
   for k in extra:
    rec[k]=int(row[k])
   for source in ['encoder','frequency','rate']:
    assert row[source+'_publication_us']<=w.decision_us
    rec[source+'_publication_age_us']=int(w.decision_us-row[source+'_publication_us']);rec[source+'_source_age_at_position_us']=int(w.initial_measurement_us-row[source+'_source_timestamp_us']) if source!='rate' else int(w.initial_measurement_us-row['rate_measurement_us']);rec[source+'_measurement_known']=bool(row[source+'_measurement_us']>=0)
   clock.append(rec)
 clock=pd.DataFrame(clock);oldclock=pd.read_csv(TIMING/'results/origin_clock_audit.csv');clock=clock.merge(oldclock[['window_id','initial_p','truth_initial_p','physical_initial_p','initial_rate_measurement_us','truth_rate_measurement_us','initial_rate_publication_us']],on='window_id',validate='one_to_one');clock.to_csv(OUT/'state_phase_clock_audit.csv',index=False)
 save(OUT/'support_receipt.json',dict(sources_sha256=receipt,unused_scope_archive_files_unavailable=unavailable,train_reference_windows=len(ref),train_reference_flights=ref.flight_id.nunique(),range_columns=cols+['flap_frequency_hz'],training_min=lo.tolist(),training_max=hi.tolist(),command_min=ulo.tolist(),command_max=uhi.tolist(),train_flights=sorted(train.flight_id.unique().tolist()),validation_flights=sorted(windows.query("split=='validation'").flight_id.unique().tolist()),test_read=False,reference_uses_validation=False,conditional_support_is_coarse_not_causal=True))
 assert not set(train.flight_id)&set(windows.query("split=='validation'").flight_id)
 print('\n'.join(line.rstrip() for line in pd.DataFrame(rows).groupby(['split','cell'])[['conditional_unknown','conditional_warning','control_outside_train_marginal','history_outside_train_marginal']].sum().to_string().splitlines()))
if __name__=='__main__':main()
