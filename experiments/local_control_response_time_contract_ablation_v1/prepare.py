"""Freeze causal input transformation before inference; no future target columns."""
from common import *
from scipy.spatial import cKDTree

def nearest_other(tree,flight,points,receiver):
 result=np.full(len(points),np.nan);todo=np.arange(len(points));k=min(8,len(flight))
 while len(todo):
  distance,index=tree.query(points[todo],k=k,workers=1);distance=np.atleast_2d(distance);index=np.atleast_2d(index);eligible=flight[index]!=receiver;good=eligible.any(1);first=np.argmax(eligible,axis=1);result[todo[good]]=distance[np.flatnonzero(good),first[good]];todo=todo[~good]
  if len(todo):
   assert k<len(flight);k=min(k*4,len(flight))
 return result

def main():
 cfg=json.loads((HERE/'config.json').read_text());w=pd.read_csv(SWAP/'results/windows.csv');cases=pd.read_csv(SWAP/'results/cases.csv');z=dict(np.load(OLD/'results/native_rollouts.npz'));x=dict(np.load(SWAP/'results/inputs.npz'));manifest=json.loads((DATA/'dataset_manifest.json').read_text());hashes={};prior={}
 for exp in [OLD,TIMING,SWAP]:
  for f in json.loads((exp/'archive_manifest.json').read_text())['files']:
   assert sha(exp/f['path'])==f['sha256'];prior[str((exp/f['path']).relative_to(ROOT))]=f['sha256']
 assert sha(DATA/'dataset_manifest.json')==json.loads((SWAP/'results/source_receipt.json').read_text())['dataset_manifest_sha256']
 cols=['flight_id','split','sampling_group','sample_in_flight','state_measurement_us','state_publication_us','encoder_source_timestamp_us','encoder_publication_us','encoder_measurement_us','frequency_source_timestamp_us','frequency_publication_us','rate_measurement_us','rate_publication_us','flap_frequency_hz','encoder_total_count','valid_history_state','valid_encoder','valid_frequency','relative_flap_phase_sin','relative_flap_phase_cos']+['velocity_ned_m_s_'+a for a in 'xyz']+['angular_velocity_body_rad_s_'+a for a in 'xyz']
 newh=z['history'].copy();rows=[];tables={};rate_tables={};distribution=[]
 for split in ['train','validation']:
  name=f'samples_{split}.parquet';hashes[name]=sha(DATA/name);assert hashes[name]==manifest['artifact_sha256'][name];df=pd.read_parquet(DATA/name,columns=cols);assert set(df.split)=={split};tables[split]=df[df.sampling_group=='main_50hz'].copy()
  name=f'native_vehicle_angular_velocity_{split}.parquet';hashes[name]=sha(DATA/name);assert hashes[name]==manifest['artifact_sha256'][name];df=pd.read_parquet(DATA/name,columns=['flight_id','timestamp_us','timestamp_sample_us']);rate_tables[split]={f:g for f,g in df[df.flight_id.isin(w[w.split==split].flight_id)].groupby('flight_id')}
 groups={sp:{f:g for f,g in frame.groupby('flight_id',sort=False)} for sp,frame in tables.items()}
 for r in w.itertuples():
  i=r.rollout_index;frame=groups[r.split][r.flight_id];hist=frame[frame.sample_in_flight.between(r.start_sample,r.origin_sample)].sort_values('sample_in_flight');n=int(z['history_length'][i]);assert len(hist)==n
  np.testing.assert_allclose(hist[['relative_flap_phase_sin','relative_flap_phase_cos','flap_frequency_hz']].to_numpy(),z['history'][i,:n,17:20],atol=2e-7)
  nt=rate_tables[r.split][r.flight_id];tt=nt.timestamp_sample_us.to_numpy();pub=nt.timestamp_us.to_numpy()
  for j,row in enumerate(hist.itertuples()):
   assert row.encoder_publication_us<=row.state_publication_us and row.frequency_publication_us<=row.state_publication_us and row.rate_publication_us<=row.state_publication_us
   age=int(row.state_measurement_us-row.encoder_source_timestamp_us);assert age>=0
   phi=float(np.arctan2(z['history'][i,j,17],z['history'][i,j,18]));freq=float(z['history'][i,j,19]);delta=2*np.pi*freq*age*1e-6;newh[i,j,17:19]=[np.sin(phi+delta),np.cos(phi+delta)]
   available=tt[pub<=row.state_publication_us];left=available[available<=row.state_measurement_us];right=available[available>=row.state_measurement_us];bracket=len(left)>0 and len(right)>0
   rows.append(dict(window_id=r.window_id,flight_id=r.flight_id,split=r.split,history_row=j,is_origin=j==n-1,sample_in_flight=row.sample_in_flight,state_us=row.state_measurement_us,decision_us=row.state_publication_us,encoder_source_us=row.encoder_source_timestamp_us,encoder_available_upper_us=row.encoder_publication_us,frequency_available_upper_us=row.frequency_publication_us,phase_age_us=age,phase_delta_rad=delta,frequency=freq,old_sin=float(z['history'][i,j,17]),old_cos=float(z['history'][i,j,18]),new_sin=float(newh[i,j,17]),new_cos=float(newh[i,j,18]),encoder_count=row.encoder_total_count,rate_measurement_us=row.rate_measurement_us,rate_available_us=row.rate_publication_us,causal_rate_bracket=bool(bracket),rate_age_us=row.state_measurement_us-row.rate_measurement_us))
 rec=pd.DataFrame(rows);rec.to_csv(OUT/'input_changes.csv.gz',index=False)
 # Origin and all historical rows require true brackets; holding an older rate is already V0.
 admission=dict(V0='admitted unchanged',V1_physical='not admitted: physical encoder measurement time not separately logged',V1_source_assumption='admitted conditional diagnostic; source is software post-I2C read anchor, physical delay not known',V2='not admitted: no causal bracket for every state; extrapolation would add an unvalidated high-frequency state model',history_rows=len(rec),history_causal_brackets=int(rec.causal_rate_bracket.sum()),origins=len(w),origin_causal_brackets=int(rec[rec.is_origin].causal_rate_bracket.sum()),phase_delta_training_quantiles=rec.query("split=='train'").phase_delta_rad.quantile([0,.25,.5,.75,1]).tolist(),parameters_chosen_without_prediction_outcomes=True,validation_tuning=False)
 assert not rec.causal_rate_bracket.all();save(OUT/'admission.json',admission)
 xi={k:v.copy() for k,v in x.items()};xi['history']=newh[cases.receiver.to_numpy()];np.savez_compressed(OUT/'V1_inputs.npz',**xi)
 # Marginal and joint feature drift measured against original train-only input representation.
 tr=tables['train'];tr=tr[tr.valid_history_state&tr.valid_encoder&tr.valid_frequency];fcols=cols[-6:]+['flap_frequency_hz','relative_flap_phase_sin','relative_flap_phase_cos'];ref=tr[fcols].to_numpy();scale=np.std(ref,axis=0);scale[-2:]=1;scale=np.maximum(scale,1e-6);refnorm=ref/scale;tree=cKDTree(refnorm);ref_flights=tr.flight_id.to_numpy();lower=ref.min(0);upper=ref.max(0)
 for r in w.itertuples():
  i=r.rollout_index;n=int(z['history_length'][i]);old=np.column_stack([z['history'][i,:n][:,CHANNELS],z['history'][i,:n,19],z['history'][i,:n,17:19]]);new=np.column_stack([newh[i,:n][:,CHANNELS],newh[i,:n,19],newh[i,:n,17:19]])
  distances=nearest_other(tree,ref_flights,np.vstack([old,new])/scale,r.flight_id);a,b=distances[:n],distances[n:];delta=np.linalg.norm(new[:,-2:]-old[:,-2:],axis=1)
  distribution.append(dict(window_id=r.window_id,split=r.split,flight_id=r.flight_id,phase_feature_change_rms=float(np.sqrt(np.mean(delta**2))),old_nn_mean=float(a.mean()),new_nn_mean=float(b.mean()),old_nn_max=float(a.max()),new_nn_max=float(b.max()),new_outside_train_marginal=bool(np.any((new<lower-1e-6)|(new>upper+1e-6)))))
 pd.DataFrame(distribution).to_csv(OUT/'feature_distribution.csv',index=False)
 trainrec=rec.query("split=='train'");counts=trainrec.drop_duplicates(['flight_id','sample_in_flight']).sort_values(['flight_id','sample_in_flight']).groupby('flight_id').encoder_count.diff().dropna();assert (counts>=0).all()
 save(OUT/'input_receipt.json',dict(data_manifest_sha256=sha(DATA/'dataset_manifest.json'),data_sha256=hashes,prior_sha256=prior,config_sha256=sha(HERE/'config.json'),reference_rows=len(tr),reference_flights=tr.flight_id.nunique(),distribution_scale=scale.tolist(),reference_uses_validation=False,train_flights=sorted(tables['train'].flight_id.unique()),validation_flights=sorted(tables['validation'].flight_id.unique()),nonpositive_count_steps=int((counts<=0).sum()),negative_count_steps=int((counts<0).sum()),test_read=False))
 assert not set(tables['train'].flight_id)&set(tables['validation'].flight_id)
 print(json.dumps(admission,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
