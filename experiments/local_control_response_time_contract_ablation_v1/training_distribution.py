"""Supplement coarse reference with the exact union of H500/H1000 training histories."""
from common import *
from prepare import nearest_other
from scipy.spatial import cKDTree

def main():
 manifest=json.loads((DATA/'dataset_manifest.json').read_text());hashes={}
 for name in ['samples_train.parquet','windows_train.parquet']:
  hashes[name]=sha(DATA/name);assert hashes[name]==manifest['artifact_sha256'][name]
 cols=['flight_id','sampling_group','sample_in_flight','valid_core']+['velocity_ned_m_s_'+a for a in 'xyz']+['angular_velocity_body_rad_s_'+a for a in 'xyz']+['flap_frequency_hz','relative_flap_phase_sin','relative_flap_phase_cos']
 samples=pd.read_parquet(DATA/'samples_train.parquet',columns=cols);samples=samples[samples.sampling_group=='main_50hz'];windows=pd.read_parquet(DATA/'windows_train.parquet');windows=windows[(windows.sampling_group=='main_50hz')&(windows.history_us==500000)&(windows.horizon_us==1000000)];assert len(windows)>0;blocks=[]
 for fid,g in windows.groupby('flight_id'):
  s=samples[samples.flight_id==fid].sort_values('sample_in_flight');np.testing.assert_array_equal(s.sample_in_flight,np.arange(len(s)));difference=np.zeros(len(s)+1,int);np.add.at(difference,g.start_sample,1);np.add.at(difference,g.origin_sample+1,-1);used=np.cumsum(difference[:-1])>0;assert s.loc[used,'valid_core'].all();blocks.append(s[used])
 ref=pd.concat(blocks,ignore_index=True);matrix=ref[cols[-9:]].to_numpy();scale=matrix.std(0);scale[-2:]=1;scale=np.maximum(scale,1e-6);tree=cKDTree(matrix/scale);flights=ref.flight_id.to_numpy();x=dict(np.load(SWAP/'results/inputs.npz'));v=dict(np.load(OUT/'V1_inputs.npz'));cases=pd.read_csv(SWAP/'results/cases.csv');rows=[]
 for r in cases.drop_duplicates('receiver').itertuples():
  ci=r.case_id;n=int(x['history_length'][ci]);a=x['history'][ci,:n];b=v['history'][ci,:n];aa=np.column_stack([a[:,CHANNELS],a[:,19],a[:,17:19]]);bb=np.column_stack([b[:,CHANNELS],b[:,19],b[:,17:19]]);dist=nearest_other(tree,flights,np.vstack([aa,bb])/scale,r.receiver_flight)
  rows.append(dict(window_id=r.receiver_window,flight_id=r.receiver_flight,split=r.split,old_nn_mean=float(dist[:n].mean()),new_nn_mean=float(dist[n:].mean()),old_nn_max=float(dist[:n].max()),new_nn_max=float(dist[n:].max()),outside_training_marginal=bool(np.any((bb<matrix.min(0)-1e-6)|(bb>matrix.max(0)+1e-6)))))
 pd.DataFrame(rows).to_csv(OUT/'training_history_distribution.csv',index=False);save(OUT/'training_distribution_receipt.json',dict(sha256=hashes,reference_windows=len(windows),reference_history_rows=len(ref),reference_flights=ref.flight_id.nunique(),scale=scale.tolist(),training_history_union_only=True,test_read=False,validation_used_for_reference=False));print(json.dumps(json.loads((OUT/'training_distribution_receipt.json').read_text()),indent=2))
if __name__=='__main__':main()
