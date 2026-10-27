"""Reuse exact archived B nodes and support; no raw-log parsing or new interpolation."""
from common import *
import shutil,subprocess

def main():
 records={};oldreceipt=json.loads((FACTOR/'results/run_receipt.json').read_text());coverage=[]
 for name in ['holdout_protocol.json','holdout_window_indices.npz','holdout_pair_assignments.csv','pairs.csv','cases.csv','cohorts.csv']:
  src=PREVIOUS/'results'/name;shutil.copyfile(src,OUT/name);records[str(src)]=sha(src)
 plan=json.loads((OUT/'holdout_protocol.json').read_text());indices=np.load(OUT/'holdout_window_indices.npz');metas={s:pd.read_parquet(TRAIN/f'{s}_windows.parquet') for s in ['train','validation']}
 for fold,expected in zip(plan['folds'],[87239,95627,87750]):
  k=fold['fold'];ids=indices[f'fold{k}_train_indices'];assert len(ids)==expected
  fit=set(metas['train'].iloc[ids].flight_id);held=set(metas['train'].iloc[indices[f'fold{k}_held_indices']].flight_id);val=set(metas['validation'].flight_id)
  assert fit==set(fold['train_flights']) and held==set(fold['held_flights'])
  for a,b in [(fit,held),(fit,val),(held,val)]:assert not a&b and not {x.split(':')[0] for x in a}&{x.split(':')[0] for x in b}
  src=PREVIOUS/'folds'/f'fold{k}';dest=HERE/'folds'/f'fold{k}';dest.mkdir(parents=True,exist_ok=True)
  receipt=json.loads((src/'normalization_receipt.json').read_text());assert receipt['passed'] and not receipt['validation_used'] and not receipt['test_read'];assert sha(src/'normalization.json')==receipt['normalization_sha256'];assert receipt['train_indices_sha256']==hashlib.sha256(ids.tobytes()).hexdigest()
  for n in ['normalization.json','normalization_receipt.json']:shutil.copyfile(src/n,dest/n);records[str(src/n)]=sha(src/n)
 hist=pd.read_parquet(CAUSAL/'results/history_states.parquet',columns=['split','flight_id','sample_index','B_p','B_q','B_r']);tail=pd.read_csv(FACTOR/'results/restored_B_tail.csv.gz');table=pd.concat([hist,tail],ignore_index=True);assert not table.duplicated(['split','flight_id','sample_index']).any();groups=dict(tuple(table.groupby(['split','flight_id'],sort=False)))
 prior=pd.read_csv(CAUSAL/'results/windows.csv.gz');counts=[]
 for split in ['train','validation']:
  raw=load_numpy(split);meta=metas[split];prev=prior[prior.split==split].set_index('registered_index').loc[meta.index];np.testing.assert_array_equal(meta.window_id,prev.window_id);np.testing.assert_array_equal(meta.flight_id,prev.flight_id)
  k=raw['target'].shape[1];mask=np.arange(k)[None,:]<prev.common_prefix_steps.to_numpy()[:,None];assert (mask.sum(1)>0).all() and np.all(~mask|raw['future_mask']);rb=raw['target'][...,10:13].copy()
  for fid,m in meta.groupby('flight_id',sort=False):
   ids=m.index.to_numpy();node=m.origin_sample.to_numpy()[:,None]+np.arange(1,k+1);hv=groups[(split,fid)].set_index('sample_index');v=hv[['B_p','B_q','B_r']].reindex(node.ravel()).to_numpy().reshape(len(ids),k,3);valid=mask[ids];assert np.isfinite(v[valid]).all();v=np.where(valid[...,None],v,rb[ids]);rb[ids]=v.astype(rb.dtype)
  assert np.isfinite(rb).all();duration=(raw['dt']*mask).sum(1);np.testing.assert_allclose(duration,prev.common_duration_s,atol=2e-7)
  np.savez_compressed(OUT/f'{split}_targets.npz',rate_B=rb,loss_mask=mask,registered_index=meta.index.to_numpy(),window_id=meta.window_id.to_numpy(str),flight_id=meta.flight_id.to_numpy(str))
  scopes=[('all',np.arange(len(meta)))]+([(f'fold{i}_fit',indices[f'fold{i}_train_indices']) for i in range(3)] if split=='train' else [])
  for scope,ids in scopes:
   coverage.append(dict(split=split,scope=scope,windows=len(ids),flights=meta.iloc[ids].flight_id.nunique(),original_steps=int(raw['future_mask'][ids].sum()),supervised_steps=int(mask[ids].sum()),mean_supervision_s=float(duration[ids].mean()),min_supervision_s=float(duration[ids].min()),complete_windows=int((mask[ids].sum(1)==raw['future_mask'][ids].sum(1)).sum()),zero_support=0))
  for fid,m in meta.groupby('flight_id'):ids=m.index.to_numpy();counts.append(dict(split=split,flight_id=fid,windows=len(ids),steps=int(mask[ids].sum()),mean_duration_s=float(duration[ids].mean())))
  records[str(OUT/f'{split}_targets.npz')]=sha(OUT/f'{split}_targets.npz');del raw,rb,mask
 pd.DataFrame(coverage).to_csv(OUT/'target_coverage.csv',index=False);pd.DataFrame(counts).to_csv(OUT/'target_coverage_flights.csv',index=False)
 save(OUT/'source_receipt.json',dict(base_sha='ac5301011dd288e29643bf85f199ce86f146b58d',config_sha256=sha(HERE/'config.json'),files=records,prior_receipt_sha256=sha(FACTOR/'results/run_receipt.json'),previous_data_hashes_reused=oldreceipt['prior_data_hashes_reused'],normalization_reused=True,partitions=['train','validation'],test_read=False))
 save(OUT/'gpu_before.json',dict(query=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv'],text=True)))
 print(pd.DataFrame(coverage).to_string(index=False),flush=True)
if __name__=='__main__':main()
