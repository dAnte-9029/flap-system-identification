from common import *
import shutil,subprocess,torch

def main():
 cfg=json.loads((HERE/'config.json').read_text());records={};protocol=json.loads((WEIGHTS/'next_stage_protocol.json').read_text());audit=json.loads((TRAIN/'data_audit.json').read_text())
 assert sha(DATA/'dataset_manifest.json')==audit['manifest_sha256']
 meta={}
 for split in ['train','validation']:
  p=CACHE/f'{split}.npz';assert sha(p)==audit['partitions'][split]['cache_sha256'];records[str(p)]=sha(p)
  p=TRAIN/f'{split}_windows.parquet';meta[split]=pd.read_parquet(p);records[str(p)]=sha(p)
 assert len(meta['train'])==107920 and meta['train'].flight_id.nunique()==94 and len(meta['validation'])==36332 and meta['validation'].flight_id.nunique()==27
 # Validate frozen weights against their archived, committed byte hashes.
 manifest=json.loads((WEIGHTS/'results/artifact_manifest.json').read_text())
 for rel in ['weighted_loss.py','weighting.py','next_stage_protocol.json','results/independent_verification.json']+[f'results/fold{k}/{n}' for k in range(3) for n in ['weights.npz','isolation.json']]:
  p=WEIGHTS/rel;assert sha(p)==manifest['files'][rel]['sha256'];records[str(p)]=sha(p)
 for name,h in protocol['holdout_source_sha256'].items():
  p=PILOT/'results'/name;assert sha(p)==h;records[str(p)]=h;shutil.copyfile(p,OUT/name)
 plan=json.loads((OUT/'holdout_protocol.json').read_text());indices=np.load(OUT/'holdout_window_indices.npz');batchchecks=[]
 for fold,expected in zip(plan['folds'],[87239,95627,87750]):
  k=fold['fold'];ids=indices[f'fold{k}_train_indices'];held=indices[f'fold{k}_held_indices'];assert len(ids)==expected
  fit=set(meta['train'].iloc[ids].flight_id);ho=set(meta['train'].iloc[held].flight_id);va=set(meta['validation'].flight_id)
  assert fit==set(fold['train_flights']) and ho==set(fold['held_flights']) and va==set(fold['validation_flights'])
  for a,b in [(fit,ho),(fit,va),(ho,va)]:assert not a&b and not {x.split(':')[0] for x in a}&{x.split(':')[0] for x in b}
  src=HOLDOUT/'folds'/f'fold{k}';dest=HERE/'folds'/f'fold{k}';dest.mkdir(exist_ok=True)
  receipt=json.loads((src/'normalization_receipt.json').read_text());assert receipt['passed'] and not receipt['validation_used'] and not receipt['test_read'];assert receipt['normalization_sha256']==sha(src/'normalization.json');assert receipt['cache_sha256']==audit['partitions']['train']['cache_sha256'];assert receipt['train_indices_sha256']==hashlib.sha256(ids.tobytes()).hexdigest();assert set(receipt['train_flights'])==fit;assert receipt['fit_function_sha256']==sha(ROOT/'scripts/train_velocity_rate_learning.py');assert max(receipt['independent_float64_max_errors'].values())<2e-5
  for name in ['normalization.json','normalization_receipt.json']:shutil.copyfile(src/name,dest/name);records[str(src/name)]=sha(src/name)
  w=np.load(WEIGHTS/f'results/fold{k}/weights.npz');assert np.array_equal(w['registered_index'],ids)
  assert np.array_equal(w['window_id'],meta['train'].iloc[ids].window_id.astype(str)) and np.array_equal(w['flight_id'],meta['train'].iloc[ids].flight_id)
  # Explicit inverse map and random permutations, not implicit local-row assumptions.
  lookup={int(i):float(v) for i,v in zip(w['registered_index'],w['w1'])}
  for epoch in [1,3,6]:
   order=torch.randperm(len(ids),generator=torch.Generator().manual_seed(170000+epoch)).numpy()
   for start in [0,len(order)//2,len(order)-1024]:
    local=order[start:start+1024];registered=ids[local];expected_weight=np.array([lookup[int(i)] for i in registered]);assert np.array_equal(w['w1'][local],expected_weight)
    expected_id=meta['train'].iloc[registered].window_id.astype(str).to_numpy();assert np.array_equal(w['window_id'][local],expected_id)
    batchchecks.append(dict(fold=k,epoch=epoch,start=start,rows=len(local),registered_index_sha256=hashlib.sha256(registered.tobytes()).hexdigest(),weights_sha256=hashlib.sha256(expected_weight.tobytes()).hexdigest(),passed=True))
  save(dest/'weight_alignment.json',dict(fold=k,windows=len(ids),weights_sha256=sha(WEIGHTS/f'results/fold{k}/weights.npz'),registered_index_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),exact_weight_dtype=str(w['w1'].dtype),identity_exact=True,flight_log_boot_disjoint=True))
 for name in ['pairs','cases','windows','cohorts','swap_support','time_support','control_differences']:
  p=SWAP/'results'/f'{name}.csv';shutil.copyfile(p,OUT/p.name);records[str(p)]=sha(p)
 for p in [SWAP/'results/inputs.npz',ABLATION/'results/factual_curves.csv.gz',ABLATION/'results/internal_curves.csv.gz',PILOT/'model.py',ROOT/'scripts/train_velocity_rate_learning.py',ROOT/'scripts/train_velocity_rate_horizons.py',ROOT/'src/system_identification/models/velocity_rate_gru.py',ROOT/'src/system_identification/models/control_events.py']:
  records[str(p)]=sha(p)
 save(OUT/'batch_identity_checks.json',dict(passed=True,checks=batchchecks))
 save(OUT/'gpu_before.json',dict(query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv'],text=True),processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv'],text=True)))
 save(OUT/'source_receipt.json',dict(base_sha=cfg['base_sha'],config_sha256=sha(HERE/'config.json'),files=records,dataset_id='velocity_rate_multistep_v1',dataset_manifest_sha256=audit['manifest_sha256'],partitions=['train','validation'],test_read=False,normalization_reused=True,normalization_independently_verified_in_source=True))
 print('Preparation passed: immutable assets, fold-fit normalization, 27 shuffled real-batch identity checks')
if __name__=='__main__':main()
