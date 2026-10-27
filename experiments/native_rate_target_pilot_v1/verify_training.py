from common import *

def main():
 summaries=[];checks=[];indices=np.load(OUT/'holdout_window_indices.npz')
 for fold in range(3):
  a={};ids=indices[f'fold{fold}_train_indices']
  for name in COMBOS:
   folder=HERE/'folds'/f'fold{fold}'/'training'/name/'seed17';s=json.loads((folder/'summary.json').read_text());r=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()];assert len(r)==6 and s['epochs']==6;assert s['updates']==6*int(np.ceil(len(ids)/1024));assert sha(folder/'last.pt')==s['last_sha256'];assert s['trainer_sha256']==sha(HERE/'train.py')
   assert [x['epoch'] for x in r]==list(range(1,7));assert np.isfinite([x['train_window_loss'] for x in r]).all();a[name]=(s,r);summaries.append(s)
  aa,bb=a['M0A'],a['M0B'];assert aa[0]['initial_model_sha256']==bb[0]['initial_model_sha256'];assert aa[0]['normalization_sha256']==bb[0]['normalization_sha256'];assert aa[0]['target_pack_sha256']==bb[0]['target_pack_sha256']
  for x,y in zip(aa[1],bb[1]):
   for k in ['epoch','updates','batch_order_sha256','lr']:assert x[k]==y[k]
   # Same registered order as the frozen W0 pilot.
   old=[json.loads(t) for t in (PREVIOUS/'folds'/f'fold{fold}'/'training/M0W0/seed17/training.jsonl').read_text().splitlines()][x['epoch']-1]
   assert old['batch_order_sha256']==x['batch_order_sha256']
  checks.append(dict(fold=fold,identical_initial_parameters=True,identical_normalization=True,identical_registered_batch_order=True,original_batch_order_reproduced=True,identical_updates=True,epoch=6))
 assert len(summaries)==6 and sum(s['updates'] for s in summaries)==3192
 pd.DataFrame(summaries).to_csv(OUT/'training_summary.csv',index=False);save(OUT/'training_verification.json',dict(passed=True,runs=6,epochs=36,updates=3192,checks=checks,test_read=False))
 print('Verified 6 runs / 36 epochs / 3192 updates')
if __name__=='__main__':main()
