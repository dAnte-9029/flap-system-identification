"""Frozen V0 regression then only one predeclared conditional phase input variant."""
from common import *
import torch,time
replay=module(OLD/'run.py','original_replay')
def main():
 if (OUT/'V1_predictions.npz').exists():raise FileExistsError('Use a fresh LOCAL_ABLATION_OUTPUT directory; frozen predictions already exist')
 cfg=json.loads((HERE/'config.json').read_text());assert json.loads((OUT/'admission.json').read_text())['parameters_chosen_without_prediction_outcomes'];assert sha(HERE/'config.json')==json.loads((OUT/'input_receipt.json').read_text())['config_sha256']
 x=dict(np.load(SWAP/'results/inputs.npz'));v=dict(np.load(OUT/'V1_inputs.npz'));oldpred=dict(np.load(SWAP/'results/predictions.npz'));checks=[];ck=replay.checkpoint_manifest();records=[];pred={}
 for k in x:
  if k=='history':
   indices=[j for j in range(20) if j not in [17,18]];np.testing.assert_array_equal(x[k][...,indices],v[k][...,indices])
  else:np.testing.assert_array_equal(x[k],v[k])
 for k in replay.KEYS:
  if k not in ['events','event_length']:np.testing.assert_array_equal(v[k][::4],v[k][1::4]);np.testing.assert_array_equal(v[k][2::4],v[k][3::4])
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device=torch.device(cfg['device']);models=[];tick=time.time()
 with torch.inference_mode():
  for record in ck:
   model,name,seed,norm=replay.model_from(record,device);p=replay.predict(model,x,device);error=float(abs(p-oldpred[f'{name}_{seed}']).max());assert error<cfg['factual_atol'];checks.append(dict(model=name,seed=seed,V0_max_error=error,normalization_sha256=norm));models.append((model,name,seed))
  print('Six V0 gates passed',flush=True)
  for model,name,seed in models:
   p=replay.predict(model,v,device);pred[f'{name}_{seed}']=p
   if name=='GRU-NoControl':np.testing.assert_array_equal(p[::4],p[1::4]);np.testing.assert_array_equal(p[2::4],p[3::4])
   small={k:v[k][:8] for k in replay.KEYS};ev,le=replay.canonical_run.refine(small['events'],small['event_length'],4)
   a=replay.predict(model,small,device);np.testing.assert_array_equal(a,replay.predict(model,dict(small,events=ev,event_length=le),device));np.testing.assert_array_equal(a,replay.predict(model,dict(small,truth=np.full((8,1,13),np.nan),future_phase=np.full((8,1),1e9)),device))
   checks.append(dict(model=name,seed=seed,variant='V1_source_assumption',NoControl_future_insensitive=name=='GRU-NoControl',redundant_events_invariant=True,extra_future_truth_phase_excluded=True))
 np.savez_compressed(OUT/'V1_predictions.npz',**pred)
 for r in ck:assert sha(ROOT/r['path'])==r['sha256']
 save(OUT/'inference_checks.json',dict(passed=True,checks=checks,checkpoints=ck,only_input_columns_changed=[17,18],initial_unchanged=True,events_unchanged=True,dt_unchanged=True,norm_unchanged=True,model_unmodified=True,seconds=time.time()-tick,gpu=torch.cuda.get_device_name(device),torch=torch.__version__,test_read=False,trained=False,variant_predictions_sha256=sha(OUT/'V1_predictions.npz')))
 print('Completed one conditional input variant; no V2 inference',flush=True)
if __name__=='__main__':main()
