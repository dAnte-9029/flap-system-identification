"""Frozen A/B/H1000-GRU evaluation on identical registered native state rows."""
import argparse,importlib.util,json,os,sys,time
from pathlib import Path
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['A','B','C'],required=True);p.add_argument('--cpu',type=int,required=True);a=p.parse_args();os.sched_setaffinity(0,{a.cpu});torch.set_num_threads(1);torch.cuda.set_device(0)
out=Path(__file__).parent;parent=out.parent;repo=Path('/home/zn/flap-system-identification/.worktrees/pahbd-compile-v4');sys.path.insert(0,str(repo/'src'))
from system_identification.training.pahbd import load_checkpoint,window_loss
from system_identification.data.pahbd import NativeWindows,ConfiguredWindows
from system_identification.models.pahbd.tensor_flow import enable_tensor_engine
prep=json.loads((out/'preparation.json').read_text());idx=json.loads((out/'index.json').read_text());summary=json.loads((parent/'summary.json').read_text());data=None
spec=importlib.util.spec_from_file_location('frozen_gru',prep['source_model']);gru=importlib.util.module_from_spec(spec);spec.loader.exec_module(gru)
keys=('history','history_mask','history_dt','history_length','initial','dt','events','event_length')
for seed in [17,29]:
 name=f'{a.variant}_seed{seed}';dest=out/name;dest.mkdir(exist_ok=False);start=time.perf_counter();max_old_difference=0.
 if a.variant=='C':
  torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=True
  checkpoint=next(x['checkpoint'] for x in prep['checkpoints'] if x['seed']==seed);b=torch.load(checkpoint,map_location='cpu',weights_only=False);model=gru.VelocityRateGRU(b['normalization'],control=True,hidden=b['config']['hidden_size'],event_hidden=b['config']['event_hidden_size']).cuda();model.load_state_dict(b['model']);model.eval()
 else:
  checkpoint=next(x['checkpoint'] for x in summary['models'] if x['task']==name);model,_,b=load_checkpoint(checkpoint,'cuda:0');model.eval();enable_tensor_engine(model,'step');cfg=b['metadata']['config']
  if data is None:data=NativeWindows(**cfg['data'],split='validation',device='cuda:0')
  lookup={x:i for i,x in enumerate(data.windows.window_id)};windows=ConfiguredWindows(data,[lookup[x['window_id']] for x in idx],model.config)
 for item in idx:
  i=item['i'];z=np.load(out/f'inputs_{i:03d}.npz')
  if a.variant=='C':
   inp={k:torch.from_numpy(z[k]).cuda() for k in keys}
   with torch.no_grad():prediction=model(inp)[0].double().cpu().numpy()
  else:
   w=windows[i]
   with torch.no_grad():loss,parts,belief,pred,times=window_loss(model,w,**cfg['loss'])
   bytime={round(t*1e6):s for t,s in zip(times,pred)};query=[round(t*1e6) for t in z['times']]
   assert all(t in bytime for t in query),'comparison would require a new integration query'
   prediction=torch.stack([bytime[t].x for t in query]).cpu().numpy()
   local=next(e for e in w.targets if e.source=='vehicle_local_position' and round(e.measurement*1e6)==query[0]);offset=local.value[:3].cpu().numpy()-z['target'][0,:3];prediction[:,:3]-=offset
   old=np.load(parent/f'direction-{name}'/f"window_{item['original_i']:03d}.npz")
   np.testing.assert_array_equal(old['origin'],z['origin_rate'])
   gyro=torch.stack([bytime[round(t*1e6)].x[10:13] for t in old['times']]).cpu().numpy();difference=float(abs(gyro-old['pred']).max());max_old_difference=max(difference,max_old_difference);assert difference<1e-10
  assert np.isfinite(prediction).all();np.savez_compressed(dest/f'window_{i:03d}.npz',pred=prediction)
  if i<1 or (i+1)%25==0:print(name,i+1,len(idx),round(time.perf_counter()-start,1),flush=True)
 (dest/'verification.json').write_text(json.dumps(dict(checkpoint=checkpoint,windows=len(idx),seconds=time.perf_counter()-start,old_pahbd_gyro_max_abs_difference=max_old_difference if a.variant!='C' else None,new_integration_queries=False,precision='float32' if a.variant=='C' else 'float64'),indent=2));print(name,'complete',flush=True)
