#!/usr/bin/env python3
"""Read-only use of a real saved run; bounded updates written to a fresh diagnostic directory."""
import argparse,copy,json,os,sys,time
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--engine',default='reference');p.add_argument('--output',type=Path,required=True);p.add_argument('--split',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);os.sched_setaffinity(0,{12});torch.set_num_threads(1);torch.cuda.set_device(0)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from system_identification.training.pahbd import load_checkpoint,formal_fit,FlightBalancedSampler,window_loss,validation_selection_score
from system_identification.data.pahbd import NativeWindows,ConfiguredWindows
from system_identification.models.pahbd.tensor_flow import enable_tensor_engine
model,opt,bundle=load_checkpoint(a.checkpoint,'cuda:0',optimizer=True)
cfg=bundle['metadata']['config'];meta=copy.deepcopy(bundle['metadata']);meta['execution_diagnostic']=a.engine
train_data=NativeWindows(**cfg['data'],split='train',device='cuda:0')
train=ConfiguredWindows(train_data,train_data.windows.index[train_data.windows.shared_periodic_eligible],model.config)
val_data=NativeWindows(**cfg['data'],split='validation',device='cuda:0');lookup={x:i for i,x in enumerate(val_data.windows.window_id)}
val_ids=[lookup[x] for x in meta['validation_window_ids']];val=ConfiguredWindows(val_data,val_ids,model.config)
meta['phase_contract']=train_data.phase_contract;meta['data']['manifest_sha256']=train_data.manifest_sha256
start_step=bundle['step'];all_draws=[];part_times=[]
for part,count in enumerate([3,2] if a.split else [5]):
 if a.engine!='reference':enable_tensor_engine(model,a.engine)
 sampler=FlightBalancedSampler(train.index,cfg['seed']);sampler.load_state_dict(bundle['training_state']['sampler'])
 next_window=train[sampler.draw()]
 # Warm compile without an Adam update, then formal_fit restores the saved RNG.
 os.sched_setaffinity(0,{8,10,12,14});opt.zero_grad(set_to_none=True);window_loss(model,next_window,**cfg['loss'])[0].backward();opt.zero_grad(set_to_none=True);torch.cuda.synchronize();os.sched_setaffinity(0,{12})
 folder=a.output/f'part{part}'
 def pause():
  path=folder/'training_draws.jsonl'
  return path.exists() and len(path.read_text().splitlines())>=count
 start=time.perf_counter()
 result=formal_fit(model,train,val,opt,cfg['training']['total_steps'],folder,meta,seed=cfg['seed'],
                   validation_interval=cfg['training']['validation_interval'],patience=cfg['training']['patience'],loss_config=cfg['loss'],resume=bundle,fixed_budget=True,pause_requested=pause)
 part_times.append(time.perf_counter()-start);assert result['state']=='paused'
 all_draws.extend(json.loads(x) for x in (folder/'training_draws.jsonl').read_text().splitlines())
 latest=json.loads((folder/'latest.json').read_text())
 model,opt,bundle=load_checkpoint(latest['checkpoint'],'cuda:0',optimizer=True)
assert bundle['step']==start_step+5
# Separate validation diagnostic; no best-model or saved run is altered.
if a.engine!='reference':enable_tensor_engine(model,a.engine)
subset=[];seen=set()
for i,row in val.index.iterrows():
 if row.horizon_us not in seen:subset.append(val[i]);seen.add(row.horizon_us)
selection=validation_selection_score(model,subset,cfg['loss'])
torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),sampler=bundle['training_state']['sampler'],schedule=bundle['training_state']['schedule'],rng=dict(torch=bundle['training_state']['torch_rng'],cuda=bundle['training_state']['cuda_rng'])),a.output/'final.pt')
(a.output/'result.json').write_text(json.dumps(dict(start_step=start_step,end_step=bundle['step'],draws=all_draws,part_seconds=part_times,validation=selection),indent=2))
print('verified saved/reloaded',bundle['step'],'validation',selection['score'],flush=True)
