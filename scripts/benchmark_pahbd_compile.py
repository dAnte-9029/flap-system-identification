#!/usr/bin/env python3
"""Fixed real train windows and original Adam; opt-in execution experiment only."""
import argparse,collections,json,os,sys,time
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--engine',choices=['reference','tensor','net','step','block4','block8'],required=True)
p.add_argument('--compile-mode',default='default',choices=['default','reduce-overhead'])
p.add_argument('--model-mode',default='A',choices=['A','B']);p.add_argument('--horizons',nargs='+',type=int,default=[200000,500000,1000000,2000000])
p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=2);p.add_argument('--updates',type=int,default=0)
p.add_argument('--cpu',type=int,default=12);p.add_argument('--warm-cpus',default='8,10,12,14');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
os.sched_setaffinity(0,{a.cpu});torch.set_num_threads(1);torch.cuda.set_device(0)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from system_identification.models.pahbd import PAHBD,Config
from system_identification.models.pahbd.tensor_flow import enable_tensor_engine
from system_identification.data.pahbd import NativeWindows,ConfiguredWindows
from system_identification.training.pahbd import window_loss,FlightBalancedSampler,train_steps
from torch._dynamo.utils import counters,compile_times
cfg=json.loads(Path('configs/pahbd/default.json').read_text());model_cfg=Config(**dict(cfg['model'],mode=a.model_mode))
start=time.perf_counter();data=NativeWindows(**cfg['data'],split='train',device='cuda:0');init_seconds=time.perf_counter()-start
indices=[int(data.windows[(data.windows.horizon_us==h)&data.windows.shared_periodic_eligible].index[0]) for h in a.horizons]
windows=ConfiguredWindows(data,indices,model_cfg)
loaded=[];load_times=[]
for i in range(len(windows)):
 start=time.perf_counter();loaded.append(windows[i]);torch.cuda.synchronize();load_times.append(time.perf_counter()-start)
torch.manual_seed(17);model=PAHBD(model_cfg).double().cuda();opt=torch.optim.Adam(model.parameters(),**cfg['optimizer'])
engine=enable_tensor_engine(model,a.engine,a.compile_mode) if a.engine!='reference' else None
trace=[]
if engine is None:
 original=model.flow.step
 def traced(s,u,dt):trace.append(dt);return original(s,u,dt)
 model.flow.step=traced

def counter_snapshot():return {k:dict(v) for k,v in counters.items()}
def measured(fn):
 torch.cuda.synchronize();start=time.perf_counter();cpu=time.process_time();e1=torch.cuda.Event(enable_timing=True);e2=torch.cuda.Event(enable_timing=True);e1.record()
 value=fn();e2.record();torch.cuda.synchronize()
 return value,dict(wall=time.perf_counter()-start,cpu=time.process_time()-cpu,cuda_elapsed=e1.elapsed_time(e2)/1000)

def compute(w,snapshot=None):
 torch.compiler.cudagraph_mark_step_begin()
 opt.zero_grad(set_to_none=True)
 if engine is None:trace.clear()
 else:engine.trace=[]
 result,forward=measured(lambda:window_loss(model,w,**cfg['loss']))
 loss,parts,belief,pred,times=result
 _,backward=measured(loss.backward)
 if snapshot:
  torch.save(dict(window_id=w.metadata['window_id'],pred={k:torch.stack([getattr(s,k) for s in [belief]+pred]).detach().cpu() for k in ('x','h','f','phi','actuator','variance','phase_quality','phase_covariance')},loss={k:v.detach().cpu() for k,v in parts.items()},grads={k:p.grad.detach().cpu() for k,p in model.named_parameters() if p.grad is not None},schedule=trace.copy() if engine is None else engine.trace.copy()),snapshot)
 return dict(forward=forward,backward=backward)

meta=dict(args=vars(a)|{'output':str(a.output)},torch=torch.__version__,dataset_init_seconds=init_seconds,manifest=data.manifest_sha256,device=torch.cuda.get_device_name(0))
(a.output/'metadata.json').write_text(json.dumps(meta,indent=2));records=[]
try:
 for h,w,load_time in zip(a.horizons,loaded,load_times):
  os.sched_setaffinity(0,set(map(int,a.warm_cpus.split(','))))
  start=time.perf_counter();compute(w);warm=time.perf_counter()-start
  os.sched_setaffinity(0,{a.cpu});before=counter_snapshot();torch.cuda.reset_peak_memory_stats()
  samples=[compute(w,a.output/f'{a.model_mode}_{h}.pt' if i==0 else None) for i in range(a.repeats)]
  row=dict(mode=a.model_mode,horizon=h,engine=a.engine,warmup_seconds=warm,load_seconds=load_time,samples=samples,peak_bytes=torch.cuda.max_memory_allocated(),counters_before=before,counters_after=counter_snapshot())
  records.append(row);(a.output/'timings.json').write_text(json.dumps(records,indent=2));print(json.dumps(row),flush=True)
 # All fixed-input comparisons above preserve the original initialized parameters.
 if a.updates:
  original_loss=train_steps.__globals__['window_loss']
  def marked(*args,**kwargs):
   torch.compiler.cudagraph_mark_step_begin();return original_loss(*args,**kwargs)
  train_steps.__globals__['window_loss']=marked
  sampler=FlightBalancedSampler(windows.index,17)
  start=time.perf_counter();draws=train_steps(model,windows,opt,a.updates,cfg['loss'],sampler=sampler);torch.cuda.synchronize()
  torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state_dict()),a.output/'updates.pt')
  (a.output/'updates.json').write_text(json.dumps(dict(wall_seconds=time.perf_counter()-start,draws=draws),indent=2))
 if engine:
  (a.output/'blocks.json').write_text(json.dumps(dict(segments=engine.segments,calls=engine.calls),indent=2))
 (a.output/'compile_times.txt').write_text(compile_times())
except BaseException as exc:
 (a.output/'FAILED.json').write_text(json.dumps(dict(type=type(exc).__name__,message=str(exc)),indent=2));raise
