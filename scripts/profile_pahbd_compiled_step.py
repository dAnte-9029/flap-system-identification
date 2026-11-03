#!/usr/bin/env python3
"""Bounded real-window profiler: execution counts, not steady speed measurements."""
import argparse,json,os,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--engine',choices=['reference','step'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);os.sched_setaffinity(0,{8,10,12,14});torch.set_num_threads(1)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from system_identification.models.pahbd import PAHBD,Config
from system_identification.models.pahbd.tensor_flow import enable_tensor_engine
from system_identification.data.pahbd import NativeWindows,ConfiguredWindows
from system_identification.training.pahbd import window_loss
cfg=json.loads(Path('configs/pahbd/default.json').read_text());mc=Config(**cfg['model']);data=NativeWindows(**cfg['data'],split='train',device='cuda:0')
i=int(data.windows[(data.windows.horizon_us==200000)&data.windows.shared_periodic_eligible].index[0]);w=ConfiguredWindows(data,[i],mc)[0]
torch.manual_seed(17);m=PAHBD(mc).double().cuda()
if a.engine!='reference':enable_tensor_engine(m,a.engine)
window_loss(m,w,**cfg['loss'])[0].backward();torch.cuda.synchronize();m.zero_grad(set_to_none=True);os.sched_setaffinity(0,{12})
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
 window_loss(m,w,**cfg['loss'])[0].backward();torch.cuda.synchronize()
events=prof.events();ops=prof.key_averages()
result=dict(engine=a.engine,window_id=w.metadata['window_id'],cuda_events=sum(e.device_type==torch.autograd.DeviceType.CUDA for e in events),cpu_events=sum(e.device_type==torch.autograd.DeviceType.CPU for e in events),cuda_event_total_ms=sum(e.device_time_total for e in events if e.device_type==torch.autograd.DeviceType.CUDA)/1000,top_cpu=[dict(name=e.key,count=e.count,self_cpu_ms=e.self_cpu_time_total/1000) for e in sorted(ops,key=lambda x:x.self_cpu_time_total,reverse=True)[:20]])
(a.output/'profile.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
