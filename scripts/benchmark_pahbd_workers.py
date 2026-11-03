#!/usr/bin/env python3
"""Short real train-only update timing; never a formal training queue."""
import argparse,json,os,sys,time
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu',type=int,required=True)
p.add_argument('--sampled-flights',action='store_true');p.add_argument('--prefetch',action='store_true');p.add_argument('--prefetch-cpu',type=int);p.add_argument('--barrier',type=Path);p.add_argument('--worker',default='single')
a=p.parse_args();os.sched_setaffinity(0,{a.cpu});torch.set_num_threads(1);torch.cuda.set_device(0)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from system_identification.models.pahbd import Config,PAHBD
from system_identification.data.pahbd import NativeWindows,ConfiguredWindows
from system_identification.training.pahbd import train_steps
cfg=json.loads(Path('configs/pahbd/default.json').read_text());model_cfg=Config(**cfg['model'])
data=NativeWindows(**cfg['data'],split='train',device='cuda:0')
indices=[int(data.windows[(data.windows.horizon_us==h)&data.windows.shared_periodic_eligible].index[0]) for h in (200000,500000,1000000,2000000)]
windows=ConfiguredWindows(data,data.windows.index[data.windows.shared_periodic_eligible] if a.sampled_flights else indices,model_cfg)
def fresh():
    torch.manual_seed(17);m=PAHBD(model_cfg).double().cuda();return m,torch.optim.Adam(m.parameters(),**cfg['optimizer'])
m,o=fresh();train_steps(m,ConfiguredWindows(data,indices[:1],model_cfg),o,1,cfg['loss']);del m,o
m,o=fresh();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
if a.barrier:
    a.barrier.mkdir(parents=True,exist_ok=True);(a.barrier/a.worker).touch()
    deadline=time.monotonic()+180
    while not (a.barrier/'GO').exists():
        if time.monotonic()>deadline:raise TimeoutError('benchmark barrier')
        time.sleep(.1)
start=time.perf_counter();cpu=time.process_time()
records=train_steps(m,windows,o,8,cfg['loss'],prefetch=a.prefetch,prefetch_cpu=a.prefetch_cpu)
torch.cuda.synchronize();wall=time.perf_counter()-start
out=dict(wall_seconds=wall,cpu_seconds=time.process_time()-cpu,steps=8,records=records,
         prefetch=a.prefetch,sampled_flights=a.sampled_flights,cpu=a.cpu,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),peak_bytes=torch.cuda.max_memory_allocated())
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2));torch.save(m.state_dict(),a.output.with_suffix('.pt'))
print(json.dumps({k:v for k,v in out.items() if k!='records'}))
