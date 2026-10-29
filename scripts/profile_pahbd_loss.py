"""Loss-only trace on saved real-window predictions, separating source and cycle reductions."""
import argparse,sys,inspect,json,os
from pathlib import Path
from types import SimpleNamespace
import torch
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--snapshots',required=True);p.add_argument('--output',required=True);a=p.parse_args()
os.sched_setaffinity(0,{12});torch.set_num_threads(1);torch.cuda.set_device(0);sys.path.insert(0,str(Path(a.source)/'src'))
from system_identification.data.pahbd import NativeWindows,ConfiguredWindows
from system_identification.models.pahbd import Config
from system_identification.training.pahbd import trajectory_loss
src=inspect.getsource(trajectory_loss)
if "record_function('state_loss')" not in src:
    start=src.index('    for e in targets:');end=src.index('    cycle_terms={}',start)
    src=src[:start]+"    with torch.profiler.record_function('state_loss'):\n"+''.join('    '+line+'\n' for line in src[start:end].splitlines())+src[end:]
    start=src.index('    cycle_terms={}');end=src.index('    total=state+',start)
    src=src[:start]+"    with torch.profiler.record_function('cycle_loss'):\n"+''.join('    '+line+'\n' for line in src[start:end].splitlines())+src[end:]
    ns=dict(trajectory_loss.__globals__);exec(src,ns);trajectory_loss=ns['trajectory_loss']
cfg=json.loads(Path('configs/pahbd/default.json').read_text());d=NativeWindows(**cfg['data'],split='train',device='cuda:0');rows=[]
for h in (200000,2000000):
    snap=torch.load(Path(a.snapshots)/f'A_{h}.pt',weights_only=True,map_location='cuda:0')
    i=int(d.windows[d.windows.window_id==snap['window_id']].index[0]);w=ConfiguredWindows(d,[i],Config(**cfg['model']))[0];times=sorted({e.measurement for e in w.targets})
    pred=[SimpleNamespace(x=x,variance=v) for x,v in zip(snap['pred']['x'],snap['pred']['variance'])]
    def loss():return trajectory_loss(pred,w.targets,times,cycles=w.metadata['cycle_pairs'],**cfg['loss'])
    for _ in range(3):loss()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:loss()
    rows.append(dict(horizon=h,stages=[dict(name=e.key,cpu_us=e.cpu_time_total,cuda_us=e.device_time_total) for e in prof.key_averages() if e.key in ('state_loss','cycle_loss')]))
Path(a.output).write_text(json.dumps(rows,indent=2));print(rows)
