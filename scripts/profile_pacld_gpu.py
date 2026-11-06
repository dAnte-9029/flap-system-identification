#!/usr/bin/env python
"""Bounded CPU/CUDA operator trace for original and compiled batch execution."""
import argparse,json,sys,time,bisect
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from system_identification.models.pacld import PACLD,Config
from system_identification.models.pacld.types import initial_state
from system_identification.models.pacld.batch import BatchedPACLD
from system_identification.data.pacld import open_windows,move_window
from system_identification.data.pacld_batch import compile_windows
from system_identification.training.pacld import window_objective

def main():
 p=argparse.ArgumentParser();p.add_argument('--reference',action='store_true');p.add_argument('--full-reference',action='store_true',help='profile a whole reference window; large host trace');p.add_argument('--output',required=True);a=p.parse_args()
 torch.set_num_threads(1);torch.manual_seed(17);torch.cuda.set_device(0)
 c=json.load(open('configs/pacld/default.json'));cfg=Config(**c['model']);d=open_windows(c,'train');ids=d.index[d.index.horizon_us==500000].groupby(['flight_id','sampling_group'],sort=True).head(1).index.tolist()[:8]
 if a.reference:
  m=PACLD(cfg).double().cuda();w=move_window(d[ids[0]],'cuda')
  event=sorted([e for e in w.history if e.measurement>=w.start and w.start<=e.availability<=0 and e.availability-e.measurement<=cfg.replay_seconds],key=lambda e:(e.measurement,e.availability,e.source))[0]
  with torch.no_grad():initial=m.flow.propagate(initial_state(w,cfg),w.start,event.measurement,w.controls)
  control=w.controls.values[bisect.bisect_right(w.controls.times,event.measurement)-1]
  def run():
   state=m.belief.update(initial,event,event.measurement-w.start,control)
   for _ in range(10):state=m.flow.step(state,control,.005)
   return state.x.square().sum()+state.z_s.square().sum()+state.c.square().sum()
  if a.full_reference:run=lambda:window_objective(m,w,.1)['total']
 else:
  m=BatchedPACLD(cfg,'mixed',True).cuda();batch=compile_windows([d[i] for i in ids],cfg).to('cuda');run=lambda:m(batch,.1)['total']
 # Warm compiled forward and backward before the trace.
 loss=run();loss.backward();del loss;m.zero_grad(set_to_none=True);torch.cuda.synchronize()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
  with torch.profiler.record_function('pacld/forward'):loss=run()
  with torch.profiler.record_function('pacld/backward'):loss.backward()
  torch.cuda.synchronize()
 stats=prof.key_averages()
 Path(a.output+'.txt').write_text(stats.table(sort_by='self_cuda_time_total',row_limit=45))
 rows=[dict(name=e.key,calls=e.count,cpu_us=e.cpu_time_total,self_cpu_us=e.self_cpu_time_total,device_us=e.device_time_total,self_device_us=e.self_device_time_total) for e in stats]
 Path(a.output+'.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
