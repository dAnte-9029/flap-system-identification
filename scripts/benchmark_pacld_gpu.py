#!/usr/bin/env python
"""Bounded real native-window performance probe, without changing selection/training."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from system_identification.models.pacld import Config,PACLD
from system_identification.models.pacld.batch import BatchedPACLD
from system_identification.data.pacld import open_windows
from system_identification.data.pacld_batch import compile_windows
from system_identification.training.pacld import window_objective

def main():
 p=argparse.ArgumentParser();p.add_argument('--batch-size',type=int,default=8);p.add_argument('--horizon',type=int,default=200000);p.add_argument('--precision',default='mixed');p.add_argument('--compile',action='store_true');p.add_argument('--device',default='cuda:0');p.add_argument('--repeats',type=int,default=2);p.add_argument('--output',required=True);p.add_argument('--profile',action='store_true');a=p.parse_args()
 torch.set_num_threads(1);torch.manual_seed(17);c=json.load(open('configs/pacld/default.json'));d=open_windows(c,'train')
 rows=d.index[d.index.horizon_us==a.horizon];ids=rows.groupby(['flight_id','sampling_group'],sort=True).head(1).index.tolist()[:a.batch_size]
 if len(ids)<a.batch_size:ids=rows.index.tolist()[:a.batch_size]
 t=time.perf_counter();ws=[d[i] for i in ids];batch=compile_windows(ws,Config(**c['model']));prep=time.perf_counter()-t
 device=torch.device(a.device);dtype=torch.float32 if a.precision=='fp32' else torch.float64
 model=BatchedPACLD(Config(**c['model']),a.precision,a.compile).to(device);batch=batch.to(device,dtype);opt=torch.optim.Adam(model.parameters(),lr=.0003)
 def sync():
  if device.type=='cuda':torch.cuda.synchronize(device)
 records=[]
 for i in range(a.repeats):
  opt.zero_grad(set_to_none=True);sync();t=time.perf_counter()
  r=model(batch,.1);sync();f=time.perf_counter()-t;t=time.perf_counter()
  r['total'].backward();sync();b=time.perf_counter()-t;t=time.perf_counter()
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);opt.step();sync();o=time.perf_counter()-t
  record=dict(iteration=i,forward_seconds=f,backward_seconds=b,optimizer_seconds=o,seconds=f+b+o,windows_per_second=len(ws)/(f+b+o),loss=float(r['total']),grad_norm=float(norm),peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
  records.append(record);print(json.dumps(record),flush=True)
 if a.profile:
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
   opt.zero_grad(set_to_none=True);r=model(batch,.1);r['total'].backward();sync()
  Path(a.output+'.profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=35))
 Path(a.output).write_text(json.dumps(dict(args=vars(a),window_ids=[w.metadata['window_id'] for w in ws],preparation_seconds=prep,records=records),indent=2))
if __name__=='__main__':main()
