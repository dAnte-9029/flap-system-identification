#!/usr/bin/env python
"""Two-rank NCCL parity and throughput using the exact same global native batch."""
import argparse,json,os,sys,time,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from system_identification.models.pacld import Config
from system_identification.models.pacld.batch import BatchedPACLD
from system_identification.data.pacld import open_windows
from system_identification.data.pacld_batch import compile_windows
from system_identification.training.pacld_gpu import BucketSampler,checkpoint_bundle,restore,gather_rng

def grads(m):return torch.cat([p.grad.flatten().double() if p.grad is not None else torch.zeros_like(p).flatten().double() for p in m.parameters()])
def main():
 p=argparse.ArgumentParser();p.add_argument('--batch-size',type=int,default=8);p.add_argument('--horizon',type=int,default=500000);p.add_argument('--compile',action='store_true');p.add_argument('--output',required=True);p.add_argument('--parity',action='store_true');a=p.parse_args()
 rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(1);torch.manual_seed(17);torch.distributed.init_process_group('nccl');world=torch.distributed.get_world_size()
 if a.batch_size%world:raise ValueError('equal shards required')
 c=json.load(open('configs/pacld/default.json'));cfg=Config(**c['model']);d=open_windows(c,'train');rows=d.index[d.index.horizon_us==a.horizon];ids=rows.groupby(['flight_id','sampling_group'],sort=True).head(1).index.tolist()[:a.batch_size]
 if len(ids)<a.batch_size:ids=rows.index.tolist()[:a.batch_size]
 t=time.perf_counter();ws=[d[i] for i in ids[rank*a.batch_size//world:(rank+1)*a.batch_size//world]];batch=compile_windows(ws,cfg).to(f'cuda:{rank}');prep=time.perf_counter()-t
 raw=BatchedPACLD(cfg,'mixed',a.compile).cuda(rank);initial={k:v.detach().cpu().clone() for k,v in raw.state_dict().items()};model=DDP(raw,device_ids=[rank],find_unused_parameters=True);opt=torch.optim.Adam(raw.parameters(),lr=.0003)
 records=[];g=None;loss=None
 for iteration in range(2):
  opt.zero_grad(set_to_none=True);torch.distributed.barrier();torch.cuda.synchronize();t=time.perf_counter();start=t;r=model(batch,.1);torch.cuda.synchronize();f=time.perf_counter()-t;t=time.perf_counter();r['total'].backward();torch.cuda.synchronize();b=time.perf_counter()-t
  if iteration==0 and a.parity:
   g=grads(raw).detach().clone();loss=r['total'].detach().clone();torch.distributed.all_reduce(loss);loss/=world
  norm=torch.nn.utils.clip_grad_norm_(raw.parameters(),10,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
  duration=torch.tensor([time.perf_counter()-start],device=rank,dtype=torch.float64);torch.distributed.all_reduce(duration,op=torch.distributed.ReduceOp.MAX)
  records.append(dict(iteration=iteration,forward_seconds=f,backward_seconds=b,seconds=float(duration),windows_per_second=a.batch_size/float(duration),peak_memory_bytes=torch.cuda.max_memory_allocated(),gradient_norm=float(norm)))
  del r
 parity=None
 if a.parity:
  torch.distributed.barrier()
  if rank==0:
   single=BatchedPACLD(cfg,'mixed').cuda(0);single.load_state_dict(initial);allbatch=compile_windows([d[i] for i in ids],cfg).to('cuda:0');r=single(allbatch,.1);r['total'].backward();sg=grads(single)
   parity=dict(loss_abs=float((loss-r['total']).abs()),gradient_max_abs=float((g-sg).abs().max()),gradient_relative_l2=float((g-sg).norm()/sg.norm()))
   assert parity['loss_abs']<1e-4 and parity['gradient_relative_l2']<1e-4,parity
 if a.parity:
  torch.distributed.barrier()
  sampler=BucketSampler(d.index,17,a.batch_size);sampler.draw();rng=gather_rng()
  saved=checkpoint_bundle(raw,opt,sampler,dict(optimizer_steps=2,windows_seen=2*a.batch_size),c,d.identity(),.1,world,rng)
  if rank==0:torch.save(saved,a.output+'.pt')
  torch.distributed.barrier();next_ids=sampler.draw()
  opt.zero_grad(set_to_none=True);r=model(batch,.1);r['total'].backward();torch.nn.utils.clip_grad_norm_(raw.parameters(),10,error_if_nonfinite=True);opt.step();expected=copy.deepcopy(raw.state_dict());del r
  restore(torch.load(a.output+'.pt',map_location='cpu',weights_only=False),raw,opt,sampler,c,d.identity(),.1,world)
  assert next_ids==sampler.draw()
  opt.zero_grad(set_to_none=True);r=model(batch,.1);r['total'].backward();torch.nn.utils.clip_grad_norm_(raw.parameters(),10,error_if_nonfinite=True);opt.step();del r
  error=max(float((v-raw.state_dict()[k]).abs().max()) for k,v in expected.items());assert error==0,error
  if rank==0:parity['restore_next_update_max_abs']=error;parity['restore_next_indices']=next_ids
 if rank==0:
  result=dict(args=vars(a),world_size=world,preparation_seconds=prep,records=records,parity=parity,global_indices=ids);Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 torch.distributed.barrier();torch.distributed.destroy_process_group()
if __name__=='__main__':main()
