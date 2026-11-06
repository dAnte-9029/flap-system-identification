#!/usr/bin/env python
"""PACLD native GPU minibatches. torchrun --nproc_per_node=2 enables NCCL DDP."""
import argparse,json,os,sys,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from run_pacld import datasets
from system_identification.models.pacld import Config
from system_identification.models.pacld.batch import BatchedPACLD
from system_identification.data.pacld_batch import compile_windows
from system_identification.training.pacld import seed_all,calibrated_weight
from system_identification.training.pacld_gpu import BucketSampler,update,validate,checkpoint_bundle,restore,gather_rng


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/pacld/default.json');p.add_argument('--output',required=True)
    p.add_argument('--steps',required=True,type=int,help='explicit total optimizer updates, including resumed updates')
    p.add_argument('--batch-size',type=int,default=8,help='global batch, divided equally between ranks')
    p.add_argument('--precision',choices=['fp64','mixed','fp32'],default='mixed')
    p.add_argument('--compile',action='store_true');p.add_argument('--checkpoint-steps',action='store_true')
    p.add_argument('--resume');p.add_argument('--device',type=int,default=0)
    a=p.parse_args();world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK',a.device))
    if a.steps<1 or a.batch_size<world or a.batch_size%world:p.error('positive steps and equally divisible global batch required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:torch.distributed.init_process_group('nccl')
    config=json.loads(Path(a.config).read_text());torch.set_num_threads(config['training']['cpu_threads']);seed_all(config['seed'])
    out=Path(a.output)
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    if world>1:torch.distributed.barrier()
    train,val,identity=datasets(config);weight=calibrated_weight(config['training']['calibration'],config,train)
    raw=BatchedPACLD(Config(**config['model']),a.precision,a.compile,a.checkpoint_steps).to(device)
    dtype=torch.float32 if a.precision=='fp32' else torch.float64
    optimizer=torch.optim.Adam(raw.parameters(),**config['optimizer']);sampler=BucketSampler(train.index,config['seed'],a.batch_size)
    schedule=dict(optimizer_steps=0,windows_seen=0,best_score=float('inf'),best_checkpoint=None)
    if a.resume:
        bundle=torch.load(a.resume,map_location='cpu',weights_only=False)
        schedule=restore(bundle,raw,optimizer,sampler,config,identity,weight,world)
    if a.steps<=schedule['optimizer_steps']:raise ValueError('steps must exceed checkpoint updates')
    # process_logit influences posterior gains, not a disconnected auxiliary model.
    model=DDP(raw,device_ids=[local],find_unused_parameters=True) if world>1 else raw
    if rank==0:(out/'configuration.json').write_text(json.dumps(dict(args=vars(a),config=config,data_identity=identity,world_size=world,lambda_macro=weight,git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2))
    while schedule['optimizer_steps']<a.steps:
        begin=time.perf_counter();ids=sampler.draw();local_ids=ids[rank*(a.batch_size//world):(rank+1)*(a.batch_size//world)]
        batch=compile_windows([train[i] for i in local_ids],raw.config).to(device,dtype)
        torch.cuda.synchronize();prep=time.perf_counter()-begin;t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        stats=update(model,batch,optimizer,weight,config['training']['max_grad_norm']);torch.cuda.synchronize()
        elapsed=time.perf_counter()-t;schedule['optimizer_steps']+=1;schedule['windows_seen']+=a.batch_size
        stats.update(optimizer_steps=schedule['optimizer_steps'],windows_seen=schedule['windows_seen'],preparation_seconds=prep,compute_seconds=elapsed,end_to_end_seconds=time.perf_counter()-begin,windows_per_second=a.batch_size/(time.perf_counter()-begin),updates_per_second=1/(time.perf_counter()-begin),peak_memory_bytes=torch.cuda.max_memory_allocated(),global_window_indices=ids)
        stats['gpu_sample']=subprocess.check_output(['nvidia-smi','-i',str(local),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
        if rank==0:
            with (out/'training.jsonl').open('a') as f:f.write(json.dumps(stats)+'\n')
            print(json.dumps(stats),flush=True)
        if schedule['optimizer_steps']%config['training']['validation_interval']==0 or schedule['optimizer_steps']==a.steps:
            if world>1:torch.distributed.barrier()
            rank_rng=gather_rng()
            if rank==0:
                result=validate(raw,val,a.batch_size//world,device,dtype);score=result['selection_score']
                if not __import__('math').isfinite(score):raise FloatingPointError('validation nonfinite')
                name=f"step_{schedule['optimizer_steps']:08d}.pt"
                if score<schedule['best_score']:schedule.update(best_score=score,best_checkpoint=name)
                torch.save(checkpoint_bundle(raw,optimizer,sampler,schedule,config,identity,weight,world,rank_rng),out/name)
                (out/f"validation_{schedule['optimizer_steps']:08d}.json").write_text(json.dumps(result,indent=2))
                (out/'best.json').write_text(json.dumps(schedule,indent=2))
            if world>1:
                obj=[schedule];torch.distributed.broadcast_object_list(obj,src=0);schedule=obj[0]
    if world>1:torch.distributed.destroy_process_group()

if __name__=='__main__':main()
