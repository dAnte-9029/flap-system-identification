"""Batched Adam and deterministic globally bucketed sampling, including DDP."""
import copy
import random
import subprocess
import numpy as np
import torch
from .pacld import FlightBalancedSampler
from system_identification.data.pacld_batch import compile_windows
from system_identification.evaluation.pacld import hierarchical

SCHEMA='pacld_gpu_batched_v1'

class BucketSampler:
    """Keep all original sampler draws; queue by horizon and drain full buckets.

    All ranks run the identical global stream and then take disjoint contiguous
    slices. Pending queues and original sampler state are checkpointed together.
    """
    def __init__(self,index,seed,batch_size):
        self.sampler=FlightBalancedSampler(index,seed); self.batch_size=batch_size
        self.pending={int(h):[] for h in sorted(index.horizon_us.unique())}
        if any(len(g)<batch_size for _,g in index.groupby('horizon_us')): raise ValueError('not enough distinct windows in a horizon')
    def draw(self):
        while True:
            i=self.sampler.draw(); h=int(self.sampler.index.iloc[i].horizon_us)
            self.pending[h].append(i)
            if len(set(self.pending[h]))>=self.batch_size:
                out=[]; retained=[]; seen=set()
                for candidate in self.pending[h]:
                    if candidate not in seen and len(out)<self.batch_size:
                        out.append(candidate);seen.add(candidate)
                    else: retained.append(candidate)
                self.pending[h]=retained
                return out
    def state_dict(self): return copy.deepcopy(dict(sampler=self.sampler.state_dict(),pending=self.pending,batch_size=self.batch_size))
    def load_state_dict(self,s):
        if s['batch_size']!=self.batch_size: raise ValueError('global batch changed')
        self.sampler.load_state_dict(s['sampler']); self.pending=copy.deepcopy(s['pending'])


def update(model,batch,optimizer,weight,max_grad_norm):
    if any(m['split']!='train' for m in batch.metadata): raise ValueError('train windows only')
    model.train(); optimizer.zero_grad(set_to_none=True)
    result=model(batch,weight)
    if not torch.isfinite(result['total']): raise FloatingPointError('nonfinite loss')
    states=torch.cat((result['predictions'].detach().reshape(-1),result['phases'].detach().reshape(-1)))
    if not torch.isfinite(states).all(): raise FloatingPointError('nonfinite rollout')
    if (result['predictions'].detach()[...,6:10].norm(dim=-1)-1).abs().max()>1e-6: raise FloatingPointError('nonunit attitude')
    result['total'].backward()  # DDP reducer completes before global norm clipping.
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm,error_if_nonfinite=True)
    optimizer.step()
    if not torch.isfinite(torch.cat([p.detach().reshape(-1) for p in model.parameters()])).all(): raise FloatingPointError('nonfinite parameters')
    valid=result['macro_valid']
    stats=torch.stack((result['full'].detach().sum(),result['macro'].detach().sum(),valid.sum().to(result['full']),result['total'].detach()*len(valid),result['full'].new_tensor(len(valid))))
    if torch.distributed.is_initialized(): torch.distributed.all_reduce(stats)
    a,b,c,d,n=stats.tolist()
    return dict(full=a/n,macro=b/c if c else None,macro_valid_windows=int(c),total=d/n,grad_norm=float(norm),windows=int(n))


@torch.no_grad()
def validate(model,windows,batch_size,device,dtype,progress=None):
    model.eval(); rows=[]
    for _,group in windows.index.groupby('horizon_us',sort=True):
        ids=group.index.tolist()
        for j in range(0,len(ids),batch_size):
            ws=[windows[i] for i in ids[j:j+batch_size]]
            if any(w.metadata['split']!='validation' for w in ws): raise ValueError('validation only')
            padded=ws+[ws[-1]]*(batch_size-len(ws))
            batch=compile_windows(padded,model.config).to(device,dtype); r=model(batch,0.)
            for i,w in enumerate(ws):
                rows.append(dict(w.metadata,full=float(r['full'][i]),macro=float(r['macro'][i]) if bool(r['macro_valid'][i]) else None))
            if progress is not None: progress(len(rows),len(windows.index))
    # Identical fixed cohort and flight/horizon hierarchical FULL selection rule.
    return dict(selection_score=hierarchical(rows,'full')['score'],full=hierarchical(rows,'full'),macro=hierarchical(rows,'macro'),windows=len(rows),macro_valid_windows=sum(r['macro'] is not None for r in rows))


def checkpoint_bundle(model,optimizer,sampler,schedule,config,identity,weight,world_size,rank_rng=None):
    return dict(schema=SCHEMA,rank_rng=rank_rng,model=model.state_dict(),optimizer=optimizer.state_dict(),sampler=sampler.state_dict(),schedule=copy.deepcopy(schedule),config=config,data_identity=identity,lambda_macro=weight,precision=model.precision,world_size=world_size,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if world_size==1 else None,python_rng=random.getstate(),numpy_rng=np.random.get_state(),git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())


def restore(bundle,model,optimizer,sampler,config,identity,weight,world_size):
    if (bundle.get('schema')!=SCHEMA or bundle['config']!=config or bundle['data_identity']!=identity or bundle['lambda_macro']!=weight or bundle['precision']!=model.precision or bundle['world_size']!=world_size): raise ValueError('GPU checkpoint contract mismatch; CPU optimizer resume forbidden')
    model.load_state_dict(bundle['model']);optimizer.load_state_dict(bundle['optimizer']);sampler.load_state_dict(bundle['sampler'])
    if world_size>1:
        if bundle.get('rank_rng') is None or len(bundle['rank_rng'])!=world_size: raise ValueError('missing distributed RNG states')
        rng=bundle['rank_rng'][torch.distributed.get_rank()]
        torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda']);random.setstate(rng['python']);np.random.set_state(rng['numpy'])
    else:
        torch.set_rng_state(bundle['torch_rng']);torch.cuda.set_rng_state_all(bundle['cuda_rng']);random.setstate(bundle['python_rng']);np.random.set_state(bundle['numpy_rng'])
    return copy.deepcopy(bundle['schedule'])


def gather_rng():
    local=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate(),numpy=np.random.get_state())
    if not torch.distributed.is_initialized(): return [local]
    states=[None]*torch.distributed.get_world_size()
    torch.distributed.all_gather_object(states,local)
    return states
