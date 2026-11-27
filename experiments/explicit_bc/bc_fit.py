import json
import math
import os
import sys
from pathlib import Path
import numpy as np
import torch
from bc_features import ROOT,EXTRA,CACHE,read
from bc_model import Predictor
from bc_arm import INCLUDE_A
from search import split_flights

def partition(part,dev):
    d=read(part,'cpu')
    keys=['y','mask','dt','forcing','memory']
    if part=='train':keys+=['target']
    data={k:d[k].to(dev) for k in keys}
    data.update({k:v.to(dev) for k,v in torch.load(CACHE/f'{part}.pt',weights_only=True).items()})
    return data

def endpoint(model,d,idx):
    x=d['x'][idx]
    coeff=model.encode_coefficients(x)
    states=model.state_path(x,d['dt'][idx],d['forcing'][idx],d['memory'][idx]) if model.include_a else None
    full,_=model.endpoints(coeff,d['query'][idx],d['base'][idx],states)
    return full,coeff,states

def loss(model,d,q,idx,scale):
    full,coeff,states=endpoint(model,d,idx)
    mask=d['mask'][idx]
    ep=((((full-d['y'][idx])/scale).square().mean(-1))*mask).sum()/mask.sum()
    lengths=q['length'][idx];ok=lengths>0
    owner=torch.repeat_interleave(torch.arange(len(idx),device=idx.device),lengths)
    start=(lengths.cumsum(0)-lengths)[owner]
    sub=q['start'][idx][owner]+torch.arange(len(owner),device=idx.device)-start
    pf,ps=model.native(coeff,q['features'][sub],d['base'][idx],states,owner,
                       q['slot'][sub] if states is not None else None,q['path'][sub] if states is not None else None)
    origin=d['origin'][idx]
    def mean_loss(pred):
        mean=torch.zeros_like(origin).index_add(0,owner,q['weight'][sub,None]*pred)+origin
        return ((((mean-d['target'][idx])/scale[0,0,1,0]).square().mean(-1))*ok).sum()/ok.sum().clamp_min(1)
    raw=mean_loss(pf);slow=mean_loss(ps)
    return ep+raw+slow,torch.stack((ep,raw,slow)).detach()

@torch.no_grad()
def select(model,d,indices,flight,scale):
    errors=[]
    for ix in indices.split(1024):
        p=endpoint(model,d,ix)[0]
        e=((p-d['y'][ix])/scale).square().mean(-1)[:,[1,3,7]]
        errors.append(e.cpu().numpy())
    a=np.concatenate(errors)
    return float(np.sqrt(np.stack([a[flight==g].mean(0,dtype=np.float64) for g in np.unique(flight)]).mean(0)).mean())

def run():
    dev='cuda';seed=17
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    d=partition('train',dev);v=partition('validation',dev)
    q0=torch.load(EXTRA/'beats.pt',weights_only=True)
    names=['weight','start','length']+(['slot','path'] if INCLUDE_A else [])
    q={k:q0[k].to(dev) for k in names}
    q['features']=torch.load(CACHE/'native.pt',weights_only=True).to(dev)
    scale=((d['y'].square()*d['mask'][...,None]).sum((0,1,3))/(3*d['mask'].sum((0,1)))).sqrt().reshape(1,1,2,1)
    with np.load(ROOT/'outputs/autopilot_flight_dynamics_20261124/validation.npz') as z:groups=z['flight']
    sel,_=split_flights(groups);ii=np.flatnonzero(np.isin(groups,list(sel)))
    valid=torch.tensor(ii,device=dev)
    model=Predictor(INCLUDE_A).to(dev)
    opt=torch.optim.Adam(model.parameters(),lr=.001,fused=True)
    gen=torch.Generator().manual_seed(seed);n=len(d['x'])
    best=select(model,v,valid,groups[ii],scale);epoch_best=0
    def state():return {k:p.detach().cpu().clone() for k,p in model.state_dict().items()}
    weights=state();history=[];steps=0
    for epoch in range(1,25):
        model.train();sums=torch.zeros(3,device=dev)
        for ix in torch.randperm(n,generator=gen).to(dev).split(1024):
            opt.param_groups[0]['lr']=.001*(.1+.9*(1+math.cos(math.pi*steps/2975))/2)
            opt.zero_grad(set_to_none=True)
            objective,terms=loss(model,d,q,ix,scale)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            sums+=len(ix)*terms;steps+=1
        model.eval();score=select(model,v,valid,groups[ii],scale)
        avg=(sums/n).tolist()
        assert np.isfinite([score,*avg]).all()
        if score<best:best=score;weights=state();epoch_best=epoch
        history.append(dict(epoch=epoch,selection=score,training_terms=avg))
        print(epoch,score,avg,flush=True)
    assert steps==2976
    out=Path(os.environ['RA_OUTPUT'])
    torch.save(dict(include_a=INCLUDE_A,state=state(),selected=weights),out/'model')
    info=dict(steps=steps,best_epoch=epoch_best,best=best,seed=seed,include_a=INCLUDE_A,
              parameters=sum(p.numel() for p in model.parameters()),history=history)
    Path(os.environ['RA_TRAIN_INFO']).write_text(json.dumps(info)+'\n')
    (out/'training.json').write_text(json.dumps(info,indent=2)+'\n')

if __name__=='__main__':
    torch.set_num_threads(3)
    run()
