import json
import math
import os
from pathlib import Path
import sys
import time
import torch
import numpy as np
from network import ROOT,Dynamics,load_tuned
from dataset import CACHE
from arm import COUPLED

BATCH=1024
EPOCHS=24
def read(part,device):
    return {k:v.to(device) for k,v in torch.load(CACHE/f'{part}.pt',weights_only=True).items()}

def call(model,d,idx,freeze=False,supervise=False):
    keys=('h','anchor','dt','forcing','memory','left')
    p,s=model(*(d[k][idx] for k in keys),freeze_s=freeze,quadrature=d['quad'][idx] if supervise else None)
    return p+d['offset'][idx],s

def loss(model,d,idx,scale):
    p,s=call(model,d,idx,supervise=model.coupled)
    m=d['mask'][idx]
    raw=((p-d['y'][idx])/scale).square().mean(-1)
    endpoint=(raw*m).sum()/m.sum().clamp_min(1)
    aux=endpoint.new_zeros(())
    if s is not None:
        ok=d['quad'][idx,:,0].sum(1)>.5
        x=((s-d['target'][idx])/scale[0,0,1]).square().mean(-1)
        aux=(x*ok).sum()/ok.sum().clamp_min(1)
    return endpoint+aux,endpoint.detach(),aux.detach()

@torch.no_grad()
def select(model,d,indices,groups,scale):
    # Identical flight-equal metric and selection flights for the two arms.
    parts=[]
    for ix in indices.split(1024):
        p,_=call(model,d,ix)
        e=((p-d['y'][ix])/scale).square().mean(-1)[:,[1,3,7]]
        assert d['mask'][ix][:,[1,3,7]].all()
        parts.append(e.cpu().numpy())
    arr=np.concatenate(parts)
    vals=np.stack([arr[groups==g].mean(0,dtype=np.float64) for g in np.unique(groups)])
    return float(np.sqrt(vals.mean(0)).mean())

def preflight():
    d=read('validation','cpu');orig=load_tuned(17,'cpu')
    a=Dynamics(orig,False);b=Dynamics(orig,True)
    ii=torch.arange(64)
    p,_=call(a,d,ii);q,_=call(b,d,ii)
    torch.testing.assert_close(q,p,atol=0,rtol=0)
    from network import wave
    raw,_,_=wave.load(ROOT,'validation','cpu')
    with torch.no_grad():
        expected=orig(*(raw[k][ii] for k in ('hist','pooled','u0','time','coef')))
        torch.testing.assert_close(p,expected,atol=3e-6,rtol=3e-6)
        b.feedback.weight.fill_(.025)
        q,_=call(b,d,ii); held,_=call(b,d,ii,True)
        torch.testing.assert_close(held[:,:1],q[:,:1],rtol=0,atol=0)
        assert (held[:,1:]-q[:,1:]).abs().max()>1e-5
        for name in ('forcing','memory','left'):
            d[name][ii,4:]+=1
        perturbed,_=call(b,d,ii)
        torch.testing.assert_close(q[:,:4],perturbed[:,:4],atol=0,rtol=0)
    print('original output parity, frozen s0 and causal prefix passed')

def run(bench=False):
    seed=17
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    dev='cuda'
    orig=load_tuned(seed,'cpu')
    model=Dynamics(orig,True if bench else COUPLED).to(dev)
    train=read('train',dev)
    n=len(train['h'])
    scale=((train['y'].square()*train['mask'][...,None]).sum((0,1,3))/(3*train['mask'].sum((0,1)))).sqrt().reshape(1,1,2,1)
    old=set(map(id, model.core.parameters()))
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam([{'params':[p for p in params if id(p) in old],'lr':1e-4},
                          {'params':[p for p in params if id(p) not in old],'lr':.001}],fused=True)
    if bench:
        tick=time.monotonic()
        for j in range(24):
            opt.zero_grad(set_to_none=True)
            loss(model,train,torch.arange(j*BATCH,(j+1)*BATCH,device=dev),scale)[0].backward()
            opt.step()
        torch.cuda.synchronize()
        print('pilot steps=24 seconds=',time.monotonic()-tick);return
    valid=read('validation',dev)
    from search import split_flights
    with np.load(ROOT/'outputs/autopilot_flight_dynamics_20261124/validation.npz') as z:ids=z['flight']
    flights,check=split_flights(ids)
    ii=np.flatnonzero(np.isin(ids,list(flights)));groups=ids[ii]
    indices=torch.tensor(ii,device=dev)
    gen=torch.Generator(device='cpu').manual_seed(seed)
    out=Path(os.environ['RA_OUTPUT'])
    history=[];steps=0;best_epoch=0;start=time.monotonic()
    initial=select(model,valid,indices,groups,scale);best=initial
    def save():
        torch.save(dict(coupled=COUPLED,state={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=best_epoch),out/'model')
    save()
    for epoch in range(1,EPOCHS+1):
        model.train();sums=torch.zeros(2,device=dev)
        for ix in torch.randperm(n,generator=gen).to(dev).split(BATCH):
            fraction=steps/(EPOCHS*math.ceil(n/BATCH)-1)
            factor=.10+.90*(1+math.cos(fraction*math.pi))/2
            for pg,lr in zip(opt.param_groups,[1e-4,1e-3]):pg['lr']=lr*factor
            opt.zero_grad(set_to_none=True)
            objective,ep,aux=loss(model,train,ix,scale)
            objective.backward();torch.nn.utils.clip_grad_norm_(params,5.)
            opt.step()
            sums+=len(ix)*torch.stack((ep,aux));steps+=1
        model.eval();value=select(model,valid,indices,groups,scale)
        if value<best:
            best=value;best_epoch=epoch;save()
        main,slow=(sums/n).tolist()
        assert np.isfinite([main,slow,value]).all()
        row=dict(epoch=epoch,step=steps,train_motion=main,train_slow=slow,val_loss=value)
        history.append(row)
        print(json.dumps(row),flush=True)
    assert steps==2976
    info=dict(steps=steps,validation_history=history,best_epoch=best_epoch,best=best,initial=initial,seed=seed,
              windows=n,auxiliary_beats=int((train['quad'][:,:,0].sum(1)>.5).sum()),seconds=time.monotonic()-start,
              scale=scale.flatten().tolist(),coupled=COUPLED,trainable=sum(p.numel() for p in params),gpu=torch.cuda.get_device_name())
    Path(os.environ['RA_TRAIN_INFO']).write_text(json.dumps(info,indent=2)+'\n')
    (out/'training.json').write_text(json.dumps(info,indent=2)+'\n')

if __name__=='__main__':
    torch.set_num_threads(3)
    if sys.argv[1]=='preflight':preflight()
    else:run(bench=sys.argv[1]=='bench')
