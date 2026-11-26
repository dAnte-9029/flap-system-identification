"""Paired fitting on fixed endpoint labels and time-weighted native beat measurements."""
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from shared import ROOT,EXTRA,read,select,load_tuned
import recipe

def native_predictions(model,hs,ss,ids,q,owner,sub):
    dt,f,m=q['path'][sub].split([1,24,16],1)
    slot=q['slot'][sub]
    start=hs[owner,slot]
    rate=model.core.rate(start,f)+model.core.memory_rate(m)
    if model.coupled:rate=rate+model.feedback(ss[owner,slot])
    answer=model.core.observe(start+dt*rate)[:,3:]-model.core.observe(hs[:,0])[owner,3:]
    return answer+q['offset'][sub]

def batch_loss(model,arm,d,q,idx,scale):
    active=arm.get('consistency') or arm.get('native')
    coupled=model.coupled
    args=[d[k][idx] for k in ('h','anchor','dt','forcing','memory','left')]
    result=model(*args,quadrature=d['quad'][idx] if coupled else None,trace=active)
    p,s=result[:2]
    mask=d['mask'][idx]
    residual=(p+d['offset'][idx]-d['y'][idx])/scale
    endpoint=(residual.square().mean(-1)*mask).sum()/mask.sum()
    ok=d['quad'][idx,:,0].sum(1)>.5
    wscale=scale[0,0,1,0]
    zero=endpoint.new_zeros(())
    def reduce(x):return (x*ok).sum()/ok.sum().clamp_min(1)
    aux=reduce(((s-d['target'][idx])/wscale).square().mean(-1)) if coupled else zero
    extras={'consistency':zero,'native':zero,'centered':zero,'direct':zero}
    if active:
        hs,ss=result[2:]
        length=q['length'][idx]
        owner=torch.repeat_interleave(torch.arange(len(idx),device=idx.device),length)
        starts=(length.cumsum(0)-length)[owner]
        sub=q['start'][idx][owner]+torch.arange(len(owner),device=idx.device)-starts
        pred=native_predictions(model,hs,ss,idx,q,owner,sub)
        wt=q['weight'][sub,None]
        mean=torch.zeros_like(d['target'][idx]).index_add(0,owner,pred*wt)
        target=d['target'][idx]
        if coupled:extras['consistency']=reduce(((mean-s.detach())/wscale).square().mean(-1))
        extras['direct']=reduce(((mean-target)/wscale).square().mean(-1))
        e=(pred-q['raw'][sub])/wscale
        native=(wt[:,0]*e.square().mean(-1)).sum()/ok.sum().clamp_min(1)
        extras['native']=native
        extras['centered']=native-extras['direct']
    extra=recipe.extra_loss(model,arm,extras)
    loss=endpoint+aux+extra
    stats=torch.stack((endpoint,aux,extras['consistency'],extras['native'])).detach()
    return loss,stats

def fit(name,arm,origdata,device,seed):
    train,valid,q,ixval,groups,scale=origdata
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=recipe.build(load_tuned(seed,'cpu'),arm).to(device)
    old=set(map(id,model.core.parameters()))
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam([{'params':[p for p in params if id(p) in old],'lr':1e-4},
                          {'params':[p for p in params if id(p) not in old],'lr':.001}],fused=True)
    gen=torch.Generator().manual_seed(seed);n=len(train['h'])
    best=select(model,valid,ixval,groups,scale);initial=best;epoch_best=0
    def state():return {k:p.detach().cpu().clone() for k,p in model.state_dict().items()}
    weights=state();history=[];steps=0;start=time.monotonic()
    for epoch in range(1,25):
        model.train();sums=torch.zeros(4,device=device)
        for ix in torch.randperm(n,generator=gen).to(device).split(1024):
            factor=.10+.90*(1+math.cos(math.pi*steps/2975))/2
            for pg,lr in zip(opt.param_groups,(1e-4,1e-3)):pg['lr']=lr*factor
            opt.zero_grad(set_to_none=True)
            obj,stats=batch_loss(model,arm,train,q,ix,scale)
            obj.backward()
            torch.nn.utils.clip_grad_norm_(params,5.);opt.step()
            sums+=len(ix)*stats;steps+=1
        model.eval();val=select(model,valid,ixval,groups,scale)
        if val<best:best=val;epoch_best=epoch;weights=state()
        avg=(sums/n).tolist()
        assert np.isfinite([val,*avg]).all()
        row=dict(epoch=epoch,step=steps,val_loss=val,terms=avg);history.append(row)
        print(name,epoch,round(val,6),round(time.monotonic()-start,1),flush=True)
    assert steps==2976
    return dict(arm=arm,state=state(),selected=weights,epoch=epoch_best),dict(steps=steps,best_epoch=epoch_best,initial=initial,best=best,
               parameters=sum(p.numel() for p in params),seconds=time.monotonic()-start,validation_history=history)

def run():
    dev='cuda';seed=17
    train=read('train',dev);valid=read('validation',dev)
    q={k:v.to(dev) for k,v in torch.load(EXTRA/'beats.pt',weights_only=True).items()}
    scale=((train['y'].square()*train['mask'][...,None]).sum((0,1,3))/(3*train['mask'].sum((0,1)))).sqrt().reshape(1,1,2,1)
    from search import split_flights
    with np.load(ROOT/'outputs/autopilot_flight_dynamics_20261124/validation.npz') as z:ids=z['flight']
    sel,_=split_flights(ids);ii=np.flatnonzero(np.isin(ids,list(sel)))
    data=train,valid,q,torch.tensor(ii,device=dev),ids[ii],scale
    packs={};info={}
    for name,arm in recipe.ARMS.items():packs[name],info[name]=fit(name,arm,data,dev,seed)
    out=Path(os.environ['RA_OUTPUT'])
    torch.save(packs,out/'model')
    (out/'training.json').write_text(json.dumps(info,indent=2)+'\n')
    target='full' if 'full' in info else 'baseline'
    Path(os.environ['RA_TRAIN_INFO']).write_text(json.dumps(dict(**info[target],arms=list(info),total_optimizer_updates=sum(v['steps'] for v in info.values())))+'\n')

if __name__=='__main__':
    torch.set_num_threads(3)
    if sys.argv[-1]=='preflight':
        from fit_coupled import preflight
        preflight()
    else:run()
