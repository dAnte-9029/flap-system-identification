"""Fixed update count, native labels and the same CUDA-graph training recipe for all five members."""
import argparse
import json
import math
import os
import time
import numpy as np
import pandas as pd
import torch
from workspace import *
from candidates import build
from common import make_model, geodesic, qmul, qexp
from train import load, statistics, atomic_save, METRICS
from harmonic import candidate_model

HORIZONS3=(.2,.5,1.,2.,3.)


def data_three(split):
    if split not in ('train','validation'):
        raise ValueError(split)
    data,ids=load(CACHE3,split)
    # Earlier prepared buffers encoded absent rows with an all-zero quaternion.
    # atan2(0,0) has an undefined backward even when multiplied by a zero mask.
    quat=data[2][:,:,6:10]
    padding=quat.abs().sum(-1)==0
    assert not (padding & data[4][:,:,1]).any()
    quat[:,:,0].masked_fill_(padding,1.)
    dist=(data[3]-3.).abs().masked_fill(~data[4],float('inf'))
    nearest,idx=dist.min(1)
    tol=torch.tensor(np.where(ids.sampling_group=='main_50hz',.035,.105),device='cuda',dtype=dist.dtype)
    idx=idx.masked_fill(nearest>tol[:,None],-1)
    data[6]=torch.cat((data[6],idx[:,None,[0,0,1,2]]),1)
    assert (data[6]>=0).all() and ids.window_id.is_unique
    tape=torch.from_numpy(np.load(CACHE3/f'{split}_commands.npy')).cuda()
    return [*data,tape],ids


def model_factory(name,mean,scale):
    if name in BASELINES:
        return make_model(name,mean,scale)
    if name=='c2':
        return candidate_model(name,mean,scale)
    return build(name,mean,scale)


def drift_scale(data):
    s,y,t,indices=data[1],data[2],data[3],data[6]
    accum=torch.zeros(indices.size(1)-2,2,device=s.device,dtype=torch.float64)
    count=torch.zeros_like(accum)
    for start in range(0,len(s),2048):
        sl=slice(start,start+2048)
        rows=torch.arange(len(s[sl]),device=s.device)[:,None]
        for col,k in enumerate([0,2]):
            ix=indices[sl,2:,k].clamp_min(0)
            dt=t[sl][rows,ix,col,None]
            if col==0:
                pred=s[sl,None,:3]+dt*s[sl,None,3:6]
                e=(pred-y[sl][rows,ix,:3]).square().sum(-1)
            else:
                pred=qmul(s[sl,None,6:10],qexp(s[sl,None,10:]*dt))
                e=geodesic(pred,y[sl][rows,ix,6:10]).square()
            valid=indices[sl,2:,k]>=0
            accum[:,col]+=(e*valid).sum(0)
            count[:,col]+=valid.sum(0)
    return (accum/count).float().clamp_min(1e-8)


@torch.no_grad()
def evaluate(model,data,ids,units,horizons):
    model.eval()
    scores=[]; errors=[]; masks=[]
    for start in range(0,len(ids),1024):
        _,s,e,ok=conditional_loss(model, select(data,slice(start,start+1024)),units)
        scores.append(s.cpu().numpy());errors.append(e.cpu().numpy());masks.append(ok.cpu().numpy())
    model.train()
    flight=ids.flight_id.to_numpy()
    score=float(pd.Series(np.concatenate(scores)).groupby(flight).mean().mean())
    e,mask=np.concatenate(errors),np.concatenate(masks)
    frames=[]
    for k,h in enumerate(horizons):
        for i,metric in enumerate(METRICS):
            ok=mask[:,k,i]
            by=pd.Series(e[ok,k,i].astype('float64')).groupby(flight[ok]).agg(['mean','size'])
            frames.append(by.rename(columns={'mean':'mse','size':'windows'}).rename_axis('flight_id').reset_index().assign(horizon_s=h,metric=metric))
    return score,pd.concat(frames,ignore_index=True)


def main(name,gpu):
    from objectives import training_loss
    locked=json.loads((RUN/'freeze.json').read_text())
    if name not in (*BASELINES,locked['representative']):
        raise ValueError(name)
    directory=RUN/'trained3'/name
    directory.mkdir(parents=True,exist_ok=False)
    torch.cuda.set_device(gpu)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32=True
    start=time.monotonic()
    data,ids=data_three('train')
    val,vid=data_three('validation')
    mean,scale,units=statistics(data)
    model=model_factory(name,mean,scale).cuda()
    drift=drift_scale(data) if model.structured else None
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,fused=True)
    gen=torch.Generator(device=f'cuda:{gpu}').manual_seed(17)
    batch,per_epoch,epochs=1024,124,24
    n=len(ids)
    cfg=dict(model=name,params=sum(p.numel() for p in model.parameters()),seed=17,units=units.tolist(),batch=batch,
             train_flights=ids.flight_id.nunique(),train_windows=n,validation_windows=len(vid),validation_flights=vid.flight_id.nunique(),
             training_s=3.,horizons=HORIZONS3,cache=str(CACHE3),epochs=epochs,updates_per_epoch=per_epoch,
             effective_epochs=2976*batch/n,gpu=gpu,device=torch.cuda.get_device_name(),git_base=locked['git_base'],
             sample_policy='124 batches per shuffled permutation; wrap from beginning',pid=os.getpid(),
             drift_scaling=None if drift is None else drift.tolist())
    (directory/'config.json').write_text(json.dumps(cfg,indent=2))
    static=[x[:batch].clone() for x in data]
    def backward():
        loss=training_loss(name,model,static,units,drift)[0]
        loss.backward()
        return loss
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        backward()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        loss=backward()
    torch.cuda.synchronize()
    tracked=next(model.parameters())
    before=tracked.detach().clone()
    print('captured',name,gpu,cfg['params'],flush=True)
    best=float('inf')
    for epoch in range(1,epochs+1):
        perm=torch.randperm(n,generator=gen,device=f'cuda:{gpu}')
        indices=perm.repeat(math.ceil(batch*per_epoch/n))[:batch*per_epoch]
        for j,ix in enumerate(indices.split(batch)):
            step=(epoch-1)*per_epoch+j+1
            for dest,x in zip(static,data):
                torch.index_select(x,0,ix,out=dest)
            graph.replay()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not bool(torch.isfinite(loss)&torch.isfinite(norm)):
                raise FloatingPointError(step)
            optimizer.param_groups[0]['lr']=.001*(.55+.45*math.cos(math.pi*(step-1)/(epochs*per_epoch)))
            optimizer.step()
            if step in (1,20) or step%500==0:
                print('update',step,float(loss),float(norm),float((tracked-before).abs().max()),flush=True)
        score,rows=evaluate(model,val,vid,units,HORIZONS3)
        if not math.isfinite(score): raise FloatingPointError(epoch)
        ck=dict(epoch=epoch,updates=step,model=model.state_dict(),optimizer=optimizer.state_dict(),generator=gen.get_state(),
                config=cfg,validation_score=score)
        atomic_save(ck,directory/'last.pt')
        if score<best:
            best=score
            atomic_save(ck,directory/'best.pt')
            rows.to_csv(directory/'best_flights.csv',index=False)
        with (directory/'epochs.jsonl').open('a') as f:
            print(json.dumps(dict(epoch=epoch,updates=step,validation_score=score)),file=f)
        print('epoch',epoch,score,'best',best,'seconds',round(time.monotonic()-start,1),flush=True)
    (directory/'completed.json').write_text(json.dumps(dict(updates=step,best=best,seconds=time.monotonic()-start)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('name');p.add_argument('--gpu',type=int,required=True)
    a=p.parse_args()
    main(a.name,a.gpu)
