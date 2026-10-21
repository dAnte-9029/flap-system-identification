"""Paired GRU64 training; targets never enter the inference input dictionary."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from system_identification.models.velocity_rate_gru import VelocityRateGRU, baseline, errors
from system_identification.models.control_events import CANONICAL_CONTROL_EVENTS
from system_identification.data.velocity_rate_dataset import write_json, file_hash
from prepare_velocity_rate_learning import EXP, CACHE

INPUT_KEYS=('history','history_mask','history_dt','history_length','initial','dt','events','event_length')


def load(split,device):
    assert split in ('train','validation')
    a=np.load(CACHE/f'{split}.npz')
    return {k:torch.from_numpy(a[k]).to(device) for k in a.files}


def batch(data,index):
    return {k:data[k][index] for k in INPUT_KEYS}


def window_loss(pred,target,mask,scales,weights):
    components=(errors(pred,target)/scales).square()*weights
    return (components.sum(-1)*mask).sum(-1)/mask.sum(-1).clamp_min(1)


def fit_normalization(data,config):
    h=data['history']; m=data['history_mask']
    columns=[3,4,5,10,11,12,19,13,14,15,16]
    mean=[]; scale=[]
    for c in columns:
        v=h[...,c][m[...,c]]
        mean.append(v.mean().item()); scale.append(max(v.std().item(),.05))
    previous=torch.cat((data['initial'][:,None,:],data['target'][:,:-1]),1)
    delta=(data['target']-previous)[..., [3,4,5,10,11,12]][data['future_mask']]
    output_scale=delta.std(0).clamp_min(.03).tolist()
    sumsq=torch.zeros(4,device=h.device); count=0
    for ix in torch.arange(len(h),device=h.device).split(2048):
        err=errors(baseline(data['initial'][ix],data['dt'][ix]),data['target'][ix])
        mask=data['future_mask'][ix]
        sumsq+=(err.square()*mask[...,None]).sum((0,1)); count+=mask.sum()
    scales=torch.maximum((sumsq/count).sqrt(),torch.tensor(config['loss_scale_floors'],device=h.device)).tolist()
    return dict(mean=mean,scale=scale,output_scale=output_scale,loss_scales=scales,
                source='train valid history samples and train future residuals only',
                features=['v_N[3]','omega_FRD[3]','frequency','commands[4]'])


@torch.no_grad()
def validate(model,data,scales,weights,flight_index,batch_size):
    model.eval(); losses=[]
    for ix in torch.arange(len(data['initial']),device=scales.device).split(batch_size):
        pred=model(batch(data,ix))
        losses.append(window_loss(pred,data['target'][ix],data['future_mask'][ix],scales,weights))
    loss=torch.cat(losses)
    count=torch.bincount(flight_index)
    flight_sum=torch.bincount(flight_index,weights=loss)
    return (flight_sum/count).mean().item(),loss.mean().item()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def train(config,normalization,train_data,val_data,device,model_name,seed):
    if config.get('control_event_policy', CANONICAL_CONTROL_EVENTS) != CANONICAL_CONTROL_EVENTS:
        raise ValueError('New training requires canonical control events')
    config = dict(config, control_event_policy=CANONICAL_CONTROL_EVENTS)
    set_seed(seed)
    model=VelocityRateGRU(normalization,control=model_name=='GRU-Control',hidden=config['hidden_size'],event_hidden=config['event_hidden_size']).to(device)
    folder=EXP/f'{model_name}/seed_{seed}'; folder.mkdir(parents=True,exist_ok=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['max_epochs'])
    scales=torch.tensor(normalization['loss_scales'],device=device)
    weights=torch.tensor(config['loss_weights'],device=device)
    meta=pd.read_parquet(EXP/'validation_windows.parquet')
    flight_index=torch.tensor(pd.factorize(meta.flight_id)[0],device=device)
    best=float('inf'); stale=0; updates=0; best_epoch=0
    start=time.time(); torch.cuda.reset_peak_memory_stats(device)
    log=[]
    for epoch in range(1,config['max_epochs']+1):
        tick=time.time(); model.train()
        # Separate CPU generator: batch order is identical across the two architectures.
        generator=torch.Generator().manual_seed(seed*10000+epoch)
        indices=torch.randperm(len(train_data['initial']),generator=generator).to(device)
        total=0.; seen=0; max_gradient=0.
        for ix in indices.split(config['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            prediction=model(batch(train_data,ix))
            loss=window_loss(prediction,train_data['target'][ix],train_data['future_mask'][ix],scales,weights).mean()
            if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
            loss.backward()
            gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'],error_if_nonfinite=True)
            optimizer.step(); updates+=1
            total+=loss.item()*len(ix); seen+=len(ix); max_gradient=max(max_gradient,gradient.item())
        vl,window_vl=validate(model,val_data,scales,weights,flight_index,config['evaluation_batch_size'])
        improved=vl < best-config['minimum_improvement']
        if improved:
            best=vl; stale=0; best_epoch=epoch
            torch.save(dict(model=model.state_dict(),normalization=normalization,config=config,seed=seed,
                            model_name=model_name,epoch=epoch,validation_flight_loss=vl,
                            input_keys=INPUT_KEYS),folder/'best.pt')
        else: stale+=1
        record=dict(epoch=epoch,train_window_loss=total/seen,validation_flight_loss=vl,
                    validation_window_loss=window_vl,lr=optimizer.param_groups[0]['lr'],
                    gradient_norm_max=max_gradient,seconds=time.time()-tick,updates=updates,
                    windows_per_second=seen/(time.time()-tick),best_epoch=best_epoch)
        log.append(record)
        with (folder/'training.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        print(model_name,seed,json.dumps(record),flush=True)
        scheduler.step()
        if stale>=config['patience']: break
    result=dict(model=model_name,seed=seed,parameters=sum(p.numel() for p in model.parameters()),
                gradient_parameters=sum(p.numel() for p in model.parameters() if p.grad is not None),
                epochs=len(log),updates=updates,examples_seen=seen*len(log),best_epoch=best_epoch,
                best_validation_flight_loss=best,seconds=time.time()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                stop_reason='patience' if stale>=config['patience'] else 'fixed_epoch_budget',
                checkpoint_sha256=file_hash(folder/'best.pt'))
    write_json(folder/'summary.json',result)
    return result


def smoke(config,norm,train_data,device):
    ix=torch.arange(64,device=device); inputs=batch(train_data,ix)
    result={}
    for control in (False,True):
        set_seed(17)
        m=VelocityRateGRU(norm,control=control).to(device)
        opt=torch.optim.AdamW(m.parameters(),lr=.001)
        tick=time.time(); grads=[]
        for _ in range(3):
            opt.zero_grad(); p=m(inputs)
            loss=window_loss(p,train_data['target'][ix],train_data['future_mask'][ix],
                             torch.tensor(norm['loss_scales'],device=device),torch.ones(4,device=device)).mean()
            loss.backward()
            grads.append(float(torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True)))
            opt.step()
        torch.cuda.synchronize()
        with torch.no_grad():
            p=m(inputs)
            changed={k:v.clone() for k,v in inputs.items()}
            changed['history'][...,:3]+=10000
            shifted=m(changed)
            torch.testing.assert_close(p,shifted,rtol=0,atol=0)
            if not control:
                changed['history'][...,13:17]=torch.randn_like(changed['history'][...,13:17])*999
                changed['events'].fill_(float('nan')); changed['event_length'].fill_(99999)
                torch.testing.assert_close(p,m(changed),rtol=0,atol=0)
            assert torch.isfinite(p).all()
        result['Control' if control else 'NoControl']=dict(device=str(p.device),steps=p.shape[1],
            updates=3,loss=loss.item(),gradient_norms=grads,seconds=time.time()-tick,
            quaternion_norm_max_error=(torch.linalg.vector_norm(p[...,6:10],dim=-1)-1).abs().max().item(),
            translation_invariance=True,no_control_isolation=not control)
    write_json(EXP/'smoke.json',result); print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke-only',action='store_true'); parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device('cuda:0')
    config=json.loads((EXP/'config.json').read_text())
    tick=time.time(); td=load('train',device); vd=load('validation',device); torch.cuda.synchronize()
    norm_path=EXP/'normalization.json'
    norm=json.loads(norm_path.read_text()) if norm_path.exists() else fit_normalization(td,config)
    if not norm_path.exists(): write_json(norm_path,norm)
    if args.smoke_only:
        smoke(config,norm,td,device)
        write_json(EXP/'runtime.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                    load_seconds=time.time()-tick,gpu_resident_data_bytes=sum(v.numel()*v.element_size() for d in (td,vd) for v in d.values()),
                    threads=4,tf32=True,cudnn_deterministic=True))
        return
    assert (EXP/'smoke.json').exists()
    for seed in ([args.seed] if args.seed is not None else config['seeds']):
        for name in ('GRU-NoControl','GRU-Control'):
            train(config,norm,td,vd,device,name,seed)

if __name__=='__main__': main()
