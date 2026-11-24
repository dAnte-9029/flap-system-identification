"""One CUDA-resident paired training member. Validation is whole-flight equal."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from model import Predictor, StandardGRU, geodesic, losses


KEYS = ['history','initial','y','t','valid','beat']
HORIZONS = [.2,.5,1.,2.]
METRICS = ['position_m','velocity_m_s','attitude_rad','rate_rad_s']


def load(cache, split):
    ids = pd.read_parquet(cache/f'{split}_windows.parquet')
    batch = [torch.from_numpy(np.load(cache/f'{split}_{key}.npy')).cuda() for key in KEYS]
    t,valid = batch[3:5]
    native_endpoint = []
    tolerance = torch.tensor(np.where(ids.sampling_group=='main_50hz',.035,.105),device='cuda',dtype=t.dtype)
    for horizon in HORIZONS:
        distance = (t-horizon).abs().masked_fill(~valid,float('inf'))
        nearest,index = distance.min(1)
        index = index.masked_fill(nearest>tolerance[:,None],-1)
        native_endpoint.append(index[:,[0,0,1,2]])
    batch.append(torch.stack(native_endpoint,1))
    return batch,ids


def statistics(train):
    history,initial,y,_,valid,_,_ = train[:7]
    mean = history.mean((0,1))
    scale = history.std((0,1)).clamp_min(.05)
    # Pooled train-only RMS displacement/rotation and variation of v and omega
    # set the relative scales of the four losses. No future signal goes to forward().
    chunks = torch.zeros(4,device='cuda',dtype=torch.float64)
    counts = torch.zeros_like(chunks)
    for j in range(0,len(initial),2048):
        sl = slice(j,j+2048)
        s = initial[sl,None]
        target = y[sl]
        sq = torch.stack([target[:,:,:3].square().sum(-1),
                          (target[:,:,3:6]-s[:,:,3:6]).square().sum(-1),
                          geodesic(s[:,:,6:10],target[:,:,6:10]).square(),
                          (target[:,:,10:]-s[:,:,10:]).square().sum(-1)],-1)
        mask = valid[sl][:,:,[0,0,1,2]]
        chunks += (sq*mask).sum((0,1))
        counts += mask.sum((0,1))
    return mean,scale,(chunks/counts).sqrt().float().clamp_min(.05)


def select(data,ix):
    return [t[ix] for t in data]


def validation(model,data,ids,units,batch_size,loss_fn=losses):
    model.eval()
    values,masks,scores = [],[],[]
    with torch.no_grad():
        for start in range(0,len(ids),batch_size):
            _,shared,sq,valid = loss_fn(model,select(data,slice(start,start+batch_size)),units)
            scores.append(shared.cpu().numpy())
            values.append(sq.cpu().numpy())
            masks.append(valid.cpu().numpy())
    score = pd.Series(np.concatenate(scores)).groupby(ids.flight_id.to_numpy()).mean().mean()
    values, masks = np.concatenate(values),np.concatenate(masks)
    rows = []
    for i,horizon in enumerate(HORIZONS):
        for k,metric in enumerate(METRICS):
            ok = masks[:,i,k]
            mse = pd.Series(values[ok,i,k]).groupby(ids.flight_id.to_numpy()[ok]).mean()
            rows.append(dict(horizon_s=horizon,metric=metric,flight_equal_rmse=float(np.sqrt(mse.mean())),
                             flights=len(mse),windows=int(ok.sum())))
    model.train()
    return float(score),rows


def atomic_save(obj,path):
    tmp = path.with_suffix('.tmp')
    torch.save(obj,tmp)
    tmp.replace(path)


def run(args,load_data=load,build_model=None,loss_fn=losses,extra_config=None):
    start = time.monotonic()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    train,ids = load_data(args.cache,'train')
    val,vid = load_data(args.cache,'validation')
    mean,scale,units = statistics(train)
    if build_model is not None:
        model = build_model(mean,scale).cuda()
    elif args.model == 'gru':
        model = StandardGRU(mean, scale).cuda()
    else:
        model = Predictor(args.model=='slowfast',mean,scale).cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,fused=True)
    gen = torch.Generator(device=f'cuda:{args.gpu}').manual_seed(args.seed)
    n = len(ids)
    per_epoch = math.ceil(n/args.batch)
    config = dict(model=args.model,seed=args.seed,epochs=args.epochs,batch=args.batch,updates_per_epoch=per_epoch,
                  params=sum(p.numel() for p in model.parameters()),units=units.tolist(),history_s=.5,rollout_dt=.02,
                  macro_dt=.02 if args.model=='gru' else .1,training_s=2.,optimizer='AdamW fused',lr=.001,weight_decay=.0001,
                  schedule='cosine to 10 percent',gradient_clip=5,tf32=True,aux_weight=.25 if model.structured else 0,
                  sample_policy='paired shuffled epochs; final batch wraps to start of permutation',
                  cache=str(args.cache.resolve()),dataset='velocity_rate_multistep_v1',train_windows=n,
                  train_flights=ids.flight_id.nunique(),validation_windows=len(vid),validation_flights=vid.flight_id.nunique(),
                  git_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip(),
                  device=torch.cuda.get_device_name(),pid=os.getpid(),gpu=args.gpu)
    config.update(extra_config or {})
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    print(json.dumps(config),flush=True)
    # Capture the repeated forward/backward kernels, keeping sampling and Adam outside.
    static = [x[:args.batch].clone() for x in train]
    def backward():
        loss,*_ = loss_fn(model,static,units)
        loss.backward()
        return loss
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        backward()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        loss = backward()
    torch.cuda.synchronize()
    print('CUDA forward/backward graph captured',flush=True)
    train_start = time.monotonic()
    tracked = model.init_hidden.weight if hasattr(model, "init_hidden") else next(model.backbone.parameters())
    before = tracked.detach().clone()
    best = float('inf')
    for epoch in range(1,args.epochs+1):
        perm = torch.randperm(n,generator=gen,device='cuda')
        perm = torch.cat([perm,perm[:per_epoch*args.batch-n]])
        for j,ix in enumerate(perm.split(args.batch)):
            step = (epoch-1)*per_epoch+j+1
            for buffer,x in zip(static,train):
                torch.index_select(x,0,ix,out=buffer)
            graph.replay()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not bool(torch.isfinite(loss)&torch.isfinite(norm)):
                raise FloatingPointError(f'nonfinite loss/gradient at {step}')
            optimizer.param_groups[0]['lr'] = .001*(.55+.45*math.cos(math.pi*(step-1)/(per_epoch*args.epochs)))
            optimizer.step()
            if step in (1,5,20,50) or step%100==0:
                elapsed = time.monotonic()-train_start
                delta = float((tracked-before).abs().max())
                if delta==0:
                    raise RuntimeError('no tracked encoder parameter update')
                print(f'update={step}/{per_epoch*args.epochs} loss={loss.item():.6f} grad={norm.item():.4f} '
                      f'parameter_delta={delta:.6g} elapsed_s={elapsed:.1f} peak_GiB={torch.cuda.max_memory_allocated()/2**30:.2f}',flush=True)
        score,metrics = validation(model,val,vid,units,args.batch,loss_fn)
        if not math.isfinite(score):
            raise FloatingPointError('invalid validation score')
        state = dict(epoch=epoch,updates=step,model=model.state_dict(),optimizer=optimizer.state_dict(),
                     generator=gen.get_state(),config=config,validation_score=score)
        atomic_save(state,args.out/'last.pt')
        if score<best:
            best = score
            atomic_save(state,args.out/'best.pt')
            pd.DataFrame(metrics).to_csv(args.out/'best_metrics.csv',index=False)
        with (args.out/'epochs.jsonl').open('a') as f:
            print(json.dumps(dict(epoch=epoch,updates=step,validation_score=score,metrics=metrics,
                                  elapsed_s=time.monotonic()-train_start)),file=f)
        print(f'epoch={epoch} validation={score:.6g} best={best:.6g} elapsed_s={time.monotonic()-train_start:.1f}',flush=True)
    (args.out/'completed.json').write_text(json.dumps(dict(updates=step,best=best,seconds=time.monotonic()-start)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--model',choices=['slowfast','single','gru'],required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--epochs',type=int,default=24)
    parser.add_argument('--batch',type=int,default=1024)
    parser.add_argument('--seed',type=int,default=17)
    run(parser.parse_args())
