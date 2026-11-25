"""Score and target-align the prior causal TCN on exactly the current eight native queries."""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
MAIN = Path('/home/zn/flap-system-identification')
HISTORY = MAIN / 'outputs/history_slow_fast_20261119_cache'
TAPES = MAIN / 'outputs/controlled_slow_fast_20261119_tapes'
BASE = MAIN / 'outputs/autopilot_flight_dynamics_20261124'
SUITE = MAIN / 'outputs/controlled_baselines_20261119_run1'
sys.path.insert(0, str(MAIN / '.worktrees/transformer-slowfast-v1/src'))
sys.path.insert(0, str(SUITE / 'source/controlled_slow_fast_v1'))
from fit import make_model
from dynamics import query, rotate, inverse

NAMES = ('v05','w05','v1','w1','v2','w2')
BATCH = 1024
START_LR = .0003
MIN_LR = .0000375


def data(split):
    assert split in ('train', 'validation')
    windows = pd.read_parquet(HISTORY/f'{split}_windows.parquet').set_index('window_id', verify_integrity=True)
    order = pd.read_parquet(TAPES/f'{split}_ids.parquet').window_id
    ix = windows.index.get_indexer(order)
    assert order.is_unique and (ix>=0).all()
    with np.load(BASE/f'{split}.npz') as arr:
        assert np.array_equal(arr['window'], order) and np.array_equal(arr['flight'], windows.flight_id.iloc[ix])
        ts, ys, valid = (arr[k] for k in ('time','y','mask'))
    get = lambda name: np.load(HISTORY/f'{split}_{name}.npy',mmap_mode='r')
    if split == 'validation':
        native = get('t')
        raw = get('y')
        ok = get('valid')
        init = get('initial')
        tdiff = targetdiff = 0.
        for start in range(0,len(ix),1024):
            sl=slice(start,start+1024)
            subix = ix[sl]
            initial = init[subix]
            frame = Rotation.from_quat(initial[:,[7,8,9,6]])
            r = np.arange(len(subix))
            time_raw, target_raw, isvalid = native[subix], raw[subix], ok[subix]
            for step,h in enumerate(np.arange(1,9)/4):
                for c,ch in enumerate((0,2)):
                    nearest = np.where(isvalid[:,:,ch],abs(time_raw[:,:,ch]-h),np.inf).argmin(1)
                    tdiff = max(tdiff,float(abs(time_raw[r,nearest,ch]-ts[sl,step,c]).max()))
                    target = (frame.apply(target_raw[r,nearest,3:6]-initial[:,3:6],inverse=True)
                              if c==0 else target_raw[r, nearest,10:13]-initial[:,10:13])
                    targetdiff=max(targetdiff,float(abs(target-ys[sl,step,c]).max()))
        assert tdiff==0 and targetdiff<1e-6 and valid.all(),(tdiff,targetdiff,valid.sum())
        validation = dict(time_max_abs_s=tdiff, target_max_abs=targetdiff, valid_integer_count=int(valid.sum()),
                          windows=int(len(ix)), flights=int(windows.flight_id.iloc[ix].nunique()))
    else:
        validation = {}
    tape=np.load(TAPES/f'{split}_commands.npy',mmap_mode='r')
    assert np.all(ts.max((1,2)) <= tape[...,4].sum((1,2))*1e-6+1e-6)
    arrays = [get('history')[ix], get('initial')[ix], np.array(tape), ts, ys, valid]
    return [torch.from_numpy(a).cuda() for a in arrays],windows.flight_id.iloc[ix].to_numpy(),validation


def square_errors(model, batch):
    hist, initial, tape, t, y, _ = batch
    rollout = model(hist, initial, tape)
    vel = query(rollout, t[...,0], 'pv')[...,3:]
    rate = query(rollout, t[...,1], 'w')
    rotated = rotate(inverse(initial[:,None,6:10]), vel-initial[:,None,3:6])
    pred = torch.stack((rotated, rate-initial[:,None,10:13]),dim=2)
    return (pred-y).square().sum(-1)


def run_score(model, dev, ids, units):
    model.eval()
    with torch.no_grad():
        static = [a[:BATCH].clone() for a in dev]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            square_errors(model,static)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sq = square_errors(model,static)
        n=len(ids)
        def score():
            err = []
            for start in range(0,n,BATCH):
                stop=min(start+BATCH,n)
                for buf, source in zip(static,dev):
                    buf[:stop-start].copy_(source[start:stop])
                graph.replay()
                err.append(sq[:stop-start].cpu().numpy())
            e=np.concatenate(err).astype(np.float64)
            mask=dev[-1].cpu().numpy()
            assert mask.all()
            keys=np.unique(ids)
            byflight = {k:e[ids==k].mean(0) for k in keys}
            all_mse=np.stack(list(byflight.values()))
            report={name:{f:float(s[[1,3,7]].reshape(-1)[j]) for f,s in byflight.items()}
                    for j,name in enumerate(NAMES)}
            scalar=float((all_mse/units.cpu().numpy()**2).mean())
            return scalar,report
        return score


def save_json(path,obj):
    path.write_text(json.dumps(obj,indent=2)+'\n')


def hydrate(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    state=ck['model']
    model=make_model('tcn', state['mean'], state['scale']).cuda()
    model.load_state_dict(state)
    return ck,model


def configure():
    torch.cuda.set_device(1)
    torch.manual_seed(17)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32=True


def main():
    configure()
    ck, model=hydrate(SUITE/'tcn/best.pt')
    assert ck['epoch']==24 and ck['updates']==2976 and ck['config']['seed']==17
    units=torch.tensor(ck['config']['units'],device='cuda')[[1,3]]
    dev, vid, audit = data('validation')
    score=run_score(model,dev,vid,units)
    raw_score, archive=score()
    ref=pd.read_csv(SUITE/'tcn/best_metrics.csv')
    old={name:float(np.sqrt(np.mean(list(rows.values())))) for name,rows in archive.items()}
    for k,metric,h in zip(NAMES,('velocity_m_s','rate_rad_s')*3,(.5,.5,1.,1.,2.,2.)):
        v=ref.query('metric==@metric and horizon_s==@h').flight_equal_rmse.item()
        np.testing.assert_allclose(old[k],v,atol=1e-5,rtol=0)
        audit.setdefault("archive_metric_signed_replay_difference", {})[k] = old[k] - v
    save_json(HERE/'archived_flight_mse.json',archive)
    save_json(HERE/'archived.json',dict(metrics=old,matched_validation_loss=raw_score,
        native_check=audit,weights=str(SUITE/'tcn/best.pt'),epoch=ck['epoch'],updates=ck['updates']))
    print('restored',old,'aligned loss',raw_score,flush=True)
    train, tids, _=data('train')
    assert len(tids)==126237 and len(set(tids))==121 and set(tids).isdisjoint(vid)
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=START_LR,weight_decay=.0001,fused=True)
    generator=torch.Generator(device='cuda').manual_seed(17)
    static=[a[:BATCH].clone() for a in train]
    def backward():
        sq=square_errors(model, static)
        mask=static[-1]
        loss=((sq / units.square())*mask).sum()/mask.sum()
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
    epochs=160
    best=reference=raw_score
    best_epoch=0
    n=len(tids)
    size=math.ceil(n/BATCH)
    lr=START_LR
    stale=0
    start=time.monotonic()
    traced=next(model.backbone.parameters())
    initial=traced.detach().clone()
    for epoch in range(1,epochs+1):
        perm=torch.randperm(n,generator=generator,device='cuda')
        schedule=torch.cat((perm,perm[:size*BATCH-n]))
        for i,ix in enumerate(schedule.split(BATCH)):
            for dst,src in zip(static, train):
                torch.index_select(src,0,ix,out=dst)
            graph.replay()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not bool(torch.isfinite(loss)&torch.isfinite(norm)):
                raise FloatingPointError(f'nonfinite at {epoch}/{i}')
            optimizer.step()
        validation, perflight=score()
        improved=validation<best
        if improved:
            best=validation
            best_epoch=epoch
            torch.save(model.state_dict(), HERE/'best.pt')
            save_json(HERE/'best_flight_mse.json',perflight)
        row=dict(epoch=epoch,steps=epoch*size,lr=lr,validation=validation,best=best,best_epoch=best_epoch,
                 train_last_batch=float(loss.detach()),encoder_parameter_change=float((traced-initial).abs().max()),
                 elapsed_s=time.monotonic()-start)
        if validation < reference*(1-.001):
            reference=validation
            stale=0
        else:
            stale+=1
        stop=lr==MIN_LR and stale>=8
        row['stale']=stale
        with (HERE/'continue_epochs.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(row,flush=True)
        if stop or epoch==epochs:
            torch.save(model.state_dict(),HERE/'terminal.pt')
            save_json(HERE/'terminal_flight_mse.json',perflight)
            save_json(HERE/'stop.json',dict(**row,reason='plateau' if stop else 'budget',
                                           original_steps=ck['updates']))
            break
        if stale>=6 and lr>MIN_LR:
            lr=max(MIN_LR,lr/2)
            optimizer.param_groups[0]['lr']=lr
            reference=best
            stale=0
    assert row['encoder_parameter_change']>0


if __name__=='__main__':
    main()
