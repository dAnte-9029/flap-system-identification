"""Fit and score the predeclared crossed scale-message ablation on native endpoints."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml

from module import ScaleExchange, encode_history, features
PROJECT = Path(__file__).resolve().parents[2]
FROZEN = PROJECT/'research/rounds/phase-module-evidence-20261125'
sys.path.insert(0,str(FROZEN/'ode_baseline'))
from train import args as ode_args, ControlledODE, NAMES, load
ARMS = ('fixed_self','clock_self','fixed_cross','full','clock_deleted')
METRICS = list(NAMES)


def prepare(root, split, device, model):
    data, time, ids, flights, info = load(root,split,device)
    folder = root/'outputs/history_slow_fast_20261119_cache'
    meta = pd.read_parquet(folder/f'{split}_windows.parquet').set_index('window_id',verify_integrity=True)
    ix = meta.index.get_indexer(ids)
    assert (ix>=0).all() and pd.Index(ids).is_unique
    np.testing.assert_array_equal(meta.iloc[ix].flight_id,flights)
    main = meta.iloc[ix].sampling_group.to_numpy()=='main_50hz'
    history = np.load(folder/f'{split}_history.npy',mmap_mode='r')
    base,coeff,eligible = [],[],[]
    for i in range(0,len(ids),512):
        sl = slice(i,i+512)
        base.append(model(*ode_args(data,sl)))
        c,ok = encode_history(history[ix[sl]],main[sl])
        coeff.append(c)
        eligible.append(ok)
    clock = torch.as_tensor(time,device=device)
    freq = data['hist'][:,165].contiguous()
    assert data['mask'].all() and (data['hist'][:,172]>0).all() and (freq>0).all()
    h = data['hist'][:,:180].reshape(-1,6,30)
    motor = h[:,-1,9].contiguous()
    coeff = torch.as_tensor(np.concatenate(coeff),device=device)
    valid = torch.as_tensor(np.concatenate(eligible),device=device)
    result = dict(coeff=coeff, valid=valid, frequency=freq, motor=motor, time=clock)
    info['eligible'] = valid.sum(0).tolist()
    info['flights'] = len(np.unique(flights))
    # Check prefix causality of the actual baseline on observed commands at native query positions.
    sl = slice(0,64)
    hist,u,end,dt,q = ode_args(data,sl)
    cut = q[:,3].amax(-1)
    suffix = torch.arange(u.shape[1],device=device)[None]>cut[:,None]
    alt = model(hist,torch.where(suffix[...,None],-u,u),end,dt,q)
    prefix_delta = (alt[:,:4]-model(hist,u,end,dt,q)[:,:4]).abs().max().item()
    assert prefix_delta < 1e-5
    info['base_suffix_prefix_max'] = prefix_delta
    return result,torch.cat(base),data['y'],flights,info


def fit(data,base,y,mean,arm):
    dims = 33 if arm.endswith('deleted') else 51
    gram = torch.zeros((2,3,dims,dims),dtype=torch.float64,device=y.device)
    rhs = torch.zeros_like(gram[...,0])
    for i in range(0,len(y),512):
        sl = slice(i,i+512)
        x = features(**{k:a[sl] for k,a in data.items()},mean_frequency=mean,arm=arm).double()
        x = x.permute(2,3,0,1,4).flatten(2,3)
        error = (y[sl]-base[sl]).permute(2,3,0,1).flatten(2,3).double()
        gram += x.transpose(-1,-2) @ x
        rhs += (x*error[...,None]).sum(-2)
    matrix = gram+1e-4*torch.diag_embed(gram.diagonal(dim1=-2,dim2=-1).clamp_min(1e-8))
    w = torch.linalg.solve(matrix,rhs[...,None]).squeeze(-1)
    residual = float(((matrix@w[...,None]).squeeze(-1)-rhs).abs().max()/rhs.abs().max())
    assert residual<1e-8
    return w.float(),residual


def score(pred,y,flights):
    e = (pred-y).square().sum(-1)[:,[1,3,7]].flatten(1).cpu().numpy()
    return pd.DataFrame([dict(flight=f,**dict(zip(METRICS,e[flights==f].mean(0,dtype=np.float64))))
                         for f in np.unique(flights)]).set_index('flight')


def paired(a,b,draws):
    old,new = a[METRICS].to_numpy(), b[METRICS].to_numpy()
    assert a.index.equals(b.index)
    diff = np.sqrt(old[draws].mean(1))-np.sqrt(new[draws].mean(1))
    intervals = np.quantile(diff,[.025,.975],axis=0)
    return {key:dict(gain=float(np.sqrt(old[:,i].mean())-np.sqrt(new[:,i].mean())),
                     ci95=intervals[:,i].tolist(),wins=int((old[:,i]>new[:,i]).sum()))
            for i,key in enumerate(METRICS)}


@torch.inference_mode()
def main(root,out,device):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    registry = yaml.safe_load((PROJECT/'configs/data/velocity_rate_dataset_registry.yaml').read_text())
    assert registry['default_dataset_id']=='velocity_rate_multistep_v1'
    out.mkdir(parents=True,exist_ok=False)
    state = torch.load(FROZEN/'ode_baseline/best.pt',map_location=device,weights_only=True)
    model = ControlledODE(state['mean'],state['scale']).to(device).eval()
    model.load_state_dict(state)
    tr,b,y,ft,it = prepare(root,'train',device,model)
    dv,bv,yv,fv,iv = prepare(root,'validation',device,model)
    assert (len(ft),len(set(ft)),len(fv),len(set(fv)))==(126237,121,44524,39) and set(ft).isdisjoint(fv)
    mean = tr['frequency'].double().mean().float()
    tables = {'baseline':score(bv,yv,fv)}
    ref = pd.read_csv(FROZEN/'ode_baseline/flight_mse.csv').set_index('flight_id').loc[tables['baseline'].index,METRICS]
    replay = float((tables['baseline']-ref).abs().max().max())
    assert replay<1e-5, replay
    # Historical waveform transfer is an additional reference on identical flights.
    old = json.loads((FROZEN/'transfer/ode_result.json').read_text())['flight_mse']
    tables['prior_waveform'] = pd.DataFrame(old['second_round']).loc[ref.index,METRICS]
    residuals = {}
    for arm in ARMS:
        w,residuals[arm] = fit(tr,b,y,mean,arm)
        module = ScaleExchange(w,mean,arm)
        torch.save(module.state_dict(),out/f'{arm}.pt')
        predictions = []
        for i in range(0,len(fv),512):
            sl = slice(i,i+512)
            predictions.append(module(bv[sl], **{k:a[sl] for k,a in dv.items()}))
        pred = torch.cat(predictions)
        assert torch.isfinite(pred).all()
        tables[arm] = score(pred,yv,fv)
    draws = np.random.default_rng(17).integers(0,39,(10000,39))
    comparisons = {ref:paired(table,tables['full'],draws) for ref,table in tables.items() if ref!='full'}
    rmse = {k:table.mean().pow(.5).to_dict() for k,table in tables.items()}
    result = dict(train=it,validation=iv,mean_frequency=float(mean),baseline_mse_replay_max=replay,
                  normal_equation_relative_residuals=residuals,rmse=rmse,reference_minus_full=comparisons)
    pd.concat(tables,names=['arm']).to_csv(out/'flight_mse.csv')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(rmse),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/home/zn/flap-system-identification'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    main(a.root,a.out,a.device)
