"""Matched ten-pass Adam schedule, with validation-only checkpoint selection."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from arm import ARM
from data import CACHE, FIELDS, OUTPUT, ROOT, DT, H, write
from network import Forecast, FlightPS, AUTHOR

SCALES=np.load(CACHE/"scaling.npz")
BATCH=256


class FlightSet:
    def __init__(self,part,device,fit=False):
        self.device=device
        d=CACHE/part
        f=FIELDS/part
        def get(path):
            return torch.tensor(np.load(path,mmap_mode="r"),device=device)
        self.ids=pd.read_parquet(f/"ids.parquet")
        self.std=torch.tensor(SCALES["std"],device=device,dtype=torch.float32).view(1,1,6)
        mean=torch.tensor(SCALES["mean"],device=device,dtype=torch.float32).view(1,1,6)
        absolute=get(d/"origin.npy").reshape(-1,1,6)
        self.x=(get(f/"origin.npy").transpose(1,2)+absolute-mean)/self.std
        self.time=get(f/"time.npy")
        self.query_u=get(d/"query_u.npy")
        self.endpoint=get(f/"y.npy")
        if fit:
            for name in ("t","y","valid","u","regular","eligible","regular_u","sparse"):
                setattr(self,name,get(d/f"{name}.npy"))


def preflight():
    assert ARM in ("mse","ps","command_ps")
    assert subprocess.check_output(["git","-C",str(AUTHOR/"official"),"rev-parse","HEAD"],text=True).strip()=="c64db5b777ec01013423bcca772e162dbdcd2b49"
    assert (SCALES["std"]>0).all()
    d=FlightSet("validation","cpu")
    torch.manual_seed(17)
    m=Forecast(ARM=="command_ps")
    y=m.at(m.grid(d.x[:3]),d.time[:3],d.query_u[:3])
    assert y.shape==(3,8,2,3) and y.isfinite().all()
    # A change to the last command segment cannot change any earlier integral.
    from data import integral
    tape=np.array([[[.2,.1,0.,1.,500000],[.2,.3,0.,1.,500000]]])
    q=np.array([[.25,.5,.9]]); first=integral(tape,q)
    tape[0,1,0]+=1
    after=integral(tape,q)
    np.testing.assert_array_equal(first[:,:2],after[:,:2])
    np.testing.assert_allclose(after[0,2,0]-first[0,2,0],.4,atol=1e-6)
    if ARM!="mse":
        # On supported patches, the adapted splitter must reproduce the authors' loss and gradients.
        from exp.exp_main import Exp_Main
        ps=FlightPS(m.core)
        x=d.x[:16]; target=m.grid(x)[:,1:96].detach()+.15*torch.sin(torch.arange(95)/8)[None,:,None]
        pred=m.grid(x)[:,1:96]
        a=ps.ps_loss(target,pred)
        orig=object.__new__(Exp_Main);orig.model=m.core;orig.kl_loss=ps.kl_loss;orig.patch_len_threshold=24
        b=orig.ps_loss(target,pred)
        ga=torch.autograd.grad(a,m.core.Linear_Trend.weight,retain_graph=True)[0]
        gb=torch.autograd.grad(b,m.core.Linear_Trend.weight)[0]
        torch.testing.assert_close(a,b);torch.testing.assert_close(ga,gb)
    print("preflight ok", ARM, flush=True)


def monitor_indices(flights):
    rng=np.random.default_rng(17)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(flights==f),min(64,(flights==f).sum()),replace=False)
                                   for f in np.unique(flights)]))


@torch.no_grad()
def monitor(model,data,idx):
    vals=[]
    for p in idx.split(512):
        pred=model.at(model.grid(data.x[p]),data.time[p],data.query_u[p])
        y=data.endpoint[p]/data.std.view(1,1,2,3)
        vals.append((pred-y).square().mean((1,2,3)).cpu())
    return float(torch.cat(vals).mean())


def train():
    seed=int(os.environ.get("RA_SEED","17"))
    torch.manual_seed(seed)
    device=torch.device("cuda:0")
    data=FlightSet("train",device,fit=True)
    dev=FlightSet("validation",device)
    model=Forecast(ARM=="command_ps").to(device)
    ps=FlightPS(model.core)
    opt=torch.optim.Adam(model.parameters(),lr=.005)
    scale=data.std.reshape(1,1,2,3)
    rng=torch.Generator().manual_seed(seed)
    idx=torch.as_tensor(monitor_indices(dev.ids.flight_id.to_numpy()),device=device)
    parameter0=[p.detach().clone() for p in model.parameters()]
    history=[];best=float("inf"); updates=0
    began=time.monotonic()
    path=Path(os.environ["RA_MODEL"])
    for epoch in range(1,11):
        model.train()
        summed=torch.zeros((),device=device)
        for ix in torch.randperm(len(data.x),generator=rng).split(BATCH):
            ix=ix.to(device)
            grid=model.grid(data.x[ix])
            pred=model.at(grid,data.t[ix],data.u[ix])
            valid=data.valid[ix]
            sq=(pred-data.y[ix]/scale).square().mean(-1)
            mse=((sq*valid).sum(1)/valid.sum(1).clamp_min(1)).mean()
            loss=mse
            if ARM!="mse":
                pg=model.regular(grid,data.regular_u[ix])
                yg=(data.regular[ix]/scale).flatten(2)
                complete=data.eligible[ix].all(1)
                structural=mse.new_zeros(())
                count=0
                for group,sl in ((False,slice(None)),(True,slice(4,None,5))):
                    chosen=(data.sparse[ix]==group)&complete
                    n=int(chosen.sum())
                    if n:
                        structural=structural+n*ps.ps_loss(yg[chosen,sl],pg[chosen,sl])
                        count+=n
                loss=mse+3*structural/max(count,1)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            summed+=loss.detach();updates+=1
        model.eval()
        value=monitor(model,dev,idx)
        average=float(summed)/((len(data.x)+BATCH-1)//BATCH)
        assert np.isfinite([value,average]).all()
        history.append(dict(step=updates,epoch=epoch,val_loss=value,train_loss=average,learning_rate=opt.param_groups[0]["lr"]))
        if value<best:
            best=value;best_epoch=epoch
            torch.save(dict(arm=ARM,seed=seed,model=model.state_dict(),scale=SCALES["std"].tolist(),mean=SCALES["mean"].tolist(),best_epoch=epoch),path)
        for group in opt.param_groups: group["lr"]=.005*.5**(epoch-1)
        print(ARM,epoch,updates,value,round(time.monotonic()-began,1),flush=True)
    movement=sum((a-b).square().sum().item() for a,b in zip(parameter0,model.parameters()))**.5
    assert movement>0 and updates==4940 and path.is_file()
    write(Path(os.environ["RA_TRAIN_INFO"]),dict(steps=updates,validation_history=history,epoch_count=10,
        best_epoch=best_epoch, best_validation=best, parameter_change=movement, seconds=time.monotonic()-began,
        ps_calls=ps.calls,min_patch_clamps=ps.clamped, training_windows=len(data.x),gpu=torch.cuda.get_device_name()))


if __name__=="__main__":
    torch.set_num_threads(2)
    preflight() if sys.argv[1]=="preflight" else train()
