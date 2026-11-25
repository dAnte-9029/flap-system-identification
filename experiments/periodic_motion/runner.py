"""Paired augmented-native training. Fixed endpoint scoring reuses the original flight evaluator."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
FLIGHT=HERE.parent/"autopilot_flight_dynamics"
sys.path.insert(0,str(FLIGHT))
spec=importlib.util.spec_from_file_location("fixed_flight_runner",FLIGHT/"runner.py")
fixed=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)
old_load=fixed.load
from model import FlightPredictor

CACHE=Path("/home/zn/flap-system-identification/outputs/autopilot_periodic_motion_20261125")
COMMON=49152
FLOOR=.000125


def load(part,*,monitor=False):
    b,ids=old_load(part,monitor=monitor)
    meta=json.loads((CACHE/"ready.json").read_text())[part]
    features=np.load(CACHE/f"{part}_features.npy")
    assert len(features)==len(ids)==meta["windows"]
    b["hist"]=torch.cat((b["hist"],torch.from_numpy(features).to("cuda")),dim=1)
    return b,ids


fixed.load=load


def grids(train):
    out={}
    for key in ("time","pooled","y","mask"):
        more=np.load(CACHE/f"train_{key}.npy",mmap_mode="c")
        out[key]=torch.cat((train[key][:,None],torch.from_numpy(more).to("cuda")),dim=1)
        assert out[key].shape[1]==7
    return out


def preflight():
    fixed.preflight()
    b,_=load("train")
    m=FlightPredictor(b["hist"].mean(0),b["hist"].std(0).clamp_min(.05)).to("cuda")
    assert b["hist"].shape[1]==194
    ix=slice(0,13)
    args=[b[k][ix] for k in fixed.INPUTS]
    pred=m(*args)
    assert torch.isfinite(pred).all()
    if hasattr(m,"components"):
        s,f,o=m.components(*args)
        torch.testing.assert_close(s+f-o,pred,atol=2e-6,rtol=2e-6)
    print("Extended-history and native mask checks passed",flush=True)


def train():
    seed=int(os.environ["RA_SEED"])
    cap=int(os.environ["RA_STEPS"])
    rule=json.loads(os.environ["RA_TRAINING_COMPLETION"])
    assert rule["metric"]=="dev_mse" and rule["eval_every"]==512
    torch.manual_seed(seed)
    b,train_ids=load("train")
    dev,ids=load("validation",monitor=True)
    assert set(ids).isdisjoint(train_ids)
    dense=grids(b)
    unique,inverse=np.unique(ids,return_inverse=True)
    counts=np.bincount(inverse,weights=dev["mask"].sum((1,2)).cpu().numpy())
    weights=torch.from_numpy(1/(3*len(unique)*counts[inverse])).to("cuda")
    x,y,mask=b["hist"],b["y"],b["mask"].unsqueeze(-1)
    mean,std=x.mean(0),x.std(0).clamp_min(.05)
    model=FlightPredictor(mean,std).to("cuda")
    scale=((y.square()*mask).sum((0,1))/mask.sum((0,1))).sqrt().clamp_min(.05)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=3,threshold=.001,threshold_mode="abs",min_lr=FLOOR)
    gen=torch.Generator().manual_seed(seed)
    init=[p.detach().clone() for p in model.parameters()]
    history=[]; losses=[]
    root=Path(os.environ["RA_OUTPUT"])
    first=COMMON
    for step in range(1,cap+1):
        ix=torch.randint(len(y),(384,),generator=gen).to("cuda")
        mix=torch.randint(12,(384,),generator=gen).to("cuda")
        g=(mix-5).clamp_min(0)
        pred=model(x[ix],dense["pooled"][ix,g],b["u0"][ix],dense["time"][ix,g])
        m=dense["mask"][ix,g].unsqueeze(-1)
        loss=(((pred-dense["y"][ix,g])/scale).square()*m).sum()/(3*m.sum())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
        opt.step()
        losses.append(float(loss.detach()))
        if step%512: continue
        score=fixed.development_loss(model,dev,weights,scale)
        assert np.isfinite(score)
        hist=dict(step=step,dev_mse=score,train_mse=float(np.mean(losses[-50:])),lr=opt.param_groups[0]["lr"])
        print(json.dumps(hist),flush=True)
        history.append(hist)
        sched.step(score)
        if step==first:
            torch.save(model.state_dict(),root/"step49152.pt")
        tail=history[-16:]
        stable=(len(history)>16 and all(r["lr"]==FLOOR for r in tail)
                and max(r["dev_mse"] for r in tail)-min(r["dev_mse"] for r in tail)<=rule["max_range"]
                and min(r["dev_mse"] for r in history[:-16])-min(r["dev_mse"] for r in tail)<=.001)
        if step>=COMMON and stable and step<cap: break
    delta=sum((p.detach()-v).square().sum() for p,v in zip(model.parameters(),init)).sqrt().item()
    assert delta>0 and np.isfinite(losses).all()
    torch.save(model.state_dict(),os.environ["RA_MODEL"])
    Path(os.environ["RA_TRAIN_INFO"]).write_text(json.dumps(dict(steps=step,samples=step*384,validation_history=history,
        train_loss=float(np.mean(losses[-50:])),parameter_change_l2=delta,parameters=sum(p.numel() for p in model.parameters()),
        stop=("sustained_plateau" if step<cap and stable else "budget_cap"),common_step=COMMON,
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),gpu=torch.cuda.get_device_name())))
    print("Completed",step,"parameter change",delta,flush=True)


if __name__=="__main__":
    torch.set_num_threads(2)
    {"train":train,"preflight":preflight,"evaluate":fixed.evaluate}[sys.argv[1]]()
