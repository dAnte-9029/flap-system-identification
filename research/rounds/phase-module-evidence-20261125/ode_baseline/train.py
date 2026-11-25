"""Train and export an independent controlled Neural ODE on whole-flight original eight targets."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from model import ControlledODE

HERE=Path(__file__).resolve().parent
SEED=17
BS=512
# The tape records actual command timestamps in integer microseconds, with four values per slot.
MAX_EDGES=160
NAMES=("v05","w05","v1","w1","v2","w2")


def segment(tape, times):
    flat=tape.reshape(-1,5)
    arr=flat[flat[:,4]>0]
    dt=arr[:,4].astype(np.float64)
    bounds=dt.cumsum()
    switches=np.any(arr[1:,:4]!=arr[:-1,:4],axis=1)
    q=np.rint(times.astype(np.float64)*1e6).astype(np.int64)
    if q.max()>bounds[-1]+1 or q.min()<=0: raise ValueError("Invalid query or insufficient command prefix")
    boundaries=np.unique(np.r_[bounds[:-1][switches],q.ravel()])
    boundaries=boundaries[boundaries<=q.max()]
    midpoint=(np.r_[0.,boundaries[:-1]]+boundaries)*.5
    controls=arr[np.searchsorted(bounds,midpoint,side="right"),:4]
    idx=np.searchsorted(boundaries,q)
    np.testing.assert_array_equal(boundaries[idx],q)
    return controls,boundaries*1e-6,idx


def load(root,part,device):
    with np.load(root/"outputs/autopilot_flight_dynamics_20261124"/f"{part}.npz") as file:
        hist,y,mask,clock,ids,flights=[file[k] for k in ("hist","y","mask","time","window","flight")]
    repo=root/"outputs/controlled_slow_fast_20261119_tapes"
    tape=np.load(repo/f"{part}_commands.npy",mmap_mode="r")
    arr_ids=pd.read_parquet(repo/f"{part}_ids.parquet",columns=["window_id"]).window_id
    np.testing.assert_array_equal(ids,arr_ids)
    aug=np.load(root/"outputs/autopilot_periodic_motion_20261125"/f"{part}_features.npy")
    hist=np.c_[hist,aug]
    n=len(ids); max_edges=0
    cmds=np.zeros((n,MAX_EDGES,4),np.float32)
    end=np.zeros((n,MAX_EDGES),np.float32); durations=np.zeros_like(end)
    query=np.zeros((n,8,2),np.uint8)
    lengths=[]
    for i in range(n):
        val,edges,ii=segment(tape[i],clock[i])
        k=len(edges)
        if k>MAX_EDGES:raise RuntimeError(f"Need {k} native segments")
        cmds[i,:k]=val
        end[i,:k]=edges;end[i,k:]=edges[-1]
        durations[i,:k]=np.diff(np.r_[0.,edges])
        query[i]=ii
        lengths.append(k);max_edges=max(max_edges,k)
    inputs={k:torch.as_tensor(v,device=device) for k,v in dict(hist=hist,command=cmds[:,:max_edges],
           end=end[:,:max_edges],dt=durations[:,:max_edges],query=query,y=y,mask=mask).items()}
    info={"split":part,"windows":n,"flights":len(set(flights)),"median_segments":float(np.median(lengths)),
          "max_segments":max_edges,"max_dt":float(durations.max())}
    print(json.dumps(info),flush=True)
    return inputs,clock,ids,flights,info


def args(d,ix):
    return d["hist"][ix],d["command"][ix],d["end"][ix],d["dt"][ix],d["query"][ix].long()


@torch.inference_mode()
def evaluate(model,d,weights,scale,output=False):
    model.eval(); score=torch.zeros((),dtype=torch.float64,device=scale.device); preds=[]
    for i in range(0,len(weights),512):
        ix=slice(i,i+512)
        pred=model(*args(d,ix))
        err=((pred-d["y"][ix])/scale).square().sum(-1)*d["mask"][ix]
        score+=(err.sum((1,2)).double()*weights[ix]).sum()
        if output:preds.append(pred.cpu().numpy())
    model.train()
    return float(score),np.concatenate(preds) if output else None


def monitor_weights(flights,mask,device):
    ids,inv=np.unique(flights,return_inverse=True)
    count=np.bincount(inv,weights=mask.sum((1,2)).cpu().numpy())
    return torch.as_tensor(1/(len(ids)*3*count[inv]),device=device)


def probe(model,dev):
    h,u,end,dt,q=args(dev,slice(0,48))
    with torch.no_grad():
        base=model(h,u,end,dt,q)
        idx=torch.arange(u.shape[1],device=u.device)[None,:]
        cut=q[:,:4].amax((1,2))[:,None]
        alt=torch.where((idx>cut)[:,:,None],-u,u)
        prefix=(model(h,alt,end,dt,q)[:,:4]-base[:,:4]).abs().max()
        refined_dt=(dt.unsqueeze(-1).expand(-1,-1,2)*.5).flatten(1)
        refined_end=torch.stack((end-dt*.5,end),-1).flatten(1)
        refined_u=u.unsqueeze(2).expand(-1,-1,2,-1).flatten(1,2)
        refined=model(h,refined_u,refined_end,refined_dt,2*q+1)
        error=(refined-base).abs()
        assert prefix < 1e-5 and torch.isfinite(error).all()
        return dict(future_suffix_max=float(prefix),half_step_max=float(error.max()),
                    half_step_rms=float(error.square().mean().sqrt()),windows=len(h))


def run(root,device,maxepochs):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    started=time.monotonic()
    train,_,_,tflights,ta=load(root,"train",device)
    dev,times,ids,flights,va=load(root,"validation",device)
    assert (ta["flights"],ta["windows"],va["flights"],va["windows"])==(121,126237,39,44524)
    assert set(tflights).isdisjoint(flights)
    weights=monitor_weights(flights,dev["mask"],device)
    mask=train["mask"].unsqueeze(-1)
    scale=(train["y"].square().mul(mask).sum((0,1))/mask.sum((0,1))).sqrt().clamp_min(.05)
    h=train["hist"];model=ControlledODE(h.mean(0),h.std(0).clamp_min(.05)).to(device)
    initial=[p.detach().clone() for p in model.parameters()]
    pre=probe(model,dev)
    lr=.001;floor=.0000625
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001,fused=True)
    best=float("inf");reference=float("inf");stale=0;history=[];step=0
    n=len(h);nsteps=(n+BS-1)//BS
    run_start=time.monotonic();generator=torch.Generator(device=device).manual_seed(SEED)
    print("ready",nsteps,"batches/epoch;",sum(p.numel() for p in model.parameters()),"parameters",
          "data elapsed",run_start-started,"probe",pre,flush=True)
    for epoch in range(1,maxepochs+1):
        order=torch.randperm(n,device=device,generator=generator)
        # The padded tail carries zero loss, so each real training window counts once per epoch.
        pad=nsteps*BS-n
        if pad:order=torch.cat((order,order[:pad]))
        total=torch.zeros((),device=device)
        for j,ix in enumerate(order.split(BS)):
            pred=model(*args(train,ix));ismask=train["mask"][ix].clone()
            if pad and j==nsteps-1:ismask[-pad:]=False
            loss=(((pred-train["y"][ix])/scale).square()*ismask[:,:,:,None]).sum()/(3*ismask.sum())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step();total+=loss.detach();step+=1
            if step==20:
                torch.cuda.synchronize()
                print("20 real updates in",time.monotonic()-run_start,"s; loss",float(loss.detach()),flush=True)
        score,_=evaluate(model,dev,weights,scale)
        if not np.isfinite(score):raise RuntimeError("Nonfinite training")
        row=dict(epoch=epoch,step=step,learning_rate=lr,train=float(total/nsteps),validation=score,
                 elapsed=time.monotonic()-run_start)
        history.append(row);print(json.dumps(row),flush=True)
        (HERE/"curve.json").write_text(json.dumps(history,indent=2)+"\n")
        if score<best:
            best=score;torch.save(model.state_dict(),HERE/"best.pt")
        if score<reference*(1-.001):
            reference=score;stale=0
        else:stale+=1
        if lr>floor and stale>=5:
            lr=max(floor,lr*.5);opt.param_groups[0]["lr"]=lr;stale=0;reference=best
        elif lr==floor and stale>=8 and epoch>=32:
            reason="minimum_learning_rate_validation_plateau";break
    else:reason="budget_cap"
    torch.save(model.state_dict(),HERE/"terminal.pt")
    delta=sum((p.detach()-a).square().sum() for p,a in zip(model.parameters(),initial)).sqrt().item()
    assert delta>0
    model.load_state_dict(torch.load(HERE/"best.pt",map_location=device,weights_only=True))
    post=probe(model,dev)
    final,pred=evaluate(model,dev,weights,scale,True)
    assert abs(final-best)<1e-6 and np.isfinite(pred).all()
    np.savez_compressed(HERE/"predictions.npz",window_id=ids,flight_id=flights,time=times,predicted_delta=pred)
    valmask=dev["mask"].cpu().numpy()[:,[1,3,7]].reshape(-1,6)
    sq=np.square(pred-dev["y"].cpu().numpy()).sum(-1)[:,[1,3,7]].reshape(-1,6)
    rows=[dict(flight_id=f,**{m:sq[(flights==f)&valmask[:,j],j].mean(dtype=np.float64)
                              for j,m in enumerate(NAMES)}) for f in np.unique(flights)]
    frame=pd.DataFrame(rows);frame.to_csv(HERE/"flight_mse.csv",index=False)
    metrics={m:float(frame[m].mean()**.5) for m in NAMES}
    result=dict(seed=SEED,parameters=sum(p.numel() for p in model.parameters()),train=ta,validation=va,epochs=epoch,
                steps=step,parameter_change_l2=delta,stop=reason,best_epoch=1+int(np.argmin([r["validation"] for r in history])),
                best_loss=best,metrics=metrics,preflight=pre,checkpoint_check=post,training_wall_s=history[-1]["elapsed"])
    (HERE/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",type=Path,default=Path("/home/zn/flap-system-identification"))
    ap.add_argument("--device",default="cuda:0");ap.add_argument("--max-epochs",type=int,default=200)
    args_=ap.parse_args();run(args_.data_root,args_.device,args_.max_epochs)
