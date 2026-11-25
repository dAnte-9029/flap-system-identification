"""Conditional two-harmonic transport in addition to the frozen first-harmonic model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
PRIOR = PROJECT / "research/rounds/periodic-motion-20261125"
sys.path.insert(0, str(PRIOR / "readout"))
from model import FlightPredictor
from native_path import query
sys.path.insert(0, str(PROJECT / "experiments/periodic_motion"))
from prepare_native import pool_commands

SEEDS = (17, 29, 43)
ARMS = ("origin", "nominal", "sham", "full")
METRICS = ("v05", "w05", "v1", "w1", "v2", "w2")


def extract(hist, main):
    time = np.broadcast_to(np.linspace(-1.,0.,hist.shape[1]), hist.shape[:2])
    phase = np.arctan2(hist[...,13],hist[...,14])
    cols = [np.ones_like(time),time]
    for k in (1,2,3):
        cols.extend([np.cos(k*phase),np.sin(k*phase)])
    design = np.stack(cols,-1)
    coef = np.zeros((len(hist),2,3,2,2),np.float32)
    support = np.zeros((len(hist),2),bool)
    for c in (0,1):
        good = (hist[...,21]>0) & (hist[...,16+c]>0)
        if c==0:
            good &= (hist[...,18]>0) & main[:,None]
        masked = design * good[...,None]
        gram = masked.transpose(0,2,1) @ masked
        eig = np.linalg.eigvalsh(gram)
        ok = (good.sum(1)>=16) & (hist[:,-1,21]>0) & (eig[:,0]>.0001*eig[:,-1])
        rhs = masked.transpose(0,2,1) @ hist[...,3*c:3*c+3]
        fit = np.linalg.solve(np.where(ok[:,None,None],gram,np.eye(8)),rhs)
        coef[ok,c] = fit[ok,4:].transpose(0,2,1).reshape(-1,3,2,2)
        support[:,c] = ok
    return coef, support


def load(root, part, device):
    old = root / "outputs/autopilot_flight_dynamics_20261124"
    aug = root / "outputs/autopilot_periodic_motion_20261125"
    source = root / "outputs/history_slow_fast_20261119_cache"
    with np.load(old / f"{part}.npz") as z:
        data = {k:z[k] for k in ("time","pooled","y","mask","hist","u0")}
        win,flight = z["window"],z["flight"]
    meta = pd.read_parquet(source / f"{part}_windows.parquet",
                           columns=["window_id","sampling_group","flight_id"]).set_index("window_id", verify_integrity=True)
    ix = meta.index.get_indexer(win)
    assert np.all(ix>=0)
    np.testing.assert_array_equal(flight,meta.iloc[ix].flight_id)
    main = meta.iloc[ix].sampling_group.to_numpy()=="main_50hz"
    history = np.load(source / f"{part}_history.npy",mmap_mode="r")
    co,support = zip(*(extract(history[ix[i:i+768]],main[i:i+768]) for i in range(0,len(ix),768)))
    data["coef"] = np.concatenate(co)
    data["hist"] = np.c_[data["hist"],np.load(aug / f"{part}_features.npy")]
    data["shift"] = np.array([int.from_bytes(hashlib.sha256(w.encode()).digest()[:4],"little")*(2*np.pi/2**32)
                              for w in win],np.float32)
    if part=="train":
        chosen = np.random.default_rng(17).integers(0,6,len(ix))
        for k in ("time","pooled","y","mask"):
            a = np.load(aug / f"{part}_{k}.npy",mmap_mode="r")
            data[k+"_early"] = a[np.arange(len(ix)),chosen,:2]
    counts = dict(flights=len(np.unique(flight)),windows=len(ix),
                  eligible_history=np.concatenate(support).sum(0).tolist(),
                  original_targets=data["mask"].sum((0,1)).tolist())
    if part=="train": counts["early_targets"]=data["mask_early"].sum((0,1)).tolist()
    return {k:torch.as_tensor(v,device=device) for k,v in data.items()},flight,counts


def design(coef,time,phase,shift,arm):
    # coef = [batch, stream, output axis, harmonic (2/3), A/B]; phase uses radians.
    gate = -torch.expm1(-time/.02)
    p0 = (coef[:,None] * gate[...,None,None,None]).flatten(-2)
    if arm=="origin":
        return p0
    if arm=="nominal":
        theta = time*(2*np.pi*3.8)
    elif arm=="sham":
        theta = phase + shift[:,None,None]
    else:
        assert arm=="full"
        theta = phase
    angle = theta[...,None,None]*time.new_tensor([2.,3.])
    a,b = coef.unbind(-1)
    co,si = torch.cos(angle),torch.sin(angle)
    rotated = torch.stack((a[:,None]*co+b[:,None]*si,-a[:,None]*si+b[:,None]*co),-1)
    envelope = gate[...,None]*torch.exp(-time[...,None]/time.new_tensor([.25,1.,4.]))
    future = (rotated[...,None]*envelope[:,:,:,None,None,None,:]).flatten(-3)
    return torch.cat([p0,future],-1)


def prediction_parts(model,d,sl,suffix):
    hist=d["hist"][sl]; u=d["u0"][sl]; t=d["time"+suffix][sl]; pool=d["pooled"+suffix][sl]
    stripped=hist[:,:model.mean.numel()]
    if suffix:
        core=query(model.core,stripped,u,d["time"][sl],d["pooled"][sl],t,pool)
    else:
        core=model.core(stripped,pool,u,t)
    beta=torch.einsum("bh,ckha->bcka",model.context(hist),model.weights)
    clock=model.clocks(hist,pool,u,t)[...,-1]
    slow=core+(-torch.expm1(-t/.35))[...,None]*beta[:,None,:,0,:]
    wave=torch.exp(-t/2)[...,None]*(clock.cos()[...,None]*beta[:,None,:,1,:]+clock.sin()[...,None]*beta[:,None,:,2,:])
    return slow+(wave-beta[:,None,:,1,:]),clock


def train(model,d):
    normal={};rhs={}
    for start in range(0,len(d["hist"]),768):
        sl=slice(start,start+768)
        for suffix in ("","_early"):
            y=d["y"+suffix][sl];mask=d["mask"+suffix][sl]
            base,clock=prediction_parts(model,d,sl,suffix)
            residual=(y-base).double()
            for arm in ARMS:
                z=design(d["coef"][sl],d["time"+suffix][sl],clock,d["shift"][sl],arm).double()
                z=z*mask[:,:,:,None,None]
                matrix=torch.einsum("bqcxm,bqcxn->cxmn",z,z)
                vector=torch.einsum("bqcxm,bqcx->cxm",z,residual)
                if arm not in normal: normal[arm],rhs[arm]=matrix,vector
                else: normal[arm]+=matrix;rhs[arm]+=vector
    weights={};errors={}
    for k in ARMS:
        reg=normal[k]+1e-5*torch.diag_embed(normal[k].diagonal(dim1=-1,dim2=-2))
        w=torch.linalg.solve(reg,rhs[k][...,None])
        errors[k]=float((reg@w-rhs[k][...,None]).abs().max()/rhs[k].abs().max())
        assert errors[k]<1e-8
        weights[k]=w.squeeze(-1).float()
    return weights,errors


def forecast(base,coef,t,clock,shift,weights):
    preds={"corrected":base}
    for arm in ARMS:
        z=design(coef,t,clock,shift,arm)
        preds[arm]=base+torch.einsum("bqcxm,cxm->bqcx",z,weights[arm])
        if arm=="sham":
            preds["full_wrong_phase"]=base+torch.einsum("bqcxm,cxm->bqcx",z,weights["full"])
        if arm=="origin":
            preds["full_drop_future"]=base+torch.einsum("bqcxm,cxm->bqcx",z,weights["full"][...,:4])
    return preds


def endpoints(model,d,flights,weights,seed):
    raw={}
    for start in range(0,len(flights),768):
        sl=slice(start,start+768)
        base,clock=prediction_parts(model,d,sl,"")
        models=forecast(base,d["coef"][sl],d["time"][sl],clock,d["shift"][sl],weights)
        for arm,p in models.items():
            e=(p-d["y"][sl]).square().sum(-1)[:,(1,3,7)].reshape(-1,6)
            raw.setdefault(arm,[]).append(e.cpu().numpy())
    rows=[]
    mask=d["mask"][:,(1,3,7)].reshape(-1,6).cpu().numpy()
    for arm,parts in raw.items():
        a=np.concatenate(parts)
        for f in np.unique(flights):
            rows.append(dict(arm=arm,seed=seed,flight=f,**{m:a[(flights==f)&mask[:,i],i].mean(dtype=np.float64)
                                                         for i,m in enumerate(METRICS)}))
    return rows


def paired(table,metrics):
    scores,comparisons={},[]
    names=sorted(table.flight.unique())
    draw=np.random.default_rng(17).integers(0,len(names),(10000,len(names)))
    for m in metrics:
        by={a:g.pivot(index="seed",columns="flight",values=m).loc[list(SEEDS),names].to_numpy()
            for a,g in table.groupby("arm")}
        means={a:vals.mean(1)**.5 for a,vals in by.items()}
        scores[m]={k:{"mean":float(v.mean()),"seeds":v.tolist()} for k,v in means.items()}
        for a in by:
            if a=="full": continue
            b,f=by[a],by["full"]
            diff=(b[:,draw].mean(2)**.5 - f[:,draw].mean(2)**.5).mean(0)
            comparisons.append(dict(metric=m,reference=a,gain=float((means[a]-means["full"]).mean()),
                                    lo=float(np.quantile(diff,.025)),hi=float(np.quantile(diff,.975)),
                                    improved=int((f.mean(0)<b.mean(0)).sum()),flights=len(names)))
    return {"scores":scores,"contrasts":comparisons}


def native_cache(root,device):
    prior=root / "research/rounds/periodic-motion-20261125/diagnostics"
    with np.load(prior/"reference.npz") as z: ref=dict(z)
    tapes=np.load(root/"outputs/controlled_slow_fast_20261119_tapes/validation_commands.npy",mmap_mode="r")
    ix=ref["window_idx"]; t=ref["time"].astype(np.float32)
    pooled=np.zeros((len(t),16),np.float32)
    for w in np.unique(ix):
        ids=np.flatnonzero(ix==w)
        pooled[ids]=pool_commands(tapes[w:w+1],t[None,ids])[0]
    ctx=torch.as_tensor(t,device=device)[:,None,None].expand(-1,1,2)
    pool=torch.as_tensor(pooled,device=device)[:,None,None,:].expand(-1,1,2,-1)
    groups=np.split(np.arange(len(t)),np.where(np.diff(ref["block"]))[0]+1)
    bins=[]
    for ids in groups:
        angle=ref["phase"][ids];wt=ref["weight"][ids]; sec=t[ids].astype(np.float64)
        low=[np.ones_like(sec),(sec-sec.mean())/np.ptp(sec),np.cos(angle),np.sin(angle)]
        high=[np.cos(2*angle),np.sin(2*angle),np.cos(3*angle),np.sin(3*angle)]
        x=np.column_stack(low+high)*np.sqrt(wt)[:,None]
        q,r=np.linalg.qr(x,mode="reduced")
        good=len(ids)>=10 and np.linalg.cond(r)<100.
        lead=wt@sec
        bins.append((ids,wt,q[:,4:] if good else None,lead))
    return prior,ref,ctx,pool,bins


def native(model,d,weights,seed,cache):
    prior,ref,t,pool,bins=cache
    ix=ref["window_idx"]; ch=ref["ch"]
    with np.load(prior/f"original_corrected{seed}_points.npz") as pred:
        base=torch.as_tensor(pred["slow_abs"]+pred["fast_abs"],device=t.device)
    predicted={}
    for start in range(0,len(ix),768):
        sl=slice(start,start+768);w=ix[sl]
        clock=model.clocks(d["hist"][w],pool[sl],d["u0"][w],t[sl])[...,-1]
        base3=base[sl,None,None,:].expand(-1,1,2,-1)
        outputs=forecast(base3,d["coef"][w],t[sl],clock,d["shift"][w],weights)
        for arm,value in outputs.items():
            predicted.setdefault(arm,[]).append(value[np.arange(len(w)),0,ch[sl]].cpu().numpy())
    rows=[]
    for arm,parts in predicted.items():
        residual=ref["data"]-np.concatenate(parts)
        for ids,wt,q,lead in bins:
            i=ids[0];err=residual[ids]*np.sqrt(wt)[:,None]
            rows.append(dict(arm=arm,seed=seed,window=ix[i],ch=ch[i],flight=ref["flight_ids"][ref["flight_index"][i]],
                             period="early" if lead<=.5 else ("late" if lead>1 else "middle"),
                             measurement=float(np.square(err).sum()),
                             higher=np.nan if q is None else float(np.square(q.T@err).sum())))
    frame=pd.DataFrame(rows)
    return frame


@torch.inference_mode()
def run(root,device):
    training,a,train_counts=load(root,"train",device)
    dev,b,dev_counts=load(root,"validation",device)
    assert train_counts["flights"]==121 and dev_counts["flights"]==39 and set(a).isdisjoint(b)
    cache=native_cache(root,device)
    rows,beats,solutions=[],[],{}
    for s in SEEDS:
        state=torch.load(PRIOR/f"readout/original_runs/incumbent{s}/corrected.pt",weights_only=True,map_location=device)
        m=FlightPredictor(state["mean"],state["std"]).to(device).eval()
        m.load_state_dict(state)
        weights,stationarity=train(m,training)
        np.savez(HERE/f"weights_{s}.npz",**{k:v.cpu().numpy() for k,v in weights.items()})
        solutions[s]=stationarity
        rows.extend(endpoints(m,dev,b,weights,s))
        beats.append(native(m,dev,weights,s,cache))
        print("finished",s,flush=True)
    ep=pd.DataFrame(rows);ep.to_csv(HERE/"flight_mse.csv",index=False)
    sample=pd.concat(beats)
    raw_coverage=sample[(sample.arm=="corrected")&(sample.seed==17)].groupby(["ch","period"]).agg(
        retained_beats=("measurement","count"),sufficient_for_projection=("higher","count"),
        flights=("flight","nunique")).reset_index()
    native_results={};native_tables=[]
    for (ch,period),g in sample.groupby(["ch","period"]):
        if period=="middle": continue
        for metric in ("measurement","higher"):
            unit=g.dropna(subset=[metric]).groupby(["arm","seed","flight","window"],as_index=False)[metric].mean()
            unit=unit.groupby(["arm","seed","flight"],as_index=False)[metric].mean()
            native_results[f"{ch}_{period}_{metric}"]=paired(unit,[metric])
            native_tables.append(unit.rename(columns={metric:"mse"}).assign(ch=ch,period=period,metric=metric))
    pd.concat(native_tables).to_csv(HERE/"native_flight_mse.csv",index=False)
    result=dict(train=train_counts,validation=dev_counts,normal_equation_relative_residual=solutions,
                endpoints=paired(ep,METRICS),native_coverage=raw_coverage.to_dict("records"),native=native_results)
    (HERE/"results.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-root",type=Path,default=Path("/home/zn/flap-system-identification"))
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    torch.set_num_threads(2)
    run(args.data_root,args.device)
