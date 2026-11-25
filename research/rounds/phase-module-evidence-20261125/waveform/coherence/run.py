"""Direct second/third circular moments for forecasts of a measured recent wingbeat waveform."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
sys.path.insert(0,str(PARENT))
import experiment as wave
sys.path.insert(0,str(PARENT.parent/"coherence"))
import module as moment

ARMS=("global","conditional","sham","unmodulated")
MOMENTS=ARMS[:3]


def learn_moments(root):
    spec=importlib.util.spec_from_file_location("wave_encoder_labels",wave.PRIOR / "clock/calibrate.py")
    labels=importlib.util.module_from_spec(spec);spec.loader.exec_module(labels)
    labels.DATA=root/"dataset/velocity_rate_multistep_v1"
    labels.MEM=root/"outputs/history_slow_fast_20261119_cache"
    labels.OLD=root/"outputs/autopilot_flight_dynamics_20261124"
    meta={r["flight_id"]:r for r in json.loads((labels.DATA/"flight_provenance.json").read_text())
          if r["split"] in ("train","validation")}
    coef=np.asarray(json.loads((wave.PRIOR/"clock/clock_fit.json").read_text())["coefficients"])
    dims=(3,45,45)
    grams={a:np.zeros((d,d)) for a,d in zip(MOMENTS,dims)}
    rhs={a:np.zeros((d,4)) for a,d in zip(MOMENTS,dims)}
    def extract(x,base,target,t):
        t=t.astype(np.float64)
        ramp=t+.2*np.expm1(-t/.2)
        raw=moment.build_design(torch.from_numpy(t),torch.from_numpy(x[...,1:]/ramp[...,None])).numpy()
        error=(2*np.pi*(target-base-x@coef))[...,None]*[2,3]
        target_pair=np.stack([np.cos(error),np.sin(error)],-1).reshape(error.shape[:-1]+(4,))
        return raw,target_pair
    rng=np.random.default_rng(17)
    n=0; pair_count=0; train_ids=set()
    for fid,x,base,target,mask,t in labels.rows("train",meta):
        raw,y=extract(x,base,target,t); perm=y.copy()
        for q in range(t.shape[1]):
            for c in range(2):
                ok=mask[:,q,c]
                a=y[ok,q,c];perm[ok,q,c]=a[rng.permutation(len(a))]
        for name,d in zip(MOMENTS,dims):
            z=raw[mask,:d];obs=(perm if name=="sham" else y)[mask]-[1,0,1,0]
            grams[name]+=z.T@z/len(z);rhs[name]+=z.T@obs/len(z)
        n+=1;pair_count+=int(mask.sum());train_ids.add(fid)
    assert n==121
    params={a:np.linalg.solve(grams[a]+.001*np.diag(np.diag(grams[a])),rhs[a]) for a in MOMENTS}
    np.savez(HERE/"clock.npz",**params)
    rows=[]
    for fid,x,base,target,mask,t in labels.rows("validation",meta):
        assert fid not in train_ids
        raw,y=extract(x,base,target,t)
        for a in MOMENTS:
            z=raw[...,:params[a].shape[0]]@params[a]+[1,0,1,0]
            z=z.reshape(z.shape[:-1]+(2,2))
            z/=np.maximum(1,np.linalg.norm(z,axis=-1,keepdims=True))
            for lo,hi,key in ((0,.5,"early"),(1,2.1,"late")):
                good=mask&(t>lo)&(t<=hi)
                yy=y[good].reshape(-1,2,2);val=z[good]
                loss=np.square(val-yy).sum(-1).mean(0)
                mag=np.linalg.norm(val,axis=-1).mean(0)
                for i,k in enumerate((2,3)):
                    rows.append(dict(flight=fid,arm=a,order=k,period=key,score=loss[i],modulus=mag[i],samples=int(good.sum())))
    frame=pd.DataFrame(rows);assert frame.flight.nunique()==39
    frame.to_csv(HERE/"clock_flights.csv",index=False)
    stats=[]
    for (order,period),g in frame.groupby(["order","period"]):
        a=g.pivot(index="flight",columns="arm",values="score")
        rng=np.random.default_rng(17);draw=rng.integers(0,len(a),(10000,len(a)))
        row=dict(order=int(order),period=period,arms=g.groupby("arm")[["score","modulus"]].mean().to_dict("index"))
        for ref in ("global","sham"):
            diff=(a[ref]-a["conditional"]).to_numpy()
            row[ref+"_gain"]=[float(diff.mean()),*np.quantile(diff[draw].mean(1),[.025,.975]).tolist()]
        stats.append(row)
    (HERE/"clock_summary.json").write_text(json.dumps(dict(train_flights=n,train_pairs=pair_count,development=stats),indent=2)+"\n")
    return params


def features(coef,t,theta,h,u,pool,shift,params,include_legacy=False):
    # Columns in the first trained waveform represent oriented C/Q, each at three decay scales.
    original=wave.design(coef,t,theta,shift,"full")
    z=original[...,4:].reshape(*original.shape[:-1],2,2,3)
    result={"unmodulated":original[...,4:]}
    if include_legacy: result["legacy"]=original
    x=moment.build_design(t,moment.causal_conditions(h,pool,u,t))
    for a in MOMENTS:
        value=(x[...,:params[a].shape[0]]@params[a]).reshape(*t.shape,2,2)+t.new_tensor([1.,0.])
        value=moment.unit_projection(value)
        cr,si=value.unbind(-1)
        cr=cr.unsqueeze(-2).unsqueeze(-1);si=si.unsqueeze(-2).unsqueeze(-1)
        c,q=z.unbind(-2)
        dense=torch.stack((cr*c+si*q,cr*q-si*c),-2)
        result[a]=dense.flatten(-3)
    return result


def train(m,data,params):
    gram={};rhs={}
    for i in range(0,len(data["hist"]),768):
        sl=slice(i,i+768)
        for suffix in ("","_early"):
            base,theta=wave.prediction_parts(m,data,sl,suffix)
            res=(data["y"+suffix][sl]-base).double()
            zall=features(data["coef"][sl],data["time"+suffix][sl],theta,data["hist"][sl],data["u0"][sl],
                          data["pooled"+suffix][sl],data["shift"][sl],params)
            for a,z in zall.items():
                z=(z*data["mask"+suffix][sl][:,:,:,None,None]).double()
                norm=torch.einsum("bqcxm,bqcxn->cxmn",z,z)
                vec=torch.einsum("bqcxm,bqcx->cxm",z,res)
                if a not in gram:gram[a],rhs[a]=norm,vec
                else:gram[a]+=norm;rhs[a]+=vec
    params={};check={}
    for a in ARMS:
        mat=gram[a]+1e-5*torch.diag_embed(gram[a].diagonal(dim1=-1,dim2=-2))
        sol=torch.linalg.solve(mat,rhs[a].unsqueeze(-1))
        check[a]=float((mat@sol-rhs[a].unsqueeze(-1)).abs().max()/rhs[a].abs().max())
        assert check[a]<1e-8
        params[a]=sol.squeeze(-1).float()
    return params,check


def output(base,z,t,weights,legacy):
    out={"corrected":base}
    for a in ARMS:
        out[a]=base+torch.einsum("bqcxm,cxm->bqcx",z[a],weights[a])
    out["conditional_with_sham_moment"]=base+torch.einsum("bqcxm,cxm->bqcx",z["sham"],weights["conditional"])
    gate=(1+torch.cos(np.pi*((t-1.125)/(1.8-1.125)).clamp(0,1)))*.5
    # Exact cutoff for queries after 1.8 s, independently of command prefix.
    gate=torch.where(t>=1.8,0.,torch.where(t<=1.125,1.,gate))
    extra=torch.einsum("bqcxm,cxm->bqcx",z["legacy"],legacy)
    out["round1"]=base+extra
    out["tapered_round1"]=base+extra*gate[...,None]
    return out


def endpoints(m,d,ids,clock_param,w,legacy,seed):
    data={}
    for i in range(0,len(ids),768):
        sl=slice(i,i+768); t=d["time"][sl];base,theta=wave.prediction_parts(m,d,sl,"")
        z=features(d["coef"][sl],t,theta,d["hist"][sl],d["u0"][sl],d["pooled"][sl],d["shift"][sl],clock_param,True)
        res=output(base,z,t,w,legacy)
        for a,p in res.items():
            err=(p-d["y"][sl]).square().sum(-1)[:,(1,3,7)].reshape(-1,6)
            data.setdefault(a,[]).append(err.cpu().numpy())
    rows=[]
    mask=d["mask"][:,(1,3,7)].reshape(-1,6).cpu().numpy()
    for a,parts in data.items():
        val=np.concatenate(parts)
        for f in np.unique(ids):
            row={k:val[(ids==f)&mask[:,j],j].mean(dtype=np.float64) for j,k in enumerate(wave.METRICS)}
            rows.append(dict(arm=a,seed=seed,flight=f,**row))
    return rows


def native(m,d,cache,param,w,legacy,seed):
    prior,ref,t,pool,bins=cache
    ix=ref["window_idx"];ch=ref["ch"]
    with np.load(prior/f"original_corrected{seed}_points.npz") as p:
        base=torch.as_tensor(p["slow_abs"]+p["fast_abs"],device=t.device)
    raw={}
    for i in range(0,len(ix),768):
        sl=slice(i,i+768);wi=ix[sl];h=d["hist"][wi];u=d["u0"][wi]
        theta=m.clocks(h,pool[sl],u,t[sl])[...,-1]
        z=features(d["coef"][wi],t[sl],theta,h,u,pool[sl],d["shift"][wi],param,True)
        outputs=output(base[sl,None,None,:].expand(-1,1,2,-1),z,t[sl],w,legacy)
        for a,out in outputs.items():
            raw.setdefault(a,[]).append(out[np.arange(len(wi)),0,ch[sl]].cpu().numpy())
    rows=[]
    for a,parts in raw.items():
        e=ref["data"]-np.concatenate(parts)
        for ids,wt,q,lead in bins:
            if .5<lead<=1:continue
            i=ids[0];err=e[ids]*np.sqrt(wt)[:,None]
            rows.append(dict(arm=a,seed=seed,flight=ref["flight_ids"][ref["flight_index"][i]],ch=ch[i],w=ix[i],
                             period="early" if lead<=.5 else "late",measurement=np.square(err).sum(),
                             higher=np.nan if q is None else np.square(q.T@err).sum()))
    return pd.DataFrame(rows).groupby(["arm","seed","flight","ch","w","period"]).mean().groupby(
        ["arm","seed","flight","ch","period"]).mean().reset_index()


@torch.inference_mode()
def run(root,dev):
    clock=learn_moments(root)
    coeff={k:torch.as_tensor(v,dtype=torch.float32,device=dev) for k,v in clock.items()}
    train_data,a,train_info=wave.load(root,"train",dev)
    val,b,val_info=wave.load(root,"validation",dev)
    assert set(a).isdisjoint(b)
    cache=wave.native_cache(root,dev)
    endpoints_rows=[];native_stats=[];checks={}
    for s in wave.SEEDS:
        state=torch.load(wave.PRIOR/f"readout/original_runs/incumbent{s}/corrected.pt",map_location=dev,weights_only=True)
        m=wave.FlightPredictor(state["mean"],state["std"]).to(dev).eval();m.load_state_dict(state)
        with np.load(PARENT/f"weights_{s}.npz") as old:
            legacy=torch.as_tensor(old["full"],device=dev)
        weights,checks[s]=train(m,train_data,coeff)
        np.savez(HERE/f"weights_{s}.npz",**{k:v.cpu().numpy() for k,v in weights.items()})
        endpoints_rows.extend(endpoints(m,val,b,coeff,weights,legacy,s))
        native_stats.append(native(m,val,cache,coeff,weights,legacy,s))
        print("finished",s,flush=True)
    ep=pd.DataFrame(endpoints_rows)
    ep.to_csv(HERE/"flight_mse.csv",index=False)
    ns=pd.concat(native_stats);ns.to_csv(HERE/"native_flight_mse.csv",index=False)
    def compare(frame,metrics):
        # The previous experiment names its primary comparison 'full'.
        return wave.paired(frame.replace({"arm":{"conditional":"full"}}),metrics)
    report=dict(train=train_info,validation=val_info,solution_residual=checks,
                endpoints=compare(ep,wave.METRICS),
                native={f"{ch}_{period}":compare(group,["measurement","higher"])
                        for (ch,period),group in ns.groupby(["ch","period"])})
    (HERE/"results.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",type=Path,default=Path("/home/zn/flap-system-identification"))
    ap.add_argument("--device",default="cuda:0")
    args=ap.parse_args();torch.set_num_threads(2);run(args.data_root,args.device)
