"""Decode first eligible future beat's motion coefficients on disjoint train/development flights."""
import json
import sys
import numpy as np
import pandas as pd
import torch
from common_paths import DIAG, HERE, OUT, MAIN, ROUND
from ordinary import make_model
sys.path.insert(0,str(DIAG))
import prepare_reference as pr
from collections import Counter
from scipy.spatial.transform import Rotation

def collect(split,velocity):
    assert split in ("train","validation")
    history=OUT/"history_slow_fast_20261119_cache"
    windows=pd.read_parquet(history/f"{split}_windows.parquet").reset_index(drop=True)
    order=pd.read_parquet(OUT/f"controlled_slow_fast_20261119_tapes/{split}_ids.parquet")
    windows=windows.reset_index(names="ix").set_index("window_id").loc[order.window_id]
    contract=json.loads((pr.SOURCE/"configs/pahbd/phase_reference.json").read_text())
    prepare=pr.load_module("phase_qc",pr.SOURCE/"src/system_identification/data/pahbd_phase.py").prepare_phase
    cols=["flight_id","state_measurement_us","segment_id","valid_core","firmware_verified","review","provenance_review","excluded"]
    samples=pd.read_parquet(pr.DATA/f"samples_{split}.parquet",columns=cols).groupby("flight_id")
    meta={m["flight_id"]:m for m in json.loads((pr.DATA/"flight_provenance.json").read_text()) if m["split"]==split}
    phase=pd.read_parquet(pr.DATA/f"native_wing_phase_{split}.parquet").groupby("flight_id")
    stream="vehicle_local_position" if velocity else "vehicle_angular_velocity"
    init=np.load(history/f"{split}_initial.npy",mmap_mode="r")
    native=pd.read_parquet(pr.DATA/f"native_{stream}_{split}.parquet").groupby("flight_id")
    inputs=[];targets=[];flights=[]
    for f,g in windows.groupby("flight_id"):
        p=prepare(phase.get_group(f),meta[f]["transmission_ratio"],contract)
        cycles=pr.make_cycles(p,meta[f],samples.get_group(f),native.get_group(f),velocity,Counter())
        eligible=[i for i,c in cycles.items() if all(j in cycles and cycles[j]["segment"]==c["segment"] for j in [i-2,i-1,i+1,i+2])]
        cs=[cycles[i] for i in sorted(eligible)]
        left=np.array([c["left"] for c in cs])
        g=g.sort_values("decision_us")
        picks=48 if split=="train" else 8
        for row in g.iloc[np.unique(np.rint(np.linspace(0,len(g)-1,picks)).astype(int))].itertuples():
            j=left.searchsorted(row.decision_us)
            if j==len(cs):continue
            c=cs[j]
            if c["right"]>row.decision_us+500000 or len(c["t"])<10:continue
            # Predictor information ends at decision time. Future measured phase only defines labels.
            t,angle,w=c["t"],c["phase"],np.sqrt(c["weight"][:,None])
            mat=np.column_stack([np.ones(len(t)),(t-np.average(t,weights=c["weight"]))/np.ptp(t),
                             *[fn(k*angle) for k in (1,2,3) for fn in (np.cos,np.sin)]])
            if np.linalg.cond(mat*w)>100:continue
            measured=c["data"]
            if velocity:
                measured=Rotation.from_quat(init[row.ix,[7,8,9,6]]).apply(measured,inverse=True)
            co=np.linalg.lstsq(mat*w,measured*w,rcond=None)[0]
            targets.append(co);inputs.append(row.ix);flights.append(f)
    x=np.load(history/f"{split}_history.npy",mmap_mode="r")[inputs]
    return x,np.stack(targets),np.array(flights)

@torch.inference_mode()
def features(x):
    out={"raw":x.reshape(len(x),-1)}
    for name in ("transformer","tcn"):
        state=torch.load(OUT/f"controlled_baselines_20261119_run1/{name}/best.pt",weights_only=False,map_location="cpu")["model"]
        model=make_model(name,state["mean"],state["scale"])
        if name=="tcn":
            state=torch.load(ROUND/"external_baselines/best.pt",weights_only=True,map_location="cpu")
        model.load_state_dict(state); model.eval()
        enc=model.backbone
        rep=[]
        for h in torch.from_numpy(x).split(128):
            z=enc.history_input((h-model.mean)/model.scale)
            if name=="transformer":
                angle=torch.arange(h.shape[1])[:,None]*enc.frequencies
                z+=torch.stack([angle.sin(),angle.cos()],-1).flatten(-2)
                for layer in enc.layers: z,_=layer(z)
                z=enc.final_norm(z)
            else:
                for layer in enc.layers:z,_=layer.initialize(z)
            rep.append(z[:,-1].numpy())
        out[name]=np.concatenate(rep)
    return out

def run(velocity):
    torch.set_num_threads(3)
    a,y,f=collect("train",velocity); b,v,g=collect("validation",velocity)
    assert not set(f)&set(g)
    print("windows/flights",len(a),len(set(f)),len(b),len(set(g)),flush=True)
    train,dev=features(a),features(b)
    rows=[]
    # Flight-balanced ridge objective; standardized feature columns, fixed 0.1 penalty.
    wt=np.array([1/(len(set(f))*(f==i).sum()) for i in f])
    y0=np.einsum("b,bka->ka",wt,y)
    for name in ("mean","raw","transformer","tcn"):
        if name=="mean":pred=np.broadcast_to(y0,v.shape)
        else:
            x,z=train[name].astype(float),dev[name].astype(float)
            xm=wt@x
            sd=np.sqrt(wt@((x-xm)**2)).clip(1e-5)
            x,z=(x-xm)/sd,(z-xm)/sd
            wx=wt[:,None]*x
            beta=np.linalg.solve(x.T@wx+.1*np.eye(x.shape[1]),wx.T@(y-y0).reshape(len(y),-1))
            pred=(z@beta).reshape(v.shape)+y0
        for j in np.unique(g):
            e=(pred[g==j]-v[g==j])**2
            for k,sl in [("level",slice(0,1)),("slope",slice(1,2)),("h1",slice(2,4)),("h2",slice(4,6)),("h3",slice(6,8))]:
                rows.append([name,j,k,*e[:,sl].mean((0,1))])
    frame=pd.DataFrame(rows,columns=["representation","flight","target","x","y","z"])
    frame.to_csv(HERE/("probe_velocity_flight_mse.csv" if velocity else "probe_flight_mse.csv"),index=False)
    print((frame.groupby(["target","representation"])[["x","y","z"]].mean().sum(axis=1)**.5).unstack().round(4),flush=True)
if __name__=="__main__":run("--velocity" in sys.argv)
