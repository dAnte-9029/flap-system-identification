"""Fixed full-development endpoint scoring plus the existing held-out-beat trend reference."""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data import ROOT, TAPE, integral, write
from network import Forecast
from train import FlightSet

METRICS=["v05","w05","v1","w1","v2","w2"]


@torch.inference_mode()
def beats(model,d):
    with np.load(ROOT/"research/rounds/periodic-motion-20261125/diagnostics/reference.npz") as z:
        ref={k:z[k] for k in ("time","weight","ch","slow","block","flight_ids","flight_index","window_idx")}
    w=ref["window_idx"];ch=ref["ch"]
    np.testing.assert_array_equal(d.ids.flight_id.to_numpy()[w],ref["flight_ids"][ref["flight_index"]])
    tape=np.load(TAPE/"validation_commands.npy",mmap_mode="r")
    vals=[]
    for start in range(0,len(w),512):
        sl=slice(start,start+512); ids=w[sl];tt=ref["time"][sl,None]
        qp=torch.tensor(np.repeat(tt,2,1)[:,None,:],dtype=torch.float32,device=d.device)
        uc=torch.tensor(integral(tape[ids],tt),device=d.device)[:,:,None,:].expand(-1,-1,2,-1)
        full=model.at(model.grid(d.x[ids]),qp,uc)*d.std.reshape(1,1,2,3)
        vals.append(full.cpu().numpy()[np.arange(len(ids)),0,ch[sl]])
    motion=np.concatenate(vals)
    starts=np.r_[0,np.flatnonzero(np.diff(ref["block"]))+1]
    def average(x):
        return np.add.reduceat(ref["weight"][:,None]*x,starts)
    np.testing.assert_allclose(average(np.ones((len(w),1))),1,atol=1e-8)
    est,reference=average(motion),average(ref["slow"])
    frame=pd.DataFrame(dict(flight=ref["flight_index"][starts],window=w[starts],ch=ch[starts]))
    for j in range(3):
        frame[f"p{j}"]=est[:,j];frame[f"r{j}"]=reference[:,j]
    columns=[f"{a}{j}" for a in "pr" for j in range(3)]
    bywin=frame.groupby(["flight","window","ch"])
    delta=(frame[columns]-bywin[columns].transform("first")).to_numpy()
    frame["model"]=np.square(delta[:,:3]-delta[:,3:]).sum(1)
    frame["hold"]=np.square(delta[:,3:]).sum(1)
    later=frame.loc[bywin.cumcount()>0]
    result={}
    for ch,name in enumerate(("v","w")):
        part=later[later.ch==ch]
        by=part.groupby(["flight","window"])[["model","hold"]].mean().groupby("flight").mean()
        result[name]=dict(model=float(np.sqrt(by.model.mean())),hold=float(np.sqrt(by.hold.mean())),flights=len(by),later_beats=len(part))
    return result


@torch.inference_mode()
def evaluate():
    pack=torch.load(os.environ["RA_MODEL"],map_location="cpu",weights_only=True)
    model=Forecast(pack["arm"]=="command_ps").eval()
    model.load_state_dict(pack["model"])
    d=FlightSet("validation","cpu")
    sq=[]
    for ix in torch.arange(len(d.x)).split(512):
        pred=model.at(model.grid(d.x[ix]),d.time[ix],d.query_u[ix])*d.std.reshape(1,1,2,3)
        sq.append((pred-d.endpoint[ix]).square().sum(-1)[:,[1,3,7]].reshape(-1,6).numpy())
    frame=pd.DataFrame(np.vstack(sq).astype("float64"),columns=METRICS)
    frame["flight"]=d.ids.flight_id.to_numpy()
    per=frame.groupby("flight")[METRICS].mean()
    assert len(per)==39 and len(frame)==44524
    out=Path(os.environ["RA_OUTPUT"])
    per.to_csv(out/"by_flight.csv")
    slow=beats(model,d)
    write(out/"slow.json",slow)
    score=np.sqrt(per.mean()).to_dict()
    score.update(slowv=slow["v"]["model"],sloww=slow["w"]["model"])
    write(Path(os.environ["RA_METRICS"]),score)
    print(score,flush=True)


if __name__=="__main__":
    torch.set_num_threads(2)
    evaluate()
