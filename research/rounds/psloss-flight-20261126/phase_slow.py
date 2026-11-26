"""Score the frozen tuned A+C on the identical held-beat reference as DLinear."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import torch

ROOT=Path("/home/zn/flap-system-identification")
sys.path.insert(0,str(ROOT/"research/rounds/wingbeat-hyperparameter-search-20261126"))
from search import wave, old_model, causal_conditions, load_tuned


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    device="cpu"; seed=17
    data, flight, _=wave.load(ROOT,"validation",device)
    directory,ref,times,pooled,bins=wave.native_cache(ROOT,device)
    win,ch=ref["window_idx"],ref["ch"]
    np.testing.assert_array_equal(np.asarray(flight)[win],ref["flight_ids"][ref["flight_index"]])
    with np.load(directory/"original_corrected17_points.npz") as p:
        base=torch.from_numpy(p["slow_abs"]+p["fast_abs"])
    model=load_tuned(seed,device)
    old=old_model(seed,device)
    predictions=[]
    for j in range(0,len(win),512):
        sl=slice(j,j+512)
        h,u,coef=[data[k][win[sl]] for k in ("hist","u0","coef")]
        t,p=times[sl],pooled[sl]
        correction=model.first_order.forecast(h,p,u,t,arm="direction")-old(h,p,u,t)
        phase=model.first_order.clocks(h,p,u,t)[...,-1]
        high=model.high_order(coef,t,phase,causal_conditions(h,p,u,t))
        predictions.append((base[sl]+(correction+high)[torch.arange(len(h)),0,ch[sl]]).numpy())
    pred=np.concatenate(predictions)
    rows=[]
    for idx,weights,_,lead in bins:
        i=idx[0]
        rows.append([ref["flight_ids"][ref["flight_index"][i]],win[i],ch[i],*(weights@pred[idx]),*(weights@ref["slow"][idx])])
    df=pd.DataFrame(rows,columns=["flight","window","ch","p0","p1","p2","r0","r1","r2"])
    cols=df.columns[3:].tolist()
    groups=df.groupby(["flight","window","ch"])
    delta=(df[cols]-groups[cols].transform("first")).to_numpy()
    df["model"]=np.square(delta[:,:3]-delta[:,3:]).sum(1)
    df["hold"]=np.square(delta[:,3:]).sum(1)
    means=df.loc[groups.cumcount()>0].groupby(["flight","window","ch"])[["model","hold"]].mean().groupby(["flight","ch"]).mean()
    result=means.groupby("ch").mean().pow(.5).to_dict("index")
    out=Path(__file__).resolve().parent/"results"
    out.mkdir(exist_ok=True)
    (out/"phase_slow.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)

if __name__=="__main__":
    main()
