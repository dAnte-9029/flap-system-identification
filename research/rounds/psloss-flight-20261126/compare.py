"""Paired flight-equal evaluation against the archived tuned, seed-17 WingbeatPredictor."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path("/home/zn/flap-system-identification")
HERE=Path(__file__).resolve().parent
STUDY=ROOT/"research/experiments/study-737f57c053b8"
PRIOR=ROOT/"research/rounds/wingbeat-hyperparameter-search-20261126/results"
KEYS=["v05","w05","v1","w1","v2","w2"]


def main():
    state=json.loads((STUDY/"state.json").read_text())
    ids={"mse":"baseline"}
    for eid,item in state["experiments"].items():
        ids["ps" if item["hypothesis"]["hypothesis_id"]=="full-paper" else "command_ps"]=eid
    assert len(ids)==3
    out=HERE/"results"
    out.mkdir(exist_ok=True)
    phase=pd.read_csv(PRIOR/"endpoints_by_flight.csv").query("arm=='full' and seed==17").set_index("flight")[KEYS].sort_index()
    frames={"phase":phase}; training={};slow={}
    for key,eid in ids.items():
        folder=STUDY/eid/"attempt-1/seed-17"
        df=pd.read_csv(folder/"output/by_flight.csv").set_index("flight")[KEYS].sort_index()
        pd.testing.assert_index_equal(phase.index,df.index)
        frozen=json.loads((folder/"metrics.json").read_text())
        np.testing.assert_allclose(np.sqrt(df.mean()),[frozen[k] for k in KEYS],atol=2e-8,rtol=0)
        frames[key]=df
        run=json.loads((folder/"training.json").read_text())
        training[key]={k:run[k] for k in ("steps","best_epoch","seconds","parameter_change","ps_calls","min_patch_clamps","gpu")}
        training[key]["output"]=str(folder/"output")
        slow[key]=json.loads((folder/"output/slow.json").read_text())
    ph=json.loads((out/"phase_slow.json").read_text())
    slow["phase"]={k:ph[str(i)] for i,k in enumerate(("v","w"))}
    for v in slow.values():
        for k in ("v","w"):
            assert abs(v[k]["hold"]-slow["mse"][k]["hold"])<1e-9
    split=json.loads((PRIOR/"split.json").read_text()); assert set(split["selection"])|set(split["confirmation"])==set(phase.index)
    summary={}
    for label,members in dict(all=list(phase.index),historical26=split["selection"],historical13=split["confirmation"]).items():
        summary[label]={k:np.sqrt(df.loc[members].mean()).to_dict() for k,df in frames.items()}
    draws=np.random.default_rng(17).integers(len(phase),size=(10000,len(phase)))
    paired={}
    for a,b in (("ps","mse"),("command_ps","ps"),("ps","phase"),("command_ps","phase")):
        x,y=frames[a].to_numpy(),frames[b].to_numpy()
        gain=np.sqrt(y[draws].mean(1))-np.sqrt(x[draws].mean(1))
        lower,upper=np.quantile(gain,[.025,.975],axis=0)
        diff=np.sqrt(y.mean(0))-np.sqrt(x.mean(0))
        paired[a+"_vs_"+b]={k:dict(gain=float(diff[i]),percent=float(100*diff[i]/np.sqrt(y.mean(0))[i]),
                                      ci95=[float(lower[i]),float(upper[i])],wins=int((x[:,i]<y[:,i]).sum())) for i,k in enumerate(KEYS)}
    result=dict(rmse=summary,paired=paired,slow=slow,training=training)
    (out/"comparison.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    pd.concat(frames,names=["model","flight"]).to_csv(out/"flight_mse.csv")
    print(json.dumps(result["rmse"]["all"],indent=2))

if __name__=="__main__":
    main()
