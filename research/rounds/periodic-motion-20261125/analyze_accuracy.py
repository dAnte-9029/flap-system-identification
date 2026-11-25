"""Re-evaluate fixed checkpoints and compare all three arms on the same flights and seeds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
METRICS=("v05","w05","v1","w1","v2","w2")
ARMS={"actual-phase-carrier":"periodic","matched-slow-basis":"matched"}
MODEL="experiments/autopilot_flight_dynamics/model.py"


def read(path):
    return json.loads(path.read_text())


def inspect(seed,study,arm,item,gpu):
    attempt=item["attempts"][-1]
    assert attempt["status"]=="measured"
    stages={v["phase"]:v for v in attempt["stages"]}
    record=item["result"]["seeds"][0]
    t=record["training"]
    assert record["seed"]==int(seed) and t["steps"]==t["samples"]//384
    assert t["parameter_change_l2"]>0 and t["steps"]>=49152
    src=Path(stages["train"]["cwd"])
    commit=subprocess.check_output(["git","-C",str(src),"rev-parse","HEAD"],text=True).strip()
    assert commit==attempt["commit"]
    run=study/attempt["directory"]/f"seed-{seed}"/"output"
    data={}
    def verify(errors,metrics):
        for k in METRICS:
            assert len(errors[k])==39 and set(errors[k])==set(errors["w2"])
            np.testing.assert_allclose(np.sqrt(np.mean(list(errors[k].values()))),metrics[k],atol=1e-10,rtol=0)
    final=read(run/"flight_mse.json")
    verify(final,record["metrics"])
    data["final"]=final
    if t["steps"]==49152:
        a,b=(torch.load(run/k,map_location="cpu",weights_only=True) for k in ("step49152.pt","model"))
        assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
        data["common"]=final
    else:
        dest=HERE/"scores"/f"seed{seed}"/arm
        dest.mkdir(parents=True,exist_ok=True)
        if not (dest/"metrics.json").exists():
            env=dict(os.environ,RA_OUTPUT=str(dest),RA_METRICS=str(dest/"metrics.json"),RA_MODEL_CODE=str(src),
                     RA_MODEL=str(run/"step49152.pt"),RA_DATA_VERSION="velocity_rate_multistep_v1",
                     RA_SPLIT="validation",PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES=str(gpu))
            env.pop("PYTHONPATH",None)
            judge=Path(stages["evaluate"]["cwd"])
            subprocess.run([sys.executable,str(judge/"experiments/periodic_motion/runner.py"),"evaluate"],env=env,check=True)
        data["common"]=read(dest/"flight_mse.json")
        verify(data["common"],read(dest/"metrics.json"))
    tail=t["validation_history"][-16:]
    if t["stop"]=="sustained_plateau":
        assert all(r["lr"]==.000125 for r in tail)
        assert max(r["dev_mse"] for r in tail)-min(r["dev_mse"] for r in tail)<=.002
        assert min(r["dev_mse"] for r in t["validation_history"][:-16])-min(r["dev_mse"] for r in tail)<=.001
    return data,dict(study=study.name,experiment=item["id"],commit=commit,parameters=t["parameters"],steps=t["steps"],
                     samples=t["samples"],parameter_change_l2=t["parameter_change_l2"],stop=t["stop"],
                     common_checkpoint=str((run/"step49152.pt").relative_to(PROJECT)),model_file=str(src/MODEL))


def aggregate(perseed, pairs_to_score=(("strong","periodic"),("matched","periodic"),("strong","matched"))):
    ids=sorted(next(iter(next(iter(perseed.values())).values()))["w2"])
    ix=np.random.default_rng(17).integers(0,len(ids),(10000,len(ids)))
    seeds=list(perseed)
    def matrix(arm,k):
        for s in seeds: assert set(perseed[s][arm][k])==set(ids)
        return np.array([[perseed[s][arm][k][f] for f in ids] for s in seeds])
    metrics={}
    for arm in next(iter(perseed.values())):
        metrics[arm]={}
        for k in METRICS:
            x=np.sqrt(matrix(arm,k).mean(1))
            metrics[arm][k]=dict(mean=float(x.mean()),seeds=dict(zip(seeds,x.tolist())))
    pairs={}
    for ref,arm in pairs_to_score:
        out={}
        for k in METRICS:
            a,b=matrix(ref,k),matrix(arm,k)
            gains=np.sqrt(a.mean(1))-np.sqrt(b.mean(1))
            boot=(np.sqrt(a[:,ix].mean(2))-np.sqrt(b[:,ix].mean(2))).mean(0)
            out[k]=dict(gain=float(gains.mean()),pct=100*float(gains.mean()/np.sqrt(a.mean(1)).mean()),
                        interval95=np.quantile(boot,[.025,.975]).tolist(),by_seed=dict(zip(seeds,gains.tolist())),
                        improved_flights=int((b.mean(0)<a.mean(0)).sum()))
        pairs[f"{arm}_vs_{ref}"]=out
    return dict(metrics=metrics,comparisons=pairs)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gpu',default='0')
    args=ap.parse_args()
    studies=read(HERE/"studies.json"); scores={"common":{},"final":{}}; fits={}
    for seed,sid in studies.items():
        study=PROJECT/"research/experiments"/sid
        state=read(study/"state.json")
        assert state.get("closed_at") and not state["pending"]
        candidates={ARMS[i["hypothesis"]["hypothesis_id"]]:i for i in state["experiments"].values()}
        assert set(candidates)=={"periodic","matched"}
        fits[seed]={}
        for stage in scores: scores[stage][seed]={}
        for arm,item in {"strong":state["baseline"],**candidates}.items():
            data,summary=inspect(seed,study,arm,item,args.gpu)
            fits[seed][arm]=summary
            for stage in data: scores[stage][seed][arm]=data[stage]
    out=dict(fits=fits,results={stage:aggregate(value) for stage,value in scores.items()})
    (HERE/"accuracy.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
