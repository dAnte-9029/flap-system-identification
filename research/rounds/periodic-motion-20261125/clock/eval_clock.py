"""Compare held-frequency and corrected-frequency predictions without changing any trained weight."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from conditional import corrected_model,load_coefficients

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DATA=ROOT/"outputs/autopilot_flight_dynamics_20261124"
EXTRA=ROOT/"outputs/autopilot_periodic_motion_20261125"
NAMES=("v05","w05","v1","w1","v2","w2")


def load_source(path):
    s=importlib.util.spec_from_file_location("trained_motions",path)
    mod=importlib.util.module_from_spec(s);s.loader.exec_module(mod)
    return mod.FlightPredictor


def run(seed,study):
    folder=ROOT/"research/experiments"/study
    s=json.loads((folder/"state.json").read_text())
    item=next(x for x in s["experiments"].values() if x["hypothesis"]["hypothesis_id"]=="actual-phase-carrier")
    q=item["attempts"][-1]
    path=folder/q["directory"]/f"seed-{seed}"/"output"/"step49152.pt"
    model=load_source(Path(item["worktree"])/"experiments/autopilot_flight_dynamics/model.py")
    state=torch.load(path,map_location="cpu",weights_only=True)
    classes=dict(held=model,corrected=corrected_model(model,load_coefficients()),identity=corrected_model(model,[0]*5))
    nets={}
    for name,cls in classes.items():
        obj=cls(state["mean"],state["std"]).eval();obj.load_state_dict(state);nets[name]=obj
    out={k:[] for k in ("held","corrected")}; identity=0.
    with np.load(DATA/"validation.npz",allow_pickle=False) as z:
        flights=z["flight"]; labels=z["y"]; mask=z["mask"]
        b={k:torch.from_numpy(z[k]) for k in ("hist","pooled","u0","time")}
        b["hist"]=torch.cat((b["hist"],torch.from_numpy(np.load(EXTRA/"validation_features.npy"))),1)
        with torch.inference_mode():
            for j in range(0,len(flights),384):
                sl=slice(j,j+384)
                a=[b[k][sl] for k in ("hist","pooled","u0","time")]
                for name in out:
                    pred=nets[name](*a)
                    if name=="held":
                        check=nets["identity"](*a)
                        identity=max(identity,float((pred-check).abs().max()))
                    error=(pred-torch.from_numpy(labels[sl])).square().sum(-1)[:,[1,3,7]].reshape(-1,6)
                    out[name].append(error.numpy())
    assert identity<2e-5,identity
    members=np.unique(flights);valid=mask[:,[1,3,7]].reshape(-1,6)
    results={}
    for name,rows in out.items():
        errors=np.concatenate(rows)
        results[name]={k:{f:float(errors[(flights==f)&valid[:,i],i].mean()) for f in members} for i,k in enumerate(NAMES)}
    (HERE/f'seed{seed}_paired.json').write_text(json.dumps(dict(identity_max=identity,flight_mse=results))+'\n')
    print('seed',seed,'2s metrics', {name:[float(np.sqrt(np.mean(list(t[k].values())))) for k in ('v2','w2')] for name,t in results.items()},flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('seed');args=a.parse_args()
    torch.set_num_threads(2)
    studies=json.loads((HERE.parent/'studies.json').read_text())
    run(args.seed,studies[args.seed])
