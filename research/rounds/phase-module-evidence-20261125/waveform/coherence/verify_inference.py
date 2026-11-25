"""Compare the standalone additive head with fixed offline training/evaluation on actual validation windows."""
import json
from pathlib import Path

import numpy as np
import torch

from inference import HarmonicTransport, causal_conditions
import run


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    source = Path("/home/zn/flap-system-identification")
    data, _, _ = run.wave.load(source, "validation", "cpu")
    ix = np.linspace(0, len(data["hist"])-1, 128).astype(int)
    data = {k:v[ix] for k,v in data.items()}
    h,p,u,t,co = [data[k] for k in ("hist","pooled","u0","time","coef")]
    rng=torch.Generator().manual_seed(17)
    records={}
    for seed in run.wave.SEEDS:
        state=torch.load(run.wave.PRIOR / f"readout/original_runs/incumbent{seed}/corrected.pt",weights_only=True,map_location="cpu")
        m=run.wave.FlightPredictor(state["mean"],state["std"]).eval();m.load_state_dict(state)
        model=HarmonicTransport.from_files(seed)
        base,theta=run.wave.prediction_parts(m,data,slice(None),"")
        cond=causal_conditions(h,p,u,t)
        public=model(co,t,theta,cond)
        zero=torch.zeros_like(t)
        zero_time=model(co,zero,torch.zeros_like(theta),causal_conditions(h,torch.zeros_like(p),u,zero)).abs().max()
        zero_history=model(torch.zeros_like(co),t,theta,cond).abs().max()
        prefix=(public[:,:4]-model(co,t[:,:4],theta[:,:4],cond[:,:4])).abs().max()
        perturb_theta=theta.clone();perturb_theta[:,4:]=torch.rand(theta[:,4:].shape,generator=rng)*100
        p2=p.clone();p2[:,4:]=torch.rand(p[:,4:].shape,generator=rng)
        change=(public[:,:4]-model(co,t,m.clocks(h,p2,u,t)[...,-1].where(t<=t[:,3:4],perturb_theta),
                                   causal_conditions(h,p2,u,t))[:,:4]).abs().max()
        with np.load(run.HERE / "clock.npz") as f:
            clock={k:torch.as_tensor(f[k],dtype=torch.float32) for k in run.MOMENTS}
        z=run.features(co,t,theta,h,u,p,data["shift"],clock)
        gold=base+torch.einsum("bqcxm,cxm->bqcx",z["conditional"],model.readout_coefficients)
        diff=(m(h,p,u,t)+public-gold).abs().max()
        assert zero_time==0 and zero_history==0 and max(prefix,change)<1e-6 and diff<2e-6
        records[seed]=dict(points=len(ix)*t.shape[1]*2,zero_time=float(zero_time),ineligible=float(zero_history),
                           prefix=float(prefix),changed_future=float(change),full_prediction=float(diff))
    (run.HERE / "inference_checks.json").write_text(json.dumps(records,indent=2)+"\n")
    print(records)


if __name__=="__main__":
    main()
