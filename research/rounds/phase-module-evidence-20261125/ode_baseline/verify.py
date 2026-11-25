"""Across all development flights, check prefix invariance, integration refinement, and export identity."""
import json

import numpy as np
import torch
from train import HERE, ControlledODE, args, load
from pathlib import Path


@torch.inference_mode()
def main():
    torch.set_num_threads(2);torch.set_float32_matmul_precision("high")
    data,t,id_,f,_ = load(Path("/home/zn/flap-system-identification"), "validation", "cuda:0")
    selected = np.concatenate([np.flatnonzero(f==fid)[np.linspace(0,(f==fid).sum()-1,4).astype(int)]
                               for fid in np.unique(f)])
    state=torch.load(HERE/"best.pt",weights_only=True,map_location="cuda:0")
    model=ControlledODE(state["mean"],state["scale"]).to("cuda:0").eval();model.load_state_dict(state)
    h,u,end,dt,q=args(data,selected)
    p=model(h,u,end,dt,q)
    with np.load(HERE/"predictions.npz") as z:
        np.testing.assert_array_equal(z["window_id"],id_)
        np.testing.assert_array_equal(z["flight_id"],f)
        np.testing.assert_array_equal(z["time"],t)
        exp=z["predicted_delta"][selected]
    different_batch=float(np.max(np.abs(exp-p.cpu().numpy())))
    # Reproduce the evaluation kernel on the same set of full-sized contiguous batches.
    saved_layout=np.empty_like(exp)
    for b in np.unique(selected//512):
        start=int(b*512); same_batch=model(*args(data,slice(start,start+512)))
        pick=np.flatnonzero(selected//512==b)
        saved_layout[pick]=same_batch[selected[pick]-start].cpu().numpy()
    cmp=float(np.max(np.abs(saved_layout-exp)))
    # Measure numerical integration sensitivity in IEEE float32, separately from TF32 GEMM selection.
    torch.set_float32_matmul_precision("highest")
    p=model(h,u,end,dt,q)
    later=torch.arange(u.shape[1],device=u.device)[None,:]>q[:,:4].amax((1,2))[:,None]
    perturbed=torch.where(later[:,:,None],u+1,u)
    prefix=float((model(h,perturbed,end,dt,q)[:,:4]-p[:,:4]).abs().max())
    halved_dt=(dt.unsqueeze(-1).expand(-1,-1,2)*.5).flatten(1)
    halved_end=torch.stack([end-dt*.5,end],-1).flatten(1)
    halved_u=u.unsqueeze(2).expand(-1,-1,2,-1).flatten(1,2)
    err=model(h,halved_u,halved_end,halved_dt,2*q+1)-p
    metrics=dict(windows=len(h),flights=len(np.unique(f[selected])),export_same_batch_max_abs=cmp,export_different_batch_max_abs=different_batch,
                 changed_control_prefix_max=prefix,
                 half_step_vrmse=float(err[:,:,0].square().sum(-1).mean().sqrt()),
                 half_step_wrmse=float(err[:,:,1].square().sum(-1).mean().sqrt()))
    print(metrics, flush=True)
    assert prefix==0 and cmp==0 and torch.isfinite(err).all()
    (HERE/"verification.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(metrics)


if __name__=="__main__":
    main()
