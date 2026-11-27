"""Query frozen causal models at complete-beat observation times."""
import sys
import numpy as np
import pandas as pd
import torch
from common_paths import DIAG, MAIN, OUT, ROUND, TMP
legacy = MAIN / ".worktrees/transformer-slowfast-v1"
sys.path[:0] = [str(legacy / "experiments/transformer_slowfast_v1"), str(legacy / "src"),
                str(legacy / "experiments/history_slow_fast_v1")]
from latent import LatentTransformerHarmonics
from fit import make_model
from dynamics import rotate, inverse
from stage_query import query

@torch.inference_mode()
def run(shift=False):
    torch.set_num_threads(4)
    ref = dict(np.load(DIAG / "reference.npz"))
    history = OUT / "history_slow_fast_20261119_cache"
    tapes = OUT / "controlled_slow_fast_20261119_tapes"
    meta = pd.read_parquet(history / "validation_windows.parquet").set_index("window_id",verify_integrity=True)
    order = pd.read_parquet(tapes / "validation_ids.parquet").window_id
    with np.load(OUT / "autopilot_flight_dynamics_20261124/validation.npz") as f:
        np.testing.assert_array_equal(order, f["window"])
        np.testing.assert_array_equal(meta.loc[order].flight_id, f["flight"])
    ix = meta.index.get_indexer(order)
    def tensor(name):
        return np.load(history / f"validation_{name}.npy",mmap_mode="r")
    hist, init = tensor("history"), tensor("initial")
    tape = np.load(tapes / "validation_commands.npy",mmap_mode="r")
    wins = np.unique(ref["window_idx"])
    for name in (("transformer","tcn") if shift else ("transformer","tcn","v2")):
        ck = OUT / "controlled_baselines_20261119_run1" / name / "best.pt"
        if name=="v2":
            ck = OUT / "transformer_slowfast_latent_v2_20261124/model/best.pt"
        state = torch.load(ck,map_location="cpu",weights_only=False)["model"]
        if name=="v2":
            model = LatentTransformerHarmonics(state["mean"],state["scale"])
        else:
            model = make_model(name,state["mean"],state["scale"])
        if name=="tcn":
            state = torch.load(ROUND / "external_baselines/best.pt",map_location="cpu",weights_only=True)
        model.load_state_dict(state)
        model.eval()
        if shift:
            def shift_phase(module, args):
                # Rotate k=1 by pi/2 and k=2 by pi at the rollout embedding only.
                token=args[0].clone()
                c,s,c2,s2=token[..., -4:].clone().unbind(-1)
                token[..., -4:]=torch.stack((-s,c,-c2,-s2),-1)
                return (token,)
            model.backbone.step_input.register_forward_pre_hook(shift_phase)
        output = np.full_like(ref["data"],np.nan)
        for start in range(0,len(wins),32):
            ws=wins[start:start+32]
            points=[np.flatnonzero(ref["window_idx"]==w) for w in ws]
            ts=np.zeros((len(ws),max(map(len,points))),np.float32)
            for i,ids in enumerate(points):
                ts[i,:len(ids)]=ref["time"][ids]
            initial=torch.from_numpy(init[ix[ws]])
            roll=model(torch.from_numpy(hist[ix[ws]]),initial,torch.from_numpy(tape[ws]))
            t=torch.from_numpy(ts)
            v=rotate(inverse(initial[:,None,6:10]),query(roll,t,"pv")[...,3:])
            rate=query(roll,t,"w")
            for i,ids in enumerate(points):
                output[ids]=np.where(ref["ch"][ids,None]==0,v[i,:len(ids)],rate[i,:len(ids)])
        assert np.isfinite(output).all()
        np.save(TMP / (name+("_shift" if shift else "")+".npy"),output)
        print(name,"validated",output.shape,flush=True)

if __name__=="__main__":
    run("--shift" in sys.argv)
