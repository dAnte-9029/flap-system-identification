"""Maintain canonical propagation for both the bare and augmented recurrent models."""
import sys
import numpy as np
import torch
from common_paths import MAIN, ROUND, TMP
sys.path.insert(0,str(ROUND/"combined"))
from predictor import WingbeatPredictor, causal_conditions
sys.path.insert(0,str(ROUND/"waveform"))
from experiment import load, native_cache, query, SEEDS

@torch.inference_mode()
def run():
    torch.set_num_threads(3)
    d, flights, _ = load(MAIN,"validation","cpu")
    _,ref,t,pool,_=native_cache(MAIN,"cpu")
    ix=ref["window_idx"]
    np.testing.assert_array_equal(flights[ix],ref["flight_ids"][ref["flight_index"]])
    for seed in SEEDS:
        full=WingbeatPredictor.from_seed(seed)
        low=full.first_order
        out={"bare":[],"wingbeat":[]}
        for start in range(0,len(ix),512):
            sl=slice(start,start+512)
            w=ix[sl]
            h=d["hist"][w]
            u=d["u0"][w]
            p=pool[sl]
            tm=t[sl]
            base=query(low.core,h[:,:low.mean.numel()],u,d["time"][w],d["pooled"][w],tm,p)
            beta=torch.einsum("bh,ckha->bcka",low.context(h),low.weights)
            first=torch.einsum("bqck,bcka->bqca",low.bases(h,p,u,tm)["direction"],beta)
            theta=low.clocks(h,p,u,tm)[...,-1]
            high=full.high_order(d["coef"][w],tm,theta,causal_conditions(h,p,u,tm))
            for name,pred in [("bare",base),("wingbeat",base+first+high)]:
                out[name].append(pred[np.arange(len(w)),0,ref["ch"][sl]].numpy()+ref["origin"][sl])
        for name,parts in out.items():
            a=np.concatenate(parts)
            assert np.isfinite(a).all()
            np.save(TMP/f"{name}{seed}.npy",a)
        print(seed,"validated",len(ix),flush=True)

if __name__=="__main__":
    run()
