"""Compare the auxiliary decoder and the actual beat-mean system output on identical beats."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'experiments/coupled_slow'))
from network import ROOT, Dynamics,load_tuned,wave
from dataset import PREV,anchor_input
sys.path.insert(0,str(HERE.parents[2]/'experiments/coupled_slow'))
from evaluate import exact_points,mse,average

@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    p=ROOT/'research/experiments/study-27657a60d6e6/exp-9a71c3328d5f/attempt-1/seed-17/output'
    pack=torch.load(p/'model',weights_only=True,map_location='cpu')
    orig=load_tuned(17,'cpu');model=Dynamics(orig,True);model.load_state_dict(pack['state'])
    d,flights,_=wave.load(ROOT,'validation','cpu')
    with np.load(PREV/'validation.npz') as z:b={k:z[k] for k in ('ok','origin','anchor')}
    b['a']=anchor_input(b);b['origin_tensor']=torch.from_numpy(b['origin']).float()
    _,ref,t,pool,_=wave.native_cache(ROOT,'cpu')
    ix=np.flatnonzero((ref['ch']==1)&b['ok'][ref['window_idx']]);wi=ref['window_idx'][ix]
    full,slow=exact_points(model,orig,d,b,wi,t[ix],pool[ix],False,True)
    first=np.r_[0,np.flatnonzero(np.diff(ref['block'][ix]))+1]
    wt=ref['weight'][ix,None]
    def mean(arr):return np.add.reduceat(wt*arr,first)
    sysmean,aux,r=mean(full),mean(slow),mean(ref['slow'][ix])
    df=pd.DataFrame(dict(flight=flights[wi[first]],window=wi[first],gap=mse(sysmean,aux),system=mse(sysmean,r),
                         aux=mse(aux,r),hold=mse(b['anchor'][wi[first]],r)))
    by=average(df,['gap','system','aux','hold'])
    out=dict(beats=len(df),flights=len(by),rmse=by.mean().pow(.5).to_dict(),aux_beats_hold_flights=int((by.aux<by.hold).sum()))
    (HERE/'readout_gap.json').write_text(json.dumps(out,indent=2)+'\n')
    print(out)
if __name__=='__main__':main()
