"""Compare the training quadrature to direct native query rollouts on selected training beats."""
from pathlib import Path
import sys
import json
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'experiments/coupled_slow'))
from network import ROOT, Dynamics,load_tuned,wave
from fit_coupled import read,call
from dataset import PREV

@torch.inference_mode()
def check():
    torch.set_num_threads(3)
    p=ROOT/'research/experiments/study-27657a60d6e6/exp-9a71c3328d5f/attempt-1/seed-17/output/model'
    ck=torch.load(p,map_location='cpu',weights_only=True)
    model=Dynamics(load_tuned(17,'cpu'),True);model.load_state_dict(ck['state'])
    cache=read('train','cpu')
    with np.load(PREV/'train.npz') as lab:
        idx=np.linspace(0,len(lab['wi'])-1,80).round().astype(int)
        wi=lab['wi'][idx];length=lab['lengths'];starts=length.cumsum()-length
        slices=[slice(starts[i],starts[i]+length[i]) for i in idx]
        times=lab['t'];weights=lab['weight']
        tt=np.concatenate([times[s] for s in slices])
        ww=np.concatenate([weights[s] for s in slices])
        reps=length[idx]
    with np.load(ROOT/'outputs/autopilot_flight_dynamics_20261124/train.npz') as src:
        h,cp,u,ct=[torch.from_numpy(src[k][wi]) for k in ('hist','pooled','u0','time')]
    tape=np.load(ROOT/'outputs/controlled_slow_fast_20261119_tapes/train_commands.npy',mmap_mode='r')
    repeats=np.repeat(np.arange(len(wi)),reps)
    pp=np.concatenate([wave.pool_commands(tape[i:i+1],times[s][None])[0] for i,s in zip(wi,slices)])
    p=torch.from_numpy(pp)[:,None,None,:].expand(-1,1,2,-1)
    t=torch.from_numpy(tt)[:,None,None].expand(-1,1,2)
    anchor=cache['anchor'][wi]
    _,sample=model.point(h[repeats],cp[repeats],u[repeats],ct[repeats],anchor[repeats],t,p,False,True)
    full=np.add.reduceat(ww[:,None]*sample.numpy(),np.r_[0,reps.cumsum()[:-1]])
    _,training=call(model,cache,torch.from_numpy(wi),supervise=True)
    diff=float(abs(training.numpy()-full).max())
    assert diff<3e-5
    result=dict(beats=len(wi),samples=len(tt),max_abs_rad_s=diff,feedback_weight_norm=float(model.feedback.weight.norm()))
    (HERE/'quadrature_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)
if __name__=='__main__':check()
