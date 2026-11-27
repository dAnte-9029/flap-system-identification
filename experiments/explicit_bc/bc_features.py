"""Causal features for explicit slow and phase-conditioned predictions; no dynamics network runs here."""
import sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'slow_iterations'))
from shared import ROOT, PREV, EXTRA, read, wave, load_tuned
from search import causal_conditions
from dataset import anchor_input
CACHE=ROOT/'.worktrees/explicit-bc-no-a/outputs/explicit_bc'

def query(clock,h,pooled,u,t,coef):
    # Three smooth time functions and two exponentially weighted prefixes of four commands.
    mass1=-torch.expm1(-t)
    mass02=-torch.expm1(-t/.2)
    slow=torch.cat((torch.stack((t,-torch.expm1(-t/.35),mass1),-1),
        pooled[...,:4]-mass02[...,None]*u[:,None,None],
        pooled[...,8:12]-mass1[...,None]*u[:,None,None]),-1)
    low=clock.first_order
    first=low.bases(h,pooled,u,t)['direction'][...,1:].clone()
    first[...,0]+=1
    theta=low.clocks(h,pooled,u,t)[...,-1]
    high=clock.high_order(coef,t,theta,causal_conditions(h,pooled,u,t))
    return torch.cat((slow,first,high),-1)

def history(raw,anchor,stats):
    x=(raw-stats['mean'])/stats['std']
    return torch.cat((x,anchor),-1)

def anchor_delta(anchor,origin):
    ang=(anchor[:,:3]-origin)*anchor[:,3:]
    return torch.stack((torch.zeros_like(ang),ang),1)

@torch.inference_mode()
def prepare():
    CACHE.mkdir(parents=True,exist_ok=True)
    clock=load_tuned(17,'cpu')
    # All operations used for the phase path must work after deleting the old A.
    del clock.first_order.core
    for part in ('train','validation'):
        d,_,_=wave.load(ROOT,part,'cpu')
        with np.load(PREV/f'{part}.npz') as z:
            a=anchor_input(z);origin=torch.from_numpy(z['origin']).float()
            with np.load(ROOT/f'outputs/autopilot_flight_dynamics_20261124/{part}.npz') as old:
                np.testing.assert_array_equal(z['window'],old['window'])
        if part=='train':
            stats=dict(mean=d['hist'].mean(0),std=d['hist'].std(0,unbiased=False).clamp_min(.001))
            torch.save(stats,CACHE/'stats.pt')
        phase=torch.cat([query(clock,*[d[k][rows] for k in ('hist','pooled','u0','time','coef')])
                         for rows in torch.arange(len(a)).split(720)])
        torch.save(dict(x=history(d['hist'],a,stats),base=anchor_delta(a,origin),
                        origin=origin,query=phase),CACHE/f'{part}.pt')
        if part=='train':native(clock,d)
        print(part,len(a),flush=True)

def native(clock,d):
    with np.load(PREV/'train.npz') as z:
        wi,length,t=[z[k] for k in ('wi','lengths','t')]
    tape=np.load(ROOT/'outputs/controlled_slow_fast_20261119_tapes/train_commands.npy',mmap_mode='r')
    ends=length.cumsum(); starts=ends-length
    out=torch.empty(len(t),16)
    for lo in range(0,len(wi),96):
        b=np.arange(lo,min(lo+96,len(wi)));ii=wi[b];r=np.arange(starts[b[0]],ends[b[-1]])
        col=np.arange(length[b].max())
        pos=starts[b,None]+np.minimum(col,length[b,None]-1)
        pp=wave.pool_commands(tape[ii],t[pos])[col<length[b,None]]
        qp=torch.from_numpy(pp)[:,None,None].expand(-1,1,2,-1)
        qt=torch.from_numpy(t[r])[:,None,None].expand(-1,1,2)
        ids=np.repeat(ii,length[b])
        out[r]=query(clock,d['hist'][ids],qp,d['u0'][ids],qt,d['coef'][ids])[:,0,1]
    torch.save(out,CACHE/'native.pt')
    print('native training queries',len(t),flush=True)
if __name__=='__main__':
    torch.set_num_threads(4)
    prepare()
