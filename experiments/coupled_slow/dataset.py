"""Fixed C contributions, native-clock inputs, and exact integration of slow-state beat labels."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from network import ROOT, Dynamics, TAU, load_tuned, wave

PREV=ROOT/'.worktrees/anchored-slow-rate-b/outputs/anchored_rate'
CACHE=ROOT/'.worktrees/coupled-slow-latent/outputs/coupled_slow'
def anchor_input(b):
    return torch.from_numpy(np.column_stack((b['anchor'],b['ok'])).astype('float32'))

def beat_quadrature(t,b):
    # For s(t)=s_left+(1-exp(-elapsed/tau))*(target-s_left), integrate both coefficients.
    wi=b['wi'];length=b['lengths'];row=np.repeat(wi,length)
    pos=b['t'].astype('float64')
    knots=t[row,:,1].numpy().astype('float64')
    index=(knots<pos[:,None]).sum(1)
    left=np.where(index==0,0,knots[np.arange(len(pos)),np.maximum(0,index-1)])
    frac=-np.expm1(-(pos-left)/TAU)
    mass=np.zeros((len(t),9,2),np.float64)
    np.add.at(mass[:,:,0],(row,index),b['weight'])
    np.add.at(mass[:,:,1],(row,index),b['weight']*frac)
    np.testing.assert_allclose(mass[wi,:,0].sum(1),1,atol=1e-7)
    assert np.all(mass[:,:,1]<=mass[:,:,0]+1e-9)
    target=np.zeros((len(t),3),np.float32);target[wi]=b['target']
    return torch.from_numpy(mass.astype('float32')),torch.from_numpy(target)

@torch.no_grad()
def prepare(part):
    label=np.load(PREV/f'{part}.npz')
    with np.load(ROOT/f'outputs/autopilot_flight_dynamics_20261124/{part}.npz') as z:
        np.testing.assert_array_equal(z['window'],label['window'])
    d,ids,_=wave.load(ROOT,part,'cpu')
    orig=load_tuned(17,'cpu');net=Dynamics(orig,False)
    stacks={k:[] for k in ('dt','forcing','memory','left','offset')}
    for start in range(0,len(ids),640):
        sl=slice(start,start+640);h,p,u,t,c=[d[k][sl] for k in ('hist','pooled','u0','time','coef')]
        raw=orig.first_order.core(h[:,:orig.first_order.mean.numel()],p,u,t)
        offset=orig(h,p,u,t,c)-raw
        for k,x in zip(stacks,(*net.inputs(p,u,t),offset)):
            stacks[k].append(x)
    out={k:torch.cat(parts) for k,parts in stacks.items()}
    out.update(h=net.context(d['hist']),anchor=anchor_input(label),y=d['y'],mask=d['mask'])
    if part=='train':
        out['quad'],out['target']=beat_quadrature(d['time'],label)
    CACHE.mkdir(parents=True,exist_ok=True)
    torch.save(out,CACHE/f'{part}.pt')
    print(part,len(ids),len(np.unique(ids)),flush=True)

if __name__=='__main__':
    torch.set_num_threads(3)
    for part in ('train','validation'):prepare(part)
