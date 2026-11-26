import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from labels import ROOT, CACHE
from anchored_model import AnchoredRate, wave
from arm import USE_B

def data(part):
    with np.load(CACHE/f'{part}.npz') as z:
        lab=dict(z)
    with np.load(ROOT/f'outputs/autopilot_flight_dynamics_20261124/{part}.npz') as z:
        np.testing.assert_array_equal(lab.pop('window'),z['window'])
    d,ids,_=wave.load(ROOT,part,'cpu')
    return d,ids,lab

def args(d,ii):
    return [d[k][ii] for k in ('hist','pooled','u0','time','coef')]

def objective(x,y):
    x,y=x.double(),y.double()
    return x.T@x,x.T@y,len(x)

@torch.inference_mode()
def fit(out):
    model=AnchoredRate()
    if not USE_B:
        torch.save(dict(use_b=False,weights=model.weights),out/'model')
        return dict(fit='frozen published A+C',steps=1)
    d,ids,b=data('train')
    tape=np.load(ROOT/'outputs/controlled_slow_fast_20261119_tapes/train_commands.npy',mmap_mode='r')
    with np.load(ROOT/'outputs/autopilot_flight_dynamics_20261124/train.npz') as z:
        import pandas as pd
        np.testing.assert_array_equal(pd.read_parquet(ROOT/'outputs/controlled_slow_fast_20261119_tapes/train_ids.parquet',columns=['window_id']).window_id,z['window'])
    anchor=torch.from_numpy(b['anchor'])
    origin=torch.from_numpy(b['origin']).float()
    w=b['wi']
    lengths=b['lengths']
    starts=lengths.cumsum()-lengths
    batches=[np.flatnonzero(b['ok']),np.arange(len(w))]
    matrices=[]; vectors=[]; counts=[]
    for mode,index in enumerate(batches):
        normal=torch.zeros((len(model.weights),len(model.weights)),dtype=torch.float64)
        rhs=torch.zeros_like(model.weights,dtype=torch.float64)
        count=0
        for start in range(0,len(index),512):
            sel=index[start:start+512]
            ii=sel if mode==0 else w[sel]
            hist,pool,u0,time,coef=args(d,ii)
            states=model.states(hist,pool,u0,time)
            if mode==0:
                feat=model.features(states,time[:,:,1],time[:,:,1],pool[:,:,1],anchor[ii])
                fixed=model.offsets(hist,pool,u0,time,coef,anchor[ii],origin[ii])
                valid=d['mask'][ii,:,1]
                x=feat[valid]
                y=(d['y'][ii,:,1]-fixed)[valid]
            else:
                col=np.arange(lengths[sel].max())
                pos=starts[sel,None]+np.minimum(col,lengths[sel,None]-1)
                t=b['t'][pos]
                weight=torch.from_numpy(np.where(col<lengths[sel,None],b['weight'][pos],0.))
                torch.testing.assert_close(weight.sum(1),torch.ones(len(ii),dtype=torch.float64))
                cmd=wave.pool_commands(tape[ii],t)
                feat=model.features(states,time[:,:,1],torch.from_numpy(t),torch.from_numpy(cmd),anchor[ii])
                x=(feat.double()*weight[:,:,None]).sum(1)
                y=torch.from_numpy(b['target'][sel])-anchor[ii]
            g,r,n=objective(x,y)
            normal+=g;rhs+=r;count+=n
        matrices.append(normal/count);vectors.append(rhs/count);counts.append(count)
    gram=sum(matrices)
    matrix=gram+1e-5*torch.diag(gram.diag().clamp_min(1e-12))
    rhs=sum(vectors)
    weights=torch.linalg.solve(matrix,rhs)
    residual=float((matrix@weights-rhs).abs().max()/rhs.abs().max())
    assert residual<1e-8 and torch.isfinite(weights).all() and weights.abs().max()>0
    torch.save(dict(use_b=True,weights=weights.float()),out/'model')
    return dict(steps=1,windows=len(ids),flights=len(np.unique(ids)),anchored_windows=len(batches[0]),endpoints=counts[0],training_beats=counts[1],
                auxiliary_native_samples=len(b['t']),relative_solver_residual=residual,weight_norm=float(weights.norm()),parameters=weights.numel())

@torch.inference_mode()
def preflight():
    net=AnchoredRate()
    with np.load(ROOT/'outputs/autopilot_flight_dynamics_20261124/train.npz') as z:
        hist=torch.from_numpy(z['hist'][:12]);u=torch.from_numpy(z['u0'][:12]);t=torch.from_numpy(z['time'][:12]);p=torch.from_numpy(z['pooled'][:12])
    core=net.ac.first_order.core
    states=net.states(hist,p,u,t)
    predicted=core.observe(states[:,1:])-core.observe(states[:,:1])
    torch.testing.assert_close(predicted[:,:,3:],core(hist,p,u,t)[:,:,1],rtol=1e-4,atol=1e-5)
    f=net.features(states,t[:,:,1],t[:,:,1],p[:,:,1],u[:,:3])
    modified=states.clone();modified[:,3:]+=100
    g=net.features(modified,t[:,:,1],t[:,:,1],p[:,:,1],u[:,:3])
    torch.testing.assert_close(f[:,:3],g[:,:3],atol=0,rtol=0)
    print('recurrent state parity and causal prefix passed')

if __name__=='__main__':
    torch.set_num_threads(3)
    if sys.argv[1]=='preflight': preflight()
    else:
        out=Path(os.environ['RA_OUTPUT'])
        result=fit(out)
        (out/'fit.json').write_text(json.dumps(result,indent=2)+'\n')
        print(result,flush=True)
        if 'RA_TRAIN_INFO' in os.environ:
            Path(os.environ['RA_TRAIN_INFO']).write_text(json.dumps({'steps':1,'stop_reason':'sufficient_steps',**result}))
