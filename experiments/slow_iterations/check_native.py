"""Check the differentiable batched native path against independent single-query rollouts."""
import numpy as np
import torch
from shared import ROOT,PREV,EXTRA,Dynamics,load_tuned,wave,read
from fit_stage import native_predictions

def check():
    d=read('train','cpu')
    orig=load_tuned(17,'cpu')
    net=Dynamics(orig,True)
    with torch.no_grad():net.feedback.weight.fill_(.025)
    raw,_,_=wave.load(ROOT,'train','cpu')
    q=torch.load(EXTRA/'beats.pt',weights_only=True)
    with np.load(PREV/'train.npz') as z:
        ii=z['wi'][::750].copy()
        sub=(q['start'][ii]+q['length'][ii]//2).numpy()
        t=z['t'][sub].copy()
    tape=np.load(ROOT/'outputs/controlled_slow_fast_20261119_tapes/train_commands.npy',mmap_mode='r')
    qt=torch.from_numpy(t)[:,None,None].expand(-1,1,2)
    qp=torch.from_numpy(wave.pool_commands(tape[ii],t[:,None]))[:,:,None].expand(-1,1,2,-1)
    args=[d[k][ii] for k in ('h','anchor','dt','forcing','memory','left')]
    _,_,hs,ss=net(*args,quadrature=d['quad'][ii],trace=True)
    p=native_predictions(net,hs,ss,ii,q,torch.arange(len(ii)),sub)
    h,cp,u,ct=[raw[k][ii] for k in ('hist','pooled','u0','time')]
    with torch.no_grad():
        expected,_=net.point(h,cp,u,ct,d['anchor'][ii],qt,qp)
        delta=(p-expected-q['offset'][sub]).abs().max().item()
        assert delta<1e-5,delta
    p.square().mean().backward()
    assert net.feedback.weight.grad.norm().item()>0
    print('native query parity samples',len(ii),'maximum absolute gap',delta,flush=True)
if __name__=='__main__':
    torch.set_num_threads(3)
    check()
