"""Check both execution paths against independent raw native-time queries before training."""
import numpy as np
import torch
from bc_features import ROOT,PREV,EXTRA,CACHE,load_tuned,wave,anchor_input
from bc_fit import partition,endpoint
from bc_model import Predictor
from bc_score import points

def run():
    torch.set_num_threads(3)
    clock=load_tuned();del clock.first_order.core
    raw,_,_=wave.load(ROOT,'train','cpu');d=partition('train','cpu')
    q=torch.load(EXTRA/'beats.pt',weights_only=True)
    nq=torch.load(CACHE/'native.pt',weights_only=True)
    with np.load(PREV/'train.npz') as z:
        ii=z['wi'][::1300].copy();sub=(q['start'][ii]+q['length'][ii]//2).numpy()
        ts=z['t'][sub].copy();a=anchor_input(z)
    ii=torch.from_numpy(ii)
    tape=np.load(ROOT/'outputs/controlled_slow_fast_20261119_tapes/train_commands.npy',mmap_mode='r')
    tt=torch.from_numpy(ts)[:,None,None].expand(-1,1,2)
    pp=torch.from_numpy(wave.pool_commands(tape[ii],ts[:,None]))[:,:,None].expand(-1,1,2,-1)
    stats=torch.load(CACHE/'stats.pt',weights_only=True)
    b=dict(a=a,origin_tensor=d['origin'],stats=stats)
    initial=None
    for use_a in (False,True):
        torch.manual_seed(17);model=Predictor(use_a)
        init=endpoint(model,d,ii)[0].detach()
        if initial is None:
            initial=init
            assert set(dict(model.named_children()))=={'coefficients'}
        else:
            torch.testing.assert_close(init,initial,atol=0,rtol=0)
            with torch.no_grad():model.a.readout_linear.weight.normal_(std=.05)
        full,c,states=endpoint(model,d,ii)
        pf,ps=model.native(c,nq[sub],d['base'][ii],states,torch.arange(len(ii)),q['slot'][sub],q['path'][sub])
        with torch.no_grad():
            direct,_=model.forecast(raw['hist'][ii],raw['pooled'][ii],raw['u0'][ii],raw['time'][ii],raw['coef'][ii],a[ii],d['origin'][ii],stats,clock)
            torch.testing.assert_close(direct,full,rtol=2e-5,atol=5e-6)
            ef,es=points(model,clock,raw,b,ii.numpy(),tt,pp,False,True)
            # Exercise a single-row group (NumPy indexing with a length-one Torch index can squeeze).
            single,_=points(model,clock,raw,b,ii[:1].numpy(),tt[:1],pp[:1],False,True)
            np.testing.assert_allclose(single,ef[:1],atol=2e-6)
            origin=d['origin'][ii]
            errors=[(torch.as_tensor(v)-origin-z).abs().max().item() for v,z in ((ef,pf),(es,ps))]
            assert max(errors)<1e-5,errors
            # The same timestamp and command prefix produce the same early prediction when later commands change.
            changed=raw['pooled'][ii].clone();changed[:,4:]+=2.
            later,_=model.forecast(raw['hist'][ii],changed,raw['u0'][ii],raw['time'][ii],raw['coef'][ii],a[ii],origin,stats,clock)
            torch.testing.assert_close(later[:,:4],direct[:,:4],rtol=0,atol=0)
        (pf.square().mean()+full.square().mean()).backward()
        assert model.coefficients[-1].weight.grad.norm()>0
        if use_a:assert model.a.encoder[0].weight.grad.norm()>0
        print('include_a',use_a,'raw/cached native parity',errors,'same initial motion; prefix passed')
if __name__=='__main__':run()
