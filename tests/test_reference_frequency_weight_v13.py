import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import train_control_windows_v2 as base
from reference_frequency_weight_v13 import frequency_weights,weighted_q_loss


def test_bounded_flight_mean_one_and_mass_changes():
    f=np.r_[np.linspace(0,1,30),np.linspace(0,3,45),np.linspace(2,3,25)]
    logs=np.array(['a']*30+['b']*45+['c']*25)
    w,c,b,r=frequency_weights(f,logs)
    assert w.min()>=.5 and w.max()<=2 and np.ptp(w)>.1
    for log in np.unique(logs):np.testing.assert_allclose(w[logs==log].mean(),1,atol=1e-12)
    assert np.all(w>0) and len(w)==len(f)


def test_tied_frequency_does_not_invent_missing_strata():
    w,_,b,_=frequency_weights(np.ones(20),np.array(['a']*10+['b']*10))
    np.testing.assert_allclose(w,1);assert set(b)=={'low'}


def test_only_q_loss_gradient_changes_and_ones_recover_baseline():
    torch.manual_seed(13);pred={k:torch.randn(4,10,n,requires_grad=True) for k,n in [('rates_b',3),('velocity_n',3),('quaternion',4)]}
    target={k:torch.randn_like(v) for k,v in pred.items()};m=SimpleNamespace(rate_scale=torch.tensor([1.,2.,3.]),velocity_scale=torch.ones(3))
    old=base.loss_fn(pred,target,m);same=weighted_q_loss(pred,target,m,torch.ones(4))
    torch.testing.assert_close(old,same,rtol=0,atol=0)
    a=torch.autograd.grad(old,tuple(pred.values()),retain_graph=True)
    weights=torch.tensor([.5,.75,1.,1.75]);b=torch.autograd.grad(weighted_q_loss(pred,target,m,weights),tuple(pred.values()))
    torch.testing.assert_close(a[0][:,:,[0,2]],b[0][:,:,[0,2]],rtol=0,atol=0)
    torch.testing.assert_close(a[0][:,:,1]*weights[:,None],b[0][:,:,1])
    for i in [1,2]:torch.testing.assert_close(a[i],b[i],rtol=0,atol=0)
