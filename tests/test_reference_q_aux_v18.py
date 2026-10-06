import sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import train_control_windows_v2 as base
from reference_q_aux_v18 import auxiliary_loss,loss_terms,AUX_WEIGHT


def fixture():
    torch.manual_seed(18)
    p={k:torch.randn(4,50,n,requires_grad=True) for k,n in [('rates_b',3),('velocity_n',3),('quaternion',4)]}
    t={k:torch.randn_like(v) for k,v in p.items()}
    m=SimpleNamespace(rate_scale=torch.tensor([1.,2.,3.]),velocity_scale=torch.ones(3))
    return p,t,m


def test_zero_weight_is_original_loss_and_gradients():
    p,t,m=fixture();old=base.loss_fn(p,t,m);new=auxiliary_loss(p,t,m,0.)
    torch.testing.assert_close(old,new,rtol=0,atol=0)
    a=torch.autograd.grad(old,tuple(p.values()),retain_graph=True)
    b=torch.autograd.grad(new,tuple(p.values()))
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_only_first_q_direct_gradient_changes_by_factor51():
    p,t,m=fixture();old=base.loss_fn(p,t,m);new=auxiliary_loss(p,t,m)
    a=torch.autograd.grad(old,tuple(p.values()),retain_graph=True)
    b=torch.autograd.grad(new,tuple(p.values()))
    expected=a[0].clone();expected[:,0,1]*=51
    torch.testing.assert_close(expected,b[0])
    for x,y in zip(a[1:],b[1:]):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_auxiliary_ignores_later_targets_but_full_loss_does_not():
    p,t,m=fixture();full,aux=loss_terms(p,t,m)
    changed={k:v.clone() for k,v in t.items()};changed['rates_b'][:,1:,1]+=30
    next_full,next_aux=loss_terms(p,changed,m)
    torch.testing.assert_close(aux,next_aux,rtol=0,atol=0)
    assert next_full>full and AUX_WEIGHT==1/3
