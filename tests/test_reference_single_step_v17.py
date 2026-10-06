import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import train_control_windows_v2 as base
from reference_single_step_v17 import first_step_inputs, first_step_loss, errors


def test_only_future_tapes_are_truncated():
    x={k:torch.ones(2,n,3) for k,n in [('history_state',26),('history_actuator',26),('future_reference',50),('dt_s',50)]}
    y=first_step_inputs(x)
    assert y['future_reference'].shape[1]==y['dt_s'].shape[1]==1
    assert y['history_state'] is x['history_state'] and y['history_actuator'] is x['history_actuator']
    assert x['future_reference'].shape[1]==50


def test_loss_and_gradients_ignore_future_targets():
    torch.manual_seed(17)
    p={k:torch.randn(3,50,n,requires_grad=True) for k,n in [('rates_b',3),('velocity_n',3),('quaternion',4)]}
    t={k:torch.randn_like(v) for k,v in p.items()}
    model=SimpleNamespace(rate_scale=torch.tensor([1.,2.,3.]),velocity_scale=torch.ones(3))
    loss=first_step_loss(p,t,model)
    expected=base.loss_fn({k:v[:,:1] for k,v in p.items()},{k:v[:,:1] for k,v in t.items()},model)
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    changed={k:v.clone() for k,v in t.items()}
    for v in changed.values():v[:,1:]+=100
    torch.testing.assert_close(loss,first_step_loss(p,changed,model),rtol=0,atol=0)
    for grad in torch.autograd.grad(loss,tuple(p.values())):assert torch.count_nonzero(grad[:,1:])==0


def test_normalized_quaternion_small_angle_and_sign():
    angle=1e-5
    a=dict(rates_b=np.zeros((1,1,3)),velocity_n=np.zeros((1,1,3)),quaternion=np.array([[[np.cos(angle/2),np.sin(angle/2),0,0]]])*1.00001)
    b=dict(rates_b=np.zeros((1,1,3)),velocity_n=np.zeros((1,1,3)),quaternion=np.array([[[-1.,0,0,0]]]))
    np.testing.assert_allclose(errors(a,b)[0,0,-1],angle**2,rtol=1e-9)
    np.testing.assert_allclose(errors(b,b),0)
