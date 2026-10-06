import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from test_controller_history_model_v25 import fixture
from system_identification.models.reference_gru_controller_no_age import ReferenceGRUControllerNoAge


def setup():
    _,parent,x,h=fixture()
    stats={k:v.numpy() for k,v in parent.named_buffers() if k not in ['controller_mean','controller_scale','derivative_scale']}
    m=ReferenceGRUControllerNoAge(stats,np.ones(8)*.3,np.ones(8)*.2)
    m.load_state_dict(parent.state_dict())
    with torch.no_grad():m.encoder.weight_ih_l0[:,25:].normal_(std=.02)
    parent.load_state_dict(m.state_dict())
    return parent,m,x,h


def test_normalized_mask_and_nonzero_readout_equal_explicit_reference():
    parent,m,x,h=setup();before=h.clone();captured=[]
    hook=m.encoder.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach()))
    a=m(**x,history_controller=h);hook.remove()
    masked=h.clone();masked[...,6:]=parent.controller_mean[6:]
    b=parent(**x,history_controller=masked)
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    torch.testing.assert_close(h,before,rtol=0,atol=0)
    assert not captured[0][...,-2:].count_nonzero()
    torch.testing.assert_close(captured[0][...,-8:-2],(h[...,:6]-m.controller_mean[:6])/m.controller_scale[:6])
    assert sum(p.numel() for p in m.parameters())==41606


def test_ages_cannot_change_prediction_or_receive_gradient():
    _,m,x,h=setup();h.requires_grad_();a=m(**x,history_controller=h)
    changed=h.detach().clone();changed[...,6:]+=100
    b=m(**x,history_controller=changed)
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    a['rates_b'].square().sum().backward()
    assert not h.grad[...,6:].count_nonzero()
    assert h.grad[...,:6].abs().sum()>0
    assert not m.encoder.weight_ih_l0.grad[:,-2:].count_nonzero()


def test_future_suffix_cannot_change_prefix():
    _,m,x,h=setup();a=m(**x,history_controller=h)
    x['future_reference'][:,8:]+=1
    b=m(**x,history_controller=h)
    for k in a:torch.testing.assert_close(a[k][:,:8],b[k][:,:8],rtol=0,atol=0)
