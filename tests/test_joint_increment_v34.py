import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from joint_increment_v34 import balanced_offsets,select_local,local_increment_loss,objective
from test_rollout_trace_v32 import case
import train_control_windows_v2 as base


def test_offsets_cover_each_parent_in_shuffled_cycles_without_rng_side_effects():
    draws=np.array([[3,9]]*117)
    torch.manual_seed(9);before=torch.get_rng_state().clone()
    offsets=balanced_offsets(draws,[3,9],2,17)
    assert torch.equal(before,torch.get_rng_state())
    np.testing.assert_array_equal(offsets,balanced_offsets(draws,[3,9],2,17))
    for j in range(2):
        for start in [0,50]:np.testing.assert_array_equal(np.sort(offsets[start:start+50,j]),np.arange(50))
        counts=np.bincount(offsets[:,j],minlength=50);assert counts.max()-counts.min()==1
    assert not np.array_equal(offsets,balanced_offsets(draws,[3,9],2,23))
    with pytest.raises(ValueError):balanced_offsets(draws,[3],2,17)


def test_local_batch_blocks_query_and_excludes_reference_inputs():
    x={'history_state':torch.arange(4*50).reshape(4,50,1),'history_reference':torch.zeros(4,50,1)}
    y={'rates_b':torch.arange(4*50*3).reshape(4,50,3)}
    a,b=select_local(x,y,np.array([1,1]),np.array([0,49]),[0,1])
    assert set(a)=={'history_state'} and b['rates_b'].shape==(2,1,3)
    assert a['history_state'].flatten().tolist()==[50,99]
    x['history_state'][2:]=999;y['rates_b'][2:]=-999
    c,d=select_local(x,y,np.array([1,1]),np.array([0,49]),[0,1])
    assert torch.equal(a['history_state'],c['history_state']) and torch.equal(b['rates_b'],d['rates_b'])
    with pytest.raises(ValueError):select_local(x,y,np.array([2]),np.array([0]),[0,1])
    with pytest.raises(ValueError):select_local(x,y,np.array([1]),np.array([50]),[0,1])


def test_rate_units_axis_average_and_increment_identity():
    prediction={'rates_b':torch.tensor([[[3.,6.,9.]]])};truth={'rates_b':torch.tensor([[[1.,2.,3.]]])}
    scale=torch.tensor([1.,2.,3.]);assert local_increment_loss(prediction,truth,scale)==4
    initial=torch.tensor([[[2.,-7.,5.]]])
    expected=(((prediction['rates_b']-initial)-(truth['rates_b']-initial))/scale).square().mean()
    assert local_increment_loss(prediction,truth,scale)==expected


def test_zero_weight_reproduces_old_loss_and_gradients_exactly():
    model,x,truth=case();truth={k:v.detach()+(.1 if k=='rates_b' else 0) for k,v in truth.items()}
    old=base.loss_fn(model(**x),truth,model);old.backward()
    gradients=[p.grad.clone() for p in model.parameters()];model.zero_grad(set_to_none=True)
    new,full,local=objective(model,x,truth,weight=0);assert local is None
    torch.testing.assert_close(new,old,rtol=0,atol=0);new.backward()
    for p,g in zip(model.parameters(),gradients):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)


def test_joint_objective_is_fixed_sum_and_has_local_gradient():
    model,x,y=case();y={k:v.detach() for k,v in y.items()}
    local_x=dict(x);local_x['future_command']=x['future_command'][:,:1];local_x['dt_s']=x['dt_s'][:,:1]
    local_y={k:v[:,:1].clone() for k,v in y.items()};local_y['rates_b']+=.25
    total,full,local=objective(model,x,y,local_x,local_y)
    torch.testing.assert_close(total,full+local,rtol=0,atol=0)
    gradient=torch.autograd.grad(local,model.head[-1].weight)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
