import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from attitude_supervision_v42 import tilt_loss,objective,calibrated_weight
from joint_increment_v34 import objective as old_objective
from system_identification.models.trajectory_main_v1 import _quaternion_multiply
from test_actuator_command_step import fixture


def test_tilt_sign_yaw_invariance_and_small_angle_units():
    identity=torch.tensor([[[1.,0,0,0]]],dtype=torch.float64)
    yaw=torch.tensor([[[np.cos(.4),0,0,np.sin(.4)]]],dtype=torch.float64)
    roll=torch.tensor([[[np.cos(.01),np.sin(.01),0,0]]],dtype=torch.float64)
    assert tilt_loss(yaw,identity)==0
    torch.testing.assert_close(tilt_loss(roll,identity),torch.tensor(4*np.sin(.01)**2/.01,dtype=torch.float64))
    torch.testing.assert_close(tilt_loss(-roll,identity),tilt_loss(roll,identity),rtol=0,atol=0)
    torch.testing.assert_close(tilt_loss(_quaternion_multiply(yaw,roll),identity),tilt_loss(roll,identity),rtol=1e-12,atol=1e-12)


def test_tilt_finite_gradient_near_euler_pitch_pole():
    q=torch.tensor([[[.70710678,0,.70710678,0]]],requires_grad=True)
    loss=tilt_loss(q,torch.tensor([[[1.,0,0,0]]]))
    loss.backward();assert torch.isfinite(q.grad).all() and q.grad.abs().sum()>0


def test_zero_weight_preserves_v35_gradient_and_positive_weight_changes_it():
    model,history,commands,dt=fixture()
    x=dict(**history,future_command=commands,dt_s=dt)
    lx=dict(**history,future_command=commands[:,:1],dt_s=dt[:,:1])
    y=dict(quaternion=torch.tensor([[[1.,0,0,0]]]).repeat(2,50,1),velocity_n=torch.zeros(2,50,3),rates_b=torch.zeros(2,50,3))
    ly={k:v[:,:1] for k,v in y.items()}
    a=old_objective(model,x,y,lx,ly,weight=1.)[0]
    b=objective(model,x,y,lx,ly,weight=0.)[0]
    ga=torch.autograd.grad(a,tuple(model.parameters()));gb=torch.autograd.grad(b,tuple(model.parameters()))
    for u,v in zip(ga,gb):torch.testing.assert_close(u,v,rtol=0,atol=0)
    c=objective(model,x,y,lx,ly,weight=.5)[0]
    gc=torch.autograd.grad(c,tuple(model.parameters()))
    assert all(torch.isfinite(v).all() for v in gc) and any(not torch.equal(u,v) for u,v in zip(ga,gc))


def test_weight_rule_has_no_query_inputs():
    weight,raw=calibrated_weight(np.full(12,4.),np.full(12,2.))
    assert weight==raw==1.
    assert calibrated_weight(np.full(12,100.),np.ones(12))==(10.,50.)
    with pytest.raises(ValueError):calibrated_weight(np.ones(12),np.zeros(12))
