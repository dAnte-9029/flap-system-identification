import torch
import pytest
import math
from system_identification.models.pahbd.dynamics import PhasePropagator

def vanloan(c,dt,q,b):
    A=c.new_tensor([[b,0.],[2*math.pi,0.]])
    E=torch.matrix_exp(torch.cat((torch.cat((A,q),1),torch.cat((torch.zeros_like(A),-A.T),1)),0)*dt)
    F=E[:2,:2];r=F@c@F.T+E[:2,2:]@F.T
    return (r+r.T)/2

@pytest.mark.parametrize('b',[-5.126725395181777,0.,1e-10,-.001])
@pytest.mark.parametrize('dt',[0.,1e-8,1e-4,.005,.1,.5,2.])
def test_covariance_value_gradient_semigroup(b,dt):
    p=PhasePropagator((3.,20.,b))
    c=torch.tensor([[.1,.02],[.02,.3]],dtype=torch.float64,requires_grad=True)
    q=torch.tensor([[.04,.003],[.003,.01]],dtype=torch.float64,requires_grad=True)
    x=p.covariance(c,dt,q);y=vanloan(c,dt,q,b)
    torch.testing.assert_close(x,y,atol=2e-10,rtol=2e-10)
    gx=torch.autograd.grad(x.square().sum(),(c,q),retain_graph=True)
    gy=torch.autograd.grad(y.square().sum(),(c,q))
    for a,z in zip(gx,gy):torch.testing.assert_close(a,z,atol=2e-9,rtol=2e-9)
    z=p.covariance(p.covariance(c,dt*.37,q),dt*.63,q)
    torch.testing.assert_close(x,z,atol=2e-12,rtol=2e-12)
    assert torch.linalg.eigvalsh(x).min()>=-1e-12

def test_control_cache_invalidation_and_differentiable_control():
    from system_identification.models.pahbd import ControlTape
    v=torch.tensor([[.5,0,0,0],[.5,0,0,0],[.5,1,0,0]],dtype=torch.float64)
    t=ControlTape([0.,.1,.2],v)
    assert t.canonical().times==[0.,.2]
    v[1,1]=.5
    assert t.canonical().times==[0.,.1,.2]
    v.requires_grad_()
    for _ in range(2):
        c=t.canonical();assert len(c.times)==3
        c.values.sum().backward();assert torch.equal(v.grad,torch.ones_like(v));v.grad=None
