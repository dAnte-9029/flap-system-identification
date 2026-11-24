import numpy as np
import pytest
import torch

from controls import pack
from dynamics import clock, query
from fit import make_model


@pytest.mark.parametrize('name',['gru','slowfast','lstm','tcn','transformer'])
def test_within_step_causality_and_control_gradient(name):
    check_control_event(lambda mean, scale: make_model(name, mean, scale), query)


def check_control_event(build, native_query):
    torch.manual_seed(17)
    torch.set_num_threads(1)
    h = torch.zeros(1,26,30)
    h[:,-1,15] = 3.
    h[:,-1,21:23] = 1.
    s = torch.zeros(1,13)
    s[:,6] = 1.
    times = np.array([0,67000])
    u = np.array([[.5,0.,0.,0.],[.8,.3,-.3,0.]])
    origins,stops = np.array([0]),np.array([200000])
    plan = torch.from_numpy(pack(times,u,origins,stops)[:,:10]).requires_grad_()
    held = torch.from_numpy(pack(times[:1],u[:1],origins,stops)[:,:10])
    assert plan[0,3,0,-1]==7000 and plan[0,3,1,-1]==13000
    m = build(torch.zeros(30),torch.ones(30))
    a,b = m(h,s,plan),m(h,s,held)
    t = torch.tensor([[0.,.06,.064,.067,.079,.18]])
    for component in ('pv','q','w'):
        x,y = native_query(a,t,component),native_query(b,t,component)
        torch.testing.assert_close(x[:,:4],y[:,:4],rtol=0,atol=1e-7)
    w = native_query(a,t,'w')
    assert not torch.equal(w[:,-1],native_query(b,t,'w')[:,-1])
    before = torch.autograd.grad(w[:,:4].sum(),plan,retain_graph=True)[0]
    assert before[0,3,1,:4].abs().max()==0
    after = torch.autograd.grad(w[:,-1].sum(),plan)[0][0,3,1,:4]
    assert after.isfinite().all() and (after.abs()>0).all()
    # The frequency at 80 ms uses 67 ms of the first command and 13 ms of the new one.
    def advance(f,m,dt):
        eq = (3.6354511983566393+23.393570068972757*m)/5.126725395181777
        return eq+(f-eq)*np.exp(-5.126725395181777*dt)
    freq = clock(h,plan)[2].reshape(1,10,2,1)
    np.testing.assert_allclose(freq[0,4,0,0].item(),advance(advance(3.,.5,.067),.8,.013),rtol=1e-6)


@pytest.mark.parametrize('name',['gru','slowfast','lstm','tcn','transformer'])
def test_mpc_rollout_prefix_and_endpoints(name):
    torch.manual_seed(8)
    h = torch.zeros(2,26,30)
    h[:,-1,15] = 3.7
    h[:,-1,21:23] = 1.
    initial = torch.zeros(2,13)
    initial[:,6] = 1.
    commands = torch.rand(2,13,4,requires_grad=True)
    m = make_model(name,torch.zeros(30),torch.ones(30))
    path = m.rollout(h,initial,commands)
    assert path.shape == (2,14,13) and torch.equal(path[:,0],initial)
    torch.testing.assert_close(path[:,:8],m.rollout(h,initial,commands[:,:7]),rtol=1e-5,atol=2e-7)
    gradient = torch.autograd.grad(path[:,7,10:].sum(),commands)[0]
    assert gradient[:,:7].isfinite().all() and gradient[:,:7].abs().sum()>0
    assert gradient[:,7:].abs().max()==0


@pytest.mark.parametrize('name',['tcn','transformer'])
def test_cached_sequence_matches_full_causal_sequence(name):
    """Exercise the 32-token attention boundary and every dilated-convolution cache."""
    from backbones import BACKBONES
    torch.manual_seed(11)
    torch.set_num_threads(1)
    model = BACKBONES[name](30,22).double()
    history = torch.randn(3,26,30,dtype=torch.float64)
    steps = torch.randn(3,17,22,dtype=torch.float64,requires_grad=True)
    cache = model.initialize(history,steps.size(1))
    streamed = []
    for token in steps.unbind(1):
        out,cache = model.step(token,cache)
        streamed.append(out)
    full = torch.cat([model.history_input(history),model.step_input(steps)],1)
    if name == 'transformer':
        angles=torch.arange(full.size(1))[:,None]*model.frequencies
        full = full + torch.stack([angles.sin(),angles.cos()],-1).flatten(-2)
        for layer in model.layers:
            full,_=layer(full)
        full=model.final_norm(full)
    else:
        for layer in model.layers:
            full,_=layer.initialize(full)
    full=full[:,history.size(1):]
    stream=torch.stack(streamed,1)
    torch.testing.assert_close(stream,full)
    weight=torch.randn_like(full)
    g1=torch.autograd.grad((stream*weight).sum(),steps)[0]
    g2=torch.autograd.grad((full*weight).sum(),steps)[0]
    torch.testing.assert_close(g1,g2)
