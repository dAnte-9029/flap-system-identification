from dataclasses import replace
import torch
from test_joint_control_model import model
from test_trajectory_main_v2 import _inputs
from system_identification.models.main_v2_simulator import MainV2Simulator


def constrained():
    m=model();m.actuator_only_rollout=True;m.signed_tail_effectiveness=True
    return m


def test_commands_only_change_proxies_on_first_step_and_have_signed_direct_effect():
    m=constrained();sim=MainV2Simulator(m);x=_inputs(2,10);s=sim.reset(**{k:v for k,v in x.items() if k not in ['future_controls','dt_s']});u=x['future_controls'][:,0];dt=x['dt_s'][:,0]
    a,_=sim.step(s,u,dt);b,_=sim.step(s,u+.1,dt)
    for name in ['position_n','velocity_n','quaternion_nb','angular_velocity_b','gru_hidden','flap_frequency_hz']:
        torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=0,atol=0)
    assert not torch.equal(a.tail_state,b.tail_state)
    assert not torch.equal(a.drive_state,b.drive_state)
    for channel,axis,sign in [(0,1,1),(1,0,-1),(2,2,1)]:
        plus=s.tail_state.clone();minus=plus.clone();plus[:,channel]+=.001;minus[:,channel]-=.001
        _,dp=sim.step(replace(s,tail_state=plus),u,dt);_,dm=sim.step(replace(s,tail_state=minus),u,dt)
        assert torch.all(sign*(dp.angular_acceleration_b[:,axis]-dm.angular_acceleration_b[:,axis])>0)


def test_constrained_training_parity_and_gradients():
    m=constrained();x=_inputs(2,30);x['future_controls']=torch.rand_like(x['future_controls']);pred=m(**x)
    sim=MainV2Simulator(m);s=sim.reset(**{k:v for k,v in x.items() if k not in ['future_controls','dt_s']});states=[s]
    for k in range(30):s,_=sim.step(s,x['future_controls'][:,k],x['dt_s'][:,k]);states.append(s)
    for name in ['position_n','velocity_n','quaternion_nb','angular_velocity_b','flap_frequency_hz']:
        torch.testing.assert_close(torch.stack([getattr(s,name) for s in states],1),getattr(pred,name),rtol=0,atol=0)
    pred.angular_velocity_b.square().sum().backward()
    for block in [m.base_model.recurrent_cell,m.base_model.derivative_head,m.tail_heads]:
        assert sum(p.grad.abs().sum() for p in block.parameters() if p.grad is not None)>0
