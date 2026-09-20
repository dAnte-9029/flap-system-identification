from dataclasses import fields,replace
import pytest
import torch
from test_main_v2_simulator import make_simulator,make_inputs,initial
from system_identification.integration.isaac_state_backend import IsaacLearnedStateBackend,observation


def test_native_rollout_parity_at_half_and_one_second_and_resume():
    sim=make_simulator();x=make_inputs();x['dt_s']=torch.full_like(x['dt_s'],.02)
    backend=IsaacLearnedStateBackend(sim,checkpoint_id='fixture')
    with torch.inference_mode():
        native=sim.model(**x);backend.reset(state=initial(sim,x))
        for k in range(50):
            backend.step(x['future_controls'][:,k])
            if k==24:
                clone=IsaacLearnedStateBackend(sim,checkpoint_id='fixture');clone.restore(backend.snapshot())
            elif k>24:
                clone.step(x['future_controls'][:,k])
                for f in fields(backend.state):torch.testing.assert_close(getattr(clone.state,f.name),getattr(backend.state,f.name),rtol=0,atol=0)
            if k in [24,49]:
                for name in ['position_n','velocity_n','quaternion_nb','angular_velocity_b','relative_phase_rad','flap_frequency_hz']:
                    torch.testing.assert_close(getattr(backend.state,name),getattr(native,name)[:,k+1],rtol=0,atol=0)


def test_frame_mapping_yaw_and_origin_and_read_only_observation():
    sim=make_simulator();state=initial(sim,make_inputs())
    q=torch.tensor([[2**-.5,0.,0.,2**-.5]]).repeat(2,1)
    state=replace(state,quaternion_nb=q,position_n=torch.tensor([[1.,2.,-3.]]).repeat(2,1),angular_velocity_b=torch.tensor([[0.,0.,1.]]).repeat(2,1))
    obs=observation(state,torch.tensor([[10.,20.,30.]]).repeat(2,1))
    torch.testing.assert_close(obs.position_w,torch.tensor([[11.,18.,33.]]).repeat(2,1))
    assert (obs.quaternion_wb[:,3]<0).all() and (obs.angular_velocity_b[:,2]==-1).all()


def test_fail_closed_clock_action_and_identity():
    sim=make_simulator()
    with pytest.raises(ValueError):IsaacLearnedStateBackend(sim,checkpoint_id='a',model_dt_s=1/480)
    b=IsaacLearnedStateBackend(sim,checkpoint_id='a')
    with pytest.raises(RuntimeError):b.observe()
    b.reset(state=initial(sim,make_inputs()))
    with pytest.raises(ValueError):b.step(torch.full((2,4),2.))
    c=IsaacLearnedStateBackend(sim,checkpoint_id='b')
    with pytest.raises(ValueError):c.restore(b.snapshot())
    before=b.snapshot()
    for _ in range(100):b.observe()
    assert b.steps==before['steps']
