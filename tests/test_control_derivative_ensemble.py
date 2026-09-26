import copy
import numpy as np
import pytest
import torch

from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from system_identification.models.control_ensemble import ControlDerivativeEnsemble


def setup():
    torch.manual_seed(8)
    m=CausalHistoryTrajectoryModel(hidden_size=8,use_controls=True,
        feature_mean=np.zeros(12),feature_std=np.ones(12),control_mean=np.zeros(4),control_std=np.ones(4),
        derivative_mean=np.zeros(7),derivative_std=np.ones(7))
    torch.nn.init.normal_(m.derivative_head[-1].weight,std=.1)
    kw=dict(history_state_features=torch.randn(2,26,12),history_controls=torch.randn(2,26,4),
        history_mask=torch.ones(2,26,dtype=torch.bool),position_n=torch.zeros(2,3),velocity_n=torch.randn(2,3),
        quaternion_nb=torch.tensor([[1.,0,0,0]]).repeat(2,1),angular_velocity_b=torch.randn(2,3),
        relative_phase_rad=torch.ones(2),flap_frequency_hz=torch.ones(2)*5,
        future_controls=torch.randn(2,10,4),dt_s=torch.ones(2,10)*.02)
    return m,kw


def test_single_member_matches_original_transition():
    m,kw=setup();e=ControlDerivativeEnsemble([m])
    with torch.inference_mode():
        for a,b in zip(m(**kw),e(**kw)):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_identical_members_match_and_no_hidden_mutation():
    m,kw=setup();e=ControlDerivativeEnsemble([m,copy.deepcopy(m),copy.deepcopy(m)])
    state=e.reset(**{k:v for k,v in kw.items() if k not in ('future_controls','dt_s')})
    before=tuple(h.clone() for h in state.hidden)
    _,d=e.step(state,kw['future_controls'][:,0],kw['dt_s'][:,0])
    for a,b in zip(state.hidden,before):torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(d['derivative_std'],torch.zeros_like(d['derivative_std']),rtol=0,atol=0)
    for a,b in zip(m(**kw),e(**kw)):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)


def test_resume_and_future_command_prefix():
    m,kw=setup();e=ControlDerivativeEnsemble([m])
    a=e(**kw);modified={**kw,'future_controls':kw['future_controls'].clone()}
    modified['future_controls'][:,5:]+=4
    b=e(**modified)
    for x,y in zip(a,b):torch.testing.assert_close(x[:,:6],y[:,:6],rtol=0,atol=0)
    state=e.reset(**{k:v for k,v in kw.items() if k not in ('future_controls','dt_s')})
    for step in range(10):
        state,_=e.step(state,kw['future_controls'][:,step],kw['dt_s'][:,step])
        if step==4:state=copy.deepcopy(state)
    torch.testing.assert_close(state.angular_velocity_b,a.angular_velocity_b[:,-1],rtol=0,atol=0)


def test_reject_invalid_time():
    m,kw=setup();e=ControlDerivativeEnsemble([m])
    state=e.reset(**{k:v for k,v in kw.items() if k not in ('future_controls','dt_s')})
    with pytest.raises(ValueError):e.step(state,kw['future_controls'][:,0],torch.zeros(2))
