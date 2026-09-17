import copy
import numpy as np
import torch
from test_trajectory_main_v1 import _samples, _windows
from test_trajectory_main_v2 import _base_model
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows, _model_call
from system_identification.training.main_v2_objectives import train_stage
from system_identification.training.main_v2_increment import increment_terms,train_increment_stage


def test_increment_uses_native_two_step_transition():
    samples = _samples()
    # Deliberately unequal native intervals: 13 ms then 27 ms.
    samples['timestamp_us'] += np.where(samples.sample_in_segment % 2, -7000, 0)
    b=assemble_history_trajectory_windows(samples,_windows(),history_steps=3)
    np.testing.assert_allclose(b.trajectory.dt_s[0], [.013, .027])
    m=_base_model();p,t=_model_call(m,b,np.arange(2),use_history=True,rollout_steps=2,device=torch.device('cpu'))
    terms=increment_terms(p,t,(.2,.3))
    expected=((p.velocity_n[:,2]-p.velocity_n[:,0])-(t.velocity_n[:,2]-t.velocity_n[:,0]))/.2
    torch.testing.assert_close(terms[0],expected.square().sum(-1).mean())
    sum(terms).backward()
    assert any(x.grad is not None and x.grad.abs().sum()>0 for x in m.derivative_head.parameters())


def test_increment_term_does_not_read_labels_after_second_step():
    from dataclasses import replace
    windows = _windows(); windows['state_sample_count'] = 5
    b=assemble_history_trajectory_windows(_samples(),windows,history_steps=3)
    p,t=_model_call(_base_model(),b,np.arange(2),use_history=True,rollout_steps=4,device=torch.device('cpu'))
    expected=increment_terms(p,t,(.2,.3))
    fields={}
    for name in ['velocity_n','angular_velocity_b']:
        value=getattr(t,name).clone();value[:,3:]=float('nan');fields[name]=value
    actual=increment_terms(p,replace(t,**fields),(.2,.3))
    for x,y in zip(actual,expected):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_zero_increment_weight_reproduces_original_training_bitwise():
    b=assemble_history_trajectory_windows(_samples(),_windows(),history_steps=3)
    m=_base_model();a,_=train_stage(copy.deepcopy(m),b,steps=2,epochs=2,seed=17,learning_rate=.0003,device='cpu',batch_size=1)
    z,_=train_increment_stage(copy.deepcopy(m),b,steps=2,epochs=2,seed=17,learning_rate=.0003,device='cpu',batch_size=1,scales=(.2,.3),weights=(0.,0.))
    for k,v in a.state_dict().items():torch.testing.assert_close(z.state_dict()[k],v,rtol=0,atol=0)


def test_increment_noise_cannot_be_rewarded_as_variation():
    from dataclasses import replace
    b=assemble_history_trajectory_windows(_samples(),_windows(),history_steps=3)
    p,t=_model_call(_base_model(),b,np.arange(2),use_history=True,rollout_steps=2,device=torch.device('cpu'))
    exact=replace(p,velocity_n=t.velocity_n.clone(),angular_velocity_b=t.angular_velocity_b.clone())
    assert sum(increment_terms(exact,t,(1.,1.)))==0
    noisy=exact.angular_velocity_b.clone();noisy[:,2,1]+=1
    assert sum(increment_terms(replace(exact,angular_velocity_b=noisy),t,(1.,1.)))>0
