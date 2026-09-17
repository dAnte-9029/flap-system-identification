import numpy as np
import torch
from test_trajectory_main_v1 import _samples,_windows
from test_trajectory_main_v2 import _base_model
from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows,fit_main_v1_stats,MainV1Config,fit_history_trajectory_model,_model_call,trajectory_rollout_loss
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from system_identification.training.main_v2_objectives import objective,loss_components,train_stage


def test_objective_original_value_and_gradients_preserved():
    b=assemble_history_trajectory_windows(_samples(),_windows(),history_steps=3)
    p,t=_model_call(_base_model(),b,np.arange(2),use_history=True,rollout_steps=2,device=torch.device('cpu'))
    a=trajectory_rollout_loss(p,t,objective_steps=2);o=objective(p,t,steps=2)
    torch.testing.assert_close(a,o,rtol=0,atol=0)
    torch.testing.assert_close(sum(loss_components(p,t,2).values()),a)
    torch.testing.assert_close(torch.autograd.grad(a,p.angular_velocity_b,retain_graph=True)[0],torch.autograd.grad(o,p.angular_velocity_b,retain_graph=True)[0])
    assert objective(p,t,steps=2,delta_lag=2,delta_weight=1)>o


def test_original_training_loop_bitwise_reproduces_legacy_on_small_batch():
    b=assemble_history_trajectory_windows(_samples(),_windows(),history_steps=3)
    stats=fit_main_v1_stats(_samples(),b)
    config=MainV1Config(model_name='x',use_controls=False,use_history=True,objective_steps=2,epochs=2,hidden_size=8,batch_size=1)
    old,_=fit_history_trajectory_model(b,stats,config,device='cpu')
    torch.manual_seed(config.seed)
    model=CausalHistoryTrajectoryModel(hidden_size=8,use_controls=False,**vars(stats))
    new,_=train_stage(model,b,steps=2,epochs=2,seed=config.seed,learning_rate=config.learning_rate,device='cpu',batch_size=1)
    for k,v in old.state_dict().items():torch.testing.assert_close(new.state_dict()[k],v,rtol=0,atol=0)


def test_delta_loss_rejects_noise_at_correct_state_and_tracks_vector_direction():
    b=assemble_history_trajectory_windows(_samples(),_windows(),history_steps=3)
    p,t=_model_call(_base_model(),b,np.arange(2),use_history=True,rollout_steps=2,device=torch.device('cpu'))
    from dataclasses import replace
    exact=replace(p,angular_velocity_b=t.angular_velocity_b.clone().requires_grad_())
    noisy=replace(exact,angular_velocity_b=exact.angular_velocity_b+torch.tensor([0.,1.,-1.])[None,:,None])
    extra=lambda x:objective(x,t,steps=2,delta_weight=1)-objective(x,t,steps=2)
    assert abs(float(extra(exact)))<1e-6
    assert extra(noisy)>0
    extra(noisy).backward()
    assert torch.isfinite(exact.angular_velocity_b.grad).all()


def test_original_actuator_training_loop_reproduces_legacy():
    from system_identification.training.trajectory_main_v2 import (
        MainV2Config, fit_main_v2_stats, fit_actuator_aware_model,
    )
    from system_identification.models.trajectory_main_v2 import ActuatorAwareTrajectoryModel
    import copy
    b = assemble_history_trajectory_windows(_samples(), _windows(), history_steps=3)
    stats = fit_main_v2_stats(b)
    config = MainV2Config('original', True, True, True, 2, epochs=2, batch_size=1)
    initial = _base_model()
    old, _ = fit_actuator_aware_model(b, copy.deepcopy(initial), stats, config, device='cpu')
    torch.manual_seed(config.seed)
    new = ActuatorAwareTrajectoryModel(
        base_model=copy.deepcopy(initial), use_drive=True, use_tail=True,
        gated_tail=True, **vars(stats),
    )
    new, _ = train_stage(new, b, steps=2, epochs=2, seed=config.seed,
        learning_rate=config.learning_rate, device='cpu', batch_size=1, actuator=True)
    for key, value in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[key], value, rtol=0, atol=0)
