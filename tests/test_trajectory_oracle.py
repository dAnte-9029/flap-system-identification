import numpy as np
import pytest
import torch
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from system_identification.evaluation.trajectory_oracle import oracle_rollout


def setup_case():
    torch.manual_seed(17)
    m=CausalHistoryTrajectoryModel(hidden_size=8,use_controls=True,feature_mean=np.zeros(12),feature_std=np.ones(12),control_mean=np.zeros(4),control_std=np.ones(4),derivative_mean=np.array([1.,0,0,0,0,0,0]),derivative_std=np.ones(7)).eval()
    x=dict(history_state_features=torch.zeros(2,3,12),history_controls=torch.zeros(2,3,4),history_mask=torch.ones(2,3,dtype=torch.bool),position_n=torch.zeros(2,3),velocity_n=torch.zeros(2,3),quaternion_nb=torch.tensor([[1.,0,0,0]]*2),angular_velocity_b=torch.zeros(2,3),relative_phase_rad=torch.zeros(2),flap_frequency_hz=torch.ones(2)*4,future_controls=torch.zeros(2,5,4),dt_s=torch.tensor([[.02,.021,.018,.019,.022]]*2))
    return m,x


def test_no_intervention_exactly_reproduces_production_with_nonzero_head():
    m,x=setup_case()
    with torch.no_grad():m.derivative_head[-1].weight.normal_(0,.05)
    a=m(**x);b=oracle_rollout(m,**x)
    for first,second in zip(a,b):torch.testing.assert_close(first,second,rtol=0,atol=0)
    with pytest.raises(ValueError):oracle_rollout(m,oracle_quaternion=a.quaternion_nb,**x)


def test_rotation_only_keeps_attitude_and_rates_free_and_changes_acceleration_direction():
    m,x=setup_case();base=m(**x)
    q=base.quaternion_nb.clone();q[:,1:]=torch.tensor([2**-.5,0.,0.,2**-.5])
    pred=oracle_rollout(m,mode='oracle_attitude_rotation',oracle_quaternion=q,**x)
    torch.testing.assert_close(pred.quaternion_nb,base.quaternion_nb)
    torch.testing.assert_close(pred.angular_velocity_b,base.angular_velocity_b)
    torch.testing.assert_close(pred.velocity_n[:,1],base.velocity_n[:,1])
    assert (pred.velocity_n[:,-1,1]>0).all()
    assert (pred.velocity_n[:,-1,0]<base.velocity_n[:,-1,0]).all()


def test_feedback_clamps_attitude_but_not_position_and_rate_integrates_attitude():
    m,x=setup_case();base=m(**x)
    q=base.quaternion_nb.clone();q[:,2:]=torch.tensor([2**-.5,0.,0.,2**-.5])
    pred=oracle_rollout(m,mode='oracle_attitude_feedback',oracle_quaternion=q,**x)
    torch.testing.assert_close(pred.quaternion_nb,q)
    torch.testing.assert_close(pred.velocity_n[:,:3],base.velocity_n[:,:3])
    rate=base.angular_velocity_b.clone();rate[:,1:,2]=1
    r=oracle_rollout(m,mode='oracle_rate',oracle_rate=rate,**x)
    torch.testing.assert_close(r.angular_velocity_b,rate)
    assert (r.quaternion_nb[:,-1,3]>0).all()
    torch.testing.assert_close(torch.linalg.vector_norm(r.quaternion_nb,dim=-1),torch.ones(2,6))


def test_feedback_enters_recurrent_update_and_future_suffix_does_not_change_prefix():
    m,x=setup_case()
    with torch.no_grad():m.derivative_head[-1].weight.normal_(0,.05)
    q=m(**x).quaternion_nb.detach().clone()
    q2=q.clone();q2[:,4:]=torch.tensor([2**-.5,0.,0.,2**-.5])
    a=oracle_rollout(m,mode='oracle_attitude_feedback',oracle_quaternion=q,**x)
    b=oracle_rollout(m,mode='oracle_attitude_feedback',oracle_quaternion=q2,**x)
    for name in ['position_n','velocity_n','angular_velocity_b']:
        torch.testing.assert_close(getattr(a,name)[:,:5],getattr(b,name)[:,:5])
    assert not torch.allclose(a.angular_velocity_b[:,-1],b.angular_velocity_b[:,-1])
