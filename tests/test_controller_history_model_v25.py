import numpy as np
import torch
import pytest
from system_identification.models.reference_gru_phase_schedule import ReferenceGRUPhaseSchedule
from system_identification.models.reference_gru_controller_history import ReferenceGRUControllerHistory


def fixture():
    torch.set_num_threads(1);torch.manual_seed(17)
    stats={}
    for k,n in [('state',14),('reference',5),('actuator',4),('rate',3),('velocity',3)]:
        if k not in ['rate','velocity']:stats[k+'_mean']=np.zeros(n)
        stats[k+'_scale']=np.ones(n)
    torch.manual_seed(17);old=ReferenceGRUPhaseSchedule(stats)
    torch.manual_seed(17);new=ReferenceGRUControllerHistory(stats,np.ones(8)*.3,np.ones(8)*.2)
    # Nonzero readout avoids a vacuous equivalence test at zero derivatives.
    torch.manual_seed(55)
    with torch.no_grad():old.head[-1].weight.normal_(std=.02);new.head[-1].weight.copy_(old.head[-1].weight)
    x=dict(history_state=torch.randn(2,26,14)*.1,history_reference=torch.randn(2,26,5)*.1,
        history_actuator=torch.randn(2,26,4)*.1,future_reference=torch.randn(2,15,5)*.1,
        initial_quaternion=torch.tensor([[1.,0,0,0]]*2),initial_velocity_n=torch.zeros(2,3),initial_rates_b=torch.zeros(2,3),
        current_mode=torch.tensor([[0.,1.]]*2),dt_s=torch.full((2,15),.02))
    return old,new,x,torch.randn(2,26,8)


def test_enlargement_preserves_common_weights_and_nontrivial_prediction():
    old,new,x,h=fixture()
    assert sum(p.numel() for p in new.parameters())==41606
    for k,v in old.named_parameters():
        got=dict(new.named_parameters())[k]
        torch.testing.assert_close(got[:,:25] if k=='encoder.weight_ih_l0' else got,v,rtol=0,atol=0)
    assert torch.count_nonzero(new.encoder.weight_ih_l0[:,25:])==0
    for active in [False,True]:
        new.use_controller=active
        for k,v in old(**x).items():torch.testing.assert_close(new(**x,history_controller=h)[k],v,atol=1e-6,rtol=1e-6)


def test_zero_arm_has_no_observation_gradient_but_observed_arm_can_learn():
    _,model,x,h=fixture()
    model.use_controller=False
    a=model(**x,history_controller=h);b=model(**x,history_controller=h+100)
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    a['rates_b'].square().sum().backward()
    assert torch.count_nonzero(model.encoder.weight_ih_l0.grad[:,25:])==0
    model.zero_grad();model.use_controller=True
    model(**x,history_controller=h)['rates_b'].square().sum().backward()
    assert torch.linalg.vector_norm(model.encoder.weight_ih_l0.grad[:,25:])>0


def test_suffix_causality_with_nonzero_controller_weights_and_bad_shape():
    _,model,x,h=fixture()
    with torch.no_grad():model.encoder.weight_ih_l0[:,25:].normal_(std=.01)
    a=model(**x,history_controller=h);changed=dict(x);changed['future_reference']=x['future_reference'].clone()
    changed['future_reference'][:,7:]+=1
    b=model(**changed,history_controller=h)
    for k in a:torch.testing.assert_close(a[k][:,:7],b[k][:,:7],rtol=0,atol=0)
    with pytest.raises(ValueError):model(**x,history_controller=h[:,:25])
