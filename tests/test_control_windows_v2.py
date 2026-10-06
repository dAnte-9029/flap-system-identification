import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_control_windows_v2 import choose_origins,input_events,cluster_events
from system_identification.models.reference_gru import ReferenceGRU


def test_event_first_windows_have_no_shared_samples_and_keep_boundaries():
    events=[(50,0,.1),(100,0,-.1),(300,1,.1)]
    origins=choose_origins(0,500,events)
    assert origins[0]==45
    for a,b in zip(origins[:-1],origins[1:]):assert a+50<b-26
    assert all(i-26>=0 and i+50<500 for i in origins)


def test_input_changes_and_cluster_context_do_not_need_outputs():
    x=np.zeros((600,5));x[100:200,0]=.1;x[250:400,1]=-.1
    events=input_events(x,0,600)
    assert any(e[2]>0 for e in events) and any(e[2]<0 for e in events)
    c=cluster_events(events,0,600)
    assert len(c)==1 and c[0]['start']==50 and c[0]['stop']==551


def model_and_input():
    stats={}
    for name,n in [('state',14),('reference',5),('actuator',4)]:
        stats[name+'_mean']=np.zeros(n);stats[name+'_scale']=np.ones(n)
    stats.update(rate_scale=np.ones(3),velocity_scale=np.ones(3))
    torch.manual_seed(3);model=ReferenceGRU(stats)
    # Nonzero head exercises command sensitivity, not just the zero-head initial state.
    torch.nn.init.normal_(model.head[-1].weight,std=.01)
    x=dict(history_state=torch.randn(2,26,14),history_reference=torch.randn(2,26,5),history_actuator=torch.randn(2,26,4),
        future_reference=torch.randn(2,10,5),initial_quaternion=torch.tensor([[1.,0,0,0]]).repeat(2,1),
        initial_velocity_n=torch.randn(2,3),initial_rates_b=torch.randn(2,3),current_mode=torch.tensor([[1.,0],[0,1.]]),dt_s=torch.full((2,10),.02))
    x['history_state'][:,:,11]=4.
    return model,x


def test_model_future_reference_tail_cannot_change_prefix():
    model,x=model_and_input();model.eval()
    with torch.no_grad():
        a=model(**x);x['future_reference'][:,5:]+=100;b=model(**x)
    for key in a:torch.testing.assert_close(a[key][:,:5],b[key][:,:5],rtol=0,atol=0)
    torch.testing.assert_close(a['quaternion'].norm(dim=-1),torch.ones(2,10))


def test_finite_gradients_and_reference_effect():
    model,x=model_and_input();x['future_reference'].requires_grad_()
    pred=model(**x);sum(v.square().mean() for v in pred.values()).backward()
    assert torch.isfinite(x['future_reference'].grad).all()
    assert x['future_reference'].grad.abs().sum()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_evaluation_primary_weights_flights_before_axes_and_horizons():
    from evaluate_control_windows_v2 import primary_mean
    cube=np.zeros((4,3,2,2,3))
    cube[:,:,0]=1
    cube[:,:,1]=3
    np.testing.assert_allclose(primary_mean(cube),2)
    # A missing event axis in one flight does not discard its other axis.
    cube[:,:,0,0]=np.nan
    np.testing.assert_allclose(primary_mean(cube),2.5)


def test_training_attitude_loss_is_quaternion_sign_invariant():
    from train_control_windows_v2 import loss_fn
    model,x=model_and_input()
    pred=model(**x)
    truth={k:v.detach().clone() for k,v in pred.items()}
    first=loss_fn(pred,truth,model)
    truth['quaternion']=-truth['quaternion']
    torch.testing.assert_close(first,loss_fn(pred,truth,model))
