import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_control_windows_v2 import model_and_input
from system_identification.models.reference_gru_unit_gravity import ReferenceGRUUnitGravity
from system_identification.models.reference_gru_direct_rate import ReferenceGRUDirectRate


def models():
    old,x=model_and_input();names=['state_mean','state_scale','reference_mean','reference_scale','actuator_mean','actuator_scale','rate_scale','velocity_scale'];stats={k:old.state_dict()[k] for k in names}
    torch.manual_seed(17);a=ReferenceGRUUnitGravity(stats)
    torch.manual_seed(17);b=ReferenceGRUDirectRate(stats)
    return a,b,x


def test_initialization_and_zero_head_are_identical():
    a,b,x=models();assert sum(v.numel() for v in a.parameters())==sum(v.numel() for v in b.parameters())==40070
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    pa=a(**x);pb=b(**x)
    for k in pa:torch.testing.assert_close(pa[k],pb[k],rtol=0,atol=0)


def test_constant_direct_head_is_origin_residual_not_accumulated_rate():
    a,b,x=models()
    with torch.no_grad():
        b.head[-1].bias[3:]=torch.tensor([.2,-.1,.3])
    p=b(**x);expected=x['initial_rates_b']+torch.tensor([.2,-.1,.3])*b.rate_scale
    torch.testing.assert_close(p['rates_b'],expected[:,None].expand_as(p['rates_b']))
    torch.testing.assert_close(p['quaternion'].norm(dim=-1),torch.ones_like(x['dt_s']))


def test_direct_rate_gradients_and_reference_prefix_are_causal():
    _,b,x=models();torch.nn.init.normal_(b.head[-1].weight,std=.01)
    x['future_reference'].requires_grad_();p=b(**x);sum(v.square().mean() for v in p.values()).backward()
    assert x['future_reference'].grad.abs().sum()>0
    assert all(v.grad is None or torch.isfinite(v.grad).all() for v in b.parameters())
    changed={k:v.detach().clone() for k,v in x.items()};changed['future_reference'][:,5:]+=10
    q=b(**changed)
    for k in p:torch.testing.assert_close(p[k][:,:5],q[k][:,:5],rtol=0,atol=0)


def test_predictability_features_cannot_read_reference_after_target():
    import numpy as np
    from diagnose_reference_predictability_v6 import features
    _,_,x=models();before=features(x,5);x['future_reference'][:,5:]+=99
    np.testing.assert_array_equal(before,features(x,5))


def test_ridge_parameters_do_not_depend_on_query_values():
    import numpy as np
    from diagnose_reference_predictability_v6 import fit_ridge
    rng=np.random.default_rng(4);a=rng.normal(size=(20,4));y=rng.normal(size=(20,3));q=rng.normal(size=(3,4));logs=np.array(['a']*10+['b']*10)
    _,p=fit_ridge(a,y,q,logs);_,r=fit_ridge(a,y,q+1000,logs)
    for k in p:np.testing.assert_array_equal(p[k],r[k])
