import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_reference_axis_diagnostics_v10 import setup_model
from reference_yaw_prefix_v11 import prefix_features,fit_prefix
from system_identification.models.reference_gru_yaw_reference import ReferenceGRUYawReference,reference_prefix_features
from reference_groupcv_v8 import STAT_KEYS


def setup_prefix():
    baseline,old,x=setup_model()
    rng=np.random.default_rng(91)
    params=dict(mean=np.zeros((50,104)),scale=np.ones((50,104)),coefficient=rng.normal(0,.02,(50,104)),target_mean=np.zeros(50))
    model=ReferenceGRUYawReference({k:old.state_dict()[k] for k in STAT_KEYS},params)
    model.load_state_dict({k:v for k,v in old.state_dict().items() if not k.startswith('yaw_')},strict=False)
    return baseline,old,model,x


def test_numpy_torch_features_and_suffix_causality():
    _,_,model,x=setup_prefix()
    actual=reference_prefix_features(x['history_state'],x['history_reference'],x['history_actuator'],x['current_mode'],x['future_reference'])
    np.testing.assert_allclose(actual.numpy(),prefix_features(x),rtol=1e-5,atol=1e-6)
    a=model(**x,diagnostics=True)
    changed={k:v.clone() for k,v in x.items()};changed['future_reference'][:,5:]+=20
    b=model(**changed,diagnostics=True)
    for k in a:torch.testing.assert_close(a[k][:,:5],b[k][:,:5],rtol=0,atol=0)
    short={k:v.clone() for k,v in x.items()};short['future_reference']=short['future_reference'][:,:5];short['dt_s']=short['dt_s'][:,:5]
    c=model(**short,diagnostics=True)
    for k in c:torch.testing.assert_close(a[k][:,:5],c[k],rtol=0,atol=0)


def test_zero_prefix_coefficients_reproduce_history_only_correction():
    _,old,model,x=setup_prefix()
    with torch.no_grad():
        model.yaw_coefficient.zero_();model.yaw_coefficient[:,:94].copy_(old.yaw_coefficient.T)
        model.yaw_target_mean.copy_(old.yaw_target_mean)
    a=old(**x,diagnostics=True);b=model(**x,diagnostics=True)
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)


def test_disabled_bounded_and_normalized():
    baseline,_,model,x=setup_prefix();model.enabled=False
    a=baseline(**x);b=model(**x)
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    model.enabled=True;a=model(**x,diagnostics=True);bias=a['yaw_correction'];limit=model.rate_scale[2]
    assert bias.abs().max()<=limit+1e-6
    assert torch.diff(torch.cat([torch.zeros_like(bias[:,:1]),bias],1),dim=1).abs().max()<=2*limit*.02+1e-6
    torch.testing.assert_close(a['quaternion'].norm(dim=-1),torch.ones_like(x['dt_s']),rtol=1e-6,atol=1e-6)


def test_reference_prefix_reaches_mean_forecast():
    _,_,model,x=setup_prefix()
    with torch.no_grad():model.yaw_coefficient.zero_();model.yaw_coefficient[:,94]=1
    a=model(**x,diagnostics=True)
    changed={k:v.clone() for k,v in x.items()};changed['future_reference'][:,:,0]+=.1
    b=model(**changed,diagnostics=True)
    assert not torch.equal(a['yaw_correction'],b['yaw_correction'])


def test_query_targets_do_not_influence_fit_and_overlap_rejected():
    rng=np.random.default_rng(3);f=rng.normal(size=(8,50,104));target=rng.normal(size=(8,50))
    logs=np.array(['a']*3+['b']*3+['c']*2);fit=np.arange(6);query=np.arange(6,8)
    a,p=fit_prefix(f,target,fit,query,logs);target[query]+=10000
    b,q=fit_prefix(f,target,fit,query,logs)
    np.testing.assert_array_equal(a,b)
    for k in p:np.testing.assert_array_equal(p[k],q[k])
    with pytest.raises(ValueError):fit_prefix(f,target,fit,np.array([0]),logs)
