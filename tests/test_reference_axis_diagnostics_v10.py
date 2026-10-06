import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reference_axis_diagnostics_v10 import align_diagnostic
from reference_error_structure_v9 import history_features
from system_identification.models.reference_gru_yaw_mean import ReferenceGRUYawMean,historical_full_features
from test_reference_capacity_v6 import models


def setup_model(enabled=True):
    baseline,_,x=models();torch.nn.init.normal_(baseline.head[-1].weight,std=.01)
    keys=['state_mean','state_scale','reference_mean','reference_scale','actuator_mean','actuator_scale','rate_scale','velocity_scale']
    params=dict(mean=np.zeros(94),scale=np.ones(94),coefficient=np.zeros((94,50)),target_mean=np.ones(50)*.5)
    model=ReferenceGRUYawMean({k:baseline.state_dict()[k] for k in keys},params,enabled=enabled)
    model.load_state_dict(baseline.state_dict(),strict=False)
    return baseline,model,x


def test_torch_history_features_match_frozen_numpy_recipe():
    _,_,x=setup_model()
    actual=historical_full_features(x['history_state'],x['history_reference'],x['history_actuator'],x['current_mode']).numpy()
    np.testing.assert_allclose(actual,history_features(x,'full'),rtol=1e-5,atol=1e-6)


def test_disabled_yaw_correction_exactly_recovers_original_decoder():
    old,model,x=setup_model(False)
    with torch.no_grad():a=old(**x);b=model(**x)
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)


def test_yaw_correction_is_prefix_causal_bounded_and_pose_normalized():
    _,model,x=setup_model();a=model(**x,diagnostics=True)
    changed={k:v.clone() for k,v in x.items()};changed['future_reference'][:,5:]+=20
    b=model(**changed,diagnostics=True)
    for k in a:torch.testing.assert_close(a[k][:,:5],b[k][:,:5],rtol=0,atol=0)
    correction=a['yaw_correction'];limit=float(model.rate_scale[2]);start=torch.zeros_like(correction[:,:1])
    assert correction.abs().max()<=limit+1e-6
    assert torch.diff(torch.cat([start,correction],1),dim=1).abs().max()<=2*limit*.02+1e-6
    torch.testing.assert_close(a['quaternion'].norm(dim=-1),torch.ones_like(x['dt_s']),atol=1e-6,rtol=1e-6)
    assert all(torch.isfinite(v).all() for v in a.values())


def test_alignment_oracle_recovers_known_positive_gain_and_lag_on_common_core():
    rng=np.random.default_rng(2);truth=rng.normal(size=(3,50));pred=np.roll(truth,2,axis=1)*.5+3
    d=align_diagnostic(pred,truth)
    np.testing.assert_allclose(d['best_lag_s'],.04)
    np.testing.assert_allclose(d['best_lag_gain'],2,atol=1e-12)
    np.testing.assert_allclose(d['best_lag_gain_mse'],0,atol=1e-24)


def test_alignment_has_no_spurious_amplitude_benefit_on_constant_prediction():
    truth=np.arange(50)[None,:];pred=np.ones((1,50))
    d=align_diagnostic(pred,truth)
    np.testing.assert_allclose(d['core_centered_mse'],d['best_lag_gain_mse'])
    np.testing.assert_allclose(d['best_lag_s'],0)
