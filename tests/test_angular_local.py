import numpy as np
import torch
from test_trajectory_oracle import setup_case
from system_identification.evaluation.angular_local import capture_rollout, angular_metrics


def test_passive_capture_preserves_prediction_and_reconstructs_rate_update():
    model,x=setup_case()
    with torch.no_grad():model.derivative_head[-1].bias.copy_(torch.tensor([0.,0.,0.,7.,-8.,2.,0.]))
    expected=model(**x);actual,raw=capture_rollout(model,x)
    for a,b in zip(expected,actual):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert not model.derivative_head._forward_hooks
    alpha=(model.derivative_mean+model.derivative_std*raw.clamp(-6,6))[...,3:6]
    torch.testing.assert_close(actual.angular_velocity_b[:,1:]-actual.angular_velocity_b[:,:-1],alpha*x['dt_s'][...,None])
    assert (raw[...,3:5].abs()>=6).all()


def test_metrics_distinguish_bias_from_rmse_and_retain_failures():
    result=angular_metrics(np.array([[3.,4.,0.],[-3.,-4.,0.]]))
    assert result['rate_vector_rmse_rad_s']==5
    assert result['x_bias_rad_s']==0
    assert result['x_rmse_rad_s']==3
    assert np.isinf(angular_metrics(np.array([[np.nan,0.,0.]]))['rate_vector_rmse_rad_s'])
