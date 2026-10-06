import numpy as np
import pytest
import torch
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from system_identification.models.reference_rate_readout import causal_average_rates
from audit_control_reference_semantics import past_values


def test_filter_uses_past_samples_and_known_step_response():
    history=torch.zeros(1,26,3);future=torch.ones(1,20,3)
    actual=causal_average_rates(history,future)
    torch.testing.assert_close(actual[0,:10,0],torch.arange(1,11)/10)
    future[:,10:]=100
    torch.testing.assert_close(actual[:,:10],causal_average_rates(history,future)[:,:10],rtol=0,atol=0)
    assert causal_average_rates(history,future).shape==future.shape


def test_filter_requires_explicit_history_and_has_finite_gradients():
    future=torch.randn(2,50,3,requires_grad=True);history=torch.randn(2,26,3)
    causal_average_rates(history,future).square().mean().backward()
    assert torch.isfinite(future.grad).all() and future.grad.abs().sum()>0
    with pytest.raises(ValueError):causal_average_rates(history[:,:2],future)
    torch.testing.assert_close(causal_average_rates(history,future,width=1),future)


def test_resampling_waits_for_both_publication_and_sample_times():
    data={'timestamp':np.array([100,200,300]),'timestamp_sample':np.array([90,220,290]),'value':np.array([1,2,3])}
    v,pa,sa=past_values(data,['value'],np.array([205,220,300]))
    np.testing.assert_array_equal(v[:,0],[1,2,3]);assert (pa>=0).all() and (sa>=0).all()
