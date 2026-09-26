import pytest
from system_identification.integration.control_error_envelope import EmpiricalErrorEnvelope


def test_two_errors_counted_for_action_comparison():
    e=EmpiricalErrorEnvelope([dict(steps=5,axis='q',absolute_error_bound=.5)])
    m=e.action_margin(.3,.1,0,steps=5,axis='q')
    assert m.nominal_improvement==pytest.approx(.2)
    assert m.two_prediction_error_allowance==1
    assert m.residual_margin==pytest.approx(-.8)
    assert not m.exceeds_empirical_allowance and not m.control_validated


def test_large_margin_is_still_not_causal_validation():
    e=EmpiricalErrorEnvelope([dict(steps=5,axis='q',absolute_error_bound=.1)])
    m=e.action_margin(1.,0.,0.,steps=5,axis='q')
    assert m.exceeds_empirical_allowance and not m.control_validated


def test_no_unassessed_horizon_or_nonfinite_values():
    e=EmpiricalErrorEnvelope([dict(steps=5,axis='q',absolute_error_bound=.1)])
    with pytest.raises(ValueError):e.bound(6,'q')
    with pytest.raises(ValueError):e.action_margin(1.,float('nan'),0.,steps=5,axis='q')
    with pytest.raises(ValueError):EmpiricalErrorEnvelope([dict(steps=5,axis='q',absolute_error_bound=-1)])
