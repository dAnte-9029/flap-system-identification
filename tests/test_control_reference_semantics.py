import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_control_reference_semantics import past_values,pitch_from_quaternion


def test_reference_requires_both_publication_and_sample_available():
    d=dict(timestamp=np.array([10,20,30]),timestamp_sample=np.array([9,25,29]),pitch=np.array([1.,2.,3.]))
    x,age,sample_age=past_values(d,['pitch'],np.array([5,22,25,30]))
    assert np.isnan(x[0,0])
    np.testing.assert_array_equal(x[1:,0],[1.,2.,3.])
    np.testing.assert_allclose(age[1:],[12e-6,5e-6,0.])
    assert (sample_age[1:]>=0).all()


def test_last_duplicate_published_reference_wins():
    d=dict(timestamp=np.array([10,10,20]),pitch=np.array([1.,2.,3.]))
    x,_,_=past_values(d,['pitch'],np.array([10]))
    assert x[0,0]==2


def test_invalid_timestamp_order_rejected():
    with pytest.raises(ValueError):past_values(dict(timestamp=np.array([20,10]),pitch=np.zeros(2)),['pitch'],np.array([30]))
    with pytest.raises(ValueError):past_values(dict(timestamp=np.array([10,20]),timestamp_sample=np.array([40,21]),pitch=np.zeros(2)),['pitch'],np.array([30]))


def test_quaternion_pitch_sign_normalization_and_invalid_input():
    theta=np.array([.3,-.2]);q=np.column_stack([np.cos(theta/2),np.zeros(2),np.sin(theta/2),np.zeros(2)])
    np.testing.assert_allclose(pitch_from_quaternion(q*2),theta)
    assert np.isnan(pitch_from_quaternion(np.zeros((1,4)))[0])
