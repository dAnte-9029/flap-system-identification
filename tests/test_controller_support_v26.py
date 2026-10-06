import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from controller_support_v26 import support,quantile,q_errors


def test_calibration_excludes_every_same_flight_neighbor():
    a=np.array([[0.],[0.],[10.],[10.]])
    r=support(a,np.array(['a','a','b','b']),np.array([[0.],[30.]]))
    np.testing.assert_allclose(r['fit_distance'],2.)
    np.testing.assert_allclose(r['query_distance'],[0.,4.])
    assert r['threshold']==pytest.approx(2.)


def test_query_does_not_affect_threshold_or_fit_scaling_and_flights_are_balanced():
    a=np.array([[0.],[10.],[12.]])
    logs=np.array(['a','b','b'])
    r=support(a,logs,np.array([[1.]]));s=support(a,logs,np.array([[1e10]]))
    for key in ['mean','scale','threshold','fit_distance']:np.testing.assert_array_equal(r[key],s[key])
    assert quantile(np.array([0.,10.,12.]),logs,.5)==0.


def test_error_decomposition_and_half_prefix_identity():
    rng=np.random.default_rng(4);truth=rng.normal(size=(7,50));pred=truth+2+rng.normal(size=(7,50))
    d=q_errors(pred,truth)
    np.testing.assert_allclose(d['mse'],d['bias_mse']+d['centered_mse'])
    np.testing.assert_allclose(d['mse'],(d['early_mse']+d['late_mse'])/2)
