import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reference_timing_v14 import curves,best_index,CORES,LAGS,effective_indices


def test_known_lag_gain_and_same_support():
    rng=np.random.default_rng(14);t=rng.normal(size=(4,50));p=np.roll(t,2,axis=1)*.5+4
    for core in CORES:
        d=curves(p,t,core);best=best_index(d['gain']);np.testing.assert_array_equal(LAGS[best],2)
        np.testing.assert_allclose(d['coefficient'][np.arange(4),best],2,atol=1e-12)
        np.testing.assert_allclose(d['gain'][np.arange(4),best],0,atol=1e-24)


def test_piecewise_shift_can_differ_without_changing_truth_support():
    rng=np.random.default_rng(2);t=rng.normal(size=(3,50));p=t.copy();p[:,:25]=np.roll(t,2,axis=1)[:,:25];p[:,25:]=np.roll(t,-2,axis=1)[:,25:]
    a=curves(p,t,CORES[0]);b=curves(p,t,CORES[1])
    np.testing.assert_array_equal(LAGS[best_index(a['unit'])],2);np.testing.assert_array_equal(LAGS[best_index(b['unit'])],-2)
    assert np.mean(np.minimum(a['unit'].min(1)+b['unit'].min(1),(a['unit']+b['unit']).min(1)))<1e-20


def test_constant_prediction_prefers_zero_lag():
    d=curves(np.ones((2,50)),np.tile(np.arange(50),(2,1)),CORES[0]);np.testing.assert_array_equal(LAGS[best_index(d['unit'])],0)


def test_publication_and_sample_must_both_precede_query():
    t=np.array([0,10,20]);s=np.array([0,15,30]);q=np.array([12,19,29,30])
    np.testing.assert_array_equal(effective_indices(t,s,q),[0,1,1,2])
    with pytest.raises(ValueError):effective_indices(t,s,[-1])
    with pytest.raises(ValueError):effective_indices([10,0],[10,0],[20])
