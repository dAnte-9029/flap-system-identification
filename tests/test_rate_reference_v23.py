import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rate_reference_v23 import FIELDS,reference_values,reference_stats


def test_rate_reference_holds_causally_and_keeps_same_message_thrust():
    d={k:np.array([1.,2.,3.]) for k in FIELDS};d['reset_integral']=np.zeros(3)
    d['timestamp']=np.array([10,20,30]);d['timestamp_sample']=np.array([9,24,29])
    v,ix=reference_values(d,np.array([10,23,24,30]))
    np.testing.assert_array_equal(ix,[0,0,1,2]);np.testing.assert_array_equal(v[:,3],[1,1,2,3])
    with pytest.raises(ValueError):reference_values(d,np.array([8]))


def test_invalid_reset_rejected_and_binary_normalization_fixed():
    d={k:np.array([0.,.5]) for k in FIELDS};d['timestamp']=np.array([10,20])
    with pytest.raises(ValueError):reference_values(d,np.array([10,20]))
    v=np.array([[0,0,0,.5,0],[1,2,3,.6,0]],float);mean,scale=reference_stats(v)
    assert mean[4]==0 and scale[4]==1
    np.testing.assert_allclose(mean[:4],[.5,1,1.5,.55])
