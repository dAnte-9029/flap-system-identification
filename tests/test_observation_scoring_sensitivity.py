import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
from scipy.signal import lsim
from run_observation_scoring_sensitivity import filter_foh,sampled_filter,interp,quaternion_at,TAU

def test_exact_foh_against_continuous_system():
    t=np.linspace(0,1,101);x=np.sin(9*t);x[0]=0
    _,truth,_=lsim(([1],[TAU**2,2*TAU,1]),U=x,T=t)
    np.testing.assert_allclose(filter_foh(t,x[:,None])[:,0],truth,atol=1e-12)
def test_constant_and_prefix():
    t=np.array([0,.01,.025,.06,.1]);x=np.full((5,3),2.)
    np.testing.assert_allclose(filter_foh(t,x),x,atol=1e-14)
    x=np.arange(15.).reshape(5,3);np.testing.assert_allclose(filter_foh(t,x)[:3],filter_foh(t[:3],x[:3]))
def test_insert_knots_no_effect_linear_input():
    t=np.array([0.,.1,.3]);x=np.stack([t,2*t,3*t],-1);q=np.array([0,.05,.1,.2,.3])
    np.testing.assert_allclose(sampled_filter(t,x,q),filter_foh(q,interp(t,x,q)),atol=1e-13)
def test_no_extrapolation_or_mutation():
    t=np.array([0.,1.]);x=np.ones((2,3));copy=x.copy()
    with pytest.raises(ValueError):interp(t,x,[-.1])
    filter_foh(t,x);np.testing.assert_array_equal(x,copy)
def test_quaternion_sign_and_midpoint():
    q=np.array([[1,0,0,0],[-1,0,0,0.]])
    out=quaternion_at(np.array([0.,1.]),q,np.array([.5]));assert abs(out[0,0])==1
