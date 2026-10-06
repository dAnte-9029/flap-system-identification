import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from residual_diagnostic_v44 import rate_parts,shapley,kinematics
from actuator_diagnostics_v31 import integrate_rates


def test_bias_cancellation_does_not_remove_window_bias_energy():
    e=np.array([[[2.],[4.]],[[-4.],[-2.]]])
    p=rate_parts(e)
    np.testing.assert_array_equal(p['bias_squared'],[[9.],[9.]])
    np.testing.assert_array_equal(p['centered_mse'],[[1.],[1.]])
    assert p['bias'].mean()==0 and p['mse'].mean()==10


def test_shapley_splits_interactions_and_has_signed_efficiency():
    values=np.zeros((1,8,1))
    for m in range(8):
        p,q,r=[bool(m&(1<<j)) for j in range(3)]
        values[0,m,0]=2*p-3*q+4*r+6*p*q+9*p*q*r
    s=shapley(values)
    np.testing.assert_allclose(s[0,:,0],[8,3,7])
    np.testing.assert_allclose(s.sum(1),values[:,7]-values[:,0])


def test_native_kinematics_and_unmodified_axis_zero_attribution():
    iq=np.array([[1,0,0,0]],dtype=np.float32);ir=np.zeros((1,3),dtype=np.float32)
    dt=np.full((1,50),.02,dtype=np.float32);a=np.zeros((1,50,3),dtype=np.float32);b=a.copy();b[:,:,0]=.2
    qa=integrate_rates(iq,ir,a,dt);qb=integrate_rates(iq,ir,b,dt)
    k=kinematics(iq,ir,dt,dict(rates_b=a,quaternion=qa),dict(rates_b=b,quaternion=qb),qa,a)
    np.testing.assert_allclose(k['shapley_mse'][:,1:],0,atol=1e-14)
    assert k['shapley_mse'][0,0,0]>0
