"""Exact error decomposition and three-axis kinematic attribution, diagnostic only."""
import math
import numpy as np
from actuator_diagnostics_v31 import integrate_rates
from audit_attitude_kinematics_v42 import errors as angle_errors


def rate_parts(error):
    e=np.asarray(error,dtype=float)
    bias=e.mean(1)
    centered=((e-bias[:,None])**2).mean(1)
    mse=(e**2).mean(1)
    np.testing.assert_allclose(mse,bias**2+centered,rtol=1e-12,atol=1e-12)
    return dict(mse=mse,bias=bias,bias_squared=bias**2,centered_mse=centered)


def shapley(values):
    """values[B,8,M], bitj selects candidate ratej instead of baseline ratej.

    Contributions sum to candidate minus baseline squared error. This is a
    symmetric allocation of kinematic interactions, not causal plant effects.
    """
    v=np.asarray(values,dtype=float)
    assert v.shape[1]==8
    result=np.zeros((v.shape[0],3,v.shape[2]))
    for j in range(3):
        for mask in range(8):
            if mask&(1<<j):continue
            k=mask.bit_count();weight=math.factorial(k)*math.factorial(2-k)/6
            result[:,j]+=weight*(v[:,mask|(1<<j)]-v[:,mask])
    np.testing.assert_allclose(result.sum(1),v[:,7]-v[:,0],rtol=1e-10,atol=1e-12)
    return result


def kinematics(initial_q,initial_r,dt,baseline,candidate,truth_q,truth_r):
    values=[]
    for mask in range(8):
        rates=baseline['rates_b'].copy()
        for j in range(3):
            if mask&(1<<j):rates[:,:,j]=candidate['rates_b'][:,:,j]
        q=integrate_rates(initial_q,initial_r,rates,dt)
        if mask in [0,7]:
            np.testing.assert_array_equal(q,(baseline if mask==0 else candidate)['quaternion'])
        values.append(angle_errors(q,truth_q).mean(1))
    values=np.stack(values,1)
    debiased=[]
    for pred in [baseline,candidate]:
        rates=pred['rates_b'];bias=(rates-truth_r).mean(1,keepdims=True)
        q=integrate_rates(initial_q,initial_r,rates-bias,dt)
        debiased.append(angle_errors(q,truth_q).mean(1))
    true_q=integrate_rates(initial_q,initial_r,truth_r,dt)
    return dict(factorial_mse=values,shapley_mse=shapley(values),debiased_mse=np.stack(debiased,1),
                true_rate_mse=angle_errors(true_q,truth_q).mean(1))
