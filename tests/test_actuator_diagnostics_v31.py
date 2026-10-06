import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from actuator_diagnostics_v31 import modal_commands, features, integrate_rates, angle_squared, lag_errors
from controller_support_v26 import support


def test_modal_commands_and_query_cannot_change_support_calibration():
    a = np.array([[[.6, -.2, .4, .1], [.7, 0., .2, -.1]]])
    m = modal_commands(a)
    np.testing.assert_allclose(m, [[[.6,.1,.3,.1],[.7,.1,.1,-.1]]])
    assert features(m).shape == (1,16)
    fit = np.array([[0.,1.],[0.,1.],[4.,3.],[4.,3.]])
    logs = np.array(['a','a','b','b'])
    one = support(fit, logs, np.array([[0.,1.]]))
    two = support(fit, logs, np.array([[1e8,1e8]]))
    for key in ['mean','scale','threshold','fit_distance']:
        np.testing.assert_array_equal(one[key],two[key])
    assert (one['fit_distance']>0).all()  # same-flight identical neighbors excluded


def test_midpoint_integration_and_quaternion_sign_invariance():
    q0 = np.array([[1.,0.,0.,0.]], dtype=np.float32)
    rates = np.zeros((1,50,3),dtype=np.float32); rates[:,:,2] = .4
    q = integrate_rates(q0, rates[:,0], rates, np.full((1,50),.02,dtype=np.float32))
    np.testing.assert_allclose(q[0,-1], [np.cos(.2),0.,0.,np.sin(.2)],atol=1e-6)
    np.testing.assert_array_equal(angle_squared(q,q),angle_squared(-q,q))
    # A one-step ramp uses mean(initial,current), not only the endpoint rate.
    ramp = integrate_rates(q0, np.zeros((1,3)), rates[:,:1], np.array([[.02]]))
    np.testing.assert_allclose(ramp[0,0,3],np.sin(.002),atol=1e-7)


def test_lag_sign_and_fixed_truth_support():
    rng = np.random.default_rng(3)
    truth = rng.normal(size=(2,50)); pred = np.zeros_like(truth)
    pred[:,3:] = truth[:,:-3]
    scores = lag_errors(pred,truth)
    assert min(scores,key=lambda k:scores[k].mean()) == 3
    np.testing.assert_allclose(scores[3],0.,atol=1e-25)
    changed = truth.copy(); changed[:,:5] += 1e6; changed[:,45:] -= 1e6
    for k,v in lag_errors(pred,changed).items(): np.testing.assert_array_equal(v,scores[k])
