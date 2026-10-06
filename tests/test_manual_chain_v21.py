import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_manual_chain_v21 import rc_normalize,servo_pwm,hold


def test_rc_asymmetric_calibration_and_reverse():
    np.testing.assert_allclose(rc_normalize([1000,1210,1420,1671,1922],1000,1420,1922,1),[-1,-.5,0,.5,1])
    np.testing.assert_allclose(rc_normalize([900,2000],1000,1420,1922,-1),[1,-1])
    with pytest.raises(ValueError):rc_normalize([1000],1000,1000,1922,1)


def test_pwm_reversal_and_rounding():
    np.testing.assert_array_equal(servo_pwm([-1,0,1],800,1800,True),[1800,1300,800])
    np.testing.assert_array_equal(servo_pwm([.001],800,1800,False),[1301])


def test_hold_requires_both_publication_and_sample_available():
    d={'timestamp':np.array([10,20,30]),'timestamp_sample':np.array([8,24,29])}
    np.testing.assert_array_equal(hold(d,np.array([10,23,24,35])),[0,0,1,2])
    with pytest.raises(ValueError):hold(d,np.array([9]))
