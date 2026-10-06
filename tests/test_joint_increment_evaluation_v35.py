import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_joint_increment_v35 import errors
from rolling_joint_increment_v35 import rate_metrics


def test_wave_endpoint_and_quaternion_sign():
    true = dict(rates_b=np.zeros((1, 3, 3)), velocity_n=np.zeros((1, 3, 3)),
                quaternion=np.array([[[1., 0, 0, 0]]*3]))
    pred = {k: v.copy() for k, v in true.items()}
    pred['rates_b'][0, :, 1] = [0, 2, 99]
    pred['quaternion'] *= -1
    wave, end = errors(pred, true, 2)
    np.testing.assert_array_equal(wave['q'], [2])
    np.testing.assert_array_equal(end['q'], [4])
    np.testing.assert_array_equal(wave['attitude'], [0])


def test_local_errors_accumulate_without_becoming_self_consistent_rollout():
    # Constant +1 rad/s2 derivative error at varying dt. State is reobserved.
    previous = np.zeros((1, 3, 3))
    actual = np.zeros_like(previous)
    dt = np.array([[.01, .02, .03]])
    derivative = np.zeros((1, 3, 6)); derivative[:, :, 3:] = 1
    pred = previous + dt[:, :, None]
    result = rate_metrics(pred, derivative, previous, actual, dt)
    np.testing.assert_allclose(result['local_mse'], np.full((1, 3), (.01**2+.02**2+.03**2)/3))
    np.testing.assert_allclose(result['accumulated_mse'], np.full((1, 3), (.01**2+.03**2+.06**2)/3))
    np.testing.assert_allclose(result['accumulated_end_mse'], .06**2)
    np.testing.assert_allclose(result['derivative_bias'], 1)
    np.testing.assert_allclose(result['persistence_mse'], 0)
