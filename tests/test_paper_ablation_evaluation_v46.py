import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_paper_ablation_v46 import distance, integrate_rates, multiply


def test_quaternion_geodesic_sign_scale_and_known_rotation():
    identity = np.array([[1., 0, 0, 0]])
    angle = .8
    q = np.array([[np.cos(angle/2), 0, np.sin(angle/2), 0]])
    np.testing.assert_allclose(distance(identity, q), angle, atol=1e-14)
    np.testing.assert_allclose(distance(identity, -3*q), angle, atol=1e-14)
    np.testing.assert_allclose(distance(q, -q), 0, atol=1e-14)


def test_independent_rate_integration_known_body_spin_and_nonuniform_dt():
    rates = np.zeros((1, 4, 3))
    rates[:, :, 2] = 2.
    dt = np.array([[.01, .02, .04, .03]])
    q0 = np.array([[np.cos(.3), np.sin(.3), 0, 0]])
    result = integrate_rates(q0, rates[:, 0], rates, dt)
    angles = 2*dt.cumsum(1)
    delta = np.stack([np.cos(angles/2), np.zeros_like(angles), np.zeros_like(angles), np.sin(angles/2)], -1)
    np.testing.assert_allclose(result, multiply(q0[:, None], delta), atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(result, axis=-1), 1., atol=1e-14)


def test_rate_integration_trapezoid_for_accelerating_single_axis():
    dt = np.full((1, 50), .02)
    times = dt.cumsum(1)
    rates = np.zeros((1, 50, 3))
    rates[:, :, 0] = 3*times
    result = integrate_rates(np.array([[1., 0, 0, 0]]), np.zeros((1, 3)), rates, dt)
    angle = 1.5*times**2
    expected = np.stack([np.cos(angle/2), np.sin(angle/2), np.zeros_like(angle), np.zeros_like(angle)], -1)
    np.testing.assert_allclose(result, expected, atol=1e-14)
