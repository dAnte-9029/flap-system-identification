import numpy as np

from scripts.prior_regression_calibration import (
    ParameterSpec,
    solve_bounded_linearized_delta,
)


def test_solve_bounded_linearized_delta_recovers_known_update():
    specs = (
        ParameterSpec("a", center=2.0, lower=0.0, upper=4.0, step=0.1, penalty_scale=2.0),
        ParameterSpec("b", center=10.0, lower=8.0, upper=12.0, step=0.2, penalty_scale=2.0),
    )
    jacobian = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 1.0], [1.0, -1.0]],
            [[-1.0, 2.0], [0.5, 0.25]],
        ],
        dtype=float,
    )
    true_delta = np.asarray([0.5, -0.25], dtype=float)
    residual = np.einsum("ntp,p->nt", jacobian, true_delta)

    result = solve_bounded_linearized_delta(
        residual=residual,
        jacobian=jacobian,
        parameter_specs=specs,
        lambda_value=0.0,
        target_scales=np.ones(2),
    )

    assert np.allclose(result.delta, true_delta, atol=1.0e-10)
    assert np.allclose(result.theta, np.asarray([2.5, 9.75]), atol=1.0e-10)
    assert result.hit_bounds == []


def test_solve_bounded_linearized_delta_respects_parameter_bounds():
    specs = (
        ParameterSpec("a", center=0.0, lower=-0.2, upper=0.2, step=0.1, penalty_scale=0.2),
        ParameterSpec("b", center=1.0, lower=0.7, upper=1.3, step=0.1, penalty_scale=0.3),
    )
    jacobian = np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=float)
    residual = np.asarray([[1.0], [-1.0]], dtype=float)

    result = solve_bounded_linearized_delta(
        residual=residual,
        jacobian=jacobian,
        parameter_specs=specs,
        lambda_value=0.0,
        target_scales=np.ones(1),
    )

    assert np.allclose(result.theta, np.asarray([0.2, 0.7]), atol=1.0e-10)
    assert set(result.hit_bounds) == {"a:upper", "b:lower"}


def test_solve_bounded_linearized_delta_uses_target_scales():
    specs = (ParameterSpec("a", center=0.0, lower=-10.0, upper=10.0, step=0.1, penalty_scale=1.0),)
    jacobian = np.asarray([[[1.0], [10.0]], [[1.0], [-10.0]]], dtype=float)
    residual = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=float)

    unscaled = solve_bounded_linearized_delta(
        residual=residual,
        jacobian=jacobian,
        parameter_specs=specs,
        lambda_value=0.0,
        target_scales=np.ones(2),
    )
    scaled = solve_bounded_linearized_delta(
        residual=residual,
        jacobian=jacobian,
        parameter_specs=specs,
        lambda_value=0.0,
        target_scales=np.asarray([1.0, 1000.0]),
    )

    assert unscaled.delta[0] < scaled.delta[0]
    assert np.isclose(scaled.delta[0], 1.0, atol=1.0e-3)
