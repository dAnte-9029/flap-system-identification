import numpy as np

from scripts.prior_nonlinear_calibration import (
    NonlinearObjectiveResult,
    ParameterSpec,
    make_theta_cache_key,
    nonlinear_objective,
    solve_bounded_nonlinear_calibration,
)


def test_nonlinear_objective_uses_target_and_parameter_scales():
    specs = (
        ParameterSpec("a", center=1.0, lower=0.0, upper=4.0, step=0.1, penalty_scale=2.0),
        ParameterSpec("b", center=0.0, lower=-2.0, upper=2.0, step=0.1, penalty_scale=0.5),
    )
    labels = np.asarray([[3.0, 10.0]], dtype=float)

    def evaluate(theta):
        return np.asarray([[theta[0] + theta[1], 0.0]], dtype=float)

    result = nonlinear_objective(
        theta=np.asarray([2.0, 1.0], dtype=float),
        labels=labels,
        evaluate_prior=evaluate,
        parameter_specs=specs,
        lambda_value=4.0,
        target_scales=np.asarray([2.0, 10.0], dtype=float),
    )

    # Residuals are [(3 - 3) / 2, (10 - 0) / 10], so data loss is 1.
    assert np.isclose(result.data_loss, 1.0)
    # Normalized parameter update is [(2 - 1) / 2, (1 - 0) / 0.5].
    assert np.isclose(result.regularization_loss, 4.0 * (0.5**2 + 2.0**2))
    assert np.isclose(result.total_loss, result.data_loss + result.regularization_loss)


def test_solve_bounded_nonlinear_calibration_recovers_exact_nonlinear_parameters():
    specs = (
        ParameterSpec("a", center=1.0, lower=0.0, upper=4.0, step=0.1, penalty_scale=2.0),
        ParameterSpec("b", center=0.0, lower=-2.0, upper=2.0, step=0.1, penalty_scale=1.0),
    )
    true_theta = np.asarray([2.0, 0.5], dtype=float)

    def evaluate(theta):
        a, b = theta
        return np.asarray(
            [
                [a * a + b, a + b * b],
                [0.5 * a * a - b, a - 0.5 * b * b],
            ],
            dtype=float,
        )

    labels = evaluate(true_theta)
    result = solve_bounded_nonlinear_calibration(
        labels=labels,
        evaluate_prior=evaluate,
        parameter_specs=specs,
        lambda_value=0.0,
        target_scales=np.ones(2),
        initial_theta=np.asarray([1.2, 0.0], dtype=float),
        maxiter=300,
    )

    assert result.success
    assert np.allclose(result.theta, true_theta, atol=1.0e-4)
    assert result.hit_bounds == []
    assert result.data_loss < 1.0e-8


def test_solve_bounded_nonlinear_calibration_respects_bounds():
    specs = (
        ParameterSpec("a", center=0.0, lower=-0.5, upper=0.5, step=0.1, penalty_scale=0.5),
    )

    def evaluate(theta):
        return np.asarray([[theta[0]]], dtype=float)

    result = solve_bounded_nonlinear_calibration(
        labels=np.asarray([[2.0]], dtype=float),
        evaluate_prior=evaluate,
        parameter_specs=specs,
        lambda_value=0.0,
        target_scales=np.ones(1),
        initial_theta=np.asarray([0.0], dtype=float),
        maxiter=100,
    )

    assert np.isclose(result.theta[0], 0.5, atol=1.0e-6)
    assert result.hit_bounds == ["a:upper"]


def test_pattern_search_nonlinear_calibration_moves_multiple_parameters():
    specs = (
        ParameterSpec("a", center=0.0, lower=-4.0, upper=4.0, step=0.1, penalty_scale=2.0),
        ParameterSpec("b", center=0.0, lower=-4.0, upper=4.0, step=0.1, penalty_scale=2.0),
    )
    true_theta = np.asarray([1.5, -1.0], dtype=float)

    def evaluate(theta):
        return np.asarray([[theta[0], theta[1]]], dtype=float)

    result = solve_bounded_nonlinear_calibration(
        labels=evaluate(true_theta),
        evaluate_prior=evaluate,
        parameter_specs=specs,
        lambda_value=0.0,
        target_scales=np.ones(2),
        initial_theta=np.asarray([0.0, 0.0], dtype=float),
        optimizer="pattern_search",
        max_function_evaluations=80,
        maxiter=20,
    )

    assert result.optimizer == "pattern_search"
    assert np.allclose(result.theta, true_theta, atol=0.11)
    assert result.data_loss < 0.03


def test_make_theta_cache_key_is_stable_for_small_roundoff():
    specs = (
        ParameterSpec("a", center=0.0, lower=-1.0, upper=1.0, step=0.1, penalty_scale=1.0),
        ParameterSpec("b", center=0.0, lower=-1.0, upper=1.0, step=0.1, penalty_scale=1.0),
    )

    key_a = make_theta_cache_key(np.asarray([0.123456789, -0.2]), specs)
    key_b = make_theta_cache_key(np.asarray([0.123456781, -0.2000000001]), specs)

    assert key_a == key_b
    assert "a_0p123457" in key_a
    assert "b_m0p2" in key_a
