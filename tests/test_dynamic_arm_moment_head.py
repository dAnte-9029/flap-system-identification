import numpy as np
import pandas as pd

from scripts.train_dynamic_arm_moment_head import (
    FORCE_TARGETS,
    MOMENT_TARGETS,
    build_arm_design_matrix,
    compute_tau_free_energy,
    cross_arm_force,
    fit_fixed_arm,
    fit_dynamic_arm_linear,
    moment_metrics,
)


def test_cross_arm_force_matches_numpy_cross() -> None:
    arm = np.array([[1.0, 2.0, 3.0], [-0.5, 0.25, 0.75]])
    force = np.array([[4.0, 5.0, 6.0], [2.0, -3.0, 1.0]])

    moment = cross_arm_force(arm, force)

    assert np.allclose(moment, np.cross(arm, force))


def test_build_arm_design_matrix_reconstructs_cross_product_coefficients() -> None:
    features = np.array([[1.0, 0.0], [1.0, 2.0], [1.0, -1.0]])
    force = np.array([[2.0, 3.0, 5.0], [-1.0, 4.0, 2.0], [3.0, -2.0, 1.0]])
    coefficients = np.array(
        [
            [0.1, -0.2, 0.3],
            [0.5, 0.25, -0.75],
        ]
    )

    design = build_arm_design_matrix(features, force)
    predicted = design @ coefficients.reshape(-1)

    assert np.allclose(predicted.reshape(-1, 3), np.cross(features @ coefficients, force))


def test_fit_fixed_arm_recovers_known_constant_arm() -> None:
    force = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 2.0, 1.0],
            [2.0, -1.0, 0.5],
            [-1.0, 1.5, 3.0],
            [4.0, 0.5, -2.0],
        ]
    )
    true_arm = np.array([0.25, -0.4, 0.15])
    moment = np.cross(np.repeat(true_arm[None, :], len(force), axis=0), force)

    model = fit_fixed_arm(force=force, moment=moment, alpha=0.0)

    assert model.model_name == "fixed_arm"
    assert np.allclose(model.predict(force=force)["r_hat"], true_arm[None, :], atol=1e-10)
    assert np.allclose(model.predict(force=force)["moment"], moment, atol=1e-10)


def test_fit_dynamic_arm_linear_recovers_known_feature_coefficients() -> None:
    frame = pd.DataFrame(
        {
            "phase_sin_1": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        }
    )
    force = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 2.0, 1.0],
            [2.0, -1.0, 0.5],
            [-1.0, 1.5, 3.0],
            [4.0, 0.5, -2.0],
            [1.5, -2.0, -1.0],
        ]
    )
    transform_columns = ["phase_sin_1"]
    # Coefficients are in standardized feature space: bias row then phase_sin_1 row.
    true_coefficients = np.array([[0.1, -0.2, 0.3], [0.5, 0.25, -0.75]])
    feature_mean = frame["phase_sin_1"].mean()
    feature_scale = frame["phase_sin_1"].std(ddof=0)
    phi = np.column_stack([np.ones(len(frame)), (frame["phase_sin_1"].to_numpy() - feature_mean) / feature_scale])
    moment = np.cross(phi @ true_coefficients, force)

    model = fit_dynamic_arm_linear(
        features=frame,
        feature_columns=transform_columns,
        force=force,
        moment=moment,
        alpha=0.0,
        force_source="true_force",
    )
    prediction = model.predict(force=force, features=frame)

    assert model.model_name == "dynamic_arm_linear"
    assert np.allclose(model.arm_coefficients, true_coefficients, atol=1e-10)
    assert np.allclose(prediction["moment"], moment, atol=1e-10)
    assert np.allclose(prediction["r_hat"], phi @ true_coefficients, atol=1e-10)


def test_compute_tau_free_energy_fraction() -> None:
    arm_moment = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    tau_free = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 0.0]])
    moment = arm_moment + tau_free

    energy = compute_tau_free_energy(moment=moment, arm_moment=arm_moment, tau_free=tau_free)

    assert energy["arm_energy"] == 25.0
    assert energy["tau_free_energy"] == 25.0
    assert energy["predicted_moment_energy"] == 50.0
    assert energy["tau_free_fraction_of_predicted"] == 0.5
    assert energy["tau_free_fraction_of_arm_plus_tau"] == 0.5


def test_moment_metrics_reports_mean_row() -> None:
    y_true = np.array([[1.0, 2.0, 3.0], [2.0, 0.0, 1.0]])
    y_pred = np.array([[1.5, 1.0, 2.0], [1.0, 1.0, 1.0]])

    metrics = moment_metrics(split="unit", model_name="demo", force_source="gt", y_true=y_true, y_pred=y_pred)

    assert set(metrics["target"]) == {*MOMENT_TARGETS, "moment_mean"}
    assert set(FORCE_TARGETS) == {"fx_b", "fy_b", "fz_b"}
    assert metrics.loc[metrics["target"].eq("moment_mean"), "rmse"].iloc[0] > 0.0
    assert metrics.loc[metrics["target"].eq("moment_mean"), "n"].iloc[0] == len(y_true)
