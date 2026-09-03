from __future__ import annotations

import numpy as np
import pandas as pd

from system_identification.evaluation.trajectory import (
    aggregate_trajectory_metrics,
    evaluate_trajectory_predictions,
)
from system_identification.models.trajectory import (
    ConstantTwistPredictor,
    InitialTrajectoryState,
    PersistencePredictor,
    RidgeDynamicsPredictor,
    attitude_error_deg,
)
from system_identification.training.trajectory_baselines import (
    build_transition_arrays,
    fit_ridge_arrays,
)


def _initial_state() -> InitialTrajectoryState:
    return InitialTrajectoryState(
        position_n=np.zeros((1, 3)),
        velocity_n=np.array([[1.0, 0.0, 0.0]]),
        quaternion_nb=np.array([[1.0, 0.0, 0.0, 0.0]]),
        angular_velocity_b=np.array([[0.0, 0.0, 1.0]]),
        relative_phase_rad=np.array([0.0]),
        flap_frequency_hz=np.array([2.0]),
    )


def test_persistence_and_constant_twist_have_distinct_nonlearned_rollouts() -> None:
    controls = np.zeros((1, 2, 4))
    dt_s = np.full((1, 2), 0.1)

    persistence = PersistencePredictor().rollout(_initial_state(), controls, dt_s)
    kinematic = ConstantTwistPredictor().rollout(_initial_state(), controls, dt_s)

    np.testing.assert_allclose(persistence.position_n[0, -1], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(kinematic.position_n[0, -1], [0.2, 0.0, 0.0])
    assert attitude_error_deg(kinematic.quaternion_nb[0, -1], [1.0, 0.0, 0.0, 0.0]) > 10.0
    np.testing.assert_allclose(kinematic.flap_frequency_hz[0], [2.0, 2.0, 2.0])


def test_attitude_error_is_invariant_to_quaternion_sign() -> None:
    q = np.array([0.5, -0.5, 0.5, -0.5])
    assert attitude_error_deg(q, -q) == 0.0


def _transition_frame() -> pd.DataFrame:
    rows = []
    for segment_id, velocities in [(0, [0.0, 0.1, 0.2]), (1, [10.0, 10.2])]:
        for sample, velocity in enumerate(velocities):
            rows.append(
                {
                    "log_id": "log_a",
                    "segment_id": segment_id,
                    "sample_in_segment": sample,
                    "timestamp_us": int((segment_id * 10 + sample) * 20_000),
                    "valid_core": True,
                    "position_ned_m_x": velocity,
                    "position_ned_m_y": 0.0,
                    "position_ned_m_z": 0.0,
                    "velocity_ned_m_s_x": velocity,
                    "velocity_ned_m_s_y": 0.0,
                    "velocity_ned_m_s_z": 0.0,
                    "attitude_q_w": 1.0,
                    "attitude_q_x": 0.0,
                    "attitude_q_y": 0.0,
                    "attitude_q_z": 0.0,
                    "angular_velocity_body_rad_s_x": 0.0,
                    "angular_velocity_body_rad_s_y": 0.0,
                    "angular_velocity_body_rad_s_z": 0.0,
                    "relative_flap_phase_rad": 0.2 * sample,
                    "relative_flap_phase_sin": np.sin(0.2 * sample),
                    "relative_flap_phase_cos": np.cos(0.2 * sample),
                    "flap_frequency_hz": 2.0,
                    "control_flap_motor_normalized": 0.5,
                    "control_left_elevon_normalized": 0.1,
                    "control_right_elevon_normalized": -0.1,
                    "control_rudder_normalized": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_transition_builder_never_crosses_segments_and_control_is_explicit() -> None:
    controlled = build_transition_arrays(_transition_frame(), use_controls=True)
    autonomous = build_transition_arrays(_transition_frame(), use_controls=False)

    assert controlled.features.shape == (3, 16)
    assert autonomous.features.shape == (3, 12)
    assert controlled.targets.shape == (3, 7)
    np.testing.assert_allclose(controlled.dt_s, [0.02, 0.02, 0.02])


def test_ridge_fit_recovers_small_linear_mapping() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(500, 5))
    weights = rng.normal(size=(5, 2))
    targets = features @ weights + np.array([0.5, -0.25])

    fit = fit_ridge_arrays(features, targets, alpha=1.0e-8, use_controls=False)
    predictor = RidgeDynamicsPredictor(**fit)

    np.testing.assert_allclose(predictor.predict_features(features), targets, atol=1.0e-6)


def test_evaluation_uses_sign_safe_attitude_and_equal_log_macro() -> None:
    true = ConstantTwistPredictor().rollout(_initial_state(), np.zeros((1, 2, 4)), np.full((1, 2), 0.1))
    predicted = ConstantTwistPredictor().rollout(_initial_state(), np.zeros((1, 2, 4)), np.full((1, 2), 0.1))
    predicted.quaternion_nb *= -1.0
    metrics = evaluate_trajectory_predictions(
        predicted,
        true,
        model_name="constant_twist",
        split="validation",
        window_ids=np.array(["w0"]),
        log_ids=np.array(["log_a"]),
        segment_ids=np.array([0]),
        horizon_steps={0.2: 2},
        dt_s=np.full((1, 2), 0.1),
    )
    assert metrics.loc[0, "attitude_error_deg"] == 0.0
    assert metrics.loc[0, "observed_horizon_s"] == 0.2

    unequal = pd.DataFrame(
        {
            "model": ["m"] * 4,
            "split": ["validation"] * 4,
            "horizon_s": [1.0] * 4,
            "log_id": ["a", "a", "a", "b"],
            "position_error_m": [1.0, 1.0, 1.0, 10.0],
            "velocity_error_m_s": [1.0, 1.0, 1.0, 10.0],
            "attitude_error_deg": [1.0, 1.0, 1.0, 10.0],
            "body_rate_error_rad_s": [1.0, 1.0, 1.0, 10.0],
        }
    )
    aggregate, per_log = aggregate_trajectory_metrics(unequal)
    assert per_log["position_rmse_m"].tolist() == [1.0, 10.0]
    assert aggregate.loc[0, "position_rmse_m_per_log_macro"] == 5.5
    assert aggregate.loc[0, "position_rmse_m_pooled"] == np.sqrt(103.0 / 4.0)
