import numpy as np
import pandas as pd

from scripts.evaluate_short_horizon_replay import (
    attitude_error_deg,
    evaluate_oracle_replay,
    integrate_oracle_teacher_forced_window,
    normalize_quaternion,
    quaternion_to_rotation_body_to_ned,
    summarize_replay_metrics,
    write_oracle_replay_artifacts,
)


def test_identity_quaternion_rotates_body_force_to_ned_identity():
    rotation = quaternion_to_rotation_body_to_ned(np.array([1.0, 0.0, 0.0, 0.0]))

    np.testing.assert_allclose(rotation @ np.array([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])


def test_constant_body_force_with_identity_attitude_produces_expected_velocity_drift():
    window = pd.DataFrame(
        {
            "time_s": [0.0, 0.1, 0.2],
            "vehicle_local_position.x": [0.0, 0.0, 0.0],
            "vehicle_local_position.y": [0.0, 0.0, 0.0],
            "vehicle_local_position.z": [0.0, 0.0, 0.0],
            "vehicle_local_position.vx": [1.0, 0.0, 0.0],
            "vehicle_local_position.vy": [2.0, 0.0, 0.0],
            "vehicle_local_position.vz": [3.0, 0.0, 0.0],
            "vehicle_attitude.q[0]": [1.0, 1.0, 1.0],
            "vehicle_attitude.q[1]": [0.0, 0.0, 0.0],
            "vehicle_attitude.q[2]": [0.0, 0.0, 0.0],
            "vehicle_attitude.q[3]": [0.0, 0.0, 0.0],
            "vehicle_angular_velocity.xyz[0]": [0.0, 0.0, 0.0],
            "vehicle_angular_velocity.xyz[1]": [0.0, 0.0, 0.0],
            "vehicle_angular_velocity.xyz[2]": [0.0, 0.0, 0.0],
            "fx_b": [2.0, 2.0, 2.0],
            "fy_b": [4.0, 4.0, 4.0],
            "fz_b": [-9.0, -9.0, -9.0],
            "mx_b": [0.0, 0.0, 0.0],
            "my_b": [0.0, 0.0, 0.0],
            "mz_b": [0.0, 0.0, 0.0],
        }
    )

    result = integrate_oracle_teacher_forced_window(
        window,
        mass_kg=2.0,
        inertia_b=np.eye(3),
        gravity_m_s2=4.5,
    )

    np.testing.assert_allclose(result["velocity_n"][-1], [1.2, 2.4, 3.0], atol=1e-12)


def test_zero_moment_and_zero_initial_rate_keep_body_rate_zero():
    window = pd.DataFrame(
        {
            "time_s": [0.0, 0.1, 0.2],
            "vehicle_local_position.x": [0.0, 0.0, 0.0],
            "vehicle_local_position.y": [0.0, 0.0, 0.0],
            "vehicle_local_position.z": [0.0, 0.0, 0.0],
            "vehicle_local_position.vx": [0.0, 0.0, 0.0],
            "vehicle_local_position.vy": [0.0, 0.0, 0.0],
            "vehicle_local_position.vz": [0.0, 0.0, 0.0],
            "vehicle_attitude.q[0]": [1.0, 1.0, 1.0],
            "vehicle_attitude.q[1]": [0.0, 0.0, 0.0],
            "vehicle_attitude.q[2]": [0.0, 0.0, 0.0],
            "vehicle_attitude.q[3]": [0.0, 0.0, 0.0],
            "vehicle_angular_velocity.xyz[0]": [0.0, 0.0, 0.0],
            "vehicle_angular_velocity.xyz[1]": [0.0, 0.0, 0.0],
            "vehicle_angular_velocity.xyz[2]": [0.0, 0.0, 0.0],
            "fx_b": [0.0, 0.0, 0.0],
            "fy_b": [0.0, 0.0, 0.0],
            "fz_b": [0.0, 0.0, 0.0],
            "mx_b": [0.0, 0.0, 0.0],
            "my_b": [0.0, 0.0, 0.0],
            "mz_b": [0.0, 0.0, 0.0],
        }
    )

    result = integrate_oracle_teacher_forced_window(
        window,
        mass_kg=1.0,
        inertia_b=np.eye(3),
        gravity_m_s2=9.81,
    )

    np.testing.assert_allclose(result["omega_b"][-1], [0.0, 0.0, 0.0], atol=1e-12)


def test_quaternion_sign_flip_has_zero_attitude_error():
    q = normalize_quaternion(np.array([0.5, -0.5, 0.5, -0.5]))

    assert attitude_error_deg(q, -q) == 0.0


def test_oracle_evaluation_writes_synthetic_multi_window_artifacts(tmp_path):
    time_s = np.round(np.arange(0.0, 1.0 + 0.1, 0.1), 10)
    acceleration_n = np.array([0.5, -0.25, 0.0])
    velocity_0 = np.array([1.0, 2.0, -0.5])
    position_0 = np.array([10.0, -3.0, 2.0])
    velocity = velocity_0 + time_s[:, None] * acceleration_n
    position = position_0 + time_s[:, None] * velocity_0 + 0.5 * (time_s**2)[:, None] * acceleration_n

    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "log_id": "synthetic_log",
            "segment_id": 0,
            "vehicle_local_position.x": position[:, 0],
            "vehicle_local_position.y": position[:, 1],
            "vehicle_local_position.z": position[:, 2],
            "vehicle_local_position.vx": velocity[:, 0],
            "vehicle_local_position.vy": velocity[:, 1],
            "vehicle_local_position.vz": velocity[:, 2],
            "vehicle_attitude.q[0]": 1.0,
            "vehicle_attitude.q[1]": 0.0,
            "vehicle_attitude.q[2]": 0.0,
            "vehicle_attitude.q[3]": 0.0,
            "vehicle_angular_velocity.xyz[0]": 0.0,
            "vehicle_angular_velocity.xyz[1]": 0.0,
            "vehicle_angular_velocity.xyz[2]": 0.0,
            "fx_b": acceleration_n[0],
            "fy_b": acceleration_n[1],
            "fz_b": -9.81,
            "mx_b": 0.0,
            "my_b": 0.0,
            "mz_b": 0.0,
        }
    )

    metrics = evaluate_oracle_replay(
        frame,
        metadata={"mass_kg": 1.0, "inertia_b": np.eye(3), "gravity_m_s2": 9.81},
        horizons_s=[0.2, 0.4],
        stride_s=0.2,
        mode="oracle_teacher_forced",
        split="test",
    )
    horizon_summary, log_summary = summarize_replay_metrics(metrics)
    write_oracle_replay_artifacts(
        tmp_path,
        metrics,
        horizon_summary,
        log_summary,
        config={"split": "test", "modes": ["oracle_teacher_forced"]},
        overwrite=True,
    )

    assert len(metrics) == 9
    assert set(horizon_summary["horizon_s"]) == {0.2, 0.4}
    assert horizon_summary["velocity_error_m_s_median"].max() < 1e-12
    assert (tmp_path / "replay_window_metrics.csv").exists()
    assert (tmp_path / "horizon_summary.csv").exists()
    assert (tmp_path / "log_summary.csv").exists()
