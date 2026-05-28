import numpy as np
import pandas as pd

from scripts.diagnose_rotational_replay_oracle import (
    apply_moment_reference_transform,
    evaluate_attitude_kinematic_closure,
    evaluate_inertia_scale_sensitivity,
    evaluate_moment_label_closure,
    evaluate_moment_lag_sweep,
    evaluate_omega_replay,
    evaluate_reference_point_sensitivity,
    evaluate_smoothing_sensitivity,
    evaluate_spike_robustness,
    fit_diagonal_inertia_from_logs,
    infer_alpha_from_moment,
    integrate_attitude_from_logged_rates,
    integrate_omega_from_moment,
    recompute_moment_from_alpha,
    run_rotational_diagnostics,
    summarize_metric_table,
)


def _base_frame(time_s, omega, alpha, inertia=None, log_id="synthetic"):
    inertia = np.eye(3) if inertia is None else np.asarray(inertia, dtype=float)
    moment = recompute_moment_from_alpha(omega, alpha, inertia)
    frame = pd.DataFrame(
        {
            "time_s": time_s,
            "log_id": log_id,
            "segment_id": 0,
            "vehicle_local_position.x": 0.0,
            "vehicle_local_position.y": 0.0,
            "vehicle_local_position.z": 0.0,
            "vehicle_local_position.vx": 0.0,
            "vehicle_local_position.vy": 0.0,
            "vehicle_local_position.vz": 0.0,
            "vehicle_attitude.q[0]": 1.0,
            "vehicle_attitude.q[1]": 0.0,
            "vehicle_attitude.q[2]": 0.0,
            "vehicle_attitude.q[3]": 0.0,
            "vehicle_angular_velocity.xyz[0]": omega[:, 0],
            "vehicle_angular_velocity.xyz[1]": omega[:, 1],
            "vehicle_angular_velocity.xyz[2]": omega[:, 2],
            "vehicle_angular_velocity.xyz_derivative[0]": alpha[:, 0],
            "vehicle_angular_velocity.xyz_derivative[1]": alpha[:, 1],
            "vehicle_angular_velocity.xyz_derivative[2]": alpha[:, 2],
            "fx_b": 0.0,
            "fy_b": 0.0,
            "fz_b": 0.0,
            "mx_b": moment[:, 0],
            "my_b": moment[:, 1],
            "mz_b": moment[:, 2],
        }
    )
    return frame


def test_constant_yaw_rate_integrates_expected_quaternion():
    window = pd.DataFrame(
        {
            "time_s": [0.0, 0.5],
            "vehicle_attitude.q[0]": [1.0, np.cos(0.25)],
            "vehicle_attitude.q[1]": [0.0, 0.0],
            "vehicle_attitude.q[2]": [0.0, 0.0],
            "vehicle_attitude.q[3]": [0.0, np.sin(0.25)],
            "vehicle_angular_velocity.xyz[0]": [0.0, 0.0],
            "vehicle_angular_velocity.xyz[1]": [0.0, 0.0],
            "vehicle_angular_velocity.xyz[2]": [1.0, 1.0],
        }
    )

    result = integrate_attitude_from_logged_rates(window)

    np.testing.assert_allclose(result[-1], [np.cos(0.25), 0.0, 0.0, np.sin(0.25)], atol=1e-12)


def test_recompute_moment_from_alpha_includes_gyro_term():
    inertia = np.diag([2.0, 3.0, 4.0])
    omega = np.array([[0.2, -0.3, 0.4]])
    alpha = np.array([[1.0, -2.0, 3.0]])

    moment = recompute_moment_from_alpha(omega, alpha, inertia)

    expected = alpha @ inertia.T + np.cross(omega, omega @ inertia.T)
    np.testing.assert_allclose(moment, expected)


def test_infer_alpha_from_moment_inverts_rigid_body_expression():
    inertia = np.array([[2.0, 0.1, 0.0], [0.1, 3.0, 0.2], [0.0, 0.2, 4.0]])
    omega = np.array([[0.2, -0.3, 0.4], [0.5, 0.1, -0.2]])
    alpha = np.array([[1.0, -2.0, 3.0], [-0.5, 0.25, 0.75]])
    moment = recompute_moment_from_alpha(omega, alpha, inertia)

    inferred = infer_alpha_from_moment(omega, moment, inertia)

    np.testing.assert_allclose(inferred, alpha, atol=1e-12)


def test_reference_transform_modes():
    moment = np.array([[1.0, 2.0, 3.0]])
    force = np.array([[4.0, 5.0, 6.0]])
    r_b = np.array([0.1, -0.2, 0.3])
    arm = np.cross(r_b, force)

    np.testing.assert_allclose(apply_moment_reference_transform(moment, force, r_b, "none"), moment)
    np.testing.assert_allclose(apply_moment_reference_transform(moment, force, r_b, "minus_r_cross_f"), moment - arm)
    np.testing.assert_allclose(apply_moment_reference_transform(moment, force, r_b, "plus_r_cross_f"), moment + arm)


def test_summarize_metric_table_groups_diagnostic_and_horizon():
    metrics = pd.DataFrame(
        {
            "diagnostic": ["a", "a", "a", "b"],
            "horizon_s": [0.1, 0.1, 0.1, 0.1],
            "error": [1.0, 2.0, 10.0, 4.0],
        }
    )

    summary = summarize_metric_table(metrics, ["diagnostic", "horizon_s"])

    row = summary[summary["diagnostic"].eq("a")].iloc[0]
    assert row["n_windows"] == 3
    assert row["error_median"] == 2.0
    assert np.isclose(row["error_p90"], 8.4)


def test_attitude_kinematic_closure_is_zero_for_synthetic_constant_rate():
    time_s = np.arange(0.0, 0.51, 0.1)
    omega = np.tile([0.0, 0.0, 0.5], (len(time_s), 1))
    alpha = np.zeros_like(omega)
    frame = _base_frame(time_s, omega, alpha)
    frame["vehicle_attitude.q[0]"] = np.cos(0.5 * 0.5 * time_s)
    frame["vehicle_attitude.q[3]"] = np.sin(0.5 * 0.5 * time_s)

    metrics = evaluate_attitude_kinematic_closure(frame, horizons_s=[0.2, 0.4], stride_s=0.2, split="test")

    nominal = metrics[metrics["variant"].eq("right_multiply_rate_plus")]
    assert set(nominal["horizon_s"]) == {0.2, 0.4}
    assert nominal["attitude_error_deg"].max() < 1e-10


def test_attitude_kinematic_closure_handles_quaternion_sign_flip():
    time_s = np.arange(0.0, 0.31, 0.1)
    omega = np.tile([0.0, 0.0, 0.5], (len(time_s), 1))
    frame = _base_frame(time_s, omega, np.zeros_like(omega))
    frame["vehicle_attitude.q[0]"] = np.cos(0.5 * 0.5 * time_s)
    frame["vehicle_attitude.q[3]"] = np.sin(0.5 * 0.5 * time_s)
    frame.loc[frame.index[-1], ["vehicle_attitude.q[0]", "vehicle_attitude.q[3]"]] *= -1.0

    metrics = evaluate_attitude_kinematic_closure(frame, horizons_s=[0.3], stride_s=0.3, split="test")

    nominal = metrics[metrics["variant"].eq("right_multiply_rate_plus")]
    assert nominal["attitude_error_deg"].max() < 1e-10


def test_moment_label_closure_zero_for_matching_labels():
    inertia = np.diag([2.0, 3.0, 4.0])
    omega = np.array([[0.1, 0.2, 0.3], [0.2, -0.1, 0.4], [0.0, 0.1, -0.2]])
    alpha = np.array([[1.0, 0.0, -1.0], [0.5, 0.25, -0.5], [0.0, 0.0, 0.25]])
    frame = _base_frame(np.array([0.0, 0.1, 0.2]), omega, alpha, inertia)

    closure = evaluate_moment_label_closure(frame, inertia)

    assert closure["rmse"].max() < 1e-12
    assert closure["alpha_rmse"].max() < 1e-12


def test_omega_replay_constant_acceleration_matches_logged_rate():
    inertia = np.diag([2.0, 3.0, 4.0])
    time_s = np.arange(0.0, 0.51, 0.1)
    alpha = np.tile([0.4, -0.2, 0.1], (len(time_s), 1))
    omega = time_s[:, None] * alpha
    frame = _base_frame(time_s, omega, alpha, inertia)

    replay = evaluate_omega_replay(frame, inertia, horizons_s=[0.2, 0.4], stride_s=0.2)

    assert replay[replay["variant"].eq("logged_gyro_euler")]["body_rate_error_rad_s"].max() < 1e-12


def test_lag_sweep_identifies_shifted_moment_labels():
    inertia = np.eye(3)
    time_s = np.round(np.arange(0.0, 2.01, 0.02), 10)
    alpha_x = np.sin(2.0 * np.pi * time_s)
    omega_x = np.concatenate([[0.0], np.cumsum(alpha_x[:-1]) * 0.02])
    omega = np.column_stack([omega_x, np.zeros_like(time_s), np.zeros_like(time_s)])
    alpha = np.column_stack([alpha_x, np.zeros_like(time_s), np.zeros_like(time_s)])
    frame = _base_frame(time_s, omega, alpha, inertia)
    imposed_lag = 0.04
    frame["mx_b"] = np.interp(time_s - imposed_lag, time_s, alpha_x)

    sweep = evaluate_moment_lag_sweep(
        frame,
        inertia,
        lags_s=[-0.04, -0.02, 0.0, 0.02, 0.04],
        horizons_s=[0.2],
        stride_s=0.2,
    )

    best = sweep.sort_values("body_rate_error_rad_s_median").iloc[0]
    assert np.isclose(best["lag_s"], imposed_lag)


def test_smoothing_sensitivity_reduces_single_moment_spike_error():
    inertia = np.eye(3)
    time_s = np.round(np.arange(0.0, 1.01, 0.02), 10)
    omega = np.zeros((len(time_s), 3))
    alpha = np.zeros_like(omega)
    frame = _base_frame(time_s, omega, alpha, inertia)
    frame.loc[len(frame) // 2, "mx_b"] = 20.0

    summary = evaluate_smoothing_sensitivity(frame, inertia, horizons_s=[1.0], stride_s=1.0)

    raw = summary[summary["variant"].eq("raw_moment")]["body_rate_error_rad_s_median"].iloc[0]
    smoothed = summary[summary["variant"].eq("moment_savgol_0p16")]["body_rate_error_rad_s_median"].iloc[0]
    assert smoothed < raw


def test_fitted_diagonal_inertia_recovers_known_values():
    inertia = np.diag([2.0, 3.0, 4.0])
    omega = np.array([[0.2, -0.3, 0.4], [0.1, 0.5, -0.2], [-0.4, 0.2, 0.3], [0.6, -0.1, 0.2]])
    alpha = np.array([[1.0, -0.5, 0.25], [0.2, 0.4, -0.3], [-0.6, 0.1, 0.7], [0.3, -0.8, 0.2]])
    frame = _base_frame(np.arange(len(omega), dtype=float) * 0.1, omega, alpha, inertia)

    fitted = fit_diagonal_inertia_from_logs(frame)

    np.testing.assert_allclose(np.diag(fitted), np.diag(inertia), rtol=1e-6, atol=1e-6)


def test_reference_point_sensitivity_reports_candidate_transforms():
    inertia = np.eye(3)
    time_s = np.arange(0.0, 0.51, 0.1)
    omega = np.zeros((len(time_s), 3))
    alpha = np.zeros_like(omega)
    frame = _base_frame(time_s, omega, alpha, inertia)
    frame["fy_b"] = 2.0
    frame["mz_b"] = 0.2

    summary = evaluate_reference_point_sensitivity(frame, inertia, np.array([0.1, 0.0, 0.0]), [0.2], 0.2)

    assert {"none", "minus_cg_cross_force", "plus_cg_cross_force"}.issubset(set(summary["variant"]))


def test_spike_robustness_winsorization_changes_spike_metric():
    inertia = np.eye(3)
    time_s = np.round(np.arange(0.0, 1.01, 0.02), 10)
    frame = _base_frame(time_s, np.zeros((len(time_s), 3)), np.zeros((len(time_s), 3)), inertia)
    frame.loc[len(frame) // 2, "mx_b"] = 30.0

    summary = evaluate_spike_robustness(frame, inertia, horizons_s=[1.0], stride_s=1.0)

    raw = summary[summary["variant"].eq("raw")]["body_rate_error_rad_s_median"].iloc[0]
    clipped = summary[summary["variant"].eq("winsorize_moment_99p0")]["body_rate_error_rad_s_median"].iloc[0]
    assert clipped <= raw


def test_run_rotational_diagnostics_writes_expected_artifacts(tmp_path):
    split_root = tmp_path / "split"
    split_root.mkdir()
    output_root = tmp_path / "out"
    metadata_path = tmp_path / "metadata.yaml"
    inertia = np.diag([1.0, 1.2, 1.4])
    time_s = np.round(np.arange(0.0, 0.61, 0.1), 10)
    omega = np.tile([0.0, 0.0, 0.2], (len(time_s), 1))
    frame = _base_frame(time_s, omega, np.zeros_like(omega), inertia)
    frame["vehicle_attitude.q[0]"] = np.cos(0.5 * 0.2 * time_s)
    frame["vehicle_attitude.q[3]"] = np.sin(0.5 * 0.2 * time_s)
    frame.to_parquet(split_root / "test_samples.parquet")
    metadata_path.write_text(
        """
mass_properties:
  mass_kg: {value: 1.0}
  inertia_b_kg_m2: {value: [[1.0, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 1.4]]}
  cg_b_m: {value: [0.0, 0.0, 0.0]}
label_definition:
  gravity_m_s2: 9.81
""",
        encoding="utf-8",
    )

    run_rotational_diagnostics(
        split_root=split_root,
        metadata_path=metadata_path,
        output_root=output_root,
        split="test",
        horizons_s=[0.2],
        stride_s=0.2,
        lags_s=[0.0],
        overwrite=False,
    )

    for name in [
        "attitude_kinematic_closure.csv",
        "moment_label_closure.csv",
        "omega_replay_summary.csv",
        "lag_sweep_summary.csv",
        "smoothing_sensitivity_summary.csv",
        "inertia_sensitivity_summary.csv",
        "reference_point_sensitivity_summary.csv",
        "spike_robustness_summary.csv",
        "diagnostic_decision.json",
        "README.md",
    ]:
        assert (output_root / name).exists()
