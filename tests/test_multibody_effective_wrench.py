from __future__ import annotations

import numpy as np
import pandas as pd


def _model():
    from system_identification.labels.multibody_effective_wrench import (
        measured_flapper_multibody_model,
    )

    return measured_flapper_multibody_model()


def test_stationary_multibody_has_zero_non_gravity_wrench():
    from system_identification.labels.multibody_effective_wrench import (
        compute_multibody_effective_wrench,
    )

    model = _model()
    result = compute_multibody_effective_wrench(
        acceleration_origin_frd_m_s2=np.array([[0.0, 0.0, 9.81]]),
        gravity_frd_m_s2=np.array([[0.0, 0.0, 9.81]]),
        body_angular_velocity_frd_rad_s=np.zeros((1, 3)),
        body_angular_acceleration_frd_rad_s2=np.zeros((1, 3)),
        flap_position_rad=np.zeros(1),
        flap_velocity_rad_s=np.zeros(1),
        flap_acceleration_rad_s2=np.zeros(1),
        model=model,
    )

    np.testing.assert_allclose(result.force_frd_n, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.moment_about_neutral_com_frd_nm, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(result.dynamic_com_frd_m[0], model.neutral_com_from_imu_frd_m, atol=1.0e-12)


def test_symmetric_wing_force_correction_matches_amini_vertical_term():
    from system_identification.labels.multibody_effective_wrench import (
        compute_multibody_effective_wrench,
    )

    model = _model()
    q = np.array([0.23])
    qd = np.array([8.7])
    qdd = np.array([-91.0])
    result = compute_multibody_effective_wrench(
        acceleration_origin_frd_m_s2=np.zeros((1, 3)),
        gravity_frd_m_s2=np.zeros((1, 3)),
        body_angular_velocity_frd_rad_s=np.zeros((1, 3)),
        body_angular_acceleration_frd_rad_s2=np.zeros((1, 3)),
        flap_position_rad=q,
        flap_velocity_rad_s=qd,
        flap_acceleration_rad_s2=qdd,
        model=model,
    )

    gamma = model.neutral_dihedral_rad + q
    common = qdd * np.cos(gamma) - qd**2 * np.sin(gamma)
    rho = 2.0 * model.wing_mass_each_kg / model.total_mass_kg
    amini_acceleration_down = rho * model.wing_spanwise_com_m * common
    expected_delta_fz = -model.total_mass_kg * amini_acceleration_down

    np.testing.assert_allclose(result.force_frd_n[:, 2], expected_delta_fz, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(result.force_frd_n[:, 1], 0.0, atol=1.0e-12)

    wing_x = model.wing_com_from_pivot_right_neutral_frame_frd_m[0]
    pivot_from_neutral_com_x = -model.neutral_com_from_imu_frd_m[0]
    expected_pitch_moment = (
        2.0
        * model.wing_spanwise_com_m
        * (wing_x + pivot_from_neutral_com_x)
        * model.wing_mass_each_kg
        * common
    )
    np.testing.assert_allclose(
        result.moment_about_neutral_com_frd_nm[:, 1],
        expected_pitch_moment,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_symmetric_wing_motion_cancels_lateral_com_motion():
    from system_identification.labels.multibody_effective_wrench import (
        compute_multibody_effective_wrench,
    )

    model = _model()
    q = np.linspace(-0.5, 0.5, 11)
    result = compute_multibody_effective_wrench(
        acceleration_origin_frd_m_s2=np.zeros((len(q), 3)),
        gravity_frd_m_s2=np.zeros((len(q), 3)),
        body_angular_velocity_frd_rad_s=np.zeros((len(q), 3)),
        body_angular_acceleration_frd_rad_s2=np.zeros((len(q), 3)),
        flap_position_rad=q,
        flap_velocity_rad_s=np.zeros(len(q)),
        flap_acceleration_rad_s2=np.zeros(len(q)),
        model=model,
    )

    np.testing.assert_allclose(result.dynamic_com_frd_m[:, 1], model.neutral_com_from_imu_frd_m[1], atol=1.0e-12)


def test_reference_shift_from_imu_to_neutral_com_is_explicit():
    from system_identification.labels.multibody_effective_wrench import (
        compute_multibody_effective_wrench,
    )

    model = _model()
    result = compute_multibody_effective_wrench(
        acceleration_origin_frd_m_s2=np.array([[2.0, -1.0, 0.5]]),
        gravity_frd_m_s2=np.zeros((1, 3)),
        body_angular_velocity_frd_rad_s=np.zeros((1, 3)),
        body_angular_acceleration_frd_rad_s2=np.zeros((1, 3)),
        flap_position_rad=np.zeros(1),
        flap_velocity_rad_s=np.zeros(1),
        flap_acceleration_rad_s2=np.zeros(1),
        model=model,
    )

    expected = result.moment_about_imu_frd_nm - np.cross(
        model.neutral_com_from_imu_frd_m,
        result.force_frd_n,
    )
    np.testing.assert_allclose(result.moment_about_neutral_com_frd_nm, expected, atol=1.0e-12)


def test_phase_kinematics_unwraps_within_each_log_without_crossing_boundaries():
    from system_identification.labels.multibody_effective_wrench import (
        compute_phase_driven_flap_kinematics,
    )

    time = np.arange(101, dtype=float) * 0.01
    phase_a = np.mod(2.0 * np.pi * 4.0 * time + 5.8, 2.0 * np.pi)
    phase_b = np.mod(2.0 * np.pi * 2.0 * time + 0.2, 2.0 * np.pi)
    samples = pd.DataFrame(
        {
            "log_id": ["a"] * len(time) + ["b"] * len(time),
            "segment_id": [0] * len(time) + [0] * len(time),
            "time_s": np.concatenate([time, time]),
            "mechanical_phase_rad": np.concatenate([phase_a, phase_b]),
            "flap_frequency_hz": [4.0] * len(time) + [2.0] * len(time),
        }
    )
    result = compute_phase_driven_flap_kinematics(
        samples,
        amplitude_rad=0.5,
        window_s=0.07,
        polyorder=3,
    )

    np.testing.assert_allclose(result.phase_rate_rad_s[: len(time)], 2.0 * np.pi * 4.0, atol=1.0e-10)
    np.testing.assert_allclose(result.phase_rate_rad_s[len(time) :], 2.0 * np.pi * 2.0, atol=1.0e-10)
    np.testing.assert_allclose(result.phase_acceleration_rad_s2, 0.0, atol=1.0e-10)
    np.testing.assert_allclose(result.position_phase_rate_rad_s, result.phase_rate_rad_s, atol=1.0e-10)


def test_sample_reconstruction_preserves_index_and_masks_invalid_phase():
    from system_identification.labels.multibody_effective_wrench import (
        reconstruct_multibody_labels_from_samples,
    )

    time = np.arange(21, dtype=float) * 0.01
    phase = np.mod(2.0 * np.pi * 4.0 * time, 2.0 * np.pi)
    samples = pd.DataFrame(
        {
            "log_id": ["a"] * len(time),
            "segment_id": [0] * len(time),
            "time_s": time,
            "mechanical_phase_rad": phase,
            "flap_frequency_hz": 4.0,
            "phase_valid": True,
            "label_reconstruction_valid": True,
            "vehicle_local_position.ax_smooth": 0.0,
            "vehicle_local_position.ay_smooth": 0.0,
            "vehicle_local_position.az_smooth": 9.81,
            "vehicle_angular_velocity.xyz[0]": 0.0,
            "vehicle_angular_velocity.xyz[1]": 0.0,
            "vehicle_angular_velocity.xyz[2]": 0.0,
            "vehicle_angular_velocity.xyz_derivative_smooth[0]": 0.0,
            "vehicle_angular_velocity.xyz_derivative_smooth[1]": 0.0,
            "vehicle_angular_velocity.xyz_derivative_smooth[2]": 0.0,
            "vehicle_attitude.q[0]": 1.0,
            "vehicle_attitude.q[1]": 0.0,
            "vehicle_attitude.q[2]": 0.0,
            "vehicle_attitude.q[3]": 0.0,
            "fx_b": 0.0,
            "fy_b": 0.0,
            "fz_b": 0.0,
            "mx_b": 0.0,
            "my_b": 0.0,
            "mz_b": 0.0,
        },
        index=np.arange(100, 121),
    )
    samples.loc[110, "phase_valid"] = False
    output = reconstruct_multibody_labels_from_samples(
        samples,
        model=_model(),
        amplitude_rad=np.deg2rad(30.0),
        phase_derivative_window_s=0.07,
        phase_derivative_polyorder=3,
    )

    assert output.index.equals(samples.index)
    assert not output.loc[110, "multibody_label_valid"]
    assert np.isnan(output.loc[110, "fz_b_multibody"])
    assert output.drop(index=110)["multibody_label_valid"].all()
    np.testing.assert_allclose(
        output["position_phase_rate_minus_logged_frequency_rad_s"].dropna(),
        0.0,
        atol=1.0e-10,
    )
