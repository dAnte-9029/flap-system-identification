from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from system_identification.baselines.isaaclab_wing_only_baseline import (
    WingOnlyBaselineConfig,
    _wing_polar_transforms_frd,
    baseline_config_from_aircraft_metadata,
    evaluate_wing_only_delaurier_segment,
)
from system_identification.physics.delaurier_airflow import (
    body_air_velocity_to_delaurier_section_velocity,
    compute_delaurier_axis_incidence,
    quaternion_wxyz_to_rotation_body_to_ned,
    reconstruct_body_airflow_from_ned,
)
from system_identification.physics.delaurier_dynamic_twist import (
    compute_delaurier_dynamic_twist,
    map_canonical_phase_to_delaurier,
)
from system_identification.physics.delaurier_strip_wrench import (
    DeLaurierParams,
    WingGeometry,
    compute_delaurier_strip_loads,
    integrate_delaurier_strip_wrench,
    transform_wrench,
    translate_wrench_moment,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "isaaclab_delaurier_wing_wrench_3b5d4ec.json"
ABS_TOL = 1.0e-10


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _geometry(data: dict) -> WingGeometry:
    geometry = data["geometry"]
    return WingGeometry(
        x_mid=np.asarray(geometry["x_mid_m"]),
        dx=np.asarray(geometry["dx_m"]),
        chord=np.asarray(geometry["chord_m"]),
        d_hat=np.asarray(geometry["d_hat"]),
        semi_span_m=float(geometry["R_m"]),
        area_m2=float(np.dot(geometry["chord_m"], geometry["dx_m"])),
        aspect_ratio=float(geometry["aspect_ratio"]),
    )


def _evaluate_fixture():
    data = _fixture()
    geometry = _geometry(data)
    inputs = data["inputs"]
    canonical_phase = np.asarray(inputs["canonical_phase_rad"])
    phase = map_canonical_phase_to_delaurier(canonical_phase)
    rate = np.asarray(inputs["phase_rate_rad_s"])
    acceleration = np.asarray(inputs["phase_acceleration_rad_s2"])
    amplitude = float(inputs["stroke_amplitude_rad"])
    q = amplitude * np.cos(phase)
    q_dot = -amplitude * np.sin(phase) * rate
    q_ddot = -amplitude * (np.cos(phase) * np.square(rate) + np.sin(phase) * acceleration)
    span = geometry.x_mid[None, :]
    mean_pitch = np.asarray(inputs["mean_pitch_rad"])
    twist = compute_delaurier_dynamic_twist(
        strip_span_m=geometry.x_mid,
        strip_width_m=geometry.dx,
        mean_pitch_rad=mean_pitch,
        tip_twist_amplitude_rad=math.radians(float(inputs["theta_tip_deg"])),
        phase_rad=phase,
        phase_rate_rad_s=rate,
        phase_acceleration_rad_s2=acceleration,
        enabled=True,
        semi_span_m=geometry.semi_span_m,
    )
    params = DeLaurierParams(
        alpha0_rad=0.0,
        eta_s=0.65,
        cd_cf=1.95,
        alpha_stall_min_rad=math.radians(-12.0),
        alpha_stall_max_rad=math.radians(12.0),
        xi=0.0,
        c_mac=0.0,
        nu=1.5e-5,
        cd_f=0.028,
        stall_smoothing_width_rad=0.0,
    )
    loads = compute_delaurier_strip_loads(
        -q[:, None] * span,
        -q_dot[:, None] * span,
        -q_ddot[:, None] * span,
        twist.theta,
        twist.theta_dot,
        twist.theta_ddot,
        geometry,
        np.asarray(inputs["rho_kg_m3"]),
        np.asarray(inputs["airspeed_m_s"]),
        theta_a=mean_pitch,
        theta_bar=mean_pitch,
        omega_ref_rad_s=rate,
        params=params,
        enable_separation=False,
    )
    wrench = integrate_delaurier_strip_wrench(loads)
    config = WingOnlyBaselineConfig(
        left_wing_root_b_m=tuple(inputs["left_root_frd_m"][0]),
        right_wing_root_b_m=tuple(inputs["right_root_frd_m"][0]),
        aircraft_cg_b_m=tuple(inputs["real_cg_frd_m"][0]),
    )
    left_transform, right_transform = _wing_polar_transforms_frd(q, config)
    left_force, left_root_moment = transform_wrench(
        wrench.force_wang, wrench.moment_wang_about_wing_origin, left_transform
    )
    right_force, right_root_moment = transform_wrench(
        wrench.force_wang, wrench.moment_wang_about_wing_origin, right_transform
    )
    left_moment = translate_wrench_moment(
        left_force, left_root_moment, np.asarray(config.left_wing_root_b_m), np.asarray(config.aircraft_cg_b_m)
    )
    right_moment = translate_wrench_moment(
        right_force,
        right_root_moment,
        np.asarray(config.right_wing_root_b_m),
        np.asarray(config.aircraft_cg_b_m),
    )
    return data, twist, loads, wrench, left_force, left_moment, right_force, right_moment


def test_float64_offline_implementation_matches_frozen_isaac_fixture() -> None:
    data, twist, loads, wrench, left_force, left_moment, right_force, right_moment = _evaluate_fixture()
    errors: list[float] = []

    def compare(actual, expected) -> None:
        errors.append(float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)))))

    for name, expected in data["expected_twist"].items():
        compare(getattr(twist, name), expected)
    for name, expected in data["expected_strip_components"].items():
        compare(getattr(loads, name), expected)
    compare(wrench.force_wang, data["expected_wang_wrench"]["force"])
    compare(wrench.moment_wang_about_wing_origin, data["expected_wang_wrench"]["moment_about_wing_origin"])
    expected = data["expected_body_cg_wrench"]
    compare(left_force, expected["left_force"])
    compare(left_moment, expected["left_moment"])
    compare(right_force, expected["right_force"])
    compare(right_moment, expected["right_moment"])
    compare(left_force + right_force, expected["total_force"])
    compare(left_moment + right_moment, expected["total_moment"])
    assert max(errors) <= ABS_TOL


def test_zero_tip_twist_matches_disabled_exactly() -> None:
    common = dict(
        strip_span_m=np.array([0.1, 0.3]),
        strip_width_m=np.array([0.2, 0.2]),
        mean_pitch_rad=np.array([0.12, -0.03]),
        tip_twist_amplitude_rad=0.0,
        phase_rad=np.array([0.0, 1.2]),
        phase_rate_rad_s=np.array([20.0, 30.0]),
        phase_acceleration_rad_s2=np.array([0.0, 4.0]),
        semi_span_m=0.4,
    )
    enabled = compute_delaurier_dynamic_twist(**common, enabled=True)
    disabled = compute_delaurier_dynamic_twist(**common, enabled=False)
    for name in ("theta", "theta_dot", "theta_ddot", "delta_theta", "delta_theta_dot", "delta_theta_ddot"):
        np.testing.assert_array_equal(getattr(enabled, name), getattr(disabled, name))


def test_dynamic_twist_is_linear_to_theoretical_tip_and_has_analytic_derivatives() -> None:
    tip = 0.2
    phase = np.array([0.0, math.pi / 2.0, math.pi])
    rate = np.array([3.0, 3.0, 3.0])
    acceleration = np.array([2.0, 2.0, 2.0])
    result = compute_delaurier_dynamic_twist(
        strip_span_m=np.array([0.1, 0.3]),
        strip_width_m=np.array([0.2, 0.2]),
        mean_pitch_rad=0.1,
        tip_twist_amplitude_rad=tip,
        phase_rad=phase,
        phase_rate_rad_s=rate,
        phase_acceleration_rad_s2=acceleration,
        enabled=True,
        semi_span_m=0.4,
    )
    np.testing.assert_allclose(result.span_fraction[0], [0.25, 0.75])
    assert result.span_fraction[0, -1] != 1.0
    expected_delta = -tip * result.span_fraction * np.sin(phase[:, None])
    expected_dot = -tip * result.span_fraction * np.cos(phase[:, None]) * rate[:, None]
    expected_ddot = tip * result.span_fraction * (
        np.sin(phase[:, None]) * rate[:, None] ** 2 - np.cos(phase[:, None]) * acceleration[:, None]
    )
    np.testing.assert_allclose(result.delta_theta, expected_delta, atol=1.0e-14)
    np.testing.assert_allclose(result.delta_theta_dot, expected_dot, atol=1.0e-14)
    np.testing.assert_allclose(result.delta_theta_ddot, expected_ddot, atol=1.0e-14)
    np.testing.assert_allclose(result.theta, 0.1 + expected_delta, atol=1.0e-14)


def test_airflow_frame_contract_and_vertical_signs() -> None:
    velocity_frd = np.array([[8.0, 1.0, 2.0], [8.0, -1.0, -2.0]])
    velocity_flu = velocity_frd * np.array([1.0, -1.0, -1.0])
    np.testing.assert_array_equal(
        body_air_velocity_to_delaurier_section_velocity(velocity_frd, body_frame="FRD"), velocity_frd
    )
    np.testing.assert_array_equal(
        body_air_velocity_to_delaurier_section_velocity(velocity_flu, body_frame="FLU"), velocity_frd
    )
    assert compute_delaurier_axis_incidence(
        air_velocity_body=np.array([[8.0, 0.0, 0.0]]), body_frame="FRD"
    )[0] == 0.0
    angles = compute_delaurier_axis_incidence(air_velocity_body=velocity_frd, body_frame="FRD")
    assert angles[0] > 0.0
    assert angles[1] < 0.0
    reverse = compute_delaurier_axis_incidence(
        air_velocity_body=np.array([[-2.0, 0.0, 1.0]]), body_frame="FRD"
    )
    np.testing.assert_allclose(reverse, [math.atan2(1.0, -2.0)], atol=1.0e-14)


def test_attitude_airflow_reconstruction_rotates_ground_minus_wind_to_body_frd() -> None:
    pitch = math.radians(25.0)
    quaternion = np.array([[math.cos(0.5 * pitch), 0.0, math.sin(0.5 * pitch), 0.0]])
    rotation, valid = quaternion_wxyz_to_rotation_body_to_ned(quaternion)
    expected_body_airflow = np.array([[8.0, 1.2, 2.0]])
    wind_ned = np.array([[1.0, -0.5, 0.0]])
    ground_ned = np.einsum("nij,nj->ni", rotation, expected_body_airflow) + wind_ned
    result = reconstruct_body_airflow_from_ned(
        ground_velocity_ned_m_s=ground_ned,
        wind_velocity_ned_m_s=wind_ned,
        quaternion_body_to_ned_wxyz=quaternion,
    )
    assert valid.tolist() == [True]
    np.testing.assert_allclose(result.velocity_body_frd_m_s, expected_body_airflow, atol=1.0e-14)
    np.testing.assert_allclose(result.alpha_rad, np.arctan2([2.0], [8.0]), atol=1.0e-14)
    np.testing.assert_allclose(
        result.beta_rad,
        np.arctan2([1.2], [math.sqrt(8.0**2 + 2.0**2)]),
        atol=1.0e-14,
    )
    flipped = reconstruct_body_airflow_from_ned(
        ground_velocity_ned_m_s=ground_ned,
        wind_velocity_ned_m_s=wind_ned,
        quaternion_body_to_ned_wxyz=-quaternion,
    )
    np.testing.assert_allclose(flipped.velocity_body_frd_m_s, result.velocity_body_frd_m_s, atol=1.0e-14)


def test_attitude_airflow_mode_changes_wrench_and_preserves_diagnostics() -> None:
    count = 7
    phase = math.radians(20.0)
    quaternion = np.tile([math.cos(0.5 * phase), 0.0, math.sin(0.5 * phase), 0.0], (count, 1))
    rotation, _ = quaternion_wxyz_to_rotation_body_to_ned(quaternion)
    body_airflow = np.tile([8.0, 0.5, 2.0], (count, 1))
    ground_ned = np.einsum("nij,nj->ni", rotation, body_airflow)
    frame = pd.DataFrame(
        {
            "time_s": np.arange(count) * 0.01,
            "log_id": "log",
            "segment_id": 1,
            "mechanical_phase_rad": np.linspace(0.0, 1.2, count),
            "flap_frequency_hz": 4.5,
            "vehicle_air_data.rho": 1.15,
            "airspeed_validated.true_airspeed_m_s": 8.0,
            "airspeed_validated.pitch_filtered": 0.0,
            "vehicle_local_position.vx": ground_ned[:, 0],
            "vehicle_local_position.vy": ground_ned[:, 1],
            "vehicle_local_position.vz": ground_ned[:, 2],
            "wind.windspeed_north": 0.0,
            "wind.windspeed_east": 0.0,
        }
    )
    for index in range(4):
        frame[f"vehicle_attitude.q[{index}]"] = quaternion[:, index]
    geometry_path = Path(__file__).parents[1] / "metadata" / "aircraft" / "flapper_01" / "wing_geometry_isaaclab_3b5d4ec.csv"
    legacy = evaluate_wing_only_delaurier_segment(
        frame,
        theta_tip_deg=[0.0],
        geometry_path=geometry_path,
        config=WingOnlyBaselineConfig(airflow_mode="legacy_scalar_true_airspeed"),
    )
    attitude = evaluate_wing_only_delaurier_segment(
        frame,
        theta_tip_deg=[0.0],
        geometry_path=geometry_path,
        config=WingOnlyBaselineConfig(airflow_mode="attitude_ground_wind_3d"),
    )
    np.testing.assert_allclose(attitude["airflow_body_u_m_s"], 8.0, atol=1.0e-13)
    np.testing.assert_allclose(attitude["airflow_body_v_m_s"], 0.5, atol=1.0e-13)
    np.testing.assert_allclose(attitude["airflow_body_w_m_s"], 2.0, atol=1.0e-13)
    np.testing.assert_allclose(attitude["airflow_alpha_rad"], math.atan2(2.0, 8.0), atol=1.0e-13)
    assert not np.allclose(attitude["pred_fz_b"], legacy["pred_fz_b"])


def test_right_reflection_uses_axial_vector_mapping_and_symmetric_wrench_relations() -> None:
    config = WingOnlyBaselineConfig(
        left_wing_fixed_roll_rad=0.0,
        right_wing_fixed_roll_rad=0.0,
        left_wing_root_b_m=(0.0, -0.1, 0.0),
        right_wing_root_b_m=(0.0, 0.1, 0.0),
        aircraft_cg_b_m=(0.0, 0.0, 0.0),
    )
    left_transform, right_transform = _wing_polar_transforms_frd(np.array([0.0]), config)
    force_wang = np.array([[0.0, 2.0, 3.0]])
    moment_wang = np.array([[4.0, 5.0, 6.0]])
    left_force, left_root_moment = transform_wrench(force_wang, moment_wang, left_transform)
    right_force, right_root_moment = transform_wrench(force_wang, moment_wang, right_transform)
    left_moment = translate_wrench_moment(left_force, left_root_moment, np.array([0.0, -0.1, 0.0]), 0.0)
    right_moment = translate_wrench_moment(right_force, right_root_moment, np.array([0.0, 0.1, 0.0]), 0.0)
    total_force = left_force + right_force
    total_moment = left_moment + right_moment
    np.testing.assert_allclose(total_force[:, 1], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(total_moment[:, [0, 2]], 0.0, atol=1.0e-14)
    assert total_force[0, 0] != 0.0 and total_force[0, 2] != 0.0
    assert total_moment[0, 1] != 0.0
    polar_right_moment = np.einsum("nij,nj->ni", right_transform, moment_wang)
    np.testing.assert_allclose(right_root_moment, -polar_right_moment, atol=1.0e-14)


def test_moment_translation_matches_hand_calculation() -> None:
    force = np.array([[2.0, -3.0, 5.0]])
    origin = np.array([1.0, 2.0, -1.0])
    reference = np.array([-2.0, 0.5, 4.0])
    moment = np.array([[0.2, -0.4, 0.8]])
    expected = moment + np.cross(origin - reference, force)
    np.testing.assert_allclose(
        translate_wrench_moment(force, moment, origin, reference), expected, atol=1.0e-14
    )


def test_invalid_phase_mapping_helper_has_no_hidden_sign_change() -> None:
    phase = np.array([0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0])
    mapped = map_canonical_phase_to_delaurier(phase)
    np.testing.assert_allclose(np.cos(mapped), np.sin(phase), atol=1.0e-14)


def test_real_cg_and_stroke_amplitude_are_loaded_from_aircraft_metadata() -> None:
    metadata_path = Path(__file__).parents[1] / "metadata" / "aircraft" / "flapper_01" / "aircraft_metadata.yaml"
    config = baseline_config_from_aircraft_metadata(metadata_path)
    assert config.aircraft_cg_b_m == (-0.12154, 0.00541, -0.04298)
    assert config.stroke_amplitude_rad == 0.5235987756
