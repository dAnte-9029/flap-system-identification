from __future__ import annotations

import importlib

import numpy as np
import pandas as pd


PHYSICS_MODULE_CASES = (
    (
        "system_identification.physics.delaurier_airflow",
        "system_identification.physics.delaurier.airflow",
        (
            "BodyFrameConvention",
            "ReconstructedBodyAirflow",
            "body_air_velocity_to_delaurier_section_velocity",
            "compute_delaurier_axis_incidence",
            "quaternion_wxyz_to_rotation_body_to_ned",
            "reconstruct_body_airflow_from_ned",
        ),
    ),
    (
        "system_identification.physics.delaurier_dynamic_twist",
        "system_identification.physics.delaurier.dynamic_twist",
        (
            "DeLaurierTwistKinematics",
            "compute_delaurier_dynamic_twist",
            "map_canonical_phase_to_delaurier",
        ),
    ),
    (
        "system_identification.physics.delaurier_strip_wrench",
        "system_identification.physics.delaurier.strip_wrench",
        (
            "DeLaurierParams",
            "DeLaurierStripLoads",
            "DeLaurierStripWrench",
            "WingGeometry",
            "compute_delaurier_strip_loads",
            "integrate_delaurier_strip_wrench",
            "load_wing_geometry_csv",
            "transform_wrench",
            "translate_wrench_moment",
        ),
    ),
    (
        "system_identification.baselines.isaaclab_wing_only_baseline",
        "system_identification.physics.baselines.wing_only",
        (
            "AIRFLOW_MODES",
            "ATTITUDE_AIRFLOW_REQUIRED_COLUMNS",
            "ISAACLAB_SOURCE_COMMIT",
            "WingOnlyBaselineConfig",
            "baseline_config_from_aircraft_metadata",
            "evaluate_wing_only_delaurier_segment",
            "required_columns_for_airflow_mode",
        ),
    ),
)


def test_legacy_and_canonical_physics_modules_export_the_same_objects():
    for legacy_name, canonical_name, symbols in PHYSICS_MODULE_CASES:
        legacy = importlib.import_module(legacy_name)
        canonical = importlib.import_module(canonical_name)
        for symbol in symbols:
            assert hasattr(legacy, symbol)
            assert hasattr(canonical, symbol)
            assert getattr(legacy, symbol) is getattr(canonical, symbol)


def test_pipeline_label_helper_delegates_to_canonical_implementation():
    pipeline = importlib.import_module("system_identification.pipeline")
    canonical = importlib.import_module("system_identification.labels.effective_wrench")

    assert pipeline._compute_effective_wrench_labels is canonical._compute_effective_wrench_labels
    assert canonical.compute_effective_wrench_labels is canonical._compute_effective_wrench_labels


def test_baseline_wrapper_preserves_tested_private_transform_helper():
    legacy = importlib.import_module("system_identification.baselines.isaaclab_wing_only_baseline")
    canonical = importlib.import_module("system_identification.physics.baselines.wing_only")

    assert legacy._wing_polar_transforms_frd is canonical._wing_polar_transforms_frd


def test_legacy_and_canonical_label_calls_match_on_effective_wrench_fixture():
    pipeline = importlib.import_module("system_identification.pipeline")
    canonical = importlib.import_module("system_identification.labels.effective_wrench")
    metadata = {
        "mass_properties": {
            "mass_kg": {"value": 1.0},
            "inertia_b_kg_m2": {"value": np.eye(3).tolist()},
        }
    }
    samples = pd.DataFrame(
        {
            "vehicle_local_position.ax": [1.0],
            "vehicle_local_position.ay": [0.0],
            "vehicle_local_position.az": [9.81],
            "vehicle_attitude.q[0]": [1.0],
            "vehicle_attitude.q[1]": [0.0],
            "vehicle_attitude.q[2]": [0.0],
            "vehicle_attitude.q[3]": [0.0],
            "vehicle_angular_velocity.xyz[0]": [0.0],
            "vehicle_angular_velocity.xyz[1]": [0.0],
            "vehicle_angular_velocity.xyz[2]": [0.0],
            "vehicle_angular_velocity.xyz_derivative[0]": [0.1],
            "vehicle_angular_velocity.xyz_derivative[1]": [0.2],
            "vehicle_angular_velocity.xyz_derivative[2]": [0.3],
        }
    )

    legacy_result = pipeline._compute_effective_wrench_labels(samples, metadata)
    canonical_result = canonical.compute_effective_wrench_labels(samples, metadata)
    for legacy_values, canonical_values in zip(legacy_result, canonical_result):
        np.testing.assert_array_equal(legacy_values, canonical_values)


def test_legacy_and_canonical_physics_calls_match_on_small_inputs():
    old_airflow = importlib.import_module("system_identification.physics.delaurier_airflow")
    new_airflow = importlib.import_module("system_identification.physics.delaurier.airflow")
    velocity = np.array([[2.0, 0.0, 0.5]])
    np.testing.assert_array_equal(
        old_airflow.compute_delaurier_axis_incidence(air_velocity_body=velocity, body_frame="FRD"),
        new_airflow.compute_delaurier_axis_incidence(air_velocity_body=velocity, body_frame="FRD"),
    )

    old_twist = importlib.import_module("system_identification.physics.delaurier_dynamic_twist")
    new_twist = importlib.import_module("system_identification.physics.delaurier.dynamic_twist")
    phase = np.array([0.0, np.pi])
    np.testing.assert_array_equal(
        old_twist.map_canonical_phase_to_delaurier(phase),
        new_twist.map_canonical_phase_to_delaurier(phase),
    )

    old_wrench = importlib.import_module("system_identification.physics.delaurier_strip_wrench")
    new_wrench = importlib.import_module("system_identification.physics.delaurier.strip_wrench")
    kwargs = {
        "force": np.array([1.0, 0.0, 0.0]),
        "moment_about_origin": np.array([0.0, 0.0, 0.0]),
        "origin_position": np.array([0.0, 1.0, 0.0]),
        "reference_position": np.zeros(3),
    }
    np.testing.assert_array_equal(
        old_wrench.translate_wrench_moment(**kwargs),
        new_wrench.translate_wrench_moment(**kwargs),
    )

    old_baseline = importlib.import_module("system_identification.baselines.isaaclab_wing_only_baseline")
    new_baseline = importlib.import_module("system_identification.physics.baselines.wing_only")
    assert old_baseline.required_columns_for_airflow_mode("legacy_scalar_true_airspeed") == (
        new_baseline.required_columns_for_airflow_mode("legacy_scalar_true_airspeed")
    )
