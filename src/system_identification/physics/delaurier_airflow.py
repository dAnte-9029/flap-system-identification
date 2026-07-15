"""Compatibility wrapper for :mod:`system_identification.physics.delaurier.airflow`."""

from system_identification.physics.delaurier.airflow import (
    BodyFrameConvention,
    ReconstructedBodyAirflow,
    body_air_velocity_to_delaurier_section_velocity,
    compute_delaurier_axis_incidence,
    quaternion_wxyz_to_rotation_body_to_ned,
    reconstruct_body_airflow_from_ned,
)

__all__ = [
    "BodyFrameConvention",
    "ReconstructedBodyAirflow",
    "body_air_velocity_to_delaurier_section_velocity",
    "compute_delaurier_axis_incidence",
    "quaternion_wxyz_to_rotation_body_to_ned",
    "reconstruct_body_airflow_from_ned",
]
