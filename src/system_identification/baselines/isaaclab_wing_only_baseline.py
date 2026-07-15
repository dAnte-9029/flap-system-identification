"""Compatibility wrapper for :mod:`system_identification.physics.baselines.wing_only`.

New code should import the frozen baseline from the canonical physics package.
"""

from system_identification.physics.baselines.wing_only import (
    AIRFLOW_MODES,
    ATTITUDE_AIRFLOW_REQUIRED_COLUMNS,
    FORCE_AXES,
    ISAACLAB_SOURCE_BRANCH,
    ISAACLAB_SOURCE_COMMIT,
    ISAACLAB_SOURCE_REPOSITORY,
    MOMENT_AXES,
    TARGETS,
    WingOnlyBaselineConfig,
    _wing_polar_transforms_frd,
    baseline_config_from_aircraft_metadata,
    evaluate_wing_only_delaurier_segment,
    required_columns_for_airflow_mode,
)

__all__ = [
    "AIRFLOW_MODES",
    "ATTITUDE_AIRFLOW_REQUIRED_COLUMNS",
    "FORCE_AXES",
    "ISAACLAB_SOURCE_BRANCH",
    "ISAACLAB_SOURCE_COMMIT",
    "ISAACLAB_SOURCE_REPOSITORY",
    "MOMENT_AXES",
    "TARGETS",
    "WingOnlyBaselineConfig",
    "baseline_config_from_aircraft_metadata",
    "evaluate_wing_only_delaurier_segment",
    "required_columns_for_airflow_mode",
]
