"""Frozen raw physics baselines."""

from system_identification.physics.baselines.wing_only import (
    AIRFLOW_MODES,
    ATTITUDE_AIRFLOW_REQUIRED_COLUMNS,
    ISAACLAB_SOURCE_COMMIT,
    WingOnlyBaselineConfig,
    baseline_config_from_aircraft_metadata,
    evaluate_wing_only_delaurier_segment,
    required_columns_for_airflow_mode,
)

__all__ = [
    "AIRFLOW_MODES",
    "ATTITUDE_AIRFLOW_REQUIRED_COLUMNS",
    "ISAACLAB_SOURCE_COMMIT",
    "WingOnlyBaselineConfig",
    "baseline_config_from_aircraft_metadata",
    "evaluate_wing_only_delaurier_segment",
    "required_columns_for_airflow_mode",
]
