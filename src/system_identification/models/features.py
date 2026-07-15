"""Deterministic feature schemas and transforms used by system-identification models."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_FEATURE_COLUMNS = [
    "phase_corrected_sin",
    "phase_corrected_cos",
    "wing_stroke_angle_rad",
    "flap_frequency_hz",
    "cycle_flap_frequency_hz",
    "motor_cmd_0",
    "servo_left_elevon",
    "servo_right_elevon",
    "servo_rudder",
    "vehicle_local_position.vx",
    "vehicle_local_position.vy",
    "vehicle_local_position.vz",
    "vehicle_local_position.ax",
    "vehicle_local_position.ay",
    "vehicle_local_position.az",
    "vehicle_angular_velocity.xyz[0]",
    "vehicle_angular_velocity.xyz[1]",
    "vehicle_angular_velocity.xyz[2]",
    "vehicle_angular_velocity.xyz_derivative[0]",
    "vehicle_angular_velocity.xyz_derivative[1]",
    "vehicle_angular_velocity.xyz_derivative[2]",
    "gravity_b.x",
    "gravity_b.y",
    "gravity_b.z",
    "airspeed_validated.true_airspeed_m_s",
    "vehicle_air_data.rho",
    "wind.windspeed_north",
    "wind.windspeed_east",
]

NO_ACCEL_NO_ALPHA_EXCLUDED_COLUMNS = [
    "vehicle_local_position.ax",
    "vehicle_local_position.ay",
    "vehicle_local_position.az",
    "vehicle_angular_velocity.xyz_derivative[0]",
    "vehicle_angular_velocity.xyz_derivative[1]",
    "vehicle_angular_velocity.xyz_derivative[2]",
]

NO_ACCEL_NO_ALPHA_FEATURE_COLUMNS = [
    column for column in DEFAULT_FEATURE_COLUMNS if column not in set(NO_ACCEL_NO_ALPHA_EXCLUDED_COLUMNS)
]

PAPER_NO_ACCEL_V2_ADDED_FEATURE_COLUMNS = [
    "vehicle_local_position.heading",
    "airspeed_validated.indicated_airspeed_m_s",
    "airspeed_validated.calibrated_airspeed_m_s",
    "airspeed_validated.calibrated_ground_minus_wind_m_s",
    "airspeed_validated.true_ground_minus_wind_m_s",
    "airspeed_validated.pitch_filtered",
    "roll_rad",
    "pitch_rad",
    "velocity_b.x",
    "velocity_b.y",
    "velocity_b.z",
    "relative_air_velocity_b.x",
    "relative_air_velocity_b.y",
    "relative_air_velocity_b.z",
    "alpha_rad",
    "beta_rad",
    "dynamic_pressure_pa",
    "elevator_like",
    "aileron_like",
]

PAPER_NO_ACCEL_V2_FEATURE_COLUMNS = NO_ACCEL_NO_ALPHA_FEATURE_COLUMNS + PAPER_NO_ACCEL_V2_ADDED_FEATURE_COLUMNS

PHASE_HARMONIC_FEATURE_COLUMNS = [
    "phase_corrected_h2_sin",
    "phase_corrected_h2_cos",
    "phase_corrected_h3_sin",
    "phase_corrected_h3_cos",
]
PHASE_CONDITIONING_COLUMNS = ["phase_corrected_sin", "phase_corrected_cos"]

PAPER_NO_ACCEL_V2_RAW_PHASE_FEATURE_COLUMNS = PAPER_NO_ACCEL_V2_FEATURE_COLUMNS + ["phase_corrected_rad"]
PAPER_NO_ACCEL_V2_PHASE_HARMONIC_FEATURE_COLUMNS = PAPER_NO_ACCEL_V2_FEATURE_COLUMNS + PHASE_HARMONIC_FEATURE_COLUMNS

PAPER_PFNN_10_FEATURE_COLUMNS = [
    "phase_corrected_rad",
    "velocity_b.x",
    "velocity_b.y",
    "velocity_b.z",
    "pitch_rad",
    "roll_rad",
    "alpha_rad",
    "beta_rad",
    "cycle_flap_frequency_hz",
    "elevator_like",
    "servo_rudder",
]

DEFAULT_FEATURE_SETS: dict[str, list[str]] = {
    "full": DEFAULT_FEATURE_COLUMNS,
    "no_accel_no_alpha": NO_ACCEL_NO_ALPHA_FEATURE_COLUMNS,
    "paper_no_accel_v2": PAPER_NO_ACCEL_V2_FEATURE_COLUMNS,
    "paper_no_accel_v2_raw_phase": PAPER_NO_ACCEL_V2_RAW_PHASE_FEATURE_COLUMNS,
    "paper_no_accel_v2_phase_harmonic": PAPER_NO_ACCEL_V2_PHASE_HARMONIC_FEATURE_COLUMNS,
    "paper_pfnn_10": PAPER_PFNN_10_FEATURE_COLUMNS,
}


def resolve_feature_set_columns(feature_set_name: str | None = None) -> list[str]:
    resolved_name = feature_set_name or "full"
    if resolved_name not in DEFAULT_FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {resolved_name}")
    return list(DEFAULT_FEATURE_SETS[resolved_name])


def _with_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    derived = frame.copy()
    if "phase_corrected_rad" not in derived.columns:
        raise ValueError("Missing required column: phase_corrected_rad")
    phase = derived["phase_corrected_rad"].to_numpy(dtype=float)
    derived["phase_corrected_sin"] = np.sin(phase)
    derived["phase_corrected_cos"] = np.cos(phase)
    for harmonic in (2, 3):
        derived[f"phase_corrected_h{harmonic}_sin"] = np.sin(float(harmonic) * phase)
        derived[f"phase_corrected_h{harmonic}_cos"] = np.cos(float(harmonic) * phase)

    quaternion_columns = [
        "vehicle_attitude.q[0]",
        "vehicle_attitude.q[1]",
        "vehicle_attitude.q[2]",
        "vehicle_attitude.q[3]",
    ]
    if all(column in derived.columns for column in quaternion_columns):
        quat = derived.loc[:, quaternion_columns].to_numpy(dtype=float, copy=True)
        quat_norm = np.linalg.norm(quat, axis=1, keepdims=True)
        quat_norm = np.where(quat_norm > 1e-8, quat_norm, 1.0)
        quat = quat / quat_norm
        w = quat[:, 0]
        x = quat[:, 1]
        y = quat[:, 2]
        z = quat[:, 3]

        # Use gravity direction in body FRD to avoid q / -q sign ambiguity.
        derived["gravity_b.x"] = 2.0 * (x * z - w * y)
        derived["gravity_b.y"] = 2.0 * (y * z + w * x)
        derived["gravity_b.z"] = 1.0 - 2.0 * (x * x + y * y)

        derived["roll_rad"] = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch_argument = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        derived["pitch_rad"] = np.arcsin(pitch_argument)

        rotation_body_to_ned = np.full((len(derived), 3, 3), np.nan, dtype=float)
        rotation_body_to_ned[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
        rotation_body_to_ned[:, 0, 1] = 2.0 * (x * y - z * w)
        rotation_body_to_ned[:, 0, 2] = 2.0 * (x * z + y * w)
        rotation_body_to_ned[:, 1, 0] = 2.0 * (x * y + z * w)
        rotation_body_to_ned[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
        rotation_body_to_ned[:, 1, 2] = 2.0 * (y * z - x * w)
        rotation_body_to_ned[:, 2, 0] = 2.0 * (x * z - y * w)
        rotation_body_to_ned[:, 2, 1] = 2.0 * (y * z + x * w)
        rotation_body_to_ned[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)

        velocity_columns = [
            "vehicle_local_position.vx",
            "vehicle_local_position.vy",
            "vehicle_local_position.vz",
        ]
        if all(column in derived.columns for column in velocity_columns):
            velocity_n = derived.loc[:, velocity_columns].to_numpy(dtype=float, copy=True)
            velocity_b = np.einsum("nji,nj->ni", rotation_body_to_ned, velocity_n)
            derived["velocity_b.x"] = velocity_b[:, 0]
            derived["velocity_b.y"] = velocity_b[:, 1]
            derived["velocity_b.z"] = velocity_b[:, 2]

            wind_columns = ["wind.windspeed_north", "wind.windspeed_east"]
            if all(column in derived.columns for column in wind_columns):
                wind_n = np.zeros_like(velocity_n)
                wind_n[:, 0] = derived["wind.windspeed_north"].to_numpy(dtype=float)
                wind_n[:, 1] = derived["wind.windspeed_east"].to_numpy(dtype=float)
                relative_air_velocity_n = velocity_n - wind_n
                relative_air_velocity_b = np.einsum("nji,nj->ni", rotation_body_to_ned, relative_air_velocity_n)
                derived["relative_air_velocity_b.x"] = relative_air_velocity_b[:, 0]
                derived["relative_air_velocity_b.y"] = relative_air_velocity_b[:, 1]
                derived["relative_air_velocity_b.z"] = relative_air_velocity_b[:, 2]

                relative_air_speed = np.linalg.norm(relative_air_velocity_b, axis=1)
                valid_speed = relative_air_speed > 1e-8
                alpha_rad = np.full(len(derived), np.nan, dtype=float)
                beta_rad = np.full(len(derived), np.nan, dtype=float)
                alpha_rad[valid_speed] = np.arctan2(
                    relative_air_velocity_b[valid_speed, 2],
                    relative_air_velocity_b[valid_speed, 0],
                )
                beta_rad[valid_speed] = np.arcsin(
                    np.clip(relative_air_velocity_b[valid_speed, 1] / relative_air_speed[valid_speed], -1.0, 1.0)
                )
                derived["alpha_rad"] = alpha_rad
                derived["beta_rad"] = beta_rad

    if {"vehicle_air_data.rho", "airspeed_validated.true_airspeed_m_s"}.issubset(derived.columns):
        true_airspeed = derived["airspeed_validated.true_airspeed_m_s"].to_numpy(dtype=float)
        rho = derived["vehicle_air_data.rho"].to_numpy(dtype=float)
        derived["dynamic_pressure_pa"] = 0.5 * rho * true_airspeed * true_airspeed

    if {"servo_left_elevon", "servo_right_elevon"}.issubset(derived.columns):
        left = derived["servo_left_elevon"].to_numpy(dtype=float)
        right = derived["servo_right_elevon"].to_numpy(dtype=float)
        derived["elevator_like"] = 0.5 * (left + right)
        derived["aileron_like"] = 0.5 * (left - right)
    return derived


WINDOW_FEATURE_MODE_COLUMNS: dict[str, list[str]] = {
    "phase_actuator": [
        "phase_corrected_sin",
        "phase_corrected_cos",
        "phase_corrected_rad",
        "wing_stroke_angle_rad",
        "flap_frequency_hz",
        "cycle_flap_frequency_hz",
        "motor_cmd_0",
        "servo_left_elevon",
        "servo_right_elevon",
        "servo_rudder",
        "elevator_like",
        "aileron_like",
    ],
    "airdata": [
        "airspeed_validated.indicated_airspeed_m_s",
        "airspeed_validated.calibrated_airspeed_m_s",
        "airspeed_validated.true_airspeed_m_s",
        "airspeed_validated.calibrated_ground_minus_wind_m_s",
        "airspeed_validated.true_ground_minus_wind_m_s",
        "airspeed_validated.pitch_filtered",
        "vehicle_air_data.rho",
        "wind.windspeed_north",
        "wind.windspeed_east",
        "dynamic_pressure_pa",
    ],
}

KINEMATIC_WINDOW_EXCLUDED_COLUMNS = {
    "vehicle_local_position.vx",
    "vehicle_local_position.vy",
    "vehicle_local_position.vz",
    "vehicle_local_position.heading",
    "vehicle_angular_velocity.xyz[0]",
    "vehicle_angular_velocity.xyz[1]",
    "vehicle_angular_velocity.xyz[2]",
    "gravity_b.x",
    "gravity_b.y",
    "gravity_b.z",
    "roll_rad",
    "pitch_rad",
    "velocity_b.x",
    "velocity_b.y",
    "velocity_b.z",
    "relative_air_velocity_b.x",
    "relative_air_velocity_b.y",
    "relative_air_velocity_b.z",
    "alpha_rad",
    "beta_rad",
}

SEQUENCE_FEATURE_MODE_COLUMNS: dict[str, list[str]] = {
    "phase_actuator": WINDOW_FEATURE_MODE_COLUMNS["phase_actuator"],
    "phase_actuator_airdata": WINDOW_FEATURE_MODE_COLUMNS["phase_actuator"] + WINDOW_FEATURE_MODE_COLUMNS["airdata"],
    "phase_harmonic": WINDOW_FEATURE_MODE_COLUMNS["phase_actuator"] + PHASE_HARMONIC_FEATURE_COLUMNS,
    "phase_harmonic_actuator_airdata": (
        WINDOW_FEATURE_MODE_COLUMNS["phase_actuator"] + PHASE_HARMONIC_FEATURE_COLUMNS + WINDOW_FEATURE_MODE_COLUMNS["airdata"]
    ),
    "raw_phase_actuator_airdata": [
        "phase_corrected_rad",
        "flap_frequency_hz",
        "cycle_flap_frequency_hz",
        "motor_cmd_0",
        "servo_left_elevon",
        "servo_right_elevon",
        "servo_rudder",
        "elevator_like",
        "aileron_like",
    ]
    + WINDOW_FEATURE_MODE_COLUMNS["airdata"],
    "no_phase_actuator_airdata": [
        "flap_frequency_hz",
        "cycle_flap_frequency_hz",
        "motor_cmd_0",
        "servo_left_elevon",
        "servo_right_elevon",
        "servo_rudder",
        "elevator_like",
        "aileron_like",
    ]
    + WINDOW_FEATURE_MODE_COLUMNS["airdata"],
}

SEQUENCE_HISTORY_DANGEROUS_COLUMNS = KINEMATIC_WINDOW_EXCLUDED_COLUMNS


def resolve_window_feature_columns(feature_columns: list[str], window_feature_mode: str = "all") -> list[str]:
    mode = (window_feature_mode or "all").lower()
    available = set(feature_columns)
    if mode == "all":
        return list(feature_columns)
    if mode == "none":
        return []
    if mode == "phase_actuator":
        return [column for column in feature_columns if column in WINDOW_FEATURE_MODE_COLUMNS["phase_actuator"]]
    if mode == "phase_actuator_airdata":
        selected = set(WINDOW_FEATURE_MODE_COLUMNS["phase_actuator"]) | set(WINDOW_FEATURE_MODE_COLUMNS["airdata"])
        return [column for column in feature_columns if column in selected]
    if mode == "no_kinematics":
        return [column for column in feature_columns if column not in KINEMATIC_WINDOW_EXCLUDED_COLUMNS]
    raise ValueError(f"Unknown window_feature_mode: {window_feature_mode}")


def resolve_sequence_feature_columns(feature_columns: list[str], sequence_feature_mode: str = "phase_actuator_airdata") -> list[str]:
    mode = (sequence_feature_mode or "phase_actuator_airdata").lower()
    if mode == "all":
        return list(feature_columns)
    if mode == "none":
        return []
    if mode in SEQUENCE_FEATURE_MODE_COLUMNS:
        selected = set(SEQUENCE_FEATURE_MODE_COLUMNS[mode])
        return [column for column in feature_columns if column in selected]
    raise ValueError(f"Unknown sequence_feature_mode: {sequence_feature_mode}")


def resolve_current_feature_columns(
    base_feature_columns: list[str],
    sequence_feature_columns: list[str],
    current_feature_mode: str = "remaining_current",
) -> list[str]:
    mode = (current_feature_mode or "remaining_current").lower()
    if mode == "remaining_current":
        sequence_set = set(sequence_feature_columns)
        return [column for column in base_feature_columns if column not in sequence_set]
    if mode == "all":
        return list(base_feature_columns)
    if mode == "none":
        return []
    raise ValueError(f"Unknown current_feature_mode: {current_feature_mode}")


def resolve_phase_conditioning_indices(sequence_feature_columns: list[str]) -> tuple[int, ...]:
    missing = [column for column in PHASE_CONDITIONING_COLUMNS if column not in sequence_feature_columns]
    if missing:
        raise ValueError(f"Phase FiLM requires sequence columns: {missing}")
    return tuple(sequence_feature_columns.index(column) for column in PHASE_CONDITIONING_COLUMNS)


def apply_sequence_order_ablation(sequence_features: np.ndarray, mode: str, *, seed: int = 42) -> np.ndarray:
    normalized = (mode or "normal").lower()
    if normalized == "normal":
        return sequence_features.astype(np.float32, copy=True)
    if normalized == "reverse":
        return sequence_features[:, ::-1, :].astype(np.float32, copy=True)
    if normalized == "shuffle":
        shuffled = sequence_features.astype(np.float32, copy=True)
        rng = np.random.default_rng(seed)
        for sample_idx in range(shuffled.shape[0]):
            shuffled[sample_idx] = shuffled[sample_idx, rng.permutation(shuffled.shape[1]), :]
        return shuffled
    raise ValueError(f"Unknown sequence order ablation mode: {mode}")
