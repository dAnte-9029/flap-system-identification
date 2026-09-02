from __future__ import annotations

import json
import math
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pyulog import ULog


AUDIT_VERSION = "august_ulg_audit_v1"

KEY_TOPIC_FIELDS: dict[str, tuple[str, ...]] = {
    "vehicle_local_position": ("x", "y", "z", "vx", "vy", "vz", "xy_valid", "z_valid", "v_xy_valid", "v_z_valid"),
    "vehicle_attitude": ("q[0]", "q[1]", "q[2]", "q[3]", "quat_reset_counter"),
    "vehicle_angular_velocity": ("xyz[0]", "xyz[1]", "xyz[2]"),
    "vehicle_acceleration": ("xyz[0]", "xyz[1]", "xyz[2]"),
    "actuator_motors": ("control[0]",),
    "actuator_servos": ("control[0]", "control[1]", "control[2]"),
    "encoder_count": ("total_count", "position_raw"),
    "flap_frequency": ("frequency_hz",),
    "rpm": ("rpm_raw", "rpm_estimate"),
    "wing_phase": ("phase_rad", "phase_unwrapped_rad", "phase_valid", "flap_frequency_hz"),
    "hall_event": (),
    "airspeed_validated": ("indicated_airspeed_m_s", "calibrated_airspeed_m_s", "true_airspeed_m_s", "airspeed_source"),
    "vehicle_air_data": ("rho",),
    "wind": ("windspeed_north", "windspeed_east"),
    "sensor_gps": ("fix_type", "eph", "epv"),
    "sensor_gnss_relative": ("relative_position_valid", "heading_valid"),
    "vehicle_global_position": ("lat", "lon", "alt", "vel_n", "vel_e", "vel_d"),
    "vehicle_status": ("arming_state", "nav_state", "failsafe"),
    "vehicle_land_detected": ("landed",),
    "manual_control_setpoint": ("roll", "pitch", "yaw", "throttle"),
    "position_setpoint_triplet": ("current.type", "current.valid", "current.vx", "current.vy", "current.vz"),
    "control_allocator_status": ("torque_setpoint_achieved", "thrust_setpoint_achieved", "actuator_saturation[0]"),
}

REQUIRED_TOPICS = {
    "vehicle_local_position",
    "vehicle_attitude",
    "vehicle_angular_velocity",
    "actuator_motors",
    "actuator_servos",
    "flap_frequency",
    "vehicle_status",
    "vehicle_land_detected",
}

SELECTED_PARAMETERS = (
    "SYS_AUTOSTART",
    "SDLOG_PROFILE",
    "FLAP_RATIO",
    "FW_USE_AIRSPD",
    "FW_AIRSPD_MIN",
    "FW_AIRSPD_TRIM",
    "FW_AIRSPD_MAX",
    "FW_AIRSPD_STALL",
    "FW_PSP_OFF",
    "ASPD_PRIMARY",
    "ASPD_SCALE_1",
    "CA_SV_CS0_TRQ_P",
    "CA_SV_CS0_TRQ_R",
    "CA_SV_CS1_TRQ_P",
    "CA_SV_CS1_TRQ_R",
    "PWM_MAIN_REV",
    "PWM_MAIN_MIN1",
    "PWM_MAIN_MAX1",
    "PWM_MAIN_MIN2",
    "PWM_MAIN_MAX2",
    "PWM_MAIN_MIN3",
    "PWM_MAIN_MAX3",
    "PWM_MAIN_MIN5",
    "PWM_MAIN_MAX5",
)

STRUCTURAL_PARAMETERS = tuple(name for name in SELECTED_PARAMETERS if name != "ASPD_SCALE_1")


def parse_log_datetime(filename: str) -> datetime | None:
    patterns = (
        re.compile(
            r"(?P<year>20\d{2})-(?P<month>\d{1,2})-(?P<day>\d{1,2})-"
            r"(?P<hour>\d{1,2})-(?P<minute>\d{1,2})-(?P<second>\d{1,2})"
        ),
        re.compile(
            r"(?P<year>20\d{2})-(?P<month>\d{2})-(?P<day>\d{2})_"
            r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})"
        ),
    )
    for pattern in patterns:
        match = pattern.search(filename)
        if match:
            return datetime(**{key: int(value) for key, value in match.groupdict().items()})
    return None


def duration_from_mask(timestamps_us: np.ndarray, mask: np.ndarray, *, max_gap_s: float = 0.05) -> float:
    timestamps = np.asarray(timestamps_us, dtype=np.int64)
    valid = np.asarray(mask, dtype=bool)
    if len(timestamps) < 2 or len(timestamps) != len(valid):
        return 0.0
    dt_s = np.diff(timestamps).astype(float) * 1e-6
    intervals = valid[:-1] & valid[1:] & (dt_s > 0.0) & (dt_s <= max_gap_s)
    return float(np.sum(dt_s[intervals]))


def durations_by_label(
    timestamps_us: np.ndarray,
    valid_mask: np.ndarray,
    labels: np.ndarray,
    *,
    max_gap_s: float,
) -> dict[str, float]:
    timestamps = np.asarray(timestamps_us, dtype=np.int64)
    valid = np.asarray(valid_mask, dtype=bool)
    categories = np.asarray(labels, dtype=object)
    unique_labels = sorted({str(value) for value in categories})
    output = {label: 0.0 for label in unique_labels}
    if len(timestamps) < 2 or len(timestamps) != len(valid) or len(timestamps) != len(categories):
        return output
    dt_s = np.diff(timestamps).astype(float) * 1e-6
    valid_intervals = valid[:-1] & valid[1:] & (dt_s > 0.0) & (dt_s <= max_gap_s)
    for label in unique_labels:
        output[label] = float(np.sum(dt_s[valid_intervals & (categories[:-1] == label)]))
    return output


def summarize_timestamps(timestamps_us: np.ndarray) -> dict[str, Any]:
    timestamps = np.asarray(timestamps_us, dtype=np.int64)
    summary: dict[str, Any] = {
        "sample_count": int(len(timestamps)),
        "duration_s": 0.0,
        "median_rate_hz": None,
        "p99_gap_s": None,
        "max_gap_s": None,
        "large_gap_count": 0,
        "duplicate_count": 0,
        "backward_count": 0,
    }
    if len(timestamps) < 2:
        return summary
    diffs_s = np.diff(timestamps).astype(float) * 1e-6
    positive = diffs_s[diffs_s > 0.0]
    summary["duration_s"] = float(max(0, int(timestamps[-1]) - int(timestamps[0])) * 1e-6)
    summary["duplicate_count"] = int(np.sum(diffs_s == 0.0))
    summary["backward_count"] = int(np.sum(diffs_s < 0.0))
    if len(positive):
        median_gap = float(np.median(positive))
        large_gap_threshold = max(0.05, 3.0 * median_gap)
        summary["median_rate_hz"] = float(round(1.0 / median_gap, 6))
        summary["p99_gap_s"] = float(np.quantile(positive, 0.99))
        summary["max_gap_s"] = float(np.max(positive))
        summary["large_gap_count"] = int(np.sum(positive > large_gap_threshold))
    return summary


def classify_maneuvers(
    vertical_velocity_ned_m_s: np.ndarray,
    roll_rad: np.ndarray,
    yaw_rate_rad_s: np.ndarray,
    ground_speed_m_s: np.ndarray,
) -> np.ndarray:
    vz = np.asarray(vertical_velocity_ned_m_s, dtype=float)
    roll = np.asarray(roll_rad, dtype=float)
    yaw_rate = np.asarray(yaw_rate_rad_s, dtype=float)
    speed = np.asarray(ground_speed_m_s, dtype=float)
    labels = np.full(len(vz), "transition", dtype=object)
    finite = np.isfinite(vz) & np.isfinite(roll) & np.isfinite(yaw_rate) & np.isfinite(speed)
    turn = finite & ((np.abs(roll) >= np.deg2rad(15.0)) | (np.abs(yaw_rate) >= np.deg2rad(15.0)))
    climb = finite & ~turn & (vz <= -0.75)
    descent = finite & ~turn & (vz >= 0.75)
    stable = (
        finite
        & ~turn
        & ~climb
        & ~descent
        & (np.abs(vz) <= 0.5)
        & (np.abs(roll) <= np.deg2rad(10.0))
        & (np.abs(yaw_rate) <= np.deg2rad(10.0))
        & (speed >= 3.0)
    )
    labels[turn] = "turn"
    labels[climb] = "climb"
    labels[descent] = "descent"
    labels[stable] = "stable_level"
    return labels


def _dataset(ulog: ULog, name: str, multi_id: int = 0):
    for dataset in ulog.data_list:
        if dataset.name == name and dataset.multi_id == multi_id:
            return dataset
    return None


def _event_time_key(dataset: Any) -> str:
    if "timestamp_sample" not in dataset.data:
        return "timestamp"
    sample_time = np.asarray(dataset.data["timestamp_sample"], dtype=float)
    finite_positive = np.isfinite(sample_time) & (sample_time > 0.0)
    if len(sample_time) and float(np.mean(finite_positive)) >= 0.95:
        return "timestamp_sample"
    return "timestamp"


def _event_times(dataset: Any) -> np.ndarray:
    key = _event_time_key(dataset)
    return np.asarray(dataset.data[key], dtype=np.int64)


def _finite_ratio(values: Any) -> float:
    array = np.asarray(values)
    if not len(array) or not np.issubdtype(array.dtype, np.number):
        return 0.0
    return float(np.mean(np.isfinite(array.astype(float))))


def _topic_summary(dataset: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    times = _event_times(dataset)
    summary = summarize_timestamps(times)
    event_time_key = _event_time_key(dataset)
    summary["timestamp_source"] = event_time_key
    summary["key_field_finite_ratio"] = {
        field: _finite_ratio(dataset.data[field]) for field in fields if field in dataset.data
    }
    if "timestamp" in dataset.data and "timestamp_sample" in dataset.data:
        timestamp_sample = np.asarray(dataset.data["timestamp_sample"], dtype=float)
        valid_sample_time = np.isfinite(timestamp_sample) & (timestamp_sample > 0.0)
        summary["timestamp_sample_valid_ratio"] = float(np.mean(valid_sample_time)) if len(timestamp_sample) else 0.0
        delay_ms = (
            np.asarray(dataset.data["timestamp"], dtype=float)[valid_sample_time]
            - timestamp_sample[valid_sample_time]
        ) * 1e-3
        delay_ms = delay_ms[np.isfinite(delay_ms)]
        if len(delay_ms):
            summary["publication_delay_ms"] = {
                "median": float(np.median(delay_ms)),
                "p99": float(np.quantile(delay_ms, 0.99)),
                "max": float(np.max(delay_ms)),
            }
    return summary


def _nearest_values(
    reference_us: np.ndarray,
    dataset: Any | None,
    field: str,
    *,
    freshness_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full(len(reference_us), np.nan, dtype=float)
    fresh = np.zeros(len(reference_us), dtype=bool)
    if dataset is None or field not in dataset.data:
        return values, fresh
    timestamps = _event_times(dataset)
    source_values = np.asarray(dataset.data[field], dtype=float)
    if not len(timestamps):
        return values, fresh
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    source_values = source_values[order]
    right = np.searchsorted(timestamps, reference_us, side="left")
    left = np.clip(right - 1, 0, len(timestamps) - 1)
    right = np.clip(right, 0, len(timestamps) - 1)
    choose_right = np.abs(timestamps[right] - reference_us) < np.abs(reference_us - timestamps[left])
    index = np.where(choose_right, right, left)
    age_s = np.abs(timestamps[index] - reference_us).astype(float) * 1e-6
    values = source_values[index]
    fresh = (age_s <= freshness_s) & np.isfinite(values)
    return values, fresh


def _zoh_values(
    reference_us: np.ndarray,
    dataset: Any | None,
    field: str,
    *,
    freshness_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full(len(reference_us), np.nan, dtype=float)
    fresh = np.zeros(len(reference_us), dtype=bool)
    if dataset is None or field not in dataset.data:
        return values, fresh
    timestamps = _event_times(dataset)
    source_values = np.asarray(dataset.data[field], dtype=float)
    if not len(timestamps):
        return values, fresh
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    source_values = source_values[order]
    index = np.searchsorted(timestamps, reference_us, side="right") - 1
    available = index >= 0
    clipped = np.clip(index, 0, len(timestamps) - 1)
    age_s = (reference_us - timestamps[clipped]).astype(float) * 1e-6
    values[available] = source_values[clipped[available]]
    fresh = available & (age_s >= 0.0) & (age_s <= freshness_s) & np.isfinite(values)
    return values, fresh


def _range_summary(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if mask is not None:
        array = array[np.asarray(mask, dtype=bool)]
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "min": float(np.min(finite)),
        "p01": float(np.quantile(finite, 0.01)),
        "p50": float(np.median(finite)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def _quaternion_euler(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    norm = np.linalg.norm(q, axis=1)
    normalized = np.full_like(q, np.nan, dtype=float)
    valid = np.isfinite(norm) & (norm > 0.0)
    normalized[valid] = q[valid] / norm[valid, None]
    w, x, y, z = normalized.T
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw, norm


def _nearest_time_offset_ms(first: Any | None, second: Any | None) -> dict[str, Any] | None:
    if first is None or second is None:
        return None
    reference = _event_times(first)
    source = _event_times(second)
    if not len(reference) or not len(source):
        return None
    source = np.sort(source, kind="stable")
    right = np.searchsorted(source, reference, side="left")
    left = np.clip(right - 1, 0, len(source) - 1)
    right = np.clip(right, 0, len(source) - 1)
    distance_us = np.minimum(np.abs(reference - source[left]), np.abs(source[right] - reference))
    return {
        "median": float(np.median(distance_us) * 1e-3),
        "p99": float(np.quantile(distance_us, 0.99) * 1e-3),
        "max": float(np.max(distance_us) * 1e-3),
    }


def _flight_analysis(ulog: ULog) -> dict[str, Any]:
    datasets = {name: _dataset(ulog, name) for name in KEY_TOPIC_FIELDS}
    local = datasets["vehicle_local_position"]
    if local is None:
        return {"airborne_duration_s": 0.0, "model_ready_duration_s": 0.0, "analysis_error": "vehicle_local_position missing"}

    reference_us = _event_times(local)
    positive_reference_gaps_s = np.diff(reference_us).astype(float) * 1e-6
    positive_reference_gaps_s = positive_reference_gaps_s[positive_reference_gaps_s > 0.0]
    median_reference_period_s = (
        float(np.median(positive_reference_gaps_s)) if len(positive_reference_gaps_s) else math.nan
    )
    reference_rate_hz = (
        float(1.0 / median_reference_period_s)
        if np.isfinite(median_reference_period_s) and median_reference_period_s > 0.0
        else 0.0
    )
    integration_gap_s = max(0.05, 2.5 * median_reference_period_s) if reference_rate_hz else 0.05
    local_values: dict[str, np.ndarray] = {}
    state_valid = np.ones(len(reference_us), dtype=bool)
    for field in ("x", "y", "z", "vx", "vy", "vz"):
        if field not in local.data:
            state_valid[:] = False
            local_values[field] = np.full(len(reference_us), np.nan)
        else:
            local_values[field] = np.asarray(local.data[field], dtype=float)
            state_valid &= np.isfinite(local_values[field])
    for field in ("xy_valid", "z_valid", "v_xy_valid", "v_z_valid"):
        if field not in local.data:
            state_valid[:] = False
        else:
            state_valid &= np.asarray(local.data[field], dtype=float) > 0.5

    arming_state, arming_fresh = _zoh_values(reference_us, datasets["vehicle_status"], "arming_state", freshness_s=1.5)
    landed, landed_fresh = _zoh_values(reference_us, datasets["vehicle_land_detected"], "landed", freshness_s=1.5)
    airborne = arming_fresh & landed_fresh & (arming_state == 2.0) & (landed < 0.5)

    q_columns = []
    attitude_fresh = np.ones(len(reference_us), dtype=bool)
    for field in ("q[0]", "q[1]", "q[2]", "q[3]"):
        values, fresh = _nearest_values(reference_us, datasets["vehicle_attitude"], field, freshness_s=0.05)
        q_columns.append(values)
        attitude_fresh &= fresh
    quaternion = np.column_stack(q_columns)
    roll, pitch, _, quaternion_norm = _quaternion_euler(quaternion)

    angular: dict[str, np.ndarray] = {}
    angular_fresh = np.ones(len(reference_us), dtype=bool)
    for field in ("xyz[0]", "xyz[1]", "xyz[2]"):
        angular[field], fresh = _nearest_values(
            reference_us, datasets["vehicle_angular_velocity"], field, freshness_s=0.05
        )
        angular_fresh &= fresh

    motor, motor_fresh = _nearest_values(reference_us, datasets["actuator_motors"], "control[0]", freshness_s=0.05)
    servos: dict[str, np.ndarray] = {}
    servo_fresh = np.ones(len(reference_us), dtype=bool)
    for field in ("control[0]", "control[1]", "control[2]"):
        servos[field], fresh = _nearest_values(reference_us, datasets["actuator_servos"], field, freshness_s=0.05)
        servo_fresh &= fresh

    wing_phase = datasets["wing_phase"]
    logged_phase, logged_phase_fresh = _nearest_values(
        reference_us, wing_phase, "phase_rad", freshness_s=0.10
    )
    if wing_phase is not None and "phase_valid" in wing_phase.data:
        phase_valid, phase_valid_fresh = _zoh_values(reference_us, wing_phase, "phase_valid", freshness_s=0.10)
        logged_phase_fresh &= phase_valid_fresh & (phase_valid > 0.5)
    encoder_phase, encoder_phase_fresh = _nearest_values(
        reference_us, datasets["encoder_count"], "position_raw", freshness_s=0.10
    )
    flap_frequency, frequency_fresh = _nearest_values(
        reference_us, wing_phase, "flap_frequency_hz", freshness_s=0.10
    )
    fallback_frequency, fallback_frequency_fresh = _nearest_values(
        reference_us, datasets["flap_frequency"], "frequency_hz", freshness_s=0.10
    )
    use_fallback_frequency = ~frequency_fresh & fallback_frequency_fresh
    flap_frequency[use_fallback_frequency] = fallback_frequency[use_fallback_frequency]
    frequency_fresh |= fallback_frequency_fresh
    frequency_ok = frequency_fresh & (flap_frequency > 0.5) & (flap_frequency < 20.0)

    failsafe, failsafe_fresh = _zoh_values(
        reference_us, datasets["vehicle_status"], "failsafe", freshness_s=1.5
    )
    failsafe_active = failsafe_fresh & (failsafe > 0.5)
    model_ready = (
        airborne
        & state_valid
        & attitude_fresh
        & angular_fresh
        & motor_fresh
        & servo_fresh
        & ~failsafe_active
    )
    logged_phase_ready = model_ready & logged_phase_fresh & np.isfinite(logged_phase) & frequency_ok
    encoder_relative_phase_ready = model_ready & encoder_phase_fresh & np.isfinite(encoder_phase) & frequency_ok
    if np.any(logged_phase_ready):
        phase_source = "logged_wing_phase"
    elif np.any(encoder_relative_phase_ready):
        phase_source = "encoder_count_relative_only"
    else:
        phase_source = "missing_or_invalid"
    airborne_duration_s = duration_from_mask(reference_us, airborne, max_gap_s=integration_gap_s)
    model_ready_duration_s = duration_from_mask(reference_us, model_ready, max_gap_s=integration_gap_s)

    true_airspeed, airspeed_fresh = _zoh_values(
        reference_us, datasets["airspeed_validated"], "true_airspeed_m_s", freshness_s=0.25
    )
    airspeed_usable = airspeed_fresh & (true_airspeed >= 0.0) & (true_airspeed <= 30.0)
    wind_north, wind_north_fresh = _zoh_values(reference_us, datasets["wind"], "windspeed_north", freshness_s=0.35)
    wind_east, wind_east_fresh = _zoh_values(reference_us, datasets["wind"], "windspeed_east", freshness_s=0.35)
    enriched = model_ready & airspeed_usable & wind_north_fresh & wind_east_fresh

    gps_fix, gps_fresh = _zoh_values(reference_us, datasets["sensor_gps"], "fix_type", freshness_s=1.5)
    relative_position_valid, relative_fresh = _zoh_values(
        reference_us, datasets["sensor_gnss_relative"], "relative_position_valid", freshness_s=1.5
    )
    nav_state, nav_fresh = _zoh_values(reference_us, datasets["vehicle_status"], "nav_state", freshness_s=1.5)

    ground_speed = np.hypot(local_values["vx"], local_values["vy"])
    maneuver_labels = classify_maneuvers(local_values["vz"], roll, angular["xyz[2]"], ground_speed)
    maneuver_duration_s = durations_by_label(
        reference_us, model_ready, maneuver_labels, max_gap_s=integration_gap_s
    )
    nav_labels = np.full(len(reference_us), "missing", dtype=object)
    nav_labels[nav_fresh & np.isfinite(nav_state)] = [
        str(int(value)) for value in nav_state[nav_fresh & np.isfinite(nav_state)]
    ]
    nav_state_duration_s = durations_by_label(
        reference_us, model_ready & nav_fresh, nav_labels, max_gap_s=integration_gap_s
    )
    nav_state_duration_s.pop("missing", None)

    model_ready_count = max(1, int(np.sum(model_ready)))
    quaternion_error = np.abs(quaternion_norm - 1.0)
    quaternion_error = quaternion_error[model_ready & np.isfinite(quaternion_error)]
    servo_saturated = np.zeros(len(reference_us), dtype=bool)
    for values in servos.values():
        servo_saturated |= np.abs(values) >= 0.98
    reset_counts: dict[str, int] = {}
    for field in (
        "xy_reset_counter",
        "z_reset_counter",
        "vxy_reset_counter",
        "vz_reset_counter",
        "heading_reset_counter",
    ):
        if field in local.data:
            values = np.asarray(local.data[field])
            reset_counts[f"vehicle_local_position.{field}"] = int(np.sum(values[1:] != values[:-1]))
    attitude = datasets["vehicle_attitude"]
    if attitude is not None and "quat_reset_counter" in attitude.data:
        values = np.asarray(attitude.data["quat_reset_counter"])
        reset_counts["vehicle_attitude.quat_reset_counter"] = int(np.sum(values[1:] != values[:-1]))

    ranges = {
        "ground_speed_m_s": _range_summary(ground_speed, model_ready),
        "vertical_velocity_ned_m_s": _range_summary(local_values["vz"], model_ready),
        "roll_deg": _range_summary(np.rad2deg(roll), model_ready),
        "pitch_deg": _range_summary(np.rad2deg(pitch), model_ready),
        "yaw_rate_deg_s": _range_summary(np.rad2deg(angular["xyz[2]"]), model_ready),
        "motor_control_0": _range_summary(motor, model_ready),
        "servo_control_0": _range_summary(servos["control[0]"], model_ready),
        "servo_control_1": _range_summary(servos["control[1]"], model_ready),
        "servo_control_2": _range_summary(servos["control[2]"], model_ready),
        "flap_frequency_hz": _range_summary(flap_frequency, model_ready),
        "true_airspeed_m_s": _range_summary(true_airspeed, model_ready & airspeed_fresh),
        "wind_north_m_s": _range_summary(wind_north, model_ready & wind_north_fresh),
        "wind_east_m_s": _range_summary(wind_east, model_ready & wind_east_fresh),
    }

    return {
        "reference_topic": "vehicle_local_position",
        "reference_sample_count": int(len(reference_us)),
        "reference_median_rate_hz": reference_rate_hz,
        "integration_max_gap_s": integration_gap_s,
        "airborne_duration_s": airborne_duration_s,
        "model_ready_duration_s": model_ready_duration_s,
        "logged_phase_valid_duration_s": duration_from_mask(
            reference_us, logged_phase_ready, max_gap_s=integration_gap_s
        ),
        "encoder_relative_phase_reconstructable_duration_s": duration_from_mask(
            reference_us, encoder_relative_phase_ready, max_gap_s=integration_gap_s
        ),
        "airdata_enriched_duration_s": duration_from_mask(
            reference_us, enriched, max_gap_s=integration_gap_s
        ),
        "model_ready_fraction_of_airborne": float(model_ready_duration_s / airborne_duration_s) if airborne_duration_s else 0.0,
        "phase_source": phase_source,
        "coverage_on_model_ready": {
            "airspeed_fresh_finite_ratio": float(np.sum(model_ready & airspeed_fresh) / model_ready_count),
            "airspeed_physically_usable_ratio": float(np.sum(model_ready & airspeed_usable) / model_ready_count),
            "wind_fresh_finite_ratio": float(np.sum(model_ready & wind_north_fresh & wind_east_fresh) / model_ready_count),
            "gps_3d_or_better_ratio": float(np.sum(model_ready & gps_fresh & (gps_fix >= 3.0)) / model_ready_count),
            "gps_rtk_fixed_ratio": float(np.sum(model_ready & gps_fresh & (gps_fix >= 6.0)) / model_ready_count),
            "relative_gnss_valid_ratio": float(
                np.sum(model_ready & relative_fresh & (relative_position_valid > 0.5)) / model_ready_count
            ),
        },
        "quality": {
            "quaternion_norm_error_p99": float(np.quantile(quaternion_error, 0.99)) if len(quaternion_error) else None,
            "motor_control_saturation_ratio": float(np.sum(model_ready & (np.abs(motor) >= 0.98)) / model_ready_count),
            "servo_control_saturation_ratio": float(np.sum(model_ready & servo_saturated) / model_ready_count),
            "negative_true_airspeed_ratio": float(
                np.sum(model_ready & airspeed_fresh & (true_airspeed < 0.0)) / model_ready_count
            ),
            "failsafe_active_sample_ratio": float(np.sum(airborne & failsafe_active) / max(1, int(np.sum(airborne)))),
            "estimator_reset_counts": reset_counts,
            "phase_encoder_nearest_offset_ms": _nearest_time_offset_ms(datasets["wing_phase"], datasets["encoder_count"]),
        },
        "maneuver_duration_s": maneuver_duration_s,
        "nav_state_duration_s": nav_state_duration_s,
        "ranges": ranges,
    }


def _admission(record: dict[str, Any]) -> tuple[str, list[str]]:
    if record.get("parse_error"):
        return "exclude_corrupt_or_unparseable", [record["parse_error"]]
    missing = record.get("missing_required_topics", [])
    if missing:
        return "exclude_missing_core_signals", [f"missing required topics: {', '.join(missing)}"]
    analysis = record["flight_analysis"]
    usable = float(analysis.get("model_ready_duration_s", 0.0))
    airborne = float(analysis.get("airborne_duration_s", 0.0))
    fraction = float(analysis.get("model_ready_fraction_of_airborne", 0.0))
    reference_rate_hz = float(analysis.get("reference_median_rate_hz", 0.0))
    reasons = [
        f"model_ready={usable:.2f}s",
        f"airborne={airborne:.2f}s",
        f"ready_fraction={fraction:.3f}",
        f"state_rate={reference_rate_hz:.3f}Hz",
    ]
    if usable >= 60.0 and fraction >= 0.75 and reference_rate_hz < 40.0:
        return "review_low_state_rate", reasons
    if usable >= 60.0 and fraction >= 0.75:
        return "eligible", reasons
    if usable >= 20.0:
        return "review_short_or_incomplete", reasons
    return "exclude_short_or_incomplete", reasons


def audit_log(path: str | Path, source_root: str | Path) -> dict[str, Any]:
    log_path = Path(path)
    root = Path(source_root)
    timestamp = parse_log_datetime(log_path.name)
    record: dict[str, Any] = {
        "path": str(log_path.resolve()),
        "relative_path": str(log_path.resolve().relative_to(root.resolve())),
        "filename": log_path.name,
        "filename_datetime": timestamp.isoformat() if timestamp else None,
        "file_size_bytes": log_path.stat().st_size,
    }
    with log_path.open("rb") as handle:
        magic = handle.read(7)
    record["valid_ulog_magic"] = magic == b"ULog\x01\x12\x35"
    if not record["valid_ulog_magic"]:
        record["parse_error"] = f"invalid ULog magic: {magic.hex()}"
        record["admission_status"], record["admission_reasons"] = _admission(record)
        return record
    try:
        ulog = ULog(str(log_path))
    except Exception as error:  # pyulog raises several format-specific exception types
        record["parse_error"] = f"{type(error).__name__}: {error}"
        record["admission_status"], record["admission_reasons"] = _admission(record)
        return record

    topics = sorted({dataset.name for dataset in ulog.data_list})
    topic_schema = {
        name: sorted(_dataset(ulog, name).data.keys())
        for name in KEY_TOPIC_FIELDS
        if _dataset(ulog, name) is not None
    }
    topic_statistics = {
        name: _topic_summary(_dataset(ulog, name), fields)
        for name, fields in KEY_TOPIC_FIELDS.items()
        if _dataset(ulog, name) is not None
    }
    firmware_keys = ("sys_name", "ver_sw", "ver_sw_release", "ver_hw", "ver_hw_subtype", "sys_uuid", "time_ref_utc")
    record.update(
        {
            "parse_error": None,
            "recorded_duration_s": float((ulog.last_timestamp - ulog.start_timestamp) * 1e-6),
            "ulog_start_timestamp_us": int(ulog.start_timestamp),
            "ulog_last_timestamp_us": int(ulog.last_timestamp),
            "dropout_count": int(len(ulog.dropouts)),
            "dropout_duration_s": float(sum(dropout.duration for dropout in ulog.dropouts) * 1e-3),
            "topic_count": int(len(ulog.data_list)),
            "topics": topics,
            "topic_schema": topic_schema,
            "topic_statistics": topic_statistics,
            "missing_required_topics": sorted(REQUIRED_TOPICS - set(topics)),
            "missing_phase_chain": not ({"wing_phase", "encoder_count"} & set(topics)),
            "firmware": {key: ulog.msg_info_dict.get(key) for key in firmware_keys},
            "initial_parameter_count": int(len(ulog.initial_parameters)),
            "selected_parameters": {key: ulog.initial_parameters.get(key) for key in SELECTED_PARAMETERS},
            "flight_analysis": _flight_analysis(ulog),
        }
    )
    record["admission_status"], record["admission_reasons"] = _admission(record)
    return record


def _assign_groups(records: list[dict[str, Any]], key_builder: Any, field: str, prefix: str) -> list[dict[str, Any]]:
    groups: dict[Any, str] = {}
    descriptions: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("parse_error"):
            record[field] = None
            continue
        key, description = key_builder(record)
        if key not in groups:
            group_id = f"{prefix}{len(groups) + 1}"
            groups[key] = group_id
            descriptions[group_id] = {"group_id": group_id, "description": description, "logs": []}
        group_id = groups[key]
        record[field] = group_id
        descriptions[group_id]["logs"].append(record["relative_path"])
    return list(descriptions.values())


def _firmware_key(record: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    description = record["firmware"]
    key = tuple((name, json.dumps(value, sort_keys=True)) for name, value in sorted(description.items()))
    return key, description


def _schema_key(record: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    description = record["topic_schema"]
    key = tuple((name, tuple(fields)) for name, fields in sorted(description.items()))
    return key, {
        "key_topic_count": len(description),
        "key_topics_present": sorted(description),
        "topic_fields": description,
    }


def _configuration_key(record: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    description = {name: record["selected_parameters"].get(name) for name in STRUCTURAL_PARAMETERS}
    key = tuple((name, json.dumps(value, sort_keys=True)) for name, value in sorted(description.items()))
    return key, description


def _git_head(repository: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parameter_changes(records: list[dict[str, Any]]) -> None:
    previous: dict[str, Any] | None = None
    for record in records:
        if record.get("parse_error"):
            record["selected_parameter_changes_from_previous"] = {}
            continue
        current = record["selected_parameters"]
        if previous is None:
            record["selected_parameter_changes_from_previous"] = {}
        else:
            record["selected_parameter_changes_from_previous"] = {
                key: {"old": previous.get(key), "new": current.get(key)}
                for key in SELECTED_PARAMETERS
                if previous.get(key) != current.get(key)
            }
        previous = current


def _recommend_splits(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [record for record in records if record.get("admission_status") == "eligible"]
    cohorts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in eligible:
        key = (record["firmware_group"], record["configuration_group"])
        cohorts.setdefault(key, []).append(record)
    if not cohorts:
        for record in records:
            record["recommended_split"] = "exclude_or_review"
        return {"primary_cohort": None, "train": [], "validation": [], "sealed_test": [], "ood_holdout": []}
    primary_key = max(
        cohorts,
        key=lambda key: sum(item["flight_analysis"]["model_ready_duration_s"] for item in cohorts[key]),
    )
    primary = cohorts[primary_key]
    dates = sorted({record["filename_datetime"][:10] for record in primary if record.get("filename_datetime")})
    test_date = dates[-1] if dates else None
    validation_date = dates[-2] if len(dates) >= 2 else None
    split = {"train": [], "validation": [], "sealed_test": [], "ood_holdout": []}
    for record in eligible:
        if (record["firmware_group"], record["configuration_group"]) != primary_key:
            target = "ood_holdout"
        else:
            date = record["filename_datetime"][:10]
            if date == test_date:
                target = "sealed_test"
            elif date == validation_date:
                target = "validation"
            else:
                target = "train"
        record["recommended_split"] = target
        split[target].append(record["relative_path"])
    for record in records:
        record.setdefault("recommended_split", "exclude_or_review")
    return {
        "policy": "largest exact firmware-plus-structural-parameter cohort; latest date sealed, previous date validation, earlier dates train",
        "primary_cohort": {"firmware_group": primary_key[0], "configuration_group": primary_key[1]},
        "test_date": test_date,
        "validation_date": validation_date,
        **split,
    }


def build_audit_summary(
    source_root: str | Path,
    *,
    year: int = 2026,
    month: int = 8,
    workers: int = 1,
    audit_repository: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(source_root).resolve()
    all_logs = sorted(root.rglob("*.ulg"))
    included: list[Path] = []
    excluded_outside_month: list[dict[str, Any]] = []
    unrecognized: list[str] = []
    for path in all_logs:
        timestamp = parse_log_datetime(path.name)
        if timestamp is None:
            unrecognized.append(str(path.relative_to(root)))
        elif timestamp.year == year and timestamp.month == month:
            included.append(path)
        else:
            excluded_outside_month.append(
                {"relative_path": str(path.relative_to(root)), "filename_datetime": timestamp.isoformat()}
            )

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(audit_log, included, [root] * len(included)))
    else:
        records = [audit_log(path, root) for path in included]
    records.sort(key=lambda item: (item.get("filename_datetime") or "", item["relative_path"]))

    firmware_groups = _assign_groups(records, _firmware_key, "firmware_group", "F")
    schema_groups = _assign_groups(records, _schema_key, "topic_schema_group", "S")
    configuration_groups = _assign_groups(records, _configuration_key, "configuration_group", "C")
    for record in records:
        record.pop("topic_schema", None)
    record_by_path = {record["relative_path"]: record for record in records}
    for group in firmware_groups:
        members = [record_by_path[path] for path in group["logs"]]
        group["median_sampling_rate_hz"] = {}
        for topic in (
            "vehicle_local_position",
            "vehicle_attitude",
            "vehicle_angular_velocity",
            "actuator_motors",
            "actuator_servos",
            "wing_phase",
            "sensor_gps",
        ):
            rates = [
                member["topic_statistics"][topic]["median_rate_hz"]
                for member in members
                if topic in member.get("topic_statistics", {})
                and member["topic_statistics"][topic]["median_rate_hz"] is not None
            ]
            if rates:
                group["median_sampling_rate_hz"][topic] = float(np.median(rates))
    _parameter_changes(records)
    splits = _recommend_splits(records)

    parsed = [record for record in records if not record.get("parse_error")]
    eligible = [record for record in parsed if record["admission_status"] == "eligible"]
    maneuver_totals = {
        label: float(sum(record["flight_analysis"]["maneuver_duration_s"].get(label, 0.0) for record in eligible))
        for label in ("stable_level", "climb", "descent", "turn", "transition")
    }
    signal_availability = {
        topic: {
            "parsed_log_count": int(sum(topic in record["topics"] for record in parsed)),
            "parsed_log_ratio": float(sum(topic in record["topics"] for record in parsed) / len(parsed)) if parsed else 0.0,
        }
        for topic in KEY_TOPIC_FIELDS
    }
    total_ready_duration_s = float(
        sum(record["flight_analysis"]["model_ready_duration_s"] for record in eligible)
    )
    weighted_coverage_fields = {
        "airspeed_validated": "airspeed_physically_usable_ratio",
        "wind": "wind_fresh_finite_ratio",
        "sensor_gps": "gps_rtk_fixed_ratio",
        "sensor_gnss_relative": "relative_gnss_valid_ratio",
    }
    for topic in (
        "vehicle_local_position",
        "vehicle_attitude",
        "vehicle_angular_velocity",
        "actuator_motors",
        "actuator_servos",
    ):
        signal_availability[topic]["eligible_model_ready_coverage_ratio"] = 1.0 if eligible else 0.0
    if total_ready_duration_s:
        for topic, coverage_field in weighted_coverage_fields.items():
            signal_availability[topic]["eligible_model_ready_coverage_ratio"] = float(
                sum(
                    record["flight_analysis"]["model_ready_duration_s"]
                    * record["flight_analysis"]["coverage_on_model_ready"][coverage_field]
                    for record in eligible
                )
                / total_ready_duration_s
            )
        logged_phase_duration = sum(
            record["flight_analysis"]["logged_phase_valid_duration_s"] for record in eligible
        )
        encoder_phase_duration = sum(
            record["flight_analysis"]["encoder_relative_phase_reconstructable_duration_s"]
            for record in eligible
        )
        signal_availability["wing_phase"]["eligible_model_ready_coverage_ratio"] = float(
            logged_phase_duration / total_ready_duration_s
        )
        signal_availability["encoder_count"]["eligible_model_ready_coverage_ratio"] = float(
            encoder_phase_duration / total_ready_duration_s
        )
        signal_availability["flap_frequency"]["eligible_model_ready_coverage_ratio"] = float(
            encoder_phase_duration / total_ready_duration_s
        )
    aggregate_range_envelopes: dict[str, dict[str, float]] = {}
    for field in (
        "ground_speed_m_s",
        "vertical_velocity_ned_m_s",
        "roll_deg",
        "pitch_deg",
        "yaw_rate_deg_s",
        "motor_control_0",
        "servo_control_0",
        "servo_control_1",
        "servo_control_2",
        "flap_frequency_hz",
        "true_airspeed_m_s",
        "wind_north_m_s",
        "wind_east_m_s",
    ):
        summaries = [
            record["flight_analysis"]["ranges"][field]
            for record in eligible
            if record["flight_analysis"]["ranges"].get(field, {}).get("count", 0) > 0
        ]
        if summaries:
            aggregate_range_envelopes[field] = {
                "minimum": float(min(item["min"] for item in summaries)),
                "minimum_per_log_p01": float(min(item["p01"] for item in summaries)),
                "maximum_per_log_p99": float(max(item["p99"] for item in summaries)),
                "maximum": float(max(item["max"] for item in summaries)),
            }
    split_summary: dict[str, Any] = {}
    for split_name in ("train", "validation", "sealed_test", "ood_holdout"):
        members = [record for record in records if record.get("recommended_split") == split_name]
        split_summary[split_name] = {
            "log_count": len(members),
            "model_ready_duration_s": float(
                sum(record["flight_analysis"]["model_ready_duration_s"] for record in members)
            ),
            "maneuver_duration_s": {
                label: float(
                    sum(record["flight_analysis"]["maneuver_duration_s"].get(label, 0.0) for record in members)
                )
                for label in ("stable_level", "climb", "descent", "turn", "transition")
            },
        }
    timestamp_fallbacks = []
    duplicate_timestamp_count = 0
    backward_timestamp_count = 0
    duplicate_timestamp_count_by_topic: dict[str, int] = {}
    backward_timestamp_count_by_topic: dict[str, int] = {}
    for record in parsed:
        for topic, statistics in record["topic_statistics"].items():
            duplicate_timestamp_count += int(statistics["duplicate_count"])
            backward_timestamp_count += int(statistics["backward_count"])
            duplicate_timestamp_count_by_topic[topic] = (
                duplicate_timestamp_count_by_topic.get(topic, 0) + int(statistics["duplicate_count"])
            )
            backward_timestamp_count_by_topic[topic] = (
                backward_timestamp_count_by_topic.get(topic, 0) + int(statistics["backward_count"])
            )
            if statistics.get("timestamp_sample_valid_ratio", 1.0) < 0.95:
                timestamp_fallbacks.append(
                    {
                        "relative_path": record["relative_path"],
                        "topic": topic,
                        "timestamp_sample_valid_ratio": statistics["timestamp_sample_valid_ratio"],
                    }
                )
    largest_dropout = max(parsed, key=lambda record: record["dropout_duration_s"], default=None)
    reset_logs = []
    for record in parsed:
        reset_counts = record["flight_analysis"]["quality"]["estimator_reset_counts"]
        if sum(reset_counts.values()):
            reset_logs.append({"relative_path": record["relative_path"], "counts": reset_counts})
    quality_summary = {
        "unparseable_logs": [
            {"relative_path": record["relative_path"], "error": record["parse_error"]}
            for record in records
            if record.get("parse_error")
        ],
        "low_state_rate_logs": [
            record["relative_path"]
            for record in parsed
            if record["flight_analysis"]["reference_median_rate_hz"] < 40.0
        ],
        "eligible_low_airspeed_coverage_logs": [
            record["relative_path"]
            for record in eligible
            if record["flight_analysis"]["coverage_on_model_ready"]["airspeed_physically_usable_ratio"] < 0.8
        ],
        "eligible_negative_true_airspeed_logs": [
            record["relative_path"]
            for record in eligible
            if record["flight_analysis"]["ranges"]["true_airspeed_m_s"].get("p01", 0.0) < 0.0
        ],
        "eligible_servo_saturation_over_5pct_logs": [
            record["relative_path"]
            for record in eligible
            if record["flight_analysis"]["quality"]["servo_control_saturation_ratio"] >= 0.05
        ],
        "largest_ulog_dropout": None
        if largest_dropout is None
        else {
            "relative_path": largest_dropout["relative_path"],
            "duration_s": largest_dropout["dropout_duration_s"],
            "count": largest_dropout["dropout_count"],
        },
        "timestamp_sample_fallbacks": timestamp_fallbacks,
        "duplicate_timestamp_count_across_key_topics": duplicate_timestamp_count,
        "backward_timestamp_count_across_key_topics": backward_timestamp_count,
        "duplicate_timestamp_count_by_topic": {
            topic: count for topic, count in duplicate_timestamp_count_by_topic.items() if count
        },
        "backward_timestamp_count_by_topic": {
            topic: count for topic, count in backward_timestamp_count_by_topic.items() if count
        },
        "logs_with_estimator_resets": reset_logs,
        "non_august_logs_inside_august_named_directory": [
            item
            for item in excluded_outside_month
            if any(part.startswith(("2026.8", "2026-08")) for part in Path(item["relative_path"]).parts[:-1])
        ],
    }
    for record in parsed:
        record["key_topics_present"] = sorted(set(record["topics"]) & set(KEY_TOPIC_FIELDS))
        record.pop("topics", None)
    audit_repo = Path(audit_repository).resolve() if audit_repository else None
    return {
        "audit_version": AUDIT_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "scope": {"source_root": str(root), "year": year, "month": month},
        "provenance": {
            "source_repository_head": _git_head(root),
            "audit_repository_head_before_outputs": _git_head(audit_repo) if audit_repo else None,
        },
        "method": {
            "event_time": "timestamp_sample when available, otherwise timestamp",
            "airborne": "arming_state == 2 and landed == 0, both fresh within 1.5 s",
            "model_ready": "airborne plus valid local state and fresh attitude/angular rate/motor/three servos; phase, airspeed, wind, and GNSS are reported separately",
            "maximum_integrated_gap": "max(0.05 s, 2.5 times the median vehicle_local_position period)",
            "physically_usable_true_airspeed": "fresh and 0 <= true_airspeed_m_s <= 30",
            "maneuver_thresholds": {
                "turn": "abs(roll) >= 15 deg or abs(body yaw rate) >= 15 deg/s",
                "climb": "NED vz <= -0.75 m/s outside turn",
                "descent": "NED vz >= 0.75 m/s outside turn",
                "stable_level": "abs(vz) <= 0.5 m/s, abs(roll) <= 10 deg, abs(yaw rate) <= 10 deg/s, ground speed >= 3 m/s",
                "transition": "remaining finite model-ready samples",
            },
            "admission": "eligible when state rate >= 40 Hz, model-ready duration >= 60 s, and at least 75% of airborne duration; 20-60 s or lower-rate data requires review",
        },
        "counts": {
            "all_ulg_under_source_root": len(all_logs),
            "included_august_logs": len(records),
            "excluded_outside_month": len(excluded_outside_month),
            "unrecognized_filename": len(unrecognized),
            "parsed": len(parsed),
            "unparseable": len(records) - len(parsed),
            "eligible": len(eligible),
            "review_or_exclude": len(records) - len(eligible),
        },
        "durations_s": {
            "recorded_parsed": float(sum(record["recorded_duration_s"] for record in parsed)),
            "airborne_parsed": float(sum(record["flight_analysis"]["airborne_duration_s"] for record in parsed)),
            "model_ready_eligible": float(sum(record["flight_analysis"]["model_ready_duration_s"] for record in eligible)),
            "logged_phase_valid_eligible": float(
                sum(record["flight_analysis"]["logged_phase_valid_duration_s"] for record in eligible)
            ),
            "encoder_relative_phase_reconstructable_eligible": float(
                sum(
                    record["flight_analysis"]["encoder_relative_phase_reconstructable_duration_s"]
                    for record in eligible
                )
            ),
            "airdata_enriched_eligible": float(
                sum(record["flight_analysis"]["airdata_enriched_duration_s"] for record in eligible)
            ),
        },
        "maneuver_duration_s_eligible": maneuver_totals,
        "aggregate_range_envelopes_eligible": aggregate_range_envelopes,
        "signal_availability": signal_availability,
        "firmware_groups": firmware_groups,
        "topic_schema_groups": schema_groups,
        "configuration_groups": configuration_groups,
        "recommended_splits": splits,
        "recommended_split_summary": split_summary,
        "quality_summary": quality_summary,
        "unrecognized_filenames": unrecognized,
        "logs": records,
    }


def write_json(summary: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _format_seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def render_markdown(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    durations = summary["durations_s"]
    splits = summary["recommended_splits"]
    split_summary = summary["recommended_split_summary"]
    quality = summary["quality_summary"]
    lines = [
        "# 2026 年 8 月 ULG 数据审计（Step 0）",
        "",
        f"生成时间：`{summary['generated_at']}`",
        f"审计版本：`{summary['audit_version']}`",
        f"QgcLogs HEAD：`{summary['provenance']['source_repository_head']}`",
        "",
        "## 结论",
        "",
        f"共识别 {counts['included_august_logs']} 条 2026 年 8 月 ULG；{counts['parsed']} 条可解析，"
        f"{counts['unparseable']} 条损坏或不可解析，{counts['eligible']} 条达到自动准入线。"
        f"达到准入线的日志提供 {durations['model_ready_eligible']:.1f} s 核心 trajectory-ready 数据，"
        f"其中 {durations['airdata_enriched_eligible']:.1f} s 同时具有新鲜空速和风估计。"
        f"有效 logged mechanical phase 仅 {durations['logged_phase_valid_eligible']:.1f} s；"
        f"另有 {durations['encoder_relative_phase_reconstructable_eligible']:.1f} s 只具备 encoder 相对 phase 重建条件。",
        "",
        "这些数据足以进入下一阶段的数据契约与基线设计，但不能直接合并训练：月份内存在明确的飞控硬件、"
        "固件和结构参数分组，必须先固定单一同构 cohort，并保持整日志、整日期的 split 边界。",
        "",
        "## 范围与方法",
        "",
        f"- 扫描根目录：`{summary['scope']['source_root']}`。按文件名时间选择 2026-08；"
        f"另有 {counts['excluded_outside_month']} 条非 8 月 ULG 被显式排除。8 月命名目录内误放的非 8 月文件为 "
        + ", ".join(
            f"`{item['relative_path']}`"
            for item in quality["non_august_logs_inside_august_named_directory"]
        )
        + "。",
        "- 事件时间在 `timestamp_sample` 有效时优先使用；全零等无效值显式回退到 `timestamp`。可用时长在 "
        "`vehicle_local_position` 时间轴上积分，仅计入 armed、airborne、状态有效且姿态/角速度/电机/三个舵面同时新鲜的相邻区间；"
        "gap 上限为 50 ms 与 2.5 倍原生状态周期中的较大者。",
        "- logged mechanical phase、encoder 相对 phase、空速、风和 RTK 均单独报告覆盖率；它们不作为核心 trajectory 准入硬条件。"
        "机动标签是审计用运动学代理，不是监督真值。",
        "- 自动准入线为状态采样率至少 40 Hz、核心可用时长至少 60 s，且不少于 airborne 时长的 75%。",
        "",
        "## 固件、硬件与 schema 变化",
        "",
        f"可解析日志形成 {len(summary['firmware_groups'])} 个固件/硬件组、"
        f"{len(summary['configuration_groups'])} 个结构参数组和 {len(summary['topic_schema_groups'])} 个关键 topic schema 组。",
        "",
        "| 固件组 | ver_sw | 硬件 | subtype | 飞控 UUID | 日志数 | 状态 Hz | actuator Hz |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for group in summary["firmware_groups"]:
        description = group["description"]
        rates = group.get("median_sampling_rate_hz", {})
        lines.append(
            f"| {group['group_id']} | `{description.get('ver_sw')}` | {description.get('ver_hw')} | "
            f"{description.get('ver_hw_subtype')} | `{description.get('sys_uuid')}` | {len(group['logs'])} | "
            f"{_format_seconds(rates.get('vehicle_local_position'))} | {_format_seconds(rates.get('actuator_motors'))} |"
        )

    lines.extend(
        [
            "",
            "| 参数组 | 日志数 | FLAP_RATIO | FW_USE_AIRSPD | AIRSPD trim | pitch offset | allocation pitch gain | PWM reverse |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group in summary["configuration_groups"]:
        description = group["description"]
        lines.append(
            f"| {group['group_id']} | {len(group['logs'])} | {description.get('FLAP_RATIO')} | "
            f"{description.get('FW_USE_AIRSPD')} | {description.get('FW_AIRSPD_TRIM')} | "
            f"{description.get('FW_PSP_OFF')} | {description.get('CA_SV_CS0_TRQ_P')} | "
            f"{description.get('PWM_MAIN_REV')} |"
        )
    lines.extend(
        [
            "",
            "| schema 组 | 日志数 | 关键 topic 数 | 缺失的审计 topic |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    all_key_topics = set(KEY_TOPIC_FIELDS)
    for group in summary["topic_schema_groups"]:
        description = group["description"]
        missing = sorted(all_key_topics - set(description["key_topics_present"]))
        lines.append(
            f"| {group['group_id']} | {len(group['logs'])} | {description['key_topic_count']} | "
            f"{', '.join(f'`{topic}`' for topic in missing) or '-'} |"
        )

    lines.extend(
        [
            "",
            "结构参数变化不能视为普通日志噪声：`FLAP_RATIO`、固定翼空速目标、pitch offset、control allocation "
            "增益以及 PWM 范围/反向均出现在审计参数集中；逐日志旧值/新值见机器汇总的 "
            "`selected_parameter_changes_from_previous`。`ASPD_SCALE_1` 属于逐次标定量，保留报告但不用于结构参数分组。",
            "",
            "## 信号覆盖",
            "",
            "| 信号 | 可解析日志 topic 覆盖 | eligible 核心时段有效覆盖 | 角色 |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    roles = {
        "vehicle_local_position": "位置/速度与主时间轴",
        "vehicle_attitude": "姿态",
        "vehicle_angular_velocity": "角速度",
        "actuator_motors": "扑翼主驱动控制",
        "actuator_servos": "三个舵面控制",
        "wing_phase": "logged mechanical phase",
        "encoder_count": "仅相对 encoder phase 重建",
        "flap_frequency": "扑翼频率",
        "airspeed_validated": "空速（附加）",
        "wind": "风估计（附加）",
        "sensor_gps": "GPS/RTK fix（附加）",
        "sensor_gnss_relative": "相对 GNSS（附加）",
    }
    for topic, role in roles.items():
        availability = summary["signal_availability"][topic]
        valid_coverage = availability.get("eligible_model_ready_coverage_ratio")
        valid_coverage_text = "-" if valid_coverage is None else f"{100.0 * valid_coverage:.1f}%"
        lines.append(
            f"| `{topic}` | {availability['parsed_log_count']}/{counts['parsed']} "
            f"({100.0 * availability['parsed_log_ratio']:.1f}%) | {valid_coverage_text} | {role} |"
        )

    lines.extend(
        [
            "",
            "## 飞行与控制覆盖",
            "",
            "达到准入线的日志按互斥运动学代理累计：",
            "",
            "| 状态 | 时长 s |",
            "| --- | ---: |",
        ]
    )
    for label, value in summary["maneuver_duration_s_eligible"].items():
        lines.append(f"| `{label}` | {value:.1f} |")

    lines.extend(
        [
            "",
            "下表是 eligible 日志的逐日志 p01/p99 外包络，避免少量单点极值主导范围判断：",
            "",
            "| 变量 | 最低 per-log p01 | 最高 per-log p99 | 单位 |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    units = {
        "ground_speed_m_s": "m/s",
        "vertical_velocity_ned_m_s": "m/s",
        "roll_deg": "deg",
        "pitch_deg": "deg",
        "yaw_rate_deg_s": "deg/s",
        "motor_control_0": "normalized",
        "servo_control_0": "normalized",
        "servo_control_1": "normalized",
        "servo_control_2": "normalized",
        "flap_frequency_hz": "Hz",
        "true_airspeed_m_s": "m/s",
        "wind_north_m_s": "m/s",
        "wind_east_m_s": "m/s",
    }
    for field, envelope in summary["aggregate_range_envelopes_eligible"].items():
        lines.append(
            f"| `{field}` | {envelope['minimum_per_log_p01']:.3f} | "
            f"{envelope['maximum_per_log_p99']:.3f} | {units[field]} |"
        )

    lines.extend(
        [
            "",
            "每条日志的 min/p01/p50/p99/max 位于机器汇总的 `flight_analysis.ranges`。该范围可用于下一阶段"
            "定义状态/控制归一化和 OOD 边界，但统计量不得从 sealed test 拟合。",
            "",
            "## 逐日志准入",
            "",
            "| 日期时间 | 日志 | 记录 s | airborne s | 核心可用 s | logged phase s | encoder 相对 phase s | 状态 | split |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for record in summary["logs"]:
        analysis = record.get("flight_analysis", {})
        lines.append(
            f"| {record.get('filename_datetime') or '-'} | `{record['filename']}` | "
            f"{_format_seconds(record.get('recorded_duration_s'))} | "
            f"{_format_seconds(analysis.get('airborne_duration_s'))} | "
            f"{_format_seconds(analysis.get('model_ready_duration_s'))} | "
            f"{_format_seconds(analysis.get('logged_phase_valid_duration_s'))} | "
            f"{_format_seconds(analysis.get('encoder_relative_phase_reconstructable_duration_s'))} | "
            f"`{record['admission_status']}` | `{record['recommended_split']}` |"
        )

    lines.extend(
        [
            "",
            "## 主要质量问题与风险",
            "",
            "- 两条文件不可用于任何建模："
            + "; ".join(
                f"`{item['relative_path']}` ({item['error']})" for item in quality["unparseable_logs"]
            )
            + "。",
            f"- {len(quality['low_state_rate_logs'])} 条 8/10 日志的主状态约 10 Hz，而后续批次约 50 Hz；"
            "它们可保留作低速率/OOD 研究，但不能直接混入主 cohort。",
            "- 全部可解析日志虽含 `wing_phase` topic，但 eligible 核心时段的有效 logged phase 为 0%；"
            "`encoder_count` 只支持相对相位重建，且所有日志均无 `hall_event`。如果 Step 1 需要跨日志一致 mechanical phase，当前数据链尚未闭合。",
            f"- eligible 中空速有效覆盖低于 80% 的日志有 {len(quality['eligible_low_airspeed_coverage_logs'])} 条；"
            f"出现负 `true_airspeed_m_s` 的 eligible 日志有 {len(quality['eligible_negative_true_airspeed_logs'])} 条；"
            f"舵面命令接近归一化边界超过 5% 的日志有 {len(quality['eligible_servo_saturation_over_5pct_logs'])} 条。"
            "名单在机器汇总 `quality_summary`。",
            f"- 最大 ULog dropout 为 `{quality['largest_ulog_dropout']['relative_path']}` 的 "
            f"{quality['largest_ulog_dropout']['duration_s']:.3f} s；所有窗口生成仍需按原始 gap 切断。",
            f"- 关键 topic 共检测到 {quality['duplicate_timestamp_count_across_key_topics']} 个重复时间戳、"
            f"{quality['backward_timestamp_count_across_key_topics']} 个倒序时间戳；"
            f"重复全部来自 `manual_control_setpoint`，倒序主要来自无效的 `sensor_gnss_relative`。"
            f"{len(quality['logs_with_estimator_resets'])} 条日志在全记录范围出现 estimator reset counter 变化。",
            "- 损坏文件、缺 topic、短航段和低同步覆盖均按显式状态保留，不静默回退。每个 topic 的采样率、"
            "finite 比例、重复/倒序时间戳、p99/max gap、发布延迟以及 ULog dropout 见机器汇总。",
            "- `time_ref_utc` 在抽取的 ULog 元数据中不能提供可靠绝对 UTC；本审计以文件名时间组织 session，"
            "不把文件名当传感器级同步证据。`sensor_gps.timestamp_sample` 在本批次为全零，审计显式回退到其 "
            "`timestamp`；下游通用 event-time 逻辑必须覆盖这一边界。",
            "- `wind` 与 `airspeed_validated` 是估计器/融合输出，未来模型若把它们同时作为输入和评估真值，"
            "可能引入闭环或 estimator leakage；应在 Step 1 明确因果可用性和部署时可得性。",
            "- 禁止随机拆 sample/window。同一次飞行、相邻 session 和相同标定必须整体进入一个 split；"
            "任何 causal window 不得跨日志、模式变化、reset 或大 gap。",
            "- 本次允许查看 sealed-test 候选仅用于输入完整性和覆盖审计。一旦接受以下划分，就冻结日志名单，"
            "不得再用 sealed test 的分布、轨迹或指标选择特征、阈值或模型。",
            "",
            "## 推荐划分",
            "",
            f"自动策略选择数据量最大的完全一致固件+结构参数 cohort：`{splits.get('primary_cohort')}`。"
            f"其中 `{splits.get('test_date')}` 整日作为 sealed test，`{splits.get('validation_date')}` 整日作为 validation，"
            "更早日期作为 train；其余达到准入线但配置不同的日志只作为 OOD holdout，不混入主模型。",
            "",
            f"- train：{len(splits['train'])} 条，{split_summary['train']['model_ready_duration_s']:.1f} s",
            f"- validation：{len(splits['validation'])} 条，{split_summary['validation']['model_ready_duration_s']:.1f} s",
            f"- sealed test：{len(splits['sealed_test'])} 条，{split_summary['sealed_test']['model_ready_duration_s']:.1f} s",
            f"- OOD holdout：{len(splits['ood_holdout'])} 条，{split_summary['ood_holdout']['model_ready_duration_s']:.1f} s",
            "",
            "机器可读日志名单以 JSON 中的 `recommended_splits` 为准。若 Step 1 决定显式建模硬件/固件/参数条件，"
            "应另立数据契约并重新冻结 split，不能直接把 OOD holdout 并入训练。",
            "",
            "## Step 1 前建议",
            "",
            "1. 冻结主 cohort、整日志 split 和 exclusion reasons，sealed test 只保存清单，不生成训练统计。",
            "2. 明确 trajectory state、control、预测 horizon、frame 和 causal availability；特别决定是否使用估计风/空速。",
            "3. 在 canonical 生成前增加 reset/gap guard，并验证 actuator 命令语义在各 schema/配置组内一致。",
            "4. 先做无训练的数据窗口计数与覆盖检查；本 Step 0 不训练、不拟合归一化，也不构造未来标签。",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown(summary: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(summary), encoding="utf-8")
    return path
