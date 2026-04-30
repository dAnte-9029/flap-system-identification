from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyulog import ULog
from scipy.signal import savgol_filter

from system_identification.metadata import (
    load_aircraft_metadata,
    metadata_has_complete_labels,
    metadata_open_warnings,
    nested_value,
)
from system_identification.phase import (
    annotate_phase_cycles,
    compute_drive_phase_rad,
    compute_wing_stroke_angle_rad,
    compute_wing_stroke_direction,
    encoder_phase_from_counts,
)
from system_identification.resample import (
    bin_mean_resample,
    build_uniform_grid_us,
    ceil_to_step_us,
    floor_to_step_us,
    linear_resample,
    zoh_resample,
)


TARGET_RATE_HZ = 100.0
DT_US = int(1e6 / TARGET_RATE_HZ)

LOW_RATE_FRESHNESS_S = {
    "airspeed_validated": 0.2,
    "vehicle_air_data": 0.2,
    "wind": 0.3,
    "vehicle_status": 1.0,
    "vehicle_land_detected": 0.5,
    "control_allocator_status": 0.5,
    "sensor_gps": 0.5,
    "sensor_gnss_relative": 0.5,
}

PHASE_FRESHNESS_S = 0.1


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ulog_dataset(ulog: ULog, topic_name: str, multi_id: int = 0):
    for dataset in ulog.data_list:
        if dataset.name == topic_name and dataset.multi_id == multi_id:
            return dataset
    return None


def _decode_debug_vect_names(data: dict[str, np.ndarray]) -> list[str]:
    name_keys = sorted(
        (key for key in data.keys() if key.startswith("name[")),
        key=lambda key: int(key[5:-1]),
    )
    if not name_keys:
        return [""] * len(data["timestamp"])

    output: list[str] = []
    n = len(data[name_keys[0]])

    for idx in range(n):
        chars: list[str] = []
        for key in name_keys:
            value = int(data[key][idx])
            if value == 0:
                break
            chars.append(chr(value))
        output.append("".join(chars))

    return output


def topic_dataframe(
    ulog: ULog,
    topic_name: str,
    multi_id: int = 0,
    debug_name_filter: str | None = None,
) -> pd.DataFrame | None:
    dataset = _ulog_dataset(ulog, topic_name, multi_id=multi_id)
    if dataset is None:
        return None

    data = {key: np.asarray(value) for key, value in dataset.data.items()}
    frame = pd.DataFrame(data)

    if topic_name == "debug_vect":
        frame["name"] = _decode_debug_vect_names(data)
        if debug_name_filter is not None:
            frame = frame.loc[frame["name"] == debug_name_filter].reset_index(drop=True)
            if frame.empty:
                return None

    event_column = "timestamp_sample" if "timestamp_sample" in frame.columns else "timestamp"
    frame["event_time_us"] = frame[event_column].astype(np.int64)
    return frame.sort_values("event_time_us").reset_index(drop=True)


def _required_frame(topic_frames: dict[str, pd.DataFrame], topic_name: str) -> pd.DataFrame:
    frame = topic_frames.get(topic_name)
    if frame is None or frame.empty:
        raise ValueError(f"Required topic missing or empty: {topic_name}")
    return frame


def build_grid_from_topic_frames(topic_frames: dict[str, pd.DataFrame], dt_us: int = DT_US) -> np.ndarray:
    core_topics = [
        name
        for name in ["encoder_count", "actuator_motors", "actuator_servos"]
        if topic_frames.get(name) is not None and not topic_frames[name].empty
    ]

    if not core_topics:
        raise ValueError("No core topics available to build canonical grid")

    start_us = max(int(topic_frames[name]["event_time_us"].min()) for name in core_topics)
    end_us = min(int(topic_frames[name]["event_time_us"].max()) for name in core_topics)

    start_us = ceil_to_step_us(start_us, dt_us)
    end_us = floor_to_step_us(end_us, dt_us)
    return build_uniform_grid_us(start_us, end_us, dt_us)


def _resample_linear_columns(frame: pd.DataFrame, grid_us: np.ndarray, columns: list[str]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for column in columns:
        if column in frame.columns:
            output[column] = linear_resample(frame["event_time_us"].to_numpy(), frame[column].to_numpy(), grid_us)
    return output


def _resample_zoh_columns(
    frame: pd.DataFrame,
    grid_us: np.ndarray,
    columns: list[str],
    freshness_s: float,
    emit_missing_columns: bool = False,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for column in columns:
        if column in frame.columns:
            values, age_s, valid = zoh_resample(
                frame["event_time_us"].to_numpy(),
                frame[column].to_numpy(),
                grid_us,
                freshness_s=freshness_s,
            )
            output[column] = values
            output[f"{column}_age_s"] = age_s
            output[f"{column}_valid"] = valid
        elif emit_missing_columns:
            output[column] = np.full(len(grid_us), np.nan, dtype=float)
            output[f"{column}_age_s"] = np.full(len(grid_us), np.nan, dtype=float)
            output[f"{column}_valid"] = np.zeros(len(grid_us), dtype=bool)
    return output


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def _frame_columns_or_none(samples: pd.DataFrame, columns: list[str]) -> np.ndarray | None:
    if any(column not in samples.columns for column in columns):
        return None
    return samples[columns].to_numpy(dtype=float, copy=True)


def _rotation_body_to_world_from_quaternions(quaternions_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quaternions = np.asarray(quaternions_wxyz, dtype=float)
    rotation = np.full((len(quaternions), 3, 3), np.nan, dtype=float)

    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        return rotation, np.zeros(len(quaternions), dtype=bool)

    norms = np.linalg.norm(quaternions, axis=1)
    valid = np.isfinite(quaternions).all(axis=1) & (norms > 0.0)

    if not np.any(valid):
        return rotation, valid

    q = np.zeros_like(quaternions)
    q[valid] = quaternions[valid] / norms[valid, None]
    w, x, y, z = q.T

    rotation[valid, 0, 0] = 1.0 - 2.0 * (y[valid] ** 2 + z[valid] ** 2)
    rotation[valid, 0, 1] = 2.0 * (x[valid] * y[valid] - z[valid] * w[valid])
    rotation[valid, 0, 2] = 2.0 * (x[valid] * z[valid] + y[valid] * w[valid])
    rotation[valid, 1, 0] = 2.0 * (x[valid] * y[valid] + z[valid] * w[valid])
    rotation[valid, 1, 1] = 1.0 - 2.0 * (x[valid] ** 2 + z[valid] ** 2)
    rotation[valid, 1, 2] = 2.0 * (y[valid] * z[valid] - x[valid] * w[valid])
    rotation[valid, 2, 0] = 2.0 * (x[valid] * z[valid] - y[valid] * w[valid])
    rotation[valid, 2, 1] = 2.0 * (y[valid] * z[valid] + x[valid] * w[valid])
    rotation[valid, 2, 2] = 1.0 - 2.0 * (x[valid] ** 2 + y[valid] ** 2)

    return rotation, valid


def _odd_window_length(sample_period_s: float, window_s: float, sample_count: int) -> int:
    if sample_count < 3 or not np.isfinite(sample_period_s) or sample_period_s <= 0.0:
        return 0
    window = max(3, int(round(window_s / sample_period_s)))
    if window % 2 == 0:
        window += 1
    if window > sample_count:
        window = sample_count if sample_count % 2 == 1 else sample_count - 1
    return window if window >= 3 else 0


def _smoothed_derivative_for_group(
    time_s: np.ndarray,
    values: np.ndarray,
    *,
    window_s: float,
    polyorder: int,
) -> np.ndarray:
    derivative = np.full(len(values), np.nan, dtype=float)
    finite = np.isfinite(time_s) & np.isfinite(values)
    if int(finite.sum()) < 3:
        return derivative

    valid_index = np.flatnonzero(finite)
    valid_time = time_s[finite]
    valid_values = values[finite]
    order = np.argsort(valid_time)
    valid_index = valid_index[order]
    valid_time = valid_time[order]
    valid_values = valid_values[order]

    dt = np.diff(valid_time)
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if len(dt) == 0:
        return derivative
    sample_period_s = float(np.median(dt))
    window = _odd_window_length(sample_period_s, window_s, len(valid_values))
    if window <= polyorder:
        estimated = np.gradient(valid_values, valid_time)
    else:
        estimated = savgol_filter(
            valid_values,
            window_length=window,
            polyorder=min(polyorder, window - 1),
            deriv=1,
            delta=sample_period_s,
            mode="interp",
        )
    derivative[valid_index] = estimated
    return derivative


def compute_smoothed_kinematic_derivatives(
    samples: pd.DataFrame,
    *,
    group_column: str = "log_id",
    window_s: float = 0.12,
    polyorder: int = 2,
) -> pd.DataFrame:
    required_columns = [
        "time_s",
        "vehicle_local_position.vx",
        "vehicle_local_position.vy",
        "vehicle_local_position.vz",
        "vehicle_angular_velocity.xyz[0]",
        "vehicle_angular_velocity.xyz[1]",
        "vehicle_angular_velocity.xyz[2]",
    ]
    missing = [column for column in required_columns if column not in samples.columns]
    if missing:
        raise ValueError(f"Missing columns for smoothed derivatives: {missing}")

    derivative_specs = [
        ("vehicle_local_position.vx", "vehicle_local_position.ax_smooth"),
        ("vehicle_local_position.vy", "vehicle_local_position.ay_smooth"),
        ("vehicle_local_position.vz", "vehicle_local_position.az_smooth"),
        ("vehicle_angular_velocity.xyz[0]", "vehicle_angular_velocity.xyz_derivative_smooth[0]"),
        ("vehicle_angular_velocity.xyz[1]", "vehicle_angular_velocity.xyz_derivative_smooth[1]"),
        ("vehicle_angular_velocity.xyz[2]", "vehicle_angular_velocity.xyz_derivative_smooth[2]"),
    ]
    output = pd.DataFrame(index=samples.index)
    for _, output_column in derivative_specs:
        output[output_column] = np.nan

    groups = samples.groupby(group_column, sort=False) if group_column in samples.columns else [(None, samples)]
    for _, group in groups:
        time_s = group["time_s"].to_numpy(dtype=float)
        for input_column, output_column in derivative_specs:
            output.loc[group.index, output_column] = _smoothed_derivative_for_group(
                time_s,
                group[input_column].to_numpy(dtype=float),
                window_s=window_s,
                polyorder=polyorder,
            )

    return output


def _compute_effective_wrench_labels(
    samples: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    linear_acceleration_columns: list[str] | None = None,
    angular_acceleration_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_count = len(samples)
    nan_vectors = np.full((sample_count, 3), np.nan, dtype=float)
    label_valid = np.zeros(sample_count, dtype=bool)

    if not metadata_has_complete_labels(metadata):
        return nan_vectors.copy(), nan_vectors.copy(), label_valid

    mass_kg = nested_value(metadata, "mass_properties", "mass_kg")
    inertia_raw = nested_value(metadata, "mass_properties", "inertia_b_kg_m2")

    try:
        mass_kg = float(mass_kg)
        inertia_b = np.asarray(inertia_raw, dtype=float)
    except (TypeError, ValueError):
        return nan_vectors.copy(), nan_vectors.copy(), label_valid

    if not np.isfinite(mass_kg) or inertia_b.shape != (3, 3) or not np.isfinite(inertia_b).all():
        return nan_vectors.copy(), nan_vectors.copy(), label_valid

    resolved_linear_acceleration_columns = linear_acceleration_columns or [
        "vehicle_local_position.ax",
        "vehicle_local_position.ay",
        "vehicle_local_position.az",
    ]
    resolved_angular_acceleration_columns = angular_acceleration_columns or [
        "vehicle_angular_velocity.xyz_derivative[0]",
        "vehicle_angular_velocity.xyz_derivative[1]",
        "vehicle_angular_velocity.xyz_derivative[2]",
    ]

    acc_n = _frame_columns_or_none(samples, resolved_linear_acceleration_columns)
    quat_nb = _frame_columns_or_none(
        samples,
        [
            "vehicle_attitude.q[0]",
            "vehicle_attitude.q[1]",
            "vehicle_attitude.q[2]",
            "vehicle_attitude.q[3]",
        ],
    )
    omega_b = _frame_columns_or_none(
        samples,
        [
            "vehicle_angular_velocity.xyz[0]",
            "vehicle_angular_velocity.xyz[1]",
            "vehicle_angular_velocity.xyz[2]",
        ],
    )
    alpha_b = _frame_columns_or_none(samples, resolved_angular_acceleration_columns)

    if acc_n is None or quat_nb is None or omega_b is None or alpha_b is None:
        return nan_vectors.copy(), nan_vectors.copy(), label_valid

    rot_nb, quat_valid = _rotation_body_to_world_from_quaternions(quat_nb)
    gravity_n = np.array(
        [0.0, 0.0, _as_float(nested_value(metadata, "label_definition", "gravity_m_s2"), default=9.81)],
        dtype=float,
    )
    specific_acc_n = acc_n - gravity_n
    force_b = mass_kg * np.einsum("nji,nj->ni", rot_nb, specific_acc_n)

    angular_momentum_b = np.einsum("ij,nj->ni", inertia_b, omega_b)
    inertia_alpha_b = np.einsum("ij,nj->ni", inertia_b, alpha_b)
    moment_b = inertia_alpha_b + np.cross(omega_b, angular_momentum_b)

    finite_mask = (
        np.isfinite(acc_n).all(axis=1)
        & np.isfinite(quat_nb).all(axis=1)
        & np.isfinite(omega_b).all(axis=1)
        & np.isfinite(alpha_b).all(axis=1)
        & quat_valid
    )

    for column in [
        "vehicle_local_position.xy_valid",
        "vehicle_local_position.z_valid",
        "vehicle_local_position.v_xy_valid",
        "vehicle_local_position.v_z_valid",
    ]:
        if column in samples.columns:
            finite_mask &= samples[column].fillna(False).astype(bool).to_numpy()

    force_b[~finite_mask] = np.nan
    moment_b[~finite_mask] = np.nan
    return force_b, moment_b, finite_mask


def assemble_canonical_samples(
    grid_us: np.ndarray,
    topic_frames: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    samples = pd.DataFrame({"timestamp_us": grid_us})
    samples["time_s"] = (samples["timestamp_us"] - samples["timestamp_us"].iloc[0]) * 1e-6

    encoder_frame = _required_frame(topic_frames, "encoder_count")
    encoder_unwrapped, encoder_wrapped = encoder_phase_from_counts(
        total_count=encoder_frame["total_count"].to_numpy(),
        position_raw=encoder_frame["position_raw"].to_numpy(),
        encoder_counts_per_rev=_as_float(
            nested_value(metadata, "flapping_drive", "encoder_counts_per_rev"),
            default=4096.0,
        ),
    )

    encoder_resampled_wrapped = linear_resample(
        encoder_frame["event_time_us"].to_numpy(),
        encoder_wrapped,
        grid_us,
    )
    encoder_resampled_unwrapped = linear_resample(
        encoder_frame["event_time_us"].to_numpy(),
        encoder_unwrapped,
        grid_us,
    )

    encoder_to_drive_ratio = _as_float(
        nested_value(metadata, "flapping_drive", "encoder_to_drive_ratio"),
        default=1.0,
    )
    encoder_to_drive_sign = _as_float(
        nested_value(metadata, "flapping_drive", "encoder_to_drive_sign"),
        default=1.0,
    )
    drive_phase_zero_offset_rad = _as_float(
        nested_value(metadata, "flapping_drive", "drive_phase_zero_offset_rad"),
        default=0.0,
    )

    drive_unwrapped, drive_wrapped = compute_drive_phase_rad(
        encoder_phase_unwrapped_rad=encoder_resampled_unwrapped,
        encoder_to_drive_ratio=encoder_to_drive_ratio,
        encoder_to_drive_sign=encoder_to_drive_sign,
        drive_phase_zero_offset_rad=drive_phase_zero_offset_rad,
    )

    wing_stroke = compute_wing_stroke_angle_rad(
        drive_phase_rad=drive_wrapped,
        wing_stroke_amplitude_rad=_as_float(
            nested_value(metadata, "flapping_drive", "wing_stroke_amplitude_rad"),
            default=float(np.deg2rad(30.0)),
        ),
        wing_stroke_phase_offset_rad=_as_float(
            nested_value(metadata, "flapping_drive", "wing_stroke_phase_offset_rad"),
            default=0.0,
        ),
    )

    samples["encoder_phase_rad"] = encoder_resampled_wrapped
    samples["encoder_phase_unwrapped_rad"] = encoder_resampled_unwrapped
    samples["drive_phase_rad"] = drive_wrapped
    samples["drive_phase_unwrapped_rad"] = drive_unwrapped
    samples["drive_phase_sin"] = np.sin(drive_wrapped)
    samples["drive_phase_cos"] = np.cos(drive_wrapped)
    samples["wing_stroke_angle_rad"] = wing_stroke
    samples["wing_stroke_angle_deg"] = np.rad2deg(wing_stroke)
    samples["wing_stroke_direction"] = compute_wing_stroke_direction(
        drive_phase_rad=drive_wrapped,
        wing_stroke_phase_offset_rad=_as_float(
            nested_value(metadata, "flapping_drive", "wing_stroke_phase_offset_rad"),
            default=0.0,
        ),
    )
    samples["encoder_total_count"] = linear_resample(
        encoder_frame["event_time_us"].to_numpy(),
        encoder_frame["total_count"].to_numpy(),
        grid_us,
    )
    samples["encoder_position_raw"] = linear_resample(
        encoder_frame["event_time_us"].to_numpy(),
        encoder_frame["position_raw"].to_numpy(),
        grid_us,
    )

    phase_source = np.full(len(samples), "encoder_count_fallback", dtype=object)
    phase_raw = drive_wrapped.copy()
    phase_raw_unwrapped = drive_unwrapped.copy()
    phase_valid = np.isfinite(phase_raw)

    if "rpm" in topic_frames and topic_frames["rpm"] is not None:
        rpm_frame = topic_frames["rpm"]
        samples["encoder_rpm_raw"] = linear_resample(
            rpm_frame["event_time_us"].to_numpy(),
            rpm_frame["rpm_raw"].to_numpy(),
            grid_us,
        )
        samples["encoder_rpm_est"] = linear_resample(
            rpm_frame["event_time_us"].to_numpy(),
            rpm_frame["rpm_estimate"].to_numpy(),
            grid_us,
        )

    if "flap_frequency" in topic_frames and topic_frames["flap_frequency"] is not None:
        flap_frame = topic_frames["flap_frequency"]
        samples["flap_frequency_hz"] = linear_resample(
            flap_frame["event_time_us"].to_numpy(),
            flap_frame["frequency_hz"].to_numpy(),
            grid_us,
        )
    else:
        samples["flap_frequency_hz"] = np.nan

    wing_phase_frame = topic_frames.get("wing_phase")
    if wing_phase_frame is not None and not wing_phase_frame.empty:
        wing_phase_raw, wing_phase_age_s, wing_phase_topic_valid = zoh_resample(
            wing_phase_frame["event_time_us"].to_numpy(),
            wing_phase_frame["phase_rad"].to_numpy(),
            grid_us,
            freshness_s=PHASE_FRESHNESS_S,
        )
        wing_phase_unwrapped, _, _ = zoh_resample(
            wing_phase_frame["event_time_us"].to_numpy(),
            wing_phase_frame["phase_unwrapped_rad"].to_numpy(),
            grid_us,
            freshness_s=PHASE_FRESHNESS_S,
        )
        wing_phase_valid_raw, _, _ = zoh_resample(
            wing_phase_frame["event_time_us"].to_numpy(),
            wing_phase_frame["phase_valid"].to_numpy(),
            grid_us,
            freshness_s=PHASE_FRESHNESS_S,
        )

        phase_source[:] = "wing_phase"
        phase_raw = wing_phase_raw
        phase_raw_unwrapped = wing_phase_unwrapped
        phase_valid = wing_phase_topic_valid & np.isfinite(phase_raw) & (wing_phase_valid_raw > 0.5)

        samples["wing_phase.phase_rad"] = wing_phase_raw
        samples["wing_phase.phase_unwrapped_rad"] = wing_phase_unwrapped
        samples["wing_phase.phase_age_s"] = wing_phase_age_s
        samples["wing_phase.phase_valid"] = phase_valid

    samples["phase_source"] = phase_source
    samples["phase_raw_rad"] = phase_raw
    samples["phase_raw_unwrapped_rad"] = phase_raw_unwrapped
    samples["phase_valid"] = phase_valid

    phase_annotations = annotate_phase_cycles(
        time_s=samples["time_s"].to_numpy(),
        phase_rad=phase_raw,
        flap_frequency_hz=samples["flap_frequency_hz"].to_numpy(),
        phase_valid=phase_valid,
    )
    for column, values in phase_annotations.items():
        samples[column] = values

    if "debug_vect" in topic_frames and topic_frames["debug_vect"] is not None:
        debug_frame = topic_frames["debug_vect"]
        samples["debug_angle_rad"] = linear_resample(
            debug_frame["event_time_us"].to_numpy(),
            debug_frame["x"].to_numpy(),
            grid_us,
        )
        samples["debug_angle_error_rad"] = samples["debug_angle_rad"] - samples["encoder_phase_rad"]

    motors_frame = _required_frame(topic_frames, "actuator_motors")
    servos_frame = _required_frame(topic_frames, "actuator_servos")
    samples["motor_cmd_0"] = bin_mean_resample(
        motors_frame["event_time_us"].to_numpy(),
        motors_frame["control[0]"].to_numpy(),
        grid_us,
        dt_us=DT_US,
    )
    samples["servo_left_elevon"] = bin_mean_resample(
        servos_frame["event_time_us"].to_numpy(),
        servos_frame["control[0]"].to_numpy(),
        grid_us,
        dt_us=DT_US,
    )
    samples["servo_right_elevon"] = bin_mean_resample(
        servos_frame["event_time_us"].to_numpy(),
        servos_frame["control[1]"].to_numpy(),
        grid_us,
        dt_us=DT_US,
    )
    samples["servo_rudder"] = bin_mean_resample(
        servos_frame["event_time_us"].to_numpy(),
        servos_frame["control[2]"].to_numpy(),
        grid_us,
        dt_us=DT_US,
    )

    linear_topics = {
        "vehicle_local_position": ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "heading"],
        "vehicle_angular_velocity": [
            "xyz[0]",
            "xyz[1]",
            "xyz[2]",
            "xyz_derivative[0]",
            "xyz_derivative[1]",
            "xyz_derivative[2]",
        ],
        "vehicle_attitude": ["q[0]", "q[1]", "q[2]", "q[3]"],
    }

    for topic_name, columns in linear_topics.items():
        frame = topic_frames.get(topic_name)
        if frame is None:
            continue
        for column, values in _resample_linear_columns(frame, grid_us, columns).items():
            samples[f"{topic_name}.{column}"] = values

    zoh_topics = {
        "vehicle_local_position": ["xy_valid", "z_valid", "v_xy_valid", "v_z_valid"],
        "airspeed_validated": [
            "indicated_airspeed_m_s",
            "calibrated_airspeed_m_s",
            "true_airspeed_m_s",
            "calibrated_ground_minus_wind_m_s",
            "true_ground_minus_wind_m_s",
            "airspeed_derivative_filtered",
            "throttle_filtered",
            "pitch_filtered",
            "airspeed_source",
        ],
        "vehicle_air_data": ["rho"],
        "wind": ["windspeed_north", "windspeed_east"],
        "vehicle_status": ["arming_state", "nav_state"],
        "vehicle_land_detected": ["landed"],
        "control_allocator_status": [
            "torque_setpoint_achieved",
            "thrust_setpoint_achieved",
            "actuator_saturation[0]",
        ],
        "sensor_gps": ["fix_type"],
        "sensor_gnss_relative": ["relative_position_valid", "heading_valid"],
    }

    for topic_name, columns in zoh_topics.items():
        frame = topic_frames.get(topic_name)
        if frame is None:
            continue
        freshness = LOW_RATE_FRESHNESS_S.get(topic_name, 0.5)
        for column, values in _resample_zoh_columns(
            frame,
            grid_us,
            columns,
            freshness,
            emit_missing_columns=topic_name == "airspeed_validated",
        ).items():
            samples[f"{topic_name}.{column}"] = values

    force_b, moment_b, label_valid = _compute_effective_wrench_labels(samples=samples, metadata=metadata)
    samples["fx_b"] = force_b[:, 0]
    samples["fy_b"] = force_b[:, 1]
    samples["fz_b"] = force_b[:, 2]
    samples["mx_b"] = moment_b[:, 0]
    samples["my_b"] = moment_b[:, 1]
    samples["mz_b"] = moment_b[:, 2]
    samples["label_valid"] = label_valid

    nav_state_column = "vehicle_status.nav_state"
    armed_column = "vehicle_status.arming_state"
    landed_column = "vehicle_land_detected.landed"
    break_mask = np.zeros(len(samples), dtype=bool)

    for column in [nav_state_column, armed_column, landed_column]:
        if column in samples.columns:
            values = samples[column].ffill()
            break_mask[1:] |= values.to_numpy()[1:] != values.to_numpy()[:-1]

    samples["segment_id"] = np.cumsum(break_mask.astype(int))
    return samples


def build_segments(samples: pd.DataFrame) -> pd.DataFrame:
    segments = []

    for segment_id, group in samples.groupby("segment_id", sort=True):
        row = {
            "segment_id": int(segment_id),
            "start_time_us": int(group["timestamp_us"].iloc[0]),
            "end_time_us": int(group["timestamp_us"].iloc[-1]),
            "duration_s": float((group["timestamp_us"].iloc[-1] - group["timestamp_us"].iloc[0]) * 1e-6),
            "sample_count": int(len(group)),
            "label_valid_ratio": float(group["label_valid"].mean()) if "label_valid" in group else 0.0,
        }
        if "cycle_valid" in group.columns:
            row["cycle_valid_ratio"] = float(group["cycle_valid"].mean())
        if "flap_active" in group.columns:
            row["flap_active_ratio"] = float(group["flap_active"].mean())
        if "vehicle_status.nav_state" in group.columns:
            row["nav_state"] = float(group["vehicle_status.nav_state"].mode(dropna=True).iloc[0])
        segments.append(row)

    return pd.DataFrame(segments)


def extract_topic_frames_from_ulog(ulog_path: str | Path) -> dict[str, pd.DataFrame]:
    ulog = ULog(str(ulog_path))
    return {
        "encoder_count": topic_dataframe(ulog, "encoder_count"),
        "rpm": topic_dataframe(ulog, "rpm"),
        "debug_vect": topic_dataframe(ulog, "debug_vect", debug_name_filter="AS5600ANG"),
        "flap_frequency": topic_dataframe(ulog, "flap_frequency"),
        "actuator_motors": topic_dataframe(ulog, "actuator_motors"),
        "actuator_servos": topic_dataframe(ulog, "actuator_servos"),
        "vehicle_local_position": topic_dataframe(ulog, "vehicle_local_position"),
        "vehicle_attitude": topic_dataframe(ulog, "vehicle_attitude"),
        "vehicle_angular_velocity": topic_dataframe(ulog, "vehicle_angular_velocity"),
        "airspeed_validated": topic_dataframe(ulog, "airspeed_validated"),
        "vehicle_air_data": topic_dataframe(ulog, "vehicle_air_data"),
        "wind": topic_dataframe(ulog, "wind"),
        "vehicle_status": topic_dataframe(ulog, "vehicle_status"),
        "vehicle_land_detected": topic_dataframe(ulog, "vehicle_land_detected"),
        "control_allocator_status": topic_dataframe(ulog, "control_allocator_status"),
        "sensor_gps": topic_dataframe(ulog, "sensor_gps"),
        "sensor_gnss_relative": topic_dataframe(ulog, "sensor_gnss_relative"),
        "wing_phase": topic_dataframe(ulog, "wing_phase"),
    }


def _log_id_from_path(ulg_path: str | Path) -> str:
    return Path(ulg_path).stem


def run_ulog_to_canonical(
    ulg_path: str | Path,
    metadata_path: str | Path,
    output_root: str | Path,
    rate_hz: float = TARGET_RATE_HZ,
) -> dict[str, Path]:
    if int(1e6 / rate_hz) != DT_US:
        raise ValueError("This first-pass pipeline currently supports only 100 Hz")

    metadata = load_aircraft_metadata(metadata_path)
    topic_frames = extract_topic_frames_from_ulog(ulg_path)
    grid_us = build_grid_from_topic_frames(topic_frames, dt_us=DT_US)
    samples = assemble_canonical_samples(grid_us=grid_us, topic_frames=topic_frames, metadata=metadata)
    segments = build_segments(samples)

    aircraft_id = metadata.get("aircraft_id", "unknown_aircraft")
    log_id = _log_id_from_path(ulg_path)
    output_dir = Path(output_root) / f"aircraft_id={aircraft_id}" / f"log_id={log_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_path = output_dir / "samples.parquet"
    segments_path = output_dir / "segments.parquet"
    manifest_path = output_dir / "source_manifest.json"
    report_path = output_dir / "preprocessing_report.json"

    samples.to_parquet(samples_path, index=False)
    segments.to_parquet(segments_path, index=False)

    manifest = {
        "pipeline_version": "ulog_to_canonical_v0.2_preliminary",
        "source_log_path": str(Path(ulg_path).resolve()),
        "source_log_sha256": _sha256_file(ulg_path),
        "source_log_size_bytes": Path(ulg_path).stat().st_size,
        "aircraft_metadata_path": str(Path(metadata_path).resolve()),
        "aircraft_metadata_sha256": _sha256_file(metadata_path),
        "aircraft_id": aircraft_id,
        "log_id": log_id,
        "rate_hz": rate_hz,
    }

    report = {
        "warning_count": len(metadata_open_warnings(metadata)),
        "warnings": metadata_open_warnings(metadata),
        "sample_count": int(len(samples)),
        "segment_count": int(len(segments)),
        "topics_present": sorted([name for name, frame in topic_frames.items() if frame is not None]),
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "output_dir": output_dir,
        "samples_path": samples_path,
        "segments_path": segments_path,
        "manifest_path": manifest_path,
        "report_path": report_path,
    }
