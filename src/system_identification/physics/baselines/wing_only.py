"""Offline wing-only six-axis DeLaurier baseline for canonical flight logs.

Pure physics is synchronized from dAnte-9029/IsaacLab ``flapping_rl`` commit
``3b5d4ec1d28f1384cf042402992ad7ea59995f49``. No Isaac, PhysX, Torch, or
simulation package is imported at runtime.

Frame/reference contract
------------------------
Canonical inputs and final outputs use body FRD at the IMU origin (force) or
the real-aircraft CG (moment). DeLaurier strip loads use the internal Wang
frame. Frozen Isaac URDF wing-link transforms start in FLU and are converted
once to FRD. Forces are polar vectors and moments are axial vectors. Returned
``pred_m*_b`` values are about the real-aircraft CG, never the wing root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from system_identification.physics import (
    DeLaurierParams,
    compute_delaurier_dynamic_twist,
    compute_delaurier_strip_loads,
    integrate_delaurier_strip_wrench,
    load_wing_geometry_csv,
    map_canonical_phase_to_delaurier,
    reconstruct_body_airflow_from_ned,
    transform_wrench,
    translate_wrench_moment,
)

ISAACLAB_SOURCE_REPOSITORY = "https://github.com/dAnte-9029/IsaacLab"
ISAACLAB_SOURCE_BRANCH = "flapping_rl"
ISAACLAB_SOURCE_COMMIT = "3b5d4ec1d28f1384cf042402992ad7ea59995f49"

TARGETS = ("fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b")
FORCE_AXES = ("fx_b", "fy_b", "fz_b")
MOMENT_AXES = ("mx_b", "my_b", "mz_b")
AIRFLOW_MODES = frozenset({"legacy_scalar_true_airspeed", "attitude_ground_wind_3d"})
ATTITUDE_AIRFLOW_REQUIRED_COLUMNS = (
    "vehicle_attitude.q[0]",
    "vehicle_attitude.q[1]",
    "vehicle_attitude.q[2]",
    "vehicle_attitude.q[3]",
    "vehicle_local_position.vx",
    "vehicle_local_position.vy",
    "vehicle_local_position.vz",
    "wind.windspeed_north",
    "wind.windspeed_east",
)


@dataclass(frozen=True)
class WingOnlyBaselineConfig:
    """Frozen physics parameters plus explicit real-aircraft reference data."""

    num_strips: int = 80
    stroke_amplitude_rad: float = 0.5235987756
    minimum_airspeed_m_s: float = 0.5
    mean_pitch_offset_rad: float = 0.0
    airflow_mode: str = "legacy_scalar_true_airspeed"
    params: DeLaurierParams = field(
        default_factory=lambda: DeLaurierParams(
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
    )
    # Positions are body FRD relative to the canonical IMU origin.
    left_wing_root_b_m: tuple[float, float, float] = (0.0, -0.056, -0.030)
    right_wing_root_b_m: tuple[float, float, float] = (0.0, 0.056, -0.030)
    aircraft_cg_b_m: tuple[float, float, float] = (-0.12154, 0.00541, -0.04298)
    # Fixed URDF roll followed by mirrored commanded joint motion.
    left_wing_fixed_roll_rad: float = -0.019391
    right_wing_fixed_roll_rad: float = 0.019391
    chunk_size: int = 4096


def baseline_config_from_aircraft_metadata(
    metadata_path: str | Path,
    *,
    chunk_size: int = 4096,
    airflow_mode: str = "legacy_scalar_true_airspeed",
) -> WingOnlyBaselineConfig:
    """Build the real-aircraft reference contract from canonical YAML metadata."""

    source = Path(metadata_path)
    with source.open(encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    frames = metadata.get("frames", {})
    if frames.get("body_frame") != "FRD":
        raise ValueError(f"Wing baseline requires metadata body_frame=FRD: {source}")
    if frames.get("body_reference_origin") != "imu_origin":
        raise ValueError(f"Wing baseline requires body_reference_origin=imu_origin: {source}")
    if frames.get("cg_reference_origin") != "imu_origin":
        raise ValueError(f"Wing baseline requires cg_reference_origin=imu_origin: {source}")
    if metadata.get("label_definition", {}).get("moment_definition") != "effective_external_moment_about_cg":
        raise ValueError(f"Wing baseline requires moment labels about the aircraft CG: {source}")
    try:
        cg = tuple(float(value) for value in metadata["mass_properties"]["cg_b_m"]["value"])
        amplitude = float(metadata["flapping_drive"]["wing_stroke_amplitude_rad"]["value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing numeric CG or wing stroke amplitude in {source}") from exc
    if len(cg) != 3 or not np.all(np.isfinite(cg)):
        raise ValueError(f"mass_properties.cg_b_m.value must contain three finite values: {source}")
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError(f"wing_stroke_amplitude_rad.value must be positive: {source}")
    if airflow_mode not in AIRFLOW_MODES:
        raise ValueError(f"Unsupported airflow_mode {airflow_mode!r}; expected one of {sorted(AIRFLOW_MODES)}")
    return WingOnlyBaselineConfig(
        stroke_amplitude_rad=amplitude,
        aircraft_cg_b_m=cg,
        chunk_size=int(chunk_size),
        airflow_mode=airflow_mode,
    )


def required_columns_for_airflow_mode(airflow_mode: str) -> set[str]:
    """Return canonical columns required by one explicit airflow contract."""

    if airflow_mode == "legacy_scalar_true_airspeed":
        return {"airspeed_validated.true_airspeed_m_s"}
    if airflow_mode == "attitude_ground_wind_3d":
        return set(ATTITUDE_AIRFLOW_REQUIRED_COLUMNS)
    raise ValueError(f"Unsupported airflow_mode {airflow_mode!r}; expected one of {sorted(AIRFLOW_MODES)}")


def _resolve_airflow_inputs(
    frame: pd.DataFrame,
    config: WingOnlyBaselineConfig,
) -> dict[str, np.ndarray]:
    mode = str(config.airflow_mode)
    if mode == "legacy_scalar_true_airspeed":
        incidence = (
            frame["airspeed_validated.pitch_filtered"].to_numpy(dtype=float)
            if "airspeed_validated.pitch_filtered" in frame.columns
            else np.zeros(len(frame), dtype=float)
        )
        true_airspeed = frame["airspeed_validated.true_airspeed_m_s"].to_numpy(dtype=float)
        forward = np.maximum(true_airspeed, float(config.minimum_airspeed_m_s))
        body_velocity = np.column_stack(
            (forward, np.zeros(len(frame), dtype=float), forward * np.tan(incidence))
        )
        return {
            "body_velocity": body_velocity,
            "speed": np.linalg.norm(body_velocity, axis=1),
            "incidence": incidence,
            "sideslip": np.zeros(len(frame), dtype=float),
            "forward_speed": forward,
            "attitude_pitch": np.full(len(frame), np.nan),
        }
    if mode != "attitude_ground_wind_3d":
        raise ValueError(f"Unsupported airflow_mode {mode!r}; expected one of {sorted(AIRFLOW_MODES)}")
    ground_velocity_ned = frame[
        ["vehicle_local_position.vx", "vehicle_local_position.vy", "vehicle_local_position.vz"]
    ].to_numpy(dtype=float)
    wind_velocity_ned = np.column_stack(
        (
            frame["wind.windspeed_north"].to_numpy(dtype=float),
            frame["wind.windspeed_east"].to_numpy(dtype=float),
            np.zeros(len(frame), dtype=float),
        )
    )
    quaternion = frame[[f"vehicle_attitude.q[{index}]" for index in range(4)]].to_numpy(dtype=float)
    airflow = reconstruct_body_airflow_from_ned(
        ground_velocity_ned_m_s=ground_velocity_ned,
        wind_velocity_ned_m_s=wind_velocity_ned,
        quaternion_body_to_ned_wxyz=quaternion,
    )
    pitch = np.arcsin(np.clip(-airflow.rotation_body_to_ned[:, 2, 0], -1.0, 1.0))
    return {
        "body_velocity": airflow.velocity_body_frd_m_s,
        "speed": airflow.speed_m_s,
        "incidence": airflow.alpha_rad,
        "sideslip": airflow.beta_rad,
        "forward_speed": np.maximum(
            airflow.velocity_body_frd_m_s[:, 0], float(config.minimum_airspeed_m_s)
        ),
        "attitude_pitch": pitch,
    }


def _rotation_x(angle_rad: np.ndarray) -> np.ndarray:
    angle = np.asarray(angle_rad, dtype=float)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    result = np.zeros((*angle.shape, 3, 3), dtype=float)
    result[..., 0, 0] = 1.0
    result[..., 1, 1] = cosine
    result[..., 1, 2] = -sine
    result[..., 2, 1] = sine
    result[..., 2, 2] = cosine
    return result


def _wing_polar_transforms_frd(q_rad: np.ndarray, config: WingOnlyBaselineConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return Wang-to-body-FRD polar transforms for left and right wings."""

    # Frozen Wang->link maps from straight_flight_env.py. Right is a reflection.
    wang_to_left_link = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    wang_to_right_link = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    flu_to_frd = np.diag([1.0, -1.0, -1.0])
    left_link_to_flu = _rotation_x(config.left_wing_fixed_roll_rad + q_rad)
    right_link_to_flu = _rotation_x(config.right_wing_fixed_roll_rad - q_rad)
    left = np.einsum(
        "ij,njk,kl->nil",
        flu_to_frd,
        left_link_to_flu,
        wang_to_left_link,
    )
    right = np.einsum(
        "ij,njk,kl->nil",
        flu_to_frd,
        right_link_to_flu,
        wang_to_right_link,
    )
    return left, right


def _phase_acceleration(
    segment: pd.DataFrame,
    phase_rate_rad_s: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    if mode == "constant_frequency_step":
        return np.zeros_like(phase_rate_rad_s)
    if mode != "frequency_derivative_experimental":
        raise ValueError(f"Unsupported phase acceleration mode: {mode}")
    time_s = segment["time_s"].to_numpy(dtype=float)
    if len(time_s) < 3 or np.any(np.diff(time_s) <= 0.0):
        return np.full_like(phase_rate_rad_s, np.nan)
    return np.gradient(phase_rate_rad_s, time_s, edge_order=2)


def _component_wrench_in_body(
    force_wang: np.ndarray,
    moment_wang: np.ndarray,
    *,
    left_transform: np.ndarray,
    right_transform: np.ndarray,
    config: WingOnlyBaselineConfig,
    translate_force_moment: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_force, left_moment = transform_wrench(force_wang, moment_wang, left_transform)
    right_force, right_moment = transform_wrench(force_wang, moment_wang, right_transform)
    if translate_force_moment:
        left_moment = translate_wrench_moment(
            left_force,
            left_moment,
            np.asarray(config.left_wing_root_b_m),
            np.asarray(config.aircraft_cg_b_m),
        )
        right_moment = translate_wrench_moment(
            right_force,
            right_moment,
            np.asarray(config.right_wing_root_b_m),
            np.asarray(config.aircraft_cg_b_m),
        )
    return left_force, left_moment, right_force, right_moment


def _chunk_result(
    frame: pd.DataFrame,
    *,
    theta_tip_deg: float,
    geometry_path: Path,
    config: WingOnlyBaselineConfig,
    phase_acceleration_mode: str,
    phase_acceleration_rad_s2: np.ndarray,
    spanwise_regions: Mapping[str, tuple[float, float]] | None = None,
    include_detailed_diagnostics: bool = False,
) -> pd.DataFrame:
    geometry = load_wing_geometry_csv(
        geometry_path,
        num_strips=config.num_strips,
        d_hat=0.0,
    )
    phase_canonical = frame["mechanical_phase_rad"].to_numpy(dtype=float)
    phase_delaurier = map_canonical_phase_to_delaurier(phase_canonical)
    frequency_hz = frame["flap_frequency_hz"].to_numpy(dtype=float)
    phase_rate = 2.0 * np.pi * frequency_hz
    amplitude = float(config.stroke_amplitude_rad)
    q = amplitude * np.cos(phase_delaurier)
    q_dot = -amplitude * np.sin(phase_delaurier) * phase_rate
    q_ddot = -amplitude * (
        np.cos(phase_delaurier) * np.square(phase_rate)
        + np.sin(phase_delaurier) * phase_acceleration_rad_s2
    )
    span = geometry.x_mid[None, :]
    h = -q[:, None] * span
    hdot = -q_dot[:, None] * span
    hddot = -q_ddot[:, None] * span

    airflow = _resolve_airflow_inputs(frame, config)
    incidence = airflow["incidence"]
    theta_bar = incidence + float(config.mean_pitch_offset_rad)
    twist = compute_delaurier_dynamic_twist(
        strip_span_m=geometry.x_mid,
        strip_width_m=geometry.dx,
        semi_span_m=geometry.semi_span_m,
        mean_pitch_rad=theta_bar,
        tip_twist_amplitude_rad=math.radians(float(theta_tip_deg)),
        phase_rad=phase_delaurier,
        phase_rate_rad_s=phase_rate,
        phase_acceleration_rad_s2=phase_acceleration_rad_s2,
        enabled=True,
    )
    airspeed = airflow["forward_speed"]
    rho = frame["vehicle_air_data.rho"].to_numpy(dtype=float)
    loads = compute_delaurier_strip_loads(
        h,
        hdot,
        hddot,
        twist.theta,
        twist.theta_dot,
        twist.theta_ddot,
        geometry,
        rho,
        airspeed,
        theta_a=incidence,
        theta_bar=theta_bar,
        omega_ref_rad_s=phase_rate,
        params=config.params,
        enable_separation=False,
    )
    wrench = integrate_delaurier_strip_wrench(loads)
    left_transform, right_transform = _wing_polar_transforms_frd(q, config)
    left_force, left_moment, right_force, right_moment = _component_wrench_in_body(
        wrench.force_wang,
        wrench.moment_wang_about_wing_origin,
        left_transform=left_transform,
        right_transform=right_transform,
        config=config,
        translate_force_moment=True,
    )
    total_force = left_force + right_force
    total_moment = left_moment + right_moment
    result = pd.DataFrame(index=frame.index)
    result["theta_tip_deg"] = float(theta_tip_deg)
    result["dynamic_twist_mode"] = (
        "disabled" if math.isclose(float(theta_tip_deg), 0.0, abs_tol=1.0e-12) else "delaurier_linear_spanwise"
    )
    result["baseline_filter_mode"] = "baseline_raw"
    result["phase_acceleration_mode"] = phase_acceleration_mode
    result["airflow_mode"] = str(config.airflow_mode)
    result["airflow_body_u_m_s"] = airflow["body_velocity"][:, 0]
    result["airflow_body_v_m_s"] = airflow["body_velocity"][:, 1]
    result["airflow_body_w_m_s"] = airflow["body_velocity"][:, 2]
    result["airflow_speed_m_s"] = airflow["speed"]
    result["airflow_forward_speed_used_m_s"] = airspeed
    result["airflow_alpha_rad"] = incidence
    result["airflow_beta_rad"] = airflow["sideslip"]
    result["attitude_pitch_rad"] = airflow["attitude_pitch"]
    result["phase_delaurier_rad"] = phase_delaurier
    result["phase_rate_rad_s"] = phase_rate
    result["phase_acceleration_rad_s2"] = phase_acceleration_rad_s2
    result["stroke_q_rad"] = q
    result["stroke_q_dot_rad_s"] = q_dot
    result["stroke_q_ddot_rad_s2"] = q_ddot
    result["theta_mean_rad"] = np.mean(twist.theta, axis=1)
    result["theta_dot_mean_rad_s"] = np.mean(twist.theta_dot, axis=1)
    result["theta_ddot_mean_rad_s2"] = np.mean(twist.theta_ddot, axis=1)
    result["theta_outer_strip_rad"] = twist.theta[:, -1]
    result["theta_outer_strip_span_fraction"] = twist.span_fraction[:, -1]
    for index, axis in enumerate(FORCE_AXES):
        result[f"pred_{axis}"] = total_force[:, index]
        result[f"pred_left_{axis}"] = left_force[:, index]
        result[f"pred_right_{axis}"] = right_force[:, index]
    for index, axis in enumerate(MOMENT_AXES):
        result[f"pred_{axis}"] = total_moment[:, index]
        result[f"pred_left_{axis}"] = left_moment[:, index]
        result[f"pred_right_{axis}"] = right_moment[:, index]

    force_components = {
        "dN_c": wrench.force_from_dN_c_wang,
        "dN_a": wrench.force_from_dN_a_wang,
        "dT_s": wrench.force_from_dT_s_wang,
        "dD_camber": wrench.force_from_dD_camber_wang,
        "dD_f": wrench.force_from_dD_f_wang,
    }
    moment_components = {
        "dN_c": wrench.moment_from_dN_c_wang,
        "dN_a": wrench.moment_from_dN_a_wang,
        "dT_s": wrench.moment_from_dT_s_wang,
        "dD_camber": wrench.moment_from_dD_camber_wang,
        "dD_f": wrench.moment_from_dD_f_wang,
    }
    diagnostic_columns: dict[str, np.ndarray] = {}
    for name, force_wang in force_components.items():
        lf, lm, rf, rm = _component_wrench_in_body(
            force_wang,
            moment_components[name],
            left_transform=left_transform,
            right_transform=right_transform,
            config=config,
            translate_force_moment=True,
        )
        for index, axis in enumerate(FORCE_AXES):
            diagnostic_columns[f"component_{name}_{axis}"] = (lf + rf)[:, index]
            if include_detailed_diagnostics:
                diagnostic_columns[f"component_{name}_left_{axis}"] = lf[:, index]
                diagnostic_columns[f"component_{name}_right_{axis}"] = rf[:, index]
        for index, axis in enumerate(MOMENT_AXES):
            diagnostic_columns[f"component_{name}_{axis}"] = (lm + rm)[:, index]

    zero_force = np.zeros_like(wrench.force_wang)
    free_moments = {
        "dM_ac": wrench.moment_from_dM_ac_wang,
        "dM_a": wrench.moment_from_dM_a_wang,
    }
    free_total = np.zeros_like(total_moment)
    for name, moment_wang in free_moments.items():
        _lf, lm, _rf, rm = _component_wrench_in_body(
            zero_force,
            moment_wang,
            left_transform=left_transform,
            right_transform=right_transform,
            config=config,
            translate_force_moment=False,
        )
        component = lm + rm
        free_total += component
        for index, axis in enumerate(MOMENT_AXES):
            diagnostic_columns[f"component_{name}_{axis}"] = component[:, index]
    force_arm_total = total_moment - free_total
    for index, axis in enumerate(MOMENT_AXES):
        diagnostic_columns[f"component_r_cross_f_{axis}"] = force_arm_total[:, index]
        diagnostic_columns[f"component_free_couple_{axis}"] = free_total[:, index]
    for name in ("dN_c", "dN_a", "dT_s", "dD_camber", "dD_f", "dM_ac", "dM_a"):
        values = getattr(loads, name)
        diagnostic_columns[f"strip_sum_{name}"] = np.sum(values, axis=1)
    diagnostic_columns["separation_ratio"] = np.average(
        loads.separation_weight,
        axis=1,
        weights=loads.chord * loads.strip_width,
    )
    if spanwise_regions:
        span_fraction = geometry.x_mid / geometry.semi_span_m
        strip_force_components = {
            "dN_c": np.stack((np.zeros_like(loads.dN_c), loads.dN_c, np.zeros_like(loads.dN_c)), axis=-1),
            "dN_a": np.stack((np.zeros_like(loads.dN_a), loads.dN_a, np.zeros_like(loads.dN_a)), axis=-1),
            "dT_s": np.stack((np.zeros_like(loads.dT_s), np.zeros_like(loads.dT_s), loads.dT_s), axis=-1),
            "dD_camber": np.stack(
                (np.zeros_like(loads.dD_camber), np.zeros_like(loads.dD_camber), -loads.dD_camber), axis=-1
            ),
            "dD_f": np.stack((np.zeros_like(loads.dD_f), np.zeros_like(loads.dD_f), -loads.dD_f), axis=-1),
        }
        zero_moment = np.zeros_like(wrench.moment_wang_about_wing_origin)
        for region_name, (lower, upper) in spanwise_regions.items():
            if not 0.0 <= float(lower) < float(upper) <= 1.0:
                raise ValueError(f"Invalid spanwise region {region_name!r}: {(lower, upper)}")
            region_mask = (span_fraction >= float(lower)) & (
                span_fraction <= float(upper) if math.isclose(float(upper), 1.0) else span_fraction < float(upper)
            )
            if not region_mask.any():
                raise ValueError(f"Spanwise region {region_name!r} contains no strips")
            region_total = np.zeros_like(total_force)
            for name, strip_force in strip_force_components.items():
                force_wang = np.sum(strip_force[:, region_mask, :], axis=1)
                left_region, _lm, right_region, _rm = _component_wrench_in_body(
                    force_wang,
                    zero_moment,
                    left_transform=left_transform,
                    right_transform=right_transform,
                    config=config,
                    translate_force_moment=False,
                )
                component_region = left_region + right_region
                region_total += component_region
                for index, axis in enumerate(FORCE_AXES):
                    diagnostic_columns[f"span_{region_name}_component_{name}_{axis}"] = component_region[:, index]
            for index, axis in enumerate(FORCE_AXES):
                diagnostic_columns[f"span_{region_name}_pred_{axis}"] = region_total[:, index]
    return pd.concat([result, pd.DataFrame(diagnostic_columns, index=result.index)], axis=1)


def evaluate_wing_only_delaurier_segment(
    segment: pd.DataFrame,
    *,
    theta_tip_deg: Iterable[float],
    geometry_path: str | Path,
    config: WingOnlyBaselineConfig | None = None,
    phase_acceleration_mode: str = "constant_frequency_step",
    spanwise_regions: Mapping[str, tuple[float, float]] | None = None,
    include_detailed_diagnostics: bool = False,
) -> pd.DataFrame:
    """Evaluate a complete canonical ``log_id + segment_id`` in long format.

    The caller must pass the full segment. Chunking only limits memory and does
    not reset phase, derivatives, or filters. One output row is produced for
    every input row and theta-tip value.
    """

    resolved = config or WingOnlyBaselineConfig()
    required = {
        "mechanical_phase_rad",
        "flap_frequency_hz",
        "vehicle_air_data.rho",
        *required_columns_for_airflow_mode(str(resolved.airflow_mode)),
    }
    missing = sorted(required - set(segment.columns))
    if missing:
        raise ValueError(f"Missing canonical baseline inputs: {missing}")
    if "log_id" in segment.columns and segment["log_id"].nunique(dropna=False) > 1:
        raise ValueError("evaluate_wing_only_delaurier_segment cannot cross log_id")
    if "segment_id" in segment.columns and segment["segment_id"].nunique(dropna=False) > 1:
        raise ValueError("evaluate_wing_only_delaurier_segment cannot cross segment_id")
    ordered = segment.sort_values("time_s", kind="stable").copy()
    phase_rate = 2.0 * np.pi * ordered["flap_frequency_hz"].to_numpy(dtype=float)
    acceleration = _phase_acceleration(ordered, phase_rate, mode=phase_acceleration_mode)
    meta_columns = [
        column
        for column in (
            "dataset_id",
            "log_id",
            "segment_id",
            "time_s",
            "timestamp_us",
            "mechanical_phase_rad",
            "phase_corrected_rad",
            "phase_valid",
            "cycle_id",
            "cycle_valid",
            "flap_frequency_hz",
            "airspeed_validated.true_airspeed_m_s",
            "airspeed_validated.pitch_filtered",
            "vehicle_air_data.rho",
            "label_valid",
            *ATTITUDE_AIRFLOW_REQUIRED_COLUMNS,
            "wind.windspeed_north_valid",
            "wind.windspeed_east_valid",
            *TARGETS,
        )
        if column in ordered.columns
    ]
    outputs: list[pd.DataFrame] = []
    for theta in theta_tip_deg:
        chunks: list[pd.DataFrame] = []
        for start in range(0, len(ordered), int(resolved.chunk_size)):
            stop = min(start + int(resolved.chunk_size), len(ordered))
            chunks.append(
                _chunk_result(
                    ordered.iloc[start:stop],
                    theta_tip_deg=float(theta),
                    geometry_path=Path(geometry_path),
                    config=resolved,
                    phase_acceleration_mode=phase_acceleration_mode,
                    phase_acceleration_rad_s2=acceleration[start:stop],
                    spanwise_regions=spanwise_regions,
                    include_detailed_diagnostics=include_detailed_diagnostics,
                )
            )
        physics = pd.concat(chunks).loc[ordered.index]
        metadata = ordered.loc[:, meta_columns].copy()
        metadata = metadata.rename(columns={target: f"true_{target}" for target in TARGETS})
        outputs.append(pd.concat([metadata.reset_index(drop=True), physics.reset_index(drop=True)], axis=1))
    return pd.concat(outputs, ignore_index=True)
