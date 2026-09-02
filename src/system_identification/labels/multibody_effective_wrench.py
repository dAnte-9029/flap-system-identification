"""Three-body inverse dynamics for whole-aircraft effective-wrench labels.

Inputs and outputs use the aircraft body FRD frame.  The fixed body origin is
the IMU origin.  Force excludes gravity, and the primary moment is about the
body-fixed whole-aircraft COM at the neutral wing pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import yaml

from system_identification.labels.effective_wrench import _rotation_body_to_world_from_quaternions


ISAACLAB_SOURCE_COMMIT = "1200ca8df1f907a515aafde6deac03cf06d0418a"
ISAACLAB_MASS_PROPERTIES_SHA256 = "3a74f2a20de4dacc658a41615d4cd2217937101d015d98841978a33f1983c826"
MEASURED_MASS_PROPERTIES_WORKBOOK_SHA256 = "4673fdee7be187278bf6ae9957b78645c781330007a43e20a4f24f5dfd24831b"


@dataclass(frozen=True)
class FlapperMultibodyModel:
    """Frozen body-plus-two-wing mass model in body FRD units."""

    body_mass_kg: float
    body_com_from_imu_frd_m: tuple[float, float, float]
    body_inertia_about_com_frd_kg_m2: tuple[tuple[float, float, float], ...]
    wing_mass_each_kg: float
    wing_pivot_from_imu_frd_m: tuple[float, float, float]
    wing_com_from_pivot_right_neutral_frame_frd_m: tuple[float, float, float]
    wing_inertia_about_com_neutral_frame_frd_kg_m2: tuple[tuple[float, float, float], ...]
    neutral_dihedral_rad: float

    @property
    def total_mass_kg(self) -> float:
        return self.body_mass_kg + 2.0 * self.wing_mass_each_kg

    @property
    def wing_spanwise_com_m(self) -> float:
        return abs(self.wing_com_from_pivot_right_neutral_frame_frd_m[1])

    @property
    def neutral_com_from_imu_frd_m(self) -> np.ndarray:
        right, left, _, _ = _wing_com_kinematics(
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
            self,
        )
        body_com = np.asarray(self.body_com_from_imu_frd_m, dtype=float)
        return (
            self.body_mass_kg * body_com
            + self.wing_mass_each_kg * (right.position_frd_m[0] + left.position_frd_m[0])
        ) / self.total_mass_kg


@dataclass(frozen=True)
class WingComKinematics:
    position_frd_m: np.ndarray
    relative_velocity_frd_m_s: np.ndarray
    relative_acceleration_frd_m_s2: np.ndarray
    rotation_frd_from_wing: np.ndarray
    relative_angular_velocity_frd_rad_s: np.ndarray
    relative_angular_acceleration_frd_rad_s2: np.ndarray


@dataclass(frozen=True)
class MultibodyEffectiveWrench:
    force_frd_n: np.ndarray
    moment_about_imu_frd_nm: np.ndarray
    moment_about_neutral_com_frd_nm: np.ndarray
    dynamic_com_frd_m: np.ndarray


@dataclass(frozen=True)
class PhaseDrivenFlapKinematics:
    position_rad: np.ndarray
    velocity_rad_s: np.ndarray
    acceleration_rad_s2: np.ndarray
    phase_rate_rad_s: np.ndarray
    phase_acceleration_rad_s2: np.ndarray
    position_phase_rate_rad_s: np.ndarray
    valid: np.ndarray


def measured_flapper_multibody_model() -> FlapperMultibodyModel:
    """Return the frozen measured model shared with the IsaacLab plant.

    The body inertia is IsaacLab's explicit nearest-diagonal projection onto
    the rigid-body inertia triangle cone.  Positions are transformed from the
    measured U_FRD origin to the IMU at ``[0, 0, 0.030]`` m.
    """

    return FlapperMultibodyModel(
        body_mass_kg=0.78261,
        body_com_from_imu_frd_m=(-0.13103, 0.00625, -0.04500),
        body_inertia_about_com_frd_kg_m2=(
            (0.0030833333333333333, 0.0, 0.0),
            (0.0, 0.023056666666666666, 0.0),
            (0.0, 0.0, 0.019973333333333333),
        ),
        wing_mass_each_kg=0.06077,
        wing_pivot_from_imu_frd_m=(0.0, 0.0, -0.03000),
        wing_com_from_pivot_right_neutral_frame_frd_m=(-0.06040, 0.29394, 0.0),
        wing_inertia_about_com_neutral_frame_frd_kg_m2=(
            (0.00270, 0.0, 0.0),
            (0.0, 0.00101, 0.0),
            (0.0, 0.0, 0.00371),
        ),
        neutral_dihedral_rad=-0.019391,
    )


def load_multibody_label_config(path: str | Path) -> tuple[dict[str, Any], FlapperMultibodyModel]:
    """Load and validate one frozen multibody label configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != "multibody_wrench_label_v1":
        raise ValueError("Expected multibody_wrench_label_v1 configuration.")
    raw_model = config.get("model")
    if not isinstance(raw_model, dict):
        raise ValueError("Multibody label configuration is missing model properties.")
    body = raw_model.get("body")
    wing = raw_model.get("wing_each")
    if not isinstance(body, dict) or not isinstance(wing, dict):
        raise ValueError("Multibody model requires body and wing_each mappings.")
    model = FlapperMultibodyModel(
        body_mass_kg=float(body["mass_kg"]),
        body_com_from_imu_frd_m=tuple(float(value) for value in body["com_from_imu_frd_m"]),
        body_inertia_about_com_frd_kg_m2=tuple(
            tuple(float(value) for value in row) for row in body["inertia_about_com_frd_kg_m2"]
        ),
        wing_mass_each_kg=float(wing["mass_kg"]),
        wing_pivot_from_imu_frd_m=tuple(float(value) for value in wing["pivot_from_imu_frd_m"]),
        wing_com_from_pivot_right_neutral_frame_frd_m=tuple(
            float(value) for value in wing["right_com_from_pivot_neutral_frame_frd_m"]
        ),
        wing_inertia_about_com_neutral_frame_frd_kg_m2=tuple(
            tuple(float(value) for value in row) for row in wing["inertia_about_com_neutral_frame_frd_kg_m2"]
        ),
        neutral_dihedral_rad=float(raw_model["neutral_dihedral_rad"]),
    )
    if not np.isclose(model.total_mass_kg, 0.90415, atol=1.0e-12):
        raise ValueError("Configured three-body mass must equal the measured 0.90415 kg total.")
    return config, model


def compute_phase_driven_flap_kinematics(
    samples: pd.DataFrame,
    *,
    amplitude_rad: float,
    window_s: float,
    polyorder: int,
) -> PhaseDrivenFlapKinematics:
    """Combine canonical phase position with measured frequency kinematics."""

    required = {"time_s", "mechanical_phase_rad", "flap_frequency_hz", "log_id"}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"Missing phase-kinematics columns: {missing}")
    position = float(amplitude_rad) * np.sin(samples["mechanical_phase_rad"].to_numpy(dtype=float))
    position_phase_rate_series = pd.Series(np.nan, index=samples.index, dtype=float)
    phase_acceleration_series = pd.Series(np.nan, index=samples.index, dtype=float)
    group_columns = [column for column in ("log_id", "segment_id") if column in samples.columns]
    grouper: Any = group_columns[0] if len(group_columns) == 1 else group_columns
    for _, group in samples.groupby(grouper, sort=False, dropna=False):
        original_index = group.index.to_numpy()
        time_s = group["time_s"].to_numpy(dtype=float)
        phase = group["mechanical_phase_rad"].to_numpy(dtype=float)
        frequency = group["flap_frequency_hz"].to_numpy(dtype=float)
        finite = np.isfinite(time_s) & np.isfinite(phase) & np.isfinite(frequency)
        if int(finite.sum()) < max(5, polyorder + 2):
            continue
        valid_index = original_index[finite]
        valid_time = time_s[finite]
        valid_phase = np.unwrap(phase[finite])
        valid_phase_rate = 2.0 * np.pi * frequency[finite]
        order = np.argsort(valid_time)
        valid_index = valid_index[order]
        valid_time = valid_time[order]
        valid_phase = valid_phase[order]
        valid_phase_rate = valid_phase_rate[order]
        dt = np.diff(valid_time)
        if len(dt) == 0 or not np.isfinite(dt).all() or np.any(dt <= 0.0):
            continue
        sample_period_s = float(np.median(dt))
        window = max(5, int(round(float(window_s) / sample_period_s)))
        if window % 2 == 0:
            window += 1
        if window > len(valid_phase):
            window = len(valid_phase) if len(valid_phase) % 2 == 1 else len(valid_phase) - 1
        if window <= polyorder:
            continue
        position_phase_rate_series.loc[valid_index] = savgol_filter(
            valid_phase,
            window_length=window,
            polyorder=polyorder,
            deriv=1,
            delta=sample_period_s,
            mode="interp",
        )
        phase_acceleration_series.loc[valid_index] = savgol_filter(
            valid_phase_rate,
            window_length=window,
            polyorder=polyorder,
            deriv=1,
            delta=sample_period_s,
            mode="interp",
        )

    position_phase_rate = position_phase_rate_series.to_numpy(dtype=float)
    phase_rate = 2.0 * np.pi * samples["flap_frequency_hz"].to_numpy(dtype=float)
    phase_acceleration = phase_acceleration_series.to_numpy(dtype=float)
    phase = samples["mechanical_phase_rad"].to_numpy(dtype=float)
    velocity = float(amplitude_rad) * np.cos(phase) * phase_rate
    acceleration = float(amplitude_rad) * (
        np.cos(phase) * phase_acceleration - np.sin(phase) * phase_rate**2
    )
    valid = np.isfinite(position) & np.isfinite(velocity) & np.isfinite(acceleration)
    return PhaseDrivenFlapKinematics(
        position_rad=position,
        velocity_rad_s=velocity,
        acceleration_rad_s2=acceleration,
        phase_rate_rad_s=phase_rate,
        phase_acceleration_rad_s2=phase_acceleration,
        position_phase_rate_rad_s=position_phase_rate,
        valid=valid,
    )


def reconstruct_multibody_labels_from_samples(
    samples: pd.DataFrame,
    *,
    model: FlapperMultibodyModel,
    amplitude_rad: float,
    phase_derivative_window_s: float,
    phase_derivative_polyorder: int,
) -> pd.DataFrame:
    """Return multibody labels and diagnostics aligned one-to-one with samples."""

    linear_acceleration_columns = [
        "vehicle_local_position.ax_smooth",
        "vehicle_local_position.ay_smooth",
        "vehicle_local_position.az_smooth",
    ]
    angular_velocity_columns = [
        "vehicle_angular_velocity.xyz[0]",
        "vehicle_angular_velocity.xyz[1]",
        "vehicle_angular_velocity.xyz[2]",
    ]
    angular_acceleration_columns = [
        "vehicle_angular_velocity.xyz_derivative_smooth[0]",
        "vehicle_angular_velocity.xyz_derivative_smooth[1]",
        "vehicle_angular_velocity.xyz_derivative_smooth[2]",
    ]
    quaternion_columns = [
        "vehicle_attitude.q[0]",
        "vehicle_attitude.q[1]",
        "vehicle_attitude.q[2]",
        "vehicle_attitude.q[3]",
    ]
    required = set(
        linear_acceleration_columns
        + angular_velocity_columns
        + angular_acceleration_columns
        + quaternion_columns
        + ["mechanical_phase_rad", "time_s", "log_id"]
    )
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"Missing multibody label columns: {missing}")

    acceleration_ned = samples[linear_acceleration_columns].to_numpy(dtype=float)
    body_omega = samples[angular_velocity_columns].to_numpy(dtype=float)
    body_alpha = samples[angular_acceleration_columns].to_numpy(dtype=float)
    quaternion = samples[quaternion_columns].to_numpy(dtype=float)
    rotation_ned_from_body, quaternion_valid = _rotation_body_to_world_from_quaternions(quaternion)
    acceleration_origin_frd = np.einsum("nji,nj->ni", rotation_ned_from_body, acceleration_ned)
    gravity_ned = np.broadcast_to(np.array([0.0, 0.0, 9.81]), acceleration_ned.shape)
    gravity_frd = np.einsum("nji,nj->ni", rotation_ned_from_body, gravity_ned)

    kinematics = compute_phase_driven_flap_kinematics(
        samples,
        amplitude_rad=amplitude_rad,
        window_s=phase_derivative_window_s,
        polyorder=phase_derivative_polyorder,
    )
    result = compute_multibody_effective_wrench(
        acceleration_origin_frd_m_s2=acceleration_origin_frd,
        gravity_frd_m_s2=gravity_frd,
        body_angular_velocity_frd_rad_s=body_omega,
        body_angular_acceleration_frd_rad_s2=body_alpha,
        flap_position_rad=kinematics.position_rad,
        flap_velocity_rad_s=kinematics.velocity_rad_s,
        flap_acceleration_rad_s2=kinematics.acceleration_rad_s2,
        model=model,
    )
    valid = (
        quaternion_valid
        & kinematics.valid
        & np.isfinite(acceleration_ned).all(axis=1)
        & np.isfinite(body_omega).all(axis=1)
        & np.isfinite(body_alpha).all(axis=1)
    )
    for column in ("label_reconstruction_valid", "phase_valid"):
        if column in samples.columns:
            valid &= samples[column].fillna(False).astype(bool).to_numpy()

    output = pd.DataFrame(index=samples.index)
    values = {
        "fx_b_multibody": result.force_frd_n[:, 0],
        "fy_b_multibody": result.force_frd_n[:, 1],
        "fz_b_multibody": result.force_frd_n[:, 2],
        "mx_b_multibody": result.moment_about_neutral_com_frd_nm[:, 0],
        "my_b_multibody": result.moment_about_neutral_com_frd_nm[:, 1],
        "mz_b_multibody": result.moment_about_neutral_com_frd_nm[:, 2],
        "dynamic_cg_x_frd_m": result.dynamic_com_frd_m[:, 0],
        "dynamic_cg_y_frd_m": result.dynamic_com_frd_m[:, 1],
        "dynamic_cg_z_frd_m": result.dynamic_com_frd_m[:, 2],
        "wing_q_rad": kinematics.position_rad,
        "wing_qd_rad_s": kinematics.velocity_rad_s,
        "wing_qdd_rad_s2": kinematics.acceleration_rad_s2,
        "mechanical_phase_rate_rad_s": kinematics.phase_rate_rad_s,
        "mechanical_phase_acceleration_rad_s2": kinematics.phase_acceleration_rad_s2,
        "position_phase_rate_rad_s": kinematics.position_phase_rate_rad_s,
    }
    for column, value in values.items():
        output[column] = value
        output.loc[~valid, column] = np.nan
    for old_column, new_column in zip(
        ("fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b"),
        ("fx_b_multibody", "fy_b_multibody", "fz_b_multibody", "mx_b_multibody", "my_b_multibody", "mz_b_multibody"),
    ):
        if old_column in samples.columns:
            output[f"delta_{old_column}_multibody_minus_rigid"] = output[new_column] - samples[old_column]
    if "flap_frequency_hz" in samples.columns:
        output["position_phase_rate_minus_logged_frequency_rad_s"] = (
            output["position_phase_rate_rad_s"]
            - 2.0 * np.pi * samples["flap_frequency_hz"].to_numpy(dtype=float)
        )
    output["multibody_label_valid"] = valid
    return output


def _rotation_x(angle_rad: np.ndarray) -> np.ndarray:
    angle = np.asarray(angle_rad, dtype=float)
    rotation = np.zeros((len(angle), 3, 3), dtype=float)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation[:, 0, 0] = 1.0
    rotation[:, 1, 1] = cosine
    rotation[:, 1, 2] = -sine
    rotation[:, 2, 1] = sine
    rotation[:, 2, 2] = cosine
    return rotation


def _one_wing_com_kinematics(
    flap_position_rad: np.ndarray,
    flap_velocity_rad_s: np.ndarray,
    flap_acceleration_rad_s2: np.ndarray,
    model: FlapperMultibodyModel,
    *,
    side_sign: float,
) -> WingComKinematics:
    gamma = model.neutral_dihedral_rad + flap_position_rad
    # Positive engineering flap position raises both wings.  Raising is FRD
    # negative z, while the mirrored wing span coordinates have opposite y.
    joint_angle = -side_sign * gamma
    joint_rate = -side_sign * flap_velocity_rad_s
    joint_acceleration = -side_sign * flap_acceleration_rad_s2
    rotation = _rotation_x(joint_angle)

    right_com = np.asarray(model.wing_com_from_pivot_right_neutral_frame_frd_m, dtype=float)
    local_com = right_com * np.array([1.0, side_sign, 1.0])
    rotated_com = np.einsum("nij,j->ni", rotation, local_com)
    pivot = np.asarray(model.wing_pivot_from_imu_frd_m, dtype=float)
    position = rotated_com + pivot

    relative_omega = np.zeros_like(position)
    relative_alpha = np.zeros_like(position)
    relative_omega[:, 0] = joint_rate
    relative_alpha[:, 0] = joint_acceleration
    relative_velocity = np.cross(relative_omega, rotated_com)
    relative_acceleration = np.cross(relative_alpha, rotated_com) + np.cross(
        relative_omega,
        np.cross(relative_omega, rotated_com),
    )
    return WingComKinematics(
        position_frd_m=position,
        relative_velocity_frd_m_s=relative_velocity,
        relative_acceleration_frd_m_s2=relative_acceleration,
        rotation_frd_from_wing=rotation,
        relative_angular_velocity_frd_rad_s=relative_omega,
        relative_angular_acceleration_frd_rad_s2=relative_alpha,
    )


def _wing_com_kinematics(
    flap_position_rad: np.ndarray,
    flap_velocity_rad_s: np.ndarray,
    flap_acceleration_rad_s2: np.ndarray,
    model: FlapperMultibodyModel,
) -> tuple[WingComKinematics, WingComKinematics, np.ndarray, np.ndarray]:
    right = _one_wing_com_kinematics(
        flap_position_rad,
        flap_velocity_rad_s,
        flap_acceleration_rad_s2,
        model,
        side_sign=-1.0,
    )
    left = _one_wing_com_kinematics(
        flap_position_rad,
        flap_velocity_rad_s,
        flap_acceleration_rad_s2,
        model,
        side_sign=1.0,
    )
    return right, left, right.position_frd_m, left.position_frd_m


def _point_acceleration(
    acceleration_origin: np.ndarray,
    body_omega: np.ndarray,
    body_alpha: np.ndarray,
    position: np.ndarray,
    relative_velocity: np.ndarray,
    relative_acceleration: np.ndarray,
) -> np.ndarray:
    return (
        acceleration_origin
        + np.cross(body_alpha, position)
        + np.cross(body_omega, np.cross(body_omega, position))
        + 2.0 * np.cross(body_omega, relative_velocity)
        + relative_acceleration
    )


def _rotational_moment_about_com(
    inertia_about_com: np.ndarray,
    body_omega: np.ndarray,
    body_alpha: np.ndarray,
    relative_omega: np.ndarray,
    relative_alpha: np.ndarray,
) -> np.ndarray:
    angular_velocity = body_omega + relative_omega
    angular_acceleration = body_alpha + relative_alpha + np.cross(body_omega, relative_omega)
    angular_momentum = np.einsum("nij,nj->ni", inertia_about_com, angular_velocity)
    return np.einsum("nij,nj->ni", inertia_about_com, angular_acceleration) + np.cross(
        angular_velocity,
        angular_momentum,
    )


def compute_multibody_effective_wrench(
    *,
    acceleration_origin_frd_m_s2: np.ndarray,
    gravity_frd_m_s2: np.ndarray,
    body_angular_velocity_frd_rad_s: np.ndarray,
    body_angular_acceleration_frd_rad_s2: np.ndarray,
    flap_position_rad: np.ndarray,
    flap_velocity_rad_s: np.ndarray,
    flap_acceleration_rad_s2: np.ndarray,
    model: FlapperMultibodyModel,
) -> MultibodyEffectiveWrench:
    """Reconstruct the non-gravity external wrench of the three-body system."""

    acceleration_origin = np.asarray(acceleration_origin_frd_m_s2, dtype=float)
    gravity = np.asarray(gravity_frd_m_s2, dtype=float)
    body_omega = np.asarray(body_angular_velocity_frd_rad_s, dtype=float)
    body_alpha = np.asarray(body_angular_acceleration_frd_rad_s2, dtype=float)
    q = np.asarray(flap_position_rad, dtype=float)
    qd = np.asarray(flap_velocity_rad_s, dtype=float)
    qdd = np.asarray(flap_acceleration_rad_s2, dtype=float)
    sample_count = len(q)
    if any(value.shape != (sample_count, 3) for value in (acceleration_origin, gravity, body_omega, body_alpha)):
        raise ValueError("Vector inputs must have shape (N, 3) matching flap kinematics.")
    if qd.shape != q.shape or qdd.shape != q.shape or q.ndim != 1:
        raise ValueError("Flap position, velocity and acceleration must have identical shape (N,).")

    body_position = np.broadcast_to(np.asarray(model.body_com_from_imu_frd_m, dtype=float), (sample_count, 3))
    zeros = np.zeros((sample_count, 3), dtype=float)
    right, left, _, _ = _wing_com_kinematics(q, qd, qdd, model)
    links = (
        (
            model.body_mass_kg,
            body_position,
            zeros,
            zeros,
            np.broadcast_to(np.asarray(model.body_inertia_about_com_frd_kg_m2), (sample_count, 3, 3)),
            zeros,
            zeros,
        ),
        *(
            (
                model.wing_mass_each_kg,
                wing.position_frd_m,
                wing.relative_velocity_frd_m_s,
                wing.relative_acceleration_frd_m_s2,
                np.einsum(
                    "nij,jk,nlk->nil",
                    wing.rotation_frd_from_wing,
                    np.asarray(model.wing_inertia_about_com_neutral_frame_frd_kg_m2),
                    wing.rotation_frd_from_wing,
                ),
                wing.relative_angular_velocity_frd_rad_s,
                wing.relative_angular_acceleration_frd_rad_s2,
            )
            for wing in (right, left)
        ),
    )

    force = np.zeros((sample_count, 3), dtype=float)
    moment_about_imu = np.zeros((sample_count, 3), dtype=float)
    weighted_com = np.zeros((sample_count, 3), dtype=float)
    for mass, position, relative_velocity, relative_acceleration, inertia, relative_omega, relative_alpha in links:
        acceleration_com = _point_acceleration(
            acceleration_origin,
            body_omega,
            body_alpha,
            position,
            relative_velocity,
            relative_acceleration,
        )
        non_gravity_force = mass * (acceleration_com - gravity)
        spin_moment = _rotational_moment_about_com(
            inertia,
            body_omega,
            body_alpha,
            relative_omega,
            relative_alpha,
        )
        force += non_gravity_force
        moment_about_imu += spin_moment + np.cross(position, non_gravity_force)
        weighted_com += mass * position

    dynamic_com = weighted_com / model.total_mass_kg
    neutral_com = model.neutral_com_from_imu_frd_m
    moment_about_neutral_com = moment_about_imu - np.cross(neutral_com, force)
    return MultibodyEffectiveWrench(
        force_frd_n=force,
        moment_about_imu_frd_nm=moment_about_imu,
        moment_about_neutral_com_frd_nm=moment_about_neutral_com,
        dynamic_com_frd_m=dynamic_com,
    )


__all__ = [
    "FlapperMultibodyModel",
    "ISAACLAB_MASS_PROPERTIES_SHA256",
    "ISAACLAB_SOURCE_COMMIT",
    "MEASURED_MASS_PROPERTIES_WORKBOOK_SHA256",
    "MultibodyEffectiveWrench",
    "compute_multibody_effective_wrench",
    "compute_phase_driven_flap_kinematics",
    "load_multibody_label_config",
    "measured_flapper_multibody_model",
    "reconstruct_multibody_labels_from_samples",
]
