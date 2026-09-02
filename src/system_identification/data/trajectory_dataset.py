from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyulog import ULog

from system_identification.data.ulg_audit import _dataset, _event_times, _zoh_values, parse_log_datetime


DATASET_VERSION = "trajectory_dataset_v1"
COUNTS_PER_MOTOR_REVOLUTION = 4096.0

STATE_COLUMNS = (
    "position_ned_m_x",
    "position_ned_m_y",
    "position_ned_m_z",
    "velocity_ned_m_s_x",
    "velocity_ned_m_s_y",
    "velocity_ned_m_s_z",
    "attitude_q_w",
    "attitude_q_x",
    "attitude_q_y",
    "attitude_q_z",
    "angular_velocity_body_rad_s_x",
    "angular_velocity_body_rad_s_y",
    "angular_velocity_body_rad_s_z",
)

CONTROL_COLUMNS = (
    "control_flap_motor_normalized",
    "control_left_elevon_normalized",
    "control_right_elevon_normalized",
    "control_rudder_normalized",
)

FLAP_STATE_COLUMNS = (
    "relative_flap_phase_rad",
    "relative_flap_phase_sin",
    "relative_flap_phase_cos",
    "flap_frequency_hz",
)

OPTIONAL_INITIAL_CONTEXT_COLUMNS = (
    "true_airspeed_m_s",
    "wind_ned_m_s_n",
    "wind_ned_m_s_e",
)

ABSOLUTE_PHASE_COLUMNS = (
    "absolute_flap_phase_rad",
    "absolute_flap_phase_sin",
    "absolute_flap_phase_cos",
    "hall_reference_valid",
)

FUTURE_FORBIDDEN_INPUT_COLUMNS = (
    *STATE_COLUMNS,
    *FLAP_STATE_COLUMNS,
    *OPTIONAL_INITIAL_CONTEXT_COLUMNS,
    *ABSOLUTE_PHASE_COLUMNS,
)

def relative_phase_from_total_count(
    total_count: np.ndarray | Sequence[float],
    *,
    transmission_ratio: float,
    counts_per_motor_revolution: float = COUNTS_PER_MOTOR_REVOLUTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log-local unwrapped and wrapped wing phase from cumulative motor counts."""
    counts = np.asarray(total_count, dtype=float)
    if transmission_ratio <= 0.0 or counts_per_motor_revolution <= 0.0:
        raise ValueError("transmission_ratio and counts_per_motor_revolution must be positive")
    finite = np.isfinite(counts)
    unwrapped = np.full(counts.shape, np.nan, dtype=float)
    if not np.any(finite):
        return unwrapped, unwrapped.copy()
    origin = counts[np.flatnonzero(finite)[0]]
    counts_per_wing_cycle = counts_per_motor_revolution * float(transmission_ratio)
    unwrapped[finite] = (counts[finite] - origin) * (2.0 * np.pi / counts_per_wing_cycle)
    wrapped = np.mod(unwrapped, 2.0 * np.pi)
    return unwrapped, wrapped


def assign_contiguous_segments(
    timestamps_us: np.ndarray | Sequence[int],
    valid: np.ndarray | Sequence[bool],
    *,
    expected_dt_us: int,
    maximum_gap_us: int,
) -> np.ndarray:
    """Assign non-negative segment IDs without bridging invalid rows or time gaps."""
    timestamps = np.asarray(timestamps_us, dtype=np.int64)
    valid_array = np.asarray(valid, dtype=bool)
    if timestamps.ndim != 1 or valid_array.ndim != 1 or len(timestamps) != len(valid_array):
        raise ValueError("timestamps_us and valid must be one-dimensional and equal length")
    if expected_dt_us <= 0 or maximum_gap_us < expected_dt_us:
        raise ValueError("gap thresholds must be positive and maximum_gap_us >= expected_dt_us")

    segments = np.full(len(timestamps), -1, dtype=np.int64)
    current_segment = -1
    previous_valid_index: int | None = None
    for index, is_valid in enumerate(valid_array):
        if not is_valid:
            previous_valid_index = None
            continue
        starts_new = previous_valid_index is None
        if previous_valid_index is not None:
            delta_us = int(timestamps[index] - timestamps[previous_valid_index])
            starts_new = delta_us <= 0 or delta_us > maximum_gap_us
        if starts_new:
            current_segment += 1
        segments[index] = current_segment
        previous_valid_index = index
    return segments


def build_window_index(
    samples: pd.DataFrame,
    *,
    horizon_steps: int,
    stride_steps: int,
    dt_s: float,
) -> pd.DataFrame:
    """Build an inclusive state / exclusive final-control trajectory window index."""
    if horizon_steps < 1 or stride_steps < 1 or dt_s <= 0.0:
        raise ValueError("horizon_steps, stride_steps, and dt_s must be positive")
    required = {"split", "log_id", "segment_id", "sample_in_segment", "timestamp_us"}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"sample table is missing columns: {missing}")

    eligible = samples.loc[samples["segment_id"].to_numpy(dtype=int) >= 0]
    if "valid_core" in eligible:
        eligible = eligible.loc[eligible["valid_core"].to_numpy(dtype=bool)]
    records: list[dict[str, Any]] = []
    group_columns = ["split", "log_id", "segment_id"]
    for (split, log_id, segment_id), group in eligible.groupby(group_columns, sort=False, dropna=False):
        ordered = group.sort_values("sample_in_segment", kind="stable")
        sample_numbers = ordered["sample_in_segment"].to_numpy(dtype=np.int64)
        if len(sample_numbers) and not np.array_equal(sample_numbers, np.arange(len(sample_numbers))):
            raise ValueError(f"non-contiguous sample_in_segment for {log_id} segment {segment_id}")
        timestamps = ordered["timestamp_us"].to_numpy(dtype=np.int64)
        sample_in_log = (
            ordered["sample_in_log"].to_numpy(dtype=np.int64)
            if "sample_in_log" in ordered
            else ordered.index.to_numpy(dtype=np.int64)
        )
        for start in range(0, len(ordered) - horizon_steps, stride_steps):
            end = start + horizon_steps
            records.append(
                {
                    "window_id": f"{split}:{log_id}:{int(segment_id)}:{int(sample_numbers[start])}",
                    "split": str(split),
                    "log_id": str(log_id),
                    "segment_id": int(segment_id),
                    "start_sample_in_segment": int(sample_numbers[start]),
                    "end_sample_in_segment": int(sample_numbers[end]),
                    "start_sample_in_log": int(sample_in_log[start]),
                    "end_sample_in_log": int(sample_in_log[end]),
                    "start_timestamp_us": int(timestamps[start]),
                    "end_timestamp_us": int(timestamps[end]),
                    "observed_horizon_s": float((timestamps[end] - timestamps[start]) * 1.0e-6),
                    "horizon_s": float(horizon_steps * dt_s),
                    "state_sample_count": int(horizon_steps + 1),
                    "control_step_count": int(horizon_steps),
                }
            )
    return pd.DataFrame.from_records(records)


def _log_date(log_id: str) -> str:
    parsed = parse_log_datetime(Path(log_id).name)
    if parsed is None:
        raise ValueError(f"cannot parse flight date from log name: {log_id}")
    return parsed.date().isoformat()


def validate_split_assignments(assignments: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    required = ("train", "validation", "sealed_test")
    missing = [split for split in required if split not in assignments]
    if missing:
        raise ValueError(f"missing required split assignments: {missing}")
    normalized = {split: [str(path) for path in assignments[split]] for split in required}
    owners: dict[str, str] = {}
    for split, paths in normalized.items():
        for path in paths:
            if path in owners:
                raise ValueError(f"log overlap between {owners[path]} and {split}: {path}")
            owners[path] = split
    split_dates = {split: sorted({_log_date(path) for path in paths}) for split, paths in normalized.items()}
    date_owners: dict[str, str] = {}
    for split, dates in split_dates.items():
        for date in dates:
            if date in date_owners:
                raise ValueError(f"date overlap between {date_owners[date]} and {split}: {date}")
            date_owners[date] = split
    return {"assignments": normalized, "split_dates": split_dates}


def _aligned(
    reference_us: np.ndarray,
    dataset: Any | None,
    field: str,
    *,
    freshness_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Past-only zero-order hold makes every aligned value causally available at its row timestamp.
    return _zoh_values(reference_us, dataset, field, freshness_s=freshness_s)


def _counter_change(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    changed = np.zeros(len(values), dtype=bool)
    if len(values) > 1:
        pair_valid = valid[1:] & valid[:-1]
        changed[1:] = pair_valid & (values[1:] != values[:-1])
    return changed


def extract_trajectory_samples(
    ulg_path: str | Path,
    *,
    log_id: str,
    split: str,
    transmission_ratio: float,
    expected_rate_hz: float = 50.0,
    maximum_gap_s: float = 0.05,
) -> pd.DataFrame:
    """Extract one causally aligned sample table from a Step 0 admitted ULog."""
    if expected_rate_hz <= 0.0 or maximum_gap_s <= 0.0:
        raise ValueError("expected_rate_hz and maximum_gap_s must be positive")
    ulog = ULog(str(ulg_path))
    local = _dataset(ulog, "vehicle_local_position")
    if local is None:
        raise ValueError(f"vehicle_local_position missing: {ulg_path}")
    reference_us = _event_times(local)
    sample_count = len(reference_us)
    datasets = {
        name: _dataset(ulog, name)
        for name in (
            "vehicle_attitude",
            "vehicle_angular_velocity",
            "actuator_motors",
            "actuator_servos",
            "encoder_count",
            "flap_frequency",
            "wing_phase",
            "airspeed_validated",
            "wind",
            "vehicle_status",
            "vehicle_land_detected",
        )
    }

    frame = pd.DataFrame(
        {
            "split": split,
            "log_id": log_id,
            "flight_date": _log_date(log_id),
            "sample_in_log": np.arange(sample_count, dtype=np.int64),
            "timestamp_us": reference_us,
            "time_since_log_start_s": (reference_us - reference_us[0]).astype(float) * 1.0e-6,
        }
    )
    state_valid = np.ones(sample_count, dtype=bool)
    for source, target in (
        ("x", "position_ned_m_x"),
        ("y", "position_ned_m_y"),
        ("z", "position_ned_m_z"),
        ("vx", "velocity_ned_m_s_x"),
        ("vy", "velocity_ned_m_s_y"),
        ("vz", "velocity_ned_m_s_z"),
    ):
        values = np.asarray(local.data.get(source, np.full(sample_count, np.nan)), dtype=float)
        frame[target] = values
        state_valid &= np.isfinite(values)
    for field in ("xy_valid", "z_valid", "v_xy_valid", "v_z_valid"):
        state_valid &= np.asarray(local.data.get(field, np.zeros(sample_count)), dtype=float) > 0.5

    q_values: list[np.ndarray] = []
    q_valid = np.ones(sample_count, dtype=bool)
    for field in ("q[0]", "q[1]", "q[2]", "q[3]"):
        values, fresh = _aligned(reference_us, datasets["vehicle_attitude"], field, freshness_s=0.05)
        q_values.append(values)
        q_valid &= fresh
    quaternion = np.column_stack(q_values)
    q_norm = np.linalg.norm(quaternion, axis=1)
    q_valid &= np.isfinite(q_norm) & (np.abs(q_norm - 1.0) <= 1.0e-3)
    quaternion[q_valid] /= q_norm[q_valid, None]
    for column, values in zip(STATE_COLUMNS[6:10], quaternion.T, strict=True):
        frame[column] = values
    state_valid &= q_valid

    angular_valid = np.ones(sample_count, dtype=bool)
    for field, target in zip(
        ("xyz[0]", "xyz[1]", "xyz[2]"), STATE_COLUMNS[10:13], strict=True
    ):
        values, fresh = _aligned(reference_us, datasets["vehicle_angular_velocity"], field, freshness_s=0.05)
        frame[target] = values
        angular_valid &= fresh
    state_valid &= angular_valid

    control_valid = np.ones(sample_count, dtype=bool)
    control_sources = (
        (datasets["actuator_motors"], "control[0]"),
        (datasets["actuator_servos"], "control[0]"),
        (datasets["actuator_servos"], "control[1]"),
        (datasets["actuator_servos"], "control[2]"),
    )
    for (dataset, field), target in zip(control_sources, CONTROL_COLUMNS, strict=True):
        values, fresh = _aligned(reference_us, dataset, field, freshness_s=0.05)
        frame[target] = values
        control_valid &= fresh

    encoder_count, encoder_fresh = _aligned(
        reference_us, datasets["encoder_count"], "total_count", freshness_s=0.10
    )
    phase_unwrapped, phase_wrapped = relative_phase_from_total_count(
        encoder_count, transmission_ratio=transmission_ratio
    )
    frequency, frequency_fresh = _aligned(
        reference_us, datasets["wing_phase"], "flap_frequency_hz", freshness_s=0.10
    )
    fallback_frequency, fallback_fresh = _aligned(
        reference_us, datasets["flap_frequency"], "frequency_hz", freshness_s=0.10
    )
    use_fallback = ~frequency_fresh & fallback_fresh
    frequency[use_fallback] = fallback_frequency[use_fallback]
    frequency_fresh |= fallback_fresh
    phase_valid = encoder_fresh & np.isfinite(phase_wrapped) & frequency_fresh & (frequency > 0.5) & (frequency < 20.0)
    frame["relative_flap_phase_unwrapped_rad"] = phase_unwrapped
    frame["relative_flap_phase_rad"] = phase_wrapped
    frame["relative_flap_phase_sin"] = np.sin(phase_wrapped)
    frame["relative_flap_phase_cos"] = np.cos(phase_wrapped)
    frame["flap_frequency_hz"] = frequency
    frame["absolute_flap_phase_rad"] = np.nan
    frame["absolute_flap_phase_sin"] = np.nan
    frame["absolute_flap_phase_cos"] = np.nan
    frame["hall_reference_valid"] = False

    airspeed, airspeed_fresh = _aligned(
        reference_us, datasets["airspeed_validated"], "true_airspeed_m_s", freshness_s=0.25
    )
    wind_north, wind_north_fresh = _aligned(
        reference_us, datasets["wind"], "windspeed_north", freshness_s=0.35
    )
    wind_east, wind_east_fresh = _aligned(
        reference_us, datasets["wind"], "windspeed_east", freshness_s=0.35
    )
    airdata_valid = (
        airspeed_fresh
        & (airspeed >= 0.0)
        & (airspeed <= 30.0)
        & wind_north_fresh
        & wind_east_fresh
    )
    frame["true_airspeed_m_s"] = airspeed
    frame["wind_ned_m_s_n"] = wind_north
    frame["wind_ned_m_s_e"] = wind_east

    arming_state, arming_fresh = _aligned(
        reference_us, datasets["vehicle_status"], "arming_state", freshness_s=1.5
    )
    landed, landed_fresh = _aligned(
        reference_us, datasets["vehicle_land_detected"], "landed", freshness_s=1.5
    )
    failsafe, failsafe_fresh = _aligned(
        reference_us, datasets["vehicle_status"], "failsafe", freshness_s=1.5
    )
    nav_state, nav_fresh = _aligned(
        reference_us, datasets["vehicle_status"], "nav_state", freshness_s=1.5
    )
    airborne = arming_fresh & landed_fresh & (arming_state == 2.0) & (landed < 0.5)
    safe = failsafe_fresh & (failsafe < 0.5)

    reset_boundary = np.zeros(sample_count, dtype=bool)
    for field in ("xy_reset_counter", "z_reset_counter", "vxy_reset_counter", "vz_reset_counter", "heading_reset_counter"):
        values = np.asarray(local.data.get(field, np.zeros(sample_count)), dtype=float)
        reset_boundary |= _counter_change(values, np.isfinite(values))
    attitude_reset, attitude_reset_fresh = _aligned(
        reference_us, datasets["vehicle_attitude"], "quat_reset_counter", freshness_s=0.05
    )
    reset_boundary |= _counter_change(attitude_reset, attitude_reset_fresh)
    encoder_reset = np.zeros(sample_count, dtype=bool)
    if sample_count > 1:
        encoder_reset[1:] = encoder_fresh[1:] & encoder_fresh[:-1] & (np.diff(encoder_count) < 0.0)
    mode_boundary = _counter_change(nav_state, nav_fresh)
    monotonic_time = np.ones(sample_count, dtype=bool)
    gap_boundary = np.zeros(sample_count, dtype=bool)
    if sample_count > 1:
        timestamp_delta_us = np.diff(reference_us)
        monotonic_time[1:] = timestamp_delta_us > 0
        gap_boundary[1:] = timestamp_delta_us > int(round(maximum_gap_s * 1.0e6))

    valid_core = (
        state_valid
        & control_valid
        & phase_valid
        & airborne
        & safe
        & nav_fresh
        & monotonic_time
        & ~reset_boundary
        & ~encoder_reset
        & ~mode_boundary
    )
    expected_dt_us = int(round(1.0e6 / expected_rate_hz))
    segment_ids = assign_contiguous_segments(
        reference_us,
        valid_core,
        expected_dt_us=expected_dt_us,
        maximum_gap_us=int(round(maximum_gap_s * 1.0e6)),
    )
    frame["nav_state"] = nav_state
    frame["valid_state"] = state_valid
    frame["valid_control"] = control_valid
    frame["valid_phase"] = phase_valid
    frame["valid_airdata"] = airdata_valid
    frame["valid_airborne_safe"] = airborne & safe
    frame["reset_boundary"] = reset_boundary | encoder_reset
    frame["mode_boundary"] = mode_boundary
    frame["gap_boundary"] = gap_boundary
    frame["valid_core"] = valid_core
    exclusion_reason = np.full(sample_count, "", dtype=object)
    reason_masks = (
        (~state_valid, "invalid_state"),
        (~control_valid, "invalid_control"),
        (~phase_valid, "invalid_phase"),
        (~(airborne & safe), "not_airborne_or_failsafe"),
        (~nav_fresh, "invalid_nav_state"),
        (reset_boundary | encoder_reset, "reset_boundary"),
        (mode_boundary, "mode_boundary"),
        (~monotonic_time, "nonmonotonic_time"),
    )
    for reason_mask, reason in reason_masks:
        for index in np.flatnonzero(reason_mask):
            exclusion_reason[index] = f"{exclusion_reason[index]}|{reason}".strip("|")
    frame["exclusion_reason"] = exclusion_reason
    frame["segment_id"] = segment_ids
    frame["sample_in_segment"] = -1
    for segment_id in np.unique(segment_ids[segment_ids >= 0]):
        mask = segment_ids == segment_id
        frame.loc[mask, "sample_in_segment"] = np.arange(int(mask.sum()), dtype=np.int64)
    return frame


def _duration_s(frame: pd.DataFrame) -> float:
    total = 0.0
    valid = frame.loc[frame["segment_id"] >= 0]
    for _, group in valid.groupby(["log_id", "segment_id"], sort=False):
        timestamps = group["timestamp_us"].to_numpy(dtype=np.int64)
        if len(timestamps) > 1:
            total += float(np.sum(np.diff(timestamps)) * 1.0e-6)
    return total


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_trajectory_dataset(
    *,
    audit_summary_path: str | Path,
    output_root: str | Path,
    partitions: Sequence[str] = ("train", "validation"),
    expected_rate_hz: float = 50.0,
    maximum_gap_s: float = 0.05,
    horizon_s: float = 2.0,
    stride_s: float = 0.2,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build sample and window Parquets from the frozen Step 0 primary cohort."""
    audit_path = Path(audit_summary_path).resolve()
    output_path = Path(output_root).resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_path}")
    if expected_rate_hz <= 0.0 or horizon_s <= 0.0 or stride_s <= 0.0:
        raise ValueError("rate, horizon, and stride must be positive")
    horizon_steps = int(round(horizon_s * expected_rate_hz))
    stride_steps = int(round(stride_s * expected_rate_hz))
    if not math.isclose(horizon_steps / expected_rate_hz, horizon_s, abs_tol=1.0e-9):
        raise ValueError("horizon_s must be an integer number of nominal samples")
    if not math.isclose(stride_steps / expected_rate_hz, stride_s, abs_tol=1.0e-9):
        raise ValueError("stride_s must be an integer number of nominal samples")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    recommended = audit["recommended_splits"]
    split_contract = validate_split_assignments(recommended)
    allowed_partitions = set(split_contract["assignments"])
    requested = [str(partition) for partition in partitions]
    unknown = sorted(set(requested) - allowed_partitions)
    if unknown:
        raise ValueError(f"unknown partitions: {unknown}")
    if len(requested) != len(set(requested)):
        raise ValueError("partitions must be unique")
    records_by_path = {record["relative_path"]: record for record in audit["logs"]}
    source_root = Path(audit["scope"]["source_root"])

    output_path.mkdir(parents=True, exist_ok=True)
    partition_summaries: dict[str, Any] = {}
    for partition in requested:
        frames: list[pd.DataFrame] = []
        for log_id in split_contract["assignments"][partition]:
            record = records_by_path[log_id]
            if record["admission_status"] != "eligible":
                raise ValueError(f"Step 0 did not admit {log_id}: {record['admission_status']}")
            source_path = source_root / log_id
            if not source_path.is_file():
                raise FileNotFoundError(f"source ULog missing: {source_path}")
            ratio = float(record["selected_parameters"]["FLAP_RATIO"])
            frames.append(
                extract_trajectory_samples(
                    source_path,
                    log_id=log_id,
                    split=partition,
                    transmission_ratio=ratio,
                    expected_rate_hz=expected_rate_hz,
                    maximum_gap_s=maximum_gap_s,
                )
            )
        samples = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        windows = build_window_index(
            samples,
            horizon_steps=horizon_steps,
            stride_steps=stride_steps,
            dt_s=1.0 / expected_rate_hz,
        )
        samples_path = output_path / f"samples_{partition}.parquet"
        windows_path = output_path / f"windows_{partition}.parquet"
        samples.to_parquet(samples_path, index=False)
        windows.to_parquet(windows_path, index=False)
        observed_error = (
            np.abs(windows["observed_horizon_s"] - horizon_s).to_numpy(dtype=float)
            if len(windows)
            else np.array([], dtype=float)
        )
        partition_summaries[partition] = {
            "log_count": len(frames),
            "source_sample_count": int(len(samples)),
            "valid_sample_count": int(samples["valid_core"].sum()),
            "valid_duration_s": _duration_s(samples),
            "segment_count": int(
                len(samples.loc[samples["segment_id"] >= 0, ["log_id", "segment_id"]].drop_duplicates())
            ),
            "window_count": int(len(windows)),
            "observed_horizon_error_s_max": float(observed_error.max()) if len(observed_error) else None,
            "airdata_valid_on_core_ratio": float(
                (samples["valid_core"] & samples["valid_airdata"]).sum()
                / max(1, int(samples["valid_core"].sum()))
            ),
            "samples_file": samples_path.name,
            "windows_file": windows_path.name,
        }

    repository = Path(repository_root).resolve() if repository_root is not None else Path(__file__).resolve().parents[3]
    manifest: dict[str, Any] = {
        "dataset_version": DATASET_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "builder_git_head": _git_head(repository),
        "source": {
            "step0_audit_path": str(audit_path),
            "step0_audit_version": audit["audit_version"],
            "step0_generated_at": audit["generated_at"],
            "qgclogs_root": str(source_root),
            "qgclogs_git_head": audit.get("provenance", {}).get("source_repository_head"),
            "primary_cohort": recommended["primary_cohort"],
        },
        "split_contract": {
            **split_contract,
            "materialized_partitions": requested,
            "sealed_test_opened": "sealed_test" in requested,
            "ood_holdout_materialized": False,
        },
        "sampling": {
            "reference_topic": "vehicle_local_position",
            "event_time": "timestamp_sample when valid, otherwise timestamp",
            "alignment": "past-only zero-order hold with per-signal freshness limits",
            "nominal_rate_hz": expected_rate_hz,
            "maximum_gap_s": maximum_gap_s,
            "history_steps": 0,
            "horizon_s": horizon_s,
            "horizon_steps": horizon_steps,
            "stride_s": stride_s,
            "stride_steps": stride_steps,
        },
        "roles": {
            "initial_state_at_t0": list(STATE_COLUMNS),
            "initial_flap_state_at_t0": list(FLAP_STATE_COLUMNS),
            "optional_initial_context_at_t0": list(OPTIONAL_INITIAL_CONTEXT_COLUMNS),
            "known_future_control_t0_to_tT_exclusive": list(CONTROL_COLUMNS),
            "trajectory_target_t0_to_tT_inclusive": [*STATE_COLUMNS, *FLAP_STATE_COLUMNS],
            "future_forbidden_as_input": list(FUTURE_FORBIDDEN_INPUT_COLUMNS),
            "absolute_phase_reserved_columns_all_invalid_in_august": list(ABSOLUTE_PHASE_COLUMNS),
        },
        "frames": {
            "position_velocity_wind": "PX4 local NED; z and vz positive down",
            "attitude": "unit quaternion q_wxyz rotating body FRD vectors into local NED",
            "angular_velocity": "body FRD rad/s",
            "control": "dimensionless normalized post-allocation actuator outputs",
            "relative_phase": "log-local [0, 2pi); arbitrary zero from first finite encoder total_count",
        },
        "normalization": "not fitted or applied by the builder",
        "partitions": partition_summaries,
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_path / "quality_summary.json").write_text(
        json.dumps(partition_summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
