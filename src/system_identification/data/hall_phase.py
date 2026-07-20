from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pyulog import ULog


TWO_PI = 2.0 * np.pi
DEFAULT_PARTITIONS = ("train", "validation")
PARTITION_FILENAMES = {"train": "train_samples.parquet", "validation": "val_samples.parquet"}

# These columns are alternative phase coordinates or features derived from the
# stale logged phase.  The rebuilt table exposes one phase coordinate only:
# ``mechanical_phase_rad``.
LEGACY_PHASE_COLUMNS = {
    "encoder_phase_rad",
    "encoder_phase_unwrapped_rad",
    "drive_phase_rad",
    "drive_phase_unwrapped_rad",
    "drive_phase_sin",
    "drive_phase_cos",
    "wing_phase.phase_rad",
    "wing_phase.phase_unwrapped_rad",
    "wing_phase.phase_age_s",
    "wing_phase.phase_valid",
    "phase_source",
    "phase_raw_rad",
    "phase_raw_unwrapped_rad",
    "mechanical_phase_unwrapped_rad",
    "mechanical_phase_source",
    "phase_corrected_rad",
    "phase_corrected_unwrapped_rad",
}

STALE_FREQUENCY_COLUMNS = {
    "flap_frequency_topic_hz",
    "wing_phase.flap_frequency_hz",
    "wing_phase.flap_frequency_age_s",
    "flap_frequency_hz_filt",
    "cycle_flap_frequency_hz",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _topic_frame(ulog: ULog, name: str, *, multi_id: int = 0) -> pd.DataFrame | None:
    dataset = next((item for item in ulog.data_list if item.name == name and item.multi_id == multi_id), None)
    if dataset is None:
        return None
    frame = pd.DataFrame({key: np.asarray(value) for key, value in dataset.data.items()})
    if frame.empty:
        return None
    return frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def _deduplicate_hall_events(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "pulse_count"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"hall_event missing required columns: {sorted(missing)}")
    hall = frame.loc[:, ["timestamp", "pulse_count"]].drop_duplicates()
    hall = hall.sort_values(["timestamp", "pulse_count"], kind="mergesort")
    hall = hall.drop_duplicates("pulse_count", keep="first").reset_index(drop=True)
    return hall


def _interpolate(source_t: np.ndarray, source_v: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    source_t = np.asarray(source_t, dtype=np.int64)
    source_v = np.asarray(source_v, dtype=float)
    target_t = np.asarray(target_t, dtype=np.int64)
    result = np.interp(target_t.astype(float), source_t.astype(float), source_v)
    outside = (target_t < source_t[0]) | (target_t > source_t[-1])
    result[outside] = np.nan
    return result


def _resample_corrected_frequency(
    flap_frequency: pd.DataFrame,
    target_timestamp_us: np.ndarray,
    *,
    logged_ratio: float,
    true_ratio: float,
) -> np.ndarray:
    if "frequency_hz" not in flap_frequency.columns:
        raise ValueError("flap_frequency missing frequency_hz")
    logged = _interpolate(
        flap_frequency["timestamp"].to_numpy(dtype=np.int64),
        flap_frequency["frequency_hz"].to_numpy(dtype=float),
        target_timestamp_us,
    )
    return logged * float(logged_ratio) / float(true_ratio)


def _hall_reconstruction(
    *,
    encoder: pd.DataFrame,
    hall_event: pd.DataFrame,
    target_timestamp_us: np.ndarray,
    frequency_hz: np.ndarray,
    counts_per_encoder_revolution: float,
    true_ratio: float,
    maximum_cycle_count_relative_error: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_encoder = {"timestamp", "total_count"}
    missing = required_encoder - set(encoder.columns)
    if missing:
        raise ValueError(f"encoder_count missing required columns: {sorted(missing)}")

    hall = _deduplicate_hall_events(hall_event)
    encoder_t = encoder["timestamp"].to_numpy(dtype=np.int64)
    encoder_count = encoder["total_count"].to_numpy(dtype=float)
    hall = hall.loc[hall["timestamp"].between(int(encoder_t[0]), int(encoder_t[-1]))].copy()
    if len(hall) < 2:
        raise ValueError("fewer than two Hall events overlap encoder_count")

    hall_t = hall["timestamp"].to_numpy(dtype=np.int64)
    hall_pulse = hall["pulse_count"].to_numpy(dtype=np.int64)
    hall_count = _interpolate(encoder_t, encoder_count, hall_t)
    target_count = _interpolate(encoder_t, encoder_count, target_timestamp_us)

    expected_counts = float(counts_per_encoder_revolution) * float(true_ratio)
    cycle_count_delta = np.diff(hall_count)
    cycle_duration_s = np.diff(hall_t).astype(float) * 1.0e-6
    pulse_delta = np.diff(hall_pulse)
    cycle_count_relative_error = np.abs(cycle_count_delta - expected_counts) / expected_counts
    valid_cycle = (
        (pulse_delta == 1)
        & np.isfinite(cycle_count_delta)
        & (cycle_count_delta > 0.0)
        & (cycle_count_relative_error <= float(maximum_cycle_count_relative_error))
        & np.isfinite(cycle_duration_s)
        & (cycle_duration_s > 0.0)
    )

    cycle_index = np.searchsorted(hall_t, target_timestamp_us, side="right") - 1
    has_complete_cycle = (cycle_index >= 0) & (cycle_index < len(hall_t) - 1)
    bounded_index = np.clip(cycle_index, 0, len(hall_t) - 2)
    sample_cycle_valid = has_complete_cycle & valid_cycle[bounded_index]

    delta_count = target_count - hall_count[bounded_index]
    phase_unwrapped = delta_count * TWO_PI / expected_counts
    phase_tolerance = TWO_PI * float(maximum_cycle_count_relative_error)
    phase_in_tolerance = (phase_unwrapped >= -phase_tolerance) & (phase_unwrapped <= TWO_PI + phase_tolerance)
    phase_valid = sample_cycle_valid & np.isfinite(phase_unwrapped) & phase_in_tolerance
    clipped_phase_count = int((phase_valid & ((phase_unwrapped < 0.0) | (phase_unwrapped >= TWO_PI))).sum())
    phase = np.clip(phase_unwrapped, 0.0, np.nextafter(TWO_PI, 0.0))
    phase[~phase_valid] = np.nan

    output = pd.DataFrame(
        {
            "mechanical_phase_rad": phase,
            "flap_frequency_hz": frequency_hz,
            "phase_valid": phase_valid,
            "cycle_id": np.where(has_complete_cycle, cycle_index, -1).astype(np.int64),
            "cycle_valid": sample_cycle_valid,
            "cycle_duration_s": np.where(
                has_complete_cycle,
                cycle_duration_s[bounded_index],
                np.nan,
            ),
        }
    )
    output["flap_active"] = np.isfinite(frequency_hz) & (frequency_hz > 0.1)

    quality = {
        "phase_method": "hall_event_encoder_count_interpolation",
        "hall_event_count": int(len(hall)),
        "candidate_cycle_count": int(len(valid_cycle)),
        "valid_cycle_count": int(valid_cycle.sum()),
        "invalid_cycle_count": int((~valid_cycle).sum()),
        "maximum_cycle_count_relative_error_observed": float(np.nanmax(cycle_count_relative_error)),
        "p95_cycle_count_relative_error": float(np.nanpercentile(cycle_count_relative_error, 95.0)),
        "phase_valid_row_count": int(phase_valid.sum()),
        "clipped_phase_row_count": clipped_phase_count,
    }
    return output, quality


def _fallback_reconstruction(
    *,
    encoder: pd.DataFrame,
    wing_phase: pd.DataFrame,
    target_timestamp_us: np.ndarray,
    frequency_hz: np.ndarray,
    logged_ratio: float,
    true_ratio: float,
    counts_per_encoder_revolution: float,
    maximum_cycle_count_relative_error: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"timestamp", "phase_unwrapped_rad", "phase_valid"}
    missing = required - set(wing_phase.columns)
    if missing:
        raise ValueError(f"wing_phase missing fallback columns: {sorted(missing)}")
    encoder_count_column = next(
        (column for column in ("encoder_total_count", "total_count") if column in wing_phase.columns),
        None,
    )
    if encoder_count_column is None:
        raise ValueError("wing_phase fallback cannot recover the Hall zero count")
    required_encoder = {"timestamp", "total_count"}
    missing_encoder = required_encoder - set(encoder.columns)
    if missing_encoder:
        raise ValueError(f"encoder_count missing fallback columns: {sorted(missing_encoder)}")

    source_t = wing_phase["timestamp"].to_numpy(dtype=np.int64)
    source_phase = wing_phase["phase_unwrapped_rad"].to_numpy(dtype=float) * float(logged_ratio) / float(true_ratio)
    reset_index = np.flatnonzero(np.diff(source_phase) < -np.pi) + 1
    if len(reset_index) < 2:
        raise ValueError("fewer than two Hall resets found in wing_phase fallback")

    expected_counts = float(counts_per_encoder_revolution) * float(true_ratio)
    logged_counts_per_cycle = float(counts_per_encoder_revolution) * float(logged_ratio)
    published_encoder_count = wing_phase[encoder_count_column].to_numpy(dtype=float)
    logged_phase_unscaled = wing_phase["phase_unwrapped_rad"].to_numpy(dtype=float)
    # PX4 logged phase = (encoder_total_count - Hall_zero_count) * 2*pi /
    # (4096 * logged_ratio).  The reset publication is delayed, but the
    # embedded zero count is the encoder count captured by the Hall ISR.  It
    # therefore lets old logs recover the Hall crossing without extrapolating
    # the post-reset phase.
    hall_count = np.rint(
        published_encoder_count[reset_index]
        - logged_phase_unscaled[reset_index] * logged_counts_per_cycle / TWO_PI
    )

    encoder_t = encoder["timestamp"].to_numpy(dtype=np.int64)
    encoder_count = encoder["total_count"].to_numpy(dtype=float)
    reset_publication_t = source_t[reset_index]
    hall_t_float = np.full(len(hall_count), np.nan, dtype=float)
    for idx, (zero_count, publication_t) in enumerate(zip(hall_count, reset_publication_t)):
        center = int(np.searchsorted(encoder_t, publication_t, side="left"))
        start = max(0, center - 6)
        stop = min(len(encoder_t) - 1, center + 2)
        candidates: list[tuple[float, float]] = []
        for pair in range(start, stop):
            count_delta = encoder_count[pair + 1] - encoder_count[pair]
            if count_delta <= 0.0:
                continue
            fraction = (zero_count - encoder_count[pair]) / count_delta
            if 0.0 <= fraction <= 1.0:
                crossing_t = encoder_t[pair] + fraction * (encoder_t[pair + 1] - encoder_t[pair])
                # Hall processing publishes after the physical crossing.  A
                # 2 ms allowance covers topic timestamp quantisation without
                # selecting a later crossing elsewhere in the log.
                if crossing_t <= publication_t + 2_000.0:
                    candidates.append((abs(publication_t - crossing_t), crossing_t))
        if candidates:
            hall_t_float[idx] = min(candidates)[1]

    recoverable = np.isfinite(hall_t_float)
    hall_count = hall_count[recoverable]
    reset_index = reset_index[recoverable]
    hall_t_float = hall_t_float[recoverable]
    if len(hall_count) < 2:
        raise ValueError("fewer than two recovered Hall zero counts overlap encoder_count")
    hall_t = np.rint(hall_t_float).astype(np.int64)

    cycle_count_delta = np.diff(hall_count)
    cycle_count_relative_error = np.abs(cycle_count_delta - expected_counts) / expected_counts
    cycle_duration_s = np.diff(hall_t_float) * 1.0e-6
    valid_cycle = (
        np.isfinite(hall_t_float[:-1])
        & np.isfinite(hall_t_float[1:])
        & np.isfinite(cycle_duration_s)
        & (cycle_duration_s > 0.0)
        & np.isfinite(cycle_count_relative_error)
        & (cycle_count_delta > 0.0)
        & (cycle_count_relative_error <= float(maximum_cycle_count_relative_error))
    )
    cycle_index = np.searchsorted(hall_t, target_timestamp_us, side="right") - 1
    has_complete_cycle = (cycle_index >= 0) & (cycle_index < len(hall_t) - 1)
    bounded_index = np.clip(cycle_index, 0, len(hall_t) - 2)

    target_count = _interpolate(encoder_t, encoder_count, target_timestamp_us)
    phase_unwrapped = (target_count - hall_count[bounded_index]) * TWO_PI / expected_counts
    phase_tolerance = TWO_PI * float(maximum_cycle_count_relative_error)
    phase_valid = (
        has_complete_cycle
        & valid_cycle[bounded_index]
        & np.isfinite(phase_unwrapped)
        & (phase_unwrapped >= -phase_tolerance)
        & (phase_unwrapped <= TWO_PI + phase_tolerance)
    )
    clipped_phase_count = int((phase_valid & ((phase_unwrapped < 0.0) | (phase_unwrapped >= TWO_PI))).sum())
    phase = np.clip(phase_unwrapped, 0.0, np.nextafter(TWO_PI, 0.0))
    phase[~phase_valid] = np.nan

    output = pd.DataFrame(
        {
            "mechanical_phase_rad": phase,
            "flap_frequency_hz": frequency_hz,
            "phase_valid": phase_valid,
            "cycle_id": np.where(has_complete_cycle, cycle_index, -1).astype(np.int64),
            "cycle_valid": has_complete_cycle & valid_cycle[bounded_index],
            "cycle_duration_s": np.where(has_complete_cycle, cycle_duration_s[bounded_index], np.nan),
        }
    )
    output["flap_active"] = np.isfinite(frequency_hz) & (frequency_hz > 0.1)
    reset_delay_ms = (source_t[reset_index].astype(float) - hall_t_float) * 1.0e-3
    quality = {
        "phase_method": "logged_hall_zero_count_encoder_inversion_fallback",
        "hall_event_count": 0,
        "candidate_cycle_count": int(len(valid_cycle)),
        "valid_cycle_count": int(valid_cycle.sum()),
        "invalid_cycle_count": int((~valid_cycle).sum()),
        "recovered_hall_zero_count": int(len(hall_count)),
        "median_reset_publication_delay_ms": float(np.nanmedian(reset_delay_ms)),
        "p95_reset_publication_delay_ms": float(np.nanpercentile(reset_delay_ms, 95.0)),
        "maximum_cycle_count_relative_error_observed": float(np.nanmax(cycle_count_relative_error)),
        "p95_cycle_count_relative_error": float(np.nanpercentile(cycle_count_relative_error, 95.0)),
        "phase_valid_row_count": int(phase_valid.sum()),
        "clipped_phase_row_count": clipped_phase_count,
    }
    return output, quality


def reconstruct_log_phase_frequency(
    *,
    ulog_path: str | Path,
    target_timestamp_us: Iterable[int],
    logged_ratio: float = 7.5,
    true_ratio: float = 8.0,
    counts_per_encoder_revolution: float = 4096.0,
    maximum_cycle_count_relative_error: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct one Hall-indexed phase and one corrected flapping frequency."""

    if logged_ratio <= 0.0 or true_ratio <= 0.0:
        raise ValueError("logged_ratio and true_ratio must be positive")
    target = np.asarray(list(target_timestamp_us), dtype=np.int64)
    if target.ndim != 1 or len(target) == 0:
        raise ValueError("target_timestamp_us must be a non-empty one-dimensional sequence")
    if np.any(np.diff(target) < 0):
        raise ValueError("target_timestamp_us must be monotonic")

    path = Path(ulog_path).resolve()
    ulog = ULog(
        str(path),
        message_name_filter_list=["encoder_count", "hall_event", "wing_phase", "flap_frequency"],
        disable_str_exceptions=True,
    )
    parameter_ratio = float(ulog.initial_parameters.get("FLAP_RATIO", np.nan))
    if not np.isfinite(parameter_ratio) or not np.isclose(parameter_ratio, logged_ratio, rtol=0.0, atol=1.0e-6):
        raise ValueError(f"{path.name}: ULog FLAP_RATIO={parameter_ratio!r}, expected {logged_ratio}")

    encoder = _topic_frame(ulog, "encoder_count")
    wing_phase = _topic_frame(ulog, "wing_phase")
    flap_frequency = _topic_frame(ulog, "flap_frequency")
    hall_event = _topic_frame(ulog, "hall_event")
    if encoder is None or wing_phase is None or flap_frequency is None:
        raise ValueError(f"{path.name}: required encoder_count/wing_phase/flap_frequency topic missing")

    frequency_hz = _resample_corrected_frequency(
        flap_frequency,
        target,
        logged_ratio=logged_ratio,
        true_ratio=true_ratio,
    )
    if hall_event is not None:
        rebuilt, quality = _hall_reconstruction(
            encoder=encoder,
            hall_event=hall_event,
            target_timestamp_us=target,
            frequency_hz=frequency_hz,
            counts_per_encoder_revolution=counts_per_encoder_revolution,
            true_ratio=true_ratio,
            maximum_cycle_count_relative_error=maximum_cycle_count_relative_error,
        )
    else:
        rebuilt, quality = _fallback_reconstruction(
            encoder=encoder,
            wing_phase=wing_phase,
            target_timestamp_us=target,
            frequency_hz=frequency_hz,
            logged_ratio=logged_ratio,
            true_ratio=true_ratio,
            counts_per_encoder_revolution=counts_per_encoder_revolution,
            maximum_cycle_count_relative_error=maximum_cycle_count_relative_error,
        )

    rebuilt.insert(0, "timestamp_us", target)
    quality.update(
        {
            "source_log_path": str(path),
            "source_log_sha256": _sha256_file(path),
            "logged_ratio": float(logged_ratio),
            "true_ratio": float(true_ratio),
            "frequency_scale": float(logged_ratio / true_ratio),
            "row_count": int(len(rebuilt)),
            "frequency_finite_row_count": int(np.isfinite(frequency_hz).sum()),
            "frequency_min_hz": float(np.nanmin(frequency_hz)),
            "frequency_max_hz": float(np.nanmax(frequency_hz)),
        }
    )
    return rebuilt, quality


def _rewrite_sample_frame(frame: pd.DataFrame, rebuilt: pd.DataFrame, *, dataset_id: str) -> pd.DataFrame:
    if frame["timestamp_us"].duplicated().any():
        raise ValueError("source sample timestamp_us is not unique within log")
    if rebuilt["timestamp_us"].duplicated().any():
        raise ValueError("rebuilt timestamp_us is not unique within log")

    drop_columns = sorted((LEGACY_PHASE_COLUMNS | STALE_FREQUENCY_COLUMNS) & set(frame.columns))
    base = frame.drop(columns=drop_columns + [column for column in rebuilt.columns if column != "timestamp_us" and column in frame])
    output = base.merge(rebuilt, on="timestamp_us", how="left", validate="one_to_one", sort=False)
    if len(output) != len(frame):
        raise ValueError("phase/frequency rewrite changed row count")
    output["dataset_id"] = dataset_id
    output["flap_frequency_hz_source"] = "ulog_flap_frequency_scaled_7p5_to_8p0"
    output["wing_stroke_angle_rad"] = np.deg2rad(30.0) * np.sin(output["mechanical_phase_rad"])
    output["wing_stroke_angle_deg"] = np.rad2deg(output["wing_stroke_angle_rad"])
    derivative = np.cos(output["mechanical_phase_rad"].to_numpy(dtype=float))
    output["wing_stroke_direction"] = np.where(derivative >= 0.0, "upstroke", "downstroke")
    return output


def _git_identity(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=project_root, text=True).strip()

    try:
        return {
            "branch": run("branch", "--show-current"),
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--short")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"branch": None, "commit": None, "dirty": None}


def build_hall_ratio8_dataset(
    *,
    source_dataset_root: str | Path,
    accepted_logs_csv: str | Path,
    output_root: str | Path,
    partitions: Iterable[str] = DEFAULT_PARTITIONS,
    logged_ratio: float = 7.5,
    true_ratio: float = 8.0,
    counts_per_encoder_revolution: float = 4096.0,
    maximum_cycle_count_relative_error: float = 0.01,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Rewrite train/validation samples with one corrected mechanical phase."""

    source_root = Path(source_dataset_root).resolve()
    accepted_path = Path(accepted_logs_csv).resolve()
    output = Path(output_root).resolve()
    selected = tuple(str(item) for item in partitions)
    if not selected or any(item not in PARTITION_FILENAMES for item in selected):
        raise ValueError("partitions must be a non-empty subset of train/validation; test is forbidden")
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output root already exists and is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    accepted = pd.read_csv(accepted_path)
    required_accepted = {"log_id", "source_log_path"}
    missing = required_accepted - set(accepted.columns)
    if missing:
        raise ValueError(f"accepted_logs.csv missing columns: {sorted(missing)}")
    log_paths = dict(zip(accepted["log_id"].astype(str), accepted["source_log_path"].astype(str)))

    input_hashes: dict[str, str] = {str(accepted_path): _sha256_file(accepted_path)}
    output_paths: dict[str, Path] = {}
    quality_rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    source_manifest = source_root / "dataset_manifest.json"
    if source_manifest.exists():
        input_hashes[str(source_manifest)] = _sha256_file(source_manifest)

    for partition in selected:
        filename = PARTITION_FILENAMES[partition]
        source_path = source_root / filename
        input_hashes[str(source_path)] = _sha256_file(source_path)
        frame = pd.read_parquet(source_path)
        required = {"log_id", "timestamp_us"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{source_path} missing columns: {sorted(missing)}")

        rewritten_logs: list[pd.DataFrame] = []
        for log_id, group in frame.groupby("log_id", sort=True):
            log_key = str(log_id)
            if log_key not in log_paths:
                raise KeyError(f"source ULog missing for {log_key}")
            ordered = group.sort_values("timestamp_us", kind="mergesort").copy()
            rebuilt, quality = reconstruct_log_phase_frequency(
                ulog_path=log_paths[log_key],
                target_timestamp_us=ordered["timestamp_us"].to_numpy(dtype=np.int64),
                logged_ratio=logged_ratio,
                true_ratio=true_ratio,
                counts_per_encoder_revolution=counts_per_encoder_revolution,
                maximum_cycle_count_relative_error=maximum_cycle_count_relative_error,
            )
            if "flap_frequency_hz" not in ordered.columns:
                raise ValueError(f"{source_path} missing canonical logged flap_frequency_hz")
            corrected_frequency = (
                pd.to_numeric(ordered["flap_frequency_hz"], errors="coerce").to_numpy(dtype=float)
                * float(logged_ratio)
                / float(true_ratio)
            )
            rebuilt["flap_frequency_hz"] = corrected_frequency
            rebuilt["flap_active"] = np.isfinite(corrected_frequency) & (corrected_frequency > 0.1)
            quality["frequency_finite_row_count"] = int(np.isfinite(corrected_frequency).sum())
            quality["frequency_min_hz"] = float(np.nanmin(corrected_frequency))
            quality["frequency_max_hz"] = float(np.nanmax(corrected_frequency))
            rewritten = _rewrite_sample_frame(ordered, rebuilt, dataset_id=output.name)
            rewritten_logs.append(rewritten)
            quality_rows.append({"partition": partition, "log_id": log_key, **quality})

        output_frame = pd.concat(rewritten_logs, ignore_index=True)
        output_path = output / filename
        output_frame.to_parquet(output_path, index=False)
        output_paths[partition] = output_path
        split_counts[partition] = int(len(output_frame))

    for filename in ("all_logs.csv", "train_logs.csv", "val_logs.csv"):
        source = source_root / filename
        if source.exists():
            logs = pd.read_csv(source)
            if "dataset_id" in logs.columns:
                logs["dataset_id"] = output.name
            if "split" in logs.columns:
                logs["included_in_data"] = logs["split"].astype(str).isin(selected)
            logs.to_csv(output / filename, index=False)

    quality = pd.DataFrame(quality_rows).sort_values(["partition", "log_id"], kind="mergesort")
    quality_path = output / "phase_frequency_quality.csv"
    quality.to_csv(quality_path, index=False)
    first_output = pd.read_parquet(output_paths[selected[0]])
    phase_columns = [
        column
        for column in first_output.columns
        if column.endswith("phase_rad") or column.endswith("phase_unwrapped_rad")
    ]
    phase_values = first_output["mechanical_phase_rad"].to_numpy(dtype=float)
    all_phase_finite = bool(np.isfinite(phase_values).all())
    all_phase_in_range = bool(((phase_values >= 0.0) & (phase_values < TWO_PI)).all())
    all_cycles_valid = bool(first_output["cycle_valid"].astype(bool).all())
    for partition in selected[1:]:
        partition_frame = pd.read_parquet(
            output_paths[partition],
            columns=["mechanical_phase_rad", "cycle_valid"],
        )
        values = partition_frame["mechanical_phase_rad"].to_numpy(dtype=float)
        all_phase_finite &= bool(np.isfinite(values).all())
        all_phase_in_range &= bool(((values >= 0.0) & (values < TWO_PI)).all())
        all_cycles_valid &= bool(partition_frame["cycle_valid"].astype(bool).all())
    input_hashes_unchanged = bool(all(_sha256_file(Path(path)) == digest for path, digest in input_hashes.items()))
    strict_checks = {
        "only_one_phase_column": phase_columns == ["mechanical_phase_rad"],
        "all_phase_rows_finite": all_phase_finite,
        "all_phase_rows_in_half_open_range": all_phase_in_range,
        "all_exported_cycles_valid": all_cycles_valid,
        "all_phase_methods_known": bool(quality["phase_method"].isin(
            ["hall_event_encoder_count_interpolation", "logged_hall_zero_count_encoder_inversion_fallback"]
        ).all()),
        "all_frequency_rows_finite": bool((quality["frequency_finite_row_count"] == quality["row_count"]).all()),
        "input_hashes_unchanged": input_hashes_unchanged,
        "test_labels_loaded": False,
    }
    strict_checks["pass"] = bool(
        all(bool(value) for key, value in strict_checks.items() if key != "test_labels_loaded")
        and strict_checks["test_labels_loaded"] is False
    )
    checks_path = output / "quality_checks.json"
    checks_path.write_text(json.dumps(strict_checks, indent=2, sort_keys=True), encoding="utf-8")

    project_root = Path(__file__).resolve().parents[3]
    manifest = {
        "schema_version": "canonical_hall_ratio8_phase_frequency_v1",
        "dataset_id": output.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_root": str(source_root),
        "accepted_logs_csv": str(accepted_path),
        "included_partitions": list(selected),
        "excluded_partitions": [item for item in ("train", "validation", "test") if item not in selected],
        "test_labels_loaded": False,
        "split_sample_counts": split_counts,
        "phase_column": "mechanical_phase_rad",
        "phase_range": "[0, 2*pi)",
        "phase_zero": "Hall event: neutral wing position starting upstroke",
        "phase_endpoint_policy": "left_closed_right_open",
        "phase_primary_method": "Hall timestamp plus interpolated encoder total_count",
        "phase_fallback_method": "Hall zero count recovered from logged phase, then inverted on encoder total_count",
        "legacy_phase_columns_exported": False,
        "frequency_column": "flap_frequency_hz",
        "frequency_formula": "logged_flap_frequency_hz * logged_ratio / true_ratio",
        "logged_ratio": float(logged_ratio),
        "true_ratio": float(true_ratio),
        "counts_per_encoder_revolution": float(counts_per_encoder_revolution),
        "counts_per_wing_cycle": float(counts_per_encoder_revolution * true_ratio),
        "maximum_cycle_count_relative_error": float(maximum_cycle_count_relative_error),
        "phase_frequency_quality_csv": str(quality_path),
        "quality_checks_json": str(checks_path),
        "input_hashes": input_hashes,
        "git": _git_identity(project_root),
        "python_version": sys.version,
    }
    manifest_path = output / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "output_root": output,
        "manifest_path": manifest_path,
        "quality_path": quality_path,
        "quality_checks_path": checks_path,
        **{f"{partition}_samples_path": path for partition, path in output_paths.items()},
    }
