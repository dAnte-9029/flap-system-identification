"""Materialize the frozen attitude-aware DeLaurier prior for canonical splits.

The authoritative output is wing-only. Forces are expressed in body FRD at
the canonical IMU origin; moments are expressed in body FRD about the measured
aircraft CG. The exporter never reads a test partition unless a future,
separately reviewed API explicitly adds that capability.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Sequence

import numpy as np
import pandas as pd

from system_identification.physics.baselines.wing_only import (
    ATTITUDE_AIRFLOW_REQUIRED_COLUMNS,
    ISAACLAB_SOURCE_BRANCH,
    ISAACLAB_SOURCE_COMMIT,
    ISAACLAB_SOURCE_REPOSITORY,
    TARGETS,
    baseline_config_from_aircraft_metadata,
    evaluate_wing_only_delaurier_segment,
)


AUTHORITATIVE_PRIOR_ID = "delaurier_attitude_aware_3b5d4ec_trainval_v1"
AUTHORITATIVE_THETA_TIP_DEG = 0.0
OUTPUT_SCHEMA_VERSION = "delaurier_keyed_prior_v2"
KEY_COLUMNS = ("dataset_id", "log_id", "segment_id", "time_s", "timestamp_us")
PARTITION_FILENAMES = {"train": "train_samples.parquet", "validation": "val_samples.parquet"}
PREDICTION_FILENAMES = {"train": "train_predictions.parquet", "validation": "val_predictions.parquet"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_status(project_root: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=project_root, text=True).strip()

    status = run("status", "--porcelain=v1")
    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "dirty_paths": status.splitlines(),
    }


def _validate_source_rows(frame: pd.DataFrame, *, partition: str) -> None:
    required = {
        "log_id",
        "segment_id",
        "time_s",
        "timestamp_us",
        "mechanical_phase_rad",
        "flap_frequency_hz",
        "vehicle_air_data.rho",
        *ATTITUDE_AIRFLOW_REQUIRED_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{partition}: missing authoritative prior inputs: {missing}")
    if frame.duplicated(["log_id", "timestamp_us"], keep=False).any():
        raise ValueError(f"{partition}: duplicate log_id/timestamp_us keys")
    numeric = sorted(required - {"log_id", "segment_id"})
    invalid = ~np.isfinite(frame[numeric].to_numpy(dtype=float)).all(axis=1)
    if invalid.any():
        raise ValueError(f"{partition}: {int(invalid.sum())} rows contain non-finite authoritative prior inputs")
    timestamp = frame["timestamp_us"].to_numpy(dtype=float)
    if not np.allclose(timestamp, np.round(timestamp), atol=0.0, rtol=0.0):
        raise ValueError(f"{partition}: timestamp_us must contain integer microseconds")
    for identity, group in frame.groupby(["log_id", "segment_id"], sort=False, dropna=False):
        time_s = group["time_s"].to_numpy(dtype=float)
        timestamp_s = group["timestamp_us"].to_numpy(dtype=float) * 1.0e-6
        delta_time = np.diff(time_s)
        delta_timestamp = np.diff(timestamp_s)
        if np.any(delta_time <= 0.0) or np.any(delta_timestamp <= 0.0):
            raise ValueError(f"{partition}: non-monotonic time in log/segment {identity!r}")
        if not np.allclose(delta_time, delta_timestamp, atol=2.0e-6, rtol=1.0e-5):
            raise ValueError(
                f"{partition}: timestamp_us increments disagree with time_s in log/segment {identity!r}"
            )


def _evaluate_partition(
    samples: pd.DataFrame,
    *,
    partition: str,
    metadata_path: Path,
    geometry_path: Path,
    chunk_size: int,
) -> pd.DataFrame:
    _validate_source_rows(samples, partition=partition)
    config = baseline_config_from_aircraft_metadata(
        metadata_path,
        chunk_size=chunk_size,
        airflow_mode="attitude_ground_wind_3d",
    )
    pieces: list[pd.DataFrame] = []
    for _, group in samples.groupby(["log_id", "segment_id"], sort=False, dropna=False):
        evaluated = evaluate_wing_only_delaurier_segment(
            group,
            theta_tip_deg=(AUTHORITATIVE_THETA_TIP_DEG,),
            geometry_path=geometry_path,
            config=config,
            phase_acceleration_mode="constant_frequency_step",
        )
        keys = [column for column in KEY_COLUMNS if column in evaluated.columns]
        prediction = evaluated.loc[:, [*keys, *(f"pred_{target}" for target in TARGETS)]].rename(
            columns={f"pred_{target}": target for target in TARGETS}
        )
        pieces.append(prediction)
    result = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if len(result) != len(samples):
        raise ValueError(f"{partition}: output row mismatch {len(result)} != {len(samples)}")
    if result.duplicated(["log_id", "timestamp_us"], keep=False).any():
        raise ValueError(f"{partition}: exporter produced duplicate log_id/timestamp_us keys")
    source_keys = samples[["log_id", "timestamp_us"]]
    output_keys = result[["log_id", "timestamp_us"]]
    joined = source_keys.merge(output_keys, how="outer", on=["log_id", "timestamp_us"], indicator=True)
    if not (joined["_merge"] == "both").all():
        raise ValueError(f"{partition}: exporter did not preserve the source key set")
    return result


def materialize_authoritative_delaurier_prior(
    *,
    dataset_root: str | Path,
    output_root: str | Path,
    metadata_path: str | Path,
    geometry_path: str | Path,
    partitions: Sequence[str] = ("train", "validation"),
    chunk_size: int = 4096,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    """Write immutable train/validation keyed predictions and their manifest."""

    requested = tuple("validation" if value == "val" else str(value) for value in partitions)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("partitions must be a non-empty unique sequence")
    unsupported = sorted(set(requested) - set(PARTITION_FILENAMES))
    if unsupported:
        raise ValueError(f"Only train/validation may be materialized by this API: {unsupported}")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")

    dataset = Path(dataset_root).resolve()
    output = Path(output_root).resolve()
    metadata = Path(metadata_path).resolve()
    geometry = Path(geometry_path).resolve()
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[4]
    for path, label in ((dataset, "dataset root"), (metadata, "aircraft metadata"), (geometry, "wing geometry")):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prior artifact: {output}")
    output.mkdir(parents=True, exist_ok=False)

    row_counts: dict[str, int] = {}
    log_ids: dict[str, list[str]] = {}
    prediction_hashes: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    try:
        for partition in requested:
            source_path = dataset / PARTITION_FILENAMES[partition]
            if not source_path.is_file():
                raise FileNotFoundError(f"{partition} samples not found: {source_path}")
            samples = pd.read_parquet(source_path)
            predictions = _evaluate_partition(
                samples,
                partition=partition,
                metadata_path=metadata,
                geometry_path=geometry,
                chunk_size=int(chunk_size),
            )
            prediction_path = output / PREDICTION_FILENAMES[partition]
            predictions.to_parquet(prediction_path, index=False)
            row_counts[partition] = int(len(predictions))
            log_ids[partition] = sorted(predictions["log_id"].astype(str).unique().tolist())
            prediction_hashes[partition] = _sha256(prediction_path)
            source_hashes[partition] = _sha256(source_path)

        dataset_manifest = dataset / "dataset_manifest.json"
        config = baseline_config_from_aircraft_metadata(
            metadata,
            chunk_size=int(chunk_size),
            airflow_mode="attitude_ground_wind_3d",
        )
        manifest: dict[str, object] = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "artifact_id": AUTHORITATIVE_PRIOR_ID,
            "lifecycle_status": "active",
            "authoritative_for": "longitudinal_force_analysis",
            "legacy": False,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset_root": str(dataset),
            "dataset_manifest": str(dataset_manifest.resolve()),
            "dataset_manifest_sha256": _sha256(dataset_manifest),
            "source_partition_sha256": source_hashes,
            "partitions": list(requested),
            "row_counts": row_counts,
            "log_ids": log_ids,
            "test_partition_loaded": False,
            "test_rows_loaded": 0,
            "alignment_keys": ["log_id", "timestamp_us"],
            "prediction_columns": list(TARGETS),
            "prediction_sha256": prediction_hashes,
            "force_prior_semantics": "two-wing attached-flow DeLaurier force in body FRD at IMU origin",
            "moment_prior_semantics": "two-wing strip-integrated DeLaurier moment in body FRD about measured CG",
            "physics_source": {
                "repository": ISAACLAB_SOURCE_REPOSITORY,
                "branch": ISAACLAB_SOURCE_BRANCH,
                "commit": ISAACLAB_SOURCE_COMMIT,
            },
            "contracts": {
                "frame_contract": "body_frd_force_at_imu_origin_moment_about_cg",
                "airflow_contract": "attitude_ground_wind_3d",
                "phase_contract": "canonical_mechanical_phase_to_delaurier_v1",
                "stroke_contract": "q=A*cos(phi_D), phi_D=canonical mechanical phase",
                "dynamic_twist_contract": "disabled_zero_tip_amplitude",
                "separation_contract": "disabled_attached_flow",
            },
            "airflow_mode": "attitude_ground_wind_3d",
            "phase_acceleration_mode": "constant_frequency_step",
            "dynamic_twist_mode": "disabled",
            "theta_tip_deg": AUTHORITATIVE_THETA_TIP_DEG,
            "enable_separation": False,
            "wing_geom_csv": str(geometry),
            "wing_geom_sha256": _sha256(geometry),
            "metadata_path": str(metadata),
            "metadata_sha256": _sha256(metadata),
            "delaurier_parameters": {
                "alpha0_deg": float(np.degrees(config.params.alpha0_rad)),
                "eta_s": float(config.params.eta_s),
                "cd_cf": float(config.params.cd_cf),
                "alpha_stall_min_deg": float(np.degrees(config.params.alpha_stall_min_rad)),
                "alpha_stall_max_deg": float(np.degrees(config.params.alpha_stall_max_rad)),
                "xi": float(config.params.xi),
                "c_mac": float(config.params.c_mac),
                "cd_f": None if config.params.cd_f is None else float(config.params.cd_f),
                "theta_w_deg": float(np.degrees(config.mean_pitch_offset_rad)),
                "twist_eta_max_deg": AUTHORITATIVE_THETA_TIP_DEG,
                "enable_separation": False,
            },
            "baseline_config": asdict(config),
            "execution_backend": "numpy_cpu_vectorized",
            "chunk_size": int(chunk_size),
            "generator_git": _git_status(root),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return manifest
    except Exception:
        # Keep a failed directory visible for diagnosis, but never leave it
        # looking like a complete artifact.
        (output / "INCOMPLETE").write_text("authoritative prior materialization failed\n", encoding="utf-8")
        raise
