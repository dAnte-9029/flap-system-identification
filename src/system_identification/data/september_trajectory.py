"""Frozen September train/validation extraction with observed phase cohorts."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pyulog import ULog

from system_identification.data.trajectory_dataset import (
    CONTROL_COLUMNS, FUTURE_FORBIDDEN_INPUT_COLUMNS, STATE_COLUMNS,
    _duration_s, assign_contiguous_segments, build_window_index,
    extract_trajectory_samples, validate_split_assignments,
)
from system_identification.data.ulg_audit import _dataset


PHASE_COLUMNS = ("logged_flap_phase_rad", "logged_flap_phase_sin", "logged_flap_phase_cos")
CRITICAL_TOPICS = (
    "vehicle_local_position", "vehicle_attitude", "vehicle_angular_velocity",
    "actuator_motors", "actuator_servos", "encoder_count", "flap_frequency",
    "wing_phase", "vehicle_status", "vehicle_land_detected",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publication_hold(reference_us, data, field, freshness_s):
    """As-of publication join; never backdate availability using timestamp_sample."""
    ref = np.asarray(reference_us, dtype=np.int64)
    values = np.full(len(ref), np.nan)
    valid = np.zeros(len(ref), dtype=bool)
    source_us = np.full(len(ref), -1, dtype=np.int64)
    if data is None or field not in data or not len(data["timestamp"]):
        return values, valid, source_us
    ts = np.asarray(data["timestamp"], dtype=np.int64)
    if np.any(np.diff(ts) <= 0):
        raise ValueError(f"non-increasing publication timestamps for {field}")
    index = np.searchsorted(ts, ref, side="right") - 1
    exists = index >= 0
    idx = np.clip(index, 0, len(ts) - 1)
    source_us[exists] = ts[idx[exists]]
    values[exists] = np.asarray(data[field], dtype=float)[idx[exists]]
    valid = exists & np.isfinite(values) & ((ref - source_us) <= freshness_s * 1e6)
    return values, valid, source_us


def observed_phase(reference_us, wing, hall):
    phase, fresh, phase_time = publication_hold(reference_us, wing, "phase_rad", 0.1)
    flag, flag_fresh, _ = publication_hold(reference_us, wing, "phase_valid", 0.1)
    valid = fresh & flag_fresh & (flag > 0.5) & (phase >= 0) & (phase < 2 * np.pi)
    pulse, hall_fresh, hall_time = publication_hold(phase_time, hall, "pulse_count", 3.0)
    hall_valid = valid & hall_fresh & (pulse > 0)
    phase[~valid] = np.nan
    return pd.DataFrame({
        "logged_flap_phase_rad": phase,
        "logged_flap_phase_sin": np.sin(phase),
        "logged_flap_phase_cos": np.cos(phase),
        "valid_logged_phase": valid,
        "hall_reference_valid": hall_valid,
        "phase_source_timestamp_us": phase_time,
        "hall_source_timestamp_us": hall_time,
    })


def cohort_samples(samples: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Retain core breaks and add phase-availability breaks for each cohort."""
    if cohort not in {"core", "logged_phase", "hall_phase"}:
        raise ValueError(f"unknown cohort: {cohort}")
    frame = samples.copy()
    valid = frame["valid_core"].to_numpy(dtype=bool).copy()
    if cohort != "core":
        column = "valid_logged_phase" if cohort == "logged_phase" else "hall_reference_valid"
        valid &= frame[column].to_numpy(dtype=bool)
    frame["valid_core"] = valid
    frame["segment_id"] = -1
    frame["sample_in_segment"] = -1
    for _, log in frame.groupby("log_id", sort=False):
        segments = assign_contiguous_segments(
            log.timestamp_us.to_numpy(), log.valid_core.to_numpy(),
            expected_dt_us=20_000, maximum_gap_us=50_000,
        )
        frame.loc[log.index, "segment_id"] = segments
        for segment in np.unique(segments[segments >= 0]):
            indices = log.index[segments == segment]
            frame.loc[indices, "sample_in_segment"] = np.arange(len(indices))
    return frame


def hall_diagnostics(encoder, hall, ratio):
    if hall is None:
        return {"status": "no_logged_hall", "events": 0}
    et = np.asarray(encoder["timestamp"], dtype=np.int64)
    ht = np.asarray(hall["timestamp"], dtype=np.int64)
    pulse = np.asarray(hall["pulse_count"], dtype=np.int64)
    overlap = (ht >= et[0]) & (ht <= et[-1])
    ht, pulse = ht[overlap], pulse[overlap]
    if len(ht) < 2:
        return {"status": "insufficient_overlap", "events": len(ht)}
    indices = np.searchsorted(et, ht)
    indices = np.clip(indices, 1, len(et) - 1)
    bracket_ok = (et[indices] - et[indices - 1]) <= 50_000
    count = np.interp(ht, et, encoder["total_count"])
    dt = np.diff(ht) * 1e-6
    dc = np.diff(count)
    good = (np.diff(pulse) == 1) & (dt > .05) & (dt < 1) & (dc > 0)
    good &= bracket_ok[:-1] & bracket_ok[1:]
    revolutions = dc[good] / 4096
    if not len(revolutions):
        return {"status": "no_valid_cycle_diagnostic", "events": len(ht)}
    median = float(np.median(revolutions))
    if abs(median / ratio - 1) > .01:
        raise ValueError(f"Hall/encoder ratio mismatch: {median} versus {ratio}")
    return {"status": "median_ratio_consistent", "events": len(ht),
            "cycle_count": int(good.sum()), "revolutions_p01_p50_p99":
            np.quantile(revolutions, [.01, .5, .99]).tolist(),
            "use": "offline provenance diagnostic only; not input or selection mask"}


def _validate_config(config):
    splits = validate_split_assignments(config["partitions"])
    expected = {"train": ["2026-09-06"], "validation": ["2026-09-07"], "sealed_test": ["2026-09-08"]}
    if splits["split_dates"] != expected:
        raise ValueError("September day split does not match frozen contract")
    paths = [p for items in splits["assignments"].values() for p in items]
    if any(Path(p).is_absolute() or ".." in Path(p).parts for p in paths):
        raise ValueError("source paths must be contained relative paths")
    if set(paths) & set(config["excluded_logs"]):
        raise ValueError("excluded log appears in admitted inventory")
    if config["sampling"] != {"nominal_rate_hz": 50, "maximum_gap_s": .05,
                               "horizons_s": [1, 2, 3, 5], "stride_s": .2}:
        raise ValueError("sampling differs from frozen v2 contract")
    return splits


def build_september_dataset(config_path: Path, repository: Path, output_root: Path | None = None):
    config = yaml.safe_load(config_path.read_text())
    splits = _validate_config(config)
    output = output_root or repository / config["output_root"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    source = Path(config["source_root"])
    audit_path = repository / config["source_audit"]
    manifest = {
        "dataset_version": "trajectory_dataset_v2_september",
        "dataset_id": config["dataset_id"], "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"root": str(source), "config_path": str(config_path.resolve()),
                   "config_sha256": file_hash(config_path), "audit_path": str(audit_path),
                   "audit_sha256": file_hash(audit_path), "ulog_sha256": {}},
        "split_contract": {**splits, "materialized_partitions": ["train", "validation"],
                           "sealed_test_opened": False,
                           "test_prior_exposure": "2026-09-08 descriptive quality audit only; now sealed"},
        "sampling": {**config["sampling"], "timestamp_basis": "publication",
                     "alignment": "past-only zero-order hold", "initial_history_requirement_s": 0},
        "phase_contract": config["phase_zero"],
        "frequency_contract": {"column": "flap_frequency_hz", "ratio": config["transmission_ratio"],
                               "rescaling": "none; September logged ratio, not April ratio8"},
        "roles": {"initial_state": list(STATE_COLUMNS), "optional_observed_phase": list(PHASE_COLUMNS),
                  "known_future_control_t0_to_tT_exclusive": list(CONTROL_COLUMNS),
                  "future_forbidden_as_input": [*FUTURE_FORBIDDEN_INPUT_COLUMNS, *PHASE_COLUMNS,
                      "valid_logged_phase", "phase_source_timestamp_us", "hall_source_timestamp_us",
                      "state_sample_timestamp_us", "nav_state", "valid_core", "segment_id"]},
        "frames": {"position_velocity": "local NED m, m/s", "quaternion": "wxyz body FRD to NED",
                   "body_rate": "FRD rad/s", "phase": "radians, logged zero; physical pose unconfirmed"},
        "normalization": "none fitted", "excluded_logs": config["excluded_logs"],
        "partitions": {}, "per_log": [], "artifact_sha256": {},
        "provenance": {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
                       "git_status": subprocess.check_output(["git", "status", "--short"], cwd=repository, text=True),
                       "python_executable": sys.executable,
                       "builder_source_sha256": {str(p.relative_to(repository)): file_hash(p) for p in
                           [Path(__file__), repository / "src/system_identification/data/trajectory_dataset.py",
                            repository / "src/system_identification/data/ulg_audit.py"]}},
    }
    # The explicit loop is also the test-data read boundary. Never traverse sealed_test.
    for partition in ("train", "validation"):
        frames = []
        for relative in splits["assignments"][partition]:
            path = source / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            before = file_hash(path)
            ulog = ULog(str(path))
            if ulog.msg_info_dict.get("ver_sw") != config["firmware"]:
                raise ValueError(f"firmware mismatch: {relative}")
            required_parameters = {**config["structural_parameters"], **config["output_mapping"]}
            for key, value in required_parameters.items():
                if ulog.initial_parameters.get(key) != value:
                    raise ValueError(f"parameter mismatch in {relative}: {key}")
            if ulog.changed_parameters:
                raise ValueError(f"parameters changed during log: {relative}")
            topics = {}
            timing = {}
            for name in (*CRITICAL_TOPICS, "hall_event"):
                dataset = _dataset(ulog, name)
                if dataset is None:
                    if name in CRITICAL_TOPICS:
                        raise ValueError(f"missing required topic {name}: {relative}")
                    topics[name] = None
                    continue
                data = dataset.data
                ts = np.asarray(data["timestamp"], dtype=np.int64)
                if len(ts) == 0 or np.any(np.diff(ts) <= 0):
                    raise ValueError(f"invalid publication timestamps in {relative}: {name}")
                topics[name] = data
                lag = ts.astype(float) - np.asarray(data.get("timestamp_sample", ts), dtype=float)
                timing[name] = {"publication_lag_ms_p50_p99": (np.quantile(lag, [.5, .99]) / 1000).tolist(),
                                "maximum_gap_s": float(np.max(np.diff(ts)) / 1e6) if len(ts)>1 else None}
            frame = extract_trajectory_samples(path, log_id=relative, split=partition,
                transmission_ratio=config["transmission_ratio"], timestamp_basis="publication")
            frame["state_sample_timestamp_us"] = topics["vehicle_local_position"].get(
                "timestamp_sample", topics["vehicle_local_position"]["timestamp"])
            phases = observed_phase(frame.timestamp_us.to_numpy(), topics["wing_phase"], topics["hall_event"])
            for column in phases:
                frame[column] = phases[column].to_numpy()
            core_n = int(frame.valid_core.sum())
            if core_n == 0:
                raise ValueError(f"no valid core samples: {relative}")
            diagnostic = hall_diagnostics(topics["encoder_count"], topics["hall_event"], config["transmission_ratio"])
            after = file_hash(path)
            if before != after:
                raise ValueError(f"source changed during read: {relative}")
            manifest["source"]["ulog_sha256"][relative] = before
            manifest["per_log"].append({"log_id": relative, "partition": partition,
                "source_rows": len(frame), "core_rows": core_n, "valid_duration_s": _duration_s(frame),
                "logged_phase_on_core_ratio": float((frame.valid_core & frame.valid_logged_phase).sum()/core_n),
                "hall_phase_on_core_ratio": float((frame.valid_core & frame.hall_reference_valid).sum()/core_n),
                "exclusion_counts": frame.loc[~frame.valid_core,"exclusion_reason"].value_counts().to_dict(),
                "timing": timing, "hall": diagnostic})
            frames.append(frame)
        samples = pd.concat(frames, ignore_index=True)
        output.mkdir(parents=True, exist_ok=True)
        filename = f"samples_{partition}.parquet"
        samples.to_parquet(output / filename, index=False)
        manifest["artifact_sha256"][filename] = file_hash(output / filename)
        info = {"log_count": len(frames), "samples_file": filename, "source_rows": len(samples),
                "valid_rows": int(samples.valid_core.sum()), "cohorts": {}}
        for cohort in ("core", "logged_phase", "hall_phase"):
            selected = cohort_samples(samples, cohort)
            # Window references use stable source row keys. Cohort segment keys are separate.
            segments = selected.loc[selected.valid_core].groupby(["log_id", "segment_id"], sort=False)
            segment_start = {(log, int(seg)): int(group.timestamp_us.iloc[0]) for (log, seg), group in segments}
            entry = {"valid_duration_s": _duration_s(selected), "segment_count": len(segment_start), "windows": {}}
            for horizon in config["sampling"]["horizons_s"]:
                windows = build_window_index(selected, horizon_steps=horizon*50, stride_steps=10, dt_s=.02)
                if windows.empty:
                    raise ValueError(f"empty windows: {partition}/{cohort}/{horizon}s")
                windows["available_history_s"] = [(int(r.start_timestamp_us)-segment_start[(r.log_id,int(r.segment_id))])/1e6 for r in windows.itertuples()]
                windows["cohort"] = cohort
                windows["window_id"] = cohort + ":" + str(horizon) + "s:" + windows.window_id
                filename = f"windows_{partition}_{cohort}_{horizon}s.parquet"
                windows.to_parquet(output / filename, index=False)
                manifest["artifact_sha256"][filename] = file_hash(output / filename)
                entry["windows"][str(horizon)] = {"file": filename, "count": len(windows),
                    "with_0p5s_history": int((windows.available_history_s >= .5).sum()),
                    "with_1s_history": int((windows.available_history_s >= 1).sum()),
                    "max_horizon_error_s": float(abs(windows.observed_horizon_s-horizon).max())}
            info["cohorts"][cohort] = entry
        manifest["partitions"][partition] = info
    manifest["window_lookup"] = "join log_id and start/end_sample_in_log; segment_id is cohort-specific, not samples.segment_id"
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_registered_dataset(registry_path: Path, repository: Path):
    """Verify the explicit trajectory registry, including every materialized artifact."""
    registry = yaml.safe_load(registry_path.read_text())
    entry = registry["datasets"][registry["default_dataset_id"]]
    path = repository / entry["manifest_path"]
    if file_hash(path) != entry["manifest_sha256"]:
        raise ValueError("registered manifest hash mismatch")
    manifest = json.loads(path.read_text())
    if manifest["dataset_id"] != registry["default_dataset_id"]:
        raise ValueError("registered dataset identity mismatch")
    if manifest["split_contract"]["sealed_test_opened"]:
        raise ValueError("test materialization is forbidden in Step 1")
    if list(path.parent.glob("*test*.parquet")):
        raise ValueError("sealed test artifacts found")
    for name, digest in manifest["artifact_sha256"].items():
        if file_hash(path.parent / name) != digest:
            raise ValueError(f"artifact hash mismatch: {name}")
    return manifest
