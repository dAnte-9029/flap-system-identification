"""Native-time velocity/rate observations and explicit, autonomous-rollout data API.

No acceleration labels, differentiation, interpolation, network, or loss weights.
History and targets deliberately have separate namespaces and timing contracts.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
from pyulog import ULog
import yaml

from system_identification.data.trajectory_dataset import (
    STATE_COLUMNS, CONTROL_COLUMNS, relative_phase_from_total_count,
)

STATE = list(STATE_COLUMNS)
CONTROL = list(CONTROL_COLUMNS)
TARGET = ["target_" + c for c in STATE]
PHASE = ["relative_flap_phase_sin", "relative_flap_phase_cos", "flap_frequency_hz"]
RESET = ["xy_reset_counter", "z_reset_counter", "vxy_reset_counter",
         "vz_reset_counter", "heading_reset_counter", "ref_timestamp"]
FIELDS = {
    "vehicle_local_position": ["x", "y", "z", "vx", "vy", "vz", "xy_valid", "z_valid",
                               "v_xy_valid", "v_z_valid", *RESET, "eph", "epv", "evh", "evv",
                               "dead_reckoning"],
    "vehicle_attitude": [f"q[{i}]" for i in range(4)] + ["quat_reset_counter"],
    "vehicle_angular_velocity": [f"xyz[{i}]" for i in range(3)],
    "actuator_motors": ["control[0]", "reversible_flags"],
    "actuator_servos": [f"control[{i}]" for i in range(3)],
    "encoder_count": ["total_count", "position_raw", "device_id"],
    "flap_frequency": ["frequency_hz"],
    "wing_phase": ["encoder_total_count", "phase_rad", "phase_unwrapped_rad",
                   "flap_frequency_hz", "phase_valid"],
    "hall_event": ["pulse_count"],
    "vehicle_status": ["nav_state", "arming_state", "failsafe"],
}
PREFIX = {"vehicle_attitude": "attitude", "vehicle_angular_velocity": "rate",
          "actuator_motors": "motor", "actuator_servos": "servo",
          "encoder_count": "encoder", "flap_frequency": "frequency",
          "vehicle_status": "status", "wing_phase": "logged_phase", "hall_event": "hall"}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                              default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")


def quantiles(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return np.quantile(x, [0, .01, .5, .99, 1]).tolist() if len(x) else None


def flight_date(path: str) -> str:
    match = re.search(r"(2026)-(\d{1,2})-(\d{1,2})", Path(path).name)
    if match:
        y, m, d = map(int, match.groups())
    else:
        match = re.match(r"(\d{1,2})\.(\d{1,2})", Path(path).parts[0])
        if not match:
            raise ValueError(f"unknown date: {path}")
        y, (m, d) = 2026, map(int, match.groups())
    return f"{y:04d}-{m:02d}-{d:02d}"


def interval_overlap(starts, ends, intervals):
    """Closed intervals, including exclusions entirely between two samples."""
    starts, ends = np.asarray(starts), np.asarray(ends)
    answer = np.zeros(starts.shape, dtype=bool)
    for a, b in np.asarray(intervals, dtype=np.int64).reshape(-1, 2):
        answer |= (starts <= b) & (ends >= a)
    return answer


def past_indices(publication, reference):
    return np.searchsorted(np.asarray(publication), np.asarray(reference), side="right") - 1


def nearest_indices(measurement, reference):
    times, ref = np.asarray(measurement), np.asarray(reference)
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("supervision measurement times must be strictly increasing")
    right = np.clip(np.searchsorted(times, ref), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(np.abs(times[left] - ref) <= np.abs(times[right] - ref), left, right)


def rotate_ned_to_body(q, velocity):
    """Hamilton wxyz q_nb; conjugate rotation, no observation renormalization."""
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q, axis=1)
    unit = q / np.where(norm > 0, norm, np.nan)[:, None]
    w, v = unit[:, :1], -unit[:, 1:]
    t = 2 * np.cross(v, velocity)
    return velocity + w * t + np.cross(v, t)


def topic_frames(ulog):
    result = {}
    for name, fields in FIELDS.items():
        ds = [d for d in ulog.data_list if d.name == name and d.multi_id == 0]
        if not ds:
            if name == "hall_event":
                result[name] = pd.DataFrame(columns=["timestamp_us", "timestamp_sample_us", *fields])
                continue
            raise ValueError(f"missing required topic instance 0: {name}")
        d = ds[0].data
        ts = np.asarray(d["timestamp"], np.int64)
        if not len(ts) or np.any(np.diff(ts) <= 0):
            raise ValueError(f"nonmonotonic native publication/source times: {name}")
        values = {"timestamp_us": ts,
                  "timestamp_sample_us": np.asarray(d.get("timestamp_sample", np.full(len(ts), -1)),
                                                    dtype=np.int64)}
        for field in fields:
            values[field] = d.get(field, np.full(len(ts), np.nan))
        result[name] = pd.DataFrame(values)
    return result


def hold(topic, reference, fields, prefix, max_age):
    n = len(reference)
    if not len(topic):
        return np.full((n, len(fields)), np.nan), np.zeros(n, bool), {
            f"{prefix}_publication_us": np.full(n, -1, np.int64),
            f"{prefix}_measurement_us": np.full(n, -1, np.int64),
            f"{prefix}_age_us": np.full(n, -1, np.int64),
            f"{prefix}_native_dt_us": np.full(n, -1, np.int64),
        }
    availability = topic.get("availability_us", topic.timestamp_us).to_numpy(dtype=np.int64)
    idx = past_indices(availability, reference)
    exists = idx >= 0
    safe = np.maximum(idx, 0)
    source_time = topic.timestamp_us.to_numpy()[safe].copy()
    pub = availability[safe].copy()
    meas = topic.timestamp_sample_us.to_numpy()[safe].copy()
    values = topic[fields].to_numpy(dtype=float)[safe].copy()
    native_dt = np.r_[-1, np.diff(topic.timestamp_us)][safe]
    pub[~exists] = -1
    meas[~exists] = -1
    values[~exists] = np.nan
    source_time[~exists] = -1
    age = np.where(exists, reference - source_time, -1)
    valid = exists & (age <= max_age) & np.isfinite(values).all(axis=1)
    # For topics with separate measurement timestamps, never accept a sample from the future.
    valid &= (meas < 0) | ((meas > 0) & (meas <= pub))
    return values, valid, {f"{prefix}_publication_us": pub, f"{prefix}_measurement_us": meas,
                          f"{prefix}_source_timestamp_us": source_time,
                          f"{prefix}_age_us": age, f"{prefix}_native_dt_us": native_dt}


def paired_target(topic, measurement, fields, prefix, tolerance):
    ts = topic.timestamp_sample_us.to_numpy(dtype=np.int64)
    idx = nearest_indices(ts, measurement)
    values = topic[fields].to_numpy(dtype=float)[idx]
    offset = ts[idx] - measurement
    pub = topic.timestamp_us.to_numpy()[idx]
    valid = (np.abs(offset) <= tolerance) & np.isfinite(values).all(axis=1)
    valid &= (ts[idx] > 0) & (ts[idx] <= pub)
    return values, valid, {f"target_{prefix}_measurement_us": ts[idx],
                          f"target_{prefix}_publication_us": pub,
                          f"target_{prefix}_offset_us": offset}


def extract_flight(row, config, exclude, review):
    source = Path(config["source_root"]) / row.log_path
    if not source.resolve().is_relative_to(Path(config["source_root"]).resolve()):
        raise ValueError("source escapes approved root")
    if file_hash(source) != row.sha256:
        raise ValueError(f"source hash mismatch: {source}")
    ulog = ULog(str(source), message_name_filter_list=list(FIELDS))
    topics = topic_frames(ulog)
    # AS5600.Run is serial. The next acquisition timestamp is a strict upper
    # bound on completion of the previous Run's publications. No future value
    # is used. The last native record has no observed upper bound.
    for name in ("encoder_count", "flap_frequency"):
        ts = topics[name].timestamp_us.to_numpy()
        topics[name]["availability_us"] = np.r_[ts[1:], np.iinfo(np.int64).max]
    params = ulog.initial_parameters
    mapping_errors = {k: params.get(k) for k, v in config["required_mapping_parameters"].items()
                      if params.get(k) != v}
    if mapping_errors:
        raise ValueError(f"unreviewed control mapping: {source}: {mapping_errors}")
    ratio = float(params["FLAP_RATIO"])
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("invalid per-log FLAP_RATIO")
    structural_changes = [(t, k, v) for t, k, v in ulog.changed_parameters
                          if k.startswith(("FLAP_RATIO", "CA_", "PWM_"))]
    if structural_changes:
        raise ValueError(f"structural parameters change inside log: {source}")
    full_local = topics["vehicle_local_position"]
    local = full_local.loc[full_local.timestamp_us.between(row.start_us, row.end_us)].copy()
    local = local.loc[local.timestamp_sample_us.between(row.start_us, row.end_us)].copy()
    pub = local.timestamp_us.to_numpy(dtype=np.int64)
    meas = local.timestamp_sample_us.to_numpy(dtype=np.int64)
    if len(pub) < 3 or np.any(np.diff(meas) <= 0):
        raise ValueError(f"insufficient/nonmonotonic native states: {source}")
    rate = 1e6 / np.median(np.diff(meas))
    group = "main_50hz" if rate >= 30 else "sparse_10hz"
    policy = config["groups"][group]
    if rate < policy["minimum_rate_hz"]:
        raise ValueError(f"unsupported sampling rate: {rate}")
    record = dict(flight_id=row.flight_id, sha256=row.sha256, log_path=row.log_path,
                  flight_index=int(row.flight_index), flight_date=row.flight_date, split=row.split,
                  sampling_group=group, start_us=int(row.start_us), end_us=int(row.end_us),
                  reviewed_duration_s=float(row.duration_s), native_rate_hz=float(rate),
                  firmware=ulog.msg_info_dict.get("ver_sw"),
                  firmware_branch=ulog.msg_info_dict.get("ver_sw_branch"),
                  sys_uuid=ulog.msg_info_dict.get("sys_uuid"),
                  boot_time_utc_us=ulog.msg_info_dict.get("boot_time_utc_us"),
                  reconstructed_source="reconstructed" in row.log_path or "offset_" in row.log_path,
                  transmission_ratio=ratio, absolute_phase_zero_verified=False,
                  parameter_changes=ulog.changed_parameters,
                  parameters={k: v for k, v in params.items()
                              if k.startswith(("CA_", "PWM_", "FLAP_", "SYS_AUTOSTART", "SDLOG_"))})
    cols = dict(flight_id=row.flight_id, split=row.split, sampling_group=group,
                sample_in_flight=np.arange(len(pub)), timestamp_us=pub,
                state_measurement_us=meas, state_publication_us=pub,
                state_age_us=pub - meas, dt_measurement_s=np.r_[np.nan, np.diff(meas) * 1e-6],
                dt_publication_s=np.r_[np.nan, np.diff(pub) * 1e-6])
    pv = local[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=float)
    pv_valid = local[["xy_valid", "z_valid", "v_xy_valid", "v_z_valid"]].to_numpy().all(axis=1)
    pv_valid &= np.isfinite(pv).all(axis=1) & (meas <= pub)
    for col, value in zip(STATE[:6], pv.T):
        cols[col] = value
        cols["target_" + col] = value
    q, q_valid, timing = hold(topics["vehicle_attitude"], pub, [f"q[{i}]" for i in range(4)],
                             "attitude", policy["history_attitude_age_us"])
    cols.update(timing)
    q_valid &= np.abs(np.linalg.norm(q, axis=1) - 1) <= .001
    omega, w_valid, timing = hold(topics["vehicle_angular_velocity"], pub,
                                 [f"xyz[{i}]" for i in range(3)], "rate", config["history_rate_age_us"])
    cols.update(timing)
    tq, tq_valid, timing = paired_target(topics["vehicle_attitude"], meas,
                                        [f"q[{i}]" for i in range(4)], "attitude",
                                        policy["target_pair_tolerance_us"])
    cols.update(timing)
    tq_valid &= np.abs(np.linalg.norm(tq, axis=1) - 1) <= .001
    tw, tw_valid, timing = paired_target(topics["vehicle_angular_velocity"], meas,
                                        [f"xyz[{i}]" for i in range(3)], "rate",
                                        policy["target_pair_tolerance_us"])
    cols.update(timing)
    for col, value, target in zip(STATE[6:], np.c_[q, omega].T, np.c_[tq, tw].T):
        cols[col], cols["target_" + col] = value, target
    for axis, value, target in zip("xyz", rotate_ned_to_body(q, pv[:, 3:]).T,
                                   rotate_ned_to_body(tq, pv[:, 3:]).T):
        cols[f"velocity_body_m_s_{axis}"] = value
        cols[f"target_velocity_body_m_s_{axis}"] = target
    controls, valid_controls = [], []
    for topic, fields in (("actuator_motors", ["control[0]"]),
                          ("actuator_servos", ["control[0]", "control[1]", "control[2]"])):
        values, valid, timing = hold(topics[topic], pub, fields, PREFIX[topic],
                                     config["control_age_us"])
        cols.update(timing)
        controls.append(values)
        valid_controls.append(valid)
    action = np.column_stack(controls)
    control_valid = np.logical_and.reduce(valid_controls) & (action[:, 0] >= 0)
    control_valid &= (action[:, 0] <= 1) & (np.abs(action[:, 1:]) <= 1).all(axis=1)
    for col, value in zip(CONTROL, action.T):
        cols[col] = value
    enc = topics["encoder_count"]
    # Origin is fixed once per log, never re-zeroed per flight/window.
    origin_count, origin_us = float(enc.total_count.iloc[0]), int(enc.timestamp_us.iloc[0])
    enc = enc.copy()
    enc["relative_unwrapped"], enc["relative_phase"] = relative_phase_from_total_count(
        enc.total_count.to_numpy(), transmission_ratio=ratio,
        counts_per_motor_revolution=config["phase"]["counts_per_motor_revolution"])
    ev, e_valid, timing = hold(enc, pub, ["total_count", "position_raw", "relative_unwrapped",
                                         "relative_phase"], "encoder", config["encoder_age_us"])
    cols.update(timing)
    cols.update(encoder_total_count=ev[:, 0], encoder_position_raw=ev[:, 1],
                relative_flap_phase_unwrapped_rad=ev[:, 2], relative_flap_phase_rad=ev[:, 3],
                relative_flap_phase_sin=np.sin(ev[:, 3]), relative_flap_phase_cos=np.cos(ev[:, 3]),
                phase_origin_count=origin_count, phase_origin_timestamp_us=origin_us)
    fv, f_valid, timing = hold(topics["flap_frequency"], pub, ["frequency_hz"], "frequency",
                               config["frequency_age_us"])
    cols.update(timing)
    f_valid &= (fv[:, 0] >= 0) & (fv[:, 0] < 30)
    if not ulog.msg_info_dict.get("ver_sw"):
        # Retain raw observations, but do not assert the driver-specific
        # availability proof for a recovered log with unknown firmware.
        e_valid[:] = False
        f_valid[:] = False
    cols["flap_frequency_hz"] = fv[:, 0]
    # wing_phase timestamp is backdated to the encoder; it is NEVER a history feature.
    wp, wp_valid, timing = hold(topics["wing_phase"], meas,
                                ["phase_rad", "phase_unwrapped_rad", "flap_frequency_hz", "phase_valid"],
                                "logged_phase", config["encoder_age_us"])
    cols.update({f"target_{k}": v for k, v in timing.items()})
    cols.update(target_logged_flap_phase_rad=wp[:, 0],
                target_logged_flap_phase_unwrapped_rad=wp[:, 1],
                target_logged_flap_frequency_hz=wp[:, 2],
                target_logged_phase_valid=wp_valid & (wp[:, 3] > .5),
                target_logged_phase_publication_known=False)
    # Native encoder/frequency targets are paired past-only at the state measurement time.
    et, et_valid, et_timing = hold(enc.drop(columns="availability_us"), meas, ["relative_phase"], "encoder",
                                  config["encoder_age_us"])
    ft, ft_valid, ft_timing = hold(topics["flap_frequency"].drop(columns="availability_us"), meas, ["frequency_hz"],
                                  "frequency", config["frequency_age_us"])
    cols.update({f"target_{k}": v for k, v in {**et_timing, **ft_timing}.items()})
    cols.update(target_relative_flap_phase_rad=et[:, 0], target_flap_frequency_hz=ft[:, 0],
                valid_encoder=e_valid, valid_frequency=f_valid,
                target_phase_valid=et_valid, target_frequency_valid=ft_valid)
    status, status_valid, timing = hold(topics["vehicle_status"], pub,
                                       ["nav_state", "arming_state", "failsafe"], "status", 2_000_000)
    cols.update(timing)
    cols.update(nav_state=status[:, 0], arming_state=status[:, 1], failsafe=status[:, 2],
                valid_status=status_valid,
                mode_change=np.r_[False, np.diff(status[:, 0]) != 0],
                high_dynamics=np.linalg.norm(omega, axis=1) > 2.0)
    reset_times = []
    for name, fields in (("vehicle_local_position", RESET),
                         ("vehicle_attitude", ["quat_reset_counter"])):
        native = topics[name]
        changed = np.r_[False, (np.diff(native[fields].to_numpy(dtype=float), axis=0) != 0).any(axis=1)]
        reset_times.extend(native.loc[changed, "timestamp_sample_us"].tolist())
        reset_times.extend(native.loc[changed, "timestamp_us"].tolist())
    # Treat backwards counts as a phase continuity break, while retaining the raw observation.
    reset_times.extend(enc.loc[np.r_[False, np.diff(enc.total_count) < 0], "timestamp_us"].tolist())
    resets = np.array(sorted(set(t for t in reset_times if t > 0)), dtype=np.int64)
    reset_intervals = np.column_stack([resets, resets])
    control_gaps = []
    for name in ("actuator_motors", "actuator_servos"):
        ts = topics[name].timestamp_us.to_numpy()
        limit = max(40_000, int(1.6 * np.median(np.diff(ts))))
        bad = np.flatnonzero(np.diff(ts) > limit)
        control_gaps.extend([[int(ts[i] + limit), int(ts[i+1] - 1)] for i in bad])
    control_gap_intervals = np.array(control_gaps, np.int64).reshape(-1, 2)
    min_support = np.minimum.reduce([meas, cols["attitude_publication_us"], cols["rate_publication_us"],
                                    cols["motor_publication_us"], cols["servo_publication_us"],
                                    cols["attitude_measurement_us"], cols["rate_measurement_us"],
                                    cols["encoder_source_timestamp_us"], cols["frequency_source_timestamp_us"],
                                    cols["target_attitude_measurement_us"], cols["target_rate_measurement_us"]])
    max_support = np.maximum.reduce([pub, cols["target_attitude_publication_us"],
                                    cols["target_rate_publication_us"]])
    excluded = interval_overlap(min_support, max_support, exclude)
    control_gap = interval_overlap(min_support, max_support, control_gap_intervals)
    reviewed = interval_overlap(min_support, max_support, review)
    target_valid = pv_valid & tq_valid & tw_valid
    history_valid = pv_valid & q_valid & w_valid & control_valid
    unique_target = np.r_[True, (np.diff(cols["target_attitude_measurement_us"]) > 0)
                          & (np.diff(cols["target_rate_measurement_us"]) > 0)]
    # Rows with held targets remain in samples with masks, but cannot enter complete windows.
    target_valid &= unique_target
    within_flight = (min_support >= row.start_us) & (max_support <= row.end_us)
    row_ok = history_valid & target_valid & within_flight & ~excluded & ~control_gap
    gap = np.r_[False, (np.diff(meas) > policy["maximum_state_gap_us"])
                | (np.diff(pub) > policy["maximum_state_gap_us"])]
    edge_reset = interval_overlap(np.r_[min_support[0], min_support[:-1]], max_support, reset_intervals)
    edge_exclude = interval_overlap(np.r_[min_support[0], min_support[:-1]], max_support, exclude)
    edge_control_gap = interval_overlap(
        np.r_[min_support[0], min_support[:-1]], max_support, control_gap_intervals)
    segments = np.full(len(pub), -1, dtype=np.int64)
    current = -1
    previous_good = False
    for i in range(len(pub)):
        if row_ok[i]:
            if not previous_good or gap[i] or edge_reset[i] or edge_exclude[i] or edge_control_gap[i]:
                current += 1
            segments[i] = current
        previous_good = bool(row_ok[i])
    cols.update(valid_history_state=pv_valid & q_valid & w_valid, valid_control=control_valid,
                target_position_velocity_valid=pv_valid, target_attitude_valid=tq_valid,
                target_rate_valid=tw_valid, target_unique_observation=unique_target,
                valid_target_state=target_valid, valid_core=row_ok, excluded=excluded, review=reviewed,
                control_gap=control_gap,
                gap_boundary=gap, reset_boundary=edge_reset, support_start_us=min_support,
                support_end_us=max_support, segment_id=segments)
    samples = pd.DataFrame(cols)
    # Boundary context is provenance-only: two previous records cover the
    # one-message availability delay; one next record covers target pairing.
    # Windows still require every source timestamp inside the reviewed flight.
    native = {}
    topic_stats = {}
    for name, frame in topics.items():
        keep = frame.timestamp_us.between(row.start_us, row.end_us)
        prior = np.flatnonzero(frame.timestamp_us.to_numpy() < row.start_us)
        if len(prior):
            keep.iloc[prior[-2:]] = True
        after = np.flatnonzero(frame.timestamp_us.to_numpy() > row.end_us)
        if len(after):
            keep.iloc[after[0]] = True
        clip = frame.loc[keep].copy()
        clip.insert(0, "flight_id", row.flight_id)
        clip["native_dt_us"] = frame.timestamp_us.diff().loc[keep].to_numpy()
        clip["in_reviewed_flight"] = clip.timestamp_us.between(row.start_us, row.end_us)
        native[name] = clip
        t = clip.loc[clip.in_reviewed_flight, "timestamp_us"].to_numpy()
        topic_stats[name] = dict(rows=len(t), dt_us_quantiles=quantiles(np.diff(t)),
                                separate_measurement_timestamp=bool((clip.timestamp_sample_us > 0).any()))
    record.update(samples=len(samples), valid_samples=int(row_ok.sum()),
                  phase_origin_count=origin_count, phase_origin_timestamp_us=origin_us,
                  phase_origin_scope="whole_log_first_encoder_record",
                  topic_support=topic_stats, reset_times_us=resets.tolist(),
                  exclude_intervals_us=exclude.tolist(), review_intervals_us=review.tolist(),
                  history_age_us={p: quantiles(samples[f"{p}_age_us"]) for p in
                                  ["attitude", "rate", "motor", "servo", "encoder", "frequency"]},
                  target_pair_offset_us={p: quantiles(samples[f"target_{p}_offset_us"])
                                         for p in ["attitude", "rate"]},
                  source_hash_verified_after_read=file_hash(source))
    if record["source_hash_verified_after_read"] != row.sha256:
        raise ValueError(f"source mutated during read: {source}")
    return samples, native, record


def build_windows(samples, record, config):
    """Duration-based windows on native measurement times; no artificial grids."""
    t = samples.state_measurement_us.to_numpy()
    pub = samples.timestamp_us.to_numpy()
    support_start = samples.support_start_us.to_numpy()
    support_end = samples.support_end_us.to_numpy()
    segments = samples.segment_id.to_numpy()
    policy = config["groups"][record["sampling_group"]]
    exclusions = np.array(record["exclude_intervals_us"], np.int64).reshape(-1, 2)
    reviews = np.array(record["review_intervals_us"], np.int64).reshape(-1, 2)
    resets = np.array(record["reset_times_us"], np.int64)
    rows, rejected = [], Counter()
    origins = []
    next_time = int(t[0])
    for i, timestamp in enumerate(t):
        if timestamp >= next_time:
            origins.append(i)
            next_time = int(timestamp) + config["origin_stride_us"]
    origins = np.asarray(origins, dtype=np.int64)
    if np.any(np.diff(support_start) < 0) or np.any(np.diff(support_end) < 0):
        raise ValueError("nonmonotonic support bounds")
    cum_mode = np.r_[0, np.cumsum(samples.mode_change)]
    cum_dynamic = np.r_[0, np.cumsum(samples.high_dynamics)]
    cum_phase_invalid = np.r_[0, np.cumsum(~(samples.valid_encoder & samples.valid_frequency))]
    for history in config["history_us"]:
        for horizon in config["horizon_us"]:
            rejected["candidate"] += len(origins)
            start = np.searchsorted(t, t[origins] - history, side="right") - 1
            end = nearest_indices(t, t[origins] + horizon)
            ok = (start >= 0) & (end > origins) & (
                np.abs(t[end] - t[origins] - horizon) <= policy["horizon_tolerance_us"])
            rejected["duration_or_edge"] += int((~ok).sum())
            origin, start, end = origins[ok], start[ok], end[ok]
            a, b = support_start[start], support_end[end]
            keep = np.ones(len(origin), bool)
            rules = [
                ("next_target_not_future_at_decision", t[origin+1] <= pub[origin]),
                ("exclude", interval_overlap(a, b, exclusions)),
                ("reset", interval_overlap(a, b, np.column_stack([resets, resets]))),
                ("invalid_or_gap_or_nonunique_target", (segments[start] < 0)
                 | (segments[start] != segments[end])),
            ]
            for reason, bad in rules:
                rejected[reason] += int((keep & bad).sum())
                keep &= ~bad
            origin, start, end, a, b = (x[keep] for x in (origin, start, end, a, b))
            if not len(origin):
                continue
            rows.append(pd.DataFrame(dict(
                window_id=[f"{record['flight_id']}:{i}:h{history}:f{horizon}" for i in origin],
                flight_id=record["flight_id"], split=record["split"],
                sampling_group=record["sampling_group"], segment_id=segments[origin],
                history_us=history, horizon_us=horizon, start_sample=start,
                origin_sample=origin, end_sample=end, history_count=origin-start+1,
                future_count=end-origin, actual_history_s=(t[origin]-t[start])*1e-6,
                actual_horizon_s=(t[end]-t[origin])*1e-6,
                initial_measurement_us=t[origin], decision_us=pub[origin],
                last_target_measurement_us=t[end], support_start_us=a, support_end_us=b,
                review=interval_overlap(a, b, reviews),
                mode_change=(cum_mode[end+1]-cum_mode[start]) > 0,
                high_dynamics=(cum_dynamic[end+1]-cum_dynamic[start]) > 0,
                complete_state_sequence=True,
                complete_phase_frequency=(cum_phase_invalid[end+1]-cum_phase_invalid[start]) == 0,
            )))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), dict(rejected)


def action_replay(motor, servo, grid, maximum_age_us):
    """u[k] is a past-only command at measurement t[k], never an interval mean.

    The separate event tape is an explicitly supplied future command plan. It
    is not silently reduced to a supposedly causal start-of-step feature.
    """
    motor = motor.sort_values("timestamp_us")
    servo = servo.sort_values("timestamp_us")
    parts, masks, times = [], [], []
    for native, fields in ((motor, ["control[0]"]), (servo, ["control[0]", "control[1]", "control[2]"])):
        value, mask, timing = hold(native, grid[:-1], fields, "u", maximum_age_us)
        parts.append(value)
        masks.append(mask)
        times.append(timing["u_publication_us"])
    starts = np.column_stack(parts)
    mask = np.logical_and.reduce(masks)
    event_t = np.unique(np.r_[grid[0], motor.timestamp_us, servo.timestamp_us]).astype(np.int64)
    event_t = event_t[(event_t >= grid[0]) & (event_t < grid[-1])]
    event_parts = [hold(n, event_t, f, "u", maximum_age_us)
                   for n, f in ((motor, ["control[0]"]),
                                (servo, ["control[0]", "control[1]", "control[2]"]))]
    return dict(step_control=starts, step_control_mask=mask,
                step_control_publication_us=np.column_stack(times),
                event_timestamp_us=event_t,
                event_control=np.column_stack([r[0] for r in event_parts]),
                event_mask=np.logical_and.reduce([r[1] for r in event_parts]),
                event_step_index=np.searchsorted(grid, event_t, side="right") - 1)


class VelocityRateDataset:
    """Lazy flight loading; only inputs/ are allowed into a recurrent predictor.

    Future truth lives solely in targets/. Variable lengths require padding
    with masks or bucketing; no architecture or fixed-step count is imposed.
    """
    def __init__(self, root, split="train", sampling_group="main_50hz", history_us=500000,
                 horizon_us=200000, *, allow_test=False, verify_hashes=True):
        self.root, self.split = Path(root), split
        if split not in {"train", "validation", "test"}:
            raise ValueError("unknown split")
        if split == "test" and not allow_test:
            raise ValueError("test is frozen; explicit allow_test required for authorized evaluation")
        self.manifest = json.loads((self.root / "dataset_manifest.json").read_text())
        if verify_hashes:
            names = [f"samples_{split}.parquet", f"windows_{split}.parquet",
                     f"native_actuator_motors_{split}.parquet", f"native_actuator_servos_{split}.parquet"]
            for name in names:
                if file_hash(self.root / name) != self.manifest["artifact_sha256"][name]:
                    raise ValueError(f"artifact hash mismatch: {name}")
        w = pd.read_parquet(self.root / f"windows_{split}.parquet")
        self.windows = w.loc[(w.sampling_group == sampling_group) & (w.history_us == history_us)
                             & (w.horizon_us == horizon_us)].reset_index(drop=True)
        self.cache = None
        self.cached_flight = None

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        row = self.windows.iloc[index]
        if self.cached_flight != row.flight_id:
            self.cache = {name: pd.read_parquet(
                self.root / f"{name}_{self.split}.parquet", filters=[("flight_id", "==", row.flight_id)]
            ).reset_index(drop=True) for name in ["samples", "native_actuator_motors", "native_actuator_servos"]}
            self.cached_flight = row.flight_id
        s = self.cache["samples"]
        history = s.iloc[int(row.start_sample):int(row.origin_sample)+1]
        future = s.iloc[int(row.origin_sample)+1:int(row.end_sample)+1]
        all_future = s.iloc[int(row.origin_sample):int(row.end_sample)+1]
        grid = all_future.state_measurement_us.to_numpy(dtype=np.int64)
        actions = action_replay(self.cache["native_actuator_motors"], self.cache["native_actuator_servos"],
                                grid, self.manifest["config"]["control_age_us"])
        dt = np.diff(grid) * 1e-6
        history_values = history[STATE + CONTROL + PHASE].to_numpy(dtype=float)
        history_mask = np.isfinite(history_values)
        history_mask[:, 13:17] &= history.valid_control.to_numpy()[:, None]
        history_mask[:, 17:19] &= history.valid_encoder.to_numpy()[:, None]
        history_mask[:, 19] &= history.valid_frequency.to_numpy()
        history_values = np.where(history_mask, history_values, 0)
        targets = future[TARGET].to_numpy(dtype=float)
        state_mask = np.column_stack([
            np.repeat(future.target_position_velocity_valid.to_numpy()[:, None], 6, axis=1),
            np.repeat(future.target_attitude_valid.to_numpy()[:, None], 4, axis=1),
            np.repeat(future.target_rate_valid.to_numpy()[:, None], 3, axis=1)])
        return {
            "inputs": {
                "history": history_values, "history_mask": history_mask,
                "history_measurement_us": history.state_measurement_us.to_numpy(),
                "history_publication_us": history.timestamp_us.to_numpy(),
                "history_dt_s": np.r_[0, np.diff(history.state_measurement_us) * 1e-6],
                "history_signal_age_us": history[
                    ["state_age_us"] + [f"{p}_age_us" for p in
                                       ["attitude", "rate", "motor", "servo", "encoder", "frequency"]]
                ].to_numpy(),
                "initial_state": history[STATE].iloc[-1].to_numpy(dtype=float),
                "initial_phase_unwrapped_rad": (
                    float(history.relative_flap_phase_unwrapped_rad.iloc[-1])
                    if history.valid_encoder.iloc[-1] else 0.0),
                "initial_phase_valid": bool(history.valid_encoder.iloc[-1]),
                "phase_origin_count": float(history.phase_origin_count.iloc[-1]),
                "phase_origin_timestamp_us": int(history.phase_origin_timestamp_us.iloc[-1]),
                "future_dt_s": dt, "future_control": actions,
            },
            "targets": {
                "state": targets, "state_mask": state_mask,
                "next_velocity_ned": targets[0, 3:6], "next_angular_rate_body": targets[0, 10:13],
                "velocity_body": future[[f"target_velocity_body_m_s_{a}" for a in "xyz"]].to_numpy(),
                "phase_frequency": future[["target_relative_flap_phase_rad", "target_flap_frequency_hz"]].to_numpy(),
                "phase_frequency_mask": future[["target_phase_valid", "target_frequency_valid"]].to_numpy(),
                "state_measurement_us": grid[1:],
                "attitude_measurement_us": future.target_attitude_measurement_us.to_numpy(),
                "rate_measurement_us": future.target_rate_measurement_us.to_numpy(),
            },
            "metadata": row.to_dict(),
        }
