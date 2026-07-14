#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_level2_replay_ready_table import KEY_COLUMNS, POSITION_COLUMNS, QUATERNION_COLUMNS, VELOCITY_COLUMNS
from scripts.evaluate_short_horizon_replay import quaternion_to_rotation_body_to_ned


FORCE_SOURCES = {
    "oracle": ("label_fx_b", "label_fz_b"),
    "raw_prior": ("raw_prior_fx_b", "raw_prior_fz_b"),
    "gain_bias": ("gain_bias_fx_b", "gain_bias_fz_b"),
    "pure_tcn": ("pure_tcn_fx_b", "pure_tcn_fz_b"),
}


def _metadata_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def load_mass_and_gravity(metadata_path: str | Path) -> tuple[float, float]:
    with Path(metadata_path).open("r", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    mass = float(_metadata_value(metadata["mass_properties"]["mass_kg"]))
    gravity = float(_metadata_value(metadata.get("label_definition", {}).get("gravity_m_s2", 9.81)))
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass_kg must be positive and finite")
    if not np.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("gravity_m_s2 must be positive and finite")
    return mass, gravity


def parse_float_list(raw: str) -> list[float]:
    values = [float(part) for part in raw.split(",") if part.strip()]
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("horizons must contain positive values")
    return values


def local_along_track_axis(velocity_n: np.ndarray, min_speed_m_s: float) -> np.ndarray | None:
    horizontal = np.array([velocity_n[0], velocity_n[1], 0.0], dtype=float)
    speed = float(np.linalg.norm(horizontal))
    if not np.isfinite(speed) or speed < min_speed_m_s:
        return None
    return horizontal / speed


def integrate_force_window(
    window: pd.DataFrame,
    source: str,
    mass_kg: float,
    gravity_m_s2: float,
) -> tuple[np.ndarray, np.ndarray]:
    if source not in FORCE_SOURCES:
        raise ValueError(f"unknown force source: {source}")
    fx_column, fz_column = FORCE_SOURCES[source]
    time_s = window["time_s"].to_numpy(dtype=float)
    velocity = window[VELOCITY_COLUMNS].to_numpy(dtype=float)
    position = window[POSITION_COLUMNS].to_numpy(dtype=float)
    quat = window[QUATERNION_COLUMNS].to_numpy(dtype=float)
    fx = window[fx_column].to_numpy(dtype=float)
    fz = window[fz_column].to_numpy(dtype=float)

    v = velocity[0].copy()
    p = position[0].copy()
    gravity_n = np.array([0.0, 0.0, gravity_m_s2], dtype=float)
    for idx in range(len(window) - 1):
        dt_s = float(time_s[idx + 1] - time_s[idx])
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("window time_s must be strictly increasing")
        force_b = np.array([fx[idx], 0.0, fz[idx]], dtype=float)
        accel_n = quaternion_to_rotation_body_to_ned(quat[idx]) @ (force_b / mass_kg) + gravity_n
        p = p + v * dt_s + 0.5 * accel_n * dt_s * dt_s
        v = v + accel_n * dt_s
    return p, v


def select_candidate_windows(
    frame: pd.DataFrame,
    horizons_s: list[float],
    stride_s: float,
    roll_threshold_deg: float,
    min_ground_speed_m_s: float,
) -> list[dict[str, Any]]:
    if stride_s <= 0.0:
        raise ValueError("stride_s must be positive")
    roll_threshold_rad = np.deg2rad(float(roll_threshold_deg))
    required = [*KEY_COLUMNS, *QUATERNION_COLUMNS, *POSITION_COLUMNS, *VELOCITY_COLUMNS, "roll_rad"]
    for fx_col, fz_col in FORCE_SOURCES.values():
        required.extend([fx_col, fz_col])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"replay-ready table missing columns: {missing}")

    sort_columns = ["outer_fold", "log_id", "segment_id", "time_s"]
    frame = frame.sort_values(sort_columns).reset_index(drop=True)
    grouped = frame.groupby(["outer_fold", "log_id", "segment_id"], sort=False, dropna=False)
    windows: list[dict[str, Any]] = []

    for (outer_fold, log_id, segment_id), group in grouped:
        times = group["time_s"].to_numpy(dtype=float)
        if len(group) < 2 or not np.isfinite(times).all():
            continue
        dt = np.diff(times)
        positive_dt = dt[np.isfinite(dt) & (dt > 0.0)]
        if len(positive_dt) == 0:
            continue
        median_dt = float(np.median(positive_dt))
        max_allowed_gap = max(0.025, 2.5 * median_dt)
        good_gap_prefix = np.concatenate([[0], np.cumsum((dt > 0.0) & (dt <= max_allowed_gap))])
        finite = np.ones(len(group), dtype=bool)
        for column in required:
            if column == "log_id":
                finite &= group[column].notna().to_numpy()
            else:
                finite &= np.isfinite(group[column].to_numpy(dtype=float, copy=False))
        finite_prefix = np.concatenate([[0], np.cumsum(finite.astype(int))])
        low_roll = np.abs(group["roll_rad"].to_numpy(dtype=float)) <= roll_threshold_rad
        low_roll_prefix = np.concatenate([[0], np.cumsum(low_roll.astype(int))])
        group_indices = group.index.to_numpy(dtype=int)
        next_start_time = float(times[0])
        while next_start_time <= float(times[-1]):
            start_idx = int(np.searchsorted(times, next_start_time, side="left"))
            if start_idx >= len(group) - 1:
                break
            axis = local_along_track_axis(group.iloc[start_idx][VELOCITY_COLUMNS].to_numpy(dtype=float), min_ground_speed_m_s)
            if axis is None:
                next_start_time += stride_s
                continue
            for horizon_s in horizons_s:
                target_time = float(times[start_idx] + horizon_s)
                end_idx = int(np.searchsorted(times, target_time, side="left"))
                if end_idx >= len(group):
                    continue
                if end_idx > start_idx and abs(times[end_idx - 1] - target_time) < abs(times[end_idx] - target_time):
                    end_idx -= 1
                if end_idx <= start_idx:
                    continue
                if good_gap_prefix[end_idx] - good_gap_prefix[start_idx] != end_idx - start_idx:
                    continue
                if finite_prefix[end_idx + 1] - finite_prefix[start_idx] != end_idx - start_idx + 1:
                    continue
                if low_roll_prefix[end_idx + 1] - low_roll_prefix[start_idx] != end_idx - start_idx + 1:
                    continue
                windows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "log_id": log_id,
                        "segment_id": segment_id,
                        "start_idx": int(group_indices[start_idx]),
                        "end_idx": int(group_indices[end_idx]),
                        "start_time_s": float(times[start_idx]),
                        "horizon_s": float(horizon_s),
                        "n_steps": int(end_idx - start_idx),
                    }
                )
            next_start_time += stride_s
    return windows


def evaluate_level2_replay(
    frame: pd.DataFrame,
    mass_kg: float,
    gravity_m_s2: float,
    horizons_s: list[float],
    stride_s: float = 0.25,
    roll_threshold_deg: float = 10.0,
    oracle_max_velocity_error_m_s: float = 1.0,
    min_ground_speed_m_s: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sorted_frame = frame.sort_values(["outer_fold", "log_id", "segment_id", "time_s"]).reset_index(drop=True)
    windows = select_candidate_windows(
        sorted_frame,
        horizons_s=horizons_s,
        stride_s=stride_s,
        roll_threshold_deg=roll_threshold_deg,
        min_ground_speed_m_s=min_ground_speed_m_s,
    )
    rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for spec in windows:
        window = sorted_frame.iloc[spec["start_idx"] : spec["end_idx"] + 1]
        start_velocity = window.iloc[0][VELOCITY_COLUMNS].to_numpy(dtype=float)
        end_velocity = window.iloc[-1][VELOCITY_COLUMNS].to_numpy(dtype=float)
        target_delta_v = end_velocity - start_velocity
        axis = local_along_track_axis(start_velocity, min_ground_speed_m_s)
        if axis is None:
            continue
        _, oracle_velocity = integrate_force_window(window, "oracle", mass_kg=mass_kg, gravity_m_s2=gravity_m_s2)
        oracle_delta_v = oracle_velocity - start_velocity
        oracle_error_vector = oracle_delta_v - target_delta_v
        oracle_velocity_error = float(np.linalg.norm(oracle_error_vector))
        passed = bool(oracle_velocity_error <= oracle_max_velocity_error_m_s)
        gate_row = {
            **{k: spec[k] for k in ["outer_fold", "log_id", "segment_id", "start_time_s", "horizon_s", "n_steps"]},
            "oracle_velocity_error_m_s": oracle_velocity_error,
            "oracle_along_track_error_m_s": float(np.dot(axis, oracle_error_vector)),
            "oracle_vertical_error_m_s": float(oracle_error_vector[2]),
            "passed_oracle_gate": passed,
        }
        gate_rows.append(gate_row)
        if not passed:
            continue
        for source in ["raw_prior", "gain_bias", "pure_tcn"]:
            _, velocity = integrate_force_window(window, source, mass_kg=mass_kg, gravity_m_s2=gravity_m_s2)
            delta_v = velocity - start_velocity
            error_vector = delta_v - target_delta_v
            rows.append(
                {
                    **{k: spec[k] for k in ["outer_fold", "log_id", "segment_id", "start_time_s", "horizon_s", "n_steps"]},
                    "force_source": source,
                    "velocity_increment_error_m_s": float(np.linalg.norm(error_vector)),
                    "along_track_velocity_increment_error_m_s": float(abs(np.dot(axis, error_vector))),
                    "vertical_velocity_increment_error_m_s": float(abs(error_vector[2])),
                    "signed_along_track_velocity_increment_error_m_s": float(np.dot(axis, error_vector)),
                    "signed_vertical_velocity_increment_error_m_s": float(error_vector[2]),
                }
            )
    window_metrics = pd.DataFrame(rows)
    gate_metrics = pd.DataFrame(gate_rows)
    summary = summarize_level2_metrics(window_metrics, gate_metrics)
    return window_metrics, gate_metrics, summary


def summarize_level2_metrics(window_metrics: pd.DataFrame, gate_metrics: pd.DataFrame) -> pd.DataFrame:
    gate_summary = pd.DataFrame()
    if not gate_metrics.empty:
        grouped_gate = gate_metrics.groupby("horizon_s", sort=True)
        gate_summary = grouped_gate.agg(
            candidate_windows=("passed_oracle_gate", "size"),
            passed_windows=("passed_oracle_gate", "sum"),
            oracle_velocity_error_m_s_median=("oracle_velocity_error_m_s", "median"),
        ).reset_index()
        gate_summary["gate_pass_rate"] = gate_summary["passed_windows"] / gate_summary["candidate_windows"]
    if window_metrics.empty:
        return gate_summary
    grouped = window_metrics.groupby(["horizon_s", "force_source"], sort=True)
    summary = grouped.agg(
        n_windows=("velocity_increment_error_m_s", "size"),
        velocity_increment_error_m_s_median=("velocity_increment_error_m_s", "median"),
        velocity_increment_error_m_s_mean=("velocity_increment_error_m_s", "mean"),
        along_track_velocity_increment_error_m_s_median=("along_track_velocity_increment_error_m_s", "median"),
        vertical_velocity_increment_error_m_s_median=("vertical_velocity_increment_error_m_s", "median"),
    ).reset_index()
    if not gate_summary.empty:
        summary = summary.merge(gate_summary, on="horizon_s", how="left")
    return summary


def write_level2_artifacts(
    output_root: str | Path,
    window_metrics: pd.DataFrame,
    gate_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    config: dict[str, Any],
    overwrite: bool = False,
) -> None:
    output_path = Path(output_root)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(f"output root exists and is not empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    window_metrics.to_csv(output_path / "level2_window_metrics.csv", index=False)
    gate_metrics.to_csv(output_path / "level2_oracle_gate_metrics.csv", index=False)
    summary.to_csv(output_path / "level2_summary.csv", index=False)
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), **config}
    with (output_path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Level 2 log-attitude translational replay.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--metadata-path", default="metadata/aircraft/flapper_01/aircraft_metadata.yaml")
    parser.add_argument("--horizons", default="0.5,1.0,2.0")
    parser.add_argument("--stride-s", type=float, default=0.25)
    parser.add_argument("--roll-threshold-deg", type=float, default=10.0)
    parser.add_argument("--oracle-max-velocity-error-m-s", type=float, default=1.0)
    parser.add_argument("--min-ground-speed-m-s", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    frame = pd.read_parquet(input_root / "level2_replay_ready_rows.parquet")
    mass_kg, gravity_m_s2 = load_mass_and_gravity(args.metadata_path)
    horizons_s = parse_float_list(args.horizons)
    window_metrics, gate_metrics, summary = evaluate_level2_replay(
        frame,
        mass_kg=mass_kg,
        gravity_m_s2=gravity_m_s2,
        horizons_s=horizons_s,
        stride_s=args.stride_s,
        roll_threshold_deg=args.roll_threshold_deg,
        oracle_max_velocity_error_m_s=args.oracle_max_velocity_error_m_s,
        min_ground_speed_m_s=args.min_ground_speed_m_s,
    )
    write_level2_artifacts(
        args.output_root,
        window_metrics,
        gate_metrics,
        summary,
        config={
            "input_root": str(input_root),
            "metadata_path": str(args.metadata_path),
            "mass_kg": mass_kg,
            "gravity_m_s2": gravity_m_s2,
            "horizons_s": horizons_s,
            "stride_s": args.stride_s,
            "roll_threshold_deg": args.roll_threshold_deg,
            "oracle_max_velocity_error_m_s": args.oracle_max_velocity_error_m_s,
            "min_ground_speed_m_s": args.min_ground_speed_m_s,
        },
        overwrite=args.overwrite,
    )
    print(f"wrote {len(window_metrics)} model metric rows and {len(gate_metrics)} gate rows to {args.output_root}")


if __name__ == "__main__":
    main()
