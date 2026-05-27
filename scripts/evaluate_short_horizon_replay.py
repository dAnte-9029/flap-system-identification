#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


QUATERNION_COLUMNS = [
    "vehicle_attitude.q[0]",
    "vehicle_attitude.q[1]",
    "vehicle_attitude.q[2]",
    "vehicle_attitude.q[3]",
]
POSITION_COLUMNS = [
    "vehicle_local_position.x",
    "vehicle_local_position.y",
    "vehicle_local_position.z",
]
VELOCITY_COLUMNS = [
    "vehicle_local_position.vx",
    "vehicle_local_position.vy",
    "vehicle_local_position.vz",
]
OMEGA_COLUMNS = [
    "vehicle_angular_velocity.xyz[0]",
    "vehicle_angular_velocity.xyz[1]",
    "vehicle_angular_velocity.xyz[2]",
]
FORCE_COLUMNS = ["fx_b", "fy_b", "fz_b"]
MOMENT_COLUMNS = ["mx_b", "my_b", "mz_b"]
METRIC_COLUMNS = [
    "position_error_m",
    "velocity_error_m_s",
    "attitude_error_deg",
    "body_rate_error_rad_s",
]
ORACLE_MODES = {"oracle_teacher_forced", "coupled_oracle"}


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    quat = np.asarray(q, dtype=float)
    norm = np.linalg.norm(quat)
    if quat.shape != (4,) or not np.isfinite(quat).all() or norm <= 0.0:
        raise ValueError("quaternion must be a finite length-4 vector with nonzero norm")
    return quat / norm


def quaternion_to_rotation_body_to_ned(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def attitude_error_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
    qa = normalize_quaternion(q_a)
    qb = normalize_quaternion(q_b)
    dot = float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _metadata_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def load_mass_properties(metadata_path: str | Path) -> tuple[float, np.ndarray, float]:
    with Path(metadata_path).open("r", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)

    mass_raw = _metadata_value(metadata["mass_properties"]["mass_kg"])
    inertia_raw = _metadata_value(metadata["mass_properties"]["inertia_b_kg_m2"])
    gravity_raw = _metadata_value(metadata.get("label_definition", {}).get("gravity_m_s2", 9.81))

    mass_kg = float(mass_raw)
    inertia_b = np.asarray(inertia_raw, dtype=float)
    gravity_m_s2 = float(gravity_raw)
    if not np.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError("metadata mass_kg must be positive and finite")
    if inertia_b.shape != (3, 3) or not np.isfinite(inertia_b).all():
        raise ValueError("metadata inertia_b_kg_m2 must be a finite 3x3 matrix")
    if not np.isfinite(gravity_m_s2):
        raise ValueError("metadata gravity_m_s2 must be finite")
    return mass_kg, inertia_b, gravity_m_s2


def _quat_multiply(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = q_left
    w1, x1, y1, z1 = q_right
    return np.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ],
        dtype=float,
    )


def _delta_quaternion_from_body_rate(omega_b: np.ndarray, dt_s: float) -> np.ndarray:
    angle = float(np.linalg.norm(omega_b) * dt_s)
    if angle <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = np.asarray(omega_b, dtype=float) / np.linalg.norm(omega_b)
    half_angle = 0.5 * angle
    return np.array([np.cos(half_angle), *(np.sin(half_angle) * axis)], dtype=float)


def _frame_values(window: pd.DataFrame, columns: list[str]) -> np.ndarray:
    missing = [column for column in columns if column not in window.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return window[columns].to_numpy(dtype=float, copy=True)


def integrate_oracle_teacher_forced_window(
    window: pd.DataFrame,
    mass_kg: float,
    inertia_b: np.ndarray,
    gravity_m_s2: float,
) -> dict[str, np.ndarray]:
    return _integrate_oracle_window(
        window,
        mass_kg=mass_kg,
        inertia_b=inertia_b,
        gravity_m_s2=gravity_m_s2,
        mode="oracle_teacher_forced",
    )


def integrate_coupled_oracle_window(
    window: pd.DataFrame,
    mass_kg: float,
    inertia_b: np.ndarray,
    gravity_m_s2: float,
) -> dict[str, np.ndarray]:
    return _integrate_oracle_window(
        window,
        mass_kg=mass_kg,
        inertia_b=inertia_b,
        gravity_m_s2=gravity_m_s2,
        mode="coupled_oracle",
    )


def _integrate_oracle_window(
    window: pd.DataFrame,
    mass_kg: float,
    inertia_b: np.ndarray,
    gravity_m_s2: float,
    mode: str,
) -> dict[str, np.ndarray]:
    if len(window) < 1:
        raise ValueError("window must contain at least one row")

    time_s = window["time_s"].to_numpy(dtype=float, copy=True)
    position_log = _frame_values(window, POSITION_COLUMNS)
    velocity_log = _frame_values(window, VELOCITY_COLUMNS)
    quat_log = _frame_values(window, QUATERNION_COLUMNS)
    omega_log = _frame_values(window, OMEGA_COLUMNS)
    force_b = _frame_values(window, FORCE_COLUMNS)
    moment_b = _frame_values(window, MOMENT_COLUMNS)

    n_rows = len(window)
    position_n = np.zeros((n_rows, 3), dtype=float)
    velocity_n = np.zeros((n_rows, 3), dtype=float)
    omega_b = np.zeros((n_rows, 3), dtype=float)
    quat_nb = np.zeros((n_rows, 4), dtype=float)

    position_n[0] = position_log[0]
    velocity_n[0] = velocity_log[0]
    omega_b[0] = omega_log[0]
    quat_nb[0] = normalize_quaternion(quat_log[0])

    inertia_b = np.asarray(inertia_b, dtype=float)
    inertia_inv = np.linalg.inv(inertia_b)
    gravity_n = np.array([0.0, 0.0, gravity_m_s2], dtype=float)

    for idx in range(n_rows - 1):
        dt_s = float(time_s[idx + 1] - time_s[idx])
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("window time_s must be strictly increasing")

        q_for_force = normalize_quaternion(quat_log[idx]) if mode == "oracle_teacher_forced" else quat_nb[idx]
        omega_for_gyro = omega_log[idx] if mode == "oracle_teacher_forced" else omega_b[idx]

        acceleration_n = quaternion_to_rotation_body_to_ned(q_for_force) @ (force_b[idx] / mass_kg) + gravity_n
        angular_momentum_b = inertia_b @ omega_for_gyro
        alpha_b = inertia_inv @ (moment_b[idx] - np.cross(omega_for_gyro, angular_momentum_b))

        position_n[idx + 1] = position_n[idx] + velocity_n[idx] * dt_s + 0.5 * acceleration_n * dt_s * dt_s
        velocity_n[idx + 1] = velocity_n[idx] + acceleration_n * dt_s
        omega_b[idx + 1] = omega_b[idx] + alpha_b * dt_s
        quat_nb[idx + 1] = normalize_quaternion(
            _quat_multiply(quat_nb[idx], _delta_quaternion_from_body_rate(omega_b[idx], dt_s))
        )

    return {
        "time_s": time_s,
        "position_n": position_n,
        "velocity_n": velocity_n,
        "omega_b": omega_b,
        "quat_nb": quat_nb,
    }


def select_replay_windows(frame: pd.DataFrame, horizons_s: list[float], stride_s: float) -> list[dict[str, Any]]:
    if stride_s <= 0.0:
        raise ValueError("stride_s must be positive")
    if not horizons_s or any(horizon <= 0.0 for horizon in horizons_s):
        raise ValueError("horizons_s must contain positive horizons")
    if "time_s" not in frame.columns:
        raise ValueError("frame must contain time_s")

    group_columns = ["log_id"] if "log_id" in frame.columns else []
    if "segment_id" in frame.columns:
        group_columns.append("segment_id")

    if group_columns:
        grouped = frame.groupby(group_columns, sort=False, dropna=False)
    else:
        grouped = [((), frame)]

    windows: list[dict[str, Any]] = []
    required_columns = POSITION_COLUMNS + VELOCITY_COLUMNS + QUATERNION_COLUMNS + OMEGA_COLUMNS + FORCE_COLUMNS + MOMENT_COLUMNS

    for group_key, group in grouped:
        group = group.sort_values("time_s")
        times = group["time_s"].to_numpy(dtype=float, copy=True)
        if len(group) < 2 or not np.isfinite(times).all():
            continue

        dt = np.diff(times)
        positive_dt = dt[np.isfinite(dt) & (dt > 0.0)]
        if len(positive_dt) == 0:
            continue
        median_dt = float(np.median(positive_dt))
        max_allowed_gap = max(0.05, 5.0 * median_dt)
        time_tolerance = max(1e-9, 1e-3 * median_dt)
        valid_gap_prefix = np.concatenate([[0], np.cumsum((dt > 0.0) & (dt <= max_allowed_gap))])

        finite_rows = np.ones(len(group), dtype=bool)
        for column in required_columns:
            if column not in group.columns:
                raise ValueError(f"frame missing required column: {column}")
            finite_rows &= np.isfinite(group[column].to_numpy(dtype=float, copy=False))
        finite_prefix = np.concatenate([[0], np.cumsum(finite_rows.astype(int))])

        group_indices = group.index.to_numpy(dtype=int)
        next_start_time = float(times[0])
        start_idx = 0
        while start_idx < len(group) - 1:
            start_idx = int(np.searchsorted(times, next_start_time - time_tolerance, side="left"))
            if start_idx >= len(group) - 1:
                break
            start_time_s = float(times[start_idx])

            for horizon_s in horizons_s:
                target_time_s = start_time_s + float(horizon_s)
                if target_time_s > times[-1] + 1e-9:
                    continue
                end_idx = int(np.searchsorted(times, target_time_s, side="left"))
                if end_idx >= len(group):
                    continue
                if end_idx > start_idx and abs(times[end_idx - 1] - target_time_s) < abs(times[end_idx] - target_time_s):
                    end_idx -= 1
                if end_idx <= start_idx:
                    continue
                if valid_gap_prefix[end_idx] - valid_gap_prefix[start_idx] != end_idx - start_idx:
                    continue
                if finite_prefix[end_idx + 1] - finite_prefix[start_idx] != end_idx - start_idx + 1:
                    continue

                if group_columns:
                    if not isinstance(group_key, tuple):
                        group_values = (group_key,)
                    else:
                        group_values = group_key
                    group_map = dict(zip(group_columns, group_values))
                else:
                    group_map = {}

                windows.append(
                    {
                        "start_idx": int(group_indices[start_idx]),
                        "end_idx": int(group_indices[end_idx]),
                        "log_id": group_map.get("log_id", ""),
                        "segment_id": group_map.get("segment_id", ""),
                        "start_time_s": start_time_s,
                        "horizon_s": float(horizon_s),
                        "n_steps": int(end_idx - start_idx),
                    }
                )

            next_start_time += stride_s

    return windows


def _coerce_metadata(metadata: dict[str, Any] | tuple[float, np.ndarray, float]) -> tuple[float, np.ndarray, float]:
    if isinstance(metadata, tuple):
        mass_kg, inertia_b, gravity_m_s2 = metadata
        return float(mass_kg), np.asarray(inertia_b, dtype=float), float(gravity_m_s2)
    if {"mass_kg", "inertia_b", "gravity_m_s2"}.issubset(metadata):
        return (
            float(metadata["mass_kg"]),
            np.asarray(metadata["inertia_b"], dtype=float),
            float(metadata["gravity_m_s2"]),
        )
    if {"mass_properties", "label_definition"}.issubset(metadata):
        mass_kg = float(_metadata_value(metadata["mass_properties"]["mass_kg"]))
        inertia_b = np.asarray(_metadata_value(metadata["mass_properties"]["inertia_b_kg_m2"]), dtype=float)
        gravity_m_s2 = float(_metadata_value(metadata["label_definition"].get("gravity_m_s2", 9.81)))
        return mass_kg, inertia_b, gravity_m_s2
    raise ValueError("metadata must provide mass_kg, inertia_b, and gravity_m_s2")


def evaluate_oracle_replay(
    samples: pd.DataFrame,
    metadata: dict[str, Any] | tuple[float, np.ndarray, float],
    horizons_s: list[float],
    stride_s: float,
    mode: str = "oracle_teacher_forced",
    split: str = "test",
) -> pd.DataFrame:
    if mode not in ORACLE_MODES:
        raise ValueError(f"unsupported oracle mode: {mode}")

    sort_columns = [column for column in ["log_id", "segment_id", "time_s"] if column in samples.columns]
    frame = samples.sort_values(sort_columns).reset_index(drop=True) if sort_columns else samples.reset_index(drop=True)
    windows = select_replay_windows(frame, horizons_s=horizons_s, stride_s=stride_s)
    mass_kg, inertia_b, gravity_m_s2 = _coerce_metadata(metadata)
    integrator = integrate_oracle_teacher_forced_window if mode == "oracle_teacher_forced" else integrate_coupled_oracle_window

    rows: list[dict[str, Any]] = []
    for window_spec in windows:
        window = frame.iloc[window_spec["start_idx"] : window_spec["end_idx"] + 1]
        result = integrator(window, mass_kg=mass_kg, inertia_b=inertia_b, gravity_m_s2=gravity_m_s2)
        target = window.iloc[-1]
        target_position = target[POSITION_COLUMNS].to_numpy(dtype=float)
        target_velocity = target[VELOCITY_COLUMNS].to_numpy(dtype=float)
        target_omega = target[OMEGA_COLUMNS].to_numpy(dtype=float)
        target_quat = target[QUATERNION_COLUMNS].to_numpy(dtype=float)

        rows.append(
            {
                "mode": mode,
                "split": split,
                "log_id": window_spec["log_id"],
                "segment_id": window_spec["segment_id"],
                "start_time_s": window_spec["start_time_s"],
                "horizon_s": window_spec["horizon_s"],
                "n_steps": window_spec["n_steps"],
                "position_error_m": float(np.linalg.norm(result["position_n"][-1] - target_position)),
                "velocity_error_m_s": float(np.linalg.norm(result["velocity_n"][-1] - target_velocity)),
                "attitude_error_deg": attitude_error_deg(result["quat_nb"][-1], target_quat),
                "body_rate_error_rad_s": float(np.linalg.norm(result["omega_b"][-1] - target_omega)),
            }
        )

    return pd.DataFrame(rows)


def _summary_for_groups(metrics: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=group_columns + ["n_windows"])

    grouped = metrics.groupby(group_columns, sort=True, dropna=False)
    summary = grouped.size().rename("n_windows").reset_index()
    for metric in METRIC_COLUMNS:
        aggregate = grouped[metric].agg(
            median="median",
            p25=lambda values: values.quantile(0.25),
            p75=lambda values: values.quantile(0.75),
            p90=lambda values: values.quantile(0.90),
            p95=lambda values: values.quantile(0.95),
            mean="mean",
        )
        aggregate = aggregate.rename(columns={column: f"{metric}_{column}" for column in aggregate.columns})
        summary = summary.merge(aggregate.reset_index(), on=group_columns, how="left")
    return summary


def summarize_replay_metrics(window_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon_summary = _summary_for_groups(window_metrics, ["mode", "split", "horizon_s"])
    log_summary = _summary_for_groups(window_metrics, ["mode", "split", "log_id", "horizon_s"])
    return horizon_summary, log_summary


def write_oracle_replay_artifacts(
    output_root: str | Path,
    window_metrics: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    log_summary: pd.DataFrame,
    config: dict[str, Any],
    overwrite: bool = False,
) -> None:
    output_path = Path(output_root)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(f"output root exists and is not empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    window_metrics.to_csv(output_path / "replay_window_metrics.csv", index=False)
    horizon_summary.to_csv(output_path / "horizon_summary.csv", index=False)
    log_summary.to_csv(output_path / "log_summary.csv", index=False)
    with (output_path / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_path / "README.md").open("w", encoding="utf-8") as handle:
        handle.write("# Oracle Short-Horizon Replay Sanity\n\n")
        handle.write("Open-loop, log-seeded oracle replay artifacts. These outputs are diagnostic only ")
        handle.write("and do not constitute closed-loop simulator validation.\n")


def _parse_float_list(raw: str) -> list[float]:
    return [float(part) for part in raw.split(",") if part.strip()]


def _load_metadata_dict(metadata_path: str | Path) -> dict[str, Any]:
    with Path(metadata_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate oracle short-horizon replay sanity.")
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--modes", default="oracle_teacher_forced")
    parser.add_argument("--horizons", default="0.10,0.25,0.50,1.00,2.00")
    parser.add_argument("--stride-s", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unsupported_modes = sorted(set(modes) - ORACLE_MODES)
    if unsupported_modes:
        raise ValueError(f"only oracle modes are implemented in this stage: {unsupported_modes}")

    horizons_s = _parse_float_list(args.horizons)
    split_root = Path(args.split_root)
    samples_path = split_root / f"{args.split}_samples.parquet"
    samples = pd.read_parquet(samples_path)
    metadata = _load_metadata_dict(args.metadata_path)

    metric_frames = [
        evaluate_oracle_replay(
            samples,
            metadata=metadata,
            horizons_s=horizons_s,
            stride_s=args.stride_s,
            mode=mode,
            split=args.split,
        )
        for mode in modes
    ]
    window_metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    horizon_summary, log_summary = summarize_replay_metrics(window_metrics)
    write_oracle_replay_artifacts(
        args.output_root,
        window_metrics,
        horizon_summary,
        log_summary,
        config={
            "split_root": str(split_root),
            "metadata_path": str(args.metadata_path),
            "split": args.split,
            "modes": modes,
            "horizons_s": horizons_s,
            "stride_s": args.stride_s,
        },
        overwrite=args.overwrite,
    )

    print(f"wrote {len(window_metrics)} replay metric rows to {args.output_root}")


if __name__ == "__main__":
    main()
