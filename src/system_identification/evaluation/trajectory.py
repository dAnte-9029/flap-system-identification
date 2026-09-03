"""Leakage-safe trajectory window assembly and rollout metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from system_identification.data.trajectory_dataset import CONTROL_COLUMNS, STATE_COLUMNS
from system_identification.models.trajectory import (
    InitialTrajectoryState,
    TrajectoryPrediction,
    attitude_error_deg,
)


POSITION_COLUMNS = STATE_COLUMNS[0:3]
VELOCITY_COLUMNS = STATE_COLUMNS[3:6]
QUATERNION_COLUMNS = STATE_COLUMNS[6:10]
BODY_RATE_COLUMNS = STATE_COLUMNS[10:13]


ERROR_COLUMNS = (
    "position_error_m",
    "velocity_error_m_s",
    "attitude_error_deg",
    "body_rate_error_rad_s",
)


@dataclass(frozen=True)
class TrajectoryWindowBatch:
    truth: TrajectoryPrediction
    controls: np.ndarray
    dt_s: np.ndarray
    window_ids: np.ndarray
    log_ids: np.ndarray
    segment_ids: np.ndarray

    def initial_state(self) -> InitialTrajectoryState:
        return InitialTrajectoryState(
            position_n=self.truth.position_n[:, 0].copy(),
            velocity_n=self.truth.velocity_n[:, 0].copy(),
            quaternion_nb=self.truth.quaternion_nb[:, 0].copy(),
            angular_velocity_b=self.truth.angular_velocity_b[:, 0].copy(),
            relative_phase_rad=self.truth.relative_phase_rad[:, 0].copy(),
            flap_frequency_hz=self.truth.flap_frequency_hz[:, 0].copy(),
        )


def assemble_trajectory_windows(samples: pd.DataFrame, windows: pd.DataFrame) -> TrajectoryWindowBatch:
    expected_counts = set(windows["state_sample_count"].astype(int))
    if len(expected_counts) != 1:
        raise ValueError("all evaluated windows must have one state_sample_count")
    state_count = expected_counts.pop()
    grouped_samples = {
        (str(log_id), int(segment_id)): group.sort_values("sample_in_segment", kind="stable")
        for (log_id, segment_id), group in samples.loc[samples["valid_core"]].groupby(
            ["log_id", "segment_id"], sort=False
        )
    }
    position: list[np.ndarray] = []
    velocity: list[np.ndarray] = []
    quaternion: list[np.ndarray] = []
    body_rate: list[np.ndarray] = []
    phase: list[np.ndarray] = []
    frequency: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    dt_values: list[np.ndarray] = []
    for row in windows.itertuples(index=False):
        key = (str(row.log_id), int(row.segment_id))
        if key not in grouped_samples:
            raise ValueError(f"window references missing valid segment: {key}")
        group = grouped_samples[key]
        start = int(row.start_sample_in_segment)
        stop = start + state_count
        selected = group.iloc[start:stop]
        if len(selected) != state_count:
            raise ValueError(f"window exceeds segment: {row.window_id}")
        sample_numbers = selected["sample_in_segment"].to_numpy(dtype=np.int64)
        if not np.array_equal(sample_numbers, np.arange(start, stop)):
            raise ValueError(f"window is not contiguous: {row.window_id}")
        timestamps = selected["timestamp_us"].to_numpy(dtype=np.int64)
        dt_s = np.diff(timestamps).astype(float) * 1.0e-6
        if np.any(dt_s <= 0.0) or np.any(dt_s > 0.05):
            raise ValueError(f"window has an invalid time gap: {row.window_id}")
        position.append(selected[list(POSITION_COLUMNS)].to_numpy(dtype=float))
        velocity.append(selected[list(VELOCITY_COLUMNS)].to_numpy(dtype=float))
        quaternion.append(selected[list(QUATERNION_COLUMNS)].to_numpy(dtype=float))
        body_rate.append(selected[list(BODY_RATE_COLUMNS)].to_numpy(dtype=float))
        phase.append(selected["relative_flap_phase_rad"].to_numpy(dtype=float))
        frequency.append(selected["flap_frequency_hz"].to_numpy(dtype=float))
        controls.append(selected.iloc[:-1][list(CONTROL_COLUMNS)].to_numpy(dtype=float))
        dt_values.append(dt_s)
    truth = TrajectoryPrediction(
        position_n=np.stack(position),
        velocity_n=np.stack(velocity),
        quaternion_nb=np.stack(quaternion),
        angular_velocity_b=np.stack(body_rate),
        relative_phase_rad=np.stack(phase),
        flap_frequency_hz=np.stack(frequency),
    )
    return TrajectoryWindowBatch(
        truth=truth,
        controls=np.stack(controls),
        dt_s=np.stack(dt_values),
        window_ids=windows["window_id"].astype(str).to_numpy(),
        log_ids=windows["log_id"].astype(str).to_numpy(),
        segment_ids=windows["segment_id"].to_numpy(dtype=np.int64),
    )


def evaluate_trajectory_predictions(
    prediction: TrajectoryPrediction,
    truth: TrajectoryPrediction,
    *,
    model_name: str,
    split: str,
    window_ids: np.ndarray,
    log_ids: np.ndarray,
    segment_ids: np.ndarray,
    horizon_steps: Mapping[float, int],
    dt_s: np.ndarray,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for horizon_s, step in horizon_steps.items():
        position_error = np.linalg.norm(
            prediction.position_n[:, step] - truth.position_n[:, step], axis=1
        )
        velocity_error = np.linalg.norm(
            prediction.velocity_n[:, step] - truth.velocity_n[:, step], axis=1
        )
        attitude_error = attitude_error_deg(
            prediction.quaternion_nb[:, step], truth.quaternion_nb[:, step]
        )
        body_rate_error = np.linalg.norm(
            prediction.angular_velocity_b[:, step] - truth.angular_velocity_b[:, step], axis=1
        )
        rows.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "split": split,
                    "window_id": window_ids,
                    "log_id": log_ids,
                    "segment_id": segment_ids,
                    "horizon_s": float(horizon_s),
                    "observed_horizon_s": np.sum(dt_s[:, :step], axis=1),
                    "position_error_m": position_error,
                    "velocity_error_m_s": velocity_error,
                    "attitude_error_deg": attitude_error,
                    "body_rate_error_rad_s": body_rate_error,
                }
            )
        )
    metrics = pd.concat(rows, ignore_index=True)
    if not np.isfinite(metrics[list(ERROR_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError(f"non-finite trajectory metric for model {model_name}")
    return metrics


def aggregate_trajectory_metrics(
    window_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = ["model", "split", "horizon_s", "log_id"]
    per_log_rows: list[dict[str, float | int | str]] = []
    metric_names = {
        "position_error_m": "position_rmse_m",
        "velocity_error_m_s": "velocity_rmse_m_s",
        "attitude_error_deg": "attitude_rmse_deg",
        "body_rate_error_rad_s": "body_rate_rmse_rad_s",
    }
    for key, group in window_metrics.groupby(group_columns, sort=True):
        model, split, horizon_s, log_id = key
        row: dict[str, float | int | str] = {
            "model": str(model),
            "split": str(split),
            "horizon_s": float(horizon_s),
            "log_id": str(log_id),
            "n_windows": int(len(group)),
        }
        for source, target in metric_names.items():
            values = group[source].to_numpy(dtype=float)
            row[target] = float(np.sqrt(np.mean(np.square(values))))
        per_log_rows.append(row)
    per_log = pd.DataFrame(per_log_rows)

    aggregate_rows: list[dict[str, float | int | str]] = []
    for key, group in window_metrics.groupby(["model", "split", "horizon_s"], sort=True):
        model, split, horizon_s = key
        log_group = per_log.loc[
            (per_log["model"] == model)
            & (per_log["split"] == split)
            & (per_log["horizon_s"] == horizon_s)
        ]
        row = {
            "model": str(model),
            "split": str(split),
            "horizon_s": float(horizon_s),
            "n_logs": int(len(log_group)),
            "n_windows": int(len(group)),
        }
        for source, target in metric_names.items():
            values = group[source].to_numpy(dtype=float)
            row[f"{target}_pooled"] = float(np.sqrt(np.mean(np.square(values))))
            row[f"{target}_per_log_macro"] = float(np.mean(log_group[target]))
            row[f"{target}_per_log_min"] = float(np.min(log_group[target]))
            row[f"{target}_per_log_max"] = float(np.max(log_group[target]))
        aggregate_rows.append(row)
    return pd.DataFrame(aggregate_rows), per_log
