"""Train-only preparation and fitting for trajectory dynamics baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from system_identification.data.trajectory_dataset import CONTROL_COLUMNS
from system_identification.models.neural import MLPRegressor
from system_identification.models.trajectory import (
    MLPDynamicsPredictor,
    RidgeDynamicsPredictor,
    dynamics_features,
)
from system_identification.physics.delaurier.airflow import quaternion_wxyz_to_rotation_body_to_ned


VELOCITY_COLUMNS = ("velocity_ned_m_s_x", "velocity_ned_m_s_y", "velocity_ned_m_s_z")
QUATERNION_COLUMNS = ("attitude_q_w", "attitude_q_x", "attitude_q_y", "attitude_q_z")
BODY_RATE_COLUMNS = (
    "angular_velocity_body_rad_s_x",
    "angular_velocity_body_rad_s_y",
    "angular_velocity_body_rad_s_z",
)


@dataclass(frozen=True)
class TransitionArrays:
    features: np.ndarray
    targets: np.ndarray
    dt_s: np.ndarray
    log_ids: np.ndarray


def build_transition_arrays(samples: pd.DataFrame, *, use_controls: bool) -> TransitionArrays:
    required = {
        "log_id",
        "segment_id",
        "sample_in_segment",
        "timestamp_us",
        "valid_core",
        *VELOCITY_COLUMNS,
        *QUATERNION_COLUMNS,
        *BODY_RATE_COLUMNS,
        "relative_flap_phase_rad",
        "flap_frequency_hz",
        *CONTROL_COLUMNS,
    }
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"trajectory samples missing columns: {missing}")

    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    dt_blocks: list[np.ndarray] = []
    log_blocks: list[np.ndarray] = []
    core = samples.loc[samples["valid_core"] & (samples["segment_id"] >= 0)]
    for (log_id, _), group in core.groupby(["log_id", "segment_id"], sort=False):
        ordered = group.sort_values("sample_in_segment", kind="stable")
        if len(ordered) < 2:
            continue
        sample_number = ordered["sample_in_segment"].to_numpy(dtype=np.int64)
        dt_s = np.diff(ordered["timestamp_us"].to_numpy(dtype=np.int64)).astype(float) * 1.0e-6
        consecutive = (np.diff(sample_number) == 1) & (dt_s > 0.0) & (dt_s <= 0.05)
        if not np.any(consecutive):
            continue
        current_indices = np.flatnonzero(consecutive)
        next_indices = current_indices + 1
        velocity = ordered[list(VELOCITY_COLUMNS)].to_numpy(dtype=float)
        quaternion = ordered[list(QUATERNION_COLUMNS)].to_numpy(dtype=float)
        body_rate = ordered[list(BODY_RATE_COLUMNS)].to_numpy(dtype=float)
        phase = ordered["relative_flap_phase_rad"].to_numpy(dtype=float)
        frequency = ordered["flap_frequency_hz"].to_numpy(dtype=float)
        controls = ordered[list(CONTROL_COLUMNS)].to_numpy(dtype=float)
        feature_blocks.append(
            dynamics_features(
                velocity_n=velocity[current_indices],
                quaternion_nb=quaternion[current_indices],
                angular_velocity_b=body_rate[current_indices],
                relative_phase_rad=phase[current_indices],
                flap_frequency_hz=frequency[current_indices],
                controls=controls[current_indices] if use_controls else None,
            )
        )
        rotation, valid = quaternion_wxyz_to_rotation_body_to_ned(quaternion[current_indices])
        if not np.all(valid):
            raise ValueError("valid_core transition contains an invalid quaternion")
        selected_dt = dt_s[current_indices]
        acceleration_n = (velocity[next_indices] - velocity[current_indices]) / selected_dt[:, None]
        acceleration_b = np.einsum("nji,nj->ni", rotation, acceleration_n)
        angular_acceleration_b = (
            body_rate[next_indices] - body_rate[current_indices]
        ) / selected_dt[:, None]
        frequency_rate = (frequency[next_indices] - frequency[current_indices]) / selected_dt
        target_blocks.append(np.column_stack([acceleration_b, angular_acceleration_b, frequency_rate]))
        dt_blocks.append(selected_dt)
        log_blocks.append(np.full(len(selected_dt), str(log_id), dtype=object))
    if not feature_blocks:
        raise ValueError("no valid consecutive transitions found")
    features = np.concatenate(feature_blocks)
    targets = np.concatenate(target_blocks)
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("transition arrays contain non-finite values")
    return TransitionArrays(
        features=features,
        targets=targets,
        dt_s=np.concatenate(dt_blocks),
        log_ids=np.concatenate(log_blocks),
    )


def _standardization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    std = np.where(std > 1.0e-8, std, 1.0)
    return mean, std


def fit_ridge_arrays(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
    use_controls: bool,
) -> dict[str, np.ndarray | bool]:
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError("features and targets must be nonempty two-dimensional arrays with equal rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("features and targets must be finite")
    feature_mean, feature_std = _standardization(x)
    target_mean, target_std = _standardization(y)
    x_scaled = (x - feature_mean) / feature_std
    y_scaled = (y - target_mean) / target_std
    regularized = x_scaled.T @ x_scaled + float(alpha) * np.eye(x.shape[1])
    coefficients = np.linalg.solve(regularized, x_scaled.T @ y_scaled)
    intercept = np.mean(y_scaled - x_scaled @ coefficients, axis=0)
    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "coefficients": coefficients,
        "intercept": intercept,
        "use_controls": bool(use_controls),
    }


def fit_ridge_dynamics(
    samples: pd.DataFrame, *, alpha: float, use_controls: bool
) -> tuple[RidgeDynamicsPredictor, TransitionArrays]:
    transitions = build_transition_arrays(samples, use_controls=use_controls)
    fit = fit_ridge_arrays(
        transitions.features,
        transitions.targets,
        alpha=alpha,
        use_controls=use_controls,
    )
    return RidgeDynamicsPredictor(**fit), transitions


def fit_mlp_dynamics(
    samples: pd.DataFrame,
    *,
    use_controls: bool = True,
    hidden_sizes: tuple[int, ...] = (64, 64),
    epochs: int = 40,
    batch_size: int = 1024,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    seed: int = 17,
) -> tuple[MLPDynamicsPredictor, TransitionArrays, pd.DataFrame]:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    transitions = build_transition_arrays(samples, use_controls=use_controls)
    feature_mean, feature_std = _standardization(transitions.features)
    target_mean, target_std = _standardization(transitions.targets)
    features = ((transitions.features - feature_mean) / feature_std).astype(np.float32)
    targets = ((transitions.targets - target_mean) / target_std).astype(np.float32)

    torch.manual_seed(seed)
    module = MLPRegressor(
        input_dim=features.shape[1],
        output_dim=targets.shape[1],
        hidden_sizes=hidden_sizes,
        dropout=0.0,
    ).cpu()
    optimizer = torch.optim.AdamW(
        module.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    x = torch.from_numpy(features)
    y = torch.from_numpy(targets)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        permutation = torch.randperm(len(x), generator=generator)
        loss_sum = 0.0
        module.train()
        for start in range(0, len(x), batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = module(x[indices])
            loss = torch.mean(torch.square(prediction - y[indices]))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        history.append({"epoch": epoch + 1, "train_standardized_mse": loss_sum / len(x)})
    predictor = MLPDynamicsPredictor(
        module=module,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        use_controls=use_controls,
    )
    return predictor, transitions, pd.DataFrame(history)
