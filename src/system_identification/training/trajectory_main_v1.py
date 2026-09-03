"""Train-only window preparation and fitting for trajectory Main V1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from system_identification.data.trajectory_dataset import CONTROL_COLUMNS
from system_identification.evaluation.trajectory import (
    BODY_RATE_COLUMNS,
    QUATERNION_COLUMNS,
    VELOCITY_COLUMNS,
    TrajectoryWindowBatch,
    assemble_trajectory_windows,
)
from system_identification.models.trajectory import InitialTrajectoryState, TrajectoryPrediction
from system_identification.models.trajectory_main_v1 import (
    CausalHistoryTrajectoryModel,
    TorchTrajectoryPrediction,
    offset_invariant_dynamics_features,
)
from system_identification.training.trajectory_baselines import build_transition_arrays


@dataclass(frozen=True)
class HistoryTrajectoryWindowBatch:
    trajectory: TrajectoryWindowBatch
    history_state_features: np.ndarray
    history_controls: np.ndarray
    history_mask: np.ndarray


@dataclass(frozen=True)
class MainV1Stats:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    control_mean: np.ndarray
    control_std: np.ndarray
    derivative_mean: np.ndarray
    derivative_std: np.ndarray


@dataclass(frozen=True)
class MainV1Config:
    model_name: str
    use_history: bool
    use_controls: bool
    objective_steps: int
    hidden_size: int = 64
    epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    gradient_clip_norm: float = 5.0
    seed: int = 17


def assemble_history_trajectory_windows(
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    history_steps: int,
) -> HistoryTrajectoryWindowBatch:
    """Attach left-padded causal history without changing Step 1 window IDs."""
    if history_steps < 1:
        raise ValueError("history_steps must be positive")
    trajectory = assemble_trajectory_windows(samples, windows)
    groups = {
        (str(log_id), int(segment_id)): group.sort_values("sample_in_segment", kind="stable")
        for (log_id, segment_id), group in samples.loc[samples["valid_core"]].groupby(
            ["log_id", "segment_id"], sort=False
        )
    }
    state_histories: list[np.ndarray] = []
    control_histories: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for row in windows.itertuples(index=False):
        key = (str(row.log_id), int(row.segment_id))
        if key not in groups:
            raise ValueError(f"window references missing valid segment: {key}")
        group = groups[key]
        start = int(row.start_sample_in_segment)
        first = max(0, start - history_steps + 1)
        selected = group.iloc[first : start + 1]
        sample_numbers = selected["sample_in_segment"].to_numpy(dtype=np.int64)
        if not np.array_equal(sample_numbers, np.arange(first, start + 1)):
            raise ValueError(f"history is not contiguous: {row.window_id}")
        actual_count = len(selected)
        if actual_count < 1:
            raise ValueError(f"history is empty: {row.window_id}")
        phase_anchor = float(selected.iloc[-1]["relative_flap_phase_rad"])
        features = offset_invariant_dynamics_features(
            velocity_n=selected[list(VELOCITY_COLUMNS)].to_numpy(dtype=float),
            quaternion_nb=selected[list(QUATERNION_COLUMNS)].to_numpy(dtype=float),
            angular_velocity_b=selected[list(BODY_RATE_COLUMNS)].to_numpy(dtype=float),
            relative_phase_rad=selected["relative_flap_phase_rad"].to_numpy(dtype=float),
            phase_anchor_rad=np.full(actual_count, phase_anchor),
            flap_frequency_hz=selected["flap_frequency_hz"].to_numpy(dtype=float),
        )
        controls = selected[list(CONTROL_COLUMNS)].to_numpy(dtype=float)
        pad_count = history_steps - actual_count
        state_histories.append(
            np.concatenate((np.repeat(features[:1], pad_count, axis=0), features), axis=0)
        )
        control_histories.append(
            np.concatenate((np.repeat(controls[:1], pad_count, axis=0), controls), axis=0)
        )
        masks.append(
            np.concatenate((np.zeros(pad_count, dtype=bool), np.ones(actual_count, dtype=bool)))
        )
    return HistoryTrajectoryWindowBatch(
        trajectory=trajectory,
        history_state_features=np.stack(state_histories),
        history_controls=np.stack(control_histories),
        history_mask=np.stack(masks),
    )


def _standardization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    return mean, np.where(std > 1.0e-8, std, 1.0)


def fit_main_v1_stats(
    train_samples: pd.DataFrame, train_batch: HistoryTrajectoryWindowBatch
) -> MainV1Stats:
    history_values = train_batch.history_state_features[train_batch.history_mask]
    feature_mean, feature_std = _standardization(history_values)
    history_controls = train_batch.history_controls[train_batch.history_mask]
    future_controls = train_batch.trajectory.controls.reshape(-1, 4)
    control_mean, control_std = _standardization(
        np.concatenate((history_controls, future_controls), axis=0)
    )
    derivatives = build_transition_arrays(train_samples, use_controls=False).targets
    derivative_mean, derivative_std = _standardization(derivatives)
    return MainV1Stats(
        feature_mean=feature_mean,
        feature_std=feature_std,
        control_mean=control_mean,
        control_std=control_std,
        derivative_mean=derivative_mean,
        derivative_std=derivative_std,
    )


def _prediction_from_truth(
    truth: TrajectoryPrediction, indices: np.ndarray, *, state_count: int
) -> TorchTrajectoryPrediction:
    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values[indices, :state_count], dtype=torch.float32)

    return TorchTrajectoryPrediction(
        position_n=tensor(truth.position_n),
        velocity_n=tensor(truth.velocity_n),
        quaternion_nb=tensor(truth.quaternion_nb),
        angular_velocity_b=tensor(truth.angular_velocity_b),
        relative_phase_rad=tensor(truth.relative_phase_rad),
        flap_frequency_hz=tensor(truth.flap_frequency_hz),
    )


def trajectory_rollout_loss(
    prediction: TorchTrajectoryPrediction,
    truth: TorchTrajectoryPrediction,
    *,
    objective_steps: int,
) -> torch.Tensor:
    if objective_steps < 1 or prediction.position_n.shape[1] < objective_steps + 1:
        raise ValueError("objective_steps exceed prediction")
    selected = slice(1, objective_steps + 1)
    position_loss = torch.mean(
        torch.sum(torch.square(prediction.position_n[:, selected] - truth.position_n[:, selected]), dim=-1)
    )
    velocity_loss = torch.mean(
        torch.sum(
            torch.square((prediction.velocity_n[:, selected] - truth.velocity_n[:, selected]) / 2.0),
            dim=-1,
        )
    )
    predicted_q = prediction.quaternion_nb[:, selected]
    true_q = truth.quaternion_nb[:, selected]
    predicted_q = predicted_q / torch.linalg.vector_norm(predicted_q, dim=-1, keepdim=True).clamp_min(
        1.0e-8
    )
    true_q = true_q / torch.linalg.vector_norm(true_q, dim=-1, keepdim=True).clamp_min(1.0e-8)
    quaternion_dot = torch.sum(predicted_q * true_q, dim=-1)
    attitude_loss = torch.mean(4.0 * (1.0 - torch.square(quaternion_dot)) / (0.35**2))
    rate_loss = torch.mean(
        torch.sum(
            torch.square(
                (prediction.angular_velocity_b[:, selected] - truth.angular_velocity_b[:, selected])
                / 2.0
            ),
            dim=-1,
        )
    )
    phase_delta = prediction.relative_phase_rad[:, selected] - truth.relative_phase_rad[:, selected]
    phase_loss = torch.mean(2.0 - 2.0 * torch.cos(phase_delta))
    frequency_loss = torch.mean(
        torch.square(
            (prediction.flap_frequency_hz[:, selected] - truth.flap_frequency_hz[:, selected]) / 3.0
        )
    )
    return (
        position_loss
        + velocity_loss
        + attitude_loss
        + rate_loss
        + 0.1 * phase_loss
        + 0.1 * frequency_loss
    )


def _model_call(
    model: CausalHistoryTrajectoryModel,
    batch: HistoryTrajectoryWindowBatch,
    indices: np.ndarray,
    *,
    use_history: bool,
    rollout_steps: int,
    device: torch.device,
) -> tuple[TorchTrajectoryPrediction, TorchTrajectoryPrediction]:
    history_slice = slice(None) if use_history else slice(-1, None)

    def tensor(values: np.ndarray, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(values[indices], dtype=dtype, device=device)

    truth = _prediction_from_truth(
        batch.trajectory.truth, indices, state_count=rollout_steps + 1
    )
    truth = TorchTrajectoryPrediction(*(value.to(device) for value in truth))
    source = batch.trajectory.truth
    prediction = model(
        history_state_features=tensor(batch.history_state_features[:, history_slice]),
        history_controls=tensor(batch.history_controls[:, history_slice]),
        history_mask=tensor(batch.history_mask[:, history_slice], dtype=torch.bool),
        position_n=tensor(source.position_n[:, 0]),
        velocity_n=tensor(source.velocity_n[:, 0]),
        quaternion_nb=tensor(source.quaternion_nb[:, 0]),
        angular_velocity_b=tensor(source.angular_velocity_b[:, 0]),
        relative_phase_rad=tensor(source.relative_phase_rad[:, 0]),
        flap_frequency_hz=tensor(source.flap_frequency_hz[:, 0]),
        future_controls=tensor(batch.trajectory.controls[:, :rollout_steps]),
        dt_s=tensor(batch.trajectory.dt_s[:, :rollout_steps]),
    )
    return prediction, truth


def fit_history_trajectory_model(
    train_batch: HistoryTrajectoryWindowBatch,
    stats: MainV1Stats,
    config: MainV1Config,
    *,
    device: str,
) -> tuple[CausalHistoryTrajectoryModel, pd.DataFrame]:
    if config.objective_steps < 1 or config.objective_steps > train_batch.trajectory.controls.shape[1]:
        raise ValueError("objective_steps outside available training horizon")
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = CausalHistoryTrajectoryModel(
        hidden_size=config.hidden_size,
        use_controls=config.use_controls,
        feature_mean=stats.feature_mean,
        feature_std=stats.feature_std,
        control_mean=stats.control_mean,
        control_std=stats.control_std,
        derivative_mean=stats.derivative_mean,
        derivative_std=stats.derivative_std,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    sample_count = len(train_batch.trajectory.window_ids)
    history: list[dict[str, float | int | str]] = []
    for epoch in range(config.epochs):
        permutation = torch.randperm(sample_count, generator=generator).numpy()
        loss_sum = 0.0
        model.train()
        for start in range(0, sample_count, config.batch_size):
            indices = permutation[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction, truth = _model_call(
                model,
                train_batch,
                indices,
                use_history=config.use_history,
                rollout_steps=config.objective_steps,
                device=torch.device(device),
            )
            loss = trajectory_rollout_loss(
                prediction, truth, objective_steps=config.objective_steps
            )
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite training loss for {config.model_name}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(indices)
        history.append(
            {
                "model": config.model_name,
                "epoch": epoch + 1,
                "train_trajectory_loss": loss_sum / sample_count,
                "last_gradient_norm": float(gradient_norm.detach().cpu()),
            }
        )
    return model.cpu().eval(), pd.DataFrame(history)


def predict_history_trajectory_model(
    model: CausalHistoryTrajectoryModel,
    batch: HistoryTrajectoryWindowBatch,
    *,
    use_history: bool,
    batch_size: int,
    device: str,
) -> TrajectoryPrediction:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model = model.to(device).eval()
    blocks: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "position_n",
            "velocity_n",
            "quaternion_nb",
            "angular_velocity_b",
            "relative_phase_rad",
            "flap_frequency_hz",
        )
    }
    with torch.no_grad():
        for start in range(0, len(batch.trajectory.window_ids), batch_size):
            indices = np.arange(start, min(start + batch_size, len(batch.trajectory.window_ids)))
            prediction, _ = _model_call(
                model,
                batch,
                indices,
                use_history=use_history,
                rollout_steps=batch.trajectory.controls.shape[1],
                device=torch.device(device),
            )
            for name, value in vars(prediction.to_numpy()).items():
                blocks[name].append(value)
    model.cpu()
    return TrajectoryPrediction(**{name: np.concatenate(values) for name, values in blocks.items()})
