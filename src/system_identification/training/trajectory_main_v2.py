"""Train actuator residuals on a frozen history-only trajectory backbone."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from system_identification.models.trajectory_main_v2 import ActuatorAwareTrajectoryModel
from system_identification.training.trajectory_main_v1 import (
    HistoryTrajectoryWindowBatch,
    _model_call,
    trajectory_rollout_loss,
)


@dataclass(frozen=True)
class MainV2Stats:
    tail_mean: np.ndarray
    tail_std: np.ndarray


@dataclass(frozen=True)
class MainV2Config:
    model_name: str
    use_drive: bool
    use_tail: bool
    gated_tail: bool
    objective_steps: int
    epochs: int = 25
    batch_size: int = 256
    learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-5
    gradient_clip_norm: float = 5.0
    drive_supervision_weight: float = 0.20
    residual_l2: float = 1.0e-3
    tail_gate_l1: float = 1.0e-2
    drive_tau_s: float = 0.10
    tail_tau_s: float = 0.04
    initial_tail_gate: float = 0.05
    seed: int = 29


def _tail_transform_numpy(controls: np.ndarray) -> np.ndarray:
    values = np.asarray(controls, dtype=float)
    if values.shape[-1] != 4:
        raise ValueError("controls must end in four Step 1 channels")
    return np.stack(
        (
            0.5 * (values[..., 1] + values[..., 2]),
            0.5 * (values[..., 1] - values[..., 2]),
            values[..., 3],
        ),
        axis=-1,
    )


def fit_main_v2_stats(train_batch: HistoryTrajectoryWindowBatch) -> MainV2Stats:
    history = _tail_transform_numpy(train_batch.history_controls)[train_batch.history_mask]
    future = _tail_transform_numpy(train_batch.trajectory.controls).reshape(-1, 3)
    values = np.concatenate((history, future), axis=0)
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    return MainV2Stats(tail_mean=mean, tail_std=np.where(std > 1.0e-8, std, 1.0))


def fit_actuator_aware_model(
    train_batch: HistoryTrajectoryWindowBatch,
    base_model: CausalHistoryTrajectoryModel,
    stats: MainV2Stats,
    config: MainV2Config,
    *,
    device: str,
) -> tuple[ActuatorAwareTrajectoryModel, pd.DataFrame]:
    if config.objective_steps < 1 or config.objective_steps > train_batch.trajectory.controls.shape[1]:
        raise ValueError("objective_steps outside available training horizon")
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = ActuatorAwareTrajectoryModel(
        base_model=base_model,
        use_drive=config.use_drive,
        use_tail=config.use_tail,
        gated_tail=config.gated_tail,
        tail_mean=stats.tail_mean,
        tail_std=stats.tail_std,
        drive_tau_s=config.drive_tau_s,
        tail_tau_s=config.tail_tau_s,
        initial_tail_gate=config.initial_tail_gate,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("actuator-aware configuration has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    sample_count = len(train_batch.trajectory.window_ids)
    history: list[dict[str, float | int | str]] = []
    for epoch in range(config.epochs):
        permutation = torch.randperm(sample_count, generator=generator).numpy()
        totals = {"loss": 0.0, "trajectory": 0.0, "drive": 0.0, "regularization": 0.0}
        model.train()
        for start in range(0, sample_count, config.batch_size):
            indices = permutation[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction, truth = _model_call(
                model,
                train_batch,
                indices,
                use_history=True,
                rollout_steps=config.objective_steps,
                device=torch.device(device),
            )
            trajectory_loss = trajectory_rollout_loss(
                prediction, truth, objective_steps=config.objective_steps
            )
            if config.use_drive:
                drive_loss = torch.mean(
                    torch.square(
                        prediction.flap_frequency_hz[:, 1 : config.objective_steps + 1]
                        - truth.flap_frequency_hz[:, 1 : config.objective_steps + 1]
                    )
                )
            else:
                drive_loss = trajectory_loss.new_zeros(())
            regularization = model.control_regularization_loss(
                residual_l2=config.residual_l2,
                tail_gate_l1=config.tail_gate_l1,
            )
            loss = trajectory_loss + config.drive_supervision_weight * drive_loss + regularization
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite training loss for {config.model_name}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip_norm)
            optimizer.step()
            batch_count = len(indices)
            totals["loss"] += float(loss.detach().cpu()) * batch_count
            totals["trajectory"] += float(trajectory_loss.detach().cpu()) * batch_count
            totals["drive"] += float(drive_loss.detach().cpu()) * batch_count
            totals["regularization"] += float(regularization.detach().cpu()) * batch_count
        gates = model.tail_gate_values().detach().cpu().numpy()
        history.append(
            {
                "model": config.model_name,
                "epoch": epoch + 1,
                "train_total_loss": totals["loss"] / sample_count,
                "train_trajectory_loss": totals["trajectory"] / sample_count,
                "train_drive_frequency_mse_hz2": totals["drive"] / sample_count,
                "train_control_regularization": totals["regularization"] / sample_count,
                "last_gradient_norm": float(gradient_norm.detach().cpu()),
                "symmetric_gate": float(gates[0]),
                "differential_gate": float(gates[1]),
                "rudder_gate": float(gates[2]),
            }
        )
    return model.cpu().eval(), pd.DataFrame(history)
