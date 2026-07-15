"""Single-epoch training and validation helpers."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from system_identification.training.losses import regression_loss


def _normalized_lr_scheduler(name: str | None) -> str:
    resolved = (name or "none").strip().lower()
    aliases = {"constant": "none", "off": "none", "disabled": "none"}
    resolved = aliases.get(resolved, resolved)
    if resolved not in {"none", "warmup_cosine"}:
        raise ValueError("lr_scheduler must be one of: none, warmup_cosine")
    return resolved


def _warmup_cosine_lr_lambda(*, warmup_steps: int, total_steps: int):
    resolved_total_steps = max(int(total_steps), 1)
    resolved_warmup_steps = max(int(warmup_steps), 0)

    def lr_lambda(step_index: int) -> float:
        step = int(step_index) + 1
        if resolved_warmup_steps > 0 and step <= resolved_warmup_steps:
            return max(float(step) / float(resolved_warmup_steps), 1e-8)
        decay_steps = max(resolved_total_steps - resolved_warmup_steps, 1)
        progress = min(max((step - resolved_warmup_steps) / decay_steps, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def _ema_update_state(ema_state: dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if not torch.is_floating_point(value):
                ema_state[name] = value.detach().clone()
                continue
            ema_state[name].mul_(float(decay)).add_(value.detach(), alpha=1.0 - float(decay))


def _train_scaled_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    *,
    use_amp: bool,
    target_loss_weights: torch.Tensor,
    loss_type: str,
    huber_delta: float,
) -> float:
    model.train()
    train_loss_sum = 0.0
    train_sample_count = 0
    amp_enabled = use_amp and device.type == "cuda"

    for batch_features, batch_targets in loader:
        batch_features = batch_features.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            predictions = model(batch_features)
            loss = regression_loss(
                predictions,
                batch_targets,
                target_loss_weights=target_loss_weights,
                loss_type=loss_type,
                huber_delta=huber_delta,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_count = len(batch_features)
        train_loss_sum += float(loss.item()) * batch_count
        train_sample_count += batch_count

    return train_loss_sum / max(train_sample_count, 1)


def _train_sequence_scaled_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    *,
    use_amp: bool,
    target_loss_weights: torch.Tensor,
    loss_type: str,
    huber_delta: float,
    prior_loss_weight: float,
    gradient_clip_norm: float | None,
    scheduler: Any = None,
    ema_state: dict[str, torch.Tensor] | None = None,
    ema_decay: float = 0.0,
) -> tuple[float, float, float]:
    model.train()
    train_loss_sum = 0.0
    train_supervised_loss_sum = 0.0
    train_prior_loss_sum = 0.0
    train_sample_count = 0
    amp_enabled = use_amp and device.type == "cuda"

    for batch in loader:
        batch_sequence, batch_current, batch_targets = batch[:3]
        batch_prior_targets = batch[3] if len(batch) == 4 else None
        batch_sequence = batch_sequence.to(device, non_blocking=True)
        batch_current = batch_current.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)
        if batch_prior_targets is not None:
            batch_prior_targets = batch_prior_targets.to(device, non_blocking=True)
        current_arg = batch_current if batch_current.shape[1] > 0 else None

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            predictions = model(batch_sequence, current_arg)
            supervised_loss = regression_loss(
                predictions,
                batch_targets,
                target_loss_weights=target_loss_weights,
                loss_type=loss_type,
                huber_delta=huber_delta,
            )
            prior_loss = (
                regression_loss(
                    predictions,
                    batch_prior_targets,
                    target_loss_weights=target_loss_weights,
                    loss_type=loss_type,
                    huber_delta=huber_delta,
                )
                if batch_prior_targets is not None
                else torch.zeros((), dtype=predictions.dtype, device=predictions.device)
            )
            loss = supervised_loss + prior_loss_weight * prior_loss

        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        if ema_state is not None:
            _ema_update_state(ema_state, model, ema_decay)

        batch_count = len(batch_sequence)
        train_loss_sum += float(loss.item()) * batch_count
        train_supervised_loss_sum += float(supervised_loss.item()) * batch_count
        train_prior_loss_sum += float(prior_loss.item()) * batch_count
        train_sample_count += batch_count

    return (
        train_loss_sum / max(train_sample_count, 1),
        train_supervised_loss_sum / max(train_sample_count, 1),
        train_prior_loss_sum / max(train_sample_count, 1),
    )


def _train_rollout_scaled_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    *,
    use_amp: bool,
    target_loss_weights: torch.Tensor,
    loss_type: str,
    huber_delta: float,
) -> float:
    model.train()
    train_loss_sum = 0.0
    train_sample_count = 0
    amp_enabled = use_amp and device.type == "cuda"

    for batch_context, batch_rollout, batch_current, batch_targets in loader:
        batch_context = batch_context.to(device, non_blocking=True)
        batch_rollout = batch_rollout.to(device, non_blocking=True)
        batch_current = batch_current.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)
        current_arg = batch_current if batch_current.shape[2] > 0 else None

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            predictions = model(batch_context, batch_rollout, current_arg)
            loss = regression_loss(
                predictions.reshape(-1, predictions.shape[-1]),
                batch_targets.reshape(-1, batch_targets.shape[-1]),
                target_loss_weights=target_loss_weights,
                loss_type=loss_type,
                huber_delta=huber_delta,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_count = len(batch_context)
        train_loss_sum += float(loss.item()) * batch_count
        train_sample_count += batch_count

    return train_loss_sum / max(train_sample_count, 1)


def _evaluate_scaled_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
    target_loss_weights: torch.Tensor,
    loss_type: str,
    huber_delta: float,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    amp_enabled = use_amp and device.type == "cuda"
    with torch.no_grad():
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions = model(batch_features)
                loss = regression_loss(
                    predictions,
                    batch_targets,
                    target_loss_weights=target_loss_weights,
                    loss_type=loss_type,
                    huber_delta=huber_delta,
                )
            batch_size = len(batch_features)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    return total_loss / max(total_samples, 1)


def _evaluate_sequence_scaled_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
    target_loss_weights: torch.Tensor,
    loss_type: str,
    huber_delta: float,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    amp_enabled = use_amp and device.type == "cuda"
    with torch.no_grad():
        for batch_sequence, batch_current, batch_targets in loader:
            batch_sequence = batch_sequence.to(device, non_blocking=True)
            batch_current = batch_current.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            current_arg = batch_current if batch_current.shape[1] > 0 else None
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions = model(batch_sequence, current_arg)
                loss = regression_loss(
                    predictions,
                    batch_targets,
                    target_loss_weights=target_loss_weights,
                    loss_type=loss_type,
                    huber_delta=huber_delta,
                )
            batch_size = len(batch_sequence)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    return total_loss / max(total_samples, 1)


def _evaluate_rollout_scaled_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
    target_loss_weights: torch.Tensor,
    loss_type: str,
    huber_delta: float,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    amp_enabled = use_amp and device.type == "cuda"
    with torch.no_grad():
        for batch_context, batch_rollout, batch_current, batch_targets in loader:
            batch_context = batch_context.to(device, non_blocking=True)
            batch_rollout = batch_rollout.to(device, non_blocking=True)
            batch_current = batch_current.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            current_arg = batch_current if batch_current.shape[2] > 0 else None
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions = model(batch_context, batch_rollout, current_arg)
                loss = regression_loss(
                    predictions.reshape(-1, predictions.shape[-1]),
                    batch_targets.reshape(-1, batch_targets.shape[-1]),
                    target_loss_weights=target_loss_weights,
                    loss_type=loss_type,
                    huber_delta=huber_delta,
                )
            batch_size = len(batch_context)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    return total_loss / max(total_samples, 1)
