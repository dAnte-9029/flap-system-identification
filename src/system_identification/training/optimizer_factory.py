"""Pure construction helpers for the existing optimizer and scheduler configuration."""

from __future__ import annotations

import torch
from torch import nn

from system_identification.training.loop import _warmup_cosine_lr_lambda


def build_adamw_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def build_training_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    warmup_steps: int,
    total_steps: int,
):
    if scheduler_name == "warmup_cosine":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=_warmup_cosine_lr_lambda(warmup_steps=warmup_steps, total_steps=total_steps),
        )
    return None
