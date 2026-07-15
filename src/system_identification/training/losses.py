"""Pure regression-loss and target-weight helpers."""

from __future__ import annotations

import math

import numpy as np
import torch

def resolve_target_loss_weights(
    target_columns: list[str],
    target_loss_weights: str | dict[str, float] | list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> np.ndarray:
    if target_loss_weights is None or target_loss_weights == "":
        return np.ones(len(target_columns), dtype=np.float32)

    if isinstance(target_loss_weights, str):
        parsed: dict[str, float] = {}
        for item in target_loss_weights.split(","):
            if not item.strip():
                continue
            if "=" not in item:
                raise ValueError(f"Invalid target loss weight item: {item!r}")
            target_name, raw_weight = item.split("=", 1)
            parsed[target_name.strip()] = float(raw_weight.strip())
        missing = [target for target in target_columns if target not in parsed]
        unknown = [target for target in parsed if target not in target_columns]
        if missing or unknown:
            raise ValueError(f"Target loss weights mismatch; missing={missing}, unknown={unknown}")
        weights = np.array([parsed[target] for target in target_columns], dtype=np.float32)
    elif isinstance(target_loss_weights, dict):
        missing = [target for target in target_columns if target not in target_loss_weights]
        unknown = [target for target in target_loss_weights if target not in target_columns]
        if missing or unknown:
            raise ValueError(f"Target loss weights mismatch; missing={missing}, unknown={unknown}")
        weights = np.array([target_loss_weights[target] for target in target_columns], dtype=np.float32)
    else:
        weights = np.asarray(target_loss_weights, dtype=np.float32)
        if weights.shape != (len(target_columns),):
            raise ValueError(f"target_loss_weights must have shape ({len(target_columns)},), got {weights.shape}")

    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("target_loss_weights must be finite and non-negative")
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("At least one target loss weight must be positive")
    return weights.astype(np.float32, copy=False)


def _target_loss_weights_as_dict(target_columns: list[str], weights: np.ndarray) -> dict[str, float]:
    return {target: float(weight) for target, weight in zip(target_columns, weights)}


def _normalized_loss_type(loss_type: str | None) -> str:
    normalized = (loss_type or "mse").lower()
    if normalized not in {"mse", "huber"}:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    return normalized


def regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_loss_weights: torch.Tensor,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    resolved_loss_type = _normalized_loss_type(loss_type)
    if huber_delta <= 0.0 or not math.isfinite(float(huber_delta)):
        raise ValueError("huber_delta must be positive and finite")

    error = predictions - targets
    if resolved_loss_type == "mse":
        per_target_loss = torch.square(error)
    else:
        abs_error = torch.abs(error)
        delta = torch.as_tensor(huber_delta, device=predictions.device, dtype=predictions.dtype)
        quadratic = torch.minimum(abs_error, delta)
        linear = abs_error - quadratic
        per_target_loss = 0.5 * torch.square(quadratic) + delta * linear

    weights = target_loss_weights.to(device=predictions.device, dtype=predictions.dtype)
    weighted = per_target_loss * weights
    return weighted.sum() / (predictions.shape[0] * torch.clamp(weights.sum(), min=1e-8))
