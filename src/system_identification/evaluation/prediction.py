"""Frozen normalization application and low-level batch prediction helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


_FOUNDATION_EXPORTS = {
    "_make_loader": ("system_identification.training.loaders", "_make_loader"),
    "_make_rollout_loader": ("system_identification.training.loaders", "_make_rollout_loader"),
    "_make_sequence_loader": ("system_identification.training.loaders", "_make_sequence_loader"),
    "_inverse_transform_targets": (
        "system_identification.training.normalization",
        "_inverse_transform_targets",
    ),
    "_transform_features": ("system_identification.training.normalization", "_transform_features"),
    "_transform_rollout_features": (
        "system_identification.training.normalization",
        "_transform_rollout_features",
    ),
    "_transform_sequence_features": (
        "system_identification.training.normalization",
        "_transform_sequence_features",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _FOUNDATION_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol_name = _FOUNDATION_EXPORTS[name]
    from importlib import import_module

    return getattr(import_module(module_name), symbol_name)

def _resolve_device(requested: str | None) -> torch.device:
    if requested and requested.lower() != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")






def _as_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)








def _predict_scaled_batches(
    model: nn.Module,
    features_scaled: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    from system_identification.training.loaders import _make_loader

    loader = _make_loader(
        features_scaled,
        None,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    outputs: list[np.ndarray] = []
    amp_enabled = use_amp and device.type == "cuda"
    model.eval()
    with torch.no_grad():
        for (batch_features,) in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                batch_predictions = model(batch_features)
            outputs.append(batch_predictions.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)

def _predict_sequence_scaled_batches(
    model: nn.Module,
    sequence_features_scaled: np.ndarray,
    current_features_scaled: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    from system_identification.training.loaders import _make_sequence_loader

    loader = _make_sequence_loader(
        sequence_features_scaled,
        current_features_scaled,
        None,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    outputs: list[np.ndarray] = []
    amp_enabled = use_amp and device.type == "cuda"
    model.eval()
    with torch.no_grad():
        for batch_sequence, batch_current in loader:
            batch_sequence = batch_sequence.to(device, non_blocking=True)
            batch_current = batch_current.to(device, non_blocking=True)
            current_arg = batch_current if batch_current.shape[1] > 0 else None
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                batch_predictions = model(batch_sequence, current_arg)
            outputs.append(batch_predictions.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _predict_rollout_scaled_batches(
    model: nn.Module,
    context_features_scaled: np.ndarray,
    rollout_features_scaled: np.ndarray,
    current_features_scaled: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    from system_identification.training.loaders import _make_rollout_loader

    loader = _make_rollout_loader(
        context_features_scaled,
        rollout_features_scaled,
        current_features_scaled,
        None,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    outputs: list[np.ndarray] = []
    amp_enabled = use_amp and device.type == "cuda"
    model.eval()
    with torch.no_grad():
        for batch_context, batch_rollout, batch_current in loader:
            batch_context = batch_context.to(device, non_blocking=True)
            batch_rollout = batch_rollout.to(device, non_blocking=True)
            batch_current = batch_current.to(device, non_blocking=True)
            current_arg = batch_current if batch_current.shape[2] > 0 else None
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                batch_predictions = model(batch_context, batch_rollout, current_arg)
            outputs.append(batch_predictions.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)
