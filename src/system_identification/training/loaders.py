"""Low-level tensor dataset and DataLoader construction helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def _make_loader(
    features: np.ndarray,
    targets: np.ndarray | None,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    feature_tensor = torch.from_numpy(features.astype(np.float32, copy=False))
    if targets is None:
        dataset = TensorDataset(feature_tensor)
    else:
        target_tensor = torch.from_numpy(targets.astype(np.float32, copy=False))
        dataset = TensorDataset(feature_tensor, target_tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _make_sequence_loader(
    sequence_features: np.ndarray,
    current_features: np.ndarray,
    targets: np.ndarray | None,
    *,
    prior_targets: np.ndarray | None = None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    sequence_tensor = torch.from_numpy(sequence_features.astype(np.float32, copy=False))
    current_tensor = torch.from_numpy(current_features.astype(np.float32, copy=False))
    if targets is None:
        if prior_targets is not None:
            raise ValueError("prior_targets require supervised targets")
        dataset = TensorDataset(sequence_tensor, current_tensor)
    else:
        target_tensor = torch.from_numpy(targets.astype(np.float32, copy=False))
        if prior_targets is None:
            dataset = TensorDataset(sequence_tensor, current_tensor, target_tensor)
        else:
            prior_tensor = torch.from_numpy(prior_targets.astype(np.float32, copy=False))
            dataset = TensorDataset(sequence_tensor, current_tensor, target_tensor, prior_tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _make_rollout_loader(
    context_features: np.ndarray,
    rollout_features: np.ndarray,
    current_features: np.ndarray,
    targets: np.ndarray | None,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    context_tensor = torch.from_numpy(context_features.astype(np.float32, copy=False))
    rollout_tensor = torch.from_numpy(rollout_features.astype(np.float32, copy=False))
    current_tensor = torch.from_numpy(current_features.astype(np.float32, copy=False))
    if targets is None:
        dataset = TensorDataset(context_tensor, rollout_tensor, current_tensor)
    else:
        target_tensor = torch.from_numpy(targets.astype(np.float32, copy=False))
        dataset = TensorDataset(context_tensor, rollout_tensor, current_tensor, target_tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
