"""Training-derived normalization fit and frozen transforms."""

from __future__ import annotations

import numpy as np

def _fit_feature_stats(
    features: np.ndarray,
    *,
    raw_feature_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    medians = np.nanmedian(features, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    imputed = np.where(np.isfinite(features), features, medians)
    means = imputed.mean(axis=0)
    stds = imputed.std(axis=0)
    stds = np.where(stds > 1e-8, stds, 1.0)
    if raw_feature_indices:
        means[raw_feature_indices] = 0.0
        stds[raw_feature_indices] = 1.0
    return medians.astype(np.float32), means.astype(np.float32), stds.astype(np.float32)


def _fit_target_stats(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(targets).all():
        raise ValueError("Targets contain non-finite values")
    means = targets.mean(axis=0)
    stds = targets.std(axis=0)
    stds = np.where(stds > 1e-8, stds, 1.0)
    return means.astype(np.float32), stds.astype(np.float32)


def _transform_targets(targets: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return ((targets - means) / stds).astype(np.float32)


def _fit_sequence_feature_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = features.reshape(-1, features.shape[-1])
    return _fit_feature_stats(flat)


def _fit_rollout_feature_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = features.reshape(-1, features.shape[-1])
    return _fit_feature_stats(flat)

def _transform_features(features: np.ndarray, medians: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    imputed = np.where(np.isfinite(features), features, medians)
    return ((imputed - means) / stds).astype(np.float32)


def _inverse_transform_targets(targets_scaled: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return (targets_scaled * stds + means).astype(np.float32)


def _transform_sequence_features(
    features: np.ndarray,
    medians: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    flat = features.reshape(-1, features.shape[-1])
    transformed = _transform_features(flat, medians, means, stds)
    return transformed.reshape(features.shape).astype(np.float32, copy=False)


def _transform_rollout_features(
    features: np.ndarray,
    medians: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    flat = features.reshape(-1, features.shape[-1])
    transformed = _transform_features(flat, medians, means, stds)
    return transformed.reshape(features.shape).astype(np.float32, copy=False)
