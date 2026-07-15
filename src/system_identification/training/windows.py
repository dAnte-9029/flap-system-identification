"""Deterministic feature/target frame and causal-window construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from system_identification.models.features import (
    DEFAULT_FEATURE_COLUMNS,
    _with_derived_columns,
)

DEFAULT_TARGET_COLUMNS = ["fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b"]

def prepare_feature_target_frames(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = feature_columns or DEFAULT_FEATURE_COLUMNS
    target_cols = target_columns or DEFAULT_TARGET_COLUMNS
    derived = _with_derived_columns(frame)

    missing_features = [column for column in feature_cols if column not in derived.columns]
    missing_targets = [column for column in target_cols if column not in derived.columns]
    if missing_features or missing_targets:
        missing = missing_features + missing_targets
        raise ValueError(f"Missing required training columns: {missing}")

    features = derived.loc[:, feature_cols].copy()
    targets = derived.loc[:, target_cols].copy()
    return features, targets


def _normalized_window_mode(window_mode: str | None) -> str:
    normalized = (window_mode or "single").lower()
    if normalized not in {"single", "causal", "centered"}:
        raise ValueError(f"Unknown window_mode: {window_mode}")
    return normalized


def _window_offsets(window_mode: str, window_radius: int) -> list[int]:
    if window_radius < 0:
        raise ValueError("window_radius must be non-negative")
    resolved_mode = _normalized_window_mode(window_mode)
    if resolved_mode == "single" or window_radius == 0:
        return [0]
    if resolved_mode == "causal":
        return list(range(-window_radius, 1))
    return list(range(-window_radius, window_radius + 1))


def _window_feature_name(column: str, offset: int) -> str:
    return f"{column}@t{offset:+d}"


def prepare_windowed_feature_target_frames(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_columns: list[str] | None = None,
    *,
    window_mode: str = "single",
    window_radius: int = 0,
    window_feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features, targets = prepare_feature_target_frames(frame, feature_columns, target_columns)
    offsets = _window_offsets(window_mode, window_radius)
    if offsets == [0]:
        return features, targets
    selected_window_features = set(window_feature_columns or list(features.columns))
    unknown_window_features = selected_window_features - set(features.columns)
    if unknown_window_features:
        raise ValueError(f"Unknown window feature columns: {sorted(unknown_window_features)}")

    group_columns = [column for column in ["log_id", "segment_id"] if column in frame.columns]
    if group_columns:
        groups = frame.groupby(group_columns, sort=False).indices.values()
    else:
        groups = [np.arange(len(frame))]

    windowed_parts: list[pd.DataFrame] = []
    target_parts: list[pd.DataFrame] = []
    for indices in groups:
        index = np.asarray(indices)
        group_features = features.iloc[index].reset_index(drop=True)
        group_targets = targets.iloc[index].reset_index(drop=True)
        if len(group_features) < len(offsets):
            continue

        shifted_columns: list[pd.DataFrame] = []
        valid = np.ones(len(group_features), dtype=bool)
        for offset in offsets:
            shifted = group_features.loc[:, [column for column in group_features.columns if column in selected_window_features]].shift(
                periods=-offset
            )
            shifted.columns = [_window_feature_name(column, offset) for column in shifted.columns]
            valid &= shifted.notna().all(axis=1).to_numpy()
            if len(shifted.columns) > 0:
                shifted_columns.append(shifted)

        current_only_columns = [column for column in group_features.columns if column not in selected_window_features]
        if current_only_columns:
            current = group_features.loc[:, current_only_columns].copy()
            current.columns = [_window_feature_name(column, 0) for column in current.columns]
            shifted_columns.append(current)

        if not np.any(valid):
            continue
        windowed_parts.append(pd.concat(shifted_columns, axis=1).loc[valid].reset_index(drop=True))
        target_parts.append(group_targets.loc[valid].reset_index(drop=True))

    if not windowed_parts:
        raise ValueError("No complete windowed samples were produced")
    return pd.concat(windowed_parts, ignore_index=True), pd.concat(target_parts, ignore_index=True)


def prepare_causal_sequence_feature_target_frames(
    frame: pd.DataFrame,
    sequence_feature_columns: list[str],
    current_feature_columns: list[str],
    target_columns: list[str],
    *,
    history_size: int,
    group_columns: tuple[str, ...] = ("log_id", "segment_id"),
    sort_column: str = "time_s",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    if history_size < 1:
        raise ValueError("history_size must be at least 1")

    derived = _with_derived_columns(frame)
    required_columns = list(sequence_feature_columns) + list(current_feature_columns) + list(target_columns)
    missing_columns = [column for column in required_columns if column not in derived.columns]
    if missing_columns:
        raise ValueError(f"Missing required sequence training columns: {missing_columns}")

    available_group_columns = [column for column in group_columns if column in derived.columns]
    metadata_columns = [column for column in [*available_group_columns, sort_column] if column in derived.columns]
    if available_group_columns:
        groups = derived.groupby(available_group_columns, sort=False, dropna=False).indices.values()
    else:
        groups = [np.arange(len(derived))]

    sequence_parts: list[np.ndarray] = []
    current_parts: list[np.ndarray] = []
    target_parts: list[pd.DataFrame] = []
    metadata_parts: list[pd.DataFrame] = []

    for indices in groups:
        group = derived.iloc[np.asarray(indices)].copy()
        if sort_column in group.columns:
            group = group.sort_values(sort_column, kind="mergesort")
        group = group.reset_index(drop=True)
        if len(group) < history_size:
            continue

        group_sequence = group.loc[:, sequence_feature_columns].to_numpy(dtype=np.float32, copy=True)
        group_current = group.loc[:, current_feature_columns].to_numpy(dtype=np.float32, copy=True)
        group_targets = group.loc[:, target_columns].copy()
        group_metadata = group.loc[:, metadata_columns].copy()
        end_indices = np.arange(history_size - 1, len(group), dtype=np.int64)

        group_windows = np.lib.stride_tricks.sliding_window_view(group_sequence, history_size, axis=0)
        sequence_parts.append(np.swapaxes(group_windows, 1, 2))
        if current_feature_columns:
            current_parts.append(group_current[end_indices])
        target_parts.append(group_targets.iloc[end_indices].reset_index(drop=True))
        metadata_parts.append(group_metadata.iloc[end_indices].reset_index(drop=True))

    if not sequence_parts:
        raise ValueError("No complete causal sequence samples were produced")

    sequence_features = np.concatenate(sequence_parts, axis=0).astype(np.float32, copy=False)
    if current_feature_columns:
        current_features = np.concatenate(current_parts, axis=0).astype(np.float32, copy=False)
    else:
        current_features = np.empty((len(sequence_features), 0), dtype=np.float32)
    targets = pd.concat(target_parts, ignore_index=True)
    metadata = pd.concat(metadata_parts, ignore_index=True) if metadata_parts else pd.DataFrame(index=range(len(sequence_features)))
    return sequence_features, current_features, targets, metadata


def prepare_causal_rollout_feature_target_frames(
    frame: pd.DataFrame,
    context_feature_columns: list[str],
    rollout_feature_columns: list[str],
    current_feature_columns: list[str],
    target_columns: list[str],
    *,
    history_size: int,
    rollout_size: int,
    rollout_stride: int | None = None,
    group_columns: tuple[str, ...] = ("log_id", "segment_id"),
    sort_column: str = "time_s",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    if history_size < 1:
        raise ValueError("history_size must be at least 1")
    if rollout_size < 1:
        raise ValueError("rollout_size must be at least 1")
    resolved_stride = rollout_size if rollout_stride is None else int(rollout_stride)
    if resolved_stride < 1:
        raise ValueError("rollout_stride must be at least 1")

    derived = _with_derived_columns(frame)
    required_columns = (
        list(context_feature_columns)
        + list(rollout_feature_columns)
        + list(current_feature_columns)
        + list(target_columns)
    )
    missing_columns = [column for column in required_columns if column not in derived.columns]
    if missing_columns:
        raise ValueError(f"Missing required rollout training columns: {missing_columns}")

    available_group_columns = [column for column in group_columns if column in derived.columns]
    metadata_columns = [column for column in [*available_group_columns, sort_column] if column in derived.columns]
    if available_group_columns:
        groups = derived.groupby(available_group_columns, sort=False, dropna=False).indices.values()
    else:
        groups = [np.arange(len(derived))]

    context_parts: list[np.ndarray] = []
    rollout_parts: list[np.ndarray] = []
    current_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    metadata_parts: list[pd.DataFrame] = []

    for indices in groups:
        group = derived.iloc[np.asarray(indices)].copy()
        if sort_column in group.columns:
            group = group.sort_values(sort_column, kind="mergesort")
        group = group.reset_index(drop=True)
        if len(group) < history_size + rollout_size:
            continue

        starts = np.arange(history_size, len(group) - rollout_size + 1, resolved_stride, dtype=np.int64)
        if len(starts) == 0:
            continue
        context_indices = starts[:, None] - history_size + np.arange(history_size, dtype=np.int64)[None, :]
        rollout_indices = starts[:, None] + np.arange(rollout_size, dtype=np.int64)[None, :]

        group_context = group.loc[:, context_feature_columns].to_numpy(dtype=np.float32, copy=True)
        group_rollout = group.loc[:, rollout_feature_columns].to_numpy(dtype=np.float32, copy=True)
        group_targets = group.loc[:, target_columns].to_numpy(dtype=np.float32, copy=True)

        context_parts.append(group_context[context_indices])
        rollout_parts.append(group_rollout[rollout_indices])
        target_parts.append(group_targets[rollout_indices])
        if current_feature_columns:
            group_current = group.loc[:, current_feature_columns].to_numpy(dtype=np.float32, copy=True)
            current_parts.append(group_current[rollout_indices])
        else:
            current_parts.append(np.empty((len(starts), rollout_size, 0), dtype=np.float32))
        if metadata_columns:
            metadata_parts.append(group.loc[:, metadata_columns].iloc[rollout_indices.reshape(-1)].reset_index(drop=True))

    if not context_parts:
        raise ValueError("No complete causal rollout subsections were produced")

    context_features = np.concatenate(context_parts, axis=0).astype(np.float32, copy=False)
    rollout_features = np.concatenate(rollout_parts, axis=0).astype(np.float32, copy=False)
    current_features = np.concatenate(current_parts, axis=0).astype(np.float32, copy=False)
    target_sequences = np.concatenate(target_parts, axis=0).astype(np.float32, copy=False)
    metadata = (
        pd.concat(metadata_parts, ignore_index=True)
        if metadata_parts
        else pd.DataFrame(index=range(context_features.shape[0] * rollout_size))
    )
    return context_features, rollout_features, current_features, target_sequences, metadata
