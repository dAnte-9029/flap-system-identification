"""Frozen inference and existing per-log, regime, and residual diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from system_identification.evaluation.metrics import _metrics_from_arrays, _validate_bin_edges
from system_identification.evaluation.prediction import (
    _as_numpy_array,
    _predict_rollout_scaled_batches,
    _predict_sequence_scaled_batches,
    _resolve_device,
)
from system_identification.evaluation.reports import _metrics_table_row
from system_identification.models.bundles import (
    _build_model_from_bundle,
    _build_rollout_model_from_bundle,
    _build_sequence_model_from_bundle,
    _is_rollout_model_type,
    _is_sequence_model_type,
)
from system_identification.models.features import NO_ACCEL_NO_ALPHA_EXCLUDED_COLUMNS, _with_derived_columns
from system_identification.training.data_preparation import _load_split_frame
from system_identification.training.loaders import _make_loader
from system_identification.training.normalization import (
    _inverse_transform_targets,
    _transform_features,
    _transform_rollout_features,
    _transform_sequence_features,
)
from system_identification.training.windows import (
    prepare_causal_rollout_feature_target_frames,
    prepare_causal_sequence_feature_target_frames,
    prepare_windowed_feature_target_frames,
)

DEFAULT_REGIME_BIN_SPECS: dict[str, list[float]] = {
    "airspeed_validated.true_airspeed_m_s": [0.0, 6.0, 8.0, 10.0, 12.0, 16.0],
    "cycle_flap_frequency_hz": [0.0, 3.0, 4.0, 5.0, 6.0, 8.0],
    "phase_corrected_rad": [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi, 2.0 * math.pi],
}


ACCELERATION_INPUT_COLUMNS = set(NO_ACCEL_NO_ALPHA_EXCLUDED_COLUMNS)


VELOCITY_HISTORY_COLUMNS = {
    "vehicle_local_position.vx",
    "vehicle_local_position.vy",
    "vehicle_local_position.vz",
    "velocity_b.x",
    "velocity_b.y",
    "velocity_b.z",
    "relative_air_velocity_b.x",
    "relative_air_velocity_b.y",
    "relative_air_velocity_b.z",
}


ANGULAR_VELOCITY_HISTORY_COLUMNS = {
    "vehicle_angular_velocity.xyz[0]",
    "vehicle_angular_velocity.xyz[1]",
    "vehicle_angular_velocity.xyz[2]",
}


ALPHA_BETA_HISTORY_COLUMNS = {"alpha_rad", "beta_rad"}


def _training_audit_flags(bundle: dict[str, Any], *, split_root: str | Path | None = None) -> dict[str, bool]:
    model_type = bundle.get("model_type", "mlp")
    if _is_sequence_model_type(model_type):
        history_columns = set(bundle.get("sequence_feature_columns", []))
        input_columns = history_columns | set(bundle.get("current_feature_columns", []))
        has_centered_window = False
    elif _is_rollout_model_type(model_type):
        history_columns = set(bundle.get("context_feature_columns", []))
        input_columns = (
            history_columns
            | set(bundle.get("rollout_feature_columns", []))
            | set(bundle.get("current_feature_columns", []))
        )
        has_centered_window = False
    else:
        history_columns = set(bundle.get("window_feature_columns", []))
        input_columns = set(bundle.get("base_feature_columns", bundle.get("feature_columns", [])))
        has_centered_window = bundle.get("window_mode") == "centered"

    uses_whole_log_split = False
    if split_root is not None:
        manifest_path = Path(split_root) / "dataset_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                uses_whole_log_split = manifest.get("split_policy") == "whole_log"
            except json.JSONDecodeError:
                uses_whole_log_split = False

    return {
        "has_acceleration_inputs": bool(input_columns & ACCELERATION_INPUT_COLUMNS),
        "has_velocity_history": bool(history_columns & VELOCITY_HISTORY_COLUMNS),
        "has_angular_velocity_history": bool(history_columns & ANGULAR_VELOCITY_HISTORY_COLUMNS),
        "has_alpha_beta_history": bool(history_columns & ALPHA_BETA_HISTORY_COLUMNS),
        "has_centered_window": bool(has_centered_window),
        "uses_whole_log_split": bool(uses_whole_log_split),
    }


def _sequence_arrays_for_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sequence_features, current_features, targets_df, _ = _sequence_arrays_with_metadata_for_bundle(bundle, frame)
    return sequence_features, current_features, targets_df


def _sequence_arrays_with_metadata_for_bundle(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    sequence_features, current_features, targets_df, metadata = prepare_causal_sequence_feature_target_frames(
        frame,
        list(bundle["sequence_feature_columns"]),
        list(bundle["current_feature_columns"]),
        list(bundle["target_columns"]),
        history_size=int(bundle["sequence_history_size"]),
        group_columns=tuple(bundle.get("sequence_group_columns", ("log_id", "segment_id"))),
        sort_column=str(bundle.get("sequence_sort_column", "time_s")),
    )
    return sequence_features, current_features, targets_df, metadata


def _rollout_arrays_for_bundle(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    return prepare_causal_rollout_feature_target_frames(
        frame,
        list(bundle["context_feature_columns"]),
        list(bundle["rollout_feature_columns"]),
        list(bundle["current_feature_columns"]),
        list(bundle["target_columns"]),
        history_size=int(bundle["sequence_history_size"]),
        rollout_size=int(bundle["rollout_size"]),
        rollout_stride=int(bundle.get("rollout_stride", bundle["rollout_size"])),
        group_columns=tuple(bundle.get("sequence_group_columns", ("log_id", "segment_id"))),
        sort_column=str(bundle.get("sequence_sort_column", "time_s")),
    )


def _targets_for_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    if _is_sequence_model_type(bundle.get("model_type", "mlp")):
        _, _, targets_df = _sequence_arrays_for_bundle(bundle, frame)
        return targets_df
    if _is_rollout_model_type(bundle.get("model_type", "mlp")):
        _, _, _, targets, _ = _rollout_arrays_for_bundle(bundle, frame)
        return pd.DataFrame(targets.reshape(-1, targets.shape[-1]), columns=bundle["target_columns"])
    _, targets_df = prepare_windowed_feature_target_frames(
        frame,
        bundle.get("base_feature_columns", bundle["feature_columns"]),
        bundle["target_columns"],
        window_mode=bundle.get("window_mode", "single"),
        window_radius=int(bundle.get("window_radius", 0)),
        window_feature_columns=bundle.get("window_feature_columns"),
    )
    return targets_df


def prediction_metadata_frame_for_bundle(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    split_name: str,
    batch_size: int = 8192,
    device: str | None = None,
) -> pd.DataFrame:
    """Align model predictions, true targets, and row metadata for diagnostics."""
    predictions_df = predict_model_bundle(bundle, frame, batch_size=batch_size, device=device)
    targets_df = _targets_for_bundle(bundle, frame)
    derived = _with_derived_columns(frame)

    if _is_sequence_model_type(bundle.get("model_type", "mlp")):
        _, _, _, metadata = _sequence_arrays_with_metadata_for_bundle(bundle, frame)
        join_keys = [column for column in ["log_id", "segment_id", "time_s"] if column in metadata.columns and column in derived.columns]
        if join_keys:
            extra_columns = [column for column in derived.columns if column not in metadata.columns]
            metadata = metadata.merge(
                derived.loc[:, [*join_keys, *extra_columns]].drop_duplicates(join_keys),
                on=join_keys,
                how="left",
            )
    elif _is_rollout_model_type(bundle.get("model_type", "mlp")):
        raise NotImplementedError("prediction metadata alignment is not implemented for rollout model bundles")
    else:
        features_df, _ = prepare_windowed_feature_target_frames(
            derived,
            bundle.get("base_feature_columns", bundle["feature_columns"]),
            bundle["target_columns"],
            window_mode=bundle.get("window_mode", "single"),
            window_radius=int(bundle.get("window_radius", 0)),
            window_feature_columns=bundle.get("window_feature_columns"),
        )
        metadata = derived.loc[features_df.index].reset_index(drop=True)

    aligned = metadata.reset_index(drop=True).copy()
    for target in bundle["target_columns"]:
        true_values = targets_df[target].to_numpy()
        pred_values = predictions_df[target].to_numpy()
        aligned[f"true_{target}"] = true_values
        aligned[f"pred_{target}"] = pred_values
        aligned[f"resid_{target}"] = true_values - pred_values
    aligned["split"] = split_name
    return aligned


def predict_model_bundle(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    batch_size: int = 8192,
    device: str | None = None,
) -> pd.DataFrame:
    resolved_device = _resolve_device(device or bundle.get("device_type"))
    if _is_sequence_model_type(bundle.get("model_type", "mlp")):
        sequence_features, current_features, _ = _sequence_arrays_for_bundle(bundle, frame)
        sequence_scaled = _transform_sequence_features(
            sequence_features,
            _as_numpy_array(bundle["sequence_feature_medians"]),
            _as_numpy_array(bundle["sequence_feature_means"]),
            _as_numpy_array(bundle["sequence_feature_stds"]),
        )
        if current_features.shape[1] > 0:
            current_scaled = _transform_features(
                current_features,
                _as_numpy_array(bundle["current_feature_medians"]),
                _as_numpy_array(bundle["current_feature_means"]),
                _as_numpy_array(bundle["current_feature_stds"]),
            )
        else:
            current_scaled = current_features.astype(np.float32, copy=False)
        model = _build_sequence_model_from_bundle(bundle, resolved_device)
        predictions_scaled_arr = _predict_sequence_scaled_batches(
            model,
            sequence_scaled,
            current_scaled,
            batch_size=batch_size,
            device=resolved_device,
            use_amp=bool(bundle.get("use_amp", False)),
        )
        predictions = _inverse_transform_targets(
            predictions_scaled_arr,
            _as_numpy_array(bundle["target_means"]),
            _as_numpy_array(bundle["target_stds"]),
        )
        return pd.DataFrame(predictions, columns=bundle["target_columns"])

    if _is_rollout_model_type(bundle.get("model_type", "mlp")):
        context_features, rollout_features, current_features, _, _ = _rollout_arrays_for_bundle(bundle, frame)
        context_scaled = _transform_rollout_features(
            context_features,
            _as_numpy_array(bundle["context_feature_medians"]),
            _as_numpy_array(bundle["context_feature_means"]),
            _as_numpy_array(bundle["context_feature_stds"]),
        )
        rollout_scaled = _transform_rollout_features(
            rollout_features,
            _as_numpy_array(bundle["rollout_feature_medians"]),
            _as_numpy_array(bundle["rollout_feature_means"]),
            _as_numpy_array(bundle["rollout_feature_stds"]),
        )
        if current_features.shape[2] > 0:
            current_scaled = _transform_rollout_features(
                current_features,
                _as_numpy_array(bundle["current_feature_medians"]),
                _as_numpy_array(bundle["current_feature_means"]),
                _as_numpy_array(bundle["current_feature_stds"]),
            )
        else:
            current_scaled = current_features.astype(np.float32, copy=False)
        model = _build_rollout_model_from_bundle(bundle, resolved_device)
        predictions_scaled_arr = _predict_rollout_scaled_batches(
            model,
            context_scaled,
            rollout_scaled,
            current_scaled,
            batch_size=batch_size,
            device=resolved_device,
            use_amp=bool(bundle.get("use_amp", False)),
        )
        predictions = _inverse_transform_targets(
            predictions_scaled_arr.reshape(-1, predictions_scaled_arr.shape[-1]),
            _as_numpy_array(bundle["target_means"]),
            _as_numpy_array(bundle["target_stds"]),
        )
        return pd.DataFrame(predictions, columns=bundle["target_columns"])

    features_df, _ = prepare_windowed_feature_target_frames(
        frame,
        bundle.get("base_feature_columns", bundle["feature_columns"]),
        bundle["target_columns"],
        window_mode=bundle.get("window_mode", "single"),
        window_radius=int(bundle.get("window_radius", 0)),
        window_feature_columns=bundle.get("window_feature_columns"),
    )
    features = features_df.to_numpy(dtype=np.float32, copy=True)
    features_scaled = _transform_features(
        features,
        _as_numpy_array(bundle["feature_medians"]),
        _as_numpy_array(bundle["feature_means"]),
        _as_numpy_array(bundle["feature_stds"]),
    )

    loader = _make_loader(
        features_scaled,
        None,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=resolved_device.type == "cuda",
    )

    model = _build_model_from_bundle(bundle, resolved_device)
    predictions_scaled: list[np.ndarray] = []
    amp_enabled = bool(bundle.get("use_amp", False)) and resolved_device.type == "cuda"

    with torch.no_grad():
        for (batch_features,) in loader:
            batch_features = batch_features.to(resolved_device, non_blocking=True)
            with torch.autocast(device_type=resolved_device.type, dtype=torch.float16, enabled=amp_enabled):
                batch_predictions = model(batch_features)
            predictions_scaled.append(batch_predictions.cpu().numpy())

    predictions_scaled_arr = np.concatenate(predictions_scaled, axis=0)
    predictions = _inverse_transform_targets(
        predictions_scaled_arr,
        _as_numpy_array(bundle["target_means"]),
        _as_numpy_array(bundle["target_stds"]),
    )
    return pd.DataFrame(predictions, columns=bundle["target_columns"])


def evaluate_model_bundle(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    split_name: str,
    batch_size: int = 8192,
    device: str | None = None,
) -> dict[str, Any]:
    targets_df = _targets_for_bundle(bundle, frame)
    predictions_df = predict_model_bundle(bundle, frame, batch_size=batch_size, device=device)

    y_true = targets_df.to_numpy(dtype=np.float64, copy=False)
    y_pred = predictions_df.to_numpy(dtype=np.float64, copy=False)
    return _metrics_from_arrays(y_true, y_pred, target_columns=bundle["target_columns"], split_name=split_name)


def evaluate_model_bundle_by_log(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    split_name: str,
    log_column: str = "log_id",
    min_samples: int = 1,
    batch_size: int = 8192,
    device: str | None = None,
) -> pd.DataFrame:
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")

    if log_column not in frame.columns:
        metrics = evaluate_model_bundle(bundle, frame, split_name=split_name, batch_size=batch_size, device=device)
        row = _metrics_table_row(
            metrics,
            split_name=split_name,
            diagnostic_type="per_log",
            group_column=log_column,
            group_value="__missing_log_id__",
        )
        row["log_id"] = "__missing_log_id__"
        return pd.DataFrame([row])

    rows: list[dict[str, Any]] = []
    for log_id, group in frame.groupby(log_column, sort=True):
        if len(group) < min_samples:
            continue
        metrics = evaluate_model_bundle(bundle, group.copy(), split_name=split_name, batch_size=batch_size, device=device)
        row = _metrics_table_row(
            metrics,
            split_name=split_name,
            diagnostic_type="per_log",
            group_column=log_column,
            group_value=str(log_id),
        )
        row["log_id"] = str(log_id)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_model_bundle_by_regime_bins(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    split_name: str,
    bin_specs: dict[str, list[float]] | None = None,
    min_samples: int = 1,
    batch_size: int = 8192,
    device: str | None = None,
) -> pd.DataFrame:
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")

    resolved_bin_specs = bin_specs or DEFAULT_REGIME_BIN_SPECS
    derived = _with_derived_columns(frame)
    rows: list[dict[str, Any]] = []

    if _is_sequence_model_type(bundle.get("model_type", "mlp")):
        _, _, targets_df, metadata = _sequence_arrays_with_metadata_for_bundle(bundle, frame)
        predictions_df = predict_model_bundle(bundle, frame, batch_size=batch_size, device=device)
        join_keys = [column for column in ["log_id", "segment_id", "time_s"] if column in metadata.columns and column in derived.columns]
        if join_keys:
            regime_columns = [column for column in resolved_bin_specs if column in derived.columns and column not in join_keys]
            meta_with_regimes = metadata.merge(
                derived.loc[:, [*join_keys, *regime_columns]].drop_duplicates(join_keys),
                on=join_keys,
                how="left",
            )
        else:
            meta_with_regimes = metadata.copy()
            for column in resolved_bin_specs:
                if column in derived.columns and len(derived) == len(meta_with_regimes):
                    meta_with_regimes[column] = derived[column].to_numpy()

        for column, raw_edges in resolved_bin_specs.items():
            if column not in meta_with_regimes.columns:
                continue
            edges = _validate_bin_edges(column, raw_edges)
            values = pd.to_numeric(meta_with_regimes[column], errors="coerce")
            binned = pd.cut(values, bins=edges, include_lowest=True)
            for interval in binned.cat.categories:
                mask = (binned == interval).to_numpy()
                if int(mask.sum()) < min_samples:
                    continue
                metrics = _metrics_from_arrays(
                    targets_df.to_numpy(dtype=np.float64, copy=False)[mask],
                    predictions_df.to_numpy(dtype=np.float64, copy=False)[mask],
                    target_columns=bundle["target_columns"],
                    split_name=split_name,
                )
                row = _metrics_table_row(
                    metrics,
                    split_name=split_name,
                    diagnostic_type="regime_bin",
                    group_column=column,
                    group_value=str(interval),
                )
                row["regime_column"] = column
                row["bin_label"] = str(interval)
                row["bin_left"] = float(interval.left)
                row["bin_right"] = float(interval.right)
                rows.append(row)
        return pd.DataFrame(rows)

    if _is_rollout_model_type(bundle.get("model_type", "mlp")):
        _, _, _, target_sequences, metadata = _rollout_arrays_for_bundle(bundle, frame)
        targets_df = pd.DataFrame(
            target_sequences.reshape(-1, target_sequences.shape[-1]),
            columns=bundle["target_columns"],
        )
        predictions_df = predict_model_bundle(bundle, frame, batch_size=batch_size, device=device)
        join_keys = [column for column in ["log_id", "segment_id", "time_s"] if column in metadata.columns and column in derived.columns]
        if join_keys:
            regime_columns = [column for column in resolved_bin_specs if column in derived.columns and column not in join_keys]
            meta_with_regimes = metadata.merge(
                derived.loc[:, [*join_keys, *regime_columns]].drop_duplicates(join_keys),
                on=join_keys,
                how="left",
            )
        else:
            meta_with_regimes = metadata.copy()
            for column in resolved_bin_specs:
                if column in derived.columns and len(derived) == len(meta_with_regimes):
                    meta_with_regimes[column] = derived[column].to_numpy()

        for column, raw_edges in resolved_bin_specs.items():
            if column not in meta_with_regimes.columns:
                continue
            edges = _validate_bin_edges(column, raw_edges)
            values = pd.to_numeric(meta_with_regimes[column], errors="coerce")
            binned = pd.cut(values, bins=edges, include_lowest=True)
            for interval in binned.cat.categories:
                mask = (binned == interval).to_numpy()
                if int(mask.sum()) < min_samples:
                    continue
                metrics = _metrics_from_arrays(
                    targets_df.to_numpy(dtype=np.float64, copy=False)[mask],
                    predictions_df.to_numpy(dtype=np.float64, copy=False)[mask],
                    target_columns=bundle["target_columns"],
                    split_name=split_name,
                )
                row = _metrics_table_row(
                    metrics,
                    split_name=split_name,
                    diagnostic_type="regime_bin",
                    group_column=column,
                    group_value=str(interval),
                )
                row["regime_column"] = column
                row["bin_label"] = str(interval)
                row["bin_left"] = float(interval.left)
                row["bin_right"] = float(interval.right)
                rows.append(row)
        return pd.DataFrame(rows)

    for column, raw_edges in resolved_bin_specs.items():
        if column not in derived.columns:
            continue
        edges = _validate_bin_edges(column, raw_edges)
        values = pd.to_numeric(derived[column], errors="coerce")
        binned = pd.cut(values, bins=edges, include_lowest=True)
        for interval in binned.cat.categories:
            mask = binned == interval
            if int(mask.sum()) < min_samples:
                continue
            group = frame.loc[mask.to_numpy()].copy()
            try:
                metrics = evaluate_model_bundle(bundle, group, split_name=split_name, batch_size=batch_size, device=device)
            except ValueError as exc:
                if "No complete causal sequence samples were produced" in str(exc):
                    continue
                raise
            row = _metrics_table_row(
                metrics,
                split_name=split_name,
                diagnostic_type="regime_bin",
                group_column=column,
                group_value=str(interval),
            )
            row["regime_column"] = column
            row["bin_label"] = str(interval)
            row["bin_left"] = float(interval.left)
            row["bin_right"] = float(interval.right)
            rows.append(row)

    return pd.DataFrame(rows)


def run_diagnostic_evaluation(
    *,
    model_bundle_path: str | Path,
    split_root: str | Path,
    output_dir: str | Path,
    split_names: tuple[str, ...] = ("test",),
    bin_specs: dict[str, list[float]] | None = None,
    min_samples: int = 16,
    batch_size: int = 8192,
    device: str | None = None,
) -> dict[str, str]:
    if not split_names:
        raise ValueError("split_names must not be empty")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(model_bundle_path, map_location="cpu", weights_only=False)

    per_log_parts: list[pd.DataFrame] = []
    per_regime_parts: list[pd.DataFrame] = []
    for split_name in split_names:
        frame = _load_split_frame(split_root, split_name, None, 0)
        per_log_parts.append(
            evaluate_model_bundle_by_log(
                bundle,
                frame,
                split_name=split_name,
                min_samples=min_samples,
                batch_size=batch_size,
                device=device,
            )
        )
        per_regime_parts.append(
            evaluate_model_bundle_by_regime_bins(
                bundle,
                frame,
                split_name=split_name,
                bin_specs=bin_specs,
                min_samples=min_samples,
                batch_size=batch_size,
                device=device,
            )
        )

    per_log = pd.concat(per_log_parts, ignore_index=True) if per_log_parts else pd.DataFrame()
    per_regime = pd.concat(per_regime_parts, ignore_index=True) if per_regime_parts else pd.DataFrame()

    per_log_metrics_path = output_path / "per_log_metrics.csv"
    per_regime_metrics_path = output_path / "per_regime_metrics.csv"
    diagnostics_config_path = output_path / "diagnostics_config.json"

    per_log.to_csv(per_log_metrics_path, index=False)
    per_regime.to_csv(per_regime_metrics_path, index=False)
    diagnostics_config_path.write_text(
        json.dumps(
            {
                "model_bundle_path": str(model_bundle_path),
                "split_root": str(split_root),
                "split_names": list(split_names),
                "bin_specs": bin_specs or DEFAULT_REGIME_BIN_SPECS,
                "min_samples": int(min_samples),
                "batch_size": int(batch_size),
                "device": device or "auto",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "per_log_metrics_path": str(per_log_metrics_path),
        "per_regime_metrics_path": str(per_regime_metrics_path),
        "diagnostics_config_path": str(diagnostics_config_path),
    }
