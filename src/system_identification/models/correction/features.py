"""Deterministic feature construction for static mean and wingbeat branches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from system_identification.models.correction.specifications import MEAN_WB_TYPES, StaticCorrectionSpec


@dataclass(frozen=True)
class DesignMatrix:
    values: np.ndarray
    feature_names: tuple[str, ...]


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")


def _checked_matrix(values: np.ndarray, names: tuple[str, ...]) -> DesignMatrix:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("Feature matrix shape does not match feature names")
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains non-finite values")
    return DesignMatrix(values=matrix, feature_names=names)


def _condition_columns(condition_set: str) -> tuple[str, ...]:
    if condition_set == "none":
        return ()
    if condition_set == "alpha":
        return ("alpha_mean_std",)
    if condition_set == "frequency":
        return ("flapping_frequency_mean_std",)
    return ("alpha_mean_std", "flapping_frequency_mean_std")


def build_mean_design(frame: pd.DataFrame, spec: StaticCorrectionSpec) -> DesignMatrix:
    if spec.model_type not in MEAN_WB_TYPES:
        raise ValueError(f"Mean design is not defined for {spec.model_type}")
    condition_columns = _condition_columns(str(spec.mean_condition_set))
    _require_columns(frame, list(condition_columns))
    columns: list[np.ndarray] = []
    names: list[str] = []
    if spec.fit_intercept:
        columns.append(np.ones(len(frame), dtype=np.float64))
        names.append("intercept")
    for column in condition_columns:
        columns.append(frame[column].to_numpy(dtype=np.float64, copy=False))
        names.append(column)
    values = np.column_stack(columns) if columns else np.empty((len(frame), 0), dtype=np.float64)
    return _checked_matrix(values, tuple(names))


def build_waveform_design(frame: pd.DataFrame, spec: StaticCorrectionSpec) -> DesignMatrix:
    if spec.model_type not in MEAN_WB_TYPES:
        raise ValueError(f"Waveform design is not defined for {spec.model_type}")
    if spec.harmonic_order is None:
        raise ValueError("harmonic_order is required")
    condition_columns = _condition_columns(str(spec.waveform_condition_set))
    required = list(condition_columns)
    for harmonic in range(1, spec.harmonic_order + 1):
        required.extend([f"sin_{harmonic}_phase_centered", f"cos_{harmonic}_phase_centered"])
    _require_columns(frame, required)

    values: list[np.ndarray] = []
    names: list[str] = []
    alpha = frame["alpha_mean_std"].to_numpy(dtype=np.float64, copy=False) if "alpha_mean_std" in condition_columns else None
    frequency = (
        frame["flapping_frequency_mean_std"].to_numpy(dtype=np.float64, copy=False)
        if "flapping_frequency_mean_std" in condition_columns
        else None
    )
    for harmonic in range(1, spec.harmonic_order + 1):
        sin_name = f"sin_{harmonic}_phase_centered"
        cos_name = f"cos_{harmonic}_phase_centered"
        sin_value = frame[sin_name].to_numpy(dtype=np.float64, copy=False)
        cos_value = frame[cos_name].to_numpy(dtype=np.float64, copy=False)
        values.extend([sin_value, cos_value])
        names.extend([sin_name, cos_name])
        if alpha is not None:
            values.extend([alpha * sin_value, alpha * cos_value])
            names.extend([f"alpha_x_sin_{harmonic}", f"alpha_x_cos_{harmonic}"])
        if frequency is not None:
            values.extend([frequency * sin_value, frequency * cos_value])
            names.extend([f"frequency_x_sin_{harmonic}", f"frequency_x_cos_{harmonic}"])
    return _checked_matrix(np.column_stack(values), tuple(names))
