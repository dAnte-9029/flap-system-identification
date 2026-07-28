"""Fail-closed Quadratic-2 dynamic-twist sensitivity experiment.

The experiment is train/validation-only. It reuses the frozen wing-only
DeLaurier evaluator, canonical alignment keys, and correction-ready accepted
cycle set. Test paths are never constructed by this module.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-quadratic2-twist-sweep")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import yaml

from system_identification.artifacts.io import sha256_file
from system_identification.artifacts.prior_registry import resolve_delaurier_prior
from system_identification.physics.baselines.wing_only import (
    ATTITUDE_AIRFLOW_REQUIRED_COLUMNS,
    ATTITUDE_KINEMATICS_REQUIRED_COLUMNS,
    WingOnlyBaselineConfig,
    baseline_config_from_aircraft_metadata,
    evaluate_wing_only_delaurier_segment,
    _resolve_airflow_inputs,
    _wing_polar_transforms_frd,
)
from system_identification.physics.delaurier.dynamic_twist import (
    compute_delaurier_dynamic_twist,
    map_canonical_phase_to_delaurier,
)
from system_identification.physics.delaurier.strip_wrench import (
    compute_delaurier_strip_loads,
    integrate_delaurier_strip_wrench,
    load_wing_geometry_csv,
)


SCHEMA_VERSION = "quadratic2_twist_sweep_v1"
PARTITION_FILES = {"train": "train_samples.parquet", "validation": "val_samples.parquet"}
PREDICTION_FILES = {
    "train": "train_predictions.parquet",
    "validation": "val_predictions.parquet",
}
FORCES = ("fx", "fz")
FRAME_CONTRACT = "body_FRD: +Fx forward, +Fz down"


@dataclass(frozen=True)
class TwistCandidate:
    profile_name: str
    A_tip_deg: float
    kappa: float
    psi_theta_deg: float
    static_twist_offset_deg: float = 0.0
    stage: str = "unspecified"
    family: str = "unspecified"

    @property
    def parameter_hash(self) -> str:
        payload = {
            "profile_name": self.profile_name,
            "A_tip_deg": round(float(self.A_tip_deg), 12),
            "kappa": round(float(self.kappa), 12),
            "psi_theta_deg": round(float(self.psi_theta_deg), 12),
            "static_twist_offset_deg": round(float(self.static_twist_offset_deg), 12),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]


@dataclass(frozen=True)
class ResolvedExperiment:
    project_root: Path
    config_path: Path
    config: Mapping[str, object]
    dataset_root: Path
    dataset_manifest_path: Path
    dataset_manifest: Mapping[str, object]
    prior_root: Path
    prior_manifest_path: Path
    prior_manifest: Mapping[str, object]
    correction_ready_root: Path
    correction_ready_manifest_path: Path
    correction_ready_manifest: Mapping[str, object]
    metadata_path: Path
    geometry_path: Path
    output_root: Path


@dataclass(frozen=True)
class CandidateEvaluation:
    metrics: pd.DataFrame
    curves: pd.DataFrame
    diagnostics: Mapping[str, object]


def _read_mapping(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def _absolute(project_root: Path, value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _git(project_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=project_root, text=True).strip()


def _validate_grid(values: Iterable[object], low: float, high: float, *, name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result or not np.isfinite(result).all():
        raise ValueError(f"{name} grid must be non-empty and finite")
    if min(result) < low or max(result) > high:
        raise ValueError(f"{name} grid exceeds configured allowed range [{low}, {high}]")
    return result


def resolve_experiment(config_path: str | Path, *, project_root: str | Path) -> ResolvedExperiment:
    root = Path(project_root).resolve()
    source = _absolute(root, config_path)
    config = _read_mapping(source)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported experiment schema: {config.get('schema_version')!r}")
    data = config.get("data")
    physics = config.get("physics")
    registries = config.get("registries")
    output = config.get("output")
    if not all(isinstance(value, Mapping) for value in (data, physics, registries, output)):
        raise ValueError("Experiment config is missing data/physics/registries/output mappings")
    assert isinstance(data, Mapping)
    assert isinstance(physics, Mapping)
    assert isinstance(registries, Mapping)
    assert isinstance(output, Mapping)
    partitions = tuple(str(value) for value in data.get("partitions", []))
    if partitions != ("train", "validation"):
        raise ValueError("Experiment partitions must be exactly [train, validation]")
    if "test" not in {str(value) for value in data.get("sealed_partitions", [])}:
        raise ValueError("Experiment config must explicitly seal test")
    if any("test" in str(value).lower() for key, value in data.items() if key != "sealed_partitions"):
        raise ValueError("Experiment data config may not contain a test path or test selection")

    dataset_registry_path = _absolute(root, registries["canonical_dataset"])
    dataset_registry = _read_mapping(dataset_registry_path)
    dataset_id = str(data["dataset_id"])
    if dataset_registry.get("default_dataset_id") != dataset_id:
        raise ValueError("Configured dataset is not the canonical registry default")
    dataset_entry = dataset_registry.get("datasets", {}).get(dataset_id)
    if not isinstance(dataset_entry, Mapping) or dataset_entry.get("lifecycle_status") != "active":
        raise ValueError("Configured canonical dataset is missing or not active")
    dataset_root = _absolute(root, data["dataset_root"])
    registered_root = _absolute(root, dataset_entry["repository_relative_root"])
    if dataset_root != registered_root:
        raise ValueError("Configured dataset path disagrees with canonical registry")
    dataset_manifest_path = _absolute(root, dataset_entry["manifest_path"])
    dataset_manifest = _read_mapping(dataset_manifest_path)
    if sha256_file(dataset_manifest_path) != dataset_entry["manifest_sha256"]:
        raise ValueError("Canonical dataset manifest hash mismatch")
    for partition in partitions:
        path = dataset_root / PARTITION_FILES[partition]
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = dataset_entry["artifact_sha256"][path.name]
        if sha256_file(path) != expected:
            raise ValueError(f"Canonical {partition} sample hash mismatch")

    prior_registry_path = _absolute(root, registries["delaurier_prior"])
    prior = resolve_delaurier_prior(
        prior_id=str(data["prior_id"]),
        registry_path=prior_registry_path,
        requested_partitions=partitions,
    )
    if prior.prior_id != str(data["prior_id"]) or prior.lifecycle_status != "active":
        raise ValueError("Configured prior is not the active requested prior")
    prior_root = prior.artifact_root
    prior_manifest_path = prior_root / "manifest.json"
    prior_manifest = _read_mapping(prior_manifest_path)
    prior_entry = prior.registry_entry
    if sha256_file(prior_manifest_path) != prior_entry.get("artifact_manifest_sha256"):
        raise ValueError("Active prior manifest hash mismatch")
    for partition in partitions:
        prediction_path = prior_root / PREDICTION_FILES[partition]
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        if sha256_file(prediction_path) != prior_manifest["prediction_sha256"][partition]:
            raise ValueError(f"Active prior {partition} prediction hash mismatch")
    if bool(prior_manifest.get("test_partition_loaded", True)):
        raise ValueError("Active prior reports test access")

    correction_root = _absolute(root, data["correction_ready_root"])
    correction_manifest_path = correction_root / "manifest.json"
    correction_manifest = _read_mapping(correction_manifest_path)
    if bool(correction_manifest.get("test_labels_loaded", True)):
        raise ValueError("Correction-ready artifact reports test labels loaded")
    if set(correction_manifest.get("included_partitions", [])) != {"train", "validation"}:
        raise ValueError("Correction-ready artifact does not contain exactly train/validation")
    if correction_manifest.get("dataset_id") != dataset_id:
        raise ValueError("Correction-ready dataset identity mismatch")
    if correction_manifest.get("resolved_prior_id") != prior.prior_id:
        raise ValueError("Correction-ready prior identity mismatch")

    allowed = physics.get("allowed")
    sweep = config.get("sweep")
    if not isinstance(allowed, Mapping) or not isinstance(sweep, Mapping):
        raise ValueError("Config is missing physics.allowed or sweep")
    for stage in ("oat", "coarse"):
        grid = sweep.get(stage)
        if not isinstance(grid, Mapping):
            raise ValueError(f"Missing sweep.{stage}")
        for name in ("A_tip_deg", "kappa", "psi_theta_deg"):
            bounds = allowed[name]
            _validate_grid(grid[name], float(bounds[0]), float(bounds[1]), name=f"{stage}.{name}")
    static_bounds = allowed["static_twist_offset_deg"]
    _validate_grid(
        sweep["oat"]["static_twist_offset_deg"],
        float(static_bounds[0]),
        float(static_bounds[1]),
        name="oat.static_twist_offset_deg",
    )
    phase = config.get("phase_binning")
    if not isinstance(phase, Mapping) or int(phase.get("bins", 0)) <= 0:
        raise ValueError("phase_binning.bins must be positive")
    smoothing = phase.get("smoothing")
    if not isinstance(smoothing, Mapping) or smoothing.get("method") != "periodic_savgol_zero_phase":
        raise ValueError("Only periodic_savgol_zero_phase smoothing is authorized")
    window = int(smoothing["window_bins"])
    order = int(smoothing["polynomial_order"])
    if window <= order or window % 2 != 1 or window > int(phase["bins"]):
        raise ValueError("Invalid zero-phase Savitzky-Golay smoothing configuration")

    metadata = _absolute(root, physics["aircraft_metadata"])
    geometry = _absolute(root, physics["wing_geometry"])
    for path in (metadata, geometry):
        if not path.is_file():
            raise FileNotFoundError(path)
    return ResolvedExperiment(
        project_root=root,
        config_path=source,
        config=config,
        dataset_root=dataset_root,
        dataset_manifest_path=dataset_manifest_path,
        dataset_manifest=dataset_manifest,
        prior_root=prior_root,
        prior_manifest_path=prior_manifest_path,
        prior_manifest=prior_manifest,
        correction_ready_root=correction_root,
        correction_ready_manifest_path=correction_manifest_path,
        correction_ready_manifest=correction_manifest,
        metadata_path=metadata,
        geometry_path=geometry,
        output_root=_absolute(root, output["root"]),
    )


def _input_columns(airflow_mode: str) -> list[str]:
    airflow = (
        ATTITUDE_AIRFLOW_REQUIRED_COLUMNS
        if airflow_mode == "attitude_ground_wind_3d"
        else ATTITUDE_KINEMATICS_REQUIRED_COLUMNS
    )
    return list(
        dict.fromkeys(
            [
                "log_id",
                "segment_id",
                "time_s",
                "timestamp_us",
                "mechanical_phase_rad",
                "flap_frequency_hz",
                "vehicle_air_data.rho",
                "fx_b",
                "fz_b",
                *airflow,
            ]
        )
    )


def load_experiment_rows(
    resolved: ResolvedExperiment,
    *,
    airflow_mode: str,
) -> dict[str, pd.DataFrame]:
    waveform_path = resolved.correction_ready_root / "waveform_table.parquet"
    if not waveform_path.is_file():
        raise FileNotFoundError(waveform_path)
    keys = pd.read_parquet(
        waveform_path,
        columns=["partition", "log_id", "timestamp_us"],
    )
    if set(keys["partition"].astype(str).unique()) != {"train", "validation"}:
        raise ValueError("Accepted-cycle key table is not exactly train/validation")
    if keys.duplicated(["partition", "log_id", "timestamp_us"]).any():
        raise ValueError("Accepted-cycle key table has duplicate stable keys")
    result: dict[str, pd.DataFrame] = {}
    for partition in ("train", "validation"):
        source = pd.read_parquet(
            resolved.dataset_root / PARTITION_FILES[partition],
            columns=_input_columns(airflow_mode),
        )
        accepted = keys.loc[keys["partition"] == partition, ["log_id", "timestamp_us"]]
        frame = source.merge(
            accepted,
            on=["log_id", "timestamp_us"],
            how="inner",
            validate="one_to_one",
        )
        if len(frame) != len(accepted):
            raise ValueError(f"{partition}: accepted key coverage mismatch")
        if frame.duplicated(["log_id", "timestamp_us"]).any():
            raise ValueError(f"{partition}: duplicate aligned keys")
        numeric = [column for column in _input_columns(airflow_mode) if column not in {"log_id", "segment_id"}]
        if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
            raise ValueError(f"{partition}: non-finite experiment inputs")
        result[partition] = frame.sort_values(
            ["log_id", "segment_id", "time_s"], kind="stable"
        ).reset_index(drop=True)
    return result


def _baseline_config(
    resolved: ResolvedExperiment,
    candidate: TwistCandidate,
    *,
    airflow_mode: str,
) -> WingOnlyBaselineConfig:
    physics = resolved.config["physics"]
    assert isinstance(physics, Mapping)
    allowed = physics["allowed"]
    assert isinstance(allowed, Mapping)
    for name, value in (
        ("A_tip_deg", candidate.A_tip_deg),
        ("kappa", candidate.kappa),
        ("psi_theta_deg", candidate.psi_theta_deg),
        ("static_twist_offset_deg", candidate.static_twist_offset_deg),
    ):
        low, high = (float(item) for item in allowed[name])
        if not np.isfinite(value) or not low <= float(value) <= high:
            raise ValueError(f"{name}={value} is outside [{low}, {high}]")
    if candidate.profile_name not in {"legacy_linear", "quadratic2_phase"}:
        raise ValueError(f"Unsupported profile {candidate.profile_name!r}")
    if candidate.profile_name == "legacy_linear" and (
        candidate.kappa != 0.0 or candidate.psi_theta_deg != 0.0
    ):
        raise ValueError("legacy_linear requires kappa=0 and psi_theta_deg=0")
    runtime = resolved.config["runtime"]
    assert isinstance(runtime, Mapping)
    base = baseline_config_from_aircraft_metadata(
        resolved.metadata_path,
        chunk_size=int(runtime["chunk_size"]),
        airflow_mode=airflow_mode,
    )
    return replace(
        base,
        mean_pitch_offset_rad=base.mean_pitch_offset_rad
        + math.radians(float(candidate.static_twist_offset_deg)),
        twist_profile_name=candidate.profile_name,
        twist_kappa=float(candidate.kappa),
        twist_phase_offset_rad=math.radians(float(candidate.psi_theta_deg)),
    )


def _phase_macro_curves(frame: pd.DataFrame, *, phase_bins: int) -> pd.DataFrame:
    work = frame.copy()
    wrapped = np.mod(work["mechanical_phase_rad"].to_numpy(dtype=float), 2.0 * np.pi)
    work["phase_bin"] = np.minimum(
        (wrapped / (2.0 * np.pi) * phase_bins).astype(int),
        phase_bins - 1,
    )
    per_log = (
        work.groupby(["partition", "log_id", "phase_bin"], as_index=False)[
            ["true_fx_b", "pred_fx_b", "true_fz_b", "pred_fz_b"]
        ]
        .mean()
    )
    macro = (
        per_log.groupby(["partition", "phase_bin"], as_index=False)
        .agg(
            data_fx=("true_fx_b", "mean"),
            model_fx=("pred_fx_b", "mean"),
            data_fz=("true_fz_b", "mean"),
            model_fz=("pred_fz_b", "mean"),
            log_count=("log_id", "nunique"),
        )
        .sort_values(["partition", "phase_bin"])
    )
    if len(macro) != 2 * phase_bins or macro[
        ["data_fx", "model_fx", "data_fz", "model_fz"]
    ].isna().any().any():
        raise ValueError("Every train/validation phase bin must be populated")
    macro["phase_rad"] = (macro["phase_bin"] + 0.5) * 2.0 * np.pi / phase_bins
    macro["phase_deg"] = np.degrees(macro["phase_rad"])
    return macro


def _smooth_periodic(values: np.ndarray, *, window: int, order: int) -> np.ndarray:
    return savgol_filter(
        np.asarray(values, dtype=float),
        window_length=window,
        polyorder=order,
        mode="wrap",
    )


def _circular_delta(angle: float, reference: float) -> float:
    return float((angle - reference + np.pi) % (2.0 * np.pi) - np.pi)


def _phase_fields(prefix: str, phase_rad: float) -> dict[str, float]:
    wrapped = float(phase_rad % (2.0 * np.pi))
    return {
        f"{prefix}_rad": wrapped,
        f"{prefix}_deg": float(np.degrees(wrapped)),
        f"{prefix}_cycle_fraction": wrapped / (2.0 * np.pi),
    }


def _half_height_width(phase: np.ndarray, values: np.ndarray, peak_index: int) -> float:
    minimum = float(np.min(values))
    threshold = minimum + 0.5 * (float(values[peak_index]) - minimum)
    mask = values >= threshold
    if not mask[peak_index]:
        return float("nan")
    count = 1
    for direction in (-1, 1):
        index = (peak_index + direction) % len(values)
        while index != peak_index and mask[index]:
            count += 1
            index = (index + direction) % len(values)
    return min(count, len(values)) * 2.0 * np.pi / len(values)


def _curve_metrics(
    phase: np.ndarray,
    data: np.ndarray,
    model: np.ndarray,
    *,
    component: str,
    smooth_window: int,
    smooth_order: int,
    fx_interval_rad: tuple[float, float],
) -> dict[str, float]:
    data = np.asarray(data, dtype=float)
    model = np.asarray(model, dtype=float)
    smooth_data = _smooth_periodic(data, window=smooth_window, order=smooth_order)
    smooth_model = _smooth_periodic(model, window=smooth_window, order=smooth_order)
    residual = model - data
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    scale = float(np.ptp(data))
    result: dict[str, float] = {
        "rmse": rmse,
        "nrmse_range": rmse / scale if scale > 1.0e-12 else float("nan"),
        "mae": float(np.mean(np.abs(residual))),
        "pearson_r": float(np.corrcoef(data, model)[0, 1])
        if np.std(data) > 1.0e-12 and np.std(model) > 1.0e-12
        else float("nan"),
    }
    data_c1 = 2.0 / len(phase) * np.sum(data * np.exp(-1j * phase))
    model_c1 = 2.0 / len(phase) * np.sum(model * np.exp(-1j * phase))
    result.update(
        {
            "data_first_harmonic_amplitude": float(abs(data_c1)),
            "model_first_harmonic_amplitude": float(abs(model_c1)),
            "data_first_harmonic_phase_rad": float((-np.angle(data_c1)) % (2.0 * np.pi)),
            "model_first_harmonic_phase_rad": float((-np.angle(model_c1)) % (2.0 * np.pi)),
            "first_harmonic_phase_error_rad": abs(
                _circular_delta(-np.angle(model_c1), -np.angle(data_c1))
            ),
        }
    )
    centered_data = data - np.mean(data)
    centered_model = model - np.mean(model)
    correlations = np.array(
        [
            np.dot(centered_data, np.roll(centered_model, shift))
            for shift in range(len(phase))
        ]
    )
    best_shift = int(np.argmax(correlations))
    signed_shift = best_shift if best_shift <= len(phase) // 2 else best_shift - len(phase)
    lag = signed_shift * 2.0 * np.pi / len(phase)
    result.update(_phase_fields("circular_xcorr_lag", lag))

    if component == "fx":
        mask = (phase >= fx_interval_rad[0]) & (phase <= fx_interval_rad[1])
        indices = np.flatnonzero(mask)
        data_primary = int(indices[np.argmax(data[mask])])
        model_primary = int(indices[np.argmax(model[mask])])
        data_primary_smooth = int(indices[np.argmax(smooth_data[mask])])
        model_primary_smooth = int(indices[np.argmax(smooth_model[mask])])
        data_full = int(np.argmax(data))
        model_full = int(np.argmax(model))
        data_full_smooth = int(np.argmax(smooth_data))
        model_full_smooth = int(np.argmax(smooth_model))
        result.update(_phase_fields("data_primary_peak_phase", phase[data_primary]))
        result.update(_phase_fields("model_primary_peak_phase", phase[model_primary]))
        result.update(_phase_fields("data_primary_peak_phase_smooth", phase[data_primary_smooth]))
        result.update(_phase_fields("model_primary_peak_phase_smooth", phase[model_primary_smooth]))
        result.update(_phase_fields("data_full_peak_phase", phase[data_full]))
        result.update(_phase_fields("model_full_peak_phase", phase[model_full]))
        result.update(_phase_fields("data_full_peak_phase_smooth", phase[data_full_smooth]))
        result.update(_phase_fields("model_full_peak_phase_smooth", phase[model_full_smooth]))
        result.update(
            {
                "data_primary_peak_magnitude": float(data[data_primary]),
                "model_primary_peak_magnitude": float(model[model_primary]),
                "primary_peak_magnitude_error": float(model[model_primary] - data[data_primary]),
                "primary_peak_phase_error_rad": abs(
                    _circular_delta(phase[model_primary], phase[data_primary])
                ),
                "primary_peak_phase_error_deg": abs(
                    math.degrees(_circular_delta(phase[model_primary], phase[data_primary]))
                ),
                "model_peak_half_height_width_rad": _half_height_width(
                    phase, model, model_primary
                ),
            }
        )
        for name, low, high in (
            ("integral_90_180", 0.5 * np.pi, np.pi),
            ("integral_180_270", np.pi, 1.5 * np.pi),
        ):
            interval = (phase >= low) & (phase <= high)
            result[f"data_{name}_n_rad"] = float(np.trapz(data[interval], phase[interval]))
            result[f"model_{name}_n_rad"] = float(np.trapz(model[interval], phase[interval]))
    else:
        data_min = int(np.argmin(data))
        model_min = int(np.argmin(model))
        data_min_smooth = int(np.argmin(smooth_data))
        model_min_smooth = int(np.argmin(smooth_model))
        result.update(_phase_fields("data_minimum_phase", phase[data_min]))
        result.update(_phase_fields("model_minimum_phase", phase[model_min]))
        result.update(_phase_fields("data_minimum_phase_smooth", phase[data_min_smooth]))
        result.update(_phase_fields("model_minimum_phase_smooth", phase[model_min_smooth]))
        result.update(
            {
                "data_minimum_magnitude": float(data[data_min]),
                "model_minimum_magnitude": float(model[model_min]),
                "minimum_magnitude_error": float(model[model_min] - data[data_min]),
                "minimum_amplitude_error_abs": abs(
                    float(model[model_min] - data[data_min])
                ),
            }
        )
    return result


def evaluate_candidate(
    resolved: ResolvedExperiment,
    rows: Mapping[str, pd.DataFrame],
    candidate: TwistCandidate,
    *,
    airflow_mode: str | None = None,
    include_raw_predictions: bool = False,
) -> CandidateEvaluation | tuple[CandidateEvaluation, pd.DataFrame]:
    airflow = resolved.config["airflow"]
    phase_config = resolved.config["phase_binning"]
    assert isinstance(airflow, Mapping)
    assert isinstance(phase_config, Mapping)
    mode = str(airflow_mode or airflow["main_mode"])
    config = _baseline_config(resolved, candidate, airflow_mode=mode)
    pieces: list[pd.DataFrame] = []
    maximum_separation_fraction = 0.0
    for partition in ("train", "validation"):
        frame = rows[partition]
        for _, segment in frame.groupby(["log_id", "segment_id"], sort=False, dropna=False):
            evaluated = evaluate_wing_only_delaurier_segment(
                segment,
                theta_tip_deg=[candidate.A_tip_deg],
                geometry_path=resolved.geometry_path,
                config=config,
                phase_acceleration_mode=str(
                    resolved.config["physics"]["phase_acceleration_mode"]
                ),
            )
            evaluated.insert(0, "partition", partition)
            pieces.append(
                evaluated[
                    [
                        "partition",
                        "log_id",
                        "segment_id",
                        "timestamp_us",
                        "mechanical_phase_rad",
                        "true_fx_b",
                        "pred_fx_b",
                        "true_fz_b",
                        "pred_fz_b",
                    ]
                ]
            )
    aligned = pd.concat(pieces, ignore_index=True)
    if not np.isfinite(
        aligned[["pred_fx_b", "pred_fz_b"]].to_numpy(dtype=float)
    ).all():
        raise ValueError(f"{candidate.parameter_hash}: non-finite force prediction")
    phase_bins = int(phase_config["bins"])
    curves = _phase_macro_curves(aligned, phase_bins=phase_bins)
    smoothing = phase_config["smoothing"]
    interval = tuple(
        math.radians(float(value)) for value in phase_config["primary_fx_interval_deg"]
    )
    metric_rows: list[dict[str, object]] = []
    for partition in ("train", "validation"):
        curve = curves.loc[curves["partition"] == partition].sort_values("phase_bin")
        phase = curve["phase_rad"].to_numpy(dtype=float)
        for component in FORCES:
            metrics = _curve_metrics(
                phase,
                curve[f"data_{component}"].to_numpy(dtype=float),
                curve[f"model_{component}"].to_numpy(dtype=float),
                component=component,
                smooth_window=int(smoothing["window_bins"]),
                smooth_order=int(smoothing["polynomial_order"]),
                fx_interval_rad=(float(interval[0]), float(interval[1])),
            )
            metric_rows.append(
                {
                    "parameter_hash": candidate.parameter_hash,
                    **asdict(candidate),
                    "partition": partition,
                    "component": component,
                    "airflow_mode": mode,
                    "frame_contract": FRAME_CONTRACT,
                    **metrics,
                }
            )
    diagnostics = {
        "finite": True,
        "maximum_separation_fraction": maximum_separation_fraction,
        "sample_rows": {key: int(len(value)) for key, value in rows.items()},
        "phase_bins": phase_bins,
    }
    result = CandidateEvaluation(
        metrics=pd.DataFrame(metric_rows),
        curves=curves.assign(
            parameter_hash=candidate.parameter_hash,
            profile_name=candidate.profile_name,
            A_tip_deg=candidate.A_tip_deg,
            kappa=candidate.kappa,
            psi_theta_deg=candidate.psi_theta_deg,
            static_twist_offset_deg=candidate.static_twist_offset_deg,
            airflow_mode=mode,
        ),
        diagnostics=diagnostics,
    )
    return (result, aligned) if include_raw_predictions else result


def _manifest_base(resolved: ResolvedExperiment) -> dict[str, object]:
    dataset_entry = resolved.config["data"]
    physics = resolved.config["physics"]
    assert isinstance(dataset_entry, Mapping)
    assert isinstance(physics, Mapping)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": resolved.config["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "branch": _git(resolved.project_root, "branch", "--show-current"),
            "commit": _git(resolved.project_root, "rev-parse", "HEAD"),
            "dirty": bool(_git(resolved.project_root, "status", "--short")),
        },
        "config_path": str(resolved.config_path),
        "config_sha256": sha256_file(resolved.config_path),
        "dataset": {
            "id": dataset_entry["dataset_id"],
            "root": str(resolved.dataset_root),
            "repository_relative_path": str(resolved.dataset_root.relative_to(resolved.project_root)),
            "manifest_path": str(resolved.dataset_manifest_path),
            "manifest_sha256": sha256_file(resolved.dataset_manifest_path),
            "sample_sha256": {
                partition: sha256_file(resolved.dataset_root / PARTITION_FILES[partition])
                for partition in ("train", "validation")
            },
            "partitions_loaded": ["train", "validation"],
            "test_partition_loaded": False,
            "test_rows_loaded": 0,
            "phase_contract": resolved.dataset_manifest["phase_contract_version"],
            "frequency_contract": resolved.dataset_manifest["frequency_contract_version"],
        },
        "prior": {
            "id": dataset_entry["prior_id"],
            "root": str(resolved.prior_root),
            "lifecycle_status": "active",
            "manifest_sha256": sha256_file(resolved.prior_manifest_path),
            "prediction_sha256": {
                partition: sha256_file(resolved.prior_root / PREDICTION_FILES[partition])
                for partition in ("train", "validation")
            },
            "physics_source_commit": resolved.prior_manifest["physics_source"]["commit"],
            "frame_contract": resolved.prior_manifest["contracts"]["frame_contract"],
            "airflow_contract": resolved.prior_manifest["contracts"]["airflow_contract"],
            "phase_contract": resolved.prior_manifest["contracts"]["phase_contract"],
            "partition_coverage": list(resolved.prior_manifest["partitions"]),
        },
        "correction_ready": {
            "root": str(resolved.correction_ready_root),
            "manifest_sha256": sha256_file(resolved.correction_ready_manifest_path),
            "accepted_cycle_count": resolved.correction_ready_manifest["accepted_cycle_count"],
            "alignment_keys": list(resolved.correction_ready_manifest["key_columns"]),
        },
        "contracts": {
            "frame": FRAME_CONTRACT,
            "phase_coordinate": physics["phase_coordinate"],
            "phase_mapping": physics["phase_mapping"],
            "frequency_column": "flap_frequency_hz",
            "separation_enabled": False,
            "test_sealed": True,
        },
        "random_seed": int(resolved.config["random_seed"]),
    }


def _ensure_output(resolved: ResolvedExperiment) -> None:
    resolved.output_root.mkdir(parents=True, exist_ok=True)
    (resolved.output_root / "figures").mkdir(exist_ok=True)
    (resolved.output_root / "logs").mkdir(exist_ok=True)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _baseline_candidate(resolved: ResolvedExperiment) -> TwistCandidate:
    baseline = resolved.config["physics"]["baseline"]
    assert isinstance(baseline, Mapping)
    return TwistCandidate(
        profile_name=str(baseline["profile_name"]),
        A_tip_deg=float(baseline["A_tip_deg"]),
        kappa=float(baseline["kappa"]),
        psi_theta_deg=float(baseline["psi_theta_deg"]),
        static_twist_offset_deg=float(baseline["static_twist_offset_deg"]),
        stage="baseline",
        family="legacy_baseline",
    )


def _plot_baseline(
    curves: pd.DataFrame,
    metrics: pd.DataFrame,
    path: Path,
    *,
    zero_wind_curves: pd.DataFrame | None = None,
    smoothing: Mapping[str, object],
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), sharex=True, constrained_layout=True)
    for column, partition in enumerate(("train", "validation")):
        table = curves.loc[curves["partition"] == partition].sort_values("phase_bin")
        for row, component in enumerate(FORCES):
            ax = axes[row, column]
            phase_deg = table["phase_deg"].to_numpy(dtype=float)
            data = table[f"data_{component}"].to_numpy(dtype=float)
            model = table[f"model_{component}"].to_numpy(dtype=float)
            ax.plot(phase_deg, data, color="#202020", linewidth=2.0, label="flight data")
            ax.plot(
                phase_deg,
                model,
                color="#d95f02",
                linewidth=1.8,
                label="legacy DeLaurier, current EKF wind",
            )
            if zero_wind_curves is not None:
                zero_table = zero_wind_curves.loc[
                    zero_wind_curves["partition"] == partition
                ].sort_values("phase_bin")
                ax.plot(
                    phase_deg,
                    zero_table[f"model_{component}"].to_numpy(dtype=float),
                    color="#1b9e77",
                    linewidth=1.2,
                    linestyle=":",
                    label="legacy DeLaurier, zero wind",
                )
            ax.plot(
                phase_deg,
                _smooth_periodic(
                    model,
                    window=int(smoothing["window_bins"]),
                    order=int(smoothing["polynomial_order"]),
                ),
                color="#7570b3",
                linewidth=1.0,
                linestyle="--",
                label="model periodic zero-phase smooth",
            )
            ax.axvspan(180.0, 270.0, color="#e6ab02", alpha=0.08)
            ax.set_title(f"{partition} | body {component.upper()}")
            ax.set_ylabel("Force (N)")
            ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Mechanical phase (deg; 0 deg = horizontal, starting upstroke)")
        ax.set_xticks([0, 90, 180, 270, 360])
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Legacy DeLaurier baseline | body FRD (+Fx forward, +Fz down) | "
        "current EKF wind | 72 bins"
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def run_baseline(resolved: ResolvedExperiment) -> dict[str, object]:
    _ensure_output(resolved)
    airflow = str(resolved.config["airflow"]["main_mode"])
    rows = load_experiment_rows(resolved, airflow_mode=airflow)
    candidate = _baseline_candidate(resolved)
    result, aligned = evaluate_candidate(
        resolved,
        rows,
        candidate,
        airflow_mode=airflow,
        include_raw_predictions=True,
    )
    zero_mode = str(resolved.config["airflow"]["sensitivity_mode"])
    zero_rows = load_experiment_rows(resolved, airflow_mode=zero_mode)
    zero_result = evaluate_candidate(
        resolved,
        zero_rows,
        candidate,
        airflow_mode=zero_mode,
    )
    parity: dict[str, float] = {}
    for partition in ("train", "validation"):
        predictions = pd.read_parquet(
            resolved.prior_root / PREDICTION_FILES[partition],
            columns=["log_id", "timestamp_us", "fx_b", "fz_b"],
        )
        current = aligned.loc[
            aligned["partition"] == partition,
            ["log_id", "timestamp_us", "pred_fx_b", "pred_fz_b"],
        ]
        joined = current.merge(
            predictions,
            on=["log_id", "timestamp_us"],
            how="left",
            validate="one_to_one",
        )
        if joined[["fx_b", "fz_b"]].isna().any().any() or len(joined) != len(current):
            raise ValueError(f"{partition}: active prior baseline coverage mismatch")
        for component in FORCES:
            parity[f"{partition}_{component}_maximum_absolute_difference_n"] = float(
                np.max(
                    np.abs(
                        joined[f"pred_{component}_b"].to_numpy(dtype=float)
                        - joined[f"{component}_b"].to_numpy(dtype=float)
                    )
                )
            )
    if max(parity.values()) > 1.0e-10:
        raise RuntimeError(f"Baseline parity gate failed: {parity}")
    fx = result.metrics.loc[result.metrics["component"] == "fx"]
    full_peaks = {
        str(row.partition): float(row.model_full_peak_phase_deg)
        for row in fx.itertuples(index=False)
    }
    bin_width_deg = 360.0 / int(resolved.config["phase_binning"]["bins"])
    if any(abs(value - 177.5) > bin_width_deg + 1.0e-12 for value in full_peaks.values()):
        raise RuntimeError(f"Baseline Fx full-cycle peak gate failed: {full_peaks}")
    result.metrics.to_csv(resolved.output_root / "baseline_metrics.csv", index=False)
    zero_result.metrics.to_csv(
        resolved.output_root / "baseline_zero_wind_metrics.csv", index=False
    )
    result.curves.to_csv(resolved.output_root / "baseline_phase_curves.csv", index=False)
    _plot_baseline(
        result.curves,
        result.metrics,
        resolved.output_root / "baseline_train_validation.png",
        zero_wind_curves=zero_result.curves,
        smoothing=resolved.config["phase_binning"]["smoothing"],
        dpi=int(resolved.config["output"]["figure_dpi"]),
    )
    zero_fx = zero_result.metrics.loc[zero_result.metrics["component"] == "fx"]
    zero_fz = zero_result.metrics.loc[zero_result.metrics["component"] == "fz"]
    zero_peak_summary = {
        str(row.partition): {
            "fx_full_cycle_peak_deg": float(row.model_full_peak_phase_deg),
            "fx_primary_interval_peak_deg": float(row.model_primary_peak_phase_deg),
        }
        for row in zero_fx.itertuples(index=False)
    }
    for row in zero_fz.itertuples(index=False):
        zero_peak_summary[str(row.partition)]["fz_minimum_phase_deg"] = float(
            row.model_minimum_phase_deg
        )
    expected_zero_peaks = {
        "train": (177.5, 182.5, 162.5),
        "validation": (177.5, 182.5, 167.5),
    }
    zero_trend_parity = {
        partition: max(
            abs(zero_peak_summary[partition]["fx_full_cycle_peak_deg"] - expected[0]),
            abs(zero_peak_summary[partition]["fx_primary_interval_peak_deg"] - expected[1]),
            abs(zero_peak_summary[partition]["fz_minimum_phase_deg"] - expected[2]),
        )
        for partition, expected in expected_zero_peaks.items()
    }
    if max(zero_trend_parity.values()) > bin_width_deg + 1.0e-12:
        raise RuntimeError(
            f"Zero-wind baseline trend parity gate failed: {zero_trend_parity}"
        )
    manifest = {
        **_manifest_base(resolved),
        "stage": "baseline",
        "baseline_candidate": asdict(candidate),
        "baseline_parameter_hash": candidate.parameter_hash,
        "baseline_full_cycle_fx_peak_deg": full_peaks,
        "baseline_primary_interval_fx_peak_deg": {
            str(row.partition): float(row.model_primary_peak_phase_deg)
            for row in fx.itertuples(index=False)
        },
        "data_primary_interval_fx_peak_deg": {
            str(row.partition): float(row.data_primary_peak_phase_deg)
            for row in fx.itertuples(index=False)
        },
        "active_prior_parity": parity,
        "baseline_gate_passed": True,
        "airflow_mode": airflow,
        "wind_used": True,
        "zero_wind_sensitivity": {
            "airflow_mode": zero_mode,
            "peak_summary": zero_peak_summary,
            "reference_output": "outputs/delaurier_airflow_comparison/20260727T111450Z_59d90e6",
            "maximum_phase_difference_from_reference_deg": zero_trend_parity,
            "trend_gate_passed": True,
        },
        "smoothing": dict(resolved.config["phase_binning"]["smoothing"]),
    }
    _write_json(resolved.output_root / "baseline_manifest.json", manifest)
    _write_json(resolved.output_root / "manifest.json", manifest)
    return manifest


def oat_candidates(resolved: ResolvedExperiment) -> list[TwistCandidate]:
    grid = resolved.config["sweep"]["oat"]
    baseline = _baseline_candidate(resolved)
    candidates: list[TwistCandidate] = []
    for name in ("A_tip_deg", "kappa", "psi_theta_deg", "static_twist_offset_deg"):
        for value in grid[name]:
            values = {
                "profile_name": "quadratic2_phase",
                "A_tip_deg": baseline.A_tip_deg,
                "kappa": baseline.kappa,
                "psi_theta_deg": baseline.psi_theta_deg,
                "static_twist_offset_deg": baseline.static_twist_offset_deg,
            }
            values[name] = float(value)
            candidates.append(
                TwistCandidate(
                    **values,
                    stage="oat",
                    family=name,
                )
            )
    return candidates


def coarse_candidates(resolved: ResolvedExperiment) -> list[TwistCandidate]:
    grid = resolved.config["sweep"]["coarse"]
    return [
        TwistCandidate(
            profile_name="quadratic2_phase",
            A_tip_deg=float(amplitude),
            kappa=float(kappa),
            psi_theta_deg=float(phase),
            static_twist_offset_deg=0.0,
            stage="coarse",
            family="A_tip_x_kappa_x_psi",
        )
        for amplitude, kappa, phase in itertools.product(
            grid["A_tip_deg"], grid["kappa"], grid["psi_theta_deg"]
        )
    ]


_WORKER_RESOLVED: ResolvedExperiment | None = None
_WORKER_ROWS: Mapping[str, pd.DataFrame] | None = None
_WORKER_AIRFLOW_MODE: str | None = None


def _worker_evaluate(candidate: TwistCandidate) -> dict[str, object]:
    if _WORKER_RESOLVED is None or _WORKER_ROWS is None or _WORKER_AIRFLOW_MODE is None:
        raise RuntimeError("Sweep worker context is not initialized")
    result = evaluate_candidate(
        _WORKER_RESOLVED,
        _WORKER_ROWS,
        candidate,
        airflow_mode=_WORKER_AIRFLOW_MODE,
    )
    assert isinstance(result, CandidateEvaluation)
    return {
        "parameter_hash": candidate.parameter_hash,
        "candidate": asdict(candidate),
        "metrics": result.metrics.to_dict(orient="records"),
        "diagnostics": dict(result.diagnostics),
    }


def _evaluate_candidate_batch(
    resolved: ResolvedExperiment,
    rows: Mapping[str, pd.DataFrame],
    candidates: Sequence[TwistCandidate],
    *,
    airflow_mode: str,
) -> list[dict[str, object]]:
    """Evaluate candidates while sharing non-twist kinematics exactly.

    The aerodynamic equations remain in ``compute_delaurier_strip_loads``.
    This routine only hoists candidate-invariant geometry, airflow, stroke
    kinematics, and body transforms out of the candidate loop.
    """

    if not candidates:
        return []
    static_offsets = {round(float(value.static_twist_offset_deg), 12) for value in candidates}
    if len(static_offsets) != 1:
        raise ValueError("A shared evaluation batch requires one static twist offset")
    configs = {
        candidate.parameter_hash: _baseline_config(
            resolved, candidate, airflow_mode=airflow_mode
        )
        for candidate in candidates
    }
    base = configs[candidates[0].parameter_hash]
    geometry = load_wing_geometry_csv(
        resolved.geometry_path,
        num_strips=base.num_strips,
        d_hat=0.0,
    )
    phase_config = resolved.config["phase_binning"]
    smoothing = phase_config["smoothing"]
    phase_bins = int(phase_config["bins"])
    interval = tuple(
        math.radians(float(value)) for value in phase_config["primary_fx_interval_deg"]
    )
    data_accumulator: dict[tuple[str, str, int], np.ndarray] = {}
    model_accumulators: dict[str, dict[tuple[str, str, int], np.ndarray]] = {
        candidate.parameter_hash: {} for candidate in candidates
    }
    chunk_size = int(resolved.config["runtime"]["chunk_size"])
    for partition in ("train", "validation"):
        for (log_id, _), segment in rows[partition].groupby(
            ["log_id", "segment_id"], sort=False, dropna=False
        ):
            ordered = segment.sort_values("time_s", kind="stable")
            for start in range(0, len(ordered), chunk_size):
                frame = ordered.iloc[start : start + chunk_size]
                phase_canonical = frame["mechanical_phase_rad"].to_numpy(dtype=float)
                phase_delaurier = map_canonical_phase_to_delaurier(phase_canonical)
                phase_rate = (
                    2.0
                    * np.pi
                    * frame["flap_frequency_hz"].to_numpy(dtype=float)
                )
                phase_acceleration = np.zeros_like(phase_rate)
                amplitude = float(base.stroke_amplitude_rad)
                q = amplitude * np.cos(phase_delaurier)
                q_dot = -amplitude * np.sin(phase_delaurier) * phase_rate
                q_ddot = -amplitude * np.cos(phase_delaurier) * np.square(phase_rate)
                span = geometry.x_mid[None, :]
                h = -q[:, None] * span
                hdot = -q_dot[:, None] * span
                hddot = -q_ddot[:, None] * span
                airflow = _resolve_airflow_inputs(frame, base)
                incidence = airflow["incidence"]
                theta_bar = incidence + float(base.mean_pitch_offset_rad)
                airspeed = airflow["forward_speed"]
                rho = frame["vehicle_air_data.rho"].to_numpy(dtype=float)
                left_transform, right_transform = _wing_polar_transforms_frd(q, base)
                wrapped = np.mod(phase_canonical, 2.0 * np.pi)
                bins = np.minimum(
                    (wrapped / (2.0 * np.pi) * phase_bins).astype(int),
                    phase_bins - 1,
                )
                true_fx = frame["fx_b"].to_numpy(dtype=float)
                true_fz = frame["fz_b"].to_numpy(dtype=float)
                for bin_index in np.unique(bins):
                    mask = bins == bin_index
                    key = (partition, str(log_id), int(bin_index))
                    values = data_accumulator.setdefault(key, np.zeros(3, dtype=float))
                    values += [
                        float(np.count_nonzero(mask)),
                        float(np.sum(true_fx[mask])),
                        float(np.sum(true_fz[mask])),
                    ]
                for candidate in candidates:
                    config = configs[candidate.parameter_hash]
                    twist = compute_delaurier_dynamic_twist(
                        strip_span_m=geometry.x_mid,
                        strip_width_m=geometry.dx,
                        semi_span_m=geometry.semi_span_m,
                        mean_pitch_rad=theta_bar,
                        tip_twist_amplitude_rad=math.radians(candidate.A_tip_deg),
                        phase_rad=phase_delaurier,
                        phase_rate_rad_s=phase_rate,
                        phase_acceleration_rad_s2=phase_acceleration,
                        enabled=True,
                        profile_name=candidate.profile_name,
                        kappa=candidate.kappa,
                        phase_offset_rad=math.radians(candidate.psi_theta_deg),
                    )
                    loads = compute_delaurier_strip_loads(
                        h,
                        hdot,
                        hddot,
                        twist.theta,
                        twist.theta_dot,
                        twist.theta_ddot,
                        geometry,
                        rho,
                        airspeed,
                        theta_a=incidence,
                        theta_bar=theta_bar,
                        omega_ref_rad_s=phase_rate,
                        params=config.params,
                        enable_separation=False,
                    )
                    wrench = integrate_delaurier_strip_wrench(loads)
                    left_force = np.einsum(
                        "nij,nj->ni", left_transform, wrench.force_wang
                    )
                    right_force = np.einsum(
                        "nij,nj->ni", right_transform, wrench.force_wang
                    )
                    total_force = left_force + right_force
                    accumulator = model_accumulators[candidate.parameter_hash]
                    for bin_index in np.unique(bins):
                        mask = bins == bin_index
                        key = (partition, str(log_id), int(bin_index))
                        values = accumulator.setdefault(key, np.zeros(3, dtype=float))
                        values += [
                            float(np.count_nonzero(mask)),
                            float(np.sum(total_force[mask, 0])),
                            float(np.sum(total_force[mask, 2])),
                        ]
    records: list[dict[str, object]] = []
    for candidate in candidates:
        curve_rows: list[dict[str, object]] = []
        model_accumulator = model_accumulators[candidate.parameter_hash]
        for partition in ("train", "validation"):
            log_ids = sorted(
                {key[1] for key in data_accumulator if key[0] == partition}
            )
            for bin_index in range(phase_bins):
                per_log: list[list[float]] = []
                for log_id in log_ids:
                    key = (partition, log_id, bin_index)
                    if key not in data_accumulator or key not in model_accumulator:
                        raise ValueError(
                            f"{candidate.parameter_hash}: missing {partition}/{log_id}/bin {bin_index}"
                        )
                    data_values = data_accumulator[key]
                    model_values = model_accumulator[key]
                    if data_values[0] != model_values[0] or data_values[0] <= 0.0:
                        raise ValueError("Batch phase-bin count mismatch")
                    per_log.append(
                        [
                            data_values[1] / data_values[0],
                            model_values[1] / model_values[0],
                            data_values[2] / data_values[0],
                            model_values[2] / model_values[0],
                        ]
                    )
                macro = np.mean(np.asarray(per_log), axis=0)
                phase_rad = (bin_index + 0.5) * 2.0 * np.pi / phase_bins
                curve_rows.append(
                    {
                        "partition": partition,
                        "phase_bin": bin_index,
                        "data_fx": macro[0],
                        "model_fx": macro[1],
                        "data_fz": macro[2],
                        "model_fz": macro[3],
                        "log_count": len(log_ids),
                        "phase_rad": phase_rad,
                        "phase_deg": math.degrees(phase_rad),
                    }
                )
        curves = pd.DataFrame(curve_rows)
        metric_rows: list[dict[str, object]] = []
        for partition in ("train", "validation"):
            curve = curves.loc[curves["partition"] == partition].sort_values("phase_bin")
            phase = curve["phase_rad"].to_numpy(dtype=float)
            for component in FORCES:
                metric_rows.append(
                    {
                        "parameter_hash": candidate.parameter_hash,
                        **asdict(candidate),
                        "partition": partition,
                        "component": component,
                        "airflow_mode": airflow_mode,
                        "frame_contract": FRAME_CONTRACT,
                        **_curve_metrics(
                            phase,
                            curve[f"data_{component}"].to_numpy(dtype=float),
                            curve[f"model_{component}"].to_numpy(dtype=float),
                            component=component,
                            smooth_window=int(smoothing["window_bins"]),
                            smooth_order=int(smoothing["polynomial_order"]),
                            fx_interval_rad=(float(interval[0]), float(interval[1])),
                        ),
                    }
                )
        records.append(
            {
                "parameter_hash": candidate.parameter_hash,
                "candidate": asdict(candidate),
                "metrics": metric_rows,
                "diagnostics": {
                    "finite": True,
                    "maximum_separation_fraction": 0.0,
                    "sample_rows": {
                        key: int(len(value)) for key, value in rows.items()
                    },
                    "phase_bins": phase_bins,
                    "shared_exact_candidate_batch": True,
                },
            }
        )
    return records


def _worker_evaluate_batch(candidates: Sequence[TwistCandidate]) -> list[dict[str, object]]:
    if _WORKER_RESOLVED is None or _WORKER_ROWS is None or _WORKER_AIRFLOW_MODE is None:
        raise RuntimeError("Sweep worker context is not initialized")
    return _evaluate_candidate_batch(
        _WORKER_RESOLVED,
        _WORKER_ROWS,
        candidates,
        airflow_mode=_WORKER_AIRFLOW_MODE,
    )


def _read_jsonl(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    if not path.is_file():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or "parameter_hash" not in value:
            raise ValueError(f"Invalid compact result at {path}:{line_number}")
        candidate_raw = value.get("candidate")
        if not isinstance(candidate_raw, dict):
            raise ValueError(f"Missing candidate mapping at {path}:{line_number}")
        canonical_hash = TwistCandidate(**candidate_raw).parameter_hash
        value["parameter_hash"] = canonical_hash
        metrics = value.get("metrics")
        if not isinstance(metrics, list):
            raise ValueError(f"Missing metrics list at {path}:{line_number}")
        for metric in metrics:
            metric["parameter_hash"] = canonical_hash
        records[canonical_hash] = value
    return records


def _clone_zero_amplitude_record(
    source: Mapping[str, object],
    candidate: TwistCandidate,
) -> dict[str, object]:
    if not np.isclose(candidate.A_tip_deg, 0.0):
        raise ValueError("Zero-amplitude equivalence may only clone A_tip=0 candidates")
    metrics: list[dict[str, object]] = []
    for raw in source["metrics"]:
        metric = dict(raw)
        metric.update(
            {
                "parameter_hash": candidate.parameter_hash,
                **asdict(candidate),
            }
        )
        metrics.append(metric)
    diagnostics = dict(source["diagnostics"])
    diagnostics["reused_exact_zero_amplitude_equivalence"] = True
    diagnostics["equivalent_source_parameter_hash"] = source["parameter_hash"]
    return {
        "parameter_hash": candidate.parameter_hash,
        "candidate": asdict(candidate),
        "metrics": metrics,
        "diagnostics": diagnostics,
    }


def run_candidates(
    resolved: ResolvedExperiment,
    candidates: Sequence[TwistCandidate],
    *,
    stage: str,
    workers: int,
    resume: bool,
) -> pd.DataFrame:
    _ensure_output(resolved)
    baseline_manifest = resolved.output_root / "baseline_manifest.json"
    if not baseline_manifest.is_file() or not _read_mapping(baseline_manifest).get(
        "baseline_gate_passed", False
    ):
        raise RuntimeError("Baseline gate must pass before any sweep stage")
    if workers <= 0:
        raise ValueError("workers must be positive")
    log_path = resolved.output_root / "logs" / f"{stage}_compact_results.jsonl"
    completed = _read_jsonl(log_path) if resume else {}
    if log_path.exists() and not resume:
        raise FileExistsError(f"{log_path} exists; use --resume")
    zero_source = next(
        (
            record
            for record in completed.values()
            if np.isclose(float(record["candidate"]["A_tip_deg"]), 0.0)
            and np.isclose(float(record["candidate"]["static_twist_offset_deg"]), 0.0)
        ),
        None,
    )
    if zero_source is not None:
        with log_path.open("a", encoding="utf-8") as stream:
            for candidate in candidates:
                if (
                    candidate.parameter_hash not in completed
                    and np.isclose(candidate.A_tip_deg, 0.0)
                    and np.isclose(candidate.static_twist_offset_deg, 0.0)
                ):
                    record = _clone_zero_amplitude_record(zero_source, candidate)
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                    completed[candidate.parameter_hash] = record
    pending = [candidate for candidate in candidates if candidate.parameter_hash not in completed]
    airflow_mode = str(resolved.config["airflow"]["main_mode"])
    rows = load_experiment_rows(resolved, airflow_mode=airflow_mode)
    global _WORKER_RESOLVED, _WORKER_ROWS, _WORKER_AIRFLOW_MODE
    _WORKER_RESOLVED = resolved
    _WORKER_ROWS = rows
    _WORKER_AIRFLOW_MODE = airflow_mode
    started = time.monotonic()
    done_at_start = len(completed)
    batch_size = (
        int(resolved.config["runtime"]["candidate_batch_size"])
        if stage in {"coarse", "refine"}
        else 1
    )
    batches = [
        pending[index : index + batch_size]
        for index in range(0, len(pending), batch_size)
    ]

    def persist(records: Sequence[Mapping[str, object]], stream) -> None:
        for raw_record in records:
            record = dict(raw_record)
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            completed[str(record["parameter_hash"])] = record
        count = len(completed)
        elapsed = max(time.monotonic() - started, 1.0e-9)
        rate = (count - done_at_start) / elapsed
        remaining = (len(candidates) - count) / rate if rate > 0.0 else float("inf")
        print(
            f"[{stage}] {count}/{len(candidates)} | {rate:.3f} candidate/s | "
            f"ETA {remaining / 60.0:.1f} min",
            flush=True,
        )

    with log_path.open("a", encoding="utf-8") as stream:
        if workers == 1:
            for batch in batches:
                records = (
                    _worker_evaluate_batch(batch)
                    if len(batch) > 1
                    else [_worker_evaluate(batch[0])]
                )
                persist(records, stream)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    (
                        executor.submit(_worker_evaluate_batch, batch)
                        if len(batch) > 1
                        else executor.submit(_worker_evaluate, batch[0])
                    ): batch
                    for batch in batches
                }
                for future in as_completed(futures):
                    value = future.result()
                    records = value if isinstance(value, list) else [value]
                    persist(records, stream)
    requested = {candidate.parameter_hash for candidate in candidates}
    missing = requested - set(completed)
    if missing:
        raise RuntimeError(f"{stage}: incomplete parameter hashes: {sorted(missing)[:5]}")
    canonical_records = [completed[key] for key in sorted(requested)]
    temporary_log = log_path.with_suffix(log_path.suffix + ".tmp")
    temporary_log.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in canonical_records),
        encoding="utf-8",
    )
    temporary_log.replace(log_path)
    metric_records: list[dict[str, object]] = []
    for candidate in candidates:
        for raw_metric in completed[candidate.parameter_hash]["metrics"]:
            metric = dict(raw_metric)
            metric.update({"parameter_hash": candidate.parameter_hash, **asdict(candidate)})
            metric_records.append(metric)
    result = pd.DataFrame(metric_records).sort_values(
        ["parameter_hash", "partition", "component"], kind="stable"
    )
    expected = len(candidates) * 4
    if len(result) != expected:
        raise RuntimeError(f"{stage}: result row mismatch {len(result)} != {expected}")
    return result.reset_index(drop=True)


def run_oat(
    resolved: ResolvedExperiment,
    *,
    workers: int,
    resume: bool,
) -> pd.DataFrame:
    result = run_candidates(
        resolved,
        oat_candidates(resolved),
        stage="oat",
        workers=workers,
        resume=resume,
    )
    result.to_csv(resolved.output_root / "oat_results.csv", index=False)
    return result


def run_coarse(
    resolved: ResolvedExperiment,
    *,
    workers: int,
    resume: bool,
) -> pd.DataFrame:
    result = run_candidates(
        resolved,
        coarse_candidates(resolved),
        stage="coarse",
        workers=workers,
        resume=resume,
    )
    result.to_parquet(resolved.output_root / "coarse_grid_results.parquet", index=False)
    return result


def _wide_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "parameter_hash",
        "profile_name",
        "A_tip_deg",
        "kappa",
        "psi_theta_deg",
        "static_twist_offset_deg",
        "stage",
        "family",
        "airflow_mode",
    ]
    base = metrics[identity].drop_duplicates("parameter_hash")
    fields = [
        "rmse",
        "mae",
        "pearson_r",
        "primary_peak_phase_error_deg",
        "model_primary_peak_phase_deg",
        "data_primary_peak_phase_deg",
        "model_full_peak_phase_deg",
        "data_full_peak_phase_deg",
        "model_primary_peak_magnitude",
        "data_primary_peak_magnitude",
        "model_first_harmonic_phase_rad",
        "first_harmonic_phase_error_rad",
        "circular_xcorr_lag_deg",
        "model_peak_half_height_width_rad",
        "model_minimum_phase_deg",
        "data_minimum_phase_deg",
        "model_minimum_magnitude",
        "data_minimum_magnitude",
        "minimum_amplitude_error_abs",
    ]
    out = base
    for partition in ("train", "validation"):
        for component in FORCES:
            subset = metrics.loc[
                (metrics["partition"] == partition) & (metrics["component"] == component),
                ["parameter_hash", *[field for field in fields if field in metrics.columns]],
            ].copy()
            subset = subset.rename(
                columns={
                    field: f"{partition}_{component}_{field}"
                    for field in subset.columns
                    if field != "parameter_hash"
                }
            )
            out = out.merge(subset, on="parameter_hash", validate="one_to_one")
    return out


def _pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    values = np.column_stack((x, y))
    mask = np.ones(len(values), dtype=bool)
    for index, value in enumerate(values):
        dominated = np.any(
            np.all(values <= value, axis=1)
            & np.any(values < value, axis=1)
        )
        mask[index] = not dominated
    return mask


def _coarse_pareto_seeds(resolved: ResolvedExperiment, coarse: pd.DataFrame) -> pd.DataFrame:
    wide = _wide_metrics(coarse)
    phase = wide["validation_fx_primary_peak_phase_error_deg"].to_numpy(dtype=float)
    fz = wide["validation_fz_rmse"].to_numpy(dtype=float)
    finite = np.isfinite(phase) & np.isfinite(fz)
    wide = wide.loc[finite].copy()
    wide["pareto"] = _pareto_mask(
        wide["validation_fx_primary_peak_phase_error_deg"].to_numpy(dtype=float),
        wide["validation_fz_rmse"].to_numpy(dtype=float),
    )
    seeds = wide.loc[wide["pareto"]].copy()
    seeds["seed_rank_score"] = (
        seeds["validation_fx_primary_peak_phase_error_deg"]
        / max(float(seeds["validation_fx_primary_peak_phase_error_deg"].max()), 1.0)
        + seeds["validation_fz_rmse"] / max(float(seeds["validation_fz_rmse"].max()), 1.0)
    )
    maximum = int(resolved.config["sweep"]["refine"]["maximum_pareto_seeds"])
    return seeds.sort_values(
        ["seed_rank_score", "A_tip_deg", "kappa", "psi_theta_deg"],
        kind="stable",
    ).head(maximum)


def refined_candidates(resolved: ResolvedExperiment, coarse: pd.DataFrame) -> list[TwistCandidate]:
    refine = resolved.config["sweep"]["refine"]
    allowed = resolved.config["physics"]["allowed"]
    seeds = _coarse_pareto_seeds(resolved, coarse)
    steps = int(refine["neighbor_steps"])
    candidates: dict[tuple[float, float, float], TwistCandidate] = {}
    for seed in seeds.itertuples(index=False):
        for da, dk, dp in itertools.product(
            range(-steps, steps + 1),
            range(-steps, steps + 1),
            range(-steps, steps + 1),
        ):
            amplitude = float(seed.A_tip_deg) + da * float(refine["A_tip_step_deg"])
            kappa = float(seed.kappa) + dk * float(refine["kappa_step"])
            phase = float(seed.psi_theta_deg) + dp * float(refine["psi_theta_step_deg"])
            if not (
                float(allowed["A_tip_deg"][0]) <= amplitude <= float(allowed["A_tip_deg"][1])
                and float(allowed["kappa"][0]) <= kappa <= float(allowed["kappa"][1])
                and float(allowed["psi_theta_deg"][0])
                <= phase
                <= float(allowed["psi_theta_deg"][1])
            ):
                continue
            key = (round(amplitude, 12), round(kappa, 12), round(phase, 12))
            candidates[key] = TwistCandidate(
                profile_name="quadratic2_phase",
                A_tip_deg=amplitude,
                kappa=kappa,
                psi_theta_deg=phase,
                static_twist_offset_deg=0.0,
                stage="refine",
                family="pareto_local_neighborhood",
            )
    completed_hashes = {
        TwistCandidate(
            profile_name=str(row.profile_name),
            A_tip_deg=float(row.A_tip_deg),
            kappa=float(row.kappa),
            psi_theta_deg=float(row.psi_theta_deg),
            static_twist_offset_deg=float(row.static_twist_offset_deg),
        ).parameter_hash
        for row in coarse[
            [
                "profile_name",
                "A_tip_deg",
                "kappa",
                "psi_theta_deg",
                "static_twist_offset_deg",
            ]
        ].drop_duplicates().itertuples(index=False)
    }
    return [
        candidate
        for candidate in candidates.values()
        if candidate.parameter_hash not in completed_hashes
    ]


def run_refine(
    resolved: ResolvedExperiment,
    *,
    workers: int,
    resume: bool,
) -> pd.DataFrame:
    coarse_path = resolved.output_root / "coarse_grid_results.parquet"
    if not coarse_path.is_file():
        raise FileNotFoundError("Coarse results are required before refine")
    coarse = pd.read_parquet(coarse_path)
    candidates = refined_candidates(resolved, coarse)
    if not candidates:
        raise RuntimeError("No valid Pareto neighborhoods were generated")
    result = run_candidates(
        resolved,
        candidates,
        stage="refine",
        workers=workers,
        resume=resume,
    )
    result.to_parquet(resolved.output_root / "refined_results.parquet", index=False)
    seeds = _coarse_pareto_seeds(resolved, coarse)
    seeds.to_csv(resolved.output_root / "refinement_seeds.csv", index=False)
    return result


def _baseline_wide(resolved: ResolvedExperiment) -> pd.Series:
    path = resolved.output_root / "baseline_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _wide_metrics(pd.read_csv(path)).iloc[0]


def build_shortlists(
    resolved: ResolvedExperiment,
    candidate_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wide = _wide_metrics(candidate_metrics)
    baseline = _baseline_wide(resolved)
    selection = resolved.config["selection"]
    assert isinstance(selection, Mapping)
    finite_columns = [
        "train_fx_rmse",
        "validation_fx_rmse",
        "train_fz_rmse",
        "validation_fz_rmse",
        "train_fx_primary_peak_phase_error_deg",
        "validation_fx_primary_peak_phase_error_deg",
    ]
    feasible = np.isfinite(wide[finite_columns].to_numpy(dtype=float)).all(axis=1)
    if bool(selection["require_train_validation_same_fx_phase_direction"]):
        feasible &= (
            wide["train_fx_primary_peak_phase_error_deg"]
            <= float(baseline["train_fx_primary_peak_phase_error_deg"]) + 1.0e-12
        ) & (
            wide["validation_fx_primary_peak_phase_error_deg"]
            <= float(baseline["validation_fx_primary_peak_phase_error_deg"]) + 1.0e-12
        )
    feasible &= wide["validation_fx_rmse"] <= float(baseline["validation_fx_rmse"]) * (
        1.0 + float(selection["maximum_validation_fx_rmse_degradation_fraction"])
    )
    feasible &= wide["validation_fz_rmse"] <= float(baseline["validation_fz_rmse"]) * (
        1.0 + float(selection["maximum_validation_fz_rmse_degradation_fraction"])
    )
    feasible &= wide["validation_fz_minimum_amplitude_error_abs"] <= float(
        baseline["validation_fz_minimum_amplitude_error_abs"]
    ) * (1.0 + float(selection["maximum_fz_minimum_amplitude_error_fraction"]))
    minimum_half_width_rad = math.radians(
        float(selection["minimum_fx_peak_half_height_width_deg"])
    )
    feasible &= (
        wide["train_fx_model_peak_half_height_width_rad"] >= minimum_half_width_rad
    ) & (
        wide["validation_fx_model_peak_half_height_width_rad"] >= minimum_half_width_rad
    )
    wide["physical_constraints_passed"] = feasible
    pool = wide.loc[wide["physical_constraints_passed"]].copy()
    if pool.empty:
        # A negative result still needs evidence. Retain finite candidates but
        # label the failed physical gate rather than manufacturing a success.
        pool = wide.loc[
            np.isfinite(wide[finite_columns].to_numpy(dtype=float)).all(axis=1)
        ].copy()
    maximum = int(selection["maximum_shortlist_size"])
    fx_phase = pool.sort_values(
        [
            "validation_fx_primary_peak_phase_error_deg",
            "train_fx_primary_peak_phase_error_deg",
            "validation_fz_rmse",
            "validation_fx_rmse",
        ],
        kind="stable",
    ).head(maximum)
    pool["pareto_balanced"] = _pareto_mask(
        pool["validation_fx_primary_peak_phase_error_deg"].to_numpy(dtype=float),
        pool["validation_fz_rmse"].to_numpy(dtype=float),
    )
    balanced = pool.loc[pool["pareto_balanced"]].sort_values(
        [
            "validation_fx_primary_peak_phase_error_deg",
            "validation_fz_rmse",
            "validation_fx_rmse",
        ],
        kind="stable",
    ).head(maximum)
    pool["physical_conservatism_score"] = (
        pool["A_tip_deg"] / 40.0
        + np.abs(pool["kappa"]) / 1.0
        + np.abs(pool["psi_theta_deg"]) / 90.0
        + np.abs(pool["static_twist_offset_deg"]) / 10.0
    )
    physical = pool.loc[
        (
            pool["validation_fx_primary_peak_phase_error_deg"]
            < float(baseline["validation_fx_primary_peak_phase_error_deg"])
        )
        & (
            pool["train_fx_primary_peak_phase_error_deg"]
            < float(baseline["train_fx_primary_peak_phase_error_deg"])
        )
    ].sort_values(
        [
            "physical_conservatism_score",
            "validation_fx_primary_peak_phase_error_deg",
            "validation_fz_rmse",
        ],
        kind="stable",
    ).head(maximum)
    return fx_phase, balanced, physical


def _candidate_from_row(row: pd.Series, *, stage: str, family: str) -> TwistCandidate:
    return TwistCandidate(
        profile_name=str(row["profile_name"]),
        A_tip_deg=float(row["A_tip_deg"]),
        kappa=float(row["kappa"]),
        psi_theta_deg=float(row["psi_theta_deg"]),
        static_twist_offset_deg=float(row["static_twist_offset_deg"]),
        stage=stage,
        family=family,
    )


def _plot_oat(oat: pd.DataFrame, output: Path, *, dpi: int) -> list[Path]:
    validation_fx = oat.loc[
        (oat["partition"] == "validation") & (oat["component"] == "fx")
    ]
    validation_fz = oat.loc[
        (oat["partition"] == "validation") & (oat["component"] == "fz")
    ]
    paths: list[Path] = []
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    for ax, family in zip(axes.ravel(), ("A_tip_deg", "kappa", "psi_theta_deg", "static_twist_offset_deg")):
        table = validation_fx.loc[validation_fx["family"] == family].sort_values(family)
        ax.plot(table[family], table["model_primary_peak_phase_deg"], marker="o")
        ax.plot(table[family], table["data_primary_peak_phase_deg"], color="black", linestyle="--")
        ax.set_xlabel(family)
        ax.set_ylabel("Body Fx peak phase (mechanical deg)")
        ax.set_title(f"OAT: {family}")
        ax.grid(alpha=0.2)
    fig.suptitle("OAT Fx peak sensitivity | validation | current EKF wind | body FRD")
    path = output / "oat_fx_peak_phase_sensitivity.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    for ax, family in zip(axes.ravel(), ("A_tip_deg", "kappa", "psi_theta_deg", "static_twist_offset_deg")):
        fx = validation_fx.loc[validation_fx["family"] == family].sort_values(family)
        fz = validation_fz.loc[validation_fz["family"] == family].sort_values(family)
        ax.plot(fx[family], fx["rmse"], marker="o", label="body Fx RMSE")
        ax.plot(fz[family], fz["rmse"], marker="s", label="body Fz RMSE")
        ax.set_xlabel(family)
        ax.set_ylabel("RMSE (N)")
        ax.set_title(f"OAT: {family}")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("OAT waveform RMSE | validation | 72 mechanical-phase bins | body FRD")
    path = output / "oat_fx_fz_rmse_sensitivity.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    paths.append(path)
    return paths


def _heatmap(
    table: pd.DataFrame,
    *,
    x: str,
    y: str,
    value: str,
    title: str,
    path: Path,
    dpi: int,
) -> Path:
    pivot = table.pivot_table(index=y, columns=x, values=value, aggfunc="mean").sort_index()
    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{value:g}" for value in pivot.columns], rotation=60)
    ax.set_yticks(np.arange(len(pivot.index)), [f"{value:g}" for value in pivot.index])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=value)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _plot_coarse(coarse: pd.DataFrame, output: Path, *, dpi: int) -> list[Path]:
    wide = _wide_metrics(coarse)
    paths: list[Path] = []
    k0 = wide.loc[np.isclose(wide["kappa"], 0.0)]
    paths.append(
        _heatmap(
            k0,
            x="psi_theta_deg",
            y="A_tip_deg",
            value="validation_fx_model_primary_peak_phase_deg",
            title="Body Fx peak phase | validation | kappa=0 | current EKF wind",
            path=output / "heatmap_A_tip_x_psi_fx_peak_phase.png",
            dpi=dpi,
        )
    )
    a0 = wide.loc[np.isclose(wide["A_tip_deg"], 0.0)]
    paths.append(
        _heatmap(
            a0,
            x="psi_theta_deg",
            y="kappa",
            value="validation_fx_model_primary_peak_phase_deg",
            title="Body Fx peak phase | validation | A_tip=0 deg | current EKF wind",
            path=output / "heatmap_kappa_x_psi_fx_peak_phase.png",
            dpi=dpi,
        )
    )
    psi0 = wide.loc[np.isclose(wide["psi_theta_deg"], 0.0)]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.5), constrained_layout=True)
    for ax, value, title in (
        (axes[0], "validation_fx_rmse", "body Fx RMSE"),
        (axes[1], "validation_fz_rmse", "body Fz RMSE"),
    ):
        pivot = psi0.pivot_table(index="kappa", columns="A_tip_deg", values=value).sort_index()
        image = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns)), [f"{item:g}" for item in pivot.columns], rotation=60)
        ax.set_yticks(np.arange(len(pivot.index)), [f"{item:g}" for item in pivot.index])
        ax.set_xlabel("A_tip_deg")
        ax.set_ylabel("kappa")
        ax.set_title(f"{title} | validation | psi=0 deg")
        fig.colorbar(image, ax=ax, label="RMSE (N)")
    path = output / "heatmap_A_tip_x_kappa_fx_fz_error.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    paths.append(path)
    return paths


def _plot_pareto(
    candidates: pd.DataFrame,
    balanced: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
) -> Path:
    wide = _wide_metrics(candidates)
    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
    scatter = ax.scatter(
        wide["validation_fx_primary_peak_phase_error_deg"],
        wide["validation_fz_rmse"],
        c=wide["validation_fx_rmse"],
        s=18,
        alpha=0.55,
        cmap="viridis",
    )
    if not balanced.empty:
        ax.scatter(
            balanced["validation_fx_primary_peak_phase_error_deg"],
            balanced["validation_fz_rmse"],
            facecolors="none",
            edgecolors="#d95f02",
            s=90,
            linewidths=1.5,
            label="balanced Pareto shortlist",
        )
    ax.set_xlabel("Absolute body Fx peak-phase error (mechanical deg)")
    ax.set_ylabel("Body Fz RMSE (N; +Fz down)")
    ax.set_title("Validation Pareto: Fx phase error vs Fz waveform RMSE")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.colorbar(scatter, ax=ax, label="Body Fx RMSE (N)")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _plot_candidate_waveforms(
    resolved: ResolvedExperiment,
    rows: Mapping[str, pd.DataFrame],
    shortlist: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
) -> tuple[Path, dict[str, CandidateEvaluation]]:
    selected = shortlist.head(3)
    evaluations: dict[str, CandidateEvaluation] = {}
    for row in selected.to_dict(orient="records"):
        series = pd.Series(row)
        candidate = _candidate_from_row(
            series,
            stage=str(series["stage"]),
            family=str(series["family"]),
        )
        evaluation = evaluate_candidate(resolved, rows, candidate)
        assert isinstance(evaluation, CandidateEvaluation)
        evaluations[str(series["parameter_hash"])] = evaluation
    baseline_curves = pd.read_csv(resolved.output_root / "baseline_phase_curves.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), sharex=True, constrained_layout=True)
    for column, partition in enumerate(("train", "validation")):
        baseline = baseline_curves.loc[
            baseline_curves["partition"] == partition
        ].sort_values("phase_bin")
        for row_index, component in enumerate(FORCES):
            ax = axes[row_index, column]
            ax.plot(
                baseline["phase_deg"],
                baseline[f"data_{component}"],
                color="#202020",
                linewidth=2.1,
                label="flight data",
            )
            ax.plot(
                baseline["phase_deg"],
                baseline[f"model_{component}"],
                color="#999999",
                linestyle="--",
                linewidth=1.4,
                label="legacy",
            )
            for index, (parameter_hash, evaluation) in enumerate(evaluations.items()):
                curve = evaluation.curves.loc[
                    evaluation.curves["partition"] == partition
                ].sort_values("phase_bin")
                candidate_row = selected.loc[
                    selected["parameter_hash"] == parameter_hash
                ].iloc[0]
                ax.plot(
                    curve["phase_deg"],
                    curve[f"model_{component}"],
                    linewidth=1.5,
                    label=(
                        f"A={candidate_row.A_tip_deg:g} deg, "
                        f"k={candidate_row.kappa:g}, psi={candidate_row.psi_theta_deg:g} deg"
                    ),
                )
            ax.set_title(f"{partition} | body {component.upper()}")
            ax.set_ylabel("Force (N)")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Mechanical phase (deg)")
        ax.set_xticks([0, 90, 180, 270, 360])
    fig.suptitle("Legacy and shortlisted Quadratic-2 waveforms | current EKF wind | body FRD")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path, evaluations


def _twist_shape(eta: np.ndarray, kappa: float) -> np.ndarray:
    shape = (1.0 - float(kappa)) * eta + float(kappa) * eta**2
    if np.any(shape < -1.0e-12):
        raise ValueError("Internal sign reversal in plotted twist shape")
    return shape


def _plot_twist_diagnostics(
    best: pd.Series,
    figure_dir: Path,
    *,
    dpi: int,
) -> list[Path]:
    eta = np.linspace(0.0, 1.0, 201)
    amplitude = math.radians(float(best["A_tip_deg"]))
    kappa = float(best["kappa"])
    psi = math.radians(float(best["psi_theta_deg"]))
    shape = _twist_shape(eta, kappa)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    ax.plot(eta, eta * amplitude * 180.0 / np.pi, color="#777777", linestyle="--", label="legacy linear amplitude")
    ax.plot(
        eta,
        shape * amplitude * 180.0 / np.pi,
        color="#d95f02",
        linewidth=2.0,
        label=f"Quadratic-2, kappa={kappa:g}",
    )
    ax.set_xlabel("Normalized span eta = |y|/R")
    ax.set_ylabel("Dynamic twist amplitude (deg)")
    ax.set_title(f"Spanwise twist amplitude | A_tip={best['A_tip_deg']:g} deg")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    path = figure_dir / "spanwise_twist_amplitude_profile.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    paths.append(path)

    checkpoints = [0.0, 90.0, 180.0, 215.0, 270.0]
    fig, ax = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    for phase_deg in checkpoints:
        internal = math.radians(phase_deg) - 0.5 * np.pi
        delta = -amplitude * shape * np.sin(internal - psi)
        ax.plot(eta, np.degrees(delta), label=f"{phase_deg:g} deg")
    ax.set_xlabel("Normalized span eta = |y|/R")
    ax.set_ylabel("delta theta (deg)")
    ax.set_title("Strip twist distribution at mechanical-phase checkpoints")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=3)
    path = figure_dir / "strip_twist_phase_checkpoints.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    paths.append(path)

    phase_deg = np.linspace(0.0, 360.0, 361)
    internal = np.radians(phase_deg) - 0.5 * np.pi
    delta = -amplitude * shape[:, None] * np.sin(internal[None, :] - psi)
    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    image = ax.imshow(
        np.degrees(delta),
        origin="lower",
        aspect="auto",
        extent=[0.0, 360.0, 0.0, 1.0],
        cmap="RdBu_r",
    )
    ax.set_xlabel("Mechanical phase (deg)")
    ax.set_ylabel("Normalized span eta")
    ax.set_title("Quadratic-2 delta theta | span x mechanical phase")
    fig.colorbar(image, ax=ax, label="delta theta (deg)")
    path = figure_dir / "span_phase_delta_theta_heatmap.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    paths.append(path)
    return paths


def _plot_fx_peak_trajectory(coarse: pd.DataFrame, path: Path, *, dpi: int) -> Path:
    wide = _wide_metrics(coarse)
    ordered = wide.sort_values(
        [
            "psi_theta_deg",
            "validation_fx_primary_peak_phase_error_deg",
            "validation_fz_rmse",
        ],
        kind="stable",
    )
    trajectory = ordered.groupby("psi_theta_deg", as_index=False).first()
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.plot(
        trajectory["psi_theta_deg"],
        trajectory["validation_fx_model_primary_peak_phase_deg"],
        marker="o",
        label="best reachable model peak at each psi",
    )
    ax.plot(
        trajectory["psi_theta_deg"],
        trajectory["validation_fx_data_primary_peak_phase_deg"],
        color="black",
        linestyle="--",
        label="validation data peak",
    )
    ax.set_xlabel("Twist phase offset psi_theta (deg)")
    ax.set_ylabel("Body Fx peak phase (mechanical deg)")
    ax.set_title("Fx peak trajectory under A_tip x kappa optimization")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _zero_wind_shortlist(
    resolved: ResolvedExperiment,
    rows: Mapping[str, pd.DataFrame],
    shortlist: pd.DataFrame,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    seen: set[str] = set()
    for row in shortlist.to_dict(orient="records"):
        original_hash = str(row["parameter_hash"])
        if original_hash in seen:
            continue
        seen.add(original_hash)
        series = pd.Series(row)
        candidate = _candidate_from_row(
            series,
            stage=str(series["stage"]),
            family=str(series["family"]),
        )
        evaluation = evaluate_candidate(
            resolved,
            rows,
            candidate,
            airflow_mode=str(resolved.config["airflow"]["sensitivity_mode"]),
        )
        assert isinstance(evaluation, CandidateEvaluation)
        table = evaluation.metrics.copy()
        table["source_parameter_hash"] = original_hash
        records.append(table)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _plot_wind_sensitivity(
    current_shortlist: pd.DataFrame,
    zero_metrics: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
) -> Path:
    current = current_shortlist[
        [
            "parameter_hash",
            "validation_fx_primary_peak_phase_error_deg",
            "validation_fz_rmse",
        ]
    ].drop_duplicates("parameter_hash")
    zero_wide = _wide_metrics(zero_metrics).rename(
        columns={
            "parameter_hash": "zero_parameter_hash",
            "validation_fx_primary_peak_phase_error_deg": "zero_fx_phase_error_deg",
            "validation_fz_rmse": "zero_fz_rmse",
        }
    )
    zero_wide["parameter_hash"] = zero_metrics[
        ["parameter_hash", "source_parameter_hash"]
    ].drop_duplicates("parameter_hash").set_index("parameter_hash").loc[
        zero_wide["zero_parameter_hash"], "source_parameter_hash"
    ].to_numpy()
    paired = current.merge(
        zero_wide[["parameter_hash", "zero_fx_phase_error_deg", "zero_fz_rmse"]],
        on="parameter_hash",
        validate="one_to_one",
    )
    x = np.arange(len(paired))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    axes[0].scatter(x, paired["validation_fx_primary_peak_phase_error_deg"], label="current EKF wind")
    axes[0].scatter(x, paired["zero_fx_phase_error_deg"], label="zero wind")
    axes[0].set_ylabel("Validation body Fx peak-phase error (deg)")
    axes[1].scatter(x, paired["validation_fz_rmse"], label="current EKF wind")
    axes[1].scatter(x, paired["zero_fz_rmse"], label="zero wind")
    axes[1].set_ylabel("Validation body Fz RMSE (N)")
    for ax in axes:
        ax.set_xlabel("Shortlist candidate index")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Shortlist wind sensitivity | body FRD | identical train/validation rows")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _oat_ranges(oat: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    for family in ("A_tip_deg", "kappa", "psi_theta_deg", "static_twist_offset_deg"):
        result[family] = {}
        for partition in ("train", "validation"):
            table = oat.loc[
                (oat["family"] == family)
                & (oat["partition"] == partition)
                & (oat["component"] == "fx")
            ]
            values = table["model_primary_peak_phase_deg"].to_numpy(dtype=float)
            result[family][partition] = {
                "minimum_deg": float(np.min(values)),
                "maximum_deg": float(np.max(values)),
                "span_deg": float(np.ptp(values)),
            }
    return result


def _conclusion(
    resolved: ResolvedExperiment,
    combined: pd.DataFrame,
    balanced: pd.DataFrame,
    zero_metrics: pd.DataFrame,
) -> tuple[str, str, Mapping[str, object]]:
    wide = _wide_metrics(combined)
    baseline = _baseline_wide(resolved)
    variable_best = wide.sort_values(
        ["validation_fx_primary_peak_phase_error_deg", "validation_fz_rmse"]
    ).iloc[0]
    psi_zero = wide.loc[np.isclose(wide["psi_theta_deg"], 0.0)]
    psi_zero_best = psi_zero.sort_values(
        ["validation_fx_primary_peak_phase_error_deg", "validation_fz_rmse"]
    ).iloc[0]
    phase_gain_from_psi = float(
        psi_zero_best["validation_fx_primary_peak_phase_error_deg"]
        - variable_best["validation_fx_primary_peak_phase_error_deg"]
    )
    reaches = (
        float(variable_best["validation_fx_primary_peak_phase_error_deg"])
        <= 360.0 / int(resolved.config["phase_binning"]["bins"])
    )
    fz_acceptable = float(variable_best["validation_fz_rmse"]) <= float(
        baseline["validation_fz_rmse"]
    ) * (
        1.0
        + float(
            resolved.config["selection"]["maximum_validation_fz_rmse_degradation_fraction"]
        )
    )
    zero_reversal = False
    if not zero_metrics.empty and not balanced.empty:
        zero_wide = _wide_metrics(zero_metrics)
        zero_map = zero_metrics[
            ["parameter_hash", "source_parameter_hash"]
        ].drop_duplicates("parameter_hash")
        zero_wide = zero_wide.merge(zero_map, on="parameter_hash", validate="one_to_one")
        current = balanced[
            ["parameter_hash", "validation_fx_primary_peak_phase_error_deg"]
        ]
        paired = current.merge(
            zero_wide[
                ["source_parameter_hash", "validation_fx_primary_peak_phase_error_deg"]
            ],
            left_on="parameter_hash",
            right_on="source_parameter_hash",
            suffixes=("_current", "_zero"),
        )
        threshold = float(resolved.config["selection"]["maximum_zero_wind_phase_reversal_deg"])
        zero_reversal = bool(
            (
                paired["validation_fx_primary_peak_phase_error_deg_zero"]
                - paired["validation_fx_primary_peak_phase_error_deg_current"]
                > threshold
            ).all()
        )
    if reaches and fz_acceptable and not zero_reversal:
        if phase_gain_from_psi > 5.0:
            code = "B"
            text = (
                "Spanwise distribution can help, but the dominant Fx phase correction "
                "comes from the independent twist-phase offset."
            )
        else:
            code = "A"
            text = "Quadratic-2 spanwise distribution itself is sufficient to materially correct Fx phase."
    elif reaches and (not fz_acceptable or zero_reversal):
        code = "C"
        text = (
            "Twist timing can reach the Fx phase, but the Fz or wind-sensitivity cost "
            "is unacceptable; span-dependent phase or a fuller structural model is needed."
        )
    else:
        code = "D"
        text = (
            "Physically bounded Quadratic-2+ parameters cannot explain the Fx phase gap; "
            "aerodynamic memory, wing bending, or synchronization should be investigated."
        )
    evidence = {
        "best_validation_phase_error_deg": float(
            variable_best["validation_fx_primary_peak_phase_error_deg"]
        ),
        "best_psi_zero_phase_error_deg": float(
            psi_zero_best["validation_fx_primary_peak_phase_error_deg"]
        ),
        "phase_gain_from_variable_psi_deg": phase_gain_from_psi,
        "fz_acceptable": fz_acceptable,
        "zero_wind_complete_reversal": zero_reversal,
        "reaches_data_within_one_bin": reaches,
    }
    return code, text, evidence


def run_report(resolved: ResolvedExperiment) -> dict[str, object]:
    _ensure_output(resolved)
    oat_path = resolved.output_root / "oat_results.csv"
    coarse_path = resolved.output_root / "coarse_grid_results.parquet"
    refined_path = resolved.output_root / "refined_results.parquet"
    for path in (oat_path, coarse_path, refined_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    oat = pd.read_csv(oat_path)
    coarse = pd.read_parquet(coarse_path)
    refined = pd.read_parquet(refined_path)
    combined = pd.concat([coarse, refined], ignore_index=True)
    fx_phase, balanced, physical = build_shortlists(resolved, combined)
    fx_phase.to_csv(resolved.output_root / "shortlist_fx_phase.csv", index=False)
    balanced.to_csv(resolved.output_root / "shortlist_balanced.csv", index=False)
    physical.to_csv(resolved.output_root / "shortlist_physical.csv", index=False)

    figure_dir = resolved.output_root / "figures"
    dpi = int(resolved.config["output"]["figure_dpi"])
    figure_paths = [
        *_plot_oat(oat, figure_dir, dpi=dpi),
        *_plot_coarse(coarse, figure_dir, dpi=dpi),
        _plot_pareto(
            combined,
            balanced,
            figure_dir / "pareto_fx_phase_error_vs_fz_rmse.png",
            dpi=dpi,
        ),
        _plot_fx_peak_trajectory(
            coarse,
            figure_dir / "fx_peak_trajectory.png",
            dpi=dpi,
        ),
    ]
    rows = load_experiment_rows(
        resolved,
        airflow_mode=str(resolved.config["airflow"]["main_mode"]),
    )
    waveform_shortlist = balanced if not balanced.empty else fx_phase
    waveform_path, _ = _plot_candidate_waveforms(
        resolved,
        rows,
        waveform_shortlist,
        figure_dir / "shortlist_train_validation_waveforms.png",
        dpi=dpi,
    )
    figure_paths.append(waveform_path)
    best = waveform_shortlist.iloc[0]
    figure_paths.extend(_plot_twist_diagnostics(best, figure_dir, dpi=dpi))

    zero_input = pd.concat([fx_phase, balanced, physical], ignore_index=True).drop_duplicates(
        "parameter_hash"
    )
    zero_metrics = _zero_wind_shortlist(resolved, rows, zero_input)
    zero_metrics.to_csv(
        resolved.output_root / "zero_wind_shortlist_results.csv", index=False
    )
    wind_path = _plot_wind_sensitivity(
        zero_input,
        zero_metrics,
        figure_dir / "shortlist_current_ekf_vs_zero_wind.png",
        dpi=dpi,
    )
    figure_paths.append(wind_path)

    ranges = _oat_ranges(oat)
    code, conclusion_text, conclusion_evidence = _conclusion(
        resolved,
        combined,
        balanced,
        zero_metrics,
    )
    baseline = _baseline_wide(resolved)
    combined_wide = _wide_metrics(combined)
    best_train = combined_wide.sort_values(
        [
            "train_fx_primary_peak_phase_error_deg",
            "train_fz_rmse",
            "train_fx_rmse",
        ],
        kind="stable",
    ).iloc[0]
    best_validation = combined_wide.sort_values(
        [
            "validation_fx_primary_peak_phase_error_deg",
            "validation_fz_rmse",
            "validation_fx_rmse",
        ],
        kind="stable",
    ).iloc[0]
    zero_wide = _wide_metrics(zero_metrics) if not zero_metrics.empty else pd.DataFrame()
    zero_wind_summary: dict[str, object] = {}
    if not zero_metrics.empty and not balanced.empty:
        zero_map = zero_metrics[
            ["parameter_hash", "source_parameter_hash"]
        ].drop_duplicates("parameter_hash")
        zero_wide = zero_wide.merge(zero_map, on="parameter_hash", validate="one_to_one")
        balanced_best = balanced.iloc[0]
        paired_zero = zero_wide.loc[
            zero_wide["source_parameter_hash"] == balanced_best["parameter_hash"]
        ].iloc[0]
        zero_wind_summary = {
            "balanced_parameter_hash": str(balanced_best["parameter_hash"]),
            "current_validation_fx_phase_error_deg": float(
                balanced_best["validation_fx_primary_peak_phase_error_deg"]
            ),
            "zero_validation_fx_phase_error_deg": float(
                paired_zero["validation_fx_primary_peak_phase_error_deg"]
            ),
            "current_validation_fx_rmse_n": float(balanced_best["validation_fx_rmse"]),
            "zero_validation_fx_rmse_n": float(paired_zero["validation_fx_rmse"]),
            "current_validation_fz_rmse_n": float(balanced_best["validation_fz_rmse"]),
            "zero_validation_fz_rmse_n": float(paired_zero["validation_fz_rmse"]),
        }
    aggregate: dict[str, object] = {
        **_manifest_base(resolved),
        "stage": "report",
        "oat_fx_primary_peak_phase_ranges_deg": ranges,
        "baseline": baseline.to_dict(),
        "best_train_candidate": best_train.to_dict(),
        "best_validation_candidate": best_validation.to_dict(),
        "shortlist_counts": {
            "fx_phase": int(len(fx_phase)),
            "balanced": int(len(balanced)),
            "physical": int(len(physical)),
        },
        "zero_wind_candidate_count": int(
            zero_metrics["source_parameter_hash"].nunique()
            if not zero_metrics.empty
            else 0
        ),
        "zero_wind_balanced_candidate_summary": zero_wind_summary,
        "conclusion_code": code,
        "conclusion_text": conclusion_text,
        "conclusion_evidence": dict(conclusion_evidence),
        "figures": [str(path) for path in figure_paths],
        "test_partition_loaded": False,
        "test_rows_loaded": 0,
    }
    _write_json(resolved.output_root / "aggregate_metrics.json", aggregate)
    manifest = {
        **aggregate,
        "artifacts": {
            path.name: sha256_file(path)
            for path in (
                resolved.output_root / "baseline_manifest.json",
                oat_path,
                coarse_path,
                refined_path,
                resolved.output_root / "shortlist_fx_phase.csv",
                resolved.output_root / "shortlist_balanced.csv",
                resolved.output_root / "shortlist_physical.csv",
                resolved.output_root / "aggregate_metrics.json",
            )
        },
    }
    _write_json(resolved.output_root / "manifest.json", manifest)

    docs_figure_dir = (
        resolved.project_root
        / "docs"
        / "analysis"
        / "figures"
        / "quadratic2_twist_sweep_v1"
    )
    docs_figure_dir.mkdir(parents=True, exist_ok=True)
    key_figures = [
        resolved.output_root / "baseline_train_validation.png",
        figure_dir / "pareto_fx_phase_error_vs_fz_rmse.png",
        figure_dir / "shortlist_train_validation_waveforms.png",
        figure_dir / "shortlist_current_ekf_vs_zero_wind.png",
    ]
    for source in key_figures:
        shutil.copy2(source, docs_figure_dir / source.name)
    docs_result_dir = (
        resolved.project_root
        / "docs"
        / "analysis"
        / "results"
        / "quadratic2_twist_sweep_v1"
    )
    docs_result_dir.mkdir(parents=True, exist_ok=True)
    for source in (
        resolved.output_root / "baseline_manifest.json",
        resolved.output_root / "manifest.json",
        resolved.output_root / "aggregate_metrics.json",
        resolved.output_root / "shortlist_fx_phase.csv",
        resolved.output_root / "shortlist_balanced.csv",
        resolved.output_root / "shortlist_physical.csv",
    ):
        shutil.copy2(source, docs_result_dir / source.name)

    def candidate_text(row: pd.Series) -> str:
        return (
            f"`A_tip={row['A_tip_deg']:.2f} deg`, `kappa={row['kappa']:.3f}`, "
            f"`psi_theta={row['psi_theta_deg']:.2f} deg`, "
            f"`static_offset={row['static_twist_offset_deg']:.2f} deg`"
        )

    report_lines = [
        "# Quadratic-2 spanwise dynamic-twist sweep v1",
        "",
        "## Contract and scope",
        "",
        "This train/validation-only experiment retains canonical body FRD: +Fx forward and +Fz down. "
        "`Fz minimum` is therefore the upward-lift extremum. Mechanical phase is the sole reported "
        "phase coordinate. Test remained sealed.",
        "",
        f"Baseline full-cycle Fx peak: train `{baseline['train_fx_model_full_peak_phase_deg']:.1f} deg`, "
        f"validation `{baseline['validation_fx_model_full_peak_phase_deg']:.1f} deg`. "
        f"Data primary-window peak: train `{baseline['train_fx_data_primary_peak_phase_deg']:.1f} deg`, "
        f"validation `{baseline['validation_fx_data_primary_peak_phase_deg']:.1f} deg`.",
        "",
        "## Direct answers",
        "",
        f"1. **kappa alone:** validation primary Fx peak range "
        f"`{ranges['kappa']['validation']['minimum_deg']:.1f}–"
        f"{ranges['kappa']['validation']['maximum_deg']:.1f} deg` "
        f"(span `{ranges['kappa']['validation']['span_deg']:.1f} deg`).",
        f"2. **A_tip alone:** validation range "
        f"`{ranges['A_tip_deg']['validation']['minimum_deg']:.1f}–"
        f"{ranges['A_tip_deg']['validation']['maximum_deg']:.1f} deg` "
        f"(span `{ranges['A_tip_deg']['validation']['span_deg']:.1f} deg`).",
        f"3. **psi_theta alone at baseline A_tip:** validation range "
        f"`{ranges['psi_theta_deg']['validation']['minimum_deg']:.1f}–"
        f"{ranges['psi_theta_deg']['validation']['maximum_deg']:.1f} deg` "
        f"(span `{ranges['psi_theta_deg']['validation']['span_deg']:.1f} deg`).",
        f"4. **static offset alone:** validation range "
        f"`{ranges['static_twist_offset_deg']['validation']['minimum_deg']:.1f}–"
        f"{ranges['static_twist_offset_deg']['validation']['maximum_deg']:.1f} deg` "
        f"(span `{ranges['static_twist_offset_deg']['validation']['span_deg']:.1f} deg`).",
        "Because the registered baseline has `A_tip=0 deg`, the OAT kappa and psi_theta rows are "
        "structurally inactive; their zero OAT spans are not evidence that they remain insensitive "
        "once a nonzero dynamic twist amplitude is introduced.",
        f"5. **Most sensitive mechanism:** variable twist timing adds "
        f"`{conclusion_evidence['phase_gain_from_variable_psi_deg']:.1f} deg` of best validation "
        "phase-error reduction relative to the best psi=0 candidate.",
        f"6. **Reachability:** best validation phase error is "
        f"`{conclusion_evidence['best_validation_phase_error_deg']:.1f} deg`; "
        f"within-one-bin reachability is `{conclusion_evidence['reaches_data_within_one_bin']}`.",
        f"7. **Fz cost:** the best validation reachability candidate changes validation Fz RMSE "
        f"from `{baseline['validation_fz_rmse']:.3f} N` to "
        f"`{best_validation['validation_fz_rmse']:.3f} N`, and its Fz-minimum amplitude error "
        f"from `{baseline['validation_fz_minimum_amplitude_error_abs']:.3f} N` to "
        f"`{best_validation['validation_fz_minimum_amplitude_error_abs']:.3f} N`; physical gate "
        f"passed=`{conclusion_evidence['fz_acceptable']}`.",
        "8. **Train/validation consistency:** shortlist admission required non-worsening Fx phase "
        "direction on both partitions plus validation Fx/Fz waveform gates.",
        f"9. **Zero-wind check:** complete conclusion reversal is "
        f"`{conclusion_evidence['zero_wind_complete_reversal']}`. For the leading balanced "
        f"candidate, validation Fx phase error changes from "
        f"`{zero_wind_summary.get('current_validation_fx_phase_error_deg', float('nan')):.1f} deg` "
        f"to `{zero_wind_summary.get('zero_validation_fx_phase_error_deg', float('nan')):.1f} deg`.",
        f"10. **Prior decision:** conclusion `{code}`; no shortlist is promoted to the default model.",
        "11. **Next structure:** conclusion B supports retaining Quadratic-2 plus independent twist "
        "phase as an experimental physical prior, not a default replacement. A spanwise phase "
        "gradient or passive-twist ODE is a robustness follow-up; circulation/LEV lag should be "
        "introduced only if the remaining waveform and wind sensitivity cannot be resolved after "
        "synchronization checks.",
        "",
        "## Best candidates",
        "",
        f"Best train reachability candidate: {candidate_text(best_train)}; "
        f"Fx phase error `{best_train['train_fx_primary_peak_phase_error_deg']:.1f} deg`, "
        f"Fx RMSE `{best_train['train_fx_rmse']:.3f} N`, "
        f"Fz RMSE `{best_train['train_fz_rmse']:.3f} N`.",
        "",
        f"Best validation reachability candidate: {candidate_text(best_validation)}; "
        f"Fx phase error `{best_validation['validation_fx_primary_peak_phase_error_deg']:.1f} deg`, "
        f"Fx RMSE `{best_validation['validation_fx_rmse']:.3f} N`, "
        f"Fz RMSE `{best_validation['validation_fz_rmse']:.3f} N`.",
        "",
        "## Final classification",
        "",
        f"**{code}. {conclusion_text}**",
        "",
        "## Key figures",
        "",
        "![Baseline](figures/quadratic2_twist_sweep_v1/baseline_train_validation.png)",
        "",
        "![Pareto](figures/quadratic2_twist_sweep_v1/pareto_fx_phase_error_vs_fz_rmse.png)",
        "",
        "![Shortlist waveforms](figures/quadratic2_twist_sweep_v1/shortlist_train_validation_waveforms.png)",
        "",
        "![Wind sensitivity](figures/quadratic2_twist_sweep_v1/shortlist_current_ekf_vs_zero_wind.png)",
        "",
        "Machine-readable metrics, full figures, compact resume logs, and manifests are under "
        "`outputs/analysis/quadratic2_twist_sweep_v1/`.",
        "Compact committed manifests and shortlist summaries are under "
        "`docs/analysis/results/quadratic2_twist_sweep_v1/`.",
    ]
    report_path = (
        resolved.project_root / "docs" / "analysis" / "quadratic2_twist_sweep_v1.md"
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return aggregate


def dry_run_summary(resolved: ResolvedExperiment) -> dict[str, object]:
    oat = oat_candidates(resolved)
    coarse = coarse_candidates(resolved)
    return {
        "dataset_id": resolved.config["data"]["dataset_id"],
        "dataset_root": str(resolved.dataset_root),
        "partitions": ["train", "validation"],
        "sealed_test": True,
        "test_partition_used": False,
        "oat_parameter_combinations": len(oat),
        "coarse_parameter_combinations": len(coarse),
        "refine_parameter_combinations": "data_dependent_after_coarse_pareto_max_10_seeds",
        "output_root": str(resolved.output_root),
        "wind_mode": resolved.config["airflow"]["main_mode"],
        "zero_wind_sensitivity_mode": resolved.config["airflow"]["sensitivity_mode"],
        "phase_convention": (
            "mechanical_phase_rad: 0 deg horizontal starting upstroke; "
            "90 deg top; 180 deg downstroke through horizontal; 270 deg bottom"
        ),
        "force_convention": FRAME_CONTRACT,
    }
