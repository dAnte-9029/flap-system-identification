"""DeLaurier longitudinal-force discrepancy attribution primitives.

The primary residual in this module is the total reconstructed effective
whole-aircraft force minus a keyed wing-only DeLaurier prior.  It is therefore
an attribution residual, not an isolated wing-aerodynamic error.

All fitting helpers in this module are low-capacity diagnostic probes.  They
return tables only and never serialize deployable models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from system_identification.conventions.phase import compute_wing_stroke_direction, wrap_to_2pi
from system_identification.physics.delaurier.airflow import reconstruct_body_airflow_from_ned


FORCE_COMPONENTS = ("fx_b", "fz_b")
PARTITION_ALIASES = {"train": "train", "validation": "val", "val": "val"}
DEFAULT_ALIGNMENT_KEYS = ("log_id", "timestamp_us")
DEFAULT_PHASE_COLUMN = "mechanical_phase_rad"
LEGACY_PHASE_COLUMN = "phase_corrected_rad"
DEFAULT_FREQUENCY_COLUMN = "cycle_flap_frequency_hz"
DEFAULT_AIRSPEED_COLUMN = "airspeed_validated.true_airspeed_m_s"
DEFAULT_RHO_COLUMN = "vehicle_air_data.rho"
SCHEMA_VERSION = "eda0.force_discrepancy_attribution.v1"


@dataclass(frozen=True)
class AuditConfig:
    """Resolved numerical settings for one attribution audit."""

    phase_bins: int = 72
    minimum_cycle_samples: int = 12
    minimum_phase_coverage_rad: float = 5.5
    maximum_cycle_missing_fraction: float = 0.0
    frequency_range_hz: tuple[float, float] = (0.5, 20.0)
    shift_search_range_rad: float = math.pi
    minimum_shift_variance: float = 1.0e-8
    reversal_band_half_width_rad: float = math.pi / 12.0
    midstroke_band_half_width_rad: float = math.pi / 12.0
    harmonic_max_order: int = 4
    condition_bins: int = 5
    ridge_alphas: tuple[float, ...] = (0.0, 0.1, 1.0, 10.0, 100.0)
    history_lengths: tuple[int, ...] = (1, 2, 4)
    random_seed: int = 20260717
    maximum_missing_alignment_fraction: float = 0.0
    phase_offset_threshold_rad: float = math.radians(10.0)
    phase_component_agreement_threshold_rad: float = math.radians(10.0)
    fixed_delay_improvement_fraction: float = 0.10
    mean_energy_threshold: float = 0.15
    phase_energy_threshold: float = 0.30
    condition_probe_gain_threshold: float = 0.05
    history_probe_gain_threshold: float = 0.05
    prior_gain_threshold: float = 0.02
    label_uncertainty_ratio_threshold: float = 1.5
    physical_similarity_threshold: float = 0.70

    def validate(self) -> None:
        if self.phase_bins < 16:
            raise ValueError("phase_bins must be at least 16")
        if self.minimum_cycle_samples < 4:
            raise ValueError("minimum_cycle_samples must be at least 4")
        if not 0.0 < self.minimum_phase_coverage_rad <= 2.0 * math.pi:
            raise ValueError("minimum_phase_coverage_rad must be in (0, 2*pi]")
        if not 0.0 <= self.maximum_cycle_missing_fraction < 1.0:
            raise ValueError("maximum_cycle_missing_fraction must be in [0, 1)")
        if self.harmonic_max_order < 1:
            raise ValueError("harmonic_max_order must be positive")
        if any(alpha < 0.0 for alpha in self.ridge_alphas):
            raise ValueError("ridge_alphas must be non-negative")
        for name in (
            "fixed_delay_improvement_fraction",
            "mean_energy_threshold",
            "phase_energy_threshold",
            "condition_probe_gain_threshold",
            "history_probe_gain_threshold",
            "prior_gain_threshold",
            "physical_similarity_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass
class AlignmentResult:
    """Keyed alignment output and machine-readable quality evidence."""

    aligned: pd.DataFrame
    report: dict[str, object]
    mismatches: pd.DataFrame


@dataclass
class CycleSelection:
    """Accepted complete-cycle rows plus explicit cycle quality tables."""

    accepted_rows: pd.DataFrame
    quality: pd.DataFrame
    rejections: pd.DataFrame


@dataclass
class RidgeState:
    """In-memory diagnostic ridge state; never written as a model artifact."""

    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of one file without interpreting its contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_partitions(partitions: Sequence[str]) -> tuple[str, ...]:
    """Normalize train/validation aliases while making test unavailable."""

    resolved: list[str] = []
    for raw in partitions:
        name = str(raw).strip().lower()
        if name == "test":
            raise ValueError("EDA0 structure audit must not load the test partition")
        if name not in PARTITION_ALIASES:
            raise ValueError(f"Unsupported partition {raw!r}; expected train and/or validation")
        normalized = PARTITION_ALIASES[name]
        if normalized not in resolved:
            resolved.append(normalized)
    if not resolved:
        raise ValueError("At least one of train or validation is required")
    return tuple(resolved)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _duplicate_examples(frame: pd.DataFrame, keys: Sequence[str], source: str) -> pd.DataFrame:
    duplicates = frame.loc[frame.duplicated(list(keys), keep=False), list(keys)].copy()
    if duplicates.empty:
        return pd.DataFrame(columns=[*keys, "mismatch_type", "source"])
    duplicates["mismatch_type"] = "duplicate_key"
    duplicates["source"] = source
    return duplicates.drop_duplicates().head(1000)


def keyed_align_label_and_prior(
    samples: pd.DataFrame,
    prior: pd.DataFrame,
    *,
    partition: str,
    keys: Sequence[str] = DEFAULT_ALIGNMENT_KEYS,
    maximum_missing_fraction: float = 0.0,
) -> AlignmentResult:
    """Align canonical labels and prior strictly by stable one-to-one keys.

    Row order, DataFrame index, and equal table length are deliberately ignored.
    Duplicate keys always fail. Missing keys are reported and fail when their
    sample-side fraction exceeds ``maximum_missing_fraction``.
    """

    normalized = normalize_partitions((partition,))[0]
    samples = samples.copy()
    if DEFAULT_PHASE_COLUMN not in samples.columns and LEGACY_PHASE_COLUMN in samples.columns:
        # Explicit historical compatibility at the analysis boundary. New
        # canonical artifacts export only mechanical_phase_rad.
        samples[DEFAULT_PHASE_COLUMN] = samples[LEGACY_PHASE_COLUMN]
    key_columns = tuple(keys)
    _require_columns(samples, [*key_columns, *FORCE_COMPONENTS], label="samples")
    _require_columns(prior, [*key_columns, *FORCE_COMPONENTS], label="prior")
    if "split" in samples.columns:
        observed = set(samples["split"].dropna().astype(str).str.lower().unique())
        aliases = {normalized, "validation" if normalized == "val" else normalized}
        if not observed.issubset(aliases):
            raise ValueError(f"Sample partition identity conflict: expected {normalized}, observed {sorted(observed)}")

    sample_duplicates = _duplicate_examples(samples, key_columns, "samples")
    prior_duplicates = _duplicate_examples(prior, key_columns, "prior")
    duplicate_count = len(sample_duplicates) + len(prior_duplicates)
    if duplicate_count:
        examples = pd.concat([sample_duplicates, prior_duplicates], ignore_index=True)
        raise ValueError(f"Duplicate alignment keys detected ({duplicate_count} example rows):\n{examples.head(10)}")

    if "timestamp_us" in key_columns:
        sample_ts = pd.to_numeric(samples["timestamp_us"], errors="coerce")
        prior_ts = pd.to_numeric(prior["timestamp_us"], errors="coerce")
        if sample_ts.isna().any() or prior_ts.isna().any():
            raise ValueError("timestamp_us contains non-numeric values")
        magnitude_ratio = max(abs(float(sample_ts.median())), 1.0) / max(abs(float(prior_ts.median())), 1.0)
        if not 0.5 <= magnitude_ratio <= 2.0:
            raise ValueError(f"timestamp unit mismatch suspected; median magnitude ratio={magnitude_ratio:.6g}")

    sample_keys = samples.loc[:, key_columns].copy()
    prior_keys = prior.loc[:, key_columns].copy()
    key_outer = sample_keys.merge(prior_keys, how="outer", on=list(key_columns), indicator=True, validate="one_to_one")
    mismatches = key_outer.loc[key_outer["_merge"] != "both"].rename(columns={"_merge": "mismatch_type"})
    mismatches["mismatch_type"] = mismatches["mismatch_type"].map(
        {"left_only": "missing_prior", "right_only": "orphan_prior"}
    )
    missing_prior = int((mismatches["mismatch_type"] == "missing_prior").sum())
    orphan_prior = int((mismatches["mismatch_type"] == "orphan_prior").sum())
    missing_fraction = missing_prior / max(len(samples), 1)

    prior_payload = prior.rename(
        columns={component: f"prior_{component}" for component in FORCE_COMPONENTS}
    )
    payload_columns = [*key_columns, *(f"prior_{component}" for component in FORCE_COMPONENTS)]
    for optional in ("dataset_id", "segment_id", "time_s"):
        if optional in prior_payload.columns and optional not in payload_columns and optional not in samples.columns:
            payload_columns.append(optional)
    aligned = samples.merge(
        prior_payload.loc[:, payload_columns],
        how="left",
        on=list(key_columns),
        validate="one_to_one",
        sort=False,
    )
    aligned["partition"] = normalized
    for component in FORCE_COMPONENTS:
        aligned[f"label_{component}"] = pd.to_numeric(aligned[component], errors="coerce")
        aligned[f"residual_{component}"] = aligned[f"label_{component}"] - aligned[f"prior_{component}"]

    report: dict[str, object] = {
        "partition": normalized,
        "alignment_keys": list(key_columns),
        "sample_rows": int(len(samples)),
        "prior_rows": int(len(prior)),
        "aligned_rows": int(aligned[f"prior_{FORCE_COMPONENTS[0]}"].notna().sum()),
        "missing_prior_rows": missing_prior,
        "orphan_prior_rows": orphan_prior,
        "missing_prior_fraction": float(missing_fraction),
        "maximum_missing_fraction": float(maximum_missing_fraction),
        "test_rows_loaded": 0,
        "row_order_used": False,
        "status": "ok" if missing_fraction <= maximum_missing_fraction else "failed_missing_threshold",
    }
    if missing_fraction > maximum_missing_fraction:
        raise ValueError(
            f"Missing prior keys exceed threshold: {missing_prior}/{len(samples)} "
            f"({missing_fraction:.3%}) > {maximum_missing_fraction:.3%}"
        )
    return AlignmentResult(aligned=aligned, report=report, mismatches=mismatches.reset_index(drop=True))


def derive_condition_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Add authoritative slow-condition columns without fitting any data."""

    result = frame.copy()
    sources: dict[str, str] = {}
    airspeed = pd.to_numeric(result[DEFAULT_AIRSPEED_COLUMN], errors="coerce")
    rho = pd.to_numeric(result[DEFAULT_RHO_COLUMN], errors="coerce")
    result["condition_airspeed_m_s"] = airspeed
    result["condition_dynamic_pressure_pa"] = 0.5 * rho * np.square(airspeed)
    sources["condition_airspeed_m_s"] = DEFAULT_AIRSPEED_COLUMN
    sources["condition_dynamic_pressure_pa"] = f"0.5*{DEFAULT_RHO_COLUMN}*{DEFAULT_AIRSPEED_COLUMN}^2"

    if DEFAULT_FREQUENCY_COLUMN in result.columns:
        frequency = pd.to_numeric(result[DEFAULT_FREQUENCY_COLUMN], errors="coerce")
        sources["condition_frequency_hz"] = DEFAULT_FREQUENCY_COLUMN
    else:
        frequency = pd.to_numeric(result["flap_frequency_hz"], errors="coerce")
        sources["condition_frequency_hz"] = "flap_frequency_hz"
    result["condition_frequency_hz"] = frequency

    attitude_columns = [f"vehicle_attitude.q[{index}]" for index in range(4)]
    velocity_columns = ["vehicle_local_position.vx", "vehicle_local_position.vy", "vehicle_local_position.vz"]
    wind_columns = ["wind.windspeed_north", "wind.windspeed_east"]
    if all(column in result.columns for column in [*attitude_columns, *velocity_columns, *wind_columns]):
        quaternion = result[attitude_columns].to_numpy(dtype=float)
        ground = result[velocity_columns].to_numpy(dtype=float)
        wind = np.column_stack(
            (
                result[wind_columns[0]].to_numpy(dtype=float),
                result[wind_columns[1]].to_numpy(dtype=float),
                np.zeros(len(result), dtype=float),
            )
        )
        finite = np.isfinite(quaternion).all(axis=1) & np.isfinite(ground).all(axis=1) & np.isfinite(wind).all(axis=1)
        alpha = np.full(len(result), np.nan, dtype=float)
        if finite.any():
            airflow = reconstruct_body_airflow_from_ned(
                ground_velocity_ned_m_s=ground[finite],
                wind_velocity_ned_m_s=wind[finite],
                quaternion_body_to_ned_wxyz=quaternion[finite],
            )
            alpha[finite] = airflow.alpha_rad
        result["condition_alpha_rad"] = alpha
        sources["condition_alpha_rad"] = "attitude_ground_minus_horizontal_wind_body_FRD_atan2(w,u)"
    elif "alpha_rad" in result.columns:
        result["condition_alpha_rad"] = pd.to_numeric(result["alpha_rad"], errors="coerce")
        sources["condition_alpha_rad"] = "alpha_rad"
    else:
        result["condition_alpha_rad"] = np.nan
        sources["condition_alpha_rad"] = "not_available"

    mean_chord_m = 0.025
    result["condition_reduced_frequency"] = (
        math.pi * result["condition_frequency_hz"] * mean_chord_m / result["condition_airspeed_m_s"].clip(lower=1.0e-6)
    )
    sources["condition_reduced_frequency"] = "pi*f*c_bar/U with c_bar=0.025 m (diagnostic only)"
    return result, sources


def _cycle_group_columns(frame: pd.DataFrame) -> list[str]:
    columns = ["partition", "log_id"]
    if "segment_id" in frame.columns:
        columns.append("segment_id")
    columns.append("cycle_id")
    return columns


def select_complete_cycles(frame: pd.DataFrame, config: AuditConfig) -> CycleSelection:
    """Select complete cycles and retain every rejection reason."""

    config.validate()
    required = [
        "partition",
        "log_id",
        "cycle_id",
        "time_s",
        DEFAULT_PHASE_COLUMN,
        "condition_frequency_hz",
        *(f"label_{component}" for component in FORCE_COMPONENTS),
        *(f"prior_{component}" for component in FORCE_COMPONENTS),
    ]
    _require_columns(frame, required, label="aligned frame")
    group_columns = _cycle_group_columns(frame)
    quality_rows: list[dict[str, object]] = []
    accepted_indices: list[int] = []
    finite_columns = [
        DEFAULT_PHASE_COLUMN,
        "time_s",
        "condition_frequency_hz",
        *(f"label_{component}" for component in FORCE_COMPONENTS),
        *(f"prior_{component}" for component in FORCE_COMPONENTS),
    ]
    for key, group in frame.groupby(group_columns, sort=True, dropna=False):
        ordered = group.sort_values("time_s", kind="stable")
        key_values = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(group_columns, key_values))
        reasons: list[str] = []
        phase = pd.to_numeric(ordered[DEFAULT_PHASE_COLUMN], errors="coerce").to_numpy(dtype=float)
        time_s = pd.to_numeric(ordered["time_s"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(ordered[finite_columns].to_numpy(dtype=float)).all(axis=1)
        missing_fraction = 1.0 - float(finite.mean()) if len(finite) else 1.0
        finite_phase = phase[np.isfinite(phase)]
        phase_unwrapped = np.unwrap(finite_phase) if len(finite_phase) else np.array([], dtype=float)
        phase_coverage = float(np.ptp(phase_unwrapped)) if len(phase_unwrapped) > 1 else 0.0
        frequency = pd.to_numeric(ordered["condition_frequency_hz"], errors="coerce").to_numpy(dtype=float)
        finite_frequency = frequency[np.isfinite(frequency)]
        mean_frequency = float(np.mean(finite_frequency)) if len(finite_frequency) else float("nan")
        duplicate_endpoint = bool(
            len(finite_phase) > 1
            and np.isclose(wrap_to_2pi(finite_phase[0]), wrap_to_2pi(finite_phase[-1]), atol=1.0e-8).item()
        )
        if len(ordered) < config.minimum_cycle_samples:
            reasons.append("too_few_samples")
        if missing_fraction > config.maximum_cycle_missing_fraction:
            reasons.append("missing_fraction")
        if len(time_s) < 2 or not np.all(np.diff(time_s) > 0.0):
            reasons.append("non_monotonic_timestamp")
        if phase_coverage < config.minimum_phase_coverage_rad:
            reasons.append("incomplete_phase_coverage")
        if len(phase_unwrapped) > 1 and np.any(np.diff(phase_unwrapped) < -1.0e-6):
            reasons.append("phase_unwrap_not_monotonic")
        if not math.isfinite(mean_frequency) or not (
            config.frequency_range_hz[0] <= mean_frequency <= config.frequency_range_hz[1]
        ):
            reasons.append("frequency_out_of_range")
        if "cycle_valid" in ordered.columns and not bool(ordered["cycle_valid"].fillna(False).all()):
            reasons.append("source_cycle_invalid")
        accepted = not reasons
        row: dict[str, object] = {
            **identity,
            "sample_count": int(len(ordered)),
            "phase_coverage_rad": phase_coverage,
            "missing_fraction": missing_fraction,
            "mean_frequency_hz": mean_frequency,
            "duplicate_endpoint": duplicate_endpoint,
            "endpoint_action": "removed_last_duplicate" if duplicate_endpoint else "none",
            "accepted": accepted,
            "rejection_reasons": ";".join(reasons),
        }
        quality_rows.append(row)
        if accepted:
            selected_index = ordered.index.tolist()
            if duplicate_endpoint:
                selected_index = selected_index[:-1]
            accepted_indices.extend(selected_index)
    quality = pd.DataFrame(quality_rows)
    rejections = quality.loc[~quality["accepted"]].copy() if not quality.empty else quality.copy()
    accepted_rows = frame.loc[sorted(accepted_indices)].copy()
    return CycleSelection(accepted_rows=accepted_rows, quality=quality, rejections=rejections)


def _periodic_interpolate(phase: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    finite = np.isfinite(phase) & np.isfinite(values)
    if finite.sum() < 3:
        return np.full_like(grid, np.nan, dtype=float)
    wrapped = wrap_to_2pi(phase[finite])
    value = np.asarray(values, dtype=float)[finite]
    order = np.argsort(wrapped, kind="stable")
    wrapped = wrapped[order]
    value = value[order]
    unique, inverse = np.unique(np.round(wrapped, 12), return_inverse=True)
    collapsed = np.array([np.mean(value[inverse == index]) for index in range(len(unique))], dtype=float)
    if len(unique) < 3:
        return np.full_like(grid, np.nan, dtype=float)
    extended_phase = np.r_[unique[-1] - 2.0 * math.pi, unique, unique[0] + 2.0 * math.pi]
    extended_value = np.r_[collapsed[-1], collapsed, collapsed[0]]
    return np.interp(grid, extended_phase, extended_value)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        return float("nan")
    x = np.asarray(a, dtype=float)[finite]
    y = np.asarray(b, dtype=float)[finite]
    if np.var(x) <= 1.0e-15 or np.var(y) <= 1.0e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def estimate_circular_phase_shift(
    label: np.ndarray,
    prior: np.ndarray,
    *,
    phase_grid: np.ndarray | None = None,
    maximum_shift_rad: float = math.pi,
    minimum_variance: float = 1.0e-8,
) -> dict[str, float | str]:
    """Estimate ``corr(label(phi), prior(phi + shift))`` with sub-bin refinement."""

    y = np.asarray(label, dtype=float)
    p = np.asarray(prior, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("label and prior must be one-dimensional arrays with equal shape")
    if phase_grid is None:
        phase_grid = np.linspace(0.0, 2.0 * math.pi, len(y), endpoint=False)
    grid = np.asarray(phase_grid, dtype=float)
    if len(grid) != len(y):
        raise ValueError("phase_grid length must match signals")
    if np.nanvar(y) < minimum_variance or np.nanvar(p) < minimum_variance:
        return {"shift_rad": float("nan"), "max_correlation": float("nan"), "status": "low_variance"}
    step = 2.0 * math.pi / len(grid)
    maximum_bins = int(math.floor(abs(maximum_shift_rad) / step + 1.0e-9))
    shifts = np.arange(-maximum_bins, maximum_bins + 1, dtype=int)
    correlations = np.array([_corr(y, np.roll(p, -shift)) for shift in shifts], dtype=float)
    if not np.isfinite(correlations).any():
        return {"shift_rad": float("nan"), "max_correlation": float("nan"), "status": "invalid"}
    best_position = int(np.nanargmax(correlations))
    best_bin = float(shifts[best_position])
    best_corr = float(correlations[best_position])
    refinement = 0.0
    if 0 < best_position < len(correlations) - 1:
        left, center, right = correlations[best_position - 1 : best_position + 2]
        denominator = left - 2.0 * center + right
        if np.isfinite([left, center, right]).all() and abs(denominator) > 1.0e-12:
            refinement = float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
            best_corr = float(center - 0.25 * (left - right) * refinement)
    return {
        "shift_rad": float((best_bin + refinement) * step),
        "max_correlation": best_corr,
        "status": "ok",
        "grid_step_rad": step,
        "sub_bin_offset": refinement,
    }


def phase_alignment_cycles(cycles: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Estimate per-cycle Fx/Fz label-prior phase alignment."""

    grid = np.linspace(0.0, 2.0 * math.pi, config.phase_bins, endpoint=False)
    rows: list[dict[str, object]] = []
    group_columns = _cycle_group_columns(cycles)
    for key, group in cycles.groupby(group_columns, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(group_columns, key_values))
        phase = group[DEFAULT_PHASE_COLUMN].to_numpy(dtype=float)
        frequency = float(group["condition_frequency_hz"].mean())
        for component in FORCE_COMPONENTS:
            label = _periodic_interpolate(phase, group[f"label_{component}"].to_numpy(dtype=float), grid)
            prior = _periodic_interpolate(phase, group[f"prior_{component}"].to_numpy(dtype=float), grid)
            estimate = estimate_circular_phase_shift(
                label - np.nanmean(label),
                prior - np.nanmean(prior),
                phase_grid=grid,
                maximum_shift_rad=config.shift_search_range_rad,
                minimum_variance=config.minimum_shift_variance,
            )
            shift = float(estimate["shift_rad"])
            rows.append(
                {
                    **identity,
                    "component": component,
                    "frequency_hz": frequency,
                    **estimate,
                    "equivalent_delay_s": shift / (2.0 * math.pi * frequency)
                    if math.isfinite(shift) and frequency > 0.0
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _circular_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan")
    return float(math.atan2(np.mean(np.sin(finite)), np.mean(np.cos(finite))))


def _circular_errors(values: np.ndarray, prediction: np.ndarray | float) -> np.ndarray:
    return np.angle(np.exp(1j * (np.asarray(values, dtype=float) - np.asarray(prediction, dtype=float))))


def summarize_phase_alignment(phase_cycles: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare no-shift, fixed-phase, and fixed-delay hypotheses."""

    by_log_rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    valid = phase_cycles.loc[phase_cycles["status"] == "ok"].copy()
    for (component, log_id), group in valid.groupby(["component", "log_id"], sort=True):
        by_log_rows.append(
            {
                "component": component,
                "log_id": log_id,
                "cycle_count": int(len(group)),
                "circular_mean_shift_rad": _circular_mean(group["shift_rad"].to_numpy(dtype=float)),
                "median_equivalent_delay_s": float(group["equivalent_delay_s"].median()),
                "median_max_correlation": float(group["max_correlation"].median()),
            }
        )
    for component, group in valid.groupby("component", sort=True):
        shift = group["shift_rad"].to_numpy(dtype=float)
        frequency = group["frequency_hz"].to_numpy(dtype=float)
        h0_error = _circular_errors(shift, 0.0)
        phase_offset = _circular_mean(shift)
        h1_error = _circular_errors(shift, phase_offset)
        unwrapped = phase_offset + _circular_errors(shift, phase_offset)
        omega = 2.0 * math.pi * frequency
        finite = np.isfinite(unwrapped) & np.isfinite(omega) & (omega > 0.0)
        tau = float(np.dot(omega[finite], unwrapped[finite]) / np.dot(omega[finite], omega[finite]))
        h2_prediction = omega * tau
        h2_error = _circular_errors(shift, h2_prediction)
        summaries[component] = {
            "cycle_count": int(len(group)),
            "H0_no_shift_rmse_rad": float(np.sqrt(np.mean(np.square(h0_error)))),
            "H1_fixed_phase_offset_rad": phase_offset,
            "H1_rmse_rad": float(np.sqrt(np.mean(np.square(h1_error)))),
            "H2_fixed_delay_s": tau,
            "H2_rmse_rad": float(np.sqrt(np.mean(np.square(h2_error)))),
            "median_equivalent_delay_s": float(np.nanmedian(group["equivalent_delay_s"])),
            "median_max_correlation": float(np.nanmedian(group["max_correlation"])),
            "grid_resolution_rad": float(np.nanmedian(group["grid_step_rad"])),
        }
    return pd.DataFrame(by_log_rows), summaries


def decompose_cycle_residuals(cycles: pd.DataFrame, *, tolerance: float = 1.0e-10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute exact cycle-mean plus zero-mean residual decomposition."""

    sample_output = cycles.copy()
    mean_rows: list[dict[str, object]] = []
    group_columns = _cycle_group_columns(cycles)
    for key, group in cycles.groupby(group_columns, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(group_columns, key_values))
        row: dict[str, object] = {**identity, "sample_count": int(len(group))}
        for component in FORCE_COMPONENTS:
            label = group[f"label_{component}"].to_numpy(dtype=float)
            prior = group[f"prior_{component}"].to_numpy(dtype=float)
            residual = label - prior
            label_mean = float(np.mean(label))
            prior_mean = float(np.mean(prior))
            residual_mean = label_mean - prior_mean
            waveform = (label - label_mean) - (prior - prior_mean)
            reconstructed = residual_mean + waveform
            max_error = float(np.max(np.abs(residual - reconstructed)))
            if max_error > tolerance:
                raise AssertionError(f"Cycle decomposition failed for {identity}, {component}: {max_error}")
            sample_output.loc[group.index, f"cycle_mean_residual_{component}"] = residual_mean
            sample_output.loc[group.index, f"waveform_residual_{component}"] = waveform
            mean_energy = float(len(group) * residual_mean**2)
            waveform_energy = float(np.sum(np.square(waveform)))
            total_energy = float(np.sum(np.square(residual)))
            row.update(
                {
                    f"label_mean_{component}": label_mean,
                    f"prior_mean_{component}": prior_mean,
                    f"mean_residual_{component}": residual_mean,
                    f"mean_energy_{component}": mean_energy,
                    f"waveform_energy_{component}": waveform_energy,
                    f"total_energy_{component}": total_energy,
                    f"mean_energy_fraction_{component}": mean_energy / total_energy if total_energy > 0.0 else float("nan"),
                    f"waveform_energy_fraction_{component}": waveform_energy / total_energy
                    if total_energy > 0.0
                    else float("nan"),
                    f"waveform_rmse_{component}": float(np.sqrt(np.mean(np.square(waveform)))),
                    f"decomposition_max_abs_error_{component}": max_error,
                }
            )
        for condition in (
            "condition_airspeed_m_s",
            "condition_alpha_rad",
            "condition_frequency_hz",
            "condition_dynamic_pressure_pa",
            "condition_reduced_frequency",
        ):
            row[condition] = float(pd.to_numeric(group[condition], errors="coerce").mean())
        mean_rows.append(row)
    return pd.DataFrame(mean_rows), sample_output


def phase_binned_waveform(cycle_samples: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Aggregate zero-mean residual by phase with equal-cycle and equal-log summaries."""

    grid = np.linspace(0.0, 2.0 * math.pi, config.phase_bins, endpoint=False)
    per_cycle_rows: list[dict[str, object]] = []
    group_columns = _cycle_group_columns(cycle_samples)
    for key, group in cycle_samples.groupby(group_columns, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(group_columns, key_values))
        phase = group[DEFAULT_PHASE_COLUMN].to_numpy(dtype=float)
        for component in FORCE_COMPONENTS:
            values = _periodic_interpolate(
                phase,
                group[f"waveform_residual_{component}"].to_numpy(dtype=float),
                grid,
            )
            label = _periodic_interpolate(phase, group[f"label_{component}"].to_numpy(dtype=float), grid)
            prior = _periodic_interpolate(phase, group[f"prior_{component}"].to_numpy(dtype=float), grid)
            for index, phase_value in enumerate(grid):
                per_cycle_rows.append(
                    {
                        **identity,
                        "component": component,
                        "phase_bin": index,
                        "phase_center_rad": phase_value,
                        "label_force": label[index],
                        "prior_force": prior[index],
                        "waveform_residual": values[index],
                    }
                )
    per_cycle = pd.DataFrame(per_cycle_rows)
    if per_cycle.empty:
        return per_cycle
    per_log = (
        per_cycle.groupby(["partition", "log_id", "component", "phase_bin", "phase_center_rad"], as_index=False)
        .agg(
            cycle_count=("cycle_id", "nunique"),
            label_force=("label_force", "mean"),
            prior_force=("prior_force", "mean"),
            waveform_residual=("waveform_residual", "mean"),
            waveform_residual_std=("waveform_residual", "std"),
        )
    )
    macro = (
        per_log.groupby(["partition", "component", "phase_bin", "phase_center_rad"], as_index=False)
        .agg(
            log_count=("log_id", "nunique"),
            label_force_macro=("label_force", "mean"),
            prior_force_macro=("prior_force", "mean"),
            waveform_residual_macro=("waveform_residual", "mean"),
            waveform_residual_log_std=("waveform_residual", "std"),
        )
    )
    macro["waveform_residual_ci95"] = 1.96 * macro["waveform_residual_log_std"] / np.sqrt(
        macro["log_count"].clip(lower=1)
    )
    return per_log.merge(
        macro,
        on=["partition", "component", "phase_bin", "phase_center_rad"],
        how="left",
        validate="many_to_one",
    )


def _circular_distance(phase: np.ndarray, center: float) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (np.asarray(phase, dtype=float) - center))))


def half_stroke_attribution(cycle_samples: pd.DataFrame, config: AuditConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure authoritative upstroke and downstroke residual structure.

    With ``q=A sin(phi)``, upstroke is ``cos(phi)>0`` and therefore crosses
    the phase wrap: ``[3*pi/2,2*pi) U [0,pi/2)``.  Downstroke occupies
    ``[pi/2,3*pi/2)``.  This deliberately avoids calling ``[0,pi)`` a physical
    half-stroke.
    """

    rows: list[dict[str, object]] = []
    group_columns = _cycle_group_columns(cycle_samples)
    for key, group in cycle_samples.groupby(group_columns, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(group_columns, key_values))
        phase = wrap_to_2pi(group[DEFAULT_PHASE_COLUMN].to_numpy(dtype=float))
        computed_direction = compute_wing_stroke_direction(phase, 0.0)
        for component in FORCE_COMPONENTS:
            residual = group[f"residual_{component}"].to_numpy(dtype=float)
            waveform = group[f"waveform_residual_{component}"].to_numpy(dtype=float)
            for half_name, expected_direction in (("upstroke", "upstroke"), ("downstroke", "downstroke")):
                mask = computed_direction == expected_direction
                if not mask.any():
                    continue
                peak_local = int(np.nanargmax(np.abs(waveform[mask])))
                phase_half = phase[mask]
                waveform_half = waveform[mask]
                if half_name == "upstroke":
                    integration_phase = np.where(phase_half < math.pi / 2.0, phase_half + 2.0 * math.pi, phase_half)
                else:
                    integration_phase = phase_half.copy()
                order = np.argsort(integration_phase, kind="stable")
                integration_phase = integration_phase[order]
                integration_waveform = waveform_half[order]
                direction_match = float(np.mean(computed_direction[mask] == expected_direction))
                reversal_mask = mask & (
                    (_circular_distance(phase, math.pi / 2.0) <= config.reversal_band_half_width_rad)
                    | (_circular_distance(phase, 3.0 * math.pi / 2.0) <= config.reversal_band_half_width_rad)
                )
                midstroke_mask = mask & (
                    (_circular_distance(phase, 0.0) <= config.midstroke_band_half_width_rad)
                    | (_circular_distance(phase, math.pi) <= config.midstroke_band_half_width_rad)
                )
                rows.append(
                    {
                        **identity,
                        "component": component,
                        "half_stroke": half_name,
                        "wing_direction": expected_direction,
                        "phase_interval": (
                            "[3pi/2,2pi) U [0,pi/2)" if half_name == "upstroke" else "[pi/2,3pi/2)"
                        ),
                        "direction_contract_match_fraction": direction_match,
                        "sample_count": int(mask.sum()),
                        "mean_residual": float(np.mean(residual[mask])),
                        "mean_waveform_residual": float(np.mean(waveform_half)),
                        "integral_waveform_residual_rad": float(
                            np.trapz(integration_waveform, integration_phase)
                        ),
                        "peak_abs_residual": float(waveform_half[peak_local]),
                        "peak_phase_rad": float(phase_half[peak_local]),
                        "positive_area": float(
                            np.trapz(np.maximum(integration_waveform, 0.0), integration_phase)
                        ),
                        "negative_area": float(
                            np.trapz(np.minimum(integration_waveform, 0.0), integration_phase)
                        ),
                        "reversal_mean_abs_residual": float(np.mean(np.abs(waveform[reversal_mask])))
                        if reversal_mask.any()
                        else float("nan"),
                        "midstroke_mean_abs_residual": float(np.mean(np.abs(waveform[midstroke_mask])))
                        if midstroke_mask.any()
                        else float("nan"),
                    }
                )
    per_cycle = pd.DataFrame(rows)
    if per_cycle.empty:
        return per_cycle, per_cycle
    wide = per_cycle.pivot_table(
        index=[*group_columns, "component"],
        columns="half_stroke",
        values="integral_waveform_residual_rad",
    )
    if {"upstroke", "downstroke"}.issubset(wide.columns):
        denominator = np.abs(wide["upstroke"]) + np.abs(wide["downstroke"])
        asymmetry = (wide["upstroke"] - wide["downstroke"]) / denominator.replace(0.0, np.nan)
        asymmetry.name = "asymmetry_index"
        per_cycle = per_cycle.merge(asymmetry.reset_index(), on=[*group_columns, "component"], how="left")
    by_log = (
        per_cycle.groupby(["partition", "log_id", "component", "half_stroke", "wing_direction"], as_index=False)
        .agg(
            cycle_count=("cycle_id", "nunique"),
            mean_residual=("mean_residual", "mean"),
            mean_waveform_residual=("mean_waveform_residual", "mean"),
            integral_waveform_residual_rad=("integral_waveform_residual_rad", "mean"),
            peak_abs_residual=("peak_abs_residual", lambda values: float(np.mean(np.abs(values)))),
            peak_phase_rad=("peak_phase_rad", _circular_mean),
            reversal_mean_abs_residual=("reversal_mean_abs_residual", "mean"),
            midstroke_mean_abs_residual=("midstroke_mean_abs_residual", "mean"),
            asymmetry_index=("asymmetry_index", "mean"),
        )
    )
    return per_cycle, by_log


def harmonic_cycle_summary(cycle_samples: pd.DataFrame, config: AuditConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute complex Fourier coefficients k=1..K on a common phase grid."""

    grid = np.linspace(0.0, 2.0 * math.pi, config.phase_bins, endpoint=False)
    rows: list[dict[str, object]] = []
    group_columns = _cycle_group_columns(cycle_samples)
    for key, group in cycle_samples.groupby(group_columns, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(group_columns, key_values))
        phase = group[DEFAULT_PHASE_COLUMN].to_numpy(dtype=float)
        for component in FORCE_COMPONENTS:
            values = _periodic_interpolate(
                phase,
                group[f"waveform_residual_{component}"].to_numpy(dtype=float),
                grid,
            )
            total_energy = float(np.mean(np.square(values)))
            cumulative = 0.0
            for order in range(1, config.harmonic_max_order + 1):
                coefficient = 2.0 / len(grid) * np.sum(values * np.exp(-1j * order * grid))
                amplitude = float(abs(coefficient))
                harmonic_energy = 0.5 * amplitude**2
                cumulative += harmonic_energy
                rows.append(
                    {
                        **identity,
                        "component": component,
                        "harmonic_order": order,
                        "coefficient_real": float(np.real(coefficient)),
                        "coefficient_imag": float(np.imag(coefficient)),
                        "amplitude": amplitude,
                        "phase_rad": float(np.angle(coefficient)),
                        "harmonic_energy": harmonic_energy,
                        "total_waveform_energy": total_energy,
                        "cumulative_energy_coverage": min(cumulative / total_energy, 1.0)
                        if total_energy > 0.0
                        else float("nan"),
                    }
                )
    cycles_table = pd.DataFrame(rows)
    if cycles_table.empty:
        return cycles_table, cycles_table
    by_log = (
        cycles_table.groupby(["partition", "log_id", "component", "harmonic_order"], as_index=False)
        .agg(
            cycle_count=("cycle_id", "nunique"),
            amplitude_mean=("amplitude", "mean"),
            amplitude_std=("amplitude", "std"),
            phase_mean_rad=("phase_rad", _circular_mean),
            cumulative_energy_coverage_mean=("cumulative_energy_coverage", "mean"),
        )
    )
    return cycles_table, by_log


def _date_from_log_id(log_id: object) -> str:
    match = re.search(r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})", str(log_id))
    if not match:
        return "unknown"
    year, month, day = (int(value) for value in match.groups())
    return f"{year:04d}-{month:02d}-{day:02d}"


def waveform_repeatability(
    cycle_samples: pd.DataFrame,
    phase_waveform: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Quantify within-log, cross-log, and cross-date waveform repeatability."""

    per_cycle_rows: list[dict[str, object]] = []
    group_columns = _cycle_group_columns(cycle_samples)
    for key, group in cycle_samples.groupby(group_columns, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(group_columns, key_values))
        for component in FORCE_COMPONENTS:
            per_cycle_rows.append(
                {
                    **identity,
                    "date": _date_from_log_id(identity["log_id"]),
                    "component": component,
                    "waveform_variance": float(np.var(group[f"waveform_residual_{component}"].to_numpy(dtype=float))),
                    "waveform_rms": float(np.sqrt(np.mean(np.square(group[f"waveform_residual_{component}"].to_numpy(dtype=float))))),
                }
            )
    cycle_stats = pd.DataFrame(per_cycle_rows)
    waveform_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    log_conditions = (
        cycle_samples.groupby(["partition", "log_id"], as_index=False)[
            ["condition_airspeed_m_s", "condition_alpha_rad", "condition_frequency_hz"]
        ].mean()
    )
    for (partition, component), group in phase_waveform.groupby(["partition", "component"], sort=True):
        pivot = group.pivot_table(index="log_id", columns="phase_bin", values="waveform_residual", aggfunc="first")
        logs = pivot.index.astype(str).tolist()
        values = pivot.to_numpy(dtype=float)
        macro = np.nanmean(values, axis=0)
        for index, log_id in enumerate(logs):
            waveform_rows.append(
                {
                    "partition": partition,
                    "log_id": log_id,
                    "date": _date_from_log_id(log_id),
                    "component": component,
                    "correlation_with_macro": _corr(values[index], macro),
                    "waveform_rms": float(np.sqrt(np.nanmean(np.square(values[index])))),
                    "between_log_variance": float(np.nanmean(np.nanvar(values, axis=0))),
                }
            )
        condition_lookup = log_conditions.set_index("log_id")
        scale = condition_lookup[["condition_airspeed_m_s", "condition_alpha_rad", "condition_frequency_hz"]].std().replace(0.0, 1.0)
        for left_index, left_log in enumerate(logs):
            for right_index, right_log in enumerate(logs):
                left_condition = condition_lookup.loc[left_log] if left_log in condition_lookup.index else None
                right_condition = condition_lookup.loc[right_log] if right_log in condition_lookup.index else None
                if left_condition is None or right_condition is None:
                    distance = float("nan")
                else:
                    delta = (
                        left_condition[["condition_airspeed_m_s", "condition_alpha_rad", "condition_frequency_hz"]]
                        - right_condition[["condition_airspeed_m_s", "condition_alpha_rad", "condition_frequency_hz"]]
                    ) / scale
                    distance = float(np.sqrt(np.nansum(np.square(delta.to_numpy(dtype=float)))))
                correlation_rows.append(
                    {
                        "partition": partition,
                        "component": component,
                        "log_id_left": left_log,
                        "log_id_right": right_log,
                        "same_date": _date_from_log_id(left_log) == _date_from_log_id(right_log),
                        "condition_distance": distance,
                        "waveform_correlation": _corr(values[left_index], values[right_index]),
                    }
                )
    repeatability = pd.DataFrame(waveform_rows)
    correlations = pd.DataFrame(correlation_rows)
    if cycle_stats.empty:
        date_summary = cycle_stats
    else:
        date_summary = (
            cycle_stats.groupby(["partition", "date", "component"], as_index=False)
            .agg(
                log_count=("log_id", "nunique"),
                cycle_count=("cycle_id", "count"),
                waveform_rms_macro=("waveform_rms", "mean"),
                within_date_cycle_variance=("waveform_variance", "mean"),
            )
        )
    return repeatability, correlations, date_summary


def condition_dependence_table(
    cycle_means: pd.DataFrame,
    half_strokes: pd.DataFrame,
    harmonics: pd.DataFrame,
    phase_alignment: pd.DataFrame,
    *,
    bins: int,
) -> pd.DataFrame:
    """Build long pooled/per-log/macro condition relationships from cycle summaries."""

    base = cycle_means.copy()
    half = (
        half_strokes.groupby([*_cycle_group_columns_from_table(half_strokes), "component"], as_index=False)
        .agg(half_stroke_abs_integral=("integral_waveform_residual_rad", lambda x: float(np.mean(np.abs(x)))))
        if not half_strokes.empty
        else pd.DataFrame()
    )
    harmonic = (
        harmonics.loc[harmonics["harmonic_order"] == 1, [*_cycle_group_columns_from_table(harmonics), "component", "amplitude", "phase_rad"]]
        .rename(columns={"amplitude": "harmonic1_amplitude", "phase_rad": "harmonic1_phase_rad"})
        if not harmonics.empty
        else pd.DataFrame()
    )
    shift = (
        phase_alignment.loc[phase_alignment["status"] == "ok", [*_cycle_group_columns_from_table(phase_alignment), "component", "shift_rad"]]
        if not phase_alignment.empty
        else pd.DataFrame()
    )
    rows: list[dict[str, object]] = []
    cycle_keys = _cycle_group_columns_from_table(base)
    for component in FORCE_COMPONENTS:
        table = base.copy()
        table["component"] = component
        table["cycle_mean_residual"] = table[f"mean_residual_{component}"]
        table["residual_rms"] = np.sqrt(table[f"total_energy_{component}"] / table["sample_count"].clip(lower=1))
        if not half.empty:
            table = table.merge(half, on=[*cycle_keys, "component"], how="left", validate="one_to_one")
        if not harmonic.empty:
            table = table.merge(harmonic, on=[*cycle_keys, "component"], how="left", validate="one_to_one")
        if not shift.empty:
            table = table.merge(shift, on=[*cycle_keys, "component"], how="left", validate="one_to_one")
        for condition in (
            "condition_airspeed_m_s",
            "condition_alpha_rad",
            "condition_frequency_hz",
            "condition_dynamic_pressure_pa",
        ):
            for summary in (
                "cycle_mean_residual",
                "residual_rms",
                "half_stroke_abs_integral",
                "harmonic1_amplitude",
                "harmonic1_phase_rad",
                "shift_rad",
            ):
                if summary not in table.columns:
                    continue
                finite = np.isfinite(table[condition]) & np.isfinite(table[summary])
                subset = table.loc[finite].copy()
                if len(subset) < 3:
                    continue
                pooled_corr = _corr(subset[condition].to_numpy(dtype=float), subset[summary].to_numpy(dtype=float))
                log_corrs = [
                    _corr(group[condition].to_numpy(dtype=float), group[summary].to_numpy(dtype=float))
                    for _, group in subset.groupby("log_id")
                    if len(group) >= 3
                ]
                rows.append(
                    {
                        "partition": "all",
                        "component": component,
                        "condition": condition,
                        "summary": summary,
                        "aggregation": "relationship",
                        "sample_count": int(len(subset)),
                        "log_count": int(subset["log_id"].nunique()),
                        "pooled_correlation": pooled_corr,
                        "per_log_macro_correlation": float(np.nanmean(log_corrs)) if log_corrs else float("nan"),
                        "simpson_sign_reversal": bool(log_corrs and np.sign(pooled_corr) != np.sign(np.nanmean(log_corrs))),
                    }
                )
                try:
                    subset["condition_bin"] = pd.qcut(subset[condition], q=min(bins, subset[condition].nunique()), duplicates="drop")
                except ValueError:
                    continue
                for (partition, condition_bin), group in subset.groupby(["partition", "condition_bin"], observed=True):
                    per_log = group.groupby("log_id")[summary].mean()
                    rows.append(
                        {
                            "partition": partition,
                            "component": component,
                            "condition": condition,
                            "summary": summary,
                            "aggregation": "binned_equal_log_macro",
                            "condition_bin": str(condition_bin),
                            "condition_center": float(group[condition].median()),
                            "sample_count": int(len(group)),
                            "log_count": int(group["log_id"].nunique()),
                            "summary_value": float(per_log.mean()),
                            "summary_log_std": float(per_log.std()),
                        }
                    )
    return pd.DataFrame(rows)


def _cycle_group_columns_from_table(frame: pd.DataFrame) -> list[str]:
    return [column for column in ("partition", "log_id", "segment_id", "cycle_id") if column in frame.columns]


def _phase_features(
    frame: pd.DataFrame,
    order: int,
    *,
    conditions: bool | Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    phase = frame[DEFAULT_PHASE_COLUMN].to_numpy(dtype=float)
    columns: list[np.ndarray] = []
    names: list[str] = []
    for harmonic in range(1, order + 1):
        columns.extend([np.sin(harmonic * phase), np.cos(harmonic * phase)])
        names.extend([f"sin_{harmonic}", f"cos_{harmonic}"])
    condition_columns = (
        ("condition_airspeed_m_s", "condition_alpha_rad", "condition_frequency_hz")
        if conditions is True
        else tuple(conditions)
        if conditions
        else ()
    )
    for name in condition_columns:
        value = frame[name].to_numpy(dtype=float)
        columns.append(value)
        names.append(name)
        for harmonic in range(1, order + 1):
            columns.extend([value * np.sin(harmonic * phase), value * np.cos(harmonic * phase)])
            names.extend([f"{name}_sin_{harmonic}", f"{name}_cos_{harmonic}"])
    return np.column_stack(columns), names


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float, weights: np.ndarray | None = None) -> RidgeState:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    finite_y = np.isfinite(y_values)
    median = np.zeros(x_values.shape[1], dtype=float)
    for column in range(x_values.shape[1]):
        finite_column = x_values[finite_y, column]
        finite_column = finite_column[np.isfinite(finite_column)]
        if len(finite_column):
            median[column] = float(np.median(finite_column))
    filled = np.where(np.isfinite(x_values), x_values, median)
    mean = np.mean(filled[finite_y], axis=0)
    scale = np.std(filled[finite_y], axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    standardized = (filled - mean) / scale
    if weights is None:
        w = np.ones(len(y_values), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    finite = finite_y & np.isfinite(w) & (w > 0.0)
    root_w = np.sqrt(w[finite] / np.mean(w[finite]))
    design = standardized[finite] * root_w[:, None]
    target = y_values[finite] * root_w
    design_augmented = np.column_stack([np.ones(len(design)), design])
    penalty = np.eye(design_augmented.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(design_augmented.T @ design_augmented + penalty) @ design_augmented.T @ target
    return RidgeState(median=median, mean=mean, scale=scale, coef=beta[1:], intercept=float(beta[0]))


def _predict_ridge(state: RidgeState, x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    filled = np.where(np.isfinite(values), values, state.median)
    standardized = (filled - state.mean) / state.scale
    return state.intercept + standardized @ state.coef


def _rmse(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(np.square(finite)))) if len(finite) else float("nan")


def equal_cycle_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    """Weight each log equally, then each cycle and sample equally within it."""

    keys = _cycle_group_columns(frame)
    cycle_counts = frame.groupby("log_id")["cycle_id"].transform("nunique").to_numpy(dtype=float)
    sample_counts = frame.groupby(keys)["cycle_id"].transform("size").to_numpy(dtype=float)
    weights = 1.0 / np.maximum(cycle_counts * sample_counts, 1.0)
    return weights / np.mean(weights)


def macro_log_rmse(frame: pd.DataFrame, errors: np.ndarray) -> float:
    """Return an equal-log macro RMSE, independent of log sample count."""

    temp = pd.DataFrame({"log_id": frame["log_id"].to_numpy(), "error": errors})
    return float(temp.groupby("log_id")["error"].apply(lambda x: _rmse(x.to_numpy(dtype=float))).mean())


def diagnostic_static_probes(cycle_samples: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Compare phase-only and phase-conditioned low-capacity probes."""

    if not {"train", "val"}.issubset(set(cycle_samples["partition"].unique())):
        return pd.DataFrame(
            [{"status": "not_available_requires_train_and_validation", "probe": "static"}]
        )
    train = cycle_samples.loc[cycle_samples["partition"] == "train"].copy()
    validation = cycle_samples.loc[cycle_samples["partition"] == "val"].copy()
    rows: list[dict[str, object]] = []
    for component in FORCE_COMPONENTS:
        target_train = train[f"residual_{component}"].to_numpy(dtype=float)
        target_validation = validation[f"residual_{component}"].to_numpy(dtype=float)
        probe_conditions: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("phase_only", ()),
            ("phase_plus_airspeed", ("condition_airspeed_m_s",)),
            ("phase_plus_angle_of_attack", ("condition_alpha_rad",)),
            ("phase_plus_flapping_frequency", ("condition_frequency_hz",)),
            (
                "phase_conditioned",
                ("condition_airspeed_m_s", "condition_alpha_rad", "condition_frequency_hz"),
            ),
        )
        for probe_name, condition_columns in probe_conditions:
            x_train, names = _phase_features(train, config.harmonic_max_order, conditions=condition_columns)
            x_validation, _ = _phase_features(validation, config.harmonic_max_order, conditions=condition_columns)
            weights = equal_cycle_sample_weights(train)
            for alpha in config.ridge_alphas:
                state = _fit_ridge(x_train, target_train, alpha, weights)
                train_error = _predict_ridge(state, x_train) - target_train
                validation_error = _predict_ridge(state, x_validation) - target_validation
                rows.append(
                    {
                        "status": "ok",
                        "probe": probe_name,
                        "component": component,
                        "alpha": alpha,
                        "feature_count": len(names),
                        "train_equal_log_macro_rmse": macro_log_rmse(train, train_error),
                        "validation_equal_log_macro_rmse": macro_log_rmse(validation, validation_error),
                        "validation_pooled_rmse": _rmse(validation_error),
                    }
                )
    return pd.DataFrame(rows)


def _lagged_phase_features(frame: pd.DataFrame, order: int, lag: int) -> tuple[np.ndarray, np.ndarray]:
    phase = frame[DEFAULT_PHASE_COLUMN].to_numpy(dtype=float)
    current, _ = _phase_features(frame, order, conditions=True)
    lag_state = np.column_stack(
        [
            np.sin(phase),
            np.cos(phase),
            frame["condition_airspeed_m_s"].to_numpy(dtype=float),
            frame["condition_alpha_rad"].to_numpy(dtype=float),
            frame["condition_frequency_hz"].to_numpy(dtype=float),
        ]
    )
    blocks = [current]
    valid = np.ones(len(frame), dtype=bool)
    group_columns = [column for column in ("partition", "log_id", "segment_id") if column in frame.columns]
    group_keys = [pd.Series(frame[column].to_numpy(), copy=False) for column in group_columns]
    for step in range(1, lag + 1):
        shifted = pd.DataFrame(lag_state).groupby(group_keys, sort=False).shift(step).to_numpy(dtype=float)
        blocks.append(shifted)
        valid &= np.isfinite(shifted).all(axis=1)
    return np.column_stack(blocks), valid


def diagnostic_history_probes(cycle_samples: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Compare static and a few predefined short-history linear probes."""

    if not {"train", "val"}.issubset(set(cycle_samples["partition"].unique())):
        return pd.DataFrame([{"status": "not_available_requires_train_and_validation", "probe": "history"}])
    train = cycle_samples.loc[cycle_samples["partition"] == "train"].sort_values(
        ["log_id", "segment_id", "time_s"], kind="stable"
    )
    validation = cycle_samples.loc[cycle_samples["partition"] == "val"].sort_values(
        ["log_id", "segment_id", "time_s"], kind="stable"
    )
    rows: list[dict[str, object]] = []
    for lag in (0, *config.history_lengths):
        x_train, valid_train = _lagged_phase_features(train, config.harmonic_max_order, lag)
        x_validation, valid_validation = _lagged_phase_features(validation, config.harmonic_max_order, lag)
        for component in FORCE_COMPONENTS:
            y_train = train[f"residual_{component}"].to_numpy(dtype=float)
            y_validation = validation[f"residual_{component}"].to_numpy(dtype=float)
            for alpha in config.ridge_alphas:
                state = _fit_ridge(
                    x_train[valid_train],
                    y_train[valid_train],
                    alpha,
                    equal_cycle_sample_weights(train.iloc[np.flatnonzero(valid_train)]),
                )
                validation_error = _predict_ridge(state, x_validation[valid_validation]) - y_validation[valid_validation]
                validation_frame = validation.iloc[np.flatnonzero(valid_validation)]
                rows.append(
                    {
                        "status": "ok",
                        "probe": "static" if lag == 0 else "short_history",
                        "history_samples": lag,
                        "component": component,
                        "alpha": alpha,
                        "validation_equal_log_macro_rmse": macro_log_rmse(validation_frame, validation_error),
                        "validation_pooled_rmse": _rmse(validation_error),
                        "validation_sample_count": int(valid_validation.sum()),
                    }
                )
    return pd.DataFrame(rows)


def matched_capacity_prior_probe(cycle_samples: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Compare prior-plus-residual and no-prior at identical diagnostic capacity."""

    if not {"train", "val"}.issubset(set(cycle_samples["partition"].unique())):
        return pd.DataFrame([{"status": "not_available_requires_train_and_validation"}])
    train = cycle_samples.loc[cycle_samples["partition"] == "train"].copy()
    validation = cycle_samples.loc[cycle_samples["partition"] == "val"].copy()
    x_train, names = _phase_features(train, config.harmonic_max_order, conditions=True)
    x_validation, _ = _phase_features(validation, config.harmonic_max_order, conditions=True)
    weights = equal_cycle_sample_weights(train)
    rows: list[dict[str, object]] = []
    for component in FORCE_COMPONENTS:
        label_train = train[f"label_{component}"].to_numpy(dtype=float)
        label_validation = validation[f"label_{component}"].to_numpy(dtype=float)
        prior_train = train[f"prior_{component}"].to_numpy(dtype=float)
        prior_validation = validation[f"prior_{component}"].to_numpy(dtype=float)
        for alpha in config.ridge_alphas:
            for model_name, target_train, base_validation in (
                ("prior_plus_delta", label_train - prior_train, prior_validation),
                ("no_prior", label_train, np.zeros(len(validation), dtype=float)),
            ):
                state = _fit_ridge(x_train, target_train, alpha, weights)
                prediction = base_validation + _predict_ridge(state, x_validation)
                error = prediction - label_validation
                validation_table = validation.assign(_prediction=prediction)
                cycle_error = validation_table.groupby(_cycle_group_columns(validation_table)).apply(
                    lambda group: float(
                        np.mean(
                            group["_prediction"].to_numpy(dtype=float)
                            - group[f"label_{component}"].to_numpy(dtype=float)
                        )
                    ),
                    include_groups=False,
                )
                rows.append(
                    {
                        "status": "ok",
                        "model": model_name,
                        "component": component,
                        "harmonic_order": config.harmonic_max_order,
                        "condition_features": "U,alpha,f with harmonic interactions",
                        "feature_count": len(names),
                        "alpha": alpha,
                        "validation_equal_log_macro_rmse": macro_log_rmse(validation, error),
                        "validation_pooled_rmse": _rmse(error),
                        "validation_cycle_mean_rmse": _rmse(cycle_error.to_numpy(dtype=float)),
                    }
                )
    return pd.DataFrame(rows)


def label_robustness(
    primary: pd.DataFrame,
    variants: Mapping[str, pd.DataFrame],
    config: AuditConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare existing keyed label variants without creating new labels."""

    if not variants:
        unavailable = pd.DataFrame([{"status": "not_available_no_label_variants"}])
        return unavailable, unavailable.copy()
    keys = [column for column in (*DEFAULT_ALIGNMENT_KEYS, "partition") if column in primary.columns]
    combined = primary.loc[
        :,
        [
            *keys,
            DEFAULT_PHASE_COLUMN,
            *(f"label_{c}" for c in FORCE_COMPONENTS),
            *(f"prior_{c}" for c in FORCE_COMPONENTS),
        ],
    ].copy()
    variant_rows: list[dict[str, object]] = []
    for name, variant in variants.items():
        _require_columns(variant, [*keys, *FORCE_COMPONENTS], label=f"label variant {name}")
        if variant.duplicated(keys).any():
            raise ValueError(f"Label variant {name} has duplicate keys")
        renamed = variant.loc[:, [*keys, *FORCE_COMPONENTS]].rename(
            columns={component: f"variant_{name}_{component}" for component in FORCE_COMPONENTS}
        )
        combined = combined.merge(renamed, on=keys, how="left", validate="one_to_one")
        for component in FORCE_COMPONENTS:
            difference = combined[f"variant_{name}_{component}"] - combined[f"label_{component}"]
            variant_rows.append(
                {
                    "status": "ok",
                    "variant": name,
                    "component": component,
                    "aligned_rows": int(difference.notna().sum()),
                    "missing_rows": int(difference.isna().sum()),
                    "label_variant_rmse": _rmse(difference.to_numpy(dtype=float)),
                    "label_variant_bias": float(difference.mean()),
                }
            )
    phase = wrap_to_2pi(combined[DEFAULT_PHASE_COLUMN].to_numpy(dtype=float))
    combined["phase_bin"] = np.floor(phase / (2.0 * math.pi) * config.phase_bins).astype(int).clip(0, config.phase_bins - 1)
    phase_grid = (np.arange(config.phase_bins, dtype=float) + 0.5) * 2.0 * math.pi / config.phase_bins
    for row in variant_rows:
        name = str(row["variant"])
        component = str(row["component"])
        variant_column = f"variant_{name}_{component}"
        per_log = (
            combined.groupby(["partition", "log_id", "phase_bin"], as_index=False)
            .agg(
                nominal_label=(f"label_{component}", "mean"),
                variant_label=(variant_column, "mean"),
                prior=(f"prior_{component}", "mean"),
            )
        )
        macro = per_log.groupby("phase_bin", as_index=False)[["nominal_label", "variant_label", "prior"]].mean()
        macro = macro.set_index("phase_bin").reindex(range(config.phase_bins)).interpolate(limit_direction="both")
        nominal_label = macro["nominal_label"].to_numpy(dtype=float)
        variant_label = macro["variant_label"].to_numpy(dtype=float)
        prior_values = macro["prior"].to_numpy(dtype=float)
        nominal_shift = estimate_circular_phase_shift(nominal_label, prior_values, phase_grid=phase_grid)
        variant_shift = estimate_circular_phase_shift(variant_label, prior_values, phase_grid=phase_grid)
        nominal_residual = nominal_label - prior_values
        variant_residual = variant_label - prior_values
        nominal_waveform = nominal_residual - np.mean(nominal_residual)
        variant_waveform = variant_residual - np.mean(variant_residual)
        row["residual_waveform_correlation"] = _corr(nominal_waveform, variant_waveform)
        nominal_shift_rad = float(nominal_shift["shift_rad"])
        variant_shift_rad = float(variant_shift["shift_rad"])
        row["nominal_optimal_shift_rad"] = nominal_shift_rad
        row["variant_optimal_shift_rad"] = variant_shift_rad
        row["optimal_shift_change_rad"] = float(
            np.angle(np.exp(1j * (variant_shift_rad - nominal_shift_rad)))
        )
        direction = compute_wing_stroke_direction(phase_grid, 0.0)
        for half_name in ("upstroke", "downstroke"):
            half_mask = direction == half_name
            half_phase = phase_grid[half_mask]
            if half_name == "upstroke":
                half_phase = np.where(half_phase < 0.5 * math.pi, half_phase + 2.0 * math.pi, half_phase)
            order = np.argsort(half_phase, kind="stable")
            row[f"nominal_{half_name}_waveform_integral_n_rad"] = float(
                np.trapz(nominal_waveform[half_mask][order], half_phase[order])
            )
            row[f"variant_{half_name}_waveform_integral_n_rad"] = float(
                np.trapz(variant_waveform[half_mask][order], half_phase[order])
            )
    phase_rows: list[dict[str, object]] = []
    for (partition, phase_bin), group in combined.groupby(["partition", "phase_bin"], sort=True):
        for component in FORCE_COMPONENTS:
            variant_columns = [column for column in group.columns if column.startswith("variant_") and column.endswith(f"_{component}")]
            if not variant_columns:
                continue
            values = group[[f"label_{component}", *variant_columns]].to_numpy(dtype=float)
            uncertainty = np.nanstd(values, axis=1, ddof=1)
            phase_rows.append(
                {
                    "status": "ok",
                    "partition": partition,
                    "component": component,
                    "phase_bin": int(phase_bin),
                    "phase_center_rad": (int(phase_bin) + 0.5) * 2.0 * math.pi / config.phase_bins,
                    "sample_count": int(len(group)),
                    "label_uncertainty_std": float(np.nanmean(uncertainty)),
                }
            )
    return pd.DataFrame(variant_rows), pd.DataFrame(phase_rows)


def trim_impact_estimate(
    cycle_means: pd.DataFrame,
    sensitivity_artifact: str | Path | None,
) -> pd.DataFrame:
    """Estimate local trim-frequency shift when a WT1 sensitivity is provided."""

    if sensitivity_artifact is None:
        return pd.DataFrame(
            [
                {
                    "status": "not_computed_missing_wt1_sensitivity",
                    "mean_fz_discrepancy_n": float(cycle_means["mean_residual_fz_b"].mean()),
                }
            ]
        )
    path = Path(sensitivity_artifact)
    table = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.DataFrame([json.loads(path.read_text())])
    derivative_columns = [
        column
        for column in table.columns
        if column in {"d_mean_fz_d_frequency_n_per_hz", "dFz_df", "sensitivity_n_per_hz"}
    ]
    if not derivative_columns:
        raise ValueError("WT1 sensitivity artifact lacks d(mean Fz)/df in N/Hz")
    derivative = float(pd.to_numeric(table[derivative_columns[0]], errors="coerce").iloc[0])
    if not math.isfinite(derivative) or abs(derivative) < 1.0e-12:
        raise ValueError("WT1 sensitivity derivative must be finite and non-zero")
    per_log = cycle_means.groupby("log_id")["mean_residual_fz_b"].mean()
    discrepancy = float(per_log.mean())
    sem = float(per_log.std(ddof=1) / math.sqrt(max(len(per_log), 1)))
    return pd.DataFrame(
        [
            {
                "status": "local_linear_estimate",
                "mean_fz_discrepancy_n": discrepancy,
                "mean_fz_discrepancy_ci95_n": 1.96 * sem,
                "d_mean_fz_d_frequency_n_per_hz": derivative,
                "estimated_trim_frequency_shift_hz": -discrepancy / derivative,
                "estimated_trim_frequency_shift_ci95_hz": 1.96 * sem / abs(derivative),
                "scope": "local only; no extrapolation beyond WT1 operating point",
            }
        ]
    )


def decision_summary(
    *,
    phase_summary: Mapping[str, object],
    cycle_means: pd.DataFrame,
    harmonic_by_log: pd.DataFrame,
    static_probes: pd.DataFrame,
    history_probes: pd.DataFrame,
    label_uncertainty_phase: pd.DataFrame,
    prior_probe: pd.DataFrame,
    config: AuditConfig,
    physical_sensitivity: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Apply configurable evidence thresholds to the correction decision matrix."""

    result: dict[str, object] = {
        "fix_phase_convention_first": "insufficient_evidence",
        "fix_fixed_delay_first": "insufficient_evidence",
        "mean_correction_fx": "insufficient_evidence",
        "mean_correction_fz": "insufficient_evidence",
        "phase_correction_fx": "insufficient_evidence",
        "phase_correction_fz": "insufficient_evidence",
        "condition_features_recommended": [],
        "harmonic_order_range_recommended": [],
        "dynamic_model_needed": "insufficient_evidence",
        "tcn_needed": "no",
        "label_uncertainty_blocks_correction": "insufficient_evidence",
        "physical_parameter_adjustment_candidate": [],
        "prior_has_incremental_value": "insufficient_evidence",
        "thresholds": {
            key: value
            for key, value in config.__dict__.items()
            if key.endswith("threshold") or key.endswith("threshold_rad") or key.endswith("threshold_fraction")
        },
    }
    phase_offsets: list[float] = []
    delay_wins: list[bool] = []
    for component in FORCE_COMPONENTS:
        summary = phase_summary.get(component) if isinstance(phase_summary, Mapping) else None
        if not isinstance(summary, Mapping):
            continue
        offset = float(summary["H1_fixed_phase_offset_rad"])
        phase_offsets.append(offset)
        h1_rmse = float(summary["H1_rmse_rad"])
        h2_rmse = float(summary["H2_rmse_rad"])
        delay_wins.append(h2_rmse <= (1.0 - config.fixed_delay_improvement_fraction) * h1_rmse)
    if phase_offsets:
        significant = min(abs(value) for value in phase_offsets) >= config.phase_offset_threshold_rad
        component_agreement = (
            len(phase_offsets) >= len(FORCE_COMPONENTS)
            and abs(float(np.angle(np.exp(1j * (phase_offsets[0] - phase_offsets[1])))))
            <= config.phase_component_agreement_threshold_rad
        )
        result["fix_phase_convention_first"] = "yes" if significant and component_agreement else "no"
        result["fix_fixed_delay_first"] = "yes" if all(delay_wins) else "no"
    for component in FORCE_COMPONENTS:
        energy_column = f"mean_energy_fraction_{component}"
        if energy_column in cycle_means.columns and len(cycle_means):
            fraction = float(cycle_means.groupby("log_id")[energy_column].mean().mean())
            result[f"mean_correction_{'fx' if component == 'fx_b' else 'fz'}"] = (
                "yes" if fraction >= config.mean_energy_threshold else "no"
            )
        harmonic = harmonic_by_log.loc[
            (harmonic_by_log.get("component") == component)
            & (harmonic_by_log.get("harmonic_order") == config.harmonic_max_order)
        ]
        if len(harmonic):
            coverage = float(harmonic["cumulative_energy_coverage_mean"].mean())
            result[f"phase_correction_{'fx' if component == 'fx_b' else 'fz'}"] = (
                "yes" if coverage >= config.phase_energy_threshold else "no"
            )
    if not harmonic_by_log.empty:
        coverage_by_order = harmonic_by_log.groupby("harmonic_order")["cumulative_energy_coverage_mean"].mean()
        recommended = [int(order) for order, value in coverage_by_order.items() if value >= 0.70]
        if recommended:
            result["harmonic_order_range_recommended"] = [min(recommended), min(max(recommended), config.harmonic_max_order)]
    if not static_probes.empty and "validation_equal_log_macro_rmse" in static_probes.columns:
        best = static_probes.groupby(["component", "probe"])["validation_equal_log_macro_rmse"].min().unstack()
        recommended: set[str] = set()
        for _, row in best.dropna().iterrows():
            if "phase_only" not in row:
                continue
            for probe, feature in (
                ("phase_plus_airspeed", "airspeed"),
                ("phase_plus_angle_of_attack", "angle_of_attack"),
                ("phase_plus_flapping_frequency", "flapping_frequency"),
            ):
                if probe in row:
                    gain = 1.0 - float(row[probe]) / float(row["phase_only"])
                    if gain >= config.condition_probe_gain_threshold:
                        recommended.add(feature)
        result["condition_features_recommended"] = sorted(recommended)
    if not history_probes.empty and "validation_equal_log_macro_rmse" in history_probes.columns:
        best = history_probes.groupby(["component", "history_samples"])["validation_equal_log_macro_rmse"].min().unstack()
        gains: list[float] = []
        for _, row in best.iterrows():
            if 0 in row and row.drop(labels=[0], errors="ignore").notna().any():
                gains.append(1.0 - float(row.drop(labels=[0]).min()) / float(row[0]))
        result["dynamic_model_needed"] = "yes" if gains and min(gains) >= config.history_probe_gain_threshold else "no"
        result["tcn_needed"] = "no"  # linear short-history evidence cannot establish a TCN requirement
    if not label_uncertainty_phase.empty and "label_uncertainty_std" in label_uncertainty_phase.columns:
        uncertainty = float(label_uncertainty_phase["label_uncertainty_std"].mean())
        residual_scale = float(
            np.mean([cycle_means[f"waveform_rmse_{component}"].mean() for component in FORCE_COMPONENTS])
        )
        result["label_uncertainty_blocks_correction"] = (
            "no" if residual_scale >= config.label_uncertainty_ratio_threshold * uncertainty else "yes"
        )
    if not prior_probe.empty and "validation_equal_log_macro_rmse" in prior_probe.columns:
        best = prior_probe.groupby(["component", "model"])["validation_equal_log_macro_rmse"].min().unstack()
        gains = []
        for _, row in best.dropna().iterrows():
            if {"prior_plus_delta", "no_prior"}.issubset(row.index):
                gains.append(1.0 - float(row["prior_plus_delta"]) / float(row["no_prior"]))
        result["prior_has_incremental_value"] = "yes" if gains and min(gains) >= config.prior_gain_threshold else "no"
    if physical_sensitivity is not None and not physical_sensitivity.empty:
        macro = physical_sensitivity.groupby("parameter", as_index=False).agg(
            shape_correlation=("shape_correlation", "mean"),
            step_size_correlation=("step_size_correlation", "mean"),
            step_size_relative_difference=("step_size_relative_difference", "mean"),
        )
        stable = macro.loc[
            macro["step_size_correlation"].fillna(1.0).ge(0.95)
            & macro["step_size_relative_difference"].fillna(0.0).le(0.10)
            & macro["shape_correlation"].abs().ge(config.physical_similarity_threshold)
        ]
        result["physical_parameter_adjustment_candidate"] = sorted(stable["parameter"].astype(str).tolist())
    return result


def audit_summary_metrics(
    cycle_means: pd.DataFrame,
    phase_summary: Mapping[str, object],
    decision: Mapping[str, object],
) -> dict[str, object]:
    """Prepare compact values for CLI/report output without recomputation."""

    summary: dict[str, object] = {"decision": dict(decision), "phase_alignment": dict(phase_summary)}
    for component in FORCE_COMPONENTS:
        if cycle_means.empty:
            continue
        per_log = cycle_means.groupby("log_id").mean(numeric_only=True)
        summary[component] = {
            "cycle_count": int(len(cycle_means)),
            "log_count": int(cycle_means["log_id"].nunique()),
            "cycle_mean_residual_macro_n": float(per_log[f"mean_residual_{component}"].mean()),
            "cycle_mean_residual_median_n": float(cycle_means[f"mean_residual_{component}"].median()),
            "mean_energy_fraction_macro": float(per_log[f"mean_energy_fraction_{component}"].mean()),
            "waveform_energy_fraction_macro": float(per_log[f"waveform_energy_fraction_{component}"].mean()),
            "waveform_rmse_macro_n": float(per_log[f"waveform_rmse_{component}"].mean()),
        }
    return summary
