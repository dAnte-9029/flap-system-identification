"""Build correction-ready longitudinal force tables without fitting a model.

This module reuses the EDA0 keyed alignment, condition derivation, and complete
cycle selection contracts.  It adds only the versioned C1 table schema,
cycle-level decomposition, centered phase basis, train-only normalization, and
explicit weighting required by later correction stages.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from system_identification.analysis.force_discrepancy_attribution import (
    AlignmentResult,
    AuditConfig,
    DEFAULT_PHASE_COLUMN,
    DEFAULT_RHO_COLUMN,
    derive_condition_columns,
    keyed_align_label_and_prior,
    select_complete_cycles,
)
from system_identification.artifacts.prior_registry import PriorResolution
from system_identification.conventions.phase import (
    compute_wing_stroke_direction,
    wrap_to_2pi,
)
from system_identification.training.normalization import (
    fit_feature_stats,
    transform_features,
)


SCHEMA_VERSION = "longitudinal_correction_ready_v1"
FORCE_COMPONENTS = ("fx", "fz")
SOURCE_FORCE_COMPONENTS = ("fx_b", "fz_b")
ALIGNMENT_KEYS = ("log_id", "timestamp_us")
CONDITION_COLUMNS = (
    "alpha_mean_rad",
    "flapping_frequency_mean_hz",
    "airspeed_mean_mps",
    "rho_mean_kgpm3",
    "dynamic_pressure_mean_pa",
)
STANDARDIZED_CONDITION_COLUMNS = {
    "alpha_mean_rad": "alpha_mean_std",
    "flapping_frequency_mean_hz": "flapping_frequency_mean_std",
    "airspeed_mean_mps": "airspeed_mean_std",
    "rho_mean_kgpm3": "rho_mean_std",
    "dynamic_pressure_mean_pa": "dynamic_pressure_mean_std",
}


@dataclass(frozen=True)
class CorrectionReadyConfig:
    """Resolved C1 data preparation settings."""

    phase_bins: int = 72
    minimum_cycle_samples: int = 12
    minimum_phase_coverage_rad: float = 5.5
    maximum_phase_gap_rad: float = 0.8
    harmonic_max_order: int = 4
    condition_aggregation: str = "arithmetic_mean_per_complete_cycle"
    reconstruction_tolerance: float = 1.0e-10
    zero_mean_tolerance: float = 1.0e-10
    normalization_std_epsilon: float = 1.0e-8
    minimum_accepted_cycle_fraction: float = 0.30
    maximum_missing_alignment_fraction: float = 0.0
    random_seed: int = 20260717

    def validate(self) -> None:
        if self.phase_bins < 16:
            raise ValueError("phase_bins must be at least 16")
        if self.minimum_cycle_samples < 4:
            raise ValueError("minimum_cycle_samples must be at least 4")
        if not 0.0 < self.minimum_phase_coverage_rad <= 2.0 * math.pi:
            raise ValueError("minimum_phase_coverage_rad must be in (0, 2*pi]")
        if not 0.0 < self.maximum_phase_gap_rad <= 2.0 * math.pi:
            raise ValueError("maximum_phase_gap_rad must be in (0, 2*pi]")
        if self.harmonic_max_order != 4:
            raise ValueError("C1 freezes harmonic_max_order at 4; order selection belongs to C2/C3")
        if self.condition_aggregation != "arithmetic_mean_per_complete_cycle":
            raise ValueError("Unsupported condition_aggregation")
        if self.reconstruction_tolerance <= 0.0 or self.zero_mean_tolerance <= 0.0:
            raise ValueError("numerical tolerances must be positive")
        if not 0.0 <= self.minimum_accepted_cycle_fraction <= 1.0:
            raise ValueError("minimum_accepted_cycle_fraction must be in [0, 1]")


@dataclass
class CyclePreparation:
    """Accepted rows and explicit cycle quality/rejection evidence."""

    accepted_rows: pd.DataFrame
    quality: pd.DataFrame
    rejections: pd.DataFrame


def normalize_correction_partitions(partitions: Sequence[str]) -> tuple[str, ...]:
    """Normalize partition aliases while making the test label unavailable."""

    result: list[str] = []
    for raw in partitions:
        value = str(raw).strip().lower()
        if value == "test":
            raise ValueError("C0/C1 must not load the test partition or test labels")
        if value not in {"train", "val", "validation"}:
            raise ValueError(f"Unsupported partition {raw!r}; use train and validation")
        normalized = "validation" if value in {"val", "validation"} else "train"
        if normalized not in result:
            result.append(normalized)
    if not result:
        raise ValueError("At least one of train or validation is required")
    return tuple(result)


def _nested(mapping: Mapping[str, object], *keys: str, default: object = None) -> object:
    value: object = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def validate_correction_contract(
    prior: PriorResolution,
    dataset_manifest: Mapping[str, object],
    aircraft_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Validate the frozen prior/label frame, phase, airflow, and scope contract."""

    if prior.lifecycle_status != "active" or prior.is_legacy:
        raise ValueError(f"Authoritative prior must be active, got {prior.lifecycle_status!r}")
    if bool(prior.manifest.get("test_partition_loaded", False)):
        raise ValueError("Authoritative prior must not be a test-window diagnostic prior")
    body_frame = str(_nested(aircraft_metadata, "frames", "body_frame", default="unknown"))
    local_frame = str(_nested(aircraft_metadata, "frames", "local_frame", default="unknown"))
    expected_frame = "body_frd_force_at_imu_origin_moment_about_cg"
    if body_frame.upper() != "FRD" or prior.frame_contract != expected_frame:
        raise ValueError(
            "prior/label frame contract mismatch: "
            f"label body={body_frame!r}, prior={prior.frame_contract!r}"
        )
    if local_frame.upper() != "NED":
        raise ValueError(f"label local frame must be NED, got {local_frame!r}")
    expected_phase = "canonical_mechanical_phase_to_delaurier_v1"
    if prior.phase_contract != expected_phase:
        raise ValueError(
            f"prior/label phase contract mismatch: expected {expected_phase!r}, got {prior.phase_contract!r}"
        )
    expected_airflow = "attitude_ground_wind_3d"
    if prior.airflow_contract != expected_airflow:
        raise ValueError(
            f"prior/label airflow contract mismatch: expected {expected_airflow!r}, got {prior.airflow_contract!r}"
        )
    dataset_id = str(dataset_manifest.get("dataset_id", ""))
    if not dataset_id:
        raise ValueError("dataset manifest has no dataset_id")
    required_ratio_fields = (
        "wing_transmission_ratio",
        "ratio_contract_version",
        "ratio_source",
        "phase_contract_version",
        "frequency_contract_version",
    )
    missing_ratio = [key for key in required_ratio_fields if not dataset_manifest.get(key)]
    if missing_ratio:
        raise ValueError(f"dataset manifest missing ratio contract fields: {missing_ratio}")
    prior_ratio = prior.manifest.get("wing_transmission_ratio")
    if prior_ratio is None or not np.isclose(
        float(prior_ratio), float(dataset_manifest["wing_transmission_ratio"]), atol=0.0, rtol=0.0
    ):
        raise ValueError("prior/dataset wing transmission ratio mismatch")
    for key in ("ratio_contract_version", "phase_contract_version", "frequency_contract_version"):
        if str(prior.manifest.get(key, "")) != str(dataset_manifest[key]):
            raise ValueError(f"prior/dataset {key} mismatch")
    label_definition = str(
        _nested(
            aircraft_metadata,
            "label_definition",
            "force_definition",
            default="unknown",
        )
    )
    if label_definition != "effective_non_gravity_external_force":
        raise ValueError(f"Unsupported reconstructed force label contract {label_definition!r}")
    source = prior.manifest.get("physics_source", {})
    source_mapping = source if isinstance(source, Mapping) else {}
    contracts = prior.manifest.get("contracts", {})
    contract_mapping = contracts if isinstance(contracts, Mapping) else {}
    return {
        "resolved_prior_id": prior.prior_id,
        "prior_lifecycle_status": prior.lifecycle_status,
        "prior_artifact_path": str(prior.artifact_root.resolve()),
        "physics_repository": str(source_mapping.get("repository", "unknown")),
        "physics_commit": prior.physics_source_commit,
        "physics_dirty": source_mapping.get("dirty", "not_recorded_in_prior_manifest"),
        "frame_contract": prior.frame_contract,
        "label_body_frame": body_frame,
        "label_local_frame": local_frame,
        "phase_contract": prior.phase_contract,
        "airflow_contract": prior.airflow_contract,
        "separation_contract": str(contract_mapping.get("separation_contract", "unknown")),
        "dynamic_twist_contract": str(contract_mapping.get("dynamic_twist_contract", "unknown")),
        "prior_partition_coverage": list(prior.required_partitions),
        "key_schema": list(ALIGNMENT_KEYS),
        "dataset_id": dataset_id,
        "label_definition": label_definition,
        "label_frame": "body_FRD",
        "label_units": "N",
        "label_sign": "+Fx forward, +Fz down",
        "attitude_quaternion_convention": "wxyz body_FRD_to_NED",
        "timestamp_units": "integer_microseconds",
        "target_scope": "provisional_effective_longitudinal_force",
        **{key: dataset_manifest[key] for key in required_ratio_fields},
    }


def align_correction_partition(
    samples: pd.DataFrame,
    prior: pd.DataFrame,
    *,
    partition: str,
    maximum_missing_fraction: float = 0.0,
) -> AlignmentResult:
    """Strictly align one label/prior partition and add authoritative conditions."""

    canonical_partition = normalize_correction_partitions((partition,))[0]
    if "timestamp_us" in samples.columns and "timestamp_us" in prior.columns:
        sample_ts = pd.to_numeric(samples["timestamp_us"], errors="coerce")
        prior_ts = pd.to_numeric(prior["timestamp_us"], errors="coerce")
        if sample_ts.isna().any() or prior_ts.isna().any():
            raise ValueError("timestamp_us contains non-numeric values")
        ratio = max(abs(float(sample_ts.median())), 1.0) / max(
            abs(float(prior_ts.median())), 1.0
        )
        if not 0.5 <= ratio <= 2.0:
            raise ValueError(f"timestamp unit mismatch suspected; median magnitude ratio={ratio:.6g}")
    prior_partition_column = next(
        (column for column in ("partition", "split") if column in prior.columns),
        None,
    )
    if prior_partition_column is not None:
        observed = {
            "validation" if str(value).lower() in {"val", "validation"} else str(value).lower()
            for value in prior[prior_partition_column].dropna().unique()
        }
        if observed != {canonical_partition}:
            raise ValueError(
                f"Prior partition identity conflict: expected {canonical_partition}, observed {sorted(observed)}"
            )
    source_alpha = (
        pd.to_numeric(samples["condition_alpha_rad"], errors="coerce").copy()
        if "condition_alpha_rad" in samples.columns
        else None
    )
    result = keyed_align_label_and_prior(
        samples,
        prior,
        partition="val" if canonical_partition == "validation" else "train",
        keys=ALIGNMENT_KEYS,
        maximum_missing_fraction=maximum_missing_fraction,
    )
    conditioned, sources = derive_condition_columns(result.aligned)
    if source_alpha is not None and conditioned["condition_alpha_rad"].isna().all():
        conditioned["condition_alpha_rad"] = source_alpha.reindex(conditioned.index)
        sources["condition_alpha_rad"] = "condition_alpha_rad"
    conditioned["condition_rho_kgpm3"] = pd.to_numeric(
        conditioned[DEFAULT_RHO_COLUMN], errors="coerce"
    )
    sources["condition_rho_kgpm3"] = DEFAULT_RHO_COLUMN
    conditioned["partition"] = canonical_partition
    result.aligned = conditioned
    result.report["partition"] = canonical_partition
    result.report["condition_sources"] = sources
    result.report["keyed_alignment"] = True
    result.report["row_order_used"] = False
    return result


def _source_key_frame(frame: pd.DataFrame) -> pd.Series:
    segment = frame["segment_id"] if "segment_id" in frame.columns else pd.Series(0, index=frame.index)
    return (
        frame["partition"].astype(str)
        + "\x1f"
        + frame["log_id"].astype(str)
        + "\x1f"
        + segment.astype(str)
        + "\x1f"
        + frame["cycle_id"].astype(str)
    )


def _flight_date(log_id: object) -> str:
    match = re.search(r"(20\d{2})-(\d{1,2})-(\d{1,2})", str(log_id))
    if not match:
        return "unknown"
    year, month, day = (int(value) for value in match.groups())
    return f"{year:04d}-{month:02d}-{day:02d}"


def _maximum_circular_gap(phase: np.ndarray) -> float:
    wrapped = np.sort(np.unique(np.round(wrap_to_2pi(phase), 12)))
    if len(wrapped) < 2:
        return 2.0 * math.pi
    return float(np.max(np.r_[np.diff(wrapped), 2.0 * math.pi - wrapped[-1] + wrapped[0]]))


def _deterministic_cycle_id(partition: str, log_id: str, sequence: int, start_us: int) -> str:
    payload = f"{partition}|{log_id}|{sequence}|{start_us}".encode("utf-8")
    return f"cycle_{hashlib.sha256(payload).hexdigest()[:20]}"


def segment_complete_cycles(
    aligned: pd.DataFrame,
    config: CorrectionReadyConfig,
) -> CyclePreparation:
    """Reuse Phase-0 cycle selection and add the complete C1 quality contract."""

    config.validate()
    base = select_complete_cycles(
        aligned,
        AuditConfig(
            phase_bins=config.phase_bins,
            minimum_cycle_samples=config.minimum_cycle_samples,
            minimum_phase_coverage_rad=config.minimum_phase_coverage_rad,
            maximum_cycle_missing_fraction=0.0,
            harmonic_max_order=config.harmonic_max_order,
            random_seed=config.random_seed,
            maximum_missing_alignment_fraction=config.maximum_missing_alignment_fraction,
        ),
    )
    group_columns = ["partition", "log_id"]
    if "segment_id" in aligned.columns:
        group_columns.append("segment_id")
    group_columns.append("cycle_id")
    records: list[dict[str, object]] = []
    base_quality = {
        str(row["partition"])
        + "\x1f"
        + str(row["log_id"])
        + "\x1f"
        + str(row.get("segment_id", 0))
        + "\x1f"
        + str(row["cycle_id"]): row
        for row in base.quality.to_dict(orient="records")
    }
    finite_conditions = [
        "condition_alpha_rad",
        "condition_frequency_hz",
        "condition_airspeed_m_s",
        "condition_rho_kgpm3",
        "condition_dynamic_pressure_pa",
    ]
    for key, group in aligned.groupby(group_columns, sort=True, dropna=False):
        ordered = group.sort_values(["timestamp_us", "time_s"], kind="stable")
        source_key = str(_source_key_frame(ordered).iloc[0])
        phase = pd.to_numeric(ordered[DEFAULT_PHASE_COLUMN], errors="coerce").to_numpy(dtype=float)
        timestamp = pd.to_numeric(ordered["timestamp_us"], errors="coerce").to_numpy(dtype=float)
        prior_reasons = str(base_quality[source_key].get("rejection_reasons", ""))
        reasons = [reason for reason in prior_reasons.split(";") if reason]
        maximum_gap = _maximum_circular_gap(phase[np.isfinite(phase)])
        if maximum_gap > config.maximum_phase_gap_rad:
            reasons.append("maximum_phase_gap_exceeded")
        if not np.isfinite(ordered[finite_conditions].to_numpy(dtype=float)).all():
            reasons.append("non_finite_label_prior_or_state")
        if len(timestamp) < 2 or not np.all(np.diff(timestamp) > 0.0):
            reasons.append("non_monotonic_timestamp")
        if not np.allclose(timestamp, np.round(timestamp), atol=0.0, rtol=0.0):
            reasons.append("timestamp_not_integer_microseconds")
        reasons = list(dict.fromkeys(reasons))
        records.append(
            {
                "source_key": source_key,
                "partition": str(ordered["partition"].iloc[0]),
                "log_id": str(ordered["log_id"].iloc[0]),
                "segment_id": ordered["segment_id"].iloc[0] if "segment_id" in ordered else 0,
                "source_cycle_id": ordered["cycle_id"].iloc[0],
                "start_timestamp_us": int(timestamp[0]),
                "end_timestamp_us": int(timestamp[-1]),
                "sample_count_raw": int(len(ordered)),
                "phase_coverage_rad": float(base_quality[source_key]["phase_coverage_rad"]),
                "maximum_phase_gap_rad": maximum_gap,
                "duration_s": float((timestamp[-1] - timestamp[0]) * 1.0e-6),
                "duplicate_endpoint": bool(base_quality[source_key]["duplicate_endpoint"]),
                "phase_endpoint_policy": str(base_quality[source_key]["endpoint_action"]),
                "accepted": not reasons,
                "rejection_reason": ";".join(reasons),
            }
        )
    quality = pd.DataFrame(records).sort_values(
        ["partition", "log_id", "start_timestamp_us", "segment_id", "source_cycle_id"],
        kind="stable",
    )
    quality["cycle_index_within_log"] = quality.groupby(
        ["partition", "log_id"], sort=False
    ).cumcount()
    quality["correction_cycle_id"] = [
        _deterministic_cycle_id(
            str(row.partition),
            str(row.log_id),
            int(row.cycle_index_within_log),
            int(row.start_timestamp_us),
        )
        for row in quality.itertuples(index=False)
    ]
    quality["flight_date"] = quality["log_id"].map(_flight_date)
    quality["cycle_frequency_hz_from_duration"] = np.where(
        quality["duration_s"] > 0.0,
        1.0 / quality["duration_s"],
        np.nan,
    )
    allowed = quality.loc[quality["accepted"]]
    source_to_cycle = dict(zip(allowed["source_key"], allowed["correction_cycle_id"]))
    source_to_index = dict(zip(allowed["source_key"], allowed["cycle_index_within_log"]))
    selected = base.accepted_rows.copy()
    selected["source_key"] = _source_key_frame(selected)
    selected = selected.loc[selected["source_key"].isin(source_to_cycle)].copy()
    selected["correction_cycle_id"] = selected["source_key"].map(source_to_cycle)
    selected["cycle_index_within_log"] = selected["source_key"].map(source_to_index).astype(int)
    selected["source_cycle_id"] = selected["cycle_id"]
    selected["flight_date"] = selected["log_id"].map(_flight_date)
    selected_phase = wrap_to_2pi(selected[DEFAULT_PHASE_COLUMN].to_numpy(dtype=float))
    selected["_endpoint"] = np.isclose(selected_phase, 0.0, atol=1.0e-10)
    endpoint_duplicate = selected["_endpoint"] & selected.duplicated(
        ["correction_cycle_id", "_endpoint"], keep="first"
    )
    removed_counts = (
        selected.loc[endpoint_duplicate]
        .groupby("correction_cycle_id")
        .size()
        .reindex(quality["correction_cycle_id"], fill_value=0)
        .to_numpy(dtype=int)
    )
    quality["removed_endpoint_sample_count"] = removed_counts
    quality.loc[
        quality["removed_endpoint_sample_count"] > 0, "phase_endpoint_policy"
    ] = "removed_duplicate_wrapped_zero_endpoint_samples"
    selected = selected.loc[~endpoint_duplicate].drop(columns="_endpoint").copy()
    remaining_counts = selected.groupby("correction_cycle_id").size()
    too_short_ids = remaining_counts.loc[
        remaining_counts < config.minimum_cycle_samples
    ].index
    if len(too_short_ids):
        mask = quality["correction_cycle_id"].isin(too_short_ids)
        quality.loc[mask, "accepted"] = False
        quality.loc[mask, "rejection_reason"] = quality.loc[mask, "rejection_reason"].map(
            lambda value: ";".join(
                filter(None, [str(value), "too_few_samples_after_endpoint_deduplication"])
            )
        )
        selected = selected.loc[~selected["correction_cycle_id"].isin(too_short_ids)].copy()
    quality_payload = quality.loc[quality["accepted"]].set_index("correction_cycle_id")
    for column in (
        "phase_coverage_rad",
        "maximum_phase_gap_rad",
        "duration_s",
        "cycle_frequency_hz_from_duration",
        "start_timestamp_us",
        "end_timestamp_us",
    ):
        selected[column] = selected["correction_cycle_id"].map(quality_payload[column])
    selected = selected.sort_values(
        ["partition", "log_id", "start_timestamp_us", "timestamp_us"],
        kind="stable",
    ).reset_index(drop=True)
    rejection_rows: list[dict[str, object]] = []
    for row in quality.loc[~quality["accepted"]].to_dict(orient="records"):
        for reason in str(row["rejection_reason"]).split(";"):
            rejection_rows.append({**row, "rejection_reason": reason})
    rejections = pd.DataFrame(rejection_rows, columns=[*quality.columns])
    return CyclePreparation(
        accepted_rows=selected,
        quality=quality.drop(columns="source_key").reset_index(drop=True),
        rejections=rejections.drop(columns="source_key", errors="ignore").reset_index(drop=True),
    )


def build_correction_tables(
    accepted_rows: pd.DataFrame,
    config: CorrectionReadyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build deterministic cycle and waveform tables from accepted cycles."""

    config.validate()
    if accepted_rows.empty:
        raise ValueError("No accepted cycles are available")
    frame = accepted_rows.copy()
    frame = frame.sort_values(
        ["partition", "log_id", "correction_cycle_id", "timestamp_us"],
        kind="stable",
    ).reset_index(drop=True)
    group = frame.groupby("correction_cycle_id", sort=False)
    frame["sample_index_within_cycle"] = group.cumcount()
    frame["sample_count"] = group["timestamp_us"].transform("size")
    phase = wrap_to_2pi(frame[DEFAULT_PHASE_COLUMN].to_numpy(dtype=float))
    frame["phase_rad"] = phase
    frame["phase_fraction"] = phase / (2.0 * math.pi)
    frame["half_stroke_id"] = compute_wing_stroke_direction(phase, 0.0)

    condition_sources = {
        "alpha_mean_rad": "condition_alpha_rad",
        "flapping_frequency_mean_hz": "condition_frequency_hz",
        "airspeed_mean_mps": "condition_airspeed_m_s",
        "rho_mean_kgpm3": "condition_rho_kgpm3",
        "dynamic_pressure_mean_pa": "condition_dynamic_pressure_pa",
    }
    for output, source in condition_sources.items():
        frame[output] = group[source].transform("mean")
    frame["airspeed_negative_fraction"] = group["condition_airspeed_m_s"].transform(
        lambda values: float((pd.to_numeric(values, errors="coerce") < 0.0).mean())
    )
    frame["airspeed_min_mps"] = group["condition_airspeed_m_s"].transform("min")
    frame["airspeed_condition_valid"] = frame["airspeed_negative_fraction"].eq(0.0)
    frame["dynamic_pressure_condition_valid"] = frame["airspeed_condition_valid"]

    for output_component, source_component in zip(FORCE_COMPONENTS, SOURCE_FORCE_COMPONENTS):
        frame[f"label_{output_component}_n"] = frame[f"label_{source_component}"].astype(float)
        frame[f"prior_{output_component}_n"] = frame[f"prior_{source_component}"].astype(float)
        frame[f"label_{output_component}_mean_n"] = group[f"label_{source_component}"].transform("mean")
        frame[f"prior_{output_component}_mean_n"] = group[f"prior_{source_component}"].transform("mean")
        frame[f"label_{output_component}_waveform_n"] = (
            frame[f"label_{output_component}_n"] - frame[f"label_{output_component}_mean_n"]
        )
        frame[f"prior_{output_component}_waveform_n"] = (
            frame[f"prior_{output_component}_n"] - frame[f"prior_{output_component}_mean_n"]
        )
        frame[f"residual_{output_component}_n"] = (
            frame[f"label_{output_component}_n"] - frame[f"prior_{output_component}_n"]
        )
        frame[f"residual_{output_component}_mean_n"] = (
            frame[f"label_{output_component}_mean_n"] - frame[f"prior_{output_component}_mean_n"]
        )
        frame[f"residual_{output_component}_waveform_n"] = (
            frame[f"label_{output_component}_waveform_n"]
            - frame[f"prior_{output_component}_waveform_n"]
        )

    for order in range(1, config.harmonic_max_order + 1):
        for name, values in (
            ("sin", np.sin(order * phase)),
            ("cos", np.cos(order * phase)),
        ):
            raw_column = f"{name}_{order}_phase"
            centered_column = f"{raw_column}_centered"
            frame[raw_column] = values
            frame[centered_column] = frame[raw_column] - group[raw_column].transform("mean")

    cycle_counts_log = frame.groupby(["partition", "log_id"])["correction_cycle_id"].transform("nunique")
    cycle_counts_date = frame.groupby(["partition", "flight_date"])["correction_cycle_id"].transform("nunique")
    frame["weight_equal_cycle_sample"] = 1.0 / frame["sample_count"]
    frame["weight_equal_log_sample"] = 1.0 / (cycle_counts_log * frame["sample_count"])
    frame["weight_equal_date_sample"] = 1.0 / (cycle_counts_date * frame["sample_count"])

    first = frame.groupby("correction_cycle_id", sort=False).first().reset_index()
    cycle_table = pd.DataFrame(
        {
            "cycle_id": first["correction_cycle_id"],
            "partition": first["partition"],
            "log_id": first["log_id"],
            "flight_date": first["flight_date"],
            "cycle_index_within_log": first["cycle_index_within_log"].astype(int),
            "start_timestamp_us": first["start_timestamp_us"].astype("int64"),
            "end_timestamp_us": first["end_timestamp_us"].astype("int64"),
            "sample_count": first["sample_count"].astype(int),
            "phase_coverage_rad": first["phase_coverage_rad"],
            "maximum_phase_gap_rad": first["maximum_phase_gap_rad"],
            "duration_s": first["duration_s"],
            "cycle_frequency_hz_from_duration": first["cycle_frequency_hz_from_duration"],
            "quality_status": "accepted",
            "rejection_reason": "",
        }
    )
    for column in CONDITION_COLUMNS:
        cycle_table[column] = first[column]
    for column in (
        "airspeed_negative_fraction",
        "airspeed_min_mps",
        "airspeed_condition_valid",
        "dynamic_pressure_condition_valid",
    ):
        cycle_table[column] = first[column]
    for component in FORCE_COMPONENTS:
        cycle_table[f"label_{component}_mean_n"] = first[f"label_{component}_mean_n"]
        cycle_table[f"prior_{component}_mean_n"] = first[f"prior_{component}_mean_n"]
        cycle_table[f"residual_{component}_mean_n"] = first[f"residual_{component}_mean_n"]
    cycles_per_log = cycle_table.groupby(["partition", "log_id"])["cycle_id"].transform("size")
    cycles_per_date = cycle_table.groupby(["partition", "flight_date"])["cycle_id"].transform("size")
    cycle_table["weight_equal_cycle"] = 1.0
    cycle_table["weight_equal_log"] = 1.0 / cycles_per_log
    cycle_table["weight_equal_date"] = 1.0 / cycles_per_date

    waveform_columns = [
        "correction_cycle_id",
        "partition",
        "log_id",
        "flight_date",
        "timestamp_us",
        "sample_index_within_cycle",
        "phase_rad",
        "phase_fraction",
        "half_stroke_id",
    ]
    waveform_columns.extend(
        column
        for column in frame.columns
        if column.startswith(("label_", "prior_", "residual_", "sin_", "cos_", "weight_"))
        or column in CONDITION_COLUMNS
        or column in {
            "airspeed_negative_fraction",
            "airspeed_min_mps",
            "airspeed_condition_valid",
            "dynamic_pressure_condition_valid",
        }
    )
    waveform = frame.loc[:, list(dict.fromkeys(waveform_columns))].rename(
        columns={"correction_cycle_id": "cycle_id"}
    )
    cycle_table = cycle_table.sort_values(
        ["partition", "log_id", "start_timestamp_us"], kind="stable"
    ).reset_index(drop=True)
    waveform = waveform.sort_values(
        ["partition", "log_id", "timestamp_us"], kind="stable"
    ).reset_index(drop=True)
    return cycle_table, waveform


def fit_condition_normalization(cycle_table: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Fit condition statistics from training cycles only."""

    training = cycle_table.loc[cycle_table["partition"] == "train"].copy()
    if training.empty:
        raise ValueError("Training partition is required to fit normalization")
    values = training.loc[:, CONDITION_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Training conditions contain non-finite values")
    medians, means, stds = fit_feature_stats(values)
    raw_stds = values.std(axis=0)
    stats: dict[str, dict[str, object]] = {}
    for index, column in enumerate(CONDITION_COLUMNS):
        stats[column] = {
            "mean": float(means[index]),
            "std": float(stds[index]),
            "minimum": float(values[:, index].min()),
            "maximum": float(values[:, index].max()),
            "median_for_imputation": float(medians[index]),
            "raw_std": float(raw_stds[index]),
            "zero_variance": bool(raw_stds[index] <= 1.0e-8),
            "near_zero_std_policy": "replace_with_1.0_when_std_le_1e-8",
            "training_row_count": int(len(training)),
            "training_cycle_count": int(training["cycle_id"].nunique()),
            "source_partition": "train",
        }
    return stats


def apply_condition_normalization(
    cycle_table: pd.DataFrame,
    waveform_table: pd.DataFrame,
    stats: Mapping[str, Mapping[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply frozen training statistics to cycle and waveform tables."""

    medians = np.asarray([stats[column]["median_for_imputation"] for column in CONDITION_COLUMNS])
    means = np.asarray([stats[column]["mean"] for column in CONDITION_COLUMNS])
    stds = np.asarray([stats[column]["std"] for column in CONDITION_COLUMNS])
    result_tables: list[pd.DataFrame] = []
    for source in (cycle_table, waveform_table):
        result = source.copy()
        transformed = transform_features(
            result.loc[:, CONDITION_COLUMNS].to_numpy(dtype=float),
            medians,
            means,
            stds,
        )
        for index, column in enumerate(CONDITION_COLUMNS):
            result[STANDARDIZED_CONDITION_COLUMNS[column]] = transformed[:, index]
        result_tables.append(result)
    return result_tables[0], result_tables[1]


def table_semantic_hash(frame: pd.DataFrame) -> str:
    """Hash a table after deterministic row/column ordering."""

    sort_columns = [
        column
        for column in ("partition", "log_id", "cycle_id", "timestamp_us", "start_timestamp_us")
        if column in frame.columns
    ]
    ordered = frame.sort_values(sort_columns, kind="stable") if sort_columns else frame
    ordered = ordered.reset_index(drop=True).reindex(sorted(ordered.columns), axis=1)
    payload = pd.util.hash_pandas_object(ordered, index=True).to_numpy(dtype="uint64").tobytes()
    schema = json.dumps(
        {column: str(dtype) for column, dtype in ordered.dtypes.items()}, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(schema + payload).hexdigest()
