"""Opt-in DeLaurier component and local-sensitivity diagnostics for EDA0.

These diagnostics use the repository's frozen offline physics implementation.
They do not modify physical parameters, production defaults, or serialized
priors.  When the primary keyed prior comes from a different exporter commit,
the caller must retain the reported prior-mismatch columns and bound causal
interpretation accordingly.
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from system_identification.analysis.force_discrepancy_attribution import (
    AuditConfig,
    DEFAULT_PHASE_COLUMN,
    FORCE_COMPONENTS,
    _corr,
    _cycle_group_columns,
    _periodic_interpolate,
)
from system_identification.physics.baselines.wing_only import (
    WingOnlyBaselineConfig,
    baseline_config_from_aircraft_metadata,
    evaluate_wing_only_delaurier_segment,
)


DELAURIER_COMPONENTS = ("dN_c", "dN_a", "dT_s", "dD_camber", "dD_f")
DEFAULT_SPANWISE_REGIONS: Mapping[str, tuple[float, float]] = {
    "root": (0.0, 1.0 / 3.0),
    "mid": (1.0 / 3.0, 2.0 / 3.0),
    "tip": (2.0 / 3.0, 1.0),
}


def _config_with_primary_parameter_overrides(
    config: WingOnlyBaselineConfig,
    parameters: Mapping[str, object] | None,
) -> WingOnlyBaselineConfig:
    """Match scalar prior parameters while retaining attached-flow closure.

    The current frozen component API exposes an additive attached-flow
    decomposition. A separation-on historical primary can only be compared
    against a parameter-matched, separation-off diagnostic counterpart; the
    active attached-flow primary has an exact component closure.
    """

    if not parameters:
        return config
    scalar_mapping = {
        "alpha0_deg": ("alpha0_rad", lambda value: math.radians(float(value))),
        "eta_s": ("eta_s", float),
        "cd_cf": ("cd_cf", float),
        "alpha_stall_min_deg": ("alpha_stall_min_rad", lambda value: math.radians(float(value))),
        "alpha_stall_max_deg": ("alpha_stall_max_rad", lambda value: math.radians(float(value))),
        "xi": ("xi", float),
        "c_mac": ("c_mac", float),
        "cd_f": ("cd_f", float),
    }
    replacements: dict[str, float] = {}
    for source, (target, converter) in scalar_mapping.items():
        if source in parameters and parameters[source] is not None:
            replacements[target] = converter(parameters[source])
    mean_pitch = math.radians(float(parameters.get("theta_w_deg", 0.0)))
    return replace(
        config,
        mean_pitch_offset_rad=mean_pitch,
        params=replace(config.params, **replacements),
    )


def symmetric_finite_difference(plus: np.ndarray, minus: np.ndarray, step: float) -> np.ndarray:
    """Return a symmetric finite difference with explicit non-zero step."""

    if not math.isfinite(float(step)) or float(step) == 0.0:
        raise ValueError("finite-difference step must be finite and non-zero")
    plus_values = np.asarray(plus, dtype=float)
    minus_values = np.asarray(minus, dtype=float)
    if plus_values.shape != minus_values.shape:
        raise ValueError("plus and minus arrays must have equal shape")
    return (plus_values - minus_values) / (2.0 * float(step))


def _evaluate_groups(
    frame: pd.DataFrame,
    *,
    geometry_path: str | Path,
    config: WingOnlyBaselineConfig,
    theta_tip_deg: float,
    detailed: bool,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    group_columns = [column for column in ("partition", "log_id", "segment_id") if column in frame.columns]
    for _, group in frame.groupby(group_columns, sort=True, dropna=False):
        evaluated = evaluate_wing_only_delaurier_segment(
            group,
            theta_tip_deg=(float(theta_tip_deg),),
            geometry_path=geometry_path,
            config=config,
            spanwise_regions=DEFAULT_SPANWISE_REGIONS if detailed else None,
            include_detailed_diagnostics=detailed,
        )
        identity = group.loc[:, [column for column in ("partition", "log_id", "segment_id", "timestamp_us") if column in group.columns]]
        evaluated = evaluated.merge(identity, on=[column for column in ("log_id", "segment_id", "timestamp_us") if column in identity.columns], how="left", validate="one_to_one")
        outputs.append(evaluated)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def _balanced_sensitivity_subset(frame: pd.DataFrame, maximum_cycles_per_log: int = 30) -> pd.DataFrame:
    """Take systematic, phase-complete cycles from every log for bounded probes."""

    pieces: list[pd.DataFrame] = []
    group_columns = _cycle_group_columns(frame)
    for _, log_frame in frame.groupby(["partition", "log_id"], sort=True):
        cycle_keys = log_frame.loc[:, group_columns].drop_duplicates().sort_values(group_columns, kind="stable")
        if len(cycle_keys) > maximum_cycles_per_log:
            positions = np.linspace(0, len(cycle_keys) - 1, maximum_cycles_per_log).round().astype(int)
            cycle_keys = cycle_keys.iloc[np.unique(positions)]
        pieces.append(log_frame.merge(cycle_keys, on=group_columns, how="inner", validate="many_to_one"))
    return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0].copy()


def evaluate_component_diagnostics(
    cycle_samples: pd.DataFrame,
    *,
    metadata_path: str | Path,
    geometry_path: str | Path,
    theta_tip_deg: float,
    airflow_mode: str = "legacy_scalar_true_airspeed",
    chunk_size: int = 4096,
    primary_prior_parameters: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    """Evaluate opt-in aggregate, left/right, and root/mid/tip diagnostics."""

    config = baseline_config_from_aircraft_metadata(
        metadata_path,
        chunk_size=chunk_size,
        airflow_mode=airflow_mode,
    )
    config = _config_with_primary_parameter_overrides(config, primary_prior_parameters)
    evaluated = _evaluate_groups(
        cycle_samples,
        geometry_path=geometry_path,
        config=config,
        theta_tip_deg=theta_tip_deg,
        detailed=True,
    )
    keys = [column for column in ("partition", "log_id", "segment_id", "timestamp_us") if column in cycle_samples.columns]
    primary = cycle_samples.loc[
        :,
        [
            *keys,
            DEFAULT_PHASE_COLUMN,
            "cycle_id",
            *(f"residual_{component}" for component in FORCE_COMPONENTS),
            *(f"prior_{component}" for component in FORCE_COMPONENTS),
        ],
    ]
    diagnostic_columns = [
        column
        for column in evaluated.columns
        if column in keys
        or column.startswith("pred_")
        or column.startswith("component_")
        or column.startswith("span_")
    ]
    aligned = primary.merge(evaluated.loc[:, diagnostic_columns], on=keys, how="inner", validate="one_to_one")
    primary_separation = bool((primary_prior_parameters or {}).get("enable_separation", False))
    scope = (
        "frozen offline wing-only diagnostic with primary scalar parameters; "
        "historical separation-on primary remains a scope mismatch"
        if primary_separation
        else "frozen offline wing-only diagnostic matching the active attached-flow primary contract"
    )
    diagnostics: dict[str, float | str] = {
        "status": "ok",
        "airflow_mode": airflow_mode,
        "theta_tip_deg": float(theta_tip_deg),
        "physics_scope": scope,
    }
    for force in FORCE_COMPONENTS:
        component_sum = sum(aligned[f"component_{name}_{force}"] for name in DELAURIER_COMPONENTS)
        diagnostics[f"component_sum_max_abs_error_{force}"] = float(
            np.max(np.abs(component_sum - aligned[f"pred_{force}"]))
        )
        left_right_error = max(
            float(
                np.max(
                    np.abs(
                        aligned[f"component_{name}_left_{force}"]
                        + aligned[f"component_{name}_right_{force}"]
                        - aligned[f"component_{name}_{force}"]
                    )
                )
            )
            for name in DELAURIER_COMPONENTS
        )
        diagnostics[f"left_right_max_abs_error_{force}"] = left_right_error
        span_error = max(
            float(
                np.max(
                    np.abs(
                        sum(aligned[f"span_{region}_component_{name}_{force}"] for region in DEFAULT_SPANWISE_REGIONS)
                        - aligned[f"component_{name}_{force}"]
                    )
                )
            )
            for name in DELAURIER_COMPONENTS
        )
        diagnostics[f"spanwise_max_abs_error_{force}"] = span_error
        diagnostics[f"primary_vs_diagnostic_prior_rmse_{force}"] = float(
            np.sqrt(np.mean(np.square(aligned[f"prior_{force}"] - aligned[f"pred_{force}"])))
        )
    return aligned, diagnostics


def summarize_component_attribution(
    aligned: pd.DataFrame,
    config: AuditConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create component phase, half-stroke, spanwise, and similarity tables."""

    grid = np.linspace(0.0, 2.0 * math.pi, config.phase_bins, endpoint=False)
    phase_rows: list[dict[str, object]] = []
    half_rows: list[dict[str, object]] = []
    span_rows: list[dict[str, object]] = []
    similarity_rows: list[dict[str, object]] = []
    per_log_waveforms: dict[tuple[str, str, str], np.ndarray] = {}
    for (partition, log_id), log_frame in aligned.groupby(["partition", "log_id"], sort=True):
        for force in FORCE_COMPONENTS:
            residual_cycles: list[np.ndarray] = []
            component_cycles: dict[str, list[np.ndarray]] = {name: [] for name in DELAURIER_COMPONENTS}
            for _, cycle in log_frame.groupby([column for column in ("segment_id", "cycle_id") if column in log_frame.columns], sort=True):
                phase = cycle[DEFAULT_PHASE_COLUMN].to_numpy(dtype=float)
                residual_cycles.append(_periodic_interpolate(phase, cycle[f"residual_{force}"].to_numpy(dtype=float), grid))
                for name in DELAURIER_COMPONENTS:
                    component_cycles[name].append(
                        _periodic_interpolate(phase, cycle[f"component_{name}_{force}"].to_numpy(dtype=float), grid)
                    )
                direction = np.cos(phase)
                for name in DELAURIER_COMPONENTS:
                    for half, mask in (("upstroke", direction > 0.0), ("downstroke", direction < 0.0)):
                        if mask.any():
                            half_phase = phase[mask].copy()
                            if half == "upstroke":
                                half_phase = np.where(half_phase < 0.5 * math.pi, half_phase + 2.0 * math.pi, half_phase)
                            contribution = cycle.loc[mask, f"component_{name}_{force}"].to_numpy(dtype=float)
                            order = np.argsort(half_phase, kind="stable")
                            half_rows.append(
                                {
                                    "partition": partition,
                                    "log_id": log_id,
                                    "segment_id": cycle["segment_id"].iloc[0] if "segment_id" in cycle else np.nan,
                                    "cycle_id": cycle["cycle_id"].iloc[0],
                                    "component": name,
                                    "force_component": force,
                                    "half_stroke": half,
                                    "mean_contribution_n": float(cycle.loc[mask, f"component_{name}_{force}"].mean()),
                                    "integral_contribution_n_rad": float(
                                        np.trapz(contribution[order], half_phase[order])
                                    ),
                                }
                            )
                for region in DEFAULT_SPANWISE_REGIONS:
                    for name in DELAURIER_COMPONENTS:
                        span_rows.append(
                            {
                                "partition": partition,
                                "log_id": log_id,
                                "segment_id": cycle["segment_id"].iloc[0] if "segment_id" in cycle else np.nan,
                                "cycle_id": cycle["cycle_id"].iloc[0],
                                "span_region": region,
                                "component": name,
                                "force_component": force,
                                "cycle_mean_contribution_n": float(
                                    cycle[f"span_{region}_component_{name}_{force}"].mean()
                                ),
                                "cycle_rms_contribution_n": float(
                                    np.sqrt(
                                        np.mean(
                                            np.square(
                                                cycle[f"span_{region}_component_{name}_{force}"].to_numpy(dtype=float)
                                            )
                                        )
                                    )
                                ),
                            }
                        )
            residual_waveform = np.nanmean(residual_cycles, axis=0)
            per_log_waveforms[(str(partition), str(log_id), force)] = residual_waveform
            for name, waveforms in component_cycles.items():
                component_waveform = np.nanmean(waveforms, axis=0)
                centered_component = component_waveform - np.nanmean(component_waveform)
                centered_residual = residual_waveform - np.nanmean(residual_waveform)
                similarity_rows.append(
                    {
                        "partition": partition,
                        "log_id": log_id,
                        "force_component": force,
                        "component": name,
                        "shape_correlation": _corr(centered_residual, centered_component),
                        "signed_projection": float(
                            np.dot(centered_residual, centered_component)
                            / max(np.dot(centered_component, centered_component), 1.0e-15)
                        ),
                        "interpretation_boundary": "shape association; not unique physical causality",
                    }
                )
                for phase_bin, phase_value in enumerate(grid):
                    phase_rows.append(
                        {
                            "partition": partition,
                            "log_id": log_id,
                            "force_component": force,
                            "component": name,
                            "phase_bin": phase_bin,
                            "phase_center_rad": phase_value,
                            "contribution_n": component_waveform[phase_bin],
                            "residual_n": residual_waveform[phase_bin],
                        }
                    )
    return (
        pd.DataFrame(phase_rows),
        pd.DataFrame(half_rows),
        pd.DataFrame(span_rows),
        pd.DataFrame(similarity_rows),
    )


def _phase_binned_sensitivity_similarity(
    subset: pd.DataFrame,
    derivative: np.ndarray,
    *,
    parameter: str,
    step: float,
    half_step_derivative: np.ndarray | None,
    config: AuditConfig,
) -> list[dict[str, object]]:
    table = subset.copy()
    rows: list[dict[str, object]] = []
    for force_index, force in enumerate(FORCE_COMPONENTS):
        table["_derivative"] = derivative[:, force_index]
        if half_step_derivative is not None:
            table["_half_derivative"] = half_step_derivative[:, force_index]
        for (partition, log_id), log_frame in table.groupby(["partition", "log_id"], sort=True):
            phase = log_frame[DEFAULT_PHASE_COLUMN].to_numpy(dtype=float)
            grid = np.linspace(0.0, 2.0 * math.pi, config.phase_bins, endpoint=False)
            residual = _periodic_interpolate(phase, log_frame[f"residual_{force}"].to_numpy(dtype=float), grid)
            jacobian = _periodic_interpolate(phase, log_frame["_derivative"].to_numpy(dtype=float), grid)
            if half_step_derivative is not None:
                half = _periodic_interpolate(phase, log_frame["_half_derivative"].to_numpy(dtype=float), grid)
                step_correlation = _corr(jacobian, half)
                step_relative_difference = float(
                    np.linalg.norm(jacobian - half) / max(np.linalg.norm(half), 1.0e-12)
                )
            else:
                step_correlation = float("nan")
                step_relative_difference = float("nan")
            rows.append(
                {
                    "partition": partition,
                    "log_id": log_id,
                    "force_component": force,
                    "parameter": parameter,
                    "symmetric_step": step,
                    "shape_correlation": _corr(
                        residual - np.nanmean(residual), jacobian - np.nanmean(jacobian)
                    ),
                    "mean_direction_product": float(np.nanmean(residual) * np.nanmean(jacobian)),
                    "step_size_correlation": step_correlation,
                    "step_size_relative_difference": step_relative_difference,
                    "interpretation_boundary": "local sensitivity similarity; not calibrated parameter",
                }
            )
    return rows


def physical_sensitivity_audit(
    cycle_samples: pd.DataFrame,
    *,
    metadata_path: str | Path,
    geometry_path: str | Path,
    theta_tip_deg: float,
    config: AuditConfig,
    phase_step_rad: float = math.radians(2.0),
    incidence_step_rad: float = math.radians(1.0),
    twist_step_deg: float = 1.0,
    maximum_cycles_per_log: int = 30,
    primary_prior_parameters: Mapping[str, object] | None = None,
    airflow_mode: str = "legacy_scalar_true_airspeed",
) -> pd.DataFrame:
    """Run bounded symmetric finite differences on an equal-log cycle subset."""

    subset = _balanced_sensitivity_subset(cycle_samples, maximum_cycles_per_log=maximum_cycles_per_log)
    base_config = baseline_config_from_aircraft_metadata(
        metadata_path,
        chunk_size=4096,
        airflow_mode=airflow_mode,
    )
    base_config = _config_with_primary_parameter_overrides(base_config, primary_prior_parameters)
    baseline = _evaluate_groups(
        subset,
        geometry_path=geometry_path,
        config=base_config,
        theta_tip_deg=theta_tip_deg,
        detailed=True,
    )
    keys = [column for column in ("partition", "log_id", "segment_id", "timestamp_us") if column in subset.columns]
    baseline_columns = [
        column
        for column in baseline.columns
        if column in keys
        or column.startswith("pred_")
        or column.startswith("component_")
        or column.startswith("span_")
    ]
    ordered = subset.merge(baseline.loc[:, baseline_columns], on=keys, how="inner", validate="one_to_one")

    def evaluate_variant(
        parameter: str,
        delta: float,
    ) -> np.ndarray:
        varied_samples = subset.copy()
        varied_config = base_config
        varied_theta = theta_tip_deg
        if parameter == "phase_offset_rad":
            varied_samples["mechanical_phase_rad"] = varied_samples["mechanical_phase_rad"] + delta
        elif parameter == "mean_incidence_rad":
            varied_config = replace(base_config, mean_pitch_offset_rad=base_config.mean_pitch_offset_rad + delta)
        elif parameter == "dynamic_twist_tip_deg":
            varied_theta = theta_tip_deg + delta
        else:
            raise ValueError(parameter)
        output = _evaluate_groups(
            varied_samples,
            geometry_path=geometry_path,
            config=varied_config,
            theta_tip_deg=varied_theta,
            detailed=False,
        )
        aligned = subset.loc[:, keys].merge(
            output.loc[:, [*keys, "pred_fx_b", "pred_fz_b"]],
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        return aligned[["pred_fx_b", "pred_fz_b"]].to_numpy(dtype=float)

    rows: list[dict[str, object]] = []
    for parameter, step in (
        ("phase_offset_rad", phase_step_rad),
        ("mean_incidence_rad", incidence_step_rad),
        ("dynamic_twist_tip_deg", twist_step_deg),
    ):
        plus = evaluate_variant(parameter, step)
        minus = evaluate_variant(parameter, -step)
        half_plus = evaluate_variant(parameter, 0.5 * step)
        half_minus = evaluate_variant(parameter, -0.5 * step)
        derivative = symmetric_finite_difference(plus, minus, step)
        half_derivative = symmetric_finite_difference(half_plus, half_minus, 0.5 * step)
        rows.extend(
            _phase_binned_sensitivity_similarity(
                ordered,
                derivative,
                parameter=parameter,
                step=step,
                half_step_derivative=half_derivative,
                config=config,
            )
        )

    scale_components = {
        "normal_force_scale": ("dN_c", "dN_a"),
        "apparent_mass_scale": ("dN_a",),
        "chordwise_force_scale": ("dT_s", "dD_camber", "dD_f"),
    }
    for parameter, names in scale_components.items():
        derivative = np.column_stack(
            [sum(ordered[f"component_{name}_{force}"] for name in names) for force in FORCE_COMPONENTS]
        )
        rows.extend(
            _phase_binned_sensitivity_similarity(
                ordered,
                derivative,
                parameter=parameter,
                step=0.05,
                half_step_derivative=derivative.copy(),
                config=config,
            )
        )
    output = pd.DataFrame(rows)
    output["sensitivity_sample_rows"] = int(len(subset))
    output["maximum_cycles_per_log"] = int(maximum_cycles_per_log)
    output["selection_policy"] = "all logs; systematic evenly spaced complete cycles; no residual-based selection"
    output["airflow_mode"] = airflow_mode
    return output
