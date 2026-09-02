"""Directed Quadratic-2 amplitude-conditioned kappa/phase maps.

This module reuses the v1 train/validation-only evaluator and exact completed
parameter hashes. It never constructs a test-partition path.
"""

from __future__ import annotations

from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from system_identification.analysis.quadratic2_twist_sweep import (
    TwistCandidate,
    ResolvedExperiment,
    _manifest_base,
    _pareto_mask,
    _plot_wind_sensitivity,
    _read_jsonl,
    _wide_metrics,
    _write_json,
    _zero_wind_shortlist,
    build_shortlists,
    load_experiment_rows,
    run_candidates,
)
from system_identification.artifacts.io import sha256_file


def conditional_candidates(resolved: ResolvedExperiment) -> list[TwistCandidate]:
    grid = resolved.config["sweep"]["conditional"]
    allowed = resolved.config["physics"]["allowed"]
    assert isinstance(grid, Mapping)
    assert isinstance(allowed, Mapping)
    values: dict[str, list[float]] = {}
    for name in ("A_tip_deg", "kappa", "psi_theta_deg"):
        sequence = [float(value) for value in grid[name]]
        low, high = (float(value) for value in allowed[name])
        if not sequence or not np.isfinite(sequence).all():
            raise ValueError(f"conditional.{name} must be non-empty and finite")
        if min(sequence) < low or max(sequence) > high:
            raise ValueError(f"conditional.{name} exceeds [{low}, {high}]")
        if len(sequence) != len(set(sequence)):
            raise ValueError(f"conditional.{name} contains duplicates")
        values[name] = sequence
    static = float(grid["static_twist_offset_deg"])
    static_low, static_high = (float(value) for value in allowed["static_twist_offset_deg"])
    if not static_low <= static <= static_high:
        raise ValueError("conditional.static_twist_offset_deg is out of range")
    result = [
        TwistCandidate(
            profile_name="quadratic2_phase",
            A_tip_deg=amplitude,
            kappa=kappa,
            psi_theta_deg=phase,
            static_twist_offset_deg=static,
            stage="conditional",
            family="A_tip_conditioned_kappa_x_psi",
        )
        for amplitude, kappa, phase in itertools.product(
            values["A_tip_deg"], values["kappa"], values["psi_theta_deg"]
        )
    ]
    if len({candidate.parameter_hash for candidate in result}) != len(result):
        raise ValueError("Conditional grid contains duplicate physical parameter hashes")
    return result


def conditional_dry_run_summary(resolved: ResolvedExperiment) -> dict[str, object]:
    candidates = conditional_candidates(resolved)
    reuse = resolved.config["sweep"]["conditional"]["reuse"]
    return {
        "dataset_id": resolved.config["data"]["dataset_id"],
        "partitions": ["train", "validation"],
        "sealed_test": True,
        "test_partition_used": False,
        "wind_mode": resolved.config["airflow"]["main_mode"],
        "phase_convention": (
            "mechanical_phase_rad: 0 deg horizontal starting upstroke; "
            "90 deg top; 180 deg downstroke through horizontal; 270 deg bottom"
        ),
        "force_convention": resolved.config["physics"]["force_sign"],
        "conditional_parameter_combinations": len(candidates),
        "expected_reused_unique": int(reuse["expected_existing_unique"]),
        "expected_new_unique": int(reuse["expected_missing_unique"]),
        "output_root": str(resolved.output_root),
        "boundary_diagnostic_enabled": bool(
            resolved.config["sweep"]["boundary_diagnostic"]["enabled"]
        ),
    }


def seed_existing_results(resolved: ResolvedExperiment) -> dict[str, int]:
    """Seed the resume log from exact v1 physical hashes, failing on drift."""
    candidates = conditional_candidates(resolved)
    requested = {candidate.parameter_hash: candidate for candidate in candidates}
    reuse = resolved.config["sweep"]["conditional"]["reuse"]
    source_root = (resolved.project_root / str(reuse["source_root"])).resolve()
    source_records: dict[str, dict[str, object]] = {}
    for stage in reuse["source_stages"]:
        path = source_root / "logs" / f"{stage}_compact_results.jsonl"
        for key, value in _read_jsonl(path).items():
            source_records[key] = value
    selected: list[dict[str, object]] = []
    for key in sorted(requested.keys() & source_records.keys()):
        candidate = requested[key]
        source = source_records[key]
        metrics = []
        for raw_metric in source["metrics"]:
            metric = dict(raw_metric)
            metric.update({"parameter_hash": key, **asdict(candidate)})
            metrics.append(metric)
        diagnostics = dict(source["diagnostics"])
        diagnostics.update(
            {
                "reused_exact_parameter_hash": True,
                "reuse_source_experiment": "quadratic2_twist_sweep_v1",
                "reuse_source_stage": source["candidate"]["stage"],
            }
        )
        selected.append(
            {
                "parameter_hash": key,
                "candidate": asdict(candidate),
                "metrics": metrics,
                "diagnostics": diagnostics,
            }
        )
    expected_existing = int(reuse["expected_existing_unique"])
    expected_missing = int(reuse["expected_missing_unique"])
    if len(selected) != expected_existing:
        raise RuntimeError(
            f"Existing-result coverage drift: {len(selected)} != {expected_existing}"
        )
    if len(candidates) - len(selected) != expected_missing:
        raise RuntimeError(
            f"Missing-result count drift: {len(candidates) - len(selected)} != {expected_missing}"
        )
    log_dir = resolved.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "conditional_compact_results.jsonl"
    if path.exists():
        existing = _read_jsonl(path)
        if not set(record["parameter_hash"] for record in selected).issubset(existing):
            raise RuntimeError("Existing conditional resume log does not contain all seed hashes")
    else:
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in selected),
            encoding="utf-8",
        )
        temporary.replace(path)
    coverage = pd.DataFrame(
        [
            {
                **asdict(candidate),
                "parameter_hash": candidate.parameter_hash,
                "reused": candidate.parameter_hash in source_records,
            }
            for candidate in candidates
        ]
    )
    coverage.to_csv(resolved.output_root / "existing_coverage.csv", index=False)
    return {
        "target_unique": len(candidates),
        "reused_unique": len(selected),
        "missing_unique": len(candidates) - len(selected),
    }


def run_conditional(
    resolved: ResolvedExperiment, *, workers: int, resume: bool
) -> pd.DataFrame:
    seed_existing_results(resolved)
    result = run_candidates(
        resolved,
        conditional_candidates(resolved),
        stage="conditional",
        workers=workers,
        # The exact v1 seed is itself a valid completed resume log.
        resume=True,
    )
    path = resolved.output_root / "conditional_grid_results.parquet"
    result.to_parquet(path, index=False)
    return result


def _signed_delta_deg(model_rad: pd.Series, data_rad: pd.Series) -> np.ndarray:
    delta = np.degrees(model_rad.to_numpy(dtype=float) - data_rad.to_numpy(dtype=float))
    return (delta + 180.0) % 360.0 - 180.0


def conditional_wide(metrics: pd.DataFrame) -> pd.DataFrame:
    wide = _wide_metrics(metrics)
    for partition in ("train", "validation"):
        fx = metrics.loc[
            (metrics["partition"] == partition) & (metrics["component"] == "fx")
        ].drop_duplicates("parameter_hash")
        derived = pd.DataFrame(
            {
                "parameter_hash": fx["parameter_hash"].to_numpy(),
                f"{partition}_fx_first_harmonic_signed_delta_deg": _signed_delta_deg(
                    fx["model_first_harmonic_phase_rad"],
                    fx["data_first_harmonic_phase_rad"],
                ),
                f"{partition}_fx_peak_signed_delta_deg": (
                    (
                        fx["model_primary_peak_phase_deg"].to_numpy(dtype=float)
                        - fx["data_primary_peak_phase_deg"].to_numpy(dtype=float)
                        + 180.0
                    )
                    % 360.0
                    - 180.0
                ),
                f"{partition}_fx_peak_smooth_signed_delta_deg": (
                    (
                        fx["model_primary_peak_phase_smooth_deg"].to_numpy(dtype=float)
                        - fx["data_primary_peak_phase_smooth_deg"].to_numpy(dtype=float)
                        + 180.0
                    )
                    % 360.0
                    - 180.0
                ),
                f"{partition}_fx_xcorr_signed_lag_deg": (
                    (
                        fx["circular_xcorr_lag_deg"].to_numpy(dtype=float)
                        + 180.0
                    )
                    % 360.0
                    - 180.0
                ),
            }
        )
        wide = wide.merge(derived, on="parameter_hash", validate="one_to_one")
    return wide


def _grid_values(
    frame: pd.DataFrame, amplitude: float, column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = frame.loc[np.isclose(frame["A_tip_deg"], amplitude)]
    pivot = subset.pivot(index="kappa", columns="psi_theta_deg", values=column)
    pivot = pivot.sort_index().sort_index(axis=1)
    if pivot.isna().any().any():
        raise ValueError(f"Incomplete conditional map for A_tip={amplitude}, {column}")
    return (
        pivot.columns.to_numpy(dtype=float),
        pivot.index.to_numpy(dtype=float),
        pivot.to_numpy(dtype=float),
    )


def _plot_map_row(
    wide: pd.DataFrame,
    *,
    column: str,
    title: str,
    output: Path,
    dpi: int,
    cmap: str,
    centered: bool = False,
) -> Path:
    amplitudes = sorted(wide["A_tip_deg"].unique())
    arrays = [_grid_values(wide, amplitude, column)[2] for amplitude in amplitudes]
    finite = np.concatenate([array[np.isfinite(array)] for array in arrays])
    norm = None
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if centered:
        limit = max(abs(vmin), abs(vmax), 1.0e-9)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(
        1, len(amplitudes), figsize=(4.0 * len(amplitudes), 3.8),
        sharex=True, sharey=True, constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    image = None
    for axis, amplitude, values in zip(axes_array, amplitudes, arrays):
        x, y, _ = _grid_values(wide, amplitude, column)
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=(x[0] - 2.5, x[-1] + 2.5, y[0] - 0.125, y[-1] + 0.125),
            cmap=cmap,
            norm=norm,
            vmin=None if norm is not None else vmin,
            vmax=None if norm is not None else vmax,
            interpolation="nearest",
        )
        axis.set_title(rf"$A_{{tip}}={amplitude:g}^\circ$")
        axis.set_xlabel(r"twist timing $\psi_\theta$ [deg mechanical]")
        axis.set_xticks([-60, -40, -20, 0, 10])
        axis.grid(False)
    axes_array[0].set_ylabel(r"span curvature $\kappa$")
    assert image is not None
    fig.colorbar(image, ax=axes_array, shrink=0.88, label=title)
    fig.suptitle(
        f"{title}\nbody FRD, current EKF wind, 72 mechanical-phase bins, "
        "periodic zero-phase Savitzky-Golay (7,3)"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def _plot_pareto(wide: pd.DataFrame, output: Path, *, dpi: int) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    colors = {20.0: "#0072B2", 30.0: "#E69F00", 35.0: "#009E73", 40.0: "#CC79A7"}
    for axis, partition in zip(axes, ("train", "validation")):
        xcol = f"{partition}_fx_primary_peak_phase_error_deg"
        ycol = f"{partition}_fz_rmse"
        for amplitude, group in wide.groupby("A_tip_deg"):
            axis.scatter(
                group[xcol], group[ycol], s=22, alpha=0.65,
                color=colors[float(amplitude)], label=rf"$A_{{tip}}={amplitude:g}^\circ$",
            )
        mask = _pareto_mask(
            wide[xcol].to_numpy(dtype=float), wide[ycol].to_numpy(dtype=float)
        )
        axis.scatter(
            wide.loc[mask, xcol], wide.loc[mask, ycol],
            s=58, facecolors="none", edgecolors="black", linewidths=1.2,
            label="global Pareto",
        )
        axis.set_title(partition)
        axis.set_xlabel(r"absolute $F_x$ peak phase error [deg mechanical]")
        axis.set_ylabel(r"body-axis $F_z$ RMSE [N]")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Conditional Quadratic-2 phase/Fz tradeoff, current EKF wind")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def _best_by_amplitude(wide: pd.DataFrame, partition: str) -> pd.DataFrame:
    columns = [
        f"{partition}_fx_primary_peak_phase_error_deg",
        f"{partition}_fx_first_harmonic_signed_delta_deg",
        f"{partition}_fx_xcorr_signed_lag_deg",
        f"{partition}_fz_rmse",
        f"{partition}_fx_rmse",
    ]
    pieces = []
    for _, group in wide.groupby("A_tip_deg", sort=True):
        ordered = group.assign(
            _harmonic_abs=np.abs(group[columns[1]]),
            _xcorr_abs=np.abs(group[columns[2]]),
        ).sort_values(
            [columns[0], "_harmonic_abs", "_xcorr_abs", columns[3], columns[4]],
            kind="stable",
        )
        pieces.append(ordered.head(1))
    return pd.concat(pieces, ignore_index=True).drop(columns=["_harmonic_abs", "_xcorr_abs"])


def _boundary_decision(wide: pd.DataFrame, resolved: ResolvedExperiment) -> dict[str, object]:
    trigger = resolved.config["sweep"]["boundary_diagnostic"]["trigger"]
    baseline = _wide_metrics(
        pd.read_csv(resolved.output_root / "baseline_metrics.csv")
    ).iloc[0]
    maximum_fz = float(baseline["validation_fz_rmse"]) * (
        1.0 + float(trigger["maximum_validation_fz_rmse_degradation_fraction"])
    )
    feasible = wide.loc[
        (wide["validation_fz_rmse"] <= maximum_fz)
        & np.isfinite(wide["validation_fx_first_harmonic_signed_delta_deg"])
        & np.isfinite(wide["validation_fx_xcorr_signed_lag_deg"])
    ].copy()
    robust_by_amplitude: dict[str, float] = {}
    for amplitude in (35.0, 40.0):
        pool = feasible.loc[np.isclose(feasible["A_tip_deg"], amplitude)]
        if pool.empty:
            robust_by_amplitude[str(int(amplitude))] = float("nan")
        else:
            robust = np.maximum.reduce(
                [
                    pool["validation_fx_primary_peak_phase_error_deg"].to_numpy(
                        dtype=float
                    ),
                    np.abs(
                        pool[
                            "validation_fx_first_harmonic_signed_delta_deg"
                        ].to_numpy(dtype=float)
                    ),
                    np.abs(
                        pool["validation_fx_xcorr_signed_lag_deg"].to_numpy(
                            dtype=float
                        )
                    ),
                ]
            )
            robust_by_amplitude[str(int(amplitude))] = float(np.min(robust))
    gain = robust_by_amplitude["35"] - robust_by_amplitude["40"]
    threshold = float(trigger["minimum_robust_phase_gain_over_A35_deg"])
    run = bool(np.isfinite(gain) and gain >= threshold)
    return {
        "run_boundary_diagnostic": run,
        "reason": (
            "A40 robust phase gain over A35 exceeds preregistered threshold"
            if run
            else "A40 does not improve robust phase metrics over A35 by the preregistered threshold"
        ),
        "A35_best_robust_phase_error_deg": robust_by_amplitude["35"],
        "A40_best_robust_phase_error_deg": robust_by_amplitude["40"],
        "A40_gain_over_A35_deg": gain,
        "threshold_deg": threshold,
        "maximum_validation_fz_rmse": maximum_fz,
    }


def run_conditional_report(resolved: ResolvedExperiment) -> dict[str, object]:
    metrics_path = resolved.output_root / "conditional_grid_results.parquet"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = pd.read_parquet(metrics_path)
    candidates = conditional_candidates(resolved)
    if metrics["parameter_hash"].nunique() != len(candidates):
        raise RuntimeError("Conditional results are incomplete")
    wide = conditional_wide(metrics)
    wide.to_csv(resolved.output_root / "conditional_grid_summary.csv", index=False)
    fx_phase, balanced, physical = build_shortlists(resolved, metrics)
    fx_phase.to_csv(resolved.output_root / "shortlist_fx_phase.csv", index=False)
    balanced.to_csv(resolved.output_root / "shortlist_balanced.csv", index=False)
    physical.to_csv(resolved.output_root / "shortlist_physical.csv", index=False)
    train_best = _best_by_amplitude(wide, "train")
    validation_best = _best_by_amplitude(wide, "validation")
    train_best.to_csv(resolved.output_root / "best_by_amplitude_train.csv", index=False)
    validation_best.to_csv(
        resolved.output_root / "best_by_amplitude_validation.csv", index=False
    )
    dpi = int(resolved.config["output"]["figure_dpi"])
    figures = resolved.output_root / "figures"
    plot_specs = []
    for partition in ("train", "validation"):
        plot_specs.extend(
            [
                (
                    f"{partition}_fx_model_primary_peak_phase_deg",
                    f"{partition}: body-axis Fx primary peak phase [deg mechanical]",
                    "viridis", False,
                ),
                (
                    f"{partition}_fx_peak_signed_delta_deg",
                    f"{partition}: signed Fx peak phase error [deg mechanical]",
                    "coolwarm", True,
                ),
                (
                    f"{partition}_fx_first_harmonic_signed_delta_deg",
                    f"{partition}: signed Fx first-harmonic phase error [deg]",
                    "coolwarm", True,
                ),
                (
                    f"{partition}_fx_xcorr_signed_lag_deg",
                    f"{partition}: Fx circular cross-correlation lag [deg]",
                    "coolwarm", True,
                ),
                (
                    f"{partition}_fx_rmse",
                    f"{partition}: body-axis Fx RMSE [N]",
                    "magma", False,
                ),
                (
                    f"{partition}_fz_rmse",
                    f"{partition}: body-axis Fz RMSE [N]",
                    "magma", False,
                ),
                (
                    f"{partition}_fz_minimum_amplitude_error_abs",
                    f"{partition}: body-axis Fz minimum amplitude error [N]",
                    "magma", False,
                ),
            ]
        )
    generated = []
    for column, title, cmap, centered in plot_specs:
        path = figures / f"{column}.png"
        generated.append(
            _plot_map_row(
                wide, column=column, title=title, output=path, dpi=dpi,
                cmap=cmap, centered=centered,
            )
        )
    generated.append(_plot_pareto(wide, figures / "conditional_pareto.png", dpi=dpi))
    wind_input = (
        wide.loc[
            wide["A_tip_deg"].isin([35.0, 40.0])
            & (wide["train_fx_primary_peak_phase_error_deg"] <= 5.0 + 1.0e-12)
            & (wide["validation_fx_primary_peak_phase_error_deg"] <= 5.0 + 1.0e-12)
        ]
        .sort_values(
            ["A_tip_deg", "validation_fz_rmse", "validation_fx_rmse"],
            kind="stable",
        )
        .groupby("A_tip_deg", as_index=False)
        .head(1)
    )
    if set(wind_input["A_tip_deg"]) != {35.0, 40.0}:
        raise RuntimeError("Could not form the preregistered A35/A40 wind shortlist")
    rows = load_experiment_rows(
        resolved, airflow_mode=str(resolved.config["airflow"]["main_mode"])
    )
    zero_metrics = _zero_wind_shortlist(resolved, rows, wind_input)
    zero_metrics.to_csv(
        resolved.output_root / "zero_wind_conditional_shortlist.csv", index=False
    )
    generated.append(
        _plot_wind_sensitivity(
            wind_input,
            zero_metrics,
            figures / "conditional_shortlist_current_ekf_vs_zero_wind.png",
            dpi=dpi,
        )
    )
    zero_wide = _wide_metrics(zero_metrics).merge(
        zero_metrics[["parameter_hash", "source_parameter_hash"]].drop_duplicates(),
        on="parameter_hash",
        validate="one_to_one",
    )
    wind_pairs = wind_input[
        [
            "parameter_hash",
            "A_tip_deg",
            "kappa",
            "psi_theta_deg",
            "validation_fx_primary_peak_phase_error_deg",
            "validation_fx_rmse",
            "validation_fz_rmse",
        ]
    ].merge(
        zero_wide[
            [
                "source_parameter_hash",
                "validation_fx_primary_peak_phase_error_deg",
                "validation_fx_rmse",
                "validation_fz_rmse",
            ]
        ],
        left_on="parameter_hash",
        right_on="source_parameter_hash",
        suffixes=("_current_ekf", "_zero_wind"),
        validate="one_to_one",
    )
    boundary = _boundary_decision(wide, resolved)
    coverage = seed_existing_results(resolved)
    summary = {
        **_manifest_base(resolved),
        "stage": "conditional_report",
        "grid": {
            "A_tip_deg": sorted(float(value) for value in wide["A_tip_deg"].unique()),
            "kappa": sorted(float(value) for value in wide["kappa"].unique()),
            "psi_theta_deg": sorted(
                float(value) for value in wide["psi_theta_deg"].unique()
            ),
            "unique_candidates": int(wide["parameter_hash"].nunique()),
            **coverage,
        },
        "metrics": {
            "peak_definition": "maximum body-axis Fx in mechanical phase [180,270] deg",
            "robust_phase_metrics": [
                "signed first-harmonic phase error",
                "signed circular cross-correlation lag",
            ],
            "smoothing": dict(resolved.config["phase_binning"]["smoothing"]),
        },
        "best_by_amplitude_train": train_best.to_dict(orient="records"),
        "best_by_amplitude_validation": validation_best.to_dict(orient="records"),
        "boundary_diagnostic_decision": boundary,
        "zero_wind_shortlist": wind_pairs.to_dict(orient="records"),
        "figures": [str(path.relative_to(resolved.output_root)) for path in generated],
        "result_sha256": sha256_file(metrics_path),
        "test_partition_loaded": False,
        "test_rows_loaded": 0,
    }
    _write_json(resolved.output_root / "conditional_manifest.json", summary)
    _write_json(resolved.output_root / "manifest.json", summary)
    return summary
