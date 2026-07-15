"""Real-log wing-only DeLaurier dynamic-twist sensitivity analysis.

This module compares a ``wing-only DeLaurier baseline`` with a
``total reconstructed effective-wrench label``. Those names are deliberate:
the label also contains tail, fuselage, disturbance, asymmetry, and label
reconstruction effects.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from system_identification.baselines.isaaclab_wing_only_baseline import (
    AIRFLOW_MODES,
    ATTITUDE_AIRFLOW_REQUIRED_COLUMNS,
    ISAACLAB_SOURCE_BRANCH,
    ISAACLAB_SOURCE_COMMIT,
    ISAACLAB_SOURCE_REPOSITORY,
    TARGETS,
    WingOnlyBaselineConfig,
    baseline_config_from_aircraft_metadata,
    evaluate_wing_only_delaurier_segment,
    required_columns_for_airflow_mode,
)
from system_identification.plotting import split_frame_on_plot_breaks


TARGET_LABELS = {
    "fx_b": "Fx",
    "fy_b": "Fy",
    "fz_b": "Fz",
    "mx_b": "Mx",
    "my_b": "My",
    "mz_b": "Mz",
}
TARGET_UNITS = {target: ("N" if target.startswith("f") else "N m") for target in TARGETS}
THETA_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#000000")
LABEL_COLOR = "#202124"


def load_canonical_samples(
    *, samples_parquet: str | Path | None = None, split_root: str | Path | None = None, partition: str = "test"
) -> tuple[pd.DataFrame, Path]:
    """Load one canonical sample table without reimplementing partition semantics."""

    if (samples_parquet is None) == (split_root is None):
        raise ValueError("Provide exactly one of samples_parquet or split_root")
    path = Path(samples_parquet) if samples_parquet is not None else Path(split_root) / f"{partition}_samples.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Canonical samples parquet not found: {path}")
    return pd.read_parquet(path), path.resolve()


def _valid_window_frame(
    frame: pd.DataFrame,
    *,
    airflow_mode: str = "legacy_scalar_true_airspeed",
) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for column in ("label_valid", "phase_valid", "cycle_valid"):
        if column in frame.columns:
            mask &= frame[column].fillna(False).to_numpy(dtype=bool)
    for column in (
        "mechanical_phase_rad",
        "flap_frequency_hz",
        "airspeed_validated.true_airspeed_m_s",
        "vehicle_air_data.rho",
        *TARGETS,
        *required_columns_for_airflow_mode(airflow_mode),
    ):
        if column not in frame.columns:
            raise ValueError(f"Canonical samples are missing required column: {column}")
        mask &= np.isfinite(frame[column].to_numpy(dtype=float))
    if "vehicle_land_detected.landed" in frame.columns:
        mask &= ~frame["vehicle_land_detected.landed"].fillna(True).to_numpy(dtype=bool)
    if airflow_mode == "attitude_ground_wind_3d":
        quaternion = frame[[f"vehicle_attitude.q[{index}]" for index in range(4)]].to_numpy(dtype=float)
        mask &= np.linalg.norm(quaternion, axis=1) > 1.0e-12
        for column in (
            "wind.windspeed_north_valid",
            "wind.windspeed_east_valid",
            "vehicle_local_position.v_xy_valid",
            "vehicle_local_position.v_z_valid",
        ):
            if column in frame.columns:
                mask &= frame[column].fillna(False).to_numpy(dtype=bool)
    return frame.loc[mask]


def _candidate_windows(
    samples: pd.DataFrame,
    *,
    duration_s: float,
    stride_s: float,
    airflow_mode: str = "legacy_scalar_true_airspeed",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (log_id, segment_id), group in samples.groupby(["log_id", "segment_id"], sort=True, dropna=False):
        group = group.sort_values("time_s", kind="stable").reset_index(drop=True)
        if len(group) < 3:
            continue
        time_values = group["time_s"].to_numpy(dtype=float)
        start = float(time_values[0])
        last_start = float(time_values[-1]) - float(duration_s)
        while start <= last_start + 1.0e-9:
            stop = start + float(duration_s)
            window = group.loc[(group["time_s"] >= start) & (group["time_s"] <= stop)]
            valid = _valid_window_frame(window, airflow_mode=airflow_mode)
            if len(window) >= 8 and len(valid) == len(window):
                delta = np.diff(window["time_s"].to_numpy(dtype=float))
                positive = delta[np.isfinite(delta) & (delta > 0.0)]
                gap_ok = len(positive) > 0 and float(np.max(positive)) <= 8.0 * float(np.median(positive))
                cycles = valid["cycle_id"].nunique() if "cycle_id" in valid.columns else 0
                if gap_ok and cycles >= 3:
                    airspeed = valid["airspeed_validated.true_airspeed_m_s"].to_numpy(dtype=float)
                    pitch_rate = (
                        valid["vehicle_angular_velocity.xyz[1]"].to_numpy(dtype=float)
                        if "vehicle_angular_velocity.xyz[1]" in valid.columns
                        else np.zeros(len(valid))
                    )
                    yaw_rate = (
                        valid["vehicle_angular_velocity.xyz[2]"].to_numpy(dtype=float)
                        if "vehicle_angular_velocity.xyz[2]" in valid.columns
                        else np.zeros(len(valid))
                    )
                    rows.append(
                        {
                            "log_id": str(log_id),
                            "segment_id": segment_id,
                            "t_start_s": float(valid["time_s"].iloc[0]),
                            "t_end_s": float(valid["time_s"].iloc[-1]),
                            "sample_count": int(len(valid)),
                            "cycle_count": int(cycles),
                            "airspeed_mean_m_s": float(np.mean(airspeed)),
                            "airspeed_std_m_s": float(np.std(airspeed)),
                            "pitch_rate_rms_rad_s": float(np.sqrt(np.mean(np.square(pitch_rate)))),
                            "yaw_rate_rms_rad_s": float(np.sqrt(np.mean(np.square(yaw_rate)))),
                        }
                    )
            start += float(stride_s)
    candidates = pd.DataFrame(rows)
    if not candidates.empty:
        candidates["stable_score"] = (
            candidates["airspeed_std_m_s"]
            + candidates["pitch_rate_rms_rad_s"]
            + candidates["yaw_rate_rms_rad_s"]
        )
    return candidates


def _overlaps(selected: list[pd.Series], candidate: pd.Series) -> bool:
    for row in selected:
        if row["log_id"] != candidate["log_id"] or row["segment_id"] != candidate["segment_id"]:
            continue
        overlap = min(float(row["t_end_s"]), float(candidate["t_end_s"])) - max(
            float(row["t_start_s"]), float(candidate["t_start_s"])
        )
        if overlap > 0.25 * (float(candidate["t_end_s"]) - float(candidate["t_start_s"])):
            return True
    return False


def select_representative_windows(
    samples: pd.DataFrame,
    *,
    window_count: int = 5,
    duration_s: float = 4.0,
    stride_s: float = 2.0,
    airflow_mode: str = "legacy_scalar_true_airspeed",
) -> pd.DataFrame:
    """Deterministically select diverse valid windows without using residuals."""

    if not 4 <= int(window_count) <= 6:
        raise ValueError("window_count must be between 4 and 6")
    candidates = _candidate_windows(
        samples,
        duration_s=duration_s,
        stride_s=stride_s,
        airflow_mode=airflow_mode,
    )
    if len(candidates) < int(window_count):
        raise ValueError(f"Only {len(candidates)} valid candidate windows; need {window_count}")
    stable_limit = float(candidates["stable_score"].median())
    candidates["stable_candidate"] = candidates["stable_score"] <= stable_limit
    candidates["pitch_mild_distance"] = np.abs(
        candidates["pitch_rate_rms_rad_s"] - float(candidates["pitch_rate_rms_rad_s"].quantile(0.75))
    )
    candidates["turn_mild_distance"] = np.abs(
        candidates["yaw_rate_rms_rad_s"] - float(candidates["yaw_rate_rms_rad_s"].quantile(0.75))
    )
    roles = [
        ("steady straight flight", "stable_score", True, False),
        ("lower-airspeed stable flight", "airspeed_mean_m_s", True, True),
        ("higher-airspeed stable flight", "airspeed_mean_m_s", False, True),
        ("mild pitch variation", "pitch_mild_distance", True, False),
        ("mild turn", "turn_mild_distance", True, False),
        ("additional stable flight", "stable_score", True, True),
    ]
    selected: list[pd.Series] = []
    descriptions: list[str] = []
    for description, column, ascending, stable_only in roles[: int(window_count)]:
        pools = [candidates.loc[candidates["stable_candidate"]].copy(), candidates] if stable_only else [candidates]
        chosen = False
        for pool in pools:
            ordered = pool.sort_values(
                [column, "log_id", "segment_id", "t_start_s"],
                ascending=[ascending, True, True, True],
                kind="stable",
            )
            for _, candidate in ordered.iterrows():
                if not _overlaps(selected, candidate):
                    selected.append(candidate)
                    descriptions.append(description)
                    chosen = True
                    break
            if chosen:
                break
    if len(selected) < int(window_count):
        raise ValueError(f"Could select only {len(selected)} non-overlapping representative windows")
    result = pd.DataFrame(selected).reset_index(drop=True)
    result.insert(0, "window_id", [f"w{index:02d}" for index in range(1, len(result) + 1)])
    result["description"] = descriptions
    result["selection_mode"] = "deterministic_auto"
    result["selection_rule"] = (
        "4 s candidates on 2 s grid; all label/phase/cycle valid, airborne, gap <= 8x median dt, >=3 cycles; "
        "ranked without wrench residuals; low/high speed restricted to the better half of stability scores, "
        "mild pitch/turn selected nearest the channel 75th percentile"
    )
    return result


def load_window_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"window_id", "log_id", "segment_id", "t_start_s", "t_end_s", "description"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Window manifest missing columns: {missing}")
    if manifest["window_id"].duplicated().any():
        raise ValueError("window_id must be unique")
    if np.any(manifest["t_end_s"].to_numpy(dtype=float) <= manifest["t_start_s"].to_numpy(dtype=float)):
        raise ValueError("Every window must have t_end_s > t_start_s")
    manifest = manifest.copy()
    manifest["selection_mode"] = "provided_manifest"
    manifest["selection_rule"] = "user-provided manifest; validated against canonical segment and validity contracts"
    return manifest


def validate_windows(
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    airflow_mode: str = "legacy_scalar_true_airspeed",
) -> None:
    for window in windows.itertuples(index=False):
        group = samples.loc[
            (samples["log_id"].astype(str) == str(window.log_id))
            & (samples["segment_id"] == window.segment_id)
            & (samples["time_s"] >= float(window.t_start_s))
            & (samples["time_s"] <= float(window.t_end_s))
        ]
        if group.empty:
            raise ValueError(f"Window {window.window_id} has no samples")
        if len(_valid_window_frame(group, airflow_mode=airflow_mode)) != len(group):
            raise ValueError(f"Window {window.window_id} contains invalid label/phase/cycle/airborne samples")
        delta = np.diff(group.sort_values("time_s")["time_s"].to_numpy(dtype=float))
        if len(delta) and np.max(delta) > 8.0 * np.median(delta[delta > 0.0]):
            raise ValueError(f"Window {window.window_id} crosses a large time gap")
        if "cycle_id" in group.columns and group["cycle_id"].nunique() < 3:
            raise ValueError(f"Window {window.window_id} contains fewer than three wingbeat cycles")


def evaluate_selected_windows(
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    theta_tip_deg: Sequence[float],
    geometry_path: str | Path,
    config: WingOnlyBaselineConfig | None = None,
) -> pd.DataFrame:
    """Evaluate each full selected segment once, then crop requested windows."""

    resolved_config = config or WingOnlyBaselineConfig()
    validate_windows(samples, windows, airflow_mode=str(resolved_config.airflow_mode))
    outputs: list[pd.DataFrame] = []
    group_columns = windows[["log_id", "segment_id"]].drop_duplicates()
    evaluated: dict[tuple[str, object], pd.DataFrame] = {}
    for row in group_columns.itertuples(index=False):
        segment = samples.loc[
            (samples["log_id"].astype(str) == str(row.log_id)) & (samples["segment_id"] == row.segment_id)
        ].copy()
        evaluated[(str(row.log_id), row.segment_id)] = evaluate_wing_only_delaurier_segment(
            segment,
            theta_tip_deg=theta_tip_deg,
            geometry_path=geometry_path,
            config=resolved_config,
            phase_acceleration_mode="constant_frequency_step",
        )
    for window in windows.itertuples(index=False):
        full_segment = evaluated[(str(window.log_id), window.segment_id)]
        cropped = full_segment.loc[
            (full_segment["time_s"] >= float(window.t_start_s))
            & (full_segment["time_s"] <= float(window.t_end_s))
        ].copy()
        cropped.insert(0, "window_id", str(window.window_id))
        cropped["window_description"] = str(window.description)
        outputs.append(cropped)
    aligned = pd.concat(outputs, ignore_index=True)
    expected_rows = sum(
        len(
            samples.loc[
                (samples["log_id"].astype(str) == str(row.log_id))
                & (samples["segment_id"] == row.segment_id)
                & (samples["time_s"] >= float(row.t_start_s))
                & (samples["time_s"] <= float(row.t_end_s))
            ]
        )
        for row in windows.itertuples(index=False)
    ) * len(theta_tip_deg)
    if len(aligned) != expected_rows:
        raise RuntimeError(f"Long-format row count mismatch: {len(aligned)} != {expected_rows}")
    return aligned


def _metric_row(true_values: np.ndarray, baseline_values: np.ndarray) -> dict[str, object]:
    valid = np.isfinite(true_values) & np.isfinite(baseline_values)
    true_values = true_values[valid]
    baseline_values = baseline_values[valid]
    if len(true_values) == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "correlation": np.nan,
            "correlation_reason": "no_valid_pairs",
            "label_std": np.nan,
            "baseline_std": np.nan,
            "amplitude_ratio": np.nan,
            "valid_sample_count": 0,
        }
    residual = baseline_values - true_values
    label_std = float(np.std(true_values))
    baseline_std = float(np.std(baseline_values))
    if label_std <= 1.0e-15 or baseline_std <= 1.0e-15:
        correlation = np.nan
        reason = "constant_label" if label_std <= 1.0e-15 else "constant_baseline"
    else:
        correlation = float(np.corrcoef(true_values, baseline_values)[0, 1])
        reason = ""
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "correlation": correlation,
        "correlation_reason": reason,
        "label_std": label_std,
        "baseline_std": baseline_std,
        "amplitude_ratio": baseline_std / label_std if label_std > 1.0e-15 else np.nan,
        "valid_sample_count": int(len(true_values)),
    }


def compute_window_metrics(aligned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (window_id, theta), group in aligned.groupby(["window_id", "theta_tip_deg"], sort=True):
        for target in TARGETS:
            rows.append(
                {
                    "window_id": window_id,
                    "theta_tip_deg": float(theta),
                    "target": target,
                    "metric_domain": "raw_sample",
                    **_metric_row(
                        group[f"true_{target}"].to_numpy(dtype=float), group[f"pred_{target}"].to_numpy(dtype=float)
                    ),
                }
            )
    return pd.DataFrame(rows)


def compute_cycle_mean_metrics(aligned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_columns = [f"{prefix}_{target}" for prefix in ("true", "pred") for target in TARGETS]
    cycle_means = (
        aligned.dropna(subset=["cycle_id"])
        .groupby(["window_id", "theta_tip_deg", "cycle_id"], sort=True, as_index=False)[value_columns]
        .mean()
    )
    rows: list[dict[str, object]] = []
    for (window_id, theta), group in cycle_means.groupby(["window_id", "theta_tip_deg"], sort=True):
        for target in TARGETS:
            rows.append(
                {
                    "window_id": window_id,
                    "theta_tip_deg": float(theta),
                    "target": target,
                    "metric_domain": "per_cycle_mean",
                    **_metric_row(
                        group[f"true_{target}"].to_numpy(dtype=float), group[f"pred_{target}"].to_numpy(dtype=float)
                    ),
                }
            )
    return pd.DataFrame(rows), cycle_means


def compute_phase_binned_curves(aligned: pd.DataFrame, *, phase_bins: int = 72) -> pd.DataFrame:
    if int(phase_bins) <= 0:
        raise ValueError("phase_bins must be positive")
    work = aligned.copy()
    work["phase_wrapped_rad"] = np.mod(work["mechanical_phase_rad"].to_numpy(dtype=float), 2.0 * np.pi)
    bin_width = 2.0 * np.pi / int(phase_bins)
    work["phase_bin"] = np.minimum((work["phase_wrapped_rad"] / bin_width).astype(int), int(phase_bins) - 1)
    rows: list[dict[str, object]] = []
    for (window_id, theta), group in work.groupby(["window_id", "theta_tip_deg"], sort=True):
        grouped = group.groupby("phase_bin", sort=True)
        for bin_index in range(int(phase_bins)):
            bin_group = grouped.get_group(bin_index) if bin_index in grouped.groups else group.iloc[0:0]
            for target in TARGETS:
                true_values = bin_group[f"true_{target}"].to_numpy(dtype=float)
                pred_values = bin_group[f"pred_{target}"].to_numpy(dtype=float)
                true_valid = true_values[np.isfinite(true_values)]
                pred_valid = pred_values[np.isfinite(pred_values)]
                rows.append(
                    {
                        "window_id": window_id,
                        "theta_tip_deg": float(theta),
                        "target": target,
                        "phase_bin": bin_index,
                        "phase_bin_center_rad": (bin_index + 0.5) * bin_width,
                        "label_mean": float(np.mean(true_valid)) if len(true_valid) else np.nan,
                        "label_std": float(np.std(true_valid)) if len(true_valid) else np.nan,
                        "label_count": int(len(true_valid)),
                        "baseline_mean": float(np.mean(pred_valid)) if len(pred_valid) else np.nan,
                        "baseline_std": float(np.std(pred_valid)) if len(pred_valid) else np.nan,
                        "baseline_count": int(len(pred_valid)),
                    }
                )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=190, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_window_time_series(aligned: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for window_id, window in aligned.groupby("window_id", sort=True):
        thetas = sorted(window["theta_tip_deg"].unique())
        base = window.loc[window["theta_tip_deg"] == thetas[0]].sort_values("time_s")
        origin = float(base["time_s"].iloc[0])
        fig, axes = plt.subplots(6, 1, figsize=(13.5, 13.0), sharex=True)
        for axis, target in zip(axes, TARGETS):
            for part_index, part in enumerate(split_frame_on_plot_breaks(base)):
                axis.plot(
                    part["time_s"] - origin,
                    part[f"true_{target}"],
                    color=LABEL_COLOR,
                    linewidth=1.8,
                    label="total reconstructed effective-wrench label" if part_index == 0 else None,
                    zorder=5,
                )
            for theta_index, theta in enumerate(thetas):
                theta_frame = window.loc[window["theta_tip_deg"] == theta].sort_values("time_s")
                for part_index, part in enumerate(split_frame_on_plot_breaks(theta_frame)):
                    axis.plot(
                        part["time_s"] - origin,
                        part[f"pred_{target}"],
                        color=THETA_COLORS[theta_index % len(THETA_COLORS)],
                        linewidth=1.0,
                        alpha=0.95,
                        label=f"wing-only DeLaurier baseline, theta tip = {theta:g} deg" if part_index == 0 else None,
                    )
            axis.set_ylabel(f"{TARGET_LABELS[target]} ({TARGET_UNITS[target]})")
            axis.grid(True, alpha=0.22)
        axes[-1].set_xlabel("time since window start (s)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.995))
        description = str(window["window_description"].iloc[0])
        fig.suptitle(f"{window_id}: wing-only baseline vs total effective wrench — {description}", y=1.02)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = output_dir / f"{window_id}_six_axis.png"
        _save_figure(fig, path)
        paths.append(path)
    return paths


def plot_phase_folded(phase_binned: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for window_id, window in phase_binned.groupby("window_id", sort=True):
        thetas = sorted(window["theta_tip_deg"].unique())
        fig, axes = plt.subplots(6, 1, figsize=(12.5, 13.0), sharex=True)
        for axis, target in zip(axes, TARGETS):
            label = window.loc[(window["target"] == target) & (window["theta_tip_deg"] == thetas[0])]
            x_values = label["phase_bin_center_rad"].to_numpy(dtype=float)
            label_mean = label["label_mean"].to_numpy(dtype=float)
            label_std = label["label_std"].to_numpy(dtype=float)
            axis.plot(x_values, label_mean, color=LABEL_COLOR, linewidth=1.8, label="total reconstructed effective-wrench label")
            axis.fill_between(x_values, label_mean - label_std, label_mean + label_std, color=LABEL_COLOR, alpha=0.12)
            for theta_index, theta in enumerate(thetas):
                curve = window.loc[(window["target"] == target) & (window["theta_tip_deg"] == theta)]
                x_values = curve["phase_bin_center_rad"].to_numpy(dtype=float)
                mean = curve["baseline_mean"].to_numpy(dtype=float)
                std = curve["baseline_std"].to_numpy(dtype=float)
                color = THETA_COLORS[theta_index % len(THETA_COLORS)]
                axis.plot(x_values, mean, color=color, linewidth=1.05, label=f"wing-only baseline, {theta:g} deg")
                axis.fill_between(x_values, mean - std, mean + std, color=color, alpha=0.08)
            axis.set_ylabel(f"{TARGET_LABELS[target]} ({TARGET_UNITS[target]})")
            axis.grid(True, alpha=0.22)
        axes[-1].set_xlabel("canonical mechanical phase (rad)")
        axes[-1].set_xticks([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi])
        axes[-1].set_xticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"])
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.995))
        fig.suptitle(f"{window_id}: phase-folded wing-only baseline vs total effective wrench", y=1.02)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = output_dir / f"{window_id}_phase_folded_six_axis.png"
        _save_figure(fig, path)
        paths.append(path)
    return paths


def plot_theta_sensitivity(metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    metric_specs = (("rmse", "RMSE"), ("bias", "mean bias"), ("correlation", "Pearson correlation"))
    for target in TARGETS:
        subset = metrics.loc[metrics["target"] == target]
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), sharex=True)
        for axis, (metric, title) in zip(axes, metric_specs):
            for window_id, window in subset.groupby("window_id", sort=True):
                window = window.sort_values("theta_tip_deg")
                axis.plot(window["theta_tip_deg"], window[metric], marker="o", linewidth=0.9, alpha=0.5, label=str(window_id))
            summary = subset.groupby("theta_tip_deg", as_index=False)[metric].mean().sort_values("theta_tip_deg")
            axis.plot(summary["theta_tip_deg"], summary[metric], color="black", marker="s", linewidth=2.0, label="window mean")
            axis.axhline(0.0, color="#888888", linewidth=0.7, alpha=0.6) if metric in {"bias", "correlation"} else None
            axis.set_title(title)
            axis.set_xlabel("theta tip (deg)")
            axis.grid(True, alpha=0.22)
        axes[0].set_ylabel(TARGET_UNITS[target])
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), fontsize=8, bbox_to_anchor=(0.5, 1.04))
        fig.suptitle(f"{TARGET_LABELS[target]} theta-tip sensitivity: wing-only baseline vs total effective wrench", y=1.14)
        fig.tight_layout()
        path = output_dir / f"{target}_theta_sensitivity.png"
        _save_figure(fig, path)
        paths.append(path)
    return paths


def plot_component_breakdown(aligned: pd.DataFrame, windows: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, Path]:
    stable_window_id = str(windows.iloc[0]["window_id"])
    frame = aligned.loc[aligned["window_id"] == stable_window_id].copy()
    component_specs = (
        ("component_dN_c_fz_b", "dN_c contribution to Fz", "N"),
        ("component_dN_a_fz_b", "dN_a contribution to Fz", "N"),
        ("component_chordwise_fx_b", "chordwise-force contribution to Fx", "N"),
        ("component_dM_a_my_b", "dM_a contribution to My", "N m"),
        ("component_r_cross_f_my_b", "r cross F contribution to My", "N m"),
    )
    frame["component_chordwise_fx_b"] = (
        frame["component_dT_s_fx_b"] + frame["component_dD_camber_fx_b"] + frame["component_dD_f_fx_b"]
    )
    frame["phase_wrapped_rad"] = np.mod(frame["mechanical_phase_rad"], 2.0 * np.pi)
    contributions = frame[
        ["window_id", "log_id", "segment_id", "time_s", "mechanical_phase_rad", "phase_wrapped_rad", "theta_tip_deg"]
        + [column for column, _, _ in component_specs]
    ].copy()
    fig, axes = plt.subplots(len(component_specs), 1, figsize=(12.5, 10.5), sharex=True)
    for axis, (column, label, unit) in zip(axes, component_specs):
        for theta_index, (theta, group) in enumerate(frame.groupby("theta_tip_deg", sort=True)):
            ordered = group.sort_values("phase_wrapped_rad")
            axis.plot(
                ordered["phase_wrapped_rad"],
                ordered[column],
                color=THETA_COLORS[theta_index % len(THETA_COLORS)],
                linewidth=0.9,
                alpha=0.75,
                label=f"theta tip = {theta:g} deg",
            )
        axis.set_ylabel(f"{label}\n({unit})")
        axis.grid(True, alpha=0.22)
    axes[-1].set_xlabel("canonical mechanical phase (rad)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(f"{stable_window_id}: DeLaurier component breakdown", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = output_dir / f"{stable_window_id}_component_breakdown.png"
    _save_figure(fig, path)
    return contributions, path


def plot_airflow_diagnostics(aligned: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot attitude and reconstructed three-dimensional airflow per window."""

    paths: list[Path] = []
    for window_id, window in aligned.groupby("window_id", sort=True):
        theta = float(np.min(window["theta_tip_deg"]))
        frame = window.loc[window["theta_tip_deg"] == theta].sort_values("time_s")
        origin = float(frame["time_s"].iloc[0])
        time = frame["time_s"].to_numpy(dtype=float) - origin
        fig, axes = plt.subplots(4, 1, figsize=(12.5, 9.0), sharex=True)
        axes[0].plot(time, np.degrees(frame["attitude_pitch_rad"]), color="#0072B2", linewidth=1.3, label="aircraft pitch")
        axes[0].plot(time, np.degrees(frame["airflow_alpha_rad"]), color="#D55E00", linewidth=1.3, linestyle="--", label="airflow incidence alpha")
        axes[0].set_ylabel("angle (deg)")
        axes[0].legend(frameon=False, ncol=2)
        axes[1].plot(time, np.degrees(frame["airflow_beta_rad"]), color="#009E73", linewidth=1.2)
        axes[1].axhline(0.0, color="#777777", linewidth=0.7)
        axes[1].set_ylabel("sideslip beta (deg)")
        for column, label, color, linestyle in (
            ("airflow_body_u_m_s", "u body forward", "#0072B2", "-"),
            ("airflow_body_v_m_s", "v body right", "#E69F00", "--"),
            ("airflow_body_w_m_s", "w body down", "#CC79A7", "-."),
        ):
            axes[2].plot(time, frame[column], color=color, linewidth=1.1, linestyle=linestyle, label=label)
        axes[2].set_ylabel("air velocity (m/s)")
        axes[2].legend(frameon=False, ncol=3)
        axes[3].plot(time, frame["airflow_speed_m_s"], color="#009E73", linewidth=1.2, label="ground-minus-wind magnitude")
        axes[3].plot(time, frame["airflow_forward_speed_used_m_s"], color="#0072B2", linewidth=1.2, linestyle="--", label="body-forward U used")
        if "airspeed_validated.true_airspeed_m_s" in frame.columns:
            axes[3].plot(time, frame["airspeed_validated.true_airspeed_m_s"], color="#202124", linewidth=1.0, linestyle=":", label="logged true airspeed diagnostic")
        axes[3].set_ylabel("speed (m/s)")
        axes[3].set_xlabel("time since window start (s)")
        axes[3].legend(frameon=False, ncol=3)
        for axis in axes:
            axis.grid(True, alpha=0.2)
        fig.suptitle(f"{window_id}: attitude-aware airflow reconstruction")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = output_dir / f"{window_id}_attitude_airflow.png"
        _save_figure(fig, path)
        paths.append(path)
    return paths


def plot_airflow_mode_comparison(
    attitude_aligned: pd.DataFrame,
    legacy_aligned: pd.DataFrame,
    output_dir: Path,
    *,
    phase_bins: int = 72,
) -> list[Path]:
    """Compare legacy scalar and attitude-aware airflow for theta tip zero."""

    paths: list[Path] = []
    targets = ("fx_b", "fz_b", "my_b")
    for window_id in sorted(set(attitude_aligned["window_id"]) & set(legacy_aligned["window_id"])):
        attitude = attitude_aligned.loc[
            (attitude_aligned["window_id"] == window_id)
            & np.isclose(attitude_aligned["theta_tip_deg"], 0.0)
        ].sort_values("time_s")
        legacy = legacy_aligned.loc[
            (legacy_aligned["window_id"] == window_id)
            & np.isclose(legacy_aligned["theta_tip_deg"], 0.0)
        ].sort_values("time_s")
        if attitude.empty or legacy.empty:
            continue
        merged = attitude.merge(
            legacy[["time_s", *[f"pred_{target}" for target in targets]]],
            on="time_s",
            how="inner",
            suffixes=("_attitude", "_legacy"),
            validate="one_to_one",
        )
        origin = float(merged["time_s"].iloc[0])
        phase = np.mod(merged["mechanical_phase_rad"].to_numpy(dtype=float), 2.0 * np.pi)
        bin_width = 2.0 * np.pi / int(phase_bins)
        phase_bin = np.minimum((phase / bin_width).astype(int), int(phase_bins) - 1)
        fig, axes = plt.subplots(3, 2, figsize=(13.5, 9.0))
        for row_index, target in enumerate(targets):
            time = merged["time_s"].to_numpy(dtype=float) - origin
            axes[row_index, 0].plot(
                time,
                merged[f"true_{target}"],
                color=LABEL_COLOR,
                linewidth=1.8,
                label="total reconstructed effective-wrench label",
            )
            axes[row_index, 0].plot(
                time,
                merged[f"pred_{target}_legacy"],
                color="#7F7F7F",
                linewidth=1.1,
                linestyle="--",
                label="wing-only baseline: legacy scalar airflow",
            )
            axes[row_index, 0].plot(
                time,
                merged[f"pred_{target}_attitude"],
                color="#0072B2",
                linewidth=1.2,
                label="wing-only baseline: attitude-aware airflow",
            )
            phase_rows: list[dict[str, float]] = []
            for bin_index in range(int(phase_bins)):
                mask = phase_bin == bin_index
                phase_rows.append(
                    {
                        "phase": (bin_index + 0.5) * bin_width,
                        "label": float(np.mean(merged.loc[mask, f"true_{target}"])) if np.any(mask) else np.nan,
                        "legacy": float(np.mean(merged.loc[mask, f"pred_{target}_legacy"])) if np.any(mask) else np.nan,
                        "attitude": float(np.mean(merged.loc[mask, f"pred_{target}_attitude"])) if np.any(mask) else np.nan,
                    }
                )
            folded = pd.DataFrame(phase_rows)
            axes[row_index, 1].plot(folded["phase"], folded["label"], color=LABEL_COLOR, linewidth=1.8)
            axes[row_index, 1].plot(folded["phase"], folded["legacy"], color="#7F7F7F", linewidth=1.1, linestyle="--")
            axes[row_index, 1].plot(folded["phase"], folded["attitude"], color="#0072B2", linewidth=1.2)
            for axis in axes[row_index]:
                axis.set_ylabel(f"{TARGET_LABELS[target]} ({TARGET_UNITS[target]})")
                axis.grid(True, alpha=0.2)
        axes[0, 0].set_title("time series, theta tip = 0 deg")
        axes[0, 1].set_title("phase-folded mean, theta tip = 0 deg")
        axes[-1, 0].set_xlabel("time since window start (s)")
        axes[-1, 1].set_xlabel("canonical mechanical phase (rad)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
        fig.suptitle(f"{window_id}: airflow-contract comparison", y=1.02)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = output_dir / f"{window_id}_legacy_vs_attitude_airflow.png"
        _save_figure(fig, path)
        paths.append(path)
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(args: list[str], root: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def write_index(
    output_dir: Path,
    *,
    windows: pd.DataFrame,
    theta_tip_deg: Sequence[float],
    metrics: pd.DataFrame,
    airflow_mode: str,
    has_legacy_comparison: bool = False,
) -> None:
    mean_metrics = metrics.groupby(["theta_tip_deg", "target"], as_index=False)[["rmse", "bias", "correlation"]].mean()
    lines = [
        "# Wing-only DeLaurier wrench theta sweep",
        "",
        "Comparison: **wing-only DeLaurier baseline** vs **total reconstructed effective-wrench label**.",
        "",
        f"Airflow mode: `{airflow_mode}`.",
        "",
        f"Theta-tip sweep: {', '.join(f'{value:g} deg' for value in theta_tip_deg)}.",
        "",
        "## Windows",
        "",
        windows.to_markdown(index=False),
        "",
        "## Window-mean raw metrics",
        "",
        mean_metrics.to_markdown(index=False),
        "",
        "## Artifacts",
        "",
        "- `aligned_predictions.parquet`: long-format raw baseline and label rows",
        "- `window_metrics.csv`: raw-sample metrics per window, theta tip, and target",
        "- `cycle_mean_metrics.csv`: per-cycle-mean metrics",
        "- `phase_binned_curves.parquet`: mean, standard deviation, and count in each phase bin",
        "- `component_contributions.parquet`: aggregate physical components for the stable window",
        "- `figures/`: PNG and PDF figures",
        "- `figures/airflow_diagnostics/`: attitude, reconstructed body airflow, and airspeed diagnostics",
        *(
            ["- `figures/airflow_mode_comparison/`: legacy scalar-airspeed vs attitude-aware airflow comparison"]
            if has_legacy_comparison
            else []
        ),
        "",
        "No baseline time shift or arbitrary bandwidth-matching filter was applied.",
    ]
    (output_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_wing_wrench_theta_sweep(
    *,
    samples: pd.DataFrame,
    samples_path: Path,
    aircraft_metadata_path: Path,
    geometry_path: Path,
    theta_tip_deg: Sequence[float],
    output_dir: Path,
    window_manifest_path: Path | None = None,
    phase_bins: int = 72,
    auto_window_count: int = 5,
    command: str | None = None,
    airflow_mode: str = "legacy_scalar_true_airspeed",
    legacy_comparison_aligned_path: Path | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    theta_values = [float(value) for value in theta_tip_deg]
    if not theta_values or len(set(theta_values)) != len(theta_values):
        raise ValueError("theta_tip_deg must be a non-empty unique list")
    if airflow_mode not in AIRFLOW_MODES:
        raise ValueError(f"Unsupported airflow_mode {airflow_mode!r}; expected one of {sorted(AIRFLOW_MODES)}")
    windows = (
        load_window_manifest(window_manifest_path)
        if window_manifest_path is not None
        else select_representative_windows(
            samples,
            window_count=auto_window_count,
            airflow_mode=airflow_mode,
        )
    )
    validate_windows(samples, windows, airflow_mode=airflow_mode)
    windows.to_csv(output_dir / "selected_windows.csv", index=False)
    config = baseline_config_from_aircraft_metadata(
        aircraft_metadata_path,
        airflow_mode=airflow_mode,
    )
    aligned = evaluate_selected_windows(
        samples,
        windows,
        theta_tip_deg=theta_values,
        geometry_path=geometry_path,
        config=config,
    )
    aligned.to_parquet(output_dir / "aligned_predictions.parquet", index=False)
    window_metrics = compute_window_metrics(aligned)
    window_metrics.to_csv(output_dir / "window_metrics.csv", index=False)
    cycle_metrics, cycle_means = compute_cycle_mean_metrics(aligned)
    cycle_metrics.to_csv(output_dir / "cycle_mean_metrics.csv", index=False)
    cycle_means.to_parquet(output_dir / "cycle_means.parquet", index=False)
    phase_binned = compute_phase_binned_curves(aligned, phase_bins=phase_bins)
    phase_binned.to_parquet(output_dir / "phase_binned_curves.parquet", index=False)
    plot_window_time_series(aligned, output_dir / "figures" / "time_series")
    plot_phase_folded(phase_binned, output_dir / "figures" / "phase_folded")
    plot_theta_sensitivity(window_metrics, output_dir / "figures" / "theta_sensitivity")
    if airflow_mode == "attitude_ground_wind_3d":
        plot_airflow_diagnostics(aligned, output_dir / "figures" / "airflow_diagnostics")
    if legacy_comparison_aligned_path is not None:
        legacy_aligned = pd.read_parquet(legacy_comparison_aligned_path)
        plot_airflow_mode_comparison(
            aligned,
            legacy_aligned,
            output_dir / "figures" / "airflow_mode_comparison",
            phase_bins=phase_bins,
        )
    contributions, _ = plot_component_breakdown(
        aligned, windows, output_dir / "figures" / "component_breakdown"
    )
    contributions.to_parquet(output_dir / "component_contributions.parquet", index=False)
    project_root = Path(__file__).resolve().parents[3]
    git_commit = _git_value(["rev-parse", "HEAD"], project_root)
    dirty = bool(_git_value(["status", "--short"], project_root))
    manifest: dict[str, object] = {
        "comparison": {
            "baseline": "wing-only DeLaurier baseline",
            "label": "total reconstructed effective-wrench label",
        },
        "git_commit": git_commit,
        "git_worktree_dirty": dirty,
        "isaaclab_source": {
            "repository": ISAACLAB_SOURCE_REPOSITORY,
            "branch": ISAACLAB_SOURCE_BRANCH,
            "commit": ISAACLAB_SOURCE_COMMIT,
        },
        "input_dataset_path": str(samples_path.resolve()),
        "input_dataset_sha256": _sha256(samples_path),
        "aircraft_metadata_path": str(aircraft_metadata_path.resolve()),
        "aircraft_metadata_sha256": _sha256(aircraft_metadata_path),
        "aircraft_reference_fields_loaded": [
            "frames.body_frame",
            "frames.body_reference_origin",
            "frames.cg_reference_origin",
            "mass_properties.cg_b_m.value",
            "flapping_drive.wing_stroke_amplitude_rad.value",
        ],
        "geometry_path": str(geometry_path.resolve()),
        "geometry_sha256": _sha256(geometry_path),
        "window_manifest_source": str(window_manifest_path.resolve()) if window_manifest_path else "deterministic_auto",
        "window_manifest_source_sha256": _sha256(window_manifest_path) if window_manifest_path else None,
        "selected_windows_path": str((output_dir / "selected_windows.csv").resolve()),
        "selected_windows_sha256": _sha256(output_dir / "selected_windows.csv"),
        "theta_tip_deg": theta_values,
        "dynamic_twist_mode": "delaurier_linear_spanwise",
        "theta_tip_semantics": "amplitude at theoretical geometric tip y=R",
        "phase_convention": {
            "canonical": "q=A sin(phi_C), phi_C=0 neutral starting upstroke",
            "mapping": "phi_D=wrap(phi_C-pi/2)",
            "frozen_motion": "q=Gamma cos(phi_D), h=-q*y",
            "phase_rate": "2*pi*flap_frequency_hz",
            "phase_acceleration": "zero within each physics sample",
        },
        "airflow_mode": airflow_mode,
        "legacy_comparison_aligned_path": (
            str(legacy_comparison_aligned_path.resolve())
            if legacy_comparison_aligned_path is not None
            else None
        ),
        "legacy_comparison_aligned_sha256": (
            _sha256(legacy_comparison_aligned_path)
            if legacy_comparison_aligned_path is not None
            else None
        ),
        "airspeed_source": (
            "body-forward component of quaternion-rotated NED ground-minus-wind velocity; minimum 0.5 m/s"
            if airflow_mode == "attitude_ground_wind_3d"
            else "airspeed_validated.true_airspeed_m_s (raw canonical row; minimum 0.5 m/s)"
        ),
        "rho_source": "vehicle_air_data.rho (raw canonical row)",
        "attitude_source": (
            "vehicle_attitude.q[0..3], PX4 wxyz body-FRD to NED"
            if airflow_mode == "attitude_ground_wind_3d"
            else None
        ),
        "ground_velocity_source": (
            "vehicle_local_position.vx/vy/vz in NED"
            if airflow_mode == "attitude_ground_wind_3d"
            else None
        ),
        "wind_used": airflow_mode == "attitude_ground_wind_3d",
        "wind_source": (
            "wind.windspeed_north/east; vertical wind explicitly set to zero"
            if airflow_mode == "attitude_ground_wind_3d"
            else None
        ),
        "vertical_airflow_used": airflow_mode == "attitude_ground_wind_3d",
        "sideslip_derived": airflow_mode == "attitude_ground_wind_3d",
        "sideslip_used_by_delaurier_strip_equations": False,
        "frame_convention": {
            "canonical_input": "body FRD",
            "internal": "DeLaurier/Wang span-normal-chord",
            "output": "body FRD",
            "force_transform": "polar vector",
            "moment_transform": "axial vector with determinant parity",
        },
        "moment_reference": "real-aircraft CG from metadata, relative to canonical IMU origin",
        "filter_mode": "baseline_raw; no label-bandwidth filter; per-cycle means also saved",
        "label_reconstruction_bandwidth": "kinematic derivatives in selected dataset use Savitzky-Golay smoothing; baseline is unfiltered",
        "phase_bins": int(phase_bins),
        "baseline_config": asdict(config),
        "script_command": command or " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_index(
        output_dir,
        windows=windows,
        theta_tip_deg=theta_values,
        metrics=window_metrics,
        airflow_mode=airflow_mode,
        has_legacy_comparison=legacy_comparison_aligned_path is not None,
    )
    return manifest
