"""Headless evidence figures for C3 static correction selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILY_COLORS = {
    "raw_prior": "#555555",
    "gain_bias": "#d95f02",
    "fixed_prior_mean_wb": "#1b9e77",
    "shaped_prior_mean_wb": "#7570b3",
    "no_prior_mean_wb": "#e7298a",
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _bar(frame: pd.DataFrame, x: str, y: str, title: str, path: Path, limit: int = 24) -> None:
    plot = frame.nsmallest(limit, y).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(plot)), plot[y], color="#4477aa")
    ax.set_xticks(np.arange(len(plot)), plot[x].astype(str), rotation=75, ha="right", fontsize=7)
    ax.set_ylabel(y)
    ax.set_title(title)
    _save(fig, path)


def generate_static_selection_figures(
    output_root: str | Path,
    *,
    validation_metrics: pd.DataFrame,
    validation_per_log: pd.DataFrame,
    candidate_tables: Mapping[tuple[str, str], pd.DataFrame],
    selected_ids: Mapping[str, str],
    stability: Mapping[str, object],
) -> list[str]:
    output = Path(output_root)
    figures = output / "figures"
    figures.mkdir(exist_ok=False)
    complete = pd.read_parquet(output / "complete_model_cv_results.parquet")
    waveform = pd.read_parquet(output / "waveform_branch_cv_results.parquet")

    paths: list[Path] = []
    path = figures / "figure_01_train_cv_complete_leaderboard.png"
    _bar(complete, "candidate_id", "macro_total_rmse", "Train CV complete candidate leaderboard", path)
    paths.append(path)
    path = figures / "figure_02_validation_candidate_leaderboard.png"
    _bar(validation_metrics, "candidate_id", "macro_total_rmse", "Validation finalist leaderboard", path)
    paths.append(path)

    for number, component in ((3, "fx"), (4, "fz")):
        subset = validation_per_log[validation_per_log["component"] == component]
        pivot = subset.pivot(index="log_id", columns="candidate_id", values="rmse")
        fig, ax = plt.subplots(figsize=(10, 5))
        for candidate_id in pivot.columns:
            ax.plot(pivot.index, pivot[candidate_id], marker="o", alpha=0.7, label=candidate_id[:18])
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("per-log RMSE [N]")
        ax.set_title(f"{component.upper()} validation per-log paired RMSE")
        ax.legend(fontsize=6, ncol=2)
        path = figures / f"figure_{number:02d}_{component}_validation_per_log_rmse.png"
        _save(fig, path)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 6))
    for component, marker in (("fx", "o"), ("fz", "s")):
        subset = validation_metrics[validation_metrics["component"] == component]
        ax.scatter(subset["macro_mean_rmse"], subset["macro_waveform_rmse"], marker=marker, label=component.upper())
    ax.set_xlabel("mean macro RMSE [N]")
    ax.set_ylabel("waveform macro RMSE [N]")
    ax.set_title("Validation mean versus waveform error")
    ax.legend()
    path = figures / "figure_05_mean_vs_waveform_rmse.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    harmonic = waveform.groupby(["component", "harmonic_order"])["macro_waveform_rmse"].min().reset_index()
    for component, group in harmonic.groupby("component"):
        ax.plot(group["harmonic_order"], group["macro_waveform_rmse"], marker="o", label=component.upper())
    ax.set_xlabel("harmonic order K")
    ax.set_ylabel("best train-CV waveform macro RMSE [N]")
    ax.set_title("Harmonic-order sensitivity")
    ax.legend()
    path = figures / "figure_06_harmonic_order_sensitivity.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    condition = (
        waveform.groupby(["component", "waveform_condition_set"])["macro_waveform_rmse"].min().reset_index()
    )
    labels = condition["component"].str.upper() + ":" + condition["waveform_condition_set"]
    ax.bar(labels, condition["macro_waveform_rmse"], color="#66c2a5")
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("best train-CV waveform macro RMSE [N]")
    ax.set_title("Mean/WB condition sensitivity")
    path = figures / "figure_07_condition_sensitivity.png"
    _save(fig, path)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, component in zip(axes, ("fx", "fz")):
        grid = (
            complete[complete["component"] == component]
            .assign(
                mean_retention=lambda frame: frame["spec_json"].map(
                    lambda value: json.loads(value).get("mean_prior_retention")
                ),
                waveform_retention=lambda frame: frame["spec_json"].map(
                    lambda value: json.loads(value).get("waveform_prior_retention")
                ),
            )
            .dropna(subset=["mean_retention", "waveform_retention"])
            .pivot_table(
                index="mean_retention",
                columns="waveform_retention",
                values="macro_total_rmse",
                aggfunc="min",
            )
        )
        image = ax.imshow(grid.to_numpy(), aspect="auto", origin="lower")
        ax.set_xticks(range(len(grid.columns)), grid.columns)
        ax.set_yticks(range(len(grid.index)), grid.index)
        ax.set_xlabel("waveform retention")
        ax.set_ylabel("mean retention")
        ax.set_title(component.upper())
        fig.colorbar(image, ax=ax, label="train-CV RMSE [N]")
    path = figures / "figure_08_retention_heatmap.png"
    _save(fig, path)
    paths.append(path)

    for number, component in ((9, "fx"), (10, "fz")):
        subset = validation_metrics[validation_metrics["component"] == component]
        fig, ax = plt.subplots(figsize=(9, 5))
        for family in (
            "raw_prior",
            "fixed_prior_mean_wb",
            "shaped_prior_mean_wb",
            "no_prior_mean_wb",
        ):
            rows = subset[subset["model_type"] == family]
            if not len(rows):
                continue
            candidate_id = str(rows.nsmallest(1, "macro_total_rmse").iloc[0]["candidate_id"])
            table = candidate_tables[(component, candidate_id)].copy()
            table["phase_bin"] = np.floor(np.mod(table["phase_rad"], 2 * np.pi) / (2 * np.pi) * 36)
            curve = table.groupby("phase_bin")[["label_n", "prediction_n"]].mean()
            phase = (curve.index.to_numpy() + 0.5) / 36 * 2 * np.pi
            ax.plot(phase, curve["prediction_n"], label=family, color=FAMILY_COLORS[family])
        first_table = next(table for (comp, _), table in candidate_tables.items() if comp == component)
        label_curve = first_table.assign(
            phase_bin=np.floor(np.mod(first_table["phase_rad"], 2 * np.pi) / (2 * np.pi) * 36)
        ).groupby("phase_bin")["label_n"].mean()
        ax.plot((label_curve.index + 0.5) / 36 * 2 * np.pi, label_curve, "k--", label="label")
        ax.set_xlabel("mechanical phase [rad]")
        ax.set_ylabel(f"{component.upper()} [N]")
        ax.set_title(f"{component.upper()} validation phase-binned finalist curves")
        ax.legend()
        path = figures / f"figure_{number:02d}_{component}_phase_binned_prior_families.png"
        _save(fig, path)
        paths.append(path)

    fz = validation_metrics[validation_metrics["component"] == "fz"].nsmallest(12, "macro_total_rmse")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(fz["candidate_id"], fz["downstroke_integral_error_abs"], color="#fc8d62")
    ax.tick_params(axis="x", rotation=70, labelsize=7)
    ax.set_ylabel("|downstroke integral error| [N rad]")
    ax.set_title("Fz downstroke integral comparison")
    path = figures / "figure_11_fz_downstroke_integral.png"
    _save(fig, path)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, component in zip(axes, ("fx", "fz")):
        table = candidate_tables[(component, selected_ids[component])]
        correction = table["prediction_n"] - table["prior_n"]
        ax.hist(correction, bins=60, alpha=0.8)
        ax.set_title(component.upper())
        ax.set_xlabel("selected correction [N]")
    path = figures / "figure_12_selected_correction_amplitude.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 6))
    for component, group in validation_metrics.groupby("component"):
        ax.scatter(group["train_cv_rank"], group["validation_rank"], label=component.upper())
    ax.plot([0, validation_metrics["train_cv_rank"].max() + 1], [0, validation_metrics["train_cv_rank"].max() + 1], "k--")
    ax.set_xlabel("train-CV rank")
    ax.set_ylabel("validation rank")
    ax.set_title("Train-CV versus validation rank consistency")
    ax.legend()
    path = figures / "figure_13_rank_consistency.png"
    _save(fig, path)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, component in zip(axes, ("fx", "fz")):
        counts = stability[component]["selection_counts"]
        ax.bar(list(counts), list(counts.values()), color="#8da0cb")
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.set_ylim(0, 5)
        ax.set_title(component.upper())
        ax.set_ylabel("LOO selected count")
    path = figures / "figure_14_leave_one_log_out_stability.png"
    _save(fig, path)
    paths.append(path)
    return [str(path.resolve()) for path in paths]
