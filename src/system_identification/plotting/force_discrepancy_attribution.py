"""Figure construction for the force-discrepancy attribution audit.

This module consumes authoritative tables produced by ``analysis``.  It does
not fit probes, choose thresholds, alter rows, or recompute audit metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OKABE_ITO = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
FORCE_LABELS = {"fx_b": r"$F_x$", "fz_b": r"$F_z$"}


@dataclass(frozen=True)
class PlotContext:
    run_id: str
    git_short_sha: str
    partitions: tuple[str, ...]
    frame: str = "body FRD"

    @property
    def suffix(self) -> str:
        return f"{self.frame} | partitions={','.join(self.partitions)} | run={self.run_id} | git={self.git_short_sha}"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _empty_figure(path: Path, title: str, reason: str, context: PlotContext) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=12)
    ax.text(0.5, 0.42, f"Not available: {reason}", ha="center", va="center")
    ax.text(0.5, 0.10, context.suffix, ha="center", va="center", fontsize=7)
    return _save(fig, path)


def figure_phase_curves(table: pd.DataFrame, path: Path, context: PlotContext) -> Path:
    if table.empty:
        return _empty_figure(path, "Figure 1 — Label, prior, and residual phase curves", "no accepted cycles", context)
    macro = table.drop_duplicates(["partition", "component", "phase_bin"])
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.0), sharex=True)
    for ax, component in zip(axes, ("fx_b", "fz_b")):
        group = macro.loc[macro["component"] == component].sort_values("phase_center_rad")
        for partition, subset in group.groupby("partition", sort=True):
            x = subset["phase_center_rad"].to_numpy(dtype=float)
            ax.plot(x, subset["label_force_macro"], label=f"label ({partition})", linestyle="-")
            ax.plot(x, subset["prior_force_macro"], label=f"DeLaurier prior ({partition})", linestyle="--")
            residual = subset["waveform_residual_macro"].to_numpy(dtype=float)
            ci = subset["waveform_residual_ci95"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, residual, label=f"zero-mean residual ({partition})", linestyle=":")
            ax.fill_between(x, residual - ci, residual + ci, alpha=0.16)
        ax.axhline(0.0, color="#555555", linewidth=0.7)
        ax.set_ylabel(f"{FORCE_LABELS[component]} (N)")
        ax.set_title(f"{FORCE_LABELS[component]} reconstructed total effective force vs wing-only prior")
        ax.legend(ncol=2)
    axes[-1].set_xlabel("Wingbeat phase (rad)")
    fig.suptitle(f"Figure 1 — Representative phase curves | {context.suffix}", fontsize=10)
    return _save(fig, path)


def figure_phase_shift(cycles: pd.DataFrame, by_log: pd.DataFrame, path: Path, context: PlotContext) -> Path:
    valid = cycles.loc[cycles.get("status") == "ok"] if not cycles.empty and "status" in cycles else pd.DataFrame()
    if valid.empty:
        return _empty_figure(path, "Figure 2 — Phase shift and fixed-delay diagnosis", "no valid phase estimates", context)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4))
    for component, color in zip(("fx_b", "fz_b"), OKABE_ITO[:2]):
        group = valid.loc[valid["component"] == component]
        axes[0, 0].scatter(group["frequency_hz"], group["shift_rad"], s=7, alpha=0.25, color=color, label=FORCE_LABELS[component])
        axes[0, 1].scatter(group["frequency_hz"], 1000.0 * group["equivalent_delay_s"], s=7, alpha=0.25, color=color, label=FORCE_LABELS[component])
        axes[1, 1].hist(group["shift_rad"], bins=30, alpha=0.45, color=color, label=FORCE_LABELS[component])
    axes[0, 0].set(xlabel="Frequency (Hz)", ylabel=r"Optimal $\Delta\phi$ (rad)", title="Circular phase shift vs frequency")
    axes[0, 1].set(xlabel="Frequency (Hz)", ylabel="Equivalent delay (ms)", title="Inferred delay vs frequency")
    if not by_log.empty:
        labels = by_log["log_id"].astype(str).unique()
        x_lookup = {label: index for index, label in enumerate(labels)}
        for component, marker in (("fx_b", "o"), ("fz_b", "s")):
            group = by_log.loc[by_log["component"] == component]
            axes[1, 0].scatter(
                [x_lookup[str(value)] for value in group["log_id"]],
                group["circular_mean_shift_rad"],
                label=FORCE_LABELS[component],
                marker=marker,
                s=24,
            )
        axes[1, 0].set_xticks(range(len(labels)), labels, rotation=90)
    axes[1, 0].set(ylabel=r"Log circular mean $\Delta\phi$ (rad)", title="Per-log distribution")
    axes[1, 1].set(xlabel=r"Optimal $\Delta\phi$ (rad)", ylabel="Cycle count", title="$F_x$/$F_z$ shift comparison")
    for ax in axes.flat:
        ax.legend()
    fig.suptitle(f"Figure 2 — Phase/fixed-delay diagnosis | {context.suffix}", fontsize=10)
    return _save(fig, path)


def figure_cycle_mean(table: pd.DataFrame, path: Path, context: PlotContext) -> Path:
    if table.empty:
        return _empty_figure(path, "Figure 3 — Cycle-mean residual", "no accepted cycles", context)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for ax, component in zip(axes, ("fx_b", "fz_b")):
        per_log = table.groupby(["partition", "log_id"], as_index=False)[f"mean_residual_{component}"].mean()
        labels = per_log["log_id"].astype(str).tolist()
        colors = [OKABE_ITO[0] if value == "train" else OKABE_ITO[1] for value in per_log["partition"]]
        ax.bar(range(len(per_log)), per_log[f"mean_residual_{component}"], color=colors, alpha=0.8)
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_xticks(range(len(labels)), labels, rotation=90)
        ax.set_ylabel("Cycle-mean residual (N)")
        ax.set_title(FORCE_LABELS[component])
    fig.suptitle(f"Figure 3 — Cycle-mean residual by log | {context.suffix}", fontsize=10)
    return _save(fig, path)


def figure_waveform(table: pd.DataFrame, path: Path, context: PlotContext) -> Path:
    if table.empty:
        return _empty_figure(path, "Figure 4 — Zero-mean wingbeat residual", "no waveform table", context)
    macro = table.drop_duplicates(["partition", "component", "phase_bin"])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), sharex=True)
    for ax, component in zip(axes, ("fx_b", "fz_b")):
        for partition, group in macro.loc[macro["component"] == component].groupby("partition"):
            group = group.sort_values("phase_center_rad")
            x = group["phase_center_rad"].to_numpy(dtype=float)
            y = group["waveform_residual_macro"].to_numpy(dtype=float)
            ci = group["waveform_residual_ci95"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, y, label=partition)
            ax.fill_between(x, y - ci, y + ci, alpha=0.18)
        ax.axhline(0.0, color="#333333", linewidth=0.7)
        ax.set(xlabel="Wingbeat phase (rad)", ylabel="Zero-mean residual (N)", title=FORCE_LABELS[component])
        ax.legend()
    fig.suptitle(f"Figure 4 — Zero-mean wingbeat residual (equal-log macro, 95% CI) | {context.suffix}", fontsize=10)
    return _save(fig, path)


def figure_half_stroke(table: pd.DataFrame, path: Path, context: PlotContext) -> Path:
    if table.empty:
        return _empty_figure(path, "Figure 5 — Half-stroke attribution", "no half-stroke table", context)
    summary = table.groupby(["component", "half_stroke"], as_index=False).agg(
        integral=("integral_waveform_residual_rad", "mean"),
        peak=("peak_abs_residual", lambda x: float(np.mean(np.abs(x)))),
        reversal=("reversal_mean_abs_residual", "mean"),
        midstroke=("midstroke_mean_abs_residual", "mean"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0))
    for row, metric, label in (
        (0, "integral", "Integral residual (N rad)"),
        (1, "peak", "Mean absolute peak (N)"),
    ):
        for column, component in enumerate(("fx_b", "fz_b")):
            group = summary.loc[summary["component"] == component]
            axes[row, column].bar(group["half_stroke"], group[metric], color=OKABE_ITO[: len(group)])
            axes[row, column].set_ylabel(label)
            axes[row, column].set_title(f"{FORCE_LABELS[component]} authoritative q=A sin(phi) halves")
    fig.suptitle(f"Figure 5 — Upstroke/downstroke attribution; reversal retained | {context.suffix}", fontsize=10)
    return _save(fig, path)


def figure_component_phase(table: pd.DataFrame, path: Path, context: PlotContext) -> Path:
    if table.empty:
        return _empty_figure(path, "Figure 6 — DeLaurier component phase contribution", "component diagnostics unavailable", context)
    summary = table.groupby(["force_component", "component", "phase_bin", "phase_center_rad"], as_index=False)["contribution_n"].mean()
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.0), sharex=True)
    for ax, force in zip(axes, ("fx_b", "fz_b")):
        for component, group in summary.loc[summary["force_component"] == force].groupby("component"):
            ax.plot(group["phase_center_rad"], group["contribution_n"], label=component)
        ax.set_ylabel(f"{FORCE_LABELS[force]} contribution (N)")
        ax.set_title(FORCE_LABELS[force])
        ax.legend(ncol=3)
    axes[-1].set_xlabel("Wingbeat phase (rad)")
    fig.suptitle(f"Figure 6 — Frozen diagnostic DeLaurier component contributions | {context.suffix}", fontsize=10)
    return _save(fig, path)


def figure_spanwise(table: pd.DataFrame, path: Path, context: PlotContext) -> Path:
    if table.empty:
        return _empty_figure(path, "Figure 7 — Spanwise attribution", "spanwise diagnostics unavailable", context)
    summary = table.groupby(["force_component", "span_region"], as_index=False)["cycle_rms_contribution_n"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3))
    for ax, force in zip(axes, ("fx_b", "fz_b")):
        group = summary.loc[summary["force_component"] == force]
        ax.bar(group["span_region"], group["cycle_rms_contribution_n"], color=OKABE_ITO[: len(group)])
        ax.set(ylabel="RMS contribution (N)", title=FORCE_LABELS[force])
    fig.suptitle(f"Figure 7 — Root/mid/tip diagnostic contribution | {context.suffix}", fontsize=10)
    return _save(fig, path)


def figure_sensitivity(table: pd.DataFrame, path: Path, context: PlotContext) -> Path:
    if table.empty:
        return _empty_figure(path, "Figure 8 — Physical sensitivity similarity", "sensitivity audit unavailable", context)
    summary = table.groupby(["force_component", "parameter"], as_index=False)["shape_correlation"].mean()
    pivot = summary.pivot(index="parameter", columns="force_component", values="shape_correlation")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [FORCE_LABELS.get(value, value) for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            ax.text(column, row, f"{pivot.iloc[row, column]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Residual–sensitivity shape correlation")
    ax.set_title(f"Figure 8 — Local physical sensitivity similarity | {context.suffix}")
    return _save(fig, path)


def figure_harmonics(table: pd.DataFrame, path: Path, context: PlotContext) -> Path:
    if table.empty:
        return _empty_figure(path, "Figure 9 — Harmonic energy", "no harmonic table", context)
    summary = table.groupby(["component", "harmonic_order"], as_index=False).agg(
        amplitude=("amplitude", "mean"), coverage=("cumulative_energy_coverage", "mean")
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))
    for component, marker in (("fx_b", "o"), ("fz_b", "s")):
        group = summary.loc[summary["component"] == component]
        axes[0].plot(group["harmonic_order"], group["amplitude"], marker=marker, label=FORCE_LABELS[component])
        axes[1].plot(group["harmonic_order"], group["coverage"], marker=marker, label=FORCE_LABELS[component])
    axes[0].set(xlabel="Fourier harmonic order K", ylabel="Mean amplitude (N)", title="Harmonic amplitude")
    axes[1].set(xlabel="Fourier harmonic order K", ylabel="Cumulative energy coverage", title="Cumulative coverage", ylim=(0, 1.05))
    for ax in axes:
        ax.legend()
    fig.suptitle(f"Figure 9 — Zero-mean residual harmonic structure | {context.suffix}", fontsize=10)
    return _save(fig, path)


def figure_conditions(table: pd.DataFrame, path: Path, context: PlotContext) -> Path:
    binned = table.loc[table.get("aggregation") == "binned_equal_log_macro"] if not table.empty else pd.DataFrame()
    if binned.empty:
        return _empty_figure(path, "Figure 10 — Condition dependence", "no binned condition table", context)
    conditions = [
        value
        for value in ("condition_airspeed_m_s", "condition_alpha_rad", "condition_frequency_hz")
        if value in set(binned["condition"])
    ]
    fig, axes = plt.subplots(2, len(conditions), figsize=(4.0 * len(conditions), 6.0), squeeze=False)
    for row, force in enumerate(("fx_b", "fz_b")):
        for column, condition in enumerate(conditions):
            ax = axes[row, column]
            group = binned.loc[
                (binned["component"] == force)
                & (binned["condition"] == condition)
                & (binned["summary"] == "residual_rms")
            ].sort_values("condition_center")
            for partition, subset in group.groupby("partition"):
                ax.plot(subset["condition_center"], subset["summary_value"], marker="o", label=partition)
            ax.set(xlabel=condition.replace("condition_", ""), ylabel="Residual RMS (N)", title=FORCE_LABELS[force])
            ax.legend()
    fig.suptitle(f"Figure 10 — Condition dependence (equal-log macro) | {context.suffix}", fontsize=10)
    return _save(fig, path)


def figure_log_correlations(table: pd.DataFrame, path: Path, context: PlotContext) -> Path:
    if table.empty:
        return _empty_figure(path, "Figure 11 — Cross-log waveform correlation", "no correlation table", context)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    for ax, force in zip(axes, ("fx_b", "fz_b")):
        group = table.loc[(table["component"] == force) & (table["partition"] == table["partition"].iloc[0])]
        pivot = group.pivot(index="log_id_left", columns="log_id_right", values="waveform_correlation")
        image = ax.imshow(pivot.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        ax.set_title(FORCE_LABELS[force])
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=90, fontsize=5)
        ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=5)
        fig.colorbar(image, ax=ax, shrink=0.75)
    fig.suptitle(f"Figure 11 — Cross-log residual-waveform correlation | {context.suffix}", fontsize=10)
    return _save(fig, path)


def figure_dates(table: pd.DataFrame, path: Path, context: PlotContext) -> Path:
    if table.empty:
        return _empty_figure(path, "Figure 12 — Cross-date comparison", "no date identity", context)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    for ax, force in zip(axes, ("fx_b", "fz_b")):
        group = table.loc[table["component"] == force]
        ax.bar(group["date"], group["waveform_rms_macro"], color=OKABE_ITO[: len(group)])
        ax.tick_params(axis="x", rotation=45)
        ax.set(ylabel="Waveform RMS (N)", title=FORCE_LABELS[force])
    fig.suptitle(f"Figure 12 — Date-level repeatability | {context.suffix}", fontsize=10)
    return _save(fig, path)


def figure_label_robustness(uncertainty: pd.DataFrame, waveform: pd.DataFrame, path: Path, context: PlotContext) -> Path:
    if uncertainty.empty or "label_uncertainty_std" not in uncertainty.columns:
        return _empty_figure(path, "Figure 13 — Label robustness band", "no comparable label variants", context)
    macro = waveform.drop_duplicates(["partition", "component", "phase_bin"])
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    for ax, force in zip(axes, ("fx_b", "fz_b")):
        residual = macro.loc[macro["component"] == force].groupby("phase_bin", as_index=False).agg(
            phase_center_rad=("phase_center_rad", "mean"), residual=("waveform_residual_macro", "mean")
        )
        band = uncertainty.loc[uncertainty["component"] == force].groupby("phase_bin", as_index=False).agg(
            phase_center_rad=("phase_center_rad", "mean"), uncertainty=("label_uncertainty_std", "mean")
        )
        merged = residual.merge(band, on="phase_bin", suffixes=("", "_band"))
        ax.plot(merged["phase_center_rad"], merged["residual"], label="residual")
        ax.fill_between(
            merged["phase_center_rad"],
            -merged["uncertainty"],
            merged["uncertainty"],
            alpha=0.25,
            label="label variant ±1 SD",
        )
        ax.set(xlabel="Wingbeat phase (rad)", ylabel="Force (N)", title=FORCE_LABELS[force])
        ax.legend()
    fig.suptitle(f"Figure 13 — Residual vs label-variant uncertainty | {context.suffix}", fontsize=10)
    return _save(fig, path)


def _best_probe_rows(table: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    if table.empty or "validation_equal_log_macro_rmse" not in table.columns:
        return pd.DataFrame()
    metric = pd.to_numeric(table["validation_equal_log_macro_rmse"], errors="coerce")
    finite = table.loc[np.isfinite(metric)].copy()
    if finite.empty:
        return pd.DataFrame()
    index = finite.groupby(group)["validation_equal_log_macro_rmse"].idxmin().dropna()
    return finite.loc[index.astype(int)].copy()


def figure_history(table: pd.DataFrame, path: Path, context: PlotContext) -> Path:
    best = _best_probe_rows(table, ["component", "history_samples"])
    if best.empty:
        return _empty_figure(path, "Figure 14 — Static versus history probe", "train+validation required", context)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for force, marker in (("fx_b", "o"), ("fz_b", "s")):
        group = best.loc[best["component"] == force].sort_values("history_samples")
        ax.plot(group["history_samples"], group["validation_equal_log_macro_rmse"], marker=marker, label=FORCE_LABELS[force])
    ax.set(xlabel="History length (samples)", ylabel="Validation equal-log macro RMSE (N)", title=f"Figure 14 — Linear static/history probes | {context.suffix}")
    ax.legend()
    return _save(fig, path)


def figure_prior_value(table: pd.DataFrame, path: Path, context: PlotContext) -> Path:
    best = _best_probe_rows(table, ["component", "model"])
    if best.empty:
        return _empty_figure(path, "Figure 15 — Matched-capacity prior value", "train+validation required", context)
    pivot = best.pivot(index="component", columns="model", values="validation_equal_log_macro_rmse")
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    pivot.plot.bar(ax=ax, color=OKABE_ITO[: len(pivot.columns)])
    ax.set(xlabel="Force component", ylabel="Validation equal-log macro RMSE (N)", title=f"Figure 15 — Matched-capacity prior vs no-prior | {context.suffix}")
    ax.legend(title="Diagnostic model")
    return _save(fig, path)


def figure_decisions(decision: Mapping[str, object], path: Path, context: PlotContext) -> Path:
    keys = [
        "fix_phase_convention_first",
        "fix_fixed_delay_first",
        "mean_correction_fx",
        "mean_correction_fz",
        "phase_correction_fx",
        "phase_correction_fz",
        "dynamic_model_needed",
        "label_uncertainty_blocks_correction",
        "prior_has_incremental_value",
    ]
    mapping = {"yes": 1.0, "no": 0.0, "insufficient_evidence": -1.0}
    values = np.array([[mapping.get(str(decision.get(key)), -1.0)] for key in keys])
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    image = ax.imshow(values, cmap=matplotlib.colors.ListedColormap(["#999999", "#56B4E9", "#D55E00"]), vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(keys)), keys)
    ax.set_xticks([0], ["status"])
    for row, key in enumerate(keys):
        ax.text(0, row, str(decision.get(key)), ha="center", va="center", color="black", fontsize=7)
    ax.set_title(f"Figure 16 — Correction decision summary | {context.suffix}")
    return _save(fig, path)


def write_audit_figures(
    *,
    output_dir: str | Path,
    context: PlotContext,
    tables: Mapping[str, pd.DataFrame],
    decision: Mapping[str, object],
) -> list[Path]:
    """Write the fixed 16-figure EDA0 diagnostic set."""

    _style()
    root = Path(output_dir)
    figures = [
        figure_phase_curves(tables.get("phase_waveform", pd.DataFrame()), root / "figure_01_phase_curves.png", context),
        figure_phase_shift(tables.get("phase_alignment_cycles", pd.DataFrame()), tables.get("phase_alignment_by_log", pd.DataFrame()), root / "figure_02_phase_shift_delay.png", context),
        figure_cycle_mean(tables.get("cycle_mean_residuals", pd.DataFrame()), root / "figure_03_cycle_mean_residual.png", context),
        figure_waveform(tables.get("phase_waveform", pd.DataFrame()), root / "figure_04_zero_mean_waveform.png", context),
        figure_half_stroke(tables.get("half_stroke_residuals", pd.DataFrame()), root / "figure_05_half_stroke_attribution.png", context),
        figure_component_phase(tables.get("component_phase_contributions", pd.DataFrame()), root / "figure_06_component_phase.png", context),
        figure_spanwise(tables.get("component_spanwise_summary", pd.DataFrame()), root / "figure_07_spanwise_attribution.png", context),
        figure_sensitivity(tables.get("physical_sensitivity_similarity", pd.DataFrame()), root / "figure_08_physical_sensitivity.png", context),
        figure_harmonics(tables.get("harmonic_cycle_summary", pd.DataFrame()), root / "figure_09_harmonic_energy.png", context),
        figure_conditions(tables.get("condition_dependence", pd.DataFrame()), root / "figure_10_condition_dependence.png", context),
        figure_log_correlations(tables.get("log_waveform_correlations", pd.DataFrame()), root / "figure_11_cross_log_correlations.png", context),
        figure_dates(tables.get("date_level_summary", pd.DataFrame()), root / "figure_12_cross_date.png", context),
        figure_label_robustness(tables.get("label_uncertainty_phase", pd.DataFrame()), tables.get("phase_waveform", pd.DataFrame()), root / "figure_13_label_robustness.png", context),
        figure_history(tables.get("static_history_probe_metrics", pd.DataFrame()), root / "figure_14_static_history.png", context),
        figure_prior_value(tables.get("matched_capacity_prior_probe", pd.DataFrame()), root / "figure_15_prior_value.png", context),
        figure_decisions(decision, root / "figure_16_decision_summary.png", context),
    ]
    return figures
