#!/usr/bin/env python3
"""Create focused fx/fz correction diagnostics for paper interpretation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGETS = ("fx_b", "fz_b")
COLORS = {
    "label": "#000000",
    "prior": "#0072B2",
    "corrected": "#D55E00",
}


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    mask = np.isfinite(y) & np.isfinite(p)
    if int(mask.sum()) == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "r2": np.nan, "corr": np.nan}
    err = p[mask] - y[mask]
    centered = y[mask] - float(np.mean(y[mask]))
    ss_tot = float(np.sum(centered * centered))
    ss_res = float(np.sum(err * err))
    corr = np.nan
    if int(mask.sum()) > 2 and np.std(y[mask]) > 1.0e-12 and np.std(p[mask]) > 1.0e-12:
        corr = float(np.corrcoef(y[mask], p[mask])[0, 1])
    return {
        "n": int(mask.sum()),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else np.nan,
        "corr": corr,
    }


def _phase_table(frame: pd.DataFrame, *, bins: int) -> pd.DataFrame:
    phase = np.mod(frame["phase_corrected_rad"].to_numpy(dtype=float), 2.0 * np.pi)
    bin_id = np.floor(phase / (2.0 * np.pi) * bins).astype(int)
    bin_id = np.clip(bin_id, 0, bins - 1)
    work = pd.DataFrame({"bin": bin_id, "phase_rad": phase})
    for target in TARGETS:
        for kind, prefix in (("label", "label"), ("prior", "prior"), ("corrected", "force_v2")):
            work[f"{kind}_{target}"] = frame[f"{prefix}_{target}"].to_numpy(dtype=float)
    rows = []
    for idx, group in work.groupby("bin", observed=True):
        row = {
            "bin": int(idx),
            "phase_center_rad": (float(idx) + 0.5) / float(bins) * 2.0 * np.pi,
            "n": int(len(group)),
        }
        for target in TARGETS:
            for kind in ("label", "prior", "corrected"):
                row[f"{kind}_{target}_mean"] = float(group[f"{kind}_{target}"].mean())
                row[f"{kind}_{target}_median"] = float(group[f"{kind}_{target}"].median())
            row[f"prior_residual_{target}_rmse"] = float(
                np.sqrt(np.mean((group[f"prior_{target}"] - group[f"label_{target}"]) ** 2))
            )
            row[f"corrected_residual_{target}_rmse"] = float(
                np.sqrt(np.mean((group[f"corrected_{target}"] - group[f"label_{target}"]) ** 2))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - np.nanmean(values)) / max(float(np.nanstd(values)), 1.0e-12)


def _choose_representative_log(frame: pd.DataFrame) -> str:
    rows = []
    for log_id, group in frame.groupby("log_id", observed=True):
        if len(group) < 1000:
            continue
        rmse = []
        for target in TARGETS:
            y = group[f"label_{target}"].to_numpy(dtype=float)
            p = group[f"force_v2_{target}"].to_numpy(dtype=float)
            rmse.append(_metrics(y, p)["rmse"])
        rows.append({"log_id": str(log_id), "score": float(np.nanmean(rmse)), "n": int(len(group))})
    if not rows:
        return str(frame["log_id"].iloc[0])
    table = pd.DataFrame(rows).sort_values("score")
    return str(table.iloc[len(table) // 2]["log_id"])


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#e5e7eb", linewidth=0.6, zorder=0)


def plot_scatter(frame: pd.DataFrame, output: Path, *, max_points: int) -> None:
    rng = np.random.default_rng(0)
    if len(frame) > max_points:
        idx = rng.choice(np.arange(len(frame)), size=max_points, replace=False)
        plot_frame = frame.iloc[np.sort(idx)].copy()
    else:
        plot_frame = frame
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    for row, target in enumerate(TARGETS):
        label = plot_frame[f"label_{target}"].to_numpy(dtype=float)
        for col, (kind, prefix) in enumerate((("prior", "prior"), ("corrected", "force_v2"))):
            pred = plot_frame[f"{prefix}_{target}"].to_numpy(dtype=float)
            ax = axes[row, col]
            ax.scatter(label, pred, s=4, alpha=0.18, color=COLORS[kind], linewidths=0)
            lim = np.nanpercentile(np.concatenate([label, pred]), [1, 99])
            pad = 0.05 * float(lim[1] - lim[0])
            ax.plot([lim[0] - pad, lim[1] + pad], [lim[0] - pad, lim[1] + pad], color="#555555", lw=0.8)
            ax.set_xlim(lim[0] - pad, lim[1] + pad)
            ax.set_ylim(lim[0] - pad, lim[1] + pad)
            met = _metrics(label, pred)
            ax.set_title(f"{target}: {kind}  corr={met['corr']:.2f}, RMSE={met['rmse']:.2f}", fontsize=9)
            ax.set_xlabel("label")
            ax.set_ylabel("prediction")
            _style_axes(ax)
    fig.savefig(output / "fx_fz_scatter_prior_vs_corrected.png", dpi=300)
    fig.savefig(output / "fx_fz_scatter_prior_vs_corrected.pdf")
    plt.close(fig)


def plot_phase(phase: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True, sharex=True)
    for row, target in enumerate(TARGETS):
        x = phase["phase_center_rad"].to_numpy(dtype=float)
        for kind in ("label", "prior", "corrected"):
            y = phase[f"{kind}_{target}_mean"].to_numpy(dtype=float)
            axes[row, 0].plot(x, y, lw=1.6, color=COLORS[kind], label=kind)
            axes[row, 1].plot(x, _zscore(y), lw=1.6, color=COLORS[kind], label=kind)
        axes[row, 0].set_ylabel(f"{target} mean")
        axes[row, 1].set_ylabel(f"{target} z-scored mean")
        axes[row, 0].set_title(f"{target}: raw phase-binned values", fontsize=9)
        axes[row, 1].set_title(f"{target}: normalized phase structure", fontsize=9)
        for ax in axes[row]:
            _style_axes(ax)
            ax.set_xlabel("wingbeat phase (rad)")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(output / "fx_fz_phase_binned_prior_vs_corrected.png", dpi=300)
    fig.savefig(output / "fx_fz_phase_binned_prior_vs_corrected.pdf")
    plt.close(fig)


def plot_timeseries(frame: pd.DataFrame, output: Path, *, log_id: str, seconds: float) -> None:
    group = frame.loc[frame["log_id"].astype(str).eq(str(log_id))].sort_values("time_s").copy()
    if group.empty:
        group = frame.sort_values("time_s").head(int(seconds * 100)).copy()
    t0 = float(group["time_s"].iloc[0])
    group = group.loc[group["time_s"].le(t0 + seconds)].copy()
    t = group["time_s"].to_numpy(dtype=float) - t0
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4), constrained_layout=True, sharex=True)
    for ax, target in zip(axes, TARGETS):
        ax.plot(t, group[f"label_{target}"], color=COLORS["label"], lw=1.4, label="label")
        ax.plot(t, group[f"prior_{target}"], color=COLORS["prior"], lw=1.0, alpha=0.8, label="prior")
        ax.plot(t, group[f"force_v2_{target}"], color=COLORS["corrected"], lw=1.2, label="corrected")
        ax.set_ylabel(target)
        ax.set_title(f"{target} on representative held-out log", fontsize=9)
        _style_axes(ax)
    axes[-1].set_xlabel("time from window start (s)")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(output / "fx_fz_representative_timeseries.png", dpi=300)
    fig.savefig(output / "fx_fz_representative_timeseries.pdf")
    plt.close(fig)


def run(*, predictions_path: Path, output_root: Path, phase_bins: int, max_points: int, seconds: float) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(predictions_path)
    required = ["log_id", "time_s", "phase_corrected_rad"]
    for target in TARGETS:
        required.extend([f"label_{target}", f"prior_{target}", f"force_v2_{target}"])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"prediction frame is missing required columns: {missing}")

    rows = []
    for target in TARGETS:
        y = frame[f"label_{target}"].to_numpy(dtype=float)
        for kind, prefix in (("prior", "prior"), ("corrected", "force_v2")):
            rows.append({"target": target, "model": kind, **_metrics(y, frame[f"{prefix}_{target}"].to_numpy(dtype=float))})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_root / "fx_fz_prior_corrected_metrics.csv", index=False)
    phase = _phase_table(frame, bins=phase_bins)
    phase.to_csv(output_root / "fx_fz_phase_binned_summary.csv", index=False)
    log_id = _choose_representative_log(frame)
    plot_scatter(frame, output_root, max_points=max_points)
    plot_phase(phase, output_root)
    plot_timeseries(frame, output_root, log_id=log_id, seconds=seconds)
    manifest = {
        "predictions_path": str(predictions_path),
        "output_root": str(output_root),
        "targets": list(TARGETS),
        "phase_bins": int(phase_bins),
        "representative_log_id": log_id,
        "interpretation": (
            "Correlation measures shape agreement, not absolute accuracy; RMSE/bias show scale and offset error."
        ),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase-bins", type=int, default=48)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--seconds", type=float, default=8.0)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                predictions_path=args.predictions_path,
                output_root=args.output_root,
                phase_bins=args.phase_bins,
                max_points=args.max_points,
                seconds=args.seconds,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
