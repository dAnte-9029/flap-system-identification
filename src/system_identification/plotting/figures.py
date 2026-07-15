"""Low-coupling figure writers used by existing training reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _save_ablation_summary_plot(summary: pd.DataFrame, output_path: str | Path) -> None:
    fig_width = max(8.0, 1.5 * len(summary))
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    x = np.arange(len(summary))
    width = 0.36
    ax.bar(x - width / 2, summary["val_overall_r2"], width=width, label="val_overall_r2")
    ax.bar(x + width / 2, summary["test_overall_r2"], width=width, label="test_overall_r2")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["variant_name"], rotation=20, ha="right")
    ax.set_ylabel("R^2")
    ax.set_title("Feature Ablation Summary")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_baseline_comparison_plot(summary: pd.DataFrame, output_path: str | Path) -> None:
    if summary.empty:
        return
    fig_width = max(8.0, 1.8 * len(summary))
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    x = np.arange(len(summary))
    if "test_overall_r2" in summary.columns:
        width = 0.36
        ax.bar(x - width / 2, summary["val_overall_r2"], width=width, label="val_overall_r2")
        ax.bar(x + width / 2, summary["test_overall_r2"], width=width, label="test_overall_r2")
    else:
        ax.bar(x, summary["val_overall_r2"], width=0.48, label="val_overall_r2")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["recipe_name"], rotation=20, ha="right")
    ax.set_ylabel("R^2")
    ax.set_title("Baseline Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
