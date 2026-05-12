#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_TARGET_COLUMNS = ["fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b"]


TARGET_LABELS = {
    "fx_b": "Fx body",
    "fy_b": "Fy body",
    "fz_b": "Fz body",
    "mx_b": "Mx roll",
    "my_b": "My pitch",
    "mz_b": "Mz yaw",
}


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def _downsample(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0 or len(frame) <= max_points:
        return frame
    stride = int(np.ceil(len(frame) / max_points))
    return frame.iloc[::stride].copy()


def _window_around_index(frame: pd.DataFrame, center_index: int, window_size: int) -> pd.DataFrame:
    if window_size <= 0 or len(frame) <= window_size:
        return frame
    half = window_size // 2
    start = max(0, min(center_index - half, len(frame) - window_size))
    end = start + window_size
    return frame.iloc[start:end].copy()


def split_frame_on_plot_breaks(
    frame: pd.DataFrame,
    *,
    time_column: str = "time_s",
    segment_column: str = "segment_id",
    gap_multiplier: float = 8.0,
) -> list[pd.DataFrame]:
    """Split a plotted time series so line plots do not bridge missing intervals."""
    if len(frame) <= 1:
        return [frame.copy()]

    break_before = np.zeros(len(frame), dtype=bool)
    if segment_column in frame.columns:
        segment_values = frame[segment_column].to_numpy()
        break_before[1:] |= segment_values[1:] != segment_values[:-1]

    if time_column in frame.columns:
        time_values = frame[time_column].to_numpy(dtype=float)
        dt = np.diff(time_values)
        valid_dt = dt[np.isfinite(dt) & (dt > 0.0)]
        if len(valid_dt):
            nominal_dt = float(np.nanmedian(valid_dt))
            if np.isfinite(nominal_dt) and nominal_dt > 0.0:
                break_before[1:] |= dt > gap_multiplier * nominal_dt

    split_indices = np.flatnonzero(break_before)
    starts = [0, *split_indices.tolist()]
    ends = [*split_indices.tolist(), len(frame)]
    return [frame.iloc[start:end].copy() for start, end in zip(starts, ends) if end > start]


def _worst_residual_window(frame: pd.DataFrame, targets: tuple[str, ...], window_size: int) -> pd.DataFrame:
    if window_size <= 0 or len(frame) <= window_size:
        return frame
    normalized_residuals: list[np.ndarray] = []
    for target in targets:
        true_column = f"true_{target}"
        pred_column = f"pred_{target}"
        if true_column not in frame.columns or pred_column not in frame.columns:
            continue
        true_values = frame[true_column].to_numpy(dtype=float)
        pred_values = frame[pred_column].to_numpy(dtype=float)
        scale = float(np.nanstd(true_values))
        if not np.isfinite(scale) or scale < 1e-12:
            scale = 1.0
        normalized_residuals.append(np.abs(pred_values - true_values) / scale)
    if not normalized_residuals:
        return frame.head(window_size).copy()
    score = np.nanmean(np.vstack(normalized_residuals), axis=0)
    center_index = int(np.nanargmax(score))
    return _window_around_index(frame, center_index, window_size)


def _target_metrics(frame: pd.DataFrame, target: str) -> tuple[float, float]:
    true_values = frame[f"true_{target}"].to_numpy(dtype=float)
    pred_values = frame[f"pred_{target}"].to_numpy(dtype=float)
    residual = pred_values - true_values
    rmse = float(np.sqrt(np.nanmean(np.square(residual))))
    ss_res = float(np.nansum(np.square(residual)))
    centered = true_values - np.nanmean(true_values)
    ss_tot = float(np.nansum(np.square(centered)))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return rmse, r2


def _plot_targets(
    frame: pd.DataFrame,
    *,
    targets: tuple[str, ...],
    title: str,
    output_path: Path,
    relative_time_axis: bool = False,
) -> None:
    if frame.empty:
        raise ValueError("Cannot plot an empty frame")
    missing = [
        column
        for target in targets
        for column in (f"true_{target}", f"pred_{target}")
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    full_x_values = frame["time_s"].to_numpy(dtype=float) if "time_s" in frame.columns else np.arange(len(frame))
    x_origin = float(np.nanmin(full_x_values)) if relative_time_axis and np.isfinite(full_x_values).any() else 0.0
    x_label = "time since window start (s)" if relative_time_axis and "time_s" in frame.columns else "flight time_s"
    if "time_s" not in frame.columns:
        x_label = "sample index"
    line_parts = split_frame_on_plot_breaks(frame)

    fig, axes = plt.subplots(len(targets), 1, figsize=(13, 1.85 * len(targets)), sharex=True)
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        rmse, r2 = _target_metrics(frame, target)
        for part_index, part in enumerate(line_parts):
            x_values = part["time_s"].to_numpy(dtype=float) if "time_s" in part.columns else part.index.to_numpy(dtype=float)
            x_values = x_values - x_origin
            true_values = part[f"true_{target}"].to_numpy(dtype=float)
            pred_values = part[f"pred_{target}"].to_numpy(dtype=float)
            ax.plot(
                x_values,
                true_values,
                color="#1f2933",
                linewidth=1.0,
                label="true" if part_index == 0 else None,
            )
            ax.plot(
                x_values,
                pred_values,
                color="#d95f02",
                linewidth=0.95,
                alpha=0.9,
                label="pred" if part_index == 0 else None,
            )
        label = TARGET_LABELS.get(target, target)
        ax.set_ylabel(label)
        ax.set_title(f"{target}: RMSE={rmse:.4g}, R2={r2:.3f}", loc="left", fontsize=9)
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1].set_xlabel(x_label)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _select_log_groups(
    aligned: pd.DataFrame,
    *,
    log_ids: tuple[str, ...] | None,
    max_logs: int | None,
) -> list[tuple[str, pd.DataFrame]]:
    if "log_id" not in aligned.columns:
        groups = [("__all__", aligned.copy())]
    else:
        groups = [(str(log_id), group.copy()) for log_id, group in aligned.groupby("log_id", sort=True)]
    if log_ids:
        requested = set(log_ids)
        groups = [(log_id, group) for log_id, group in groups if log_id in requested]
    groups = sorted(groups, key=lambda item: len(item[1]), reverse=True)
    if max_logs is not None:
        groups = groups[:max_logs]
    return groups


def write_prediction_curve_plots(
    aligned: pd.DataFrame,
    *,
    output_dir: Path,
    split_name: str,
    targets: Iterable[str] = DEFAULT_TARGET_COLUMNS,
    log_ids: tuple[str, ...] | None = None,
    max_logs: int | None = None,
    zoom_samples: int = 2000,
    max_overview_points: int = 6000,
    relative_time_axis: bool = False,
) -> pd.DataFrame:
    resolved_targets = tuple(targets)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for log_id, group in _select_log_groups(aligned, log_ids=log_ids, max_logs=max_logs):
        if "time_s" in group.columns:
            group = group.sort_values("time_s").reset_index(drop=True)
        else:
            group = group.reset_index(drop=True)
        views = [
            ("overview", _downsample(group, max_overview_points)),
            ("zoom_start", group.head(zoom_samples).copy() if zoom_samples > 0 else group.copy()),
            ("zoom_worst_residual", _worst_residual_window(group, resolved_targets, zoom_samples)),
        ]
        for view_name, view_frame in views:
            safe_log_id = _sanitize_filename(log_id)
            output_path = output_dir / f"{split_name}_{safe_log_id}_{view_name}.png"
            title = f"{split_name} / {log_id} / {view_name}"
            _plot_targets(
                view_frame,
                targets=resolved_targets,
                title=title,
                output_path=output_path,
                relative_time_axis=relative_time_axis,
            )
            time_values = view_frame["time_s"].to_numpy(dtype=float) if "time_s" in view_frame.columns else np.arange(len(view_frame))
            rows.append(
                {
                    "split": split_name,
                    "log_id": log_id,
                    "view": view_name,
                    "sample_count": int(len(group)),
                    "plotted_points": int(len(view_frame)),
                    "t_start_s": float(np.nanmin(time_values)) if len(time_values) else np.nan,
                    "t_end_s": float(np.nanmax(time_values)) if len(time_values) else np.nan,
                    "plot_path": str(output_path),
                }
            )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "prediction_curve_manifest.csv", index=False)
    _write_markdown_index(manifest, output_dir / "index.md")
    return manifest


def _write_markdown_index(manifest: pd.DataFrame, output_path: Path) -> None:
    lines = ["# Prediction Curves", ""]
    for _, row in manifest.iterrows():
        title = f"{row['split']} / {row['log_id']} / {row['view']}"
        plot_path = Path(str(row["plot_path"]))
        rel_path = plot_path.name if plot_path.parent == output_path.parent else str(plot_path)
        lines.extend(
            [
                f"## {title}",
                "",
                f"- samples: {row['sample_count']}",
                f"- plotted_points: {row['plotted_points']}",
                f"- time_window_s: {row['t_start_s']:.3f} to {row['t_end_s']:.3f}",
                "",
                f"![{title}]({rel_path})",
                "",
            ]
        )
    output_path.write_text("\n".join(lines))


def _parse_csv_tuple(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or raw.strip() == "":
        return None
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    return values or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot model prediction curves against true target time series.")
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGET_COLUMNS), help="Comma-separated target names")
    parser.add_argument("--log-ids", default=None, help="Optional comma-separated log_id subset")
    parser.add_argument("--max-logs", type=int, default=None)
    parser.add_argument("--zoom-samples", type=int, default=2000)
    parser.add_argument("--max-overview-points", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-aligned-parquet", action="store_true")
    parser.add_argument(
        "--relative-time-axis",
        action="store_true",
        help="Plot x-axis relative to each window start instead of absolute flight time_s.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from system_identification.training import prediction_metadata_frame_for_bundle

    bundle = torch.load(args.model_bundle, map_location="cpu", weights_only=False)
    frame = pd.read_parquet(args.split_root / f"{args.split}_samples.parquet")
    aligned = prediction_metadata_frame_for_bundle(
        bundle,
        frame,
        split_name=args.split,
        batch_size=args.batch_size,
        device=args.device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_aligned_parquet:
        aligned.to_parquet(args.output_dir / f"aligned_{args.split}_predictions.parquet", index=False)
    manifest = write_prediction_curve_plots(
        aligned,
        output_dir=args.output_dir,
        split_name=args.split,
        targets=_parse_csv_tuple(args.targets) or tuple(DEFAULT_TARGET_COLUMNS),
        log_ids=_parse_csv_tuple(args.log_ids),
        max_logs=args.max_logs,
        zoom_samples=args.zoom_samples,
        max_overview_points=args.max_overview_points,
        relative_time_axis=args.relative_time_axis,
    )
    print(f"wrote {len(manifest)} plots")
    print(f"manifest: {args.output_dir / 'prediction_curve_manifest.csv'}")
    print(f"index: {args.output_dir / 'index.md'}")


if __name__ == "__main__":
    main()
