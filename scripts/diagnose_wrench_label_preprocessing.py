#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.signal_preprocessing import highpass_energy_fraction, nominal_sample_rate_hz  # noqa: E402


TARGET_COLUMNS = ["fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b"]
HIGHPASS_CUTOFFS_HZ = [8.0, 12.0, 20.0]


def _finite_pair(raw: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(raw) & np.isfinite(clean)
    return raw[finite], clean[finite]


def _corr(raw: np.ndarray, clean: np.ndarray) -> float:
    raw_valid, clean_valid = _finite_pair(raw, clean)
    if len(raw_valid) < 3 or np.std(raw_valid) < 1e-12 or np.std(clean_valid) < 1e-12:
        return float("nan")
    return float(np.corrcoef(raw_valid, clean_valid)[0, 1])


def _rmse(raw: np.ndarray, clean: np.ndarray) -> float:
    raw_valid, clean_valid = _finite_pair(raw, clean)
    if len(raw_valid) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(clean_valid - raw_valid))))


def _top_overlap(raw: np.ndarray, clean: np.ndarray, quantile: float) -> float:
    raw_valid, clean_valid = _finite_pair(raw, clean)
    if len(raw_valid) == 0:
        return float("nan")
    raw_abs = np.abs(raw_valid)
    clean_abs = np.abs(clean_valid)
    raw_top = raw_abs >= float(np.quantile(raw_abs, quantile))
    clean_top = clean_abs >= float(np.quantile(clean_abs, quantile))
    denom = int(np.sum(raw_top))
    if denom == 0:
        return float("nan")
    return float(np.sum(raw_top & clean_top) / denom)


def _jump_p99(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    return float(np.quantile(np.abs(np.diff(values)), 0.99))


def compute_channel_metrics(
    raw_frame: pd.DataFrame,
    clean_frame: pd.DataFrame,
    *,
    split: str,
    channel: str,
    log_id: str | None = None,
) -> dict[str, Any]:
    raw = raw_frame[channel].to_numpy(dtype=float)
    clean = clean_frame[channel].to_numpy(dtype=float)
    raw_valid, clean_valid = _finite_pair(raw, clean)
    sample_rate_hz = nominal_sample_rate_hz(raw_frame)
    row: dict[str, Any] = {
        "split": split,
        "log_id": "__all__" if log_id is None else log_id,
        "channel": channel,
        "sample_count": int(len(raw_valid)),
        "finite_ratio": float(len(raw_valid) / len(raw)) if len(raw) else float("nan"),
        "std_raw": float(np.std(raw_valid)) if len(raw_valid) else float("nan"),
        "std_clean": float(np.std(clean_valid)) if len(clean_valid) else float("nan"),
        "rmse_clean_vs_raw": _rmse(raw, clean),
        "corr_clean_vs_raw": _corr(raw, clean),
        "p95_abs_raw": float(np.quantile(np.abs(raw_valid), 0.95)) if len(raw_valid) else float("nan"),
        "p95_abs_clean": float(np.quantile(np.abs(clean_valid), 0.95)) if len(clean_valid) else float("nan"),
        "p99_abs_raw": float(np.quantile(np.abs(raw_valid), 0.99)) if len(raw_valid) else float("nan"),
        "p99_abs_clean": float(np.quantile(np.abs(clean_valid), 0.99)) if len(clean_valid) else float("nan"),
        "p999_abs_raw": float(np.quantile(np.abs(raw_valid), 0.999)) if len(raw_valid) else float("nan"),
        "p999_abs_clean": float(np.quantile(np.abs(clean_valid), 0.999)) if len(clean_valid) else float("nan"),
        "max_abs_raw": float(np.max(np.abs(raw_valid))) if len(raw_valid) else float("nan"),
        "max_abs_clean": float(np.max(np.abs(clean_valid))) if len(clean_valid) else float("nan"),
        "sample_to_sample_jump_p99_raw": _jump_p99(raw),
        "sample_to_sample_jump_p99_clean": _jump_p99(clean),
        "top1pct_overlap_fraction": _top_overlap(raw, clean, 0.99),
        "top5pct_overlap_fraction": _top_overlap(raw, clean, 0.95),
    }
    for cutoff_hz in HIGHPASS_CUTOFFS_HZ:
        suffix = str(cutoff_hz).replace(".", "p")
        row[f"highpass_energy_fraction_raw_{suffix}hz"] = highpass_energy_fraction(
            raw,
            sample_rate_hz=sample_rate_hz,
            cutoff_hz=cutoff_hz,
        )
        row[f"highpass_energy_fraction_clean_{suffix}hz"] = highpass_energy_fraction(
            clean,
            sample_rate_hz=sample_rate_hz,
            cutoff_hz=cutoff_hz,
        )
    return row


def compute_metrics_tables(
    *,
    raw_split_root: Path,
    clean_split_root: Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []
    for split in splits:
        raw_frame = pd.read_parquet(raw_split_root / f"{split}_samples.parquet")
        clean_frame = pd.read_parquet(clean_split_root / f"{split}_samples.parquet")
        if len(raw_frame) != len(clean_frame):
            raise ValueError(f"Raw and clean split lengths differ for {split}: {len(raw_frame)} vs {len(clean_frame)}")
        for channel in TARGET_COLUMNS:
            split_rows.append(compute_channel_metrics(raw_frame, clean_frame, split=split, channel=channel))
        if "log_id" in raw_frame.columns:
            for log_id, raw_log in raw_frame.groupby("log_id", sort=False):
                clean_log = clean_frame.loc[raw_log.index]
                for channel in TARGET_COLUMNS:
                    log_rows.append(
                        compute_channel_metrics(
                            raw_log,
                            clean_log,
                            split=split,
                            channel=channel,
                            log_id=str(log_id),
                        )
                    )
    return pd.DataFrame(split_rows), pd.DataFrame(log_rows)


def write_report(
    split_metrics: pd.DataFrame,
    log_metrics: pd.DataFrame,
    *,
    raw_split_root: Path,
    clean_split_root: Path,
    output_dir: Path,
) -> Path:
    lines = [
        "# Smoothed Time-Aligned Wrench Label Diagnostics",
        "",
        f"- raw_split_root: `{raw_split_root}`",
        f"- clean_split_root: `{clean_split_root}`",
        "",
        "## Split-Level Metrics",
        "",
        split_metrics.to_csv(index=False),
        "",
        "## Log-Level Summary",
        "",
    ]
    if log_metrics.empty:
        lines.append("No `log_id` column was available for log-level metrics.")
    else:
        summary_cols = [
            "split",
            "channel",
            "corr_clean_vs_raw",
            "rmse_clean_vs_raw",
            "p99_abs_raw",
            "p99_abs_clean",
            "sample_to_sample_jump_p99_raw",
            "sample_to_sample_jump_p99_clean",
        ]
        summary = log_metrics[summary_cols].groupby(["split", "channel"], as_index=False).median(numeric_only=True)
        lines.append(summary.to_csv(index=False))
    report_path = output_dir / "label_quality_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_diagnostics(
    *,
    raw_split_root: str | Path,
    clean_split_root: str | Path,
    output_dir: str | Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, str]:
    raw_split_root = Path(raw_split_root)
    clean_split_root = Path(clean_split_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_metrics, log_metrics = compute_metrics_tables(
        raw_split_root=raw_split_root,
        clean_split_root=clean_split_root,
        splits=splits,
    )
    split_path = output_dir / "label_quality_by_split_channel.csv"
    log_path = output_dir / "label_quality_by_log_channel.csv"
    split_metrics.to_csv(split_path, index=False)
    log_metrics.to_csv(log_path, index=False)
    report_path = write_report(
        split_metrics,
        log_metrics,
        raw_split_root=raw_split_root,
        clean_split_root=clean_split_root,
        output_dir=output_dir,
    )
    return {
        "split_metrics": str(split_path),
        "log_metrics": str(log_path),
        "report": str(report_path),
    }


def _parse_splits(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("splits must not be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare raw and preprocessed effective-wrench labels.")
    parser.add_argument("--raw-split-root", required=True, type=Path)
    parser.add_argument("--clean-split-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--splits", type=_parse_splits, default=("train", "val", "test"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_diagnostics(
        raw_split_root=args.raw_split_root,
        clean_split_root=args.clean_split_root,
        output_dir=args.output_dir,
        splits=args.splits,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
