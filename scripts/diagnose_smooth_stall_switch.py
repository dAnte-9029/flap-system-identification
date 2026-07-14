#!/usr/bin/env python3
"""Compare hard and smooth stall switching for the nominal DeLaurier prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_nested_prior_nonlinear_calibration_exp3 import FIXED_PRIOR_PARAMS, SPLITS, TARGETS
from scripts.run_nested_prior_regression_calibration_exp2 import (
    DEFAULT_EXPORT_SPLIT_ROOT,
    DEFAULT_EXPORTER,
    DEFAULT_METADATA,
    DEFAULT_PYTHON_EXE,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/20260706_smooth_stall_switch_diagnostic_v1"


def _parse_deltas(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def _export_command(
    *,
    python_exe: Path,
    exporter: Path,
    split_root: Path,
    metadata: Path,
    output_root: Path,
    params: dict[str, float | bool],
    stall_smoothing_width_deg: float,
    chunk_size: int,
    device: str,
    max_rows_for_tests: int | None,
) -> list[str]:
    cmd = [
        str(python_exe),
        str(exporter),
        "--split-root",
        str(split_root),
        "--metadata",
        str(metadata),
        "--output-root",
        str(output_root),
        "--overwrite",
        "--chunk-size",
        str(int(chunk_size)),
        "--device",
        str(device),
        "--theta-w-deg",
        "0.0",
        "--twist-eta-max-deg",
        str(float(params["twist_eta_max_deg"])),
        "--twist-eta-limit-deg",
        str(float(params["twist_eta_max_deg"])),
        "--alpha0-deg",
        str(float(params["alpha0_deg"])),
        "--eta-s",
        str(float(params["eta_s"])),
        "--cd-f",
        str(float(params["cd_f"])),
        "--alpha-stall-min-deg",
        str(float(params["alpha_stall_min_deg"])),
        "--alpha-stall-max-deg",
        str(float(params["alpha_stall_max_deg"])),
        "--cd-cf",
        str(float(params["cd_cf"])),
        "--xi",
        str(float(params["xi"])),
        "--stall-smoothing-width-deg",
        str(float(stall_smoothing_width_deg)),
        "--include-diagnostics",
    ]
    if bool(params.get("enable_separation", False)):
        cmd.append("--enable-separation")
    if max_rows_for_tests is not None:
        cmd.extend(["--max-rows-for-tests", str(int(max_rows_for_tests))])
    return cmd


def _run_export(
    *,
    python_exe: Path,
    exporter: Path,
    split_root: Path,
    metadata: Path,
    output_root: Path,
    params: dict[str, float | bool],
    stall_smoothing_width_deg: float,
    chunk_size: int,
    device: str,
    max_rows_for_tests: int | None,
    reuse_existing: bool,
) -> None:
    if reuse_existing and all((output_root / f"{split}_predictions.parquet").exists() for split in SPLITS):
        return
    cmd = _export_command(
        python_exe=python_exe,
        exporter=exporter,
        split_root=split_root,
        metadata=metadata,
        output_root=output_root,
        params=params,
        stall_smoothing_width_deg=stall_smoothing_width_deg,
        chunk_size=chunk_size,
        device=device,
        max_rows_for_tests=max_rows_for_tests,
    )
    completed = subprocess.run(
        cmd,
        cwd=exporter.parents[2],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            cmd,
            output=completed.stdout,
            stderr=completed.stderr,
        )


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _mae(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values)))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if float(np.std(a)) <= 1.0e-12 or float(np.std(b)) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _force_metrics(samples: pd.DataFrame, hard: pd.DataFrame, smooth: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    hard_force = hard.loc[:, TARGETS].to_numpy(dtype=float)
    smooth_force = smooth.loc[:, TARGETS].to_numpy(dtype=float)
    labels = samples.loc[:, TARGETS].to_numpy(dtype=float)
    diff = smooth_force - hard_force

    metrics["force_drift_rmse_mean"] = _rmse(diff)
    metrics["force_drift_mae_mean"] = _mae(diff)
    metrics["force_drift_over_hard_rms"] = _rmse(diff) / max(_rmse(hard_force), 1.0e-12)
    metrics["hard_prior_rmse_mean"] = _rmse(hard_force - labels)
    metrics["smooth_prior_rmse_mean"] = _rmse(smooth_force - labels)
    metrics["smooth_minus_hard_prior_rmse_mean"] = metrics["smooth_prior_rmse_mean"] - metrics["hard_prior_rmse_mean"]

    for target in TARGETS:
        h = hard[target].to_numpy(dtype=float)
        s = smooth[target].to_numpy(dtype=float)
        y = samples[target].to_numpy(dtype=float)
        d = s - h
        metrics[f"{target}_drift_rmse"] = _rmse(d)
        metrics[f"{target}_drift_mae"] = _mae(d)
        metrics[f"{target}_hard_smooth_corr"] = _corr(h, s)
        metrics[f"{target}_hard_prior_rmse"] = _rmse(h - y)
        metrics[f"{target}_smooth_prior_rmse"] = _rmse(s - y)

    for column in (
        "sep_ratio",
        "sep_weight_mean",
        "sep_weight_mid_area_ratio",
        "stall_transition_3delta_area_ratio",
        "alpha_le_mean",
        "alpha_stall_mean",
    ):
        if column in smooth.columns:
            values = smooth[column].to_numpy(dtype=float)
            metrics[f"smooth_{column}_mean"] = float(np.mean(values))
            metrics[f"smooth_{column}_p50"] = float(np.quantile(values, 0.50))
            metrics[f"smooth_{column}_p95"] = float(np.quantile(values, 0.95))
    if "sep_weight_mean" in smooth.columns:
        sep = smooth["sep_weight_mean"].to_numpy(dtype=float)
        metrics["smooth_mid_sep_weight_sample_fraction"] = float(np.mean((sep > 0.1) & (sep < 0.9)))
    if "stall_transition_3delta_area_ratio" in smooth.columns:
        band = smooth["stall_transition_3delta_area_ratio"].to_numpy(dtype=float)
        metrics["smooth_transition_band_any_sample_fraction"] = float(np.mean(band > 0.0))

    return metrics


def _summarize_variant(
    *,
    split_root: Path,
    hard_root: Path,
    smooth_root: Path,
    variant: str,
    max_rows_for_tests: int | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pooled_samples: list[pd.DataFrame] = []
    pooled_hard: list[pd.DataFrame] = []
    pooled_smooth: list[pd.DataFrame] = []
    for split in SPLITS:
        samples = pd.read_parquet(split_root / f"{split}_samples.parquet").reset_index(drop=True)
        if max_rows_for_tests is not None:
            samples = samples.head(int(max_rows_for_tests)).copy()
        hard = pd.read_parquet(hard_root / f"{split}_predictions.parquet").reset_index(drop=True)
        smooth = pd.read_parquet(smooth_root / f"{split}_predictions.parquet").reset_index(drop=True)
        if len(samples) != len(hard) or len(samples) != len(smooth):
            raise ValueError(f"row count mismatch for {split}: samples={len(samples)}, hard={len(hard)}, smooth={len(smooth)}")
        metrics = _force_metrics(samples, hard, smooth)
        rows.append({"variant": variant, "split": split, "n_rows": int(len(samples)), **metrics})
        pooled_samples.append(samples)
        pooled_hard.append(hard)
        pooled_smooth.append(smooth)

    metrics = _force_metrics(
        pd.concat(pooled_samples, ignore_index=True),
        pd.concat(pooled_hard, ignore_index=True),
        pd.concat(pooled_smooth, ignore_index=True),
    )
    rows.append(
        {
            "variant": variant,
            "split": "all",
            "n_rows": int(sum(len(frame) for frame in pooled_samples)),
            **metrics,
        }
    )
    return rows


def run_diagnostic(args: argparse.Namespace) -> dict[str, object]:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    params = dict(FIXED_PRIOR_PARAMS)

    hard_root = output_root / "hard_nominal"
    _run_export(
        python_exe=args.python_exe,
        exporter=args.exporter,
        split_root=args.split_root,
        metadata=args.metadata,
        output_root=hard_root,
        params=params,
        stall_smoothing_width_deg=0.0,
        chunk_size=args.chunk_size,
        device=args.device,
        max_rows_for_tests=args.max_rows_for_tests,
        reuse_existing=args.reuse_existing,
    )

    rows: list[dict[str, object]] = []
    for delta in args.deltas_deg:
        variant = f"smooth_delta_{delta:g}deg"
        smooth_root = output_root / variant
        _run_export(
            python_exe=args.python_exe,
            exporter=args.exporter,
            split_root=args.split_root,
            metadata=args.metadata,
            output_root=smooth_root,
            params=params,
            stall_smoothing_width_deg=float(delta),
            chunk_size=args.chunk_size,
            device=args.device,
            max_rows_for_tests=args.max_rows_for_tests,
            reuse_existing=args.reuse_existing,
        )
        rows.extend(
            _summarize_variant(
                split_root=args.split_root,
                hard_root=hard_root,
                smooth_root=smooth_root,
                variant=variant,
                max_rows_for_tests=args.max_rows_for_tests,
            )
        )

    summary = pd.DataFrame(rows)
    summary_path = output_root / "smooth_stall_switch_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "output_root": str(output_root),
        "split_root": str(args.split_root),
        "metadata": str(args.metadata),
        "exporter": str(args.exporter),
        "python_exe": str(args.python_exe),
        "params": params,
        "deltas_deg": [float(value) for value in args.deltas_deg],
        "device": args.device,
        "chunk_size": int(args.chunk_size),
        "max_rows_for_tests": None if args.max_rows_for_tests is None else int(args.max_rows_for_tests),
        "summary_csv": str(summary_path),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_EXPORT_SPLIT_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--exporter", type=Path, default=DEFAULT_EXPORTER)
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON_EXE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--deltas-deg", type=_parse_deltas, default=(1.0, 2.0, 3.0))
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--max-rows-for-tests", type=int, default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    manifest = run_diagnostic(_parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
