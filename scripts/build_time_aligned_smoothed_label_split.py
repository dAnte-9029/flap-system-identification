#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.metadata import load_aircraft_metadata  # noqa: E402
from system_identification.pipeline import _compute_effective_wrench_labels, compute_kinematic_derivatives  # noqa: E402
from system_identification.signal_preprocessing import (  # noqa: E402
    apply_groupwise_time_shift,
    groupwise_lowpass_filter,
)


TARGET_COLUMNS = ["fx_b", "fy_b", "fz_b", "mx_b", "my_b", "mz_b"]
LINEAR_ACCELERATION_COLUMNS = [
    "vehicle_local_position.ax_smooth",
    "vehicle_local_position.ay_smooth",
    "vehicle_local_position.az_smooth",
]
ANGULAR_ACCELERATION_COLUMNS = [
    "vehicle_angular_velocity.xyz_derivative_smooth[0]",
    "vehicle_angular_velocity.xyz_derivative_smooth[1]",
    "vehicle_angular_velocity.xyz_derivative_smooth[2]",
]
DEFAULT_LAG_CANDIDATES = {
    "phase_raw_rad": [-0.04, -0.02, 0.0, 0.02, 0.04],
    "phase_raw_unwrapped_rad": [-0.04, -0.02, 0.0, 0.02, 0.04],
    "flap_frequency_hz": [-0.04, -0.02, 0.0, 0.02, 0.04],
    "airspeed_validated.true_airspeed_m_s": [-0.10, -0.05, 0.0, 0.05, 0.10],
    "servo_left_elevon": [-0.08, -0.04, 0.0, 0.04, 0.08],
    "servo_right_elevon": [-0.08, -0.04, 0.0, 0.04, 0.08],
    "servo_rudder": [-0.08, -0.04, 0.0, 0.04, 0.08],
}
DEFAULT_FILTER_CONFIG = {
    "flap_frequency_hz": {"method": "butterworth", "order": 2, "cutoff_hz": 12.0},
    "airspeed_validated.true_airspeed_m_s": {"method": "butterworth", "order": 2, "cutoff_hz": 5.0},
    "servo_left_elevon": {"method": "first_order", "time_constant_s": 0.04},
    "servo_right_elevon": {"method": "first_order", "time_constant_s": 0.04},
    "servo_rudder": {"method": "first_order", "time_constant_s": 0.04},
}


def _load_json_config(path: Path | None, default: dict[str, Any]) -> dict[str, Any]:
    if path is None:
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_corr(a: pd.Series, b: pd.Series) -> float:
    x = a.to_numpy(dtype=float)
    y = b.to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 3:
        return float("nan")
    x = x[finite]
    y = y[finite]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def run_train_only_lag_sweep(
    train_frame: pd.DataFrame,
    *,
    lag_candidates: dict[str, list[float]],
    artifact_dir: Path,
) -> dict[str, float]:
    rows: list[dict[str, Any]] = []
    selected: dict[str, float] = {}
    for column, candidates in lag_candidates.items():
        if column not in train_frame.columns:
            continue
        for lag_s in candidates:
            shifted = apply_groupwise_time_shift(train_frame, column, lag_s=float(lag_s))
            rows.append(
                {
                    "column": column,
                    "lag_s": float(lag_s),
                    "finite_ratio": float(np.isfinite(shifted.to_numpy(dtype=float)).mean()),
                    "corr_fx_b": _finite_corr(shifted, train_frame["fx_b"]) if "fx_b" in train_frame.columns else float("nan"),
                    "corr_fz_b": _finite_corr(shifted, train_frame["fz_b"]) if "fz_b" in train_frame.columns else float("nan"),
                }
            )
        selected[column] = 0.0 if 0.0 in candidates else float(min(candidates, key=lambda value: abs(value)))

    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(artifact_dir / "lag_sweep_train_metrics.csv", index=False)
    selection_payload = {
        "policy": "conservative_zero_lag_default_after_train_only_diagnostics",
        "selected_lags_s": selected,
    }
    (artifact_dir / "lag_selection.json").write_text(json.dumps(selection_payload, indent=2, sort_keys=True), encoding="utf-8")
    return selected


def _aligned_column_name(column: str) -> str:
    return f"{column}_aligned"


def apply_selected_lags(frame: pd.DataFrame, selected_lags_s: dict[str, float]) -> pd.DataFrame:
    output = frame.copy()
    for column, lag_s in selected_lags_s.items():
        if column in output.columns:
            output[_aligned_column_name(column)] = apply_groupwise_time_shift(output, column, lag_s=float(lag_s))
    return output


def _filtered_column_name(column: str) -> str:
    return f"{column}_filt"


def apply_input_filters(frame: pd.DataFrame, filter_config: dict[str, dict[str, Any]]) -> pd.DataFrame:
    output = frame.copy()
    for column, config in filter_config.items():
        source_column = _aligned_column_name(column) if _aligned_column_name(column) in output.columns else column
        if source_column not in output.columns:
            continue
        output[_filtered_column_name(column)] = groupwise_lowpass_filter(
            output,
            source_column,
            method=str(config.get("method", "butterworth")),
            order=int(config.get("order", 2)),
            cutoff_hz=float(config["cutoff_hz"]) if "cutoff_hz" in config else None,
            time_constant_s=float(config["time_constant_s"]) if "time_constant_s" in config else None,
        )
    return output


def rewrite_frame(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    derivative_method: str,
    window_s: float,
    polyorder: int,
    force_label_source: str,
    moment_label_source: str,
    selected_lags_s: dict[str, float] | None = None,
    filter_config: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    output = frame.copy()
    if selected_lags_s:
        output = apply_selected_lags(output, selected_lags_s)
    if filter_config:
        output = apply_input_filters(output, filter_config)

    derivatives = compute_kinematic_derivatives(
        output,
        method=derivative_method,
        window_s=window_s,
        polyorder=polyorder,
        group_columns=["log_id", "segment_id"],
    )
    for column in derivatives.columns:
        output[column] = derivatives[column].to_numpy()

    force_b, moment_b, label_valid = _compute_effective_wrench_labels(
        output,
        metadata,
        linear_acceleration_columns=LINEAR_ACCELERATION_COLUMNS if force_label_source == "smooth" else None,
        angular_acceleration_columns=ANGULAR_ACCELERATION_COLUMNS if moment_label_source == "smooth" else None,
    )
    for idx, column in enumerate(["fx_b", "fy_b", "fz_b"]):
        output[column] = force_b[:, idx]
    for idx, column in enumerate(["mx_b", "my_b", "mz_b"]):
        output[column] = moment_b[:, idx]
    output["label_valid"] = label_valid
    output["label_variant"] = "smoothed_time_aligned_wrench"
    output["linear_derivative_source"] = force_label_source
    output["angular_derivative_source"] = moment_label_source
    output["label_reconstruction_valid"] = label_valid
    return output


def build_time_aligned_smoothed_label_split(
    *,
    split_root: str | Path,
    metadata_path: str | Path,
    output_root: str | Path,
    artifact_dir: str | Path,
    derivative_method: str = "savgol",
    window_s: float = 0.12,
    polyorder: int = 2,
    force_label_source: str = "smooth",
    moment_label_source: str = "smooth",
    enable_lag_sweep: bool = False,
    enable_input_filtering: bool = False,
    lag_config_path: str | Path | None = None,
    filter_config_path: str | Path | None = None,
) -> dict[str, str]:
    if force_label_source not in {"raw", "smooth"}:
        raise ValueError(f"Unknown force_label_source: {force_label_source}")
    if moment_label_source not in {"raw", "smooth"}:
        raise ValueError(f"Unknown moment_label_source: {moment_label_source}")

    split_root = Path(split_root)
    metadata_path = Path(metadata_path)
    output_root = Path(output_root)
    artifact_dir = Path(artifact_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_aircraft_metadata(metadata_path)

    lag_candidates = _load_json_config(Path(lag_config_path) if lag_config_path else None, DEFAULT_LAG_CANDIDATES)
    filter_config = _load_json_config(Path(filter_config_path) if filter_config_path else None, DEFAULT_FILTER_CONFIG)
    selected_lags_s: dict[str, float] = {}
    if enable_lag_sweep:
        train_frame = pd.read_parquet(split_root / "train_samples.parquet")
        selected_lags_s = run_train_only_lag_sweep(train_frame, lag_candidates=lag_candidates, artifact_dir=artifact_dir)

    active_filter_config = filter_config if enable_input_filtering else None
    split_sample_counts: dict[str, int] = {}
    label_valid_ratios: dict[str, float] = {}
    for split_name in ["train", "val", "test"]:
        frame = pd.read_parquet(split_root / f"{split_name}_samples.parquet")
        rewritten = rewrite_frame(
            frame,
            metadata,
            derivative_method=derivative_method,
            window_s=window_s,
            polyorder=polyorder,
            force_label_source=force_label_source,
            moment_label_source=moment_label_source,
            selected_lags_s=selected_lags_s,
            filter_config=active_filter_config,
        )
        rewritten.to_parquet(output_root / f"{split_name}_samples.parquet", index=False)
        split_sample_counts[split_name] = int(len(rewritten))
        label_valid_ratios[split_name] = float(rewritten["label_valid"].mean())

    for filename in ["train_logs.csv", "val_logs.csv", "test_logs.csv", "all_logs.csv"]:
        source = split_root / filename
        if source.exists():
            shutil.copy2(source, output_root / filename)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": output_root.name,
        "source_split_root": str(split_root),
        "metadata_path": str(metadata_path),
        "label_policy": "effective wrench recomputed from pre-smoothed kinematic derivatives",
        "derivative": {
            "method": derivative_method,
            "window_s": float(window_s),
            "polyorder": int(polyorder),
            "force_label_source": force_label_source,
            "moment_label_source": moment_label_source,
            "linear_acceleration_columns": LINEAR_ACCELERATION_COLUMNS,
            "angular_acceleration_columns": ANGULAR_ACCELERATION_COLUMNS,
        },
        "lag_sweep": {
            "enabled": bool(enable_lag_sweep),
            "selected_lags_s": selected_lags_s,
        },
        "input_filtering": {
            "enabled": bool(enable_input_filtering),
            "config": active_filter_config or {},
        },
        "split_sample_counts": split_sample_counts,
        "label_valid_ratios": label_valid_ratios,
    }
    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"output_root": str(output_root), "manifest_path": str(manifest_path), "artifact_dir": str(artifact_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a split with pre-smoothed, optionally time-aligned effective-wrench labels.")
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--derivative-method", choices=["raw", "savgol", "cubic_spline"], default="savgol")
    parser.add_argument("--window-s", type=float, default=0.12)
    parser.add_argument("--polyorder", type=int, default=2)
    parser.add_argument("--force-label-source", choices=["raw", "smooth"], default="smooth")
    parser.add_argument("--moment-label-source", choices=["raw", "smooth"], default="smooth")
    parser.add_argument("--enable-lag-sweep", action="store_true")
    parser.add_argument("--enable-input-filtering", action="store_true")
    parser.add_argument("--lag-config", type=Path)
    parser.add_argument("--filter-config", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_time_aligned_smoothed_label_split(
        split_root=args.split_root,
        metadata_path=args.metadata,
        output_root=args.output,
        artifact_dir=args.artifact_dir,
        derivative_method=args.derivative_method,
        window_s=args.window_s,
        polyorder=args.polyorder,
        force_label_source=args.force_label_source,
        moment_label_source=args.moment_label_source,
        enable_lag_sweep=args.enable_lag_sweep,
        enable_input_filtering=args.enable_input_filtering,
        lag_config_path=args.lag_config,
        filter_config_path=args.filter_config,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
