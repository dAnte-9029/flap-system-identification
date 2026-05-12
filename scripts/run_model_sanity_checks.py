#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import system_identification.training as training_module
from system_identification.training import (
    DEFAULT_TARGET_COLUMNS,
    evaluate_model_bundle,
    evaluate_model_bundle_by_log,
    predict_model_bundle,
    prepare_feature_target_frames,
)


SPLIT_NAMES = ("train", "val", "test")
ACCELERATION_PATTERNS = (
    re.compile(r"(^|[._])a[xyz]($|[^A-Za-z0-9])"),
    re.compile(r"accel", re.IGNORECASE),
    re.compile(r"acceleration", re.IGNORECASE),
    re.compile(r"xyz_derivative", re.IGNORECASE),
)
SIMPLE_LINEAR_FEATURES = [
    "phase_corrected_sin",
    "phase_corrected_cos",
    "phase_corrected_rad",
    "wing_stroke_angle_rad",
    "flap_frequency_hz",
    "cycle_flap_frequency_hz",
    "motor_cmd_0",
    "servo_left_elevon",
    "servo_right_elevon",
    "servo_rudder",
    "elevator_like",
    "aileron_like",
    "airspeed_validated.true_airspeed_m_s",
    "airspeed_validated.calibrated_airspeed_m_s",
    "airspeed_validated.pitch_filtered",
    "vehicle_air_data.rho",
    "dynamic_pressure_pa",
    "vehicle_local_position.vx",
    "vehicle_local_position.vy",
    "vehicle_local_position.vz",
    "vehicle_angular_velocity.xyz[0]",
    "vehicle_angular_velocity.xyz[1]",
    "vehicle_angular_velocity.xyz[2]",
]


def _read_split_frame(split_root: Path, split_name: str) -> pd.DataFrame:
    path = split_root / f"{split_name}_samples.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing split parquet: {path}")
    return pd.read_parquet(path)


def load_split_frames(split_root: Path, split_names: tuple[str, ...] = SPLIT_NAMES) -> dict[str, pd.DataFrame]:
    return {split_name: _read_split_frame(split_root, split_name) for split_name in split_names}


def _split_log_ids(frame: pd.DataFrame) -> set[str]:
    if "log_id" not in frame.columns:
        return set()
    return set(frame["log_id"].astype(str).dropna().unique())


def build_split_protocol_table(split_root: Path) -> pd.DataFrame:
    frames = load_split_frames(split_root)
    rows: list[dict[str, Any]] = []

    for split_name, frame in frames.items():
        has_log_id = "log_id" in frame.columns
        rows.append(
            {
                "check": "split_summary",
                "split": split_name,
                "split_a": split_name,
                "split_b": "",
                "passed": bool(has_log_id),
                "sample_count": int(len(frame)),
                "log_count": int(len(_split_log_ids(frame))),
                "matched_count": 0,
                "matched_columns": "",
                "details": "log_id column present" if has_log_id else "log_id column missing",
            }
        )

    log_ids = {split_name: _split_log_ids(frame) for split_name, frame in frames.items()}
    for split_a, split_b in combinations(SPLIT_NAMES, 2):
        overlap = sorted(log_ids[split_a] & log_ids[split_b])
        rows.append(
            {
                "check": "log_overlap",
                "split": "",
                "split_a": split_a,
                "split_b": split_b,
                "passed": bool(len(overlap) == 0),
                "sample_count": 0,
                "log_count": 0,
                "matched_count": int(len(overlap)),
                "matched_columns": ",".join(overlap),
                "details": "whole-log split ids are disjoint" if not overlap else "shared log_id across splits",
            }
        )

    manifest_path = split_root / "split_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        split_policy = str(manifest.get("split_policy", ""))
        rows.append(
            {
                "check": "manifest_split_policy",
                "split": "",
                "split_a": "",
                "split_b": "",
                "passed": bool(split_policy == "whole_log"),
                "sample_count": 0,
                "log_count": 0,
                "matched_count": 0,
                "matched_columns": split_policy,
                "details": f"split_policy={split_policy}",
            }
        )

    return pd.DataFrame(rows)


def _base_feature_name(column: str) -> str:
    return str(column).split("@", 1)[0]


def _bundle_input_columns(bundle: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    for key in ("feature_columns", "sequence_feature_columns", "current_feature_columns"):
        value = bundle.get(key)
        if value is None:
            continue
        columns.extend(str(column) for column in value)
    return sorted(set(columns))


def _acceleration_like_columns(columns: list[str]) -> list[str]:
    matched: list[str] = []
    for column in columns:
        base = _base_feature_name(column)
        if any(pattern.search(base) for pattern in ACCELERATION_PATTERNS):
            matched.append(column)
    return matched


def scan_bundle_inputs(bundle: dict[str, Any]) -> pd.DataFrame:
    input_columns = _bundle_input_columns(bundle)
    target_columns = {str(column) for column in bundle.get("target_columns", DEFAULT_TARGET_COLUMNS)}
    base_columns = {_base_feature_name(column) for column in input_columns}

    acceleration_columns = _acceleration_like_columns(input_columns)
    target_leak_columns = sorted(column for column in input_columns if _base_feature_name(column) in target_columns)
    suspicious_columns = sorted(
        column
        for column in input_columns
        if any(token in _base_feature_name(column).lower() for token in ("target", "label", "wrench"))
    )

    checks = [
        (
            "no_acceleration_inputs",
            acceleration_columns,
            "No linear/angular acceleration or derivative-like inputs should be present.",
        ),
        (
            "no_target_columns_in_inputs",
            target_leak_columns,
            "Target columns must not be present as current or sequence inputs.",
        ),
        (
            "no_suspicious_target_named_inputs",
            suspicious_columns,
            "Columns with target/label/wrench names are suspicious and should be reviewed.",
        ),
    ]
    rows = [
        {
            "check": check,
            "passed": bool(len(columns) == 0),
            "matched_count": int(len(columns)),
            "matched_columns": ",".join(columns),
            "input_column_count": int(len(input_columns)),
            "base_input_column_count": int(len(base_columns)),
            "details": details,
        }
        for check, columns, details in checks
    ]
    return pd.DataFrame(rows)


def _flatten_metrics(metrics: dict[str, Any], *, baseline_name: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "baseline": baseline_name,
        "sample_count": int(metrics["sample_count"]),
        "overall_mae": float(metrics["overall_mae"]),
        "overall_rmse": float(metrics["overall_rmse"]),
        "overall_r2": float(metrics["overall_r2"]),
    }
    for target_name, target_metrics in metrics["per_target"].items():
        row[f"{target_name}_mae"] = float(target_metrics["mae"])
        row[f"{target_name}_rmse"] = float(target_metrics["rmse"])
        row[f"{target_name}_r2"] = float(target_metrics["r2"])
    return row


def compute_mean_baseline_metrics(
    train_targets: pd.DataFrame,
    test_targets: pd.DataFrame,
    *,
    target_columns: list[str],
) -> dict[str, Any]:
    train_mean = train_targets.loc[:, target_columns].mean(axis=0).to_numpy(dtype=np.float64)
    y_true = test_targets.loc[:, target_columns].to_numpy(dtype=np.float64, copy=False)
    y_pred = np.repeat(train_mean[None, :], len(test_targets), axis=0)
    metrics = training_module._metrics_from_arrays(
        y_true,
        y_pred,
        target_columns=target_columns,
        split_name="test",
    )
    return _flatten_metrics(metrics, baseline_name="train_target_mean")


def compute_permuted_target_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_columns: list[str],
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y_true))
    permuted_true = y_true[order]
    metrics = training_module._metrics_from_arrays(
        permuted_true,
        y_pred,
        target_columns=target_columns,
        split_name="test_permuted_targets",
    )
    return _flatten_metrics(metrics, baseline_name="model_predictions_vs_permuted_test_targets")


def _select_available_features(frame: pd.DataFrame, target_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    available_features: list[str] = []
    for column in SIMPLE_LINEAR_FEATURES:
        try:
            prepare_feature_target_frames(frame.head(2).copy(), [column], target_columns)
        except ValueError:
            continue
        available_features.append(column)
    features, _ = prepare_feature_target_frames(frame, available_features, target_columns)
    return features, available_features


def _standardize_matrix(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_median = np.nanmedian(train_x, axis=0)
    train_x = np.where(np.isfinite(train_x), train_x, train_median)
    test_x = np.where(np.isfinite(test_x), test_x, train_median)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return (train_x - mean) / std, (test_x - mean) / std


def compute_simple_linear_baseline_metrics(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    target_columns: list[str],
    ridge_alpha: float = 1e-6,
) -> tuple[dict[str, Any], list[str]]:
    train_features, feature_columns = _select_available_features(train_frame, target_columns)
    test_features, _ = prepare_feature_target_frames(test_frame, feature_columns, target_columns)
    _, train_targets = prepare_feature_target_frames(train_frame, feature_columns, target_columns)
    _, test_targets = prepare_feature_target_frames(test_frame, feature_columns, target_columns)

    train_x, test_x = _standardize_matrix(
        train_features.to_numpy(dtype=np.float64, copy=False),
        test_features.to_numpy(dtype=np.float64, copy=False),
    )
    train_x = np.concatenate([np.ones((len(train_x), 1), dtype=np.float64), train_x], axis=1)
    test_x = np.concatenate([np.ones((len(test_x), 1), dtype=np.float64), test_x], axis=1)

    train_y = train_targets.loc[:, target_columns].to_numpy(dtype=np.float64, copy=False)
    test_y = test_targets.loc[:, target_columns].to_numpy(dtype=np.float64, copy=False)
    regularizer = ridge_alpha * np.eye(train_x.shape[1], dtype=np.float64)
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(train_x.T @ train_x + regularizer, train_x.T @ train_y)
    predictions = test_x @ weights
    metrics = training_module._metrics_from_arrays(
        test_y,
        predictions,
        target_columns=target_columns,
        split_name="test",
    )
    return _flatten_metrics(metrics, baseline_name="simple_linear_physics_features"), feature_columns


def _model_metrics_row(bundle: dict[str, Any], test_frame: pd.DataFrame, *, batch_size: int, device: str) -> dict[str, Any]:
    metrics = evaluate_model_bundle(bundle, test_frame, split_name="test", batch_size=batch_size, device=device)
    return _flatten_metrics(metrics, baseline_name="trained_model")


def _write_markdown_report(
    output_path: Path,
    *,
    model_bundle_path: Path,
    split_root: Path,
    protocol_table: pd.DataFrame,
    input_scan: pd.DataFrame,
    baseline_table: pd.DataFrame,
    per_log_table: pd.DataFrame,
    linear_features: list[str],
) -> None:
    failed_protocol = protocol_table.loc[~protocol_table["passed"]]
    failed_inputs = input_scan.loc[~input_scan["passed"]]

    lines = [
        "# Model Sanity Checks",
        "",
        f"- model_bundle: `{model_bundle_path}`",
        f"- split_root: `{split_root}`",
        f"- protocol_failed_checks: {len(failed_protocol)}",
        f"- input_failed_checks: {len(failed_inputs)}",
        f"- simple_linear_feature_count: {len(linear_features)}",
        "",
        "## Protocol Checks",
        "",
        _dataframe_to_markdown(protocol_table),
        "",
        "## Input Leakage Scan",
        "",
        _dataframe_to_markdown(input_scan),
        "",
        "## Test Metrics and Negative Controls",
        "",
        _dataframe_to_markdown(baseline_table),
        "",
        "## Per-log Model Metrics",
        "",
        _dataframe_to_markdown(per_log_table),
        "",
        "## Simple Linear Features",
        "",
        ", ".join(f"`{column}`" for column in linear_features),
        "",
    ]
    output_path.write_text("\n".join(lines))


def _dataframe_to_markdown(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_empty_"
    display = frame if max_rows is None else frame.head(max_rows)
    columns = [str(column) for column in display.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row[column]) for column in display.columns]
        values = [value.replace("|", "\\|").replace("\n", " ") for value in values]
        rows.append("| " + " | ".join(values) + " |")
    if max_rows is not None and len(frame) > max_rows:
        rows.append(f"| ... {len(frame) - max_rows} more rows |" + " |" * (len(columns) - 1))
    return "\n".join(rows)


def run_sanity_checks(
    *,
    model_bundle_path: Path,
    split_root: Path,
    output_dir: Path,
    batch_size: int = 8192,
    device: str = "auto",
    seed: int = 20260508,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_split_frames(split_root)
    bundle = torch.load(model_bundle_path, map_location="cpu", weights_only=False)
    target_columns = list(bundle.get("target_columns", DEFAULT_TARGET_COLUMNS))

    protocol_table = build_split_protocol_table(split_root)
    input_scan = scan_bundle_inputs(bundle)
    _, train_targets = prepare_feature_target_frames(frames["train"], target_columns=target_columns)
    _, test_targets = prepare_feature_target_frames(frames["test"], target_columns=target_columns)
    aligned_model_test_targets = training_module._targets_for_bundle(bundle, frames["test"])

    model_row = _model_metrics_row(bundle, frames["test"], batch_size=batch_size, device=device)
    mean_row = compute_mean_baseline_metrics(train_targets, test_targets, target_columns=target_columns)
    linear_row, linear_features = compute_simple_linear_baseline_metrics(
        frames["train"],
        frames["test"],
        target_columns=target_columns,
    )
    predictions = predict_model_bundle(bundle, frames["test"], batch_size=batch_size, device=device)
    permuted_row = compute_permuted_target_metrics(
        aligned_model_test_targets.loc[:, target_columns].to_numpy(dtype=np.float64, copy=False),
        predictions.loc[:, target_columns].to_numpy(dtype=np.float64, copy=False),
        target_columns=target_columns,
        seed=seed,
    )

    baseline_table = pd.DataFrame([model_row, mean_row, linear_row, permuted_row])
    per_log_table = evaluate_model_bundle_by_log(
        bundle,
        frames["test"],
        split_name="test",
        min_samples=16,
        batch_size=batch_size,
        device=device,
    )

    paths = {
        "protocol_checks": output_dir / "protocol_checks.csv",
        "input_leakage_scan": output_dir / "input_leakage_scan.csv",
        "baseline_sanity_metrics": output_dir / "baseline_sanity_metrics.csv",
        "per_log_model_metrics": output_dir / "per_log_model_metrics.csv",
        "report": output_dir / "sanity_report.md",
    }
    protocol_table.to_csv(paths["protocol_checks"], index=False)
    input_scan.to_csv(paths["input_leakage_scan"], index=False)
    baseline_table.to_csv(paths["baseline_sanity_metrics"], index=False)
    per_log_table.to_csv(paths["per_log_model_metrics"], index=False)
    _write_markdown_report(
        paths["report"],
        model_bundle_path=model_bundle_path,
        split_root=split_root,
        protocol_table=protocol_table,
        input_scan=input_scan,
        baseline_table=baseline_table,
        per_log_table=per_log_table,
        linear_features=linear_features,
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage and plausibility sanity checks for a trained model bundle.")
    parser.add_argument("--model-bundle", required=True, type=Path)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260508)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_sanity_checks(
        model_bundle_path=args.model_bundle,
        split_root=args.split_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
