#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEY_COLUMNS = ["outer_fold", "log_id", "segment_id", "time_s"]
QUATERNION_COLUMNS = [
    "vehicle_attitude.q[0]",
    "vehicle_attitude.q[1]",
    "vehicle_attitude.q[2]",
    "vehicle_attitude.q[3]",
]
POSITION_COLUMNS = [
    "vehicle_local_position.x",
    "vehicle_local_position.y",
    "vehicle_local_position.z",
]
VELOCITY_COLUMNS = [
    "vehicle_local_position.vx",
    "vehicle_local_position.vy",
    "vehicle_local_position.vz",
]
OMEGA_COLUMNS = [
    "vehicle_angular_velocity.xyz[0]",
    "vehicle_angular_velocity.xyz[1]",
    "vehicle_angular_velocity.xyz[2]",
]
STATE_COLUMNS = [
    "log_id",
    "segment_id",
    "time_s",
    *QUATERNION_COLUMNS,
    *POSITION_COLUMNS,
    *VELOCITY_COLUMNS,
    *OMEGA_COLUMNS,
    "fx_b",
    "fz_b",
]

DEFAULT_MODEL_MAP = {
    "Raw prior": "raw_prior",
    "Conditioned gain-bias": "gain_bias",
    "Pure TCN": "pure_tcn",
}


def normalize_quaternion_array(q: np.ndarray) -> np.ndarray:
    quat = np.asarray(q, dtype=float)
    norm = np.linalg.norm(quat, axis=1)
    if np.any(~np.isfinite(quat)) or np.any(norm <= 0.0):
        raise ValueError("quaternion rows must be finite and nonzero")
    return quat / norm[:, None]


def roll_from_quaternion(q: np.ndarray) -> np.ndarray:
    quat = normalize_quaternion_array(q)
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    return np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fold_index_from_name(path: Path) -> int:
    name = path.name
    if not name.startswith("fold_"):
        raise ValueError(f"unexpected fold directory name: {name}")
    return int(name.split("_", 1)[1])


def _read_state_rows(samples_path: Path, outer_fold: int) -> pd.DataFrame:
    samples = pd.read_parquet(samples_path)
    missing = [column for column in STATE_COLUMNS if column not in samples.columns]
    if missing:
        raise ValueError(f"{samples_path} missing required state columns: {missing}")

    state = samples[STATE_COLUMNS].copy()
    state.insert(0, "outer_fold", int(outer_fold))
    state["roll_rad"] = roll_from_quaternion(state[QUATERNION_COLUMNS].to_numpy(dtype=float))
    state = state.rename(columns={"fx_b": "label_fx_b_state", "fz_b": "label_fz_b_state"})
    return state


def _pivot_predictions(predictions: pd.DataFrame, model_map: dict[str, str]) -> pd.DataFrame:
    required = [*KEY_COLUMNS, "model", "label_fx_b", "label_fz_b", "pred_fx_b", "pred_fz_b"]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"prediction table missing required columns: {missing}")

    selected = predictions[predictions["model"].isin(model_map)].copy()
    found = set(selected["model"].unique())
    missing_models = sorted(set(model_map) - found)
    if missing_models:
        raise ValueError(f"prediction table missing required models: {missing_models}")

    selected["force_source"] = selected["model"].map(model_map)
    duplicates = selected.duplicated(KEY_COLUMNS + ["force_source"])
    if duplicates.any():
        raise ValueError("prediction table contains duplicate key/model rows")

    base = selected[KEY_COLUMNS + ["label_fx_b", "label_fz_b"]].drop_duplicates(KEY_COLUMNS)
    if len(base) != selected[KEY_COLUMNS].drop_duplicates().shape[0]:
        raise ValueError("label columns are not unique for each prediction key")

    force = selected.pivot(index=KEY_COLUMNS, columns="force_source", values=["pred_fx_b", "pred_fz_b"])
    force.columns = [f"{source}_{axis.replace('pred_', '')}" for axis, source in force.columns]
    force = force.reset_index()
    expected_columns = [f"{source}_{axis}" for source in model_map.values() for axis in ("fx_b", "fz_b")]
    missing_force = [column for column in expected_columns if column not in force.columns]
    if missing_force:
        raise ValueError(f"pivoted prediction table missing force columns: {missing_force}")

    return base.merge(force, on=KEY_COLUMNS, how="inner", validate="one_to_one")


def build_replay_ready_table(
    comparison_root: str | Path,
    model_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(comparison_root)
    resolved_model_map = dict(DEFAULT_MODEL_MAP if model_map is None else model_map)
    frames: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold_dir in sorted(root.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        outer_fold = _fold_index_from_name(fold_dir)
        manifest_path = fold_dir / "manifest.json"
        predictions_path = fold_dir / "test_predictions_common_rows.parquet"
        if not manifest_path.exists() or not predictions_path.exists():
            continue

        manifest = _load_json(manifest_path)
        fold_root = Path(manifest["fold_root"])
        samples_path = fold_root / "test_samples.parquet"
        state = _read_state_rows(samples_path, outer_fold=outer_fold)
        predictions = pd.read_parquet(predictions_path)
        pivoted = _pivot_predictions(predictions, resolved_model_map)
        joined = pivoted.merge(state, on=KEY_COLUMNS, how="inner", validate="one_to_one")
        if joined.empty:
            raise ValueError(f"fold {outer_fold} join produced no rows")

        label_delta_fx = float(np.nanmax(np.abs(joined["label_fx_b"] - joined["label_fx_b_state"])))
        label_delta_fz = float(np.nanmax(np.abs(joined["label_fz_b"] - joined["label_fz_b_state"])))
        joined = joined.drop(columns=["label_fx_b_state", "label_fz_b_state"])
        frames.append(joined)
        fold_summaries.append(
            {
                "outer_fold": outer_fold,
                "prediction_rows_per_model": int(manifest.get("common_test_rows", len(pivoted))),
                "state_rows": int(len(state)),
                "joined_rows": int(len(joined)),
                "max_abs_label_fx_mismatch_n": label_delta_fx,
                "max_abs_label_fz_mismatch_n": label_delta_fz,
                "manifest_path": str(manifest_path),
                "predictions_path": str(predictions_path),
                "samples_path": str(samples_path),
            }
        )

    if not frames:
        raise ValueError(f"no fold prediction artifacts found under {root}")

    table = pd.concat(frames, ignore_index=True)
    table = table.sort_values(KEY_COLUMNS).reset_index(drop=True)
    metadata = {
        "comparison_root": str(root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "join_key": KEY_COLUMNS,
        "join_policy": "exact inner join on outer_fold/log_id/segment_id/time_s",
        "model_map": resolved_model_map,
        "n_rows": int(len(table)),
        "folds": fold_summaries,
    }
    return table, metadata


def write_replay_ready_artifacts(
    output_root: str | Path,
    table: pd.DataFrame,
    metadata: dict[str, Any],
    overwrite: bool = False,
) -> None:
    output_path = Path(output_root)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(f"output root exists and is not empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path / "level2_replay_ready_rows.parquet", index=False)
    with (output_path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Level 2 replay-ready rows from frozen force predictions.")
    parser.add_argument("--comparison-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gain-bias-model", default="Conditioned gain-bias")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_map = {
        "Raw prior": "raw_prior",
        args.gain_bias_model: "gain_bias",
        "Pure TCN": "pure_tcn",
    }
    table, metadata = build_replay_ready_table(args.comparison_root, model_map=model_map)
    write_replay_ready_artifacts(args.output_root, table, metadata, overwrite=args.overwrite)
    print(f"wrote {len(table)} replay-ready rows to {args.output_root}")


if __name__ == "__main__":
    main()
