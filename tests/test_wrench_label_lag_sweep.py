from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_time_aligned_smoothed_label_split import (
    apply_selected_lags,
    build_time_aligned_smoothed_label_split,
    run_train_only_lag_sweep,
)


def _write_metadata(path: Path) -> None:
    path.write_text(
        """
mass_properties:
  mass_kg:
    value: 1.0
  inertia_b_kg_m2:
    value:
      - [1.0, 0.0, 0.0]
      - [0.0, 1.0, 0.0]
      - [0.0, 0.0, 1.0]
label_definition:
  gravity_m_s2: 9.81
""",
        encoding="utf-8",
    )


def _frame(split: str, log_id: str) -> pd.DataFrame:
    time_s = np.arange(40, dtype=float) * 0.02
    vx = 0.5 * time_s**2
    wz = 0.2 * time_s**2
    return pd.DataFrame(
        {
            "time_s": time_s,
            "log_id": log_id,
            "split": split,
            "vehicle_local_position.vx": vx,
            "vehicle_local_position.vy": np.sin(time_s),
            "vehicle_local_position.vz": np.zeros(len(time_s)),
            "vehicle_local_position.ax": time_s,
            "vehicle_local_position.ay": np.cos(time_s),
            "vehicle_local_position.az": np.full(len(time_s), 9.81),
            "vehicle_attitude.q[0]": np.ones(len(time_s)),
            "vehicle_attitude.q[1]": np.zeros(len(time_s)),
            "vehicle_attitude.q[2]": np.zeros(len(time_s)),
            "vehicle_attitude.q[3]": np.zeros(len(time_s)),
            "vehicle_angular_velocity.xyz[0]": np.zeros(len(time_s)),
            "vehicle_angular_velocity.xyz[1]": np.zeros(len(time_s)),
            "vehicle_angular_velocity.xyz[2]": wz,
            "vehicle_angular_velocity.xyz_derivative[0]": np.zeros(len(time_s)),
            "vehicle_angular_velocity.xyz_derivative[1]": np.zeros(len(time_s)),
            "vehicle_angular_velocity.xyz_derivative[2]": 0.4 * time_s,
            "phase_raw_rad": time_s,
            "phase_raw_unwrapped_rad": time_s,
            "flap_frequency_hz": np.full(len(time_s), 6.0),
            "servo_left_elevon": np.sin(time_s),
            "servo_right_elevon": np.cos(time_s),
            "servo_rudder": np.sin(2.0 * time_s),
            "fx_b": time_s,
            "fy_b": np.sin(time_s),
            "fz_b": np.cos(time_s),
            "mx_b": np.zeros(len(time_s)),
            "my_b": np.zeros(len(time_s)),
            "mz_b": 0.4 * time_s,
        }
    )


def _write_split(root: Path) -> None:
    root.mkdir()
    for split in ("train", "val", "test"):
        frame = _frame(split, f"{split}_log")
        frame.to_parquet(root / f"{split}_samples.parquet", index=False)
        pd.DataFrame({"log_id": [f"{split}_log"], "split": [split]}).to_csv(root / f"{split}_logs.csv", index=False)
    pd.DataFrame({"log_id": ["train_log", "val_log", "test_log"]}).to_csv(root / "all_logs.csv", index=False)


def test_run_train_only_lag_sweep_writes_train_metrics(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts"
    train = _frame("train", "train_log")

    selected = run_train_only_lag_sweep(
        train,
        lag_candidates={"phase_raw_rad": [-0.02, 0.0, 0.02]},
        artifact_dir=artifact_dir,
    )

    assert selected == {"phase_raw_rad": 0.0}
    assert (artifact_dir / "lag_sweep_train_metrics.csv").exists()
    payload = json.loads((artifact_dir / "lag_selection.json").read_text(encoding="utf-8"))
    assert payload["selected_lags_s"]["phase_raw_rad"] == 0.0


def test_apply_selected_lags_adds_groupwise_aligned_columns():
    frame = pd.concat([_frame("train", "a"), _frame("train", "b")], ignore_index=True)

    shifted = apply_selected_lags(frame, {"phase_raw_rad": 0.04})

    assert "phase_raw_rad_aligned" in shifted.columns
    assert np.isnan(shifted.loc[0, "phase_raw_rad_aligned"])
    assert np.isnan(shifted.loc[len(_frame("train", "a")), "phase_raw_rad_aligned"])


def test_builder_preserves_splits_and_records_lag_selection(tmp_path: Path):
    split_root = tmp_path / "split"
    output_root = tmp_path / "out"
    artifact_dir = tmp_path / "artifacts"
    metadata_path = tmp_path / "metadata.yaml"
    _write_split(split_root)
    _write_metadata(metadata_path)

    outputs = build_time_aligned_smoothed_label_split(
        split_root=split_root,
        metadata_path=metadata_path,
        output_root=output_root,
        artifact_dir=artifact_dir,
        enable_lag_sweep=True,
    )

    assert Path(outputs["manifest_path"]).exists()
    rewritten = pd.read_parquet(output_root / "val_samples.parquet")
    assert len(rewritten) == 40
    assert "phase_raw_rad_aligned" in rewritten.columns
    assert "label_reconstruction_valid" in rewritten.columns
