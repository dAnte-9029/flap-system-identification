from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_lateral_smoothed_label_split import build_lateral_smoothed_label_split


def _write_metadata(path: Path) -> None:
    path.write_text(
        """
mass_properties:
  mass_kg:
    value: 1.0
  inertia_b_kg_m2:
    value:
      - [2.0, 0.0, 0.0]
      - [0.0, 3.0, 0.0]
      - [0.0, 0.0, 4.0]
label_definition:
  gravity_m_s2: 9.81
""",
        encoding="utf-8",
    )


def _sample_frame(split: str) -> pd.DataFrame:
    time_s = np.arange(25, dtype=float) * 0.05
    vy = np.sin(2.0 * np.pi * time_s)
    p = 0.5 * time_s**2
    r = -0.25 * time_s**2
    return pd.DataFrame(
        {
            "time_s": time_s,
            "log_id": ["log_a"] * len(time_s),
            "split": [split] * len(time_s),
            "vehicle_local_position.ax": np.zeros(len(time_s)),
            "vehicle_local_position.ay": np.zeros(len(time_s)),
            "vehicle_local_position.az": np.full(len(time_s), 9.81),
            "vehicle_local_position.vx": np.zeros(len(time_s)),
            "vehicle_local_position.vy": vy,
            "vehicle_local_position.vz": np.zeros(len(time_s)),
            "vehicle_attitude.q[0]": np.ones(len(time_s)),
            "vehicle_attitude.q[1]": np.zeros(len(time_s)),
            "vehicle_attitude.q[2]": np.zeros(len(time_s)),
            "vehicle_attitude.q[3]": np.zeros(len(time_s)),
            "vehicle_angular_velocity.xyz[0]": p,
            "vehicle_angular_velocity.xyz[1]": np.zeros(len(time_s)),
            "vehicle_angular_velocity.xyz[2]": r,
            "vehicle_angular_velocity.xyz_derivative[0]": np.full(len(time_s), 99.0),
            "vehicle_angular_velocity.xyz_derivative[1]": np.zeros(len(time_s)),
            "vehicle_angular_velocity.xyz_derivative[2]": np.full(len(time_s), 99.0),
            "fx_b": np.full(len(time_s), 10.0),
            "fy_b": np.full(len(time_s), 20.0),
            "fz_b": np.full(len(time_s), 30.0),
            "mx_b": np.full(len(time_s), 0.1),
            "my_b": np.full(len(time_s), 0.2),
            "mz_b": np.full(len(time_s), 0.3),
            "label_valid": np.ones(len(time_s), dtype=bool),
        }
    )


def test_build_lateral_smoothed_label_split_rewrites_only_lateral_targets(tmp_path: Path):
    split_root = tmp_path / "split"
    output_root = tmp_path / "lateral_smooth"
    split_root.mkdir()
    metadata_path = tmp_path / "metadata.yaml"
    _write_metadata(metadata_path)

    for split in ("train", "val", "test"):
        _sample_frame(split).to_parquet(split_root / f"{split}_samples.parquet", index=False)
        pd.DataFrame({"log_id": ["log_a"], "split": [split]}).to_csv(split_root / f"{split}_logs.csv", index=False)
    pd.DataFrame({"log_id": ["log_a"], "split": ["train"]}).to_csv(split_root / "all_logs.csv", index=False)

    outputs = build_lateral_smoothed_label_split(
        split_root=split_root,
        metadata_path=metadata_path,
        output_root=output_root,
        window_s=0.2,
        polyorder=2,
    )

    rewritten = pd.read_parquet(output_root / "train_samples.parquet")
    assert Path(outputs["manifest_path"]).exists()
    assert not np.allclose(rewritten["fy_b"].to_numpy(), 20.0)
    assert not np.allclose(rewritten["mx_b"].to_numpy(), 0.1)
    assert not np.allclose(rewritten["mz_b"].to_numpy(), 0.3)
    np.testing.assert_allclose(rewritten["fx_b"].to_numpy(), 10.0)
    np.testing.assert_allclose(rewritten["fz_b"].to_numpy(), 30.0)
    np.testing.assert_allclose(rewritten["my_b"].to_numpy(), 0.2)
    assert rewritten["lateral_label_valid"].all()
