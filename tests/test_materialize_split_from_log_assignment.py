from __future__ import annotations

import json

import pandas as pd

from scripts.materialize_split_from_log_assignment import materialize_split_from_assignment


def test_materialize_split_from_assignment_reuses_log_splits(tmp_path):
    dataset = tmp_path / "dataset"
    log_a = dataset / "aircraft_id=flapper_01" / "log_id=log_a"
    log_b = dataset / "aircraft_id=flapper_01" / "log_id=log_b"
    log_a.mkdir(parents=True)
    log_b.mkdir(parents=True)
    for log_dir, x in [(log_a, 1.0), (log_b, 2.0)]:
        pd.DataFrame(
            {
                "time_s": [0.0, 0.01],
                "label_valid": [True, True],
                "flap_active": [True, True],
                "cycle_valid": [True, True],
                "cycle_id": [0, 0],
                "vehicle_local_position.z": [-10.0, -10.5],
                "fx_b": [x, x + 1.0],
            }
        ).to_parquet(log_dir / "samples.parquet", index=False)

    accepted_json = dataset / "accepted_logs.json"
    accepted_json.write_text(
        json.dumps(
            [
                {"log_id": "log_a", "samples_path": str(log_a / "samples.parquet")},
                {"log_id": "log_b", "samples_path": str(log_b / "samples.parquet")},
            ]
        ),
        encoding="utf-8",
    )
    manifest = dataset / "dataset_manifest.json"
    manifest.write_text(
        json.dumps({"dataset_id": "source_dataset", "accepted_logs_json": str(accepted_json)}),
        encoding="utf-8",
    )
    assignment = tmp_path / "all_logs.csv"
    pd.DataFrame(
        [
            {"log_id": "log_a", "split": "train"},
            {"log_id": "log_b", "split": "test"},
        ]
    ).to_csv(assignment, index=False)

    output = tmp_path / "split"
    materialize_split_from_assignment(
        source_manifest=manifest,
        assignment_csv=assignment,
        output_root=output,
        altitude_window_min_m=5.0,
    )

    train = pd.read_parquet(output / "train_samples.parquet")
    test = pd.read_parquet(output / "test_samples.parquet")
    val = pd.read_parquet(output / "val_samples.parquet")
    all_logs = pd.read_csv(output / "all_logs.csv")

    assert train["log_id"].unique().tolist() == ["log_a"]
    assert test["log_id"].unique().tolist() == ["log_b"]
    assert val.empty
    assert all_logs.set_index("log_id").loc["log_a", "valid_sample_count"] == 2
    assert json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))["split_policy"] == "whole_log_reused_assignment"
