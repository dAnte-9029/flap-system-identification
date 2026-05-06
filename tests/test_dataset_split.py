from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from system_identification.dataset_split import (
    assign_cycle_block_splits,
    build_train_purge_intervals,
    extract_cycle_blocks,
    materialize_cycle_block_split,
    materialize_log_split,
)


def _samples_for_cycles(cycle_ids: list[int], rows_per_cycle: int = 2) -> pd.DataFrame:
    rows = []
    timestamp_us = 0
    for cycle_id in cycle_ids:
        for sample_idx in range(rows_per_cycle):
            rows.append(
                {
                    "timestamp_us": timestamp_us,
                    "cycle_id": cycle_id,
                    "label_valid": True,
                    "cycle_valid": True,
                    "flap_active": True,
                    "vehicle_land_detected.landed": 0.0,
                    "feature": float(cycle_id * 10 + sample_idx),
                }
            )
            timestamp_us += 10_000
    return pd.DataFrame(rows)


def test_build_cycle_blocks_groups_valid_cycles():
    samples = _samples_for_cycles([0, 1, 2, 3, 4], rows_per_cycle=2)
    samples.loc[samples["cycle_id"] == 2, "flap_active"] = False

    blocks = extract_cycle_blocks(
        samples,
        dataset_id="cohort_a",
        log_id="log_a",
        source_samples_path="/tmp/log_a/samples.parquet",
        block_size_cycles=2,
    )

    assert blocks["block_id"].tolist() == [
        "cohort_a:log_a:block_0000",
        "cohort_a:log_a:block_0001",
    ]
    assert blocks["cycle_start"].tolist() == [0, 3]
    assert blocks["cycle_end"].tolist() == [1, 4]
    assert blocks["cycle_count"].tolist() == [2, 2]
    assert blocks["sample_count"].tolist() == [4, 4]


def test_assign_blocks_purges_nearby_train_cycles():
    blocks = pd.DataFrame(
        [
            {
                "block_id": "b0",
                "dataset_id": "cohort_a",
                "log_id": "log_a",
                "source_samples_path": "/tmp/log_a/samples.parquet",
                "cycle_start": 0,
                "cycle_end": 1,
                "cycle_count": 2,
                "sample_count": 20,
            },
            {
                "block_id": "b1",
                "dataset_id": "cohort_a",
                "log_id": "log_a",
                "source_samples_path": "/tmp/log_a/samples.parquet",
                "cycle_start": 2,
                "cycle_end": 3,
                "cycle_count": 2,
                "sample_count": 20,
            },
            {
                "block_id": "b2",
                "dataset_id": "cohort_a",
                "log_id": "log_a",
                "source_samples_path": "/tmp/log_a/samples.parquet",
                "cycle_start": 4,
                "cycle_end": 5,
                "cycle_count": 2,
                "sample_count": 20,
            },
            {
                "block_id": "b3",
                "dataset_id": "cohort_b",
                "log_id": "log_b",
                "source_samples_path": "/tmp/log_b/samples.parquet",
                "cycle_start": 0,
                "cycle_end": 1,
                "cycle_count": 2,
                "sample_count": 20,
            },
        ]
    )

    assigned = assign_cycle_block_splits(
        blocks,
        seed=7,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
    )

    assert assigned["block_id"].is_unique
    assert set(assigned["split"]) == {"train", "val", "test"}

    assigned_for_purge = pd.DataFrame(
        [
            {
                "block_id": "train_left",
                "dataset_id": "cohort_a",
                "log_id": "log_a",
                "cycle_start": 0,
                "cycle_end": 1,
                "split": "train",
            },
            {
                "block_id": "val_mid",
                "dataset_id": "cohort_a",
                "log_id": "log_a",
                "cycle_start": 2,
                "cycle_end": 3,
                "split": "val",
            },
            {
                "block_id": "train_right",
                "dataset_id": "cohort_a",
                "log_id": "log_a",
                "cycle_start": 4,
                "cycle_end": 5,
                "split": "train",
            },
            {
                "block_id": "test_far",
                "dataset_id": "cohort_a",
                "log_id": "log_a",
                "cycle_start": 8,
                "cycle_end": 9,
                "split": "test",
            },
        ]
    )

    purge = build_train_purge_intervals(assigned_for_purge, purge_cycles=1)

    assert purge[["cycle_start", "cycle_end"]].to_dict("records") == [
        {"cycle_start": 1, "cycle_end": 4},
        {"cycle_start": 7, "cycle_end": 10},
    ]


def test_materialize_split_writes_manifests_and_parquets(tmp_path: Path):
    dataset_root = tmp_path / "cohort_a"
    log_a_dir = dataset_root / "aircraft_id=flapper_01" / "log_id=log_a"
    log_b_dir = dataset_root / "aircraft_id=flapper_01" / "log_id=log_b"
    log_a_dir.mkdir(parents=True)
    log_b_dir.mkdir(parents=True)

    samples_a = _samples_for_cycles([0, 1, 2, 3, 4, 5], rows_per_cycle=2)
    samples_b = _samples_for_cycles([0, 1, 2, 3], rows_per_cycle=2)
    samples_a.to_parquet(log_a_dir / "samples.parquet", index=False)
    samples_b.to_parquet(log_b_dir / "samples.parquet", index=False)

    accepted_logs = [
        {
            "log_id": "log_a",
            "samples_path": str(log_a_dir / "samples.parquet"),
        },
        {
            "log_id": "log_b",
            "samples_path": str(log_b_dir / "samples.parquet"),
        },
    ]
    accepted_logs_path = dataset_root / "accepted_logs.json"
    accepted_logs_path.write_text(json.dumps(accepted_logs, indent=2), encoding="utf-8")

    dataset_manifest = {
        "dataset_id": "cohort_a",
        "accepted_logs_json": str(accepted_logs_path),
    }
    dataset_manifest_path = dataset_root / "dataset_manifest.json"
    dataset_manifest_path.write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")

    output_root = tmp_path / "training_ready"
    outputs = materialize_cycle_block_split(
        manifest_paths=[dataset_manifest_path],
        output_root=output_root,
        block_size_cycles=2,
        purge_cycles=1,
        seed=3,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
    )

    assert Path(outputs["dataset_manifest_path"]).exists()
    assert (output_root / "all_blocks.csv").exists()
    assert (output_root / "train_blocks.csv").exists()
    assert (output_root / "val_blocks.csv").exists()
    assert (output_root / "test_blocks.csv").exists()
    assert (output_root / "train_samples.parquet").exists()
    assert (output_root / "val_samples.parquet").exists()
    assert (output_root / "test_samples.parquet").exists()

    train_samples = pd.read_parquet(output_root / "train_samples.parquet")
    val_samples = pd.read_parquet(output_root / "val_samples.parquet")
    test_samples = pd.read_parquet(output_root / "test_samples.parquet")

    for split_name, frame in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        assert not frame.empty
        assert frame["split"].eq(split_name).all()
        assert frame["label_valid"].all()
        assert frame["cycle_valid"].all()
        assert frame["flap_active"].all()
        assert frame["cycle_id"].ge(0).all()
        assert {"dataset_id", "log_id", "block_id", "source_samples_path"}.issubset(frame.columns)


def test_materialize_split_excludes_landed_samples(tmp_path: Path):
    dataset_root = tmp_path / "cohort_a"
    log_a_dir = dataset_root / "aircraft_id=flapper_01" / "log_id=log_a"
    log_a_dir.mkdir(parents=True)

    samples_a = _samples_for_cycles([0, 1, 2, 3], rows_per_cycle=2)
    samples_a.loc[samples_a["cycle_id"] == 0, "vehicle_land_detected.landed"] = 1.0
    samples_a.loc[samples_a["cycle_id"] == 3, "vehicle_land_detected.landed"] = 1.0
    samples_a.to_parquet(log_a_dir / "samples.parquet", index=False)

    accepted_logs = [
        {
            "log_id": "log_a",
            "samples_path": str(log_a_dir / "samples.parquet"),
        }
    ]
    accepted_logs_path = dataset_root / "accepted_logs.json"
    accepted_logs_path.write_text(json.dumps(accepted_logs, indent=2), encoding="utf-8")

    dataset_manifest = {
        "dataset_id": "cohort_a",
        "accepted_logs_json": str(accepted_logs_path),
    }
    dataset_manifest_path = dataset_root / "dataset_manifest.json"
    dataset_manifest_path.write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")

    output_root = tmp_path / "training_ready"
    materialize_cycle_block_split(
        manifest_paths=[dataset_manifest_path],
        output_root=output_root,
        block_size_cycles=1,
        purge_cycles=0,
        seed=1,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
    )

    merged = pd.concat(
        [
            pd.read_parquet(output_root / "train_samples.parquet"),
            pd.read_parquet(output_root / "val_samples.parquet"),
            pd.read_parquet(output_root / "test_samples.parquet"),
        ],
        ignore_index=True,
    )

    assert not merged.empty
    assert (merged["vehicle_land_detected.landed"] < 0.5).all()
    assert set(merged["cycle_id"].astype(int).unique()) == {1, 2}


def test_materialize_log_split_keeps_logs_disjoint(tmp_path: Path):
    dataset_root = tmp_path / "cohort_logsplit"
    accepted_logs = []

    for log_index in range(5):
        log_id = f"log_{log_index}"
        log_dir = dataset_root / "aircraft_id=flapper_01" / f"log_id={log_id}"
        log_dir.mkdir(parents=True)
        samples = _samples_for_cycles([0, 1, 2, 3], rows_per_cycle=2)
        samples["feature"] = samples["feature"] + 100.0 * log_index
        samples.to_parquet(log_dir / "samples.parquet", index=False)
        accepted_logs.append(
            {
                "log_id": log_id,
                "samples_path": str(log_dir / "samples.parquet"),
            }
        )

    accepted_logs_path = dataset_root / "accepted_logs.json"
    accepted_logs_path.write_text(json.dumps(accepted_logs, indent=2), encoding="utf-8")
    dataset_manifest_path = dataset_root / "dataset_manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "cohort_logsplit",
                "accepted_logs_json": str(accepted_logs_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "logsplit"
    outputs = materialize_log_split(
        manifest_paths=[dataset_manifest_path],
        output_root=output_root,
        seed=9,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )

    assert Path(outputs["dataset_manifest_path"]).exists()
    assert (output_root / "train_logs.csv").exists()
    assert (output_root / "val_logs.csv").exists()
    assert (output_root / "test_logs.csv").exists()
    assert (output_root / "train_samples.parquet").exists()
    assert (output_root / "val_samples.parquet").exists()
    assert (output_root / "test_samples.parquet").exists()

    split_tables = {
        split_name: pd.read_parquet(output_root / f"{split_name}_samples.parquet")
        for split_name in ["train", "val", "test"]
    }
    logs_by_split = {
        split_name: set(zip(frame["dataset_id"], frame["log_id"]))
        for split_name, frame in split_tables.items()
    }

    assert logs_by_split["train"].isdisjoint(logs_by_split["val"])
    assert logs_by_split["train"].isdisjoint(logs_by_split["test"])
    assert logs_by_split["val"].isdisjoint(logs_by_split["test"])
    assert sum(len(logs) for logs in logs_by_split.values()) == 5

    for split_name, frame in split_tables.items():
        assert not frame.empty
        assert frame["split"].eq(split_name).all()
        assert frame["label_valid"].all()
        assert frame["cycle_valid"].all()
        assert frame["flap_active"].all()
        assert (frame["vehicle_land_detected.landed"] < 0.5).all()
        assert {"dataset_id", "log_id", "source_samples_path", "split"}.issubset(frame.columns)


def test_materialize_log_split_can_trim_to_altitude_window(tmp_path: Path):
    dataset_root = tmp_path / "cohort_altitude"
    accepted_logs = []

    for log_index in range(3):
        log_id = f"log_{log_index}"
        log_dir = dataset_root / "aircraft_id=flapper_01" / f"log_id={log_id}"
        log_dir.mkdir(parents=True)
        samples = _samples_for_cycles(list(range(8)), rows_per_cycle=1)
        samples["time_s"] = samples["timestamp_us"] * 1e-6
        samples["vehicle_local_position.z"] = [
            -1.0,
            -4.0,
            -5.1,
            -8.0,
            -3.0,
            -7.0,
            -5.2,
            -2.0,
        ]
        samples.to_parquet(log_dir / "samples.parquet", index=False)
        accepted_logs.append(
            {
                "log_id": log_id,
                "samples_path": str(log_dir / "samples.parquet"),
            }
        )

    accepted_logs_path = dataset_root / "accepted_logs.json"
    accepted_logs_path.write_text(json.dumps(accepted_logs, indent=2), encoding="utf-8")
    dataset_manifest_path = dataset_root / "dataset_manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "cohort_altitude",
                "accepted_logs_json": str(accepted_logs_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "logsplit_altitude"
    materialize_log_split(
        manifest_paths=[dataset_manifest_path],
        output_root=output_root,
        seed=1,
        train_ratio=0.34,
        val_ratio=0.33,
        test_ratio=0.33,
        altitude_window_min_m=5.0,
    )

    merged = pd.concat(
        [
            pd.read_parquet(output_root / "train_samples.parquet"),
            pd.read_parquet(output_root / "val_samples.parquet"),
            pd.read_parquet(output_root / "test_samples.parquet"),
        ],
        ignore_index=True,
    )

    assert not merged.empty
    assert set(merged["cycle_id"].astype(int).unique()) == {2, 3, 4, 5, 6}

    for _, group in merged.groupby("log_id"):
        assert group["cycle_id"].astype(int).tolist() == [2, 3, 4, 5, 6]
